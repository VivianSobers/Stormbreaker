"""Tests for the attribution model.

The interesting property is not predictive accuracy — plain least squares
predicts fine. It is whether the *coefficients* land on the truth, because
that is what gets shown to a user as "Slack costs you 1.9 W".
"""

import numpy as np
import pytest

from stormbreaker.model import (
    Dataset,
    _solve_bounded_ridge,
    attribute,
    choose_budget,
    coefficient_table,
    fit,
    mean_watts,
    predict,
)


def make_dataset(n=600, seed=0, noise=0.05):
    """Synthetic machine with three apps and a known cost structure."""
    rng = np.random.default_rng(seed)
    truth = {"heavy": 6.0, "light": 1.5, "io_hog": 0.0}

    cpu_heavy = rng.random(n) * 0.8
    cpu_light = rng.random(n) * 0.3
    io = rng.random(n) * 40.0

    X = np.column_stack([cpu_heavy, cpu_light, io])
    columns = [("heavy", "cpu"), ("light", "cpu"), ("io_hog", "io_mb")]

    baseline = 3.2
    y = (
        baseline
        + truth["heavy"] * cpu_heavy
        + truth["light"] * cpu_light
        + 0.02 * io
        + rng.normal(0, noise, n)
    )

    ds = Dataset(
        X=X,
        y=y,
        columns=columns,
        ts=np.arange(n, dtype=float) * 5.0,
        freq_edges=[],
        target="soc_w",
        win_ids=list(range(n)),
        globals_={
            "dt": np.full(n, 5.0),
            "discharging": np.zeros(n),
            "batt_w": np.zeros(n),
            "charge": np.zeros(n),
            "volt_v": np.full(n, 11.5),
        },
    )
    return ds, truth, baseline


def test_recovers_known_coefficients():
    ds, truth, baseline = make_dataset()
    f = fit(ds)

    assert f.baseline == pytest.approx(baseline, abs=0.15)
    by_col = {lab: f.coef[i] for i, (lab, _) in enumerate(f.columns)}
    assert by_col["heavy"] == pytest.approx(truth["heavy"], rel=0.05)
    assert by_col["light"] == pytest.approx(truth["light"], rel=0.10)
    assert f.r2 > 0.99


def test_coefficients_are_never_negative():
    """The whole point of NNLS. Feed it a feature that anti-correlates with
    power and check it is clamped to zero rather than allowed to cancel."""
    ds, _truth, _baseline = make_dataset()
    # A column that genuinely reduces measured power would tempt OLS negative.
    ds.X = np.column_stack([ds.X, ds.X[:, 0]])
    ds.columns = [*ds.columns, ("phantom", "cpu")]
    ds.y = ds.y - 2.0 * ds.X[:, 3]

    f = fit(ds)
    assert (f.coef >= 0).all()
    assert f.baseline >= 0


def test_baseline_is_not_penalised():
    """A large ridge penalty must shrink app coefficients but leave the idle
    baseline free, or the static draw gets pushed onto applications."""
    ds, _truth, baseline = make_dataset(noise=0.01)
    strong = fit(ds, lam=1e4)
    assert strong.baseline > baseline * 0.5
    assert strong.coef.max() < 6.0


def test_attribution_sums_to_prediction():
    ds, _truth, _baseline = make_dataset()
    f = fit(ds)
    attr = attribute(ds, f)
    total = sum(attr.values()) + f.baseline
    np.testing.assert_allclose(total, predict(ds, f), rtol=1e-9)


def test_idle_app_gets_no_power():
    """An app that never does anything must not be assigned watts."""
    ds, _truth, _baseline = make_dataset()
    ds.X = np.column_stack([ds.X, np.zeros(len(ds.y))])
    ds.columns = [*ds.columns, ("ghost", "cpu")]
    f = fit(ds)
    ranked = dict(mean_watts(ds, f))
    assert ranked.get("ghost", 0.0) == 0.0


def test_ranking_is_by_watts_not_cpu():
    """A low-CPU, high-cost workload must outrank a high-CPU, cheap one —
    this is the behaviour that distinguishes the tool from `top`."""
    n = 400
    rng = np.random.default_rng(3)
    cheap_cpu = np.full(n, 0.9) + rng.normal(0, 0.02, n)  # lots of CPU
    pricey_cpu = np.full(n, 0.1) + rng.normal(0, 0.02, n)  # little CPU
    y = 2.0 + 0.5 * cheap_cpu + 20.0 * pricey_cpu + rng.normal(0, 0.01, n)

    ds = Dataset(
        X=np.column_stack([cheap_cpu, pricey_cpu]),
        y=y,
        columns=[("cheap", "cpu"), ("pricey", "cpu")],
        ts=np.arange(n, dtype=float),
        freq_edges=[],
        target="soc_w",
        win_ids=list(range(n)),
        globals_={"dt": np.full(n, 5.0), "discharging": np.zeros(n)},
    )
    ranked = mean_watts(ds, fit(ds))
    assert ranked[0][0] == "pricey"


def test_solver_handles_collinear_features():
    """Two identical columns are individually unidentifiable. The split between
    them is arbitrary, but their total must stay sane and non-negative."""
    rng = np.random.default_rng(7)
    x = rng.random(300)
    X = np.column_stack([np.ones(300), x, x.copy()])
    y = 1.0 + 4.0 * x
    KINDS = ["baseline", "cpu", "cpu"]
    COLS = [("", "baseline"), ("a", "cpu"), ("b", "cpu")]

    exact = _solve_bounded_ridge(X, y, 0.0, KINDS, COLS)
    assert (exact >= 0).all()
    assert exact[1] + exact[2] == pytest.approx(4.0, rel=1e-6)
    assert exact[0] == pytest.approx(1.0, abs=1e-6)


def test_ridge_shrinks_toward_the_baseline_never_away():
    """Ridge biases app coefficients downward. Because the baseline is
    unpenalised it takes up the slack, so regularisation makes attribution
    *conservative* — watts move from applications to the unattributable floor,
    never the other way round. Worth knowing when reading a report.
    """
    rng = np.random.default_rng(7)
    x = rng.random(300)
    X = np.column_stack([np.ones(300), x, x.copy()])
    y = 1.0 + 4.0 * x
    KINDS = ["baseline", "cpu", "cpu"]
    COLS = [("", "baseline"), ("a", "cpu"), ("b", "cpu")]

    exact = _solve_bounded_ridge(X, y, 0.0, KINDS, COLS)
    shrunk = _solve_bounded_ridge(X, y, 0.1, KINDS, COLS)

    assert (shrunk >= 0).all()
    assert shrunk[1] + shrunk[2] < exact[1] + exact[2]
    assert shrunk[0] > exact[0]
    # The baseline takes up as much slack as physics permits, but it is capped
    # at the minimum observed power, so heavy shrinkage leaves the model
    # slightly under-predicting rather than inventing an impossible floor.
    assert shrunk[0] <= np.min(y) + 1e-9
    assert (X @ shrunk).mean() == pytest.approx((X @ exact).mean(), rel=0.05)


def test_baseline_cannot_exceed_minimum_observed_power():
    """Every non-baseline term is non-negative, so the constant term can never
    be larger than the smallest power reading. A fit that violates this is
    claiming the machine idles at more than it was ever measured drawing.
    """
    rng = np.random.default_rng(11)
    n = 400
    cpu = rng.random(n) * 2.0
    y = 4.0 + 3.0 * cpu + rng.normal(0, 0.05, n)

    ds = Dataset(
        X=cpu.reshape(-1, 1),
        y=y,
        columns=[("app", "cpu")],
        ts=np.arange(n, dtype=float),
        freq_edges=[],
        target="soc_w",
        win_ids=list(range(n)),
        globals_={"dt": np.full(n, 5.0), "discharging": np.zeros(n)},
    )
    f = fit(ds)
    assert f.baseline <= float(np.min(y)) + 1e-9


def test_near_idle_column_cannot_absorb_watts():
    """Regression test for the failure that made the first real run nonsense.

    A process using a thousandth of a core was handed a coefficient of ~24000
    W per core, because per-column normalisation had rescaled its near-empty
    column up to parity with genuinely busy ones. It must now be dropped for
    lack of identifiable activity instead.
    """
    rng = np.random.default_rng(5)
    n = 500
    busy = rng.random(n) * 1.5
    ghost = rng.random(n) * 1e-4  # a daemon doing essentially nothing
    y = 3.0 + 5.0 * busy + rng.normal(0, 0.2, n)

    ds = Dataset(
        X=np.column_stack([busy, ghost]),
        y=y,
        columns=[("busy", "cpu"), ("ghost", "cpu")],
        ts=np.arange(n, dtype=float),
        freq_edges=[],
        target="soc_w",
        win_ids=list(range(n)),
        globals_={"dt": np.full(n, 5.0), "discharging": np.zeros(n)},
    )
    f = fit(ds)
    by_label = {lab: f.coef[i] for i, (lab, _) in enumerate(f.columns)}
    assert by_label["ghost"] == 0.0
    assert by_label["busy"] == pytest.approx(5.0, rel=0.1)
    ranked = dict(mean_watts(ds, f))
    assert ranked.get("ghost", 0.0) == 0.0


def test_coefficient_table_reports_identifying_activity():
    """A high per-unit rate on a barely-exercised feature must be flagged, and
    the table must rank by actual contribution rather than by the raw rate —
    otherwise it reads as "this process costs 60 W"."""
    rng = np.random.default_rng(17)
    n = 500
    cpu = rng.random(n) * 2.0  # routinely ~1 busy core
    gpu = rng.random(n) * 0.02  # never near a busy GPU
    y = 3.0 + 5.0 * cpu + 60.0 * gpu + rng.normal(0, 0.05, n)

    ds = Dataset(
        X=np.column_stack([cpu, gpu]),
        y=y,
        columns=[("worker", "cpu"), ("compositor", "gpu")],
        ts=np.arange(n, dtype=float),
        freq_edges=[],
        target="soc_w",
        win_ids=list(range(n)),
        globals_={"dt": np.full(n, 5.0), "discharging": np.zeros(n)},
    )
    rows = coefficient_table(fit(ds), ds)
    by_feature = {r.feature: r for r in rows}

    gpu_row = by_feature["gpu"]
    assert gpu_row.coef > 10.0  # a large per-unit rate...
    assert gpu_row.watts_mean < 1.5  # ...that contributes very little
    assert gpu_row.extrapolated

    cpu_row = by_feature["cpu"]
    assert not cpu_row.extrapolated
    assert cpu_row.watts_mean == pytest.approx(cpu_row.coef * cpu.mean(), rel=1e-9)

    # ranked by contribution, so the big rate does not lead the table
    assert rows[0].feature == "cpu"


def test_coefficient_table_works_without_a_dataset():
    ds, _t, _b = make_dataset()
    rows = coefficient_table(fit(ds))
    assert rows
    assert all(r.activity_p95 != r.activity_p95 for r in rows)  # NaN


def test_column_budget_scales_with_available_data():
    """Guards the failure that made the defaults produce a model worse than
    predicting the mean: 186 columns fitted on 77 training windows."""
    small_top, small_buckets = choose_budget(120, None, None)
    big_top, big_buckets = choose_budget(5000, None, None)

    assert small_top < big_top
    assert small_buckets <= big_buckets
    assert small_buckets == 1  # no frequency resolution on a short recording

    for n in (40, 120, 400, 2000):
        top, buckets = choose_budget(n, None, None)
        columns = (top + 1) * (buckets + 3)
        assert columns <= max(n, 20), f"{n} windows would fit {columns} columns"


def test_explicit_budget_is_honoured():
    """An analyst asking for a specific shape gets it, data volume aside."""
    assert choose_budget(50, 25, 3) == (25, 3)
    assert choose_budget(9999, 4, 1) == (4, 1)


def test_too_little_data_is_an_error_not_a_guess():
    ds, _t, _b = make_dataset(n=10)
    with pytest.raises(ValueError, match="usable windows"):
        fit(ds)
