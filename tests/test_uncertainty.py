"""Tests for the confidence range on a single application's watts.

The point estimate for one application is the weakest number this tool prints,
and the bootstrap exists to say so per row rather than in a footnote. What is
worth testing is therefore not that an interval has some particular width, but
that it is wide exactly where the estimate deserves no trust — and, just as
importantly, that the two failures it *cannot* see stay documented rather than
quietly assumed away.
"""

import numpy as np
import pytest

from stormbreaker.model import Dataset, fit
from stormbreaker.uncertainty import (
    MIN_RESAMPLES,
    _resample_index,
    block_length,
    bootstrap_watts,
    render_intervals,
)


def _ds(X, y, labels):
    n = X.shape[0]
    return Dataset(
        X=X,
        y=y,
        columns=[(lab, "cpu0") for lab in labels],
        ts=np.arange(n, dtype=float),
        freq_edges=[],
        target="soc_w",
        win_ids=list(range(n)),
        globals_={"dt": np.full(n, 5.0), "discharging": np.zeros(n)},
    )


def _bursty(rng, n, period=40):
    """An application that runs in bursts, the way real ones do."""
    phase = (np.arange(n) // period) % 2
    return phase * (0.8 + rng.normal(0, 0.05, n)).clip(0.1)


def test_resample_index_stays_in_range_and_keeps_length():
    rng = np.random.default_rng(0)
    idx = _resample_index(100, 7, rng)
    assert len(idx) == 100
    assert idx.min() >= 0 and idx.max() < 100


def test_block_length_grows_slowly_with_the_series():
    assert block_length(27) == 3
    assert block_length(2600) == 14
    assert block_length(10) >= 2


def test_too_few_resamples_is_refused():
    """Reading a 90th percentile off five points is not an interval."""
    rng = np.random.default_rng(0)
    X = rng.random((200, 1))
    ds = _ds(X, 3.0 + X[:, 0] * 4.0, ["a"])
    with pytest.raises(ValueError, match="too few"):
        bootstrap_watts(ds, fit(ds), n_resamples=MIN_RESAMPLES - 1)


def test_an_independent_application_gets_a_tight_interval():
    rng = np.random.default_rng(1)
    n = 400
    a = _bursty(rng, n, 40)
    b = _bursty(rng, n, 31)  # different period, so the two come apart
    X = np.column_stack([a, b])
    y = 3.0 + 4.0 * a + 2.0 * b + rng.normal(0, 0.1, n)
    ds = _ds(X, y, ["a", "b"])

    iv = bootstrap_watts(ds, fit(ds), n_resamples=30, seed=0).intervals()
    truth_a = float((4.0 * a).mean())

    assert iv["a"].separable
    assert iv["a"].relative_width < 0.2
    assert abs(iv["a"].point - truth_a) / truth_a < 0.10


def test_the_interval_does_not_promise_to_cover_the_truth():
    """A documented limit, pinned so it cannot be forgotten.

    The estimator is regularised and bounded; both shrink coefficients. The
    bootstrap resamples that estimator, so it reports the spread of a biased
    thing and says nothing about the bias. Here the point estimate sits ~6%
    above the true watts while the interval spans ~3%, missing truth entirely.
    Only `selftest` measures absolute accuracy.
    """
    rng = np.random.default_rng(1)
    n = 400
    a = _bursty(rng, n, 40)
    b = _bursty(rng, n, 31)
    X = np.column_stack([a, b])
    y = 3.0 + 4.0 * a + 2.0 * b + rng.normal(0, 0.1, n)
    ds = _ds(X, y, ["a", "b"])

    iv = bootstrap_watts(ds, fit(ds), n_resamples=30, seed=0).intervals()["a"]
    truth = float((4.0 * a).mean())

    assert iv.point > truth  # shrinkage does not always land low
    assert not (iv.lo <= truth <= iv.hi)


def test_an_unseparable_pair_gets_a_confident_wrong_split():
    """The other documented limit, and the reason group totals exist.

    Two applications that are always active together can be split any way at
    all without changing the prediction. Ridge breaks the tie, and breaks it
    the *same* way in every resample — so the interval is narrow around a
    number that is simply wrong. Their total, however, is sound.
    """
    rng = np.random.default_rng(2)
    n = 400
    shared = _bursty(rng, n, 40)
    X = np.column_stack([shared, shared * 0.9])
    y = 3.0 + 4.0 * X[:, 0] + 4.0 * X[:, 1] + rng.normal(0, 0.1, n)
    ds = _ds(X, y, ["a", "b"])

    bs = bootstrap_watts(ds, fit(ds), n_resamples=30, seed=0, groups=[["a", "b"]])
    iv = bs.intervals()

    truth_a = float((4.0 * X[:, 0]).mean())
    truth_total = truth_a + float((4.0 * X[:, 1]).mean())

    # narrow, and wrong by far more than its own width
    assert iv["a"].relative_width < 0.05
    assert abs(iv["a"].point - truth_a) / truth_a > 3 * iv["a"].relative_width

    # the total is the number that survives
    total = bs.combined(["a", "b"])
    assert abs(total.point - truth_total) / truth_total < 0.06


def test_combined_sums_within_each_resample_not_after():
    """Order of operations. Summing intervals would give a wider range than
    summing draws whenever two applications trade the same watts back and
    forth, which is exactly the case group totals exist for."""
    rng = np.random.default_rng(2)
    n = 400
    shared = _bursty(rng, n, 40)
    X = np.column_stack([shared, shared * 0.9])
    y = 3.0 + 4.0 * X.sum(axis=1) + rng.normal(0, 0.1, n)
    ds = _ds(X, y, ["a", "b"])

    bs = bootstrap_watts(ds, fit(ds), n_resamples=30, seed=0)
    iv = bs.intervals()
    total = bs.combined(["a", "b"])

    assert total.width <= iv["a"].width + iv["b"].width + 1e-9


def test_blocks_are_wider_than_independent_resampling():
    """The reason for blocks at all.

    Adjacent windows are correlated, so resampling them one at a time treats
    correlated observations as independent evidence and reports an interval
    several times too narrow.
    """
    rng = np.random.default_rng(3)
    n = 400
    a = _bursty(rng, n, 40)
    b = _bursty(rng, n, 31)
    X = np.column_stack([a, b])
    y = 3.0 + 4.0 * a + 2.0 * b + rng.normal(0, 0.4, n)
    ds = _ds(X, y, ["a", "b"])
    f = fit(ds)

    blocked = bootstrap_watts(ds, f, n_resamples=30, block=40, seed=0).intervals()
    single = bootstrap_watts(ds, f, n_resamples=30, block=1, seed=0).intervals()
    assert blocked["a"].width > single["a"].width


def test_the_same_seed_gives_the_same_interval():
    rng = np.random.default_rng(4)
    n = 300
    a = _bursty(rng, n, 30)
    ds = _ds(a.reshape(-1, 1), 3.0 + 4.0 * a + rng.normal(0, 0.1, n), ["a"])
    f = fit(ds)

    one = bootstrap_watts(ds, f, n_resamples=25, seed=7).intervals()["a"]
    two = bootstrap_watts(ds, f, n_resamples=25, seed=7).intervals()["a"]
    assert one.lo == two.lo and one.hi == two.hi


def test_a_wider_interval_covers_a_narrower_one():
    rng = np.random.default_rng(5)
    n = 300
    a = _bursty(rng, n, 30)
    ds = _ds(a.reshape(-1, 1), 3.0 + 4.0 * a + rng.normal(0, 0.3, n), ["a"])
    f = fit(ds)

    wide = bootstrap_watts(ds, f, n_resamples=40, ci=0.98, seed=1).intervals()
    narrow = bootstrap_watts(ds, f, n_resamples=40, ci=0.50, seed=1).intervals()
    assert wide["a"].lo <= narrow["a"].lo
    assert wide["a"].hi >= narrow["a"].hi


def test_rendering_warns_about_grouped_rows():
    rng = np.random.default_rng(6)
    n = 300
    shared = _bursty(rng, n, 40)
    X = np.column_stack([shared, shared * 0.9])
    y = 3.0 + 4.0 * X.sum(axis=1) + rng.normal(0, 0.1, n)
    ds = _ds(X, y, ["a", "b"])

    bs = bootstrap_watts(ds, fit(ds), n_resamples=25, seed=0, groups=[["a", "b"]])
    text = render_intervals(bs)
    assert "[?]" in text
    assert "Trust their total instead" in text
    assert "90%" in text
