"""Tests for "what changed between two stretches of time".

The dangerous output here is a plausible one. A ranked list of applications
whose watts moved reads like an answer whether or not the model can explain
what actually happened, so most of these tests are about the cases where it
cannot: a plug-in between the two periods, a profile change, and a measured
change that the applications do not account for.
"""

import numpy as np
import pytest

from stormbreaker.compare import (
    MIN_PERIOD_WINDOWS,
    compare,
    render_comparison,
    split_periods,
)
from stormbreaker.model import Dataset


def _ds(a_before, a_after, extra_after=0.0, discharging=None, profiles=None):
    """Two back-to-back periods of equal length, one column per application."""
    nb, na = len(a_before), len(a_after)
    n = nb + na
    a = np.concatenate([a_before, a_after])
    b = np.concatenate([np.full(nb, 0.4), np.full(na, 0.4)])
    rng = np.random.default_rng(0)
    y = 3.0 + 4.0 * a + 2.0 * b + rng.normal(0, 0.05, n)
    y[nb:] += extra_after
    ts = np.arange(n, dtype=float) * 5.0
    disc = np.ones(n) if discharging is None else np.asarray(discharging, float)
    return Dataset(
        X=np.column_stack([a, b]),
        y=y,
        columns=[("app", "cpu0"), ("daemon", "cpu0")],
        ts=ts,
        freq_edges=[],
        target="soc_w",
        win_ids=list(range(n)),
        globals_={"dt": np.full(n, 5.0), "discharging": disc},
        profiles=None if profiles is None else np.array(profiles, dtype=object),
    )


def _busy(n, level, rng, period=13):
    """Duty-cycled activity, so the column is identifiable at all."""
    phase = (np.arange(n) // period) % 2
    return phase * level * (1.0 + rng.normal(0, 0.05, n)).clip(0.5)


def test_split_is_anchored_on_the_newest_window():
    """Not on the wall clock: a comparison run a day after collection stopped
    must mean the same thing as one run during it."""
    ds = _ds(np.zeros(60), np.zeros(60))
    before, after = split_periods(ds, recent_min=2.5, baseline_min=2.5)
    assert after[-1]
    assert not before[-1]
    assert before.sum() and after.sum()


def test_ending_bounds_the_recent_period_at_both_ends():
    """With --ending set, 'recent' is a window in the past, not everything
    after a start point."""
    ds = _ds(np.zeros(120), np.zeros(120))
    _before, after = split_periods(ds, recent_min=2.0, baseline_min=2.0,
                                   ending_min=3.0)
    assert after.sum() > 0
    assert not after[-1]  # the newest windows are excluded


def test_a_short_period_is_refused():
    ds = _ds(np.zeros(MIN_PERIOD_WINDOWS - 5), np.zeros(60))
    with pytest.raises(ValueError, match="earlier period"):
        compare(ds, recent_min=5.0, baseline_min=1.0)


def test_a_plug_in_between_the_periods_is_refused():
    """The case that produced the first real run's nonsense: package power up
    1.26 W with busy cores down from 1.73 to 0.42, because the machine had
    been plugged in."""
    n = 120
    rng = np.random.default_rng(1)
    ds = _ds(
        _busy(n, 1.0, rng),
        _busy(n, 1.0, rng),
        discharging=np.concatenate([np.ones(n), np.zeros(n)]),
    )
    with pytest.raises(ValueError, match="on battery.*on mains"):
        compare(ds, recent_min=10.0, baseline_min=10.0)


def test_a_period_straddling_a_plug_in_is_refused():
    n = 120
    rng = np.random.default_rng(1)
    half = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    ds = _ds(
        _busy(n, 1.0, rng),
        _busy(n, 1.0, rng),
        discharging=np.concatenate([np.ones(n), half]),
    )
    with pytest.raises(ValueError, match="itself part plugged in"):
        compare(ds, recent_min=10.0, baseline_min=10.0)


def test_a_profile_change_between_the_periods_is_refused():
    n = 120
    rng = np.random.default_rng(1)
    ds = _ds(
        _busy(n, 1.0, rng),
        _busy(n, 1.0, rng),
        profiles=["balanced"] * n + ["performance"] * n,
    )
    with pytest.raises(ValueError, match="more than one power regime"):
        compare(ds, recent_min=10.0, baseline_min=10.0)


def test_an_application_that_started_working_harder_is_found():
    n = 160
    rng = np.random.default_rng(2)
    ds = _ds(_busy(n, 0.5, rng), _busy(n, 1.5, rng))

    c = compare(ds, recent_min=13.0, baseline_min=13.0, n_resamples=25)
    app = next(ch for ch in c.changes if ch.label == "app")

    assert app.delta > 0
    assert app.real
    assert app.lo > 0
    assert c.trustworthy
    assert 0.7 <= c.explained <= 1.3


def test_an_unchanged_application_is_not_reported_as_a_change():
    n = 160
    rng = np.random.default_rng(3)
    ds = _ds(_busy(n, 1.0, rng), _busy(n, 1.0, rng))

    c = compare(ds, recent_min=13.0, baseline_min=13.0, n_resamples=25)
    daemon = next(ch for ch in c.changes if ch.label == "daemon")
    assert not daemon.real


def test_power_that_no_application_explains_lands_in_unexplained():
    """A change in the *cost* of work — heat, a device waking up — must not be
    charged to whichever application happens to be busiest."""
    n = 160
    rng = np.random.default_rng(4)
    ds = _ds(_busy(n, 1.0, rng), _busy(n, 1.0, rng), extra_after=3.0)

    c = compare(ds, recent_min=13.0, baseline_min=13.0, n_resamples=25)
    assert c.unexplained > 2.0
    assert abs(c.attributed_delta) < 1.0
    assert not c.trustworthy


def test_an_untrustworthy_comparison_says_so_before_ranking_anything():
    n = 160
    rng = np.random.default_rng(4)
    ds = _ds(_busy(n, 1.0, rng), _busy(n, 1.0, rng), extra_after=3.0)

    text = render_comparison(compare(ds, recent_min=13.0, baseline_min=13.0,
                                     n_resamples=25))
    assert "partial account, not the answer" in text
    assert text.index("partial account") < text.index("unexplained")


def test_a_tiny_change_is_not_called_untrustworthy():
    """Dividing a near-zero attributed change by a near-zero measured one
    produces a meaningless ratio; no claim is being made either way."""
    n = 160
    rng = np.random.default_rng(5)
    ds = _ds(_busy(n, 1.0, rng), _busy(n, 1.0, rng))

    c = compare(ds, recent_min=13.0, baseline_min=13.0, n_resamples=25)
    assert abs(c.measured_delta) < 0.25
    assert c.trustworthy
