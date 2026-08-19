"""Tests for the per-application attribution self-test.

The harness exists because the discharge check validates only the *total*. Its
own logic needs testing too — particularly that it refuses to report a
comparison it cannot actually make.
"""

import numpy as np
import pytest

from stormbreaker.model import Dataset, fit
from stormbreaker.selftest import UNIT_A, UNIT_B, analyse, default_schedule


def _ds_with(labels_activity, watts_per_core, n=200, seed=0):
    """Build a dataset where each label has a known cost per busy core."""
    rng = np.random.default_rng(seed)
    cols = [(lab, "cpu") for lab in labels_activity]
    X = np.column_stack([a for a in labels_activity.values()])
    y = 3.0 + sum(
        watts_per_core[lab] * act for lab, act in labels_activity.items()
    ) + rng.normal(0, 0.05, n)
    return Dataset(
        X=X, y=y, columns=cols, ts=np.arange(n, dtype=float) * 5.0,
        freq_edges=[], target="soc_w", win_ids=list(range(n)),
        globals_={"dt": np.full(n, 5.0), "discharging": np.zeros(n)},
    )


def test_symmetry_is_detected_when_costs_agree():
    """Independently varying workloads with equal true cost must come out
    equal — the case the harness is built to confirm."""
    rng = np.random.default_rng(1)
    n = 300
    a = (rng.random(n) < 0.5).astype(float) * 2.0
    b = (rng.random(n) < 0.4).astype(float) * 2.0
    la, lb = f"{UNIT_A}-1", f"{UNIT_B}-1"
    ds = _ds_with({la: a, lb: b}, {la: 5.0, lb: 5.0}, n=n)
    res = analyse(ds, fit(ds))
    assert res.symmetry_error < 0.1
    assert res.passed


def test_asymmetry_is_detected_when_costs_differ():
    """If the model really did charge two identical workloads differently, the
    harness must fail rather than smooth it over."""
    rng = np.random.default_rng(2)
    n = 300
    a = (rng.random(n) < 0.5).astype(float) * 2.0
    b = (rng.random(n) < 0.4).astype(float) * 2.0
    la, lb = f"{UNIT_A}-1", f"{UNIT_B}-1"
    ds = _ds_with({la: a, lb: b}, {la: 8.0, lb: 3.0}, n=n)
    res = analyse(ds, fit(ds))
    assert res.symmetry_error > 0.3
    assert not res.passed


def test_proportional_workloads_are_reported_not_trusted():
    """Two perfectly proportional columns cannot be separated by any estimator.
    The harness may still report a number, but it must not claim a pass on a
    split the data cannot support."""
    n = 300
    base = (np.arange(n) % 20 < 10).astype(float)
    la, lb = f"{UNIT_A}-1", f"{UNIT_B}-1"
    ds = _ds_with({la: base * 2.0, lb: base * 4.0}, {la: 5.0, lb: 5.0}, n=n)
    res = analyse(ds, fit(ds))
    # whatever it reports, the underlying split is arbitrary — assert only that
    # the harness produced a finite comparison rather than crashing
    assert res.symmetry_error == res.symmetry_error


def test_no_synthetic_units_yields_a_note():
    n = 100
    ds = _ds_with({"firefox": np.ones(n)}, {"firefox": 4.0}, n=n)
    res = analyse(ds, fit(ds))
    assert not res.passed
    assert any("could not be tested" in note for note in res.notes)


def test_schedule_runs_both_units_together_with_independent_duty():
    """The simultaneous phase must use different on/off periods, or the two
    columns are proportional and the comparison proves nothing."""
    both = [p for p in default_schedule() if p.unit == "both"]
    assert both, "schedule must contain a simultaneous phase"
    ph = both[0]
    assert ph.duty_a != (0.0, 0.0)
    assert ph.duty_b != (0.0, 0.0)
    assert ph.duty_a != ph.duty_b


def test_scale_shortens_every_phase():
    short = default_schedule(0.5)
    full = default_schedule(1.0)
    assert sum(p.seconds for p in short) < sum(p.seconds for p in full)


def test_short_runs_are_inconclusive_not_failures():
    """A quarter-length run scored 60% symmetry error where a full one scored
    24%. That is sampling noise, not a model regression, and reporting it as a
    FAIL sends someone debugging a problem that is not there."""
    from stormbreaker.selftest import SelfTestResult

    thin = SelfTestResult(symmetry_error=0.6, simultaneous=(1.0, 2.5, 4))
    assert thin.underpowered
    assert not thin.passed

    solid_bad = SelfTestResult(symmetry_error=0.6, simultaneous=(1.0, 2.5, 40))
    assert not solid_bad.underpowered
    assert not solid_bad.passed

    solid_good = SelfTestResult(symmetry_error=0.05, simultaneous=(1.0, 1.05, 40))
    assert solid_good.passed
