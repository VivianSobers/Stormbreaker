"""Tests for the package-to-system model.

The attribution model explains the *package*. Turning that into "minutes of
battery" needs whole-system draw, which includes the panel, the radios and —
the part that took a real measurement to notice — the fans. Fan power tracks
temperature rather than instantaneous compute, and no package sensor sees it.
"""

import numpy as np
import pytest

from stormbreaker.model import Dataset
from stormbreaker.report import SystemModel, fit_system_model


def _ds(package, batt, temp, n=None):
    n = len(package)
    return Dataset(
        X=np.zeros((n, 1)),
        y=np.asarray(package, float),
        columns=[("a", "cpu")],
        ts=np.arange(n, dtype=float) * 5.0,
        freq_edges=[],
        target="soc_w",
        win_ids=list(range(n)),
        globals_={
            "discharging": np.ones(n),
            "batt_w": np.asarray(batt, float),
            "temp_c": np.asarray(temp, float),
            "dt": np.full(n, 5.0),
        },
    )


def test_constant_temperature_contributes_nothing():
    """A machine whose temperature never moves gives the term no information,
    and it must be left out rather than fitted to noise."""
    rng = np.random.default_rng(0)
    n = 200
    pkg = rng.random(n) * 20 + 3
    batt = 5.0 + 1.3 * pkg + rng.normal(0, 0.1, n)
    f = fit_system_model(_ds(pkg, batt, np.full(n, 45.0)))
    assert f.temp_coef == 0.0
    assert f.usable
    assert f.slope == pytest.approx(1.3, rel=0.1)


def test_temperature_term_is_recovered_when_it_matters():
    """Fan power: real watts drawn from the battery that the package sensor
    cannot see, rising with temperature."""
    rng = np.random.default_rng(1)
    n = 300
    pkg = rng.random(n) * 25 + 3
    temp = 40.0 + rng.random(n) * 40.0
    batt = 4.0 + 1.2 * pkg + 0.25 * (temp - 40.0) + rng.normal(0, 0.1, n)

    f = fit_system_model(_ds(pkg, batt, temp))
    assert f.temp_coef > 0.1
    assert f.temp_ref == pytest.approx(temp.min(), abs=1e-6)
    assert f.r2 > 0.95


def test_system_watts_applies_the_temperature_term():
    m = SystemModel(intercept=5.0, slope=1.0, r2=0.9, n=100, temp_coef=0.2, temp_ref=40.0)
    assert m.system_watts(10.0, 40.0) == pytest.approx(15.0)
    assert m.system_watts(10.0, 60.0) == pytest.approx(19.0)
    # below the reference the term must not go negative
    assert m.system_watts(10.0, 30.0) == pytest.approx(15.0)


def test_system_watts_without_a_temperature_reading():
    """Callers that have no temperature must still get the package-only answer
    rather than a crash or a silently wrong number."""
    m = SystemModel(intercept=5.0, slope=1.0, r2=0.9, n=100, temp_coef=0.2, temp_ref=40.0)
    assert m.system_watts(10.0) == pytest.approx(15.0)


def test_too_few_discharge_windows_is_not_usable():
    rng = np.random.default_rng(2)
    n = 10
    pkg = rng.random(n) * 10
    f = fit_system_model(_ds(pkg, 5 + pkg, np.full(n, 50.0)))
    assert not f.usable


def test_system_model_never_goes_negative():
    """Non-negativity again: a system draw that falls with package power would
    be unphysical, and would make 'close this app to gain time' advice wrong."""
    rng = np.random.default_rng(3)
    n = 200
    pkg = rng.random(n) * 20 + 3
    batt = 12.0 - 0.5 * pkg + rng.normal(0, 0.1, n)  # anti-correlated on purpose
    f = fit_system_model(_ds(pkg, batt, np.full(n, 45.0)))
    assert f.slope >= 0.0
    assert f.intercept >= 0.0
