"""Tests for the synthetic attribution benchmark.

The benchmark's own claims need checking: that it recovers what it planted for
identifiable applications, and that it *fails* in the specific ways theory
predicts for the ones that cannot be identified. A benchmark that quietly
passed the unidentifiable cases would be worse than none.
"""

import numpy as np
import pytest

from stormbreaker.bench import Scenario, make_scenario, run


def test_independent_applications_are_recovered():
    """Applications with their own duty cycles are identifiable, and the
    estimator should land close on a clean signal."""
    res = run(Scenario(n_windows=1200, noise_w=0.05, n_correlated=0, seed=3))
    assert res.independent_error < 0.06
    assert res.r2 > 0.99


def test_baseline_is_recovered():
    res = run(Scenario(n_windows=1200, noise_w=0.05, seed=4))
    assert res.baseline_fit == pytest.approx(res.baseline_true, rel=0.25)


def test_constant_daemon_is_not_recoverable():
    """A constant-load service is collinear with the baseline. The benchmark
    must show this failing, because the real system cannot fix it either."""
    res = run(Scenario(n_windows=1200, noise_w=0.05, constant_daemon=True, seed=5))
    assert res.daemon_error > 0.4


def test_co_varying_applications_are_worse_than_independent_ones():
    res = run(Scenario(n_windows=1200, noise_w=0.05, n_correlated=3, seed=6))
    assert res.correlated_error > res.independent_error


def test_co_varying_error_does_not_improve_with_a_better_sensor():
    """The key structural result: independent-app error is noise-limited and
    falls with a cleaner sensor, while co-varying error does not move, because
    it is set by identifiability rather than measurement."""
    clean = [run(Scenario(n_windows=900, noise_w=0.05, n_correlated=3, seed=s))
             for s in range(4)]
    noisy = [run(Scenario(n_windows=900, noise_w=1.0, n_correlated=3, seed=s))
             for s in range(4)]

    ind_clean = np.nanmean([r.independent_error for r in clean])
    ind_noisy = np.nanmean([r.independent_error for r in noisy])
    corr_clean = np.nanmean([r.correlated_error for r in clean])
    corr_noisy = np.nanmean([r.correlated_error for r in noisy])

    assert ind_clean < ind_noisy * 0.7, "independent error should be noise-limited"
    # co-varying error is structural: it barely responds to sensor quality
    assert corr_clean > corr_noisy * 0.4


def test_scenario_shape_is_as_declared():
    ds, truth, baseline = make_scenario(
        Scenario(n_windows=300, n_apps=4, n_correlated=2, constant_daemon=True)
    )
    assert ds.X.shape == (300, 7)  # 4 apps + 2 correlated + 1 daemon
    assert len(truth) == 7
    assert baseline > 0
    assert np.all(ds.X >= 0)
