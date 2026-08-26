"""Tests for power-regime awareness.

A power profile change rewrites the machine's cost structure: the same workload
draws different power under 'performance' than under 'power-saver'. Fitting
across a change yields coefficients that describe neither regime while
appearing to fit both — which is worse than refusing, because it looks fine.
"""

import numpy as np

from stormbreaker.model import (
    Dataset,
    dominant_profile,
    filter_to_profile,
    profile_mix,
)
from stormbreaker.validate import find_discharge_segments

PERF = "performance|performance|powersave"
SAVER = "low-power|power|powersave"


def _ds(profiles, discharging=None, n=None):
    n = len(profiles)
    return Dataset(
        X=np.arange(n, dtype=float).reshape(-1, 1),
        y=np.linspace(5.0, 15.0, n),
        columns=[("a", "cpu")],
        ts=np.arange(n, dtype=float) * 5.0,
        freq_edges=[],
        target="soc_w",
        win_ids=list(range(n)),
        globals_={
            "discharging": np.ones(n) if discharging is None else np.array(discharging, float),
            "charge": np.linspace(3_200_000, 3_100_000, n),
            "dt": np.full(n, 5.0),
        },
        profiles=np.array(profiles, dtype=object),
    )


def test_profile_mix_counts_regimes():
    ds = _ds([PERF] * 30 + [SAVER] * 10)
    assert profile_mix(ds) == {PERF: 30, SAVER: 10}
    assert dominant_profile(ds) == PERF


def test_blank_profiles_are_ignored():
    """Windows recorded before profile tracking existed carry no regime, and
    must not be counted as one."""
    ds = _ds([""] * 20 + [PERF] * 5)
    assert profile_mix(ds) == {PERF: 5}


def test_no_profiles_at_all_is_not_an_error():
    ds = _ds([""] * 20)
    assert profile_mix(ds) == {}
    assert dominant_profile(ds) is None


def test_filter_keeps_only_one_regime():
    ds = _ds([PERF] * 30 + [SAVER] * 10)
    only = filter_to_profile(ds, PERF)
    assert len(only.y) == 30
    assert set(only.profiles) == {PERF}
    assert only.X.shape[0] == 30
    assert len(only.win_ids) == 30
    assert len(only.globals_["dt"]) == 30


def test_profile_change_splits_a_discharge_segment():
    """The core of the feature: a regime change is a hard boundary, exactly
    like a sampling gap. Training across it fits neither side."""
    ds = _ds([PERF] * 100 + [SAVER] * 100)
    segs = find_discharge_segments(ds, min_windows=60)
    assert [(s.start, s.stop) for s in segs] == [(0, 100), (100, 200)]


def test_unchanged_profile_leaves_one_segment():
    ds = _ds([PERF] * 200)
    segs = find_discharge_segments(ds, min_windows=60)
    assert [(s.start, s.stop) for s in segs] == [(0, 200)]


def test_missing_profiles_do_not_split():
    """Old data with no regime recorded must behave as it always did, rather
    than fragmenting into unusable pieces."""
    ds = _ds([""] * 200)
    segs = find_discharge_segments(ds, min_windows=60)
    assert [(s.start, s.stop) for s in segs] == [(0, 200)]


def test_a_brief_profile_flip_still_splits():
    """Even a short excursion is a real regime change; it is not smoothed."""
    ds = _ds([PERF] * 80 + [SAVER] * 5 + [PERF] * 80)
    segs = find_discharge_segments(ds, min_windows=60)
    starts = [(s.start, s.stop) for s in segs]
    assert (0, 80) in starts
    assert (85, 165) in starts
