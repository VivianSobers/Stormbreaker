"""Tests for inseparable-application detection.

If two applications are always busy together in the same proportion, the split
of power between them is decided by the regulariser rather than the data.
Detecting that is the difference between reporting a number and reporting a
number that means something.
"""

import numpy as np

from stormbreaker.model import Dataset, inseparable_groups


def _ds(activities):
    n = len(next(iter(activities.values())))
    return Dataset(
        X=np.column_stack(list(activities.values())),
        y=np.ones(n),
        columns=[(k, "cpu") for k in activities],
        ts=np.arange(n, dtype=float),
        freq_edges=[], target="soc_w", win_ids=list(range(n)),
        globals_={"dt": np.full(n, 5.0), "discharging": np.zeros(n)},
    )


def test_independent_applications_are_not_grouped():
    rng = np.random.default_rng(0)
    n = 400
    ds = _ds({
        "alpha": (rng.random(n) < 0.5).astype(float),
        "beta": (rng.random(n) < 0.4).astype(float),
        "gamma": (rng.random(n) < 0.3).astype(float),
    })
    assert inseparable_groups(ds) == []


def test_proportional_applications_are_grouped():
    """The canonical case: a browser and its GPU process, always busy together."""
    rng = np.random.default_rng(1)
    n = 400
    base = (rng.random(n) < 0.5).astype(float) * 1.5
    ds = _ds({"browser": base, "browser_gpu": base * 0.4,
              "unrelated": (rng.random(n) < 0.3).astype(float)})
    groups = inseparable_groups(ds)
    assert len(groups) == 1
    assert groups[0] == {"browser", "browser_gpu"}


def test_grouping_is_transitive():
    """Three mutually co-varying processes form one group, not three pairs."""
    rng = np.random.default_rng(2)
    n = 400
    base = (rng.random(n) < 0.5).astype(float) * 2.0
    ds = _ds({"a": base, "b": base * 0.5, "c": base * 1.7})
    groups = inseparable_groups(ds)
    assert len(groups) == 1
    assert groups[0] == {"a", "b", "c"}


def test_partial_correlation_is_not_grouped():
    """Applications that merely often coincide are still separable, and must
    not be lumped together — that would hide real information."""
    rng = np.random.default_rng(3)
    n = 600
    shared = (rng.random(n) < 0.4).astype(float)
    a = shared + (rng.random(n) < 0.3).astype(float)
    b = shared + (rng.random(n) < 0.3).astype(float)
    groups = inseparable_groups(_ds({"a": a, "b": b}))
    assert groups == []


def test_idle_applications_are_ignored():
    """A label with no activity correlates with nothing meaningfully."""
    rng = np.random.default_rng(4)
    n = 300
    ds = _ds({"busy": (rng.random(n) < 0.5).astype(float),
              "idle": np.zeros(n)})
    assert inseparable_groups(ds) == []
