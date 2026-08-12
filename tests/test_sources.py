"""Tests for counter reading and differencing.

Process churn is the thing that breaks naive counter arithmetic, so most of
these tests are about processes and GPU clients appearing and disappearing
between two snapshots.
"""

import numpy as np
import pytest

from stormbreaker.caps import Caps
from stormbreaker.sources import CgCounters, Sampler, Snapshot, SubSample, pretty_unit
from stormbreaker.validate import find_discharge_segments
from stormbreaker.model import Dataset


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("NetworkManager.service", "NetworkManager"),
        ("app-flatpak-com.slack.Slack-2b41f9.scope", "com.slack.Slack"),
        ("app-code@284624f764254643ab8485dc69383c87.service", "code"),
        ("app-google-chrome@be27b1a40aeb4a898417b4db924e6039.service", "google-chrome"),
        ("dbus-:1.3-org.kde.kwalletd6@0.service", "org.kde.kwalletd6"),
        ("app-firefox-1234.scope", "firefox"),
        (r"foo\x2dbar.service", "foo-bar"),
        ("init.scope", "init"),
    ],
)
def test_pretty_unit(raw, expected):
    assert pretty_unit(raw) == expected


def _caps():
    c = Caps()
    c.ncpu = 8
    c.cgroup_root = None
    return c


def _snap(mono, cgroups, pid_ctxt=None, label_pids=None, gpu_ns=None, gpu_label=None):
    return Snapshot(
        mono=mono,
        wall=mono,
        cgroups=cgroups,
        pid_ctxt=pid_ctxt or {},
        label_pids=label_pids or {},
        gpu_ns=gpu_ns or {},
        gpu_label=gpu_label or {},
    )


def test_cpu_delta_is_a_rate():
    s = Sampler(_caps(), gpu_fdinfo=False)
    prev = _snap(0.0, {"app": CgCounters(cpu_usec=1_000_000)})
    cur = _snap(10.0, {"app": CgCounters(cpu_usec=6_000_000)})
    _dt, feats, _g = s.window(prev, cur, SubSample())
    # 5 core-seconds over a 10 s window = 0.5 busy cores
    assert feats["app"]["cpu"] == pytest.approx(0.5)


def test_new_process_contributes_its_whole_counter():
    """A pid created mid-window started its context-switch count at zero, so
    its current value is the delta. Treating it as if it had a prior value
    would silently drop the work it did."""
    s = Sampler(_caps(), gpu_fdinfo=False)
    prev = _snap(0.0, {"app": CgCounters(cpu_usec=0)}, pid_ctxt={}, label_pids={"app": []})
    cur = _snap(
        10.0,
        {"app": CgCounters(cpu_usec=1_000_000)},
        pid_ctxt={99: 5000},
        label_pids={"app": [99]},
    )
    _dt, feats, _g = s.window(prev, cur, SubSample())
    assert feats["app"]["ctxt_k"] == pytest.approx(5000 / 1e3 / 10.0)


def test_exited_process_does_not_produce_negative_delta():
    """An aggregate context-switch total falls when a process exits. Summing
    per-pid deltas instead keeps the result non-negative."""
    s = Sampler(_caps(), gpu_fdinfo=False)
    prev = _snap(
        0.0,
        {"app": CgCounters(cpu_usec=0)},
        pid_ctxt={1: 10_000, 2: 90_000},
        label_pids={"app": [1, 2]},
    )
    cur = _snap(
        10.0,
        {"app": CgCounters(cpu_usec=1_000_000)},
        pid_ctxt={1: 12_000},  # pid 2 exited
        label_pids={"app": [1]},
    )
    _dt, feats, _g = s.window(prev, cur, SubSample())
    assert feats["app"]["ctxt_k"] == pytest.approx(2000 / 1e3 / 10.0)
    assert feats["app"]["ctxt_k"] >= 0


def test_gpu_differenced_per_client():
    s = Sampler(_caps(), gpu_fdinfo=False)
    prev = _snap(
        0.0,
        {"game": CgCounters(cpu_usec=0)},
        label_pids={"game": [7]},
        gpu_ns={5: 1_000_000_000},
        gpu_label={5: "game"},
    )
    cur = _snap(
        10.0,
        {"game": CgCounters(cpu_usec=1)},
        label_pids={"game": [7]},
        gpu_ns={5: 4_000_000_000, 6: 2_000_000_000},  # client 6 is new
        gpu_label={5: "game", 6: "game"},
    )
    _dt, feats, _g = s.window(prev, cur, SubSample())
    # 3 s from the existing client + 2 s from the new one, over 10 s
    assert feats["game"]["gpu"] == pytest.approx(0.5)


def test_recreated_unit_is_dropped_not_spiked():
    """If a unit is destroyed and a new one takes its label, the counter
    restarts and the delta is meaningless. Better to drop the window for that
    label than to report a machine-melting spike."""
    s = Sampler(_caps(), gpu_fdinfo=False)
    prev = _snap(0.0, {"app": CgCounters(cpu_usec=0)})
    cur = _snap(1.0, {"app": CgCounters(cpu_usec=999_000_000)})
    _dt, feats, _g = s.window(prev, cur, SubSample())
    assert "app" not in feats


def test_subsample_is_a_running_mean():
    acc = SubSample()
    for w in (10.0, 20.0, 30.0):
        acc.add(w * 1e6, 0.0, 0.0, 0.0, 11.0, 0)
    assert acc.soc_uw == pytest.approx(20e6)
    assert acc.n == 3


def _discharge_ds(flags, ts=None, dt=5.0):
    n = len(flags)
    ts = np.arange(n, dtype=float) * dt if ts is None else np.asarray(ts, float)
    return Dataset(
        X=np.zeros((n, 1)),
        y=np.ones(n),
        columns=[("a", "cpu")],
        ts=ts,
        freq_edges=[],
        target="soc_w",
        win_ids=list(range(n)),
        globals_={
            "discharging": np.array(flags, float),
            "charge": np.full(n, 3_000_000.0),
            "dt": np.full(n, dt),
        },
    )


def test_discharge_segments_need_to_be_contiguous():
    ds = _discharge_ds([1] * 100 + [0] * 10 + [1] * 100)
    segs = find_discharge_segments(ds, min_windows=60)
    assert [(s.start, s.stop) for s in segs] == [(0, 100), (110, 210)]


def test_sampling_gap_splits_a_segment():
    """A suspend or a collector restart leaves a hole. Energy consumed while
    we were not sampling cannot be attributed, so the segment must break."""
    ts = list(np.arange(80) * 5.0) + list(np.arange(80) * 5.0 + 5000.0)
    ds = _discharge_ds([1] * 160, ts=ts)
    segs = find_discharge_segments(ds, min_windows=60)
    assert len(segs) == 2
    assert segs[0].stop == 80


def test_short_segments_are_ignored():
    ds = _discharge_ds([1] * 10 + [0] * 50)
    assert find_discharge_segments(ds, min_windows=60) == []
