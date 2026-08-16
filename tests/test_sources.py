"""Tests for counter reading and differencing.

Process churn is the thing that breaks naive counter arithmetic, so most of
these tests are about processes and GPU clients appearing and disappearing
between two snapshots.
"""

import numpy as np
import pytest

from stormbreaker.caps import Caps
from stormbreaker.sources import CgCounters, Sampler, Snapshot, SubSample, pretty_unit
from stormbreaker.validate import (
    discharge_readiness,
    find_discharge_segments,
    validate_discharge,
)
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
    """Endpoint sampling misestimates a bursty window, so instantaneous sensors
    are averaged over sub-samples rather than read twice."""
    acc = SubSample()
    for w, t in ((10.0, 40.0), (20.0, 50.0), (30.0, 60.0)):
        acc.add(w * 1e6, 0.0, 0.0, 0.0, 11.0, t, 0)
    assert acc.soc_uw == pytest.approx(20e6)
    assert acc.temp_c == pytest.approx(50.0)
    assert acc.n == 3


def test_subsample_tolerates_missing_sensors():
    """A machine without a temperature or battery sensor must still sample."""
    acc = SubSample()
    acc.add(5e6, None, None, None, None, None, 0)
    assert acc.soc_uw == pytest.approx(5e6)
    assert acc.temp_c == 0.0


def _sampler_with_ac(tmp_path, online: str | None, battery=True):
    c = Caps()
    c.battery = "/nonexistent" if battery else None
    if online is not None:
        p = tmp_path / "online"
        p.write_text(online)
        c.ac_online_path = str(p)
    return Sampler(c, gpu_fdinfo=False)


def test_full_battery_on_mains_power_still_counts_as_discharging(tmp_path):
    """Regression test for a bug that blocked discharge validation entirely.

    A pack at 100% reports ``Full`` rather than ``Discharging`` for the first
    minutes after unplugging, and firmware alternates between the two. Trusting
    the status string chopped a real discharge run into 1-2 window fragments,
    so no segment ever reached the length needed to validate.
    """
    s = _sampler_with_ac(tmp_path, "0")
    assert s.on_battery("Full") is True
    assert s.on_battery("Discharging") is True
    assert s.on_battery("Unknown") is True


def test_mains_present_overrides_battery_status(tmp_path):
    s = _sampler_with_ac(tmp_path, "1")
    assert s.on_battery("Discharging") is False
    assert s.on_battery("Full") is False


def test_falls_back_to_status_without_a_mains_adapter(tmp_path):
    """A desktop or an unusual firmware may expose no Mains supply at all."""
    s = _sampler_with_ac(tmp_path, None)
    assert s.on_battery("Discharging") is True
    assert s.on_battery("Full") is False


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


def _charge_ds(flags, charge):
    ds = _discharge_ds(flags)
    ds.globals_["charge"] = np.asarray(charge, float)
    ds.globals_["batt_w"] = np.full(len(flags), 10.0)
    return ds


def test_readiness_reports_no_unplugged_windows():
    ds = _charge_ds([0] * 100, [3_000_000.0] * 100)
    assert "no unplugged windows" in discharge_readiness(ds)


def test_readiness_reports_fragmented_run():
    """Alternating flags — the exact symptom of trusting the battery status
    string on a full pack — must be named as a fragmentation problem."""
    flags = [1, 1, 0] * 40
    ds = _charge_ds(flags, [3_000_000.0] * len(flags))
    msg = discharge_readiness(ds)
    assert "consecutive" in msg


def test_readiness_reports_an_unmoving_fuel_gauge():
    """A long clean run against a gauge stuck on one value is not validatable,
    and must say so rather than divide by a near-zero energy drop."""
    n = 120
    ds = _charge_ds([1] * n, [3_214_000.0] * n)
    msg = discharge_readiness(ds)
    assert "distinct value" in msg
    assert validate_discharge(ds) is None


def test_readiness_passes_once_the_gauge_moves():
    n = 120
    charge = np.linspace(3_214_000.0, 3_100_000.0, n).round(-4)
    ds = _charge_ds([1] * n, charge)
    assert discharge_readiness(ds) == "ready"


def test_charge_to_energy_is_monotonic_under_load_swings():
    """Regression test: converting charge with the instantaneous terminal
    voltage made "energy remaining" *rise* during a discharge.

    Terminal voltage is open-circuit minus I*R, so it sags under load and
    recovers when load drops. Across a 3.2 Ah pack a 1 V swing is a 3.2 Wh
    swing — far larger than the energy actually drawn over a few minutes.
    """
    from stormbreaker.validate import _wh_from_charge

    n = 60
    charge = np.linspace(3_200_000, 3_150_000, n)  # strictly draining
    # voltage collapses mid-segment under a load burst, then recovers
    volts = np.full(n, 12.6)
    volts[20:40] = 11.4

    wh = _wh_from_charge(charge, volts)
    assert np.all(np.diff(wh) < 0), "energy must fall while charge falls"
    assert wh[0] > wh[-1]


def test_charge_to_energy_needs_a_voltage():
    from stormbreaker.validate import _wh_from_charge

    with pytest.raises(ValueError, match="no pack voltage"):
        _wh_from_charge(np.array([3_200_000.0]), np.array([np.nan]))


def test_gauge_plateau_is_trimmed():
    """A pack unplugged at full pins its gauge at the maximum for minutes, then
    catches up in one jump. Scoring across that boundary compares the model
    against an instrument that was not yet reporting."""
    from stormbreaker.validate import Segment, trim_gauge_plateau

    n = 100
    charge = np.concatenate([np.full(30, 3_214_000.0),
                             np.linspace(3_213_000, 3_150_000, 70)])
    ds = _charge_ds([1] * n, charge)
    trimmed = trim_gauge_plateau(ds, Segment(0, n))
    assert trimmed.start == 30
    assert trimmed.stop == n


def test_trimming_a_moving_gauge_changes_nothing():
    from stormbreaker.validate import Segment, trim_gauge_plateau

    n = 50
    ds = _charge_ds([1] * n, np.linspace(3_200_000, 3_150_000, n))
    trimmed = trim_gauge_plateau(ds, Segment(0, n))
    assert trimmed.start == 1  # only the first reading itself
    assert len(trimmed) >= n - 1
