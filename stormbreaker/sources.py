"""Counter and energy readers.

Two kinds of signal are collected:

*Counters* are monotonic (cgroup CPU microseconds, IO bytes, context switches,
DRM engine nanoseconds). They are read at each window boundary and differenced.
Wraparound and process churn are handled explicitly: aggregate context-switch
and GPU totals cannot be differenced directly, because a process exiting makes
the aggregate fall and a process starting makes it jump. Both are therefore
differenced per-pid and per-DRM-client, then summed.

*Instantaneous sensors* (SoC watts, GPU busy %, CPU frequency, battery current)
have no counter form, so they are sub-sampled several times per window and
averaged. Averaging matters: a 5 s window sampled only at its endpoints badly
misestimates a bursty workload.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from .caps import Caps

# ---------------------------------------------------------------------------
# small sysfs/procfs helpers
# ---------------------------------------------------------------------------


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    txt = _read(path)
    if txt is None:
        return None
    try:
        return int(txt.strip())
    except ValueError:
        return None


_ESCAPE = re.compile(r"\\x([0-9a-fA-F]{2})")


def _unescape(name: str) -> str:
    return _ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), name)


_APP_AT_HEX = re.compile(r"^app-(?:flatpak-|snap\.)?(.+?)@[0-9a-f]{6,}$")
_APP_SCOPE_HEX = re.compile(r"^app-(?:flatpak-|snap\.)?(.+?)-[0-9a-f]{6,}$")
_APP_SCOPE_NUM = re.compile(r"^app-(?:flatpak-|snap\.)?(.+?)-\d+$")
_DBUS_UNIT = re.compile(r"^dbus-:[\d.]+-(.+?)(?:@\d+)?$")
_UNIT_SUFFIXES = (".service", ".scope", ".socket", ".mount")


def pretty_unit(component: str) -> str:
    """Turn a systemd cgroup directory name into something a human recognises.

    ``app-flatpak-com.slack.Slack-2b41f9.scope`` -> ``com.slack.Slack``
    ``dbus-:1.3-org.kde.kwalletd6@0.service``    -> ``org.kde.kwalletd6``
    ``NetworkManager.service``                   -> ``NetworkManager``
    """
    name = _unescape(component)
    for suffix in (*_UNIT_SUFFIXES, ".slice"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for pattern in (_APP_AT_HEX, _APP_SCOPE_HEX, _APP_SCOPE_NUM, _DBUS_UNIT):
        m = pattern.match(name)
        if m:
            return m.group(1)
    return name


def _label_for(rel_path: str) -> str:
    """Nearest unit ancestor of a leaf cgroup, walking from the leaf upward."""
    parts = [p for p in rel_path.split("/") if p]
    for comp in reversed(parts):
        if comp.endswith(_UNIT_SUFFIXES):
            return pretty_unit(comp)
    return pretty_unit(parts[-1]) if parts else "root"


def find_leaf_cgroups(root: str) -> dict[str, list[str]]:
    """Map label -> leaf cgroup paths.

    Only leaves are used. cgroup v2 counters are hierarchical, so summing a
    parent together with its children would double-count; the set of leaves
    partitions the tree exactly once. Several leaves may share a label (a
    delegated unit that spawned sub-scopes), so they are summed under it.
    """
    groups: dict[str, list[str]] = {}
    for dirpath, dirnames, _ in os.walk(root):
        if dirnames:
            continue
        if not os.path.exists(os.path.join(dirpath, "cpu.stat")):
            continue
        rel = os.path.relpath(dirpath, root)
        label = "root" if rel == "." else _label_for(rel)
        groups.setdefault(label, []).append(dirpath)
    return groups


# ---------------------------------------------------------------------------
# per-cgroup counters
# ---------------------------------------------------------------------------


@dataclass
class CgCounters:
    cpu_usec: int = 0
    user_usec: int = 0
    system_usec: int = 0
    rbytes: int = 0
    wbytes: int = 0
    pgfault: int = 0
    nr_procs: int = 0


def _parse_cpu_stat(path: str, out: CgCounters) -> None:
    txt = _read(os.path.join(path, "cpu.stat"))
    if not txt:
        return
    for line in txt.splitlines():
        key, _, val = line.partition(" ")
        if key == "usage_usec":
            out.cpu_usec += int(val)
        elif key == "user_usec":
            out.user_usec += int(val)
        elif key == "system_usec":
            out.system_usec += int(val)


def _parse_memory_stat(path: str, out: CgCounters) -> None:
    """Page faults as a proxy for memory traffic.

    True memory bandwidth needs resctrl (root) or the uncore PMUs (root), so
    this is the best per-cgroup signal available unprivileged. A minor fault is
    raised on first touch of a page, so the count tracks allocation and
    streaming rather than steady-state bandwidth — a related quantity, not the
    same one. Whether it earns a column is decided by measurement, not by the
    plausibility of the story.
    """
    txt = _read(os.path.join(path, "memory.stat"))
    if not txt:
        return
    for line in txt.splitlines():
        if line.startswith("pgfault "):
            out.pgfault += int(line.split()[1])
            return


_DM_CACHE: dict[str, bool] = {}


def _is_device_mapper(devno: str) -> bool:
    """A device-mapper target mirrors the IO of the block device beneath it, so
    counting both double-counts every byte. Skip the dm layer."""
    hit = _DM_CACHE.get(devno)
    if hit is None:
        hit = os.path.exists(f"/sys/dev/block/{devno}/dm")
        _DM_CACHE[devno] = hit
    return hit


def _parse_io_stat(path: str, out: CgCounters) -> None:
    txt = _read(os.path.join(path, "io.stat"))
    if not txt:
        return
    for line in txt.splitlines():
        fields = line.split()
        if not fields or _is_device_mapper(fields[0]):
            continue
        for f in fields[1:]:
            key, _, val = f.partition("=")
            if key == "rbytes":
                out.rbytes += int(val)
            elif key == "wbytes":
                out.wbytes += int(val)


def _read_pids(path: str) -> list[int]:
    txt = _read(os.path.join(path, "cgroup.procs"))
    if not txt:
        return []
    return [int(p) for p in txt.split()]


_CTXT_KEYS = ("voluntary_ctxt_switches:", "nonvoluntary_ctxt_switches:")


def _pid_ctxt(pid: int) -> int:
    """Context switches for one process. /proc/PID/status is the only place
    these are exposed; both voluntary and involuntary counts are summed."""
    txt = _read(f"/proc/{pid}/status")
    if not txt:
        return 0
    total = 0
    for line in txt.splitlines():
        if line.startswith(_CTXT_KEYS):
            total += int(line.split()[-1])
    return total


_DRM_ENGINE = re.compile(r"^drm-engine-\w+:\s+(\d+)\s*ns", re.MULTILINE)
_DRM_CLIENT = re.compile(r"^drm-client-id:\s+(\d+)", re.MULTILINE)


def _pid_gpu_clients(pid: int) -> dict[int, int]:
    """DRM engine nanoseconds per client id for one process.

    A process usually holds several fds against the same DRM client and the
    kernel reports identical totals on each, so results are keyed by
    drm-client-id and recorded once. Client ids are globally unique and
    monotonic, which is also what lets us difference across fd churn.
    """
    out: dict[int, int] = {}
    try:
        fds = os.listdir(f"/proc/{pid}/fdinfo")
    except OSError:
        return out
    for fd in fds:
        blob = _read(f"/proc/{pid}/fdinfo/{fd}")
        if not blob or "drm-client-id" not in blob:
            continue
        cm = _DRM_CLIENT.search(blob)
        if not cm:
            continue
        busy = sum(int(m.group(1)) for m in _DRM_ENGINE.finditer(blob))
        if busy:
            out[int(cm.group(1))] = busy
    return out


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    mono: float
    wall: float
    rapl_uj: dict[str, int] = field(default_factory=dict)
    cgroups: dict[str, CgCounters] = field(default_factory=dict)
    label_pids: dict[str, list[int]] = field(default_factory=dict)
    pid_ctxt: dict[int, int] = field(default_factory=dict)
    gpu_ns: dict[int, int] = field(default_factory=dict)
    gpu_label: dict[int, str] = field(default_factory=dict)
    charge_uah: int | None = None
    battery_status: str = "Unknown"
    profile: str = ""


@dataclass
class SubSample:
    """Running mean of the instantaneous sensors over one window."""

    n: int = 0
    soc_uw: float = 0.0
    gpu_busy: float = 0.0
    freq_khz: float = 0.0
    batt_w: float = 0.0
    volt_v: float = 0.0
    temp_c: float = 0.0
    discharging: int = 0

    def add(self, soc_uw, gpu_busy, freq_khz, batt_w, volt_v, temp_c, discharging) -> None:
        self.n += 1
        k = self.n
        # incremental mean: allocation-free however long the window runs
        self.soc_uw += ((soc_uw or 0.0) - self.soc_uw) / k
        self.gpu_busy += ((gpu_busy or 0.0) - self.gpu_busy) / k
        self.freq_khz += ((freq_khz or 0.0) - self.freq_khz) / k
        self.batt_w += ((batt_w or 0.0) - self.batt_w) / k
        self.volt_v += ((volt_v or 0.0) - self.volt_v) / k
        self.temp_c += ((temp_c or 0.0) - self.temp_c) / k
        self.discharging += discharging


class Sampler:
    """Reads one machine. Holds the state needed to difference counters."""

    def __init__(
        self,
        caps: Caps,
        gpu_fdinfo: bool = True,
        gpu_rescan: int = 12,
        memory_stat: bool = False,
    ):
        self.caps = caps
        # Off by default: measured to add 54% to the per-window scan cost while
        # making held-out prediction slightly *worse*. See README.
        self.memory_stat = memory_stat and caps.has_memory_stat
        self.gpu_enabled = gpu_fdinfo and caps.drm_fdinfo
        self.gpu_rescan = gpu_rescan
        self._gpu_pids: set[int] = set()
        self._scans = 0
        self._freq_paths = (
            [os.path.join(p, "cpuinfo_avg_freq") for p in caps.cpufreq_policies]
            if caps.has_avg_freq
            else [os.path.join(p, "scaling_cur_freq") for p in caps.cpufreq_policies]
        )

    # -- instantaneous sensors ------------------------------------------------

    def read_soc_uw(self) -> float | None:
        if not self.caps.soc_power_path:
            return None
        v = _read_int(self.caps.soc_power_path)
        return float(v) if v is not None else None

    def read_profile(self) -> str:
        """An opaque identifier for the current power regime.

        The individual values are never interpreted — only compared against the
        previous window's, so that a change can be treated as a boundary.
        """
        parts = []
        for path in self.caps.profile_paths:
            v = _read(path)
            parts.append(v.strip() if v else "?")
        return "|".join(parts)

    def read_temp_c(self) -> float | None:
        if not self.caps.temp_path:
            return None
        v = _read_int(self.caps.temp_path)
        return v / 1000.0 if v is not None else None

    def read_gpu_busy(self) -> float | None:
        if not self.caps.gpu_busy_path:
            return None
        v = _read_int(self.caps.gpu_busy_path)
        return float(v) if v is not None else None

    def read_freq_khz(self) -> float | None:
        vals = [v for v in (_read_int(p) for p in self._freq_paths) if v]
        return sum(vals) / len(vals) if vals else None

    def on_battery(self, status: str) -> bool:
        """Whether the machine is currently running from the battery.

        The battery's own status string is not trustworthy for this. A pack at
        100% reports ``Full`` rather than ``Discharging`` immediately after the
        mains is pulled, and firmware flips between the two as it updates —
        which chops a genuine discharge run into fragments too short to
        validate against. The mains adapter's ``online`` flag has no such
        ambiguity, so it wins wherever it exists.
        """
        if self.caps.ac_online_path:
            v = _read_int(self.caps.ac_online_path)
            if v is not None:
                return v == 0
        return status == "Discharging"

    def read_battery(
        self,
    ) -> tuple[float | None, str, int | None, float | None, bool]:
        """Returns (watts, status, charge, volts, on_battery). Watts is positive
        while discharging and negative while charging. ``charge`` is in uAh on
        charge-based batteries and uWh on energy-based ones."""
        bat = self.caps.battery
        if not bat:
            return None, "Unknown", None, None, False
        status = (_read(os.path.join(bat, "status")) or "Unknown").strip()
        vol = _read_int(os.path.join(bat, "voltage_now"))
        volts = vol / 1e6 if vol else None
        if self.caps.battery_charge_based:
            cur = _read_int(os.path.join(bat, "current_now"))
            charge = _read_int(os.path.join(bat, "charge_now"))
            watts = (cur * vol) / 1e12 if (cur is not None and vol is not None) else None
        else:
            pw = _read_int(os.path.join(bat, "power_now"))
            charge = _read_int(os.path.join(bat, "energy_now"))
            watts = pw / 1e6 if pw is not None else None
        discharging = self.on_battery(status)
        if watts is not None and not discharging:
            # On mains the current reading describes charging, not consumption.
            watts = -abs(watts)
        return watts, status, charge, volts, discharging

    def subsample(self, acc: SubSample) -> None:
        watts, _status, _charge, volts, discharging = self.read_battery()
        acc.add(
            self.read_soc_uw(),
            self.read_gpu_busy(),
            self.read_freq_khz(),
            watts if watts is not None else 0.0,
            volts,
            self.read_temp_c(),
            1 if discharging else 0,
        )

    # -- counter snapshot -----------------------------------------------------

    def _gpu_scan_pids(self, all_pids: list[int]) -> list[int]:
        """Full fdinfo scans are the most expensive thing we do, so they run
        only every ``gpu_rescan`` windows. In between, only processes already
        known to hold a DRM client are read. A newly launched GPU application
        is therefore picked up within one rescan interval."""
        self._scans += 1
        if self._scans % self.gpu_rescan == 1:
            return all_pids
        return [p for p in all_pids if p in self._gpu_pids]

    def snapshot(self) -> Snapshot:
        snap = Snapshot(mono=time.monotonic(), wall=time.time())

        for dom in self.caps.rapl:
            v = _read_int(os.path.join(dom.path, "energy_uj"))
            if v is not None:
                snap.rapl_uj[dom.name] = v

        _, status, charge, _, _ = self.read_battery()
        snap.battery_status = status
        snap.charge_uah = charge
        snap.profile = self.read_profile()

        if not self.caps.cgroup_root:
            return snap

        groups = find_leaf_cgroups(self.caps.cgroup_root)
        all_pids: list[tuple[int, str]] = []

        for label, paths in groups.items():
            agg = CgCounters()
            pids: list[int] = []
            for path in paths:
                _parse_cpu_stat(path, agg)
                if self.caps.has_io_stat:
                    _parse_io_stat(path, agg)
                if self.memory_stat:
                    _parse_memory_stat(path, agg)
                pids.extend(_read_pids(path))
            agg.nr_procs = len(pids)
            if agg.cpu_usec or pids:
                snap.cgroups[label] = agg
                snap.label_pids[label] = pids
                all_pids.extend((p, label) for p in pids)

        for pid, _label in all_pids:
            c = _pid_ctxt(pid)
            if c:
                snap.pid_ctxt[pid] = c

        if self.gpu_enabled:
            owner = dict(all_pids)
            found: set[int] = set()
            for pid in self._gpu_scan_pids([p for p, _ in all_pids]):
                clients = _pid_gpu_clients(pid)
                if not clients:
                    continue
                found.add(pid)
                for cid, ns in clients.items():
                    # Highest total wins if two pids share a client id; they
                    # report the same underlying counter.
                    if ns >= snap.gpu_ns.get(cid, -1):
                        snap.gpu_ns[cid] = ns
                        snap.gpu_label[cid] = owner.get(pid, "root")
            if self._scans % self.gpu_rescan == 1:
                self._gpu_pids = found
            else:
                self._gpu_pids |= found

        return snap

    # -- window deltas --------------------------------------------------------

    def window(
        self, prev: Snapshot, cur: Snapshot, subs: SubSample
    ) -> tuple[float, dict[str, dict[str, float]], dict[str, float]]:
        """Difference two snapshots into per-second rates.

        Returns (duration_s, per-cgroup features, global measurements).
        """
        dt = max(cur.mono - prev.mono, 1e-6)

        globals_: dict[str, float] = {"dt": dt}
        for name, uj in cur.rapl_uj.items():
            before = prev.rapl_uj.get(name)
            if before is None:
                continue
            delta = uj - before
            if delta < 0:  # counter wrapped
                rng = next((d.max_range_uj for d in self.caps.rapl if d.name == name), 0)
                delta += rng
            globals_[f"rapl_{name}_w"] = delta / 1e6 / dt

        globals_["soc_w"] = subs.soc_uw / 1e6
        globals_["gpu_busy"] = subs.gpu_busy / 100.0
        globals_["freq_ghz"] = subs.freq_khz / 1e6
        globals_["batt_w"] = subs.batt_w
        globals_["discharging"] = 1.0 if subs.discharging > subs.n / 2 else 0.0
        globals_["charge"] = float(cur.charge_uah or 0)
        globals_["volt_v"] = subs.volt_v
        globals_["temp_c"] = subs.temp_c
        globals_["profile"] = cur.profile

        # Context switches, differenced per pid. A pid absent from the previous
        # snapshot started during this window, and its counter began at zero, so
        # its current value *is* the delta.
        ctxt_by_label: dict[str, int] = {}
        for label, pids in cur.label_pids.items():
            total = 0
            for pid in pids:
                now = cur.pid_ctxt.get(pid)
                if now is None:
                    continue
                total += max(now - prev.pid_ctxt.get(pid, 0), 0)
            ctxt_by_label[label] = total

        # GPU engine time, differenced per DRM client id for the same reason.
        gpu_by_label: dict[str, int] = {}
        for cid, ns in cur.gpu_ns.items():
            delta = max(ns - prev.gpu_ns.get(cid, 0), 0)
            if delta:
                label = cur.gpu_label.get(cid, "root")
                gpu_by_label[label] = gpu_by_label.get(label, 0) + delta

        feats: dict[str, dict[str, float]] = {}
        for label, cnt in cur.cgroups.items():
            before = prev.cgroups.get(label) or CgCounters()
            cpu_s = max(cnt.cpu_usec - before.cpu_usec, 0) / 1e6
            if cpu_s > dt * self.caps.ncpu * 1.5:
                # a unit was destroyed and recreated under the same label;
                # the delta is meaningless, so drop the window for this label
                continue
            rb = max(cnt.rbytes - before.rbytes, 0)
            wb = max(cnt.wbytes - before.wbytes, 0)
            pgf = max(cnt.pgfault - before.pgfault, 0)
            row = {
                "cpu": cpu_s / dt,  # busy cores
                "io_mb": (rb + wb) / 1e6 / dt,  # MB/s
                "ctxt_k": ctxt_by_label.get(label, 0) / 1e3 / dt,
                "gpu": gpu_by_label.get(label, 0) / 1e9 / dt,  # busy GPU-s/s
                "pgflt_k": pgf / 1e3 / dt,  # thousand page faults/s
                "nr_procs": float(cnt.nr_procs),
            }
            if row["cpu"] or row["io_mb"] or row["gpu"] or row["ctxt_k"] or row["pgflt_k"]:
                feats[label] = row
        return dt, feats, globals_
