"""Capability detection.

Nothing in Stormbreaker assumes a particular vendor or kernel config. Every
energy source and every feature source is probed once at startup, and the rest
of the system adapts to whatever is actually readable by *this* process. The
honest failure mode is a smaller feature vector, never a crash or a guess.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field


def _readable(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.read(1)
        return True
    except OSError:
        return False


def _read_text(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


@dataclass
class RaplDomain:
    path: str
    name: str
    max_range_uj: int


@dataclass
class Caps:
    # --- energy targets ---
    rapl: list[RaplDomain] = field(default_factory=list)
    """Powercap RAPL domains whose energy_uj we can actually read."""

    soc_power_path: str | None = None
    """hwmon power sensor reporting whole-package/SoC draw, in microwatts."""

    soc_power_label: str | None = None

    battery: str | None = None
    """sysfs power_supply directory for the battery, if present."""

    ac_online_path: str | None = None
    """Mains adapter 'online' flag. This, not the battery status string, is the
    authoritative answer to 'are we on battery right now'."""

    battery_charge_based: bool = False
    """True when the battery reports charge_now/current_now (uAh/uA) rather
    than energy_now/power_now (uWh/uW). Both are handled; they need different
    arithmetic and different unit assumptions."""

    # --- feature sources ---
    cgroup_root: str | None = None
    cgroup_v2: bool = False
    has_cpu_stat: bool = False
    has_io_stat: bool = False
    has_memory_stat: bool = False
    mbm_capable: bool = False
    """CPU supports hardware memory-bandwidth monitoring, but reading it needs
    resctrl mounted, which needs root. Recorded so the capability report can
    say the signal exists but is out of reach."""
    cpufreq_policies: list[str] = field(default_factory=list)
    has_avg_freq: bool = False
    has_time_in_state: bool = False
    gpu_busy_path: str | None = None
    profile_paths: list[str] = field(default_factory=list)
    """Files whose combined value identifies the current power regime.

    A power profile change rewrites the machine's cost structure: the same
    workload draws different power under 'performance' than under
    'power-saver'. Coefficients fitted under one do not describe the other, so
    the regime is recorded per window and treated as a hard boundary.
    """

    temp_path: str | None = None
    temp_label: str | None = None
    drm_fdinfo: bool = False
    max_freq_khz: int = 0
    ncpu: int = 1

    def energy_target(self) -> str:
        """Which signal we regress against. RAPL package is the most direct
        measurement of the domain our features describe; the SoC hwmon sensor
        is a close second and is usually readable without privileges."""
        for d in self.rapl:
            if d.name.startswith("package"):
                return "rapl:" + d.name
        if self.soc_power_path:
            return "soc"
        if self.battery:
            return "battery"
        return "none"

    def describe(self) -> str:
        lines = []
        if self.rapl:
            for d in self.rapl:
                lines.append(f"  rapl:{d.name:<12} {d.path}/energy_uj")
        else:
            lines.append("  rapl:            unavailable (root-only or absent)")
        if self.soc_power_path:
            lines.append(
                f"  soc power:       {self.soc_power_path} "
                f"({self.soc_power_label or 'unlabelled'})"
            )
        else:
            lines.append("  soc power:       unavailable")
        if self.battery:
            kind = "charge-based (I*V)" if self.battery_charge_based else "energy-based"
            lines.append(f"  battery:         {self.battery} [{kind}]")
            lines.append(f"  mains adapter:   {self.ac_online_path or 'unavailable'}")
        else:
            lines.append("  battery:         unavailable (desktop?)")
        lines.append(f"  cgroup v2:       {self.cgroup_root or 'unavailable'}")
        lines.append(f"  cpu.stat:        {self.has_cpu_stat}")
        lines.append(f"  io.stat:         {self.has_io_stat}")
        mem = "pgfault (memory.stat)" if self.has_memory_stat else "unavailable"
        if self.mbm_capable:
            mem += "; hardware MBM present but needs root (resctrl)"
        lines.append(f"  memory traffic:  {mem}")
        freq = "time_in_state" if self.has_time_in_state else (
            "cpuinfo_avg_freq" if self.has_avg_freq else "none")
        lines.append(f"  cpu frequency:   {freq} ({len(self.cpufreq_policies)} policies)")
        lines.append(f"  gpu busy:        {self.gpu_busy_path or 'unavailable'}")
        lines.append(
            f"  package temp:    {self.temp_path or 'unavailable'}"
            + (f" ({self.temp_label})" if self.temp_label else "")
        )
        lines.append(
            f"  power profile:   "
            + (", ".join(os.path.basename(p) for p in self.profile_paths)
               if self.profile_paths else "unavailable")
        )
        lines.append(f"  per-proc gpu:    {'drm fdinfo' if self.drm_fdinfo else 'unavailable'}")
        lines.append(f"  energy target:   {self.energy_target()}")
        return "\n".join(lines)


def _probe_rapl() -> list[RaplDomain]:
    out = []
    for path in sorted(glob.glob("/sys/class/powercap/*rapl*")):
        ej = os.path.join(path, "energy_uj")
        if not os.path.exists(ej) or not _readable(ej):
            continue
        name = _read_text(os.path.join(path, "name")) or os.path.basename(path)
        mx = _read_text(os.path.join(path, "max_energy_range_uj"))
        out.append(RaplDomain(path=path, name=name, max_range_uj=int(mx) if mx else 0))
    return out


def _probe_soc_power() -> tuple[str | None, str | None]:
    """Find a hwmon sensor that reports package/SoC power in microwatts.

    On AMD APUs the amdgpu driver exposes the SMU's package power tracking
    (labelled PPT) and, unlike RAPL, it is world-readable. That covers CPU +
    iGPU + fabric, which is precisely the domain our per-cgroup features drive.
    """
    preferred = ("amdgpu", "k10temp", "coretemp")
    candidates: list[tuple[int, str, str | None]] = []
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = _read_text(os.path.join(hw, "name")) or ""
        for attr in ("power1_average", "power1_input"):
            p = os.path.join(hw, attr)
            if not _readable(p):
                continue
            label = _read_text(os.path.join(hw, "power1_label"))
            rank = preferred.index(name) if name in preferred else len(preferred)
            candidates.append((rank, p, label))
            break
    if not candidates:
        return None, None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1], candidates[0][2]


def _probe_mains() -> str | None:
    for path in sorted(glob.glob("/sys/class/power_supply/*")):
        if (_read_text(os.path.join(path, "type")) or "") != "Mains":
            continue
        online = os.path.join(path, "online")
        if _readable(online):
            return online
    return None


def _probe_battery() -> tuple[str | None, bool]:
    for path in sorted(glob.glob("/sys/class/power_supply/*")):
        if (_read_text(os.path.join(path, "type")) or "") != "Battery":
            continue
        if _readable(os.path.join(path, "power_now")) and _readable(
            os.path.join(path, "energy_now")
        ):
            return path, False
        if _readable(os.path.join(path, "current_now")) and _readable(
            os.path.join(path, "charge_now")
        ):
            return path, True
    return None, False


def _probe_temp() -> tuple[str | None, str | None]:
    """Find a package temperature sensor, in millidegrees Celsius.

    Silicon leakage current rises with temperature, so a machine at 90 C draws
    measurably more than the same machine at 40 C doing identical work. The
    sensor is recorded now so that effect can be tested for later; it is not
    yet a model feature.
    """
    preferred = ("k10temp", "coretemp", "amdgpu", "acpitz")
    best: tuple[int, str, str | None] | None = None
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = _read_text(os.path.join(hw, "name")) or ""
        if name not in preferred:
            continue
        for temp in sorted(glob.glob(os.path.join(hw, "temp*_input"))):
            if not _readable(temp):
                continue
            label = _read_text(temp.replace("_input", "_label"))
            rank = preferred.index(name)
            if best is None or rank < best[0]:
                best = (rank, temp, label or name)
            break
    return (best[1], best[2]) if best else (None, None)


def _probe_profile_paths() -> list[str]:
    """Knobs that together define the current power regime.

    Read as opaque strings and compared for equality — the point is only to
    notice that the regime *changed*, not to interpret what it means. Each read
    costs ~5 us, so all of them together are far below the sampling noise.
    """
    candidates = [
        "/sys/firmware/acpi/platform_profile",
        "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference",
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
    ]
    return [p for p in candidates if _readable(p)]


def _probe_gpu_busy() -> str | None:
    for p in sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")):
        if _readable(p):
            return p
    return None


def _probe_drm_fdinfo() -> bool:
    """True if any process exposes per-client DRM engine busy time.

    We only need one positive example to know the kernel/driver supports it;
    the collector re-scans every window anyway.
    """
    checked = 0
    for pid_dir in glob.glob("/proc/[0-9]*/fdinfo"):
        try:
            fds = os.listdir(pid_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                with open(os.path.join(pid_dir, fd)) as fh:
                    blob = fh.read(4096)
            except OSError:
                continue
            if "drm-client-id" not in blob:
                continue
            if "drm-engine-" in blob:
                return True
            checked += 1
            if checked > 200:
                return False
    return False


def probe() -> Caps:
    c = Caps()
    c.ncpu = os.cpu_count() or 1
    c.rapl = _probe_rapl()
    c.soc_power_path, c.soc_power_label = _probe_soc_power()
    c.battery, c.battery_charge_based = _probe_battery()
    c.ac_online_path = _probe_mains()

    if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
        c.cgroup_root = "/sys/fs/cgroup"
        c.cgroup_v2 = True
        c.has_cpu_stat = os.path.exists("/sys/fs/cgroup/cpu.stat")
        c.has_io_stat = os.path.exists("/sys/fs/cgroup/io.stat")
        c.has_memory_stat = os.path.exists("/sys/fs/cgroup/user.slice/memory.stat")
    flags = _read_text("/proc/cpuinfo") or ""
    c.mbm_capable = "cqm_mbm_total" in flags

    c.cpufreq_policies = sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*"))
    if c.cpufreq_policies:
        p0 = c.cpufreq_policies[0]
        c.has_avg_freq = _readable(os.path.join(p0, "cpuinfo_avg_freq"))
        c.has_time_in_state = _readable(os.path.join(p0, "stats/time_in_state"))
        mx = _read_text(os.path.join(p0, "cpuinfo_max_freq"))
        c.max_freq_khz = int(mx) if mx else 0
    if not c.max_freq_khz:
        c.max_freq_khz = 4_000_000

    c.gpu_busy_path = _probe_gpu_busy()
    c.temp_path, c.temp_label = _probe_temp()
    c.profile_paths = _probe_profile_paths()
    c.drm_fdinfo = _probe_drm_fdinfo()
    return c
