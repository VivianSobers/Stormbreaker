"""The sampling loop.

Runs unprivileged wherever the capability probe allows it. The loop is written
to be cheap: the expensive part is walking /proc for context switches and DRM
clients, and its cost is measured every window so the user can see the observer
effect rather than having to guess at it.
"""

from __future__ import annotations

import json
import signal
import sys
import time

from .caps import Caps, probe
from .sources import Sampler, SubSample
from .store import DEFAULT_DB, Store


class Collector:
    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        window_s: float = 5.0,
        subsample_s: float = 0.5,
        gpu_fdinfo: bool = True,
        caps: Caps | None = None,
    ):
        self.caps = caps or probe()
        self.sampler = Sampler(self.caps, gpu_fdinfo=gpu_fdinfo)
        self.store = Store(db_path)
        self.window_s = window_s
        self.subsample_s = min(subsample_s, window_s / 2)
        self._stop = False
        self._record_caps()

    def _record_caps(self) -> None:
        s = self.store
        s.set_meta("ncpu", str(self.caps.ncpu))
        s.set_meta("max_freq_khz", str(self.caps.max_freq_khz))
        s.set_meta("energy_target", self.caps.energy_target())
        s.set_meta("battery", self.caps.battery or "")
        s.set_meta("battery_charge_based", str(int(self.caps.battery_charge_based)))
        s.set_meta(
            "caps_json",
            json.dumps(
                {
                    "rapl": [d.name for d in self.caps.rapl],
                    "soc_power": self.caps.soc_power_path,
                    "soc_label": self.caps.soc_power_label,
                    "gpu_busy": self.caps.gpu_busy_path,
                    "drm_fdinfo": self.caps.drm_fdinfo,
                    "avg_freq": self.caps.has_avg_freq,
                }
            ),
        )
        s.commit()

    def stop(self, *_a) -> None:
        self._stop = True

    def run(self, duration_s: float | None = None, verbose: bool = False) -> int:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        started = time.monotonic()
        prev = self.sampler.snapshot()
        n = 0
        commit_every = max(int(30.0 / self.window_s), 1)

        while not self._stop:
            if duration_s is not None and time.monotonic() - started >= duration_s:
                break

            subs = SubSample()
            deadline = time.monotonic() + self.window_s
            while not self._stop:
                self.sampler.subsample(subs)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(self.subsample_s, remaining))

            t0 = time.monotonic()
            cur = self.sampler.snapshot()
            scan_ms = (time.monotonic() - t0) * 1e3

            dt, feats, g = self.sampler.window(prev, cur, subs)
            self.store.add_window(g, feats)
            prev = cur
            n += 1
            if n % commit_every == 0:
                self.store.commit()

            if verbose:
                target = g.get("rapl_package-0_w") or g.get("soc_w") or 0.0
                top = sorted(feats.items(), key=lambda kv: -kv[1]["cpu"])[:3]
                busy = " ".join(f"{k}={v['cpu']:.2f}" for k, v in top)
                print(
                    f"[{n:5d}] dt={dt:.2f}s target={target:5.2f}W "
                    f"gpu={g['gpu_busy']*100:4.1f}% f={g['freq_ghz']:.2f}GHz "
                    f"cg={len(feats):3d} scan={scan_ms:5.1f}ms  {busy}",
                    flush=True,
                )

        self.store.commit()
        return n


def run_collect(
    db_path: str,
    window_s: float,
    duration_s: float | None,
    verbose: bool,
    gpu_fdinfo: bool = True,
) -> int:
    caps = probe()
    if caps.energy_target() == "none":
        print(
            "No energy source is readable on this machine. Stormbreaker needs at "
            "least one of: powercap RAPL, an hwmon package-power sensor, or a "
            "battery reporting power/current.",
            file=sys.stderr,
        )
        return 1
    c = Collector(db_path, window_s=window_s, gpu_fdinfo=gpu_fdinfo, caps=caps)
    print(f"stormbreaker collecting -> {db_path}")
    print(caps.describe())
    if not caps.rapl:
        print(
            "\nnote: RAPL energy counters are root-only on this kernel "
            "(CVE-2020-8694 mitigation).\n"
            "      Falling back to the hwmon package sensor, which needs no "
            "privileges.\n"
            "      For RAPL, run the collector as root or grant it "
            "CAP_DAC_READ_SEARCH."
        )
    print()
    n = c.run(duration_s=duration_s, verbose=verbose)
    c.store.close()
    print(f"\ncollected {n} windows -> {db_path}")
    return 0
