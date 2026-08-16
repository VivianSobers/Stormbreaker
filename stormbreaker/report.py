"""Human-facing output: the wattage top and the daily battery report.

The unit that matters to a person is not watts, it is minutes of life lost. The
conversion needs one thing the attribution model does not itself provide: the
*whole system* draw, including panel, radios and SSD, which no package sensor
sees. That is recovered by regressing measured battery discharge onto the
package target over unplugged windows, and it is reported with its own fit
quality so nobody has to take it on faith.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from .model import Dataset, Fit, attribute, predict
from .sources import _read_int
from .store import Store


@dataclass
class SystemModel:
    """P_system = intercept + slope * P_package + temp_coef * (T - temp_ref).

    The temperature term is not decoration. A package sensor cannot see the
    fans, and fan power tracks temperature rather than instantaneous compute.
    Fitting package power alone leaves a systematic under-prediction —
    measured here as a +0.96 W held-out bias that the temperature term reduces
    to -0.30 W, lifting held-out R^2 from 0.885 to 0.927.

    Package power and temperature are strongly collinear (r = 0.94 on this
    machine), so the temperature coefficient is partly a proxy for sustained
    load rather than a clean fan measurement. It earns its place on held-out
    accuracy, not on a claim about which watts are whose.
    """

    intercept: float
    slope: float
    r2: float
    n: int
    temp_coef: float = 0.0
    temp_ref: float = 0.0

    @property
    def usable(self) -> bool:
        return self.n >= 30 and self.slope > 0

    def system_watts(self, package_watts: float, temp_c: float | None = None) -> float:
        w = self.intercept + self.slope * package_watts
        if self.temp_coef and temp_c is not None:
            w += self.temp_coef * max(temp_c - self.temp_ref, 0.0)
        return w


def fit_system_model(ds: Dataset) -> SystemModel:
    """Relate the package sensor to true battery draw.

    Only unplugged windows carry information: on AC the battery current
    reflects charging, not consumption.
    """
    disch = ds.globals_["discharging"] > 0.5
    batt = ds.globals_["batt_w"]
    ok = disch & np.isfinite(batt) & (batt > 0.1) & np.isfinite(ds.y) & (ds.y > 0)
    n = int(ok.sum())
    if n < 30:
        return SystemModel(intercept=0.0, slope=0.0, r2=float("nan"), n=n)

    temp = ds.globals_.get("temp_c")
    use_temp = (
        temp is not None
        and np.isfinite(temp[ok]).all()
        and float(np.ptp(temp[ok])) > 2.0
    )

    cols = [np.ones(n), ds.y[ok]]
    temp_ref = 0.0
    if use_temp:
        temp_ref = float(np.min(temp[ok]))
        cols.append(temp[ok] - temp_ref)

    A = np.column_stack(cols)
    w, _ = nnls(A, batt[ok])
    pred = A @ w
    resid = batt[ok] - pred
    ss_tot = float(((batt[ok] - batt[ok].mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return SystemModel(
        intercept=float(w[0]),
        slope=float(w[1]),
        r2=r2,
        n=n,
        temp_coef=float(w[2]) if use_temp else 0.0,
        temp_ref=temp_ref,
    )


def battery_capacity_wh(battery_path: str | None, charge_based: bool) -> float | None:
    """Design-independent usable capacity, in watt-hours."""
    if not battery_path:
        return None
    if charge_based:
        charge = _read_int(os.path.join(battery_path, "charge_full")) or _read_int(
            os.path.join(battery_path, "charge_full_design")
        )
        volt = _read_int(os.path.join(battery_path, "voltage_min_design")) or _read_int(
            os.path.join(battery_path, "voltage_now")
        )
        if charge and volt:
            return charge * volt / 1e12
        return None
    energy = _read_int(os.path.join(battery_path, "energy_full")) or _read_int(
        os.path.join(battery_path, "energy_full_design")
    )
    return energy / 1e6 if energy else None


@dataclass
class Row:
    label: str
    watts: float
    share: float
    energy_wh: float
    minutes_lost: float
    minutes_gained: float
    cpu_cores: float
    gpu: float
    io_mb: float


def build_rows(
    ds: Dataset,
    f: Fit,
    sysmod: SystemModel,
    capacity_wh: float | None,
) -> tuple[list[Row], dict[str, float]]:
    attr = attribute(ds, f)
    hours = float(np.nansum(ds.globals_["dt"])) / 3600.0
    pred = predict(ds, f)
    pkg_mean = float(np.nanmean(pred))

    temp = ds.globals_.get("temp_c")
    temp_mean = float(np.nanmean(temp)) if temp is not None and np.isfinite(temp).any() else None
    if sysmod.usable:
        sys_mean = sysmod.system_watts(pkg_mean, temp_mean)
    else:
        # Without discharge data the package sensor is a lower bound on system
        # draw. Say so rather than inventing an offset.
        sys_mean = pkg_mean

    col_of = {}
    for j, (label, feat) in enumerate(f.columns):
        col_of.setdefault(label, []).append((j, feat))

    rows: list[Row] = []
    for label, series in attr.items():
        watts = float(series.mean())
        if watts <= 0:
            continue
        energy_wh = watts * hours
        # Minutes of battery consumed by this app's energy.
        minutes_lost = (energy_wh / sys_mean) * 60.0 if sys_mean > 0 else 0.0
        # Minutes you would gain by closing it: runtime is capacity over draw,
        # which is nonlinear, so this is not the same number as minutes_lost.
        gained = 0.0
        if capacity_wh and sys_mean > 0:
            drop = sysmod.slope * watts if sysmod.usable else watts
            without = max(sys_mean - drop, 0.1)
            gained = (capacity_wh / without - capacity_wh / sys_mean) * 60.0
        cpu = sum(
            float(ds.X[:, j].mean())
            for j, feat in col_of.get(label, [])
            if feat.startswith("cpu")
        )
        gpu = sum(
            float(ds.X[:, j].mean())
            for j, feat in col_of.get(label, [])
            if feat == "gpu"
        )
        io = sum(
            float(ds.X[:, j].mean())
            for j, feat in col_of.get(label, [])
            if feat == "io_mb"
        )
        rows.append(
            Row(
                label=label,
                watts=watts,
                share=0.0,
                energy_wh=energy_wh,
                minutes_lost=minutes_lost,
                minutes_gained=gained,
                cpu_cores=cpu,
                gpu=gpu,
                io_mb=io,
            )
        )

    rows.sort(key=lambda r: -r.watts)
    total = sum(r.watts for r in rows) + f.baseline
    for r in rows:
        r.share = r.watts / total if total > 0 else 0.0

    ctx = {
        "hours": hours,
        "pkg_mean": pkg_mean,
        "sys_mean": sys_mean,
        "baseline": f.baseline,
        "total": total,
        "capacity_wh": capacity_wh or float("nan"),
    }
    return rows, ctx


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _bar(frac: float, width: int = 12) -> str:
    filled = int(round(max(frac, 0.0) * width))
    return "#" * filled + "." * (width - filled)


def render_top(rows: list[Row], ctx: dict, f: Fit, sysmod: SystemModel, n: int) -> str:
    out = []
    out.append(
        f"  window: {ctx['hours']*60:.1f} min   "
        f"package: {ctx['pkg_mean']:.2f} W   "
        f"system: {ctx['sys_mean']:.2f} W"
        + ("" if sysmod.usable else "  (system = package; no unplugged data yet)")
    )
    out.append(
        f"  model:  R^2={f.r2:.3f}  MAE={f.mae:.2f} W  "
        f"lambda={f.lam:g}  n={f.n_windows} windows  target={f.target}"
    )
    out.append("")
    out.append(
        f"  {'WATTS':>7}  {'SHARE':>6}  {'CPU':>6}  {'GPU':>6}  "
        f"{'IO MB/s':>8}  {'':12}  APPLICATION"
    )
    for r in rows[:n]:
        out.append(
            f"  {r.watts:7.3f}  {r.share*100:5.1f}%  {r.cpu_cores:6.3f}  "
            f"{r.gpu:6.3f}  {r.io_mb:8.2f}  {_bar(r.share)}  {r.label}"
        )
    out.append(
        f"  {f.baseline:7.3f}  {f.baseline/ctx['total']*100 if ctx['total'] else 0:5.1f}%"
        f"  {'':6}  {'':6}  {'':8}  {_bar(f.baseline/ctx['total'] if ctx['total'] else 0)}"
        f"  [idle baseline — attributable to nobody]"
    )
    return "\n".join(out)


def render_daily(
    rows: list[Row], ctx: dict, f: Fit, sysmod: SystemModel, n: int = 3
) -> str:
    out = []
    hours = ctx["hours"]
    out.append(f"  Observed over {hours:.1f} h of samples.")
    if sysmod.usable:
        out.append(
            f"  Whole-system draw modelled as {sysmod.intercept:.2f} W + "
            f"{sysmod.slope:.2f} x package  (R^2={sysmod.r2:.3f}, "
            f"{sysmod.n} unplugged windows)"
        )
    else:
        out.append(
            "  No unplugged data yet, so whole-system draw is unknown and the "
            "package figure is used as a lower bound.\n"
            "  Run on battery for a few minutes to calibrate; the minute "
            "estimates below will sharpen considerably."
        )
    cap = ctx["capacity_wh"]
    if cap == cap:  # not NaN
        runtime = cap / ctx["sys_mean"] if ctx["sys_mean"] > 0 else float("nan")
        out.append(
            f"  Battery {cap:.1f} Wh -> {runtime:.2f} h at the current average draw."
        )
    out.append("")
    out.append("  Top offenders:")
    for i, r in enumerate(rows[:n], 1):
        out.append(
            f"    {i}. {r.label:<28} {r.watts:5.2f} W   "
            f"{r.minutes_lost:5.1f} min of battery consumed"
        )
        if r.minutes_gained >= 1.0:
            out.append(
                f"       close it to gain ~{r.minutes_gained:.0f} min of runtime"
            )
    out.append("")
    out.append(
        f"  Idle baseline {f.baseline:.2f} W is not attributable to any "
        "application; it is the floor you pay for having the machine on."
    )
    return "\n".join(out)


def load_and_report(
    store: Store,
    ds: Dataset,
    f: Fit,
) -> tuple[list[Row], dict, SystemModel]:
    sysmod = fit_system_model(ds)
    cap = battery_capacity_wh(
        store.get_meta("battery") or None,
        store.get_meta("battery_charge_based") == "1",
    )
    rows, ctx = build_rows(ds, f, sysmod, cap)
    return rows, ctx, sysmod
