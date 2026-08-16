"""Validation.

Two checks, in increasing order of how much they prove.

**Held-out target error** works any time, plugged in or not. Fit on the first
part of the record, predict the package sensor on the last part. It shows the
model generalises across time rather than memorising the window it was fitted
on.

**Discharge-curve tracking** is the real one, and the reason the project is
worth building. Take an unplugged session, fit on its first half, then predict
the second half's battery trajectory *without ever looking at it*, and compare
against what the fuel gauge actually reported. The gauge's charge reading is an
integral measurement, accumulated independently of every counter we regress on,
so agreement is not something the fit can manufacture. If the predicted curve
tracks the real one, the attribution is doing its job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Dataset, fit, predict
from .report import SystemModel, fit_system_model


@dataclass
class Segment:
    start: int
    stop: int  # exclusive

    def __len__(self) -> int:
        return self.stop - self.start


def find_discharge_segments(ds: Dataset, min_windows: int = 60) -> list[Segment]:
    """Contiguous runs of unplugged windows with no sampling gap.

    A gap (suspend, collector restart) breaks a segment: energy consumed while
    we were not looking cannot be attributed, and silently bridging the gap
    would make the model look wrong for a reason that is not its fault.

    A **power profile change** breaks a segment for the same reason. Switching
    between performance and power-saver rewrites the machine's cost structure,
    so coefficients fitted before the switch do not describe the windows after
    it. Training across that boundary produces a model that describes neither
    regime while appearing to fit both.
    """
    disch = ds.globals_["discharging"] > 0.5
    charge = ds.globals_["charge"]
    ts = ds.ts
    dt = ds.globals_["dt"]
    profiles = ds.profiles

    segs: list[Segment] = []
    start = None
    for i in range(len(disch)):
        gap = i > 0 and (ts[i] - ts[i - 1]) > max(3.0 * dt[i], 15.0)
        if (
            i > 0
            and profiles is not None
            and profiles[i]
            and profiles[i - 1]
            and profiles[i] != profiles[i - 1]
        ):
            gap = True
        valid = disch[i] and np.isfinite(charge[i]) and charge[i] > 0
        if valid and not gap:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_windows:
                segs.append(Segment(start, i))
            start = i if valid else None
    if start is not None and len(disch) - start >= min_windows:
        segs.append(Segment(start, len(disch)))
    return segs


def _wh_from_charge(charge_uah: np.ndarray, volts: np.ndarray) -> np.ndarray:
    """uAh -> Wh, using a single nominal voltage for the whole segment.

    Converting with the *instantaneous* terminal voltage seems more precise and
    is badly wrong. Terminal voltage is open-circuit voltage minus I*R, so it
    sags the moment load rises and recovers when load drops. Across a 3.2 Ah
    pack a 1 V swing is a 3.2 Wh swing — larger than the energy actually drawn
    over several minutes, and enough to make the converted "energy remaining"
    curve rise while the battery is definitely discharging.

    The gauge's charge reading is a coulomb count, so a constant conversion
    keeps it monotonic and preserves the quantity being validated. The median
    is used rather than the mean because load spikes drag the mean down.

    The real open-circuit voltage does fall as the pack empties, but over a
    validation window of minutes that drift is far smaller than the IR noise
    this avoids.
    """
    ok = np.isfinite(volts) & (volts > 1.0)
    if not ok.any():
        raise ValueError("no pack voltage recorded; cannot convert charge to energy")
    nominal = float(np.median(volts[ok]))
    return charge_uah * nominal / 1e6


@dataclass
class DischargeResult:
    n_train: int
    n_test: int
    hours: float
    predicted_wh: np.ndarray
    measured_wh: np.ndarray
    ts: np.ndarray
    mae_wh: float
    final_error_wh: float
    final_error_pct: float
    predicted_runtime_h: float
    measured_runtime_h: float
    sysmod: SystemModel


def trim_gauge_plateau(ds: Dataset, seg: Segment) -> Segment:
    """Drop the leading windows where the fuel gauge has not started moving.

    A pack unplugged at full sits pinned at its maximum reading for minutes,
    then catches up in one jump. Total energy over a long segment still comes
    out right, but *within* the segment the reported rate is badly distorted —
    measured here as a held-out span appearing to drain at 25 W while the
    current sensor read 12.5 W. Fitting or scoring across that boundary
    compares a model against an instrument that was not yet reporting.
    """
    charge = ds.globals_["charge"]
    start = seg.start
    first = charge[start]
    while start < seg.stop - 1 and charge[start] >= first:
        start += 1
    return Segment(start, seg.stop)


MIN_GAUGE_STEPS = 4
"""Distinct fuel-gauge readings needed across the held-out span.

A charge gauge reports in coarse quantised steps — 1% of a 38 Wh pack is
0.38 Wh here. Comparing a predicted curve against a measurement that took two
or three values is not a validation, it is a coincidence, and dividing by the
resulting near-zero energy drop yields impressive-looking nonsense.
"""


def discharge_readiness(ds: Dataset, charge_based: bool = True) -> str:
    """Explain why a discharge validation cannot run yet."""
    segs = find_discharge_segments(ds)
    if not segs:
        disch = int((ds.globals_["discharging"] > 0.5).sum())
        if disch == 0:
            return "no unplugged windows recorded yet"
        return (
            f"{disch} unplugged windows, but no run of {60} consecutive ones "
            "(a gap, a suspend, or a collector restart splits a segment)"
        )
    seg = max(segs, key=len)
    idx = np.arange(seg.start, seg.stop)
    steps = len(np.unique(ds.globals_["charge"][idx]))
    if steps < MIN_GAUGE_STEPS:
        drawn = float(np.nansum(ds.globals_["batt_w"][idx] * ds.globals_["dt"][idx]))
        return (
            f"longest unplugged run is {len(seg)} windows "
            f"({len(seg)*5/60:.0f} min), but the fuel gauge has only reported "
            f"{steps} distinct value(s) over it — roughly {drawn/3600:.2f} Wh drawn "
            "with no resolvable change. Draw more power or run longer"
        )
    return "ready"


def validate_discharge(
    ds: Dataset,
    charge_based: bool = True,
    holdout: float = 0.5,
    segment: Segment | None = None,
) -> DischargeResult | None:
    """Fit on the first part of an unplugged session, predict the rest."""
    segs = find_discharge_segments(ds)
    if not segs:
        return None
    seg = segment or max(segs, key=len)
    seg = trim_gauge_plateau(ds, seg)
    if len(seg) < 40:
        return None
    if len(np.unique(ds.globals_["charge"][seg.start : seg.stop])) < MIN_GAUGE_STEPS:
        return None
    cut = seg.start + max(int(len(seg) * (1 - holdout)), 30)
    if cut >= seg.stop - 10:
        return None

    train = np.arange(seg.start, cut)
    test = np.arange(cut, seg.stop)

    sub_train = _subset(ds, train)
    f = fit(sub_train)
    sysmod = fit_system_model(sub_train)
    if not sysmod.usable:
        return None

    sub_test = _subset(ds, test)
    pkg_pred = predict(sub_test, f)
    sys_pred = sysmod.intercept + sysmod.slope * pkg_pred
    if sysmod.temp_coef:
        t = ds.globals_["temp_c"][test]
        sys_pred = sys_pred + sysmod.temp_coef * np.clip(t - sysmod.temp_ref, 0, None)

    dt = ds.globals_["dt"][test]
    volts = ds.globals_["volt_v"][test]
    if charge_based:
        measured = _wh_from_charge(ds.globals_["charge"][test], volts)
    else:
        measured = ds.globals_["charge"][test] / 1e6

    # Integrate predicted draw forward from the true starting energy.
    consumed = np.cumsum(sys_pred * dt / 3600.0)
    predicted = measured[0] - consumed

    err = predicted - measured
    hours = float(dt.sum()) / 3600.0
    mean_sys = float(np.mean(sys_pred))
    measured_rate = (measured[0] - measured[-1]) / hours if hours > 0 else float("nan")

    return DischargeResult(
        n_train=len(train),
        n_test=len(test),
        hours=hours,
        predicted_wh=predicted,
        measured_wh=measured,
        ts=ds.ts[test],
        mae_wh=float(np.abs(err).mean()),
        final_error_wh=float(err[-1]),
        final_error_pct=float(
            err[-1] / max(measured[0] - measured[-1], 1e-6) * 100.0
        ),
        predicted_runtime_h=float(measured[0] / mean_sys) if mean_sys > 0 else float("nan"),
        measured_runtime_h=float(measured[0] / measured_rate)
        if measured_rate > 0
        else float("nan"),
        sysmod=sysmod,
    )


def _subset(ds: Dataset, idx: np.ndarray) -> Dataset:
    return Dataset(
        X=ds.X[idx],
        y=ds.y[idx],
        columns=ds.columns,
        ts=ds.ts[idx],
        freq_edges=ds.freq_edges,
        target=ds.target,
        win_ids=[ds.win_ids[i] for i in idx],
        globals_={k: v[idx] for k, v in ds.globals_.items()},
        profiles=None if ds.profiles is None else ds.profiles[idx],
    )


@dataclass
class HoldoutResult:
    n_train: int
    n_test: int
    r2: float
    mae: float
    rmse: float
    mean_measured: float
    mean_predicted: float
    naive_mae: float


def validate_holdout(ds: Dataset, holdout: float = 0.3) -> HoldoutResult:
    """Chronological train/test split on the package target."""
    n = len(ds.y)
    cut = max(int(n * (1 - holdout)), 20)
    train, test = np.arange(cut), np.arange(cut, n)
    if len(test) < 10:
        raise ValueError("not enough windows for a hold-out split")

    f = fit(_subset(ds, train))
    sub_test = _subset(ds, test)
    pred = predict(sub_test, f)
    ok = np.isfinite(sub_test.y) & (sub_test.y > 0)
    y, p = sub_test.y[ok], pred[ok]

    resid = y - p
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    # A model that just predicts the training mean is the bar to clear.
    naive = float(np.abs(y - ds.y[train][np.isfinite(ds.y[train])].mean()).mean())

    return HoldoutResult(
        n_train=cut,
        n_test=int(ok.sum()),
        r2=r2,
        mae=float(np.abs(resid).mean()),
        rmse=float(np.sqrt((resid @ resid) / len(y))),
        mean_measured=float(y.mean()),
        mean_predicted=float(p.mean()),
        naive_mae=naive,
    )


def plot_discharge(res: DischargeResult, path: str) -> bool:
    """Write the predicted-vs-measured discharge plot. Returns False if
    matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    mins = (res.ts - res.ts[0]) / 60.0
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    ax.plot(mins, res.measured_wh, label="measured (fuel gauge)", lw=2.2, color="#1b3a5c")
    ax.plot(
        mins,
        res.predicted_wh,
        label="predicted (attribution model)",
        lw=2.0,
        ls="--",
        color="#c2410c",
    )
    ax.fill_between(
        mins, res.measured_wh, res.predicted_wh, color="#c2410c", alpha=0.12
    )
    ax.set_xlabel("minutes since start of held-out segment")
    ax.set_ylabel("battery energy remaining (Wh)")
    ax.set_title(
        f"Predicted vs measured discharge  —  "
        f"MAE {res.mae_wh:.3f} Wh, final error {res.final_error_pct:+.1f}%"
    )
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True
