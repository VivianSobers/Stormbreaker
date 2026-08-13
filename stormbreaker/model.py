"""The attribution model.

Per-process power is not measurable. The hardware reports one number for the
whole package, so the only honest route is to regress that one number onto the
per-cgroup activity vectors and let the coefficients fall out.

    measured_watts[t] = baseline + SUM_k SUM_j  w[k,j] * activity[t,k,j]

Two constraints make the coefficients physically meaningful rather than merely
predictive:

* **Non-negativity.** No workload can consume negative power. Without this
  constraint, collinear features happily produce large positive and negative
  coefficients that cancel, which predicts well and attributes nonsense.
* **A free baseline.** The static draw of an idle machine belongs to nobody. It
  is fitted as an unpenalised intercept so it cannot be smeared across
  applications.

Ridge regularisation handles the collinearity that remains (CPU time and
context switches move together), and is applied by augmenting the system rather
than by any bespoke solver, so ``scipy.optimize.nnls`` still does the work.

CPU time is bucketed by clock frequency because the energy cost of a busy core
is strongly superlinear in frequency; a core-second at 5 GHz and one at 1.2 GHz
are different goods and must not share a coefficient.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear

from .store import Store

# Per-cgroup features other than CPU. CPU is handled separately because it is
# split across frequency buckets.
FLAT_FEATURES = ("io_mb", "gpu", "ctxt_k")

# Below these activity levels a feature carries no identifiable information:
# the column is essentially zero, and any coefficient fitted to it is noise
# scaled up by a near-singular inverse. Such columns are dropped outright.
ACTIVITY_FLOOR = {
    "cpu": 0.01,  # busy cores
    "io_mb": 0.05,  # MB/s
    "gpu": 0.002,  # busy GPU-seconds/s
    "ctxt_k": 0.05,  # thousand context switches/s
}


def feature_kind(feat: str) -> str:
    """Frequency-bucketed CPU columns all share the 'cpu' kind."""
    return "cpu" if feat.startswith("cpu") else feat


@dataclass
class Fit:
    columns: list[tuple[str, str]]  # (label, feature name)
    coef: np.ndarray  # non-negative, aligned with columns
    baseline: float  # watts attributable to nobody
    freq_edges: list[float]
    target: str
    lam: float
    r2: float
    mae: float
    rmse: float
    n_windows: int
    labels: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "columns": [list(c) for c in self.columns],
                "coef": self.coef.tolist(),
                "baseline": self.baseline,
                "freq_edges": self.freq_edges,
                "target": self.target,
                "lam": self.lam,
                "r2": self.r2,
                "mae": self.mae,
                "rmse": self.rmse,
                "n_windows": self.n_windows,
                "labels": self.labels,
            }
        )

    @staticmethod
    def from_json(blob: str) -> "Fit":
        d = json.loads(blob)
        return Fit(
            columns=[tuple(c) for c in d["columns"]],
            coef=np.array(d["coef"]),
            baseline=d["baseline"],
            freq_edges=d["freq_edges"],
            target=d["target"],
            lam=d["lam"],
            r2=d["r2"],
            mae=d["mae"],
            rmse=d["rmse"],
            n_windows=d["n_windows"],
            labels=d.get("labels", []),
        )


@dataclass
class Dataset:
    X: np.ndarray  # (n_windows, n_columns), design matrix without intercept
    y: np.ndarray  # (n_windows,) measured watts
    columns: list[tuple[str, str]]
    ts: np.ndarray
    freq_edges: list[float]
    target: str
    win_ids: list[int]
    globals_: dict[str, np.ndarray]


def _bucket_edges(freq: np.ndarray, n_buckets: int) -> list[float]:
    """Frequency bucket edges from the observed distribution.

    Quantiles rather than fixed cut-points: a machine that never leaves its
    efficiency range should not waste columns on frequencies it never reaches.
    """
    usable = freq[freq > 0]
    if usable.size < 50 or n_buckets <= 1:
        return []
    qs = np.linspace(0, 100, n_buckets + 1)[1:-1]
    edges = sorted(set(np.percentile(usable, qs).round(3).tolist()))
    return [float(e) for e in edges]


def _bucket_of(freq: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if freq < e:
            return i
    return len(edges)


def load_dataset(
    store: Store,
    since: float | None = None,
    limit: int | None = None,
    target: str | None = None,
    top_n: int = 30,
    n_buckets: int = 3,
    min_cpu: float = 1e-4,
) -> Dataset:
    """Build the design matrix from stored windows.

    Labels beyond ``top_n`` (ranked by total CPU) are merged into ``other``.
    Keeping every short-lived scope as its own column would add hundreds of
    near-empty columns whose coefficients are pure noise.
    """
    rows = store.windows(since=since, limit=limit)
    if not rows:
        raise ValueError("no windows in database; run `stormbreaker collect` first")

    win_ids = [r["id"] for r in rows]
    samples = store.samples_for(win_ids)

    if target is None:
        target = "rapl_pkg_w" if rows[0]["rapl_pkg_w"] is not None else "soc_w"
    y = np.array([(r[target] if r[target] is not None else np.nan) for r in rows], float)
    ts = np.array([r["ts"] for r in rows], float)
    freq = np.array([r["freq_ghz"] or 0.0 for r in rows], float)

    totals: dict[str, float] = {}
    for wid in win_ids:
        for label, f in samples.get(wid, {}).items():
            totals[label] = totals.get(label, 0.0) + f["cpu"] + f["gpu"]
    keep = {
        lab
        for lab, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:top_n]
        if totals[lab] > min_cpu
    }

    edges = _bucket_edges(freq, n_buckets)
    n_bucket_cols = len(edges) + 1
    labels = sorted(keep | {"other"})

    columns: list[tuple[str, str]] = []
    col_index: dict[tuple[str, str], int] = {}
    for lab in labels:
        for b in range(n_bucket_cols):
            name = f"cpu@{b}" if n_bucket_cols > 1 else "cpu"
            col_index[(lab, name)] = len(columns)
            columns.append((lab, name))
        for feat in FLAT_FEATURES:
            col_index[(lab, feat)] = len(columns)
            columns.append((lab, feat))

    X = np.zeros((len(win_ids), len(columns)), float)
    for i, wid in enumerate(win_ids):
        b = _bucket_of(freq[i], edges)
        cpu_name = f"cpu@{b}" if n_bucket_cols > 1 else "cpu"
        for label, f in samples.get(wid, {}).items():
            lab = label if label in keep else "other"
            X[i, col_index[(lab, cpu_name)]] += f["cpu"]
            for feat in FLAT_FEATURES:
                X[i, col_index[(lab, feat)]] += f[feat] or 0.0

    globals_ = {
        k: np.array([r[k] if r[k] is not None else np.nan for r in rows], float)
        for k in (
            "soc_w",
            "rapl_pkg_w",
            "batt_w",
            "gpu_busy",
            "freq_ghz",
            "charge",
            "volt_v",
            "dt",
        )
    }
    globals_["discharging"] = np.array([r["discharging"] or 0 for r in rows], float)

    return Dataset(
        X=X,
        y=y,
        columns=columns,
        ts=ts,
        freq_edges=edges,
        target=target,
        win_ids=win_ids,
        globals_=globals_,
    )


def _group_scales(X: np.ndarray, kinds: list[str]) -> np.ndarray:
    """One ridge scale per *feature kind*, shared by every application.

    Scaling each column to unit norm individually — the textbook default —
    is actively wrong here. It rescales a nearly-idle application's tiny
    activity column up to the same footing as a saturated one, so the ridge
    penalty no longer restrains that application's raw watts-per-core, and a
    process using a thousandth of a core can be handed a four-figure
    coefficient that fits noise.

    Columns of one kind are already in identical units (busy cores, MB/s), so
    they share a scale. The penalty then expresses the prior we actually hold:
    no application should have an extreme cost *per unit of work* relative to
    its peers.
    """
    scale = np.ones(X.shape[1])
    for kind in set(kinds[1:]):
        cols = [j for j, k in enumerate(kinds) if k == kind]
        norms = np.linalg.norm(X[:, cols], axis=0)
        rms = float(np.sqrt((norms**2).mean()))
        scale[cols] = rms if rms > 0 else 1.0
    return scale


def _upper_bounds(
    X: np.ndarray, y: np.ndarray, kinds: list[str], columns: list[tuple[str, str]]
) -> np.ndarray:
    """Physical ceilings on each coefficient.

    Non-negativity is only half of what physics tells us. The other half:

    * The idle baseline cannot exceed the least power ever observed, because
      every other term in the sum is non-negative. This one is exact.
    * No single feature can, at its own typical-high activity, account for more
      than the whole package draw. Anything above that is the solver fitting
      noise through a near-empty column.
    Note what is deliberately *not* bounded. A coefficient is a cost per unit
    of activity, and it is legitimate for it to exceed the total package power
    when the feature is never anywhere near one full unit: a process using a
    hundredth of a GPU may honestly cost 100 W per fully-busy-GPU-second, since
    that rate is only ever evaluated at a hundredth of it. Clamping such
    coefficients to the observed maximum power sounds physical but assumes a
    saturated unit was actually seen, and measurably degrades the fit when it
    was not. Extrapolation is instead flagged for the reader by
    :func:`coefficient_table`, which reports the activity each coefficient was
    identified at.
    """
    ub = np.full(X.shape[1], np.inf)
    ub[0] = max(float(np.min(y)), 0.0)
    y_max = float(np.max(y))
    for j in range(1, X.shape[1]):
        floor = ACTIVITY_FLOOR.get(kinds[j], 1e-3)
        ref = max(float(np.percentile(X[:, j], 95)), floor)
        ub[j] = y_max / ref
    return ub


def _solve_bounded_ridge(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    kinds: list[str],
    columns: list[tuple[str, str]],
) -> np.ndarray:
    """min ||Xw - y||^2 + lam*||w_scaled||^2  subject to  0 <= w <= physical.

    Column 0 is the intercept, penalised not at all and bounded above by the
    minimum observed power. Bounded-variable least squares replaces plain NNLS
    so the upper limits can be imposed; with the upper bounds at infinity the
    two agree exactly.
    """
    scale = _group_scales(X, kinds)
    Xs = X / scale
    ub = _upper_bounds(X, y, kinds, columns) * scale

    n_col = Xs.shape[1]
    penalty = np.sqrt(lam) * np.eye(n_col)
    penalty[0, 0] = 0.0  # free baseline
    X_aug = np.vstack([Xs, penalty])
    y_aug = np.concatenate([y, np.zeros(n_col)])

    res = lsq_linear(
        X_aug, y_aug, bounds=(np.zeros(n_col), ub), method="bvls", lsq_solver="exact"
    )
    return res.x / scale


def _metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    resid = y - pred
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.abs(resid).mean())
    rmse = math.sqrt(ss_res / len(y))
    return r2, mae, rmse


def fit(
    ds: Dataset,
    lam: float | None = None,
    holdout: float = 0.25,
) -> Fit:
    """Fit the model, choosing ridge strength by chronological hold-out.

    The split is chronological, never random: adjacent windows are correlated,
    so a shuffled split leaks the answer across the boundary and reports a
    hold-out score that cannot be reproduced in use.
    """
    ok = np.isfinite(ds.y) & (ds.y > 0)
    X_all = np.hstack([np.ones((ds.X.shape[0], 1)), ds.X])[ok]
    y_all = ds.y[ok]
    if len(y_all) < 20:
        raise ValueError(
            f"only {len(y_all)} usable windows; collect a few minutes of data first"
        )

    # Drop columns that never move, and columns whose activity never rises
    # above the level at which a coefficient means anything. Both are
    # unidentifiable; keeping them lets the solver explain noise with a huge
    # coefficient on a near-empty column.
    kinds_full = ["baseline"] + [feature_kind(feat) for _lab, feat in ds.columns]
    active = X_all.std(axis=0) > 0
    for j in range(1, X_all.shape[1]):
        if active[j]:
            floor = ACTIVITY_FLOOR.get(kinds_full[j], 1e-3)
            active[j] = float(X_all[:, j].max()) >= floor
    active[0] = True

    Xa = X_all[:, active]
    kinds_a = [k for k, keep in zip(kinds_full, active) if keep]
    cols_a = [c for c, keep in zip([("", "baseline"), *ds.columns], active) if keep]

    lams = [lam] if lam is not None else [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    cut = max(int(len(y_all) * (1 - holdout)), 10)
    best, best_score = None, float("inf")

    for cand in lams:
        if len(lams) == 1 or cut >= len(y_all) - 5:
            score = 0.0
            w = _solve_bounded_ridge(Xa, y_all, cand, kinds_a, cols_a)
        else:
            w_tr = _solve_bounded_ridge(
                Xa[:cut], y_all[:cut], cand, kinds_a, cols_a
            )
            score = float(np.abs(y_all[cut:] - Xa[cut:] @ w_tr).mean())
            w = _solve_bounded_ridge(Xa, y_all, cand, kinds_a, cols_a)
        if score < best_score:
            best_score, best = score, (cand, w)

    lam_used, w_active = best
    w_full = np.zeros(X_all.shape[1])
    w_full[active] = w_active

    pred = X_all @ w_full
    r2, mae, rmse = _metrics(y_all, pred)

    labels = sorted({lab for lab, _ in ds.columns})
    return Fit(
        columns=ds.columns,
        coef=w_full[1:],
        baseline=float(w_full[0]),
        freq_edges=ds.freq_edges,
        target=ds.target,
        lam=lam_used,
        r2=r2,
        mae=mae,
        rmse=rmse,
        n_windows=int(len(y_all)),
        labels=labels,
    )


def predict(ds: Dataset, f: Fit) -> np.ndarray:
    """Predicted package watts per window."""
    return f.baseline + ds.X @ f.coef


def attribute(ds: Dataset, f: Fit) -> dict[str, np.ndarray]:
    """Per-label watts per window. Sums to ``predict() - baseline``."""
    out: dict[str, np.ndarray] = {}
    for j, (label, _feat) in enumerate(f.columns):
        if f.coef[j] == 0:
            continue
        contrib = ds.X[:, j] * f.coef[j]
        if label in out:
            out[label] += contrib
        else:
            out[label] = contrib.copy()
    return out


def mean_watts(ds: Dataset, f: Fit) -> list[tuple[str, float]]:
    """Average watts per label over the dataset, descending."""
    attr = attribute(ds, f)
    ranked = [(lab, float(v.mean())) for lab, v in attr.items()]
    ranked.sort(key=lambda kv: -kv[1])
    return ranked


@dataclass
class CoefRow:
    label: str
    feature: str
    coef: float  # watts per unit of activity
    activity_p95: float  # the activity level it was identified at
    watts_p95: float  # what it contributes at that activity
    watts_mean: float  # what it contributes on average
    extrapolated: bool  # identified far below one full unit


def coefficient_table(f: Fit, ds: Dataset | None = None) -> list[CoefRow]:
    """Non-zero coefficients with the activity each was identified at.

    A coefficient is a cost *per unit* of activity — per busy core, per MB/s,
    per busy GPU-second, per 1000 context switches. Read alone it invites a
    category error: a coefficient of 130 W per fully-busy-GPU-second looks
    impossible on a 45 W package until you notice it was only ever evaluated at
    a hundredth of a busy GPU, contributing 1.3 W.

    So the activity level travels with the number. Rows identified far below
    one full unit are marked as extrapolated, meaning the rate is real but the
    machine was never observed anywhere near that regime.
    """
    rows: list[CoefRow] = []
    for j, (lab, feat) in enumerate(f.columns):
        coef = float(f.coef[j])
        if coef <= 0:
            continue
        if ds is not None and j < ds.X.shape[1]:
            p95 = float(np.percentile(ds.X[:, j], 95))
            mean = float(ds.X[:, j].mean())
        else:
            p95 = mean = float("nan")
        kind = feature_kind(feat)
        rows.append(
            CoefRow(
                label=lab,
                feature=feat,
                coef=coef,
                activity_p95=p95,
                watts_p95=coef * p95,
                watts_mean=coef * mean,
                extrapolated=bool(kind in ("cpu", "gpu") and p95 == p95 and p95 < 0.1),
            )
        )
    # Ranked by what each actually contributes, not by the raw rate, so the
    # table cannot be read as "this daemon costs 130 W".
    rows.sort(key=lambda r: -(r.watts_mean if r.watts_mean == r.watts_mean else 0.0))
    return rows
