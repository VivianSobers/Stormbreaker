"""A synthetic benchmark for attribution accuracy.

`selftest` measures attribution error on real hardware, which is the honest
test but costs minutes per run. This generates data with *known* per-application
costs so a model change can be scored in milliseconds, and so the error can be
decomposed into causes that real hardware confounds together.

The generator deliberately reproduces the structures that make real attribution
hard:

* applications that switch on and off independently (identifiable),
* applications that co-vary with another (only partly identifiable),
* an always-on daemon at constant load (indistinguishable from the baseline),
* measurement noise on the power sensor.

Being synthetic, it can only validate that the estimator recovers what it was
given. It cannot tell us the feature set is right — real workloads consume
power through paths these columns do not model. `selftest` remains the
authority; this is the fast iteration loop underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Dataset, fit


@dataclass
class Scenario:
    n_windows: int = 600
    n_apps: int = 6
    noise_w: float = 0.30
    baseline_w: float = 4.0
    n_correlated: int = 0
    """Applications forced to co-vary with app 0, which caps how well they can
    be separated no matter how much data is collected."""
    constant_daemon: bool = True
    seed: int = 0


@dataclass
class BenchResult:
    truth: dict[str, float] = field(default_factory=dict)
    recovered: dict[str, float] = field(default_factory=dict)
    rel_error: dict[str, float] = field(default_factory=dict)
    baseline_true: float = 0.0
    baseline_fit: float = 0.0
    r2: float = 0.0

    def _group(self, pred) -> float:
        vals = [v for k, v in self.rel_error.items() if pred(k)]
        return float(np.median(vals)) if vals else float("nan")

    @property
    def independent_error(self) -> float:
        """Applications that switch on and off on their own schedule. These are
        the ones attribution can genuinely resolve."""
        return self._group(lambda k: k.startswith("app"))

    @property
    def correlated_error(self) -> float:
        """Applications welded to another's schedule. Bounded below by the
        identifiability limit, not by estimator quality."""
        return self._group(lambda k: k.startswith("corr"))

    @property
    def daemon_error(self) -> float:
        """A constant-load service, collinear with the baseline by
        construction. Expected to fail; reported so the failure stays visible
        rather than hiding inside an average."""
        return self._group(lambda k: k == "daemon")


def make_scenario(sc: Scenario) -> tuple[Dataset, dict[str, float], float]:
    rng = np.random.default_rng(sc.seed)
    n = sc.n_windows
    cols: list[tuple[str, str]] = []
    acts: list[np.ndarray] = []
    truth: dict[str, float] = {}

    def duty(period: int, frac: float, level: float) -> np.ndarray:
        phase = rng.integers(0, period)
        on = ((np.arange(n) + phase) % period) < max(int(period * frac), 1)
        return on.astype(float) * level * (0.85 + 0.3 * rng.random(n))

    base = None
    for i in range(sc.n_apps):
        name = f"app{i}"
        a = duty(int(rng.integers(7, 40)), float(rng.uniform(0.2, 0.7)),
                 float(rng.uniform(0.3, 2.5)))
        if base is None:
            base = a
        cols.append((name, "cpu"))
        acts.append(a)
        truth[name] = float(rng.uniform(1.5, 7.0))

    # Applications welded to app0: the identifiability limit, made explicit.
    for j in range(sc.n_correlated):
        name = f"corr{j}"
        a = base * float(rng.uniform(0.4, 1.6))
        cols.append((name, "cpu"))
        acts.append(a)
        truth[name] = float(rng.uniform(1.5, 7.0))

    if sc.constant_daemon:
        cols.append(("daemon", "cpu"))
        acts.append(np.full(n, 0.06) + rng.normal(0, 1e-3, n))
        truth["daemon"] = 2.0

    X = np.column_stack(acts)
    y = sc.baseline_w + X @ np.array([truth[c[0]] for c in cols])
    y = y + rng.normal(0, sc.noise_w, n)

    ds = Dataset(
        X=X, y=y, columns=cols, ts=np.arange(n, dtype=float) * 5.0,
        freq_edges=[], target="soc_w", win_ids=list(range(n)),
        globals_={"dt": np.full(n, 5.0), "discharging": np.zeros(n)},
    )
    return ds, truth, sc.baseline_w


def run(sc: Scenario, **fit_kw) -> BenchResult:
    ds, truth, baseline = make_scenario(sc)
    f = fit(ds, **fit_kw)
    res = BenchResult(truth=truth, baseline_true=baseline,
                      baseline_fit=f.baseline, r2=f.r2)
    for j, (label, _feat) in enumerate(f.columns):
        got = float(f.coef[j])
        res.recovered[label] = got
        want = truth[label]
        res.rel_error[label] = abs(got - want) / want
    return res


def sweep(name: str, values, build, **fit_kw) -> list[tuple]:
    out = []
    for v in values:
        ind, corr, dae, r2s = [], [], [], []
        for seed in range(5):
            sc = build(v, seed)
            r = run(sc, **fit_kw)
            ind.append(r.independent_error)
            corr.append(r.correlated_error)
            dae.append(r.daemon_error)
            r2s.append(r.r2)
        out.append((
            v,
            float(np.nanmean(ind)),
            float(np.nanmean(corr)),
            float(np.nanmean(dae)),
            float(np.mean(r2s)),
        ))
    return out
