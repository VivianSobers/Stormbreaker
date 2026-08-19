"""Self-test: does the per-application split actually hold up?

The discharge check validates the *total* — how fast the battery drains. It
says nothing about whether the split between applications is right, and a model
can get the total exactly right while dividing it between applications wrongly.

Testing the split looks like it needs ground truth that does not exist: nobody
can measure what a single process really costs. But two properties can be
checked without ever knowing the true watts.

**Symmetry.** Run byte-identical workloads in two differently-named cgroups.
Whatever a busy core really costs, both must be charged the *same* amount. Any
divergence is attribution error, full stop — there is no interpretation under
which the same work in a different cgroup legitimately costs more.

**Linearity.** Double the cores, and the attributed power should roughly
double. The relationship is not exactly linear on a real chip — frequency and
voltage move with load — so this is a sanity bound rather than an assertion.

**Separation.** Run both cgroups at once. Each should receive about half of the
combined draw, rather than one absorbing the other's share.

The harness drives these workloads itself, into dedicated systemd scopes, so
the cgroup boundaries are unambiguous.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

from .caps import probe
from .collect import Collector
from .model import Dataset, Fit, attribute, feature_kind, fit, load_dataset
from .store import DEFAULT_DB, Store

UNIT_A = "sb-selftest-a"
UNIT_B = "sb-selftest-b"

DEFAULT_SELFTEST_DB = os.path.join(
    os.path.dirname(DEFAULT_DB), "selftest.db"
)
"""Runs are kept next to the main database rather than in a temporary
directory. A run costs minutes of wall time and a burst of CPU; throwing the
data away means paying that again for every re-analysis, and re-analysis is
milliseconds."""


@dataclass
class Phase:
    label: str
    unit: str | None
    threads: int
    seconds: float
    threads_b: int = 0
    """Thread count for unit B when both run together."""
    duty_a: tuple[float, float] = (0.0, 0.0)
    duty_b: tuple[float, float] = (0.0, 0.0)
    """On/off seconds per unit. Different periods make the two activity
    columns uncorrelated, which is what makes them separable."""


def default_schedule(scale: float = 1.0) -> list[Phase]:
    """Idle gaps between phases let the package settle, so a phase is not
    contaminated by the thermal and frequency tail of the one before it."""
    s = lambda x: max(x * scale, 5.0)  # noqa: E731
    # Settling gaps are the cheapest thing to shorten: they exist only to keep
    # one phase's thermal tail out of the next, which takes a few seconds, not
    # twenty. Trimming them cut a default run from 289s to 189s without
    # touching the phases that carry signal.
    return [
        Phase("settle", None, 0, s(10)),
        Phase("A x2", UNIT_A, 2, s(35)),
        Phase("idle", None, 0, s(8)),
        Phase("B x2", UNIT_B, 2, s(35)),
        Phase("idle", None, 0, s(8)),
        Phase("A x4", UNIT_A, 4, s(35)),
        Phase("idle", None, 0, s(8)),
        # The two units run together but on *different duty cycles*, and this
        # is the crux of the whole test.
        #
        # Two workloads that run simultaneously at constant levels have
        # proportional activity columns (X_b = k * X_a), and no estimator can
        # split two proportional columns — the division between them is set by
        # the penalty, not by the data. Making them merely different sizes does
        # not help; that is still proportional. Only independent variation over
        # time makes the columns distinguishable.
        #
        # Measured: identical steady loads scored 0.0% symmetry error (forced by
        # ridge symmetry), proportional steady loads scored 50%, and only
        # independently duty-cycled loads test anything real.
        Phase("A/B duty", "both", 3, s(50), threads_b=3,
              duty_a=(4.0, 3.0), duty_b=(7.0, 5.0)),
        Phase("idle", None, 0, s(8)),
    ]


def have_systemd_run() -> bool:
    return shutil.which("systemd-run") is not None


def _spawn(
    unit: str, threads: int, seconds: float, on: float = 0.0, off: float = 0.0
) -> subprocess.Popen | None:
    """Start a busy-loop workload inside its own transient scope.

    With ``on``/``off`` set, the load is duty-cycled rather than steady. That
    is what makes two simultaneous workloads separable at all — see
    :func:`default_schedule`.
    """
    if on > 0:
        script = (
            f"end=$((SECONDS+{int(seconds)})); "
            f"while [ $SECONDS -lt $end ]; do "
            f"for i in $(seq {threads}); do (while :; do :; done) & done; "
            f"sleep {on:g}; kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; "
            f"sleep {off:g}; done"
        )
    else:
        script = (
            f"for i in $(seq {threads}); do (while :; do :; done) & done; "
            f"sleep {seconds:.0f}; kill $(jobs -p) 2>/dev/null; wait 2>/dev/null"
        )
    try:
        return subprocess.Popen(
            [
                "systemd-run", "--user", "--scope", "--quiet",
                f"--unit={unit}-{int(time.time())}",
                "bash", "-c", script,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None


@dataclass
class SelfTestResult:
    watts_per_core: dict[str, float] = field(default_factory=dict)
    cores: dict[str, float] = field(default_factory=dict)
    watts: dict[str, float] = field(default_factory=dict)
    symmetry_error: float = float("nan")
    solo_error: float = float("nan")
    simultaneous: tuple | None = None
    solo: tuple | None = None
    linearity: tuple | None = None
    fit: Fit | None = None
    n_windows: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Judged on the simultaneous comparison, which is the only one with no
        confound. The solo comparison is reported but not gated on."""
        return (
            self.symmetry_error == self.symmetry_error
            and self.symmetry_error < 0.15
        )


def run_workload(schedule: list[Phase], verbose: bool = True) -> None:
    for ph in schedule:
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] {ph.label:<8} {ph.seconds:.0f}s", flush=True)
        procs = []
        if ph.unit == "both":
            procs.append(_spawn(UNIT_A, ph.threads, ph.seconds, *ph.duty_a))
            procs.append(
                _spawn(UNIT_B, ph.threads_b or ph.threads, ph.seconds, *ph.duty_b)
            )
        elif ph.unit:
            procs.append(_spawn(ph.unit, ph.threads, ph.seconds))
        time.sleep(ph.seconds + 1.0)
        for p in procs:
            if p is not None:
                p.wait(timeout=10)


def _unit_side(label: str) -> str | None:
    """systemd appends a unique suffix per scope, so each phase becomes its own
    label. Map them back to the A/B workload they belonged to."""
    if label.startswith(UNIT_A):
        return "A"
    if label.startswith(UNIT_B):
        return "B"
    return None


def analyse(ds: Dataset, f: Fit) -> SelfTestResult:
    """Compare what the model charged each synthetic unit per busy core.

    Two comparisons are made, and they answer different questions.

    *Simultaneous* is the strict one: both workloads running at the same
    instant, so frequency, temperature and every other shared condition are
    identical by construction. Any difference is attribution error alone.

    The two are run on *independent duty cycles* while together. Simultaneous
    workloads at constant levels have proportional activity columns, and no
    estimator can divide two proportional columns — the split is decided by the
    penalty rather than the data. Only independent variation over time makes
    the columns distinguishable and the comparison meaningful.

    *Solo* compares the two workloads run at separate times. It is the weaker
    test, because the machine genuinely differed between the two phases, but it
    is closer to how attribution gets used in practice.
    """
    res = SelfTestResult(fit=f, n_windows=len(ds.y))
    attr = attribute(ds, f)

    col_of: dict[str, list[int]] = {}
    for j, (label, feat) in enumerate(f.columns):
        if feat.startswith("cpu"):
            col_of.setdefault(label, []).append(j)

    active: dict[str, np.ndarray] = {}
    for label in sorted(attr):
        if _unit_side(label) is None:
            continue
        cols = col_of.get(label, [])
        if not cols:
            continue
        per_window = ds.X[:, cols].sum(axis=1)
        cores = float(per_window.sum())
        if cores <= 0:
            continue
        active[label] = per_window > 0.05
        res.cores[label] = cores
        res.watts[label] = float(attr[label].sum()) / len(ds.y)
        res.watts_per_core[label] = float(attr[label].sum()) / cores

    a_labels = [l for l in res.watts_per_core if _unit_side(l) == "A"]
    b_labels = [l for l in res.watts_per_core if _unit_side(l) == "B"]

    # Strict test: the pair whose active windows overlap the most.
    best, best_overlap = None, 0
    for la in a_labels:
        for lb in b_labels:
            ov = int((active[la] & active[lb]).sum())
            if ov > best_overlap:
                best, best_overlap = (la, lb), ov
    if best and best_overlap >= 3:
        va, vb = res.watts_per_core[best[0]], res.watts_per_core[best[1]]
        res.symmetry_error = abs(va - vb) / max(va, vb)
        res.simultaneous = (va, vb, best_overlap)

    # Weaker test: solo phases with the most similar amount of work done.
    solo_a = [l for l in a_labels if best is None or l != best[0]]
    solo_b = [l for l in b_labels if best is None or l != best[1]]
    pair, diff = None, float("inf")
    for la in solo_a:
        for lb in solo_b:
            d = abs(res.cores[la] - res.cores[lb]) / max(res.cores[la], res.cores[lb])
            if d < diff:
                pair, diff = (la, lb), d
    if pair and diff < 0.25:
        va, vb = res.watts_per_core[pair[0]], res.watts_per_core[pair[1]]
        res.solo_error = abs(va - vb) / max(va, vb)
        res.solo = (va, vb)

    if res.symmetry_error != res.symmetry_error and res.solo_error != res.solo_error:
        res.notes.append(
            "did not observe a comparable pair of synthetic units — symmetry "
            "could not be tested"
        )

    # Linearity: the same unit at two different core counts.
    by_cores = sorted(((res.cores[l], l) for l in a_labels))
    if len(by_cores) >= 2:
        (c_lo, l_lo), (c_hi, l_hi) = by_cores[0], by_cores[-1]
        if c_hi > c_lo * 1.5:
            res.linearity = (
                res.watts_per_core[l_lo],
                res.watts_per_core[l_hi],
                c_hi / c_lo,
            )
    return res


def render(res: SelfTestResult) -> str:
    out = ["", "  Per-application attribution self-test", "  " + "=" * 56]
    if res.fit is not None:
        out.append(
            f"  fitted {res.n_windows} windows, R^2={res.fit.r2:.3f}, "
            f"MAE={res.fit.mae:.2f} W, baseline={res.fit.baseline:.2f} W"
        )
    out.append("")
    out.append(f"  {'UNIT':<24} {'CORE-SEC':>9} {'MEAN W':>8} {'W/CORE':>8}")
    for label in sorted(res.watts_per_core):
        out.append(
            f"  {label:<24} {res.cores[label]*5:9.1f} {res.watts[label]:8.3f} "
            f"{res.watts_per_core[label]:8.3f}"
        )
    out.append("")
    if res.simultaneous:
        va, vb, ov = res.simultaneous
        verdict = "PASS" if res.passed else "FAIL"
        out.append(
            f"  symmetry, simultaneous  A={va:.3f}  B={vb:.3f} W/core over {ov} "
            f"shared windows"
        )
        out.append(
            "                          (independent duty cycles, so equal cost "
            "per core is not forced)"
        )
        out.append(f"                          error {res.symmetry_error*100:5.1f}%   [{verdict}]")
    if res.solo:
        va, vb = res.solo
        out.append(
            f"  symmetry, run apart     A={va:.3f}  B={vb:.3f} W/core   "
            f"error {res.solo_error*100:5.1f}%"
        )
    if res.linearity:
        lo, hi, ratio = res.linearity
        out.append(
            f"  linearity               {lo:.3f} -> {hi:.3f} W/core at {ratio:.1f}x "
            f"the work ({(hi/lo-1)*100:+.0f}% per core)"
        )
    out.append("")
    out.append(
        "  Identical work in two differently-named cgroups must cost the same.\n"
        "  Run simultaneously, frequency and temperature are shared, so any gap\n"
        "  is attribution error alone. Run apart, the machine genuinely differed\n"
        "  between phases, so that figure is the weaker of the two."
    )
    for n in res.notes:
        out.append(f"  note: {n}")
    return "\n".join(out)


def ablate(db_path: str, minutes: float | None = None, holdout: float = 0.35):
    """Score each feature by removing it, on data already collected.

    Feature decisions used to cost a fresh recording each. They do not need
    one: zeroing a feature's columns in a stored dataset and re-scoring answers
    "does this earn its place" in milliseconds. A feature that does not improve
    held-out error is costing collection time for nothing.

    Returns (baseline_result, [(feature, r2, mae, delta_mae), ...]).
    """
    import time as _time

    from .validate import _subset, validate_holdout

    store = Store(db_path)
    try:
        since = None if minutes is None else _time.time() - minutes * 60.0
        ds = load_dataset(store, since=since)
    finally:
        store.close()

    base = validate_holdout(ds, holdout=holdout)
    kinds = sorted({feature_kind(feat) for _lab, feat in ds.columns})

    rows = []
    for kind in kinds:
        idx = [j for j, (_l, f) in enumerate(ds.columns) if feature_kind(f) == kind]
        if not idx:
            continue
        cut = _subset(ds, np.arange(len(ds.y)))
        cut.X = cut.X.copy()
        cut.X[:, idx] = 0.0
        try:
            r = validate_holdout(cut, holdout=holdout)
        except ValueError:
            continue
        rows.append((kind, r.r2, r.mae, r.mae - base.mae))
    # Largest positive delta = removing it hurt most = most valuable feature.
    rows.sort(key=lambda t: -t[3])
    return base, rows


def render_ablation(base, rows) -> str:
    out = ["", "  Feature ablation (on already-collected data)", "  " + "=" * 56]
    out.append(
        f"  all features: held-out R^2={base.r2:+.4f}  MAE={base.mae:.3f} W  "
        f"({base.n_train} train / {base.n_test} test)"
    )
    out.append("")
    out.append(f"  {'REMOVED':<10} {'R^2':>9} {'MAE':>8} {'MAE change':>11}  verdict")
    for kind, r2, mae, d in rows:
        verdict = (
            "carries signal" if d > 0.02
            else "no measurable value" if d > -0.02
            else "HURTS - consider dropping"
        )
        out.append(f"  {kind:<10} {r2:+9.4f} {mae:8.3f} {d:+11.3f}  {verdict}")
    out.append("")
    out.append(
        "  MAE change is what happens when the feature is REMOVED, so positive\n"
        "  means it was helping. Anything near zero is paying collection cost\n"
        "  for nothing."
    )
    return "\n".join(out)


def analyse_db(db_path: str) -> SelfTestResult:
    """Score a previously collected run. Milliseconds, no workload, no battery.

    Every question that does not need *new* measurements should come through
    here rather than through a fresh collection.
    """
    store = Store(db_path)
    try:
        ds = load_dataset(store, top_n=12, n_buckets=1)
        f = fit(ds)
        return analyse(ds, f)
    finally:
        store.close()


def run_selftest(
    db_path: str,
    scale: float = 1.0,
    window_s: float = 2.0,
    verbose: bool = True,
    reuse: bool = False,
) -> SelfTestResult:
    """Drive known workloads, collect, fit, and check the split."""
    if reuse:
        return analyse_db(db_path)
    if not have_systemd_run():
        raise RuntimeError(
            "systemd-run is required to place workloads in their own cgroups"
        )

    caps = probe()
    collector = Collector(
        db_path, window_s=window_s, caps=caps, refit_every_s=0
    )
    schedule = default_schedule(scale)
    total = sum(p.seconds + 1 for p in schedule)

    import threading

    stop = threading.Event()

    def _collect():
        collector.run(duration_s=total + 10)
        stop.set()

    t = threading.Thread(target=_collect, daemon=True)
    t.start()
    try:
        run_workload(schedule, verbose=verbose)
    finally:
        collector.stop()
        t.join(timeout=30)
    collector.store.commit()
    collector.store.close()
    return analyse_db(db_path)
