"""How much to trust a single application's watts.

The README is blunt that the energy *total* is validated to ~4% while the
*split* between applications is only good to 10-25%. That number is a global
average, and a global average is the wrong thing to hand someone reading one
row: the split is tight for an application that runs on its own schedule and
close to meaningless for one that only ever runs while something else does.

This module puts a range on each row, by moving-block bootstrap.

Two details make it a bootstrap of the right thing:

**Blocks, not individual windows.** Consecutive windows are strongly
autocorrelated — a video keeps playing, a compile keeps compiling. Resampling
windows independently would treat 2600 correlated observations as 2600
independent ones and report intervals several times too narrow. Contiguous
blocks preserve the correlation inside them, so the resamples carry roughly the
information the original series did.

**Activity is held fixed; only coefficients are resampled.** The question is
"how much of *my* battery did this application use over this period?" — the
activity over that period was observed, not sampled. So each resample refits
coefficients, then scores them against the original activity. Resampling the
activity too would answer a different and less useful question, about a
hypothetical period in which the machine was used differently.

Ridge strength is held at the value the base fit chose, rather than re-searched
per resample. Re-searching would cost 6x for a second-order effect, and a
lambda that changed between resamples would mix regularisation drift into what
is supposed to be sampling spread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Dataset, Fit, fit

# With fewer resamples than this the tail percentiles are being read off a
# handful of points and the interval is mostly noise about noise.
MIN_RESAMPLES = 20


@dataclass
class Interval:
    label: str
    point: float  # watts from the base fit
    lo: float
    hi: float
    median: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def relative_width(self) -> float:
        """Interval width as a fraction of the point estimate.

        Above ~1.0 the row is order-of-magnitude information only.
        """
        return self.width / self.point if self.point > 0 else float("inf")

    @property
    def separable(self) -> bool:
        """Whether this application is distinguishable from drawing nothing.

        A lower bound at zero means some resamples explained the data without
        the application at all — usually because it is collinear with a
        baseline or with another application that was always running.
        """
        return self.lo > 0.0


def block_length(n: int) -> int:
    """Blocks of about n**(1/3), the standard rule for a stationary series.

    2600 windows -> 14, which at a 5 s window is a bit over a minute: long
    enough to contain a whole burst of activity, short enough that the
    resamples still differ from each other.
    """
    return max(int(round(n ** (1.0 / 3.0))), 2)


def _resample_index(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of a moving-block resample of length n."""
    starts = rng.integers(0, max(n - block, 1), size=n // block + 1)
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
    return idx[:n]


def _slice(ds: Dataset, idx: np.ndarray) -> Dataset:
    return Dataset(
        X=ds.X[idx],
        y=ds.y[idx],
        columns=ds.columns,
        ts=ds.ts[idx],
        freq_edges=ds.freq_edges,
        target=ds.target,
        win_ids=[ds.win_ids[i] for i in idx],
        globals_={k: v[idx] for k, v in ds.globals_.items()},
    )


def _watts_by_label(
    X: np.ndarray, coef: np.ndarray, columns: list[tuple[str, str]]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for j, (label, _feat) in enumerate(columns):
        if coef[j] == 0.0:
            out.setdefault(label, 0.0)
            continue
        out[label] = out.get(label, 0.0) + float((X[:, j] * coef[j]).mean())
    return out


def bootstrap_watts(
    ds: Dataset,
    f: Fit,
    n_resamples: int = 60,
    ci: float = 0.90,
    block: int | None = None,
    seed: int = 0,
) -> dict[str, Interval]:
    """Per-label watts with a confidence interval, by moving-block bootstrap.

    Returns one :class:`Interval` per label in ``f``. Labels whose columns were
    dropped by every resample come back with an interval pinned at zero, which
    is the honest answer: the data does not distinguish them from nothing.
    """
    if n_resamples < MIN_RESAMPLES:
        raise ValueError(
            f"{n_resamples} resamples is too few to read an interval from; "
            f"use at least {MIN_RESAMPLES}"
        )
    n = ds.X.shape[0]
    blk = block or block_length(n)
    rng = np.random.default_rng(seed)

    point = _watts_by_label(ds.X, f.coef, f.columns)
    draws: dict[str, list[float]] = {lab: [] for lab in point}

    for _ in range(n_resamples):
        idx = _resample_index(n, blk, rng)
        try:
            fb = fit(_slice(ds, idx), lam=f.lam)
        except ValueError:
            # Too few usable windows in this resample; a skipped draw is
            # better than a fabricated one.
            continue
        w = _watts_by_label(ds.X, fb.coef, fb.columns)
        for lab in draws:
            draws[lab].append(w.get(lab, 0.0))

    tail = (1.0 - ci) / 2.0 * 100.0
    out: dict[str, Interval] = {}
    for lab, vals in draws.items():
        if not vals:
            out[lab] = Interval(lab, point[lab], point[lab], point[lab], point[lab])
            continue
        a = np.asarray(vals, dtype=float)
        out[lab] = Interval(
            label=lab,
            point=point[lab],
            lo=float(np.percentile(a, tail)),
            hi=float(np.percentile(a, 100.0 - tail)),
            median=float(np.median(a)),
        )
    return out


def render_intervals(
    intervals: dict[str, Interval], n: int = 15, ci: float = 0.90
) -> str:
    """A table of watts with ranges, worst-understood rows flagged."""
    rows = sorted(intervals.values(), key=lambda i: -i.point)[:n]
    out = [
        f"  {'WATTS':>7}  {'LOW':>7}  {'HIGH':>7}  {'+/-':>6}  APPLICATION",
    ]
    for r in rows:
        rel = r.relative_width
        rel_s = "  n/a" if rel == float("inf") else f"{rel*100:5.0f}%"
        mark = "" if r.separable else "  <- not distinguishable from zero"
        out.append(
            f"  {r.point:7.3f}  {r.lo:7.3f}  {r.hi:7.3f}  {rel_s}  {r.label}{mark}"
        )
    out.append("")
    out.append(
        f"  Ranges are {ci*100:.0f}% moving-block bootstrap intervals over the "
        "same data.\n  They cover sampling spread only — a coefficient that is "
        "biased on every\n  resample stays biased here."
    )
    return "\n".join(out)
