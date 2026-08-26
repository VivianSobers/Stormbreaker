"""How much to trust a single application's watts.

The README is blunt that the energy *total* is validated to ~4% while the
*split* between applications is only good to 10-25%. That is a global average,
and a global average is the wrong thing to hand someone reading one row: the
split is tight for an application that runs on its own schedule and close to
meaningless for one that only ever runs while something else does.

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
activity too would answer a different, less useful question about a
hypothetical period in which the machine was used differently.

What the interval does not cover
--------------------------------

**Bias.** The estimator is regularised and bounded, both of which shrink
coefficients. Measured on synthetic data with a known answer, the point
estimate sat 5.8% above truth while the interval spanned 2.5% — so the interval
missed the true value entirely. It describes the spread of the estimator, not
its distance from reality. Only :mod:`stormbreaker.selftest` measures the
latter, against workloads whose real cost is known.

**Unidentifiability.** This one is worth stating twice, because the failure is
silent. Two applications that are always active together can be split any way
at all without changing the prediction. Ridge breaks that tie, and it breaks it
*the same way in every resample* — so the bootstrap reports a narrow interval
around an arbitrary split. Measured on a perfectly proportional pair: a 0.7%
interval around a number 9% wrong, while their combined total was good to 4%.

So a narrow interval is necessary but not sufficient for trusting a row. For
labels that :func:`stormbreaker.model.inseparable_groups` puts in a group, the
number that means something is the group's total, and
:meth:`Bootstrap.combined` is what computes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
        the application at all.
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


@dataclass
class Bootstrap:
    """Resampled watts per label, plus the machinery to read them.

    Draws are kept rather than reduced to intervals immediately, because the
    interval that means something for a group of inseparable applications is
    the one on their *sum*, which cannot be recovered from their individual
    intervals afterwards.
    """

    point: dict[str, float]
    draws: dict[str, np.ndarray]
    ci: float
    block: int
    n_windows: int
    n_draws: int
    groups: list[list[str]] = field(default_factory=list)

    def _interval(self, label: str, point: float, vals: np.ndarray) -> Interval:
        if vals.size == 0:
            return Interval(label, point, point, point, point)
        tail = (1.0 - self.ci) / 2.0 * 100.0
        return Interval(
            label=label,
            point=point,
            lo=float(np.percentile(vals, tail)),
            hi=float(np.percentile(vals, 100.0 - tail)),
            median=float(np.median(vals)),
        )

    def interval(self, label: str) -> Interval:
        return self._interval(label, self.point[label], self.draws[label])

    def intervals(self) -> dict[str, Interval]:
        return {lab: self.interval(lab) for lab in self.point}

    def combined(self, labels: list[str]) -> Interval:
        """The interval on a set of labels taken together.

        Summing *within* each resample before taking percentiles is the whole
        point: when two applications trade the same watts back and forth
        between resamples, their individual spreads are large and their total's
        is small, and only this order of operations shows that.
        """
        name = " + ".join(labels)
        point = sum(self.point.get(lab, 0.0) for lab in labels)
        stacks = [self.draws[lab] for lab in labels if lab in self.draws]
        if not stacks:
            return Interval(name, point, point, point, point)
        return self._interval(name, point, np.sum(stacks, axis=0))

    def group_of(self, label: str) -> list[str] | None:
        for g in self.groups:
            if label in g:
                return g
        return None


def bootstrap_watts(
    ds: Dataset,
    f: Fit,
    n_resamples: int = 60,
    ci: float = 0.90,
    block: int | None = None,
    seed: int = 0,
    groups: list[list[str]] | None = None,
) -> Bootstrap:
    """Per-label watts with a confidence interval, by moving-block bootstrap.

    Ridge strength is held at the value the base fit chose rather than
    re-searched per resample: re-searching costs 6x for a second-order effect,
    and a lambda that moved between resamples would mix regularisation drift
    into what is meant to be sampling spread.
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
    collected: dict[str, list[float]] = {lab: [] for lab in point}

    for _ in range(n_resamples):
        idx = _resample_index(n, blk, rng)
        try:
            fb = fit(ds.select(idx), lam=f.lam)
        except ValueError:
            # Too few usable windows in this resample; a skipped draw is
            # better than a fabricated one.
            continue
        w = _watts_by_label(ds.X, fb.coef, fb.columns)
        for lab in collected:
            collected[lab].append(w.get(lab, 0.0))

    draws = {lab: np.asarray(v, dtype=float) for lab, v in collected.items()}
    n_draws = max((len(v) for v in collected.values()), default=0)
    return Bootstrap(
        point=point,
        draws=draws,
        ci=ci,
        block=blk,
        n_windows=n,
        n_draws=n_draws,
        groups=[list(g) for g in (groups or [])],
    )


def render_intervals(bs: Bootstrap, n: int = 15) -> str:
    """A table of watts with ranges, with the misleading rows called out."""
    rows = sorted(bs.intervals().values(), key=lambda i: -i.point)[:n]
    shown = {r.label for r in rows}

    out = [
        f"  {bs.n_windows} windows, {bs.n_draws} moving-block resamples of "
        f"{bs.block} windows each",
        "",
        f"  {'WATTS':>7}  {'LOW':>7}  {'HIGH':>7}  {'+/-':>6}  {'':3} APPLICATION",
    ]
    for r in rows:
        rel = r.relative_width
        rel_s = "  n/a" if rel == float("inf") else f"{rel*100:5.0f}%"
        g = bs.group_of(r.label)
        tag = "[?]" if g else "   "
        note = "" if r.separable else "  <- not distinguishable from zero"
        out.append(
            f"  {r.point:7.3f}  {r.lo:7.3f}  {r.hi:7.3f}  {rel_s}  {tag} "
            f"{r.label}{note}"
        )

    pct = f"{bs.ci*100:.0f}%"
    groups = [g for g in bs.groups if shown & set(g)]
    if groups:
        out.append("")
        out.append(
            "  [?] These run together, so how the watts divide between them is "
            "arbitrary. The\n      range beside each one measures how much the "
            "split moved between resamples,\n      which is not the same as how "
            "arbitrary it is — ridge can pick the same\n      wrong split every "
            "time. Trust their total instead:"
        )
        for g in groups:
            c = bs.combined(g)
            out.append(
                f"        {c.point:7.3f}  [{c.lo:.3f} - {c.hi:.3f}]  "
                f"{' + '.join(g)}"
            )
    out.append("")
    out.append(
        f"  Ranges are {pct} moving-block bootstrap intervals. They cover "
        "sampling spread\n  only: an estimate that is biased on every resample "
        "is biased here too, and\n  the range will not say so. For absolute "
        "accuracy see `stormbreaker selftest`."
    )
    return "\n".join(out)
