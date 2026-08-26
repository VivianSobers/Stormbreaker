"""What changed between two stretches of time.

"My battery is worse than it was this morning" is the question a battery tool
should be able to answer, and it is not the same question as "what is using
power now". Answering it needs a difference, and a difference of two
independently fitted models is not one: two fits have different column sets,
different chosen ridge strengths, and different shrinkage because they saw
different amounts of data. Subtracting them would report changes that are
artefacts of fitting twice.

So one model is fitted across both periods and used as a single price list,
and each period's activity is scored against it. What comes out is the change
in *work done*, priced consistently. That is the common case by a wide margin:
a browser tab started spinning, an indexer woke up, a video began playing.

The other kind of change — the same work costing more, because the machine is
hotter, or throttled, or on a different power profile — deliberately does not
land on any application. It shows up as the gap between the measured change in
package power and the sum of the attributed changes, reported as unexplained.
Blaming an application for a change in the cost of its work would be wrong, and
the residual is the honest place for it.

Both halves must come from the same regime, in two senses, and the second was
found by measurement rather than foresight. Coefficients fitted under
`balanced` do not describe `performance`, so a comparison straddling a profile
change measures the profile. And a comparison straddling a *plug-in* measures
the charger: the first real run of this code reported package power up 1.26 W
with the busy-core count down from 1.73 to 0.42, because the earlier period was
on battery and the later one was charging. No arrangement of application
coefficients can explain that, and none should be asked to. Both cases refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Dataset, fit, profile_mix, reduce_to_budget
from .uncertainty import _resample_index, _watts_by_label, block_length

# Below this the two periods barely overlap in anything and the comparison is
# reading noise. Two minutes at a 5 s window.
MIN_PERIOD_WINDOWS = 24

# A period with a mains fraction between these is itself straddling a plug-in
# and cannot be given a single regime.
MIXED_LOW, MIXED_HIGH = 0.2, 0.8

# A measured change smaller than this is not worth explaining, and dividing by
# it to compute an explained fraction produces nonsense.
SMALL_CHANGE_W = 0.25


@dataclass
class Change:
    label: str
    before: float
    after: float
    delta: float
    lo: float
    hi: float

    @property
    def real(self) -> bool:
        """Whether the change is larger than the model's own uncertainty.

        An interval spanning zero means some resamples of the same data saw
        the change go the other way.
        """
        return self.lo > 0.0 or self.hi < 0.0

    @property
    def relative(self) -> float:
        return self.delta / self.before if self.before > 0 else float("inf")


@dataclass
class Comparison:
    before_windows: int
    after_windows: int
    before_minutes: float
    after_minutes: float
    measured_before: float
    measured_after: float
    changes: list[Change]
    ci: float
    block: int
    n_draws: int
    profile: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def measured_delta(self) -> float:
        return self.measured_after - self.measured_before

    @property
    def attributed_delta(self) -> float:
        return sum(c.delta for c in self.changes)

    @property
    def unexplained(self) -> float:
        """Measured change the applications do not account for.

        This is where a change in the *cost* of work lands — thermal drift,
        throttling, a background device waking up — as opposed to a change in
        the amount of work done.
        """
        return self.measured_delta - self.attributed_delta

    @property
    def explained(self) -> float:
        """Fraction of the measured change the applications account for."""
        if abs(self.measured_delta) < 1e-9:
            return float("nan")
        return self.attributed_delta / self.measured_delta

    @property
    def trustworthy(self) -> bool:
        """Whether the attributed changes are worth reading as an answer.

        A comparison in which the applications account for a third of the
        measured change — or, worse, move the other way — has ranked rows that
        look like findings and are not. Reporting them without saying so is the
        failure this whole tool is arranged to avoid.
        """
        if abs(self.measured_delta) < SMALL_CHANGE_W:
            return True  # nothing much moved; no claim is being made
        return 0.5 <= self.explained <= 2.0


def split_periods(
    ds: Dataset,
    recent_min: float,
    baseline_min: float,
    ending_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for the earlier and later periods, newest-anchored.

    Anchored on the newest window rather than on the wall clock, so the
    comparison means the same thing whether it is run during collection or a
    day afterwards. ``ending_min`` walks the whole pair back in time, which is
    how a past event gets examined — and, in practice, how a pair is moved off
    a plug-in that would otherwise sit between them.
    """
    end = float(ds.ts.max()) - ending_min * 60.0
    after_from = end - recent_min * 60.0
    before_from = after_from - baseline_min * 60.0
    # Both ends bounded: with ``ending_min`` set, "recent" is a window in the
    # past, not everything after a start point.
    after = (ds.ts >= after_from) & (ds.ts <= end)
    before = (ds.ts >= before_from) & (ds.ts < after_from)
    return before, after


def _check_mains(ds: Dataset, before: np.ndarray, after: np.ndarray) -> None:
    """Refuse a comparison that straddles a plug-in.

    Mains state is the coarsest regime boundary there is. Charging warms the
    package, changes what the SoC sensor reads, and lifts clocks that were
    being held down — none of which any application did.
    """
    disc = ds.globals_.get("discharging")
    if disc is None:
        return
    frac = {
        "earlier": float(np.nanmean(disc[before])),
        "recent": float(np.nanmean(disc[after])),
    }

    def describe(v: float) -> str:
        return f"on battery ({v*100:.0f}% of windows)" if v >= 0.5 else (
            f"on mains ({(1-v)*100:.0f}% of windows)"
        )

    for name, v in frac.items():
        if MIXED_LOW < v < MIXED_HIGH:
            raise ValueError(
                f"the {name} period is itself part plugged in and part on "
                f"battery ({v*100:.0f}% discharging). Charging changes what "
                "the package sensor reads, so pick a period that sits on one "
                "side of the plug-in."
            )
    if (frac["earlier"] >= 0.5) != (frac["recent"] >= 0.5):
        raise ValueError(
            f"the earlier period was {describe(frac['earlier'])} and the "
            f"recent one {describe(frac['recent'])}. Charging warms the "
            "package and lifts clocks on its own, so this comparison would be "
            "measuring the charger rather than the applications."
        )


def compare(
    ds: Dataset,
    recent_min: float = 60.0,
    baseline_min: float | None = None,
    ending_min: float = 0.0,
    n_resamples: int = 60,
    ci: float = 0.90,
    seed: int = 0,
) -> Comparison:
    """Attribute the change in power between two periods to applications."""
    baseline_min = recent_min if baseline_min is None else baseline_min
    before, after = split_periods(ds, recent_min, baseline_min, ending_min)

    for name, mask, span in (
        ("earlier", before, baseline_min),
        ("recent", after, recent_min),
    ):
        if int(mask.sum()) < MIN_PERIOD_WINDOWS:
            raise ValueError(
                f"only {int(mask.sum())} windows in the {name} period "
                f"({span:g} min); need {MIN_PERIOD_WINDOWS}. Collect for longer "
                "or widen the period."
            )

    notes: list[str] = []
    _check_mains(ds, before, after)
    union = before | after
    sub = ds.select(union)

    profile = None
    if sub.profiles is not None:
        mix = profile_mix(sub)
        if len(mix) > 1:
            top = ", ".join(f"{k} ({v})" for k, v in sorted(mix.items()))
            raise ValueError(
                "the two periods span more than one power regime "
                f"({top}). Coefficients fitted under one profile do not "
                "describe another, so this comparison would be measuring the "
                "profile change, not the applications."
            )
        profile = next(iter(mix), None)

    # The column budget was sized for however much data was loaded, which is
    # not what is being fitted here.
    sub = reduce_to_budget(sub)
    f = fit(sub)

    in_after = after[union]
    X_before, X_after = sub.X[~in_after], sub.X[in_after]
    point_before = _watts_by_label(X_before, f.coef, f.columns)
    point_after = _watts_by_label(X_after, f.coef, f.columns)

    draws: dict[str, list[float]] = {lab: [] for lab in point_after}
    n = sub.X.shape[0]
    blk = block_length(n)
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        try:
            fb = fit(sub.select(_resample_index(n, blk, rng)), lam=f.lam)
        except ValueError:
            continue
        wb = _watts_by_label(X_before, fb.coef, fb.columns)
        wa = _watts_by_label(X_after, fb.coef, fb.columns)
        for lab in draws:
            draws[lab].append(wa.get(lab, 0.0) - wb.get(lab, 0.0))

    tail = (1.0 - ci) / 2.0 * 100.0
    changes: list[Change] = []
    for lab in point_after:
        b, a = point_before.get(lab, 0.0), point_after[lab]
        vals = np.asarray(draws[lab], dtype=float)
        lo, hi = (a - b, a - b)
        if vals.size:
            lo = float(np.percentile(vals, tail))
            hi = float(np.percentile(vals, 100.0 - tail))
        changes.append(Change(lab, b, a, a - b, lo, hi))
    changes.sort(key=lambda c: -abs(c.delta))

    dt = ds.globals_.get("dt")

    def mins(mask: np.ndarray) -> float:
        if dt is None:
            return float(mask.sum())
        return float(np.nansum(dt[mask])) / 60.0

    n_draws = max((len(v) for v in draws.values()), default=0)
    if n_draws == 0:
        notes.append("no resample produced a usable fit; ranges are point values")

    return Comparison(
        before_windows=int(before.sum()),
        after_windows=int(after.sum()),
        before_minutes=mins(before),
        after_minutes=mins(after),
        measured_before=float(np.nanmean(ds.y[before])),
        measured_after=float(np.nanmean(ds.y[after])),
        changes=changes,
        ci=ci,
        block=blk,
        n_draws=n_draws,
        profile=profile,
        notes=notes,
    )


def _arrow(d: float) -> str:
    return "up" if d > 0 else "down" if d < 0 else "--"


def render_comparison(c: Comparison, n: int = 10) -> str:
    out = [
        f"  earlier: {c.before_minutes:6.1f} min  ({c.before_windows} windows)  "
        f"{c.measured_before:6.2f} W",
        f"  recent:  {c.after_minutes:6.1f} min  ({c.after_windows} windows)  "
        f"{c.measured_after:6.2f} W",
        "",
        f"  package power changed by {c.measured_delta:+.2f} W"
        + (f"   [{c.profile}]" if c.profile else ""),
        "",
    ]

    if not c.trustworthy:
        out.append(
            "  ! The applications account for "
            f"{c.explained*100:.0f}% of that change, so most of it came from "
            "something\n    they did not do — a device, the panel, heat, or a "
            "counter this tool does not\n    read. Treat the ranking below as "
            "a partial account, not the answer.\n"
        )

    real = [ch for ch in c.changes if ch.real][:n]
    if real:
        out.append(
            f"  {'BEFORE':>7}  {'AFTER':>7}  {'CHANGE':>8}  "
            f"{'RANGE':>17}  APPLICATION"
        )
        for ch in real:
            out.append(
                f"  {ch.before:7.3f}  {ch.after:7.3f}  {ch.delta:+8.3f}  "
                f"  [{ch.lo:+.3f}, {ch.hi:+.3f}]  {ch.label} "
                f"({_arrow(ch.delta)})"
            )
    else:
        out.append(
            "  No application's change was larger than the model's own "
            "uncertainty."
        )

    quiet = len(c.changes) - len([ch for ch in c.changes if ch.real])
    out.append("")
    out.append(
        f"  attributed: {c.attributed_delta:+.2f} W    "
        f"unexplained: {c.unexplained:+.2f} W"
    )
    out.append(
        "  Unexplained is the same work costing more or less — heat, "
        "throttling, a device\n  waking up. It is deliberately not charged to "
        "any application."
    )
    if quiet:
        out.append("")
        out.append(
            f"  {quiet} other application(s) moved by less than the "
            f"{c.ci*100:.0f}% interval on their\n  own change, which is not "
            "evidence that anything happened."
        )
    for note in c.notes:
        out.append(f"  note: {note}")
    return "\n".join(out)
