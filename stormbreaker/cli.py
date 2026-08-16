"""Command line interface."""

from __future__ import annotations

import argparse
import sys
import time

from . import __version__
from .caps import probe
from .collect import run_collect
from .model import (
    align_to_fit,
    coefficient_table,
    fit,
    load_dataset,
    load_fit,
    save_fit,
    unknown_labels,
)
from .report import load_and_report, render_daily, render_top
from .store import DEFAULT_DB, Store
from .validate import plot_discharge, validate_discharge, validate_holdout


def _since(minutes: float | None) -> float | None:
    return None if minutes is None else time.time() - minutes * 60.0


def _reload(store, args):
    return load_dataset(
        store,
        since=_since(getattr(args, "minutes", None)),
        target=getattr(args, "target", None),
        top_n=getattr(args, "top_n", None),
        n_buckets=getattr(args, "buckets", None),
    )


def _load(args, min_minutes: float | None = None):
    store = Store(args.db)
    if store.window_count() == 0:
        print(
            f"No data in {args.db}.\n"
            f"Run:  stormbreaker collect --duration 300 -v",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return store, _reload(store, args)


def _fit_for(args, store, ds):
    """Either reuse the stored model or fit a fresh one.

    Reusing returns the dataset re-aligned onto the stored model's columns, so
    the caller must use the dataset this hands back rather than the one it
    passed in.
    """
    if not getattr(args, "saved", False):
        return ds, fit(ds, lam=args.lam)

    hit = load_fit(store)
    if hit is None:
        print(
            "No stored model. Run `stormbreaker fit` first, or drop --saved to "
            "fit on the fly.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    f, age = hit
    missing = unknown_labels(ds, f)
    if missing:
        shown = ", ".join(sorted(missing)[:4])
        print(
            f"note: {len(missing)} application(s) started since the model was "
            f"fitted and are\n      not attributed: {shown}"
            f"{' ...' if len(missing) > 4 else ''}. Re-run `stormbreaker fit` "
            "to include them.",
            file=sys.stderr,
        )
    print(f"using stored model, fitted {age/3600:.1f} h ago", file=sys.stderr)
    return align_to_fit(ds, f), f


def cmd_fit(args) -> int:
    store, ds = _load(args)
    f = fit(ds, lam=args.lam)
    save_fit(store, f)
    print(
        f"fitted {f.n_windows} windows against {f.target}: "
        f"R^2={f.r2:.4f}, MAE={f.mae:.3f} W, lambda={f.lam:g}"
    )
    print(f"idle baseline {f.baseline:.3f} W over {len(f.labels)} applications")
    print(f"saved to {args.db}")
    return 0


def cmd_caps(args) -> int:
    caps = probe()
    print("stormbreaker capability probe")
    print(caps.describe())
    if not caps.rapl:
        print(
            "\nRAPL is unreadable by this user. The hwmon package sensor is used "
            "instead;\nit measures the same silicon and needs no privileges."
        )
    return 0


def cmd_collect(args) -> int:
    return run_collect(
        args.db,
        window_s=args.window,
        duration_s=args.duration,
        verbose=args.verbose,
        gpu_fdinfo=not args.no_gpu,
        refit_every_s=args.refit_every,
        rolling_hours=args.rolling_hours,
    )


def cmd_top(args) -> int:
    store, ds = _load(args)
    if not args.watch:
        ds, f = _fit_for(args, store, ds)
        rows, ctx, sysmod = load_and_report(store, ds, f)
        print()
        print(render_top(rows, ctx, f, sysmod, args.number))
        print()
        return 0

    # Watch mode refits at most once per --refit-watch seconds. Refitting every
    # frame would make the tool a heavier consumer than most of what it ranks,
    # which for a battery tool would be self-defeating.
    cached: tuple = ()
    last_fit = 0.0
    try:
        while True:
            ds = _reload(store, args)
            now = time.monotonic()
            if not cached or now - last_fit >= args.refit_watch:
                cached = _fit_for(args, store, ds)
                last_fit = now
            _ds, f = cached
            rows, ctx, sysmod = load_and_report(store, _reload(store, args), f)
            stamp = time.strftime("%H:%M:%S")
            print("\033[2J\033[H", end="")
            print(f"  stormbreaker top — {stamp}   (ctrl-c to exit)")
            print(render_top(rows, ctx, f, sysmod, args.number))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


def cmd_report(args) -> int:
    store, ds = _load(args)
    ds, f = _fit_for(args, store, ds)
    rows, ctx, sysmod = load_and_report(store, ds, f)
    print("\nStormbreaker battery report")
    print("=" * 60)
    print(render_daily(rows, ctx, f, sysmod, n=args.number))
    print()
    return 0


def cmd_coefs(args) -> int:
    store, ds = _load(args)
    ds, f = _fit_for(args, store, ds)
    print(f"\nfitted on {f.n_windows} windows, target={f.target}, lambda={f.lam:g}")
    print(f"R^2={f.r2:.4f}  MAE={f.mae:.3f} W  RMSE={f.rmse:.3f} W")
    if f.freq_edges:
        edges = ", ".join(f"{e:.2f}" for e in f.freq_edges)
        print(f"frequency bucket edges (GHz): {edges}")
    print(f"idle baseline: {f.baseline:.3f} W\n")
    print(
        f"  {'COEF':>10}  {'ACT p95':>9}  {'W @ p95':>8}  {'W avg':>7}  "
        f"{'FEATURE':<9}  APPLICATION"
    )
    rows = coefficient_table(f, ds)[: args.number]
    for r in rows:
        mark = " *" if r.extrapolated else "  "
        print(
            f"  {r.coef:10.3f}  {r.activity_p95:9.4f}  {r.watts_p95:8.3f}  "
            f"{r.watts_mean:7.3f}  {r.feature:<9}{mark} {r.label}"
        )
    print(
        "\nunits: cpu@N = W per busy core in frequency bucket N, "
        "io_mb = W per MB/s,\n       gpu = W per busy GPU-second/s, "
        "ctxt_k = W per 1000 context switches/s"
    )
    if any(r.extrapolated for r in rows):
        print(
            "\n  * identified below a tenth of one full unit. The rate is real but "
            "the\n    machine was never observed near that regime — read 'W avg', "
            "not 'COEF'."
        )
    return 0


def cmd_validate(args) -> int:
    store, ds = _load(args)

    print("\nStormbreaker validation")
    print("=" * 60)

    try:
        h = validate_holdout(ds, holdout=args.holdout)
        print("\n1. Held-out package power (works on AC)")
        print(f"   trained on {h.n_train} windows, tested on {h.n_test}")
        print(f"   R^2  {h.r2:+.4f}")
        print(f"   MAE  {h.mae:.3f} W   (predict-the-mean baseline: {h.naive_mae:.3f} W)")
        print(f"   RMSE {h.rmse:.3f} W")
        print(
            f"   mean measured {h.mean_measured:.2f} W  vs  "
            f"predicted {h.mean_predicted:.2f} W"
        )
    except ValueError as e:
        print(f"\n1. Held-out package power: skipped ({e})")

    print("\n2. Discharge-curve tracking (needs an unplugged session)")
    charge_based = store.get_meta("battery_charge_based") == "1"
    res = validate_discharge(ds, charge_based=charge_based, holdout=args.holdout)
    if res is None:
        print(
            "   No usable unplugged segment yet. Unplug the machine and let the\n"
            "   collector run for ~20 minutes, then re-run this command. This is\n"
            "   the check that actually proves the attribution, so it is worth it."
        )
    else:
        print(f"   trained on {res.n_train} windows, predicted {res.n_test} forward")
        print(f"   held-out span            {res.hours*60:.1f} min")
        print(f"   MAE vs fuel gauge        {res.mae_wh:.4f} Wh")
        print(
            f"   error at end of segment  {res.final_error_wh:+.4f} Wh "
            f"({res.final_error_pct:+.1f}% of energy drawn)"
        )
        print(
            f"   runtime estimate         {res.predicted_runtime_h:.2f} h predicted "
            f"vs {res.measured_runtime_h:.2f} h measured"
        )
        print(
            f"   system model             {res.sysmod.intercept:.2f} W + "
            f"{res.sysmod.slope:.2f} x package (R^2={res.sysmod.r2:.3f})"
        )
        if args.plot:
            if plot_discharge(res, args.plot):
                print(f"\n   plot written to {args.plot}")
            else:
                print("\n   matplotlib not installed; skipping plot "
                      "(pip install 'stormbreaker[plot]')")
    print()
    return 0


def cmd_prune(args) -> int:
    store = Store(args.db)
    n = store.prune(args.keep_days)
    store.db.execute("VACUUM")
    print(f"removed {n} windows older than {args.keep_days} days")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="stormbreaker",
        description="Per-process battery attribution, learned on this machine.",
    )
    p.add_argument("--version", action="version", version=f"stormbreaker {__version__}")
    p.add_argument("--db", default=DEFAULT_DB, help="sqlite database path")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("caps", help="show what this machine exposes")
    sp.set_defaults(func=cmd_caps)

    sp = sub.add_parser("collect", help="sample counters and energy into the database")
    sp.add_argument("--window", type=float, default=5.0, help="window length, seconds")
    sp.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    sp.add_argument("--no-gpu", action="store_true", help="skip per-process GPU scan")
    sp.add_argument("--refit-every", type=float, default=600.0, dest="refit_every",
                    help="seconds between background refits (0 disables)")
    sp.add_argument("--rolling-hours", type=float, default=4.0, dest="rolling_hours",
                    help="trailing window the background refit uses")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_collect)

    def add_model_args(sp):
        sp.add_argument("--minutes", type=float, default=None,
                        help="only use the last N minutes of data")
        sp.add_argument("--lam", type=float, default=None,
                        help="ridge strength (default: chosen by hold-out)")
        sp.add_argument("--top-n", type=int, default=None, dest="top_n",
                        help="how many cgroups get their own columns "
                             "(default: sized to the data)")
        sp.add_argument("--buckets", type=int, default=None,
                        help="CPU frequency buckets (default: sized to the data)")
        sp.add_argument("--target", default=None,
                        help="energy column to regress on (soc_w | rapl_pkg_w)")

    def add_saved_arg(sp):
        sp.add_argument("--saved", action="store_true",
                        help="reuse the model stored by `stormbreaker fit`")

    sp = sub.add_parser("fit", help="fit a model and store it in the database")
    add_model_args(sp)
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser("top", help="rank applications by watts")
    add_model_args(sp)
    add_saved_arg(sp)
    sp.add_argument("-n", "--number", type=int, default=15)
    sp.add_argument("-w", "--watch", action="store_true",
                    help="refresh continuously instead of printing once")
    sp.add_argument("--interval", type=float, default=3.0,
                    help="seconds between refreshes in watch mode")
    sp.add_argument("--refit-watch", type=float, default=60.0, dest="refit_watch",
                    help="seconds between refits in watch mode")
    sp.set_defaults(func=cmd_top)

    sp = sub.add_parser("report", help="battery report in minutes of life")
    add_model_args(sp)
    add_saved_arg(sp)
    sp.add_argument("-n", "--number", type=int, default=3)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("coefs", help="show the learned coefficients")
    add_model_args(sp)
    add_saved_arg(sp)
    sp.add_argument("-n", "--number", type=int, default=25)
    sp.set_defaults(func=cmd_coefs)

    sp = sub.add_parser("validate", help="check the model against reality")
    add_model_args(sp)
    sp.add_argument("--holdout", type=float, default=0.4)
    sp.add_argument("--plot", default=None, help="write a PNG of the discharge curve")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("prune", help="drop old windows")
    sp.add_argument("--keep-days", type=float, default=30.0)
    sp.set_defaults(func=cmd_prune)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
