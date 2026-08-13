# Stormbreaker

Per-process battery attribution for Linux, learned per machine.

Tells you what is actually draining your battery, in watts rather than CPU
percent:

```
    WATTS   SHARE     CPU     GPU   IO MB/s                APPLICATION
    5.323   23.4%   1.094   0.000      0.00  ###.........  sb-beta
    5.001   21.9%   1.224   0.000      0.00  ###.........  sb-alpha
    1.798    7.9%   0.117   0.009      0.00  #...........  code
    1.653    7.3%   0.154   0.005      0.08  #...........  google-chrome
    1.639    7.2%   0.141   0.000      0.20  #...........  org.chromium.Chromium
    0.962    4.2%   0.037   0.000      0.00  #...........  plasma-kwin_wayland
    5.238   23.0%                            ###.........  [idle baseline — attributable to nobody]
```

## Why this needs a model at all

You cannot measure per-process power. The hardware reports one number for the
whole package. Windows and macOS solve this with proprietary attribution
models; powertop on Linux uses hard-coded constants from 2010-era hardware.

So the coefficients have to be learned, per machine, because a 15 W ultrabook
and a 45 W workstation have completely different cost structures. Stormbreaker
regresses the one number the hardware *does* give you onto per-cgroup activity
vectors:

```
measured_watts[t] = baseline + SUM_k SUM_j  w[k,j] * activity[t,k,j]
```

What makes the coefficients physically meaningful rather than merely predictive
is the constraint set:

- **Non-negativity.** Nothing consumes negative power. Without this, collinear
  features produce large positive and negative coefficients that cancel — which
  predicts well and attributes nonsense.
- **A free baseline.** The static draw of an idle machine belongs to nobody, so
  it is fitted as an unpenalised intercept and cannot be smeared across
  applications.
- **`baseline <= min(observed power)`.** Exact, and follows directly from every
  other term being non-negative. An earlier build violated it — claiming a
  7.66 W idle floor on a machine never seen drawing under 5.24 W.
- **An activity floor.** A process using a thousandth of a core cannot have its
  cost identified. Such columns are dropped rather than fitted.
- **Ridge scaled per feature *kind*, not per column.** Textbook per-column
  normalisation is actively wrong here: it rescales a nearly-idle application's
  near-empty column up to parity with a saturated one, so the penalty stops
  restraining its watts-per-core. That is how a daemon doing nothing was once
  assigned 23,938 W per core. Columns of one kind share a scale, so the penalty
  expresses the prior actually held — no application should have an extreme cost
  *per unit of work* relative to its peers.

Solved with bounded-variable least squares (`scipy.optimize.lsq_linear`), which
is NNLS plus the upper bounds. Fits in milliseconds on a few thousand windows.

## What it collects

Locally, into SQLite, a few MB per week. Nothing leaves the machine.

**Energy targets**, in order of preference — the first readable one wins:

| Source | Notes |
|---|---|
| `powercap` RAPL `energy_uj` | Most direct, but **root-only** on kernels carrying the CVE-2020-8694 mitigation |
| hwmon package power (`power1_average`) | Same silicon, usually **world-readable**. On AMD APUs this is the SMU's PPT sensor |
| battery `power_now`, or `current_now * voltage_now` | Whole system including panel and radios; used as cross-check and for validation |

**Per-cgroup features**, all from cgroup v2 leaves so the tree is partitioned
exactly once:

| Feature | Source |
|---|---|
| CPU seconds, bucketed by clock frequency | `cpu.stat`, bucketed by sampled CPU frequency |
| Bytes read/written | `io.stat`, with device-mapper layers skipped to avoid double counting |
| Context switches | `/proc/PID/status`, differenced per pid |
| GPU engine time | `drm-engine-*` in `/proc/PID/fdinfo`, differenced per DRM client |

Frequency bucketing matters because the energy cost of a busy core is strongly
superlinear in clock: a core-second at 5 GHz and one at 1.2 GHz are different
goods and must not share a coefficient.

Counters that cannot be differenced in aggregate are not. Context switches and
GPU time are differenced per-pid and per-client respectively, because a process
exiting makes an aggregate total *fall* and a process starting makes it *jump* —
both of which corrupt a naive delta.

## Not implemented

- **Per-cgroup network packets.** Needs an eBPF program; there is no procfs
  path to per-cgroup packet counts. The spec called for it; it is absent.
- **CPU frequency residency by bucket.** `cpufreq` `stats/time_in_state` is not
  exported by `amd-pstate`, so frequency is sampled per window rather than
  integrated over residency. Coarser, but available.

## Install

```sh
pip install -e .          # numpy, scipy
pip install -e '.[plot]'  # adds matplotlib for the discharge plot
```

## Use

```sh
stormbreaker caps                        # what this machine exposes
stormbreaker collect --window 5 -v       # sample into ~/.local/share/stormbreaker
stormbreaker top                         # rank applications by watts
stormbreaker report                      # battery report in minutes of life
stormbreaker coefs                       # inspect the learned coefficients
stormbreaker validate --plot out.png     # check the model against reality
```

By default each command fits a fresh model from the stored windows. To fit once
and reuse it:

```sh
stormbreaker fit                         # fit and store the model
stormbreaker top --saved                 # reuse it instead of refitting
```

A stored model is a vector of coefficients whose meaning is positional, and the
set of running applications changes between runs, so `--saved` re-aligns the
data onto the model's columns by name. Applications the model has never seen
contribute zero and are reported as unattributed, rather than borrowing whatever
coefficient happens to sit at their column index.

For continuous collection, `packaging/stormbreaker.service` is a user unit:

```sh
cp packaging/stormbreaker.service ~/.config/systemd/user/
systemctl --user enable --now stormbreaker
```

It needs no privileges unless you want RAPL specifically.

## Validation

Two checks, in increasing order of what they prove.

**Held-out package power** works any time, plugged in or not. Fit on the first
part of the record, predict the last part.

**Discharge-curve tracking** is the real one. Take an unplugged session, fit on
its first half, predict the second half's battery trajectory without looking at
it, and compare against the fuel gauge. The gauge's charge reading is an
integral measurement accumulated independently of every counter being regressed
on, so agreement is not something the fit can manufacture.

### Status: unproven

The discharge check has **never been run against real data** — the development
machine was on AC for the whole session, so it has only ever taken its
no-unplugged-data path. Until it runs, the attribution is plausible but not
demonstrated. Treat the numbers accordingly.

What *has* been measured, on one machine (AMD Ryzen AI 7 350, Radeon 860M,
Fedora, kernel 7.0.10), over 5.4 minutes and 159 windows:

- R² 0.896, MAE 2.92 W, against package power ranging 5.24–48.85 W
- Idle baseline 5.238 W, sitting exactly on its physical bound
- Collector cost **13 ms per snapshot** — 0.26% duty cycle at a 5 s window

Cross-check on the learned CPU cost: ~5.5 W per busy core × 8 cores + 5.24 W
baseline ≈ 49 W, against a 48.85 W peak actually measured under an 8-thread
load. Two synthetic loads pinned in their own cgroups at 2 and 6 cores were
recovered as the top two consumers.

Five minutes is a small sample and one machine is one machine.

## Reading the output

Coefficients are costs *per unit of activity*, and may legitimately exceed total
package power when the feature never approaches one full unit — a process using
a hundredth of a GPU can honestly cost 100 W per fully-busy-GPU-second, since
that rate is only ever evaluated at a hundredth of it. Clamping such
coefficients to the observed maximum sounds physical, but assumes a saturated
unit was actually observed; doing so measurably degrades the fit when it was
not.

Ridge shrinkage biases application coefficients downward, and the unpenalised
baseline absorbs the difference. Attribution is therefore **conservative**:
watts move from applications to the unattributable floor, never the other way.

## Tests

```sh
python -m pytest tests/ -q
```

28 tests. The interesting ones are not about predictive accuracy — plain least
squares predicts fine — but about whether the *coefficients* land on the truth,
since that is what gets shown to a user as "Slack costs you 1.9 W". The rest
cover counter differencing under process churn, which is where naive arithmetic
breaks.
