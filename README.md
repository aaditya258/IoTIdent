# Operating-point transfer for open-world IoT device identification

Code, measurement records and manuscript source for a study of what a rejection
threshold actually delivers after the monitor that owns it has moved.

A detector configured to accept 95% of known devices accepts **49.9%** on a
second network. The area under the ROC curve for that same comparison reads
**0.918**. Threshold-free scores do not see the failure, because they average
over every threshold the model might have used and a deployed monitor uses one.

> This repository accompanies a manuscript under review at *Computer Networks*.
> Citation details will be added on publication.

## What is measured

For each deployment shift, two quantities:

| Quantity | Meaning |
|---|---|
| **Δ_A** | displacement of the realised known-acceptance from its nominal setting. Zero means the stored threshold still buys what it was bought for. |
| **R@τ\*** | unknown-rejection that survives once the threshold is refitted on the shifted condition. It separates loss that recalibration recovers from loss that no threshold on that condition recovers. |

Four shifts: a different network, five months of elapsed time, reduced handshake
visibility, and an unseen physical unit of a product already known.

## Layout

```
src/vaid/          the measurement framework (22 modules)
  customflow.py      reads the Custom Flow Parquet format into windows
  transport.py       operating-point transfer: Δ_A, R@τ*, the verdicts
  openset.py         prototype and energy rejection layers
  splits.py          leakage-resistant partitions, including instance held-out
  audit.py           fail-closed forbidden-feature audit
  selftest.py        17 guard-rail checks; run this first
run/               laptop runbook: probe, full-span driver, flow counter
figures/           figure scripts and the rendered PDFs used in the paper
paper/             main_CN.tex, refs_CN.bib, verify_claims.py
results/           measurement records (JSON) and run logs
```

## Data

The capture data is third-party, roughly 30 GB, and is **not** redistributed
here. It is DATA2025 by Sivanathan *et al.*, released under CC0 on Dryad:
[doi:10.5061/dryad.7h44j1098](https://doi.org/10.5061/dryad.7h44j1098).
Download it, point `--data` at the folder holding the per-network directories,
and the pipeline builds its own window cache.

## Reproducing the measurement

```bash
pip install -r requirements.txt
export PYTHONPATH=src            # Windows: set PYTHONPATH=src
python -m vaid.selftest          # 17 checks, all should pass

python run/probe.py --data <DATA_ROOT> --root .
```

`probe.py` reports what is on disk, times a real capture day from each network,
and recommends the sampling rate and per-class cap for your machine. Then:

```bash
python -m vaid.cf_cli build     --data <DATA_ROOT> --root . --min-day-size-frac 0.25
python -m vaid.cf_cli audit     --data <DATA_ROOT> --root . --min-day-size-frac 0.25
python -m vaid.cf_cli threshold --data <DATA_ROOT> --root . --min-day-size-frac 0.25 \
                                --methods rf --repeats 5
```

On Windows, `run/run_all.ps1` chains the whole sequence with per-step logs and
resume markers. Expect six to ten hours, nearly all of it in the build, which is
cached afterwards.

## Choices that affect the numbers

Four are worth knowing before reading the code.

**Upstream-only flows, first five packets.** One capture records both directions
and the other records only upstream. Admitting downstream features where they
exist inflates the apparent difference between the networks by 0.084 macro-F1 —
exporter mismatch that reads as a site effect. `Config.cf_upstream_only` stays
true for any cross-network claim.

**Truncated capture days are dropped in code.** `--min-day-size-frac 0.25`
removes days far smaller than their neighbours and names them in the log, which
is what makes the exclusion reproducible by someone else.

**Scores are checked for applicability.** Free energy is identically zero for a
model that emits normalised posteriors, and prototype distance needs an
embedding. An inapplicable score is refused with a reason rather than reported;
an early version of this experiment printed τ = −0.0000 and a chance-level AUROC
for both, which reads as a finding rather than a bug.

**An operating point must be worth transporting.** A condition is reported only
when the source detector rejects at least half of its unknowns. Without that
guard, a detector that separates almost nothing reports a perfect transport.

## Checking the paper against the records

Every number in the manuscript can be checked without the capture data:

```bash
python figures/derive_transport.py   # rebuilds results/derived_transport.json from the per-seed files
python paper/verify_claims.py        # 147 checks; prints any mismatch
```

## Measurement records

`results/` holds the JSON written by each experiment, including per-seed detail,
and `results/logs/` holds the run logs. The logs matter as much as the JSON:
they record which capture days were dropped, which device types were capped, and
what the leakage audit found. Local paths have been rewritten to `<DATA_ROOT>`
and `<REPO_ROOT>`.

## Figures

Both figures are produced from the result JSON by the scripts beside them, with
no drawing package:

```bash
python figures/fig_process_flow.py --out figures/fig_process_flow
python figures/fig_threshold_displacement.py --root .
```
