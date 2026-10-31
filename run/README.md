# Laptop run plan — steps 1 to 6

Everything here runs on your machine, because your machine is where the ~29 GB
of DATA2025 lives. Steps 7 to 10 (external baseline, significance tests,
figures, manuscript) happen on the Claude side and need only the JSON files this
produces.

## Before you start

```powershell
pip install psutil          # optional, but the probe's RAM check needs it
python -c "import pyarrow, sklearn, torch; print('deps ok')"
```

`torch` is needed only for the neural arm. If it is missing, everything still
runs — the framework falls back to a shallow model and says so.

## Step 0 — five minutes of housekeeping

Three things in `<DATA_ROOT>`:

1. **`ae-numeric\` is extracted wrong.** It is triple-nested:
   `ae-numeric\ae-numeric\ae-numeric\blocks4_latent40_epochs100\final_model`,
   and it also holds a stray copy of `ae-numeric-limited`. Flatten it so the
   layout matches the other four encoders. Not needed until step 7, but easier
   to fix now than to rediscover later.
2. **Reclaim ~1.3 GB**: the nested `ae-numeric` duplicates, plus
   `doi_10_5061_dryad_7h44j1098__v20260713\vae-numeric.tar.gz` (538 MB) and the
   already-extracted copy beside it.
3. **Leave the truncated days alone** — do *not* delete `25-10-13` and
   `25-10-14`. The pipeline drops them itself via `--min-day-size-frac 0.25` and
   prints their names into the log, which is what makes the exclusion
   reproducible. Deleting them by hand would make the run unrepeatable by anyone
   else.

## Step 1 — the probe. Run this first.

```powershell
cd "<REPO_ROOT>"
python run\probe.py --data "<DATA_ROOT>" --root "<REPO_ROOT>"
```

A few minutes. It prints, and writes to `results\probe.json`:

* what is on disk — file counts, date ranges, sizes, and which days are
  truncated;
* how long the full build will take, measured by actually building one
  median-sized day from each dataset rather than guessing;
* whether the result will fit in RAM, and a recommended `--every` and
  `--cap-per-class`.

**Read the RECOMMENDATION block and put those two numbers into `run_all.ps1`**
(`$Every`, `$CapPerClass`) if they differ from the defaults.

Why bother: the full span is roughly twenty times the development set we have
been working on. Five minutes here beats discovering a memory problem four hours
into a build.

## Steps 2 to 6 — the run

```powershell
powershell -ExecutionPolicy Bypass -File run\run_all.ps1
```

Then leave it. Expect something like 6 to 10 hours in total, nearly all of it in
step 2.

It is **resumable**. Each step writes `results\logs\<step>.log` and a `.done`
marker; re-running skips finished steps. A crash or a reboot costs you one step.
To force a step again, delete its `.done` file.

What the steps are, and why each is in the paper:

| Step | What it does | Why it is needed |
|---|---|---|
| 2 build | Reads every day, builds 30-minute windows, caches them | Everything else reads this cache |
| 3 audit | Re-checks that no MAC, IP, DNS name or vendor token reached the model, and that no adjacency group straddles a split | Nothing is quotable until this passes |
| 3 baseline | Closed-set macro-F1, 5 folds | Shows the model is competent, so "the threshold fails to transport" is a finding about thresholds and not about a weak model |
| 4 cross | Train on one network, test on the other, with a within-source control | The shift magnitude in familiar closed-set terms |
| 4 K-sensitivity | The same, with K = 3, 5, 8 and views F+C only | Justifies leading with the v2 → v1 direction |
| **5 threshold** | **Calibrate τ on one condition, measure what acceptance it delivers elsewhere** | **The central result** |
| 6 instance | Hold out whole physical units | Product-versus-box; also partly deconfounds step 4 |

Step 5 runs twice on purpose — the forest arm uncapped on the full span, the
neural arm capped — so each JSON is quotable on its own terms. The cap exists
because the threshold command refits its model twenty times (5 seeds × 4 axes);
at full span an uncapped neural fit would take hours each. Capping bounds the
cost **without shortening the calendar span**, which matters because the span is
exactly what the time axis measures.

## When it finishes

Send back:

```
results\cf_*.json
results\probe.json
results\logs\*.log
```

That is everything steps 7 to 10 need. The logs matter as much as the JSONs —
they record which days were dropped, which classes were capped, and what the
audit said.

## If something goes wrong

| Symptom | What to do |
|---|---|
| `ModuleNotFoundError: pyarrow` | `pip install pyarrow` |
| Memory error during step 2 or 3 | Set `$Every = 2` in `run_all.ps1`, delete `results\logs\*.done`, re-run. Say so in the paper. |
| Step 2 is far slower than the probe predicted | Likely another process competing for disk. Check it is not running from an external drive. |
| A step fails | The script stops and names its log. Fix, re-run; finished steps are skipped. |
| `torch` missing and you want the neural arm | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Need to stop | Ctrl-C. Re-run to resume. |
