<#
  Full-span DATA2025 run — steps 2 to 6 of claude/the-ten-steps.md
  ================================================================

      powershell -ExecutionPolicy Bypass -File run\run_all.ps1

  Edit $Data / $Root below if your paths differ.

  Resumable: every step writes results\logs\<step>.log and a .done marker.
  Re-running skips finished steps, so a crash costs one step, not the run.
  Delete a .done file to force that step again. Ctrl-C is safe.

  v2, after the step4_ksens_K3 crash. Three things changed:

    1. Each experiment that needs a NEW cache now gets its own `build` step
       first, in a SEPARATE PROCESS. Building 273k windows and then
       materialising them in the same process holds two large allocations at
       once; splitting them halves peak memory. That is what killed K=3.
    2. stderr is captured as text, so a Python traceback reaches the log
       instead of being eaten by PowerShell's NativeCommandError wrapper.
       Logs are UTF-8, not UTF-16.
    3. The system drive is checked before anything runs. Windows grows the
       pagefile there, so a full C: turns a large allocation into a
       MemoryError with no useful message.
#>

# ---------------------------------------------------------------- settings --
$Data = "<DATA_ROOT>"
$Root = "<REPO_ROOT>"

$Every = 1
$MinDaySizeFrac = 0.25

# Capping policy, stated once so the paper can state it once:
#   * experiments that use ONLY the random forest run on all 273,037 windows;
#   * experiments that involve the NEURAL model cap each class at $CapPerClass
#     windows by whole adjacency group.
# The cap is a cost control for the neural arm, which is refit 20 times by the
# threshold command. Capping by group keeps the grouped split honest, and it
# preserves every class and the full calendar span (verified: v1 still 4
# months, v2 still 10). It also balances a 29,565-to-3 class imbalance that
# otherwise lets one class dominate a macro average.
$CapPerClass = 2200

$Seeds  = 5
$Folds  = 5
$Epochs = 40

# K values for the sensitivity sweep. Each K needs its own ~7 h cache build,
# so this is the most expensive thing per unit of evidence in the whole run.
# At full span K=3 and K=5 already agree to 0.003 in BOTH directions, which is
# the invariance the sweep exists to demonstrate; add 8 only if a reviewer asks
# for a third point.
$KValues = @(3, 5)
$MinFreeGb = 3      # hard stop below this on the system drive

# -------------------------------------------------------------- scaffolding --
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
Set-Location $Root

# Keep scratch off the system drive. scikit-learn fits the forest with
# n_jobs=-1 and joblib memory-maps the feature matrix into %TEMP% for each
# worker; at 273k windows that is ~0.5 GB a copy, and the folders are only
# cleaned up on a clean exit.
$ScratchDir = Join-Path (Split-Path $Root -Qualifier) "\tmp"
New-Item -ItemType Directory -Force -Path $ScratchDir | Out-Null
$env:TEMP = $ScratchDir
$env:TMP  = $ScratchDir
$env:JOBLIB_TEMP_FOLDER = $ScratchDir

$LogDir = Join-Path $Root "results\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Get-SystemFreeGb {
    $letter = (Split-Path $env:SystemRoot -Qualifier).TrimEnd(":")
    [math]::Round((Get-PSDrive -Name $letter).Free / 1GB, 1)
}

function Invoke-Step {
    param([string]$Name, [string[]]$CliArgs)

    $done = Join-Path $LogDir "$Name.done"
    $log  = Join-Path $LogDir "$Name.log"
    if (Test-Path $done) {
        Write-Host "[skip] $Name" -ForegroundColor DarkGray
        return
    }

    $free = Get-SystemFreeGb
    if ($free -lt $MinFreeGb) {
        Write-Host ""
        Write-Host "[STOP] only $free GB free on the system drive." -ForegroundColor Red
        Write-Host "       Windows grows the pagefile there. A large allocation will" -ForegroundColor Red
        Write-Host "       fail as a MemoryError. Free space, then re-run this script;" -ForegroundColor Red
        Write-Host "       finished steps are skipped." -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "[run ] $Name   $(Get-Date -Format 'HH:mm:ss')   (C: $free GB free)" -ForegroundColor Cyan
    Write-Host ("  python -m vaid.cf_cli " + ($CliArgs -join " "))
    Write-Host ("=" * 70)

    $t0 = Get-Date
    # ErrorActionPreference must relax around a native call: with "Stop",
    # anything python writes to stderr becomes a terminating NativeCommandError
    # and PowerShell truncates the traceback. Stringifying each record keeps the
    # text; Out-File -Encoding utf8 keeps the log readable.
    $ErrorActionPreference = "Continue"
    & python -m vaid.cf_cli @CliArgs 2>&1 |
        ForEach-Object { $line = "$_"; Write-Host $line; $line } |
        Out-File -FilePath $log -Encoding utf8
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

    if ($code -ne 0) {
        Write-Host "[FAIL] $Name exited $code after $mins min." -ForegroundColor Red
        Write-Host "       Traceback is at the end of $log" -ForegroundColor Red
        Write-Host "       Fix the cause and re-run; finished steps are skipped."
        exit $code
    }
    "completed $(Get-Date -Format s) in $mins min" | Set-Content $done
    Write-Host "[ok  ] $Name in $mins min" -ForegroundColor Green
}

$Common = @("--data", $Data, "--root", $Root,
            "--every", $Every, "--min-day-size-frac", $MinDaySizeFrac)
$Capped = $Common + @("--cap-per-class", $CapPerClass)

Write-Host ""
Write-Host "Full-span DATA2025 run (v2)" -ForegroundColor Yellow
Write-Host "  data    $Data"
Write-Host "  repo    $Root"
Write-Host "  scratch $ScratchDir"
Write-Host "  C: free $(Get-SystemFreeGb) GB   (hard stop below $MinFreeGb GB)"
Write-Host ""

# -- STEP 2 -- build and cache. Already done; kept so a fresh clone works.
Invoke-Step "step2_build" ($Common + @("build"))

# -- STEP 3 -- audit (uncapped: a leakage check should see every window),
#    then the closed-set baseline (capped: it trains the TCN).
Invoke-Step "step3_audit"    ($Common + @("audit"))
Invoke-Step "step3_baseline" ($Capped + @("baseline", "--folds", $Folds,
                                          "--epochs", $Epochs))

# -- STEP 4 -- cross-network. Forest only, so uncapped on all 273k windows.
#    Renamed from "step4_cross" on purpose: the earlier run's JSON was
#    overwritten by the K sweep (all three wrote one fixed filename), so this
#    has to produce its own tagged file. ~4 minutes.
Invoke-Step "step4_cross_FHC" ($Common + @("cross", "--method", "rf"))

# -- STEP 5 -- THE CENTRAL RESULT: threshold transport.
#    Deliberately ahead of the K sweep. The K sweep is a methods-validity
#    check that costs a ~3.5 h rebuild for K=8; the threshold result is the
#    paper. Bank the paper first, so an overnight failure costs a validity
#    check and not the contribution.
#    Forest arm uncapped; neural arm capped. Separate JSONs, each quotable on
#    its own terms. This is the heaviest step: 20 refits (5 seeds x 4 axes).
Invoke-Step "step5_threshold_rf" ($Common + @("threshold",
                                              "--methods", "rf",
                                              "--repeats", $Seeds))
Invoke-Step "step5_threshold_gated" ($Capped + @("threshold",
                                                 "--methods", "gated",
                                                 "--repeats", $Seeds,
                                                 "--epochs", $Epochs))

# -- STEP 6 -- instance-held-out: product or box? Forest only, uncapped.
Invoke-Step "step6_instance" ($Common + @("instance", "--method", "rf"))

# -- STEP 7 -- K-sensitivity, last because it is the cheapest to lose.
#    K is baked into the sequence features, so each K needs its own cache.
#    ONE DATASET PER PROCESS. load_records accumulates every dataset it reads,
#    so building v2 while v1 is still in memory meant 273k records resident at
#    the moment of the final pickle -- which is what raised MemoryError seven
#    hours into the K=8 build. Building them separately halves peak memory.
foreach ($k in $KValues) {
    foreach ($ds in "DATA2025v1", "DATA2025v2") {
        Invoke-Step ("step7_ksens_K{0}_build_{1}" -f $k, $ds) `
            ($Common + @("build", "--datasets", $ds, "--upstream-packets", $k))
    }
    Invoke-Step ("step7_ksens_K{0}" -f $k) ($Common + @("cross", "--method", "rf",
                                                        "--views", "F", "C",
                                                        "--upstream-packets", $k))
}

# ------------------------------------------------------------------ wrap-up --
Write-Host ""
Write-Host ("=" * 70)
Write-Host "ALL STEPS COMPLETE" -ForegroundColor Green
Write-Host ("=" * 70)
Get-ChildItem (Join-Path $Root "results") -Filter "cf_*.json" |
    Sort-Object LastWriteTime |
    ForEach-Object { Write-Host ("  {0,-44} {1,8:N0} KB  {2}" -f $_.Name, ($_.Length / 1KB), $_.LastWriteTime) }
Write-Host ""
Write-Host "Send back:  results\cf_*.json   results\probe.json   results\logs\*.log"
