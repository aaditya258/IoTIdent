#!/usr/bin/env python3
"""Step 1 of the laptop plan: size the job before paying for it.

Run this FIRST. It costs a few minutes and answers the three questions that
decide every setting in `run_all.ps1`:

  1. What is actually on disk?  File counts, date ranges, total size, and any
     truncated captures (a day far smaller than its neighbours is a capture that
     stopped early, not a quiet day).
  2. How long will the full build take?  Times one real day from each dataset,
     both passes, then extrapolates.
  3. Will it fit in RAM?  Measures the per-day window count and peak memory,
     then projects the full-span feature matrices.

It writes `results/probe.json` and prints a recommended command line. Nothing
else in the pipeline depends on it, so it is safe to re-run.

    python run/probe.py --data <DATA_ROOT> --root <REPO_ROOT>

`--data` is the folder holding the per-network capture directories; `--root`
is this repository. On Windows, quote any path that contains a space.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from vaid.config import Config, Paths                      # noqa: E402
from vaid.customflow import (build_windows_customflow, load_device_map,  # noqa: E402
                             _day_files)

MIN_SIZE_FRAC = 0.25
GB = 1024 ** 3


def _peak_rss_mb() -> float:
    """Peak resident memory, by whatever this platform offers."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        pass
    try:                                   # Linux / macOS
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / 1e3 if sys.platform == "darwin" else ru / 1e3
    except Exception:
        return float("nan")


def inventory(root: str, dataset: str) -> dict:
    files = sorted(glob.glob(os.path.join(root, dataset, "*.parquet")))
    if not files:
        return {"dataset": dataset, "n_files": 0}
    sizes = np.array([os.path.getsize(f) for f in files], dtype=float)
    med = float(np.median(sizes))
    stems = [os.path.basename(f).split(".")[0] for f in files]
    trunc = [(s, round(z / 1e6, 1)) for s, z in zip(stems, sizes)
             if z < MIN_SIZE_FRAC * med]
    return {
        "dataset": dataset,
        "n_files": len(files),
        "first_day": stems[0], "last_day": stems[-1],
        "total_gb": round(sizes.sum() / GB, 2),
        "median_mb": round(med / 1e6, 1),
        "min_mb": round(sizes.min() / 1e6, 1),
        "max_mb": round(sizes.max() / 1e6, 1),
        "truncated_days": trunc,
        "n_after_size_filter": int((sizes >= MIN_SIZE_FRAC * med).sum()),
    }


def time_one_day(data_root: str, dataset: str, cfg: Config, device_map: dict,
                 with_handshake: bool) -> dict:
    """Build the median-sized day so the timing is representative."""
    files = _day_files(data_root, dataset, 1, 0, min_size_frac=MIN_SIZE_FRAC,
                       verbose=False)
    if not files:
        return {}
    sizes = [os.path.getsize(f) for f in files]
    pick = files[int(np.argsort(sizes)[len(sizes) // 2])]
    stem = os.path.basename(pick).split(".")[0]

    before = _peak_rss_mb()
    t0 = time.time()
    recs = build_windows_customflow(data_root, dataset, cfg, device_map,
                                    every=1, limit=0, with_handshake=with_handshake,
                                    verbose=False, only_days=(stem,))
    dt = time.time() - t0
    after = _peak_rss_mb()
    return {
        "day": stem,
        "mb": round(os.path.getsize(pick) / 1e6, 1),
        "seconds": round(dt, 1),
        "windows": len(recs),
        "rss_delta_mb": round(after - before, 1) if np.isfinite(after) else None,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser("probe")
    p.add_argument("--data", required=True, help="folder with DATA2025v1/ v2/ scripts/")
    p.add_argument("--root", default=None, help="research repo (cache + results)")
    p.add_argument("--datasets", nargs="+", default=["DATA2025v1", "DATA2025v2"])
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--no-handshake", action="store_true")
    a = p.parse_args(argv)

    cfg = Config()
    cfg.T_minutes = a.T
    paths = Paths(root=a.root) if a.root else Paths()

    print("=" * 74)
    print("1. WHAT IS ON DISK")
    print("=" * 74)
    inv = [inventory(a.data, d) for d in a.datasets]
    total_days = 0
    for v in inv:
        if not v["n_files"]:
            print(f"  {v['dataset']}: NOT FOUND under {a.data}")
            continue
        total_days += v["n_after_size_filter"]
        print(f"  {v['dataset']}: {v['n_files']} files, "
              f"{v['first_day']} .. {v['last_day']}, {v['total_gb']} GB, "
              f"median {v['median_mb']} MB (min {v['min_mb']}, max {v['max_mb']})")
        if v["truncated_days"]:
            names = ", ".join(f"{s} ({z} MB)" for s, z in v["truncated_days"])
            print(f"     TRUNCATED, will be dropped at --min-day-size-frac "
                  f"{MIN_SIZE_FRAC}: {names}")
        else:
            print("     no truncated captures")
    if not total_days:
        print("\nNothing to probe. Check --data.")
        return 2

    device_map = load_device_map(os.path.join(a.data, "scripts", "scripts", "configs")) \
        or load_device_map(os.path.join(a.data, "scripts", "configs"))
    if not device_map:
        print("\nFATAL: no *-config.yaml found under scripts/. Labels need it.")
        return 2
    print(f"\n  device map: {len(device_map)} MAC addresses")

    print()
    print("=" * 74)
    print("2. HOW LONG WILL THE BUILD TAKE")
    print("=" * 74)
    print("  building one median-sized day per dataset ...")
    timings, est_total_s, est_windows = {}, 0.0, 0
    for v in inv:
        if not v["n_files"]:
            continue
        t = time_one_day(a.data, v["dataset"], cfg, device_map, not a.no_handshake)
        if not t:
            continue
        timings[v["dataset"]] = t
        n = v["n_after_size_filter"]
        est_total_s += t["seconds"] * n
        est_windows += t["windows"] * n
        print(f"  {v['dataset']}: {t['day']} ({t['mb']} MB) -> "
              f"{t['seconds']} s, {t['windows']:,} windows"
              + (f", +{t['rss_delta_mb']} MB RSS" if t["rss_delta_mb"] else ""))
        print(f"     x {n} days = {t['seconds'] * n / 3600:.1f} h, "
              f"~{t['windows'] * n:,} windows")

    print(f"\n  TOTAL BUILD (every=1): {est_total_s / 3600:.1f} h, "
          f"~{est_windows:,} windows")
    print(f"  at --every 2         : {est_total_s / 7200:.1f} h, "
          f"~{est_windows // 2:,} windows")

    print()
    print("=" * 74)
    print("3. WILL IT FIT IN RAM")
    print("=" * 74)
    L, F = cfg.max_flows_per_window, 14
    n_feat_flat = 250          # measured: 189 audited + sequence summaries + masks
    xseq = est_windows * L * F * 4
    flat = est_windows * n_feat_flat * 4
    bundle = (xseq + flat) / GB
    print(f"  Xseq  [{est_windows:,} x {L} x {F}] float32 = {xseq / GB:.2f} GB")
    print(f"  flat  [{est_windows:,} x ~{n_feat_flat}]   float32 = {flat / GB:.2f} GB")
    print(f"  one materialised bundle                  ~ {bundle:.2f} GB")
    print(f"  with the copies made by fit/subset (x3)   ~ {bundle * 3:.2f} GB peak")
    try:
        import psutil
        ram = psutil.virtual_memory().total / GB
        print(f"\n  this machine has {ram:.1f} GB RAM")
        headroom = ram - bundle * 3
        verdict = ("comfortable" if headroom > 8 else
                   "tight -- close other applications" if headroom > 2 else
                   "NOT ENOUGH at every=1")
        print(f"  projected headroom {headroom:.1f} GB -> {verdict}")
    except Exception:
        ram, verdict = None, "unknown (install psutil for a RAM check)"
        print(f"\n  {verdict}")

    print()
    print("=" * 74)
    print("RECOMMENDATION")
    print("=" * 74)
    every = 1
    if ram and (bundle * 3) > (ram - 2):
        every = 2 if (bundle * 1.5) < (ram - 2) else 3
        print(f"  Use --every {every}: the full span does not leave headroom here.")
    else:
        print("  Use --every 1 (the full span).")
    cap = 0 if est_windows < 60_000 else max(1000, int(60_000 / 27))
    if cap:
        print(f"  Use --cap-per-class {cap} for the NEURAL arm only "
              f"(`--methods gated`). The forest arm runs uncapped.")
        print("  The cap bounds the 20 refits the threshold command performs "
              "without shortening the calendar span.")
    else:
        print("  No cap needed; the neural arm can run uncapped.")
    print(f"  Always pass --min-day-size-frac {MIN_SIZE_FRAC} so the truncated "
          f"captures are dropped and named in the log.")

    out = {
        "inventory": inv, "timings": timings,
        "estimated_build_hours_every1": round(est_total_s / 3600, 2),
        "estimated_windows_every1": int(est_windows),
        "estimated_bundle_gb": round(bundle, 2),
        "machine_ram_gb": round(ram, 1) if ram else None,
        "recommended_every": every,
        "recommended_cap_per_class": cap,
        "min_day_size_frac": MIN_SIZE_FRAC,
        "n_devices_in_map": len(device_map),
    }
    os.makedirs(paths.results, exist_ok=True)
    dst = os.path.join(paths.results, "probe.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwritten: {dst}")
    print("\nNext: run run\\run_all.ps1 with the settings above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
