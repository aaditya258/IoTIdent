#!/usr/bin/env python3
"""Exact flow totals per dataset, for the data-availability and limitations text.

Reads only each Parquet file's footer, where the row count is already stored, so
323 files take seconds rather than hours. Prints the total, the number of rows
whose device MAC is outside the released configuration map, and the share that
represents -- the denominator the manuscript needs and that cannot be taken from
the dataset README, whose per-dataset totals are scoped to the authors' own
subset rather than to every row shipped in the Parquet files.

    python run/count_flows.py --data <DATA_ROOT> --root .
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from vaid.customflow import _day_files, load_device_map   # noqa: E402

MIN_SIZE_FRAC = 0.25


def main(argv=None) -> int:
    p = argparse.ArgumentParser("count_flows")
    p.add_argument("--data", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--datasets", nargs="+", default=["DATA2025v1", "DATA2025v2"])
    p.add_argument("--batch-size", type=int, default=65536)
    a = p.parse_args(argv)

    dm = load_device_map(os.path.join(a.data, "scripts", "scripts", "configs")) \
        or load_device_map(os.path.join(a.data, "scripts", "configs"))
    if not dm:
        print("no device map found under scripts/")
        return 2
    print(f"device map: {len(dm)} MAC addresses\n")

    out = {}
    for ds in a.datasets:
        files = _day_files(a.data, ds, 1, 0, min_size_frac=MIN_SIZE_FRAC,
                           verbose=False)
        total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)

        # The MAC column is small; one pass over it gives the unmapped split.
        unmapped = Counter()
        n_unmapped = 0
        for i, f in enumerate(files, 1):
            pf = pq.ParquetFile(f)
            for b in pf.iter_batches(batch_size=a.batch_size, columns=["Device"]):
                for m in b.column(0).to_pylist():
                    m = (m.decode() if isinstance(m, bytes) else str(m)).lower()
                    if m not in dm:
                        unmapped[m] += 1
                        n_unmapped += 1
            if i % 25 == 0 or i == len(files):
                print(f"  {ds}: {i}/{len(files)} days", end="\r")

        share = n_unmapped / total if total else 0.0
        print(f"\n{ds}: {len(files)} days, {total:,} flows")
        print(f"   outside the device map: {n_unmapped:,} ({share:.1%}) "
              f"from {len(unmapped)} MAC(s)")
        for m, n in unmapped.most_common(10):
            print(f"      {m}  {n:,}  ({n / total:.1%})")
        out[ds] = {"days": len(files), "flows": total,
                   "unmapped_flows": n_unmapped, "unmapped_share": share,
                   "unmapped_by_mac": dict(unmapped.most_common())}

    dst = os.path.join(a.root, "results", "flow_counts.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwritten: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
