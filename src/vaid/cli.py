"""Command line entry point.

    python -m vaid.cli audit                     # §4.1 audit + §4.3 spec check
    python -m vaid.cli build --T 30              # build and cache windows
    python -m vaid.cli exp1                      # closed-set baseline (Table 3)
    python -m vaid.cli exp3 --folds 5            # visibility-aware fusion V0-V5
    python -m vaid.cli all --quick               # everything, small budget

Run from the repository root, or pass --root explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .audit import audit_dataset, write_audit
from .config import Config, Paths
from .dataset import class_counts, load_records


def _cfg_from_args(a) -> Config:
    cfg = Config()
    cfg.T_minutes = a.T
    cfg.seed = a.seed
    cfg.backend = a.backend
    cfg.min_windows_per_class = a.min_windows
    if a.epochs:
        cfg.epochs = a.epochs
    if a.quick:
        cfg.epochs = min(cfg.epochs, 8)
        cfg.n_repeats = 2
        cfg.n_bootstrap = 200
    if a.allow_endpoint:
        cfg.allow_endpoint_upper_bound = True
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser("vaid", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["audit", "build", "exp1", "exp2", "exp3",
                                       "exp4", "exp5", "exp8", "exp9", "exp10", "all"])
    p.add_argument("--root", default=None, help="repository root")
    p.add_argument("--T", type=int, default=30, help="observation window in minutes")
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "shallow"])
    p.add_argument("--min-windows", type=int, default=30,
                   help="minimum windows for a class to be trainable")
    p.add_argument("--quick", action="store_true", help="small budget smoke run")
    p.add_argument("--allow-endpoint", action="store_true",
                   help="enable V6, the endpoint-assisted upper bound (reported separately)")
    a = p.parse_args(argv)

    paths = Paths(root=a.root) if a.root else Paths()
    cfg = _cfg_from_args(a)
    os.makedirs(paths.results, exist_ok=True)
    print(f"root={paths.root}\nT={cfg.T_minutes} min  backend={cfg.backend}  seed={cfg.seed}")

    if a.command == "audit":
        out = audit_dataset(paths)
        print(json.dumps(out["summary"], indent=1))
        print("written:", write_audit(paths, out))
        return 0

    if a.command == "build":
        recs = load_records(paths, cfg)
        counts = class_counts(recs)
        print(f"windows={len(recs)}  classes={len(counts)}")
        for k, v in counts.items():
            print(f"  {k:36s} {v:6d}")
        return 0

    from . import experiments as E
    table = {"exp1": E.experiment_1, "exp2": E.experiment_2, "exp3": E.experiment_3,
             "exp4": E.experiment_4, "exp5": E.experiment_5, "exp8": E.experiment_8,
             "exp9": E.experiment_9, "exp10": E.experiment_10}
    if a.command == "all":
        for name, fn in table.items():
            print(f"\n=== {name} ===")
            try:
                fn(paths, cfg) if name not in ("exp1", "exp2", "exp3") else fn(
                    paths, cfg, n_folds=a.folds)
            except NotImplementedError as e:
                print(f"  skipped: {e}")
        return 0

    fn = table[a.command]
    if a.command in ("exp1", "exp2", "exp3"):
        fn(paths, cfg, n_folds=a.folds)
    else:
        fn(paths, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
