"""Command line for the Custom Flow source (DATA2016 / DATA2025).

    python -m vaid.cf_cli build    --data <DATA_ROOT> --every 7
    python -m vaid.cf_cli audit    --data "..."          # §4.1 + §4.3 on the new source
    python -m vaid.cf_cli baseline --data "..."          # the sanity gate
    python -m vaid.cf_cli instance --data "..."          # instance-held-out check

`--data` is the folder holding DATA2025v1/, DATA2025v2/ and scripts/.
`--root` is the research repository (for data/cache and results/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

from .audit import (ForbiddenFeatureError, assert_clean_feature_space,
                    load_forbidden_spec, scan_feature_values)
from .config import Config, Paths
from .customflow import load_records
from .dataset import cap_records_per_class, make_index, materialise
from .metrics import closed_set_metrics, mean_ci
from .models_shallow import make_model
from .splits import (e1_session_grouped, e4_cross_network, e6_instance_heldout,
                     trainable_classes)
from .views import all_feature_names
from .visibility import apply_profile


def _prepare(a):
    cfg = Config()
    cfg.T_minutes = a.T
    cfg.seed = a.seed
    cfg.backend = a.backend
    cfg.min_windows_per_class = a.min_windows
    cfg.cf_upstream_only = not a.allow_bidirectional
    cfg.cf_upstream_packets = a.upstream_packets
    cfg.cf_exclude_days = tuple(a.exclude or ())
    cfg.cf_min_day_size_frac = a.min_day_size_frac
    cfg.cf_cap_per_class = a.cap_per_class
    if a.epochs:
        cfg.epochs = a.epochs
    if a.quick:
        cfg.epochs = min(cfg.epochs, 8)
        cfg.n_repeats = 2
        cfg.n_bootstrap = 200
    paths = Paths(root=a.root) if a.root else Paths()
    recs = load_records(paths, cfg, a.data, a.datasets, every=a.every,
                        limit=a.limit, with_handshake=not a.no_handshake)
    # The cap is applied AFTER the cache is read, so one expensive build serves
    # both the full-span and the capped runs.
    recs = cap_records_per_class(recs, a.cap_per_class, cfg, seed=cfg.seed)
    return cfg, paths, recs


def _class_report(recs, cfg):
    counts = Counter(r.label for r in recs)
    keep = sorted(c for c, n in counts.items() if n >= cfg.min_windows_per_class)
    tail = sorted(c for c, n in counts.items() if n < cfg.min_windows_per_class)
    print(f"\nwindows={len(recs):,} classes={len(counts)} "
          f"trainable={len(keep)} tail={len(tail)}")
    if tail:
        print("  tail: " + ", ".join(f"{c}({counts[c]})" for c in tail))
    return counts, keep, tail


def main(argv=None) -> int:
    p = argparse.ArgumentParser("vaid.cf_cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build", "audit", "baseline", "instance",
                                       "cross", "threshold"])
    p.add_argument("--data", required=True, help="folder with DATA2025v1/ v2/ scripts/")
    p.add_argument("--root", default=None, help="research repo (cache + results)")
    p.add_argument("--datasets", nargs="+", default=["DATA2025v1", "DATA2025v2"])
    p.add_argument("--every", type=int, default=1, help="read every Nth day")
    p.add_argument("--limit", type=int, default=0, help="at most N days per dataset")
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--upstream-packets", type=int, default=5,
                   help="K: upstream packets per flow in the sequence view")
    p.add_argument("--allow-bidirectional", action="store_true",
                   help="use downstream features too. INVALID for cross-network "
                        "work: v1 is bidirectional, v2 is not")
    p.add_argument("--no-handshake", action="store_true",
                   help="skip the payload pass (much faster, no view H)")
    p.add_argument("--min-windows", type=int, default=30)
    p.add_argument("--exclude", nargs="*", default=None,
                   help="day stems to skip, e.g. 25-10-13 25-10-14")
    p.add_argument("--min-day-size-frac", type=float, default=0.0,
                   help="drop days smaller than this fraction of the median "
                        "day size. 0.25 removes the two truncated v2 captures; "
                        "the dropped filenames are printed")
    p.add_argument("--cap-per-class", type=int, default=0,
                   help="cap windows per class, by whole adjacency group. Use "
                        "for the neural arm at full span; 0 keeps everything")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "shallow"])
    p.add_argument("--method", default="rf", help="model for cross/instance runs")
    p.add_argument("--views", nargs="+", default=["F", "H", "C"], choices=["F", "H", "C"],
                   help="views to use. F C isolates the sequence view, which is "
                        "what the K-sensitivity check needs")
    p.add_argument("--quick", action="store_true")
    # threshold-transport only
    p.add_argument("--axes", nargs="+", default=["network", "time", "visibility",
                                                 "instance"],
                   choices=["network", "time", "visibility", "instance"])
    p.add_argument("--methods", nargs="+", default=None,
                   help="models for the threshold run (default: --method)")
    p.add_argument("--scorers", nargs="+",
                   default=["msp", "entropy", "energy", "proto"],
                   choices=["msp", "entropy", "energy", "proto"],
                   help="energy and proto are dropped automatically for models "
                        "that emit normalised posteriors instead of logits and "
                        "an embedding -- they are undefined there, not merely "
                        "weak")
    p.add_argument("--alpha", type=float, default=None,
                   help="nominal known-acceptance level (default "
                        "Config.target_known_tpr)")
    p.add_argument("--repeats", type=int, default=1,
                   help="reseed the split and the unknown-class choice N times "
                        "and report dA as mean +- 95%% CI")
    p.add_argument("--n-unknown", type=int, default=6,
                   help="classes held out of training as unknowns")
    a = p.parse_args(argv)

    cfg, paths, recs = _prepare(a)
    counts, keep, tail = _class_report(recs, cfg)

    if a.command == "build":
        by_net = Counter(getattr(r, "network", "") for r in recs)
        print("\nwindows per network:", dict(by_net))
        hs = sum(1 for r in recs if r.hs.get("tls_present"))
        dh = sum(1 for r in recs if r.hs.get("dhcp_present"))
        t13 = sum(1 for r in recs if r.hs.get("tls13"))
        ech = sum(1 for r in recs if r.hs.get("has_ech"))
        print(f"handshake view available in {hs:,}/{len(recs):,} windows "
              f"({hs/max(len(recs),1):.1%}); TLS 1.3 in {t13:,}; ECH in {ech:,}; "
              f"DHCP option 55 in {dh:,}")
        print("\nper class (windows / instances / networks):")
        inst = {}
        nets = {}
        for r in recs:
            inst.setdefault(r.label, set()).add(r.device)
            nets.setdefault(r.label, set()).add(getattr(r, "network", ""))
        for c, n in counts.most_common():
            print(f"  {c:26s} {n:>7,}  units={len(inst[c])}  nets={len(nets[c])}")
        return 0

    if a.command == "audit":
        idx = make_index(recs, cfg, labels=keep)
        tr, va, te = next(iter(e1_session_grouped(idx, 4, cfg.seed)))
        b, _ = materialise(recs, cfg, tr, labels=keep)
        names = all_feature_names(b)
        spec = load_forbidden_spec(paths)
        try:
            assert_clean_feature_space(names, spec)
            print(f"\n§4.3 forbidden-feature audit: PASS ({len(names)} features)")
        except ForbiddenFeatureError as e:
            print(f"\n§4.3 audit FAILED: {e}")
            return 1
        X = np.concatenate([b.Xagg, b.Xh, b.Xc], axis=1)
        nm = [n for n in names if not n.startswith("seq__")]
        rep = scan_feature_values(X, nm)
        print(f"constant (zero-variance) features: {len(rep['constant_features'])}")
        for n in rep["constant_features"][:20]:
            print(f"   {n}")
        print("\nNOTE: with cf_upstream_only=True the direction features are "
              "constant by construction on v2 and should appear above.")
        print(f"\ngroups straddling train/test: "
              f"{len(set(b.group[tr]) & set(b.group[te]))}")
        return 0

    if a.command == "baseline":
        from .dataset import filter_records
        recs2, _ = filter_records(recs, keep)
        idx = make_index(recs2, cfg, labels=keep)
        rows, folds = [], {}
        methods = ("rf", "gbm", "tcn") if not a.quick else ("rf", "gbm")
        for f, (tr, va, te) in enumerate(e1_session_grouped(idx, a.folds, cfg.seed)):
            b, _ = materialise(recs2, cfg, tr, labels=keep)
            TR, VA, TE = b.subset(tr), b.subset(va), b.subset(te)
            for m in methods:
                model = make_model(cfg, m, seed=cfg.seed + f, views=tuple(a.views))
                model.fit(TR, VA)
                pr = model.predict(TE)
                mt = closed_set_metrics(TE.y, pr["y_pred"], pr["proba"],
                                        getattr(model, "classes_", None))
                mt.update({"method": m, "fold": f, "n_test": len(TE.y)})
                rows.append(mt)
                folds.setdefault(m, []).append(mt["macro_f1"])
            print(f"  fold {f}: " + "  ".join(f"{m}={folds[m][-1]:.3f}" for m in methods))
        print("\nclosed-set, session-grouped, all views:")
        for m, v in folds.items():
            s = mean_ci(v)
            print(f"  {m:6s} macro-F1 {s['mean']:.3f} ± {s['ci95']:.3f}")
        os.makedirs(paths.results, exist_ok=True)
        out = os.path.join(paths.results, "cf_baseline.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"config": cfg.to_dict(), "rows": rows,
                       "summary": {m: mean_ci(v) for m, v in folds.items()},
                       "trainable_classes": keep, "tail_classes": tail}, fh,
                      indent=1, default=float)
        print(f"written: {out}")
        return 0

    if a.command == "cross":
        # Train on one network, test on the other, restricted to shared classes,
        # with a within-source control so the drop is attributable.
        from sklearn.metrics import f1_score
        from .dataset import filter_records
        from .splits import _group_holdout
        recs2, _ = filter_records(recs, keep)
        idx = make_index(recs2, cfg, labels=keep)
        nets = sorted(set(idx.network.tolist()))
        if len(nets) < 2:
            print(f"need two networks, have {nets}")
            return 1
        out = {}
        for src, tgt in ((nets[0], nets[1]), (nets[1], nets[0])):
            tr, va, te, shared = e4_cross_network(idx, src, tgt, seed=cfg.seed)
            b, _ = materialise(recs2, cfg, tr, labels=keep)
            model = make_model(cfg, a.method, seed=cfg.seed, views=tuple(a.views))
            model.fit(b.subset(tr), b.subset(va))
            # control: held-out groups inside the SOURCE network, same classes
            _, ctrl = _group_holdout(idx.group[tr], idx.y[tr], 0.25,
                                     np.random.default_rng(cfg.seed + 3))
            res = {}
            for name, sel in (("source_control", tr[ctrl]), ("target", te)):
                TE = b.subset(sel)
                pr = model.predict(TE)
                present = sorted(set(TE.y.tolist()))
                res[name] = {
                    "n_windows": int(len(sel)),
                    "n_classes_present": len(present),
                    "macro_f1": float(f1_score(TE.y, pr["y_pred"], average="macro",
                                               labels=present, zero_division=0)),
                    "accuracy": float(np.mean(TE.y == pr["y_pred"])),
                }
            gap = res["source_control"]["macro_f1"] - res["target"]["macro_f1"]
            print(f"\n{src} -> {tgt}  ({len(shared)} shared classes, "
              f"method={a.method}, views={''.join(a.views)}, K={cfg.cf_upstream_packets})")
            print(f"  source control : macro-F1 {res['source_control']['macro_f1']:.3f} "
                  f"(n={res['source_control']['n_windows']}, "
                  f"{res['source_control']['n_classes_present']} classes)")
            print(f"  target network : macro-F1 {res['target']['macro_f1']:.3f} "
                  f"(n={res['target']['n_windows']}, "
                  f"{res['target']['n_classes_present']} classes)")
            print(f"  cross-network gap: {gap:+.3f}")
            out[f"{src}->{tgt}"] = {"shared_classes": shared, "gap": gap, **res}
        os.makedirs(paths.results, exist_ok=True)
        # The filename carries the configuration. It used to be a fixed
        # "cf_cross_network.json", so the K-sensitivity sweep silently
        # overwrote the headline all-views result with its own views=FC run --
        # three runs, one file, last writer wins.
        tag = f"{a.method}_{''.join(a.views)}_K{cfg.cf_upstream_packets}"
        pth = os.path.join(paths.results, f"cf_cross_network_{tag}.json")
        with open(pth, "w", encoding="utf-8") as fh:
            json.dump({"config": cfg.to_dict(), "runs": out}, fh, indent=1, default=float)
        print(f"\nwritten: {pth}")
        print("\nCAVEAT: a class shared between the networks is usually a "
              "DIFFERENT physical unit in each, so this gap mixes the network "
              "effect with the unit effect. Run `instance` to size the latter.")
        return 0

    if a.command == "instance":
        # Does the model recognise the PRODUCT or the BOX? Holding out a whole
        # physical unit answers that, but only against a matched control: the
        # same classes, the same trained model, and test windows that differ
        # from the held-out ones ONLY in which unit produced them. Without the
        # control, a low score could just mean those classes are hard.
        from sklearn.metrics import f1_score, recall_score
        from .dataset import filter_records
        from .splits import _group_holdout
        recs2, _ = filter_records(recs, keep)
        idx = make_index(recs2, cfg, labels=keep)
        labs = np.array(idx.labels)
        out = {}
        for net in sorted(set(idx.network.tolist())):
            m = idx.network == net
            rng = np.random.default_rng(cfg.seed)
            held, tested = [], []
            for cls in sorted(set(labs[idx.y[m]].tolist())):
                units = sorted(set(idx.instance[m & (labs[idx.y] == cls)].tolist()))
                if len(units) < 2:
                    continue
                held.append(units[int(rng.integers(len(units)))])
                tested.append(cls)
            if not held:
                print(f"\n{net}: no device type has two units here")
                continue

            is_held = np.isin(idx.instance, held)
            i_held = np.where(m & is_held)[0]
            i_rest = np.where(m & ~is_held)[0]
            # carve a control test set out of the retained units, by group
            sub_tr, sub_ctrl = _group_holdout(idx.group[i_rest], idx.y[i_rest], 0.25,
                                              np.random.default_rng(cfg.seed + 1))
            i_ctrl = i_rest[sub_ctrl]
            i_fit = i_rest[sub_tr]
            fit2, val2 = _group_holdout(idx.group[i_fit], idx.y[i_fit], 0.2,
                                        np.random.default_rng(cfg.seed + 2))

            b, _ = materialise(recs2, cfg, i_fit[fit2], labels=keep)
            model = make_model(cfg, a.method, seed=cfg.seed, views=tuple(a.views))
            model.fit(b.subset(i_fit[fit2]), b.subset(i_fit[val2]))

            tested_ids = [keep.index(c) for c in tested]
            res = {}
            for name, sel in (("unseen_unit", i_held),
                              ("seen_unit_control", i_ctrl[np.isin(idx.y[i_ctrl], tested_ids)])):
                if len(sel) == 0:
                    continue
                TE = b.subset(sel)
                pr = model.predict(TE)
                present = sorted(set(TE.y.tolist()))
                res[name] = {
                    "n_windows": int(len(sel)),
                    "macro_f1_over_present_classes": float(
                        f1_score(TE.y, pr["y_pred"], average="macro",
                                 labels=present, zero_division=0)),
                    "accuracy": float(np.mean(TE.y == pr["y_pred"])),
                    "per_class_recall": {
                        keep[c]: float(r) for c, r in zip(
                            present, recall_score(TE.y, pr["y_pred"], labels=present,
                                                  average=None, zero_division=0))},
                }
            a_, b_ = res.get("unseen_unit"), res.get("seen_unit_control")
            print(f"\n{net}: {len(tested)} device types have >1 unit; held out {held}")
            if b_:
                print(f"  seen unit,   same classes : macro-F1 {b_['macro_f1_over_present_classes']:.3f} "
                      f"acc {b_['accuracy']:.3f}  (n={b_['n_windows']})")
            if a_:
                print(f"  UNSEEN unit, same classes : macro-F1 {a_['macro_f1_over_present_classes']:.3f} "
                      f"acc {a_['accuracy']:.3f}  (n={a_['n_windows']})")
            if a_ and b_:
                print(f"  drop attributable to the physical unit: "
                      f"{b_['macro_f1_over_present_classes'] - a_['macro_f1_over_present_classes']:+.3f}")
                print("  per-class recall (seen unit -> unseen unit):")
                for c in tested:
                    sr = b_["per_class_recall"].get(c)
                    ur = a_["per_class_recall"].get(c)
                    f = lambda x: "  n/a" if x is None else f"{x:5.3f}"
                    print(f"    {c:24s} {f(sr)} -> {f(ur)}")
            out[net] = {"held_units": held, "tested_classes": tested, **res}

        os.makedirs(paths.results, exist_ok=True)
        pth = os.path.join(paths.results, "cf_instance_heldout.json")
        with open(pth, "w", encoding="utf-8") as fh:
            json.dump({"config": cfg.to_dict(), "networks": out}, fh, indent=1,
                      default=float)
        print(f"\nwritten: {pth}")
        return 0

    if a.command == "threshold":
        # THE decisive experiment. Calibrate the rejection threshold on one
        # condition at a nominal acceptance level, then measure what acceptance
        # it actually delivers somewhere else. AUROC is printed beside it so the
        # two failure modes stay distinguishable: a displaced operating point
        # with AUROC intact is a calibration problem and recalibration recovers
        # it; a displaced operating point with AUROC gone is a ranking problem
        # and no threshold saves it.
        from .dataset import filter_records
        from . import transport as TR

        recs2, _ = filter_records(recs, keep)
        idx = make_index(recs2, cfg, labels=keep)
        alpha = a.alpha if a.alpha is not None else cfg.target_known_tpr
        methods = a.methods or [a.method]
        nets = sorted(set(idx.network.tolist()))
        biggest = max(nets, key=lambda n: int((idx.network == n).sum())) if nets else None

        def build_specs(seed):
            out = []
            for ax in a.axes:
                try:
                    if ax == "network":
                        if len(nets) < 2:
                            print(f"skipping axis 'network': only {nets}")
                            continue
                        # v2 -> v1 first: the K-sensitivity result says that
                        # direction is the representation-invariant one, so it
                        # is the direction a transport claim should rest on.
                        for s, t in ((nets[-1], nets[0]), (nets[0], nets[-1])):
                            out.append(TR.axis_network(idx, cfg, s, t,
                                                       a.n_unknown, seed))
                    elif ax == "time":
                        out.append(TR.axis_time(idx, cfg, biggest, a.n_unknown, seed))
                    elif ax == "visibility":
                        out.append(TR.axis_visibility(idx, cfg, biggest,
                                                      ("V4", "V5"), a.n_unknown,
                                                      seed))
                    elif ax == "instance":
                        out.append(TR.axis_instance(idx, cfg, biggest, 2, seed))
                except ValueError as e:
                    print(f"skipping axis {ax!r}: {e}")
            return out

        # Repeats vary the seed, which varies BOTH the group split and which
        # classes are held out as unknown. That is the right notion of
        # variability for this experiment: a displacement that only appears for
        # one choice of unknown classes is a property of those classes.
        runs, verdicts = {}, {}
        repeat_dA: dict = {}
        specs = [(r, sp) for r in range(max(1, a.repeats))
                 for sp in build_specs(cfg.seed + 1000 * r)]
        for rep, spec in specs:
            print(f"\n{'=' * 72}\naxis: {spec.axis}")
            print(f"  known classes   {len(spec.known_classes)}")
            print(f"  unknown classes {len(spec.unknown_classes)}: "
                  f"{', '.join(spec.unknown_classes)}")
            print(f"  fit {len(spec.fit):,} windows   calibration "
                  f"{len(spec.calib):,} windows   alpha {alpha:.2f}")
            if spec.meta:
                print(f"  {spec.meta}")
            b, _ = materialise(recs2, cfg, spec.fit, labels=keep)
            for meth in methods:
                model = make_model(cfg, meth, seed=cfg.seed, views=tuple(a.views))
                model.fit(b.subset(spec.fit), b.subset(spec.calib))
                res = TR.run_transport(model, b, cfg, spec.calib, spec.conditions,
                                       alpha=alpha, scorers=tuple(a.scorers),
                                       proto_fit=spec.fit)
                res["meta"] = spec.meta
                res["known_classes"] = spec.known_classes
                res["unknown_classes"] = spec.unknown_classes
                key = f"{spec.axis}|{meth}|r{rep}"
                runs[key] = res
                for sc, why in res["scorers_skipped"].items():
                    print(f"\n  --- {meth}: scorer {sc!r} NOT APPLICABLE: {why}")
                for sc in res["scorers_used"]:
                    print(f"\n  --- {meth}, scorer={sc}, "
                          f"tau={res['tau'][sc]:.4f} ---")
                    for name, rec in res["conditions"].items():
                        for line in TR.format_condition(name, rec, sc, alpha):
                            print(line)
                    for name in list(res["conditions"])[1:]:
                        v = TR.transport_verdict(res, "source_control", name, sc)
                        verdicts[f"{key}|{sc}|{name}"] = v
                        agg = repeat_dA.setdefault(
                            f"{spec.axis.split(':')[0]}|{meth}|{sc}|{name}",
                            {"dA": [], "R_ctrl": [], "R_star": [],
                             "auroc": [], "csF1": [], "verdicts": []})
                        agg["dA"].append(v["delta_A_shifted"])
                        agg["R_ctrl"].append(v["rejection_control"])
                        agg["R_star"].append(v["rejection_shifted_at_oracle_tau"])
                        agg["auroc"].append(v["auroc_shifted"])
                        agg["csF1"].append(v["closed_set_f1_shifted"])
                        agg["verdicts"].append(v["verdict"].split(":")[0]
                                               .split(" (")[0])
                        print(f"      => {name}: {v['verdict']}  "
                              f"(control {v['delta_A_control']:+.3f}, "
                              f"shifted {v['delta_A_shifted']:+.3f}, "
                              f"floor {v['noise_floor']:.3f})")

        if a.repeats > 1:
            print(f"\n{'=' * 72}\nacross {a.repeats} seeds "
                  f"(each reseeds the split AND the unknown class choice)")
            print(f"{'axis / condition':34s} {'method':6s} {'scorer':7s} "
                  f"{'dA mean+-ci':>16s} {'R_ctrl':>7s} {'R*':>7s} "
                  f"{'AUROC':>6s}  modal verdict")
            for k, agg in repeat_dA.items():
                axis, meth, sc, cond = k.split("|")
                d, rc, rs, au = (mean_ci(agg["dA"]), mean_ci(agg["R_ctrl"]),
                                 mean_ci(agg["R_star"]), mean_ci(agg["auroc"]))
                modal = Counter(agg["verdicts"]).most_common(1)[0]
                print(f"{axis + ' / ' + cond:34.34s} {meth:6s} {sc:7s} "
                      f"{d['mean']:+8.3f} +-{d['ci95']:5.3f} "
                      f"{rc['mean']:7.3f} {rs['mean']:7.3f} {au['mean']:6.3f}  "
                      f"{modal[0]} ({modal[1]}/{len(agg['verdicts'])})")

        print(f"\n{'=' * 72}\nsummary: displacement of the realised known-acceptance")
        print(f"{'axis / condition':46s} {'method':6s} {'scorer':7s} "
              f"{'dA':>7s} {'R@tau':>6s} {'R@tau*':>7s} {'AUROC':>6s} "
              f"{'csF1':>6s}  verdict")
        for k, v in verdicts.items():
            axis, meth, rep, sc, cond = k.split("|")
            print(f"{axis + ' / ' + cond:46.46s} {meth:6s} {sc:7s} "
                  f"{v['delta_A_shifted']:+7.3f} "
                  f"{v['rejection_shifted_at_stale_tau']:6.3f} "
                  f"{v['rejection_shifted_at_oracle_tau']:7.3f} "
                  f"{v['auroc_shifted']:6.3f} "
                  f"{v['closed_set_f1_shifted']:6.3f}  {v['verdict']}")
        strong = {k: v for k, v in verdicts.items()
                  if v["source_operating_point_useful"]}
        print(f"\nheadline: the {len(strong)} of {len(verdicts)} "
              f"(axis, method, scorer) combinations whose SOURCE operating "
              f"point was worth transporting (rejection >= 0.5)")
        for k, v in strong.items():
            axis, meth, rep, sc, cond = k.split("|")
            print(f"  {axis} / {cond}  [{meth}, {sc}, {rep}]")
            print(f"      acceptance {v['delta_A_control'] * -1 + 0:+.3f} -> "
                  f"displacement {v['delta_A_shifted']:+.3f};  "
                  f"rejection {v['rejection_control']:.3f} -> "
                  f"{v['rejection_shifted_at_oracle_tau']:.3f} even after "
                  f"recalibration;  closed-set macro-F1 "
                  f"{v['closed_set_f1_control']:.3f} -> "
                  f"{v['closed_set_f1_shifted']:.3f}")
        print("\ndA  = nominal acceptance minus realised acceptance "
              "(0 means the operating point transported)")
        print("R@tau  = unknown rejection at the STALE threshold;  "
              "R@tau* = at the threshold the condition needed")
        print("csF1   = closed-set macro-F1 on the shifted condition, i.e. the "
              "number a closed-set paper would print")

        os.makedirs(paths.results, exist_ok=True)
        # Tagged, for the same reason `cross` is: this command used to write one
        # fixed "cf_threshold_transport.json", so a forest run followed by a
        # neural run left only the neural one on disk. The forest arm is the
        # headline, and it was the arm that got overwritten.
        tag = "_".join(methods) + "_" + "".join(a.views) + f"_r{a.repeats}"
        pth = os.path.join(paths.results, f"cf_threshold_transport_{tag}.json")
        with open(pth, "w", encoding="utf-8") as fh:
            json.dump({"config": cfg.to_dict(), "alpha": alpha,
                       "methods": methods, "scorers": list(a.scorers),
                       "views": list(a.views), "repeats": a.repeats,
                       "runs": runs, "verdicts": verdicts,
                       "across_seeds": {
                           k: {"n": len(v["dA"]),
                               "delta_A": mean_ci(v["dA"]),
                               "rejection_control": mean_ci(v["R_ctrl"]),
                               "rejection_at_oracle": mean_ci(v["R_star"]),
                               "auroc": mean_ci(v["auroc"]),
                               "closed_set_f1": mean_ci(v["csF1"]),
                               "verdicts": dict(Counter(v["verdicts"]))}
                           for k, v in repeat_dA.items()}},
                      fh, indent=1, default=float)
        print(f"\nwritten: {pth}")
        print("\nNOTE: the unknown CLASS pool is the same on both sides of each "
              "comparison, which is sound here because tau is the alpha-quantile "
              "of KNOWN scores only -- no unknown window ever moves it. Report "
              "it anyway: it is the difference from the E3 protocol.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
