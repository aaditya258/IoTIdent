"""Guard-rail tests. Run with:  python -m vaid.selftest

These check the properties the experimental protocol depends on. They are cheap
and they fail loudly, because every one of them is a way a result could look
good for the wrong reason.
"""
from __future__ import annotations

import sys

import numpy as np

from .audit import ForbiddenFeatureError, assert_clean_feature_space, content_scan, load_forbidden_spec
from .config import Config, Paths
from .dataset import load_records, make_index, materialise
from .handshake import parse_client_hello_extensions, parse_hex
from .naming import class_label, device_key
from .splits import e1_session_grouped, e2_chronological, e3_class_heldout, e5_temporal_AB
from .views import all_feature_names
from .visibility import apply_profile

OK, FAIL = "ok  ", "FAIL"
_results: list[tuple[str, bool, str]] = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print(f"[{OK if cond else FAIL}] {name}" + (f"  -- {detail}" if detail else ""))


def main(root: str | None = None) -> int:
    paths = Paths(root=root) if root else Paths()
    cfg = Config()

    # 1. device-key normalisation: the double-underscore trap
    check("WithingsSmartScale spellings collapse to one key",
          device_key("WithingsSmartScale__0024e41b6f96")
          == device_key("WithingsSmartScale_0024e41b6f96_flows.csv"))
    check("class label carries no MAC",
          class_label("TribySpeaker_18B79E022044") == "TribySpeaker")

    # 2. the SNI never leaves the handshake parser
    blob = ("0000002100 1f00001c" + "device-metrics-us.amazon.com".encode().hex()
            ).replace(" ", "")
    ch = parse_client_hello_extensions(blob)
    check("SNI value is discarded, only presence and length survive",
          ch.has_sni and ch.sni_len >= 0 and "amazon" not in str(ch.__dict__).lower(),
          f"sni_len={ch.sni_len}")

    # 3. content scan catches identity inside an innocuous-looking column
    hits = content_scan(["16030100 " + "ws.invoxia.io".encode().hex(),
                         "192.168.1.120", "18:B7:9E:02:20:44"],
                        "TribySpeaker_18b79e022044", ["triby"])
    check("content scan flags MAC / RFC1918 / hostname",
          {"mac_colon_or_dash", "rfc1918_literal"} <= set(hits), str(sorted(hits)))

    # 4. forbidden-name assertion fails closed
    spec = load_forbidden_spec(paths)
    try:
        assert_clean_feature_space(["agg__n_flows", "h__chaddr"], spec)
        check("forbidden feature name is rejected", False, "no exception raised")
    except ForbiddenFeatureError:
        check("forbidden feature name is rejected", True)

    # everything below needs the data
    try:
        recs = load_records(paths, cfg, verbose=False)
    except FileNotFoundError as e:
        check("sample data present", False, str(e))
        return 1
    idx = make_index(recs, cfg)
    check("windows built", len(recs) > 0, f"{len(recs)} windows, {len(idx.labels)} classes")

    tr, va, te = next(iter(e1_session_grouped(idx, 5, cfg.seed)))
    b, vs = materialise(recs, cfg, tr)
    names = all_feature_names(b)
    try:
        assert_clean_feature_space(names, spec)
        check("materialised feature space passes the §4.3 audit", True,
              f"{len(names)} features")
    except ForbiddenFeatureError as e:
        check("materialised feature space passes the §4.3 audit", False, str(e))

    # 5. no group straddles a partition
    g_tr, g_te = set(b.group[tr]), set(b.group[te])
    check("E1 groups do not straddle train/test", not (g_tr & g_te),
          f"{len(g_tr & g_te)} shared groups")

    # 6. chronological split really is chronological, per device
    c_tr, c_va, c_te = e2_chronological(idx)
    bad = [d for d in np.unique(idx.device)
           if len(np.intersect1d(c_tr, np.where(idx.device == d)[0])) and
           len(np.intersect1d(c_te, np.where(idx.device == d)[0])) and
           idx.t_start[np.intersect1d(c_tr, np.where(idx.device == d)[0])].max() >
           idx.t_start[np.intersect1d(c_te, np.where(idx.device == d)[0])].min()]
    check("E2 test windows are strictly later than training windows", not bad, str(bad[:3]))

    # 7. unknown-class sets are disjoint
    known, u_val, u_test = e3_class_heldout(idx, 0.3, cfg.seed)
    check("E3 U_val and U_test are disjoint", not (set(u_val) & set(u_test)))
    check("E3 known classes exclude both unknown sets",
          not (set(known) & (set(u_val) | set(u_test))))

    # 8. E5 uses only devices present in both periods
    try:
        a_tr, a_va, a_te, devs = e5_temporal_AB(idx, seed=cfg.seed)
        check("E5 restricted to devices in both periods",
              set(idx.device[a_te]) <= set(devs), f"{len(devs)} devices")
    except ValueError as e:
        check("E5 restricted to devices in both periods", False, str(e))

    # 9. visibility profiles actually remove information
    b2 = apply_profile(b, "V5", cfg)
    check("V5 removes the handshake view entirely",
          float(np.abs(b2.Xh).sum()) == 0.0 and float(b2.mH.sum()) == 0.0)
    b3 = apply_profile(b, "V4", cfg)
    lo, hi = b.h_blocks["tls_detail"]
    check("V4 removes handshake detail but keeps presence/version",
          float(np.abs(b3.Xh[:, lo:hi]).sum()) == 0.0 and float(b3.mH.sum()) > 0)
    b4 = apply_profile(b, "V3", cfg)
    clo, chi = b.c_blocks["dest_groups"]
    check("V3 suppresses destination-group identity",
          float(np.abs(b4.Xc[:, clo:chi]).sum()) == 0.0)

    # 10. V6 is refused unless explicitly enabled
    try:
        apply_profile(b, "V6", cfg)
        check("V6 refused without explicit opt-in", False)
    except PermissionError:
        check("V6 refused without explicit opt-in", True)

    # 11. scalers are fitted on training data only
    check("scaler fitted on the training partition only",
          np.allclose(b.Xagg[tr].mean(0), 0, atol=0.15)
          and not np.allclose(b.Xagg[te].mean(0), 0, atol=1e-6),
          "train mean ~ 0, test mean free")

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
