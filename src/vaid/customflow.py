"""Adapter for the Custom Flow representation (DATA2016 / DATA2025).

Produces the same `WindowRecord` objects the UNSW-IoTraffic path produces, so
everything above the data layer -- partitions, visibility profiles, the open-set
layer, metrics and statistics -- is shared between the two sources.

THE REPRESENTATION MISMATCH THAT DECIDES THIS MODULE'S DESIGN
-------------------------------------------------------------
The two released 2025 networks are NOT in the same representation, and nothing
in the README says so. Measured over the staged days:

    DATA2025v1   2,225,537 upstream + 1,173,878 downstream captured packets
                 (34.5% downstream)
    DATA2025v2   4,775,706 upstream +         0 downstream

v2 is upstream-only. Worse, `TotalFlowSize` and `PacketCount` count BOTH
directions on v1 and upstream only on v2 -- verified by reconstruction: on v1
`TotalFlowSize` equals the sum of all captured packet sizes 100% of the time and
the upstream-only sum 50.7% of the time; on v2 both are 100%.

Train on v1 with those columns, test on v2, and most of the "cross-network drop"
would be the exporter difference rather than the network. That is the exact
confound this project refused to accept from a self-built pcap pipeline, and it
is present inside the released pair.

So, with `upstream_only=True` (the default and the only setting valid for a
cross-network claim):

  * every flow's captured packets are filtered to direction == 1 before any
    feature is computed;
  * `TotalFlowSize` and `PacketCount` are NEVER used -- byte and packet counts
    are recomputed from the surviving upstream packets;
  * the per-flow sequence uses the first `K` upstream packets (`Config
    .cf_upstream_packets`, default 5). The cap matters: "first 10 packets of the
    flow" covers about 6.5 upstream packets on v1 but 10 on v2, so a fixed small
    K is what makes the two sides comparable. Vary K and confirm the
    cross-network gap does not move with it.

Set `upstream_only=False` only for a within-network experiment, and report it
separately, exactly as the V6 endpoint upper bound is reported separately.
"""
from __future__ import annotations

import glob
import hashlib
import math
import os
import pickle
import time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from .config import SERVICE_BUCKETS, SERVICE_ORDER, Config, Paths
from .handshake import (ClientHello, parse_client_hello_record,
                        parse_dhcp_parameter_request_list_from_payload,
                        split_payload_packets)
from .windows import WindowRecord, _entropy, _is_broadcast_or_multicast, _is_rfc1918

META_COLS = ["Device", "FirstSeen", "RemIP", "Proto", "DevPort", "RemPort"]
PKT_SIZE = [f"P{i:02d}_PS" for i in range(10)]
PKT_TIME = [f"P{i:02d}_TO" for i in range(10)]
PKT_DIR = [f"P{i:02d}_D" for i in range(10)]

#: Per-flow-event features. One row per flow inside an observation window.
CF_SEQ_FEATURES = [
    "log_up_bytes", "log_n_up_packets", "log_mean_up_size", "std_log_up_size",
    "log_max_up_size", "log_min_up_size", "log_burst_span_s",
    "mean_log_interpacket_s", "std_log_interpacket_s", "log_dt_prev_flow_s",
    "is_tcp", "is_udp", "is_other_proto", "capture_truncated",
]

#: `FirstSeen` and `P*_TO` are microseconds in DATA2025.
US = 1e6

NON_IOT_TYPE_ID = 24          # "IT" in the released configs


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def load_device_map(configs_dir: str) -> dict:
    """MAC -> (device-type name, type_id), from the released config YAMLs.

    The datasets ship this map, so no label guessing is needed. Note that the
    MAC is used ONLY to attach a label and an instance id at ingest; it never
    becomes a feature (§4.3).
    """
    import yaml
    out: dict[str, tuple[str, int]] = {}
    for p in sorted(glob.glob(os.path.join(configs_dir, "*-config.yaml"))):
        with open(p, "r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh) or {}
        for e in spec.get("device_info", []) or []:
            mac = str(e.get("mac_address", "")).strip().lower()
            if mac:
                out[mac] = (str(e.get("name", "unknown")), int(e.get("type_id", -1)))
    return out


# --------------------------------------------------------------------------
# Per-flow feature extraction
# --------------------------------------------------------------------------

def _flow_rows(sizes, times, dirs, upstream_only: bool, K: int):
    """Return (sizes, times) of the first K packets kept for one flow.

    `sizes`/`times`/`dirs` are length-10 arrays with NaN in unused slots.
    """
    keep = ~np.isnan(sizes)
    if upstream_only:
        keep &= (dirs == 1.0)
    s, t = sizes[keep], times[keep]
    if len(s) > K:
        s, t = s[:K], t[:K]
    return s, t


def _seq_row(s, t, proto, dt_prev, truncated) -> list[float]:
    ls = np.log1p(np.clip(s, 0, None)) if len(s) else np.array([0.0])
    gaps = np.diff(np.sort(t)) / US if len(t) > 1 else np.array([0.0])
    lg = np.log1p(np.clip(gaps, 0, None))
    span = (t.max() - t.min()) / US if len(t) > 1 else 0.0
    return [
        float(np.log1p(s.sum() if len(s) else 0.0)),
        float(np.log1p(len(s))),
        float(np.log1p(s.mean() if len(s) else 0.0)),
        float(ls.std()),
        float(np.log1p(s.max() if len(s) else 0.0)),
        float(np.log1p(s.min() if len(s) else 0.0)),
        float(np.log1p(span)),
        float(lg.mean()), float(lg.std()),
        float(np.log1p(max(dt_prev, 0.0))),
        1.0 if proto == 6 else 0.0,
        1.0 if proto == 17 else 0.0,
        0.0 if proto in (6, 17) else 1.0,
        1.0 if truncated else 0.0,
    ]


def _aggregates(seq: np.ndarray, t_starts: np.ndarray, protos: np.ndarray,
                T_seconds: float, n_trunc: int) -> dict:
    n = len(seq)
    dt = np.diff(np.sort(t_starts)) if n > 1 else np.array([0.0])
    ldt = np.log1p(np.clip(dt, 0, None))
    up_bytes = np.expm1(seq[:, 0])
    n_up = np.expm1(seq[:, 1])
    return {
        "n_flows": float(n),
        "log_n_flows": float(np.log1p(n)),
        "flows_per_min": float(n / (T_seconds / 60.0)),
        "log_total_up_bytes": float(np.log1p(up_bytes.sum())),
        "log_total_up_packets": float(np.log1p(n_up.sum())),
        "mean_log_up_bytes": float(seq[:, 0].mean()),
        "std_log_up_bytes": float(seq[:, 0].std()),
        "mean_log_up_packets": float(seq[:, 1].mean()),
        "std_log_up_packets": float(seq[:, 1].std()),
        "mean_log_burst_span": float(seq[:, 6].mean()),
        "std_log_burst_span": float(seq[:, 6].std()),
        "mean_log_interflow": float(ldt.mean()),
        "std_log_interflow": float(ldt.std()),
        "burst_frac": float(np.mean(dt < 1.0)) if n > 1 else 0.0,
        "frac_tcp": float(seq[:, 10].mean()),
        "frac_udp": float(seq[:, 11].mean()),
        "frac_other_proto": float(seq[:, 12].mean()),
        "frac_flows_truncated": float(n_trunc / max(n, 1)),
        "active_span_frac": float((t_starts.max() - t_starts.min()) / T_seconds) if n > 1 else 0.0,
        "mean_up_packets_captured": float(n_up.mean()),
    }


def _communication(remips, rports, protos, t_starts, cfg: Config):
    n = len(remips)
    remote = np.array([not (_is_rfc1918(ip) or _is_broadcast_or_multicast(ip))
                       for ip in remips])
    bcast = np.array([_is_broadcast_or_multicast(ip) for ip in remips])
    buckets = []
    for p, q in zip(rports, protos):
        try:
            pi = int(p)
        except Exception:
            buckets.append("other_eph")
            continue
        buckets.append(SERVICE_BUCKETS.get(pi, "other_priv" if pi < 1024 else "other_eph"))
    bc = Counter(buckets)

    groups = []
    for ip, r in zip(remips, remote):
        if not r or not ip:
            continue
        parts = ip.split(".")
        net = ".".join(parts[:3]) if len(parts) == 4 else ip
        h = hashlib.blake2b(f"{cfg.hash_salt}|{net}".encode(), digest_size=4).digest()
        groups.append(int.from_bytes(h, "big") % cfg.dest_group_vocab)

    seen, repeats = set(), 0
    for ip in remips:
        if ip in seen:
            repeats += 1
        seen.add(ip)
    gaps = np.diff(np.sort(t_starts)) if n > 1 else np.array([0.0])
    cv = float(gaps.std() / (gaps.mean() + 1e-9)) if n > 1 else 0.0
    trans = sum(1 for i in range(1, len(buckets)) if buckets[i] != buckets[i - 1])

    feats = {
        "n_dst_distinct": float(len(set(remips))),
        "log_n_dst_distinct": float(np.log1p(len(set(remips)))),
        "dst_cardinality_norm": float(len(set(remips)) / max(1, n)),
        "n_service_groups": float(len(bc)),
        "service_entropy": _entropy(bc.values()),
        "frac_remote": float(remote.mean()) if n else 0.0,
        "frac_broadcast": float(bcast.mean()) if n else 0.0,
        "frac_local": float(np.mean([_is_rfc1918(ip) for ip in remips])) if n else 0.0,
        "n_dst_ports_distinct": float(len(set(int(p) for p in rports))),
        "repeat_rate": float(repeats / max(1, n)),
        "periodicity": float(1.0 / (1.0 + cv)),
        "service_transitions_norm": float(trans / max(1, n - 1)),
    }
    for b in SERVICE_ORDER:
        feats[f"frac_service_{b}"] = float(bc.get(b, 0) / max(1, n))
    return feats, groups


def _empty_handshake() -> dict:
    return {"tls_present": 0.0, "dhcp_present": 0.0, "n_tls": 0.0, "n_dhcp": 0.0,
            "version": "", "ciphers": [], "ext_types": [], "ext_len_total": 0.0,
            "groups": [], "ecpf": [], "n_sig_algs": 0.0, "has_alpn": 0.0,
            "has_ticket": 0.0, "has_keyshare": 0.0, "has_sni": 0.0, "sni_len": 0.0,
            "n_ciphers": 0.0, "n_exts": 0.0, "n_fingerprints": 0.0, "prl": [],
            "tls13": 0.0, "has_ech": 0.0, "has_psk": 0.0, "frac_truncated": 0.0}


def _fold_client_hello(hs: dict, ch: ClientHello) -> None:
    """Accumulate one ClientHello into a window's handshake view.

    Presence flags latch on, because seeing an extension is sound evidence even
    from a truncated record. Field-valued features take the latest COMPLETE
    handshake, because absence in a cut record is not evidence.
    """
    hs["tls_present"] = 1.0
    hs["n_tls"] += 1.0
    hs["_fp"].add((ch.version, tuple(ch.cipher_suites), tuple(ch.ext_types)))
    if not ch.complete:
        hs["_trunc"] += 1
    for flag, attr in (("has_alpn", "has_alpn"), ("has_ticket", "has_session_ticket"),
                       ("has_keyshare", "has_key_share"), ("has_sni", "has_sni"),
                       ("has_ech", "has_ech"), ("has_psk", "has_psk")):
        if getattr(ch, attr):
            hs[flag] = 1.0
    if ch.tls13:
        hs["tls13"] = 1.0
    if ch.complete:
        hs.update({
            "version": ch.version, "ciphers": ch.cipher_suites,
            "ext_types": ch.ext_types, "ext_len_total": float(sum(ch.ext_lengths)),
            "groups": ch.supported_groups, "ecpf": ch.ec_point_formats,
            "n_sig_algs": float(ch.n_signature_algorithms),
            "sni_len": float(np.log1p(ch.sni_len)),
            "n_ciphers": float(len(ch.cipher_suites)),
            "n_exts": float(len(ch.ext_types)),
        })


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def _day_files(root: str, dataset: str, every: int, limit: int,
               exclude=(), min_size_frac: float = 0.0, only_days=None,
               verbose: bool = True) -> list[str]:
    """Day files, after dropping excluded and truncated captures.

    `min_size_frac` drops any day smaller than that fraction of the median day
    size and PRINTS which ones. This is not cosmetic: DATA2025v2 contains
    25-10-13 (4.3 MB) and 25-10-14 (2.5 MB) against a ~62 MB median -- captures
    that stopped early. 25-10-14 was in the original 15-day development set, so
    the Sep-Nov side of the time axis was thinner than it looked. A truncated
    day is not a quiet day: it has fewer windows per device for a reason that
    has nothing to do with the device.

    Exclusion happens BEFORE the `every` stride, so the stride lands on the
    retained days rather than leaving holes where a dropped day used to be.
    """
    files = sorted(glob.glob(os.path.join(root, dataset, "*.parquet")))
    if only_days:
        want = {str(d).strip() for d in only_days}
        return [f for f in files if os.path.basename(f).split(".")[0] in want]
    ex = {str(e).strip() for e in (exclude or ()) if str(e).strip()}
    if ex:
        before = len(files)
        files = [f for f in files
                 if os.path.basename(f).split(".")[0] not in ex]
        if verbose and len(files) != before:
            print(f"  {dataset}: excluded {before - len(files)} day(s) by name")
    if min_size_frac > 0 and files:
        sizes = np.array([os.path.getsize(f) for f in files], dtype=float)
        med = float(np.median(sizes))
        keep = sizes >= min_size_frac * med
        dropped = [os.path.basename(f) for f, k in zip(files, keep) if not k]
        if dropped and verbose:
            print(f"  {dataset}: dropped {len(dropped)} truncated day(s) "
                  f"(< {min_size_frac:.0%} of the {med/1e6:.0f} MB median): "
                  + ", ".join(dropped))
        files = [f for f, k in zip(files, keep) if k]
    files = files[::max(1, every)]
    return files[:limit] if limit else files


def build_windows_customflow(root: str, dataset: str, cfg: Config,
                             device_map: dict, every: int = 1, limit: int = 0,
                             with_handshake: bool = True, batch_size: int = 4096,
                             include_non_iot: bool = False, only_days=None,
                             verbose: bool = True) -> list[WindowRecord]:
    """Two passes per day.

    Pass 1 reads the ~36 metadata and per-packet columns and builds views F and
    C. Pass 2 reads the 3,000 payload columns and builds view H, but only for
    the port-443 and port-67 rows that can carry a handshake. Splitting them
    matters: pass 1 over 36 columns is seconds, and pass 2 is the expensive one,
    so a run that does not need the handshake view skips it entirely.
    """
    T = cfg.T_minutes * 60
    K = getattr(cfg, "cf_upstream_packets", 5)
    upstream_only = getattr(cfg, "cf_upstream_only", True)
    files = _day_files(root, dataset, every, limit,
                       exclude=getattr(cfg, "cf_exclude_days", ()),
                       min_size_frac=getattr(cfg, "cf_min_day_size_frac", 0.0),
                       only_days=only_days, verbose=verbose)
    if not files:
        raise FileNotFoundError(f"no parquet files under {os.path.join(root, dataset)}")

    buckets: dict[tuple[str, int], dict] = {}
    n_unmapped = Counter()

    for fi, path in enumerate(files, 1):
        t0 = time.time()
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size,
                                     columns=META_COLS + PKT_SIZE + PKT_TIME + PKT_DIR):
            d = batch.to_pydict()
            dev_raw = d["Device"]
            ts = np.asarray(d["FirstSeen"], dtype=np.float64) / US
            proto = np.asarray(d["Proto"], dtype=np.float64)
            rport = np.asarray(d["RemPort"], dtype=np.float64)
            remip = d["RemIP"]
            S = np.column_stack([np.asarray(d[c], dtype=np.float64) for c in PKT_SIZE])
            Tm = np.column_stack([np.asarray(d[c], dtype=np.float64) for c in PKT_TIME])
            Dr = np.column_stack([np.asarray(d[c], dtype=np.float64) for c in PKT_DIR])

            for r in range(len(ts)):
                mac = dev_raw[r]
                mac = (mac.decode() if isinstance(mac, bytes) else str(mac)).lower()
                ent = device_map.get(mac)
                if ent is None:
                    n_unmapped[mac] += 1
                    continue
                name, type_id = ent
                if type_id == NON_IOT_TYPE_ID and not include_non_iot:
                    continue
                s, t = _flow_rows(S[r], Tm[r], Dr[r], upstream_only, K)
                if len(s) == 0:
                    continue
                wid = int(ts[r] // T)
                key = (mac, wid)
                b = buckets.get(key)
                if b is None:
                    ip = remip[r]
                    b = buckets[key] = {
                        "mac": mac, "label": name, "type_id": type_id, "wid": wid,
                        "seq": [], "t": [], "proto": [], "remip": [], "rport": [],
                        "n_trunc": 0, "hs": None}
                dt_prev = ts[r] - b["t"][-1] if b["t"] else 0.0
                n_cap = int(np.sum(~np.isnan(S[r])))
                truncated = n_cap >= 10
                b["seq"].append(_seq_row(s, t, proto[r], dt_prev, truncated))
                b["t"].append(float(ts[r]))
                b["proto"].append(float(proto[r]))
                b["rport"].append(float(rport[r]))
                ipv = remip[r]
                b["remip"].append(ipv.decode() if isinstance(ipv, bytes) else str(ipv))
                b["n_trunc"] += int(truncated)
        if verbose:
            print(f"  [{fi}/{len(files)}] pass1 {os.path.basename(path):20s} "
                  f"windows={len(buckets):>7,} ({time.time()-t0:.1f}s)")

    # ---- pass 2: the handshake view ---------------------------------------
    if with_handshake:
        ccols = [c for c in pq.ParquetFile(files[0]).schema_arrow.names
                 if c.startswith("C_")]
        for fi, path in enumerate(files, 1):
            t0 = time.time()
            pf = pq.ParquetFile(path)
            n_ch = 0
            for batch in pf.iter_batches(batch_size=batch_size,
                                         columns=META_COLS + ccols):
                d = batch.to_pydict()
                dev_raw = d["Device"]
                ts = np.asarray(d["FirstSeen"], dtype=np.float64) / US
                proto = np.asarray(d["Proto"], dtype=np.float64)
                rport = np.asarray(d["RemPort"], dtype=np.float64)
                payload = None
                for r in range(len(ts)):
                    rp, pr = rport[r], proto[r]
                    is_tls = (rp == 443 and pr == 6)
                    is_dhcp = (rp == 67 and pr == 17)
                    if not (is_tls or is_dhcp):
                        continue
                    mac = dev_raw[r]
                    mac = (mac.decode() if isinstance(mac, bytes) else str(mac)).lower()
                    key = (mac, int(ts[r] // T))
                    b = buckets.get(key)
                    if b is None:
                        continue
                    if payload is None:
                        payload = np.column_stack(
                            [np.asarray(d[c], dtype=np.float32) for c in ccols])
                    if b["hs"] is None:
                        b["hs"] = _empty_handshake()
                        b["hs"]["_fp"] = set()
                        b["hs"]["_trunc"] = 0
                    pkts = split_payload_packets(payload[r])
                    if is_tls:
                        for pkt in pkts:
                            ch = parse_client_hello_record(pkt)
                            if ch is not None:
                                _fold_client_hello(b["hs"], ch)
                                n_ch += 1
                                break
                    else:
                        for pkt in pkts:
                            prl = parse_dhcp_parameter_request_list_from_payload(pkt)
                            if prl:
                                b["hs"]["dhcp_present"] = 1.0
                                b["hs"]["n_dhcp"] += 1.0
                                b["hs"]["prl"] = prl
                                break
            if verbose:
                print(f"  [{fi}/{len(files)}] pass2 {os.path.basename(path):20s} "
                      f"handshakes+={n_ch:>6,} ({time.time()-t0:.1f}s)")

    # ---- finalise ---------------------------------------------------------
    out: list[WindowRecord] = []
    for (mac, wid), b in buckets.items():
        if len(b["seq"]) < cfg.min_flows_per_window:
            continue
        seq = np.asarray(b["seq"], dtype=np.float64)
        t_arr = np.asarray(b["t"], dtype=np.float64)
        protos = np.asarray(b["proto"], dtype=np.float64)
        comm, groups = _communication(b["remip"], b["rport"], protos, t_arr, cfg)
        hs = b["hs"] or _empty_handshake()
        if "_fp" in hs:
            hs["n_fingerprints"] = float(len(hs.pop("_fp")))
            hs["frac_truncated"] = float(hs.pop("_trunc") / max(hs["n_tls"], 1.0))
        order = np.argsort(t_arr)
        out.append(WindowRecord(
            device=mac, label=b["label"], window_id=wid,
            t_start=float(wid) * T,
            period=time.strftime("%y-%m", time.gmtime(float(wid) * T)),
            n_flows=len(seq),
            seq=np.nan_to_num(seq[order][: cfg.max_flows_per_window],
                              nan=0.0, posinf=0.0, neginf=0.0),
            agg=_aggregates(seq, t_arr, protos, T, b["n_trunc"]),
            comm=comm, dest_groups=groups, hs=hs, network=dataset))

    if verbose:
        lab = Counter(r.label for r in out)
        print(f"  {dataset}: {len(out):,} windows, {len(lab)} device types")
        if n_unmapped:
            print(f"  WARNING {len(n_unmapped)} MAC(s) not in the config map, "
                  f"{sum(n_unmapped.values()):,} flows dropped: "
                  f"{list(n_unmapped)[:4]}")
    return out


def cache_key(cfg: Config, dataset: str, every: int, limit: int, hs: bool) -> str:
    payload = (f"cf|{dataset}|T{cfg.T_minutes}|K{getattr(cfg,'cf_upstream_packets',5)}"
               f"|up{getattr(cfg,'cf_upstream_only',True)}|L{cfg.max_flows_per_window}"
               f"|every{every}|limit{limit}|hs{hs}|salt{cfg.hash_salt}"
               f"|dg{cfg.dest_group_vocab}"
               f"|ex{sorted(getattr(cfg, 'cf_exclude_days', ()) or ())}"
               f"|msf{getattr(cfg, 'cf_min_day_size_frac', 0.0)}")
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


#: Records per pickle chunk. See `_dump_records`.
CACHE_CHUNK = 20_000


def _dump_records(records: list, path: str) -> None:
    """Write a window cache in chunks, atomically.

    Two separate failures, both paid for. First MEMORY: one `pickle.dump` over
    the whole list keeps the pickler's memo -- an entry per object, and each
    WindowRecord holds several -- alive for the duration of the call, on top of
    the records themselves. At 169k windows that was enough to raise
    MemoryError at the end of a seven-hour build. Chunking gives every call a
    fresh memo.

    Second ATOMICITY: a dump that dies part-way leaves a truncated .pkl that
    the next run finds, trusts and tries to load. Writing to `.part` and
    renaming means a cache file is either complete or absent; the record count
    in the header lets the loader catch a truncation that slips through anyway.
    """
    tmp = path + ".part"
    try:
        with open(tmp, "wb") as fh:
            pickle.dump({"_cf_cache": 2, "n": len(records),
                         "chunk": CACHE_CHUNK}, fh, protocol=4)
            for i in range(0, len(records), CACHE_CHUNK):
                pickle.dump(records[i: i + CACHE_CHUNK], fh, protocol=4)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def _read_records(path: str) -> list:
    """Read a cache written by `_dump_records`, or a legacy single-blob one."""
    with open(path, "rb") as fh:
        head = pickle.load(fh)
        if not (isinstance(head, dict) and head.get("_cf_cache") == 2):
            fh.seek(0)
            return pickle.load(fh)
        out: list = []
        while True:
            try:
                out.extend(pickle.load(fh))
            except EOFError:
                break
    n = head.get("n")
    if n is not None and len(out) != n:
        raise EOFError(
            f"{os.path.basename(path)} is truncated: {len(out):,} of {n:,} "
            f"records. Delete it and rebuild.")
    return out


def load_records(paths: Paths, cfg: Config, root: str, datasets: list[str],
                 every: int = 1, limit: int = 0, with_handshake: bool = True,
                 use_cache: bool = True, verbose: bool = True) -> list[WindowRecord]:
    """Build (or load) windows for several Custom Flow datasets at once."""
    device_map = load_device_map(os.path.join(root, "scripts", "scripts", "configs"))
    if not device_map:
        device_map = load_device_map(os.path.join(root, "scripts", "configs"))
    if not device_map:
        raise FileNotFoundError(
            f"no *-config.yaml found under {root}/scripts -- the MAC to "
            "device-type map lives there and is required for labels")
    os.makedirs(paths.cache, exist_ok=True)
    recs: list[WindowRecord] = []
    for ds in datasets:
        p = os.path.join(paths.cache,
                         f"cf_{ds}_{cache_key(cfg, ds, every, limit, with_handshake)}.pkl")
        if use_cache and os.path.exists(p):
            part = _read_records(p)
            if verbose:
                print(f"  {ds}: {len(part):,} windows from cache")
        else:
            if verbose:
                print(f"building {ds} (T={cfg.T_minutes} min, "
                      f"upstream_only={getattr(cfg,'cf_upstream_only',True)}, "
                      f"K={getattr(cfg,'cf_upstream_packets',5)})")
            part = build_windows_customflow(root, ds, cfg, device_map, every, limit,
                                            with_handshake, verbose=verbose)
            _dump_records(part, p)
        recs += part
    return recs
