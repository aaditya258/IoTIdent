"""Observation construction (§4.4) and the three views (§4.5-4.7).

One sample = one device x one non-overlapping wall-clock window of T minutes
containing at least `min_flows_per_window` flows. Non-overlapping is the
proposal's default; nothing here generates overlapping windows, so the
"same parent interval must stay in one partition" rule cannot be violated.

What each view may contain is decided here, once:

  F  temporal flow behaviour  -- sizes, counts, durations, interarrival, transport
                                 category. A SEQUENCE of flow events, because
                                 flow-level data carries no packet sequences and
                                 no TCP flags (data/schema/flow_schema.md).
  H  handshake                -- parsed ClientHello types/lengths/permitted values
                                 and the DHCP parameter request list. Never raw
                                 SNI, never certificates.
  C  communication structure  -- counts, diversity, periodicity and hashed
                                 destination GROUPS. Never a raw address.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SERVICE_BUCKETS, SERVICE_ORDER, Config, Paths
from .handshake import client_hello_from_row, parse_dhcp_parameter_request_list
from .ingest import flow_path, read_flows, read_parameter, to_epoch_seconds

SEQ_FEATURES = [
    "log_src_bytes", "log_dst_bytes", "log_src_packets", "log_dst_packets",
    "log_duration_s", "log_avg_iat_s", "log_std_iat_s",
    "log_src_max_payload", "log_dst_max_payload", "log_dt_prev_s",
    "dir_ratio", "is_tcp", "is_udp", "is_other_proto",
]


@dataclass
class WindowRecord:
    device: str
    label: str
    window_id: int
    t_start: float          # epoch seconds
    period: str             # UNSW: "A"/"B"; Custom Flow: "YY-MM"
    n_flows: int
    seq: np.ndarray                       # [n, len(SEQ_FEATURES)]
    agg: dict = field(default_factory=dict)
    comm: dict = field(default_factory=dict)
    dest_groups: list = field(default_factory=list)   # hashed /24 buckets
    hs: dict = field(default_factory=dict)
    network: str = ""       # which capture environment this window came from


def _service_bucket(port, proto) -> str:
    try:
        p = int(port)
    except Exception:
        return "other_eph"
    if p in SERVICE_BUCKETS:
        return SERVICE_BUCKETS[p]
    return "other_priv" if p < 1024 else "other_eph"


def _is_rfc1918(ip: str) -> bool:
    if not isinstance(ip, str):
        return False
    if ip.startswith("192.168.") or ip.startswith("10."):
        return True
    if ip.startswith("172."):
        try:
            return 16 <= int(ip.split(".")[1]) <= 31
        except Exception:
            return False
    return False


def _is_broadcast_or_multicast(ip: str) -> bool:
    if not isinstance(ip, str):
        return False
    if ip == "255.255.255.255" or ip.endswith(".255"):
        return True
    try:
        first = int(ip.split(".")[0])
    except Exception:
        return False
    return 224 <= first <= 239


def _dest_group(ip: str, salt: str, k: int) -> int:
    """Hashed /24 destination group.

    The abstraction the forbidden-field spec permits: the raw address is
    discarded immediately and only a salted bucket index survives, so the model
    can see "these flows went to two different destination groups" without ever
    seeing where.
    """
    if not isinstance(ip, str) or not ip:
        return -1
    parts = ip.split(".")
    net = ".".join(parts[:3]) if len(parts) == 4 else ip
    h = hashlib.blake2b(f"{salt}|{net}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "big") % k


def _safe(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _entropy(counts) -> float:
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    p = np.array([c / tot for c in counts if c > 0])
    return float(-(p * np.log(p)).sum())


def _flow_sequence(g: pd.DataFrame, L: int) -> np.ndarray:
    g = g.head(L)
    t = to_epoch_seconds(g["time"])
    dt = np.diff(t, prepend=t[0] if len(t) else 0.0)
    sb = g["srcPayloadSize"].fillna(0).to_numpy(dtype=float)
    db = g["dstPayloadSize"].fillna(0).to_numpy(dtype=float)
    proto = g["ipProto"].fillna(-1).to_numpy(dtype=float)
    out = np.column_stack([
        np.log1p(np.clip(sb, 0, None)),
        np.log1p(np.clip(db, 0, None)),
        np.log1p(np.clip(g["srcNumPackets"].fillna(0).to_numpy(dtype=float), 0, None)),
        np.log1p(np.clip(g["dstNumPackets"].fillna(0).to_numpy(dtype=float), 0, None)),
        np.log1p(np.clip(g["flowDuration"].fillna(0).to_numpy(dtype=float), 0, None)),
        np.log1p(np.clip(g["avgInterarrivalTime"].fillna(0).to_numpy(dtype=float), 0, None)),
        np.log1p(np.clip(g["stdDevInterarrivalTime"].fillna(0).to_numpy(dtype=float), 0, None)),
        np.log1p(np.clip(g["srcMaxPayloadSize"].fillna(0).to_numpy(dtype=float), 0, None)),
        np.log1p(np.clip(g["dstMaxPayloadSize"].fillna(0).to_numpy(dtype=float), 0, None)),
        np.log1p(np.clip(dt, 0, None)),
        (sb - db) / (sb + db + 1.0),
        (proto == 6).astype(float),
        (proto == 17).astype(float),
        ((proto != 6) & (proto != 17)).astype(float),
    ])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _aggregates(g: pd.DataFrame, T_seconds: float) -> dict:
    t = to_epoch_seconds(g["time"])
    dt = np.diff(t) if len(t) > 1 else np.array([0.0])
    sb = g["srcPayloadSize"].fillna(0).to_numpy(dtype=float)
    db = g["dstPayloadSize"].fillna(0).to_numpy(dtype=float)
    dur = g["flowDuration"].fillna(0).to_numpy(dtype=float)
    iat = g["avgInterarrivalTime"].fillna(0).to_numpy(dtype=float)
    proto = g["ipProto"].fillna(-1).to_numpy(dtype=float)
    labels = g.get("protocol", pd.Series([""] * len(g))).fillna("").to_numpy()
    n = len(g)
    ld = np.log1p(np.clip(dur, 0, None))
    li = np.log1p(np.clip(iat, 0, None))
    lsb, ldb = np.log1p(np.clip(sb, 0, None)), np.log1p(np.clip(db, 0, None))
    ldt = np.log1p(np.clip(dt, 0, None))
    cnt = Counter(labels)
    return {
        "n_flows": float(n),
        "log_n_flows": float(np.log1p(n)),
        "flows_per_min": float(n / (T_seconds / 60.0)),
        "log_total_src_bytes": float(np.log1p(sb.sum())),
        "log_total_dst_bytes": float(np.log1p(db.sum())),
        "up_down_ratio": float((sb.sum() - db.sum()) / (sb.sum() + db.sum() + 1.0)),
        "mean_log_duration": float(ld.mean()), "std_log_duration": float(ld.std()),
        "mean_log_iat": float(li.mean()), "std_log_iat": float(li.std()),
        "mean_log_src_bytes": float(lsb.mean()), "std_log_src_bytes": float(lsb.std()),
        "mean_log_dst_bytes": float(ldb.mean()), "std_log_dst_bytes": float(ldb.std()),
        "mean_log_dt": float(ldt.mean()), "std_log_dt": float(ldt.std()),
        "burst_frac": float(np.mean(dt < 1.0)) if len(t) > 1 else 0.0,
        "frac_tcp": float(np.mean(proto == 6)),
        "frac_udp": float(np.mean(proto == 17)),
        "frac_other_proto": float(np.mean((proto != 6) & (proto != 17))),
        "protocol_label_entropy": _entropy(cnt.values()),
        "n_protocol_labels": float(len(cnt)),
        "active_span_frac": float((t.max() - t.min()) / T_seconds) if n > 1 else 0.0,
    }


def _communication(g: pd.DataFrame, cfg: Config) -> tuple[dict, list]:
    n = len(g)
    dst = g.get("dstIp", pd.Series([""] * n)).fillna("").to_numpy()
    dport = g.get("dstPort", pd.Series([0] * n)).fillna(0).to_numpy()
    proto = g.get("ipProto", pd.Series([0] * n)).fillna(0).to_numpy()
    t = to_epoch_seconds(g["time"])

    remote = np.array([not (_is_rfc1918(d) or _is_broadcast_or_multicast(d)) for d in dst])
    bcast = np.array([_is_broadcast_or_multicast(d) for d in dst])
    buckets = [_service_bucket(p, q) for p, q in zip(dport, proto)]
    bc = Counter(buckets)
    groups = [_dest_group(d, cfg.hash_salt, cfg.dest_group_vocab)
              for d, r in zip(dst, remote) if r]

    seen, repeats = set(), 0
    for d in dst:
        if d in seen:
            repeats += 1
        seen.add(d)
    gaps = np.diff(t) if n > 1 else np.array([0.0])
    cv = float(gaps.std() / (gaps.mean() + 1e-9)) if n > 1 else 0.0
    transitions = sum(1 for i in range(1, len(buckets)) if buckets[i] != buckets[i - 1])

    feats = {
        "n_dst_distinct": float(len(set(dst))),
        "log_n_dst_distinct": float(np.log1p(len(set(dst)))),
        "dst_cardinality_norm": float(len(set(dst)) / max(1, n)),
        "n_service_groups": float(len(bc)),
        "service_entropy": _entropy(bc.values()),
        "frac_remote": float(remote.mean()),
        "frac_broadcast": float(bcast.mean()),
        "frac_local": float(np.mean([_is_rfc1918(d) for d in dst])),
        "n_dst_ports_distinct": float(len(set(dport.tolist()))),
        "repeat_rate": float(repeats / max(1, n)),
        "periodicity": float(1.0 / (1.0 + cv)),
        "service_transitions_norm": float(transitions / max(1, n - 1)),
    }
    for b in SERVICE_ORDER:
        feats[f"frac_service_{b}"] = float(bc.get(b, 0) / max(1, n))
    return feats, groups


def _handshake(tls: pd.DataFrame, dhcp: pd.DataFrame) -> dict:
    hs = {"tls_present": 0.0, "dhcp_present": 0.0, "n_tls": 0.0, "n_dhcp": 0.0,
          "version": "", "ciphers": [], "ext_types": [], "ext_len_total": 0.0,
          "groups": [], "ecpf": [], "n_sig_algs": 0.0, "has_alpn": 0.0,
          "has_ticket": 0.0, "has_keyshare": 0.0, "has_sni": 0.0, "sni_len": 0.0,
          "n_ciphers": 0.0, "n_exts": 0.0, "n_fingerprints": 0.0, "prl": []}
    if tls is not None and len(tls):
        hs["tls_present"] = 1.0
        hs["n_tls"] = float(len(tls))
        fps = set()
        last = None
        for _, r in tls.iterrows():
            ch = client_hello_from_row(r)
            fps.add(ch.fingerprint())
            last = ch
        hs["n_fingerprints"] = float(len(fps))
        if last is not None:
            hs.update({
                "version": last.version,
                "ciphers": last.cipher_suites,
                "ext_types": last.ext_types,
                "ext_len_total": float(sum(last.ext_lengths)),
                "groups": last.supported_groups,
                "ecpf": last.ec_point_formats,
                "n_sig_algs": float(last.n_signature_algorithms),
                "has_alpn": float(last.has_alpn),
                "has_ticket": float(last.has_session_ticket),
                "has_keyshare": float(last.has_key_share),
                "has_sni": float(last.has_sni),
                "sni_len": float(np.log1p(last.sni_len)),
                "n_ciphers": float(len(last.cipher_suites)),
                "n_exts": float(len(last.ext_types)),
            })
    if dhcp is not None and len(dhcp):
        hs["dhcp_present"] = 1.0
        hs["n_dhcp"] = float(len(dhcp))
        col = "parameter request list"
        if col in dhcp.columns:
            vals = [v for v in dhcp[col].tolist() if isinstance(v, str) and v]
            if vals:
                hs["prl"] = parse_dhcp_parameter_request_list(vals[-1])
    return hs


def build_windows(paths: Paths, cfg: Config, devices: list[str] | None = None,
                  verbose: bool = True) -> list[WindowRecord]:
    from .ingest import list_devices
    devices = devices or list_devices(paths)
    T = cfg.T_minutes * 60
    out: list[WindowRecord] = []

    for dev in devices:
        ft = read_flows(flow_path(paths, dev))
        df = ft.df
        if df.empty:
            continue
        tls = read_parameter(paths, dev, "request", "tls")
        dhcp = read_parameter(paths, dev, "request", "dhcp")
        epoch = to_epoch_seconds(df["time"])
        df = df.assign(_win=(epoch // T).astype(np.int64))
        tls_win = (to_epoch_seconds(tls["time"]) // T).astype(np.int64) if len(tls) else np.array([], dtype=np.int64)
        dhcp_win = (to_epoch_seconds(dhcp["time"]) // T).astype(np.int64) if len(dhcp) else np.array([], dtype=np.int64)

        for wid, g in df.groupby("_win", sort=True):
            if len(g) < cfg.min_flows_per_window:
                continue
            g = g.sort_values("time", kind="stable")
            t0 = float(wid) * T
            per = "A" if pd.Timestamp(t0, unit="s").year <= 2016 else "B"
            comm, groups = _communication(g, cfg)
            hs = _handshake(tls[tls_win == wid] if len(tls) else None,
                            dhcp[dhcp_win == wid] if len(dhcp) else None)
            out.append(WindowRecord(
                device=dev, label=ft.label, window_id=int(wid), t_start=t0,
                period=per, n_flows=len(g),
                seq=_flow_sequence(g, cfg.max_flows_per_window),
                agg=_aggregates(g, T), comm=comm, dest_groups=groups, hs=hs))
        if verbose:
            print(f"  {dev:40s} windows={sum(1 for r in out if r.device == dev):5d}")
    return out
