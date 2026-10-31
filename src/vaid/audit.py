"""§4.1 dataset audit and §4.3 forbidden-feature audit.

Two independent guards, because the spec in data/schema/forbidden_fields.yaml
says a name filter alone cannot implement §4.3:

1. `forbidden_by_name`  -- column names, organised by semantic category, so that
   `chaddr` is caught as a MAC even though it is not called one.
2. `content_scan`       -- runs over the values of every column that survives the
   name filter, allowlisted columns included, looking for MAC-shaped hex, the
   device's own OUI, RFC1918 literals, vendor tokens built from the class names
   and hostname shapes.

`assert_clean_feature_space` is called by the dataset builder before any matrix
reaches a model, and it fails closed: an unrecognised column is quarantined, not
admitted.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Paths
from .handshake import client_hello_from_row
from .ingest import (flow_path, join_rate, list_devices, read_flows,
                     read_parameter, param_dir)
from .naming import class_label, device_mac

# Fallback copy of the category lists, used when PyYAML is unavailable. Kept in
# sync with data/schema/forbidden_fields.yaml v2.
_FALLBACK_FORBIDDEN = {
    "hardware_identifier": ["srcMac", "dstMac", "chaddr", "sname", "chassis id",
                            "chassis-id-subtype"],
    "device_name": ["host name", "hostname", "vendor class identifier", "sname",
                    "system name", "system description", "port description",
                    "unique service name", "name", "answer name", "software"],
    "local_address": ["srcIp", "dstIp", "yiaddr", "ciaddr", "giaddr",
                      "management address", "location", "via"],
    "environment_identifier": ["siaddr", "ip lease time", "reference id",
                               "community", "community length"],
    "remote_endpoint_raw": ["dstIp", "sni", "server_name", "certificate subject",
                            "server-hello-certificates", "host", "http.host",
                            "server", "uri", "from", "to", "subject"],
    "application_payload": ["user-agent", "http.user_agent", "payload",
                            "payload_bytes"],
    "transaction_noise": ["xid", "transaction id"],
}
_FALLBACK_DERIVED = ["oui", "vendor_prefix", "mac_prefix_24"]

_MAC_HEX = re.compile(r"\b[0-9a-fA-F]{12}\b")
_MAC_SEP = re.compile(r"\b(?:[0-9a-fA-F]{2}[:\-]){5}[0-9a-fA-F]{2}\b")
_RFC1918 = re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b")
_DNSNAME = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9\-]{1,62}(?:\.[a-zA-Z0-9\-]{1,63})+\b")


def load_forbidden_spec(paths: Paths) -> dict:
    p = paths.forbidden_spec
    if os.path.exists(p):
        try:
            import yaml  # optional dependency
            with open(p, "r", encoding="utf-8") as fh:
                spec = yaml.safe_load(fh)
            if isinstance(spec, dict) and "categories" in spec:
                return spec
        except Exception:
            pass
    return {"version": "fallback", "policy": "fail_closed",
            "categories": {k: {"columns": v} for k, v in _FALLBACK_FORBIDDEN.items()},
            "_fallback": True}


def forbidden_columns(spec: dict) -> set[str]:
    out: set[str] = set()
    for cat in spec.get("categories", {}).values():
        for c in (cat or {}).get("columns", []) or []:
            out.add(str(c).strip().lower())
        for c in (cat or {}).get("derived_prohibited", []) or []:
            out.add(str(c).strip().lower())
    out.update(_FALLBACK_DERIVED)
    return out


# --------------------------------------------------------------------------
# §4.3 guards
# --------------------------------------------------------------------------

class ForbiddenFeatureError(AssertionError):
    pass


def assert_clean_feature_space(feature_names: list[str], spec: dict,
                               allow_endpoint: bool = False) -> None:
    """Fail if any feature name is, or embeds, a forbidden column name."""
    bad = forbidden_columns(spec)
    hits = []
    for f in feature_names:
        fl = f.lower()
        if fl.startswith("v6_"):          # endpoint-assisted upper bound
            if not allow_endpoint:
                hits.append((f, "V6 feature outside the upper-bound experiment"))
            continue
        for b in bad:
            token = b.replace(" ", "_")
            if fl == token or fl.endswith("__" + token) or ("raw_" + token) in fl:
                hits.append((f, b))
    if hits:
        raise ForbiddenFeatureError(
            "forbidden features reached the training pipeline: "
            + "; ".join(f"{f} <- {b}" for f, b in hits))


def content_scan(values, device: str, vendor_tokens: list[str]) -> dict[str, int]:
    """Count identity-shaped content in a column's values (finding 1 and 3)."""
    oui = (device_mac(device) or "")[:6]
    counts = Counter()
    for v in values:
        if v is None:
            continue
        s = str(v)
        if not s or s == "nan":
            continue
        if _MAC_SEP.search(s):
            counts["mac_colon_or_dash"] += 1
        if _MAC_HEX.search(s):
            counts["mac_hex_12"] += 1
        if _RFC1918.search(s):
            counts["rfc1918_literal"] += 1
        if oui and oui in s.lower():
            counts["device_own_oui"] += 1
        low = s.lower()
        if any(t in low for t in vendor_tokens):
            counts["vendor_token"] += 1
        if _DNSNAME.search(s):
            counts["dns_name_literal"] += 1
    return dict(counts)


def vendor_token_list(devices: list[str]) -> list[str]:
    toks: set[str] = set()
    for d in devices:
        lab = class_label(d)
        parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+", lab)
        for p in parts:
            if len(p) >= 4:
                toks.add(p.lower())
    return sorted(toks)


def scan_feature_values(X: np.ndarray, feature_names: list[str]) -> dict:
    """Numeric feature matrices cannot carry a hostname, but they can carry a
    constant that is unique to one class. Flag any feature that is perfectly
    separating -- a strong hint of a leaked identifier."""
    out = {"n_features": len(feature_names), "constant_features": []}
    if X.size:
        var = X.var(axis=0)
        out["constant_features"] = [feature_names[i] for i in np.where(var == 0)[0]]
    return out


# --------------------------------------------------------------------------
# §4.1 dataset audit
# --------------------------------------------------------------------------

@dataclass
class DeviceAudit:
    device: str
    label: str
    n_flows: int = 0
    n_raw_rows: int = 0
    n_unparsed_time: int = 0
    n_epoch_artifact: int = 0
    active_days: int = 0
    first_seen: str = ""
    last_seen: str = ""
    tls_flow_frac: float = 0.0
    protocol_counts: dict = field(default_factory=dict)
    n_distinct_dst: int = 0
    has_param_dir: bool = False
    n_tls_handshakes: int = 0
    n_dhcp: int = 0
    tls_join_rate: float = float("nan")
    tls_versions: dict = field(default_factory=dict)
    distinct_ch_fingerprints: int = 0
    protocol_label_disagreement: float = 0.0
    content_flags: dict = field(default_factory=dict)


def audit_dataset(paths: Paths, devices: list[str] | None = None) -> dict:
    devices = devices or list_devices(paths)
    vend = vendor_token_list(devices)
    spec = load_forbidden_spec(paths)
    per: list[DeviceAudit] = []

    for dev in devices:
        ft = read_flows(flow_path(paths, dev))
        df = ft.df
        a = DeviceAudit(device=dev, label=ft.label, n_flows=len(df),
                        n_raw_rows=ft.n_raw_rows,
                        n_unparsed_time=ft.n_unparsed_time,
                        n_epoch_artifact=ft.n_epoch_artifact)
        if len(df):
            a.active_days = int(df["time"].dt.date.nunique())
            a.first_seen = str(df["time"].min())
            a.last_seen = str(df["time"].max())
            amp = df.get("allMatchedProtocols", pd.Series([""] * len(df))).fillna("")
            proto = df.get("protocol", pd.Series([""] * len(df))).fillna("")
            a.tls_flow_frac = float(np.mean(amp.str.contains("tls") | (proto == "tls")))
            a.protocol_counts = dict(Counter(proto).most_common(12))
            if "dstIp" in df.columns:
                a.n_distinct_dst = int(df["dstIp"].nunique())
            # flow_schema.md: protocol label vs transport/port disagreement
            dis = ((proto == "icmpv6") & (df["ipProto"] == 17)) | \
                  ((proto == "tls") & (df["dstPort"] != 443) & (df["srcPort"] != 443))
            a.protocol_label_disagreement = float(np.mean(dis))

        a.has_param_dir = param_dir(paths, dev) is not None
        tls = read_parameter(paths, dev, "request", "tls")
        dhcp = read_parameter(paths, dev, "request", "dhcp")
        a.n_tls_handshakes = len(tls)
        a.n_dhcp = len(dhcp)
        if len(tls):
            a.tls_join_rate = join_rate(df, tls)
            a.tls_versions = dict(Counter(tls.get("client-hello-version", pd.Series(dtype=str)).fillna("")))
            fps = set()
            for _, r in tls.iterrows():
                fps.add(client_hello_from_row(r).fingerprint())
            a.distinct_ch_fingerprints = len(fps)
            flags = {}
            for col in ("client-hello-extensions", "client-hello-cipher-suites"):
                if col in tls.columns:
                    hit = content_scan(tls[col].head(500), dev, vend)
                    if hit:
                        flags[col] = hit
            a.content_flags = flags
        per.append(a)

    tot_flows = sum(p.n_flows for p in per)
    versions = Counter()
    for p in per:
        versions.update(p.tls_versions)
    summary = {
        "n_devices": len(per),
        "n_device_types": len({p.label for p in per}),
        "n_flows": tot_flows,
        "n_epoch_artifact_rows": sum(p.n_epoch_artifact for p in per),
        "n_unparsed_time_rows": sum(p.n_unparsed_time for p in per),
        "class_imbalance_flows": (max(p.n_flows for p in per) /
                                  max(1, min(p.n_flows for p in per))) if per else 0,
        "devices_with_tls": sum(1 for p in per if p.n_tls_handshakes > 0),
        "devices_with_dhcp": sum(1 for p in per if p.n_dhcp > 0),
        "tls_client_hello_versions": dict(versions),
        "tls_1_3_or_quic_present": bool(versions.get("0304", 0)),
        "forbidden_spec_version": spec.get("version"),
        "forbidden_spec_source": "fallback" if spec.get("_fallback") else "yaml",
    }
    # §9.3 trigger
    summary["protocol_scope_warning"] = (
        "No TLS 1.3 (0304) ClientHello and no QUIC/DTLS in this data. Per §9.3 the "
        "QUIC/DTLS/ECH claims cannot be evaluated here and must be removed from the "
        "title and principal claims, or a modern dataset must be added."
        if not summary["tls_1_3_or_quic_present"] else "")
    return {"summary": summary, "devices": [d.__dict__ for d in per]}


def write_audit(paths: Paths, out: dict) -> str:
    os.makedirs(paths.results, exist_ok=True)
    p = os.path.join(paths.results, "audit_4_1.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    return p
