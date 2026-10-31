"""Configuration for the visibility-aware open-world IoT identification framework.

Every constant that the proposal fixes by name lives here so that a reviewer can
check the implementation against the manuscript without reading the model code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Sequence

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def default_repo_root() -> str:
    """Repository root, assuming this file lives at <root>/src/vaid/config.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


@dataclass
class Paths:
    root: str = field(default_factory=default_repo_root)

    @property
    def samples(self) -> str:
        return os.path.join(self.root, "data", "samples")

    @property
    def raw(self) -> str:
        return os.path.join(self.root, "data", "raw")

    @property
    def flows(self) -> str:
        return os.path.join(self.samples, "flows")

    @property
    def parameters(self) -> str:
        return os.path.join(self.samples, "protocols", "parameters")

    @property
    def forbidden_spec(self) -> str:
        return os.path.join(self.root, "data", "schema", "forbidden_fields.yaml")

    @property
    def cache(self) -> str:
        return os.path.join(self.root, "data", "cache")

    @property
    def results(self) -> str:
        return os.path.join(self.root, "results")


# --------------------------------------------------------------------------
# Observation construction (§4.4)
# --------------------------------------------------------------------------

#: Candidate observation durations in minutes. T* is selected on validation only.
T_CANDIDATES: tuple[int, ...] = (5, 15, 30, 60, 120)

#: Flow-schema columns. `flowSeqNum` is the join key to the parameter files and
#: must survive ingest (data/schema/flow_schema.md).
FLOW_NUMERIC = [
    "srcNumPackets", "dstNumPackets",
    "srcPayloadSize", "dstPayloadSize",
    "srcAvgPayloadSize", "dstAvgPayloadSize",
    "srcMaxPayloadSize", "dstMaxPayloadSize",
    "srcStdDevPayloadSize", "dstStdDevPayloadSize",
    "flowDuration",
    "srcAvgInterarrivalTime", "dstAvgInterarrivalTime", "avgInterarrivalTime",
    "srcStdDevInterarrivalTime", "dstStdDevInterarrivalTime", "stdDevInterarrivalTime",
]

#: The six-field key that aligns protocol-parameter rows to flow rows.
JOIN_KEY = ["srcIp", "dstIp", "ipProto", "srcPort", "dstPort", "flowSeqNum"]

#: Service buckets for the communication-structure view (§4.7). Ports only —
#: never the remote address itself.
SERVICE_BUCKETS = {
    443: "tls", 80: "http", 8080: "http", 53: "dns", 123: "ntp",
    1900: "ssdp", 5353: "mdns", 67: "dhcp", 68: "dhcp", 137: "nbns",
    138: "nbns", 139: "smb", 445: "smb", 554: "rtsp", 5060: "sip",
    3478: "stun", 1883: "mqtt", 8883: "mqtt", 514: "syslog", 161: "snmp",
}
SERVICE_ORDER = ["tls", "http", "dns", "ntp", "ssdp", "mdns", "dhcp", "nbns",
                 "smb", "rtsp", "sip", "stun", "mqtt", "syslog", "snmp",
                 "other_priv", "other_eph"]


@dataclass
class Config:
    # -- observation construction -----------------------------------------
    T_minutes: int = 30
    max_flows_per_window: int = 32       # L in §4.4 ("first L relevant flows")
    min_flows_per_window: int = 1
    #: Classes with fewer windows than this are not trainable and are reported
    #: separately (see claude/sample-sufficiency-assessment.md).
    min_windows_per_class: int = 30
    #: Route excluded tail classes into the open-set unknown pool instead of
    #: discarding them.
    tail_classes_to_unknown: bool = True

    # -- Custom Flow (DATA2016 / DATA2025) --------------------------------
    #: Keep only device->remote packets and recompute byte/packet counts from
    #: them. REQUIRED for any cross-network claim: DATA2025v1 is bidirectional
    #: while v2 is upstream-only, and TotalFlowSize/PacketCount mean different
    #: things in the two. See customflow.py.
    #: Day files to exclude by basename stem, e.g. ("25-10-13", "25-10-14").
    cf_exclude_days: tuple = ()
    #: Drop days smaller than this fraction of the median day size. 0 disables.
    #: 0.25 removes the two truncated DATA2025v2 captures.
    cf_min_day_size_frac: float = 0.0
    #: Cap windows per class before splitting, by whole adjacency group. 0 keeps
    #: everything. The full span is ~20x the development set and the neural arm
    #: is refit 20 times by the threshold command, so the expensive arm needs a
    #: bound that does not shorten the calendar span.
    cf_cap_per_class: int = 0
    cf_upstream_only: bool = True
    #: Number of upstream packets per flow used for the sequence view. "First
    #: 10 packets of the flow" is ~6.5 upstream packets on v1 but 10 on v2, so a
    #: small fixed cap is what makes the two comparable.
    cf_upstream_packets: int = 5

    # -- leakage control (§6.1) -------------------------------------------
    #: Adjacency/session grouping block in hours. All windows inside one block
    #: for one device stay in the same partition.
    group_block_hours: float = 3.0

    # -- view construction -------------------------------------------------
    cipher_vocab: int = 24
    ext_vocab: int = 20
    group_vocab: int = 8        # supported_groups multi-hot
    prl_vocab: int = 16         # DHCP parameter-request-list options
    dest_group_vocab: int = 32  # hashed /24 destination buckets (V2 only)
    hash_salt: str = "vaid-v1"  # salt for destination-group hashing

    # -- model -------------------------------------------------------------
    backend: str = "auto"       # auto | torch | shallow
    embed_dim: int = 64
    tcn_channels: int = 64
    tcn_dilations: Sequence[int] = (1, 2, 4)
    dropout: float = 0.1
    epochs: int = 40
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 1e-4
    supcon_lambda: float = 0.1
    supcon_temperature: float = 0.1
    balanced_sampling: bool = True
    early_stop_patience: int = 8

    # -- view dropout (§5.6) ----------------------------------------------
    p_drop_H: float = 0.5
    p_drop_C: float = 0.25
    #: If True, p_v is sampled per batch from U(0, 2*p) rather than fixed.
    sample_dropout_rate: bool = True

    # -- open set (§5.9) ---------------------------------------------------
    openset_metric: str = "cosine"      # cosine | euclidean
    target_known_tpr: float = 0.95      # threshold calibrated on validation-unknowns

    # -- evaluation --------------------------------------------------------
    n_repeats: int = 5                  # §6.6 requires >= 5
    n_bootstrap: int = 2000
    seed: int = 20260906

    # -- safety ------------------------------------------------------------
    #: V6 (endpoint-assisted upper bound) is refused unless this is set.
    allow_endpoint_upper_bound: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tcn_dilations"] = list(self.tcn_dilations)
        d["cf_exclude_days"] = list(self.cf_exclude_days)
        return d
