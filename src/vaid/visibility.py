"""Visibility profiles V0-V6 (§6.2) and view dropout (§5.6).

A profile is applied at EVALUATION time and is a pure function of a Bundle: it
returns a copy in which the suppressed information is genuinely zeroed and the
corresponding mask bit is cleared. Nothing is imputed. A model that wants to
know whether a view was observed must read the mask, which is exactly the
distinction §3.2 insists on -- a missing handshake is not a handshake whose
fields happen to be zero.

Profiles, from the manuscript's Table:
    V0  flow timing/size/direction/transport only
    V1  V0 + observable handshake metadata
    V2  V1 + abstract communication-structure metadata
    V3  V2 with raw endpoint identity suppressed
    V4  partial handshake visibility (version and presence survive; ciphers,
        extensions, groups and the DHCP option list are removed)
    V5  no handshake view
    V6  endpoint-assisted upper bound -- NOT part of the principal model
"""
from __future__ import annotations

import copy

import numpy as np

from .config import Config
from .views import Bundle

PROFILES = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]
PRINCIPAL_PROFILES = ["V0", "V1", "V2", "V3", "V4", "V5"]

PROFILE_DOC = {
    "V0": "flow only",
    "V1": "flow + handshake",
    "V2": "flow + handshake + communication structure",
    "V3": "V2, raw endpoint identity suppressed",
    "V4": "partial handshake (version/presence only)",
    "V5": "no handshake",
    "V6": "endpoint-assisted upper bound (non-principal)",
}


def apply_profile(b: Bundle, profile: str, cfg: Config | None = None) -> Bundle:
    if profile not in PROFILES:
        raise ValueError(f"unknown visibility profile {profile}")
    out = copy.copy(b)
    out.Xh = b.Xh.copy()
    out.Xc = b.Xc.copy()
    out.mH = b.mH.copy()
    out.mC = b.mC.copy()
    out.h_field_mask = b.h_field_mask.copy()

    def kill_h():
        out.Xh[:] = 0.0
        out.mH[:] = 0.0
        out.h_field_mask[:] = 0.0

    def kill_c():
        out.Xc[:] = 0.0
        out.mC[:] = 0.0

    def kill_block(blocks, name):
        if name in blocks:
            lo, hi = blocks[name]
            out.Xh[:, lo:hi] = 0.0

    if profile == "V0":
        kill_h(); kill_c()
    elif profile == "V1":
        kill_c()
    elif profile == "V2":
        pass
    elif profile == "V3":
        lo, hi = b.c_blocks.get("dest_groups", (0, 0))
        if hi > lo:
            out.Xc[:, lo:hi] = 0.0
    elif profile == "V4":
        kill_block(b.h_blocks, "tls_detail")
        kill_block(b.h_blocks, "dhcp_detail")
        out.h_field_mask[:, 1] = 0.0
        out.h_field_mask[:, 3] = 0.0
    elif profile == "V5":
        kill_h()
    elif profile == "V6":
        if cfg is not None and not cfg.allow_endpoint_upper_bound:
            raise PermissionError(
                "V6 is the endpoint-assisted upper bound. Set "
                "Config.allow_endpoint_upper_bound=True and report it separately; "
                "it is not comparable with the principal tables (§4.3).")
        if b.Xv6 is not None:
            out.Xc = np.concatenate([out.Xc, b.Xv6], axis=1).astype(np.float32)
    return out


def sample_view_dropout(n: int, cfg: Config, rng: np.random.Generator
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Controlled view dropout for training (§5.6).

    Returns keep-masks for H and C. The flow view is never dropped: m_F = 1.
    With `sample_dropout_rate` the per-batch probability is itself sampled, so
    the model sees a range of deployment visibilities rather than one rate.
    """
    pH = rng.uniform(0.0, min(1.0, 2 * cfg.p_drop_H)) if cfg.sample_dropout_rate else cfg.p_drop_H
    pC = rng.uniform(0.0, min(1.0, 2 * cfg.p_drop_C)) if cfg.sample_dropout_rate else cfg.p_drop_C
    keep_H = (rng.random(n) >= pH).astype(np.float32)
    keep_C = (rng.random(n) >= pC).astype(np.float32)
    return keep_H, keep_C
