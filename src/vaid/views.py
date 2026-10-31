"""Turning window records into model matrices, with train-only fitting.

Everything vocabulary-shaped (cipher suites, extension types, supported groups,
DHCP option lists, destination-group buckets) and every scaler is fitted on the
TRAINING indices only and then applied to validation and test. That is what
makes §4.5's "normalized using parameters estimated from the training partition
only" true in code rather than in prose.

The handshake matrix is deliberately built in blocks, because the visibility
profiles need to remove *part* of a view:

    tls_basic   presence, version, handshake count      -- survives V4
    tls_detail  ciphers, extension types, groups, ...   -- removed by V4
    dhcp_basic  presence, count                          -- survives V4
    dhcp_detail parameter request list                   -- removed by V4

Field-level masks are carried alongside so that "absent" is representable
distinctly from "present and zero" (§3.2, §5.3).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .handshake import TLS_VERSIONS
from .windows import SEQ_FEATURES, WindowRecord

AGG_KEYS: list[str] = []
COMM_KEYS: list[str] = []


@dataclass
class Bundle:
    """Everything a model needs, plus everything an auditor needs."""
    Xseq: np.ndarray
    seq_mask: np.ndarray
    Xagg: np.ndarray
    Xh: np.ndarray
    Xc: np.ndarray
    mH: np.ndarray
    mC: np.ndarray
    h_field_mask: np.ndarray          # [N, 4] tls_basic, tls_detail, dhcp_basic, dhcp_detail
    y: np.ndarray
    labels: list[str]                 # class index -> label
    device: np.ndarray
    t_start: np.ndarray
    period: np.ndarray
    group: np.ndarray                 # session/adjacency group id
    n_flows: np.ndarray
    feature_names: dict = field(default_factory=dict)
    h_blocks: dict = field(default_factory=dict)
    c_blocks: dict = field(default_factory=dict)
    Xv6: np.ndarray | None = None
    #: capture environment per window (Custom Flow); "" for a single-source run
    network: np.ndarray | None = None
    #: physical unit per window. Equal to `device`, kept separate so that
    #: instance-held-out splits read the right thing even when `device` is
    #: repurposed. Never a feature.
    instance: np.ndarray | None = None

    def subset(self, idx) -> "Bundle":
        idx = np.asarray(idx)
        return Bundle(
            Xseq=self.Xseq[idx], seq_mask=self.seq_mask[idx], Xagg=self.Xagg[idx],
            Xh=self.Xh[idx], Xc=self.Xc[idx], mH=self.mH[idx], mC=self.mC[idx],
            h_field_mask=self.h_field_mask[idx], y=self.y[idx], labels=self.labels,
            device=self.device[idx], t_start=self.t_start[idx],
            period=self.period[idx], group=self.group[idx], n_flows=self.n_flows[idx],
            feature_names=self.feature_names, h_blocks=self.h_blocks,
            c_blocks=self.c_blocks,
            Xv6=None if self.Xv6 is None else self.Xv6[idx],
            network=None if self.network is None else self.network[idx],
            instance=None if self.instance is None else self.instance[idx])

    def __len__(self) -> int:
        return len(self.y)


class ViewSpace:
    """Fits vocabularies and scalers on training windows only."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cipher_vocab: list[int] = []
        self.ext_vocab: list[int] = []
        self.group_vocab: list[int] = []
        self.prl_vocab: list[int] = []
        self.agg_keys: list[str] = []
        self.comm_keys: list[str] = []
        self.mu_seq = self.sd_seq = None
        self.mu_agg = self.sd_agg = None
        self.mu_c = self.sd_c = None
        self.fitted = False

    # -- fitting -----------------------------------------------------------
    @staticmethod
    def _top(counter: Counter, k: int) -> list:
        return [v for v, _ in counter.most_common(k)]

    def fit(self, recs: list[WindowRecord], train_idx: np.ndarray) -> "ViewSpace":
        cfg = self.cfg
        cc, ec, gc, pc = Counter(), Counter(), Counter(), Counter()
        for i in train_idx:
            hs = recs[i].hs
            cc.update(set(hs.get("ciphers", [])))
            ec.update(set(hs.get("ext_types", [])))
            gc.update(set(hs.get("groups", [])))
            pc.update(set(hs.get("prl", [])))
        self.cipher_vocab = sorted(self._top(cc, cfg.cipher_vocab))
        self.ext_vocab = sorted(self._top(ec, cfg.ext_vocab))
        self.group_vocab = sorted(self._top(gc, cfg.group_vocab))
        self.prl_vocab = sorted(self._top(pc, cfg.prl_vocab))
        self.agg_keys = sorted(recs[train_idx[0]].agg.keys())
        self.comm_keys = sorted(recs[train_idx[0]].comm.keys())

        seq = np.concatenate([recs[i].seq for i in train_idx], axis=0)
        self.mu_seq, self.sd_seq = seq.mean(0), seq.std(0) + 1e-6
        agg = np.array([[recs[i].agg[k] for k in self.agg_keys] for i in train_idx])
        self.mu_agg, self.sd_agg = agg.mean(0), agg.std(0) + 1e-6
        comm = np.array([self._comm_row(recs[i]) for i in train_idx])
        self.mu_c, self.sd_c = comm.mean(0), comm.std(0) + 1e-6
        self.fitted = True
        return self

    # -- per-record rows ---------------------------------------------------
    def _comm_row(self, r: WindowRecord) -> np.ndarray:
        base = [r.comm[k] for k in self.comm_keys]
        bag = np.zeros(self.cfg.dest_group_vocab)
        for g in r.dest_groups:
            if 0 <= g < len(bag):
                bag[g] += 1.0
        bag = bag / max(1.0, bag.sum())
        return np.array(base + bag.tolist(), dtype=float)

    def _h_row(self, r: WindowRecord) -> tuple[np.ndarray, dict]:
        # .get() throughout: UNSW-IoTraffic records carry a subset of these
        # keys and Custom Flow records carry all of them.
        hs = r.hs
        g = hs.get
        tls_basic = [g("tls_present", 0.0), np.log1p(g("n_tls", 0.0)),
                     g("frac_truncated", 0.0)]
        tls_basic += [1.0 if g("version", "") == v else 0.0 for v in TLS_VERSIONS]
        tls_basic += [1.0 if (g("version", "") == "other") else 0.0]

        cipher = np.zeros(len(self.cipher_vocab))
        for c in g("ciphers", []):
            if c in self.cipher_vocab:
                cipher[self.cipher_vocab.index(c)] = 1.0
        ext = np.zeros(len(self.ext_vocab))
        for e in g("ext_types", []):
            if e in self.ext_vocab:
                ext[self.ext_vocab.index(e)] = 1.0
        grp = np.zeros(len(self.group_vocab))
        for gg in g("groups", []):
            if gg in self.group_vocab:
                grp[self.group_vocab.index(gg)] = 1.0
        tls_detail = (cipher.tolist() + ext.tolist() + grp.tolist() + [
            np.log1p(g("n_ciphers", 0.0)), np.log1p(g("n_exts", 0.0)),
            np.log1p(g("ext_len_total", 0.0)), np.log1p(g("n_sig_algs", 0.0)),
            len(g("ecpf", [])) / 4.0, g("has_alpn", 0.0), g("has_ticket", 0.0),
            g("has_keyshare", 0.0), g("has_sni", 0.0), g("sni_len", 0.0),
            np.log1p(g("n_fingerprints", 0.0)),
            g("tls13", 0.0), g("has_ech", 0.0), g("has_psk", 0.0),
        ])

        dhcp_basic = [g("dhcp_present", 0.0), np.log1p(g("n_dhcp", 0.0))]
        prl = np.zeros(len(self.prl_vocab))
        for o in g("prl", []):
            if o in self.prl_vocab:
                prl[self.prl_vocab.index(o)] = 1.0
        dhcp_detail = prl.tolist() + [len(g("prl", [])) / 16.0]

        row = np.array(tls_basic + tls_detail + dhcp_basic + dhcp_detail, dtype=float)
        blocks = {
            "tls_basic": (0, len(tls_basic)),
            "tls_detail": (len(tls_basic), len(tls_basic) + len(tls_detail)),
            "dhcp_basic": (len(tls_basic) + len(tls_detail),
                           len(tls_basic) + len(tls_detail) + len(dhcp_basic)),
            "dhcp_detail": (len(tls_basic) + len(tls_detail) + len(dhcp_basic), len(row)),
        }
        return np.nan_to_num(row), blocks

    def _h_names(self) -> list[str]:
        n = ["tls_present", "tls_n_handshakes", "tls_frac_truncated"] \
            + [f"tls_version_{v}" for v in TLS_VERSIONS] + ["tls_version_other"]
        n += [f"tls_cipher_{c:04x}" for c in self.cipher_vocab]
        n += [f"tls_ext_type_{e:04x}" for e in self.ext_vocab]
        n += [f"tls_supported_group_{g:04x}" for g in self.group_vocab]
        n += ["tls_n_ciphers", "tls_n_exts", "tls_ext_len_total", "tls_n_sig_algs",
              "tls_n_ec_point_formats", "tls_alpn_present", "tls_session_ticket_present",
              "tls_key_share_present", "tls_sni_present", "tls_sni_length_log",
              "tls_n_distinct_fingerprints", "tls13_offered",
              "tls_ech_extension_present", "tls_psk_present"]
        n += ["dhcp_present", "dhcp_n"]
        n += [f"dhcp_prl_option_{o}" for o in self.prl_vocab] + ["dhcp_prl_len"]
        return n

    # -- transform ---------------------------------------------------------
    def transform(self, recs: list[WindowRecord], labels: list[str] | None = None,
                  allow_v6: bool = False) -> Bundle:
        assert self.fitted, "ViewSpace.fit must run on training indices first"
        cfg = self.cfg
        L, dF = cfg.max_flows_per_window, len(SEQ_FEATURES)
        N = len(recs)

        Xseq = np.zeros((N, L, dF), dtype=np.float32)
        seq_mask = np.zeros((N, L), dtype=np.float32)
        Xagg = np.zeros((N, len(self.agg_keys)), dtype=np.float32)
        h_rows, blocks = [], None
        Xc = np.zeros((N, len(self.comm_keys) + cfg.dest_group_vocab), dtype=np.float32)
        mH = np.zeros(N, dtype=np.float32)
        mC = np.zeros(N, dtype=np.float32)
        hfm = np.zeros((N, 4), dtype=np.float32)

        for i, r in enumerate(recs):
            s = (r.seq - self.mu_seq) / self.sd_seq
            k = min(len(s), L)
            Xseq[i, :k] = s[:k]
            seq_mask[i, :k] = 1.0
            Xagg[i] = (np.array([r.agg[q] for q in self.agg_keys]) - self.mu_agg) / self.sd_agg
            row, blocks = self._h_row(r)
            h_rows.append(row)
            Xc[i] = (self._comm_row(r) - self.mu_c) / self.sd_c
            mH[i] = 1.0 if (r.hs["tls_present"] or r.hs["dhcp_present"]) else 0.0
            mC[i] = 1.0 if r.n_flows >= 2 else 0.0
            hfm[i] = [r.hs["tls_present"], r.hs["tls_present"],
                      r.hs["dhcp_present"], r.hs["dhcp_present"]]

        Xh = np.asarray(h_rows, dtype=np.float32)
        labs = labels or sorted({r.label for r in recs})
        y = np.array([labs.index(r.label) if r.label in labs else -1 for r in recs])

        c_blocks = {"structure": (0, len(self.comm_keys)),
                    "dest_groups": (len(self.comm_keys), Xc.shape[1])}
        names = {
            "seq": SEQ_FEATURES,
            "agg": list(self.agg_keys),
            "h": self._h_names(),
            "c": list(self.comm_keys) + [f"dest_group_bucket_{i}" for i in range(cfg.dest_group_vocab)],
        }

        Xv6 = None
        if allow_v6:
            # Endpoint-assisted upper bound only (§4.3 final paragraph). Named
            # with a v6_ prefix so the audit refuses it in every other setting.
            import hashlib
            K = 256
            Xv6 = np.zeros((N, K), dtype=np.float32)
            for i, r in enumerate(recs):
                for g in r.dest_groups:
                    h = hashlib.blake2b(str(g).encode(), digest_size=2).digest()
                    Xv6[i, int.from_bytes(h, "big") % K] += 1.0
                if Xv6[i].sum() > 0:
                    Xv6[i] /= Xv6[i].sum()
            names["v6"] = [f"v6_endpoint_bucket_{i}" for i in range(K)]

        return Bundle(
            Xseq=Xseq, seq_mask=seq_mask, Xagg=Xagg, Xh=Xh, Xc=Xc, mH=mH, mC=mC,
            h_field_mask=hfm, y=y, labels=labs,
            device=np.array([r.device for r in recs]),
            t_start=np.array([r.t_start for r in recs], dtype=float),
            period=np.array([r.period for r in recs]),
            group=np.array([f"{getattr(r, 'network', '')}|{r.device}|"
                            f"{int(r.t_start // (self.cfg.group_block_hours * 3600))}"
                            for r in recs]),
            n_flows=np.array([r.n_flows for r in recs]),
            feature_names=names, h_blocks=blocks or {}, c_blocks=c_blocks, Xv6=Xv6,
            network=np.array([getattr(r, "network", "") for r in recs]),
            instance=np.array([r.device for r in recs]))


def all_feature_names(b: Bundle) -> list[str]:
    out = []
    for k in ("seq", "agg", "h", "c"):
        out += [f"{k}__{n}" for n in b.feature_names.get(k, [])]
    return out
