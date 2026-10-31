"""The proposed framework (§5) in PyTorch, plus the fusion baselines it is
compared against (§6.3).

    E_F  temporal convolutional encoder over the flow-event sequence. A TCN and
         not a Transformer by default, per §5.2; a compact Transformer and a
         1D-CNN are provided as baselines.
    E_H  set/multi-hot encoder over parsed handshake fields, multiplied by the
         FIELD-level availability mask so an absent field is not a zero-valued
         observed field.
    E_C  MLP over communication-structure statistics.

    fusion='gated'      a_v = g_v(h_v, m); alpha = softmax over AVAILABLE views;
                        z = sum alpha_v h_v.  alpha_v is exactly 0 when m_v = 0.
    fusion='attention'  same gate WITHOUT the mask input and softmaxed over all
                        three views -- the visibility-unaware control.
    fusion='concat'     early concatenation with zero-imputation (+ mask appended)
    fusion='late'       per-view heads, logits averaged over available views
    fusion='fixed'      equal weights over available views

The difference between 'gated' and 'attention'/'concat' is the paper's claim, so
they share every other component: same encoders, same optimiser, same schedule,
same seed.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:                                    # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = object

from .config import Config
from .views import Bundle
from .visibility import sample_view_dropout

VIEWS = ("F", "H", "C")


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class TCNEncoder(nn.Module):
        def __init__(self, d_in: int, ch: int, dilations, d_out: int, p: float):
            super().__init__()
            layers = []
            c_prev = d_in
            for d in dilations:
                layers.append(nn.Conv1d(c_prev, ch, kernel_size=3, padding=d, dilation=d))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(p))
                c_prev = ch
            self.net = nn.Sequential(*layers)
            self.proj = nn.Linear(2 * ch, d_out)

        def forward(self, x, mask):                    # x [B,L,d_in]
            h = self.net(x.transpose(1, 2)).transpose(1, 2)    # [B,L,ch]
            m = mask.unsqueeze(-1)
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            mx = (h + (m - 1) * 1e4).max(dim=1).values
            return self.proj(torch.cat([mean, mx], dim=-1))

    class CNN1DEncoder(nn.Module):
        def __init__(self, d_in: int, ch: int, d_out: int, p: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(d_in, ch, 3, padding=1), nn.GELU(), nn.Dropout(p),
                nn.Conv1d(ch, ch, 3, padding=1), nn.GELU())
            self.proj = nn.Linear(ch, d_out)

        def forward(self, x, mask):
            h = self.net(x.transpose(1, 2)).transpose(1, 2)
            m = mask.unsqueeze(-1)
            return self.proj((h * m).sum(1) / m.sum(1).clamp(min=1.0))

    class TransformerEncoder(nn.Module):
        def __init__(self, d_in: int, ch: int, d_out: int, p: float, heads: int = 4):
            super().__init__()
            self.inp = nn.Linear(d_in, ch)
            self.pos = nn.Parameter(torch.randn(1, 512, ch) * 0.02)
            layer = nn.TransformerEncoderLayer(ch, heads, ch * 2, dropout=p,
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=2)
            self.proj = nn.Linear(ch, d_out)

        def forward(self, x, mask):
            h = self.inp(x) + self.pos[:, : x.shape[1]]
            h = self.enc(h, src_key_padding_mask=(mask < 0.5))
            m = mask.unsqueeze(-1)
            return self.proj((h * m).sum(1) / m.sum(1).clamp(min=1.0))

    class MLPEncoder(nn.Module):
        def __init__(self, d_in: int, d_hidden: int, d_out: int, p: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(max(1, d_in), d_hidden), nn.GELU(), nn.Dropout(p),
                nn.Linear(d_hidden, d_out))

        def forward(self, x):
            return self.net(x)

    class VisibilityAwareModel(nn.Module):
        def __init__(self, cfg: Config, d_seq: int, d_agg: int, d_h: int, d_c: int,
                     n_classes: int, fusion: str = "gated",
                     seq_encoder: str = "tcn", use_views=VIEWS):
            super().__init__()
            self.cfg, self.fusion, self.use_views = cfg, fusion, tuple(use_views)
            d = cfg.embed_dim
            if seq_encoder == "tcn":
                self.enc_seq = TCNEncoder(d_seq, cfg.tcn_channels, cfg.tcn_dilations, d, cfg.dropout)
            elif seq_encoder == "cnn1d":
                self.enc_seq = CNN1DEncoder(d_seq, cfg.tcn_channels, d, cfg.dropout)
            elif seq_encoder == "transformer":
                self.enc_seq = TransformerEncoder(d_seq, cfg.tcn_channels, d, cfg.dropout)
            else:
                raise ValueError(seq_encoder)
            self.enc_agg = MLPEncoder(d_agg, 128, d, cfg.dropout)
            self.mix_F = nn.Linear(2 * d, d)
            self.enc_H = MLPEncoder(d_h, 128, d, cfg.dropout)
            self.enc_C = MLPEncoder(d_c, 128, d, cfg.dropout)

            gate_in = d + (len(VIEWS) if fusion == "gated" else 0)
            self.gates = nn.ModuleDict({
                v: nn.Sequential(nn.Linear(gate_in, 32), nn.Tanh(), nn.Linear(32, 1))
                for v in VIEWS})
            if fusion == "concat":
                self.head = nn.Sequential(
                    nn.Linear(3 * d + len(VIEWS), d), nn.GELU(), nn.Linear(d, n_classes))
                self.zproj = nn.Linear(3 * d + len(VIEWS), d)
            else:
                self.head = nn.Linear(d, n_classes)
                self.per_view_head = nn.ModuleDict({v: nn.Linear(d, n_classes) for v in VIEWS})

        def encode(self, batch):
            h = {}
            hs = self.enc_seq(batch["Xseq"], batch["seq_mask"])
            ha = self.enc_agg(batch["Xagg"])
            h["F"] = self.mix_F(torch.cat([hs, ha], dim=-1))
            h["H"] = self.enc_H(batch["Xh"] * batch["h_field_expand"])
            h["C"] = self.enc_C(batch["Xc"])
            return h

        def forward(self, batch):
            m = torch.stack([batch["mF"], batch["mH"], batch["mC"]], dim=1)  # [B,3]
            for i, v in enumerate(VIEWS):
                if v not in self.use_views:
                    m[:, i] = 0.0
            m[:, 0] = torch.clamp(m[:, 0], min=1.0 if "F" in self.use_views else 0.0)
            h = self.encode(batch)
            hv = torch.stack([h[v] for v in VIEWS], dim=1)                   # [B,3,d]
            hv = hv * m.unsqueeze(-1)

            if self.fusion == "concat":
                flat = torch.cat([hv[:, i] for i in range(3)] + [m], dim=-1)
                return self.head(flat), self.zproj(flat), {"m": m, "alpha": m / m.sum(1, keepdim=True).clamp(min=1e-6)}
            if self.fusion in ("late", "fixed"):
                logits_v = torch.stack(
                    [self.per_view_head[v](h[v]) for v in VIEWS], dim=1)      # [B,3,K]
                w = m / m.sum(1, keepdim=True).clamp(min=1e-6)
                if self.fusion == "late":
                    logits = (logits_v * w.unsqueeze(-1)).sum(1)
                else:
                    logits = self.head((hv * w.unsqueeze(-1)).sum(1))
                z = (hv * w.unsqueeze(-1)).sum(1)
                return logits, z, {"m": m, "alpha": w}

            a = []
            for i, v in enumerate(VIEWS):
                inp = torch.cat([h[v], m], dim=-1) if self.fusion == "gated" else h[v]
                a.append(self.gates[v](inp))
            a = torch.cat(a, dim=1)                                           # [B,3]
            if self.fusion == "gated":
                a = a.masked_fill(m < 0.5, float("-inf"))                     # alpha_v = 0
            alpha = torch.softmax(a, dim=1)
            alpha = torch.nan_to_num(alpha)
            z = (hv * alpha.unsqueeze(-1)).sum(1)
            return self.head(z), z, {"m": m, "alpha": alpha}


# --------------------------------------------------------------------------
# Training wrapper
# --------------------------------------------------------------------------

def _batch_from(b: Bundle, idx, device, keep_H=None, keep_C=None):
    t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)
    mH = b.mH[idx].copy()
    mC = b.mC[idx].copy()
    hfm = b.h_field_mask[idx].copy()
    Xh = b.Xh[idx].copy()
    Xc = b.Xc[idx].copy()
    if keep_H is not None:
        mH = mH * keep_H
        hfm = hfm * keep_H[:, None]
        Xh = Xh * keep_H[:, None]
    if keep_C is not None:
        mC = mC * keep_C
        Xc = Xc * keep_C[:, None]
    # expand the 4 field-mask bits over the H columns they gate
    blocks = b.h_blocks
    exp = np.ones_like(Xh)
    if blocks:
        for j, name in enumerate(("tls_basic", "tls_detail", "dhcp_basic", "dhcp_detail")):
            lo, hi = blocks[name]
            exp[:, lo:hi] = hfm[:, j: j + 1]
    return {
        "Xseq": t(b.Xseq[idx]), "seq_mask": t(b.seq_mask[idx]), "Xagg": t(b.Xagg[idx]),
        "Xh": t(Xh), "Xc": t(Xc), "mF": torch.ones(len(idx), device=device),
        "mH": t(mH), "mC": t(mC), "h_field_expand": t(exp),
        "y": torch.as_tensor(b.y[idx], dtype=torch.long, device=device),
    }


def supcon_loss(z, y, temperature: float):
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / temperature
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    pos = (y[:, None] == y[None, :]) & ~eye
    if pos.sum() == 0:
        return z.new_zeros(())
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -(logp * pos).sum(1) / pos.sum(1).clamp(min=1)
    return loss[pos.sum(1) > 0].mean()


@dataclass
class FitReport:
    epochs_run: int
    best_val_macro_f1: float
    train_seconds: float
    n_parameters: int


class NeuralClassifier:
    """Sklearn-shaped wrapper so experiments treat every method the same way."""

    def __init__(self, cfg: Config, fusion: str = "gated", seq_encoder: str = "tcn",
                 use_views=VIEWS, view_dropout: bool = True, seed: int = 0):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed; use backend='shallow'")
        self.cfg, self.fusion, self.seq_encoder = cfg, fusion, seq_encoder
        self.use_views, self.view_dropout, self.seed = tuple(use_views), view_dropout, seed
        self.model = None
        self.report: FitReport | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, tr: Bundle, va: Bundle | None = None):
        cfg = self.cfg
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = torch.device("cpu")
        self.classes_ = np.unique(tr.y)
        remap = {c: i for i, c in enumerate(self.classes_)}
        y_tr = np.array([remap[c] for c in tr.y])

        self.model = VisibilityAwareModel(
            cfg, tr.Xseq.shape[2], tr.Xagg.shape[1], tr.Xh.shape[1], tr.Xc.shape[1],
            len(self.classes_), self.fusion, self.seq_encoder, self.use_views).to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        n = len(tr.y)
        if cfg.balanced_sampling:
            cnt = np.bincount(y_tr, minlength=len(self.classes_)).astype(float)
            w = 1.0 / np.maximum(cnt[y_tr], 1.0)
            w = w / w.sum()
        else:
            w = None

        best, best_state, patience = -1.0, None, 0
        t0 = time.time()
        epochs_run = 0
        for ep in range(cfg.epochs):
            epochs_run = ep + 1
            self.model.train()
            order = (rng.choice(n, size=n, replace=True, p=w) if w is not None
                     else rng.permutation(n))
            for s in range(0, n, cfg.batch_size):
                idx = order[s: s + cfg.batch_size]
                if len(idx) < 4:
                    continue
                kH = kC = None
                if self.view_dropout:
                    kH, kC = sample_view_dropout(len(idx), cfg, rng)
                batch = _batch_from(tr, idx, dev, kH, kC)
                yb = torch.as_tensor(y_tr[idx], dtype=torch.long, device=dev)
                logits, z, _aux = self.model(batch)
                loss = F.cross_entropy(logits, yb)
                if cfg.supcon_lambda > 0:
                    loss = loss + cfg.supcon_lambda * supcon_loss(z, yb, cfg.supcon_temperature)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()

            if va is not None and len(va):
                from sklearn.metrics import f1_score
                pred = self.predict(va)["y_pred"]
                f1 = f1_score(va.y, pred, average="macro", zero_division=0)
                if f1 > best + 1e-4:
                    best, patience = f1, 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= cfg.early_stop_patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.report = FitReport(epochs_run, float(best), time.time() - t0,
                                sum(p.numel() for p in self.model.parameters()))
        return self

    @torch.no_grad()
    def predict(self, b: Bundle, batch_size: int = 512) -> dict:
        self.model.eval()
        dev = torch.device("cpu")
        logits, zs, alphas, masks = [], [], [], []
        for s in range(0, len(b.y), batch_size):
            idx = np.arange(s, min(s + batch_size, len(b.y)))
            batch = _batch_from(b, idx, dev)
            lg, z, aux = self.model(batch)
            logits.append(lg.numpy()); zs.append(z.numpy())
            alphas.append(aux["alpha"].numpy()); masks.append(aux["m"].numpy())
        logits = np.concatenate(logits); z = np.concatenate(zs)
        proba = np.exp(logits - logits.max(1, keepdims=True))
        proba = proba / proba.sum(1, keepdims=True)
        pred_local = logits.argmax(1)
        return {"logits": logits, "proba": proba, "z": z,
                "y_pred": self.classes_[pred_local],
                "alpha": np.concatenate(alphas),
                "view_mask": np.concatenate(masks)}
