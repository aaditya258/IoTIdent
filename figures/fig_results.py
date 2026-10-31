#!/usr/bin/env python3
"""Result figures, drawn straight from the measurement records.

Every number here is read from results/derived_transport.json, which is itself
produced from the per-seed JSON written by the experiment. Nothing is typed in
by hand, so a figure cannot drift away from the table beside it.

    python figures/fig_results.py --root .

Produces, in figures/:
    fig_displacement.pdf     realised acceptance against the nominal setting
    fig_recoverability.pdf   rejection kept, at the stale and at the refitted
                             threshold
    fig_threshold_shift.pdf  the threshold stored against the one the shifted
                             condition needs
    fig_model_agreement.pdf  the same displacement under a second model family
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# Light, print-safe, and distinguishable under the common colour-vision
# deficiencies: a pale blue, a pale orange and a neutral grey.
BLUE, ORANGE, GREY = "#a6cee3", "#fdbf6f", "#d9d9d9"
EDGE = "#333333"

SHORT = {"Different\nnetwork": "Different\nnetwork",
         "Five months\nlater": "Five months\nlater",
         "Handshake\nreduced": "Handshake\nreduced",
         "Handshake\nabsent": "Handshake\nabsent",
         "Unseen unit of\na known product": "Unseen\nunit"}

AXES = [
    ("network|{m}|{s}|target_network", "Different\nnetwork"),
    ("time|{m}|{s}|later_months", "Five months\nlater"),
    ("visibility|{m}|{s}|visibility_V4", "Handshake\nreduced"),
    ("visibility|{m}|{s}|visibility_V5", "Handshake\nabsent"),
    ("instance|{m}|{s}|unseen_unit", "Unseen unit of\na known product"),
]
CONTROL = {
    "network": "network|{m}|{s}|control",
    "time": "time|{m}|{s}|control",
    "visibility": "visibility|{m}|{s}|control",
    "instance": "instance|{m}|{s}|control",
}


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "font.size": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def cell(d, key, model="forest", m="rf", s="msp"):
    return d[model]["cells"][key.format(m=m, s=s)]


def save(fig, root, name):
    out = os.path.join(root, "figures", name)
    fig.savefig(out + ".pdf", dpi=600, facecolor="white", bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(out + ".png", dpi=300, facecolor="white", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print("written:", out + ".pdf")


# ---------------------------------------------------------------- figure 2 --
def fig_displacement(d, root, alpha=0.95):
    """Where the acceptance actually lands, against where it was set."""
    labels, acc, err, ctrl = [], [], [], []
    for key, lab in AXES:
        c = cell(d, key)
        labels.append(lab)
        acc.append(c["known_acceptance"]["mean"])
        err.append(c["known_acceptance"]["ci95"])
        ctrl.append(cell(d, CONTROL[key.split("|")[0]])["known_acceptance"]["mean"])

    y = list(range(len(labels)))[::-1]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    ax.barh(y, acc, height=0.52, color=BLUE, edgecolor=EDGE, linewidth=0.6,
            xerr=err, error_kw=dict(ecolor=EDGE, elinewidth=0.7, capsize=2.0),
            label="realised on the shifted condition", zorder=3)
    ax.scatter(ctrl, y, marker="D", s=14, facecolor="white", edgecolor=EDGE,
               linewidth=0.7, zorder=5, label="realised on the source condition")
    ax.axvline(alpha, color=EDGE, linewidth=0.9, linestyle=(0, (4, 2)), zorder=4)
    ax.text(alpha - 0.015, len(labels) - 0.55, "nominal 0.95", fontsize=7,
            ha="right", va="center", color=EDGE)

    for yy, a in zip(y, acc):
        ax.text(1.13, yy, f"{a:.2f}", va="center", ha="left", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_xlim(0, 1.26)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("share of known devices accepted", fontsize=7.8)
    ax.set_ylim(-0.6, len(labels) - 0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.grid(True, color=GREY, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.9, loc="lower center", bbox_to_anchor=(0.5, -0.48),
              frameon=False, ncol=1, handlelength=1.4, handletextpad=0.5,
              labelspacing=0.3)
    fig.tight_layout(pad=0.2)
    save(fig, root, "fig_displacement")


# ---------------------------------------------------------------- figure 3 --
def fig_recoverability(d, root):
    """What recalibration returns, and what it does not."""
    labels, r_ctrl, r_stale, r_or, e_or = [], [], [], [], []
    for key, lab in AXES:
        c = cell(d, key)
        labels.append(lab)
        r_ctrl.append(cell(d, CONTROL[key.split("|")[0]])["unknown_rejection"]["mean"])
        r_stale.append(c["unknown_rejection"]["mean"])
        r_or.append(c["unknown_rejection_at_oracle"]["mean"])
        e_or.append(c["unknown_rejection_at_oracle"]["ci95"])

    x = list(range(len(labels)))
    w = 0.26
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    ax.bar([i - w for i in x], r_ctrl, w, color=GREY, edgecolor=EDGE,
           linewidth=0.6, label="source condition", zorder=3)
    ax.bar(x, r_stale, w, color=BLUE, edgecolor=EDGE, linewidth=0.6,
           label="shifted, stored threshold", zorder=3)
    ax.bar([i + w for i in x], r_or, w, color=ORANGE, edgecolor=EDGE,
           linewidth=0.6, yerr=e_or,
           error_kw=dict(ecolor=EDGE, elinewidth=0.7, capsize=1.8),
           label="shifted, threshold refitted", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[l] for l in labels], fontsize=6.7)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("share of unknown devices rejected", fontsize=7.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color=GREY, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.9, loc="upper center", bbox_to_anchor=(0.5, 1.22),
              frameon=False, ncol=3, handlelength=1.1, handletextpad=0.4,
              columnspacing=1.0)
    fig.tight_layout(pad=0.2)
    save(fig, root, "fig_recoverability")


# ---------------------------------------------------------------- figure 4 --
def fig_threshold_shift(d, root):
    """The stored threshold against the one the shifted condition needs."""
    labels, used, need = [], [], []
    for key, lab in AXES:
        c = cell(d, key)
        labels.append(lab.replace("\n", " "))
        used.append(c["tau_used"]["mean"])
        need.append(c["tau_oracle"]["mean"])

    y = list(range(len(labels)))[::-1]
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    for yy, u, n in zip(y, used, need):
        ax.plot([u, n], [yy, yy], color=EDGE, linewidth=0.8, zorder=2)
    ax.scatter(used, y, s=22, marker="o", facecolor=BLUE, edgecolor=EDGE,
               linewidth=0.6, zorder=4, label="threshold stored at calibration")
    ax.scatter(need, y, s=26, marker="^", facecolor=ORANGE, edgecolor=EDGE,
               linewidth=0.6, zorder=4, label="threshold the shifted condition needs")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.set_xlim(-0.03, 0.78)
    ax.set_xlabel("rejection threshold on the score", fontsize=7.8)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.grid(True, color=GREY, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.9, loc="lower center", bbox_to_anchor=(0.5, -0.52),
              frameon=False, handlelength=1.0, handletextpad=0.4,
              labelspacing=0.3)
    fig.tight_layout(pad=0.2)
    save(fig, root, "fig_threshold_shift")


# ---------------------------------------------------------------- figure 5 --
def fig_model_agreement(d, root):
    """The displacement under a second model family and a second score."""
    labels, LAB2, forest, gated, ef, eg = [], [], [], [], [], []
    for key, lab in AXES:
        labels.append(lab.replace("\n", " ")); LAB2.append(lab)
        cf = cell(d, key, "forest", "rf", "msp")
        cg = cell(d, key, "gated", "gated", "proto")
        forest.append(cf["delta_A"]["mean"]); ef.append(cf["delta_A"]["ci95"])
        gated.append(cg["delta_A"]["mean"]); eg.append(cg["delta_A"]["ci95"])

    x = list(range(len(labels)))
    w = 0.34
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    ax.bar([i - w / 2 for i in x], forest, w, color=BLUE, edgecolor=EDGE,
           linewidth=0.6, yerr=ef,
           error_kw=dict(ecolor=EDGE, elinewidth=0.7, capsize=1.8),
           label="forest, maximum posterior", zorder=3)
    ax.bar([i + w / 2 for i in x], gated, w, color=ORANGE, edgecolor=EDGE,
           linewidth=0.6, yerr=eg,
           error_kw=dict(ecolor=EDGE, elinewidth=0.7, capsize=1.8),
           label="gated network, prototype distance", zorder=3)
    ax.axhline(0, color=EDGE, linewidth=0.7, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[l] for l in LAB2], fontsize=6.7)
    ax.set_ylabel("displacement of known acceptance", fontsize=7.8)
    ax.set_ylim(-0.08, 0.68)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color=GREY, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.9, loc="upper center", bbox_to_anchor=(0.5, 1.20),
              frameon=False, ncol=1, handlelength=1.1, handletextpad=0.4)
    fig.tight_layout(pad=0.2)
    save(fig, root, "fig_model_agreement")


def main(argv=None) -> int:
    p = argparse.ArgumentParser("fig_results")
    p.add_argument("--root", default=".")
    a = p.parse_args(argv)
    d = json.load(open(os.path.join(a.root, "results", "derived_transport.json")))
    style()
    os.makedirs(os.path.join(a.root, "figures"), exist_ok=True)
    fig_displacement(d, a.root)
    fig_recoverability(d, a.root)
    fig_threshold_shift(d, a.root)
    fig_model_agreement(d, a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
