#!/usr/bin/env python3
"""Figure 1 — the operating point moves; the reported metric does not.

Reads results/cf_threshold_transport_*.json and draws, per shift axis:

  * a bar running from the REALISED known-acceptance A(c) up to the nominal
    alpha. The bar's LENGTH IS the displacement Delta_A, which is the
    quantity the paper introduces -- so the reader measures the finding
    directly off the page rather than decoding it from two bar heights.
  * the AUROC on the same condition, as a separate mark.

Both quantities are rates on [0, 1], so they share one axis. That is the whole
argument in one picture: the orange marks sit high and flat -- the number a
closed-set or threshold-free paper would report, and it looks healthy -- while
the blue ends have slid left, because the threshold that produced them no
longer does the job it was set to do.

    python figures/fig_threshold_displacement.py --root .

Writes figures/fig_threshold_displacement.{pdf,png}. PDF is the one to
\\includegraphics; the PNG is for looking at.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# Palette: categorical slots 1 and 2 of the reference instance, validated for
# this pair (CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the surface).
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK_2 = "#0b0b0b", "#52514e"

#: axis key -> (row label, sort hint). The keys are what cf_cli writes.
LABELS = {
    "network|target_network":        "Different network",
    "time|later_months":             "Five months later",
    "visibility|visibility_V5":      "No handshake (V5)",
    "visibility|visibility_V4":      "Partial handshake (V4)",
    "instance|unseen_unit":          "Unseen physical unit",
}


def load(root: str, pattern: str, scorer: str = "msp",
         method: str = "rf") -> tuple[list, float]:
    """Newest matching run, filtered to one method and one scorer.

    Both filters earn their place. The results directory accumulates runs from
    several configurations, and an older file can sort after a newer one by
    name -- so pick by mtime, not alphabetically. And a single file may hold
    more than one method, in which case taking every row silently draws each
    condition twice.
    """
    hits = glob.glob(os.path.join(root, "results", pattern))
    if not hits:
        raise SystemExit(f"no results matching {pattern} under {root}/results")
    newest = max(hits, key=os.path.getmtime)
    print(f"reading {os.path.basename(newest)}  (method={method}, scorer={scorer})")
    with open(newest, encoding="utf-8") as fh:
        d = json.load(fh)
    alpha = float(d["alpha"])
    rows = []
    for key, v in d["across_seeds"].items():
        ax, meth, sc, cond = key.split("|")
        if sc != scorer or meth != method:
            continue
        name = LABELS.get(f"{ax}|{cond}")
        if name is None:
            continue
        rows.append({
            "label": name,
            "dA": v["delta_A"]["mean"],
            "ci": v["delta_A"]["ci95"],
            "auroc": v["auroc"]["mean"],
        })
    rows.sort(key=lambda r: r["dA"])          # smallest at the bottom
    return rows, alpha


def draw(rows: list, alpha: float, out: str, title_note: str = "") -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed as TrueType, not Type 3
    })
    # IEEE single column.
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y = range(len(rows))
    realised = [alpha - r["dA"] for r in rows]

    # The displacement itself, as a bar from realised acceptance up to nominal.
    ax.hlines(list(y), realised, [alpha] * len(rows),
              color=BLUE, linewidth=5, alpha=0.30, zorder=2,
              capstyle="butt")
    ax.errorbar(realised, list(y), xerr=[r["ci"] for r in rows],
                fmt="o", color=BLUE, markersize=5, linewidth=1.2,
                capsize=2.5, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8,
                label="Realised acceptance")
    ax.plot([r["auroc"] for r in rows], list(y), "D", color=ORANGE,
            markersize=4.5, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=0.8,
            label="AUROC (what is reported)")

    ax.axvline(alpha, color=INK_2, linestyle=(0, (3, 2)), linewidth=0.9, zorder=3)
    ax.annotate(f"nominal $\\alpha$ = {alpha:.2f}", (alpha, len(rows) - 0.55),
                xytext=(-4, 0), textcoords="offset points",
                fontsize=7, color=INK_2, va="center", ha="right")

    # Values on the headline series only -- AUROC is in the legend and the
    # companion table; a number on every mark would be noise.
    for i, (r, a) in enumerate(zip(rows, realised)):
        ax.annotate(f"{a:.2f}", (a - r["ci"], i), xytext=(-5, 0),
                    textcoords="offset points", fontsize=7,
                    color=INK, va="center", ha="right")

    ax.set_yticks(list(y))
    ax.set_yticklabels([r["label"] for r in rows], color=INK)
    ax.set_xlabel("Rate", color=INK)
    ax.set_xlim(0.30, 1.045)
    ax.set_ylim(-0.55, len(rows) - 0.2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, length=3, width=0.8)

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("Realised acceptance"),
             labels.index("AUROC (what is reported)")]
    leg = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                    loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
                    frameon=False, handletextpad=0.4, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK)

    # Any run note belongs in the LaTeX caption, not on the canvas.
    if title_note:
        ax.set_title(title_note, fontsize=7.5, color=INK_2, loc="left", pad=6)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, facecolor=SURFACE,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"written: {out}.pdf and {out}.png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser("fig_threshold_displacement")
    p.add_argument("--root", default=".")
    p.add_argument("--pattern", default="cf_threshold_transport_rf_*.json")
    p.add_argument("--scorer", default="msp")
    p.add_argument("--method", default="rf")
    p.add_argument("--note", default="")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    rows, alpha = load(a.root, a.pattern, a.scorer, a.method)
    out = a.out or os.path.join(a.root, "figures", "fig_threshold_displacement")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    draw(rows, alpha, out, a.note)

    print(f"\n{'condition':26s}{'realised':>9s}{'dA':>8s}{'+-ci':>7s}{'AUROC':>8s}")
    for r in rows[::-1]:
        print(f"{r['label']:26s}{alpha - r['dA']:9.3f}{r['dA']:+8.3f}"
              f"{r['ci']:7.3f}{r['auroc']:8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
