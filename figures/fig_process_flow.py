#!/usr/bin/env python3
"""Figure 1 - the measurement procedure, in plain words.

Each step names the action taken and, beneath a short rule, the objective that
action serves. Two phases side by side, so the whole protocol fits across the
page in a two-column layout. Deliberately plain: black outlines on white, no
fill, no shading, no symbols and no notation.

    python figures/fig_process_flow.py --out figures/fig_process_flow
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

# (what was done, what it achieves), four steps per phase.
PHASES = [
    ("Phase 1  Prepare the observations", [
        ("Read ten months of traffic captured from two separate home networks",
         "two deployments that differ in site and in season, so a shift can be "
         "observed rather than assumed"),
        ("Group each device's traffic into fixed half-hour observation windows",
         "fixes the unit a monitor would actually score, so every later number "
         "refers to one decision"),
        ("Describe each window three ways: packet timing and size, handshake "
         "metadata, and communication structure",
         "keeps a description available when payload is encrypted and when the "
         "handshake is missing altogether"),
        ("Remove features that name a device outright, and keep a device's "
         "neighbouring activity on one side of every split",
         "stops an identifier or a repeated session standing in for the "
         "behaviour the model is meant to learn"),
    ]),
    ("Phase 2  Measure the operating point", [
        ("Train on one condition, holding out several device types so they "
         "stand in for products never installed before",
         "builds the closed part of the task and the open part from one "
         "capture, without borrowing unknowns from elsewhere"),
        ("Set the rejection threshold on held-out known traffic so the "
         "detector accepts a chosen share of the devices it knows",
         "fixes the single operating point an installed monitor would store"),
        ("Carry that model and that stored threshold unchanged to a shifted "
         "condition, and record the share it now accepts",
         "measures how far the promised acceptance has moved, which is the "
         "quantity an operator was given"),
        ("Refit the threshold on the shifted condition and record the "
         "rejection that survives",
         "separates loss a recalibration would return from loss no threshold "
         "on that condition can return"),
    ]),
]


def draw(out: str) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    wrap_a, wrap_w = 54, 58
    cols = []
    for title, steps in PHASES:
        act = [textwrap.fill(a, wrap_a) for a, _ in steps]
        why = [textwrap.fill(w, wrap_w) for _, w in steps]
        cols.append((title, act, why,
                     [t.count("\n") + 1 for t in act],
                     [t.count("\n") + 1 for t in why]))

    unit_a, unit_w = 0.132, 0.114
    gap, pad, rule_gap = 0.115, 0.044, 0.032
    heights = [[a * unit_a + w * unit_w + 2 * pad + rule_gap
                for a, w in zip(c[3], c[4])] for c in cols]
    col_h = [sum(h) + gap * (len(h) - 1) for h in heights]
    title_h = 0.26
    total = max(col_h) + title_h

    fig_w, fig_h = 7.16, total + 0.12
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total)
    ax.axis("off")

    col_x = [0.015, 0.515]
    col_w = 0.470
    box_bottoms = []

    for ci, (title, act, why, na, nw) in enumerate(cols):
        x0 = col_x[ci]
        ax.text(x0 + col_w / 2, total - 0.085, title, ha="center", va="center",
                fontsize=8.2, color="black")
        y = total - title_h
        for i, h in enumerate(heights[ci]):
            ax.add_patch(Rectangle((x0, y - h), col_w, h, facecolor="white",
                                   edgecolor="black", linewidth=0.9, zorder=2))
            ax.text(x0 + col_w / 2, y - pad, act[i], ha="center", va="top",
                    fontsize=7.3, color="black", linespacing=1.16, zorder=3)
            y_rule = y - pad - na[i] * unit_a - rule_gap / 2
            ax.plot([x0 + 0.030, x0 + col_w - 0.030], [y_rule, y_rule],
                    color="black", linewidth=0.45, zorder=3)
            ax.text(x0 + col_w / 2, y_rule - rule_gap / 2, why[i],
                    ha="center", va="top", fontsize=6.4, color="black",
                    linespacing=1.15, zorder=3)
            if i < len(heights[ci]) - 1:
                ax.add_patch(FancyArrow(
                    x0 + col_w / 2, y - h - 0.012, 0, -(gap - 0.032),
                    width=0.0012, head_width=0.013, head_length=0.030,
                    length_includes_head=True, facecolor="black",
                    edgecolor="black", linewidth=0.5, zorder=2))
            y -= h + gap
        box_bottoms.append(y + gap)

    # Link the end of phase one to the start of phase two.
    y_from = box_bottoms[0] + 0.010
    x_from = col_x[0] + col_w
    y_to = total - title_h - heights[1][0] / 2
    x_mid = (col_x[0] + col_w + col_x[1]) / 2
    ax.plot([x_from, x_mid], [y_from, y_from], color="black", linewidth=0.7,
            zorder=1)
    ax.plot([x_mid, x_mid], [y_from, y_to], color="black", linewidth=0.7,
            zorder=1)
    ax.add_patch(FancyArrow(x_mid, y_to, col_x[1] - x_mid - 0.004, 0,
                            width=0.0012, head_width=0.013, head_length=0.008,
                            length_includes_head=True, facecolor="black",
                            edgecolor="black", linewidth=0.5, zorder=2))

    fig.tight_layout(pad=0.10)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=600, facecolor="white",
                    bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"written: {out}.pdf and {out}.png  ({fig_w:.2f} x {fig_h:.2f} in)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser("fig_process_flow")
    p.add_argument("--out", default="figures/fig_process_flow")
    a = p.parse_args(argv)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    draw(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
