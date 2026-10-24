"""Schematic diagram of the low-pass wavelet filter.

Output: results/week3/wavelet_lowpass_diagram.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUTPUT = Path(__file__).resolve().parent / "wavelet_lowpass_diagram.png"

BLUE = "#6d8fc2"
GREEN = "#2d6a4f"
RED = "#b3402e"
GRAY = "#e8e6e1"
INK = "#222222"


def _box(ax, xy, w, h, text, face, fontsize=10, textcolor=INK,
         boxstyle="round,pad=0.04"):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=boxstyle, facecolor=face, edgecolor=INK,
        linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, linespacing=1.5)


def _arrow(ax, x1, y1, x2, y2, label=""):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.6, color=INK, shrinkA=2, shrinkB=4))
    if label:
        ax.text((x1 + x2) / 2, y1 + 0.45, label, ha="center", va="bottom",
                fontsize=8.5, color="#555555", linespacing=1.4)


def main():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    # ---- input ----
    _box(ax, (0.2, 1.1), 1.7, 1.6, "RGB channel\nH \u00d7 W \u00b7 uint8",
         face="white", fontsize=10)

    _arrow(ax, 1.95, 1.9, 2.95, 1.9, "pywt.wavedec2\n(db2, level 3)")

    # ---- coefficient pyramid ----
    ax.add_patch(Rectangle((3.05, 0.75), 2.3, 2.3, facecolor="#f3d9d3",
                           edgecolor=RED, linewidth=1.2))
    ax.add_patch(Rectangle((3.4, 1.1), 1.6, 1.6, facecolor="#f7e8e4",
                           edgecolor=RED, linewidth=1.2))
    ax.add_patch(Rectangle((3.75, 1.45), 0.9, 0.9, facecolor="#dcefe4",
                           edgecolor=GREEN, linewidth=1.6))
    ax.text(4.2, 1.9, "cA3\nkeep", ha="center", va="center", fontsize=9.5,
            color=GREEN, weight="bold")
    ax.text(4.2, 0.92, "L1 details  \u2192  0", ha="center", va="center",
            fontsize=8.5, color=RED)
    ax.text(3.5, 2.55, "L2 det. \u2192 0", ha="left", va="center",
            fontsize=8.5, color=RED)
    ax.text(4.9, 2.55, "L3 det. \u2192 0", ha="right", va="center",
            fontsize=8.5, color=RED)
    ax.text(4.2, 3.25, "coefficients per channel", ha="center", va="center",
            fontsize=9, color="#555555")
    ax.text(3.2, 0.62, "H\u00d7W", ha="center", fontsize=8, color="#777777")
    ax.text(4.2, 1.32, "H/8\u00d7W/8", ha="center", fontsize=7.5,
            color="#777777")

    _arrow(ax, 5.4, 1.9, 6.4, 1.9, "zero details,\nkeep LL only")

    # ---- reconstruction ----
    _box(ax, (6.5, 1.1), 1.9, 1.6,
         "pywt.waverec2\ncrop to H\u00d7W \u2192 uint8", face="white",
         fontsize=10)

    _arrow(ax, 8.45, 1.9, 8.85, 1.9)

    _box(ax, (8.95, 1.1), 1.7, 1.6, "smoothed\nchannel", face=BLUE,
         fontsize=10, textcolor="white")

    ax.set_title("Low-pass wavelet filter (one RGB channel at a time)",
                 fontsize=12, pad=10)
    fig.text(0.5, 0.02,
             "Approximation cA3 keeps coarse structure and color; "
             "zeroed details carry edges, texture and impulse noise.  "
             "Best: db2 level 3.",
             ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUTPUT, dpi=150)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
