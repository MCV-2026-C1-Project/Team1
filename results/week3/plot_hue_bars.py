"""Paired bars: mAP@1 and mAP@5 on hue-altered QSD1_W3 queries.

Reads results/week3/hue_normalization/summary.csv and writes
results/week3/hue_normalization/map_hue_bars.png.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "hue_normalization"
SUMMARY = OUTPUT_DIR / "summary.csv"
FIGURE = OUTPUT_DIR / "map_hue_bars.png"


def main():
    with SUMMARY.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["subset"] == "hue"]
    by_method = {row["method"]: row for row in rows}
    baseline = by_method["baseline"]
    method = by_method["hue_reference"]
    map1_col = "mAP@1"
    map5_col = next(key for key in baseline if key.startswith("mAP@")
                    and key != map1_col)
    values = np.array([
        [float(baseline[map1_col]), float(baseline[map5_col])],
        [float(method[map1_col]), float(method[map5_col])],
    ])

    labels = ["Default", "Hue-invariance"]
    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = ["#6d8fc2", "#2d6a4f"]
    for pos, (name, color) in enumerate(zip((map1_col, map5_col), colors)):
        bars = ax.bar(x + (pos - 0.5) * width, values[:, pos], width,
                      label=name, color=color, edgecolor="white")
        for bar, value in zip(bars, values[:, pos]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                    f"{value:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mAP")
    ax.set_title("QSD1_W3: hue-altered queries (None-MultiplyHue)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150)
    print(f"Saved {FIGURE}")


if __name__ == "__main__":
    main()
