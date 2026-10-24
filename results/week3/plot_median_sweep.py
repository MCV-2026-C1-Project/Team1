"""Grouped bars: median filter sweep 3x3 to 15x15 (query-only, L1).

Reads results/week3/filters_comparison.csv and regenerates
results/week3/median_kernel_bars.png.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMPARISON = ROOT / "results" / "week3" / "filters_comparison.csv"
OUTPUT = ROOT / "results" / "week3" / "median_kernel_bars.png"
KERNELS = (3, 5, 7, 9, 11, 13, 15)


def main():
    with COMPARISON.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["filter"] == "median"
                and row["parameter"] in {f"{k}x{k}" for k in KERNELS}]
    rows.sort(key=lambda row: int(row["parameter"].split("x")[0]))
    assert [int(r["parameter"].split("x")[0]) for r in rows] == list(KERNELS), (
        "median sweep rows missing in filters_comparison.csv")
    map1 = np.array([float(r["mAP@1"]) for r in rows])
    map5 = np.array([float(r["mAP@5"]) for r in rows])

    x = np.arange(len(KERNELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.2))
    bars1 = ax.bar(x - width / 2, map1, width, label="mAP@1",
                   color="#0f766e", edgecolor="white")
    bars2 = ax.bar(x + width / 2, map5, width, label="mAP@5",
                   color="#d9a441", edgecolor="white")
    for bars, values, color in ((bars1, map1, "#0f766e"),
                                (bars2, map5, "#d9a441")):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=10,
                    color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in KERNELS])
    ax.set_xlabel("median kernel size k")
    ax.set_ylabel("mAP")
    ax.set_ylim(0.60, 0.85)
    ax.set_title("Median kernel on QSD1-W3 (L1)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.text(0.5, 0.01,
             "median filter applied to queries only; "
             "museum/reference images are never filtered.",
             ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT, dpi=150)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
