"""Best-result bars for the requested W3 filters (query-only scope).

For Median/Open/Close/Gaussian/Wavelet the best kernel is taken from
results/week3/filters_comparison.csv.  Average/Open+Close/Close+Open are not
in that file, so they are evaluated here with the same query-only protocol
(museum unchanged, Lab blocks 1D + L1) and the best kernel is selected by
(mAP@1, mAP@5).

Output: results/week3/filter_best_bars.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")

from methods_w3 import evaluate_filter_sweep  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COMPARISON = ROOT / "results" / "week3" / "filters_comparison.csv"
OUTPUT = ROOT / "results" / "week3" / "filter_best_bars.png"
KERNELS = (3, 5, 7, 9, 11, 13, 15)

ORDER = [
    ("none", "No filter"),
    ("median", "Median"),
    ("open", "Open"),
    ("close", "Close"),
    ("open+close", "Open+Close"),
    ("close+open", "Close+Open"),
    ("average", "Average"),
    ("gaussian", "Gaussian"),
    ("wavelet_lowpass", "Low-pass Wavelet"),
]


def _load_csv_rows():
    with COMPARISON.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _best(candidates):
    return max(candidates, key=lambda row: (row["mAP@1"], row["mAP@5"]))


def main():
    rows = _load_csv_rows()
    csv_rows = [
        {
            "filter": row["filter"],
            "parameter": row["parameter"],
            "mAP@1": float(row["mAP@1"]),
            "mAP@5": float(row["mAP@5"]),
        }
        for row in rows
    ]

    missing_specs = (
        [(name, k) for name in ("average", "open+close", "close+open")
         for k in KERNELS]
    )
    computed = evaluate_filter_sweep(
        "data/BBDD", "data/qsd1_w3", "data/qsd1_w3/gt_corresps.pkl",
        filter_specs=missing_specs)
    computed_rows = [
        {
            "filter": row["filter"],
            "parameter": f"{row['ksize']}x{row['ksize']}",
            "mAP@1": float(row["mAP@1"]),
            "mAP@5": float(row["mAP@5"]),
        }
        for row in computed
    ]
    available = csv_rows + computed_rows

    best = {}
    for key, _label in ORDER:
        if key == "none":
            candidates = [row for row in available
                          if row["filter"].lower() == "no filter"]
        else:
            candidates = [row for row in available if row["filter"] == key]
        if key == "wavelet_lowpass":
            candidates = [row for row in candidates
                          if row["parameter"] == "db2_lvl3"]
        best[key] = _best(candidates)

    labels = []
    map1 = []
    map5 = []
    for key, label in ORDER:
        row = best[key]
        if key == "none":
            parameter = "\u2013"
        else:
            parameter = ("lvl3" if key == "wavelet_lowpass"
                         else row["parameter"])
        labels.append(f"{label}\n({parameter})")
        map1.append(row["mAP@1"])
        map5.append(row["mAP@5"])
        print(f"{label:<16} {row['parameter']:<10} "
              f"mAP@1={row['mAP@1']:.4f} mAP@5={row['mAP@5']:.4f}")

    x = np.arange(len(ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    bars1 = ax.bar(x - width / 2, map1, width, label="mAP@1",
                   color="#6d8fc2", edgecolor="white")
    bars2 = ax.bar(x + width / 2, map5, width, label="mAP@5",
                   color="#2d6a4f", edgecolor="white")
    for bars, values in ((bars1, map1), (bars2, map5)):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008,
                    f"{value:.4f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("mAP")
    ax.set_title("Results on QSD1_W3. Showing best kernel size per filter. "
                 "Applying hue-invariance.")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.12), ncol=2,
              frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
