"""Best kernel per filter, WITH hue-invariance (query-only scope + hue).

For every group the filter is applied at its best kernel to BOTH queries and
museum (Task 1 contract, ``filter_scope="both"``) and the ranking uses the HSV hue-invariance mode of methods_w3 (per-candidate hue
scale, blend of raw and hue-normalized LAB distances).

Outputs (recomputed only with --recompute):
    results/week3/filter_best_bars_hue.csv
    results/week3/filter_best_bars.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")

from k_similarity import mean_average_precision_at_k  # noqa: E402
from methods_w2 import _load_ground_truth, _ordered_image_paths  # noqa: E402
from methods_w3 import evaluate_folders_w3  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MUSEUM = ROOT / "data" / "BBDD"
QUERIES = ROOT / "data" / "qsd1_w3"
GROUND_TRUTH = QUERIES / "gt_corresps.pkl"
HUE_REFERENCE = (ROOT / "results" / "week3" / "color_reference"
                 / "museum_hue_references.npz")
OUTPUT_DIR = ROOT / "results" / "week3"
CSV = OUTPUT_DIR / "filter_best_bars_hue.csv"
FIGURE = OUTPUT_DIR / "filter_best_bars.png"

# (label, W3 method, kernel or None, parameter tag for the x axis)
CONFIGS = [
    ("No filter", "Lab blocks 1D", None, "\u2013"),
    ("Median", "Lab blocks 1D median", 7, "7x7"),
    ("Open", "Lab blocks 1D open", 9, "9x9"),
    ("Close", "Lab blocks 1D close", 7, "7x7"),
    ("Open+Close", "Lab blocks 1D open+close", 11, "11x11"),
    ("Close+Open", "Lab blocks 1D close+open", 7, "7x7"),
    ("Average", "Lab blocks 1D average", 5, "5x5"),
    ("Gaussian", "Lab blocks 1D gaussian", 7, "7x7"),
    ("Low-pass Wavelet", "Lab blocks 1D wavelet", None, "lvl3"),
]


def _evaluate(label, method, ksize):
    ranked_ids = evaluate_folders_w3(
        str(MUSEUM), str(QUERIES), method,
        ground_truth_path=str(GROUND_TRUTH),
        ksize=3 if ksize is None else ksize,
        wavelet_level=3,
        hue_invariant=True,
        hue_reference=str(HUE_REFERENCE),
    )
    _, query_paths = _ordered_image_paths(str(QUERIES))
    query_ids = [int(Path(p).stem) for p in query_paths]
    ground_truth = _load_ground_truth(str(GROUND_TRUTH), query_ids)
    map1 = mean_average_precision_at_k(ranked_ids, ground_truth, 1)
    map5 = mean_average_precision_at_k(ranked_ids, ground_truth, 5)
    print(f"{label:<16} mAP@1={map1:.4f} mAP@5={map5:.4f}", flush=True)
    return map1, map5


def _plot(rows):
    labels = [f"{row['filter']}\n({row['parameter']})" for row in rows]
    map1 = np.array([row["mAP@1"] for row in rows])
    map5 = np.array([row["mAP@5"] for row in rows])
    x = np.arange(len(rows))
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
    fig.savefig(FIGURE, dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURE}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true",
                        help="Re-run the hue-invariant evaluations "
                             "(otherwise only replot from the CSV)")
    options = parser.parse_args(argv)

    if options.recompute or not CSV.exists():
        rows = []
        for label, method, ksize, tag in CONFIGS:
            map1, map5 = _evaluate(label, method, ksize)
            rows.append({"filter": label, "parameter": tag,
                         "mAP@1": map1, "mAP@5": map5})
        with CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["filter", "parameter", "mAP@1", "mAP@5"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {CSV}")
    else:
        with CSV.open(encoding="utf-8") as handle:
            rows = [{**row, "mAP@1": float(row["mAP@1"]),
                     "mAP@5": float(row["mAP@5"])}
                    for row in csv.DictReader(handle)]
    _plot(rows)


if __name__ == "__main__":
    main()
