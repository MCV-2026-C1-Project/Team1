"""MSE of the hue-altered query and its hue-corrected version vs the BBDD GT.

Reproduces the exact correction of results/week3/run_qualitatives_hue.py
(sample 00014, scale estimated against its GT reference), resizes the GT
image to the query size and reports:

    MSE(altered query, GT) and MSE(corrected query, GT) in RGB.

Output: results/week3/hue_normalization/hue_correction_mse.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from color_reference import (  # noqa: E402
    DEFAULT_SATURATION_THRESHOLD,
    DEFAULT_SCALE_MAX,
    DEFAULT_SCALE_MIN,
    DEFAULT_SCALE_STEP,
    HueReference,
    correct_hue_scale,
    estimate_hue_scale,
    hue_histogram,
    hue_scale_grid,
)
from methods_w2 import _load_ground_truth, _ordered_image_paths, _read_rgb  # noqa: E402

MUSEUM = ROOT / "data" / "BBDD"
QUERIES = ROOT / "data" / "qsd1_w3"
GROUND_TRUTH = QUERIES / "gt_corresps.pkl"
REFERENCE = (ROOT / "results" / "week3" / "color_reference"
             / "museum_hue_references.npz")
OUTPUT = ROOT / "results" / "week3" / "hue_normalization"
SAMPLE = 14


def _mse(a, b):
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float((diff ** 2).mean())


def main():
    museum_ids, museum_paths = _ordered_image_paths(MUSEUM)
    query_ids, query_paths = _ordered_image_paths(QUERIES)
    ground_truth = _load_ground_truth(GROUND_TRUTH, query_ids)
    reference = HueReference.load(REFERENCE).reorder(museum_ids)

    query_rgb = _read_rgb(query_paths[SAMPLE])
    gt_idx = int(ground_truth[SAMPLE][0])
    gt_rgb = _read_rgb(museum_paths[gt_idx])

    scales = hue_scale_grid(DEFAULT_SCALE_MIN, DEFAULT_SCALE_MAX,
                            DEFAULT_SCALE_STEP)
    query_hist = hue_histogram(
        query_rgb, saturation_threshold=DEFAULT_SATURATION_THRESHOLD)
    scale, _score = estimate_hue_scale(
        query_hist, reference.histograms[gt_idx], scales=scales)
    corrected_rgb = correct_hue_scale(
        query_rgb, scale,
        saturation_threshold=DEFAULT_SATURATION_THRESHOLD)

    gt_resized = cv2.resize(
        gt_rgb, (query_rgb.shape[1], query_rgb.shape[0]),
        interpolation=cv2.INTER_AREA)
    mse_altered = _mse(query_rgb, gt_resized)
    mse_corrected = _mse(corrected_rgb, gt_resized)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "hue_correction_mse.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample", "reference_file", "gt_id", "hue_scale",
                        "mse_altered_vs_gt", "mse_corrected_vs_gt"])
        writer.writeheader()
        writer.writerow({
            "sample": Path(query_paths[SAMPLE]).name,
            "reference_file": Path(museum_paths[gt_idx]).name,
            "gt_id": museum_ids[gt_idx],
            "hue_scale": f"{scale:.4f}",
            "mse_altered_vs_gt": f"{mse_altered:.4f}",
            "mse_corrected_vs_gt": f"{mse_corrected:.4f}",
        })

    print(f"sample={Path(query_paths[SAMPLE]).name} "
          f"GT={Path(museum_paths[gt_idx]).name} hue_scale={scale:.4f}")
    print(f"MSE(altered, GT)   = {mse_altered:.2f}")
    print(f"MSE(corrected, GT) = {mse_corrected:.2f}")


if __name__ == "__main__":
    main()
