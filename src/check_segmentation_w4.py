"""Compare segmentation thresholds on the Week 4 development set."""

import argparse
import pickle

import cv2
import numpy as np

from methods_w2 import _ordered_image_paths, _read_rgb
from paintings_w3 import PaintingParams, segment_paintings
from utils import compute_segmentation_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    args = parser.parse_args()

    image_ids, paths = _ordered_image_paths(args.queries)

    # The correspondence list tells us how many paintings each query has.
    # An entry [-1] means ONE unknown painting, not zero paintings.
    ground_truth_path = paths[0].parent / "gt_corresps.pkl"

    with ground_truth_path.open("rb") as stream:
        correspondences = pickle.load(stream)

    if len(correspondences) != len(paths):
        raise ValueError("Ground-truth count does not match query count")

    settings = {
        "default": None,
        "threshold_3": PaintingParams(thr_min=3.0, thr_max=3.0),
        "threshold_4": PaintingParams(thr_min=4.0, thr_max=4.0),
        "threshold_5": PaintingParams(thr_min=5.0, thr_max=5.0),
        "threshold_6": PaintingParams(thr_min=6.0, thr_max=6.0),
    }

    results = {name: [] for name in settings}

    # Read each image once, then try every setting.
    for position, (image_id, path) in enumerate(
        zip(image_ids, paths)
    ):
        image = _read_rgb(path)

        truth_mask = cv2.imread(
            str(path.with_suffix(".png")),
            cv2.IMREAD_GRAYSCALE,
        )

        if truth_mask is None:
            raise ValueError(f"Cannot read ground-truth mask for {path}")

        expected_count = len(correspondences[position])

        for name, params in settings.items():
            predicted_mask, boxes = segment_paintings(
                image,
                params=params,
            )

            scores = compute_segmentation_metrics(
                predicted_mask,
                truth_mask,
            )

            results[name].append({
                "image_id": image_id,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["F1"],
                "count_correct": len(boxes) == expected_count,
                "predicted_count": len(boxes),
                "expected_count": expected_count,
            })

        print(
            f"Evaluated image {image_id}: "
            f"{position + 1}/{len(paths)}",
            flush=True,
        )

    print("\nSUMMARY")
    print(
        f"{'Setting':<16}"
        f"{'Precision':>11}"
        f"{'Recall':>10}"
        f"{'Mask F1':>10}"
        f"{'Correct count':>16}"
    )

    for name, rows in results.items():
        precision = np.mean([row["precision"] for row in rows])
        recall = np.mean([row["recall"] for row in rows])
        f1 = np.mean([row["f1"] for row in rows])
        correct = sum(row["count_correct"] for row in rows)

        print(
            f"{name:<16}"
            f"{precision:>11.4f}"
            f"{recall:>10.4f}"
            f"{f1:>10.4f}"
            f"{correct:>12}/{len(rows)}"
        )

    # Show where threshold 4 helps or hurts compared with the defaults.
    print("\nTHRESHOLD 4 COMPARED WITH DEFAULT")
    print("Positive F1 change means threshold 4 improved the mask.")

    for default, candidate in zip(
        results["default"],
        results["threshold_4"],
    ):
        difference = candidate["f1"] - default["f1"]

        print(
            f"Image {default['image_id']:05d}: "
            f"F1 {default['f1']:.3f} -> {candidate['f1']:.3f} "
            f"({difference:+.3f}); "
            f"count {default['predicted_count']} -> "
            f"{candidate['predicted_count']}, "
            f"expected {candidate['expected_count']}"
        )


if __name__ == "__main__":
    main()