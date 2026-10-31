"""Automatic Week 4 matching, development calibration, and prediction."""

import argparse
import json
import pickle
from numbers import Integral
from pathlib import Path

import cv2

from matching_w4 import load_features, match_sift
from paintings_w3 import evaluate_painting_rankings


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def ordered_images(index):
    images = sorted(index["images"], key=lambda image: image["image_id"])
    ids = [image["image_id"] for image in images]

    if not images or len(ids) != len(set(ids)):
        raise ValueError("Expected nonempty, unique image IDs")

    return images


def run_matching(args):
    museum_index = read_json(args.museum_features / "index.json")
    query_index = read_json(args.query_features / "index.json")

    if museum_index["role"] != "museum" or query_index["role"] != "queries":
        raise ValueError("Incorrect museum/query feature folders")

    settings = {
        "method": "SIFT",
        "distance": getattr(args, "distance", "L2"),
        "ratio": args.ratio,
        "ransac_threshold": args.ransac_threshold,
        "museum_nfeatures": museum_index["nfeatures"],
        "query_nfeatures": query_index["nfeatures"],
        "museum_preprocessing": museum_index["preprocessing"],
        "query_preprocessing": query_index["preprocessing"],
        "matching_opencv_version": cv2.__version__,
        "museum_feature_opencv_version": museum_index["opencv_version"],
        "query_feature_opencv_version": query_index["opencv_version"],
        "query_segmentation": query_index.get("segmentation"),
    }

    if museum_index["method"] != "SIFT" or query_index["method"] != "SIFT":
        raise ValueError("This matcher requires SIFT features")

    if not 0 < args.ratio < 1 or args.ransac_threshold <= 0:
        raise ValueError("Invalid matching thresholds")

    # Load museum features once, rather than for every query crop.
    museum = []

    for record in ordered_images(museum_index):
        if len(record["paintings"]) != 1:
            raise ValueError("Expected one painting per museum image")

        painting = record["paintings"][0]
        features = load_features(
            args.museum_features / painting["feature_file"]
        )
        museum.append((int(record["image_id"]), features))

    if not 1 <= args.top_k <= len(museum):
        raise ValueError("top_k must be between 1 and the museum size")

    report = {
        "settings": settings,
        "top_k": args.top_k,
        "query_features": str(args.query_features.resolve()),
        "museum_features": str(args.museum_features.resolve()),
        "images": [],
    }

    images = ordered_images(query_index)

    for position, image in enumerate(images, start=1):
        row = {
            "image_id": int(image["image_id"]),
            "segmentation_fallback": image["segmentation_fallback"],
            "paintings": [],
        }

        paintings = sorted(
            image["paintings"],
            key=lambda painting: painting["painting_index"],
        )

        # Painting count comes from the extraction index.
        # No manual painting count or ground truth is used.
        for painting in paintings:
            query = load_features(
                args.query_features / painting["feature_file"]
            )
            candidates = []

            for museum_id, features in museum:
                result = match_sift(
                    query,
                    features,
                    ratio=args.ratio,
                    ransac_threshold=args.ransac_threshold,
                    distance=settings["distance"],
                )

                candidates.append({
                    "museum_id": museum_id,
                    "verified": result["inlier_count"],
                    "tentative": result["tentative_count"],
                })

            candidates.sort(key=lambda item: (
                -item["verified"],
                -item["tentative"],
                item["museum_id"],
            ))

            row["paintings"].append({
                "painting_index": painting["painting_index"],
                "box": painting["box"],
                "candidates": candidates[:args.top_k],
            })

        report["images"].append(row)

        scores = [
            painting["candidates"][0]["verified"]
            for painting in row["paintings"]
        ]

        print(
            f"[{position}/{len(images)}] Query {row['image_id']}: "
            f"{len(paintings)} painting(s); best inliers={scores}",
            flush=True,
        )

    write_json(args.output, report)
    print(f"Saved matching scores: {args.output}")


def make_predictions(report, min_inliers):
    """Apply rejection without using ground truth."""

    if min_inliers < 1:
        raise ValueError("min_inliers must be positive")

    predictions = []

    for image in report["images"]:
        painting_results = []

        for painting in image["paintings"]:
            candidates = painting["candidates"]

            if (
                not candidates
                or candidates[0]["verified"] < min_inliers
            ):
                painting_results.append([-1])
            else:
                painting_results.append([
                    int(item["museum_id"])
                    for item in candidates
                ])

        predictions.append(painting_results)

    return predictions


def unknown_metrics(predictions, truth):
    """Unknown is the positive class; compare slots in reading order."""

    truth = normalize_truth(truth, len(predictions))
    tp = fp = fn = 0

    for predicted, actual in zip(predictions, truth):
        for slot in range(max(len(predicted), len(actual))):
            actual_unknown = (
                slot < len(actual)
                and actual[slot] == -1
            )

            rejected = (
                slot < len(predicted)
                and predicted[slot] == [-1]
            )

            tp += int(actual_unknown and rejected)
            fp += int(rejected and not actual_unknown)
            fn += int(actual_unknown and not rejected)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def normalize_truth(truth, count):
    """Accept integer IDs or singleton ID lists, including unknown (-1)."""
    if len(truth) != count:
        raise ValueError("Ground truth must match numeric query order")
    normalized = []
    for row in truth:
        if not isinstance(row, (list, tuple)) or not 1 <= len(row) <= 2:
            raise ValueError("Expected one or two ground-truth paintings")
        values = []
        for item in row:
            if isinstance(item, (list, tuple)) and len(item) == 1:
                item = item[0]
            if isinstance(item, bool) or not isinstance(item, Integral) or item < -1:
                raise ValueError("Expected museum IDs or -1 in ground truth")
            values.append(int(item))
        normalized.append(values)
    return normalized


def calibrate_threshold(report, truth, thresholds=None):
    """Choose the rejection threshold on development unknown-positive F1."""
    truth = normalize_truth(truth, len(report["images"]))

    if not any(value == -1 for row in truth for value in row):
        raise ValueError("Unknown calibration requires unknown examples")

    if not any(value != -1 for row in truth for value in row):
        raise ValueError("Unknown calibration requires known examples")

    if thresholds is None:
        # Predictions change only when a threshold passes a best-match score.
        thresholds = sorted({1} | {
            painting["candidates"][0]["verified"] + 1
            for image in report["images"] for painting in image["paintings"]
            if painting["candidates"]
        })
    if not thresholds or any(isinstance(t, bool) or not isinstance(t, Integral)
                             or t < 1 for t in thresholds):
        raise ValueError("Expected positive integer thresholds")
    rows = []
    for threshold in sorted(set(thresholds)):
        predictions = make_predictions(report, threshold)

        metrics = unknown_metrics(predictions, truth)
        row = {
            "min_inliers": threshold,
            **metrics,
        }
        for k in (1, 5, 10):
            row.update(evaluate_painting_rankings(predictions, truth, k))
        rows.append(row)

    # Optimize unknown F1, as required by Task 2.
    # Break equal-F1 ties using the smaller threshold.
    best = max(
        rows,
        key=lambda row: (row["f1"], -row["min_inliers"]),
    )

    return {
        "min_inliers": best["min_inliers"],
        "settings": report["settings"],
        "selection_metric": "unknown-positive F1, reading-order slots",
        "results": rows,
    }


def tune_threshold(args):
    report = read_json(args.scores)
    with args.ground_truth.open("rb") as stream:
        truth = pickle.load(stream)
    calibration = calibrate_threshold(report, truth, args.thresholds)
    calibration["development_scores"] = str(args.scores.resolve())
    print("Threshold  Unknown-P  Unknown-R  Unknown-F1   mAP@1   mAP@5")
    for row in calibration["results"]:
        print(f"{row['min_inliers']:9d} {row['precision']:10.4f} "
              f"{row['recall']:10.4f} {row['f1']:11.4f} "
              f"{row['mAP@1']:8.4f} {row['mAP@5']:8.4f}")
    write_json(args.output, calibration)

    print(f"Selected min_inliers={calibration['min_inliers']}")
    print("These are development tuning scores, not held-out test scores.")
    print(f"Saved calibration: {args.output}")


def predict(args):
    report = read_json(args.scores)
    calibration = read_json(args.calibration)

    if report["settings"] != calibration["settings"]:
        raise ValueError(
            "Matching settings differ from calibration settings"
        )

    predictions = make_predictions(
        report,
        calibration["min_inliers"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("wb") as stream:
        pickle.dump(predictions, stream, protocol=4)

    # Keep query IDs alongside the positional result list.
    write_json(args.output.with_suffix(".json"), {
        "query_ids": [
            row["image_id"] for row in report["images"]
        ],
        "min_inliers": calibration["min_inliers"],
        "settings": report["settings"],
        "predictions": predictions,
    })

    print(
        f"Saved {len(predictions)} query results: {args.output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    match = commands.add_parser("match")
    match.add_argument(
        "--museum-features", type=Path, required=True
    )
    match.add_argument(
        "--query-features", type=Path, required=True
    )
    match.add_argument("--output", type=Path, required=True)
    match.add_argument("--ratio", type=float, default=0.75)
    match.add_argument("--distance", choices=("L1", "L2"), default="L2")
    match.add_argument(
        "--ransac-threshold", type=float, default=5.0
    )
    match.add_argument("--top-k", type=int, default=10)
    match.set_defaults(run=run_matching)

    tune = commands.add_parser("tune")
    tune.add_argument("--scores", type=Path, required=True)
    tune.add_argument(
        "--ground-truth", type=Path, required=True
    )
    tune.add_argument("--output", type=Path, required=True)
    tune.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=None,
        help="Optional thresholds; by default test every distinct decision",
    )
    tune.set_defaults(run=tune_threshold)

    prediction = commands.add_parser("predict")
    prediction.add_argument(
        "--scores", type=Path, required=True
    )
    prediction.add_argument(
        "--calibration", type=Path, required=True
    )
    prediction.add_argument(
        "--output", type=Path, required=True
    )
    prediction.set_defaults(run=predict)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
