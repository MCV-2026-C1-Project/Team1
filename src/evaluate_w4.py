"""Task 3: evaluate SIFT and the unchanged Week 3 texture system on QSD1-W4."""

import argparse
import csv
import pickle
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from batch_matching_w4 import (
    calibrate_threshold, make_predictions, normalize_truth,
    read_json, run_matching, unknown_metrics, write_json,
)
from k_similarity import average_precision_at_k
from methods_w2 import _ordered_image_paths, _read_rgb
from paintings_w3 import (
    build_query_filter, evaluate_painting_rankings, retrieve_folder_w3,
)


def evaluate_predictions(predictions, truth):
    """Score each painting in reading order, including missing/extra crops."""
    truth = normalize_truth(truth, len(predictions))
    metrics = {}
    for k in (1, 5, 10):
        metrics.update(evaluate_painting_rankings(predictions, truth, k))
        known = [
            average_precision_at_k(predicted[i] if i < len(predicted) else [],
                                   [target], k)
            for predicted, actual in zip(predictions, truth)
            for i, target in enumerate(actual) if target != -1
        ]
        metrics[f"known_mAP@{k}"] = float(np.mean(known)) if known else None
    metrics["unknown"] = unknown_metrics(predictions, truth)
    return metrics


def validate_scores(report, query_ids, museum_ids):
    """Check alignment before comparing positional predictions with the GT."""
    if [row["image_id"] for row in report["images"]] != query_ids:
        raise ValueError("Matching scores must follow numeric query filename order")
    if report["top_k"] < 10:
        raise ValueError("At least 10 ranked candidates are needed for mAP@10")
    valid_ids = set(museum_ids)
    for image in report["images"]:
        paintings = image["paintings"]
        if [p["painting_index"] for p in paintings] != list(range(len(paintings))):
            raise ValueError("Painting scores must follow crop reading order")
        if not 1 <= len(paintings) <= 2:
            raise ValueError("Expected one or two painting crops per query")
        for painting in paintings:
            candidates = painting["candidates"]
            ids = [item["museum_id"] for item in candidates]
            if len(ids) != report["top_k"] or len(set(ids)) != len(ids):
                raise ValueError("Expected top-k unique museum candidates")
            if not set(ids) <= valid_ids:
                raise ValueError("Candidate IDs are absent from the museum")
            expected = sorted(candidates, key=lambda c: (
                -c["verified"], -c["tentative"], c["museum_id"]))
            if candidates != expected:
                raise ValueError("Candidates must be ordered by verified matches")


def save_details(output, query_ids, truth, methods, scores):
    fields = ["query_id", "painting", "ground_truth", "method", "top1",
              "rank_of_truth", "ap1", "ap5", "ap10", "count_correct",
              "best_inliers", "segmentation_fallback"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (qid, actual, score) in enumerate(zip(query_ids, truth,
                                                        scores["images"])):
            for name, predictions in methods.items():
                predicted = predictions[index]
                for slot in range(max(len(actual), len(predicted))):
                    target = actual[slot] if slot < len(actual) else None
                    ranking = predicted[slot] if slot < len(predicted) else []
                    row = {
                        "query_id": f"{qid:05d}", "painting": slot,
                        "ground_truth": target, "method": name,
                        "top1": ranking[0] if ranking else None,
                        "rank_of_truth": ranking.index(target) + 1
                        if target in ranking else None,
                        "count_correct": len(predicted) == len(actual),
                    }
                    for k in (1, 5, 10):
                        row[f"ap{k}"] = average_precision_at_k(ranking, [target], k)
                    if name.startswith("SIFT") and slot < len(score["paintings"]):
                        row["best_inliers"] = score["paintings"][slot]["candidates"][0]["verified"]
                        row["segmentation_fallback"] = score["segmentation_fallback"]
                    writer.writerow(row)


def comparison_markdown(metrics, threshold, nqueries, nknown, nunknown):
    lines = [
        "# Week 4 - Task 3", "",
        f"QSD1-W4: {nqueries} queries, {nknown} known paintings and "
        f"{nunknown} unknown paintings.", "",
        "| Method | mAP@1 | mAP@5 | mAP@10 | Count accuracy | Unknown F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        lines.append(f"| {name} | {row['mAP@1']:.4f} | {row['mAP@5']:.4f} | "
                     f"{row['mAP@10']:.4f} | {row['count_accuracy']:.4f} | "
                     f"{row['unknown']['f1']:.4f} |")
    lines += [
        "", f"SIFT rejects a crop when its best candidate has fewer than {threshold} "
        "RANSAC inliers. The threshold maximizes unknown-positive F1 on this "
        "development set; equal-F1 ties use the smaller threshold.", "",
        "These are development results, including threshold tuning on the same "
        "set. They do not estimate held-out test performance.", "",
        "AP is computed per painting in reading order. Missing and extra crops "
        "score zero. A correctly rejected unknown has target -1 and scores one.", "",
        "The W3 baseline keeps its original configuration: Texture fusion, L1, "
        "3x3 median on query crops, and the original adaptive W3 segmentation. "
        "It always retrieves museum IDs, so it cannot recognize unknown paintings. "
        "Its unknown F1 is therefore zero.", "",
        "SIFT uses the fixed segmentation threshold 4 selected on QSD1-W4. "
        "This compares complete pipelines: both segmentation and retrieval differ.", "",
        "## Retrieval on annotated known paintings", "",
        "This view excludes unknown annotations and counts missing known crops "
        "as zero. The main table also penalizes extra crops.", "",
        "| Method | Known mAP@1 | Known mAP@5 | Known mAP@10 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        values = [f"{row[f'known_mAP@{k}']:.4f}" if row[f"known_mAP@{k}"] is not None
                  else "n/a" for k in (1, 5, 10)]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def save_error_examples(output, query_paths, museum_paths, truth, predictions, scores):
    errors = [i for i, (predicted, actual) in enumerate(zip(predictions, truth))
              if [ranking[0] for ranking in predicted] != actual]
    if not errors:
        return
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    errors = errors[:4]
    fig = Figure(figsize=(12, 4 * len(errors)))
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(errors), 3, squeeze=False)
    for row, i in enumerate(errors):
        image = scores["images"][i]
        ax = axes[row, 0]
        ax.imshow(_read_rgb(query_paths[i]))
        top1 = [ranking[0] for ranking in predictions[i]]
        ax.set_title(f"Query {image['image_id']:05d}: predicted {top1}")
        for painting in image["paintings"]:
            x0, y0, x1, y1 = painting["box"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor="yellow", linewidth=2))
        for slot, target in enumerate(truth[i]):
            ax = axes[row, slot + 1]
            if target != -1:
                ax.imshow(_read_rgb(museum_paths[target]))
                ax.set_title(f"Expected painting {slot}: museum {target:05d}")
            else:
                ax.text(.5, .5, "Expected: unknown", ha="center", va="center")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=140)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--museum", type=Path, default=Path("data/BBDD"))
    parser.add_argument("--queries", type=Path, default=Path("data/qsd1_w4"))
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--scores", type=Path,
                        default=Path("results/week4/batch/matching_scores.json"))
    parser.add_argument("--museum-features", type=Path,
                        default=Path("results/week4/sift/museum"))
    parser.add_argument("--query-features", type=Path,
                        default=Path("results/week4/sift/qsd1_w4_threshold4"))
    parser.add_argument("--rematch", action="store_true",
                        help="Recompute matching from the saved SIFT features")
    parser.add_argument("--distance", choices=("L1", "L2"), default="L2",
                        help="SIFT matching distance when using --rematch")
    parser.add_argument("--output", type=Path, default=Path("results/week4/evaluation"))
    args = parser.parse_args()
    cv2.setNumThreads(1)

    query_ids, query_paths = _ordered_image_paths(args.queries)
    museum_ids, museum_paths = _ordered_image_paths(args.museum)
    truth_path = args.ground_truth or args.queries / "gt_corresps.pkl"
    with truth_path.open("rb") as stream:
        truth = normalize_truth(pickle.load(stream), len(query_ids))
    if any(value != -1 and value not in set(museum_ids)
           for row in truth for value in row):
        raise ValueError("Ground truth contains IDs absent from the museum")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.rematch:
        args.scores = args.output / "sift" / "matching_scores.json"
        run_matching(SimpleNamespace(
            museum_features=args.museum_features, query_features=args.query_features,
            ratio=0.75, ransac_threshold=5.0, distance=args.distance,
            top_k=10, output=args.scores))
    scores = read_json(args.scores)
    validate_scores(scores, query_ids, museum_ids)
    calibration = calibrate_threshold(scores, truth)
    calibration["development_scores"] = str(args.scores)
    write_json(args.output / "sift" / "calibration.json", calibration)
    sift = make_predictions(scores, calibration["min_inliers"])
    unrejected = [[
        [int(candidate["museum_id"]) for candidate in painting["candidates"]]
        for painting in image["paintings"]] for image in scores["images"]]

    baseline, baseline_report = retrieve_folder_w3(
        args.museum, args.queries, args.output / "w3_baseline",
        distance_name="L1", top_k=10, background_removal=True,
        query_filter=build_query_filter("median", kernel=3),
        descriptor_name="W3 Texture fusion, L1, median 3x3", filter_scope="query")
    methods = {"W3 Texture fusion": baseline, "SIFT without rejection": unrejected,
               "SIFT with rejection": sift}
    metrics = {name: evaluate_predictions(predicted, truth)
               for name, predicted in methods.items()}
    nunknown = sum(target == -1 for row in truth for target in row)
    nknown = sum(len(row) for row in truth) - nunknown
    report = {
        "dataset": "QSD1-W4", "query_ids": query_ids, "ground_truth": str(truth_path),
        "known_paintings": nknown, "unknown_paintings": nunknown,
        "min_inliers": calibration["min_inliers"], "sift_settings": scores["settings"],
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "baseline": {key: value for key, value in baseline_report.items()
                     if key not in ("per_image", "metrics")},
        "metrics": metrics,
    }
    write_json(args.output / "comparison.json", report)
    write_json(args.output / "predictions.json", {"query_ids": query_ids, **methods})
    with (args.output / "sift" / "result.pkl").open("wb") as stream:
        pickle.dump(sift, stream, protocol=4)
    with (args.output / "sift" / "result.pkl").open("rb") as stream:
        if pickle.load(stream) != sift:
            raise ValueError("Saved predictions differ from evaluated results")
    save_details(args.output / "per_painting.csv", query_ids, truth, methods, scores)
    save_error_examples(args.output / "errors.png", query_paths,
                        dict(zip(museum_ids, museum_paths)), truth, sift, scores)
    text = comparison_markdown(metrics, calibration["min_inliers"], len(truth),
                               nknown, nunknown)
    (args.output / "comparison.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
