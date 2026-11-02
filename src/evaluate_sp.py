"""Evaluate SuperPoint+LightGlue retrieval and compare with SIFT.

Produces the same ranking JSON schema as Week 4 so calibration, rejection
and per-painting metrics can be reused unchanged.
"""

import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace

from batch_matching_sp import run_matching as run_matching_sp
from batch_matching_w4 import (calibrate_threshold, make_predictions,
                               read_json, write_json)
from batch_matching_w4 import run_matching as run_matching_sift
from evaluate_w4 import evaluate_predictions
from methods_w2 import _ordered_image_paths


def evaluate_scores(scores, truth):
    calibration = calibrate_threshold(scores, truth) if any(
        v == -1 for row in truth for v in row) else None
    if calibration is not None:
        predictions = make_predictions(scores, calibration["min_inliers"])
    else:
        predictions = [[[int(c["museum_id"]) for c in p["candidates"]]
                        for p in img["paintings"]] for img in scores["images"]]
    metrics = evaluate_predictions(predictions, truth)
    return predictions, metrics, calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--museum", type=Path, default=Path("data/BBDD"))
    parser.add_argument("--queries", type=Path, default=Path("data/qsd1_w3"))
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--sp-museum", type=Path, required=True)
    parser.add_argument("--sp-queries", type=Path, required=True)
    parser.add_argument("--sift-scores", type=Path, default=None,
                        help="Reuse saved SIFT matching scores instead of rematching")
    parser.add_argument("--sift-museum", type=Path, default=None)
    parser.add_argument("--sift-queries", type=Path, default=None)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/week4/sp/evaluation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    truth_path = args.ground_truth or args.queries / "gt_corresps.pkl"
    with truth_path.open("rb") as stream:
        truth = pickle.load(stream)
    # Normalize singleton lists to ints (same convention as Week 4).
    norm = []
    for row in truth:
        norm.append([item[0] if isinstance(item, (list, tuple)) else item for item in row])
    truth = norm

    sp_scores_path = args.output / "matching_scores.json"
    run_matching_sp(SimpleNamespace(
        museum_features=args.sp_museum, query_features=args.sp_queries,
        ransac_threshold=args.ransac_threshold, top_k=args.top_k,
        device=args.device, output=sp_scores_path))
    sp_scores = read_json(sp_scores_path)
    sp_pred, sp_metrics, sp_calib = evaluate_scores(sp_scores, truth)

    sift_scores = None
    if args.sift_scores is not None:
        sift_scores = read_json(args.sift_scores)
    elif args.sift_museum is not None and args.sift_queries is not None:
        sift_scores_path = args.output / "sift_matching_scores.json"
        run_matching_sift(SimpleNamespace(
            museum_features=args.sift_museum, query_features=args.sift_queries,
            ratio=0.75, ransac_threshold=args.ransac_threshold,
            distance="L2", top_k=args.top_k, output=sift_scores_path))
        sift_scores = read_json(sift_scores_path)

    report = {"dataset": str(args.queries), "ground_truth": str(truth_path),
              "sp_settings": sp_scores["settings"],
              "sp_metrics": sp_metrics,
              "sp_calibration": sp_calib["min_inliers"] if sp_calib else None}
    if sift_scores is not None:
        sift_pred, sift_metrics, sift_calib = evaluate_scores(sift_scores, truth)
        report.update({"sift_settings": sift_scores["settings"],
                       "sift_metrics": sift_metrics,
                       "sift_calibration": sift_calib["min_inliers"] if sift_calib else None})
        print(f"SIFT: mAP@1={sift_metrics['mAP@1']:.4f} "
              f"mAP@5={sift_metrics['mAP@5']:.4f} "
              f"unknownF1={sift_metrics['unknown']['f1']:.4f}")
    print(f"SP+LG: mAP@1={sp_metrics['mAP@1']:.4f} "
          f"mAP@5={sp_metrics['mAP@5']:.4f} "
          f"unknownF1={sp_metrics['unknown']['f1']:.4f}")
    write_json(args.output / "comparison_sp.json", report)
    with (args.output / "predictions_sp.pkl").open("wb") as stream:
        pickle.dump(sp_pred, stream, protocol=4)


if __name__ == "__main__":
    main()
