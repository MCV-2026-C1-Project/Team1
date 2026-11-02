"""SuperPoint+LightGlue batch matching with the Week 4 scoring schema."""

import argparse
from pathlib import Path

import cv2

from batch_matching_w4 import calibrate_threshold, ordered_images, read_json, write_json
from keypoints_superpoint import match_superpoint
from matching_w4 import load_features


def run_matching(args):
    museum_index = read_json(args.museum_features / "index.json")
    query_index = read_json(args.query_features / "index.json")
    if museum_index["role"] != "museum" or query_index["role"] != "queries":
        raise ValueError("Incorrect museum/query feature folders")
    settings = {
        "method": "SuperPoint+LightGlue",
        "ransac_threshold": args.ransac_threshold,
        "museum_max_keypoints": museum_index.get("nfeatures"),
        "query_max_keypoints": query_index.get("nfeatures"),
        "museum_preprocessing": museum_index.get("preprocessing"),
        "query_preprocessing": query_index.get("preprocessing"),
        "matching_opencv_version": cv2.__version__,
        "museum_feature_opencv_version": museum_index.get("opencv_version"),
        "query_feature_opencv_version": query_index.get("opencv_version"),
        "query_segmentation": query_index.get("segmentation"),
    }
    if museum_index["method"] != "SuperPoint" or query_index["method"] != "SuperPoint":
        raise ValueError("This matcher requires SuperPoint features")
    if args.ransac_threshold <= 0:
        raise ValueError("Invalid matching thresholds")

    museum = []
    for record in ordered_images(museum_index):
        if len(record["paintings"]) != 1:
            raise ValueError("Expected one painting per museum image")
        painting = record["paintings"][0]
        features = load_features(args.museum_features / painting["feature_file"])
        museum.append((int(record["image_id"]), features))
    if not 1 <= args.top_k <= len(museum):
        raise ValueError("top_k must be between 1 and the museum size")

    report = {"settings": settings, "top_k": args.top_k,
              "query_features": str(args.query_features.resolve()),
              "museum_features": str(args.museum_features.resolve()), "images": []}
    images = ordered_images(query_index)
    for position, image in enumerate(images, start=1):
        row = {"image_id": int(image["image_id"]),
               "segmentation_fallback": image["segmentation_fallback"], "paintings": []}
        paintings = sorted(image["paintings"], key=lambda p: p["painting_index"])
        for painting in paintings:
            query = load_features(args.query_features / painting["feature_file"])
            candidates = []
            for museum_id, features in museum:
                result = match_superpoint(query, features,
                                          ransac_threshold=args.ransac_threshold,
                                          device=args.device)
                candidates.append({"museum_id": museum_id,
                                   "verified": result["inlier_count"],
                                   "tentative": result["tentative_count"]})
            candidates.sort(key=lambda item: (-item["verified"], -item["tentative"],
                                              item["museum_id"]))
            row["paintings"].append({"painting_index": painting["painting_index"],
                                     "box": painting["box"],
                                     "candidates": candidates[:args.top_k]})
        report["images"].append(row)
        scores = [p["candidates"][0]["verified"] for p in row["paintings"]]
        print(f"[{position}/{len(images)}] Query {row['image_id']}: "
              f"{len(paintings)} painting(s); best inliers={scores}", flush=True)
    write_json(args.output, report)
    print(f"Saved matching scores: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    match = commands.add_parser("match")
    match.add_argument("--museum-features", type=Path, required=True)
    match.add_argument("--query-features", type=Path, required=True)
    match.add_argument("--output", type=Path, required=True)
    match.add_argument("--ransac-threshold", type=float, default=5.0)
    match.add_argument("--top-k", type=int, default=10)
    match.add_argument("--device", type=str, default=None)
    match.set_defaults(run=run_matching)
    from batch_matching_w4 import tune_threshold, predict
    tune = commands.add_parser("tune")
    tune.add_argument("--scores", type=Path, required=True)
    tune.add_argument("--ground-truth", type=Path, required=True)
    tune.add_argument("--output", type=Path, required=True)
    tune.add_argument("--thresholds", nargs="+", type=int, default=None)
    tune.set_defaults(run=tune_threshold)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--scores", type=Path, required=True)
    prediction.add_argument("--calibration", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.set_defaults(run=predict)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
