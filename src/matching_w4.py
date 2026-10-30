"""Week 4: match saved SIFT features and visualize verified matches."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from methods_w2 import _read_rgb


def load_features(path):
    """Load numerical feature arrays without rerunning SIFT."""
    with np.load(path, allow_pickle=False) as saved:
        return {
            name: saved[name]
            for name in ("points", "descriptors", "box")
        }


def find_image(index, image_id):
    """Find an image by its actual filename ID."""
    for image in index["images"]:
        if image["image_id"] == image_id:
            return image
    raise ValueError(f"Image ID {image_id} is absent from the index")


def match_sift(query, museum, ratio=0.75, ransac_threshold=5.0):
    """Find tentative matches, then geometrically verify them."""

    if not 0 < ratio < 1 or ransac_threshold <= 0:
        raise ValueError("Invalid matching thresholds")

    matches = []

    # Need at least two museum descriptors for the ratio test.
    if (
        len(query["descriptors"]) > 0
        and len(museum["descriptors"]) >= 2
    ):
        matcher = cv2.BFMatcher(cv2.NORM_L2)

        neighbors = matcher.knnMatch(
            query["descriptors"],
            museum["descriptors"],
            k=2,
        )

        # Retain a match only if the best neighbor is clearly better
        # than the second-best neighbor.
        candidates = [
            pair[0]
            for pair in neighbors
            if len(pair) == 2
            and pair[0].distance < ratio * pair[1].distance
        ]

        # Avoid counting multiple query descriptors matched to the
        # same museum descriptor. Keep the closest such match.
        used = set()

        for match in sorted(candidates, key=lambda m: m.distance):
            if match.trainIdx not in used:
                matches.append(match)
                used.add(match.trainIdx)

    inliers = np.zeros(len(matches), dtype=bool)

    # Four points can define a homography exactly.
    # Require additional evidence before attempting verification.
    if len(matches) >= 8:
        src = np.float32([
            query["points"][m.queryIdx]
            for m in matches
        ])

        dst = np.float32([
            museum["points"][m.trainIdx]
            for m in matches
        ])

        def reference_points(points, box):
            x0, y0, x1, y1 = box

            # Express coordinates at a reference longest side of 1000.
            # This avoids making the RANSAC tolerance depend on the
            # original image resolution.
            return points * (1000.0 / max(x1 - x0, y1 - y0))

        src = reference_points(src, query["box"])
        dst = reference_points(dst, museum["box"])

        # Make repeated runs reproducible.
        cv2.setRNGSeed(0)

        try:
            homography, mask = cv2.findHomography(
                src,
                dst,
                cv2.RANSAC,
                ransac_threshold,
            )
        except cv2.error:
            homography, mask = None, None

        if (
            homography is not None
            and mask is not None
            and np.isfinite(homography).all()
        ):
            inliers = mask.ravel().astype(bool)

    return {
        "matches": matches,
        "inliers": inliers,
        "tentative_count": len(matches),
        "inlier_count": int(inliers.sum()),
    }


def read_crop(image_record, features):
    """Read the same crop that was used during feature extraction."""
    image = _read_rgb(image_record["source_path"])
    x0, y0, x1, y1 = map(int, features["box"])
    return image[y0:y1, x0:x1]


def save_visualization(
    query_record,
    query,
    museum_record,
    museum,
    result,
    output,
):
    """Draw up to 60 geometrically verified matches."""

    query_image = read_crop(query_record, query)
    museum_image = read_crop(museum_record, museum)

    # Reconstruct minimal KeyPoint objects for drawing.
    # Matching itself uses the saved coordinates and descriptors.
    query_keypoints = [
        cv2.KeyPoint(float(x), float(y), 1.0)
        for x, y in query["points"]
    ]

    museum_keypoints = [
        cv2.KeyPoint(float(x), float(y), 1.0)
        for x, y in museum["points"]
    ]

    verified = [
        match
        for match, keep in zip(
            result["matches"],
            result["inliers"],
        )
        if keep
    ]

    visualization = cv2.drawMatches(
        query_image,
        query_keypoints,
        museum_image,
        museum_keypoints,
        verified[:60],
        None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Reduce only the visualization size, not the matching coordinates.
    if visualization.shape[1] > 1800:
        scale = 1800 / visualization.shape[1]
        visualization = cv2.resize(
            visualization,
            (
                1800,
                max(1, round(visualization.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    visualization = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR,
    )

    if not cv2.imwrite(str(output), visualization):
        raise OSError(f"Could not write {output}")

    print(f"\nMatch visualization: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--museum-features",
        type=Path,
        default=Path("results/week4/sift/museum"),
    )
    parser.add_argument(
        "--query-features",
        type=Path,
        default=Path("results/week4/sift/qsd1_w4_threshold4"),
    )
    parser.add_argument("--query-id", type=int, required=True)
    parser.add_argument("--painting-index", type=int, default=0)
    parser.add_argument(
        "--museum-id",
        type=int,
        help="Match only this museum ID; otherwise rank all museum images",
    )
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/week4/matches/example.png"),
    )
    args = parser.parse_args()

    query_index = json.loads(
        (args.query_features / "index.json").read_text()
    )
    museum_index = json.loads(
        (args.museum_features / "index.json").read_text()
    )

    query_record = find_image(query_index, args.query_id)

    paintings = {
        p["painting_index"]: p
        for p in query_record["paintings"]
    }

    if args.painting_index not in paintings:
        raise ValueError(
            f"Available painting indices: {list(paintings)}"
        )

    query_painting = paintings[args.painting_index]
    query = load_features(
        args.query_features / query_painting["feature_file"]
    )

    if query_record["segmentation_fallback"]:
        print("WARNING: this query uses a whole-image fallback.")

    museum_records = museum_index["images"]

    if args.museum_id is not None:
        museum_records = [
            find_image(museum_index, args.museum_id)
        ]

    results = []

    for record in museum_records:
        painting = record["paintings"][0]
        museum = load_features(
            args.museum_features / painting["feature_file"]
        )

        result = match_sift(
            query,
            museum,
            ratio=args.ratio,
            ransac_threshold=args.ransac_threshold,
        )

        results.append((record, painting, result))

    # Verified matches are the primary score.
    # Tentative matches break ties, followed by museum ID.
    results.sort(
        key=lambda item: (
            -item[2]["inlier_count"],
            -item[2]["tentative_count"],
            item[0]["image_id"],
        )
    )

    print(
        f"\nQuery {args.query_id}, painting {args.painting_index}"
        f", descriptors: {len(query['descriptors'])}"
    )
    print(f"{'Museum ID':>12} {'Tentative':>12} {'Verified':>12}")

    for record, painting, result in results[:10]:
        print(
            f"{record['image_id']:>12}"
            f"{result['tentative_count']:>12}"
            f"{result['inlier_count']:>12}"
        )

    best_record, best_painting, best_result = results[0]
    best_features = load_features(
        args.museum_features / best_painting["feature_file"]
    )

    save_visualization(
        query_record,
        query,
        best_record,
        best_features,
        best_result,
        args.output,
    )

    print(
        "\nThis is a matching diagnostic. "
        "Unknown rejection is not calibrated yet."
    )


if __name__ == "__main__":
    main()