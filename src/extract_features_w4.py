"""Batch SIFT extraction for museum images and query painting crops."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from keypoints_w4 import extract_sift
from methods_w2 import _ordered_image_paths, _read_rgb
from paintings_w3 import PaintingParams, segment_paintings

def save_features(destination, keypoints, descriptors, box):
    """Store numerical arrays instead of OpenCV KeyPoint objects."""

    points = np.asarray(
        [kp.pt for kp in keypoints],
        dtype=np.float32,
    ).reshape(-1, 2)

    np.savez_compressed(
        destination,
        points=points,
        sizes=np.asarray(
            [kp.size for kp in keypoints],
            dtype=np.float32,
        ),
        angles=np.asarray(
            [kp.angle for kp in keypoints],
            dtype=np.float32,
        ),
        responses=np.asarray(
            [kp.response for kp in keypoints],
            dtype=np.float32,
        ),
        descriptors=descriptors,
        box=np.asarray(box, dtype=np.int32),
    )


def extract_folder(
    folder,
    output_dir,
    is_query=False,
    nfeatures=1000,
    limit=None,
    segmentation_threshold=4.0,
):
    """Save features for every museum image or detected query painting."""

    if nfeatures < 1:
        raise ValueError("nfeatures must be positive")

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    if segmentation_threshold is not None and segmentation_threshold <= 0:
        raise ValueError("segmentation_threshold must be positive")
    params = PaintingParams()
    if segmentation_threshold is not None:
        params.thr_min = params.thr_max = segmentation_threshold

    image_ids, paths = _ordered_image_paths(folder)

    # Useful for checking a few images before processing everything.
    if limit is not None:
        image_ids = image_ids[:limit]
        paths = paths[:limit]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "schema_version": 2,
        "method": "SIFT",
        "nfeatures": nfeatures,
        "opencv_version": cv2.__version__,
        "role": "queries" if is_query else "museum",
        "source_folder": str(Path(folder).resolve()),
        "preprocessing": "grayscale; no extra denoising or resizing",
        "coordinate_system": "keypoints relative to each crop",
        "segmentation": asdict(params) if is_query else None,
        "images": [],
    }

    for position, (image_id, path) in enumerate(
        zip(image_ids, paths),
        start=1,
    ):
        image_rgb = _read_rgb(path)
        height, width = image_rgb.shape[:2]

        fallback = False

        if is_query:
            _, boxes = segment_paintings(image_rgb, params=params)

            if not boxes:
                # Keep the query represented even if segmentation fails.
                boxes = [(0, 0, width, height)]
                fallback = True
                print(
                    f"WARNING: query {image_id}: no painting detected; "
                    "using the whole image."
                )
        else:
            # Museum images already contain an individual painting.
            boxes = [(0, 0, width, height)]

        image_record = {
            "image_id": int(image_id),
            "source_path": str(path.resolve()),
            "image_shape": list(image_rgb.shape),
            "segmentation_fallback": fallback,
            "paintings": [],
        }

        for painting_index, box in enumerate(boxes):
            x0, y0, x1, y1 = box

            # NumPy indexes rows first, so y comes before x.
            crop_rgb = image_rgb[y0:y1, x0:x1]

            keypoints, descriptors = extract_sift(
                crop_rgb,
                nfeatures=nfeatures,
            )

            filename = (
                f"{image_id:05d}_painting{painting_index}.npz"
            )

            save_features(
                output_dir / filename,
                keypoints,
                descriptors,
                box,
            )

            image_record["paintings"].append({
                "painting_index": painting_index,
                "box": [int(value) for value in box],
                "feature_file": filename,
                "keypoint_count": len(keypoints),
            })

            if not keypoints:
                print(
                    f"WARNING: image {image_id}, "
                    f"painting {painting_index}: no SIFT features."
                )

        index["images"].append(image_record)

        counts = [
            painting["keypoint_count"]
            for painting in image_record["paintings"]
        ]

        print(
            f"[{position}/{len(paths)}] image {image_id}: "
            f"{len(boxes)} painting(s), keypoints={counts}",
            flush=True,
        )

    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved feature index: {index_path}")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--museum", type=Path)
    source.add_argument("--queries", type=Path)

    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nfeatures", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--segmentation-threshold", type=float, default=4.0,
                        help="Fixed wall-distance threshold for query crops")

    args = parser.parse_args()

    is_query = args.queries is not None
    folder = args.queries if is_query else args.museum

    extract_folder(
        folder=folder,
        output_dir=args.output,
        is_query=is_query,
        nfeatures=args.nfeatures,
        limit=args.limit,
        segmentation_threshold=args.segmentation_threshold,
    )


if __name__ == "__main__":
    main()
