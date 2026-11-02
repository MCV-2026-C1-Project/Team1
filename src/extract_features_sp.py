"""Batch SuperPoint extraction for museum images and query painting crops."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from keypoints_superpoint import extract_superpoint
from methods_w2 import _ordered_image_paths, _read_rgb
from paintings_w3 import PaintingParams, segment_paintings


def save_features(destination, keypoints, descriptors, box):
    points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
    np.savez_compressed(
        destination,
        points=points,
        sizes=np.asarray([kp.size for kp in keypoints], dtype=np.float32),
        angles=np.asarray([kp.angle for kp in keypoints], dtype=np.float32),
        responses=np.asarray([kp.response for kp in keypoints], dtype=np.float32),
        descriptors=np.asarray(descriptors, dtype=np.float32),
        box=np.asarray(box, dtype=np.int32),
    )


def extract_folder(folder, output_dir, is_query=False, max_keypoints=1024,
                   image_size=960, limit=None, segmentation_threshold=4.0,
                   device=None):
    if max_keypoints < 1:
        raise ValueError("max_keypoints must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    params = PaintingParams()
    if segmentation_threshold is not None:
        if segmentation_threshold <= 0:
            raise ValueError("segmentation_threshold must be positive")
        params.thr_min = params.thr_max = segmentation_threshold

    image_ids, paths = _ordered_image_paths(folder)
    if limit is not None:
        image_ids, paths = image_ids[:limit], paths[:limit]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "schema_version": 2,
        "method": "SuperPoint",
        "nfeatures": int(max_keypoints),
        "image_size": int(image_size),
        "opencv_version": cv2.__version__,
        "role": "queries" if is_query else "museum",
        "source_folder": str(Path(folder).resolve()),
        "preprocessing": f"grayscale; longest side {image_size}; no denoising",
        "coordinate_system": "keypoints relative to each crop",
        "segmentation": asdict(params) if is_query else None,
        "images": [],
    }

    for position, (image_id, path) in enumerate(zip(image_ids, paths), start=1):
        image_rgb = _read_rgb(path)
        height, width = image_rgb.shape[:2]
        fallback = False
        if is_query:
            _, boxes = segment_paintings(image_rgb, params=params)
            if not boxes:
                boxes = [(0, 0, width, height)]
                fallback = True
                print(f"WARNING: query {image_id}: no painting detected; using whole image.")
        else:
            boxes = [(0, 0, width, height)]

        image_record = {"image_id": int(image_id), "source_path": str(path.resolve()),
                        "image_shape": list(image_rgb.shape),
                        "segmentation_fallback": fallback, "paintings": []}
        for painting_index, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            crop_rgb = image_rgb[y0:y1, x0:x1]
            keypoints, descriptors = extract_superpoint(
                crop_rgb, max_num_keypoints=max_keypoints,
                image_size=image_size, device=device)
            filename = f"{image_id:05d}_painting{painting_index}.npz"
            save_features(output_dir / filename, keypoints, descriptors, box)
            image_record["paintings"].append({
                "painting_index": painting_index,
                "box": [int(v) for v in box],
                "feature_file": filename,
                "keypoint_count": len(keypoints)})
            if not keypoints:
                print(f"WARNING: image {image_id}, painting {painting_index}: no SuperPoint features.")
        index["images"].append(image_record)
        counts = [p["keypoint_count"] for p in image_record["paintings"]]
        print(f"[{position}/{len(paths)}] image {image_id}: "
              f"{len(boxes)} painting(s), keypoints={counts}", flush=True)

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nSaved feature index: {index_path}")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--museum", type=Path)
    source.add_argument("--queries", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--segmentation-threshold", type=float, default=4.0)
    parser.add_argument("--no-segmentation", action="store_true",
                        help="Treat query images as already cropped (skip wall segmentation)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    is_query = args.queries is not None
    extract_folder(args.queries if is_query else args.museum, args.output,
                   is_query=is_query and not args.no_segmentation,
                   max_keypoints=args.max_keypoints,
                   image_size=args.image_size, limit=args.limit,
                   segmentation_threshold=args.segmentation_threshold,
                   device=args.device)
    if is_query and args.no_segmentation:
        # Keep the queries role while skipping crop detection.
        index_path = Path(args.output) / "index.json"
        index = json.loads(index_path.read_text())
        index["role"] = "queries"
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
