

import argparse
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import cv2

from main import compute_descriptors, DESCRIPTOR_NAMES
from main import METHODS_W1_NAMES  # Frozen QST1 submission in W1; QST1/QST2-W2 will go in task4_w2.py
from k_similarity import retrieve_all_queries, DISTANCE_FUNCTIONS
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# QST1 submission plan: at most two methods (statement). Grayscale and RGB
# descriptors are kept in utils.py / main.py for validation/ablation only.
#
# Why HSV: painting retrieval is driven by chromatic content. HSV decouples
# chromaticity (Hue, Saturation) from illumination (Value), so it is more
# robust than RGB — where brightness and chroma are coupled in all three
# channels — to lighting/exposure changes between query photos and museum
# scans, and more discriminative than grayscale, which discards color. On
# QSD1 validation HSV clearly beats RGB and grayscale (mAP@5 0.59 vs 0.40
# vs 0.29), and the 4x4 HSV grid adds spatial layout while still using only
# concatenated 1D histograms (mAP@5 0.82). Hence method1 = HSV (ex method2),
# method2 = HSV grid (ex method4), both compared with L1 (tied best on QSD1).
SUBMISSION_PLAN = (
    ("method1", "HSV", "L1"),
    ("method2", "HSV grid", "L1"),
)

# Kept for reference: validation-tuned comparison per descriptor, in
# DESCRIPTOR_NAMES order (Grayscale, HSV, RGB, HSV grid). Only the HSV-based
# entries above are submitted.
DEFAULT_DISTANCES = ["Chi-square", "L1", "L1", "L1"]


def image_id(name):
    """Read the numeric suffix in 00007.jpg or bbdd_00007.jpg."""
    match = re.search(r"(\d+)$", Path(name).stem)
    if match is None:
        raise ValueError(f"Image filename has no numeric ID: {name}")
    return int(match.group(1))


def load_descriptors(path):
    """Read JPGs from an image folder in numeric order, one image at a time."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Expected an image folder: {path}")
    names = sorted(
        (p for p in path.iterdir() if p.is_file()
         and p.suffix.lower() in (".jpg", ".jpeg")),
        key=lambda p: (image_id(p), p.name),
    )
    ids = [image_id(p) for p in names]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Expected nonempty, unique image IDs in {path}")
    descriptors = [[] for _ in DESCRIPTOR_NAMES]
    for name in names:
        img = cv2.imread(str(name), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot decode image: {name}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for destination, computed in zip(
            descriptors, compute_descriptors(SimpleNamespace(images=[rgb]))
        ):
            destination.extend(computed)
    return ids, descriptors


def check_submission_ids(result, museum_ids, query_ids):
    """Minimal ID check: K=10 unique Python-int museum IDs per query."""
    museum_ids = set(museum_ids)
    if len(result) != len(query_ids):
        raise ValueError(
            f"Expected {len(query_ids)} query rows, got {len(result)}"
        )
    for row in result:
        if len(row) != 10 or len(set(row)) != 10:
            raise ValueError(f"Each query needs 10 unique IDs, got {row}")
        if any(type(v) is not int for v in row):
            raise ValueError(f"Museum IDs must be Python ints, got {row}")
        unknown = set(row) - museum_ids
        if unknown:
            raise ValueError(f"Unknown museum IDs {sorted(unknown)}")


def generate_submissions(museum_path, query_path, output_dir, distance=None):
    museum_ids, museum_descriptors = load_descriptors(museum_path)
    query_ids, query_descriptors = load_descriptors(query_path)
    if len(museum_ids) < 10:
        raise ValueError("Task 4 requires at least 10 museum images")
    by_name = dict(zip(DESCRIPTOR_NAMES, museum_descriptors))
    by_query = dict(zip(DESCRIPTOR_NAMES, query_descriptors))
    for output_name, descriptor_name, default_distance in SUBMISSION_PLAN:
        museum = by_name[descriptor_name]
        queries = by_query[descriptor_name]
        distance_name = distance or default_distance
        rankings = retrieve_all_queries(queries, museum, DISTANCE_FUNCTIONS[distance_name], top_k=10)
        result = [[museum_ids[idx] for idx, score in ranking] for ranking in rankings]
        check_submission_ids(result, museum_ids, query_ids)
        destination = Path(output_dir) / output_name / "result.pkl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            pickle.dump(result, stream, protocol=4)
        print(f"{output_name} ({descriptor_name}, {distance_name}): "
              f"{len(query_ids)} queries × 10 integer IDs -> {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--museum", type=Path, required=True, help="Museum image folder")
    parser.add_argument("--queries", type=Path, required=True, help="Test query image folder")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "week1" / "QST1")
    parser.add_argument("--distance", choices=DISTANCE_FUNCTIONS,
                        help="Override the validation-based comparison for all methods")
    args = parser.parse_args()
    generate_submissions(args.museum, args.queries, args.output_dir, args.distance)


if __name__ == "__main__":
    main()
