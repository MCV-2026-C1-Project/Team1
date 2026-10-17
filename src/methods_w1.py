"""Methods_W1: descriptors implemented and validated in Week 1.

Includes the QST1 submission setup (ex task4.py): at most two methods
(statement). Grayscale and RGB descriptors are kept for
validation/ablation only.

Canonical order (METHODS_W1_NAMES):
    0. Grayscale  (256 bins, L1)
    1. HSV        (H 30 + S 32 + V 32 = 94, concatenated 1D marginal, L1)
    2. RGB        (32*3 = 96, concatenated 1D marginal, L1)
    3. HSV grid  (4x4 blocks, per block H 16 + S 8 + V 8 = 32 -> 512, L1 per
                  block and global L1)

On QSD1 the best was HSV grid (mAP@5 0.82, L1) and the best global was HSV
(mAP@5 0.59, L1). Hence the QST1-W1 submission uses only HSV and HSV grid.

Run as a script (argparse selects the action):
    python methods_w1.py --list
    python methods_w1.py --museum <dir> --queries <dir> [--method NAME] [--distance L1]
    python methods_w1.py --museum <dir> --queries <dir> --ground-truth <pkl> [...]
    python methods_w1.py --museum <dir> --queries <dir> --output-dir <dir> [...]
"""

import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace

import cv2

from utils import (
    compute_color_hsv_histogram,
    compute_color_rgb_histogram,
    compute_gray_histogram,
    compute_hsv_grid_histogram,
    image_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# QST1 submission plan: at most two methods (statement). Grayscale and RGB
# descriptors are kept in utils.py / METHODS_W1 for validation/ablation only.
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
# METHODS_W1_NAMES order (Grayscale, HSV, RGB, HSV grid). Only the HSV-based
# entries above are submitted.
DEFAULT_DISTANCES = ["Chi-square", "L1", "L1", "L1"]

# Name -> function(img) where img is RGB except HSV/grid which expect HSV.
# Kept as an ordered dict so the order is explicit.
METHODS_W1 = {
    "Grayscale": lambda img_rgb: compute_gray_histogram(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    ),
    "HSV": lambda img_rgb: compute_color_hsv_histogram(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    ),
    "RGB": compute_color_rgb_histogram,
    "HSV grid": lambda img_rgb: compute_hsv_grid_histogram(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    ),
}

METHODS_W1_NAMES = list(METHODS_W1.keys())


def compute_descriptors_w1(dataset):
    """Task 1 (W1): one descriptor per W1 method, in METHODS_W1_NAMES order."""
    methods = [[] for _ in METHODS_W1_NAMES]
    for img in dataset.images:
        for i, name in enumerate(METHODS_W1_NAMES):
            methods[i].append(METHODS_W1[name](img))
    return methods


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
    descriptors = [[] for _ in METHODS_W1_NAMES]
    for name in names:
        img = cv2.imread(str(name), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot decode image: {name}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for destination, computed in zip(
            descriptors, compute_descriptors_w1(SimpleNamespace(images=[rgb]))
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


def generate_submissions(museum_path, query_path, output_dir, distance=None,
                         plan=SUBMISSION_PLAN):
    """Write one result.pkl per plan entry (10 unique museum IDs per query)."""
    # Local import: k_similarity imports METHODS_W1 at module level.
    from k_similarity import DISTANCE_FUNCTIONS, retrieve_all_queries

    museum_ids, museum_descriptors = load_descriptors(museum_path)
    query_ids, query_descriptors = load_descriptors(query_path)
    if len(museum_ids) < 10:
        raise ValueError("Task 4 requires at least 10 museum images")
    by_name = dict(zip(METHODS_W1_NAMES, museum_descriptors))
    by_query = dict(zip(METHODS_W1_NAMES, query_descriptors))
    for output_name, descriptor_name, default_distance in plan:
        museum = by_name[descriptor_name]
        queries = by_query[descriptor_name]
        distance_name = distance or default_distance
        rankings = retrieve_all_queries(
            queries, museum, DISTANCE_FUNCTIONS[distance_name], top_k=10)
        result = [[museum_ids[idx] for idx, score in ranking]
                  for ranking in rankings]
        check_submission_ids(result, museum_ids, query_ids)
        destination = Path(output_dir) / output_name / "result.pkl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            pickle.dump(result, stream, protocol=4)
        print(f"{output_name} ({descriptor_name}, {distance_name}): "
              f"{len(query_ids)} queries x 10 integer IDs -> {destination}")


def evaluate_folders(museum_path, queries_path, method, distance_name,
                     top_k, ground_truth_path=None):
    """Retrieve top-k museum IDs per query; print mAP if ground truth given."""
    # Local import: k_similarity imports METHODS_W1 at module level.
    from k_similarity import (
        DISTANCE_FUNCTIONS,
        mean_average_precision_at_k,
        retrieve_all_queries,
    )

    museum_ids, museum_descriptors = load_descriptors(museum_path)
    query_ids, query_descriptors = load_descriptors(queries_path)
    museum = dict(zip(METHODS_W1_NAMES, museum_descriptors))[method]
    queries = dict(zip(METHODS_W1_NAMES, query_descriptors))[method]
    rankings = retrieve_all_queries(
        queries, museum, DISTANCE_FUNCTIONS[distance_name], top_k=top_k)
    ranked_ids = [[museum_ids[idx] for idx, _ in row] for row in rankings]
    print(f"{method} + {distance_name}: "
          f"{len(museum_ids)} museum, {len(query_ids)} queries, "
          f"dim {len(museum[0])}")
    if ground_truth_path is None:
        for qid, row in zip(query_ids, ranked_ids):
            print(f"Query {qid}: {row}")
        return ranked_ids
    with Path(ground_truth_path).open("rb") as stream:
        ground_truth = pickle.load(stream)
    if len(ground_truth) != len(query_ids):
        raise ValueError("Ground-truth count must match numeric query order")
    for k in (1, top_k) if top_k != 5 else (1, 5):
        print(f"mAP@{k}: "
              f"{mean_average_precision_at_k(ranked_ids, ground_truth, k):.4f}")
    return ranked_ids


def build_parser():
    # Local import so --help works even without the retrieval stack installed.
    try:
        from k_similarity import DISTANCE_FUNCTIONS
        distances = sorted(DISTANCE_FUNCTIONS)
    except Exception:
        distances = ["L1"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="List available W1 methods and exit")
    parser.add_argument("--method", choices=METHODS_W1_NAMES, default="HSV grid",
                        help="Descriptor for retrieval evaluation")
    parser.add_argument("--museum", type=Path, help="Museum image folder")
    parser.add_argument("--queries", type=Path, help="Query image folder")
    parser.add_argument("--ground-truth", type=Path, default=None,
                        help="Trusted correspondence pickle (IDs, query order)")
    parser.add_argument("--distance", choices=distances, default=None,
                        help="Comparison for evaluation; submission uses plan "
                             "defaults unless overridden here")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Write QST1 submission (method1/method2 result.pkl)")
    return parser


def main(args=None):
    parser = build_parser()
    opts = parser.parse_args(args)
    if opts.list:
        for name in METHODS_W1_NAMES:
            print(name)
        return
    if not opts.museum or not opts.queries:
        parser.error("--museum and --queries are required (or use --list)")
    if opts.output_dir is not None:
        plan = SUBMISSION_PLAN
        if opts.method and opts.method not in (p[1] for p in SUBMISSION_PLAN):
            plan = (("method1", opts.method, opts.distance or "L1"),)
        generate_submissions(opts.museum, opts.queries, opts.output_dir,
                             distance=opts.distance, plan=plan)
        return
    evaluate_folders(opts.museum, opts.queries, opts.method,
                     opts.distance or "L1", opts.top_k, opts.ground_truth)


if __name__ == "__main__":
    main()
