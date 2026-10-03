"""Generate Week 1 QST1 submissions without ground-truth correspondences."""
import argparse
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import cv2

from main import compute_descriptors, DESCRIPTOR_NAMES
from k_similarity import retrieve_all_queries, DISTANCE_FUNCTIONS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Best mAP@5 comparisons per descriptor in the supplied results.txt.
DEFAULT_DISTANCES = ["Chi-square", "L1", "L1", "L1", "Chi-square"]


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


def generate_submissions(museum_path, query_path, output_dir, distance=None):
    museum_ids, museum_descriptors = load_descriptors(museum_path)
    query_ids, query_descriptors = load_descriptors(query_path)
    if len(museum_ids) < 10:
        raise ValueError("Task 4 requires at least 10 museum images")
    for index, (museum, queries) in enumerate(
        zip(museum_descriptors, query_descriptors), start=1
    ):
        distance_name = distance or DEFAULT_DISTANCES[index - 1]
        rankings = retrieve_all_queries(queries, museum, DISTANCE_FUNCTIONS[distance_name], top_k=10)
        result = [[museum_ids[idx] for idx, score in ranking] for ranking in rankings]
        destination = Path(output_dir) / f"method{index}" / "result.pkl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            pickle.dump(result, stream, protocol=4)
        print(f"method{index} ({DESCRIPTOR_NAMES[index - 1]}, {distance_name}): "
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
