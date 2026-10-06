"""Evaluate one spatial-pyramid configuration using Week 1 retrieval metrics."""

import argparse
from pathlib import Path
import pickle

from spatial_pyramid import compute_spatial_pyramid
from utils import load_image
from task4 import image_id
from k_similarity import DISTANCE_FUNCTIONS, retrieve_all_queries, mean_average_precision_at_k


def load_pyramid_descriptors(folder, **configuration):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Expected an image folder: {folder}")
    paths = sorted((p for p in folder.iterdir() if p.is_file()
                    and p.suffix.lower() in (".jpg", ".jpeg")),
                   key=lambda p: (image_id(p), p.name))
    ids = [image_id(p) for p in paths]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"Expected nonempty, unique image IDs in {folder}")
    descriptors = [compute_spatial_pyramid(load_image(str(p)), **configuration)
                   for p in paths]
    return ids, descriptors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--museum", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path,
                        help="Trusted correspondence pickle, aligned to numeric query order")
    parser.add_argument("--pyramid-levels", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--histogram-type", choices=["1d", "3d"], default="1d")
    parser.add_argument("--color-space", choices=["RGB", "HSV", "Lab", "YCrCb", "YCbCr"], default="HSV")
    parser.add_argument("--bins", nargs="+", type=int, default=[32])
    parser.add_argument("--normalization", choices=["l1", "l2", "none"], default="l1")
    parser.add_argument("--distance", choices=DISTANCE_FUNCTIONS, default="L1")
    args = parser.parse_args()
    if len(args.bins) not in (1, 3):
        parser.error("--bins requires one or three integers")
    configuration = dict(pyramid_levels=args.pyramid_levels,
                         histogram_type=args.histogram_type, color_space=args.color_space,
                         bins=args.bins[0] if len(args.bins) == 1 else args.bins,
                         normalization=args.normalization)
    museum_ids, museum = load_pyramid_descriptors(args.museum, **configuration)
    query_ids, queries = load_pyramid_descriptors(args.queries, **configuration)
    rankings = retrieve_all_queries(queries, museum, DISTANCE_FUNCTIONS[args.distance], top_k=5)
    ranked_ids = [[museum_ids[index] for index, score in row] for row in rankings]
    print(f"Configuration: {configuration}; comparison: {args.distance}")
    print(f"Museum: {len(museum_ids)}; queries: {len(query_ids)}; dimension: {len(museum[0])}")
    if args.ground_truth:
        with args.ground_truth.open("rb") as stream:
            ground_truth = pickle.load(stream)
        for k in (1, 5):
            print(f"mAP@{k}: {mean_average_precision_at_k(ranked_ids, ground_truth, k):.6f}")
    else:
        for query_id, row in zip(query_ids, ranked_ids):
            print(f"Query {query_id}: {row}")


if __name__ == "__main__":
    main()
