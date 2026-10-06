"""Cached spatial-pyramid configuration sweep using Week 1 retrieval/evaluation."""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from spatial_pyramid import compute_spatial_pyramid
from utils import load_image, _l1_normalize
from task4 import image_id
from k_similarity import DISTANCE_FUNCTIONS, retrieve_all_queries, mean_average_precision_at_k

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_sweep_descriptors(folder, configurations, levels, normalization, cache_dir):
    """Decode each image once for all uncached configurations; retain only descriptors."""
    folder = Path(folder)
    paths = sorted((p for p in folder.iterdir() if p.is_file()
                    and p.suffix.lower() in (".jpg", ".jpeg")),
                   key=lambda p: (image_id(p), p.name))
    ids = [image_id(p) for p in paths]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"Expected nonempty, unique image IDs in {folder}")
    # Invalidate caches when inputs, descriptor code, utilities, or OpenCV change.
    manifest = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    source = hashlib.sha256()
    for filename in ("spatial_pyramid.py", "utils.py", "spatial_pyramid_sweep.py"):
        source.update(Path(__file__).with_name(filename).read_bytes())
    descriptors, missing, destinations = {}, [], {}
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for config in configurations:
        key = hashlib.sha256(json.dumps([manifest, config, levels, normalization,
                                         cv2.__version__, source.hexdigest()]).encode()).hexdigest()
        destination = cache_dir / f"{key}.npz"
        destinations[config] = destination
        if destination.exists():
            with np.load(destination, allow_pickle=False) as cached:
                descriptors[config] = cached["descriptors"]
        else:
            missing.append(config)
            descriptors[config] = []
    print(f"{folder.name}: {len(ids)} images; {len(configurations) - len(missing)} cached, "
          f"{len(missing)} configurations to compute", flush=True)
    if missing:
        for index, path in enumerate(paths, start=1):
            image = load_image(str(path))
            for config in missing:
                mode, space, bins = config
                descriptors[config].append(compute_spatial_pyramid(
                    image, pyramid_levels=levels, histogram_type=mode,
                    color_space=space, bins=bins, normalization=normalization))
            if index % 50 == 0 or index == len(paths):
                print(f"  Descriptors: {index}/{len(paths)} images", flush=True)
        for config in missing:
            descriptors[config] = np.asarray(descriptors[config])
            np.savez_compressed(destinations[config], descriptors=descriptors[config])
    return ids, descriptors


def pyramid_prefix(descriptors, levels, mode, bins, normalization):
    """Recover a smaller prefix pyramid without recomputing any region histogram."""
    region_dimension = 3 * bins if mode == "1d" else bins ** 3
    dimension = sum(grid ** 2 for grid in levels) * region_dimension
    prefix = descriptors[:, :dimension]
    if normalization == "l1":
        return np.asarray([_l1_normalize(row) for row in prefix])
    if normalization == "l2":
        return np.asarray([cv2.normalize(row, None, alpha=1, norm_type=cv2.NORM_L2).flatten()
                           for row in prefix])
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--museum", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--color-spaces", nargs="+", choices=["RGB", "HSV", "Lab", "YCrCb", "YCbCr"],
                        default=["HSV", "RGB", "Lab"])
    parser.add_argument("--bins-1d", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--bins-3d", nargs="+", type=int, default=[8])
    parser.add_argument("--pyramid-levels", nargs="+", type=int, default=[1, 2, 4],
                        help="Evaluate every prefix, e.g. [1], [1,2], [1,2,4]")
    parser.add_argument("--normalization", choices=["l1", "l2", "none"], default="l1")
    parser.add_argument("--distances", nargs="+", choices=DISTANCE_FUNCTIONS,
                        default=["L1", "Hellinger"])
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "results/spatial_pyramid/cache")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/spatial_pyramid/sweep.csv")
    args = parser.parse_args()
    if (any(b <= 0 for b in args.bins_1d + args.bins_3d)
            or any(g <= 0 for g in args.pyramid_levels)
            or len(set(args.pyramid_levels)) != len(args.pyramid_levels)):
        parser.error("Bins and grid sizes must be positive; grid sizes must be unique")
    configurations = list(dict.fromkeys(
        (mode, space, bins)
        for mode, choices in (("1d", args.bins_1d), ("3d", args.bins_3d))
        for space in args.color_spaces for bins in choices))
    with args.ground_truth.open("rb") as stream:
        ground_truth = pickle.load(stream)
    start = perf_counter()
    museum_ids, museum = load_sweep_descriptors(args.museum, configurations,
                                               args.pyramid_levels, args.normalization, args.cache_dir)
    query_ids, queries = load_sweep_descriptors(args.queries, configurations,
                                               args.pyramid_levels, args.normalization, args.cache_dir)
    if len(query_ids) != len(ground_truth):
        raise ValueError("Ground-truth count must match numerically ordered queries")
    if any(int(i) not in set(museum_ids) for relevant in ground_truth for i in relevant):
        raise ValueError("Ground truth refers to IDs outside this museum")
    rows = []
    total = len(configurations) * len(args.pyramid_levels) * len(args.distances)
    for mode, space, bins in configurations:
        config = (mode, space, bins)
        for count in range(1, len(args.pyramid_levels) + 1):
            levels = args.pyramid_levels[:count]
            museum_prefix = pyramid_prefix(museum[config], levels, mode, bins, args.normalization)
            query_prefix = pyramid_prefix(queries[config], levels, mode, bins, args.normalization)
            for distance in args.distances:
                ranked = retrieve_all_queries(query_prefix, museum_prefix,
                                               DISTANCE_FUNCTIONS[distance], top_k=5)
                ranked_ids = [[museum_ids[i] for i, score in ranking] for ranking in ranked]
                row = dict(histogram_type=mode, color_space=space, bins=bins,
                           pyramid_levels=" ".join(map(str, levels)),
                           normalization=args.normalization, distance=distance,
                           dimension=len(museum_prefix[0]),
                           **{f"mAP@{k}": mean_average_precision_at_k(ranked_ids, ground_truth, k)
                              for k in (1, 5)})
                rows.append(row)
                print(f"[{len(rows)}/{total}] {mode} {space} bins={bins} levels={levels} "
                      f"{distance}: mAP@1={row['mAP@1']:.4f}, mAP@5={row['mAP@5']:.4f}", flush=True)
    rows.sort(key=lambda row: (-row["mAP@5"], -row["mAP@1"], row["dimension"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"museum": str(args.museum.resolve()), "queries": str(args.queries.resolve()),
                "ground_truth": str(args.ground_truth.resolve()), "query_ids": query_ids,
                "museum_count": len(museum_ids), "configuration_count": len(rows),
                "elapsed_seconds": perf_counter() - start, "sort_metric": "mAP@5"}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"\nTop configurations (sorted by mAP@5):")
    for row in rows[:10]:
        print(row)
    print(f"\nSaved {args.output}; elapsed {perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
