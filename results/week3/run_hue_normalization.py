"""Evaluate HSV hue normalization while keeping retrieval in LAB.

The method estimates a global hue scale for every (query, museum reference)
pair, applies it in HSV, converts the corrected image back to RGB and computes
the usual W3 "Lab blocks 1D" descriptor.  The final ranking blends the raw LAB
distance and the color-normalized LAB distance.

Outputs are written to a new folder, so the earlier W3 filter results are not
modified:

    results/week3/color_reference/
        museum_hue_references.npz
        reference_metadata.json
        reference_hue_histogram.png
    results/week3/hue_normalization/
        summary.csv
        per_query.csv
        README.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from color_reference import (  # noqa: E402
    DEFAULT_REGULARIZATION,
    DEFAULT_SATURATION_THRESHOLD,
    DEFAULT_SCALE_MAX,
    DEFAULT_SCALE_MIN,
    DEFAULT_SCALE_STEP,
    HueReference,
    build_hue_references_from_images,
    correct_hue_scale,
    estimate_hue_scale,
    hue_histogram,
    hue_scale_grid,
    rank_with_hue_references,
)
from k_similarity import (  # noqa: E402
    average_precision_at_k,
    mean_average_precision_at_k,
)
from methods_w2 import (  # noqa: E402
    _load_ground_truth,
    _ordered_image_paths,
    _read_rgb,
)
from methods_w3 import baseline_descriptor  # noqa: E402

MUSEUM = ROOT / "data" / "BBDD"
QUERIES = ROOT / "data" / "qsd1_w3"
GROUND_TRUTH = QUERIES / "gt_corresps.pkl"
AUGMENTATIONS = QUERIES / "augmentations.pkl"
REFERENCE = (ROOT / "results" / "week3" / "color_reference"
             / "museum_hue_references.npz")
OUTPUT = ROOT / "results" / "week3" / "hue_normalization"


def _subsets(augmentations):
    return {
        "all": list(range(len(augmentations))),
        "hue": [i for i, label in enumerate(augmentations)
                if str(label) == "None-MultiplyHue"],
        "clean": [i for i, label in enumerate(augmentations)
                  if str(label) == "None"],
        "impulse": [i for i, label in enumerate(augmentations)
                    if str(label) == "UnnamedImpulseNoise"],
    }


def _rank_of(ranking, target):
    try:
        return ranking.index(target) + 1
    except ValueError:
        return len(ranking) + 1


def _global_reference_ranking(query_rgb, museum_descriptors, reference,
                             descriptor_fn, scales, saturation_threshold,
                             regularization):
    query_hist = hue_histogram(
        query_rgb, saturation_threshold=saturation_threshold)
    scale, score = estimate_hue_scale(
        query_hist, reference.global_histogram(), scales=scales,
        regularization=regularization)
    corrected = correct_hue_scale(
        query_rgb, scale, saturation_threshold=saturation_threshold)
    descriptor = np.asarray(descriptor_fn(corrected), dtype=np.float64)
    distances = np.abs(museum_descriptors - descriptor).sum(axis=1)
    ranking = [int(index) for index in np.argsort(distances, kind="stable")]
    return ranking, float(scale), float(score)


def _save_reference_artifacts(reference, reference_path, museum_dir,
                              output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "n_references": len(reference),
        "bins": reference.bins,
        "saturation_threshold": reference.saturation_threshold,
        "museum_dir": str(museum_dir),
        "reference_file": str(reference_path),
    }
    (output_dir / "reference_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(9, 3.5))
    bins = np.arange(reference.bins)
    axis.plot(bins, reference.global_histogram(), color="#1f77b4", lw=1.5)
    axis.fill_between(
        bins, reference.global_histogram(), color="#1f77b4", alpha=0.18)
    axis.set_xlabel("Hue (OpenCV 0-179)")
    axis.set_ylabel("Weighted probability")
    axis.set_title("Average hue reference built from BBDD")
    axis.set_xlim(0, reference.bins - 1)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "reference_hue_histogram.png", dpi=150)
    plt.close(fig)


def _summary_rows(rankings, ground_truth, museum_ids, augmentations, top_k):
    rows = []
    for subset, indices in _subsets(augmentations).items():
        gt_subset = [ground_truth[i] for i in indices]
        for method, all_rankings in rankings.items():
            ranked_ids = [
                [museum_ids[index] for index in all_rankings[i]]
                for i in indices
            ]
            rows.append({
                "subset": subset,
                "n_queries": len(indices),
                "method": method,
                "mAP@1": mean_average_precision_at_k(
                    ranked_ids, gt_subset, 1),
                f"mAP@{top_k}": mean_average_precision_at_k(
                    ranked_ids, gt_subset, top_k),
            })
    return rows


def _write_summary(rows, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_per_query(rows, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(output, reference_path, museum_dir, parameters, summary):
    lines = [
        "Normalizacion de tono en HSV con retrieval en LAB",
        "=" * 49,
        "",
        "Referencias: " + str(reference_path),
        "Codigo de referencias: src/color_reference/",
        f"Museo: {museum_dir}",
        "",
        "Metodo",
        "------",
        "1. Se construye un histograma de tono por imagen de BBDD, ponderado",
        "   por saturacion y ignorando pixeles con S <= umbral.",
        "2. Para cada candidato del museo se estima un factor global de escala",
        "   de tono que alinea el histograma de la query con su referencia.",
        "3. La correccion se aplica solo al canal H de la query (HSV), dejando",
        "   S y V intactos, y se vuelve a RGB.",
        "4. El descriptor sigue siendo 'Lab blocks 1D' y la distancia L1; el",
        "   ranking final mezcla la distancia LAB original y la normalizada,",
        "   ponderando cada candidato por la confianza del alineamiento de tono.",
        "",
        "No se usa la etiqueta de augmentacion en ningun paso.",
        "",
        "Parametros",
        "----------",
    ]
    for key, value in parameters.items():
        lines.append(f"  {key}: {value}")
    lines += [
        "",
        "Resultados",
        "----------",
    ]
    top_k_column = f"mAP@{parameters['top_k']}"
    for row in summary:
        lines.append(
            f"  {row['subset']:<7} {row['method']:<16} "
            f"mAP@1={row['mAP@1']:.4f} "
            f"{top_k_column}={row[top_k_column]:.4f}"
        )
    lines += [
        "",
        "Notas",
        "-----",
        "- La referencia global (media de BBDD) se evalua como control; no usa",
        "  una referencia distinta por candidato.",
        "- La mejora se concentra en las queries None-MultiplyHue; las queries",
        "  con ruido impulsivo no se corrigen con este metodo.",
        "- El ranking de cada query y el factor estimado para su GT estan en",
        "  per_query.csv.",
    ]
    (output / "README.txt").write_text("\n".join(lines) + "\n",
                                       encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--museum", type=Path, default=MUSEUM)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH)
    parser.add_argument("--augmentations", type=Path, default=AUGMENTATIONS)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rebuild-reference", action="store_true")
    parser.add_argument("--blocks", type=int, default=25)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sat-threshold", type=int,
                        default=DEFAULT_SATURATION_THRESHOLD)
    parser.add_argument("--scale-min", type=float,
                        default=DEFAULT_SCALE_MIN)
    parser.add_argument("--scale-max", type=float,
                        default=DEFAULT_SCALE_MAX)
    parser.add_argument("--scale-step", type=float,
                        default=DEFAULT_SCALE_STEP)
    parser.add_argument("--descriptor-step", type=float, default=0.01)
    parser.add_argument("--regularization", type=float,
                        default=DEFAULT_REGULARIZATION)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args(argv)

    start = time.time()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_path = args.reference.resolve()
    reference_path.parent.mkdir(parents=True, exist_ok=True)

    museum_ids, museum_paths = _ordered_image_paths(args.museum)
    query_ids, query_paths = _ordered_image_paths(args.queries)
    ground_truth = _load_ground_truth(args.ground_truth, query_ids)
    with args.augmentations.open("rb") as handle:
        augmentations = pickle.load(handle)
    if len(augmentations) != len(query_paths):
        raise ValueError("augmentation count does not match query count")

    print("Loading museum images...", flush=True)
    museum_images = [_read_rgb(path) for path in museum_paths]
    descriptor_fn = lambda rgb: baseline_descriptor(
        rgb, blocks=args.blocks, bins=args.bins)
    museum_descriptors = np.asarray([
        descriptor_fn(image) for image in museum_images
    ])
    print(f"Museum descriptors: {museum_descriptors.shape} "
          f"({time.time() - start:.1f}s)", flush=True)

    if args.rebuild_reference or not reference_path.exists():
        print("Building hue references...", flush=True)
        reference = build_hue_references_from_images(
            museum_ids, museum_images,
            saturation_threshold=args.sat_threshold)
        reference.save(reference_path)
        print(f"Reference cache: {reference_path}", flush=True)
    else:
        reference = HueReference.load(reference_path)
        print(f"Loaded reference cache: {reference_path}", flush=True)
    reference = reference.reorder(museum_ids)

    scales = hue_scale_grid(
        args.scale_min, args.scale_max, args.scale_step)
    rankings = {
        "baseline": [],
        "global_reference": [],
        "hue_reference": [],
    }
    per_query = []
    for qi, path in enumerate(query_paths):
        query_rgb = _read_rgb(path)
        result = rank_with_hue_references(
            query_rgb, museum_descriptors, reference, descriptor_fn,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            scale_step=args.scale_step,
            descriptor_step=args.descriptor_step,
            saturation_threshold=args.sat_threshold,
            regularization=args.regularization,
            alpha=args.alpha,
            temperature=args.temperature,
        )
        global_ranking, global_scale, global_score = (
            _global_reference_ranking(
                query_rgb, museum_descriptors, reference, descriptor_fn,
                scales, args.sat_threshold, args.regularization)
        )
        rankings["baseline"].append(result.baseline_ranking)
        rankings["global_reference"].append(global_ranking)
        rankings["hue_reference"].append(result.ranking)

        gt = ground_truth[qi]
        gt_id = gt[0]
        baseline_rank = result.baseline_ranking
        color_rank = result.ranking
        baseline_ids = [museum_ids[index] for index in baseline_rank]
        color_ids = [museum_ids[index] for index in color_rank]
        global_ids = [museum_ids[index] for index in global_ranking]
        per_query.append({
            "query_id": query_ids[qi],
            "augmentation": str(augmentations[qi]),
            "gt_id": gt_id,
            "baseline_top1": baseline_ids[0],
            "color_top1": color_ids[0],
            "baseline_gt_rank": _rank_of(baseline_rank, gt_id),
            "color_gt_rank": _rank_of(color_rank, gt_id),
            "global_gt_rank": _rank_of(global_ranking, gt_id),
            "scale_at_gt": float(result.scales[gt_id]),
            "global_scale": global_scale,
            "baseline_ap@1": average_precision_at_k(
                baseline_ids, gt, 1),
            "color_ap@1": average_precision_at_k(color_ids, gt, 1),
            "global_ap@1": average_precision_at_k(global_ids, gt, 1),
            "baseline_ap@5": average_precision_at_k(
                baseline_ids, gt, args.top_k),
            "color_ap@5": average_precision_at_k(
                color_ids, gt, args.top_k),
            "global_ap@5": average_precision_at_k(
                global_ids, gt, args.top_k),
            "top1_changed": int(baseline_ids[0] != color_ids[0]),
        })
        print(f"  {query_ids[qi]:05d} {str(augmentations[qi]):18s} "
              f"baseline={baseline_ids[0]} color={color_ids[0]} "
              f"rank {baseline_rank.index(gt_id) + 1} -> "
              f"{color_rank.index(gt_id) + 1}", flush=True)

    summary = _summary_rows(
        rankings, ground_truth, museum_ids, augmentations, args.top_k)
    _write_summary(summary, output / "summary.csv")
    _write_per_query(per_query, output / "per_query.csv")

    _save_reference_artifacts(
        reference, reference_path, args.museum.resolve(),
        reference_path.parent)
    parameters = {
        "top_k": args.top_k,
        "sat_threshold": args.sat_threshold,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "scale_step": args.scale_step,
        "descriptor_step": args.descriptor_step,
        "regularization": args.regularization,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "blocks": args.blocks,
        "bins": args.bins,
    }
    _write_readme(
        output, reference_path, args.museum.resolve(), parameters, summary)

    print("\nSummary")
    for row in summary:
        print(f"{row['subset']:<7} {row['method']:<16} "
              f"mAP@1={row['mAP@1']:.4f} "
              f"mAP@{args.top_k}={row[f'mAP@{args.top_k}']:.4f}")
    print(f"\nDone in {time.time() - start:.1f}s -> {output}")


if __name__ == "__main__":
    main()
