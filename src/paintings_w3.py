"""Week 3, Tasks 3 and 4: background removal and retrieval of up to two paintings.

The wall model, mask metrics and visualizations come from Week 2. Each
foreground component is cropped separately before computing its descriptor.
Paintings are returned from left to right, or top to bottom when stacked.

    python src/paintings_w3.py --segment data/qsd2_w3 --out results/week3/QSD2/masks
    python src/paintings_w3.py --museum data/BBDD --queries data/qsd2_w3 \
        --background-removal --ground-truth data/qsd2_w3/gt_corresps.pkl

The default descriptor is the W2 Lab pyramid baseline (7 levels). Task 2
descriptors can be passed as RGB callables, without changing this pipeline.
"""

import argparse
import importlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from methods_w2 import (
    BGParams,
    _load_ground_truth,
    _ordered_image_paths,
    _read_rgb,
    _save_overlays,
    background_distance,
    build_method,
    evaluate_segmentation,
    fit_background_model,
    overlay,
)
from utils import compute_segmentation_metrics


@dataclass
class PaintingParams(BGParams):
    border: float = 0.02
    max_paintings: int = 2


def _reading_order(boxes):
    """Boxes are (x0, y0, x1, y1), with exclusive right/bottom coordinates."""
    if len(boxes) < 2:
        return boxes
    a, b = boxes
    overlap_y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    same_row = overlap_y >= 0.5 * min(a[3] - a[1], b[3] - b[1])
    return sorted(boxes, key=lambda box: (box[0], box[1]) if same_row
                  else (box[1], box[0]))


def _component_boxes(mask, min_area, max_paintings):
    """Keep large components; merge overlapping boxes from the same painting."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area >= min_area:
            candidates.append((area, (x, y, x + w, y + h)))
    candidates.sort(reverse=True)
    boxes = []
    for _, box in candidates:
        index = 0
        while index < len(boxes):
            other = boxes[index]
            if (max(box[0], other[0]) < min(box[2], other[2])
                    and max(box[1], other[1]) < min(box[3], other[3])):
                box = (min(box[0], other[0]), min(box[1], other[1]),
                       max(box[2], other[2]), max(box[3], other[3]))
                boxes.pop(index)
                index = 0
            else:
                index += 1
        boxes.append(box)
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return _reading_order(boxes[:max_paintings])


def segment_paintings(img_rgb, params=None):
    """Return the union mask (0/255, original size) and individual crop boxes.

    Unlike W2, connected components are kept separately. The two largest
    candidates above the area threshold are retained, so the wall between
    two paintings is not included in a single bounding box.
    """
    p = params or PaintingParams()
    if p.max_paintings not in (1, 2):
        raise ValueError("max_paintings must be 1 or 2")
    if img_rgb.dtype != np.uint8 or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image")
    H, W = img_rgb.shape[:2]
    scale = min(1.0, p.max_side / max(H, W))
    rgb = (cv2.resize(img_rgb, (max(1, round(W * scale)), max(1, round(H * scale))),
                      interpolation=cv2.INTER_AREA) if scale < 1 else img_rgb)
    h, w = rgb.shape[:2]
    smooth = cv2.GaussianBlur(rgb, (p.blur, p.blur), 0) if p.blur > 1 else rgb
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    model = fit_background_model(lab, p.border, p.poly_degree)
    dist = background_distance(lab, model, p.poly_degree)
    d8 = np.clip(dist * (255.0 / 20.0), 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(np.clip(otsu * 20.0 / 255.0, p.thr_min, p.thr_max))
    mask = (dist > threshold).astype(np.uint8) * 255
    kernel = lambda fraction: cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, int(fraction * min(h, w))) | 1,) * 2)
    # Close first to connect thin frame edges before removing small speckles.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(p.close_frac))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(p.open_frac))
    boxes = _component_boxes(mask, p.min_area_frac * h * w, p.max_paintings)
    boxes = [(int(np.floor(x0 * W / w)), int(np.floor(y0 * H / h)),
              min(W, int(np.ceil(x1 * W / w))), min(H, int(np.ceil(y1 * H / h))))
             for x0, y0, x1, y1 in boxes]
    union = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in boxes:
        union[y0:y1, x0:x1] = 255
    return union, boxes


def segment_folder_w3(folder, out_dir, viz_dir=None, params=None):
    """Save masks, crop boxes and mask metrics; optionally save W2 overlays."""
    ids, paths = _ordered_image_paths(folder)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows, tiles, labels = [], [], []
    for qid, path in zip(ids, paths):
        rgb = _read_rgb(path)
        mask, boxes = segment_paintings(rgb, params)
        if not cv2.imwrite(str(destination / (path.stem + ".png")), mask):
            raise OSError(f"Cannot write mask for {path}")
        row = {"id": qid, "boxes": boxes, "paintings": len(boxes)}
        gt_path = path.with_suffix(".png")
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else None
        label = f"{path.stem}  n={len(boxes)}"
        if gt is not None:
            row.update(compute_segmentation_metrics(mask, gt))
            label += f"  F1={row['F1']:.2f}"
        rows.append(row)
        if viz_dir is not None:
            tiles.append(overlay(rgb, mask, gt))
            labels.append(label)
    report = {"parameters": asdict(params or PaintingParams()), "per_image": rows}
    scored = [row for row in rows if "F1" in row]
    if scored:
        report["mean"] = {key: float(np.mean([row[key] for row in scored]))
                          for key in ("precision", "recall", "F1")}
        print("Mask means: " + " ".join(f"{key}={value:.4f}"
                                        for key, value in report["mean"].items()))
    (destination / "segmentation.json").write_text(json.dumps(report, indent=2) + "\n")
    if viz_dir is not None:
        _save_overlays(viz_dir, [path.stem for path in paths], tiles, labels)
    print(f"{len(ids)} images segmented, masks -> {destination}")
    return report


def evaluate_painting_rankings(rankings, ground_truth, k):
    """AP per painting in reading order; missing/extra detections score zero.

    W3 ground truth stores one or two museum IDs per query. Each ID is one
    painting, rather than two relevant results for a single crop. The mean
    includes max(predicted count, true count) slots per image, so an extra
    detection cannot improve the score.
    """
    from k_similarity import average_precision_at_k

    if len(rankings) != len(ground_truth):
        raise ValueError("Number of rankings must match number of queries")
    scores, correct_counts = [], 0
    for predicted, actual in zip(rankings, ground_truth):
        if not isinstance(actual, (list, tuple)) or not 1 <= len(actual) <= 2:
            raise ValueError("Expected one or two ground-truth paintings per query")
        targets = [[int(item)] if np.isscalar(item) else list(item) for item in actual]
        correct_counts += len(predicted) == len(targets)
        for index in range(max(len(predicted), len(targets))):
            score = (average_precision_at_k(predicted[index], targets[index], k)
                     if index < min(len(predicted), len(targets)) else 0.0)
            scores.append(score)
    return {f"mAP@{k}": float(np.mean(scores)) if scores else 0.0,
            "count_accuracy": correct_counts / len(rankings) if rankings else 0.0}


def _rgb_lab_descriptor(method="Lab pyramid", blocks=36, levels=7):
    """Adapt the existing W2 Lab descriptors to the RGB callable interface."""
    descriptor = build_method(method, blocks=blocks, levels=levels)
    return lambda rgb: descriptor(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB))


def _descriptor_vector(descriptor, rgb):
    vector = np.asarray(descriptor(rgb), dtype=np.float32)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("Descriptors must return a nonempty, finite 1D vector")
    return vector


def retrieve_folder_w3(museum_path, queries_path, out_dir, descriptor=None,
                       distance_name="L1", top_k=10, ground_truth_path=None,
                       background_removal=True, params=None, query_filter=None,
                       viz_dir=None, descriptor_name="W2 Lab pyramid, 7 levels"):
    """Rank each crop independently and save query -> painting -> museum IDs.

    descriptor(rgb) is shared by museum images and query crops. query_filter
    is optional and applies only to query crops, reusing Task 1 if requested.
    An empty segmentation falls back to the whole image for retrieval; the
    saved mask remains empty and the fallback is recorded in the report.
    """
    from k_similarity import DISTANCE_FUNCTIONS, rank_museum_images

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    museum_ids, museum_paths = _ordered_image_paths(museum_path)
    if len(museum_ids) < top_k:
        raise ValueError(f"Need at least {top_k} museum images")
    query_ids, query_paths = _ordered_image_paths(queries_path)
    ground_truth = _load_ground_truth(ground_truth_path, query_ids)
    descriptor = descriptor or _rgb_lab_descriptor()
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    segmentation = (segment_folder_w3(queries_path, destination, viz_dir, params)
                    if background_removal else None)
    museum = [_descriptor_vector(descriptor, _read_rgb(path)) for path in museum_paths]
    dimension = museum[0].size
    if any(vector.size != dimension for vector in museum):
        raise ValueError("All museum descriptors must have the same dimension")
    rankings, rows = [], []
    for index, (qid, path) in enumerate(zip(query_ids, query_paths)):
        rgb = _read_rgb(path)
        h, w = rgb.shape[:2]
        boxes = (segmentation["per_image"][index]["boxes"] if segmentation
                 else [(0, 0, w, h)])
        fallback = not boxes
        crop_boxes = boxes or [(0, 0, w, h)]
        painting_rankings = []
        for x0, y0, x1, y1 in crop_boxes:
            crop = rgb[y0:y1, x0:x1]
            if query_filter is not None:
                crop = query_filter(crop)
            vector = _descriptor_vector(descriptor, crop)
            if vector.size != dimension:
                raise ValueError(f"Query {qid}: descriptor dimension differs from museum")
            ranked = rank_museum_images(vector, museum, DISTANCE_FUNCTIONS[distance_name], top_k)
            painting_rankings.append([int(museum_ids[pos]) for pos, _ in ranked])
        rankings.append(painting_rankings)
        rows.append({"id": qid, "boxes": boxes, "fallback": fallback,
                     "rankings": painting_rankings})
    report = {"descriptor": descriptor_name, "distance": distance_name,
              "top_k": top_k, "dimension": dimension,
              "background_removal": background_removal,
              "evaluation": "AP per painting in reading order; unmatched slots score zero",
              "query_filter": getattr(query_filter, "__name__", None),
              "museum_count": len(museum_ids), "per_image": rows}
    if ground_truth is not None:
        metrics = {}
        for k in sorted({1, min(5, top_k), top_k}):
            metrics.update(evaluate_painting_rankings(rankings, ground_truth, k))
        if segmentation:
            metrics["segmentation_count_accuracy"] = float(np.mean([
                len(row["boxes"]) == len(actual)
                for row, actual in zip(rows, ground_truth)]))
        report["metrics"] = metrics
        print(" ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    with (destination / "result.pkl").open("wb") as stream:
        pickle.dump(rankings, stream)
    with (destination / "result.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    valid_ids = set(museum_ids)
    for paintings in saved:
        if not 1 <= len(paintings) <= 2:
            raise ValueError("Each query must contain one or two painting rankings")
        for ranking in paintings:
            if (len(ranking) != top_k or len(set(ranking)) != top_k
                    or any(type(item) is not int or item not in valid_ids for item in ranking)):
                raise ValueError("Invalid ranked museum IDs")
    (destination / "retrieval.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(query_ids)} queries retrieved, results -> {destination}")
    return rankings, report


def _import_callable(value):
    """Load a local RGB descriptor or query filter as module:function."""
    module_name, separator, function_name = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("Use module:function")
    try:
        function = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not callable(function):
        raise argparse.ArgumentTypeError(f"{value} is not callable")
    return function


def build_parser():
    from k_similarity import DISTANCE_FUNCTIONS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=Path, help="Query folder for Task 3 only")
    parser.add_argument("--out", type=Path, default=Path("results/week3/QSD2/masks"))
    parser.add_argument("--viz", type=Path, help="Save W2 overlays and overview.jpg")
    parser.add_argument("--segmentation", nargs=2, type=Path, metavar=("PRED", "GT"))
    parser.add_argument("--save", type=Path, help="Save segmentation metrics as text")
    parser.add_argument("--museum", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--background-removal", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/week3/QSD2/lab_pyramid_baseline"))
    descriptor_group = parser.add_mutually_exclusive_group()
    descriptor_group.add_argument("--method", choices=("Lab blocks 1D", "Lab 2D", "Lab 3D", "Lab pyramid"))
    descriptor_group.add_argument("--descriptor", type=_import_callable,
                                  help="Task 2 RGB descriptor, module:function")
    parser.add_argument("--query-filter", type=_import_callable,
                        help="Task 1 RGB filter, e.g. methods_w3:median_filter")
    parser.add_argument("--blocks", type=int, default=36)
    parser.add_argument("--levels", type=int, default=7)
    parser.add_argument("--distance", choices=sorted(DISTANCE_FUNCTIONS), default="L1")
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main(args=None):
    parser = build_parser()
    opts = parser.parse_args(args)
    if opts.segment:
        segment_folder_w3(opts.segment, opts.out, opts.viz)
        return
    if opts.segmentation:
        evaluate_segmentation(*opts.segmentation, save_path=opts.save)
        return
    if not opts.museum or not opts.queries:
        parser.error("Use --segment, --segmentation, or --museum and --queries")
    method = opts.method or "Lab pyramid"
    descriptor = opts.descriptor or _rgb_lab_descriptor(method, opts.blocks, opts.levels)
    name = (f"{opts.descriptor.__module__}:{opts.descriptor.__name__}" if opts.descriptor
            else f"W2 {method}, blocks={opts.blocks}, levels={opts.levels}")
    retrieve_folder_w3(
        opts.museum, opts.queries, opts.output_dir, descriptor=descriptor,
        distance_name=opts.distance, top_k=opts.top_k,
        ground_truth_path=opts.ground_truth, background_removal=opts.background_removal,
        query_filter=opts.query_filter, viz_dir=opts.viz, descriptor_name=name)


if __name__ == "__main__":
    main()
