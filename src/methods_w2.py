"""Methods_W2: Lab color descriptors to test in Week 2 (Task 1).

Run as a script (argparse selects the action):
    python methods_w2.py --list
    python methods_w2.py --museum <dir> --queries <dir> [--method NAME]
    python methods_w2.py --museum <dir> --queries <dir> --ground-truth <pkl>
    python methods_w2.py --segmentation <pred_dir> <gt_dir> [--save FILE]

Functions take uint8 OpenCV Lab images (all channels span [0, 256)) and
accept an optional foreground mask (Task 3/5): if ``mask`` is None the full
image is used; if given, only foreground pixels contribute to the histogram.

One rule everywhere in W2: a single uniform bin count shared by every
channel (no per-channel bins, no Max variants). Dims are therefore ``bins``
(marginal 1D per channel x3), ``bins**2`` (2D), ``bins**3`` (3D), times the
number of cells for blocks/pyramids.

METHODS_W2 registry (all L1), validated on QSD1-W2 with L1 only
(see results/week2/qsd1w2_registry_l1.txt), in three families:

Marginal 1D (three per-channel histograms of `bins`, concatenated):
    - "Lab 1D 32" global (96, mAP@5 0.5861): 1 block.
    - "Lab block 2x2 1D 32" (384, mAP@5 0.7106): 4 blocks.
    - "Lab block 2x4 1D 32" (768, mAP@5 0.8278): 8 blocks.
    - "Lab block 4x4 1D 32" (1536, mAP@5 0.8722): 16 blocks.
    - "Lab block 6x6 1D 32" (3456).
    - "Lab pyramid 1D 32" (2016).

Joint/mixed (2D and 3D histograms, global or per cell):
    - Globals: "Lab 3D 4" (64, 0.4733), "Lab 2D La 32" (1024, 0.4806),
      "Lab 2D Lb 8" (64, 0.4428), "Lab 2D ab 16" (256, 0.5150),
      "Lab 2D La+Lb 16" (512, 0.5167).
    - Partitioned (1D marginal + 2D joint, equal weight per part):
      "Lab L+ab 16" (272, mAP@5 0.5111) = L 16 + ab 16x16, global;
      "Lab block 4x4 L+ab 16" (4352, mAP@5 0.7583), the same split in
      16 blocks; best Lab joint result after the marginal family.
    - Pyramids (levels (1, 2, 4) = 21 cells; each cell carries the same
      combination, L1-normalized per cell and L1-renormalized on
      concatenation, no level weights): "Lab pyramid 3D 4" (1344, 0.7417),
      "Lab pyramid La 32" (21504, 0.6678), "Lab pyramid Lb 8" (1344, 0.6733),
      "Lab pyramid ab 16" (5376, 0.6733),
      "Lab pyramid La+Lb 16" (10752, 0.6444).

Channel pairs keep the Lab indices, i.e. (L,a) and (L,b), plus the
chromaticity-only pair (a,b) at 16 bins (remote sweep: ab-16 pyramid 0.6733
vs ab-32 0.7011 at 4x the dims, so 16 is the registered tier).
These are QSD1 development results with L1 distance, not blind test scores.

Masked variants (QSD2/QST2) reuse these same functions by passing
``mask``; they are not separate registry entries until the Task 1
ablation is closed on QSD1-W2 (same dev set as W1, direct comparison).
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from utils import _l1_normalize, compute_segmentation_metrics, image_id


def _check_bins(bins):
    """Uniform bin count shared by every channel (no per-channel bins)."""
    assert isinstance(bins, int) and not isinstance(bins, bool) and bins > 0, (
        f"bins must be a positive integer, got {bins!r}")
    return bins


def _split_with_mask(image, mask, grid_y, grid_x):
    """Yield (block, block_mask) pairs; block_mask is None if mask is None."""
    height, width = image.shape[:2]
    for y in range(grid_y):
        for x in range(grid_x):
            y0, y1 = y * height // grid_y, (y + 1) * height // grid_y
            x0, x1 = x * width // grid_x, (x + 1) * width // grid_x
            m = None if mask is None else mask[y0:y1, x0:x1]
            yield image[y0:y1, x0:x1], m


def _grid_size(grid):
    """Normalize a grid to (grid_y, grid_x); int means square."""
    if isinstance(grid, int):
        assert grid > 0, f"grid must be positive, got {grid!r}"
        return (grid, grid)
    assert (isinstance(grid, (tuple, list)) and len(grid) == 2
            and all(isinstance(g, int) and not isinstance(g, bool) and g > 0
                    for g in grid)), f"grid must be int or (gy, gx), got {grid!r}"
    return (grid[0], grid[1])


def _compute_spatial_histogram(image, levels, block_fn, mask):
    parts = [
        _l1_normalize(block_fn(block, m))
        for grid in levels
        for block, m in _split_with_mask(image, mask, *_grid_size(grid))
    ]
    return _l1_normalize(np.concatenate(parts))


def compute_lab_histogram(img_lab, bins=32, mask=None):
    """Marginal L, a and b histograms for uint8 OpenCV Lab, L1 normalized."""
    assert len(img_lab.shape) == 3, "Expected Lab image"
    _check_bins(bins)
    parts = [
        cv2.calcHist([img_lab], [ch], mask, [bins], [0, 256]).flatten()
        for ch in range(3)
    ]
    return _l1_normalize(np.concatenate(parts))


# OpenCV uint8 Lab ranges (L, a, b all span [0, 256)).
_LAB_RANGES = {0: (0, 256), 1: (0, 256), 2: (0, 256)}


def _lab_ranges_for(channels):
    ranges = []
    for ch in channels:
        lo, hi = _LAB_RANGES[ch]
        ranges.extend([lo, hi])
    return ranges


def compute_lab_3d_histogram(img_lab, bins=32, mask=None):
    """Joint 3D L-a-b histogram with `bins` per axis, L1 normalized."""
    assert len(img_lab.shape) == 3, "Expected Lab image"
    _check_bins(bins)
    hist = cv2.calcHist(
        [img_lab], [0, 1, 2], mask, [bins, bins, bins],
        [0, 256, 0, 256, 0, 256],
    ).flatten()
    return _l1_normalize(hist)


def compute_lab_2d_histogram(img_lab, channels=(0, 1), bins=32, mask=None):
    """Joint 2D Lab histogram (L-a by default) with `bins` per axis, L1."""
    assert len(img_lab.shape) == 3, "Expected Lab image"
    assert len(channels) == 2
    _check_bins(bins)
    hist = cv2.calcHist(
        [img_lab], list(channels), mask, [bins, bins],
        _lab_ranges_for(channels),
    ).flatten()
    return _l1_normalize(hist)


def compute_lab_2d_concat_histogram(img_lab, parts=(((0, 1), 32),), mask=None):
    """Concatenate several Lab 2D histograms (e.g. La+Lb), L1 normalized.

    Equal-weight-per-pair rule, with the same uniform-bins rule
    (no per-channel bins anywhere in W2).
    """
    assert len(img_lab.shape) == 3, "Expected Lab image"
    assert len(parts) >= 1
    for _, b in parts:
        _check_bins(b)
    normed_parts = [
        _l1_normalize(compute_lab_2d_histogram(
            img_lab, channels=ch, bins=b, mask=mask
        ))
        for ch, b in parts
    ]
    return _l1_normalize(np.concatenate(normed_parts))


def compute_lab_1d_histogram(img_lab, channel=0, bins=16, mask=None):
    """Single-channel Lab marginal (L by default), L1 normalized."""
    assert len(img_lab.shape) == 3, "Expected Lab image"
    assert channel in (0, 1, 2), f"channel must be 0, 1 or 2, got {channel!r}"
    _check_bins(bins)
    hist = cv2.calcHist([img_lab], [channel], mask, [bins], [0, 256]).flatten()
    return _l1_normalize(hist)


def compute_lab_mixed_concat_histogram(img_lab, parts=(((0,), 32), ((1, 2), 32)),
                                       mask=None):
    """Concatenate 1D marginal and 2D joint Lab parts (e.g. L+ab), L1.

    parts: sequence of (channels, bins) where channels has 1 entry
    (marginal) or 2 entries (joint), e.g. [((0,), 16), ((1, 2), 16)]
    for L + ab. Same uniform-bins rule and equal-weight-per-part
    L1 normalization as the other concats.
    """
    assert len(img_lab.shape) == 3, "Expected Lab image"
    assert len(parts) >= 1
    normed_parts = []
    for channels, b in parts:
        _check_bins(b)
        assert len(channels) in (1, 2), f"channels must have 1 or 2 entries, got {channels!r}"
        if len(channels) == 1:
            hist = compute_lab_1d_histogram(
                img_lab, channel=channels[0], bins=b, mask=mask)
        else:
            hist = compute_lab_2d_histogram(
                img_lab, channels=channels, bins=b, mask=mask)
        normed_parts.append(_l1_normalize(hist))
    return _l1_normalize(np.concatenate(normed_parts))


def _lab_concat_fn(*parts):
    """Build fn(lab, mask) for a Lab 2D concat with equal weight per pair."""
    spec = list(parts)
    return lambda lab, mask=None: compute_lab_2d_concat_histogram(
        lab, parts=spec, mask=mask
    )


def _lab_mixed_concat_fn(*parts):
    """Build fn(lab, mask) for a mixed Lab concat."""
    spec = list(parts)
    return lambda lab, mask=None: compute_lab_mixed_concat_histogram(
        lab, parts=spec, mask=mask
    )


def compute_lab_pyramid_histogram(img_lab, levels=(1, 2, 4),
                                  bins=32, block_fn=None, mask=None):
    """Lab histograms per cell, L1-normalized per cell and renormalized
    on concatenation.

    block_fn(block_lab, block_mask) -> 1D vector; defaults to the marginal
    1D histogram with `bins` per channel.
    """
    _check_bins(bins)
    if block_fn is None:
        block_fn = lambda b, m: compute_lab_histogram(b, bins=bins, mask=m)
    return _compute_spatial_histogram(img_lab, levels, block_fn, mask)


def compute_lab_block_histogram(img_lab, grid=4, bins=32, mask=None):
    """Grid of Lab marginals, concatenated in row-major order.

    grid: int for a square grid (4 -> 4x4 = 16 blocks) or (gy, gx)
    tuple for a rectangular one ((2, 4) -> 8 blocks).
    """
    _grid_size(grid)
    return compute_lab_pyramid_histogram(
        img_lab, levels=(grid,), bins=bins, mask=mask
    )


METHODS_W2 = {
    "Lab block 4x4 1D 32": lambda lab, mask=None: compute_lab_block_histogram(
        lab, grid=4, mask=mask
    ),
    "Lab block 6x6 1D 32": lambda lab, mask=None: compute_lab_block_histogram(
        lab, grid=6, mask=mask
    ),
    "Lab block 2x2 1D 32": lambda lab, mask=None: compute_lab_block_histogram(
        lab, grid=2, mask=mask
    ),
    "Lab block 2x4 1D 32": lambda lab, mask=None: compute_lab_block_histogram(
        lab, grid=(2, 4), mask=mask
    ),
    "Lab pyramid 1D 32": lambda lab, mask=None: compute_lab_pyramid_histogram(
        lab, mask=mask
    ),
    "Lab 1D 32": lambda lab, mask=None: compute_lab_histogram(
        lab, bins=32, mask=mask
    ),
    "Lab 3D 4": lambda lab, mask=None: compute_lab_3d_histogram(
        lab, bins=4, mask=mask
    ),
    "Lab 2D La 32": lambda lab, mask=None: compute_lab_2d_histogram(
        lab, channels=(0, 1), bins=32, mask=mask
    ),
    "Lab 2D Lb 8": lambda lab, mask=None: compute_lab_2d_histogram(
        lab, channels=(0, 2), bins=8, mask=mask
    ),
    "Lab 2D ab 16": lambda lab, mask=None: compute_lab_2d_histogram(
        lab, channels=(1, 2), bins=16, mask=mask
    ),
    "Lab 2D La+Lb 16": _lab_concat_fn(((0, 1), 16), ((0, 2), 16)),
    "Lab L+ab 16": _lab_mixed_concat_fn(((0,), 16), ((1, 2), 16)),
    "Lab pyramid 3D 4": lambda lab, mask=None: compute_lab_pyramid_histogram(
        lab, levels=(1, 2, 4),
        block_fn=lambda b, m: compute_lab_3d_histogram(b, bins=4, mask=m),
        mask=mask,
    ),
    "Lab pyramid La 32": lambda lab, mask=None: compute_lab_pyramid_histogram(
        lab, levels=(1, 2, 4),
        block_fn=lambda b, m: compute_lab_2d_histogram(
            b, channels=(0, 1), bins=32, mask=m),
        mask=mask,
    ),
    "Lab pyramid Lb 8": lambda lab, mask=None: compute_lab_pyramid_histogram(
        lab, levels=(1, 2, 4),
        block_fn=lambda b, m: compute_lab_2d_histogram(
            b, channels=(0, 2), bins=8, mask=m),
        mask=mask,
    ),
    "Lab pyramid ab 16": lambda lab, mask=None: compute_lab_pyramid_histogram(
        lab, levels=(1, 2, 4),
        block_fn=lambda b, m: compute_lab_2d_histogram(
            b, channels=(1, 2), bins=16, mask=m),
        mask=mask,
    ),
    "Lab pyramid La+Lb 16": lambda lab, mask=None: compute_lab_pyramid_histogram(
        lab, levels=(1, 2, 4),
        block_fn=lambda b, m: compute_lab_2d_concat_histogram(
            b, parts=[((0, 1), 16), ((0, 2), 16)], mask=m),
        mask=mask,
    ),
    "Lab block 4x4 L+ab 16": lambda lab, mask=None: compute_lab_pyramid_histogram(
        lab, levels=(4,),
        block_fn=lambda b, m: compute_lab_mixed_concat_histogram(
            b, parts=[((0,), 16), ((1, 2), 16)], mask=m),
        mask=mask,
    ),
}

METHODS_W2_NAMES = list(METHODS_W2.keys())
METHODS_W2_COLOR_SPACES = dict.fromkeys(METHODS_W2, "Lab")


def compute_descriptors_w2(dataset, masks=None):
    """W2 descriptors for a dataset; masks optional (QSD2 foreground).

    dataset.images in RGB (like Dataset); masks list of HxW uint8 or None.
    Returns one list per W2 method, in METHODS_W2_NAMES order.
    """
    methods = [[] for _ in METHODS_W2_NAMES]
    n = len(dataset.images)
    if masks is not None and len(masks) != n:
        raise ValueError(f"Expected {n} masks, got {len(masks)}")
    for i, img in enumerate(dataset.images):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        mask = None if masks is None else masks[i]
        for j, name in enumerate(METHODS_W2_NAMES):
            methods[j].append(METHODS_W2[name](lab, mask=mask))
    return methods


def _ordered_image_paths(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Expected an image folder: {folder}")
    paths = sorted(
        (p for p in folder.iterdir() if p.is_file()
         and p.suffix.lower() in (".jpg", ".jpeg")),
        key=lambda p: (image_id(p), p.name),
    )
    ids = [image_id(p) for p in paths]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Expected nonempty, unique image IDs in {folder}")
    return ids, paths


def _ordered_masks(folder, ids):
    """Load grayscale PNG masks aligned to image IDs (nonzero=foreground)."""
    if folder is None:
        return None
    folder = Path(folder)
    found = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() == ".png":
            key = image_id(p.name)
            if key in found:
                raise ValueError(f"Duplicate mask ID {key} in {folder}")
            found[key] = p
    if set(found) != set(ids):
        raise ValueError("Mask IDs must match image IDs")
    masks = []
    for key in ids:
        mask = cv2.imread(str(found[key]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot decode mask: {found[key]}")
        masks.append(mask)
    return masks


def load_descriptors_w2(path, mask_folder=None, method=None):
    """Descriptors per W2 method for a folder (numeric order + optional masks).

    method: a single registry name (computed alone, fast) or None for all.
    Returns (ids, descriptors); descriptors is a single list when method is
    given, else one list per method in METHODS_W2_NAMES order.
    """
    ids, paths = _ordered_image_paths(path)
    masks = _ordered_masks(mask_folder, ids) if mask_folder else None
    names = [method] if method is not None else list(METHODS_W2_NAMES)
    if method is not None and method not in METHODS_W2:
        raise ValueError(f"Unknown W2 method: {method!r}")
    descriptors = [[] for _ in names]
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot decode image: {p}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        mask = None if masks is None else masks[i]
        for j, name in enumerate(names):
            descriptors[j].append(METHODS_W2[name](lab, mask=mask))
    if method is not None:
        return ids, descriptors[0]
    return ids, descriptors


def evaluate_folders_w2(museum_path, queries_path, method, distance_name,
                        top_k, ground_truth_path=None, museum_masks=None,
                        query_masks=None):
    """Retrieve top-k museum IDs per query; print mAP if ground truth given."""
    # Local import: k_similarity imports METHODS_W2 at module level.
    from k_similarity import (
        DISTANCE_FUNCTIONS,
        mean_average_precision_at_k,
        retrieve_all_queries,
    )

    museum_ids, museum = load_descriptors_w2(
        museum_path, museum_masks, method=method)
    query_ids, queries = load_descriptors_w2(
        queries_path, query_masks, method=method)
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


def _mask_paths(folder):
    """Map numeric mask ID -> PNG path in `folder`."""
    paths = {}
    for path in Path(folder).iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        index = image_id(path.name)
        if index in paths:
            raise ValueError(f"Duplicate mask ID {index} in {folder}")
        paths[index] = path
    return paths


def evaluate_segmentation(pred_dir, gt_dir, save_path=None):
    """Precision/recall/F1 of predicted masks vs trusted masks (by ID).

    Both folders hold binary PNG masks, paired by the numeric ID in the
    filename. Returns {"per_image": [...], "mean": {...}}, prints a table
    and optionally writes it to `save_path`. Task 4 (QSD2 segmentation).
    """
    predicted = _mask_paths(pred_dir)
    truth = _mask_paths(gt_dir)
    if not truth or predicted.keys() != truth.keys():
        raise ValueError("Both folders must contain the same mask IDs")
    per_image = []
    for index in sorted(truth):
        predicted_mask = cv2.imread(str(predicted[index]), cv2.IMREAD_GRAYSCALE)
        truth_mask = cv2.imread(str(truth[index]), cv2.IMREAD_GRAYSCALE)
        if predicted_mask is None or truth_mask is None:
            raise ValueError(f"Cannot read mask {index}")
        scores = compute_segmentation_metrics(predicted_mask, truth_mask)
        per_image.append({"id": index, **scores})
    mean = {key: sum(row[key] for row in per_image) / len(per_image)
            for key in ("precision", "recall", "F1")}
    lines = [f"{'Image':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}"]
    for row in per_image:
        lines.append(f"{row['id']:10d} {row['precision']:10.4f} "
                     f"{row['recall']:10.4f} {row['F1']:10.4f}")
    lines.append(f"{'Mean':>10s} {mean['precision']:10.4f} "
                 f"{mean['recall']:10.4f} {mean['F1']:10.4f}")
    print("\n".join(lines))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved to {save_path}")
    return {"per_image": per_image, "mean": mean}


def build_parser():
    try:
        from k_similarity import DISTANCE_FUNCTIONS
        distances = sorted(DISTANCE_FUNCTIONS)
    except Exception:
        distances = ["L1"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="List available W2 methods and exit")
    parser.add_argument("--method", choices=METHODS_W2_NAMES,
                        default="Lab block 4x4 1D 32",
                        help="Descriptor for retrieval evaluation")
    parser.add_argument("--museum", type=Path, help="Museum image folder")
    parser.add_argument("--queries", type=Path, help="Query image folder")
    parser.add_argument("--museum-masks", type=Path, default=None,
                        help="Optional PNG mask folder for the museum set")
    parser.add_argument("--query-masks", type=Path, default=None,
                        help="Optional PNG mask folder for the queries")
    parser.add_argument("--ground-truth", type=Path, default=None,
                        help="Trusted correspondence pickle (IDs, query order)")
    parser.add_argument("--distance", choices=distances, default="L1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--segmentation", nargs=2, type=Path,
                        metavar=("PRED_DIR", "GT_DIR"),
                        help="Evaluate predicted masks vs trusted masks "
                             "(precision/recall/F1) and exit")
    parser.add_argument("--save", type=Path, default=None,
                        help="Write the segmentation table to this file")
    return parser


def main(args=None):
    parser = build_parser()
    opts = parser.parse_args(args)
    if opts.list:
        for name in METHODS_W2_NAMES:
            print(name)
        return
    if opts.segmentation:
        evaluate_segmentation(*opts.segmentation, save_path=opts.save)
        return
    if not opts.museum or not opts.queries:
        parser.error("--museum and --queries are required "
                     "(or use --list / --segmentation)")
    evaluate_folders_w2(opts.museum, opts.queries, opts.method,
                        opts.distance, opts.top_k, opts.ground_truth,
                        opts.museum_masks, opts.query_masks)


if __name__ == "__main__":
    main()
