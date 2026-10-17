"""Methods_W2: Lab color descriptors for Week 2 (Task 1).

Run as a script (argparse selects the action):
    python methods_w2.py --list
    python methods_w2.py --museum <dir> --queries <dir> --method NAME
    python methods_w2.py --museum <dir> --queries <dir> --method all
    python methods_w2.py --museum <dir> --queries <dir> --method NAME --ground-truth <pkl>
    python methods_w2.py --segmentation <pred_dir> <gt_dir> [--save FILE]

Retrieval requires --method explicitly: a single descriptor name, or "all"
to evaluate every method in a single pass over the images.

Functions take uint8 OpenCV Lab images (all channels span [0, 256)) and
accept an optional foreground mask (Task 3/5): if ``mask`` is None the full
image is used; if given, only foreground pixels contribute to the histogram.

Only four W2 methods are kept (every part is L1-normalized and the
concatenation is L1-renormalized, so each part carries equal weight):

    - "Lab blocks 1D"  the image is split into a grid of `blocks` total
      cells (a perfect square, so n x n); each cell carries a marginal 1D
      histogram per channel (32 bins), and all cells are concatenated
      row-major.  Parameter: --blocks (total cells, default 25 -> 5x5).
    - "Lab 2D"         the image is split into the same `blocks` grid; each
      cell carries the L channel as a 1D marginal (32 bins) plus the (a, b)
      pair as a joint 2D histogram (32 bins per axis); all cells are
      concatenated row-major.  Parameter: --blocks (same as 1D).
    - "Lab 3D"         the image is split into the same `blocks` grid; each
      cell carries a joint 3D L-a-b histogram (32 bins per axis); all cells
      are concatenated row-major.  Parameter: --blocks (same as 1D).
    - "Lab pyramid"    hierarchy of 1D-block levels: level 1 is a single
      block (whole image, like "Lab blocks 1D" with 1 block) and every extra
      level doubles the number of blocks, arranged as square as possible
      (1, 2, 4, 8 ... -> grids (1,1), (1,2), (2,2), (2,4) ...); the 1D
      histograms of every cell and level are concatenated.  Parameter:
      --levels (number of hierarchy levels, default 3).

Masked variants (QSD2/QST2) reuse these same functions by passing ``mask``.
"""

import argparse
import math
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


LAB_2D_PARTS = (((0,), 32), ((1, 2), 32))
LAB_3D_BINS = 32


def compute_lab_block_2d_histogram(img_lab, grid=4, parts=LAB_2D_PARTS,
                                   mask=None):
    """Grid of mixed 1D+2D Lab parts (L + ab), concatenated row-major.

    Same grid convention as compute_lab_block_histogram; each cell carries
    the per-part mixed histogram and every cell is L1-renormalized on
    concatenation (equal weight per cell).
    """
    _grid_size(grid)
    block_fn = (lambda block, m:
                compute_lab_mixed_concat_histogram(block, parts=parts, mask=m))
    return _compute_spatial_histogram(img_lab, (grid,), block_fn, mask)


def compute_lab_block_3d_histogram(img_lab, grid=4, bins=LAB_3D_BINS,
                                   mask=None):
    """Grid of joint 3D L-a-b histograms, concatenated in row-major order.

    Same grid convention as compute_lab_block_histogram; each cell carries
    the joint 3D histogram and every cell is L1-renormalized on
    concatenation (equal weight per cell).
    """
    _grid_size(grid)
    block_fn = (lambda block, m:
                compute_lab_3d_histogram(block, bins=bins, mask=m))
    return _compute_spatial_histogram(img_lab, (grid,), block_fn, mask)


DEFAULT_BLOCKS = 25
DEFAULT_PYRAMID_LEVELS = 3


def blocks_grid(total):
    """Grid (n, n) for `total` cells, which must be a perfect square.

    1 -> (1, 1), 4 -> (2, 2), 9 -> (3, 3), 16 -> (4, 4), 25 -> (5, 5).
    """
    assert (isinstance(total, int) and not isinstance(total, bool)
            and total >= 1), f"blocks must be a positive integer, got {total!r}"
    n = math.isqrt(total)
    assert n * n == total, f"blocks must be a perfect square, got {total!r}"
    return (n, n)


def pyramid_grids(divisions):
    """Grids for `divisions` pyramid levels, block count doubling per level.

    Level 1 is a single block (whole image); each extra level doubles the
    number of blocks, arranged as square as possible:
    1, 2, 4, 8, ... -> (1, 1), (1, 2), (2, 2), (2, 4), ...
    """
    assert (isinstance(divisions, int) and not isinstance(divisions, bool)
            and divisions >= 1), (
        f"divisions must be a positive integer, got {divisions!r}")
    return tuple((2 ** (i // 2), 2 ** (i - i // 2)) for i in range(divisions))


def build_method(name, blocks=DEFAULT_BLOCKS, levels=DEFAULT_PYRAMID_LEVELS):
    """Return fn(lab, mask) for a W2 method built with explicit parameters.

    blocks: total cells (a perfect square -> n x n) for "Lab blocks 1D",
        "Lab 2D" and "Lab 3D".
    levels: number of hierarchy levels for "Lab pyramid".
    """
    if name == "Lab blocks 1D":
        return lambda lab, mask=None: compute_lab_block_histogram(
            lab, grid=blocks_grid(blocks), mask=mask)
    if name == "Lab 2D":
        return lambda lab, mask=None: compute_lab_block_2d_histogram(
            lab, grid=blocks_grid(blocks), mask=mask)
    if name == "Lab 3D":
        return lambda lab, mask=None: compute_lab_block_3d_histogram(
            lab, grid=blocks_grid(blocks), mask=mask)
    if name == "Lab pyramid":
        return lambda lab, mask=None: compute_lab_pyramid_histogram(
            lab, levels=pyramid_grids(levels), mask=mask)
    raise ValueError(f"Unknown W2 method: {name!r}")


METHODS_W2_NAMES = ("Lab blocks 1D", "Lab 2D", "Lab 3D", "Lab pyramid")
METHODS_W2 = {name: build_method(name) for name in METHODS_W2_NAMES}
METHODS_W2_COLOR_SPACES = dict.fromkeys(METHODS_W2_NAMES, "Lab")

METHOD_ALL = "all"
METHOD_CHOICES = METHODS_W2_NAMES + (METHOD_ALL,)


def compute_descriptors_w2(dataset, masks=None, blocks=DEFAULT_BLOCKS,
                           levels=DEFAULT_PYRAMID_LEVELS):
    """W2 descriptors for a dataset; masks optional (QSD2 foreground).

    dataset.images in RGB (like Dataset); masks list of HxW uint8 or None.
    Returns one list per W2 method, in METHODS_W2_NAMES order.
    """
    methods = [[] for _ in METHODS_W2_NAMES]
    fns = {name: build_method(name, blocks, levels)
           for name in METHODS_W2_NAMES}
    n = len(dataset.images)
    if masks is not None and len(masks) != n:
        raise ValueError(f"Expected {n} masks, got {len(masks)}")
    for i, img in enumerate(dataset.images):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        mask = None if masks is None else masks[i]
        for j, name in enumerate(METHODS_W2_NAMES):
            methods[j].append(fns[name](lab, mask=mask))
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


def load_descriptors_w2(path, mask_folder=None, method=None,
                        blocks=DEFAULT_BLOCKS, levels=DEFAULT_PYRAMID_LEVELS):
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
    fns = {name: build_method(name, blocks, levels) for name in names}
    descriptors = [[] for _ in names]
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot decode image: {p}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        mask = None if masks is None else masks[i]
        for j, name in enumerate(names):
            descriptors[j].append(fns[name](lab, mask=mask))
    if method is not None:
        return ids, descriptors[0]
    return ids, descriptors


def _load_ground_truth(ground_truth_path, query_ids):
    """Load the trusted correspondences and check the query count."""
    if ground_truth_path is None:
        return None
    with Path(ground_truth_path).open("rb") as stream:
        ground_truth = pickle.load(stream)
    if len(ground_truth) != len(query_ids):
        raise ValueError("Ground-truth count must match numeric query order")
    return ground_truth


def _report_rankings(method, distance_name, museum_ids, query_ids, dim,
                     rankings, top_k, ground_truth):
    """Print the header plus ranked IDs (or mAP); return the ranked IDs."""
    from k_similarity import mean_average_precision_at_k

    ranked_ids = [[museum_ids[idx] for idx, _ in row] for row in rankings]
    print(f"{method} + {distance_name}: "
          f"{len(museum_ids)} museum, {len(query_ids)} queries, dim {dim}")
    if ground_truth is None:
        for qid, row in zip(query_ids, ranked_ids):
            print(f"Query {qid}: {row}")
        return ranked_ids
    for k in (1, top_k) if top_k != 5 else (1, 5):
        print(f"mAP@{k}: "
              f"{mean_average_precision_at_k(ranked_ids, ground_truth, k):.4f}")
    return ranked_ids


def evaluate_folders_w2(museum_path, queries_path, method, distance_name,
                        top_k, ground_truth_path=None, museum_masks=None,
                        query_masks=None, blocks=DEFAULT_BLOCKS,
                        levels=DEFAULT_PYRAMID_LEVELS):
    """Retrieve top-k museum IDs per query; print mAP if ground truth given."""
    # Local import: k_similarity imports METHODS_W2 at module level.
    from k_similarity import DISTANCE_FUNCTIONS, retrieve_all_queries

    museum_ids, museum = load_descriptors_w2(
        museum_path, museum_masks, method=method, blocks=blocks, levels=levels)
    query_ids, queries = load_descriptors_w2(
        queries_path, query_masks, method=method, blocks=blocks, levels=levels)
    ground_truth = _load_ground_truth(ground_truth_path, query_ids)
    rankings = retrieve_all_queries(
        queries, museum, DISTANCE_FUNCTIONS[distance_name], top_k=top_k)
    return _report_rankings(method, distance_name, museum_ids, query_ids,
                            len(museum[0]), rankings, top_k, ground_truth)


def evaluate_all_folders_w2(museum_path, queries_path, distance_name, top_k,
                            ground_truth_path=None, museum_masks=None,
                            query_masks=None, blocks=DEFAULT_BLOCKS,
                            levels=DEFAULT_PYRAMID_LEVELS):
    """Run every W2 method in one pass; print mAP if ground truth given.

    Returns a dict mapping each method name to its ranked IDs.
    """
    from k_similarity import DISTANCE_FUNCTIONS, retrieve_all_queries

    museum_ids, museum = load_descriptors_w2(
        museum_path, museum_masks, method=None, blocks=blocks, levels=levels)
    query_ids, queries = load_descriptors_w2(
        queries_path, query_masks, method=None, blocks=blocks, levels=levels)
    ground_truth = _load_ground_truth(ground_truth_path, query_ids)
    results = {}
    for j, name in enumerate(METHODS_W2_NAMES):
        rankings = retrieve_all_queries(
            queries[j], museum[j], DISTANCE_FUNCTIONS[distance_name],
            top_k=top_k)
        results[name] = _report_rankings(
            name, distance_name, museum_ids, query_ids, len(museum[j][0]),
            rankings, top_k, ground_truth)
    return results


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


def _perfect_square(value):
    """argparse type: a positive perfect square (total number of cells)."""
    try:
        total = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"blocks must be an integer, got {value!r}") from error
    if total < 1 or math.isqrt(total) ** 2 != total:
        raise argparse.ArgumentTypeError(
            "blocks must be a positive perfect square (e.g. 1, 4, 9, 16, 25)")
    return total


def build_parser():
    try:
        from k_similarity import DISTANCE_FUNCTIONS
        distances = sorted(DISTANCE_FUNCTIONS)
    except Exception:
        distances = ["L1"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="List available W2 methods and exit")
    parser.add_argument("--method", choices=METHOD_CHOICES, default=None,
                        help="Descriptor for retrieval evaluation, or 'all' "
                             "to evaluate every method (required for "
                             "retrieval)")
    parser.add_argument("--blocks", type=_perfect_square,
                        default=DEFAULT_BLOCKS,
                        help="Total spatial cells (a perfect square, default "
                             "25 -> 5x5) for 'Lab blocks 1D', 'Lab 2D' and "
                             "'Lab 3D'")
    parser.add_argument("--levels", type=int, default=DEFAULT_PYRAMID_LEVELS,
                        help="Number of hierarchy levels for 'Lab pyramid'")
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
    if opts.method is None:
        parser.error("--method is required: one of "
                     + ", ".join(METHOD_CHOICES))
    if opts.method == METHOD_ALL:
        evaluate_all_folders_w2(opts.museum, opts.queries, opts.distance,
                                opts.top_k, opts.ground_truth,
                                opts.museum_masks, opts.query_masks,
                                opts.blocks, opts.levels)
        return
    evaluate_folders_w2(opts.museum, opts.queries, opts.method,
                        opts.distance, opts.top_k, opts.ground_truth,
                        opts.museum_masks, opts.query_masks,
                        opts.blocks, opts.levels)


if __name__ == "__main__":
    main()
