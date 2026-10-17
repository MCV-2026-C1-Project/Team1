"""Methods_W2: Lab color descriptors for Week 2 (Task 1).

Run as a script (argparse selects the action):
    python methods_w2.py --list
    python methods_w2.py --museum <dir> --queries <dir> --method NAME
    python methods_w2.py --museum <dir> --queries <dir> --method all
    python methods_w2.py --museum <dir> --queries <dir> --method NAME --ground-truth <pkl>
    python methods_w2.py --segmentation <pred_dir> <gt_dir> [--images DIR] [--viz DIR] [--save FILE]
    python methods_w2.py --segment <queries_dir> --out <masks_dir> [--viz DIR]
    python methods_w2.py --museum <dir> --queries <dir> --method NAME --blocks 36 \
        --ground-truth <pkl> --background-removal
    python methods_w2.py --museum <dir> --qst1 <dir> --qst2 <dir> [--submit-out DIR]

Week 2 tasks and where they are:
    Task 1  the descriptors below (Lab blocks 1D, 2D, 3D, pyramid)
    Task 2  retrieval + mAP on QSD1                  --method ... --ground-truth
    Task 3  background removal (in this file): masks + overlay images
                                                      --segment ... --out ... --viz ...
    Task 4  mask precision / recall / F1             --segmentation PRED GT
    Task 5  retrieval on QSD2 with Task 3 masks      --background-removal
    Task 6  blind QST1 / QST2 submission             --qst1 ... --qst2 ...

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
    - "Lab 2D"         the L channel as a 1D marginal (32 bins) plus the
      (a, b) pair as a joint 2D histogram (32 bins per axis), concatenated.
      No specific parameter.
    - "Lab 3D"         joint 3D L-a-b histogram (32 bins per axis). No
      specific parameter.
    - "Lab pyramid"    hierarchy of 1D-block levels: level 1 is a single
      block (whole image, like "Lab blocks 1D" with 1 block) and every extra
      level doubles the number of blocks, arranged as square as possible
      (1, 2, 4, 8 ... -> grids (1,1), (1,2), (2,2), (2,4) ...); the 1D
      histograms of every cell and level are concatenated.  Parameter:
      --levels (number of hierarchy levels, default 3).

Masked variants (QSD2/QST2) reuse these same functions by passing ``mask``.

Crop (Tasks 5 and 6): museum images contain only the painting, but a QSD2 /
QST2 query has wall around it. A block grid is laid over the *image*, so with
the grid over the whole photo, cell (i, j) of the query covers another part of
the painting than cell (i, j) of the museum image, even when the mask removes
the wall pixels. With ``crop=True`` the query is first cropped to the bounding
box of its mask, so both grids cover the same area (QSD2, 6x6 blocks: mAP@5
0.61 with the mask alone, 0.81 with mask + crop). ``--no-crop`` turns it off.
"""

import argparse
import math
import pickle
from pathlib import Path

from dataclasses import dataclass

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

    blocks: grid side for "Lab blocks 1D" (blocks x blocks cells).
    levels: number of hierarchy levels for "Lab pyramid".
    """
    if name == "Lab blocks 1D":
        return lambda lab, mask=None: compute_lab_block_histogram(
            lab, grid=blocks_grid(blocks), mask=mask)
    if name == "Lab 2D":
        return lambda lab, mask=None: compute_lab_mixed_concat_histogram(
            lab, parts=[((0,), 32), ((1, 2), 32)], mask=mask)
    if name == "Lab 3D":
        return lambda lab, mask=None: compute_lab_3d_histogram(
            lab, bins=32, mask=mask)
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


def _find_folder(folder, suffixes):
    """The folder that holds the files: ``folder`` itself, or the only folder
    inside it (up to two levels down) that has files with these suffixes.

    So ``data/qsd2_w2`` works when the images are in ``data/qsd2_w2/qsd2_w1``
    (a zip often adds one extra folder). Several candidates -> error that
    lists them, so the choice is never silent.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Expected a folder: {folder}")

    def has_files(d):
        return any(p.is_file() and p.suffix.lower() in suffixes
                   for p in d.iterdir())

    if has_files(folder):
        return folder
    children = sorted(d for d in folder.iterdir() if d.is_dir())
    found = [d for d in children if has_files(d)]
    if not found:      # one more level down
        found = [g for d in children for g in sorted(d.iterdir())
                 if g.is_dir() and has_files(g)]
    kind = "/".join(suffixes)
    if not found:
        raise ValueError(f"No {kind} files in {folder} or in the folders inside it")
    if len(found) > 1:
        raise ValueError(f"{folder} has several folders with {kind} files "
                         f"({[str(d.name) for d in found]}): pass one of them")
    print(f"Using {found[0]} (the files are one folder inside {folder})")
    return found[0]


def _ordered_image_paths(folder):
    folder = _find_folder(folder, (".jpg", ".jpeg"))
    paths = sorted(
        (p for p in folder.iterdir() if p.is_file()
         and p.suffix.lower() in (".jpg", ".jpeg")),
        key=lambda p: (image_id(p), p.name),
    )
    ids = [image_id(p) for p in paths]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Expected unique image IDs in {folder}")
    return ids, paths


def _ordered_masks(folder, ids):
    """Load grayscale PNG masks aligned to image IDs (nonzero=foreground)."""
    if folder is None:
        return None
    folder = _find_folder(folder, (".png",))
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


# ----------------------------------------------------- Task 3: background
# Background removal using only the background color (it used to live in
# background_removal.py; everything is in this file now).
#
# Week 2, Task 3: background removal using only the background color.
#
# Idea
# ----
# The wall behind a painting is (almost) uniform in color, and the painting never
# touches the image border. So:
#
# 1. Model the background as a Gaussian in CIE-Lab, fitted on the pixels of a
#    thin strip along the four image borders (robust: median + MAD-based
#    covariance, so a frame/painting poking into the strip does not break it).
# 2. Per-pixel Mahalanobis distance to that model -> foreground where the
#    distance is large. The threshold is chosen per image with Otsu on the
#    distance map, clipped to a sane range (no tuning per image by hand).
# 3. Cleanup with morphology + keep the largest connected blob + fill holes.
#    (No contour / edge / object detectors: the mask is still decided by color.)
# 4. Optional: replace the blob by its bounding box. Paintings are rectangles,
#    so this recovers dark regions of the painting that look like the wall.
#
# Masks are uint8 HxW with 255 = painting (foreground) and 0 = background, which
# is the format of the ground-truth .png files and what ``cv2.calcHist`` expects
# as ``mask`` (any non-zero value is selected).


@dataclass
class BGParams:
    border: float = 0.04          # border strip width, fraction of min(H, W)
    blur: int = 5                 # Gaussian blur (odd) before the color model
    thr_min: float = 3.0          # clip Otsu threshold (Mahalanobis units)
    thr_max: float = 12.0
    open_frac: float = 0.004      # morphology kernel size, fraction of min(H, W)
    close_frac: float = 0.02
    use_bbox: bool = True         # fill the bounding box of the largest blob
    poly_degree: int = 1          # wall illumination surface (0 = constant)
    max_side: int = 400           # work at this resolution, upsample the mask
    min_area_frac: float = 0.02   # discard blobs smaller than this fraction


def _border_pixels(img, border):
    h, w = img.shape[:2]
    b = max(2, int(round(border * min(h, w))))
    strip = np.zeros((h, w), bool)
    strip[:b, :] = strip[-b:, :] = strip[:, :b] = strip[:, -b:] = True
    return strip


def _poly_features(h, w, degree, ys=None, xs=None):
    """Polynomial terms in normalized (x, y) in [-1, 1]; full grid if ys is None."""
    if ys is None:
        ys, xs = np.mgrid[0:h, 0:w]
    y = (ys.astype(np.float32) / max(h - 1, 1)) * 2 - 1
    x = (xs.astype(np.float32) / max(w - 1, 1)) * 2 - 1
    cols = [x ** i * y ** j for i in range(degree + 1)
            for j in range(degree + 1 - i)]
    return np.stack(cols, axis=-1)


def fit_background_model(lab, border=0.04, degree=2):
    """Smooth wall model: per-channel polynomial surface fitted on the border.

    Walls are lit unevenly (vignetting, spot lights), so a constant color is
    too rigid. The surface is fitted with a few robust re-weighting rounds so
    frame corners / labels in the strip are ignored. Returns the coefficients
    and the residual covariance (inverse) used for the Mahalanobis distance.
    """
    h, w = lab.shape[:2]
    ys, xs = np.nonzero(_border_pixels(lab, border))
    A = _poly_features(h, w, degree, ys, xs)
    Y = lab[ys, xs].astype(np.float32)
    keep = np.ones(len(ys), bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], Y[keep], rcond=None)
        res = Y - A @ coef
        mad = np.median(np.abs(res), axis=0) * 1.4826 + 1.0
        keep = np.all(np.abs(res) < 3.0 * mad, axis=1)
        if keep.sum() < 50:
            keep[:] = True
            break
    res = (Y - A @ coef)[keep]
    cov = np.cov(res.T) + np.eye(3, dtype=np.float32) * 2.0   # regularize
    return coef, np.linalg.inv(cov)


def background_distance(lab, model, degree=2):
    """Mahalanobis distance of every pixel to the (spatially varying) wall."""
    coef, icov = model
    h, w = lab.shape[:2]
    pred = _poly_features(h, w, degree) @ coef                 # H x W x 3
    d = (lab.astype(np.float32) - pred).reshape(-1, 3)
    m2 = np.einsum("ij,jk,ik->i", d, icov, d)
    return np.sqrt(np.maximum(m2, 0)).reshape(h, w)


def _largest_blob(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < min_area:
        return mask
    return (labels == idx).astype(np.uint8) * 255


def _fill_holes(mask):
    inv = cv2.bitwise_not(mask)
    h, w = mask.shape
    flood = inv.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 0)      # background connected to corner
    return cv2.bitwise_or(mask, flood)       # whatever is left = holes


def remove_background(img_rgb, params=None, return_distance=False):
    """Foreground mask (uint8, 255 = painting) for an RGB image."""
    p = params or BGParams()
    H, W = img_rgb.shape[:2]
    scale = min(1.0, p.max_side / max(H, W))
    if scale < 1.0:
        img_rgb = cv2.resize(img_rgb, (round(W * scale), round(H * scale)),
                             interpolation=cv2.INTER_AREA)
    h, w = img_rgb.shape[:2]
    s = min(h, w)

    img = cv2.GaussianBlur(img_rgb, (p.blur, p.blur), 0) if p.blur > 1 else img_rgb
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    model = fit_background_model(lab, p.border, p.poly_degree)
    dist = background_distance(lab, model, p.poly_degree)

    d8 = np.clip(dist * (255.0 / 20.0), 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(otsu * 20.0 / 255.0, p.thr_min, p.thr_max))
    mask = (dist > thr).astype(np.uint8) * 255

    k = lambda f: cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, int(f * s)) | 1,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k(p.open_frac))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k(p.close_frac))
    mask = _largest_blob(mask, int(p.min_area_frac * h * w))
    mask = _fill_holes(mask)

    if p.use_bbox:
        ys, xs = np.nonzero(mask)
        if len(ys):
            out = np.zeros_like(mask)
            out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 255
            mask = out
    if scale < 1.0:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return (mask, dist) if return_distance else mask


def mask_scores(pred, gt):
    """Pixel-level precision, recall and F1 (foreground = positive class)."""
    a, o = gt > 0, pred > 0
    tp = np.logical_and(a, o).sum()
    fp = np.logical_and(~a, o).sum()
    fn = np.logical_and(a, ~o).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


# A mask smaller than this fraction of the image is treated as a failed
# segmentation and the whole image is used instead.
MIN_FOREGROUND_FRAC = 0.02


def _crop_to_foreground(rgb, mask, min_frac=MIN_FOREGROUND_FRAC):
    """Crop image and mask to the bounding box of the foreground.

    Returns ``(rgb, mask)`` unchanged when there is no mask, and ``(rgb, None)``
    when the mask is almost empty, so a failed segmentation degrades to "use
    the whole image" instead of crashing.
    """
    if mask is None:
        return rgb, None
    ys, xs = np.nonzero(mask)
    if len(ys) < min_frac * mask.size:
        return rgb, None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def background_segmenter(params=None):
    """Task 3: ``fn(rgb) -> 0/255 mask`` (``remove_background`` below)."""
    if params is None:
        return remove_background
    return lambda rgb: remove_background(rgb, params)


def load_descriptors_w2(path, mask_folder=None, method=None,
                        blocks=DEFAULT_BLOCKS, levels=DEFAULT_PYRAMID_LEVELS,
                        segmenter=None, crop=False):
    """Descriptors per W2 method for a folder (numeric order + optional masks).

    method: a single registry name (computed alone, fast) or None for all.
    mask_folder: saved PNG masks; segmenter: ``fn(rgb) -> mask`` computed on the
    fly (Task 3, see ``background_segmenter``); not both.
    crop: lay the grid over the bounding box of the mask (see module docstring).
    Returns (ids, descriptors); descriptors is a single list when method is
    given, else one list per method in METHODS_W2_NAMES order.
    """
    if mask_folder and segmenter is not None:
        raise ValueError("Use either mask_folder or segmenter, not both")
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
        mask = None if masks is None else masks[i]
        if segmenter is not None:
            mask = segmenter(rgb)
        if crop:
            rgb, mask = _crop_to_foreground(rgb, mask)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
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
                        levels=DEFAULT_PYRAMID_LEVELS, query_segmenter=None,
                        crop=False):
    """Retrieve top-k museum IDs per query; print mAP if ground truth given.

    Task 5: ``query_segmenter`` (Task 3 background removal) or ``query_masks``
    restrict the query descriptor to the painting; ``crop`` aligns the grid.
    """
    # Local import: k_similarity imports METHODS_W2 at module level.
    from k_similarity import DISTANCE_FUNCTIONS, retrieve_all_queries

    museum_ids, museum = load_descriptors_w2(
        museum_path, museum_masks, method=method, blocks=blocks, levels=levels)
    query_ids, queries = load_descriptors_w2(
        queries_path, query_masks, method=method, blocks=blocks, levels=levels,
        segmenter=query_segmenter, crop=crop)
    ground_truth = _load_ground_truth(ground_truth_path, query_ids)
    rankings = retrieve_all_queries(
        queries, museum, DISTANCE_FUNCTIONS[distance_name], top_k=top_k)
    return _report_rankings(method, distance_name, museum_ids, query_ids,
                            len(museum[0]), rankings, top_k, ground_truth)


def evaluate_all_folders_w2(museum_path, queries_path, distance_name, top_k,
                            ground_truth_path=None, museum_masks=None,
                            query_masks=None, blocks=DEFAULT_BLOCKS,
                            levels=DEFAULT_PYRAMID_LEVELS, query_segmenter=None,
                            crop=False):
    """Run every W2 method in one pass; print mAP if ground truth given.

    Returns a dict mapping each method name to its ranked IDs.
    """
    from k_similarity import DISTANCE_FUNCTIONS, retrieve_all_queries

    museum_ids, museum = load_descriptors_w2(
        museum_path, museum_masks, method=None, blocks=blocks, levels=levels)
    query_ids, queries = load_descriptors_w2(
        queries_path, query_masks, method=None, blocks=blocks, levels=levels,
        segmenter=query_segmenter, crop=crop)
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
    folder = _find_folder(folder, (".png",))
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        index = image_id(path.name)
        if index in paths:
            raise ValueError(f"Duplicate mask ID {index} in {folder}")
        paths[index] = path
    return paths


def overlay(img_rgb, mask, gt=None):
    """Dim the removed background; red = predicted outline, green = GT outline."""
    vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    vis[mask == 0] = (vis[mask == 0] * 0.25).astype(np.uint8)
    thickness = max(2, min(vis.shape[:2]) // 150)
    for m, color in ((gt, (0, 255, 0)), (mask, (0, 0, 255))):
        if m is not None:
            contours, _ = cv2.findContours((m > 0).astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, thickness)
    return vis


def contact_sheet(tiles, labels, cols=6, size=240):
    """Grid of resized overlays with a label (name, F1) written on each."""
    cells = []
    for tile, label in zip(tiles, labels):
        h, w = tile.shape[:2]
        k = size / max(h, w)
        cell = np.zeros((size, size, 3), np.uint8)
        small = cv2.resize(tile, (round(w * k), round(h * k)))
        cell[:small.shape[0], :small.shape[1]] = small
        cv2.putText(cell, label, (4, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    while len(cells) % cols:
        cells.append(np.zeros((size, size, 3), np.uint8))
    return np.vstack([np.hstack(cells[i:i + cols])
                      for i in range(0, len(cells), cols)])


def _read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_overlays(viz_dir, stems, tiles, labels):
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stem, tile in zip(stems, tiles):
        cv2.imwrite(str(viz_dir / f"{stem}_overlay.jpg"), tile)
    cv2.imwrite(str(viz_dir / "overview.jpg"), contact_sheet(tiles, labels))
    print(f"Overlays + overview.jpg saved to {viz_dir}")


def segment_folder(folder, out_dir=None, viz_dir=None, segmenter=None):
    """Task 3: background removal for every image of ``folder``.

    Saves ``<name>.png`` masks (0 / 255, same name as the image) in ``out_dir``
    and, with ``viz_dir``, overlay images + ``overview.jpg``. When a ground-truth
    ``<name>.png`` sits next to the image (QSD2) it is drawn in green and the
    F1 is printed. Returns ``(ids, masks)``.
    """
    segmenter = segmenter or background_segmenter()
    ids, paths = _ordered_image_paths(folder)
    masks, tiles, labels, scores = [], [], [], []
    for path in paths:
        rgb = _read_rgb(path)
        mask = segmenter(rgb)
        masks.append(mask)
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(out_dir) / (path.stem + ".png")), mask)
        gt_path = path.with_suffix(".png")      # only QSD2 has ground-truth masks
        gt = (cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
              if gt_path.exists() else None)
        label = path.stem
        if gt is not None:
            scores.append(compute_segmentation_metrics(mask, gt))
            label += f"  F1={scores[-1]['F1']:.2f}"
        if viz_dir is not None:
            tiles.append(overlay(rgb, mask, gt))
            labels.append(label)
    if viz_dir is not None:
        _save_overlays(viz_dir, [p.stem for p in paths], tiles, labels)
    print(f"{len(ids)} images segmented"
          + (f", masks -> {out_dir}" if out_dir is not None else ""))
    if scores:
        mean = {k: sum(s[k] for s in scores) / len(scores)
                for k in ("precision", "recall", "F1")}
        print(f"Against the ground-truth masks next to the images: "
              f"precision={mean['precision']:.4f} recall={mean['recall']:.4f} "
              f"F1={mean['F1']:.4f}")
    return ids, masks


def evaluate_segmentation(pred_dir, gt_dir, save_path=None, images_dir=None,
                          viz_dir=None):
    """Precision/recall/F1 of predicted masks vs trusted masks (by ID).

    Both folders hold binary PNG masks, paired by the numeric ID in the
    filename. Returns {"per_image": [...], "mean": {...}}, prints a table
    and optionally writes it to `save_path`. Task 4 (QSD2 segmentation).
    With `viz_dir`, also saves overlays + overview.jpg (red = predicted,
    green = trusted) drawn on the .jpg images of `images_dir` (default: the
    ground-truth folder, where QSD2 keeps the .jpg next to its .png mask).
    """
    predicted = _mask_paths(pred_dir)
    truth = _mask_paths(gt_dir)
    if not truth:
        raise ValueError(f"No ground-truth .png masks in {gt_dir}")
    if predicted.keys() != truth.keys():
        missing = sorted(truth.keys() - predicted.keys())[:10]
        extra = sorted(predicted.keys() - truth.keys())[:10]
        raise ValueError("Both folders must contain the same mask IDs "
                         f"(missing in {pred_dir}: {missing}; "
                         f"not in {gt_dir}: {extra})")
    images = {}
    if viz_dir is not None:
        image_ids, image_paths = _ordered_image_paths(images_dir or gt_dir)
        images = dict(zip(image_ids, image_paths))
        if not truth.keys() <= images.keys():
            raise ValueError("Every mask needs its .jpg image (use --images)")
    per_image, tiles, labels = [], [], []
    for index in sorted(truth):
        predicted_mask = cv2.imread(str(predicted[index]), cv2.IMREAD_GRAYSCALE)
        truth_mask = cv2.imread(str(truth[index]), cv2.IMREAD_GRAYSCALE)
        if predicted_mask is None or truth_mask is None:
            raise ValueError(f"Cannot read mask {index}")
        scores = compute_segmentation_metrics(predicted_mask, truth_mask)
        per_image.append({"id": index, **scores})
        if viz_dir is not None:
            tiles.append(overlay(_read_rgb(images[index]), predicted_mask,
                                 truth_mask))
            labels.append(f"{truth[index].stem}  F1={scores['F1']:.2f}")
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
    if viz_dir is not None:
        _save_overlays(viz_dir, [truth[i].stem for i in sorted(truth)], tiles,
                       labels)
    return {"per_image": per_image, "mean": mean}


# ------------------------------------------------------ Task 6: submission
K_SUBMISSION = 10
# (output folder, descriptor, its parameters, comparison). method1 = Lab 6x6
# blocks, method2 = Lab 4x4 blocks, 32 bins per channel, L1: the best two on
# QSD1 / QSD2. Change here to change the submission.
SUBMISSION_METHODS = (
    ("method1", "Lab blocks 1D", {"blocks": 36}, "L1"),
    ("method2", "Lab blocks 1D", {"blocks": 16}, "L1"),
)


def _rank_ids(queries, museum, museum_ids, distance_name, k=K_SUBMISSION):
    """For each query, the k closest museum image ids (Python ints), best first."""
    from k_similarity import DISTANCE_FUNCTIONS, retrieve_all_queries
    if len(museum_ids) < k:
        raise ValueError(f"Need at least {k} museum images, got {len(museum_ids)}")
    rankings = retrieve_all_queries(
        queries, museum, DISTANCE_FUNCTIONS[distance_name], top_k=k)
    return [[int(museum_ids[idx]) for idx, _ in row] for row in rankings]


def _verify_submission(destination, museum_ids, n_queries, k=K_SUBMISSION):
    """Read the pickle back and check: list of lists, k unique known int ids."""
    with Path(destination).open("rb") as stream:
        result = pickle.load(stream)
    known = set(museum_ids)
    assert isinstance(result, list) and len(result) == n_queries, "wrong query count"
    for row in result:
        assert isinstance(row, list) and len(row) == k, f"need {k} ids per query"
        assert all(type(v) is int for v in row), "ids must be Python ints"
        assert len(set(row)) == k and set(row) <= known, "ids must be unique museum ids"


def _verify_masks(mask_dir, query_paths):
    """Every query has a <name>.png mask of the same size, values 0 / 255."""
    for path in query_paths:
        mask = cv2.imread(str(Path(mask_dir) / (path.stem + ".png")),
                          cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Missing mask {path.stem}.png in {mask_dir}")
        if mask.shape != cv2.imread(str(path), cv2.IMREAD_COLOR).shape[:2]:
            raise ValueError(f"{path.stem}.png has the wrong size")
        if not set(np.unique(mask)) <= {0, 255}:
            raise ValueError(f"{path.stem}.png is not binary (0/255)")


def write_submission(museum_path, qst1=None, qst2=None,
                     output_dir=Path("results/week2"), distance=None):
    """Task 6: ``result.pkl`` (K=10 ids per query) for QST1 and / or QST2.

    QST1 (cropped paintings): descriptor on the whole image.
    QST2 (paintings with background): Task 3 masks are saved as <name>.png in
    each method folder, then the descriptor uses mask + crop.
    Layout: <output_dir>/QST1|QST2/method1|method2/result.pkl (+ QST2 *.png).
    """
    import shutil
    museum_cache = {}
    for split, folder in (("QST1", qst1), ("QST2", qst2)):
        if folder is None:
            continue
        use_masks = split == "QST2"
        _, query_paths = _ordered_image_paths(folder)
        first_dir = None
        for out_name, name, params, default_distance in SUBMISSION_METHODS:
            dest = Path(output_dir) / split / out_name
            dest.mkdir(parents=True, exist_ok=True)
            if use_masks:
                if first_dir is None:       # segment once, copy to the others
                    segment_folder(folder, out_dir=dest)
                    first_dir = dest
                else:
                    for path in query_paths:
                        shutil.copy(first_dir / (path.stem + ".png"),
                                    dest / (path.stem + ".png"))
                _verify_masks(dest, query_paths)
            key = (name, tuple(sorted(params.items())))
            if key not in museum_cache:
                museum_cache[key] = load_descriptors_w2(
                    museum_path, method=name, **params)
            museum_ids, museum = museum_cache[key]
            query_ids, queries = load_descriptors_w2(
                folder, dest if use_masks else None, method=name,
                crop=use_masks, **params)
            distance_name = distance or default_distance
            result = _rank_ids(queries, museum, museum_ids, distance_name)
            with (dest / "result.pkl").open("wb") as stream:
                pickle.dump(result, stream, protocol=4)
            _verify_submission(dest / "result.pkl", museum_ids, len(query_ids))
            print(f"{split} {out_name} ({name} {params}, {distance_name}): "
                  f"{len(query_ids)} queries x {K_SUBMISSION} ids"
                  + (f" + {len(query_ids)} masks" if use_masks else "")
                  + f" -> {dest}")


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
                        help="Total cells for 'Lab blocks 1D' (a perfect "
                             "square, default 25 -> 5x5)")
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
    parser.add_argument("--images", type=Path, default=None,
                        help="--segmentation --viz: folder with the .jpg images "
                             "(default: the ground-truth folder)")
    parser.add_argument("--viz", type=Path, default=None,
                        help="--segment / --segmentation: folder for overlay "
                             "images + overview.jpg")
    parser.add_argument("--segment", type=Path, metavar="QUERIES_DIR",
                        help="Task 3: remove the background of every image and "
                             "exit (needs --out)")
    parser.add_argument("--out", type=Path, default=None,
                        help="--segment: folder for the <name>.png masks")
    parser.add_argument("--background-removal", action="store_true",
                        help="Task 5: segment the queries with Task 3 on the fly")
    parser.add_argument("--no-crop", action="store_true",
                        help="with query masks: keep the grid over the whole "
                             "image instead of the mask bounding box")
    parser.add_argument("--qst1", type=Path, default=None,
                        help="Task 6: QST1 folder (cropped paintings)")
    parser.add_argument("--qst2", type=Path, default=None,
                        help="Task 6: QST2 folder (paintings with background)")
    parser.add_argument("--submit-out", type=Path, default=Path("results/week2"),
                        help="Task 6: output folder (default results/week2)")
    return parser


def main(args=None):
    parser = build_parser()
    opts = parser.parse_args(args)
    if opts.list:
        for name in METHODS_W2_NAMES:
            print(name)
        return
    if opts.segmentation:
        evaluate_segmentation(*opts.segmentation, save_path=opts.save,
                              images_dir=opts.images, viz_dir=opts.viz)
        return
    if opts.segment:
        if opts.out is None:
            parser.error("--segment needs --out (folder for the masks)")
        segment_folder(opts.segment, opts.out, opts.viz)
        return
    if opts.qst1 or opts.qst2:
        if not opts.museum:
            parser.error("--museum is required with --qst1 / --qst2")
        write_submission(opts.museum, opts.qst1, opts.qst2, opts.submit_out,
                         opts.distance if opts.distance != "L1" else None)
        print("\nDone. Upload the folder as Team1/week2/ (QST1/ and QST2/).")
        return
    if not opts.museum or not opts.queries:
        parser.error("--museum and --queries are required "
                     "(or use --list / --segmentation)")
    if opts.method is None:
        parser.error("--method is required: one of "
                     + ", ".join(METHOD_CHOICES))
    if opts.background_removal and opts.query_masks:
        parser.error("use --background-removal or --query-masks, not both")
    segmenter = background_segmenter() if opts.background_removal else None
    crop = (segmenter is not None or opts.query_masks is not None) \
        and not opts.no_crop
    if opts.method == METHOD_ALL:
        evaluate_all_folders_w2(opts.museum, opts.queries, opts.distance,
                                opts.top_k, opts.ground_truth,
                                opts.museum_masks, opts.query_masks,
                                opts.blocks, opts.levels, segmenter, crop)
        return
    evaluate_folders_w2(opts.museum, opts.queries, opts.method,
                        opts.distance, opts.top_k, opts.ground_truth,
                        opts.museum_masks, opts.query_masks,
                        opts.blocks, opts.levels, segmenter, crop)


if __name__ == "__main__":
    main()