"""Week 3 in one file: texture descriptors (Task 2), multi-painting background
removal (Task 3), retrieval per painting (Task 4) and blind submission (Task 5).

Part 1 - texture descriptors. All work on the grayscale image, split into
5x5 = 25 blocks whose features are concatenated:

    "LBP"             3x3 Local Binary Pattern, 256-bin histogram per block.
    "LBP uniform"     58 uniform patterns plus one bin for the rest (59 bins).
    "Multiscale LBP"  circular LBP at (P,R) = (8,1), (12,2.5), (16,4), bilinear
                      interpolation, uniform patterns, three histograms/block.
    "DCT"             2D DCT per block, zig-zag scan, first N coefficients
                      (the DC term is dropped: it is brightness, not texture).
    "Wavelet"         3-level Haar; mean absolute value of every detail subband.
    "DCT log"         DCT with sign(c) * log(1 + |c|) coefficients.
    "Texture fusion"  DCT log + Multiscale LBP + Wavelet (default), each part
                      with unit L2 norm; use it with L1.

DCT-based methods are signed, so Chi-square, Histogram Intersection and
Hellinger are rejected for them. Only numpy and OpenCV are needed.

Part 2 - paintings. The wall model, mask metrics and visualizations come from
Week 2. Each foreground component is cropped separately before its descriptor
is computed. Paintings are returned from left to right, or top to bottom when
stacked.

    python src/paintings_w3.py --list
    python src/paintings_w3.py --segment data/qsd2_w3 --out results/week3/QSD2/masks
    python src/paintings_w3.py --museum data/BBDD --queries data/qsd2_w3 \
        --background-removal --ground-truth data/qsd2_w3/gt_corresps.pkl \
        --query-filter methods_w3:median_filter
    # Task 2 table on QSD1-W3 (texture only, no segmentation)
    python src/paintings_w3.py --museum data/BBDD --queries data/qsd1_w3 \
        --ground-truth data/qsd1_w3/gt_corresps.pkl --table --denoise 3
"""

import argparse
import importlib
import json
import math
import pickle
from dataclasses import asdict, dataclass
from functools import lru_cache
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
    blocks_grid,
    evaluate_segmentation,
    fit_background_model,
    overlay,
)
from utils import compute_segmentation_metrics


# ===========================================================================
# PART 1 - TEXTURE DESCRIPTORS (Task 2)
# ===========================================================================

DEFAULT_BLOCKS = 25
DEFAULT_DISTANCE = "L1"
DEFAULT_SIZE = 256           # DCT / wavelet work on a fixed-size gray image
DEFAULT_LBP_SIDE = 256       # LBP works with the longest side at this size
DEFAULT_DCT_COEFFS = 15      # zig-zag coefficients kept per block (no DC)
DEFAULT_WAVELET_LEVELS = 3
LBP_SCALES = ((8, 1.0), (12, 2.5), (16, 4.0))   # (points P, radius R)

LBP_METHOD = "LBP"
LBP_UNIFORM_METHOD = "LBP uniform"
MULTISCALE_METHOD = "Multiscale LBP"
DCT_METHOD = "DCT"
WAVELET_METHOD = "Wavelet"
DCT_LOG_METHOD = "DCT log"
FUSION_METHOD = "Texture fusion"
TEXTURE_METHOD_NAMES = (LBP_METHOD, LBP_UNIFORM_METHOD, MULTISCALE_METHOD,
                        DCT_METHOD, WAVELET_METHOD, DCT_LOG_METHOD,
                        FUSION_METHOD)
# DCT coefficients are signed, so the distances that assume non-negative
# histograms (they take square roots or minima of the values) do not apply.
SIGNED_METHODS = (DCT_METHOD, DCT_LOG_METHOD, FUSION_METHOD)
NON_NEGATIVE_DISTANCES = ("Chi-square", "Histogram Intersection", "Hellinger")
METHOD_ALL = "all"
METHOD_CHOICES = TEXTURE_METHOD_NAMES + (METHOD_ALL,)
DISTANCE_ALL = "all"

_METHOD_ALIASES = {
    "lbp": LBP_METHOD,
    "lbp uniform": LBP_UNIFORM_METHOD,
    "uniform lbp": LBP_UNIFORM_METHOD,
    "multiscale lbp": MULTISCALE_METHOD,
    "multi-scale lbp": MULTISCALE_METHOD,
    "dct": DCT_METHOD,
    "wavelet": WAVELET_METHOD,
    "haar": WAVELET_METHOD,
    "dct log": DCT_LOG_METHOD,
    "dct_log": DCT_LOG_METHOD,
    "texture fusion": FUSION_METHOD,
    "fusion": FUSION_METHOD,
}


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def to_gray(img_rgb):
    """Grayscale float32 image in [0, 255] from an RGB uint8 image."""
    image = np.asarray(img_rgb)
    assert image.dtype == np.uint8, f"expected a uint8 image, got {image.dtype}"
    if image.ndim == 2:
        return image.astype(np.float32)
    assert image.ndim == 3 and image.shape[2] == 3, (
        f"expected HxW or HxWx3, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def working_gray(img_rgb, side=DEFAULT_LBP_SIDE):
    """Gray image whose longest side is ``side`` pixels (aspect kept).

    Museum and query photos have very different resolutions; fixing the size
    makes LBP see the same texture scale in both, and it is also much faster.
    """
    gray = to_gray(img_rgb)
    scale = side / max(gray.shape)
    if scale == 1.0:
        return gray
    size = (max(1, round(gray.shape[1] * scale)),
            max(1, round(gray.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(gray, size, interpolation=interpolation)


def median_denoise(img_rgb, ksize):
    """Median filter per channel (queries only). ksize <= 1 returns the image.

    Delegates to the Task 1 implementation in ``methods_w3`` so there is a
    single median filter in the project.
    """
    if ksize is None or ksize <= 1:
        return img_rgb
    from methods_w3 import median_filter
    return median_filter(img_rgb, ksize=ksize)


def _cell_edges(size, parts):
    """Integer cell borders that split ``size`` pixels into ``parts`` cells."""
    return np.linspace(0, size, parts + 1).round().astype(int)


def _l1_normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    total = float(np.abs(vector).sum())
    return vector / total if total > 0 else vector


def _grid_cells(shape, grid):
    """Yield (y0, y1, x0, x1) for every cell of a (rows, cols) grid."""
    rows, cols = grid
    ys = _cell_edges(shape[0], rows)
    xs = _cell_edges(shape[1], cols)
    for i in range(rows):
        for j in range(cols):
            yield ys[i], ys[i + 1], xs[j], xs[j + 1]


# ---------------------------------------------------------------------------
# LBP
# ---------------------------------------------------------------------------

# The 8 neighbors of the 3x3 neighborhood, clockwise from the top-left one.
_NEIGHBORS_3X3 = ((-1, -1), (-1, 0), (-1, 1), (0, 1),
                  (1, 1), (1, 0), (1, -1), (0, -1))


def lbp_codes_3x3(gray):
    """Classic LBP: compare each pixel with its 8 neighbors (3x3).

    A bit is 1 when the neighbor is >= the center and 0 when the center is
    greater, as in the slide. Returns an int array with codes 0..255.
    """
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    h, w = gray.shape
    codes = np.zeros((h, w), np.int64)
    for bit, (dy, dx) in enumerate(_NEIGHBORS_3X3):
        neighbor = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        codes |= (neighbor >= gray).astype(np.int64) << bit
    return codes


def lbp_codes_circular(gray, points, radius):
    """Circular LBP with ``points`` samples at distance ``radius``.

    Samples that do not fall on a pixel are bilinearly interpolated
    (multiscale LBP). Returns an int array with codes 0..2**points - 1.
    """
    h, w = gray.shape
    pad = int(math.ceil(radius)) + 1
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad,
                                cv2.BORDER_REFLECT_101)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    codes = np.zeros((h, w), np.int64)
    for p in range(points):
        angle = 2.0 * math.pi * p / points
        map_x = xs + pad + np.float32(radius * math.cos(angle))
        map_y = ys + pad - np.float32(radius * math.sin(angle))
        neighbor = cv2.remap(padded, map_x, map_y, cv2.INTER_LINEAR)
        codes |= (neighbor >= gray).astype(np.int64) << p
    return codes


@lru_cache(maxsize=None)
def uniform_lookup(points):
    """Table code -> bin for the uniform LBP patterns of ``points`` bits.

    A pattern is uniform when, read circularly, it has at most two 0/1
    transitions. Each uniform pattern gets its own bin (points*(points-1)+2
    of them); all other patterns share the last one.
    """
    codes = np.arange(2 ** points, dtype=np.int64)
    mask = (1 << points) - 1
    rotated = ((codes << 1) | (codes >> (points - 1))) & mask
    changes = codes ^ rotated
    transitions = np.zeros_like(codes)
    for bit in range(points):
        transitions += (changes >> bit) & 1
    uniform = transitions <= 2
    table = np.full(codes.shape, int(uniform.sum()), np.int64)
    table[uniform] = np.arange(int(uniform.sum()))
    return table


def _block_code_histograms(codes, n_bins, grid):
    """L1-normalized histogram of the codes in every block, concatenated."""
    parts = []
    for y0, y1, x0, x1 in _grid_cells(codes.shape, grid):
        counts = np.bincount(codes[y0:y1, x0:x1].ravel(), minlength=n_bins)
        parts.append(_l1_normalize(counts))
    return np.concatenate(parts)


def lbp_descriptor(img_rgb, blocks=DEFAULT_BLOCKS, uniform=False):
    """Block LBP histograms (3x3 neighbors): 256 bins, or 59 if ``uniform``."""
    codes = lbp_codes_3x3(working_gray(img_rgb))
    n_bins = 256
    if uniform:
        codes = uniform_lookup(8)[codes]
        n_bins = 59
    return _l1_normalize(
        _block_code_histograms(codes, n_bins, blocks_grid(blocks)))


def multiscale_lbp_descriptor(img_rgb, blocks=DEFAULT_BLOCKS,
                              scales=LBP_SCALES):
    """Block histograms of uniform circular LBP at several scales."""
    gray = working_gray(img_rgb)
    grid = blocks_grid(blocks)
    per_scale = []
    for points, radius in scales:
        table = uniform_lookup(points)
        codes = table[lbp_codes_circular(gray, points, radius)]
        n_bins = points * (points - 1) + 3
        per_scale.append(_block_code_histograms(codes, n_bins, grid))
    return _l1_normalize(np.concatenate(per_scale))


# ---------------------------------------------------------------------------
# DCT
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def zigzag_order(size):
    """(row, col) indices of a size x size block in JPEG zig-zag order."""
    # Anti-diagonals in order; odd ones run top to bottom (row grows), even
    # ones bottom to top (column grows): (0,0) (0,1) (1,0) (2,0) (1,1) (0,2)...
    order = sorted(((r, c) for r in range(size) for c in range(size)),
                   key=lambda rc: (rc[0] + rc[1],
                                   rc[0] if (rc[0] + rc[1]) % 2 else rc[1]))
    rows, cols = zip(*order)
    return np.array(rows), np.array(cols)


def dct_descriptor(img_rgb, blocks=DEFAULT_BLOCKS, n_coeffs=DEFAULT_DCT_COEFFS,
                   size=DEFAULT_SIZE):
    """Block DCT: 2D DCT per block, zig-zag scan, first N coefficients.

    The gray image is resized to a multiple of the grid so every block has the
    same pixel size. The DC coefficient (mean brightness) is skipped. The
    result is not a histogram: it can be compared with L1 / Euclidean /
    cosine distances.
    """
    grid = blocks_grid(blocks)
    cell = max(8, size // max(grid))
    gray = cv2.resize(to_gray(img_rgb), (cell * grid[1], cell * grid[0]),
                      interpolation=cv2.INTER_AREA)
    rows, cols = zigzag_order(cell)
    keep = slice(1, 1 + n_coeffs)            # skip the DC term
    features = []
    for y0, y1, x0, x1 in _grid_cells(gray.shape, grid):
        coeffs = cv2.dct(np.ascontiguousarray(gray[y0:y1, x0:x1]))
        features.append(coeffs[rows, cols][keep])
    return np.concatenate(features).astype(np.float64)


# ---------------------------------------------------------------------------
# Wavelet (Haar)
# ---------------------------------------------------------------------------

def haar_level(approx):
    """One Haar step: returns (LL, horizontal, vertical, diagonal) details."""
    h, w = approx.shape
    a = approx[:h // 2 * 2, :w // 2 * 2]
    p, q = a[0::2, 0::2], a[0::2, 1::2]
    r, s = a[1::2, 0::2], a[1::2, 1::2]
    return ((p + q + r + s) / 4.0, (p + q - r - s) / 4.0,
            (p - q + r - s) / 4.0, (p - q - r + s) / 4.0)


def wavelet_descriptor(img_rgb, blocks=DEFAULT_BLOCKS,
                       levels=DEFAULT_WAVELET_LEVELS, size=DEFAULT_SIZE):
    """Mean absolute Haar detail coefficient per subband, level and block."""
    grid = blocks_grid(blocks)
    approx = cv2.resize(to_gray(img_rgb), (size, size),
                        interpolation=cv2.INTER_AREA)
    features = []
    for _ in range(levels):
        approx, horizontal, vertical, diagonal = haar_level(approx)
        for detail in (horizontal, vertical, diagonal):
            magnitude = np.abs(detail)
            for y0, y1, x0, x1 in _grid_cells(magnitude.shape, grid):
                features.append(float(magnitude[y0:y1, x0:x1].mean()))
    return _l1_normalize(np.array(features))


# ---------------------------------------------------------------------------
# Improved descriptors (Task 4): normalized DCT and texture fusion
# ---------------------------------------------------------------------------

def _unit(vector):
    """Scale a vector to unit L2 norm (zero vectors are returned as they are)."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def dct_log_descriptor(img_rgb, blocks=DEFAULT_BLOCKS,
                       n_coeffs=DEFAULT_DCT_COEFFS, size=DEFAULT_SIZE):
    """DCT descriptor with log-compressed coefficients: sign(c) * log(1 + |c|).

    The low-frequency coefficients of a block are far larger than the others,
    so they dominate every distance. The log compresses that dynamic range and
    makes the descriptor less sensitive to contrast and to noise. Signed, so
    use it with L1, Euclidean, Cosine or Correlation (L1 works best).
    """
    vector = dct_descriptor(img_rgb, blocks=blocks, n_coeffs=n_coeffs,
                            size=size)
    return np.sign(vector) * np.log1p(np.abs(vector))


def fusion_descriptor(img_rgb, blocks=DEFAULT_BLOCKS,
                      n_coeffs=DEFAULT_DCT_COEFFS):
    """Concatenation of DCT-log, Multiscale LBP and Wavelet, one weight each.

    Each part is made comparable before concatenating: histograms (LBP and
    wavelet energies) go through a square root, which turns the Euclidean
    distance into the Hellinger distance, and every part gets unit L2 norm so
    that none of the three dominates. Use it with Euclidean or Cosine.
    """
    parts = (dct_log_descriptor(img_rgb, blocks=blocks, n_coeffs=n_coeffs),
             np.sqrt(multiscale_lbp_descriptor(img_rgb, blocks=blocks)),
             np.sqrt(wavelet_descriptor(img_rgb, blocks=blocks)))
    return np.concatenate([_unit(part) for part in parts])


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def canonical_method_name(name):
    """Resolve a method name or alias to its canonical registry name."""
    if name in TEXTURE_METHOD_NAMES:
        return name
    if isinstance(name, str):
        canonical = _METHOD_ALIASES.get(name.strip().lower())
        if canonical is not None:
            return canonical
    raise ValueError(f"Unknown texture method: {name!r}")


def build_descriptor(name, blocks=DEFAULT_BLOCKS, n_coeffs=DEFAULT_DCT_COEFFS,
                     levels=DEFAULT_WAVELET_LEVELS):
    """Return ``fn(img_rgb)`` that computes the descriptor of ``name``."""
    name = canonical_method_name(name)
    if name == LBP_METHOD:
        return lambda img: lbp_descriptor(img, blocks=blocks)
    if name == LBP_UNIFORM_METHOD:
        return lambda img: lbp_descriptor(img, blocks=blocks, uniform=True)
    if name == MULTISCALE_METHOD:
        return lambda img: multiscale_lbp_descriptor(img, blocks=blocks)
    if name == DCT_METHOD:
        return lambda img: dct_descriptor(img, blocks=blocks,
                                          n_coeffs=n_coeffs)
    if name == DCT_LOG_METHOD:
        return lambda img: dct_log_descriptor(img, blocks=blocks,
                                              n_coeffs=n_coeffs)
    if name == FUSION_METHOD:
        return lambda img: fusion_descriptor(img, blocks=blocks,
                                             n_coeffs=n_coeffs)
    return lambda img: wavelet_descriptor(img, blocks=blocks, levels=levels)


def load_descriptors_texture(path, names, query=False, denoise=0,
                             blocks=DEFAULT_BLOCKS, n_coeffs=DEFAULT_DCT_COEFFS,
                             levels=DEFAULT_WAVELET_LEVELS):
    """Descriptors of every image of a folder, in numeric ID order.

    ``names`` is a list of methods; the result is one descriptor list per
    method. The median filter (``denoise``) is applied only when
    ``query=True``; museum images are never filtered.
    """
    ids, paths = _ordered_image_paths(path)
    functions = [build_descriptor(n, blocks, n_coeffs, levels) for n in names]
    descriptors = [[] for _ in names]
    for image_path in paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if query:
            rgb = median_denoise(rgb, denoise)
        for index, function in enumerate(functions):
            descriptors[index].append(function(rgb))
    return ids, descriptors


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _check_distance(name, distance_name):
    """Reject a histogram-only distance for a method with signed features."""
    if name in SIGNED_METHODS and distance_name in NON_NEGATIVE_DISTANCES:
        raise ValueError(
            f"{distance_name} needs non-negative histograms, but {name} has "
            f"signed coefficients: use L1, Euclidean, Cosine or Correlation")


def _distance_names(distance_name):
    from k_similarity import DISTANCE_FUNCTIONS
    if distance_name == DISTANCE_ALL:
        return list(DISTANCE_FUNCTIONS)
    return [distance_name]



def evaluate_table_texture(museum_path, queries_path, methods, distances,
                           top_k=5, ground_truth_path=None, denoise=0,
                           blocks=DEFAULT_BLOCKS, n_coeffs=DEFAULT_DCT_COEFFS,
                           levels=DEFAULT_WAVELET_LEVELS):
    """mAP@1 / mAP@K of every method x distance; prints a table.

    Descriptors are computed once per method. Returns
    ``{(method, distance): (map1, mapk)}``. Needs a ground truth.
    """
    from k_similarity import (DISTANCE_FUNCTIONS, mean_average_precision_at_k,
                              retrieve_all_queries)

    names = [canonical_method_name(m) for m in methods]
    museum_ids, museum = load_descriptors_texture(
        museum_path, names, query=False, blocks=blocks, n_coeffs=n_coeffs,
        levels=levels)
    query_ids, queries = load_descriptors_texture(
        queries_path, names, query=True, denoise=denoise, blocks=blocks,
        n_coeffs=n_coeffs, levels=levels)
    ground_truth = _load_ground_truth(ground_truth_path, query_ids)
    if ground_truth is None:
        raise ValueError("A ground truth (--ground-truth) is needed for the "
                         "comparison table")
    results = {}
    print(f"{len(museum_ids)} museum images, {len(query_ids)} queries, "
          f"{blocks} blocks, median filter on queries: "
          f"{'off' if not denoise or denoise <= 1 else f'{denoise}x{denoise}'}")
    print(f"{'Method':<16}{'Distance':<24}{'dim':>7}"
          f"{'mAP@1':>9}{'mAP@' + str(top_k):>9}")
    for index, name in enumerate(names):
        for distance_name in distances:
            if (name in SIGNED_METHODS
                    and distance_name in NON_NEGATIVE_DISTANCES):
                print(f"{name:<16}{distance_name:<24}{'n/a (signed)':>25}")
                continue
            rankings = retrieve_all_queries(
                queries[index], museum[index],
                DISTANCE_FUNCTIONS[distance_name], top_k=top_k)
            ranked = [[museum_ids[i] for i, _ in row] for row in rankings]
            map1 = mean_average_precision_at_k(ranked, ground_truth, 1)
            mapk = mean_average_precision_at_k(ranked, ground_truth, top_k)
            results[(name, distance_name)] = (map1, mapk)
            print(f"{name:<16}{distance_name:<24}{len(museum[index][0]):>7}"
                  f"{map1:>9.4f}{mapk:>9.4f}")
    return results




# ===========================================================================
# PART 2 - PAINTINGS (Tasks 3, 4 and 5)
# ===========================================================================

DEFAULT_METHOD = FUSION_METHOD
DEFAULT_RETRIEVAL_DISTANCE = "L1"


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


def _texture_descriptor(method=DEFAULT_METHOD, blocks=DEFAULT_BLOCKS,
                        n_coeffs=DEFAULT_DCT_COEFFS,
                        levels=DEFAULT_WAVELET_LEVELS):
    """Build a W3 texture descriptor with the RGB callable interface."""
    return build_descriptor(method, blocks=blocks, n_coeffs=n_coeffs,
                            levels=levels)


def _descriptor_vector(descriptor, rgb):
    vector = np.asarray(descriptor(rgb), dtype=np.float32)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("Descriptors must return a nonempty, finite 1D vector")
    return vector


def retrieve_folder_w3(museum_path, queries_path, out_dir, descriptor=None,
                       distance_name=DEFAULT_RETRIEVAL_DISTANCE, top_k=10, ground_truth_path=None,
                       background_removal=True, params=None, query_filter=None,
                       viz_dir=None, descriptor_name="W3 Texture fusion"):
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
    descriptor = descriptor or _texture_descriptor()
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
                        default=Path("results/week3/QSD2/texture_baseline"))
    descriptor_group = parser.add_mutually_exclusive_group()
    descriptor_group.add_argument("--method",
                                  choices=TEXTURE_METHOD_NAMES + ("all",))
    descriptor_group.add_argument("--descriptor", type=_import_callable,
                                  help="Task 2 RGB descriptor, module:function")
    parser.add_argument("--query-filter", type=_import_callable,
                        help="Task 1 RGB filter, e.g. methods_w3:median_filter")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--n-coeffs", type=int, default=DEFAULT_DCT_COEFFS)
    parser.add_argument("--levels", type=int, default=DEFAULT_WAVELET_LEVELS)
    parser.add_argument("--distance",
                        choices=sorted(DISTANCE_FUNCTIONS) + [DISTANCE_ALL],
                        default=DEFAULT_RETRIEVAL_DISTANCE,
                        help="Distance, or 'all' with --table to compare every one")
    parser.add_argument("--list", action="store_true",
                        help="List the texture methods and exit")
    parser.add_argument("--table", action="store_true",
                        help="Task 2: method x distance mAP table on the whole "
                             "query images (no segmentation)")
    parser.add_argument("--denoise", type=int, default=0,
                        help="Median size on the queries for --table: 0 = off, "
                             "or an odd number >= 3")
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main(args=None):
    parser = build_parser()
    opts = parser.parse_args(args)
    if opts.list:
        for name in TEXTURE_METHOD_NAMES:
            print(name)
        return
    if opts.segment:
        segment_folder_w3(opts.segment, opts.out, opts.viz)
        return
    if opts.segmentation:
        evaluate_segmentation(*opts.segmentation, save_path=opts.save)
        return
    if not opts.museum or not opts.queries:
        parser.error("Use --segment, --segmentation, or --museum and --queries")
    if opts.method == "all" and opts.descriptor:
        parser.error("--method all cannot be combined with --descriptor")
    methods = (TEXTURE_METHOD_NAMES if opts.method == "all"
               else (opts.method or DEFAULT_METHOD,))
    if opts.table:
        evaluate_table_texture(
            opts.museum, opts.queries, methods, _distance_names(opts.distance),
            opts.top_k, opts.ground_truth, opts.denoise, opts.blocks,
            opts.n_coeffs, opts.levels)
        return
    if opts.distance == DISTANCE_ALL:
        parser.error("--distance all needs --table")
    for method in methods:
        if not opts.descriptor:
            _check_distance(method, opts.distance)
        descriptor = (opts.descriptor or
                      _texture_descriptor(method, opts.blocks, opts.n_coeffs,
                                          opts.levels))
        name = (f"{opts.descriptor.__module__}:{opts.descriptor.__name__}"
                if opts.descriptor else
                f"W3 {method}, blocks={opts.blocks}, levels={opts.levels}")
        output_dir = (opts.output_dir / method.lower().replace(" ", "_")
                      if opts.method == "all" else opts.output_dir)
        method_viz = (opts.viz / method.lower().replace(" ", "_")
                      if opts.viz is not None and opts.method == "all"
                      else opts.viz)
        retrieve_folder_w3(
            opts.museum, opts.queries, output_dir, descriptor=descriptor,
            distance_name=opts.distance, top_k=opts.top_k,
            ground_truth_path=opts.ground_truth,
            background_removal=opts.background_removal,
            query_filter=opts.query_filter, viz_dir=method_viz,
            descriptor_name=name)


if __name__ == "__main__":
    main()