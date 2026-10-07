"""Methods_W2: color descriptors to test in Week 2 (Task 1).

Functions take images in their own color space (HSV or Lab) and accept
an optional foreground mask (Task 3/5):
if ``mask`` is None the full image is used; if given, only foreground
pixels contribute to the histogram.

HSV families (all L1 normalized):
- Globals (no spatial subdivision).
- Pyramids: levels (1, 2, 4) = 21 cells; each cell carries the same
  global combination, L1-normalized per cell, L1-renormalized on
  concatenation (no level weights, as in ``spatial_pyramid.py``).

HSV entries in METHODS_W2:
Globals validated on QSD1 with L1 only (see results/week2/color_sweep_l1.txt;
W1 baselines: marginal HSV mAP@5 0.59, HSV grid mAP@5 0.82):
    - "HSV 3D 4x4x4" (64): best 3D (0.5389); finer bins scatter under L1.
    - "HSV 2D HS 30x32" (960): best single 2D (0.5067).
    - "HSV 2D HV 8x8" (64): best HV (0.4883); SV discarded alone (<=0.41).
    - "HSV 2D HS+HV 16x16" (512): best W2 global (mAP@5 0.6167, mAP@1
      0.5667); beats the W1 marginal. HS+HV+SV (0.5528) performs worse
      in this comparison and is not registered.
Pyramids mirror those 4 combinations (21 cells), validated on QSD1-W2,
L1 only (see results/week2/qsd1w2_registry_l1.txt):
    - "HSV pyramid 3D 4x4x4" (1344): mAP@5 0.7067.
    - "HSV pyramid HS 30x32" (20160): mAP@5 0.6233.
    - "HSV pyramid HV 8x8" (1344): mAP@5 0.6167.
    - "HSV pyramid HS+HV 16x16" (10752): mAP@5 0.7722, best HSV W2.

Lab entries use three marginal 1D histograms (32 bins per channel):
    - "Lab block 4x4 1D 32" (1536): mAP@1 0.8333, mAP@5 0.8722.
    - "Lab block 6x6 1D 32" (3456): mAP@1 0.8667, mAP@5 0.9000.
    - "Lab pyramid 1D 32" (2016): mAP@1 0.8333, mAP@5 0.8694.
These are QSD1 development results with L1 distance, not blind test scores.

Masked variants (QSD2/QST2) reuse these same functions by passing
``mask``; they are not separate registry entries until the Task 1
ablation is closed on QSD1-W2 (same dev set as W1, direct comparison).
"""

import cv2
import numpy as np

from utils import _l1_normalize

# OpenCV ranges for HSV (H: 0-180, S/V: 0-256).
_HSV_RANGES = {0: (0, 180), 1: (0, 256), 2: (0, 256)}


def _ranges_for(channels):
    ranges = []
    for ch in channels:
        lo, hi = _HSV_RANGES[ch]
        ranges.extend([lo, hi])
    return ranges


def compute_hsv_3d_histogram(img_hsv, bins=(8, 8, 8), mask=None):
    """Joint 3D H-S-V histogram, L1 normalized."""
    assert len(img_hsv.shape) == 3, "Expected HSV image"
    hist = cv2.calcHist(
        [img_hsv], [0, 1, 2], mask, list(bins),
        [0, 180, 0, 256, 0, 256],
    ).flatten()
    return _l1_normalize(hist)


def compute_hsv_2d_histogram(img_hsv, channels=(0, 1), bins=(16, 16), mask=None):
    """Joint 2D histogram (H-S by default), L1 normalized."""
    assert len(img_hsv.shape) == 3, "Expected HSV image"
    assert len(channels) == 2 and len(bins) == 2
    hist = cv2.calcHist(
        [img_hsv], list(channels), mask, list(bins),
        _ranges_for(channels),
    ).flatten()
    return _l1_normalize(hist)


def compute_hsv_2d_concat_histogram(img_hsv, parts=(((0, 1), (16, 16)),), mask=None):
    """Concatenate several 2D histograms (e.g. HS+HV), L1 normalized.

    parts: sequence of (channels, bins), e.g. [((0,1),(16,16)),
        ((0,2),(16,16))] for HS+HV.
    Each part is L1-normalized before concatenation and the final vector is
    L1-renormalized: each 2D pair carries equal weight (1/N), just as each
    channel carries 1/3 in the W1 marginal and each block carries equal
    weight in the W1 grid.
    """
    assert len(img_hsv.shape) == 3, "Expected HSV image"
    assert len(parts) >= 1
    normed_parts = [
        _l1_normalize(compute_hsv_2d_histogram(
            img_hsv, channels=ch, bins=b, mask=mask
        ))
        for ch, b in parts
    ]
    return _l1_normalize(np.concatenate(normed_parts))


def _concat_fn(*parts):
    """Build fn(hsv, mask) for a 2D concat with equal weight per pair."""
    spec = list(parts)
    return lambda hsv, mask=None: compute_hsv_2d_concat_histogram(
        hsv, parts=spec, mask=mask
    )


def _split_with_mask(image, mask, grid_y, grid_x):
    """Yield (block, block_mask) pairs; block_mask is None if mask is None."""
    height, width = image.shape[:2]
    for y in range(grid_y):
        for x in range(grid_x):
            y0, y1 = y * height // grid_y, (y + 1) * height // grid_y
            x0, x1 = x * width // grid_x, (x + 1) * width // grid_x
            m = None if mask is None else mask[y0:y1, x0:x1]
            yield image[y0:y1, x0:x1], m


def _compute_spatial_histogram(image, levels, block_fn, mask):
    parts = [
        _l1_normalize(block_fn(block, m))
        for grid in levels
        for block, m in _split_with_mask(image, mask, grid, grid)
    ]
    return _l1_normalize(np.concatenate(parts))


def compute_hsv_pyramid_histogram(img_hsv, levels=(1, 2, 4),
                                  block_fn=None, mask=None):
    """Spatial pyramid of one HSV combination, L1 normalized.

    levels: grid per level, e.g. (1, 2, 4) -> 1 + 4 + 16 = 21 cells.
    block_fn(block_hsv, block_mask) -> 1D vector; defaults to HS 8x8.
    Each cell is L1-normalized and the concatenation is L1-renormalized,
    so each cell carries equal mass (finer levels weigh more in total,
    as in the W1 grid and ``spatial_pyramid.py``).
    """
    assert len(img_hsv.shape) == 3, "Expected HSV image"
    if block_fn is None:
        block_fn = lambda b, m: compute_hsv_2d_histogram(
            b, channels=(0, 1), bins=(8, 8), mask=m
        )
    return _compute_spatial_histogram(img_hsv, levels, block_fn, mask)


def compute_lab_histogram(img_lab, bins=(32, 32, 32), mask=None):
    """Marginal L, a and b histograms for uint8 OpenCV Lab, L1 normalized."""
    assert len(img_lab.shape) == 3, "Expected Lab image"
    parts = [
        cv2.calcHist([img_lab], [ch], mask, [b], [0, 256]).flatten()
        for ch, b in enumerate(bins)
    ]
    return _l1_normalize(np.concatenate(parts))


def compute_lab_pyramid_histogram(img_lab, levels=(1, 2, 4),
                                  bins=(32, 32, 32), mask=None):
    """Lab marginals per cell, with the same normalization as HSV pyramids."""
    block_fn = lambda b, m: compute_lab_histogram(b, bins=bins, mask=m)
    return _compute_spatial_histogram(img_lab, levels, block_fn, mask)


def compute_lab_block_histogram(img_lab, grid=4, bins=(32, 32, 32), mask=None):
    """Single grid of Lab marginals, concatenated in row-major order."""
    return compute_lab_pyramid_histogram(
        img_lab, levels=(grid,), bins=bins, mask=mask
    )


METHODS_W2 = {
    "HSV 3D 4x4x4": lambda hsv, mask=None: compute_hsv_3d_histogram(
        hsv, bins=(4, 4, 4), mask=mask
    ),
    "HSV 2D HS 30x32": lambda hsv, mask=None: compute_hsv_2d_histogram(
        hsv, channels=(0, 1), bins=(30, 32), mask=mask
    ),
    "HSV 2D HV 8x8": lambda hsv, mask=None: compute_hsv_2d_histogram(
        hsv, channels=(0, 2), bins=(8, 8), mask=mask
    ),
    "HSV 2D HS+HV 16x16": _concat_fn(((0, 1), (16, 16)), ((0, 2), (16, 16))),
    "HSV pyramid 3D 4x4x4": lambda hsv, mask=None: compute_hsv_pyramid_histogram(
        hsv, levels=(1, 2, 4),
        block_fn=lambda b, m: compute_hsv_3d_histogram(b, bins=(4, 4, 4), mask=m),
        mask=mask,
    ),
    "HSV pyramid HS 30x32": lambda hsv, mask=None: compute_hsv_pyramid_histogram(
        hsv, levels=(1, 2, 4),
        block_fn=lambda b, m: compute_hsv_2d_histogram(
            b, channels=(0, 1), bins=(30, 32), mask=m),
        mask=mask,
    ),
    "HSV pyramid HV 8x8": lambda hsv, mask=None: compute_hsv_pyramid_histogram(
        hsv, levels=(1, 2, 4),
        block_fn=lambda b, m: compute_hsv_2d_histogram(
            b, channels=(0, 2), bins=(8, 8), mask=m),
        mask=mask,
    ),
    "HSV pyramid HS+HV 16x16": lambda hsv, mask=None: compute_hsv_pyramid_histogram(
        hsv, levels=(1, 2, 4),
        block_fn=lambda b, m: compute_hsv_2d_concat_histogram(
            b, parts=[((0, 1), (16, 16)), ((0, 2), (16, 16))], mask=m),
        mask=mask,
    ),
    "Lab block 4x4 1D 32": lambda lab, mask=None: compute_lab_block_histogram(
        lab, grid=4, mask=mask
    ),
    "Lab block 6x6 1D 32": lambda lab, mask=None: compute_lab_block_histogram(
        lab, grid=6, mask=mask
    ),
    "Lab pyramid 1D 32": lambda lab, mask=None: compute_lab_pyramid_histogram(
        lab, mask=mask
    ),
}

METHODS_W2_NAMES = list(METHODS_W2.keys())
METHODS_W2_COLOR_SPACES = dict.fromkeys(METHODS_W2, "HSV")
METHODS_W2_COLOR_SPACES.update({
    "Lab block 4x4 1D 32": "Lab",
    "Lab block 6x6 1D 32": "Lab",
    "Lab pyramid 1D 32": "Lab",
})


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
        sources = {
            "HSV": cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
            "Lab": cv2.cvtColor(img, cv2.COLOR_RGB2LAB),
        }
        mask = None if masks is None else masks[i]
        for j, name in enumerate(METHODS_W2_NAMES):
            source = sources[METHODS_W2_COLOR_SPACES[name]]
            methods[j].append(METHODS_W2[name](source, mask=mask))
    return methods
