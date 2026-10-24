"""HSV hue statistics and global hue-scale correction.

The retrieval descriptor is never computed here: this module only estimates
and applies a global hue transform.  The corrected RGB image is later turned
into the usual "Lab blocks 1D" descriptor, so retrieval is still done in LAB.

OpenCV stores the hue channel of an 8-bit HSV image in ``[0, 179]``.  All
functions in this module use that convention unless ``bins`` is passed
explicitly to :func:`hue_histogram`.
"""

from __future__ import annotations

import cv2
import numpy as np

HUE_BINS = 180
DEFAULT_SATURATION_THRESHOLD = 40
DEFAULT_SCALE_MIN = 0.75
DEFAULT_SCALE_MAX = 1.30
DEFAULT_SCALE_STEP = 0.005
DEFAULT_REGULARIZATION = 0.10


def _check_rgb(rgb):
    image = np.asarray(rgb)
    assert image.dtype == np.uint8, (
        f"expected a uint8 RGB image, got {image.dtype}")
    assert image.ndim == 3 and image.shape[2] == 3, (
        f"expected an HxWx3 RGB image, got {image.shape}")
    return image


def _check_bins(bins):
    assert isinstance(bins, int) and not isinstance(bins, bool) and bins > 0, (
        f"bins must be a positive integer, got {bins!r}")
    return bins


def hue_histogram(rgb, bins=HUE_BINS,
                  saturation_threshold=DEFAULT_SATURATION_THRESHOLD):
    """Saturation-weighted hue histogram, L1-normalized.

    Pixels with ``S <= saturation_threshold`` are ignored, so white walls,
    black borders and other nearly neutral areas do not pull the histogram.
    """
    image = _check_rgb(rgb)
    bins = _check_bins(bins)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int64)
    saturation = hsv[..., 1].astype(np.float64)
    mask = saturation > saturation_threshold
    if not np.any(mask):
        return np.zeros(bins, dtype=np.float64)

    hue_bins = np.minimum((hue * bins) // HUE_BINS, bins - 1)
    hist = np.bincount(
        hue_bins[mask], weights=saturation[mask], minlength=bins
    ).astype(np.float64)
    total = hist.sum()
    return hist / total if total > 0 else hist


def hue_scale_grid(scale_min=DEFAULT_SCALE_MIN, scale_max=DEFAULT_SCALE_MAX,
                   scale_step=DEFAULT_SCALE_STEP):
    """Candidate multiplicative corrections for the hue channel."""
    assert 0 < scale_min <= 1.0 <= scale_max, (
        f"scale range must contain 1.0, got {scale_min!r}..{scale_max!r}")
    assert scale_step > 0, f"scale_step must be positive, got {scale_step!r}"
    values = np.arange(scale_min, scale_max + scale_step / 2.0, scale_step)
    values = np.round(values, 6)
    # Make the identity correction exact even with floating-point grids.
    values = np.unique(np.append(values, 1.0))
    return values


def warp_hue_histogram(hist, scale, bins=None):
    """Apply ``hue -> scale * hue (mod bins)`` to a hue histogram.

    This is a nearest-bin approximation of the correction.  It is used only
    to estimate a global scale; the actual correction is applied per pixel
    in :func:`correct_hue_scale`.
    """
    hist = np.asarray(hist, dtype=np.float64)
    assert hist.ndim == 1, f"hist must be 1-D, got {hist.shape}"
    assert scale > 0, f"scale must be positive, got {scale!r}"
    bins = len(hist) if bins is None else _check_bins(bins)
    assert bins == len(hist), (
        f"bins={bins} does not match histogram length {len(hist)}")
    warped = np.zeros_like(hist)
    source = np.arange(bins, dtype=np.float64)
    destination = np.rint((source * scale) % bins).astype(np.int64) % bins
    np.add.at(warped, destination, hist)
    return warped


def estimate_hue_scale(query_hist, reference_hist, scales=None,
                       regularization=DEFAULT_REGULARIZATION):
    """Estimate the hue scale that best aligns a query with a reference.

    The query histogram is warped by every candidate scale ``s`` and compared
    with ``reference_hist``.  A small quadratic penalty toward ``s = 1``
    avoids large corrections when the histograms are already close.

    Returns ``(scale, score)``, where lower scores are better.
    """
    query = np.asarray(query_hist, dtype=np.float64)
    reference = np.asarray(reference_hist, dtype=np.float64)
    assert query.shape == reference.shape and query.ndim == 1, (
        "query and reference histograms must be 1-D and have the same length")
    if query.sum() <= 0 or reference.sum() <= 0:
        return 1.0, float("inf")
    scales = hue_scale_grid() if scales is None else np.asarray(
        scales, dtype=np.float64)
    best_scale = 1.0
    best_score = float("inf")
    for scale in scales:
        warped = warp_hue_histogram(query, float(scale), bins=len(query))
        score = float(np.abs(warped - reference).sum())
        score += float(regularization) * float(scale - 1.0) ** 2
        if score < best_score:
            best_score = score
            best_scale = float(scale)
    return best_scale, best_score


def correct_hue_scale(rgb, scale,
                      saturation_threshold=DEFAULT_SATURATION_THRESHOLD):
    """Apply a global hue scaling to an RGB uint8 image.

    Only pixels with ``S > saturation_threshold`` are modified; saturation and
    value are kept.  The result is an RGB uint8 image ready for ``RGB -> LAB``.
    """
    image = _check_rgb(rgb)
    assert scale > 0, f"scale must be positive, got {scale!r}"
    if np.isclose(scale, 1.0):
        return image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float64)
    saturation = hsv[..., 1]
    corrected = (hue * float(scale)) % HUE_BINS
    out_hue = hue.copy()
    mask = saturation > saturation_threshold
    out_hue[mask] = corrected[mask]
    hsv[..., 0] = np.clip(np.rint(out_hue), 0, HUE_BINS - 1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def estimate_and_correct(rgb, reference_hist, scales=None,
                         saturation_threshold=DEFAULT_SATURATION_THRESHOLD,
                         regularization=DEFAULT_REGULARIZATION):
    """Convenience wrapper returning ``(corrected_rgb, scale, score)``."""
    query_hist = hue_histogram(
        rgb, saturation_threshold=saturation_threshold)
    scale, score = estimate_hue_scale(
        query_hist, reference_hist, scales=scales,
        regularization=regularization)
    return correct_hue_scale(
        rgb, scale, saturation_threshold=saturation_threshold), scale, score
