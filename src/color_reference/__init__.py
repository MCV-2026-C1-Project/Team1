"""HSV hue references and color-normalized LAB retrieval.

This package is intentionally separate from ``methods_w3``: it owns the
reference hue histograms and the HSV hue correction, while the actual
descriptor and distance remain the W3/W2 LAB ones.
"""

from .hue import (
    DEFAULT_REGULARIZATION,
    DEFAULT_SATURATION_THRESHOLD,
    DEFAULT_SCALE_MAX,
    DEFAULT_SCALE_MIN,
    DEFAULT_SCALE_STEP,
    HUE_BINS,
    correct_hue_scale,
    estimate_and_correct,
    estimate_hue_scale,
    hue_histogram,
    hue_scale_grid,
    warp_hue_histogram,
)
from .reference import (
    DEFAULT_REFERENCE_SOURCE,
    REFERENCE_SOURCES,
    HueReference,
    build_hue_references,
    build_hue_references_from_images,
)
from .retrieval import (
    HueRetrievalResult,
    confidence_weights,
    rank_with_hue_references,
)

__all__ = [
    "DEFAULT_REFERENCE_SOURCE",
    "DEFAULT_REGULARIZATION",
    "DEFAULT_SATURATION_THRESHOLD",
    "DEFAULT_SCALE_MAX",
    "DEFAULT_SCALE_MIN",
    "DEFAULT_SCALE_STEP",
    "HUE_BINS",
    "HueReference",
    "HueRetrievalResult",
    "REFERENCE_SOURCES",
    "build_hue_references",
    "build_hue_references_from_images",
    "confidence_weights",
    "correct_hue_scale",
    "estimate_and_correct",
    "estimate_hue_scale",
    "hue_histogram",
    "hue_scale_grid",
    "rank_with_hue_references",
    "warp_hue_histogram",
]
