"""Configurable color histograms over a spatial pyramid of RGB input images."""

from numbers import Integral

import cv2
import numpy as np

if __package__:
    from .utils import (compute_color_rgb_histogram, compute_color_hsv_histogram,
                        _split_in_blocks, _l1_normalize)
else:
    from utils import (compute_color_rgb_histogram, compute_color_hsv_histogram,
                       _split_in_blocks, _l1_normalize)


def _positive_integers(values, name):
    values = tuple(values)
    if not values or any(isinstance(v, bool) or not isinstance(v, Integral)
                         or v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def compute_spatial_pyramid(image_rgb, pyramid_levels=(1, 2, 4),
                            histogram_type="1d", color_space="HSV", bins=32,
                            normalization="l1"):
    """Return levels concatenated in supplied order, with row-major cells.

    Input is uint8 RGB, as returned by utils.load_image. Levels specify grid
    widths, rather than exponents. Bins accepts one integer or three integers.
    Each region is normalized independently (l1, l2, or none); concatenation
    receives the same normalization. With none, histogram counts are retained.
    No extra level weights are applied. Thus l1 gives each cell equal mass,
    and finer levels collectively receive more mass because they have more cells.
    HSV uses OpenCV's hue range [0,180); other uint8 channels use [0,256).
    YCbCr is reordered from OpenCV YCrCb to Y,Cb,Cr before histogramming.
    """
    image_rgb = np.asarray(image_rgb)
    if (image_rgb.ndim != 3 or image_rgb.shape[2] != 3
            or image_rgb.dtype != np.uint8 or image_rgb.size == 0):
        raise ValueError("Expected a nonempty H x W x 3 uint8 RGB image")
    levels = _positive_integers(pyramid_levels, "pyramid_levels")
    if len(set(levels)) != len(levels):
        raise ValueError("pyramid_levels must be unique")
    if max(levels) > min(image_rgb.shape[:2]):
        raise ValueError("Grid dimensions cannot exceed image dimensions")
    bin_counts = _positive_integers(
        (bins,) * 3 if isinstance(bins, Integral) else bins, "bins")
    if len(bin_counts) != 3:
        raise ValueError("bins must be an integer or three integers")
    histogram_type = histogram_type.lower()
    normalization = normalization.lower()
    if histogram_type not in ("1d", "3d"):
        raise ValueError("histogram_type must be 1d or 3d")
    if normalization not in ("l1", "l2", "none"):
        raise ValueError("normalization must be l1, l2, or none")
    space = color_space.upper()
    conversions = {"HSV": cv2.COLOR_RGB2HSV, "LAB": cv2.COLOR_RGB2LAB,
                   "YCRCB": cv2.COLOR_RGB2YCrCb, "YCBCR": cv2.COLOR_RGB2YCrCb}
    if space == "RGB":
        converted = image_rgb
    elif space in conversions:
        converted = cv2.cvtColor(image_rgb, conversions[space])
        if space == "YCBCR":
            converted = converted[:, :, [0, 2, 1]]
    else:
        raise ValueError("color_space must be RGB, HSV, Lab, YCrCb, or YCbCr")

    def normalize(vector):
        vector = np.asarray(vector, dtype=np.float32).flatten()
        if normalization == "l1":
            return _l1_normalize(vector)
        if normalization == "l2":
            return cv2.normalize(vector, None, alpha=1,
                                 norm_type=cv2.NORM_L2).flatten()
        return vector

    ranges = [0, 180 if space == "HSV" else 256, 0, 256, 0, 256]
    regions = []
    for grid in levels:
        for block in _split_in_blocks(converted, grid, grid):
            if histogram_type == "3d":
                histogram = cv2.calcHist([block], [0, 1, 2], None,
                                         list(bin_counts), ranges).flatten()
            else:
                # Lab/YCrCb/YCbCr uint8 channels share RGB histogram ranges.
                function = (compute_color_hsv_histogram if space == "HSV"
                            else compute_color_rgb_histogram)
                histogram = function(block, *bin_counts)
                if normalization == "none":
                    histogram = histogram * (3 * block.shape[0] * block.shape[1])
            regions.append(normalize(histogram))
    return normalize(np.concatenate(regions))
