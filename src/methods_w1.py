"""Methods_W1: descriptors implemented and validated in Week 1.

Do not modify without updating tests/test_descriptors.py and
tests/test_task4.py. W2 descriptors live in methods_w2.py.

Canonical order (DESCRIPTOR_NAMES):
    0. Grayscale  (256 bins, L1)
    1. HSV        (H 30 + S 32 + V 32 = 94, concatenated 1D marginal, L1)
    2. RGB        (32*3 = 96, concatenated 1D marginal, L1)
    3. HSV grid  (4x4 blocks, per block H 16 + S 8 + V 8 = 32 -> 512, L1 per
                  block and global L1)

On QSD1 the best was HSV grid (mAP@5 0.82, L1) and the best global was HSV
(mAP@5 0.59, L1). Hence the QST1-W1 submission uses only HSV and HSV grid.
"""

import cv2

from utils import (
    compute_color_hsv_histogram,
    compute_color_rgb_histogram,
    compute_gray_histogram,
    compute_hsv_grid_histogram,
)

# Name -> function(img) where img is RGB except HSV/grid which expect HSV.
# Kept as an ordered dict so the order is explicit.
METHODS_W1 = {
    "Grayscale": lambda img_rgb: compute_gray_histogram(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    ),
    "HSV": lambda img_rgb: compute_color_hsv_histogram(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    ),
    "RGB": compute_color_rgb_histogram,
    "HSV grid": lambda img_rgb: compute_hsv_grid_histogram(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    ),
}

METHODS_W1_NAMES = list(METHODS_W1.keys())


def compute_descriptors_w1(dataset):
    """Task 1 (W1): one descriptor per W1 method, in METHODS_W1_NAMES order."""
    methods = [[] for _ in METHODS_W1_NAMES]
    for img in dataset.images:
        for i, name in enumerate(METHODS_W1_NAMES):
            methods[i].append(METHODS_W1[name](img))
    return methods
