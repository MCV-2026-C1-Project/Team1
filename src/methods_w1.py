"""Methods_W1: descriptores ya implementados y validados en Week 1.

No modificar sin actualizar los tests de tests/test_descriptors.py y
tests/test_task4.py. Los descriptores W2 viven en methods_w2.py.

Orden canonico (DESCRIPTOR_NAMES):
    0. Grayscale  (256 bins, L1)
    1. HSV        (H 30 + S 32 + V 32 = 94, 1D marginal concatenado, L1)
    2. RGB        (32*3 = 96, 1D marginal concatenado, L1)
    3. HSV grid  (4x4 bloques, por bloque H 16 + S 8 + V 8 = 32 -> 512, L1 por
                  bloque y L1 global)

En QSD1 el mejor fue HSV grid (mAP@5 0.82, L1) y el mejor global HSV
(mAP@5 0.59, L1). Por eso la entrega QST1-W1 usa solo HSV y HSV grid.
"""

import cv2

from utils import (
    compute_color_hsv_histogram,
    compute_color_rgb_histogram,
    compute_gray_histogram,
    compute_hsv_grid_histogram,
)

# Nombre -> funcion(img) donde img es RGB salvo HSV/grid que esperan HSV.
# Se mantiene como dict ordenado para que el orden sea explicito.
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
    """Task 1 (W1): un descriptor por metodo W1, en orden METHODS_W1_NAMES."""
    methods = [[] for _ in METHODS_W1_NAMES]
    for img in dataset.images:
        for i, name in enumerate(METHODS_W1_NAMES):
            methods[i].append(METHODS_W1[name](img))
    return methods
