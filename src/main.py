"""Entry point W1 (+ compatibilidad W2).

Separacion Week 2:
    - Methods_W1 (methods_w1.py): Grayscale, HSV, RGB, HSV grid. Congelados.
    - Methods_W2 (methods_w2.py): HSV 3D / 2D / bloque / piramide. En prueba.

DESCRIPTOR_NAMES y compute_descriptors() se conservan como alias de W1
para no romper k_similarity.py, task4.py ni los tests de Week 1.
El codigo nuevo debe importar explicitamente desde methods_w1 / methods_w2.
"""

from methods_w1 import METHODS_W1, METHODS_W1_NAMES, compute_descriptors_w1
from methods_w2 import METHODS_W2, METHODS_W2_NAMES, compute_descriptors_w2
from utils import Dataset, visualize_histograms
import cv2
import numpy as np

# Alias de compatibilidad W1.
DESCRIPTOR_NAMES = METHODS_W1_NAMES


def compute_descriptors(dataset):
    """ Task 1 (W1): alias de compute_descriptors_w1. """
    return compute_descriptors_w1(dataset)

def main():
    train = Dataset("data/BBDD")
    descriptors = compute_descriptors(train)

    """ Visualize histograms """
    idx = np.random.randint(0, len(train.images))
    visualize_histograms(train, idx)

if __name__ == "__main__":
    main()