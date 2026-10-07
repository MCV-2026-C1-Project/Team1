"""W1 entry point (+ W2 compatibility).

Week 2 split:
    - Methods_W1 (methods_w1.py): Grayscale, HSV, RGB, HSV grid. Frozen.
    - Methods_W2 (methods_w2.py): global + pyramid HSV 3D / 2D. Under test.

DESCRIPTOR_NAMES and compute_descriptors() are kept as W1 aliases
so k_similarity.py, task4.py and the Week 1 tests keep working.
New code must import explicitly from methods_w1 / methods_w2.
"""

from methods_w1 import METHODS_W1, METHODS_W1_NAMES, compute_descriptors_w1
from methods_w2 import METHODS_W2, METHODS_W2_NAMES, compute_descriptors_w2
from utils import Dataset, visualize_histograms
import cv2
import numpy as np

# W1 compatibility alias.
DESCRIPTOR_NAMES = METHODS_W1_NAMES


def compute_descriptors(dataset):
    """ Task 1 (W1): alias for compute_descriptors_w1. """
    return compute_descriptors_w1(dataset)

def main():
    train = Dataset("data/BBDD")
    descriptors = compute_descriptors(train)

    """ Visualize histograms """
    idx = np.random.randint(0, len(train.images))
    visualize_histograms(train, idx)

if __name__ == "__main__":
    main()
