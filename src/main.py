from utils import (Dataset, compute_gray_histogram, compute_color_hsv_histogram, compute_color_rgb_histogram, compute_hsv_grid_histogram, visualize_histograms)
import cv2
import numpy as np

DESCRIPTOR_NAMES = ["Grayscale", "HSV", "RGB", "HSV grid"]

def compute_descriptors(dataset):
    """ Task 1: Compute image descriptors (QSD1). One list per method, in DESCRIPTOR_NAMES order."""
    methods = [[] for _ in DESCRIPTOR_NAMES]

    for img in dataset.images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        methods[0].append(compute_gray_histogram(img_gray))
        methods[1].append(compute_color_hsv_histogram(img_hsv))
        methods[2].append(compute_color_rgb_histogram(img))
        methods[3].append(compute_hsv_grid_histogram(img_hsv))

    return methods

def main():
    train = Dataset("data/BBDD")
    descriptors = compute_descriptors(train)

    """ Visualize histograms """
    idx = np.random.randint(0, len(train.images))
    visualize_histograms(train, idx)

if __name__ == "__main__":
    main()