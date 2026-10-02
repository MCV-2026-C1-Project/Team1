from utils import Dataset, compute_gray_histogram, compute_color_hsv_histogram, compute_color_rgb_histogram, visualize_histograms
import cv2
import numpy as np

def compute_descriptors(dataset):
    """ Task 1: Compute image descriptors (QSD1) (up to three methods)"""
    method1, method2, method3 = [], [], []
    
    for img in dataset.images:
        method1.append(compute_gray_histogram(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)))
        method2.append(compute_color_hsv_histogram(cv2.cvtColor(img, cv2.COLOR_RGB2HSV)))
        method3.append(compute_color_rgb_histogram(img))

    return method1, method2, method3

def main():
    train = Dataset("data/BBDD")
    descriptors_mthd1, descriptors_mthd2, descriptors_mthd3 = compute_descriptors(train)

    """ Visualize histograms """
    idx = np.random.randint(0, len(train.images))
    visualize_histograms(train, idx)

if __name__ == "__main__":
    main()