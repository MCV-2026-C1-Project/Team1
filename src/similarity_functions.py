# Task 2: Similarity measures. Comparing 1D histograms

import numpy as np

def euclidean_distance(hist_a, hist_b):
    """Euclidean distance: lower is better."""
    hist_a = np.asarray(hist_a, dtype=float)
    hist_b = np.asarray(hist_b, dtype=float)
    return np.sqrt(np.sum((hist_a - hist_b) ** 2))


def l1_distance(hist_a, hist_b):
    """L1 distance: lower is better."""
    hist_a = np.asarray(hist_a, dtype=float)
    hist_b = np.asarray(hist_b, dtype=float)
    return np.sum(np.abs(hist_a - hist_b))


def chi_square_distance(hist_a, hist_b):
    """Chi-square distance: lower is better."""
    hist_a = np.asarray(hist_a, dtype=float)
    hist_b = np.asarray(hist_b, dtype=float)
    total = hist_a + hist_b
    mask = total > 0
    return np.sum((hist_a[mask] - hist_b[mask]) ** 2 / total[mask])


def histogram_intersection(hist_a, hist_b):
    """Histogram intersection: higher is better."""
    hist_a = np.asarray(hist_a, dtype=float)
    hist_b = np.asarray(hist_b, dtype=float)
    return np.sum(np.minimum(hist_a, hist_b))


def hellinger_kernel(hist_a, hist_b):
    """Hellinger kernel: higher is better."""
    hist_a = np.asarray(hist_a, dtype=float)
    hist_b = np.asarray(hist_b, dtype=float)
    return np.sum(np.sqrt(hist_a * hist_b))

def cosine_similarity(hist_a, hist_b):
    """Cosine similarity: higher is better."""
    hist_a = np.asarray(hist_a, dtype=float)
    hist_b = np.asarray(hist_b, dtype=float)
    denom = np.linalg.norm(hist_a) * np.linalg.norm(hist_b)
    return np.dot(hist_a, hist_b) / denom if denom > 0 else 0.0


def correlation_similarity(hist_a, hist_b):
    """Pearson correlation: higher is better."""
    hist_a = np.asarray(hist_a, dtype=float)
    hist_b = np.asarray(hist_b, dtype=float)
    return cosine_similarity(hist_a - hist_a.mean(), hist_b - hist_b.mean())