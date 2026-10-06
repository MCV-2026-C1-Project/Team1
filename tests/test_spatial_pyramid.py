"""Synthetic sanity checks; no datasets or output files required."""

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_pyramid import compute_spatial_pyramid
from utils import compute_color_hsv_histogram
from k_similarity import retrieve_all_queries, mean_average_precision_at_k, DISTANCE_FUNCTIONS


class SpatialPyramidTests(unittest.TestCase):
    def test_dimensions_and_normalization(self):
        image = np.full((9, 11, 3), 100, dtype=np.uint8)
        for space in ("RGB", "HSV", "Lab", "YCrCb", "YCbCr"):
            for mode, dimension in (("1d", 21 * 24), ("3d", 10752)):
                result = compute_spatial_pyramid(image, color_space=space,
                                                 histogram_type=mode, bins=8)
                self.assertEqual(result.shape, (dimension,))
                self.assertTrue(np.isfinite(result).all())
                self.assertAlmostEqual(float(result.sum()), 1, places=5)
        result = compute_spatial_pyramid(image, normalization="l2")
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1, places=5)

    def test_global_matches_week1(self):
        image = np.random.default_rng(42).integers(0, 256, (9, 11, 3), dtype=np.uint8)
        expected = compute_color_hsv_histogram(cv2.cvtColor(image, cv2.COLOR_RGB2HSV), 30, 32, 32)
        actual = compute_spatial_pyramid(image, pyramid_levels=[1], bins=[30, 32, 32])
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_joint_captures_channel_relationships(self):
        a = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        b = np.array([[[0, 255, 0], [255, 0, 255]]], dtype=np.uint8)
        config = dict(pyramid_levels=[1], color_space="RGB", bins=2)
        np.testing.assert_allclose(compute_spatial_pyramid(a, **config),
                                   compute_spatial_pyramid(b, **config))
        self.assertFalse(np.array_equal(compute_spatial_pyramid(a, histogram_type="3d", **config),
                                        compute_spatial_pyramid(b, histogram_type="3d", **config)))

    def test_cell_order_and_full_coverage(self):
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        colors = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for index, color in enumerate(colors):
            y, x = divmod(index, 2)
            image[y * 5 // 2:(y + 1) * 5 // 2, x * 7 // 2:(x + 1) * 7 // 2] = color
        result = compute_spatial_pyramid(image, pyramid_levels=[1, 2],
                                         histogram_type="3d", color_space="RGB",
                                         bins=2, normalization="none").reshape(5, 8)
        np.testing.assert_array_equal(result[1:].argmax(axis=1), [0, 1, 2, 4])
        self.assertEqual(float(result.sum()), 2 * 5 * 7)
        np.testing.assert_array_equal(result[0], result[1:].sum(axis=0))

    def test_invalid_settings(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        for config in (dict(pyramid_levels=[0]), dict(pyramid_levels=[1, 1]),
                       dict(pyramid_levels=[8]), dict(bins=[8, 8]), dict(bins=0),
                       dict(color_space="invalid"), dict(histogram_type="2d"),
                       dict(normalization="invalid")):
            with self.subTest(config=config), self.assertRaises(ValueError):
                compute_spatial_pyramid(image, **config)

    def test_week1_retrieval_integration(self):
        images = [np.full((4, 4, 3), value, dtype=np.uint8) for value in (0, 255)]
        descriptors = [compute_spatial_pyramid(image, color_space="RGB", bins=2)
                       for image in images]
        for function in DISTANCE_FUNCTIONS.values():
            rankings = retrieve_all_queries(descriptors, descriptors, function, top_k=2)
            ids = [[index for index, score in row] for row in rankings]
            self.assertEqual(mean_average_precision_at_k(ids, [[0], [1]], 1), 1)


if __name__ == "__main__":
    unittest.main()
