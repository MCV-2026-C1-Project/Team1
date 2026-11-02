"""SuperPoint+LightGlue extraction and matching."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keypoints_superpoint import extract_superpoint, match_superpoint


def record(points, descriptors, box):
    return {"points": points, "descriptors": descriptors, "box": box}


class TestSuperPoint(unittest.TestCase):
    def test_flat_image_has_empty_descriptor_matrix(self):
        _, descriptors = extract_superpoint(
            np.zeros((100, 100, 3), np.uint8), max_num_keypoints=128)
        self.assertEqual(descriptors.shape, (0, 256))
        self.assertEqual(descriptors.dtype, np.float32)

    def test_extract_and_match_same_image(self):
        rng = np.random.default_rng(4)
        rgb = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
        _, descriptors = extract_superpoint(rgb, max_num_keypoints=128)
        self.assertEqual(descriptors.shape[1], 256)
        n = len(descriptors)
        if n == 0:
            self.skipTest("no SuperPoint features on synthetic image")
        rng2 = np.random.default_rng(5)
        points = (rng2.uniform(0, 160, (n, 2))).astype(np.float32)
        query = record(points, descriptors, [0, 0, 160, 160])
        result = match_superpoint(query, query)
        self.assertGreater(result["inlier_count"], 8)

    def test_empty_features_produce_no_matches(self):
        empty = record(np.empty((0, 2), np.float32),
                       np.empty((0, 256), np.float32), [0, 0, 100, 100])
        result = match_superpoint(empty, empty)
        self.assertEqual(result["tentative_count"], 0)
        self.assertEqual(result["inlier_count"], 0)


if __name__ == "__main__":
    unittest.main()
