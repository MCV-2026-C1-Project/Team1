"""Verify prefix reuse and persistent descriptor caching on tiny images."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_pyramid import compute_spatial_pyramid
from spatial_pyramid_sweep import pyramid_prefix, load_sweep_descriptors


class SweepTests(unittest.TestCase):
    def test_prefix_matches_direct_descriptor(self):
        image = np.random.default_rng(7).integers(0, 256, (9, 11, 3), dtype=np.uint8)
        for mode in ("1d", "3d"):
            for normalization in ("l1", "l2", "none"):
                full = compute_spatial_pyramid(image, histogram_type=mode, bins=4,
                                               normalization=normalization)
                for levels in ([1], [1, 2], [1, 2, 4]):
                    direct = compute_spatial_pyramid(image, pyramid_levels=levels,
                                                     histogram_type=mode, bins=4,
                                                     normalization=normalization)
                    reused = pyramid_prefix(np.asarray([full]), levels, mode, 4, normalization)[0]
                    np.testing.assert_allclose(reused, direct, atol=1e-7, rtol=1e-6)

    def test_cache_and_numeric_order(self):
        with TemporaryDirectory() as root:
            images, cache = Path(root) / "images", Path(root) / "cache"
            images.mkdir()
            for index in (10, 2):
                cv2.imwrite(str(images / f"{index}.jpg"), np.full((4, 4, 3), index, np.uint8))
            configurations = [("1d", "RGB", 2), ("3d", "HSV", 2)]
            ids, descriptors = load_sweep_descriptors(images, configurations, [1, 2], "l1", cache)
            self.assertEqual(ids, [2, 10])
            with patch("spatial_pyramid_sweep.load_image", side_effect=AssertionError("Decoded cached image")):
                cached_ids, cached = load_sweep_descriptors(images, configurations, [1, 2], "l1", cache)
            self.assertEqual(cached_ids, ids)
            for config in configurations:
                np.testing.assert_array_equal(cached[config], descriptors[config])


if __name__ == "__main__":
    unittest.main()
