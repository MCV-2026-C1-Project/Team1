"""Tests Methods_W2 (methods_w2.py): formas, normalizacion L1 y mascaras."""
import os
import sys
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from methods_w2 import (  # noqa: E402
    METHODS_W2,
    METHODS_W2_NAMES,
    compute_descriptors_w2,
    compute_hsv_2d_histogram,
    compute_hsv_3d_histogram,
    compute_hsv_block_2d_hs_histogram,
    compute_hsv_block_3d_histogram,
    compute_hsv_pyramid_histogram,
)


def _red_hsv(h=32, w=32):
    rgb = np.full((h, w, 3), (255, 0, 0), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


class TestW2Shapes(unittest.TestCase):
    def test_registry_is_hsv_only_five_methods(self):
        self.assertEqual(len(METHODS_W2_NAMES), 5)
        self.assertTrue(all(n.startswith("HSV") for n in METHODS_W2_NAMES))

    def test_shapes_and_l1(self):
        hsv = _red_hsv()
        expected = {
            "HSV 3D 8x8x8": (512,),
            "HSV 2D HS 30x32": (960,),
            "HSV block 4x4 HS 8x8": (1024,),
            "HSV block 4x4 3D 4x4x4": (1024,),
            "HSV pyramid HS 8x8": (1344,),
        }
        for name, shape in expected.items():
            v = METHODS_W2[name](hsv)
            self.assertEqual(v.shape, shape, name)
            self.assertAlmostEqual(float(v.sum()), 1.0, places=5, msg=name)
            self.assertTrue(np.all(v >= 0))

    def test_3d_custom_bins(self):
        v = compute_hsv_3d_histogram(_red_hsv(), bins=(4, 4, 4))
        self.assertEqual(v.shape, (64,))

    def test_2d_custom_channels(self):
        hsv = _red_hsv()
        v = compute_hsv_2d_histogram(hsv, channels=(1, 2), bins=(8, 8))
        self.assertEqual(v.shape, (64,))
        self.assertAlmostEqual(float(v.sum()), 1.0, places=5)

    def test_mask_keeps_shape_and_l1(self):
        hsv = _red_hsv()
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        for fn in (
            compute_hsv_3d_histogram,
            compute_hsv_block_2d_hs_histogram,
            compute_hsv_block_3d_histogram,
            compute_hsv_pyramid_histogram,
        ):
            full = fn(hsv)
            masked = fn(hsv, mask=mask)
            self.assertEqual(full.shape, masked.shape)
            self.assertAlmostEqual(float(masked.sum()), 1.0, places=5)
        # La mascara parcial debe cambiar el descriptor global en imagen bicolor
        # (en imagen uniforme el histograma no cambia: test anterior usaba rojo plano).
        bicolor = np.zeros((32, 32, 3), dtype=np.uint8)
        bicolor[:, :16] = (255, 0, 0)
        bicolor[:, 16:] = (0, 0, 255)
        bicolor_hsv = cv2.cvtColor(bicolor, cv2.COLOR_RGB2HSV)
        half_mask = np.zeros((32, 32), dtype=np.uint8)
        half_mask[:, :16] = 255  # solo mitad roja
        self.assertGreater(
            float(np.abs(
                compute_hsv_3d_histogram(bicolor_hsv)
                - compute_hsv_3d_histogram(bicolor_hsv, mask=half_mask)
            ).sum()), 0.0)

    def test_spatial_sensitivity(self):
        left = np.zeros((32, 32, 3), dtype=np.uint8)
        left[:, :16] = (255, 0, 0)
        left[:, 16:] = (0, 0, 255)
        uniform = np.full((32, 32, 3), (255, 0, 0), dtype=np.uint8)
        hl = cv2.cvtColor(left, cv2.COLOR_RGB2HSV)
        hu = cv2.cvtColor(uniform, cv2.COLOR_RGB2HSV)
        for fn in (
            compute_hsv_block_2d_hs_histogram,
            compute_hsv_block_3d_histogram,
            compute_hsv_pyramid_histogram,
        ):
            self.assertGreater(float(np.abs(fn(hl) - fn(hu)).sum()), 0.1)

    def test_compute_descriptors_w2_order(self):
        rng = np.random.RandomState(0)
        images = [rng.randint(0, 256, size=(24, 24, 3)).astype(np.uint8)
                  for _ in range(2)]
        methods = compute_descriptors_w2(SimpleNamespace(images=images))
        self.assertEqual(len(methods), 5)
        for m in methods:
            self.assertEqual(len(m), 2)

    def test_compute_descriptors_w2_masks(self):
        rng = np.random.RandomState(1)
        images = [rng.randint(0, 256, size=(24, 24, 3)).astype(np.uint8)]
        masks = [np.full((24, 24), 255, dtype=np.uint8)]
        methods = compute_descriptors_w2(
            SimpleNamespace(images=images), masks=masks)
        self.assertEqual(len(methods), 5)
        with self.assertRaises(ValueError):
            compute_descriptors_w2(SimpleNamespace(images=images), masks=[])


if __name__ == "__main__":
    unittest.main()
