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
    METHODS_W2_COLOR_SPACES,
    compute_descriptors_w2,
    compute_hsv_2d_histogram,
    compute_hsv_3d_histogram,
    compute_hsv_pyramid_histogram,
    compute_lab_histogram,
    compute_lab_block_histogram,
)
from spatial_pyramid import compute_spatial_pyramid  # noqa: E402


def _red_hsv(h=32, w=32):
    rgb = np.full((h, w, 3), (255, 0, 0), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


class TestW2Shapes(unittest.TestCase):
    def test_registry_color_spaces(self):
        self.assertEqual(len(METHODS_W2_NAMES), 11)
        self.assertEqual(list(METHODS_W2), METHODS_W2_NAMES)
        self.assertEqual(set(METHODS_W2_COLOR_SPACES), set(METHODS_W2))

    def test_shapes_and_l1(self):
        hsv = _red_hsv()
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        sources = {"HSV": hsv, "Lab": cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)}
        expected = {
            "HSV 3D 4x4x4": (64,),
            "HSV 2D HS 30x32": (960,),
            "HSV 2D HV 8x8": (64,),
            "HSV 2D HS+HV 16x16": (512,),
            "HSV pyramid 3D 4x4x4": (1344,),
            "HSV pyramid HS 30x32": (20160,),
            "HSV pyramid HV 8x8": (1344,),
            "HSV pyramid HS+HV 16x16": (10752,),
            "Lab block 4x4 1D 32": (1536,),
            "Lab block 6x6 1D 32": (3456,),
            "Lab pyramid 1D 32": (2016,),
        }
        for name, shape in expected.items():
            v = METHODS_W2[name](sources[METHODS_W2_COLOR_SPACES[name]])
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
            compute_hsv_2d_histogram,
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
        hl = cv2.cvtColor(left, cv2.COLOR_RGB2HSV)
        swapped = cv2.cvtColor(left[:, ::-1].copy(), cv2.COLOR_RGB2HSV)
        np.testing.assert_allclose(
            compute_hsv_2d_histogram(hl), compute_hsv_2d_histogram(swapped))
        self.assertGreater(float(np.abs(
            compute_hsv_pyramid_histogram(hl)
            - compute_hsv_pyramid_histogram(swapped)).sum()), 0.1)
        lab = cv2.cvtColor(left, cv2.COLOR_RGB2LAB)
        swapped_lab = cv2.cvtColor(left[:, ::-1].copy(), cv2.COLOR_RGB2LAB)
        np.testing.assert_allclose(
            compute_lab_histogram(lab), compute_lab_histogram(swapped_lab))
        self.assertGreater(float(np.abs(
            compute_lab_block_histogram(lab)
            - compute_lab_block_histogram(swapped_lab)).sum()), 0.1)

    def test_compute_descriptors_w2_order(self):
        rng = np.random.RandomState(0)
        images = [rng.randint(0, 256, size=(24, 24, 3)).astype(np.uint8)
                  for _ in range(2)]
        methods = compute_descriptors_w2(SimpleNamespace(images=images))
        self.assertEqual(len(methods), len(METHODS_W2_NAMES))
        for m in methods:
            self.assertEqual(len(m), 2)
        levels = {"Lab block 4x4 1D 32": (4,),
                  "Lab block 6x6 1D 32": (6,),
                  "Lab pyramid 1D 32": (1, 2, 4)}
        for name, grids in levels.items():
            descriptors = methods[METHODS_W2_NAMES.index(name)]
            for image, descriptor in zip(images, descriptors):
                reference = compute_spatial_pyramid(
                    image, pyramid_levels=grids, color_space="Lab", bins=32)
                np.testing.assert_allclose(descriptor, reference, atol=1e-7)

    def test_compute_descriptors_w2_masks(self):
        rng = np.random.RandomState(1)
        images = [rng.randint(0, 256, size=(24, 24, 3)).astype(np.uint8)]
        masks = [np.full((24, 24), 255, dtype=np.uint8)]
        methods = compute_descriptors_w2(
            SimpleNamespace(images=images), masks=masks)
        self.assertEqual(len(methods), len(METHODS_W2_NAMES))
        with self.assertRaises(ValueError):
            compute_descriptors_w2(SimpleNamespace(images=images), masks=[])

    def test_lab_mask_ignores_background(self):
        image = np.full((36, 36, 3), (255, 0, 0), dtype=np.uint8)
        image[:, 18:] = (0, 0, 255)
        changed = image.copy()
        changed[:, 18:] = (0, 255, 0)
        mask = np.zeros((36, 36), dtype=np.uint8)
        mask[:, :18] = 255
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        other = cv2.cvtColor(changed, cv2.COLOR_RGB2LAB)
        for name, fn in METHODS_W2.items():
            if METHODS_W2_COLOR_SPACES[name] != "Lab":
                continue
            np.testing.assert_array_equal(fn(lab, mask), fn(other, mask))
            self.assertGreater(float(np.abs(fn(lab) - fn(other)).sum()), 0.0)
            empty = fn(lab, np.zeros_like(mask))
            self.assertTrue(np.all(np.isfinite(empty)))
            self.assertEqual(float(empty.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
