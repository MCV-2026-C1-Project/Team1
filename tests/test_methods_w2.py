"""Tests for Methods_W2 (methods_w2.py): HSV + Lab globals, pyramids and
Lab spatial entries, L1, masks.

Lab mirrors every HSV combination 1:1 (same channel indices, same bins).
Spatial pyramids/blocks are the canonical W2 spatial layout (see
compute_lab_pyramid_histogram / compute_lab_block_histogram).
"""
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
    compute_hsv_2d_concat_histogram,
    compute_hsv_2d_histogram,
    compute_hsv_3d_histogram,
    compute_hsv_pyramid_histogram,
    compute_lab_1d_histogram,
    compute_lab_2d_concat_histogram,
    compute_lab_2d_histogram,
    compute_lab_3d_histogram,
    compute_lab_histogram,
    compute_lab_block_histogram,
    compute_lab_mixed_concat_histogram,
    compute_lab_pyramid_histogram,
)


def _red_hsv(h=32, w=32):
    rgb = np.full((h, w, 3), (255, 0, 0), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def _red_lab(h=32, w=32):
    rgb = np.full((h, w, 3), (255, 0, 0), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


class TestW2Shapes(unittest.TestCase):
    def test_registry_color_spaces(self):
        self.assertEqual(len(METHODS_W2_NAMES), 26)
        self.assertEqual(list(METHODS_W2), METHODS_W2_NAMES)
        self.assertEqual(set(METHODS_W2_COLOR_SPACES), set(METHODS_W2))

    def test_shapes_and_l1(self):
        hsv = _red_hsv()
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        sources = {"HSV": hsv, "Lab": cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)}
        expected = {
            "HSV 3D 4": (64,),
            "HSV 2D HS 32": (1024,),
            "HSV 2D HV 8": (64,),
            "HSV 2D HS+HV 16": (512,),
            "HSV pyramid 3D 4": (21 * 64,),
            "HSV pyramid HS 32": (21 * 1024,),
            "HSV pyramid HV 8": (21 * 64,),
            "HSV pyramid HS+HV 16": (21 * 512,),
            "Lab block 4x4 1D 32": (1536,),
            "Lab block 6x6 1D 32": (3456,),
            "Lab pyramid 1D 32": (2016,),
            "Lab block 2x2 1D 32": (384,),
            "Lab block 2x4 1D 32": (768,),
            "Lab 1D 32": (96,),
            "Lab 3D 4": (64,),
            "Lab 2D La 32": (1024,),
            "Lab 2D Lb 8": (64,),
            "Lab 2D ab 16": (256,),
            "Lab 2D La+Lb 16": (512,),
            "Lab L+ab 16": (272,),
            "Lab block 4x4 L+ab 16": (16 * 272,),
            "Lab pyramid 3D 4": (21 * 64,),
            "Lab pyramid La 32": (21 * 1024,),
            "Lab pyramid Lb 8": (21 * 64,),
            "Lab pyramid ab 16": (21 * 256,),
            "Lab pyramid La+Lb 16": (21 * 512,),
        }
        for name, shape in expected.items():
            v = METHODS_W2[name](sources[METHODS_W2_COLOR_SPACES[name]])
            self.assertEqual(v.shape, shape, name)
            self.assertAlmostEqual(float(v.sum()), 1.0, places=5, msg=name)
            self.assertTrue(np.all(v >= 0))

    def test_3d_custom_bins(self):
        v = compute_hsv_3d_histogram(_red_hsv(), bins=4)
        self.assertEqual(v.shape, (64,))

    def test_2d_custom_channels(self):
        hsv = _red_hsv()
        v = compute_hsv_2d_histogram(hsv, channels=(1, 2), bins=8)
        self.assertEqual(v.shape, (64,))
        self.assertAlmostEqual(float(v.sum()), 1.0, places=5)

    def test_uniform_bins_rejected_tuples(self):
        with self.assertRaises(AssertionError):
            compute_hsv_2d_histogram(_red_hsv(), bins=(8, 8))

    def test_rectangular_grid(self):
        lab = _red_lab(h=32, w=32)
        v = METHODS_W2["Lab block 2x4 1D 32"](lab)
        self.assertEqual(v.shape, (8 * 96,))
        self.assertAlmostEqual(float(v.sum()), 1.0, places=5)
        with self.assertRaises(AssertionError):
            compute_lab_block_histogram(lab, grid=0)
        with self.assertRaises(AssertionError):
            compute_lab_block_histogram(lab, grid=(2, 0))

    def test_2d_concat_shape_and_l1(self):
        hsv = _red_hsv()
        v = compute_hsv_2d_concat_histogram(
            hsv, parts=[((0, 1), 16), ((0, 2), 16)]
        )
        self.assertEqual(v.shape, (512,))
        self.assertAlmostEqual(float(v.sum()), 1.0, places=5)

    def test_lab_3d_2d_concat_shapes_and_l1(self):
        lab = _red_lab()
        cases = [
            (compute_lab_3d_histogram(lab, bins=4), (64,)),
            (compute_lab_2d_histogram(
                lab, channels=(0, 1), bins=32), (1024,)),
            (compute_lab_2d_histogram(
                lab, channels=(0, 2), bins=8), (64,)),
            (compute_lab_2d_histogram(
                lab, channels=(1, 2), bins=16), (256,)),
            (compute_lab_2d_concat_histogram(
                lab, parts=[((0, 1), 16), ((0, 2), 16)]),
             (512,)),
        ]
        for v, shape in cases:
            self.assertEqual(v.shape, shape)
            self.assertAlmostEqual(float(v.sum()), 1.0, places=5)
            self.assertTrue(np.all(v >= 0))

    def test_lab_mixed_concat_l_ab(self):
        lab = _red_lab()
        v = compute_lab_mixed_concat_histogram(
            lab, parts=[((0,), 16), ((1, 2), 16)])
        self.assertEqual(v.shape, (272,))
        self.assertAlmostEqual(float(v.sum()), 1.0, places=5)
        self.assertTrue(np.all(v >= 0))
        # Equal weight per part: L takes the first half of the mass.
        self.assertAlmostEqual(float(v[:16].sum()), 0.5, places=5)
        self.assertAlmostEqual(float(v[16:].sum()), 0.5, places=5)
        np.testing.assert_allclose(
            v[:16] * 2, compute_lab_1d_histogram(lab, channel=0, bins=16),
            atol=1e-6)
        with self.assertRaises(AssertionError):
            compute_lab_mixed_concat_histogram(lab, parts=[((0, 1, 2), 16)])

    def test_pyramid_levels_and_l1(self):
        hsv = _red_hsv(h=32, w=32)
        full = compute_hsv_pyramid_histogram(hsv)
        self.assertEqual(full.shape, (21 * 32 * 32,))
        self.assertAlmostEqual(float(full.sum()), 1.0, places=5)
        single = compute_hsv_pyramid_histogram(hsv, levels=(1,))
        self.assertEqual(single.shape, (32 * 32,))
        np.testing.assert_allclose(
            single, compute_hsv_2d_histogram(hsv, channels=(0, 1), bins=32),
            atol=1e-6)

    def test_mask_keeps_shape_and_l1(self):
        hsv = _red_hsv()
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        concat = lambda im, mask=mask: compute_hsv_2d_concat_histogram(
            im, parts=[((0, 1), 16), ((0, 2), 16)],
            mask=mask,
        )
        for fn in (
            compute_hsv_3d_histogram,
            compute_hsv_2d_histogram,
            concat,
            compute_hsv_pyramid_histogram,
        ):
            full = fn(hsv)
            masked = fn(hsv, mask=mask)
            self.assertEqual(full.shape, masked.shape)
            self.assertAlmostEqual(float(masked.sum()), 1.0, places=5)
        # A partial mask must change the global descriptor on a two-color
        # image (on a flat image the histogram does not change: the test
        # above used flat red).
        bicolor = np.zeros((32, 32, 3), dtype=np.uint8)
        bicolor[:, :16] = (255, 0, 0)
        bicolor[:, 16:] = (0, 0, 255)
        bicolor_hsv = cv2.cvtColor(bicolor, cv2.COLOR_RGB2HSV)
        half_mask = np.zeros((32, 32), dtype=np.uint8)
        half_mask[:, :16] = 255  # red half only
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
        levels = {"Lab block 2x2 1D 32": (2,),
                  "Lab block 2x4 1D 32": ((2, 4),),
                  "Lab block 4x4 1D 32": (4,),
                  "Lab block 6x6 1D 32": (6,),
                  "Lab pyramid 1D 32": (1, 2, 4)}
        for name, grids in levels.items():
            descriptors = methods[METHODS_W2_NAMES.index(name)]
            for image, descriptor in zip(images, descriptors):
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                reference = compute_lab_pyramid_histogram(
                    lab, levels=grids)
                np.testing.assert_allclose(descriptor, reference, atol=1e-7)

    def test_pyramid_dimensions_and_l1(self):
        """Cell count x dims per cell; L1 mass is 1 (ported spatial check)."""
        lab = _red_lab(h=9, w=11)
        for grids, cells in (((1,), 1), ((1, 2), 5), ((1, 2, 4), 21)):
            result = compute_lab_pyramid_histogram(lab, levels=grids)
            self.assertEqual(result.shape, (cells * 96,))
            self.assertTrue(np.isfinite(result).all())
            self.assertAlmostEqual(float(result.sum()), 1, places=5)
            result3d = compute_lab_pyramid_histogram(
                lab, levels=grids,
                block_fn=lambda b, m: compute_lab_3d_histogram(
                    b, bins=4, mask=m))
            self.assertEqual(result3d.shape, (cells * 64,))
            self.assertAlmostEqual(float(result3d.sum()), 1, places=5)

    def test_single_level_matches_global(self):
        """levels=(1,) with the default block equals the global descriptor."""
        lab = cv2.cvtColor(
            np.random.default_rng(42).integers(
                0, 256, (9, 11, 3), dtype=np.uint8),
            cv2.COLOR_RGB2LAB)
        np.testing.assert_allclose(
            compute_lab_pyramid_histogram(lab, levels=(1,)),
            compute_lab_histogram(lab, bins=32), atol=1e-7)

    def test_joint_captures_channel_relationships(self):
        """Same Lab marginals, different joint 3D (ported spatial check)."""
        a = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        b = np.array([[[0, 255, 0], [255, 0, 255]]], dtype=np.uint8)
        np.testing.assert_allclose(
            compute_lab_histogram(a, bins=2),
            compute_lab_histogram(b, bins=2))
        self.assertFalse(np.array_equal(
            compute_lab_3d_histogram(a, bins=2),
            compute_lab_3d_histogram(b, bins=2)))

    def test_invalid_bins_and_grids(self):
        lab = _red_lab(h=8, w=8)
        for fn in (lambda: compute_lab_histogram(lab, bins=0),
                   lambda: compute_lab_2d_histogram(lab, bins=(8, 8)),
                   lambda: compute_lab_3d_histogram(lab, bins=0),
                   lambda: compute_lab_block_histogram(lab, grid=0),
                   lambda: compute_lab_block_histogram(lab, grid=(2, 0)),
                   lambda: compute_lab_pyramid_histogram(lab, levels=(0,)),
                   lambda: compute_lab_mixed_concat_histogram(
                       lab, parts=[((0, 1, 2), 16)])):
            with self.subTest(fn=fn), self.assertRaises(AssertionError):
                fn()

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
