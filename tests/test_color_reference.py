"""Tests for the separate HSV hue-reference package."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from color_reference import (  # noqa: E402
    HueReference,
    confidence_weights,
    correct_hue_scale,
    estimate_hue_scale,
    hue_histogram,
    rank_with_hue_references,
    warp_hue_histogram,
)


def _yellow_and_gray():
    image = np.empty((2, 2, 3), dtype=np.uint8)
    image[0, 0] = (255, 255, 0)      # saturated yellow
    image[0, 1] = (255, 255, 0)
    image[1, 0] = (128, 128, 128)    # neutral gray
    image[1, 1] = (128, 128, 128)
    return image


def _normalized(values):
    values = np.asarray(values, dtype=np.float64)
    return values / values.sum()


class TestHueStatistics(unittest.TestCase):
    def test_histogram_is_normalized_and_ignores_gray(self):
        hist = hue_histogram(_yellow_and_gray(), saturation_threshold=40)
        self.assertEqual(hist.shape, (180,))
        self.assertAlmostEqual(float(hist.sum()), 1.0, places=12)
        self.assertEqual(int(np.argmax(hist)), 30)

    def test_warp_preserves_mass(self):
        hist = _normalized(np.arange(1, 181, dtype=np.float64))
        warped = warp_hue_histogram(hist, 1.17)
        self.assertAlmostEqual(float(warped.sum()), 1.0, places=12)
        self.assertEqual(warped.shape, hist.shape)

    def test_estimate_scale_recovers_inverse(self):
        reference = np.zeros(180)
        reference[[20, 60, 100, 140]] = 1.0
        reference = _normalized(reference)
        augmented = warp_hue_histogram(reference, 1.20)
        scale, score = estimate_hue_scale(
            augmented, reference,
            scales=np.arange(0.70, 1.301, 0.005),
            regularization=0.0,
        )
        self.assertAlmostEqual(scale, 1.0 / 1.20, delta=0.02)
        self.assertLess(score, 1e-12)


class TestHueCorrection(unittest.TestCase):
    def test_correction_keeps_gray_and_changes_color(self):
        image = _yellow_and_gray()
        corrected = correct_hue_scale(image, 1.20, saturation_threshold=40)
        self.assertEqual(corrected.shape, image.shape)
        self.assertEqual(corrected.dtype, np.uint8)
        self.assertTrue(np.array_equal(corrected[1, 0], image[1, 0]))
        self.assertFalse(np.array_equal(corrected[0, 0], image[0, 0]))

    def test_identity_returns_a_copy(self):
        image = _yellow_and_gray()
        corrected = correct_hue_scale(image, 1.0)
        self.assertTrue(np.array_equal(corrected, image))
        self.assertIsNot(corrected, image)


class TestHueReference(unittest.TestCase):
    def test_save_load_and_reorder(self):
        hist_a = _normalized([1.0, 2.0, 3.0] + [0.0] * 177)
        hist_b = _normalized([0.0, 1.0, 1.0] + [0.0] * 177)
        reference = HueReference(("a", "b"), np.stack([hist_a, hist_b]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.npz"
            reference.save(path)
            loaded = HueReference.load(path)
        self.assertEqual(loaded.ids, ("a", "b"))
        np.testing.assert_allclose(loaded.histograms, reference.histograms)
        reordered = loaded.reorder(("b", "a"))
        self.assertEqual(reordered.ids, ("b", "a"))
        np.testing.assert_allclose(
            reordered.histograms[0], reference.histograms[1])
        self.assertAlmostEqual(
            float(loaded.global_histogram().sum()), 1.0, places=12)

    def test_missing_id_is_rejected(self):
        reference = HueReference(
            ("a",), _normalized(np.ones(180)).reshape(1, -1))
        with self.assertRaises(ValueError):
            reference.reorder(("a", "b"))

    def test_source_defaults_to_raw(self):
        hist = _normalized(np.ones(180))
        reference = HueReference(("a",), hist.reshape(1, -1))
        self.assertEqual(reference.source, "raw")

    def test_invalid_source_rejected(self):
        hist = _normalized(np.ones(180))
        with self.assertRaises(ValueError):
            HueReference(("a",), hist.reshape(1, -1), source="median7x7")

    def test_source_survives_save_load_and_reorder(self):
        hist_a = _normalized([1.0, 2.0, 3.0] + [0.0] * 177)
        hist_b = _normalized([0.0, 1.0, 1.0] + [0.0] * 177)
        reference = HueReference(("a", "b"), np.stack([hist_a, hist_b]),
                                 source="filtered")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.npz"
            reference.save(path)
            loaded = HueReference.load(path)
        self.assertEqual(loaded.source, "filtered")
        self.assertEqual(loaded.reorder(("b", "a")).source, "filtered")

    def test_legacy_cache_without_source_loads_as_legacy(self):
        hist_a = _normalized(np.ones(180))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.npz"
            np.savez_compressed(
                path,
                ids=np.asarray(("a",)),
                histograms=hist_a.reshape(1, -1),
                bins=np.asarray(180),
                saturation_threshold=np.asarray(40))
            loaded = HueReference.load(path)
        self.assertEqual(loaded.source, "legacy")


class TestRanking(unittest.TestCase):
    def _reference(self):
        yellow = _yellow_and_gray()
        hist_yellow = hue_histogram(yellow)
        hist_neutral = hue_histogram(
            np.full((4, 4, 3), 128, dtype=np.uint8))
        hist_other = _normalized(np.roll(np.ones(180), 45))
        return HueReference(
            ("a", "b", "c"),
            np.stack([hist_yellow, hist_neutral, hist_other]),
        )

    @staticmethod
    def _descriptor(rgb):
        return np.asarray(
            [rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean()],
            dtype=np.float64,
        ) / 255.0

    def test_ranking_is_a_permutation(self):
        museum = np.asarray([
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        result = rank_with_hue_references(
            _yellow_and_gray(), museum, self._reference(), self._descriptor,
            scale_step=0.02, descriptor_step=0.05,
        )
        self.assertEqual(sorted(result.ranking), [0, 1, 2])
        self.assertEqual(sorted(result.baseline_ranking), [0, 1, 2])
        self.assertEqual(result.base_distances.shape, (3,))
        self.assertEqual(result.corrected_distances.shape, (3,))
        self.assertEqual(result.final_distances.shape, (3,))
        self.assertEqual(result.scales.shape, (3,))
        self.assertEqual(result.hist_distances.shape, (3,))
        self.assertEqual(result.confidence.shape, (3,))
        self.assertAlmostEqual(float(result.confidence.max()), 1.0)

    def test_alpha_zero_matches_baseline(self):
        museum = np.asarray([
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        result = rank_with_hue_references(
            _yellow_and_gray(), museum, self._reference(), self._descriptor,
            scale_step=0.02, descriptor_step=0.05, alpha=0.0,
        )
        self.assertEqual(result.ranking, result.baseline_ranking)

    def test_confidence_weights_decay(self):
        weights = confidence_weights([0.1, 0.2, 0.4], temperature=1.0)
        self.assertAlmostEqual(float(weights[0]), 1.0)
        self.assertGreater(float(weights[1]), float(weights[2]))
        self.assertTrue(np.all(weights > 0))


if __name__ == "__main__":
    unittest.main()
