"""Tests for Methods_W3 (methods_w3.py): the query-side noise filters used
by Task 1 and the unfiltered "Lab blocks 1D" baseline.

The descriptors are the W2 "Lab blocks 1D" ones; here we only check that the
filters keep the RGB/gray contract, that they are applied to the queries only
and that the method registry/hyperparameters are wired correctly.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from methods_w3 import (  # noqa: E402
    AVERAGE_METHOD,
    BASELINE_METHOD,
    CLOSE_METHOD,
    CLOSE_OPEN_METHOD,
    DEFAULT_BLOCKS,
    DEFAULT_BINS,
    DEFAULT_FILTER_SCOPE,
    DEFAULT_MEDIAN_KSIZE,
    DEFAULT_WAVELET_LEVEL,
    FILTER_SCOPES,
    AVERAGE_SWEEP_SIZES,
    CHAIN_SWEEP_SIZES,
    GAUSSIAN_SWEEP_SIZES,
    WAVELET_SWEEP_LEVELS,
    GAUSSIAN_METHOD,
    IMAGE_FILTERS,
    MEDIAN_METHOD,
    METHOD_ALL,
    METHOD_CHOICES,
    METHODS_W3,
    METHODS_W3_NAMES,
    MORPH_OPERATIONS,
    OPEN_CLOSE_METHOD,
    OPEN_METHOD,
    WAVELET_METHOD,
    average_filter,
    baseline_descriptor,
    bilateral_filter,
    build_descriptor,
    build_image_filter,
    build_method,
    build_query_filter,
    canonical_method_name,
    compute_baseline_descriptors_w3,
    compute_lab_blocks_1d_histogram,
    compute_descriptors_w3,
    default_filter_specs,
    dilate_filter,
    erode_filter,
    evaluate_folders_w3,
    gaussian_filter,
    load_museum_descriptors_w3,
    median_descriptor,
    median_filter,
    morphology_filter,
    open_filter,
    close_filter,
    wavelet_lowpass_filter,
)


def _red_rgb(h=32, w=32):
    return np.full((h, w, 3), (255, 0, 0), dtype=np.uint8)


def _lab_blocks_1d_reference(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return compute_lab_blocks_1d_histogram(lab)


class TestW3Registry(unittest.TestCase):
    def test_registry_names(self):
        self.assertEqual(
            METHODS_W3_NAMES,
            ("Lab blocks 1D", "Lab blocks 1D median", "Lab blocks 1D average",
             "Lab blocks 1D gaussian", "Lab blocks 1D open",
             "Lab blocks 1D close", "Lab blocks 1D open+close",
             "Lab blocks 1D close+open", "Lab blocks 1D wavelet"))
        self.assertEqual(list(METHODS_W3), list(METHODS_W3_NAMES))
        self.assertEqual(METHOD_CHOICES, METHODS_W3_NAMES + (METHOD_ALL,))
        self.assertNotIn(METHOD_ALL, METHODS_W3)

    def test_aliases(self):
        self.assertEqual(canonical_method_name("baseline"), BASELINE_METHOD)
        self.assertEqual(canonical_method_name("Median"), MEDIAN_METHOD)
        self.assertEqual(
            canonical_method_name("median + lab blocks 1d"), MEDIAN_METHOD)
        self.assertEqual(canonical_method_name("average"), AVERAGE_METHOD)
        self.assertEqual(canonical_method_name("gaussian"), GAUSSIAN_METHOD)
        self.assertEqual(canonical_method_name("open"), OPEN_METHOD)
        self.assertEqual(canonical_method_name("close"), CLOSE_METHOD)
        self.assertEqual(
            canonical_method_name("open+close"), OPEN_CLOSE_METHOD)
        self.assertEqual(
            canonical_method_name("close+open"), CLOSE_OPEN_METHOD)
        self.assertEqual(
            canonical_method_name("low-pass wavelet"), WAVELET_METHOD)
        with self.assertRaises(ValueError):
            canonical_method_name("not-a-method")

    def test_method_all_rejected(self):
        with self.assertRaises(ValueError):
            build_method(METHOD_ALL)


class TestW3Filters(unittest.TestCase):
    def test_median_shapes_and_dtype(self):
        rgb = _red_rgb()
        self.assertEqual(median_filter(rgb).shape, rgb.shape)
        self.assertEqual(median_filter(rgb).dtype, np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.assertEqual(median_filter(gray).shape, gray.shape)
        self.assertEqual(median_filter(gray).dtype, np.uint8)

    def test_median_removes_impulse(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        gray[2, 2] = 255
        filtered = median_filter(gray, ksize=3)
        self.assertEqual(int(filtered[2, 2]), 0)

    def test_median_rejects_even_or_small_ksize(self):
        with self.assertRaises(AssertionError):
            median_filter(_red_rgb(), ksize=4)
        with self.assertRaises(AssertionError):
            median_filter(_red_rgb(), ksize=2)

    def test_morphology_shapes_and_direction(self):
        gray = np.zeros((7, 7), dtype=np.uint8)
        gray[3, 3] = 255
        self.assertGreaterEqual(
            int(dilate_filter(gray, 3).sum()), int(gray.sum()))
        self.assertLessEqual(
            int(erode_filter(gray, 3).sum()), int(gray.sum()))
        self.assertEqual(open_filter(gray, 3).shape, gray.shape)
        self.assertEqual(close_filter(gray, 3).shape, gray.shape)
        self.assertEqual(morphology_filter(gray, "dilate").dtype, np.uint8)
        for operation in MORPH_OPERATIONS:
            self.assertEqual(
                morphology_filter(_red_rgb(), operation).shape,
                _red_rgb().shape)
        with self.assertRaises(ValueError):
            morphology_filter(gray, "not-an-operation")

    def test_morphology_rejects_bad_ksize(self):
        with self.assertRaises(AssertionError):
            morphology_filter(np.zeros((5, 5), np.uint8), "dilate", ksize=2)

    def test_average_shapes(self):
        rgb = _red_rgb()
        self.assertEqual(average_filter(rgb, 5).shape, rgb.shape)
        self.assertEqual(average_filter(rgb, 5).dtype, np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.assertEqual(average_filter(gray, 5).shape, gray.shape)
        with self.assertRaises(AssertionError):
            average_filter(rgb, ksize=4)

    def test_bilateral_shapes(self):
        rgb = _red_rgb()
        self.assertEqual(bilateral_filter(rgb, 5).shape, rgb.shape)
        self.assertEqual(bilateral_filter(rgb, 5).dtype, np.uint8)

    def test_gaussian_shapes(self):
        rgb = _red_rgb()
        self.assertEqual(gaussian_filter(rgb, 5).shape, rgb.shape)
        self.assertEqual(gaussian_filter(rgb, 5).dtype, np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.assertEqual(gaussian_filter(gray, 5).shape, gray.shape)
        with self.assertRaises(AssertionError):
            gaussian_filter(rgb, ksize=4)

    def test_image_filter_factory(self):
        rgb = _red_rgb()
        identity = build_image_filter("none")
        self.assertTrue(np.array_equal(identity(rgb), rgb))
        average = build_image_filter("average", ksize=5)
        self.assertTrue(np.array_equal(
            average(rgb), average_filter(rgb, ksize=5)))
        gaussian = build_image_filter("gaussian", ksize=5)
        self.assertTrue(np.array_equal(
            gaussian(rgb), gaussian_filter(rgb, ksize=5)))
        chain = build_image_filter("median+open", ksize=5)
        expected = open_filter(median_filter(rgb, ksize=5), ksize=5)
        self.assertTrue(np.array_equal(chain(rgb), expected))
        open_close = build_image_filter("open+close", ksize=5)
        self.assertEqual(open_close(rgb).shape, rgb.shape)
        close_open = build_image_filter("close+open", ksize=5)
        self.assertEqual(close_open(rgb).shape, rgb.shape)
        with self.assertRaises(ValueError):
            build_image_filter("not-a-filter")

    def test_wavelet_shapes(self):
        rgb = _red_rgb(h=32, w=32)
        for level in (1, 2, 3):
            filtered = wavelet_lowpass_filter(rgb, level=level)
            self.assertEqual(filtered.shape, rgb.shape)
            self.assertEqual(filtered.dtype, np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.assertEqual(
            wavelet_lowpass_filter(gray, level=2).shape, gray.shape)
        factory = build_image_filter(
            "wavelet_lowpass", wavelet_level=DEFAULT_WAVELET_LEVEL)
        self.assertTrue(np.array_equal(
            factory(rgb),
            wavelet_lowpass_filter(rgb, level=DEFAULT_WAVELET_LEVEL)))
        with self.assertRaises(AssertionError):
            wavelet_lowpass_filter(rgb, level=0)

    def test_every_registered_method_builds(self):
        rgb = _red_rgb()
        for name in METHODS_W3_NAMES:
            query_filter = build_query_filter(name)
            method = build_method(name)
            if name == BASELINE_METHOD:
                self.assertIsNone(query_filter)
            else:
                self.assertTrue(callable(query_filter))
                self.assertEqual(query_filter(rgb).shape, rgb.shape)
            self.assertEqual(method(rgb).shape,
                             baseline_descriptor(rgb).shape)


class TestW3Descriptors(unittest.TestCase):
    def test_baseline_is_w2_blocks_1d(self):
        rgb = _red_rgb()
        self.assertTrue(np.allclose(
            baseline_descriptor(rgb, blocks=DEFAULT_BLOCKS, bins=DEFAULT_BINS),
            _lab_blocks_1d_reference(rgb)))

    def test_build_method_filters_queries_only(self):
        rgb = _red_rgb()
        baseline = build_method(BASELINE_METHOD)
        median = build_method(MEDIAN_METHOD, ksize=DEFAULT_MEDIAN_KSIZE)
        self.assertTrue(np.array_equal(baseline(rgb), baseline_descriptor(rgb)))
        self.assertTrue(np.array_equal(median(rgb), median_descriptor(rgb)))
        reference = build_descriptor()
        self.assertTrue(np.array_equal(reference(rgb), baseline_descriptor(rgb)))

    def test_compute_descriptors_w3(self):
        dataset = SimpleNamespace(images=[_red_rgb(), _red_rgb()])
        descriptors = compute_descriptors_w3(dataset)
        self.assertEqual(len(descriptors), len(METHODS_W3_NAMES))
        self.assertEqual(len(descriptors[0]), 2)
        self.assertTrue(np.array_equal(
            descriptors[0][0], baseline_descriptor(dataset.images[0])))
        self.assertTrue(np.array_equal(
            compute_baseline_descriptors_w3(dataset)[0],
            descriptors[0][0]))

    def test_sweep_specs(self):
        specs = default_filter_specs()
        self.assertEqual(specs[0], ("none", 1))
        self.assertIn(("average", AVERAGE_SWEEP_SIZES[0]), specs)
        self.assertIn(("median", 7), specs)
        self.assertIn(("bilateral", 7), specs)
        self.assertIn(("gaussian", GAUSSIAN_SWEEP_SIZES[0]), specs)
        for operation in MORPH_OPERATIONS:
            self.assertIn((operation, 3), specs)
        self.assertIn(("open+close", CHAIN_SWEEP_SIZES[0]), specs)
        self.assertIn(("close+open", CHAIN_SWEEP_SIZES[0]), specs)
        for level in WAVELET_SWEEP_LEVELS:
            self.assertIn(("wavelet_lowpass", level), specs)


class TestW3FilterScope(unittest.TestCase):
    def test_scope_constants(self):
        self.assertEqual(FILTER_SCOPES, ("query", "both"))
        self.assertEqual(DEFAULT_FILTER_SCOPE, "query")

    def test_invalid_scope_rejected(self):
        with self.assertRaises(AssertionError):
            evaluate_folders_w3(
                "missing", "missing", BASELINE_METHOD, filter_scope="museum")

    def test_museum_filter_is_optional(self):
        noisy = np.zeros((32, 32, 3), dtype=np.uint8)
        noisy[8:24, 8:24] = (180, 80, 40)
        noisy[3, 3] = (255, 255, 255)
        with tempfile.TemporaryDirectory() as tmp:
            cv2.imwrite(
                str(Path(tmp) / "00000.jpg"),
                cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))
            ids_raw, raw = load_museum_descriptors_w3(tmp)
            ids_identity, identity = load_museum_descriptors_w3(
                tmp, image_filter=lambda image: image)
            ids_median, filtered = load_museum_descriptors_w3(
                tmp, image_filter=lambda image: median_filter(image, 5))
        self.assertEqual(ids_raw, ids_identity)
        self.assertEqual(ids_raw, ids_median)
        self.assertTrue(np.allclose(raw[0], identity[0]))
        self.assertFalse(np.allclose(raw[0], filtered[0]))

    def test_hue_invariant_mode_runs(self):
        rng = np.random.default_rng(7)
        museum_colors = [(200, 60, 60), (60, 200, 60), (60, 60, 200)]
        with tempfile.TemporaryDirectory() as tmp:
            museum_dir = Path(tmp) / "museum"
            query_dir = Path(tmp) / "queries"
            museum_dir.mkdir()
            query_dir.mkdir()
            for index, color in enumerate(museum_colors):
                image = np.full((32, 32, 3), color, dtype=np.uint8)
                cv2.imwrite(
                    str(museum_dir / f"{index:05d}.jpg"),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            tinted = np.full((32, 32, 3), (200, 120, 60), dtype=np.uint8)
            tinted = np.clip(
                tinted.astype(np.int32)
                + rng.integers(-8, 9, tinted.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(
                str(query_dir / "00000.jpg"),
                cv2.cvtColor(tinted, cv2.COLOR_RGB2BGR))
            ranked = evaluate_folders_w3(
                str(museum_dir), str(query_dir), BASELINE_METHOD,
                hue_invariant=True)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(len(ranked[0]), 3)


if __name__ == "__main__":
    unittest.main()
