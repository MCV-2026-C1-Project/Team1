"""Week 4 matching, rejection and painting-level evaluation."""

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from batch_matching_w4 import calibrate_threshold, make_predictions, unknown_metrics
from evaluate_w4 import evaluate_predictions, main as evaluate_main, validate_scores
from extract_features_w4 import extract_folder
from keypoints_w4 import extract_sift
from matching_w4 import match_sift
from paintings_w3 import _dct2, _dct_basis, dct_descriptor


def score_report():
    return {
        "settings": {"method": "SIFT"}, "top_k": 10,
        "images": [
            {"image_id": i, "paintings": [{"painting_index": 0, "candidates": [
                {"museum_id": mid, "verified": best - rank, "tentative": 30 - rank}
                for rank, mid in enumerate(range(10))
            ]}]} for i, best in enumerate((9, 15))
        ],
    }


class TestSIFT(unittest.TestCase):
    def test_flat_image_has_empty_descriptor_matrix(self):
        points, descriptors = extract_sift(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(len(points), 0)
        self.assertEqual(descriptors.shape, (0, 128))
        self.assertEqual(descriptors.dtype, np.float32)

    def test_extract_save_and_match_same_image(self):
        rgb = np.random.default_rng(4).integers(0, 256, (160, 160, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            cv2.imwrite(str(images / "00007.jpg"), rgb)
            index = extract_folder(images, root / "features", nfeatures=100)
            saved_index = json.loads((root / "features/index.json").read_text())
            self.assertEqual(index, saved_index)
            self.assertEqual(index["images"][0]["image_id"], 7)
            with np.load(root / "features/00007_painting0.npz") as data:
                features = {name: data[name] for name in ("points", "descriptors", "box")}
            result = match_sift(features, features)
            self.assertGreater(result["inlier_count"], 8)
            self.assertEqual(result["inlier_count"], result["tentative_count"])

    def test_homography_verifies_perspective_correspondences(self):
        rng = np.random.default_rng(12)
        points = rng.uniform(30, 270, (20, 2)).astype(np.float32)
        transform = np.array([[1.1, .1, 12], [.04, .9, 15], [.0003, .0001, 1.]])
        transformed = cv2.perspectiveTransform(points[None], transform)[0]
        descriptors = rng.uniform(0, 100, (20, 128)).astype(np.float32)
        query = {"points": points, "descriptors": descriptors, "box": [0, 0, 300, 300]}
        museum = {"points": transformed, "descriptors": descriptors.copy(),
                  "box": [0, 0, 400, 400]}
        for distance in ("L1", "L2"):
            with self.subTest(distance=distance):
                result = match_sift(query, museum, distance=distance)
                self.assertEqual(result["inlier_count"], 20)

    def test_insufficient_features_do_not_produce_verified_matches(self):
        features = {"points": np.empty((0, 2), np.float32),
                    "descriptors": np.empty((0, 128), np.float32),
                    "box": [0, 0, 100, 100]}
        result = match_sift(features, features)
        self.assertEqual(result["tentative_count"], 0)
        self.assertEqual(result["inlier_count"], 0)


class TestW3Portability(unittest.TestCase):
    def test_odd_dct_keeps_energy_and_can_be_inverted(self):
        block = np.random.default_rng(3).normal(size=(51, 51))
        coefficients = _dct2(block)
        basis = _dct_basis(51)
        np.testing.assert_allclose(basis.T @ coefficients @ basis, block, atol=1e-12)
        self.assertAlmostEqual(float((coefficients ** 2).sum()), float((block ** 2).sum()))

    def test_basis_matches_opencv_on_supported_sizes(self):
        block = np.random.default_rng(4).normal(size=(50, 50))
        basis = _dct_basis(50)
        np.testing.assert_allclose(basis @ block @ basis.T, cv2.dct(block), atol=1e-12)

    def test_default_w3_dct_runs_without_changing_block_dimensions(self):
        descriptor = dct_descriptor(np.full((80, 80, 3), 128, np.uint8))
        self.assertEqual(descriptor.shape, (375,))
        np.testing.assert_allclose(descriptor, 0, atol=1e-8)


class TestRejectionAndEvaluation(unittest.TestCase):
    def test_threshold_sweep_includes_decision_boundaries(self):
        report = score_report()
        calibrated = calibrate_threshold(report, [[-1], [0]])
        self.assertEqual(calibrated["min_inliers"], 10)
        predictions = make_predictions(report, calibrated["min_inliers"])
        self.assertEqual(predictions[0], [[-1]])
        self.assertEqual(predictions[1][0][0], 0)
        self.assertEqual(unknown_metrics(predictions, [[-1], [0]])["f1"], 1.)

    def test_calibration_requires_both_known_and_unknown(self):
        for truth in ([[0], [1]], [[-1], [-1]]):
            with self.assertRaises(ValueError):
                calibrate_threshold(score_report(), truth)

    def test_unknown_singleton_lists_are_supported(self):
        predictions = [[[-1]], [[0, 1]]]
        metrics = unknown_metrics(predictions, [[[-1]], [[0]]])
        self.assertEqual(metrics["f1"], 1.)
        with self.assertRaises(ValueError):
            unknown_metrics(predictions, [[-1]])

    def test_missing_extra_and_unknown_paintings(self):
        # First image misses one known painting; the second has an extra crop.
        predictions = [[[4]], [[-1], [8]]]
        metrics = evaluate_predictions(predictions, [[4, 7], [-1]])
        self.assertEqual(metrics["mAP@1"], .5)
        self.assertEqual(metrics["known_mAP@1"], .5)
        self.assertEqual(metrics["count_accuracy"], 0.)
        self.assertEqual(metrics["unknown"]["f1"], 1.)

    def test_extra_rejected_crop_counts_as_false_positive(self):
        metrics = unknown_metrics([[[-1], [-1]]], [[-1]])
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 0))

    def test_retrieval_rank_is_scored_per_painting(self):
        metrics = evaluate_predictions([[[8, 7], [-1]]], [[7, -1]])
        self.assertEqual(metrics["mAP@1"], .5)
        self.assertEqual(metrics["mAP@5"], .75)
        self.assertEqual(metrics["known_mAP@5"], .5)

    def test_score_cache_rejects_wrong_order_and_duplicate_candidates(self):
        report = score_report()
        validate_scores(report, [0, 1], list(range(10)))
        with self.assertRaises(ValueError):
            validate_scores(report, [1, 0], list(range(10)))
        report["images"][0]["paintings"][0]["candidates"][1]["museum_id"] = 0
        with self.assertRaises(ValueError):
            validate_scores(report, [0, 1], list(range(10)))

    def test_full_evaluation_writes_both_methods_and_reloadable_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            museum, queries, output = [root / name for name in ("museum", "queries", "out")]
            museum.mkdir()
            queries.mkdir()
            rng = np.random.default_rng(5)
            for i in range(10):
                image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(museum / f"{i:05d}.jpg"), image)
                if i < 2:
                    cv2.imwrite(str(queries / f"{i:05d}.jpg"), image)
            with (queries / "gt_corresps.pkl").open("wb") as stream:
                pickle.dump([[-1], [0]], stream)
            report = score_report()
            for image in report["images"]:
                image["segmentation_fallback"] = False
            (root / "scores.json").write_text(json.dumps(report))
            argv = ["evaluate_w4.py", "--museum", str(museum), "--queries", str(queries),
                    "--scores", str(root / "scores.json"), "--output", str(output)]
            with patch.object(sys, "argv", argv):
                evaluate_main()
            saved = json.loads((output / "comparison.json").read_text())
            self.assertEqual(saved["metrics"]["SIFT with rejection"]["mAP@1"], 1.)
            self.assertEqual(saved["metrics"]["W3 Texture fusion"]["unknown"]["f1"], 0.)
            with (output / "sift/result.pkl").open("rb") as stream:
                predictions = pickle.load(stream)
            self.assertEqual(predictions[0], [[-1]])
            self.assertEqual(predictions[1][0], list(range(10)))
            self.assertTrue((output / "per_painting.csv").is_file())
            self.assertTrue((output / "w3_baseline/retrieval.json").is_file())


if __name__ == "__main__":
    unittest.main()
