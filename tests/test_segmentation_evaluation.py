"""Check pixel metrics and matching of predicted masks to reference masks."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval_w2_masks import evaluate_mask_folders
from utils import compute_segmentation_metrics


class SegmentationTests(unittest.TestCase):
    def test_known_counts_and_scores(self):
        truth = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
        predicted = np.array([[255, 255, 0], [0, 255, 0], [0, 0, 0]], dtype=np.uint8)
        scores = compute_segmentation_metrics(predicted, truth)
        self.assertAlmostEqual(scores["precision"], 2 / 3)
        self.assertAlmostEqual(scores["recall"], 1 / 2)
        self.assertAlmostEqual(scores["F1"], 4 / 7)

    def test_perfect_missing_and_inverted_foreground(self):
        truth = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        perfect = compute_segmentation_metrics(truth != 0, truth)
        for key in ("precision", "recall", "F1"):
            self.assertEqual(perfect[key], 1.0)
        for predicted in (np.zeros_like(truth), 255 - truth):
            scores = compute_segmentation_metrics(predicted, truth)
            for key in ("precision", "recall", "F1"):
                self.assertEqual(scores[key], 0.0)

    def test_empty_foreground_has_defined_scores(self):
        empty = np.zeros((2, 2), dtype=np.uint8)
        scores = compute_segmentation_metrics(empty, empty)
        for key in ("precision", "recall", "F1"):
            self.assertEqual(scores[key], 0.0)

    def test_identical_large_masks_preserve_background_and_inputs(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[20:110, 30:100] = 255
        before = mask.copy()
        scores = compute_segmentation_metrics(mask, mask)
        for key in ("precision", "recall", "F1"):
            self.assertEqual(scores[key], 1.0)
        np.testing.assert_array_equal(mask, before)

    def test_wrong_mask_shape(self):
        truth = np.ones((2, 2), dtype=np.uint8)
        for predicted in (np.ones((3, 2)), np.ones((2, 2, 3))):
            with self.subTest(shape=predicted.shape):
                with self.assertRaises(ValueError):
                    compute_segmentation_metrics(predicted, truth)

    def test_folder_order_and_aggregation(self):
        with TemporaryDirectory() as temp:
            predicted, truth = Path(temp) / "predicted", Path(temp) / "truth"
            predicted.mkdir()
            truth.mkdir()
            for index, size in ((10, 3), (2, 2)):
                reference = np.full((size, size), 255, dtype=np.uint8)
                estimate = reference if index == 2 else np.zeros_like(reference)
                cv2.imwrite(str(truth / f"{index}.png"), reference)
                cv2.imwrite(str(predicted / f"mask_{index:05d}.png"), estimate)
            result = evaluate_mask_folders(predicted, truth)
            self.assertEqual([row["id"] for row in result["per_image"]], [2, 10])
            self.assertEqual(result["mean"], {"precision": 0.5, "recall": 0.5, "F1": 0.5})

    def test_missing_extra_and_duplicate_ids(self):
        with TemporaryDirectory() as temp:
            predicted, truth = Path(temp) / "predicted", Path(temp) / "truth"
            predicted.mkdir()
            truth.mkdir()
            mask = np.ones((2, 2), dtype=np.uint8)
            cv2.imwrite(str(truth / "00001.png"), mask)
            cv2.imwrite(str(predicted / "00002.png"), mask)
            with self.assertRaisesRegex(ValueError, "same mask IDs"):
                evaluate_mask_folders(predicted, truth)
            cv2.imwrite(str(predicted / "mask_2.png"), mask)
            with self.assertRaisesRegex(ValueError, "Duplicate mask ID 2"):
                evaluate_mask_folders(predicted, truth)

    def test_unreadable_masks_and_wrong_size(self):
        with TemporaryDirectory() as temp:
            predicted, truth = Path(temp) / "predicted", Path(temp) / "truth"
            predicted.mkdir()
            truth.mkdir()
            cv2.imwrite(str(truth / "00001.png"), np.ones((2, 2), dtype=np.uint8))
            (predicted / "00001.png").write_text("not an image")
            with self.assertRaisesRegex(ValueError, "Cannot read mask"):
                evaluate_mask_folders(predicted, truth)
            cv2.imwrite(str(predicted / "00001.png"), np.ones((3, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "same shape"):
                evaluate_mask_folders(predicted, truth)


if __name__ == "__main__":
    unittest.main()
