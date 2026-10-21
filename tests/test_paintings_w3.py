"""Check separate painting crops, W3 rankings and reuse of W2 descriptors."""

import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paintings_w3 import (
    PaintingParams,
    _reading_order,
    evaluate_painting_rankings,
    retrieve_folder_w3,
    segment_paintings,
)


class PaintingSegmentationTests(unittest.TestCase):
    def test_two_paintings_preserve_the_wall_between_them(self):
        rgb = np.full((240, 400, 3), 190, dtype=np.uint8)
        rgb[35:200, 25:155] = (150, 20, 30)
        rgb[65:210, 245:375] = (20, 40, 150)
        mask, boxes = segment_paintings(rgb)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, rgb.shape[:2])
        self.assertEqual(set(np.unique(mask)), {0, 255})
        self.assertTrue(np.all(mask[:, 175:225] == 0))
        for box, point in zip(boxes, ((80, 100), (300, 120))):
            x0, y0, x1, y1 = box
            self.assertTrue(x0 <= point[0] < x1 and y0 <= point[1] < y1)

    def test_small_label_is_not_a_second_painting(self):
        rgb = np.full((240, 400, 3), 190, dtype=np.uint8)
        rgb[30:210, 45:290] = (150, 20, 30)
        rgb[165:173, 325:340] = 10
        _, boxes = segment_paintings(rgb)
        self.assertEqual(len(boxes), 1)

    def test_resize_preserves_original_mask_coordinates(self):
        rgb = np.full((900, 1400, 3), 190, dtype=np.uint8)
        rgb[120:780, 150:1200] = (20, 50, 140)
        mask, boxes = segment_paintings(rgb, PaintingParams(max_side=400))
        self.assertEqual(mask.shape, (900, 1400))
        self.assertEqual(len(boxes), 1)
        for got, expected in zip(boxes[0], (150, 120, 1200, 780)):
            self.assertLess(abs(got - expected), 15)

    def test_uniform_wall_returns_no_detections(self):
        rgb = np.full((120, 200, 3), 190, dtype=np.uint8)
        mask, boxes = segment_paintings(rgb)
        self.assertEqual(boxes, [])
        self.assertFalse(mask.any())

    def test_reading_order_for_side_by_side_and_stacked_paintings(self):
        left, right = (10, 50, 100, 200), (140, 20, 300, 190)
        self.assertEqual(_reading_order([right, left]), [left, right])
        top, bottom = (50, 10, 200, 100), (20, 140, 190, 300)
        self.assertEqual(_reading_order([bottom, top]), [top, bottom])


class PaintingRetrievalTests(unittest.TestCase):
    def test_each_painting_has_its_own_relevant_museum_id(self):
        rankings = [[[7, 44], [44, 7]], [[9, 3]]]
        result = evaluate_painting_rankings(rankings, [[7, 44], [3]], 2)
        self.assertAlmostEqual(result["mAP@2"], (1 + 1 + 0.5) / 3)
        self.assertEqual(result["count_accuracy"], 1)

    def test_missing_extra_and_reversed_paintings_are_penalized(self):
        for predicted in ([[7]], [[7], [44], [8]]):
            result = evaluate_painting_rankings([predicted], [[7, 44]], 1)
            self.assertLess(result["mAP@1"], 1)
            self.assertEqual(result["count_accuracy"], 0)
        self.assertEqual(evaluate_painting_rankings([[[44], [7]]], [[7, 44]], 1)["mAP@1"], 0)

    def test_crop_retrieval_uses_filename_ids_and_nested_python_integers(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            museum, queries, output = root / "museum", root / "queries", root / "out"
            museum.mkdir()
            queries.mkdir()
            colors = [(30 + 12 * i, 40 + 5 * i, 120 - 5 * i) for i in range(12)]
            for i, color in enumerate(colors):
                cv2.imwrite(str(museum / f"bbdd_{100 + i:05d}.jpg"),
                            np.full((80, 100, 3), color, np.uint8))
            rgb = np.full((240, 400, 3), 190, np.uint8)
            rgb[35:205, 25:155] = colors[2]
            rgb[35:205, 245:375] = colors[9]
            cv2.imwrite(str(queries / "00002.jpg"), rgb)
            calls = []

            def query_filter(crop):
                calls.append(crop.shape)
                return crop

            rankings, report = retrieve_folder_w3(
                museum, queries, output,
                descriptor=lambda crop: np.median(crop, axis=(0, 1)),
                query_filter=query_filter)
            self.assertEqual([row[0] for row in rankings[0]], [102, 109])
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(width < 200 for height, width, channels in calls))
            self.assertFalse(report["per_image"][0]["fallback"])
            with (output / "result.pkl").open("rb") as stream:
                saved = pickle.load(stream)
            self.assertEqual(saved, rankings)
            for ranking in saved[0]:
                self.assertEqual(len(ranking), 10)
                self.assertEqual(len(set(ranking)), 10)
                self.assertTrue(all(type(item) is int for item in ranking))

    def test_failed_segmentation_keeps_empty_mask_and_reports_fallback(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            museum, queries, output = root / "museum", root / "queries", root / "out"
            museum.mkdir()
            queries.mkdir()
            image = np.full((80, 120, 3), 190, np.uint8)
            cv2.imwrite(str(museum / "00007.jpg"), image)
            cv2.imwrite(str(queries / "00001.jpg"), image)
            rankings, report = retrieve_folder_w3(
                museum, queries, output, descriptor=lambda rgb: rgb.mean(axis=(0, 1)), top_k=1)
            self.assertEqual(rankings, [[[7]]])
            self.assertTrue(report["per_image"][0]["fallback"])
            self.assertFalse(cv2.imread(str(output / "00001.png"), 0).any())


if __name__ == "__main__":
    unittest.main()
