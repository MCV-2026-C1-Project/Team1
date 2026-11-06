# Week 4 - Task 3

QSD1-W4: 30 queries, 29 known paintings and 10 unknown paintings.

| Method | mAP@1 | mAP@5 | mAP@10 | Count accuracy | Unknown F1 |
|---|---:|---:|---:|---:|---:|
| W3 Texture fusion | 0.5385 | 0.5513 | 0.5556 | 0.9000 | 0.0000 |
| SIFT without rejection | 0.6667 | 0.6752 | 0.6752 | 0.9667 | 0.0000 |
| SIFT with rejection | 0.9231 | 0.9231 | 0.9231 | 0.9667 | 0.9091 |

SIFT rejects a crop when its best candidate has fewer than 8 RANSAC inliers. The threshold maximizes unknown-positive F1 on this development set; equal-F1 ties use the smaller threshold.

These are development results, including threshold tuning on the same set. They do not estimate held-out test performance.

AP is computed per painting in reading order. Missing and extra crops score zero. A correctly rejected unknown has target -1 and scores one.

The W3 baseline keeps its original configuration: Texture fusion, L1, 3x3 median on query crops, and the original adaptive W3 segmentation. It always retrieves museum IDs, so it cannot recognize unknown paintings. Its unknown F1 is therefore zero.

SIFT uses the fixed segmentation threshold 4 selected on QSD1-W4. This compares complete pipelines: both segmentation and retrieval differ.

## Retrieval on annotated known paintings

This view excludes unknown annotations and counts missing known crops as zero. The main table also penalizes extra crops.

| Method | Known mAP@1 | Known mAP@5 | Known mAP@10 |
|---|---:|---:|---:|
| W3 Texture fusion | 0.7241 | 0.7414 | 0.7471 |
| SIFT without rejection | 0.8966 | 0.9080 | 0.9080 |
| SIFT with rejection | 0.8966 | 0.8966 | 0.8966 |
