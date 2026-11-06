# SIFT descriptor distances

Same query crops, ratio 0.75 and RANSAC tolerance 5 at reference side 1000.
Each distance selects its rejection threshold on development unknown-positive F1.

| Distance | Minimum inliers | mAP@1 | mAP@5 | mAP@10 | Unknown F1 |
|---|---:|---:|---:|---:|---:|
| L2 | 8 | 0.9231 | 0.9231 | 0.9231 | 0.9091 |
| L1 | 8 | 0.9231 | 0.9231 | 0.9231 | 0.9091 |

The final metrics tie, so L2 is retained.

The L1 experiment uses an exact uint8 representation of the saved SIFT values to speed up OpenCV matching. Every value was checked to be unchanged by conversion. Integer and float L1 distances are identical for these descriptors.

Reproduce the L1 scores and calibration:

```bash
python src/batch_matching_w4.py match --museum-features results/week4/sift/museum --query-features results/week4/sift/qsd1_w4_threshold4 --distance L1 --output results/week4/evaluation/distance_comparison/l1_scores.json
python src/batch_matching_w4.py tune --scores results/week4/evaluation/distance_comparison/l1_scores.json --ground-truth data/qsd1_w4/gt_corresps.pkl --output results/week4/evaluation/distance_comparison/l1_calibration.json
```
