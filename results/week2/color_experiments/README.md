# Color histogram experiments

Results on 287 museum images and the 30 QSD1 development queries used in W1.
The tests cover color spaces, histogram bins, grids, pyramids and distance
functions.

| Descriptor | Comparison | Values | mAP@1 | mAP@5 |
|---|---|---:|---:|---:|
| W1 HSV grid 4x4 | L1 | 512 | 0.7667 | 0.8194 |
| HSV HS+HV pyramid, 16x16 bins | L1 | 10752 | 0.7333 | 0.7722 |
| Lab pyramid (1, 2, 4), 32 bins/channel | L1 | 2016 | 0.8333 | 0.8694 |
| Lab blocks 4x4, 32 bins/channel | L1 | 1536 | 0.8333 | 0.8722 |
| Lab blocks 6x6, 32 bins/channel | L1 | 3456 | 0.8667 | 0.9000 |
| Lab pyramid (1, 3, 6), 32 bins/channel | L1 | 4416 | 0.8667 | 0.9000 |
| 75% Lab blocks 4x4 + 25% W1 HSV grid | Weighted L1 | 2048 | 0.8667 | 0.8917 |

Lab 6x6 was added to `src/methods_w2.py` along with Lab 4x4 and the Lab
pyramid. Adding the coarser levels to the 6x6 grid did not improve the score.
The 4x4 grid and the original Lab pyramid are almost tied: the small mAP@5
difference comes from one query moving from rank 4 to rank 3.

These configurations were selected using QSD1. Scores on new queries and
segmented paintings still need to be checked.

## Saved results

- `controlled_study.csv/json`: color spaces, bins and layouts with L1,
  chi-square and Hellinger (378 evaluations).
- `brightness_study.csv/json`: different weights for the Lab channels.
- `fusion_study.csv/json`: fixed combinations of HSV and Lab distances.
- `extra_grids_study.csv/json`: 3x3 and 6x6 grids, plus the (1, 3, 6) pyramid.
- `metadata.json`: dataset sizes, correspondences and settings.

The CSV files contain scores; the JSON files also contain the rankings and
AP for each query.

To repeat the experiments, run these from the project root:

```bash
python results/week2/color_experiments/study.py
python results/week2/color_experiments/followups.py
```

Run `study.py` first: it generates the cached histograms needed by
`followups.py`. Both scripts save their results in this folder.

For the shorter comparison using the integrated methods:

```bash
python src/eval_w2_color.py --sections A E F
```
