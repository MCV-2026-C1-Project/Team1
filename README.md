# C1-PROJECT

## Initial Setup

Create a virtual environment to encapsulate the required installations.

1. In your project directory, create the environment:
```powershell
   python -m venv .venv
```
2. Upgrade pip, in case the versions do not coincide:
```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
```
3. Install the required packages:
```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
4. Activate the environment (Windows):
```powershell
   .\.venv\Scripts\Activate.ps1
```

> **Note:** Windows PowerShell may restrict running scripts by default. If activation fails, allow execution for the current session with:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

---

## Week 1

### Task 1: Image Descriptors

Run:
```powershell
python .\src\methods_w1.py --list
```

The W1 descriptors live in `methods_w1.py` (`METHODS_W1`), with the histogram
implementations in `utils.py`:
- Grayscale
- RGB
- HSV (one component per histogram)
- HSV grid (4×4 cells, 512 values)

### Task 2: Similarity Functions

All similarity formulas are implemented in `k_similarity.py`, and fall into two groups:

| Type | Functions | Interpretation |
|------|-----------|----------------|
| Distances | Euclidean, L1, Chi-square | Lower value = more similar |
| Similarities | Histogram Intersection, Hellinger kernel, Cosine, Correlation | Higher value = more similar |

### Task 3: Retrieval and Evaluation

Run:
```powershell
python .\src\k_similarity.py
```

This script retrieves the top K museum images for each query by comparing image descriptors (Grayscale, HSV, RGB, HSV grid) using the similarity functions.

**Ranking order**
- Distances are sorted in ascending order.
- Similarities (Histogram Intersection and Hellinger kernel) are sorted in descending order. They are listed in the `SIMILARITY_FUNCTIONS` constant, used by `rank_museum_images`.

Sorting a similarity in ascending order would return the least similar images first (scores of 0), which is why this distinction is needed.

**Evaluation metrics**
- **AP@K** measures how accurately the relevant museum images are ranked within the top K results for a single query, using the ground-truth correspondences.
- **mAP@K** averages AP@K across all queries, allowing retrieval performance to be compared across descriptor methods and distance functions for K=1 and K=5.

### Task 4: Blind QST1 submission

From the project directory, give the folders containing the museum and test images:
```bash
python .\src\methods_w2.py --museum .\data\BBDD --queries .\data\qsd1_w2 --method "Lab blocks 1D" --blocks 25 
```

Both arguments accept relative or absolute image-folder paths. Only JPG/JPEG
files directly inside each folder are read. Queries are ordered by numeric filename ID. Museum IDs come from
filenames: `bbdd_00007.jpg` becomes integer `7`.

Each `result.pkl` contains a Python list of lists, with exactly 10 unique Python
integer museum IDs per query, ranked best first.

| Output under `results/week1/QST1/` | Descriptor | Default comparison |
|---|---|---|
| `method1/result.pkl` | HSV | L1 |
| `method2/result.pkl` | HSV grid | L1 |

At most two methods are allowed (statement), so only HSV and HSV grid are
submitted; Grayscale and RGB remain in `METHODS_W1` for validation only.

To override the comparison for both methods or change the output location:
```bash
python src/methods_w1.py --museum data/BBDD --queries data/qst1_w1 --distance Hellinger --output-dir results/hellinger/week1/QST1
```

Available comparisons: `Euclidean`, `L1`, `Chi-square`, `Histogram Intersection`,
`Hellinger`, `Cosine`, and `Correlation`. Quote names containing spaces.

---

## Week 2

### Task 1: Color Histograms

`methods_w2.py` keeps four Lab descriptors (all L1-normalized, with an
optional foreground mask):

| Method | Description | Parameter |
|---|---|---|
| `Lab blocks 1D` | grid of marginal 1D histograms (32 bins per channel), concatenated in row-major order | `--blocks T` total cells, a perfect square (default 25 → 5×5) |
| `Lab 2D` | same grid of cells; each cell is L as a 1D marginal (32 bins) plus the (a, b) pair as a joint 2D histogram (32 bins per axis), concatenated row-major | `--blocks T` (default 25 → 5×5) |
| `Lab 3D` | same grid of cells; each cell is a joint 3D L-a-b histogram (32 bins per axis), concatenated row-major | `--blocks T` (default 25 → 5×5) |
| `Lab pyramid` | hierarchy of 1D-block levels: level 1 is one block, each extra level doubles the number of blocks (1, 2, 4, 8 …) | `--levels K` (default 3) |

Run `python .\src\methods_w2.py --list` to list them.

### Task 2: Retrieval Comparison

Run a W2 method on a museum/query pair (add `--ground-truth` to print mAP):
```powershell
python .\src\methods_w2.py --museum ..\..\datasets\BBDD --queries ..\..\datasets\qsd1_w2 --method "Lab blocks 1D" --blocks 25
```

`--method` is required: a descriptor name or `all` (evaluates the four methods
in one pass). Use `--blocks`/`--levels` for the parametrized methods and
`--distance` for the comparison (default L1).

For example, on QSD1-W2:
```powershell
python .\src\methods_w2.py --museum data\BBDD --queries data\qsd1_w2 --ground-truth data\qsd1_w2\qsd1_w1\gt_corresps.pkl --method all
```

mAP on the 30 QSD1-W2 queries against the BBDD (L1):

| Descriptor | Config | mAP@1 | mAP@5 |
|---|---|---:|---:|
| W1 HSV grid | 4×4 | 0.7667 | 0.8194 |
| W1 HSV grid | 6×6 | 0.8000 | 0.8500 |
| `Lab blocks 1D` | `--blocks 16` | 0.8333 | 0.8722 |
| `Lab blocks 1D` | `--blocks 25` (default) | 0.8667 | 0.8944 |
| `Lab blocks 1D` | `--blocks 36` | 0.8667 | 0.9000 |
| `Lab pyramid` | `--levels 3` | 0.6333 | 0.6744 |
| `Lab pyramid` | `--levels 4` | 0.7333 | 0.7778 |
| `Lab 2D` | `--blocks 1` (1×1) | 0.5333 | 0.6028 |
| `Lab 2D` | `--blocks 25` (default) | 0.8667 | 0.8917 |
| `Lab 3D` | `--blocks 1` (1×1) | 0.4000 | 0.4833 |
| `Lab 3D` | `--blocks 25` (default) | 0.7000 | 0.7472 |

`Lab blocks 1D --blocks 36` gives the best mAP@5 on this set (tied with the
default `--blocks 25` at rank 1). Evaluation with segmentation and blind
queries is still pending.

The extra experiments and their results are in
[results/week2/color_experiments](results/week2/color_experiments).

### Task 3: Background Removal

Method:
1. The outer 4% of the image is used as a sample of the wall.
2. A smooth color model of the wall is fitted in CIE-Lab (a polynomial surface,
   so uneven lighting is handled; frame corners or labels in the strip are
   down-weighted).
3. Each pixel's Mahalanobis distance to that wall model is computed; pixels far
   from the wall are foreground. The threshold is chosen per image (Otsu,
   clipped to 3-12).
4. Cleanup: morphology, keep the largest connected blob, fill holes, and replace
   the blob by its bounding box (paintings are rectangles). Use
   `BGParams(use_bbox=False)` for the pure color mask.

No edge, contour or object detector is used. Masks are saved with the same name
as the image and a `.png` extension (`00001.jpg` -> `00001.png`), values 0
(background) / 255 (painting). `--viz` saves one overlay per image plus
`overview.jpg` (dimmed = removed background, red = our mask, green =
ground truth, F1 on every tile). If the ground-truth masks sit next to the
images (QSD2), the mean P/R/F1 is printed.

### Task 4: Segmentation Evaluation

Run after generating the masks from Task 3:
```powershell
python src\methods_w2.py --segment data\qsd2_w2 --out results\week2\QSD2\masks --viz results\week2\QSD2\viz 
```

The script matches PNG masks by numeric filename ID and computes foreground
precision, recall and F1 for each query. Masks must have the same dimensions
as their reference, with 0 for background and nonzero pixels for foreground.

The table shows the scores for each query and their mean. Use `--save` to
write it to a text file.

Add `--viz` to also draw the masks over the images (`--images` if the `.jpg`
files are not in the ground-truth folder):
```powershell
python src\methods_w2.py --segmentation results\week2\QSD2\masks data\qsd2_w2 --viz results\week2\QSD2\viz --save results\week2\QSD2\mask_metrics.txt
```

Result of Task 3 on the 30 QSD2 images:

| | Precision | Recall | F1 |
|---|---:|---:|---:|
| Mean over the 30 images | 0.9370 | 0.9827 | **0.9569** |

19 of 30 masks have F1 >= 0.95. The weakest are 00020 (white painting on a
light wall, 0.84), 00026, 00014, 00000 and 00013. Most of the lost precision is
the shadow or lighter strip of wall just below a painting.

### Task 5: Retrieval on QSD2 with Background Removal

Add `--background-removal` to the retrieval command: each query is segmented
with Task 3 and the descriptor only counts the painting pixels.
```powershell
python src\methods_w2.py --museum data\BBDD --queries data\qsd2_w2 --ground-truth data\qsd2_w2\qsd2_w1\gt_corresps.pkl --method "Lab pyramid" --levels 7 --background-removal
```

Museum images contain only the painting, but a query has wall around it. The
block grid is laid over the image, so cell (i, j) of the query would cover a
different part of the painting than cell (i, j) of the museum image, even when
the wall pixels are masked out. The query is therefore **cropped to the
bounding box of its mask** before the descriptor is computed. `--no-crop` turns
this off.

mAP@1 / mAP@5 on QSD2 (L1):

| Descriptor | **Mask + crop** | Ground-truth mask + crop |
|---|---:|---:|
| `Lab pyramid --levels 7` | **0.8333 / 0.8483** | 0.8667 / 0.8750 |
| `Lab blocks 1D --blocks 36` | 0.7667 / 0.8139 | 0.8667 / 0.8844 |
| `Lab blocks 1D --blocks 16` | 0.7000 / 0.7556 | 0.7333 / 0.7890 |

The last column uses the ground-truth masks (`--query-masks data\qsd2_w1`) and is
a comparison with the annotated masks. `--background-removal` and
`--query-masks` cannot be used together.

### Task 6: Blind Submission (QST1 and QST2)

```powershell
python src\methods_w2.py --museum data\BBDD --qst1 data\qst_w2\qst1_w2 --qst2 data\qst_w2\qst2_w2
```

Give `--qst1` and/or `--qst2` (the folders must contain the `.jpg` files
directly). Use `--submit-out` to change the output folder (default
`results/week2`) and `--distance` to change the comparison.

| Output under `results/week2/` | Content |
|---|---|
| `QST1/method1/result.pkl` | `Lab blocks 1D`, 6x6 cells (`--blocks 36`), L1, whole image |
| `QST1/method2/result.pkl` | `Lab pyramid`, 7 levels (`--levels 7`), L1, whole image |
| `QST2/method1/result.pkl`, `QST2/method2/result.pkl` | same methods on the painting only (Task 3 mask + crop) |
| `QST2/method1/*.png`, `QST2/method2/*.png` | background masks, same name as the query, 0/255 |

Each `result.pkl` is a Python list of lists with exactly 10 unique integer
museum IDs per query, best first. After writing, the script reads every file
back and checks the IDs, and that each mask exists, has the size of its query
and contains only 0 and 255. The methods are defined in `SUBMISSION_METHODS`
in `methods_w2.py`.

---

## Week 3

The development sets are in `data/qsd1_w3` and `data/qsd2_w3`. Use the JPG
files in these folders; `non_augmented/` contains the clean reference copies
and is not used as query input.

Tasks 3 and 4 are in `paintings_w3.py`. This file reuses the W2 wall model,
descriptor functions, ranking, mask evaluation and overlays. It is separate
from `methods_w3.py` so work on filtering (Task 1) and texture descriptors
(Task 2) can continue independently.

### Task 3: Background Removal for Multiple Paintings

```powershell
python src/paintings_w3.py --segment data/qsd2_w3 --out results/week3/QSD2/masks --viz results/week3/QSD2/viz
```

The method follows the W2 Lab wall model and Mahalanobis distance. The wall
is sampled from the outer 2% of the image. Closing connects thin frame edges
before opening removes small speckles. Instead of keeping only the largest
component, up to two components are kept, each above 2% of the image area.
Overlapping boxes are merged and each painting gets its own bounding box.
The mask is the union of these boxes, with the wall between them left out.

Masks keep the query name and original dimensions, with 0 for background and
255 for foreground. `segmentation.json` records crop boxes and per-image
metrics. The overlays use the W2 convention: red prediction, green ground
truth. Evaluate saved masks with the same W2 metrics:

```powershell
python src/paintings_w3.py --segmentation results/week3/QSD2/masks data/qsd2_w3 --save results/week3/QSD2/mask_metrics.txt
```

Results on the 30 augmented QSD2-W3 queries:

| Method | Precision | Recall | F1 | Correct painting count |
|---|---:|---:|---:|---:|
| W2, largest component only | 0.9631 | 0.7330 | 0.8076 | 21/30 |
| W3, two components, W2 4% border | 0.9622 | 0.9227 | 0.9348 | 29/30 |
| W3, two components, 2% border | 0.9577 | 0.9573 | **0.9526** | **29/30** |

The border width was selected on this development set. The comparison is
saved in `results/week3/QSD2/segmentation_comparison.csv`. Query 00002 still
loses the smaller painting; 00010 loses the dark lower part of the painting.

### Task 4: Retrieval per Painting

```powershell
python src/paintings_w3.py --museum data/BBDD --queries data/qsd2_w3 --background-removal --ground-truth data/qsd2_w3/gt_corresps.pkl --output-dir results/week3/QSD2/lab_pyramid_baseline
```

Each painting is cropped separately, then compared with the museum images.
Paintings in the same row are ordered left to right; vertically separated
paintings are ordered top to bottom. Queries follow numeric filename order.
If segmentation finds nothing, retrieval uses the full image and records
the fallback; the saved segmentation mask remains empty.

The default is the W2 Lab pyramid with 7 levels and L1 distance. This is a
baseline for checking the multiple-painting pipeline, before integrating
the Task 2 texture descriptor:

| Descriptor | mAP@1 | mAP@5 |
|---|---:|---:|
| W2 Lab pyramid, 7 levels, predicted crops | 0.7436 | 0.7564 |

Evaluation computes AP separately for each painting in annotation order.
Missing or extra paintings contribute zero; the mean is taken over painting
slots, so an image with two paintings contributes two scores. Count accuracy
is reported separately. These results use the development correspondences.

`retrieve_folder_w3` accepts a `descriptor(rgb)` callable, shared by museum
images and query crops, and an optional `query_filter(rgb)` applied only to
queries. The CLI accepts them as `--descriptor module:function` and
`--query-filter module:function`. The existing Task 1 median filter can be
selected with `--query-filter methods_w3:median_filter`. The baseline above
does not apply a query filter.

Outputs in the chosen folder:

| File | Content |
|---|---|
| `result.pkl` | query -> painting -> 10 ranked integer museum IDs |
| `*.png`, `segmentation.json` | masks and crop boxes when using background removal |
| `retrieval.json` | rankings, configuration, fallbacks and evaluation metrics |

Even a query with one painting has the extra list level: `[[[7, 25, ...]],
[[44, 81, ...], [3, 9, ...]], ...]`. The result file is read back and checked
for list sizes, unique IDs and Python integers. Blind QST outputs and the
final descriptor selection are pending.
