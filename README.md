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

The development sets are in `data/qsd1_w3` (30 cropped queries, one painting each)
and `data/qsd2_w3` (30 queries with up to two paintings). Both contain noise and
hue changes. Use the JPG files in these folders; `non_augmented/` contains the
clean reference copies and is not used as query input.

| File | Content |
|---|---|
| `methods_w3.py` | Task 1: query filters (Median, Open, Close, Open+Close, Close+Open, Average, Gaussian, Low-pass Wavelet) plus the HSV hue-invariance mode |
| `paintings_w3.py` | Task 2: texture descriptors (LBP, Multiscale LBP, DCT, Wavelet, and the improved DCT log and Texture fusion); Tasks 3, 4 and 5: multi-painting segmentation, retrieval per painting, blind submission |

`paintings_w3.py` reuses the W2 wall model, ranking, mask evaluation and
overlays. The texture descriptors are the first part of the same file, so
`paintings_w3.py` is self-contained apart from the W2 modules; the median filter
of Task 1 stays in `methods_w3.py` and is selected with `--query-filter`.

### Task 1: Noise Filtering

`methods_w3.py` registers one `Lab blocks 1D <filter>` method per filter
(Median, Open, Close, Open+Close, Close+Open, Average, Gaussian, Low-pass
Wavelet) sharing a `--kernel` size, plus an HSV hue-invariance mode
(`--hue-invariant`). The filter is applied to **both** the queries and the
museum images by default (`--filter-scope both`), so both sides share the
same filtered domain; `--filter-scope query` restores queries-only.

```powershell
python src/methods_w3.py --museum data/BBDD --queries data/qsd1_w3 --method all --ground-truth data/qsd1_w3/gt_corresps.pkl
```

### Task 2: Texture Descriptors

All descriptors work on the grayscale image, so the random hue changes of the
W3 queries cannot affect them. The image is split into 5x5 = 25 blocks and the
block features are concatenated, which keeps the position of the texture.

| Method | Description | Size |
|---|---|---:|
| `LBP` | 3x3 Local Binary Pattern, 256-bin histogram per block | 6400 |
| `LBP uniform` | same codes, 58 uniform patterns plus one bin for the rest | 1475 |
| `Multiscale LBP` | circular LBP at (P,R) = (8,1), (12,2.5), (16,4), bilinear interpolation, uniform patterns, three histograms per block | 10925 |
| `DCT` | 2D DCT per block, zig-zag scan, first 15 coefficients without the DC term | 375 |
| `Wavelet` | 3-level Haar; mean absolute value of each detail subband per block | 225 |
| `DCT log` | DCT with log-compressed coefficients, sign(c)*log(1+abs(c)) | 375 |
| `Texture fusion` | `DCT log`, square-root `Multiscale LBP` and square-root `Wavelet`, each scaled to unit L2 norm and concatenated | 11525 |

Evaluation on QSD1-W3 with texture only: no segmentation (the queries are
already cropped), no color. `--table` prints a method x distance table; run it
without a filter and with the median filter (best distance per method, 30
queries, mAP@1):

```powershell
python src/paintings_w3.py --museum data/BBDD --queries data/qsd1_w3 --ground-truth data/qsd1_w3/gt_corresps.pkl --table --method all --distance all
python src/paintings_w3.py --museum data/BBDD --queries data/qsd1_w3 --ground-truth data/qsd1_w3/gt_corresps.pkl --table --method all --distance all --denoise 3
python src/paintings_w3.py --museum data/BBDD --queries data/qsd1_w3 --ground-truth data/qsd1_w3/gt_corresps.pkl --table --method all --distance all --denoise 5
```

| Descriptor | No filter | Median 3x3 | Median 5x5 |
|---|---:|---:|---:|
| `LBP` | 0.800 | 0.600 | 0.533 |
| `LBP uniform` | 0.667 | 0.600 | 0.533 |
| `Multiscale LBP` | 0.900 | 0.967 | 0.967 |
| `DCT` (L1 or Cosine) | 0.967 | 0.967 | 0.967 |
| `Wavelet` | 0.833 | 0.867 | 0.867 |
| `DCT log` (L1, Cosine, Correlation) | **1.000** | **1.000** | **1.000** |
| `Texture fusion` (L1, Euclidean, Cosine, Correlation) | **1.000** | **1.000** | **1.000** |

`DCT log` and `Texture fusion` find the right painting at rank 1 for all 30
queries, with and without the median filter, so they are robust to the noise
and the hue changes of QSD1-W3. Plain `DCT` is also stable (0.967), while the
LBP histograms depend on the filter: the median hurts plain LBP, which reads the
smoothed texture as a different pattern, but helps Multiscale LBP (best with
Chi-square or Hellinger) and Wavelet. The best distance for histogram
descriptors is Chi-square, Hellinger or L1; Euclidean and Cosine are poor for
Multiscale LBP.

#### Texture fusion (final descriptor)

`Texture fusion` is the descriptor submitted for Tasks 4 and 5. It joins three
texture descriptors that look at the image in different ways, using only the
grayscale image (no color):

| Part | What it captures | How it is computed | Size |
|---|---|---|---:|
| `DCT log` | Frequency content of each block (coarse to fine texture) | gray image resized to 255x255, 25 blocks of 51x51; 2D DCT per block; zig-zag scan; the DC term (mean brightness) is dropped and the next 15 coefficients are kept; each coefficient c becomes sign(c)*log(1+abs(c)) | 375 |
| Multiscale LBP | Local micro-patterns (edges, spots, flat areas) at three scales | gray image with the longest side at 256 px; circular LBP at (P,R) = (8,1), (12,2.5), (16,4); uniform patterns; one histogram per block and scale; L1 normalization; square root | 10925 |
| Wavelet | Oriented detail energy at three scales | gray image resized to 256x256; 3-level Haar; mean absolute value of the horizontal, vertical and diagonal details in each block; L1 normalization; square root | 225 |

The final vector is `[DCT log / ||DCT log||, MLBP / ||MLBP||, Wavelet / ||Wavelet||]`:
every part is divided by its own L2 norm and the three are concatenated with
the same weight (11525 values). No weight was tuned.

Why each step is there:

- **DCT without the DC term.** The DC coefficient is the mean brightness of
  the block, which is not texture and changes with lighting. Skipping it makes
  the descriptor insensitive to brightness shifts.
- **Log on the DCT coefficients.** The low-frequency coefficients are much
  larger than the others, so in a plain L1 or Euclidean distance they dominate
  and the fine texture is ignored. The log compresses that range, so every
  coefficient counts, and it also reduces the effect of noise and contrast
  changes. This step alone moves `DCT` from 0.7436 to 0.7692 on QSD2-W3.
- **Square root on the histograms.** LBP and wavelet values are L1-normalized
  histograms. After a square root, the Euclidean distance between two vectors is
  the Hellinger distance between the histograms, which is a better measure for
  histograms and reduces the weight of very frequent patterns.
- **Unit L2 norm per part.** The three parts have very different sizes (375,
  10925, 225 values) and scales. Without this step the long LBP part would
  dominate the distance; with it, each part contributes comparably.
- **Grayscale only.** The queries have random hue changes, so color is not used.
  LBP is also invariant to any monotonic change of the gray levels.
- **Same treatment for museum and query.** The museum images are never
  filtered; only the query crops go through the 3x3 median filter of Task 1.

Why the fusion works: the three parts are complementary. The DCT describes
global frequency content per block, LBP describes the local structure of the
pixels, and the wavelet describes how much detail there is and in which
direction. A painting that is ambiguous for one of them is usually
separated by another. This is visible in the results: on QSD2-W3 the best single
descriptor reaches mAP@1 0.7692, while the fusion reaches 0.9231 (+0.154), and
on QSD1-W3 it finds all 30 paintings at rank 1.

The default distance is L1; on QSD1-W3 the fusion gives 1.000 with L1,
Euclidean, Cosine and Correlation, so it does not depend on one distance.
Because the vector contains signed DCT values, Chi-square, Histogram
Intersection and Hellinger are not valid for it. The cost is a longer vector
(11525 values) and a slower descriptor computation than a single descriptor,
which is acceptable for 287 museum images and 30 to 50 queries.

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

To evaluate several method outputs stored below one directory (for example
`all_methods/dct`, `all_methods/lbp`, and `all_methods/wavelet`), pass the
parent directory. The saved report contains one labeled section per method:

```powershell
python src/paintings_w3.py --segmentation results/week3/QSD2/all_methods data/qsd2_w3 --save results/week3/QSD2/mask_metrics_all.txt
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
python src/paintings_w3.py --museum data/BBDD --queries data/qsd2_w3 --background-removal --ground-truth data/qsd2_w3/gt_corresps.pkl --query-filter methods_w3:median_filter --output-dir results/week3/QSD2/texture_fusion
```

Each painting is cropped separately, then compared with the museum images.
Paintings in the same row are ordered left to right; vertically separated
paintings are ordered top to bottom. Queries follow numeric filename order.
If segmentation finds nothing, retrieval uses the full image and records
the fallback; the saved segmentation mask remains empty.

The default method is `Texture fusion` with the L1 distance. Choose another one
with `--method NAME` (`--method all` runs every texture method, saving each one
in its own subfolder) and `--distance NAME`.

mAP on the 30 QSD2-W3 queries, L1, predicted crops, 3x3 median filter on the
queries (`--method all`):

| Descriptor | mAP@1 | mAP@5 | mAP@10 |
|---|---:|---:|---:|
| W2 Lab pyramid, 7 levels (baseline, no median) | 0.7436 | 0.7564 | - |
| `LBP uniform` | 0.5128 | 0.5650 | 0.5723 |
| `LBP` | 0.6154 | 0.6603 | 0.6639 |
| `Wavelet` | 0.6923 | 0.7205 | 0.7248 |
| `Multiscale LBP` | 0.7179 | 0.7231 | 0.7231 |
| `DCT` | 0.7436 | 0.7436 | 0.7436 |
| `DCT log` | 0.7692 | 0.8085 | 0.8085 |
| **`Texture fusion`** | **0.9231** | **0.9231** | **0.9231** |

Without the median filter, the first run gave `LBP` 0.6154, `LBP uniform`
0.5641, `Multiscale LBP` 0.6410, `DCT` 0.7436 and `Wavelet` 0.6923 (mAP@1). The
filter helps Multiscale LBP (+0.077) and lowers LBP uniform (-0.051). All
methods share the same masks (F1 0.9526, 29/30 correct painting counts), so
the differences come only from the descriptor.

`Texture fusion` is 0.18 above the best single descriptor. Of the 39 paintings,
36 are correct at rank 1; the missing ones come from the image with a wrong
painting count and from weak masks, so segmentation is now the main limit.

Evaluation computes AP separately for each painting in annotation order.
Missing or extra paintings contribute zero; the mean is taken over painting
slots, so an image with two paintings contributes two scores. Count accuracy
is reported separately. These results use the development correspondences.

`retrieve_folder_w3` accepts a `descriptor(rgb)` callable, shared by museum
images and query crops, and an optional `query_filter(rgb)` applied only to
queries. The CLI accepts them as `--descriptor module:function` and
`--query-filter module:function`, for example
`--descriptor paintings_w3:dct_log_descriptor`.

Outputs in the chosen folder:

| File | Content |
|---|---|
| `result.pkl` | query -> painting -> 10 ranked integer museum IDs |
| `*.png`, `segmentation.json` | masks and crop boxes when using background removal |
| `retrieval.json` | rankings, configuration, fallbacks and evaluation metrics |

Even a query with one painting has the extra list level: `[[[7, 25, ...]],
[[44, 81, ...], [3, 9, ...]], ...]`. The result file is read back and checked
for list sizes, unique IDs and Python integers.

### Task 5: Blind Submission (QST1 and QST2)

Only the best method is submitted, one per test set: `Texture fusion`, L1,
median filter on the queries, top 10. Every `result.pkl` has the extra list
level (a list of one or two painting rankings per query).

QST1-W3 has cropped pictures with one painting, so segmentation is not used and
the whole image is the painting. QST2-W3 has pictures with background and up to
two paintings, so segmentation is used and the masks are submitted.

```powershell
python src/paintings_w3.py --museum data/BBDD --queries data/qst1_w3 --method "Texture fusion" --distance L1 --query-filter methods_w3:median_filter --top-k 10 --output-dir results/week3/QST1/method1
python src/paintings_w3.py --museum data/BBDD --queries data/qst2_w3 --background-removal --method "Texture fusion" --distance L1 --query-filter methods_w3:median_filter --top-k 10 --output-dir results/week3/QST2/method1
```

| Output | Content |
|---|---|
| `Team1/week3/QST1/method1/result.pkl` | 50 queries, one painting ranking each |
| `Team1/week3/QST2/method1/result.pkl` | 30 queries, one or two painting rankings each |
| `Team1/week3/QST2/method1/*.png` | background masks, same name as the query, 0/255 |
