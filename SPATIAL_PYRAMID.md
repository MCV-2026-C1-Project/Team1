# Spatial pyramid experiments

This implementation is isolated from the Week 1 entry points and descriptors.
Run commands from the repository root with requirements.txt installed.

```bash
python -m unittest discover -s tests -p 'test_spatial_pyramid.py' -v
```

Run 1D histograms:

```bash
python src/spatial_pyramid_experiment.py --museum ../../datasets/BBDD --queries ../../datasets/dev_set/P1/qsd1_w1 --ground-truth ../../datasets/dev_set/P1/qsd1_w1/gt_corresps.pkl --histogram-type 1d --color-space HSV --bins 32 --pyramid-levels 1 2 4 --normalization l1 --distance L1
```

Run true joint 3D histograms:

```bash
python src/spatial_pyramid_experiment.py --museum ../../datasets/BBDD --queries ../../datasets/dev_set/P1/qsd1_w1 --ground-truth ../../datasets/dev_set/P1/qsd1_w1/gt_corresps.pkl --histogram-type 3d --color-space HSV --bins 8 8 8 --pyramid-levels 1 2 4 --normalization l1 --distance L1
```

Adjust folder and ground-truth paths to your dataset. Ground truth contains lists
of relevant numeric museum IDs, aligned with queries sorted by numeric filename
suffix. Load only trusted pickle files. Omitting ground truth prints top-five
retrieval IDs without evaluation. The runner writes no result files.

For the spatial ablation, repeat a command with `--pyramid-levels 1`, then
`--pyramid-levels 1 2`, then `--pyramid-levels 1 2 4`, keeping every other
setting fixed. For the histogram ablation, keep the pyramid and color space
fixed and change `--histogram-type 1d` to `3d`. Keeping bins fixed isolates
the histogram structure, though descriptor dimensions differ. Using 32 bins
for 1D and 8 for 3D also changes quantization, so report both settings.
Use `--histogram-type 3d --pyramid-levels 1` for the global joint baseline.

The API is `compute_spatial_pyramid(image_rgb, pyramid_levels=(1,2,4),
histogram_type="1d", color_space="HSV", bins=32, normalization="l1")`.
It accepts uint8 RGB input, integer or three-channel bin counts, and RGB, HSV,
Lab, YCrCb, or YCbCr. YCbCr orders channels Y,Cb,Cr, while YCrCb orders
Y,Cr,Cb. OpenCV uint8 HSV hue spans [0,180); all other channels span [0,256).

Levels are grid sizes, processed in supplied order. Cells are nonoverlapping,
row-major, and cover the full image including uneven dimensions. Images smaller
than the largest grid are rejected. Joint histograms are computed in three
dimensions before C-order flattening. Dimensions are
`sum(grid**2) * sum(bins)` for 1D and `sum(grid**2) * product(bins)` for 3D.
The examples produce 2,016 and 10,752 values respectively.

L1 or L2 normalization is applied per region and again after concatenation;
`none` retains raw counts. There are no additional level weights. With L1,
each cell has equal mass, so levels 1,2,4 have mass proportions 1:4:16.
Normalization and quantization should stay fixed within each spatial ablation.

Existing utilities provide RGB image loading, region splitting, RGB/HSV 1D
histograms, L1 normalization, numeric filename IDs, all seven comparisons,
retrieval, and mAP. New code adds configurable conversion, joint histograms,
pyramid concatenation, and the experiment CLI. Existing files are unchanged.
