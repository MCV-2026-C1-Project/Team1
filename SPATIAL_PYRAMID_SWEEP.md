# Cached configuration sweep

From `src/`, run:

```bash
python3 spatial_pyramid_sweep.py \
  --museum /home/asingh/Desktop/uni/Master/C1/datasets/BBDD \
  --queries /home/asingh/Desktop/uni/Master/C1/datasets/dev_set/P2/qsd1_w1 \
  --ground-truth /home/asingh/Desktop/uni/Master/C1/datasets/dev_set/P2/qsd1_w1/gt_corresps.pkl
```

The default sweep evaluates 54 combinations: HSV/RGB/Lab, 1D with 8/32 bins
or joint 3D with 8 bins per channel, pyramid prefixes [1]/[1,2]/[1,2,4],
and L1/Hellinger. Normalization defaults to L1. Results are sorted by mAP@5,
then mAP@1, then smaller descriptor dimension. Outputs are separate from
Week 1 submissions:

- `results/spatial_pyramid/sweep.csv`: all configuration scores.
- `results/spatial_pyramid/sweep.json`: dataset paths, query order, and run metadata.
- `results/spatial_pyramid/cache/`: reusable compressed descriptor arrays.

Each image is decoded once while computing all missing descriptor configurations.
Smaller pyramids reuse prefixes of the largest descriptor and renormalize them,
which preserves the original descriptor definition. All comparisons reuse the
existing Week 1 functions. Repeated sweeps reuse disk caches and skip histogram
computation; cache keys include input paths, sizes, modification times, descriptor
settings, implementation source, and OpenCV version. Images are streamed rather
than all kept in memory. Cache files contain arrays loaded without pickle.

To broaden the sweep, append options such as:

```bash
--color-spaces HSV RGB Lab YCrCb --bins-1d 8 16 32 --bins-3d 4 8 --distances L1 Hellinger Chi-square
```

For a smaller sweep:

```bash
--color-spaces HSV --bins-1d 8 --bins-3d 8 --distances L1
```

`--pyramid-levels 1 2 4` evaluates every prefix of that list. Use
`--normalization l2` or `none` to compare normalization in a separate run;
use `--output ../results/spatial_pyramid/sweep_l2.csv` to preserve earlier
results. `--cache-dir` changes the cache location. Rerunning the same output
path replaces that sweep table. Large joint bin counts grow cubically and can
consume substantial descriptor memory.

With equal per-channel bins, 1D and 3D comparisons keep quantization fixed.
Comparing 1D/32 against 3D/8 also changes quantization; report that distinction.
Within a spatial ablation, keep color space, histogram type, bins,
normalization, and comparison fixed. The top score identifies the best tested
configuration on this development set, not performance on unseen queries.

From the repository root, run the synthetic checks:

```bash
python -m unittest discover -s tests -p 'test_spatial_pyramid*.py' -v
```

The added sweep runner and its tests do not modify the existing descriptor,
single-configuration runner, or other group implementations.
