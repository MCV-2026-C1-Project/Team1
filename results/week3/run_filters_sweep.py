import csv
import pickle
import sys
from pathlib import Path
import numpy as np
import pywt

sys.path.insert(0, "src")
from methods_w2 import _load_ground_truth, _ordered_image_paths, _read_rgb
from methods_w3 import baseline_descriptor, median_filter, morphology_filter
from k_similarity import DISTANCE_FUNCTIONS, mean_average_precision_at_k, retrieve_all_queries


def wavelet_lowpass_filter(img_rgb, wavelet="db2", level=1):
    img = np.asarray(img_rgb)
    H, W = img.shape[:2]
    channels = []
    for c in range(img.shape[2]):
        channel = img[..., c].astype(np.float64)
        coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=level)
        # Keep approximation (LL), zero out all detail coefficients
        filtered_coeffs = [coeffs[0]] + [
            (np.zeros_like(h), np.zeros_like(v), np.zeros_like(d))
            for (h, v, d) in coeffs[1:]
        ]
        rec = pywt.waverec2(filtered_coeffs, wavelet=wavelet)
        rec = rec[:H, :W]
        channels.append(np.clip(np.rint(rec), 0, 255).astype(np.uint8))
    return np.stack(channels, axis=-1)


def main():
    museum_path = "data/BBDD"
    queries_path = "data/qsd1_w3"
    gt_path = "data/qsd1_w3/gt_corresps.pkl"
    cache_file = Path("results/week3/bbdd_descs_cache.pkl")
    output_csv = Path("results/week3/filters_comparison.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Museum descriptors (with cache)
    museum_ids, museum_paths = _ordered_image_paths(museum_path)
    if cache_file.exists():
        print(f"Loading museum descriptors from {cache_file}...")
        with open(cache_file, "rb") as f:
            museum_descs = pickle.load(f)
    else:
        print("Computing museum descriptors...")
        raw_museum = [_read_rgb(p) for p in museum_paths]
        museum_descs = [baseline_descriptor(img) for img in raw_museum]
        with open(cache_file, "wb") as f:
            pickle.dump(museum_descs, f)
        print("Saved museum descriptors cache.")

    # 2. Queries and Ground Truth
    print("Loading queries...")
    query_ids, query_paths = _ordered_image_paths(queries_path)
    raw_queries = [_read_rgb(p) for p in query_paths]
    ground_truth = _load_ground_truth(gt_path, query_ids)

    dist_fn = DISTANCE_FUNCTIONS["L1"]
    results = []

    def evaluate(filter_name, param_str, filter_fn):
        q_descs = [baseline_descriptor(filter_fn(img)) for img in raw_queries]
        rankings = retrieve_all_queries(q_descs, museum_descs, dist_fn, top_k=5)
        ranked_ids = [[museum_ids[idx] for idx, _ in row] for row in rankings]
        map1 = mean_average_precision_at_k(ranked_ids, ground_truth, 1)
        map5 = mean_average_precision_at_k(ranked_ids, ground_truth, 5)
        print(f"{filter_name:<16} {param_str:>10} | mAP@1: {map1:.4f}  mAP@5: {map5:.4f}")
        results.append({
            "filter": filter_name,
            "parameter": param_str,
            "mAP@1": f"{map1:.4f}",
            "mAP@5": f"{map5:.4f}"
        })

    print(f"\n{'Filter':<16} {'Parameter':>10} | {'mAP@1':>7}  {'mAP@5':>7}")
    print("-" * 46)

    # Baseline: No filter
    evaluate("No filter", "-", lambda img: img)

    # Median filter (odd kernels 3 to 15)
    ksizes = [3, 5, 7, 9, 11, 13, 15]
    for k in ksizes:
        evaluate("median", f"{k}x{k}", lambda img, k=k: median_filter(img, ksize=k))

    # Morphological filters
    for op in ["erode", "dilate", "open", "close"]:
        for k in ksizes:
            evaluate(op, f"{k}x{k}", lambda img, op=op, k=k: morphology_filter(img, op, ksize=k))

    # Wavelet low-pass filters
    wavelets = ["haar", "db2", "db4", "sym4"]
    levels = [1, 2, 3]
    for w in wavelets:
        for lvl in levels:
            evaluate("wavelet_lowpass", f"{w}_lvl{lvl}",
                     lambda img, w=w, lvl=lvl: wavelet_lowpass_filter(img, wavelet=w, level=lvl))

    # Save CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filter", "parameter", "mAP@1", "mAP@5"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll results saved to {output_csv}")


if __name__ == "__main__":
    main()
