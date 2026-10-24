import csv
import sys
from pathlib import Path

sys.path.insert(0, "src")
from methods_w2 import _load_ground_truth, _ordered_image_paths, _read_rgb
from methods_w3 import baseline_descriptor, morphology_filter
from k_similarity import DISTANCE_FUNCTIONS, mean_average_precision_at_k, retrieve_all_queries

def run_morphology_sweep():
    museum_path = "data/BBDD"
    queries_path = "data/qsd1_w3"
    gt_path = "data/qsd1_w3/gt_corresps.pkl"
    output_csv = Path("results/week3/morphology_sweep.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("Loading museum descriptors...")
    museum_ids, museum_paths = _ordered_image_paths(museum_path)
    raw_museum = [_read_rgb(p) for p in museum_paths]
    museum_descs = [baseline_descriptor(img) for img in raw_museum]

    print("Loading queries...")
    query_ids, query_paths = _ordered_image_paths(queries_path)
    raw_queries = [_read_rgb(p) for p in query_paths]
    ground_truth = _load_ground_truth(gt_path, query_ids)

    dist_fn = DISTANCE_FUNCTIONS["L1"]

    results = []

    # 1. No filter baseline
    print("Evaluating No filter...")
    query_descs_none = [baseline_descriptor(img) for img in raw_queries]
    rankings_none = retrieve_all_queries(query_descs_none, museum_descs, dist_fn, top_k=5)
    ranked_ids_none = [[museum_ids[idx] for idx, _ in row] for row in rankings_none]
    map1_none = mean_average_precision_at_k(ranked_ids_none, ground_truth, 1)
    map5_none = mean_average_precision_at_k(ranked_ids_none, ground_truth, 5)
    results.append({"filter": "No filter", "k": "-", "mAP@1": map1_none, "mAP@5": map5_none})
    print(f"No filter: mAP@1={map1_none:.4f}, mAP@5={map5_none:.4f}")

    # 2. Morphological filters
    operations = ["erode", "dilate", "open", "close"]
    ksizes = [3, 5, 7, 9, 11, 13, 15]

    for op in operations:
        for k in ksizes:
            print(f"Evaluating {op} (k={k})...")
            q_descs = [
                baseline_descriptor(morphology_filter(img, op, ksize=k))
                for img in raw_queries
            ]
            rankings = retrieve_all_queries(q_descs, museum_descs, dist_fn, top_k=5)
            ranked_ids = [[museum_ids[idx] for idx, _ in row] for row in rankings]
            map1 = mean_average_precision_at_k(ranked_ids, ground_truth, 1)
            map5 = mean_average_precision_at_k(ranked_ids, ground_truth, 5)
            results.append({"filter": op, "k": k, "mAP@1": map1, "mAP@5": map5})
            print(f"{op} k={k}: mAP@1={map1:.4f}, mAP@5={map5:.4f}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filter", "k", "mAP@1", "mAP@5"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filter": r["filter"],
                "k": r["k"],
                "mAP@1": f"{r['mAP@1']:.4f}",
                "mAP@5": f"{r['mAP@5']:.4f}"
            })

    print(f"\nSaved results to {output_csv}")

if __name__ == "__main__":
    run_morphology_sweep()
