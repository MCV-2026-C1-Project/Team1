"""Open+Close kernel sweep: does 7x7 really beat 11x11?

Checks the claim under every relevant configuration, because the answer
depends on two flags that change the numbers:

    scope  : "query" (filters only the queries) vs "both" (Task 1 contract,
             the current default)
    hue    : hue-invariant on (the figures) vs off

Outputs (appended as it goes, safe to interrupt):
    results/week3/kernel_check/open_close_kernels.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from k_similarity import mean_average_precision_at_k  # noqa: E402
from methods_w2 import _load_ground_truth, _ordered_image_paths  # noqa: E402
from methods_w3 import DEFAULT_FILTER_SCOPE, evaluate_folders_w3  # noqa: E402

MUSEUM = ROOT / "data" / "BBDD"
QUERIES = ROOT / "data" / "qsd1_w3"
GROUND_TRUTH = QUERIES / "gt_corresps.pkl"
HUE_REFERENCE = (ROOT / "results" / "week3" / "color_reference"
                 / "museum_hue_references.npz")
CSV = Path(__file__).resolve().parent / "open_close_kernels.csv"

KERNELS = (3, 5, 7, 9, 11, 13, 15)
CONFIGS = [("query", False), ("both", False),
           ("query", True), ("both", True)]
FIELDS = ["filter_scope", "hue_invariant", "kernel", "mAP@1", "mAP@5"]


def _metrics(ranked_ids):
    _, query_paths = _ordered_image_paths(str(QUERIES))
    query_ids = [int(Path(p).stem) for p in query_paths]
    ground_truth = _load_ground_truth(str(GROUND_TRUTH), query_ids)
    return (mean_average_precision_at_k(ranked_ids, ground_truth, 1),
            mean_average_precision_at_k(ranked_ids, ground_truth, 5))


def _run(scope, hue, kernel):
    ranked_ids = evaluate_folders_w3(
        str(MUSEUM), str(QUERIES), "Lab blocks 1D open+close",
        ground_truth_path=str(GROUND_TRUTH), ksize=kernel,
        filter_scope=scope, hue_invariant=hue,
        hue_reference=str(HUE_REFERENCE) if hue else None)
    return _metrics(ranked_ids)


def main():
    done = set()
    if CSV.exists():
        with CSV.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                done.add((row["filter_scope"], row["hue_invariant"],
                          int(row["kernel"])))
    new = not CSV.exists()
    with CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for scope, hue in CONFIGS:
            for kernel in KERNELS:
                if (scope, str(hue), kernel) in done:
                    continue
                map1, map5 = _run(scope, hue, kernel)
                writer.writerow({"filter_scope": scope, "hue_invariant": hue,
                                 "kernel": kernel, "mAP@1": map1,
                                 "mAP@5": map5})
                handle.flush()
                print(f"scope={scope:<5} hue={str(hue):<5} k={kernel:>2} "
                      f"mAP@1={map1:.4f} mAP@5={map5:.4f}", flush=True)
    with CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for scope, hue in CONFIGS:
        part = [row for row in rows
                if row["filter_scope"] == scope
                and row["hue_invariant"] == str(hue)]
        if part:
            top = max(part, key=lambda row: float(row["mAP@5"]))
            print(f"best for scope={scope} hue={hue}: "
                  f"{top['kernel']}x{top['kernel']} "
                  f"(mAP@1={float(top['mAP@1']):.4f} "
                  f"mAP@5={float(top['mAP@5']):.4f})", flush=True)
    print(f"Default filter scope in code: {DEFAULT_FILTER_SCOPE}")


if __name__ == "__main__":
    main()
