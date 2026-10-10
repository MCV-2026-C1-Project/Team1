"""Task 4: evaluate predicted foreground masks against QSD2 ground truth."""

import argparse
from pathlib import Path

import cv2

from task4 import image_id
from utils import compute_segmentation_metrics


def _mask_paths(folder):
    paths = {}
    for path in Path(folder).iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        index = image_id(path.name)
        if index in paths:
            raise ValueError(f"Duplicate mask ID {index} in {folder}")
        paths[index] = path
    return paths


def evaluate_mask_folders(predicted_folder, ground_truth_folder):
    """Compute the three metrics for each mask and average over queries."""
    predicted = _mask_paths(predicted_folder)
    truth = _mask_paths(ground_truth_folder)
    if not truth or predicted.keys() != truth.keys():
        raise ValueError("Both folders must contain the same mask IDs")

    per_image = []
    for index in sorted(truth):
        predicted_mask = cv2.imread(str(predicted[index]), cv2.IMREAD_GRAYSCALE)
        truth_mask = cv2.imread(str(truth[index]), cv2.IMREAD_GRAYSCALE)
        if predicted_mask is None or truth_mask is None:
            raise ValueError(f"Cannot read mask {index}")
        scores = compute_segmentation_metrics(predicted_mask, truth_mask)
        per_image.append({"id": index, **scores})

    mean = {key: sum(row[key] for row in per_image) / len(per_image)
            for key in ("precision", "recall", "F1")}
    return {"per_image": per_image, "mean": mean}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", type=Path, required=True,
                        help="Folder containing predicted binary PNG masks")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/qsd2_w2"),
                        help="Folder containing the reference PNG masks")
    parser.add_argument("--save", type=Path, help="Save the results table")
    args = parser.parse_args()
    try:
        result = evaluate_mask_folders(args.predicted, args.ground_truth)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    lines = [f"{'Image':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}"]
    for row in result["per_image"]:
        lines.append(f"{row['id']:10d} {row['precision']:10.4f} "
                     f"{row['recall']:10.4f} {row['F1']:10.4f}")
    mean = result["mean"]
    lines.append(f"{'Mean':>10s} {mean['precision']:10.4f} "
                 f"{mean['recall']:10.4f} {mean['F1']:10.4f}")
    print("\n".join(lines))
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
