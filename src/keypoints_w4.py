"""Extract and visualize SIFT local features."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reuse the project's existing image-loading and grayscale helpers.
from methods_w2 import _read_rgb
from paintings_w3 import to_gray


def extract_sift(image_rgb, nfeatures=1000):
    """Return keypoints and their corresponding 128-value descriptors."""

    if (
        isinstance(nfeatures, bool)
        or not isinstance(nfeatures, int)
        or nfeatures < 1
    ):
        raise ValueError("nfeatures must be a positive integer")

    # The existing helper returns float32; SIFT expects uint8 input.
    gray = to_gray(image_rgb).astype(np.uint8)

    # Create the SIFT detector and descriptor extractor.
    sift = cv2.SIFT_create(nfeatures=nfeatures)

    # Detect keypoints and compute one descriptor per returned keypoint.
    # None means that no foreground mask is supplied.
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # A flat image may have no detectable features.
    # Return an empty matrix so downstream code can handle it explicitly.
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)

    return keypoints, descriptors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--nfeatures", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        help="Save the visualization instead of opening a plot window",
    )
    args = parser.parse_args()

    image_rgb = _read_rgb(args.image)

    keypoints, descriptors = extract_sift(
        image_rgb,
        nfeatures=args.nfeatures,
    )

    print(f"Image shape: {image_rgb.shape}")
    print(f"Detected keypoints: {len(keypoints)}")
    print(f"Descriptor matrix shape: {descriptors.shape}")
    print(f"Descriptor data type: {descriptors.dtype}")

    if keypoints:
        first = keypoints[0]
        print("\nFirst keypoint:")
        print(f"  Position (x, y): {first.pt}")
        print(f"  Size: {first.size:.2f}")
        print(f"  Orientation: {first.angle:.2f} degrees")
        print(f"  Response: {first.response:.6f}")
        print(f"  First 8 descriptor values: {descriptors[0, :8]}")

    visualization = cv2.drawKeypoints(
        image_rgb,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Visualization is RGB; OpenCV's image writer expects BGR.
        visualization_bgr = cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR,
        )

        if not cv2.imwrite(str(args.output), visualization_bgr):
            raise OSError(f"Could not save visualization: {args.output}")

        print(f"\nVisualization saved to: {args.output}")
    else:
        plt.figure(figsize=(14, 8))
        plt.imshow(visualization)
        plt.title(f"SIFT: {len(keypoints)} keypoints")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()