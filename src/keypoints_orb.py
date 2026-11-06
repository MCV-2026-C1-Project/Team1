"""Extract and visualize ORB (oriented FAST + rotated BRIEF) local features.

Descriptors are 256-bit binary strings (32 uint8 values) compared with the
Hamming distance. This file is the ORB counterpart of keypoints_w4.py (SIFT);
extract_features_w4_orb.py saves the features of a folder.

Two variants share this code and differ only in two extraction settings:

    v1  FAST threshold 20 (OpenCV default), no contrast equalisation
    v2  FAST threshold 7 and CLAHE on the grayscale crop (default, final method):
        faded or colour-changed paintings give far more keypoints

    python keypoints_w4_orb.py --image data/qsd1_w4/00000.jpg --variant v2 --output results/week4/orb_v2/00000_kp.jpg
    python keypoints_w4_orb.py --image data/qsd1_w4/00000.jpg --variant v1 --output results/week4/orb/00000_kp.jpg

--fast-threshold N and --clahe/--no-clahe override a variant's setting
(the variant is then recorded as "custom").
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# Reuse the project's existing image-loading and grayscale helpers.
from methods_w2 import _read_rgb
from paintings_w3 import to_gray

DEFAULT_NFEATURES = 1000
DEFAULT_MAX_SIDE = 384          # crops are resized to this longest side
DESCRIPTOR_BYTES = 32           # 256-bit BRIEF

ORB_VARIANTS = {
    "v1": {"fast_threshold": 20, "clahe": False},   # OpenCV defaults
    "v2": {"fast_threshold": 7, "clahe": True},     # low-contrast paintings
}
DEFAULT_VARIANT = "v2"


def variant_settings(variant=DEFAULT_VARIANT, fast_threshold=None, clahe=None):
    """Resolve a variant name plus optional overrides to the extraction settings.

    Returns {"variant": name or "custom", "fast_threshold": int, "clahe": bool}.
    """
    if variant not in ORB_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(ORB_VARIANTS)}")
    settings = dict(ORB_VARIANTS[variant])
    if fast_threshold is not None:
        if isinstance(fast_threshold, bool) or not isinstance(fast_threshold, int) \
                or fast_threshold < 1:
            raise ValueError("fast_threshold must be a positive integer")
        settings["fast_threshold"] = fast_threshold
    if clahe is not None:
        settings["clahe"] = bool(clahe)
    name = variant if settings == ORB_VARIANTS[variant] else "custom"
    return {"variant": name, **settings}


def extract_orb(image_rgb, nfeatures=DEFAULT_NFEATURES, max_side=DEFAULT_MAX_SIDE,
                variant=DEFAULT_VARIANT, fast_threshold=None, clahe=None):
    """Return keypoints (in the coordinates of ``image_rgb``) and uint8 descriptors.

    variant: "v1" or "v2" (see the module docstring). fast_threshold and clahe
    override the variant: a lower FAST threshold finds keypoints in faded or
    low-contrast paintings, and CLAHE equalises the local contrast first.
    """
    settings = variant_settings(variant, fast_threshold, clahe)
    if isinstance(nfeatures, bool) or not isinstance(nfeatures, int) or nfeatures < 1:
        raise ValueError("nfeatures must be a positive integer")
    gray = to_gray(image_rgb).astype(np.uint8)

    # Work at a fixed size so that the detector is tuned the same way for every crop.
    h, w = gray.shape
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                          interpolation=cv2.INTER_AREA)

    if settings["clahe"]:
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8,
                         fastThreshold=settings["fast_threshold"])
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        return [], np.empty((0, DESCRIPTOR_BYTES), dtype=np.uint8)

    # Map the keypoints back to the crop's own resolution.
    keypoints = [cv2.KeyPoint(k.pt[0] / scale, k.pt[1] / scale, k.size / scale,
                              k.angle, k.response, k.octave) for k in keypoints]
    return keypoints, descriptors


def show_one(args):
    image = _read_rgb(args.image)
    keypoints, descriptors = extract_orb(image, args.nfeatures, args.max_side, args.variant,
                                         args.fast_threshold, args.clahe)
    print(f"Variant: {variant_settings(args.variant, args.fast_threshold, args.clahe)}")
    print(f"Image shape: {image.shape}")
    print(f"Detected keypoints: {len(keypoints)}")
    print(f"Descriptor matrix shape: {descriptors.shape}, dtype {descriptors.dtype}")
    if keypoints:
        first = keypoints[0]
        print(f"First keypoint: position {first.pt}, size {first.size:.1f}, "
              f"angle {first.angle:.1f}, response {first.response:.5f}")
        print(f"First 8 descriptor bytes: {descriptors[0, :8]}")
    drawing = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.cvtColor(drawing, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save {args.output}")
    print(f"Visualization saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--nfeatures", type=int, default=DEFAULT_NFEATURES)
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument("--variant", choices=sorted(ORB_VARIANTS), default=DEFAULT_VARIANT)
    parser.add_argument("--fast-threshold", type=int, default=None,
                        help="override the variant's FAST threshold")
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=None,
                        help="override the variant's contrast equalisation")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to save the keypoint drawing")
    show_one(parser.parse_args())


if __name__ == "__main__":
    main()
