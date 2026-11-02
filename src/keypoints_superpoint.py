"""SuperPoint detection + LightGlue matching for Week 4 comparison.

Mirrors ``keypoints_w4`` / ``matching_w4`` but replaces SIFT + ratio-test
matching with a learned detector + LightGlue assignment, keeping the same
RANSAC homography verification so inlier counts stay comparable.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from methods_w2 import _read_rgb
from paintings_w3 import to_gray

_MODELS = {}


def _device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _models(device=None, max_num_keypoints=1024):
    from lightglue import LightGlue, SuperPoint

    device = _device(device)
    key = (str(device), int(max_num_keypoints))
    if key not in _MODELS:
        extractor = SuperPoint(max_num_keypoints=int(max_num_keypoints)).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)
        _MODELS[key] = (extractor, matcher)
    return _MODELS[key], device


def extract_superpoint(image_rgb, max_num_keypoints=1024, image_size=960, device=None):
    """Detect keypoints and compute 256-dim descriptors.

    Returns a list of ``cv2.KeyPoint`` (for API parity with SIFT) and a
    ``(N, 256)`` float32 descriptor matrix. Coordinates are expressed in the
    original crop resolution.
    """
    if isinstance(max_num_keypoints, bool) or not isinstance(max_num_keypoints, int):
        raise ValueError("max_num_keypoints must be a positive integer")
    if max_num_keypoints < 1:
        raise ValueError("max_num_keypoints must be a positive integer")
    if image_size < 64:
        raise ValueError("image_size must be at least 64")

    (extractor, _), dev = _models(device, max_num_keypoints)
    gray = to_gray(image_rgb).astype(np.float32) / 255.0
    h0, w0 = gray.shape[:2]
    scale = image_size / max(h0, w0)
    if scale < 1.0:
        nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
        small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        small, scale = gray, 1.0
    image_t = torch.from_numpy(small)[None, None].to(dev)
    with torch.no_grad():
        feats = extractor({"image": image_t})
    kpts = feats["keypoints"][0].cpu().numpy() / scale
    desc = feats["descriptors"][0].cpu().numpy().astype(np.float32)
    scores = feats["scores"][0].cpu().numpy() if "scores" in feats else np.zeros(len(kpts))
    keypoints = [
        cv2.KeyPoint(float(x), float(y), 8.0, -1.0, float(s), 0, -1)
        for (x, y), s in zip(kpts, scores)
    ]
    if len(keypoints) == 0:
        desc = np.empty((0, 256), dtype=np.float32)
    return keypoints, desc


def match_superpoint(query, museum, ransac_threshold=5.0, device=None):
    """Match with LightGlue, then verify with a RANSAC homography.

    ``query`` / ``museum`` are dicts with ``points`` (N, 2), ``descriptors``
    (N, 256) and ``box``. Returns the same keys as ``match_sift`` so the
    Week 4 scoring, calibration and rejection code can be reused unchanged.
    """
    if ransac_threshold <= 0:
        raise ValueError("Invalid matching thresholds")

    nq = len(query["descriptors"])
    nm = len(museum["descriptors"])
    if nq == 0 or nm == 0:
        return {"matches": [], "inliers": np.zeros(0, dtype=bool),
                "tentative_count": 0, "inlier_count": 0}

    (_, matcher), dev = _models(device)
    q_pts = torch.from_numpy(np.asarray(query["points"], dtype=np.float32))[None].to(dev)
    m_pts = torch.from_numpy(np.asarray(museum["points"], dtype=np.float32))[None].to(dev)
    q_desc = torch.from_numpy(np.asarray(query["descriptors"], dtype=np.float32))[None].to(dev)
    m_desc = torch.from_numpy(np.asarray(museum["descriptors"], dtype=np.float32))[None].to(dev)
    q_box = np.asarray(query["box"], dtype=np.float32)
    m_box = np.asarray(museum["box"], dtype=np.float32)
    q_size = torch.tensor([[q_box[2] - q_box[0], q_box[3] - q_box[1]]]).to(dev)
    m_size = torch.tensor([[m_box[2] - m_box[0], m_box[3] - m_box[1]]]).to(dev)
    with torch.no_grad():
        out = matcher({
            "image0": {"keypoints": q_pts, "descriptors": q_desc, "image_size": q_size},
            "image1": {"keypoints": m_pts, "descriptors": m_desc, "image_size": m_size},
        })
    pairs = out["matches"][0].cpu().numpy()
    tentative = len(pairs)

    inliers = np.zeros(tentative, dtype=bool)
    matches = []
    if tentative >= 8:
        qp = np.asarray(query["points"], dtype=np.float32)[pairs[:, 0]]
        mp = np.asarray(museum["points"], dtype=np.float32)[pairs[:, 1]]
        ref_q = qp * (1000.0 / max(q_box[2] - q_box[0], q_box[3] - q_box[1]))
        ref_m = mp * (1000.0 / max(m_box[2] - m_box[0], m_box[3] - m_box[1]))
        cv2.setRNGSeed(0)
        try:
            homography, mask = cv2.findHomography(ref_q, ref_m, cv2.RANSAC, ransac_threshold)
        except cv2.error:
            homography, mask = None, None
        if homography is not None and mask is not None and np.isfinite(homography).all():
            inliers = mask.ravel().astype(bool)

    class _M:
        def __init__(self, qi, ti):
            self.queryIdx, self.trainIdx = int(qi), int(ti)

    matches = [_M(qi, ti) for qi, ti in pairs]
    return {"matches": matches, "inliers": inliers,
            "tentative_count": int(tentative), "inlier_count": int(inliers.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--output", type=Path,
                        help="Save the visualization instead of opening a plot window")
    args = parser.parse_args()

    image_rgb = _read_rgb(args.image)
    keypoints, descriptors = extract_superpoint(
        image_rgb, max_num_keypoints=args.max_keypoints, image_size=args.image_size)
    print(f"Image shape: {image_rgb.shape}")
    print(f"Detected keypoints: {len(keypoints)}")
    print(f"Descriptor matrix shape: {descriptors.shape}")

    visualization = cv2.drawKeypoints(image_rgb, keypoints, None, color=(0, 255, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not save visualization: {args.output}")
        print(f"Visualization saved to: {args.output}")


if __name__ == "__main__":
    main()
