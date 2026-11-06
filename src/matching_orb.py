"""Match saved ORB features (Hamming distance) and verify them geometrically.

ORB counterpart of match_sift in matching_w4.py: the same ratio test,
one-to-one filtering, RANSAC homography at a reference size of 1000 px and
result keys, so the scoring, calibration and rejection code is shared.
Features are loaded with load_features from matching_w4.py.
"""

import cv2
import numpy as np


def match_orb(query, museum, ratio=0.75, ransac_threshold=5.0, distance="Hamming"):
    """Find tentative matches with the Hamming distance, then verify them."""

    if not 0 < ratio < 1 or ransac_threshold <= 0:
        raise ValueError("Invalid matching thresholds")
    if distance != "Hamming":
        raise ValueError("ORB distance must be Hamming")

    matches = []

    # Need at least two museum descriptors for the ratio test.
    if len(query["descriptors"]) > 0 and len(museum["descriptors"]) >= 2:
        # Binary descriptors are stored as uint8 (32 bytes = 256 bits).
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        neighbors = matcher.knnMatch(
            np.ascontiguousarray(query["descriptors"], dtype=np.uint8),
            np.ascontiguousarray(museum["descriptors"], dtype=np.uint8),
            k=2,
        )

        # Keep a match only if the best neighbor is clearly better
        # than the second-best neighbor.
        candidates = [
            pair[0] for pair in neighbors
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
        ]

        # Avoid counting several query descriptors matched to the same
        # museum descriptor. Keep the closest such match.
        used = set()
        for match in sorted(candidates, key=lambda m: m.distance):
            if match.trainIdx not in used:
                matches.append(match)
                used.add(match.trainIdx)

    inliers = np.zeros(len(matches), dtype=bool)

    # Four points define a homography exactly; require extra evidence.
    if len(matches) >= 8:
        src = np.float32([query["points"][m.queryIdx] for m in matches])
        dst = np.float32([museum["points"][m.trainIdx] for m in matches])

        def reference_points(points, box):
            x0, y0, x1, y1 = box
            # Express coordinates at a reference longest side of 1000, so the
            # RANSAC tolerance does not depend on the image resolution.
            return points * (1000.0 / max(x1 - x0, y1 - y0))

        src = reference_points(src, query["box"])
        dst = reference_points(dst, museum["box"])

        # Make repeated runs reproducible.
        cv2.setRNGSeed(0)
        try:
            homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
        except cv2.error:
            homography, mask = None, None

        if homography is not None and mask is not None and np.isfinite(homography).all():
            inliers = mask.ravel().astype(bool)

    return {
        "matches": matches,
        "inliers": inliers,
        "tentative_count": len(matches),
        "inlier_count": int(inliers.sum()),
    }