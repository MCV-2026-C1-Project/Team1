"""Color-normalized retrieval that keeps the LAB descriptor and L1 distance.

For every museum candidate we estimate a global HSV hue scale that aligns the
query hue histogram with that candidate's reference histogram, correct the
query, convert it back to RGB and compute the usual "Lab blocks 1D"
descriptor.  The final ranking blends the raw LAB distance and the
color-normalized LAB distance.

Only the hue correction uses HSV.  The descriptor and the distance remain the
LAB / L1 ones used by the rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hue import (
    DEFAULT_REGULARIZATION,
    DEFAULT_SATURATION_THRESHOLD,
    DEFAULT_SCALE_MAX,
    DEFAULT_SCALE_MIN,
    DEFAULT_SCALE_STEP,
    correct_hue_scale,
    hue_histogram,
    hue_scale_grid,
    warp_hue_histogram,
)
from .reference import HueReference


@dataclass
class HueRetrievalResult:
    """Ranking plus the intermediate values, useful for diagnostics."""

    ranking: list
    baseline_ranking: list
    base_distances: np.ndarray
    corrected_distances: np.ndarray
    final_distances: np.ndarray
    scales: np.ndarray
    hist_distances: np.ndarray
    confidence: np.ndarray


def confidence_weights(hist_distances, temperature=1.0):
    """Down-weight candidates whose hue histogram does not align well.

    ``hist_distances`` are the regularized histogram distances for the best
    scale of every candidate.  The best candidate gets weight 1 and the weight
    decays with the distance relative to the median candidate.
    """
    distances = np.asarray(hist_distances, dtype=np.float64)
    assert distances.ndim == 1, (
        f"hist_distances must be 1-D, got {distances.shape}")
    assert temperature > 0, (
        f"temperature must be positive, got {temperature!r}")
    if distances.size == 0:
        return distances
    best = float(distances.min())
    spread = float(np.median(distances) - best)
    if spread <= 1e-12:
        return np.ones_like(distances)
    return np.exp(-(distances - best) / (temperature * spread))


def _descriptor_scale_grid(scale_min, scale_max, descriptor_step):
    values = np.arange(
        scale_min, scale_max + descriptor_step / 2.0, descriptor_step)
    values = np.round(values, 6)
    return np.unique(np.append(values, 1.0))


def rank_with_hue_references(
        query_rgb,
        museum_descriptors,
        reference,
        descriptor_fn,
        *,
        scale_min=DEFAULT_SCALE_MIN,
        scale_max=DEFAULT_SCALE_MAX,
        scale_step=DEFAULT_SCALE_STEP,
        descriptor_step=0.01,
        saturation_threshold=DEFAULT_SATURATION_THRESHOLD,
        regularization=DEFAULT_REGULARIZATION,
        alpha=1.0,
        temperature=1.0,
):
    """Rank museum descriptors with a per-candidate HSV hue correction.

    Parameters
    ----------
    query_rgb:
        Query image as an RGB uint8 array.
    museum_descriptors:
        One LAB descriptor per museum image, in the same order as
        ``reference.histograms``.
    reference:
        :class:`HueReference` with one hue histogram per museum image.
    descriptor_fn:
        Callable ``descriptor_fn(rgb) -> 1-D LAB descriptor``.  This is where
        the descriptor stays in LAB (for example ``baseline_descriptor``).
    alpha:
        Blend between the raw LAB distance (``0``) and the normalized LAB
        distance (``1``).  The chosen correction is weighted per candidate by
        the hue-alignment confidence.
    """
    if not isinstance(reference, HueReference):
        raise TypeError("reference must be a HueReference")
    descriptors = np.asarray(museum_descriptors, dtype=np.float64)
    if descriptors.ndim != 2:
        raise ValueError(
            "museum_descriptors must be a 2-D array, got "
            f"{descriptors.shape}")
    if descriptors.shape[0] != len(reference):
        raise ValueError(
            "museum descriptor/reference count mismatch: "
            f"{descriptors.shape[0]} != {len(reference)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")

    scales = hue_scale_grid(scale_min, scale_max, scale_step)
    descriptor_scales = _descriptor_scale_grid(
        scale_min, scale_max, descriptor_step)
    scale_to_descriptor = np.abs(
        descriptor_scales[:, None] - scales[None, :]
    ).argmin(axis=0)

    query_hist = hue_histogram(
        query_rgb, saturation_threshold=saturation_threshold)
    warped = np.stack([
        warp_hue_histogram(query_hist, float(scale), bins=reference.bins)
        for scale in scales
    ])
    # Penalize corrections away from the identity.
    hist_distances = np.abs(
        warped[:, None, :] - reference.histograms[None, :, :]
    ).sum(axis=2)
    hist_distances = hist_distances + (
        float(regularization) * (scales[:, None] - 1.0) ** 2
    )

    best_scale_index = np.argmin(hist_distances, axis=0)
    best_hist_distances = hist_distances[
        best_scale_index, np.arange(len(reference))]
    best_scales = scales[best_scale_index]
    best_descriptor_index = scale_to_descriptor[best_scale_index]

    base_descriptor = np.asarray(descriptor_fn(query_rgb), dtype=np.float64)
    if base_descriptor.ndim != 1:
        raise ValueError(
            "descriptor_fn must return a 1-D vector, got "
            f"{base_descriptor.shape}")
    if base_descriptor.shape[0] != descriptors.shape[1]:
        raise ValueError(
            "query/museum descriptor dimension mismatch: "
            f"{base_descriptor.shape[0]} != {descriptors.shape[1]}")

    corrected_descriptors = np.stack([
        np.asarray(
            descriptor_fn(correct_hue_scale(
                query_rgb, float(scale),
                saturation_threshold=saturation_threshold)),
            dtype=np.float64,
        )
        for scale in descriptor_scales
    ])
    base_distances = np.abs(
        descriptors - base_descriptor).sum(axis=1)
    corrected_distances = np.abs(
        corrected_descriptors[best_descriptor_index] - descriptors
    ).sum(axis=1)

    weights = confidence_weights(best_hist_distances, temperature=temperature)
    final_distances = base_distances + (
        float(alpha) * (corrected_distances - base_distances) * weights
    )
    ranking = [int(index) for index in np.argsort(
        final_distances, kind="stable")]
    baseline_ranking = [int(index) for index in np.argsort(
        base_distances, kind="stable")]
    return HueRetrievalResult(
        ranking=ranking,
        baseline_ranking=baseline_ranking,
        base_distances=base_distances,
        corrected_distances=corrected_distances,
        final_distances=final_distances,
        scales=best_scales,
        hist_distances=best_hist_distances,
        confidence=weights,
    )
