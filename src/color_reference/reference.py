"""Reference hue histograms for the museum images.

The references live in their own package so they can be built, cached and
inspected without touching the W3 noise-filter code.  The normal cache path is
``results/week3/color_reference/museum_hue_references.npz``.

A reference must always describe the museum images as they really are, so the
canonical source is the UNFILTERED museum (``source="raw"``). Filters change
the per-channel color, so a reference built from filtered images would make the
estimated hue scale depend on the filter and kernel being tested, and two runs
with the same command would disagree depending on which cache file they
happened to load. That is recorded in the cache and checked on load; caches
written before the field existed are reported as ``legacy`` and refused, so
they must be rebuilt (see
``results/week3/color_reference/migrate_reference_provenance.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from methods_w2 import _ordered_image_paths, _read_rgb

from .hue import (
    DEFAULT_SATURATION_THRESHOLD,
    HUE_BINS,
    hue_histogram,
)

REFERENCE_SOURCES = ("raw", "filtered")
# Caches written before provenance existed: their origin cannot be trusted.
LEGACY_REFERENCE_SOURCE = "legacy"
DEFAULT_REFERENCE_SOURCE = "raw"


@dataclass
class HueReference:
    """One saturation-weighted hue histogram per museum image."""

    ids: tuple[str, ...]
    histograms: np.ndarray
    bins: int = HUE_BINS
    saturation_threshold: int = DEFAULT_SATURATION_THRESHOLD
    source: str = DEFAULT_REFERENCE_SOURCE

    def __post_init__(self):
        self.ids = tuple(str(image_id) for image_id in self.ids)
        self.histograms = np.asarray(self.histograms, dtype=np.float64)
        if self.source not in REFERENCE_SOURCES + (LEGACY_REFERENCE_SOURCE,):
            raise ValueError(
                f"source must be one of {REFERENCE_SOURCES}, "
                f"got {self.source!r}")
        if self.histograms.ndim != 2:
            raise ValueError(
                "histograms must be a 2-D array, got "
                f"{self.histograms.shape}")
        if self.histograms.shape[0] != len(self.ids):
            raise ValueError(
                "one histogram is required per id: "
                f"{self.histograms.shape[0]} != {len(self.ids)}")
        if self.histograms.shape[1] != self.bins:
            raise ValueError(
                f"expected {self.bins} hue bins, got "
                f"{self.histograms.shape[1]}")
        if not np.all(np.isfinite(self.histograms)):
            raise ValueError("histograms contain non-finite values")

    def __len__(self):
        return len(self.ids)

    def global_histogram(self):
        """Average reference histogram (useful as a single global baseline)."""
        hist = self.histograms.mean(axis=0)
        total = hist.sum()
        return hist / total if total > 0 else hist

    def reorder(self, ids):
        """Return a copy with histograms in the order of ``ids``."""
        ids = tuple(str(image_id) for image_id in ids)
        index = {image_id: position
                 for position, image_id in enumerate(self.ids)}
        missing = [image_id for image_id in ids if image_id not in index]
        if missing:
            raise ValueError(f"reference is missing image IDs: {missing}")
        order = [index[image_id] for image_id in ids]
        return HueReference(
            ids=ids,
            histograms=self.histograms[order],
            bins=self.bins,
            saturation_threshold=self.saturation_threshold,
            source=self.source,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ids=np.asarray(self.ids),
            histograms=self.histograms,
            bins=np.asarray(self.bins),
            saturation_threshold=np.asarray(self.saturation_threshold),
            source=np.asarray(self.source),
        )
        return path

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            # A cache without the provenance field was written before it
            # existed: its origin is unknown, so it is reported as "legacy"
            # and the retrieval code refuses to use it.
            source = (str(data["source"]) if "source" in data.files
                      else LEGACY_REFERENCE_SOURCE)
            return cls(
                ids=tuple(str(image_id) for image_id in data["ids"]),
                histograms=data["histograms"],
                bins=int(data["bins"]),
                saturation_threshold=int(data["saturation_threshold"]),
                source=source,
            )


def build_hue_references_from_images(ids, images,
                                     bins=HUE_BINS,
                                     saturation_threshold=DEFAULT_SATURATION_THRESHOLD,
                                     source=DEFAULT_REFERENCE_SOURCE):
    """Build references from already-loaded RGB uint8 images."""
    ids = tuple(str(image_id) for image_id in ids)
    if len(ids) != len(images):
        raise ValueError(
            f"ids/images length mismatch: {len(ids)} != {len(images)}")
    histograms = np.stack([
        hue_histogram(
            image, bins=bins,
            saturation_threshold=saturation_threshold)
        for image in images
    ])
    return HueReference(
        ids=ids,
        histograms=histograms,
        bins=bins,
        saturation_threshold=saturation_threshold,
        source=source,
    )


def build_hue_references(museum_dir, bins=HUE_BINS,
                         saturation_threshold=DEFAULT_SATURATION_THRESHOLD,
                         source=DEFAULT_REFERENCE_SOURCE):
    """Build one hue reference per image in a museum folder."""
    ids, paths = _ordered_image_paths(museum_dir)
    images = [_read_rgb(path) for path in paths]
    return build_hue_references_from_images(
        ids, images, bins=bins,
        saturation_threshold=saturation_threshold, source=source)
