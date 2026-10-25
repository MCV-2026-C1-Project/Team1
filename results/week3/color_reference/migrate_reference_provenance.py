"""One-off migration: record hue-reference provenance in the canonical cache.

The cache at results/week3/color_reference/museum_hue_references.npz was written
before the ``source`` field existed. This rebuilds it from the UNFILTERED museum
images, checks that every histogram is unchanged, and only then saves the file
with ``source="raw"``.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from color_reference import (  # noqa: E402
    HueReference,
    build_hue_references,
)

CACHE = (ROOT / "results" / "week3" / "color_reference"
         / "museum_hue_references.npz")
MUSEUM = ROOT / "data" / "BBDD"


def main():
    old = HueReference.load(CACHE)
    print(f"old cache provenance: {old.source}")
    new = build_hue_references(str(MUSEUM))
    if old.ids != new.ids:
        raise SystemExit("ID order differs from the cache: refusing to migrate")
    diff = float(np.abs(old.histograms - new.histograms).max())
    print(f"max histogram difference vs rebuilt raw: {diff:.12f}")
    if diff > 1e-12:
        raise SystemExit("Histograms differ: refusing to overwrite the cache")
    path = new.save(CACHE)
    print(f"migrated {path} (source={new.source}, {len(new)} references)")
    print(f"reloaded provenance: {HueReference.load(path).source}")


if __name__ == "__main__":
    main()
