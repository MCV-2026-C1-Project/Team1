"""Build or inspect the cached museum hue references.

Examples:
    python -m color_reference build --museum data/BBDD \
        --output results/week3/color_reference/museum_hue_references.npz
    python -m color_reference inspect \
        --reference results/week3/color_reference/museum_hue_references.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .reference import HueReference, build_hue_references


def _build(args):
    reference = build_hue_references(
        args.museum,
        saturation_threshold=args.saturation_threshold,
    )
    path = reference.save(args.output)
    print(f"Wrote {len(reference)} hue references to {path}")
    return 0


def _inspect(args):
    reference = HueReference.load(args.reference)
    print(f"references: {len(reference)}")
    print(f"bins: {reference.bins}")
    print(f"saturation threshold: {reference.saturation_threshold}")
    print(f"ids: {reference.ids[:5]} ... {reference.ids[-5:]}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the reference cache")
    build.add_argument("--museum", type=Path, default=Path("data/BBDD"))
    build.add_argument(
        "--output", type=Path,
        default=Path("results/week3/color_reference/"
                     "museum_hue_references.npz"))
    build.add_argument("--saturation-threshold", type=int, default=40)
    build.set_defaults(func=_build)

    inspect = subparsers.add_parser("inspect", help="Inspect a cache")
    inspect.add_argument(
        "--reference", type=Path,
        default=Path("results/week3/color_reference/"
                     "museum_hue_references.npz"))
    inspect.set_defaults(func=_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
