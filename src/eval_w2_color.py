"""Global HSV comparison, L1 distance only (QSD1 = QSD1-W2).

Sections:
  A. Baselines: gray, marginal RGB and marginal HSV with 32, 64 and Max bins
     per channel (Max = 1 bin per value: gray/RGB 256; HSV H 180, S/V 256),
     plus the historic Week 1 HSV 30+32+32 reference.
  B. Global HSV 3D (bins per axis). 3D-Max (180x256x256 = 11.8M dims) is
     skipped as infeasible when dense.
  C. Single HSV 2D: HS, HV, SV.
  D. 2D concatenations: HS+HV, HS+SV, HV+SV and HS+HV+SV. Each part is
     L1-normalized and the concat is L1-renormalized (equal weight per pair).
  E. HSV pyramids (registry combos, levels (1, 2, 4) = 21 cells, L1).

No other color space: HSV only, same combinations as METHODS_W2.

All with L1, top-5, mAP@1/mAP@5.

Usage from the project root:
    python src/eval_w2_color.py
    python src/eval_w2_color.py --queries data/qsd1_w2 --save results/week2/color_sweep_l1.txt
"""

import argparse
import time
from pathlib import Path

import cv2

from k_similarity import (
    evaluate_retrieval,
    retrieve_all_queries,
    validate_ground_truth,
)
from methods_w2 import (
    METHODS_W2,
    compute_hsv_2d_concat_histogram,
    compute_hsv_2d_histogram,
    compute_hsv_3d_histogram,
)
from similarity_functions import l1_distance
from utils import (
    Dataset,
    compute_color_hsv_histogram,
    compute_color_rgb_histogram,
    compute_gray_histogram,
)

HS, HV, SV = (0, 1), (0, 2), (1, 2)
MAX_2D = {HS: (180, 256), HV: (180, 256), SV: (256, 256)}


def _3d(bins):
    return lambda hsv: compute_hsv_3d_histogram(hsv, bins=bins)


def _2d(pair, bins):
    return lambda hsv: compute_hsv_2d_histogram(hsv, channels=pair, bins=bins)


def _cat(pairs_bins):
    parts = [(pair, bins) for pair, bins in pairs_bins]
    return lambda hsv: compute_hsv_2d_concat_histogram(hsv, parts=parts)


def _dims_label(pairs_bins):
    """'HS 8x8' or 'HS 8x8 + HV 8x8' plus total dimension in parentheses."""
    total = sum(b[0] * b[1] for _, b in pairs_bins)
    return " + ".join(f"{t} {b[0]}x{b[1]}" for t, (_, b) in
                      zip(_tags(pairs_bins), pairs_bins)) + f" ({total})"


def _tags(pairs_bins):
    names = {HS: "HS", HV: "HV", SV: "SV"}
    return [names[p] for p, _ in pairs_bins]


def build_sections():
    """[(title, [(name, fn, source)])]; source: hsv/gray/rgb."""
    # A. Baselines (gray: 256 bins hardcoded in the W1 code).
    base = [
        ("Gray 256", lambda g: compute_gray_histogram(g), "gray"),
        ("RGB 32+32+32 (96)",
         lambda im: compute_color_rgb_histogram(im, 32, 32, 32), "rgb"),
        ("RGB 64+64+64 (192)",
         lambda im: compute_color_rgb_histogram(im, 64, 64, 64), "rgb"),
        ("RGB Max 256+256+256 (768)",
         lambda im: compute_color_rgb_histogram(im, 256, 256, 256), "rgb"),
        ("HSV 30+32+32 / W1 ref (94)",
         lambda hsv: compute_color_hsv_histogram(hsv), "hsv"),
        ("HSV 32+32+32 (96)",
         lambda hsv: compute_color_hsv_histogram(hsv, 32, 32, 32), "hsv"),
        ("HSV 64+64+64 (192)",
         lambda hsv: compute_color_hsv_histogram(hsv, 64, 64, 64), "hsv"),
        ("HSV Max 180+256+256 (692)",
         lambda hsv: compute_color_hsv_histogram(hsv, 180, 256, 256), "hsv"),
    ]
    # B. Global 3D.
    three = [((f"3D {b[0]}x{b[1]}x{b[2]} ({b[0]*b[1]*b[2]})", _3d(b), "hsv"))
             for b in [(4, 4, 4), (8, 8, 8), (16, 8, 8),
                       (16, 16, 8), (32, 32, 32), (64, 64, 64)]]
    # C. Single 2D (same bins on every channel).
    solo = []
    for tag, pair in (("HS", HS), ("HV", HV), ("SV", SV)):
        for b in [(8, 8), (16, 16), (30, 32), (32, 32), (64, 64),
                  MAX_2D[pair]]:
            maxmark = " Max" if b == MAX_2D[pair] else ""
            solo.append((f"{tag}{maxmark} {b[0]}x{b[1]} ({b[0]*b[1]})",
                         _2d(pair, b), "hsv"))
    # D. 2D concatenations.
    concat = []
    for tags, pairs in (("HS+HV", (HS, HV)), ("HS+SV", (HS, SV)),
                        ("HV+SV", (HV, SV)), ("HS+HV+SV", (HS, HV, SV))):
        for b in [(8, 8), (16, 16), (32, 32), (64, 64)]:
            parts = [(p, b) for p in pairs]
            concat.append((f"{tags} {b[0]}x{b[1]}/part " +
                           f"({_dims_label(parts).split('(')[-1]}",
                           _cat(parts), "hsv"))
        parts = [(p, MAX_2D[p]) for p in pairs]
        concat.append((f"{tags} Max " +
                       "+".join(f"{b[0]}x{b[1]}" for _, b in parts) +
                       f" ({sum(b[0]*b[1] for _, b in parts)})",
                       _cat(parts), "hsv"))
    # E. Pyramids: same 4 HSV combos as the registry (21 cells, L1).
    pyramids = [
        ("Pyramid 3D 4x4x4 (1344)",
         METHODS_W2["HSV pyramid 3D 4x4x4"], "hsv"),
        ("Pyramid HS 30x32 (20160)",
         METHODS_W2["HSV pyramid HS 30x32"], "hsv"),
        ("Pyramid HV 8x8 (1344)",
         METHODS_W2["HSV pyramid HV 8x8"], "hsv"),
        ("Pyramid HS+HV 16x16 (10752)",
         METHODS_W2["HSV pyramid HS+HV 16x16"], "hsv"),
    ]
    return [
        ("A. Baselines (grayscale / RGB / 1D HSV)", base),
        ("B. Global HSV 3D", three),
        ("C. Single HSV 2D (HS, HV, SV)", solo),
        ("D. 2D concatenations (equal weight per pair)", concat),
        ("E. HSV pyramids (registry combos, 1+2+4, L1)", pyramids),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--museum", default="data/BBDD")
    parser.add_argument("--queries", default="data/qsd1_w1")
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    museum = Dataset(args.museum)
    queries = Dataset(args.queries)
    print(f"Museum: {len(museum.images)} | Queries: {len(queries.images)}")
    validate_ground_truth(queries.correspondances, len(museum.images),
                           len(queries.images))

    t0 = time.time()
    museum_hsv = [cv2.cvtColor(im, cv2.COLOR_RGB2HSV) for im in museum.images]
    queries_hsv = [cv2.cvtColor(im, cv2.COLOR_RGB2HSV) for im in queries.images]
    museum_gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in museum.images]
    queries_gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in queries.images]
    print(f"HSV/gray conversion: {time.time()-t0:.1f}s")

    sources = {"hsv": (museum_hsv, queries_hsv),
               "gray": (museum_gray, queries_gray),
               "rgb": (museum.images, queries.images)}
    width = 44
    lines = []
    for title, exps in build_sections():
        lines.append("")
        lines.append(title)
        lines.append(f"{'Method':{width}s} {'dims':>7s} "
                     f"{'mAP@1':>7s} {'mAP@5':>7s}")
        print("\n" + "\n".join(lines[-2:]))
        for name, fn, src in exps:
            m_src, q_src = sources[src]
            m_desc = [fn(im) for im in m_src]
            q_desc = [fn(im) for im in q_src]
            dims = m_desc[0].shape[0]
            rankings = retrieve_all_queries(q_desc, m_desc, l1_distance,
                                            top_k=5)
            ranked = [[i for i, _ in r] for r in rankings]
            res = evaluate_retrieval(ranked, queries.correspondances, 5)
            line = (f"{name:{width}s} {dims:7d} "
                    f"{res[1]['mAP']:7.4f} {res[5]['mAP']:7.4f}")
            print(line)
            lines.append(line)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
