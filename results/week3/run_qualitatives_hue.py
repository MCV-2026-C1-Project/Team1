"""Qualitative correction example for the HSV hue normalization + LAB method.

Output shows two panels:

    query con hue alterado   ->   query corregida por el metodo

Outputs are written to:

    results/week3/qualitatives_hue/
        00014.jpg
        summary.csv
        README.txt
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from color_reference import (  # noqa: E402
    DEFAULT_REGULARIZATION,
    DEFAULT_SATURATION_THRESHOLD,
    DEFAULT_SCALE_MAX,
    DEFAULT_SCALE_MIN,
    DEFAULT_SCALE_STEP,
    HueReference,
    build_hue_references,
    correct_hue_scale,
    estimate_hue_scale,
    hue_histogram,
    hue_scale_grid,
    rank_with_hue_references,
)
from methods_w2 import _load_ground_truth, _ordered_image_paths, _read_rgb  # noqa: E402
from methods_w3 import baseline_descriptor  # noqa: E402

MUSEUM = ROOT / "data" / "BBDD"
QUERIES = ROOT / "data" / "qsd1_w3"
GROUND_TRUTH = QUERIES / "gt_corresps.pkl"
AUGMENTATIONS = QUERIES / "augmentations.pkl"
REFERENCE = (ROOT / "results" / "week3" / "color_reference"
             / "museum_hue_references.npz")
OUTPUT = ROOT / "results" / "week3" / "qualitatives_hue"

CANVAS_WIDTH = 1400
CANVAS_HEIGHT = 860
IMAGE_BOX = 460
IMAGE_TOP = 140
GAP = 260


def _font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


FONT_TITLE = _font(27, bold=True)
FONT_SUBTITLE = _font(19)
FONT_LABEL = _font(18, bold=True)
FONT_ARROW = _font(21, bold=True)
FONT_FOOTER = _font(19)
FONT_SMALL = _font(16)


def _display(image):
    pil = Image.fromarray(np.asarray(image)).convert("RGB")
    pil.thumbnail((IMAGE_BOX, IMAGE_BOX), Image.Resampling.LANCZOS)
    return pil


def _draw_arrow(draw, x1, x2, y):
    color = (45, 45, 45)
    draw.line((x1, y, x2 - 22, y), fill=color, width=6)
    draw.polygon(
        [(x2, y), (x2 - 24, y - 14), (x2 - 24, y + 14)],
        fill=color,
    )


def _centered_text(draw, text, font, center_x, y, fill):
    width = draw.textlength(text, font=font)
    draw.text((center_x - width / 2, y), text, font=font, fill=fill)


def render_sample(path, sample_name, augmentation, reference_name, scale,
                  hist_before, hist_after, gt_id, baseline_rank, method_rank,
                  n_museum, altered, corrected):
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((45, 26),
              f"Correccion de tono HSV + retrieval LAB   |   Muestra: {sample_name}",
              font=FONT_TITLE, fill=(20, 20, 20))
    draw.text((45, 68),
              f"Aumentacion: {augmentation}   |   Estimada contra: {reference_name}",
              font=FONT_SUBTITLE, fill=(70, 70, 70))

    panels = [_display(altered), _display(corrected)]
    total_width = sum(panel.width for panel in panels) + GAP
    x = (CANVAS_WIDTH - total_width) // 2
    xs = []
    for panel in panels:
        xs.append(x)
        x += panel.width + GAP

    for x_pos, panel in zip(xs, panels):
        canvas.paste(panel, (x_pos, IMAGE_TOP))

    colors = [(80, 80, 80), (30, 90, 180)]
    for x_pos, panel, color in zip(xs, panels, colors):
        draw.rectangle(
            (x_pos, IMAGE_TOP, x_pos + panel.width - 1,
             IMAGE_TOP + panel.height - 1),
            outline=color, width=4)

    center_y = IMAGE_TOP + max(p.height for p in panels) // 2
    arrow_x1 = xs[0] + panels[0].width + 25
    arrow_x2 = xs[1] - 25
    _draw_arrow(draw, arrow_x1, arrow_x2, center_y)
    _centered_text(
        draw,
        f"H x {scale:.3f}   |   delta hist {hist_before:.2f} -> {hist_after:.2f}",
        FONT_ARROW, (arrow_x1 + arrow_x2) / 2, center_y - 42, (30, 30, 30))

    label_y = IMAGE_TOP + max(p.height for p in panels) + 12
    draw.text((xs[0], label_y), "Query (Hue alterado)",
              font=FONT_LABEL, fill=(50, 50, 50))
    draw.text((xs[1], label_y), "Query corregida",
              font=FONT_LABEL, fill=(30, 90, 180))

    rank_color = (0, 125, 60) if method_rank == 1 else (
        (180, 100, 0) if method_rank <= 5 else (170, 35, 35))
    draw.text(
        (45, CANVAS_HEIGHT - 88),
        f"GT: bbdd_{gt_id:05d}   |   Ranking L1: {baseline_rank} -> "
        f"{method_rank} de {n_museum}",
        font=FONT_FOOTER, fill=rank_color)
    draw.text(
        (45, CANVAS_HEIGHT - 58),
        "La escala de tono se estima contra la referencia indicada arriba; el "
        "descriptor final sigue siendo Lab blocks 1D + L1.",
        font=FONT_SMALL, fill=(90, 90, 90))
    canvas.save(path, quality=95, subsampling=0, optimize=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=14)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--rebuild-reference", action="store_true")
    options = parser.parse_args(argv)

    museum_ids, museum_paths = _ordered_image_paths(MUSEUM)
    query_ids, query_paths = _ordered_image_paths(QUERIES)
    ground_truth = _load_ground_truth(GROUND_TRUTH, query_ids)
    with AUGMENTATIONS.open("rb") as handle:
        augmentations = pickle.load(handle)

    sample = options.sample
    if not 0 <= sample < len(query_paths):
        parser.error(f"sample index out of range: {sample}")

    reference_path = options.reference.resolve()
    if options.rebuild_reference or not reference_path.exists():
        reference = build_hue_references(MUSEUM)
        reference.save(reference_path)
    else:
        reference = HueReference.load(reference_path)
    reference = reference.reorder(museum_ids)

    descriptor_fn = lambda image: baseline_descriptor(image)
    museum_descriptors = np.asarray([
        descriptor_fn(_read_rgb(path)) for path in museum_paths])

    query_rgb = _read_rgb(query_paths[sample])
    gt_idx = int(ground_truth[sample][0])
    reference_rgb = _read_rgb(museum_paths[gt_idx])
    reference_hist = reference.histograms[gt_idx]
    scales = hue_scale_grid(DEFAULT_SCALE_MIN, DEFAULT_SCALE_MAX,
                            DEFAULT_SCALE_STEP)
    query_hist = hue_histogram(
        query_rgb, saturation_threshold=DEFAULT_SATURATION_THRESHOLD)
    scale, _score = estimate_hue_scale(
        query_hist, reference_hist, scales=scales,
        regularization=DEFAULT_REGULARIZATION)
    corrected_rgb = correct_hue_scale(
        query_rgb, scale,
        saturation_threshold=DEFAULT_SATURATION_THRESHOLD)
    corrected_hist = hue_histogram(
        corrected_rgb, saturation_threshold=DEFAULT_SATURATION_THRESHOLD)
    hist_before = float(np.abs(query_hist - reference_hist).sum())
    hist_after = float(np.abs(corrected_hist - reference_hist).sum())

    result = rank_with_hue_references(
        query_rgb, museum_descriptors, reference, descriptor_fn)
    baseline_rank = result.baseline_ranking.index(gt_idx) + 1
    method_rank = result.ranking.index(gt_idx) + 1

    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_name = Path(query_paths[sample]).name
    render_sample(
        output / f"{query_ids[sample]:05d}.jpg",
        sample_name,
        str(augmentations[sample]),
        Path(museum_paths[gt_idx]).name,
        scale,
        hist_before,
        hist_after,
        museum_ids[gt_idx],
        baseline_rank,
        method_rank,
        len(museum_ids),
        query_rgb,
        corrected_rgb,
    )

    with (output / "summary.csv").open("w", newline="",
                                       encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample", "augmentation", "reference_file", "gt_id",
                        "hue_scale", "hist_distance_before",
                        "hist_distance_after", "baseline_rank", "method_rank"])
        writer.writeheader()
        writer.writerow({
            "sample": sample_name,
            "augmentation": str(augmentations[sample]),
            "reference_file": Path(museum_paths[gt_idx]).name,
            "gt_id": museum_ids[gt_idx],
            "hue_scale": f"{scale:.4f}",
            "hist_distance_before": f"{hist_before:.4f}",
            "hist_distance_after": f"{hist_after:.4f}",
            "baseline_rank": baseline_rank,
            "method_rank": method_rank,
        })

    (output / "README.txt").write_text(
        "Caso cualitativo: correccion de tono HSV\n"
        "========================================\n\n"
        f"Muestra: {sample_name}\n"
        "Referencia para estimar s: la imagen de BBDD que es la GT.\n\n"
        "Izquierda: query original con el tono alterado.\n"
        "Derecha: query tras aplicar la escala global de H.\n\n"
        "La flecha indica el factor estimado y la reduccion de la\n"
        "distancia de histogramas contra esa referencia.\n"
        "El retrieval final sigue usando 'Lab blocks 1D' + L1.\n",
        encoding="utf-8")

    print(f"Saved {output / f'{query_ids[sample]:05d}.jpg'}")


if __name__ == "__main__":
    main()
