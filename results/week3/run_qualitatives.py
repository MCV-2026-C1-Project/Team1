"""Generate qualitative retrieval results for every W3 Task 1 filter.

For each filter configuration this script writes

    results/week3/qualitatives/<filter>[/<parameter>]/
        overview.jpg
        summary.txt

The overview has one row per representative query: original query, filtered
query and the top-5 museum images. Relevant images (ground truth) are framed
in green, the rest in red. The summary lists the metrics and the ranking of
every query.

The default is the Task 1 protocol (filter applied to queries only). Pass
``--scope both`` to filter the museum references too.
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from methods_w2 import _load_ground_truth, _ordered_image_paths, _read_rgb  # noqa: E402
from methods_w3 import (  # noqa: E402
    baseline_descriptor,
    bilateral_filter,
    gaussian_filter,
    median_filter,
    morphology_filter,
)
from k_similarity import (  # noqa: E402
    DISTANCE_FUNCTIONS,
    average_precision_at_k,
    mean_average_precision_at_k,
    retrieve_all_queries,
)

MUSEUM = ROOT / "data" / "BBDD"
QUERIES = ROOT / "data" / "qsd1_w3"
GROUND_TRUTH = QUERIES / "gt_corresps.pkl"
OUTPUT = ROOT / "results" / "week3" / "qualitatives"

# Fixed across filters so the rows can be compared side by side.
# They cover the three augmentation types and both easy and hard cases.
SELECTED_QUERIES = (0, 1, 2, 8, 16, 20, 22, 23)

TILE = 220
COL_GAP = 10
MARGIN = 16
ROW_LABEL_HEIGHT = 30
TILE_LABEL_HEIGHT = 42
ROW_GAP = 14
HEADER_HEIGHT = 82
FOOTER_HEIGHT = 34


def _wavelet_lowpass(img_rgb, wavelet, level):
    """DWT low-pass: keep LL, zero every detail coefficient."""
    image = np.asarray(img_rgb)
    height, width = image.shape[:2]
    channels = []
    for channel in range(image.shape[2]):
        data = image[..., channel].astype(np.float64)
        coeffs = pywt.wavedec2(data, wavelet=wavelet, level=level)
        filtered = [coeffs[0]] + [
            (np.zeros_like(h), np.zeros_like(v), np.zeros_like(d))
            for h, v, d in coeffs[1:]
        ]
        reconstructed = pywt.waverec2(filtered, wavelet=wavelet)
        reconstructed = reconstructed[:height, :width]
        channels.append(
            np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8))
    return np.stack(channels, axis=-1)


def _open_close(image, ksize):
    return morphology_filter(
        morphology_filter(image, "open", ksize=ksize), "close", ksize=ksize)


def _close_open(image, ksize):
    return morphology_filter(
        morphology_filter(image, "close", ksize=ksize), "open", ksize=ksize)


def filter_specs():
    specs = [("none", "-", "none", lambda image: image)]
    for ksize in (3, 5, 7, 9, 11, 13, 15):
        specs.append((
            "median", f"{ksize}x{ksize}", "median",
            lambda image, k=ksize: median_filter(image, ksize=k)))
    for ksize in (3, 5, 7, 9, 11, 13, 15):
        specs.append((
            "gaussian", f"{ksize}x{ksize}", "gaussian",
            lambda image, k=ksize: gaussian_filter(image, ksize=k)))
    for ksize in (5, 7, 9):
        specs.append((
            "bilateral", f"{ksize}x{ksize}", "bilateral",
            lambda image, k=ksize: bilateral_filter(image, ksize=k)))
    for operation in ("erode", "dilate", "open", "close"):
        for ksize in (3, 5, 7, 9, 11, 13, 15):
            specs.append((
                operation, f"{ksize}x{ksize}", operation,
                lambda image, op=operation, k=ksize: morphology_filter(
                    image, op, ksize=k)))
    for ksize in (3, 5, 7, 9, 11, 13, 15):
        specs.append((
            "open+close", f"{ksize}x{ksize}", "open_close",
            lambda image, k=ksize: _open_close(image, k)))
    for ksize in (3, 5, 7, 9, 11, 13, 15):
        specs.append((
            "close+open", f"{ksize}x{ksize}", "close_open",
            lambda image, k=ksize: _close_open(image, k)))
    for wavelet in ("haar", "db2", "db4", "sym4"):
        for level in (1, 2, 3):
            specs.append((
                "wavelet_lowpass", f"{wavelet}_lvl{level}", "wavelet_lowpass",
                lambda image, w=wavelet, lvl=level: _wavelet_lowpass(
                    image, w, lvl)))
    return specs


def _font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


FONT_HEADER = _font(24, bold=True)
FONT_SUBHEADER = _font(18)
FONT_ROW = _font(17, bold=True)
FONT_TILE = _font(14)
FONT_FOOTER = _font(14)


def _thumbnail(image, size=TILE):
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = Image.fromarray(np.asarray(image)).convert("RGB")
    return ImageOps_contain(pil, (size, size))


def ImageOps_contain(image, size):
    """Small local copy of ImageOps.contain to avoid another import alias."""
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (246, 246, 246))
    offset = ((size[0] - image.width) // 2,
              (size[1] - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def _fit_text(draw, text, font, max_width):
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "...", font=font) > max_width:
        text = text[:-1]
    return text + "..."


def _draw_tile(canvas, draw, image, x, y, label, border):
    thumb = _thumbnail(image)
    canvas.paste(thumb, (x, y))
    draw.rectangle((x, y, x + TILE - 1, y + TILE - 1),
                   outline=border, width=4)
    draw.text((x + 4, y + TILE + 7),
              _fit_text(draw, label, FONT_TILE, TILE - 8),
              fill=(25, 25, 25), font=FONT_TILE)


def render_overview(path, family, parameter, scope, map1, map5,
                    query_data, museum_thumbs):
    rows = len(query_data)
    row_height = ROW_LABEL_HEIGHT + TILE + TILE_LABEL_HEIGHT
    width = MARGIN * 2 + 7 * TILE + 6 * COL_GAP
    height = (HEADER_HEIGHT + rows * row_height
              + (rows - 1) * ROW_GAP + FOOTER_HEIGHT)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title = f"{family}  {parameter}" if parameter != "-" else family
    draw.text((MARGIN, 16), f"Resultados cualitativos: {title}",
              fill=(20, 20, 20), font=FONT_HEADER)
    draw.text(
        (MARGIN, 48),
        f"scope={scope}   |   mAP@1={map1:.4f}   |   mAP@5={map5:.4f}",
        fill=(60, 60, 60), font=FONT_SUBHEADER)

    for row_index, row in enumerate(query_data):
        base_y = HEADER_HEIGHT + row_index * (row_height + ROW_GAP)
        query_id = row["query_id"]
        verdict = "acierto" if row["top1_ok"] else "fallo"
        color = (0, 140, 70) if row["top1_ok"] else (185, 35, 35)
        row_text = (f"q{query_id:05d}  |  {row['augmentation']}  |  "
                    f"top-1: {verdict}  |  AP@5={row['ap5']:.2f}")
        draw.text((MARGIN, base_y + 2), row_text, fill=color, font=FONT_ROW)

        tile_y = base_y + ROW_LABEL_HEIGHT
        columns = [
            (row["original"], "original", (90, 90, 90)),
            (row["filtered"], "filtrada", (90, 90, 90)),
        ]
        for rank, museum_id in enumerate(row["top5"], start=1):
            relevant = museum_id in row["gt"]
            border = (0, 150, 70) if relevant else (190, 40, 40)
            marker = "  [GT]" if relevant else ""
            columns.append((
                museum_thumbs[museum_id],
                f"#{rank} bbdd_{museum_id:05d}{marker}",
                border,
            ))
        for column_index, (image, label, border) in enumerate(columns):
            x = MARGIN + column_index * (TILE + COL_GAP)
            _draw_tile(canvas, draw, image, x, tile_y, label, border)

    footer = ("Verde: imagen relevante (GT). Rojo: no relevante. "
              "Las dos primeras columnas son la consulta original y filtrada.")
    draw.text((MARGIN, height - FOOTER_HEIGHT + 4), footer,
              fill=(70, 70, 70), font=FONT_FOOTER)
    canvas.save(path, quality=92, subsampling=0, optimize=True)


def write_summary(path, family, parameter, scope, map1, map5, query_data,
                  all_rows):
    lines = [
        f"Filter: {family}",
        f"Parameter: {parameter}",
        f"Scope: {scope}",
        f"mAP@1: {map1:.4f}",
        f"mAP@5: {map5:.4f}",
        "",
        "Representative queries:",
        "query\tnoise\ttop1\ttop1_ok\tAP@5",
    ]
    for row in query_data:
        lines.append(
            f"{row['query_id']:05d}\t{row['augmentation']}\t"
            f"{row['top5'][0]:05d}\t"
            f"{'OK' if row['top1_ok'] else 'FAIL'}\t{row['ap5']:.4f}")
    lines += [
        "",
        "All queries:",
        "query\tnoise\tgt\ttop1\ttop5\ttop1_ok\tAP@5",
    ]
    for row in all_rows:
        lines.append(
            f"{row['query_id']:05d}\t{row['augmentation']}\t"
            f"{row['gt'][0]:05d}\t{row['top5'][0]:05d}\t"
            f"{row['top5']}\t{'OK' if row['top1_ok'] else 'FAIL'}\t"
            f"{row['ap5']:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sanitize_parameter(parameter):
    return parameter.replace("/", "_").replace("\\", "_")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("query", "both"), default="query",
                        help="Filter queries only (Task 1) or both sides")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    options = parser.parse_args(argv)

    if not pywt.__version__:
        raise RuntimeError("PyWavelets is required for the wavelet filters")

    museum_ids, museum_paths = _ordered_image_paths(MUSEUM)
    query_ids, query_paths = _ordered_image_paths(QUERIES)
    ground_truth = _load_ground_truth(GROUND_TRUTH, query_ids)
    augmentations = [path.stem for path in query_paths]  # placeholder

    with (QUERIES / "augmentations.pkl").open("rb") as handle:
        raw_augmentations = pickle.load(handle)
    augmentations = [
        str(label) if label is not None else "None"
        for label in raw_augmentations
    ]

    raw_queries = [_read_rgb(path) for path in query_paths]
    raw_museum = None
    museum_descriptors = []
    museum_thumbs = {}
    if options.scope == "both":
        raw_museum = [_read_rgb(path) for path in museum_paths]
        for museum_id, image in zip(museum_ids, raw_museum):
            museum_descriptors.append(baseline_descriptor(image))
            museum_thumbs[museum_id] = _thumbnail(image)
    else:
        for museum_id, path in zip(museum_ids, museum_paths):
            image = _read_rgb(path)
            museum_descriptors.append(baseline_descriptor(image))
            museum_thumbs[museum_id] = _thumbnail(image)
    print(f"Loaded {len(museum_descriptors)} museum descriptors and "
          f"{len(raw_queries)} queries.")

    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for family, parameter, folder_name, image_filter in filter_specs():
        print(f"  {family:<16} {parameter:>10}", flush=True)
        if options.scope == "both":
            filtered_museum = [
                baseline_descriptor(image_filter(image))
                for image in raw_museum
            ]
        else:
            filtered_museum = museum_descriptors

        query_descriptors = []
        selected_filtered = {}
        for query_index, image in enumerate(raw_queries):
            filtered = image_filter(image)
            query_descriptors.append(baseline_descriptor(filtered))
            if query_index in SELECTED_QUERIES:
                selected_filtered[query_index] = Image.fromarray(filtered)

        rankings = retrieve_all_queries(
            query_descriptors, filtered_museum,
            DISTANCE_FUNCTIONS["L1"], top_k=5)
        ranked_ids = [
            [museum_ids[index] for index, _ in ranking]
            for ranking in rankings
        ]
        map1 = mean_average_precision_at_k(ranked_ids, ground_truth, 1)
        map5 = mean_average_precision_at_k(ranked_ids, ground_truth, 5)

        all_rows = []
        for query_index, (ranking, relevant) in enumerate(
                zip(ranked_ids, ground_truth)):
            relevant_set = set(relevant)
            all_rows.append({
                "query_id": query_ids[query_index],
                "augmentation": augmentations[query_index],
                "gt": relevant,
                "top5": ranking,
                "top1_ok": ranking[0] in relevant_set,
                "ap5": average_precision_at_k(ranking, relevant_set, 5),
            })

        selected_rows = []
        for query_index in SELECTED_QUERIES:
            row = dict(all_rows[query_index])
            row["original"] = Image.fromarray(raw_queries[query_index])
            row["filtered"] = selected_filtered[query_index]
            selected_rows.append(row)

        parameter_folder = output / folder_name
        if parameter != "-":
            parameter_folder = parameter_folder / _sanitize_parameter(parameter)
        parameter_folder.mkdir(parents=True, exist_ok=True)
        render_overview(
            parameter_folder / "overview.jpg", family, parameter,
            options.scope, map1, map5, selected_rows, museum_thumbs)
        write_summary(
            parameter_folder / "summary.txt", family, parameter,
            options.scope, map1, map5, selected_rows, all_rows)
        index_rows.append({
            "filter": family,
            "parameter": parameter,
            "scope": options.scope,
            "mAP@1": f"{map1:.4f}",
            "mAP@5": f"{map5:.4f}",
            "folder": parameter_folder.relative_to(output).as_posix(),
        })

    with (output / "index.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filter", "parameter", "scope", "mAP@1", "mAP@5",
                        "folder"])
        writer.writeheader()
        writer.writerows(index_rows)

    readme = [
        "Resultados cualitativos de los filtros de Week 3, Task 1",
        "=" * 55,
        "",
        f"Scope: {options.scope} (query = solo consultas).",
        "",
        "Cada carpeta contiene:",
        "  overview.jpg  consulta original, consulta filtrada y top-5.",
        "                Verde = imagen relevante (GT); rojo = no relevante.",
        "  summary.txt   metricas y ranking de las 30 consultas.",
        "",
        "La estructura es qualitatives/<filtro>/<parametro>/.",
        "Para los filtros sin parametro (none) los archivos estan en",
        "qualitatives/none/. Los nombres open+close y close+open se",
        "guardan como open_close y close_open.",
        "",
        "Consultas representativas (las mismas en todos los filtros):",
        "  00000  None-MultiplyHue      facil",
        "  00001  None                  la mediana corrige el top-1",
        "  00002  None-MultiplyHue      dificil",
        "  00008  None                  facil",
        "  00016  UnnamedImpulseNoise   dificil",
        "  00020  UnnamedImpulseNoise   la mediana corrige el top-1",
        "  00022  None                  dificil",
        "  00023  UnnamedImpulseNoise   la mediana corrige el top-1",
        "",
        "El indice con mAP@1/mAP@5 y la ruta de cada configuracion esta en",
        "index.csv.",
        "",
        "Regenerar:",
        "  python results/week3/run_qualitatives.py",
        "  python results/week3/run_qualitatives.py --scope both",
    ]
    (output / "README.txt").write_text(
        "\n".join(readme) + "\n", encoding="utf-8")
    print(f"Done. Qualitative results in {output}")


if __name__ == "__main__":
    main()
