"""Qualitative filter comparison with MSE and L1 ground-truth rank.

New output, kept separate from ``results/week3/qualitatives``:

    results/week3/qualitatives_mse/<filter>/
        00018.jpg
        00020.jpg

Each image shows the noisy query on the left, an arrow with the pixel MSE,
and the filtered query on the right. The bottom caption gives the position of
the ground-truth museum image in the full L1 ranking.

All kernel-based filters use the same 9x9 kernel. The wavelet low-pass uses
db2 at level 3 (fixed configuration, no kernel). Use ``--filters`` and
``--samples`` to generate a subset in a separate ``--output`` folder.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from methods_w2 import _load_ground_truth, _ordered_image_paths, _read_rgb  # noqa: E402
from methods_w3 import (  # noqa: E402
    average_filter,
    baseline_descriptor,
    bilateral_filter,
    gaussian_filter,
    median_filter,
    morphology_filter,
)
from k_similarity import (  # noqa: E402
    DISTANCE_FUNCTIONS,
    rank_museum_images,
)

MUSEUM = ROOT / "data" / "BBDD"
QUERIES = ROOT / "data" / "qsd1_w3"
GROUND_TRUTH = QUERIES / "gt_corresps.pkl"
AUGMENTATIONS = QUERIES / "augmentations.pkl"
OUTPUT = ROOT / "results" / "week3" / "qualitatives_mse"

DEFAULT_SAMPLES = (18, 20)
KERNEL = 9
WAVELET = "db2"
WAVELET_LEVEL = 3

CANVAS_WIDTH = 1400
CANVAS_HEIGHT = 820
IMAGE_BOX = 520
IMAGE_TOP = 120
ARROW_GAP = 220


def _wavelet_lowpass(img_rgb, wavelet=WAVELET, level=WAVELET_LEVEL):
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


def _open_close(image):
    return morphology_filter(
        morphology_filter(image, "open", ksize=KERNEL), "close", ksize=KERNEL)


def _close_open(image):
    return morphology_filter(
        morphology_filter(image, "close", ksize=KERNEL), "open", ksize=KERNEL)


def filter_specs():
    return [
        ("none", "sin filtro", lambda image: image),
        ("average", f"{KERNEL}x{KERNEL}", lambda image: average_filter(image, KERNEL)),
        ("median", f"{KERNEL}x{KERNEL}", lambda image: median_filter(image, KERNEL)),
        ("gaussian", f"{KERNEL}x{KERNEL}", lambda image: gaussian_filter(image, KERNEL)),
        ("bilateral", f"{KERNEL}x{KERNEL}", lambda image: bilateral_filter(image, KERNEL)),
        ("erode", f"{KERNEL}x{KERNEL}", lambda image: morphology_filter(image, "erode", KERNEL)),
        ("dilate", f"{KERNEL}x{KERNEL}", lambda image: morphology_filter(image, "dilate", KERNEL)),
        ("open", f"{KERNEL}x{KERNEL}", lambda image: morphology_filter(image, "open", KERNEL)),
        ("close", f"{KERNEL}x{KERNEL}", lambda image: morphology_filter(image, "close", KERNEL)),
        ("open+close", f"{KERNEL}x{KERNEL}", _open_close),
        ("close+open", f"{KERNEL}x{KERNEL}", _close_open),
        ("wavelet_lowpass", f"{WAVELET}_lvl{WAVELET_LEVEL}", _wavelet_lowpass),
    ]


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
FONT_MSE = _font(24, bold=True)
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


def render_sample(path, filter_name, parameter, sample_name,
                  augmentation, original, filtered, mse, gt_id, gt_rank,
                  n_museum):
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title = f"Filtro: {filter_name} ({parameter})   |   Muestra: {sample_name}"
    draw.text((45, 26), title, font=FONT_TITLE, fill=(20, 20, 20))
    draw.text((45, 68), f"Aumentación: {augmentation}",
              font=FONT_SUBTITLE, fill=(70, 70, 70))

    left = _display(original)
    right = _display(filtered)
    pair_width = left.width + ARROW_GAP + right.width
    left_x = (CANVAS_WIDTH - pair_width) // 2
    right_x = left_x + left.width + ARROW_GAP
    image_y = IMAGE_TOP

    canvas.paste(left, (left_x, image_y))
    canvas.paste(right, (right_x, image_y))
    draw.rectangle(
        (left_x, image_y, left_x + left.width - 1, image_y + left.height - 1),
        outline=(80, 80, 80), width=4)
    draw.rectangle(
        (right_x, image_y, right_x + right.width - 1, image_y + right.height - 1),
        outline=(30, 90, 180), width=4)

    arrow_y = image_y + left.height // 2
    arrow_x1 = left_x + left.width + 25
    arrow_x2 = right_x - 25
    _draw_arrow(draw, arrow_x1, arrow_x2, arrow_y)

    mse_text = f"MSE = {mse:.2f}"
    _centered_text(
        draw, mse_text, FONT_MSE,
        (arrow_x1 + arrow_x2) / 2, arrow_y - 48, (30, 30, 30))

    label_y = image_y + left.height + 12
    draw.text((left_x, label_y), "Original (con ruido)",
              font=FONT_LABEL, fill=(50, 50, 50))
    draw.text((right_x, label_y), "Filtrada",
              font=FONT_LABEL, fill=(30, 90, 180))

    rank_color = (0, 125, 60) if gt_rank == 1 else (
        (180, 100, 0) if gt_rank <= 5 else (170, 35, 35))
    footer = (f"GT: bbdd_{gt_id:05d}   |   Ranking L1: "
              f"{gt_rank}º de {n_museum}")
    draw.text((45, CANVAS_HEIGHT - 78), footer,
              font=FONT_FOOTER, fill=rank_color)
    draw.text(
        (45, CANVAS_HEIGHT - 48),
        "MSE calculado sobre píxeles RGB (escala 0-255); ranking 1-based.",
        font=FONT_SMALL, fill=(90, 90, 90))

    canvas.save(path, quality=95, subsampling=0, optimize=True)


def main(argv=None):
    specs = filter_specs()
    names = [name for name, _, _ in specs]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--filters", nargs="+", choices=names, default=None,
                        help="Subset of filters (default: all)")
    parser.add_argument("--samples", nargs="+", type=int,
                        default=list(DEFAULT_SAMPLES),
                        help="Query indices, e.g. 16 or 18 20")
    options = parser.parse_args(argv)

    museum_ids, museum_paths = _ordered_image_paths(MUSEUM)
    query_ids, query_paths = _ordered_image_paths(QUERIES)
    ground_truth = _load_ground_truth(GROUND_TRUTH, query_ids)
    import pickle
    with AUGMENTATIONS.open("rb") as handle:
        augmentations = pickle.load(handle)

    for sample_index in options.samples:
        if not 0 <= sample_index < len(query_paths):
            parser.error(f"sample index out of range: {sample_index}")

    museum_descriptors = [
        baseline_descriptor(_read_rgb(path)) for path in museum_paths
    ]
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    selected_names = set(options.filters) if options.filters else set(names)
    specs = [spec for spec in specs if spec[0] in selected_names]

    index_rows = []
    for filter_name, parameter, image_filter in specs:
        folder = output / filter_name.replace("+", "_")
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  {filter_name:<16} {parameter:>12}", flush=True)

        for sample_index in options.samples:
            original = _read_rgb(query_paths[sample_index])
            filtered = image_filter(original)
            mse = float(np.mean(
                (original.astype(np.float64) - filtered.astype(np.float64)) ** 2))
            query_descriptor = baseline_descriptor(filtered)
            full_ranking = rank_museum_images(
                query_descriptor, museum_descriptors,
                DISTANCE_FUNCTIONS["L1"])
            ranked_ids = [museum_ids[index] for index, _ in full_ranking]
            gt_id = int(ground_truth[sample_index][0])
            try:
                gt_rank = ranked_ids.index(gt_id) + 1
            except ValueError:
                gt_rank = len(ranked_ids) + 1

            sample_name = Path(query_paths[sample_index]).name
            image_path = folder / f"{Path(query_paths[sample_index]).stem}.jpg"
            render_sample(
                image_path, filter_name, parameter, sample_name,
                str(augmentations[sample_index]), original, filtered,
                mse, gt_id, gt_rank, len(museum_ids))
            index_rows.append({
                "filter": filter_name,
                "parameter": parameter,
                "sample": sample_name,
                "augmentation": str(augmentations[sample_index]),
                "mse": f"{mse:.4f}",
                "gt_id": gt_id,
                "gt_rank": gt_rank,
                "n_museum": len(museum_ids),
                "folder": folder.relative_to(output).as_posix(),
            })

    with (output / "index.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filter", "parameter", "sample", "augmentation",
                        "mse", "gt_id", "gt_rank", "n_museum", "folder"])
        writer.writeheader()
        writer.writerows(index_rows)

    sample_files = [
        Path(query_paths[index]).name for index in options.samples
    ]
    sample_augmentations = [
        str(augmentations[index]) for index in options.samples
    ]
    readme = [
        "Resultados cualitativos con MSE y ranking L1",
        "=" * 47,
        "",
        "Carpeta nueva; los resultados anteriores no se han modificado.",
        "",
        "Filtros: " + ", ".join(name for name, _, _ in specs),
        "Muestras: " + ", ".join(
            f"{name} ({noise})"
            for name, noise in zip(sample_files, sample_augmentations)
        ),
        f"Kernel fijo: {KERNEL}x{KERNEL} para todos los filtros con kernel.",
        f"Wavelet: {WAVELET}, nivel {WAVELET_LEVEL}.",
        "",
        "Cada filtro tiene una imagen por muestra seleccionada:",
    ]
    readme += [f"  <filtro>/{Path(name).stem}.jpg" for name in sample_files]
    if {"open+close", "close+open"} & selected_names:
        readme += [
            "Los filtros open+close y close+open se guardan como open_close",
            "y close_open.",
        ]
    readme += [
        "",
        "En cada imagen:",
        "  izquierda = consulta original con ruido",
        "  centro    = flecha con el MSE entre original y filtrada",
        "  derecha   = consulta tras aplicar el filtro",
        "  pie       = ID de la imagen GT y su posicion en el ranking L1",
        "",
        "MSE: media de (original - filtrada)^2 sobre los tres canales RGB,",
        "con valores de pixel en 0-255.",
        "Ranking: 1-based, sobre las 287 imagenes de BBDD, usando L1.",
        "",
        "El resumen numerico esta en index.csv.",
    ]
    (output / "README.txt").write_text(
        "\n".join(readme) + "\n", encoding="utf-8")
    print(f"Done. New qualitative results in {output}")


if __name__ == "__main__":
    main()
