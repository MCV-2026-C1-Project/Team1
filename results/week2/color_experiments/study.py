"""Controlled QSD1 development study; reuse the project's descriptors and metrics.

This writes only to its output directory. No test queries or QSD2 masks are used.
"""
import csv
import hashlib
import io
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from k_similarity import average_precision_at_k, DISTANCE_FUNCTIONS
from methods_w1 import METHODS_W1
from methods_w2 import METHODS_W2
from spatial_pyramid import compute_spatial_pyramid


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('Expected plain dataset annotations')


def normalize(a):
    a = np.asarray(a, dtype=np.float32)
    return a / (a.sum(axis=-1, keepdims=True) + 1e-12)


def template(name, space, parts):
    return {'name': name, 'space': space, 'parts': parts}


TEMPLATES = [
    template('HSV 1D W1-global bins', 'HSV', [((0,), (30,)), ((1,), (32,)), ((2,), (32,))]),
    template('HSV 1D W1-grid bins', 'HSV', [((0,), (16,)), ((1,), (8,)), ((2,), (8,))]),
    template('HSV 1D 16', 'HSV', [((c,), (16,)) for c in range(3)]),
    template('HSV 1D 32', 'HSV', [((c,), (32,)) for c in range(3)]),
    *[template(f'Lab 1D {bins}', 'Lab', [((c,), (bins,)) for c in range(3)]) for bins in (16, 32, 64)],
    template('RGB 1D 32', 'RGB', [((c,), (32,)) for c in range(3)]),
    template('YCrCb 1D 32', 'YCrCb', [((c,), (32,)) for c in range(3)]),
    *[template(f'HSV 2D HS {h}x{s}', 'HSV', [((0, 1), (h, s))]) for h, s in ((8, 8), (16, 16), (30, 32))],
    template('HSV 2D HV 8x8', 'HSV', [((0, 2), (8, 8))]),
    template('HSV 2D HS+HV 16x16', 'HSV', [((0, 1), (16, 16)), ((0, 2), (16, 16))]),
    *[template(f'HSV 3D {bins}', 'HSV', [((0, 1, 2), (bins,) * 3)]) for bins in (4, 8)],
    *[template(f'Lab 2D ab {bins}', 'Lab', [((1, 2), (bins, bins))]) for bins in (8, 16, 32)],
    *[template(f'Lab 3D {bins}', 'Lab', [((0, 1, 2), (bins,) * 3)]) for bins in (4, 8)],
]
LAYOUTS = [
    ('global', (1,), 'cells'),
    ('blocks 2x2', (2,), 'cells'),
    ('blocks 4x4', (4,), 'cells'),
    ('pyramid 1+2', (1, 2), 'cells'),
    ('pyramid 1+2+4', (1, 2, 4), 'cells'),
    ('pyramid 1+2+4 equal levels', (1, 2, 4), 'levels'),
]
METRICS = ('L1', 'Chi-square', 'Hellinger')


def region_histogram(image, spec):
    parts = []
    for channels, bins in spec['parts']:
        ranges = []
        for c in channels:
            ranges.extend((0, 180 if spec['space'] == 'HSV' and c == 0 else 256))
        hist = cv2.calcHist([image], list(channels), None, list(bins), ranges).ravel()
        parts.append(normalize(hist))
    return normalize(np.concatenate(parts))


def image_regions(bgr, templates=TEMPLATES, grids=(1, 2, 4)):
    spaces = {'HSV': cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
              'Lab': cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab),
              'RGB': cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
              'YCrCb': cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)}
    result = {}
    for spec in templates:
        im = spaces[spec['space']]
        h, w = im.shape[:2]
        for grid in grids:
            parts = []
            for y in range(grid):
                for x in range(grid):
                    block = im[y*h//grid:(y+1)*h//grid, x*w//grid:(x+1)*w//grid]
                    parts.append(region_histogram(block, spec))
            result[f"{spec['name']}|{grid}"] = np.concatenate(parts)
    return result


def descriptor(regions, name, grids, weighting):
    levels = [regions[f'{name}|{g}'] for g in grids]
    if weighting == 'levels':
        levels = [normalize(level) for level in levels]
    return normalize(np.concatenate(levels, axis=-1))


def validate_definitions():
    rgb = np.random.default_rng(73).integers(0, 256, (19, 23, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    regions = image_regions(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    checks = [
        (descriptor(regions, 'HSV 1D W1-global bins', (1,), 'cells'), METHODS_W1['HSV'](rgb)),
        (descriptor(regions, 'HSV 1D W1-grid bins', (4,), 'cells'), METHODS_W1['HSV grid'](rgb)),
        (descriptor(regions, 'Lab 1D 32', (1, 2, 4), 'cells'), compute_spatial_pyramid(rgb, color_space='Lab', bins=32)),
        (descriptor(regions, 'HSV 2D HS+HV 16x16', (1, 2, 4), 'cells'), METHODS_W2['HSV pyramid HS+HV 16x16'](hsv)),
    ]
    for actual, expected in checks:
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
    print('Descriptor definitions match W1, Andreu and Amrit reference code.', flush=True)


def numeric_id(path):
    return int(path.stem.split('_')[-1])


def compute_set(paths, label):
    cache = OUT / f'{label}_regions.npz'
    signature = hashlib.sha256(json.dumps([
        [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths], TEMPLATES
    ]).encode()).hexdigest()
    if cache.exists() and cache.with_suffix('.signature').read_text() == signature:
        with np.load(cache, allow_pickle=False) as data:
            print(f'{label}: using cached region histograms.', flush=True)
            return {key: data[key] for key in data.files}
    rows = {}
    for i, path in enumerate(paths, 1):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f'Cannot decode {path}')
        for key, values in image_regions(bgr).items():
            rows.setdefault(key, []).append(values)
        if i % 25 == 0 or i == len(paths):
            print(f'{label}: {i}/{len(paths)} images', flush=True)
    result = {key: np.asarray(values) for key, values in rows.items()}
    np.savez_compressed(cache, **result)
    cache.with_suffix('.signature').write_text(signature)
    return result


def rank_matrix(museum, queries, metric):
    # Float64 matches the comparison functions in similarity_functions.py.
    museum = np.asarray(museum, dtype=np.float64)
    rankings = []
    for query in queries:
        query = np.asarray(query, dtype=np.float64)
        if metric == 'L1':
            scores = np.abs(museum-query).sum(axis=1)
        elif metric == 'Chi-square':
            den = museum+query
            numerator = (museum-query)**2
            np.divide(numerator, den, out=numerator, where=den > 0)
            scores = numerator.sum(axis=1)
        elif metric == 'Hellinger':
            scores = -np.sqrt(museum*query).sum(axis=1)
        else:
            raise ValueError(metric)
        rankings.append(np.argsort(scores, kind='stable')[:10])
    return np.asarray(rankings)


def evaluate(museum, queries, metric, museum_ids, gt):
    ranked = rank_matrix(museum, queries, metric)
    predicted = [[museum_ids[i] for i in row] for row in ranked]
    aps = {k: [average_precision_at_k(row, truth, k) for row, truth in zip(predicted, gt)] for k in (1, 5)}
    return float(np.mean(aps[1])), float(np.mean(aps[5])), predicted, aps


def save_rows(rows, filename):
    rows = sorted(rows, key=lambda row: (-row['mAP@5'], -row['mAP@1'], row['dimension']))
    summary = [{k: v for k, v in row.items() if k not in ('predicted', 'AP@1', 'AP@5')} for row in rows]
    with (OUT / filename).with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUT / filename).with_suffix('.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows


def main():
    start = time.time()
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True)
    validate_definitions()
    museum_paths = sorted((ROOT / 'data/BBDD').glob('*.jpg'), key=numeric_id)
    query_paths = sorted((ROOT / 'data/qsd1_w1').glob('*.jpg'), key=numeric_id)
    museum_ids = [numeric_id(p) for p in museum_paths]
    gt_bytes = (ROOT / 'data/qsd1_w1/gt_corresps.pkl').read_bytes()
    gt = DataOnlyUnpickler(io.BytesIO(gt_bytes)).load()
    assert len(gt) == len(query_paths)
    museum = compute_set(museum_paths, 'museum')
    queries = compute_set(query_paths, 'queries')
    rows = []
    for spec in TEMPLATES:
        for layout, levels, weighting in LAYOUTS:
            m = descriptor(museum, spec['name'], levels, weighting)
            q = descriptor(queries, spec['name'], levels, weighting)
            for metric in METRICS:
                ap1, ap5, predicted, aps = evaluate(m, q, metric, museum_ids, gt)
                rows.append({'template': spec['name'], 'space': spec['space'], 'layout': layout,
                             'distance': metric, 'dimension': int(m.shape[1]),
                             'mAP@1': ap1, 'mAP@5': ap5, 'predicted': predicted,
                             'AP@1': aps[1], 'AP@5': aps[5]})
        print(f"Evaluated {spec['name']}; {len(rows)} configurations.", flush=True)
    rows = save_rows(rows, 'controlled_study')
    metadata = {'museum_count': len(museum_paths), 'query_count': len(query_paths),
                'query_ids': [numeric_id(p) for p in query_paths], 'ground_truth': gt,
                'configuration_count': len(rows), 'elapsed_seconds': time.time()-start,
                'opencv_version': cv2.__version__, 'selection': 'development mAP@5, mAP@1, dimension',
                'scope': 'QSD1 only; exploratory development comparison, no unseen-test claim'}
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('\nTOP CONFIGURATIONS:', flush=True)
    for row in rows[:15]:
        print({k:v for k,v in row.items() if k not in ('predicted','AP@1','AP@5')}, flush=True)
    print(f'Completed {len(rows)} configurations in {time.time()-start:.1f}s.', flush=True)


if __name__ == '__main__':
    main()
