"""Reproduce brightness weighting, fixed HSV/Lab fusion and extra grids.

Run study.py first, then this file. Results are QSD1 development experiments.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import study as s


def load_regions(label):
    with np.load(s.OUT / f'{label}_regions.npz', allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def row_result(m, q, metric, ids, truth, **settings):
    ap1, ap5, predicted, aps = s.evaluate(m, q, metric, ids, truth)
    return dict(settings, distance=metric, dimension=int(m.shape[1]),
                **{'mAP@1': ap1, 'mAP@5': ap5, 'predicted': predicted,
                   'AP@1': aps[1], 'AP@5': aps[5]})


def brightness(museum, queries, ids, truth):
    rows = []
    for bins in (16, 32, 64):
        name = f'Lab 1D {bins}'
        for layout, grids in [('global', (1,)), ('blocks 4x4', (4,)),
                              ('pyramid 1+2+4', (1, 2, 4))]:
            m = s.descriptor(museum, name, grids, 'cells')
            q = s.descriptor(queries, name, grids, 'cells')
            cells = sum(g*g for g in grids)
            for weight in (0.0, 0.2, 1/3, 0.5):
                weights = np.array([weight, (1-weight)/2, (1-weight)/2], dtype=np.float32)*3
                wm = s.normalize((m.reshape(-1, cells, 3, bins)*weights[None, None, :, None]).reshape(len(m), -1))
                wq = s.normalize((q.reshape(-1, cells, 3, bins)*weights[None, None, :, None]).reshape(len(q), -1))
                for metric in s.METRICS:
                    rows.append(row_result(wm, wq, metric, ids, truth,
                                           template=name, layout=layout, L_weight=weight))
        print(f'Brightness weights: Lab {bins}', flush=True)
    return s.save_rows(rows, 'brightness_study')


def fusion(museum, queries, ids, truth):
    def scores(name, grids):
        m = np.asarray(s.descriptor(museum, name, grids, 'cells'), dtype=np.float64)
        q = s.descriptor(queries, name, grids, 'cells')
        return np.asarray([np.abs(m-np.asarray(row, dtype=np.float64)).sum(axis=1)/2 for row in q]), m.shape[1]
    lab_blocks, db = scores('Lab 1D 32', (4,))
    lab_pyramid, dp = scores('Lab 1D 32', (1, 2, 4))
    hsv_grid, dh = scores('HSV 1D W1-grid bins', (4,))
    hsv_concat, dc = scores('HSV 2D HS+HV 16x16', (1, 2, 4))
    rows = []
    pairs = [('Lab blocks + W1 HSV grid', lab_blocks, hsv_grid, db+dh),
             ('Lab pyramid + W1 HSV grid', lab_pyramid, hsv_grid, dp+dh),
             ('Lab blocks + Andreu HSV pyramid', lab_blocks, hsv_concat, db+dc)]
    for label, a, b, dim in pairs:
        for weight in (0.25, 0.5, 0.75):
            ranking = np.argsort(weight*a+(1-weight)*b, axis=1, kind='stable')[:, :10]
            predicted = [[ids[i] for i in row] for row in ranking]
            aps = {k: [s.average_precision_at_k(p, t, k) for p, t in zip(predicted, truth)] for k in (1, 5)}
            rows.append({'template': label, 'Lab_weight': weight, 'distance': 'weighted L1',
                         'dimension': int(dim), 'mAP@1': float(np.mean(aps[1])),
                         'mAP@5': float(np.mean(aps[5])), 'predicted': predicted,
                         'AP@1': aps[1], 'AP@5': aps[5]})
    return s.save_rows(rows, 'fusion_study')


def extra_grids(ids, truth):
    selected = [t for t in s.TEMPLATES if t['name'] in ('Lab 1D 32', 'HSV 1D W1-grid bins')]
    sets = []
    for label, folder in [('museum', s.ROOT / 'data/BBDD'), ('queries', s.ROOT / 'data/qsd1_w1')]:
        paths = sorted(folder.glob('*.jpg'), key=s.numeric_id)
        cache = s.OUT / f'{label}_extra_grids.npz'
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                regions = {key: data[key] for key in data.files}
        else:
            regions = {}
            for index, path in enumerate(paths, 1):
                for key, values in s.image_regions(cv2.imread(str(path)), selected, (3, 6)).items():
                    regions.setdefault(key, []).append(values)
                if index % 50 == 0 or index == len(paths):
                    print(f'Extra grids {label}: {index}/{len(paths)}', flush=True)
            regions = {key: np.asarray(values) for key, values in regions.items()}
            np.savez_compressed(cache, **regions)
        base = load_regions(label)
        for name in [t['name'] for t in selected]:
            regions[f'{name}|1'] = base[f'{name}|1']
        sets.append(regions)
    rows = []
    for spec in selected:
        for label, levels in [('blocks 3x3', (3,)), ('blocks 6x6', (6,)),
                              ('pyramid 1+3+6', (1, 3, 6))]:
            m = s.descriptor(sets[0], spec['name'], levels, 'cells')
            q = s.descriptor(sets[1], spec['name'], levels, 'cells')
            for metric in s.METRICS:
                rows.append(row_result(m, q, metric, ids, truth, template=spec['name'], layout=label))
    return s.save_rows(rows, 'extra_grids_study')


def main():
    cv2.setNumThreads(1)
    meta = json.loads((s.OUT / 'metadata.json').read_text())
    ids = [s.numeric_id(p) for p in sorted((s.ROOT / 'data/BBDD').glob('*.jpg'), key=s.numeric_id)]
    museum, queries = load_regions('museum'), load_regions('queries')
    for fn in (brightness, fusion):
        result = fn(museum, queries, ids, meta['ground_truth'])
        print({k:v for k,v in result[0].items() if k not in ('predicted', 'AP@1', 'AP@5')}, flush=True)
    result = extra_grids(ids, meta['ground_truth'])
    print({k:v for k,v in result[0].items() if k not in ('predicted', 'AP@1', 'AP@5')}, flush=True)


if __name__ == '__main__':
    main()
