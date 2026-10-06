"""Methods_W2: descriptores HSV a probar en Week 2 (Task 1).

Todos trabajan en HSV y aceptan mascara opcional de foreground (Task 3/5):
si ``mask`` es None se usa la imagen completa; si se da, solo los pixeles
de foreground contribuyen al histograma.

Bloques primitivos (ya implementados y listos para ablacion):
    - 3D global: histograma conjunto H-S-V (correlacion color que el
      marginal 1D de W1 pierde).
    - 2D global: histograma conjunto H-S (cromaticidad, invariante a V).
    - Bloque: grid GxG, un histograma conjunto por bloque, concatenados.
    - Piramide (SPM): niveles 1x1 + 2x2 + 4x4 = 21 bloques, concatenados
      con pesos de nivel, L1 final.

Registro propuesto METHODS_W2 (todos HSV, todos L1):
    0. "HSV 3D 8x8x8"            512 dims, comparable al grid W1 (512).
    1. "HSV 2D HS 30x32"         960 dims, croma de alta resolucion.
    2. "HSV block 4x4 HS 8x8"    16*64 = 1024 dims, espacial + croma.
    3. "HSV block 4x4 3D 4x4x4"  16*64 = 1024 dims, espacial + conjunto.
    4. "HSV pyramid HS 8x8"      21*64 = 1344 dims, jerarquico SPM.

Las variantes con mascara (QSD2/QST2) reutilizan estas mismas funciones
pasando ``mask``; no son entradas separadas del registro hasta cerrar la
ablacion de Task 1 en QSD1-W2 (mismo dev set que W1, comparacion directa).
"""

import cv2
import numpy as np

from utils import _l1_normalize, _split_in_blocks

# Rangos OpenCV para HSV (H: 0-180, S/V: 0-256).
_HSV_RANGES = {0: (0, 180), 1: (0, 256), 2: (0, 256)}


def _ranges_for(channels):
    ranges = []
    for ch in channels:
        lo, hi = _HSV_RANGES[ch]
        ranges.extend([lo, hi])
    return ranges


def compute_hsv_3d_histogram(img_hsv, bins=(8, 8, 8), mask=None):
    """Histograma 3D conjunto H-S-V, L1 normalizado."""
    assert len(img_hsv.shape) == 3, "Se espera imagen HSV"
    hist = cv2.calcHist(
        [img_hsv], [0, 1, 2], mask, list(bins),
        [0, 180, 0, 256, 0, 256],
    ).flatten()
    return _l1_normalize(hist)


def compute_hsv_2d_histogram(img_hsv, channels=(0, 1), bins=(16, 16), mask=None):
    """Histograma 2D conjunto (por defecto H-S), L1 normalizado."""
    assert len(img_hsv.shape) == 3, "Se espera imagen HSV"
    assert len(channels) == 2 and len(bins) == 2
    hist = cv2.calcHist(
        [img_hsv], list(channels), mask, list(bins),
        _ranges_for(channels),
    ).flatten()
    return _l1_normalize(hist)


def _split_with_mask(img_hsv, mask, grid_y, grid_x):
    """Bloques de imagen y mascara alineados (mascara puede ser None)."""
    height, width = img_hsv.shape[:2]
    for y in range(grid_y):
        for x in range(grid_x):
            y0, y1 = y * height // grid_y, (y + 1) * height // grid_y
            x0, x1 = x * width // grid_x, (x + 1) * width // grid_x
            m = None if mask is None else mask[y0:y1, x0:x1]
            yield img_hsv[y0:y1, x0:x1], m


def compute_hsv_block_histogram(img_hsv, grid=4, block_fn=None, mask=None):
    """Bloques GxG con un histograma conjunto por bloque, concatenados.

    block_fn(bloque_hsv, mascara_bloque) -> vector 1D. Por defecto HS 8x8.
    Cada bloque se L1-normaliza y el concatenado final tambien (cada
    bloque pesa igual, como en el grid W1).
    """
    assert len(img_hsv.shape) == 3, "Se espera imagen HSV"
    if block_fn is None:
        block_fn = lambda b, m: compute_hsv_2d_histogram(
            b, channels=(0, 1), bins=(8, 8), mask=m
        )
    parts = [
        _l1_normalize(block_fn(block, m))
        for block, m in _split_with_mask(img_hsv, mask, grid, grid)
    ]
    return _l1_normalize(np.concatenate(parts))


def compute_hsv_block_3d_histogram(img_hsv, grid=4, bins=(4, 4, 4), mask=None):
    """Bloques GxG con histograma 3D por bloque."""
    return compute_hsv_block_histogram(
        img_hsv, grid=grid,
        block_fn=lambda b, m: compute_hsv_3d_histogram(b, bins=bins, mask=m),
        mask=mask,
    )


def compute_hsv_block_2d_hs_histogram(img_hsv, grid=4, bins=(8, 8), mask=None):
    """Bloques GxG con histograma 2D H-S por bloque."""
    return compute_hsv_block_histogram(
        img_hsv, grid=grid,
        block_fn=lambda b, m: compute_hsv_2d_histogram(
            b, channels=(0, 1), bins=bins, mask=m
        ),
        mask=mask,
    )


def compute_hsv_pyramid_histogram(
    img_hsv, levels=(1, 2, 4), level_weights=None, mask=None,
    block_fn=None,
):
    """Piramide espacial: bloques en cada nivel, concatenados con pesos.

    levels: tupla de grids (1,2,4) -> 1 + 4 + 16 = 21 bloques.
    level_weights: peso por nivel; por defecto pesos SPM
        w_l = 1 / 2^{L-l} (nivel mas fino pesa mas), normalizados a suma 1.
    block_fn: descriptor por bloque; por defecto HS 8x8.
    Cada nivel se L1-normaliza antes de ponderar; el vector final se
    L1-normaliza para comparar con L1/Chi-square como en W1.
    """
    assert len(img_hsv.shape) == 3, "Se espera imagen HSV"
    if block_fn is None:
        block_fn = lambda b, m: compute_hsv_2d_histogram(
            b, channels=(0, 1), bins=(8, 8), mask=m
        )
    if level_weights is None:
        n = len(levels)
        level_weights = np.array(
            [1.0 / (2 ** (n - 1 - i)) for i in range(n)], dtype=np.float32
        )
        level_weights /= level_weights.sum()
    assert len(level_weights) == len(levels)

    level_vectors = []
    for grid, weight in zip(levels, level_weights):
        parts = [
            _l1_normalize(block_fn(block, m))
            for block, m in _split_with_mask(img_hsv, mask, grid, grid)
        ]
        level_vectors.append(float(weight) * _l1_normalize(np.concatenate(parts)))
    return _l1_normalize(np.concatenate(level_vectors))


# --- Registro W2: nombre -> funcion(img_hsv, mask=None) ---
# Todas reciben HSV (convertir desde RGB antes) para que la comparacion
# con W1 sea directa. Envoltorios lambda fijan los hiperparametros
# propuestos; la ablacion variara bins/grid/pesos desde aqui.
METHODS_W2 = {
    "HSV 3D 8x8x8": lambda hsv, mask=None: compute_hsv_3d_histogram(
        hsv, bins=(8, 8, 8), mask=mask
    ),
    "HSV 2D HS 30x32": lambda hsv, mask=None: compute_hsv_2d_histogram(
        hsv, channels=(0, 1), bins=(30, 32), mask=mask
    ),
    "HSV block 4x4 HS 8x8": lambda hsv, mask=None: compute_hsv_block_2d_hs_histogram(
        hsv, grid=4, bins=(8, 8), mask=mask
    ),
    "HSV block 4x4 3D 4x4x4": lambda hsv, mask=None: compute_hsv_block_3d_histogram(
        hsv, grid=4, bins=(4, 4, 4), mask=mask
    ),
    "HSV pyramid HS 8x8": lambda hsv, mask=None: compute_hsv_pyramid_histogram(
        hsv, levels=(1, 2, 4), level_weights=None, mask=mask
    ),
}

METHODS_W2_NAMES = list(METHODS_W2.keys())


def compute_descriptors_w2(dataset, masks=None):
    """Descriptores W2 para un dataset; masks opcional (foreground QSD2).

    dataset.images en RGB (como Dataset); masks lista de HxW uint8 o None.
    Devuelve una lista por metodo W2, en orden METHODS_W2_NAMES.
    """
    import cv2 as _cv2

    methods = [[] for _ in METHODS_W2_NAMES]
    n = len(dataset.images)
    if masks is not None and len(masks) != n:
        raise ValueError(f"Esperaba {n} mascaras, recibi {len(masks)}")
    for i, img in enumerate(dataset.images):
        hsv = _cv2.cvtColor(img, _cv2.COLOR_RGB2HSV)
        mask = None if masks is None else masks[i]
        for j, name in enumerate(METHODS_W2_NAMES):
            methods[j].append(METHODS_W2[name](hsv, mask))
    return methods
