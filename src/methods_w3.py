"""Methods_W3: Week 3, Task 1 (noise filtering) and the unfiltered baseline.

Task 1 asks for a linear or non-linear filter for an unknown noise model,
where only some queries are noisy. This module implements the non-linear
median filter and applies it to the queries by default; the museum/reference
images (BBDD) stay unfiltered unless the symmetric variant is requested with
``filter_scope="both"``.

Two retrieval methods share the Week 2 "Lab blocks 1D" descriptor:

    - "Lab blocks 1D"         baseline: the query is used as it is.
    - "Lab blocks 1D median"  the query is median-filtered before the
                              descriptor.

Both use the W2 configuration: CIE-Lab, 32 bins per channel, 5x5 = 25
spatial cells, L1-normalized histograms and the L1 distance.

Besides the median filter, the module exposes linear Average and Gaussian
filters, per-channel morphological opening and closing, chained opening+closing
in both orders and a low-pass wavelet filter (keep the approximation
coefficients, zero the details). Filters can be chained with "+" (for example
"open+close"). These are query-side candidates for the same Task 1
comparison; pass ``--filter-scope both`` to apply the same filter to the
references too. Every filter is registered as a ``"Lab blocks 1D <filter>"``
method sharing one ``--kernel`` size parameter (the wavelet uses
``--wavelet-level`` instead).

The HSV hue-invariance approach lives in ``src/color_reference`` and is
available here as an additional retrieval mode: pass ``hue_invariant=True``
(or ``--hue-invariant``) to estimate a per-candidate global hue scale and
blend the raw and hue-normalized LAB distances. The descriptor and distance
stay "Lab blocks 1D" + L1 in every case.

Run as a script (argparse selects the action):
    python methods_w3.py --list
    python methods_w3.py --museum <dir> --queries <dir> --method NAME
    python methods_w3.py --museum <dir> --queries <dir> --method all \
        --ground-truth <pkl>
    python methods_w3.py --museum <dir> --queries <dir> --method NAME \
        --pre-filter {none,average,median,gaussian,wavelet_lowpass,open,close,open+close,close+open} \
        --kernel SIZE [--wavelet-level LEVEL] [--filter-scope {query,both}] \
        [--hue-invariant [--hue-reference <npz>]] [--ground-truth <pkl>]
    python methods_w3.py --museum <dir> --queries <dir> --sweep \
        --ground-truth <pkl> [--median-sizes 3 5 7 ...] \
        [--morph-sizes 3 5] [--chain-sizes 3 5 7 ...] \
        [--wavelet-levels 1 2 3] [--filter-scope {query,both}] \
        [--stop-on-drop]
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from methods_w2 import (
    _load_ground_truth,
    _ordered_image_paths,
    _read_rgb,
    _report_rankings,
    blocks_grid,
    compute_lab_block_histogram,
)

DEFAULT_BINS = 32
DEFAULT_BLOCKS = 25
DEFAULT_AVERAGE_KSIZE = 3
DEFAULT_MEDIAN_KSIZE = 3
DEFAULT_MORPH_KSIZE = 3
DEFAULT_GAUSSIAN_KSIZE = 3
DEFAULT_GAUSSIAN_SIGMA = 0
DEFAULT_WAVELET = "db2"
DEFAULT_WAVELET_LEVEL = 3
DEFAULT_DISTANCE = "L1"
DEFAULT_HUE_ALPHA = 1.0
DEFAULT_HUE_TEMPERATURE = 1.0
FILTER_SCOPES = ("query", "both")
DEFAULT_FILTER_SCOPE = "query"

MORPH_OPERATIONS = ("open", "close")
WAVELET_FILTER_NAMES = ("wavelet", "wavelet_lowpass")
IMAGE_FILTERS = (
    "none", "average", "median", "gaussian",
    "wavelet_lowpass", "open+close", "close+open",
) + MORPH_OPERATIONS
AVERAGE_SWEEP_SIZES = (3, 5, 7, 9, 11, 13, 15)
MEDIAN_SWEEP_SIZES = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
MORPH_SWEEP_SIZES = (3, 5)
CHAIN_SWEEP_SIZES = (3, 5, 7, 9, 11)
GAUSSIAN_SWEEP_SIZES = (3, 5, 7, 9, 11, 13, 15)
WAVELET_SWEEP_LEVELS = (1, 2, 3)

BASELINE_METHOD = "Lab blocks 1D"
MEDIAN_METHOD = "Lab blocks 1D median"
AVERAGE_METHOD = "Lab blocks 1D average"
GAUSSIAN_METHOD = "Lab blocks 1D gaussian"
OPEN_METHOD = "Lab blocks 1D open"
CLOSE_METHOD = "Lab blocks 1D close"
OPEN_CLOSE_METHOD = "Lab blocks 1D open+close"
CLOSE_OPEN_METHOD = "Lab blocks 1D close+open"
WAVELET_METHOD = "Lab blocks 1D wavelet"
METHODS_W3_NAMES = (
    BASELINE_METHOD,
    MEDIAN_METHOD,
    AVERAGE_METHOD,
    GAUSSIAN_METHOD,
    OPEN_METHOD,
    CLOSE_METHOD,
    OPEN_CLOSE_METHOD,
    CLOSE_OPEN_METHOD,
    WAVELET_METHOD,
)

_METHOD_FILTERS = {
    BASELINE_METHOD: None,
    MEDIAN_METHOD: "median",
    AVERAGE_METHOD: "average",
    GAUSSIAN_METHOD: "gaussian",
    OPEN_METHOD: "open",
    CLOSE_METHOD: "close",
    OPEN_CLOSE_METHOD: "open+close",
    CLOSE_OPEN_METHOD: "close+open",
    WAVELET_METHOD: "wavelet_lowpass",
}

METHOD_ALL = "all"
METHOD_CHOICES = METHODS_W3_NAMES + (METHOD_ALL,)

_METHOD_ALIASES = {
    "baseline": BASELINE_METHOD,
    "lab blocks 1d": BASELINE_METHOD,
    "lab blocks 1d baseline": BASELINE_METHOD,
    "median": MEDIAN_METHOD,
    "median filter": MEDIAN_METHOD,
    "lab blocks 1d median": MEDIAN_METHOD,
    "lab blocks 1d + median": MEDIAN_METHOD,
    "median + lab blocks 1d": MEDIAN_METHOD,
    "average": AVERAGE_METHOD,
    "mean": AVERAGE_METHOD,
    "lab blocks 1d average": AVERAGE_METHOD,
    "lab blocks 1d + average": AVERAGE_METHOD,
    "average + lab blocks 1d": AVERAGE_METHOD,
    "gaussian": GAUSSIAN_METHOD,
    "lab blocks 1d gaussian": GAUSSIAN_METHOD,
    "lab blocks 1d + gaussian": GAUSSIAN_METHOD,
    "gaussian + lab blocks 1d": GAUSSIAN_METHOD,
    "open": OPEN_METHOD,
    "opening": OPEN_METHOD,
    "lab blocks 1d open": OPEN_METHOD,
    "lab blocks 1d + open": OPEN_METHOD,
    "close": CLOSE_METHOD,
    "closing": CLOSE_METHOD,
    "lab blocks 1d close": CLOSE_METHOD,
    "lab blocks 1d + close": CLOSE_METHOD,
    "open+close": OPEN_CLOSE_METHOD,
    "open + close": OPEN_CLOSE_METHOD,
    "lab blocks 1d open+close": OPEN_CLOSE_METHOD,
    "close+open": CLOSE_OPEN_METHOD,
    "close + open": CLOSE_OPEN_METHOD,
    "lab blocks 1d close+open": CLOSE_OPEN_METHOD,
    "wavelet": WAVELET_METHOD,
    "wavelet_lowpass": WAVELET_METHOD,
    "wavelet lowpass": WAVELET_METHOD,
    "low-pass wavelet": WAVELET_METHOD,
    "lowpass wavelet": WAVELET_METHOD,
    "lab blocks 1d wavelet": WAVELET_METHOD,
    "lab blocks 1d + wavelet": WAVELET_METHOD,
}


def _check_median_ksize(ksize):
    assert (isinstance(ksize, int) and not isinstance(ksize, bool)
            and ksize >= 3 and ksize % 2 == 1), (
        f"median ksize must be an odd integer >= 3, got {ksize!r}")
    return ksize


def _check_linear_ksize(name, ksize):
    assert (isinstance(ksize, int) and not isinstance(ksize, bool)
            and ksize >= 1 and ksize % 2 == 1), (
        f"{name} ksize must be an odd integer >= 1, got {ksize!r}")
    return ksize


def average_filter(img_rgb, ksize=DEFAULT_AVERAGE_KSIZE):
    """Linear box (average) filter for a grayscale or RGB uint8 image."""
    ksize = _check_linear_ksize("average", ksize)
    image = np.asarray(img_rgb)
    assert image.dtype == np.uint8, (
        f"average_filter expects a uint8 image, got {image.dtype}")
    assert image.ndim in (2, 3), (
        f"average_filter expects HxW or HxWxC, got shape {image.shape}")
    return cv2.blur(image, (ksize, ksize))


def median_filter(img_rgb, ksize=DEFAULT_MEDIAN_KSIZE):
    """Non-linear median filter, applied per channel.

    Accepts a grayscale HxW or an RGB HxWx3 uint8 image and returns an image
    of the same shape and dtype. The median is taken independently in every
    channel, so no color from one channel leaks into another.
    """
    ksize = _check_median_ksize(ksize)
    image = np.asarray(img_rgb)
    assert image.dtype == np.uint8, (
        f"median_filter expects a uint8 image, got {image.dtype}")
    if image.ndim == 2:
        return cv2.medianBlur(image, ksize)
    assert image.ndim == 3, (
        f"median_filter expects HxW or HxWxC, got shape {image.shape}")
    channels = [
        cv2.medianBlur(np.ascontiguousarray(image[..., channel]), ksize)
        for channel in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1)


def _check_morph_ksize(ksize):
    assert (isinstance(ksize, int) and not isinstance(ksize, bool)
            and ksize >= 1 and ksize % 2 == 1), (
        f"morphology ksize must be an odd integer >= 1, got {ksize!r}")
    return ksize


def morphology_filter(img_rgb, operation, ksize=DEFAULT_MORPH_KSIZE):
    """Morphological filter applied per channel.

    operation: "open" or "close". The structuring element
    is an ellipse of size ksize x ksize. Works on grayscale HxW or RGB HxWx3
    uint8 images and keeps the shape and dtype.
    """
    if operation not in MORPH_OPERATIONS:
        raise ValueError(f"Unknown morphology operation: {operation!r}")
    ksize = _check_morph_ksize(ksize)
    image = np.asarray(img_rgb)
    assert image.dtype == np.uint8, (
        f"morphology_filter expects a uint8 image, got {image.dtype}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    def apply(channel):
        code = cv2.MORPH_OPEN if operation == "open" else cv2.MORPH_CLOSE
        return cv2.morphologyEx(channel, code, kernel)

    if image.ndim == 2:
        return apply(image)
    assert image.ndim == 3, (
        f"morphology_filter expects HxW or HxWxC, got shape {image.shape}")
    channels = [
        apply(np.ascontiguousarray(image[..., channel]))
        for channel in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1)


def open_filter(img_rgb, ksize=DEFAULT_MORPH_KSIZE):
    """Morphological opening: erosion followed by dilation."""
    return morphology_filter(img_rgb, "open", ksize)


def close_filter(img_rgb, ksize=DEFAULT_MORPH_KSIZE):
    """Morphological closing: dilation followed by erosion."""
    return morphology_filter(img_rgb, "close", ksize)


opening_filter = open_filter
closing_filter = close_filter


def gaussian_filter(img_rgb, ksize=DEFAULT_GAUSSIAN_KSIZE,
                    sigma=DEFAULT_GAUSSIAN_SIGMA):
    """Linear Gaussian blur for a grayscale or RGB uint8 image.

    ``sigma=0`` lets OpenCV derive the standard deviation from ``ksize``.
    """
    assert (isinstance(ksize, int) and not isinstance(ksize, bool)
            and ksize >= 1 and ksize % 2 == 1), (
        f"gaussian ksize must be an odd integer >= 1, got {ksize!r}")
    assert sigma >= 0, f"gaussian sigma must be >= 0, got {sigma!r}"
    image = np.asarray(img_rgb)
    assert image.dtype == np.uint8, (
        f"gaussian_filter expects a uint8 image, got {image.dtype}")
    assert image.ndim in (2, 3), (
        f"gaussian_filter expects HxW or HxWxC, got shape {image.shape}")
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def _check_wavelet_level(level):
    assert (isinstance(level, int) and not isinstance(level, bool)
            and 1 <= level <= 5), (
        f"wavelet level must be an integer in [1, 5], got {level!r}")
    return level


def wavelet_lowpass_filter(img_rgb, wavelet=DEFAULT_WAVELET,
                           level=DEFAULT_WAVELET_LEVEL):
    """Low-pass wavelet filter applied per channel.

    Each channel is decomposed with ``pywt.wavedec2``; only the approximation
    (LL) coefficients are kept, every detail coefficient is zeroed, and the
    image is reconstructed with ``pywt.waverec2`` and cropped back to HxW.
    Accepts a grayscale HxW or an RGB HxWx3 uint8 image and returns an image
    of the same shape and dtype.
    """
    try:
        import pywt
    except ImportError as error:
        raise ImportError(
            "wavelet_lowpass_filter requires the 'pywt' (PyWavelets) "
            "package") from error
    level = _check_wavelet_level(level)
    image = np.asarray(img_rgb)
    assert image.dtype == np.uint8, (
        f"wavelet_lowpass_filter expects a uint8 image, got {image.dtype}")
    assert image.ndim in (2, 3), (
        f"wavelet_lowpass_filter expects HxW or HxWxC, got shape "
        f"{image.shape}")

    def lowpass(channel):
        height, width = channel.shape[:2]
        coeffs = pywt.wavedec2(
            np.asarray(channel, dtype=np.float64), wavelet=wavelet,
            level=level)
        filtered = [coeffs[0]] + [
            (np.zeros_like(h), np.zeros_like(v), np.zeros_like(d))
            for (h, v, d) in coeffs[1:]
        ]
        rec = pywt.waverec2(filtered, wavelet=wavelet)
        return np.clip(np.rint(rec[:height, :width]), 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return lowpass(image)
    channels = [lowpass(np.ascontiguousarray(image[..., channel]))
                for channel in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def _chain_filters(names, ksize=None, wavelet=DEFAULT_WAVELET,
                   wavelet_level=DEFAULT_WAVELET_LEVEL):
    filters = [
        build_image_filter(name, ksize=ksize, wavelet=wavelet,
                           wavelet_level=wavelet_level)
        for name in names
    ]

    def apply_chain(image):
        for image_filter in filters:
            image = image_filter(image)
        return image

    return apply_chain


def build_image_filter(name, ksize=None, wavelet=DEFAULT_WAVELET,
                       wavelet_level=DEFAULT_WAVELET_LEVEL):
    """Return ``fn(rgb) -> rgb`` for a preprocessing filter name.

    ``ksize`` is the shared spatial kernel size for average/median/gaussian/
    morphology filters (chained names with "+" reuse it for every stage).
    The wavelet filter ignores ``ksize`` and uses ``wavelet``/``wavelet_level``
    instead.
    """
    key = name.strip().lower()
    aliases = {
        "identity": "none",
        "mean": "average",
        "box": "average",
        "opening": "open",
        "closing": "close",
        "wavelet": "wavelet_lowpass",
        "wavelet-lowpass": "wavelet_lowpass",
        "lowpass": "wavelet_lowpass",
        "low-pass": "wavelet_lowpass",
        "low_pass": "wavelet_lowpass",
    }
    key = aliases.get(key, key)
    if key == "wavelet_lowpass":
        level = (DEFAULT_WAVELET_LEVEL if wavelet_level is None
                 else wavelet_level)
        family = DEFAULT_WAVELET if wavelet is None else wavelet
        return lambda image: wavelet_lowpass_filter(
            image, wavelet=family, level=level)
    if "+" in key:
        return _chain_filters(key.split("+"), ksize=ksize, wavelet=wavelet,
                              wavelet_level=wavelet_level)
    if key == "none":
        return lambda image: image
    if key == "average":
        ksize = DEFAULT_AVERAGE_KSIZE if ksize is None else ksize
        return lambda image: average_filter(image, ksize=ksize)
    if key == "median":
        ksize = DEFAULT_MEDIAN_KSIZE if ksize is None else ksize
        return lambda image: median_filter(image, ksize=ksize)
    if key == "gaussian":
        ksize = DEFAULT_GAUSSIAN_KSIZE if ksize is None else ksize
        return lambda image: gaussian_filter(image, ksize=ksize)
    if key in MORPH_OPERATIONS:
        ksize = DEFAULT_MORPH_KSIZE if ksize is None else ksize
        return lambda image: morphology_filter(image, key, ksize=ksize)
    raise ValueError(f"Unknown image filter: {name!r}")


def compute_lab_blocks_1d_histogram(img_lab, blocks=DEFAULT_BLOCKS,
                                    bins=DEFAULT_BINS, mask=None):
    """W2 "Lab blocks 1D" descriptor on an already converted Lab image.

    blocks is the total number of cells (a perfect square, 25 -> 5x5); the
    W2 histogram function normalizes every cell and the concatenation with L1.
    """
    assert len(img_lab.shape) == 3, "Expected Lab image"
    return compute_lab_block_histogram(
        img_lab, grid=blocks_grid(blocks), bins=bins, mask=mask)


def baseline_descriptor(img_rgb, blocks=DEFAULT_BLOCKS, bins=DEFAULT_BINS,
                        mask=None):
    """Unfiltered Lab blocks 1D descriptor for an RGB uint8 image."""
    assert len(img_rgb.shape) == 3, "Expected RGB image"
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return compute_lab_blocks_1d_histogram(
        lab, blocks=blocks, bins=bins, mask=mask)


def median_descriptor(img_rgb, ksize=DEFAULT_MEDIAN_KSIZE,
                      blocks=DEFAULT_BLOCKS, bins=DEFAULT_BINS, mask=None):
    """Lab blocks 1D descriptor of the median-filtered RGB image."""
    return baseline_descriptor(
        median_filter(img_rgb, ksize=ksize),
        blocks=blocks, bins=bins, mask=mask)


def canonical_method_name(name):
    """Resolve a W3 method name or alias to its canonical registry name."""
    if name in METHODS_W3_NAMES:
        return name
    if isinstance(name, str):
        canonical = _METHOD_ALIASES.get(name.strip().lower())
        if canonical is not None:
            return canonical
    raise ValueError(f"Unknown W3 method: {name!r}")


def build_descriptor(blocks=DEFAULT_BLOCKS, bins=DEFAULT_BINS):
    """Descriptor-only function ``fn(rgb, mask=None)`` with no filtering.

    This is what must be used for the museum / reference images, which are
    never filtered.
    """
    def descriptor(img_rgb, mask=None):
        return baseline_descriptor(
            img_rgb, blocks=blocks, bins=bins, mask=mask)
    return descriptor


def build_query_filter(name, ksize=DEFAULT_MEDIAN_KSIZE,
                       wavelet=DEFAULT_WAVELET,
                       wavelet_level=DEFAULT_WAVELET_LEVEL):
    """Image filter for a method, or None for the unfiltered baseline."""
    filt = _METHOD_FILTERS[canonical_method_name(name)]
    if filt is None:
        return None
    return build_image_filter(filt, ksize=ksize, wavelet=wavelet,
                              wavelet_level=wavelet_level)


def build_method(name, ksize=DEFAULT_MEDIAN_KSIZE, blocks=DEFAULT_BLOCKS,
                 bins=DEFAULT_BINS, wavelet=DEFAULT_WAVELET,
                 wavelet_level=DEFAULT_WAVELET_LEVEL):
    """Return ``fn(rgb, mask=None)`` for a W3 method, including its filter.

    The returned callable applies the method's query filter (median, average,
    morphology, wavelet, ...) before the descriptor; the baseline applies no
    filter. Use ``build_descriptor`` for reference images so they are never
    filtered.
    """
    descriptor = build_descriptor(blocks=blocks, bins=bins)
    query_filter = build_query_filter(
        name, ksize=ksize, wavelet=wavelet, wavelet_level=wavelet_level)
    if query_filter is None:
        return descriptor
    return lambda img_rgb, mask=None: descriptor(
        query_filter(img_rgb), mask=mask)


class _MethodRegistry(dict):
    """Name -> method mapping that also resolves the aliases above."""

    def __getitem__(self, name):
        return super().__getitem__(canonical_method_name(name))

    def __contains__(self, name):
        try:
            return super().__contains__(canonical_method_name(name))
        except (ValueError, TypeError):
            return False

    def get(self, name, default=None):
        try:
            return self[name]
        except (ValueError, KeyError, TypeError):
            return default


METHODS_W3 = _MethodRegistry(
    {name: build_method(name) for name in METHODS_W3_NAMES})
METHODS_W3_COLOR_SPACES = dict.fromkeys(METHODS_W3_NAMES, "Lab")


def compute_descriptors_w3(dataset, ksize=DEFAULT_MEDIAN_KSIZE,
                           blocks=DEFAULT_BLOCKS, bins=DEFAULT_BINS,
                           wavelet=DEFAULT_WAVELET,
                           wavelet_level=DEFAULT_WAVELET_LEVEL):
    """Descriptors for every W3 method, in METHODS_W3_NAMES order.

    dataset.images are assumed to be RGB query images, so each method applies
    its own query filter. For reference / museum images use
    ``compute_baseline_descriptors_w3`` instead.
    """
    functions = {
        name: build_method(name, ksize=ksize, blocks=blocks, bins=bins,
                           wavelet=wavelet, wavelet_level=wavelet_level)
        for name in METHODS_W3_NAMES
    }
    return [
        [functions[name](np.asarray(image)) for image in dataset.images]
        for name in METHODS_W3_NAMES
    ]


def compute_baseline_descriptors_w3(dataset, blocks=DEFAULT_BLOCKS,
                                    bins=DEFAULT_BINS):
    """Unfiltered descriptors for reference / museum dataset images."""
    descriptor = build_descriptor(blocks=blocks, bins=bins)
    return [descriptor(np.asarray(image)) for image in dataset.images]


def load_descriptors_w3(path, method=None, query=False,
                        ksize=DEFAULT_MEDIAN_KSIZE, blocks=DEFAULT_BLOCKS,
                        bins=DEFAULT_BINS, masks=None,
                        wavelet=DEFAULT_WAVELET,
                        wavelet_level=DEFAULT_WAVELET_LEVEL):
    """Descriptors from a folder in numeric order.

    query=True applies the method's filter; query=False always uses the
    unfiltered descriptor (reference images). With method=None one list per
    method is returned, in METHODS_W3_NAMES order.
    """
    ids, paths = _ordered_image_paths(path)
    if masks is not None and len(masks) != len(paths):
        raise ValueError(f"Expected {len(paths)} masks, got {len(masks)}")
    names = (list(METHODS_W3_NAMES) if method is None
             else [canonical_method_name(method)])
    if query:
        functions = {
            name: build_method(name, ksize=ksize, blocks=blocks, bins=bins,
                               wavelet=wavelet, wavelet_level=wavelet_level)
            for name in names
        }
    else:
        descriptor = build_descriptor(blocks=blocks, bins=bins)
        functions = dict.fromkeys(names, descriptor)
    descriptors = [[] for _ in names]
    for index, image_path in enumerate(paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = None if masks is None else masks[index]
        for method_index, name in enumerate(names):
            descriptors[method_index].append(functions[name](rgb, mask=mask))
    if method is not None:
        return ids, descriptors[0]
    return ids, descriptors


def load_museum_descriptors_w3(path, blocks=DEFAULT_BLOCKS,
                               bins=DEFAULT_BINS, image_filter=None):
    """Reference descriptors, unfiltered unless ``image_filter`` is given.

    The default (``image_filter=None``) preserves the Task 1 query-only
    contract. Passing a filter computes the same preprocessing on the museum
    images, which is the symmetric variant.
    """
    if image_filter is None:
        return load_descriptors_w3(path, method=BASELINE_METHOD, query=False,
                                   blocks=blocks, bins=bins)
    return load_preprocessed_descriptors_w3(
        path, image_filter, blocks=blocks, bins=bins)


def load_query_descriptors_w3(path, method, ksize=DEFAULT_MEDIAN_KSIZE,
                              blocks=DEFAULT_BLOCKS, bins=DEFAULT_BINS,
                              wavelet=DEFAULT_WAVELET,
                              wavelet_level=DEFAULT_WAVELET_LEVEL):
    """Query descriptors: the method's filter is applied here."""
    return load_descriptors_w3(path, method=method, query=True, ksize=ksize,
                               blocks=blocks, bins=bins, wavelet=wavelet,
                               wavelet_level=wavelet_level)


def load_preprocessed_descriptors_w3(path, image_filter,
                                     blocks=DEFAULT_BLOCKS,
                                     bins=DEFAULT_BINS):
    """Descriptors after an arbitrary ``fn(rgb) -> rgb`` preprocessing.

    Used for query images and, in the symmetric variant, for reference images
    as well.
    """
    ids, paths = _ordered_image_paths(path)
    descriptor = build_descriptor(blocks=blocks, bins=bins)
    descriptors = [
        descriptor(image_filter(_read_rgb(image_path)))
        for image_path in paths
    ]
    return ids, descriptors


def _check_filter_scope(filter_scope):
    assert filter_scope in FILTER_SCOPES, (
        f"filter_scope must be one of {FILTER_SCOPES}, got {filter_scope!r}")
    return filter_scope


def _resolve_hue_reference(hue_reference, museum_ids, museum_images,
                           saturation_threshold=40):
    """Load (or build) the per-museum hue references for hue-invariant mode.

    ``hue_reference`` may be a path to a cached ``.npz`` (loaded when it
    exists, refreshed from ``museum_images`` otherwise) or ``None`` to build
    the references in memory without touching the disk.
    """
    from color_reference import (
        DEFAULT_SATURATION_THRESHOLD,
        HueReference,
        build_hue_references_from_images,
    )

    threshold = (DEFAULT_SATURATION_THRESHOLD if saturation_threshold is None
                 else saturation_threshold)
    if hue_reference is not None:
        path = Path(hue_reference)
        if path.exists():
            return HueReference.load(path).reorder(museum_ids)
        reference = build_hue_references_from_images(
            museum_ids, museum_images, saturation_threshold=threshold)
        reference.save(path)
        return reference
    return build_hue_references_from_images(
        museum_ids, museum_images, saturation_threshold=threshold)


def evaluate_folders_w3(museum_path, queries_path, method,
                        distance_name=DEFAULT_DISTANCE, top_k=5,
                        ground_truth_path=None, ksize=DEFAULT_MEDIAN_KSIZE,
                        blocks=DEFAULT_BLOCKS, bins=DEFAULT_BINS,
                        image_filter=None,
                        filter_scope=DEFAULT_FILTER_SCOPE,
                        wavelet=DEFAULT_WAVELET,
                        wavelet_level=DEFAULT_WAVELET_LEVEL,
                        hue_invariant=False, hue_reference=None,
                        hue_alpha=DEFAULT_HUE_ALPHA,
                        hue_temperature=DEFAULT_HUE_TEMPERATURE):
    """Retrieval with a W3 method; print mAP if a ground truth is given.

    With filter_scope="query" (default) only the queries are filtered, which
    is the Task 1 contract. With filter_scope="both" the same filter is also
    applied to the museum/reference images. ``image_filter`` overrides the
    method's own filter with any preprocessing function.

    With hue_invariant=True the HSV hue-invariance approach is enabled on top
    of the method's filter: a global hue scale is estimated per
    (query, museum candidate) pair and the raw LAB distance is blended with
    the hue-normalized LAB distance (``hue_alpha`` weights the correction,
    ``hue_temperature`` the per-candidate confidence). ``hue_reference`` is
    an optional cache path for the museum hue histograms.
    """
    from k_similarity import DISTANCE_FUNCTIONS, retrieve_all_queries

    _check_filter_scope(filter_scope)
    name = canonical_method_name(method)
    query_filter = image_filter
    if query_filter is None:
        query_filter = build_query_filter(
            name, ksize=ksize, wavelet=wavelet, wavelet_level=wavelet_level)
    if not hue_invariant:
        reference_filter = query_filter if filter_scope == "both" else None
        museum_ids, museum = load_museum_descriptors_w3(
            museum_path, blocks=blocks, bins=bins,
            image_filter=reference_filter)
        if query_filter is None:
            query_ids, queries = load_query_descriptors_w3(
                queries_path, name, ksize=ksize, blocks=blocks, bins=bins,
                wavelet=wavelet, wavelet_level=wavelet_level)
        else:
            query_ids, queries = load_preprocessed_descriptors_w3(
                queries_path, query_filter, blocks=blocks, bins=bins)
        ground_truth = _load_ground_truth(ground_truth_path, query_ids)
        rankings = retrieve_all_queries(
            queries, museum, DISTANCE_FUNCTIONS[distance_name], top_k=top_k)
        return _report_rankings(name, distance_name, museum_ids, query_ids,
                                len(museum[0]), rankings, top_k, ground_truth)

    from color_reference import rank_with_hue_references

    museum_ids, museum_paths = _ordered_image_paths(museum_path)
    query_ids, query_paths = _ordered_image_paths(queries_path)
    raw_museum = [_read_rgb(path) for path in museum_paths]
    raw_queries = [_read_rgb(path) for path in query_paths]
    queries = [query_filter(image) if query_filter is not None else image
               for image in raw_queries]
    reference_images = (
        [query_filter(image) if query_filter is not None else image
         for image in raw_museum]
        if filter_scope == "both" else raw_museum)
    descriptor_fn = lambda image: baseline_descriptor(
        image, blocks=blocks, bins=bins)
    museum = [descriptor_fn(image) for image in reference_images]
    reference = _resolve_hue_reference(hue_reference, museum_ids,
                                       reference_images)
    museum_array = np.asarray(museum, dtype=np.float64)
    rankings = []
    for image in queries:
        result = rank_with_hue_references(
            image, museum_array, reference, descriptor_fn,
            alpha=hue_alpha, temperature=hue_temperature)
        ranking = [(index, 0.0) for index in result.ranking[:top_k]]
        rankings.append(ranking)
    ground_truth = _load_ground_truth(ground_truth_path, query_ids)
    return _report_rankings(f"{name} + hue-invariant", distance_name,
                            museum_ids, query_ids, len(museum[0]), rankings,
                            top_k, ground_truth)


def evaluate_all_folders_w3(museum_path, queries_path,
                            distance_name=DEFAULT_DISTANCE, top_k=5,
                            ground_truth_path=None,
                            ksize=DEFAULT_MEDIAN_KSIZE, blocks=DEFAULT_BLOCKS,
                            bins=DEFAULT_BINS,
                            filter_scope=DEFAULT_FILTER_SCOPE,
                            wavelet=DEFAULT_WAVELET,
                            wavelet_level=DEFAULT_WAVELET_LEVEL):
    """Run every registered W3 method in one pass."""
    from k_similarity import DISTANCE_FUNCTIONS, retrieve_all_queries

    _check_filter_scope(filter_scope)
    museum_ids, raw_museum = load_museum_descriptors_w3(
        museum_path, blocks=blocks, bins=bins)
    query_ids, queries = load_descriptors_w3(
        queries_path, method=None, query=True, ksize=ksize, blocks=blocks,
        bins=bins, wavelet=wavelet, wavelet_level=wavelet_level)
    ground_truth = _load_ground_truth(ground_truth_path, query_ids)
    results = {}
    for index, name in enumerate(METHODS_W3_NAMES):
        query_filter = build_query_filter(
            name, ksize=ksize, wavelet=wavelet, wavelet_level=wavelet_level)
        if filter_scope == "both" and query_filter is not None:
            _, museum = load_museum_descriptors_w3(
                museum_path, blocks=blocks, bins=bins,
                image_filter=query_filter)
        else:
            museum = raw_museum
        rankings = retrieve_all_queries(
            queries[index], museum, DISTANCE_FUNCTIONS[distance_name],
            top_k=top_k)
        results[name] = _report_rankings(
            name, distance_name, museum_ids, query_ids, len(museum[0]),
            rankings, top_k, ground_truth)
    return results


def default_filter_specs(median_sizes=None, morph_sizes=None,
                         chain_sizes=None, wavelet_levels=None):
    """Filter grid for the sweep: baseline, average, median, Gaussian,
    morphology (open/close), chained open+close / close+open and low-pass
    wavelet levels."""
    median_sizes = (MEDIAN_SWEEP_SIZES if median_sizes is None
                    else tuple(median_sizes))
    morph_sizes = (MORPH_SWEEP_SIZES if morph_sizes is None
                   else tuple(morph_sizes))
    chain_sizes = (CHAIN_SWEEP_SIZES if chain_sizes is None
                   else tuple(chain_sizes))
    wavelet_levels = (WAVELET_SWEEP_LEVELS if wavelet_levels is None
                      else tuple(wavelet_levels))
    specs = [("none", 1)]
    specs += [("average", ksize) for ksize in AVERAGE_SWEEP_SIZES]
    specs += [("median", ksize) for ksize in median_sizes]
    specs += [("gaussian", ksize) for ksize in GAUSSIAN_SWEEP_SIZES]
    specs += [(operation, ksize) for operation in MORPH_OPERATIONS
              for ksize in morph_sizes]
    specs += [("open+close", ksize) for ksize in chain_sizes]
    specs += [("close+open", ksize) for ksize in chain_sizes]
    specs += [("wavelet_lowpass", level) for level in wavelet_levels]
    return specs


def evaluate_filter_sweep(museum_path, queries_path, ground_truth_path,
                          filter_specs=None, distance_name=DEFAULT_DISTANCE,
                          top_k=5, blocks=DEFAULT_BLOCKS, bins=DEFAULT_BINS,
                          stop_on_drop=False,
                          filter_scope=DEFAULT_FILTER_SCOPE):
    """Compare filters and print mAP@1 / mAP@top_k.

    With filter_scope="query" (default) only the queries are preprocessed.
    With filter_scope="both" the same filter is applied to the museum
    references as well. With stop_on_drop, the sweep stops after the first
    filter whose mAP@top_k is lower than the best seen so far.
    """
    from k_similarity import (
        DISTANCE_FUNCTIONS,
        mean_average_precision_at_k,
        retrieve_all_queries,
    )

    if ground_truth_path is None:
        raise ValueError("A ground-truth pickle is required for the sweep")
    _check_filter_scope(filter_scope)
    museum_ids, museum_paths = _ordered_image_paths(museum_path)
    query_ids, query_paths = _ordered_image_paths(queries_path)
    ground_truth = _load_ground_truth(ground_truth_path, query_ids)
    raw_museum = [_read_rgb(path) for path in museum_paths]
    raw_queries = [_read_rgb(path) for path in query_paths]
    raw_museum_descriptors = [
        baseline_descriptor(image, blocks=blocks, bins=bins)
        for image in raw_museum
    ]
    fixed_museum = (raw_museum_descriptors if filter_scope == "query" else None)
    specs = default_filter_specs() if filter_specs is None else filter_specs
    results = []
    best = None
    print(f"Filter scope: {filter_scope}")
    print(f"{'Filter':<14} {'k':>3} {'mAP@1':>8} {f'mAP@{top_k}':>8}")
    for name, ksize in specs:
        if name in WAVELET_FILTER_NAMES:
            image_filter = build_image_filter(
                name, wavelet_level=int(ksize))
        else:
            image_filter = build_image_filter(name, ksize)
        if filter_scope == "both":
            if name == "none":
                museum = raw_museum_descriptors
            else:
                museum = [
                    baseline_descriptor(
                        image_filter(image), blocks=blocks, bins=bins)
                    for image in raw_museum
                ]
        else:
            museum = fixed_museum
        descriptors = [
            baseline_descriptor(image_filter(image), blocks=blocks, bins=bins)
            for image in raw_queries
        ]
        rankings = retrieve_all_queries(
            descriptors, museum, DISTANCE_FUNCTIONS[distance_name],
            top_k=top_k)
        ranked_ids = [[museum_ids[index] for index, _ in row]
                      for row in rankings]
        map_at_1 = mean_average_precision_at_k(ranked_ids, ground_truth, 1)
        map_at_k = mean_average_precision_at_k(ranked_ids, ground_truth, top_k)
        results.append(
            {"filter": name, "ksize": ksize, "filter_scope": filter_scope,
             "mAP@1": map_at_1, f"mAP@{top_k}": map_at_k})
        print(f"{name:<14} {ksize:>3d} {map_at_1:>8.4f} {map_at_k:>8.4f}")
        if stop_on_drop and best is not None and map_at_k < best:
            break
        if best is None or map_at_k > best:
            best = map_at_k
    return results


def _perfect_square(value):
    try:
        total = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"blocks must be an integer, got {value!r}") from error
    if total < 1 or math.isqrt(total) ** 2 != total:
        raise argparse.ArgumentTypeError(
            "blocks must be a positive perfect square (e.g. 1, 4, 9, 16, 25)")
    return total


def _odd_kernel(value):
    try:
        ksize = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"median size must be an integer, got {value!r}") from error
    if ksize < 3 or ksize % 2 == 0:
        raise argparse.ArgumentTypeError(
            "median size must be an odd integer >= 3 (e.g. 3, 5, 7)")
    return ksize


def _positive_bins(value):
    try:
        bins = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"bins must be an integer, got {value!r}") from error
    if bins < 1:
        raise argparse.ArgumentTypeError("bins must be a positive integer")
    return bins


def build_parser():
    try:
        from k_similarity import DISTANCE_FUNCTIONS
        distances = sorted(DISTANCE_FUNCTIONS)
    except Exception:
        distances = [DEFAULT_DISTANCE]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="List available W3 methods and exit")
    parser.add_argument("--sweep", action="store_true",
                        help="Compare the registered filters on the queries "
                             "(needs --ground-truth)")
    parser.add_argument("--median-sizes", nargs="+", type=_odd_kernel,
                        default=None,
                        help="Median sizes for --sweep (default 3..21 odd)")
    parser.add_argument("--morph-sizes", nargs="+", type=_odd_kernel,
                        default=None,
                        help="Morphology sizes for --sweep (default 3 5)")
    parser.add_argument("--chain-sizes", nargs="+", type=_odd_kernel,
                        default=None,
                        help="Open+close / close+open sizes for --sweep "
                             "(default 3 5 7 9 11)")
    parser.add_argument("--wavelet-levels", nargs="+", type=int,
                        default=None,
                        help="Wavelet low-pass levels for --sweep "
                             "(default 1 2 3)")
    parser.add_argument("--stop-on-drop", action="store_true",
                        help="--sweep: stop after the first mAP drop")
    parser.add_argument("--method", choices=METHOD_CHOICES, default=None,
                        help="W3 method for retrieval, or 'all' to evaluate "
                             "every registered method")
    parser.add_argument("--pre-filter", choices=IMAGE_FILTERS, default=None,
                        help="Override the query preprocessing filter "
                             "(single-method retrieval only)")
    parser.add_argument("--filter-scope", choices=FILTER_SCOPES,
                        default=DEFAULT_FILTER_SCOPE,
                        help="Apply the filter to queries only (default) or "
                             "to both queries and museum references")
    parser.add_argument("--museum", type=Path, help="Museum image folder")
    parser.add_argument("--queries", type=Path, help="Query image folder")
    parser.add_argument("--ground-truth", type=Path, default=None,
                        help="Trusted correspondence pickle (IDs, query order)")
    parser.add_argument("--distance", choices=distances,
                        default=DEFAULT_DISTANCE)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--kernel", "--median-size", dest="kernel",
                        type=_odd_kernel, default=None,
                        help="REQUIRED (except with --list): spatial filter "
                             "size, odd and >= 3 (e.g. 7 means 7x7). There "
                             "is no default on purpose: every reported "
                             "number must state its kernel.")
    parser.add_argument("--wavelet", default=DEFAULT_WAVELET,
                        help="Wavelet family for the wavelet method "
                             f"(default {DEFAULT_WAVELET})")
    parser.add_argument("--wavelet-level", type=int,
                        default=DEFAULT_WAVELET_LEVEL,
                        help="Wavelet low-pass decomposition level "
                             f"(default {DEFAULT_WAVELET_LEVEL})")
    parser.add_argument("--hue-invariant", action="store_true",
                        help="Enable the HSV hue-invariance approach on top "
                             "of the method's filter")
    parser.add_argument("--hue-reference", type=Path, default=None,
                        help="Cache path for the museum hue histograms "
                             "(built in memory when omitted)")
    parser.add_argument("--hue-alpha", type=float, default=DEFAULT_HUE_ALPHA,
                        help="Blend weight of the hue-normalized distance "
                             f"(default {DEFAULT_HUE_ALPHA})")
    parser.add_argument("--hue-temperature", type=float,
                        default=DEFAULT_HUE_TEMPERATURE,
                        help="Confidence temperature for hue-invariant "
                             "ranking "
                             f"(default {DEFAULT_HUE_TEMPERATURE})")
    parser.add_argument("--blocks", type=_perfect_square,
                        default=DEFAULT_BLOCKS,
                        help="Total spatial cells, a perfect square "
                             "(default 25 -> 5x5)")
    parser.add_argument("--bins", type=_positive_bins, default=DEFAULT_BINS,
                        help="Histogram bins per channel (default 32)")
    return parser


def _warn_without_hue_invariant():
    """Loud warning: plain numbers are not hue-invariant numbers."""
    print("WARNING: --hue-invariant NOT set. This run does NOT apply hue "
          "correction, so its numbers are NOT comparable with any figure "
          "titled 'Applying hue-invariance'. Re-run with --hue-invariant "
          "(and --hue-reference) for hue-invariant numbers.",
          file=sys.stderr, flush=True)


def main(args=None):
    parser = build_parser()
    opts = parser.parse_args(args)
    if opts.list:
        for name in METHODS_W3_NAMES:
            print(name)
        return
    if opts.kernel is None:
        parser.error("--kernel is required: pass an explicit odd size "
                     ">= 3 (e.g. --kernel 7). Silent defaults caused "
                     "3x3 numbers to be mistaken for best-kernel numbers.")
    if not opts.hue_invariant and (opts.method is not None or opts.sweep):
        _warn_without_hue_invariant()
    if opts.sweep:
        if not opts.museum or not opts.queries:
            parser.error("--sweep needs --museum and --queries")
        if not opts.ground_truth:
            parser.error("--sweep needs --ground-truth")
        specs = default_filter_specs(
            opts.median_sizes, opts.morph_sizes, opts.chain_sizes,
            opts.wavelet_levels)
        evaluate_filter_sweep(
            opts.museum, opts.queries, opts.ground_truth, specs,
            opts.distance, opts.top_k, opts.blocks, opts.bins,
            opts.stop_on_drop, opts.filter_scope)
        return
    if not opts.museum or not opts.queries:
        parser.error("--museum and --queries are required (or use --list)")
    if opts.method is None:
        parser.error("--method is required: one of "
                     + ", ".join(METHOD_CHOICES))
    if opts.method == METHOD_ALL:
        if opts.pre_filter is not None:
            parser.error("--pre-filter cannot be combined with --method all")
        if opts.hue_invariant:
            parser.error("--hue-invariant cannot be combined with --method "
                         "all (it would be silently ignored): run one "
                         "--method at a time with --kernel K "
                         "--hue-invariant instead.")
        print("NOTE: --method all applies the same --kernel "
              f"({opts.kernel}x{opts.kernel}) to every method; "
              "per-filter best kernels need single-method runs.",
              file=sys.stderr, flush=True)
        evaluate_all_folders_w3(
            opts.museum, opts.queries, opts.distance, opts.top_k,
            opts.ground_truth, opts.kernel, opts.blocks, opts.bins,
            opts.filter_scope, wavelet=opts.wavelet,
            wavelet_level=opts.wavelet_level)
        return
    image_filter = (None if opts.pre_filter is None else
                    build_image_filter(opts.pre_filter, opts.kernel,
                                       wavelet=opts.wavelet,
                                       wavelet_level=opts.wavelet_level))
    evaluate_folders_w3(
        opts.museum, opts.queries, opts.method, opts.distance, opts.top_k,
        opts.ground_truth, opts.kernel, opts.blocks, opts.bins,
        image_filter=image_filter, filter_scope=opts.filter_scope,
        wavelet=opts.wavelet, wavelet_level=opts.wavelet_level,
        hue_invariant=opts.hue_invariant, hue_reference=opts.hue_reference,
        hue_alpha=opts.hue_alpha, hue_temperature=opts.hue_temperature)


if __name__ == "__main__":
    main()
