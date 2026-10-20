"""Methods_W3: Week 3, Task 1 (noise filtering) and the unfiltered baseline.

Task 1 asks for a linear or non-linear filter for an unknown noise model,
where only some queries are noisy. This module implements the non-linear
median filter and applies it **only to the queries**; the museum/reference
images (BBDD) are never filtered.

Two retrieval methods share the Week 2 "Lab blocks 1D" descriptor:

    - "Lab blocks 1D"         baseline: the query is used as it is.
    - "Lab blocks 1D median"  the query is median-filtered before the
                              descriptor.

Both use the W2 configuration: CIE-Lab, 32 bins per channel, 5x5 = 25
spatial cells, L1-normalized histograms and the L1 distance.

Run as a script (argparse selects the action):
    python methods_w3.py --list
    python methods_w3.py --museum <dir> --queries <dir> --method NAME
    python methods_w3.py --museum <dir> --queries <dir> --method all \
        --ground-truth <pkl>
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from methods_w2 import (
    _load_ground_truth,
    _ordered_image_paths,
    _report_rankings,
    blocks_grid,
    compute_lab_block_histogram,
)

DEFAULT_BINS = 32
DEFAULT_BLOCKS = 25
DEFAULT_MEDIAN_KSIZE = 3
DEFAULT_DISTANCE = "L1"

BASELINE_METHOD = "Lab blocks 1D"
MEDIAN_METHOD = "Lab blocks 1D median"
METHODS_W3_NAMES = (BASELINE_METHOD, MEDIAN_METHOD)

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
}


def _check_median_ksize(ksize):
    assert (isinstance(ksize, int) and not isinstance(ksize, bool)
            and ksize >= 3 and ksize % 2 == 1), (
        f"median ksize must be an odd integer >= 3, got {ksize!r}")
    return ksize


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


def build_query_filter(name, ksize=DEFAULT_MEDIAN_KSIZE):
    """Image filter for a method, or None for the unfiltered baseline."""
    if canonical_method_name(name) == MEDIAN_METHOD:
        return lambda img_rgb: median_filter(img_rgb, ksize=ksize)
    return None


def build_method(name, ksize=DEFAULT_MEDIAN_KSIZE, blocks=DEFAULT_BLOCKS,
                 bins=DEFAULT_BINS):
    """Return ``fn(rgb, mask=None)`` for a W3 method, including its filter.

    The returned callable applies the method's query filter (median for
    "Lab blocks 1D median", none for the baseline). Use ``build_descriptor``
    for reference images so they are never filtered.
    """
    descriptor = build_descriptor(blocks=blocks, bins=bins)
    query_filter = build_query_filter(name, ksize=ksize)
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
                           blocks=DEFAULT_BLOCKS, bins=DEFAULT_BINS):
    """Descriptors for every W3 method, in METHODS_W3_NAMES order.

    dataset.images are assumed to be RGB query images, so the median method
    filters them. For reference / museum images use
    ``compute_baseline_descriptors_w3`` instead.
    """
    functions = {
        name: build_method(name, ksize=ksize, blocks=blocks, bins=bins)
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
                        bins=DEFAULT_BINS, masks=None):
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
            name: build_method(name, ksize=ksize, blocks=blocks, bins=bins)
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
                               bins=DEFAULT_BINS):
    """Reference descriptors: always unfiltered."""
    return load_descriptors_w3(path, method=BASELINE_METHOD, query=False,
                               blocks=blocks, bins=bins)


def load_query_descriptors_w3(path, method, ksize=DEFAULT_MEDIAN_KSIZE,
                              blocks=DEFAULT_BLOCKS, bins=DEFAULT_BINS):
    """Query descriptors: the method's filter is applied here."""
    return load_descriptors_w3(path, method=method, query=True, ksize=ksize,
                               blocks=blocks, bins=bins)


def evaluate_folders_w3(museum_path, queries_path, method,
                        distance_name=DEFAULT_DISTANCE, top_k=5,
                        ground_truth_path=None, ksize=DEFAULT_MEDIAN_KSIZE,
                        blocks=DEFAULT_BLOCKS, bins=DEFAULT_BINS):
    """Retrieval with a W3 method; print mAP if a ground truth is given.

    Museum descriptors are always computed without filtering; only the query
    descriptors go through the method's filter (median for the median method).
    """
    from k_similarity import DISTANCE_FUNCTIONS, retrieve_all_queries

    name = canonical_method_name(method)
    museum_ids, museum = load_museum_descriptors_w3(
        museum_path, blocks=blocks, bins=bins)
    query_ids, queries = load_query_descriptors_w3(
        queries_path, name, ksize=ksize, blocks=blocks, bins=bins)
    ground_truth = _load_ground_truth(ground_truth_path, query_ids)
    rankings = retrieve_all_queries(
        queries, museum, DISTANCE_FUNCTIONS[distance_name], top_k=top_k)
    return _report_rankings(name, distance_name, museum_ids, query_ids,
                            len(museum[0]), rankings, top_k, ground_truth)


def evaluate_all_folders_w3(museum_path, queries_path,
                            distance_name=DEFAULT_DISTANCE, top_k=5,
                            ground_truth_path=None,
                            ksize=DEFAULT_MEDIAN_KSIZE, blocks=DEFAULT_BLOCKS,
                            bins=DEFAULT_BINS):
    """Run the baseline and the median method in one pass."""
    from k_similarity import DISTANCE_FUNCTIONS, retrieve_all_queries

    museum_ids, museum = load_museum_descriptors_w3(
        museum_path, blocks=blocks, bins=bins)
    query_ids, queries = load_descriptors_w3(
        queries_path, method=None, query=True, ksize=ksize, blocks=blocks,
        bins=bins)
    ground_truth = _load_ground_truth(ground_truth_path, query_ids)
    results = {}
    for index, name in enumerate(METHODS_W3_NAMES):
        rankings = retrieve_all_queries(
            queries[index], museum, DISTANCE_FUNCTIONS[distance_name],
            top_k=top_k)
        results[name] = _report_rankings(
            name, distance_name, museum_ids, query_ids, len(museum[0]),
            rankings, top_k, ground_truth)
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
    parser.add_argument("--method", choices=METHOD_CHOICES, default=None,
                        help="W3 method for retrieval, or 'all' to evaluate "
                             "the baseline and the median filter")
    parser.add_argument("--museum", type=Path, help="Museum image folder")
    parser.add_argument("--queries", type=Path, help="Query image folder")
    parser.add_argument("--ground-truth", type=Path, default=None,
                        help="Trusted correspondence pickle (IDs, query order)")
    parser.add_argument("--distance", choices=distances,
                        default=DEFAULT_DISTANCE)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--kernel", "--median-size", dest="kernel",
                        type=_odd_kernel, default=DEFAULT_MEDIAN_KSIZE,
                        help="Median filter size, odd and >= 3 (default 3)")
    parser.add_argument("--blocks", type=_perfect_square,
                        default=DEFAULT_BLOCKS,
                        help="Total spatial cells, a perfect square "
                             "(default 25 -> 5x5)")
    parser.add_argument("--bins", type=_positive_bins, default=DEFAULT_BINS,
                        help="Histogram bins per channel (default 32)")
    return parser


def main(args=None):
    parser = build_parser()
    opts = parser.parse_args(args)
    if opts.list:
        for name in METHODS_W3_NAMES:
            print(name)
        return
    if not opts.museum or not opts.queries:
        parser.error("--museum and --queries are required (or use --list)")
    if opts.method is None:
        parser.error("--method is required: one of "
                     + ", ".join(METHOD_CHOICES))
    if opts.method == METHOD_ALL:
        evaluate_all_folders_w3(
            opts.museum, opts.queries, opts.distance, opts.top_k,
            opts.ground_truth, opts.kernel, opts.blocks, opts.bins)
        return
    evaluate_folders_w3(
        opts.museum, opts.queries, opts.method, opts.distance, opts.top_k,
        opts.ground_truth, opts.kernel, opts.blocks, opts.bins)


if __name__ == "__main__":
    main()
