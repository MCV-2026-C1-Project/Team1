import numpy as np
from similarity_functions import (euclidean_distance, l1_distance, chi_square_distance, histogram_intersection, hellinger_kernel, cosine_similarity, correlation_similarity)
from utils import Dataset
from main import compute_descriptors, DESCRIPTOR_NAMES, compute_descriptors_w2, METHODS_W2_NAMES

SIMILARITY_FUNCTIONS = (histogram_intersection, hellinger_kernel, cosine_similarity, correlation_similarity)

DISTANCE_FUNCTIONS = {
    "Euclidean": euclidean_distance,
    "L1": l1_distance,
    "Chi-square": chi_square_distance,
    "Histogram Intersection": histogram_intersection,
    "Hellinger": hellinger_kernel,
    "Cosine": cosine_similarity,
    "Correlation": correlation_similarity,
}

def rank_museum_images(query_descriptor, museum_descriptors, distance_function, top_k=None):
    distances = []
    for museum_idx, museum_descriptor in enumerate(museum_descriptors):
        distance = distance_function(query_descriptor, museum_descriptor)
        distances.append((museum_idx, float(distance)))

    # For similarity functions, we want higher scores to be better
    higher_is_better = distance_function in SIMILARITY_FUNCTIONS
    distances.sort(key=lambda x: x[1], reverse=higher_is_better)
    return distances[:top_k]


def retrieve_all_queries(query_descriptors, museum_descriptors, distance_function, top_k=None): # Retrieve the ranked museum images for every query
    all_rankings = []

    for query_descriptor in query_descriptors:

        ranking = rank_museum_images(query_descriptor, museum_descriptors, distance_function, top_k)
        all_rankings.append(ranking)

    return all_rankings

def average_precision_at_k(ranking, relevant_images, k): # Calculate AP@K for one query.

    relevant_images = set(relevant_images)

    if len(relevant_images) == 0:
        return 0.0

    top_k = ranking[:k]

    num_relevant = 0
    precision_sum = 0.0

    for rank, museum_idx in enumerate(top_k, start=1):
        if museum_idx in relevant_images:
            num_relevant += 1
            precision_at_rank = num_relevant / rank
            precision_sum += precision_at_rank

    # Normalize by the maximum possible relevant results at K
    denominator = min(len(relevant_images), k)

    return precision_sum / denominator

def mean_average_precision_at_k(rankings, ground_truth, k): #Calculate mAP@K over all query images.
    if len(rankings) != len(ground_truth):
        raise ValueError(
            "Number of rankings must match number of queries"
        )

    if len(rankings) == 0:
        return 0.0

    ap_values = []

    for ranking, relevant_images in zip(rankings, ground_truth):

        ap = average_precision_at_k(ranking, relevant_images, k)
        ap_values.append(ap)

    return float(np.mean(ap_values))

def validate_ground_truth(ground_truth, n_museum, n_queries):
    """Minimal ID check: one entry per query, each relevant ID a valid
    museum position in [0, n_museum). Evaluation ranks by position in the
    sorted load order; submission (task4) maps those positions to filename
    IDs, so an out-of-range ID here means a corrupt/mismatched pkl."""
    if len(ground_truth) != n_queries:
        raise ValueError(
            f"Expected {n_queries} ground-truth entries, got {len(ground_truth)}"
        )
    for qi, relevant in enumerate(ground_truth):
        for museum_idx in relevant:
            if type(museum_idx) is not int or not 0 <= museum_idx < n_museum:
                raise ValueError(
                    f"Query {qi}: invalid museum index {museum_idx!r} "
                    f"(museum size {n_museum})"
                )

def evaluate_retrieval(rankings, ground_truth, max_k): # Calculate AP@K for each query and mAP@K for K=1...max_k.
    results = {}

    for k in range(1, max_k + 1):

        ap_per_query = []

        for ranking, relevant_images in zip(rankings, ground_truth):

            ap = average_precision_at_k(ranking, relevant_images, k)
            ap_per_query.append(ap)

        results[k] = {
            "AP_per_query": ap_per_query,
            "mAP": float(np.mean(ap_per_query))
        }

    return results

def main():
    # Load datasets
    museum = Dataset("data/BBDD")
    queries = Dataset("data/qsd1_w1")

    print("Museum images:", len(museum.images))
    print("Query images:", len(queries.images))


    descriptor_names = DESCRIPTOR_NAMES

    # Compute descriptors
    museum_descriptors = compute_descriptors(museum)
    query_descriptors = compute_descriptors(queries)

    ground_truth = queries.correspondances

    if ground_truth is None:
        raise ValueError("Ground-truth correspondences were not loaded.")

    validate_ground_truth(ground_truth, len(museum.images), len(queries.images))

    print("Ground-truth queries:", len(ground_truth))

    max_k = 5
    for method_idx, method_name in enumerate(descriptor_names):
        print(f"\nEvaluating descriptor method: {method_name} [W1]")
        museum_descriptor = museum_descriptors[method_idx]
        query_descriptor = query_descriptors[method_idx]

        for distance_name, distance_function in DISTANCE_FUNCTIONS.items():
            print(f"\nEvaluating distance function: {distance_name}")
            rankings = retrieve_all_queries(query_descriptor, museum_descriptor, distance_function, top_k=max_k)
            ranked_indices = [[museum_idx for museum_idx, distance in ranking] for ranking in rankings]
            evaluation_results = evaluate_retrieval(ranked_indices, ground_truth, max_k)

            for k in range(1, max_k + 1):
                if k == 1 or k == max_k:
                    print(f"mAP@{k}: {evaluation_results[k]['mAP']:.4f}")

            print("Top-1 per query:", [r[0][0] for r in rankings])

    # Week 2: same queries (QSD1-W2 == QSD1-W1), HSV and Lab descriptors.
    # Compare directly against the best W1 (HSV grid + L1, mAP@5 0.82).
    print("\n--- Methods_W2 (HSV 3D/2D and spatial Lab) ---")
    museum_descriptors_w2 = compute_descriptors_w2(museum)
    query_descriptors_w2 = compute_descriptors_w2(queries)
    for method_idx, method_name in enumerate(METHODS_W2_NAMES):
        print(f"\nEvaluating descriptor method: {method_name} [W2]")
        museum_descriptor = museum_descriptors_w2[method_idx]
        query_descriptor = query_descriptors_w2[method_idx]

        for distance_name, distance_function in DISTANCE_FUNCTIONS.items():
            print(f"\nEvaluating distance function: {distance_name}")
            rankings = retrieve_all_queries(query_descriptor, museum_descriptor, distance_function, top_k=max_k)
            ranked_indices = [[museum_idx for museum_idx, distance in ranking] for ranking in rankings]
            evaluation_results = evaluate_retrieval(ranked_indices, ground_truth, max_k)

            for k in range(1, max_k + 1):
                if k == 1 or k == max_k:
                    print(f"mAP@{k}: {evaluation_results[k]['mAP']:.4f}")

            print("Top-1 per query:", [r[0][0] for r in rankings])

if __name__ == "__main__":
    main()
