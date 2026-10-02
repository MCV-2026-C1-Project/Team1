import numpy as np
from similarity_functions import euclidean_distance, l1_distance, chi_square_distance, histogram_intersection, hellinger_kernel
from utils import Dataset
from main import compute_descriptors

SIMILARITY_FUNCTIONS = (histogram_intersection, hellinger_kernel)

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

# Load datasets
museum = Dataset("data/BBDD")
queries = Dataset("data/qsd1_w1")

print("Museum images:", len(museum.images))
print("Query images:", len(queries.images))


descriptor_names = ["Grayscale", "HSV", "RGB"]

# Compute descriptors
museum_descriptors = compute_descriptors(museum)
query_descriptors = compute_descriptors(queries)

ground_truth = queries.correspondances

if ground_truth is None:
    raise ValueError("Ground-truth correspondences were not loaded.")

print("Ground-truth queries:", len(ground_truth))

# Select descriptor method
for method_idx, method_name in enumerate(descriptor_names):
    print(f"\nEvaluating descriptor method: {method_name}")

    museum_descriptor = museum_descriptors[method_idx]
    query_descriptor = query_descriptors[method_idx]

    # Retrieve ranked museum images
    rankings_eucdis = retrieve_all_queries(query_descriptor, museum_descriptor, euclidean_distance, top_k=5)
    rankings_l1 = retrieve_all_queries(query_descriptor, museum_descriptor, l1_distance, top_k=5)
    rankings_chi_square = retrieve_all_queries(query_descriptor, museum_descriptor, chi_square_distance, top_k=5)
    rankings_histogram_intersection = retrieve_all_queries(query_descriptor, museum_descriptor, histogram_intersection, top_k=5)
    rankings_hellinger = retrieve_all_queries(query_descriptor, museum_descriptor, hellinger_kernel, top_k=5)

    # Evaluate retrieval performance for each distance function
    max_k = 5   
    rankings = [rankings_eucdis, rankings_l1, rankings_chi_square, rankings_histogram_intersection, rankings_hellinger]
    for rankings, distance_function in zip(rankings, ["Euclidean", "L1", "Chi-square", "Histogram Intersection", "Hellinger"]):
        print(f"\nEvaluating distance function: {distance_function}")
        ranked_indices =  [[museum_idx for museum_idx, distance in ranking]for ranking in rankings]
        evaluation_results = evaluate_retrieval(ranked_indices, ground_truth, max_k)

        for k in range(1, max_k + 1):
            if k == 1 or k == max_k:
                mAP = evaluation_results[k]["mAP"]
                print(f"mAP@{k}: {mAP:.4f}")
            #print(f"AP@{k} for each query:")
            #print(evaluation_results[k]["AP_per_query"])

        print("Top-1 per query:", [r[0][0] for r in rankings]) # list of top-1 museum indices for each query