import numpy as np
import heapq

def linear_scan_knn(dataset_dict, query_point, k=5):
    """
    Performs a brute-force K-Nearest Neighbor search by scanning the entire dataset.
    
    :param dataset_dict: Dictionary mapping point IDs to high-dimensional numpy arrays.
    :param query_point: 1D numpy array representing the query vector.
    :param k: The number of nearest neighbors to retrieve.
    :return: A list of tuples (distance, point_id, point_vector) representing the k nearest neighbors.
    """
    query_point = np.array(query_point, dtype=np.float32)
    
    # Priority queue (min-heap) to keep track of the top k closest points
    nearest_neighbors = []
    
    for point_id, point_vector in dataset_dict.items():
        dist = np.linalg.norm(query_point - point_vector)
        
        # We use negative distance so the heap acts as a max-heap (keeping the largest at index 0)
        if len(nearest_neighbors) < k:
            heapq.heappush(nearest_neighbors, (-dist, point_id, point_vector))
        else:
            kth_farthest_dist = -nearest_neighbors[0][0]
            if dist < kth_farthest_dist:
                heapq.heapreplace(nearest_neighbors, (-dist, point_id, point_vector))
                
    # Format and sort results
    results = [(-dist, pid, vec) for dist, pid, vec in nearest_neighbors]
    results.sort(key=lambda x: x[0])
    
    return results