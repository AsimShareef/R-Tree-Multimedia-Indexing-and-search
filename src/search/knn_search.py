import numpy as np
import heapq

def knn_search(tree, query_point, k=5):
    """
    Performs a K-Nearest Neighbor search on the R-Tree using a Priority Queue.
    
    :param tree: The populated RTree object.
    :param query_point: 1D numpy array representing the query vector.
    :param k: The number of nearest neighbors to retrieve.
    :return: A list of tuples (distance, point_id, point_vector) representing the k nearest neighbors.
    """
    query_point = np.array(query_point, dtype=np.float32)
    
    # Priority queue to explore nodes. 
    # Stores tuples of (min_distance_to_mbr, node_id, node)
    # node_id is used as a tie-breaker to prevent heapq from comparing Node objects
    pq = []
    
    # Max-heap to store the k nearest neighbors found so far.
    # Stores tuples of (-distance, point_id, point_vector)
    # We use negative distance because Python's heapq is a min-heap, 
    # and we want to easily pop the farthest of the k neighbors.
    nearest_neighbors = []
    
    # Start at the root
    if tree.root.mbr is None:
        return [] # Tree is empty
        
    # Push root onto queue
    heapq.heappush(pq, (tree.root.mbr.min_distance(query_point), id(tree.root), tree.root))
    
    while pq:
        min_dist, _, current_node = heapq.heappop(pq)
        
        # Branch Pruning:
        # If the minimum possible distance to this node is greater than the 
        # distance to our current k-th closest neighbor, we can stop exploring this branch.
        if len(nearest_neighbors) == k:
            kth_farthest_dist = -nearest_neighbors[0][0]
            if min_dist > kth_farthest_dist:
                continue # Prune branch
                
        if current_node.is_leaf():
            # If it's a leaf, calculate exact distances to all points inside
            for point_id, point_vector in current_node.entries:
                dist = np.linalg.norm(query_point - point_vector)
                
                if len(nearest_neighbors) < k:
                    heapq.heappush(nearest_neighbors, (-dist, point_id, point_vector))
                else:
                    # If we found a closer point than our current k-th farthest, replace it
                    kth_farthest_dist = -nearest_neighbors[0][0]
                    if dist < kth_farthest_dist:
                        heapq.heapreplace(nearest_neighbors, (-dist, point_id, point_vector))
        else:
            # If it's an internal node, add its children to the priority queue
            for child in current_node.children:
                if child.mbr is not None:
                    child_min_dist = child.mbr.min_distance(query_point)
                    
                    # Only add to queue if it has a chance of containing a nearer neighbor
                    if len(nearest_neighbors) < k or child_min_dist < -nearest_neighbors[0][0]:
                        heapq.heappush(pq, (child_min_dist, id(child), child))
                        
    # Format the results (convert negative distances back to positive and sort closest to farthest)
    results = [(-dist, pid, vec) for dist, pid, vec in nearest_neighbors]
    results.sort(key=lambda x: x[0])
    
    return results