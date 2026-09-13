import numpy as np
import heapq

class KDNode:
    def __init__(self, point_id, point, axis):
        self.id = point_id
        self.point = point
        self.axis = axis
        self.left = None
        self.right = None

class KDTree:
    def __init__(self, data):
        """
        Builds the KD-Tree recursively.
        :param data: List of tuples -> (point_id, numpy_vector)
        """
        self.dimensions = len(data[0][1]) if data else 0
        self.root = self._build(data, depth=0)

    def _build(self, points, depth):
        if not points:
            return None

        # 1. Determine splitting axis
        axis = depth % self.dimensions

        # 2. Sort by axis and find median
        points.sort(key=lambda x: x[1][axis])
        median_idx = len(points) // 2

        # 3. Create Node
        node = KDNode(
            point_id=points[median_idx][0],
            point=points[median_idx][1],
            axis=axis
        )

        # 4. Recurse for left and right children
        node.left = self._build(points[:median_idx], depth + 1)
        node.right = self._build(points[median_idx + 1:], depth + 1)

        return node

    def query(self, query_point, k=5):
        """
        Performs an exact K-Nearest Neighbor search using recursive backtracking.
        """
        # Python's heapq is a min-heap. To make a bounded max-heap, we push negative distances.
        # Format: (-distance, tie_breaker, point_id)
        max_heap = []
        tie_breaker = 0 # Prevents heapq from throwing an error if distances are identical

        def search(node):
            nonlocal tie_breaker
            if node is None:
                return

            # Calculate actual Euclidean distance
            dist = np.linalg.norm(node.point - query_point)

            # Push to priority queue (Max-Heap logic)
            if len(max_heap) < k:
                heapq.heappush(max_heap, (-dist, tie_breaker, node.id))
                tie_breaker += 1
            elif dist < -max_heap[0][0]:
                heapq.heappop(max_heap)
                heapq.heappush(max_heap, (-dist, tie_breaker, node.id))
                tie_breaker += 1

            # Determine which side of the splitting plane the query falls on
            axis_dist = query_point[node.axis] - node.point[node.axis]
            
            if axis_dist < 0:
                good_side, bad_side = node.left, node.right
            else:
                good_side, bad_side = node.right, node.left

            # 1. Always explore the side the query point falls into first
            search(good_side)

            # 2. Pruning Check: Do we need to check the "bad" side?
            # We only check if the orthogonal distance to the boundary is smaller 
            # than the worst distance currently in our K-nearest neighbors heap.
            if len(max_heap) < k or abs(axis_dist) < -max_heap[0][0]:
                search(bad_side)

        # Start recursive search from root
        search(self.root)
        
        # Format output to match linear_scan: list of (distance, point_id), closest first
        results = [(-dist, p_id) for dist, _, p_id in max_heap]
        results.sort(key=lambda x: x[0])
        return results