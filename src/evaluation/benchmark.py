import os
import sys
import time
import gc
import statistics
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

from src.index.rtree import RTree
from src.search.knn_search import knn_search
from src.index.kdtree import KDTree 

def linear_scan(dataset, query, k):
    distances = []
    for point_id, vector in dataset:
        dist = np.linalg.norm(vector - query)
        distances.append((dist, point_id))
    distances.sort(key=lambda x: x[0])
    return distances[:k]

def warmup_system():
    dummy_dataset = np.random.rand(5000, 50)
    dummy_query = np.random.rand(50)
    for _ in range(500):
        _ = np.linalg.norm(dummy_dataset - dummy_query, axis=1)

def get_benchmark_data(N=10000, k=5, dimensions_to_test=[2, 5, 10, 20, 50]):
    """Runs the benchmark and returns the data as a dictionary for the Flask API."""
    warmup_system()
    
    # Dictionary to hold data for the frontend
    results = {
        "dims": dimensions_to_test,
        "linear": [],
        "kdtree": [],
        "rtree": []
    }
    
    iterations = 20

    for d in dimensions_to_test:
        print(f"Benchmarking {d}D space...")
        data_vectors = np.random.rand(N, d)
        dataset = [(f"item_{i}", vec) for i, vec in enumerate(data_vectors)]
        query_vector = np.random.rand(d)
        
        kd_tree = KDTree(dataset)
        
        r_tree = RTree(capacity=50)
        for point_id, vec in dataset:
            r_tree.insert(point_id, vec)
            
        # 1. Linear Scan
        linear_times = []
        gc.disable()
        for _ in range(iterations):
            start = time.perf_counter()
            _ = linear_scan(dataset, query_vector, k)
            linear_times.append((time.perf_counter() - start) * 1000)
        gc.enable()
        results["linear"].append(statistics.median(linear_times))
        
        # 2. KD-Tree
        kd_times = []
        gc.disable()
        for _ in range(iterations):
            start = time.perf_counter()
            _ = kd_tree.query(query_vector, k=k)
            kd_times.append((time.perf_counter() - start) * 1000)
        gc.enable()
        results["kdtree"].append(statistics.median(kd_times))
        
        # 3. R-Tree
        rtree_times = []
        gc.disable()
        for _ in range(iterations):
            start = time.perf_counter()
            _ = knn_search(r_tree, query_vector, k)
            rtree_times.append((time.perf_counter() - start) * 1000)
        gc.enable()
        results["rtree"].append(statistics.median(rtree_times))
        
    return results

if __name__ == "__main__":
    # Fallback to CLI printing if run directly
    data = get_benchmark_data()
    print(data)