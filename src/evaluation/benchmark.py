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

def compute_recall(approx_ids, exact_ids):
    """
    Recall@k: fraction of the brute-force top-k ("ground truth") that also
    shows up in the tree-based result set. Both the R-Tree and KD-Tree here
    are exact (not approximate) KNN algorithms, so this is really a
    correctness check -- recall should sit at (or extremely near) 1.0.
    Anything consistently lower would indicate a pruning bug.
    """
    exact_set = set(exact_ids)
    if not exact_set:
        return 1.0
    hits = sum(1 for pid in approx_ids if pid in exact_set)
    return hits / len(exact_set)

def get_benchmark_data(N=10000, k=5, dimensions_to_test=[2, 5, 10, 20, 50], recall_trials=10):
    """Runs the benchmark and returns the data as a dictionary for the Flask API."""
    warmup_system()

    # Dictionary to hold data for the frontend
    results = {
        "dims": dimensions_to_test,
        "linear": [],
        "kdtree": [],
        "rtree": [],
        "kdtree_recall": [],
        "rtree_recall": []
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

        # 4. Correctness check: does each index actually agree with brute-force
        # ground truth on a fresh batch of queries? (separate from the timed
        # queries above so it can't skew the latency numbers.)
        kd_recalls, rtree_recalls = [], []
        for _ in range(recall_trials):
            trial_query = np.random.rand(d)
            exact_ids = [pid for _, pid in linear_scan(dataset, trial_query, k)]
            kd_ids = [pid for _, pid in kd_tree.query(trial_query, k=k)]
            rtree_ids = [pid for _, pid, _ in knn_search(r_tree, trial_query, k)]
            kd_recalls.append(compute_recall(kd_ids, exact_ids))
            rtree_recalls.append(compute_recall(rtree_ids, exact_ids))

        results["kdtree_recall"].append(statistics.mean(kd_recalls))
        results["rtree_recall"].append(statistics.mean(rtree_recalls))

    return results

if __name__ == "__main__":
    # Fallback to CLI printing if run directly
    data = get_benchmark_data()
    print(f"\n{'Dim':>5} | {'Linear (ms)':>12} | {'KD-Tree (ms)':>13} | {'R-Tree (ms)':>12} | {'KD Recall':>10} | {'R Recall':>9}")
    print("-" * 76)
    for i, d in enumerate(data["dims"]):
        print(f"{d:>5} | {data['linear'][i]:>12.4f} | {data['kdtree'][i]:>13.4f} | "
              f"{data['rtree'][i]:>12.4f} | {data['kdtree_recall'][i]:>9.1%} | {data['rtree_recall'][i]:>8.1%}")