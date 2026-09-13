import numpy as np
import pytest

from src.index.kdtree import KDTree
from src.search.linear_scan import linear_scan_knn


def make_dataset(n, dims, seed):
    rng = np.random.default_rng(seed)
    vectors = rng.random((n, dims))
    return {f"p{i}": vectors[i] for i in range(n)}


@pytest.mark.parametrize("n,dims,k,seed", [
    (50, 2, 3, 20),
    (200, 5, 5, 21),
    (300, 10, 10, 22),
])
def test_kdtree_matches_linear_scan(n, dims, k, seed):
    dataset = make_dataset(n, dims, seed)
    tree = KDTree(list(dataset.items()))

    rng = np.random.default_rng(seed + 1000)
    query = rng.random(dims)

    kd_ids = [pid for _, pid in tree.query(query, k=k)]
    exact_ids = [pid for _, pid, _ in linear_scan_knn(dataset, query, k=k)]
    assert kd_ids == exact_ids


def test_kdtree_handles_single_point():
    tree = KDTree([("only", np.array([1.0, 2.0]))])
    result = tree.query(np.array([0.0, 0.0]), k=1)
    assert [pid for _, pid in result] == ["only"]
