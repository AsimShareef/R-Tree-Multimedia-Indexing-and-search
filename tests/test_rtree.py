import numpy as np
import pytest

from src.index.node import InternalNode
from src.index.rtree import RTree
from src.search.knn_search import knn_search
from src.search.linear_scan import linear_scan_knn


def make_dataset(n, dims, seed):
    rng = np.random.default_rng(seed)
    vectors = rng.random((n, dims))
    return {f"p{i}": vectors[i] for i in range(n)}


def count_leaf_points(node):
    if node.is_leaf():
        return len(node.entries)
    return sum(count_leaf_points(child) for child in node.children)


def all_point_ids(node):
    if node.is_leaf():
        return {pid for pid, _ in node.entries}
    ids = set()
    for child in node.children:
        ids |= all_point_ids(child)
    return ids


def assert_mbrs_are_consistent(node):
    """Every child's MBR must fit entirely within its parent's MBR."""
    if node.is_leaf():
        for _, vector in node.entries:
            assert node.mbr.contains_point(vector)
        return
    for child in node.children:
        assert node.mbr.contains_point(child.mbr.min_bounds)
        assert node.mbr.contains_point(child.mbr.max_bounds)
        assert_mbrs_are_consistent(child)


class TestInsert:
    def test_single_insert_sets_mbr_to_the_point(self):
        tree = RTree(capacity=10)
        tree.insert("a", np.array([1.0, 2.0]))
        assert tree.root.is_leaf()
        assert list(tree.root.mbr.min_bounds) == pytest.approx([1.0, 2.0])
        assert list(tree.root.mbr.max_bounds) == pytest.approx([1.0, 2.0])

    def test_insert_below_capacity_does_not_split(self):
        tree = RTree(capacity=4)
        for i in range(4):
            tree.insert(f"p{i}", np.random.rand(3))
        assert tree.root.is_leaf()
        assert count_leaf_points(tree.root) == 4

    def test_overflow_triggers_split_into_internal_root(self):
        tree = RTree(capacity=4)
        for i in range(5):
            tree.insert(f"p{i}", np.random.rand(3))
        assert isinstance(tree.root, InternalNode)
        assert len(tree.root.children) == 2
        assert count_leaf_points(tree.root) == 5

    def test_no_node_exceeds_capacity_after_many_inserts(self):
        tree = RTree(capacity=6)
        dataset = make_dataset(200, dims=4, seed=1)
        for pid, vec in dataset.items():
            tree.insert(pid, vec)

        def check(node):
            if node.is_leaf():
                assert len(node.entries) <= node.capacity
            else:
                assert len(node.children) <= node.capacity
                for child in node.children:
                    check(child)

        check(tree.root)
        assert count_leaf_points(tree.root) == len(dataset)
        assert all_point_ids(tree.root) == set(dataset.keys())

    def test_mbrs_stay_consistent_after_many_inserts(self):
        tree = RTree(capacity=5)
        dataset = make_dataset(150, dims=3, seed=2)
        for pid, vec in dataset.items():
            tree.insert(pid, vec)
        assert_mbrs_are_consistent(tree.root)


class TestKnnSearchMatchesLinearScan:
    @pytest.mark.parametrize("n,dims,capacity,k,seed", [
        (50, 2, 4, 3, 10),
        (200, 5, 10, 5, 11),
        (300, 8, 50, 10, 12),
    ])
    def test_exact_agreement_with_brute_force(self, n, dims, capacity, k, seed):
        dataset = make_dataset(n, dims, seed)
        tree = RTree(capacity=capacity)
        for pid, vec in dataset.items():
            tree.insert(pid, vec)

        rng = np.random.default_rng(seed + 1000)
        query = rng.random(dims)

        rtree_results = knn_search(tree, query, k=k)
        exact_results = linear_scan_knn(dataset, query, k=k)

        rtree_ids = [pid for _, pid, _ in rtree_results]
        exact_ids = [pid for _, pid, _ in exact_results]
        assert rtree_ids == exact_ids

    def test_k_larger_than_dataset_returns_everything(self):
        dataset = make_dataset(5, dims=2, seed=3)
        tree = RTree(capacity=3)
        for pid, vec in dataset.items():
            tree.insert(pid, vec)
        results = knn_search(tree, np.random.rand(2), k=100)
        assert len(results) == 5

    def test_empty_tree_returns_no_results(self):
        tree = RTree(capacity=10)
        assert knn_search(tree, np.array([0.0, 0.0]), k=5) == []


class TestDelete:
    def test_delete_removes_point_from_results(self):
        dataset = make_dataset(30, dims=3, seed=4)
        tree = RTree(capacity=5)
        for pid, vec in dataset.items():
            tree.insert(pid, vec)

        target_id, target_vec = next(iter(dataset.items()))
        assert tree.delete(target_id, target_vec) is True
        assert target_id not in all_point_ids(tree.root)
        assert count_leaf_points(tree.root) == len(dataset) - 1

        results = knn_search(tree, target_vec, k=len(dataset))
        returned_ids = {pid for _, pid, _ in results}
        assert target_id not in returned_ids

    def test_delete_nonexistent_point_returns_false(self):
        tree = RTree(capacity=5)
        tree.insert("a", np.array([1.0, 1.0]))
        assert tree.delete("does-not-exist", np.array([9.0, 9.0])) is False
        assert count_leaf_points(tree.root) == 1

    def test_delete_matches_linear_scan_after_partial_deletion(self):
        dataset = make_dataset(300, dims=6, seed=5)
        tree = RTree(capacity=8)
        for pid, vec in dataset.items():
            tree.insert(pid, vec)

        to_delete = list(dataset.keys())[::3]  # delete a third of the points
        for pid in to_delete:
            assert tree.delete(pid, dataset[pid]) is True

        surviving = {pid: vec for pid, vec in dataset.items() if pid not in set(to_delete)}
        assert count_leaf_points(tree.root) == len(surviving)
        assert all_point_ids(tree.root) == set(surviving.keys())

        rng = np.random.default_rng(999)
        query = rng.random(6)
        rtree_ids = [pid for _, pid, _ in knn_search(tree, query, k=10)]
        exact_ids = [pid for _, pid, _ in linear_scan_knn(surviving, query, k=10)]
        assert rtree_ids == exact_ids

    def test_no_node_underflows_below_min_fill_after_deletions(self):
        capacity = 6
        min_fill = capacity // 2
        dataset = make_dataset(400, dims=4, seed=6)
        tree = RTree(capacity=capacity)
        for pid, vec in dataset.items():
            tree.insert(pid, vec)

        for pid in list(dataset.keys())[:250]:
            tree.delete(pid, dataset[pid])

        def check(node, is_root):
            if node.is_leaf():
                if not is_root:
                    assert len(node.entries) >= min_fill
            else:
                if not is_root:
                    assert len(node.children) >= min_fill
                for child in node.children:
                    check(child, is_root=False)

        check(tree.root, is_root=True)

    def test_deleting_every_point_leaves_an_empty_leaf_root(self):
        dataset = make_dataset(60, dims=3, seed=7)
        tree = RTree(capacity=5)
        for pid, vec in dataset.items():
            tree.insert(pid, vec)
        for pid, vec in dataset.items():
            assert tree.delete(pid, vec) is True

        assert tree.root.is_leaf()
        assert tree.root.entries == []
        assert tree.root.mbr is None
        assert knn_search(tree, np.random.rand(3), k=5) == []

    def test_random_insert_delete_sequence_stays_consistent_with_ground_truth(self):
        rng = np.random.default_rng(42)
        tree = RTree(capacity=6)
        ground_truth = {}

        for i in range(500):
            pid = f"p{i % 120}"
            if pid in ground_truth and rng.random() < 0.5:
                tree.delete(pid, ground_truth.pop(pid))
            else:
                vec = rng.random(4)
                if pid in ground_truth:
                    tree.delete(pid, ground_truth[pid])
                ground_truth[pid] = vec
                tree.insert(pid, vec)

        assert all_point_ids(tree.root) == set(ground_truth.keys())
        assert count_leaf_points(tree.root) == len(ground_truth)

        query = rng.random(4)
        k = min(10, len(ground_truth))
        rtree_ids = [pid for _, pid, _ in knn_search(tree, query, k=k)]
        exact_ids = [pid for _, pid, _ in linear_scan_knn(ground_truth, query, k=k)]
        assert rtree_ids == exact_ids
