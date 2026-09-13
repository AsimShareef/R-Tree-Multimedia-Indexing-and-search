import numpy as np
import pytest

from src.index.geometry import MBR


def test_volume_2d():
    mbr = MBR(min_bounds=[0, 0], max_bounds=[2, 3])
    assert mbr.volume() == pytest.approx(6.0)


def test_volume_is_zero_for_a_single_point():
    mbr = MBR(min_bounds=[1, 1], max_bounds=[1, 1])
    assert mbr.volume() == pytest.approx(0.0)


def test_enlarge_with_point_outside_bounds():
    mbr = MBR(min_bounds=[0, 0], max_bounds=[2, 2])
    enlarged = mbr.enlarge(np.array([5, -1]))
    assert list(enlarged.min_bounds) == pytest.approx([0, -1])
    assert list(enlarged.max_bounds) == pytest.approx([5, 2])


def test_enlarge_with_another_mbr():
    a = MBR(min_bounds=[0, 0], max_bounds=[2, 2])
    b = MBR(min_bounds=[1, 3], max_bounds=[4, 5])
    enlarged = a.enlarge(b)
    assert list(enlarged.min_bounds) == pytest.approx([0, 0])
    assert list(enlarged.max_bounds) == pytest.approx([4, 5])


def test_enlargement_area_is_zero_for_point_already_inside():
    mbr = MBR(min_bounds=[0, 0], max_bounds=[10, 10])
    assert mbr.enlargement_area(np.array([5, 5])) == pytest.approx(0.0)


def test_enlargement_area_grows_for_point_outside():
    mbr = MBR(min_bounds=[0, 0], max_bounds=[2, 2])
    # New MBR would be [0,0]-[4,2] -> area 8, original area 4 -> enlargement 4
    assert mbr.enlargement_area(np.array([4, 1])) == pytest.approx(4.0)


@pytest.mark.parametrize(
    "a_bounds, b_bounds, expected",
    [
        (([0, 0], [2, 2]), ([1, 1], [3, 3]), True),   # overlapping
        (([0, 0], [1, 1]), ([1, 1], [2, 2]), True),   # touching at a corner
        (([0, 0], [1, 1]), ([5, 5], [6, 6]), False),  # disjoint
    ],
)
def test_intersects(a_bounds, b_bounds, expected):
    a = MBR(*a_bounds)
    b = MBR(*b_bounds)
    assert a.intersects(b) is expected
    assert b.intersects(a) is expected  # intersection should be symmetric


def test_min_distance_is_zero_when_point_is_inside():
    mbr = MBR(min_bounds=[0, 0], max_bounds=[10, 10])
    assert mbr.min_distance(np.array([5, 5])) == pytest.approx(0.0)


def test_min_distance_to_a_point_outside():
    mbr = MBR(min_bounds=[0, 0], max_bounds=[2, 2])
    # Closest point on the box to (5, 0) is (2, 0) -> distance 3
    assert mbr.min_distance(np.array([5, 0])) == pytest.approx(3.0)


def test_contains_point():
    mbr = MBR(min_bounds=[0, 0], max_bounds=[2, 2])
    assert mbr.contains_point(np.array([1, 1])) is True
    assert mbr.contains_point(np.array([0, 0])) is True  # boundary is inclusive
    assert mbr.contains_point(np.array([3, 1])) is False
