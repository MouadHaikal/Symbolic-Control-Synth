import pytest
import numpy as np
import time
from symControl.space.discreteSpace import DiscreteSpace


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def simple_space():
    """Creates a small 2D discrete space for logic and error testing.

    Returns:
        DiscreteSpace: A 2D grid space from [0,0] to [10,10] with 5x5 resolution.
    """
    return DiscreteSpace("SmallGrid", 2, [0, 0], [10, 10], [5, 5])


@pytest.fixture
def large_space():
    """Creates a large 3D discrete space for performance testing.

    Returns:
        DiscreteSpace: A 3D grid space from [0,0,0] to [100,100,100] with 100x100x100 resolution.
    """
    return DiscreteSpace("BigGrid", 3, [0, 0, 0], [100, 100, 100], [100, 100, 100])


# -----------------------
# Tests
# -----------------------

def test_out_of_bounds_point(simple_space):
    """Tests that points outside the space bounds raise RuntimeError.

    Args:
        simple_space (DiscreteSpace): 2D discrete space fixture.

    Raises:
        RuntimeError: If the point is out of bounds.
    """
    # Point outside upper bounds
    with pytest.raises(RuntimeError, match="vector out of bounds"):
        simple_space.getCellCoords([11, 5])

    # Point below lower bounds
    with pytest.raises(RuntimeError, match="vector out of bounds"):
        simple_space.getCellCoords([-0.1, 3])


def test_invalid_dimensions(simple_space):
    """Tests that providing a point with wrong dimensions raises RuntimeError.

    Args:
        simple_space (DiscreteSpace): 2D discrete space fixture.

    Raises:
        RuntimeError: If the point has incorrect dimensionality.
    """
    with pytest.raises(RuntimeError, match="Dimension mismatch"):
        simple_space.getCellCoords([1, 2, 3])


def test_inverted_bounds():
    """Tests that initializing a space with lower bounds > upper bounds raises RuntimeError.

    Raises:
        RuntimeError: If lower bounds are greater than upper bounds.
    """
    with pytest.raises(RuntimeError, match="Inconsistent bounds"):
        DiscreteSpace("Invalid", 2, [5, 5], [1, 1], [2, 2])


def test_point_within_cell(simple_space):
    """Tests that points are contained within the bounds of their assigned cell.

    Args:
        simple_space (DiscreteSpace): 2D discrete space fixture.
    """
    points = [
        [0.1, 0.1],
        [4.9, 4.9],
        [5.1, 5.1],
        [9.9, 9.9]
    ]
    for p in points:
        coords = simple_space.getCellCoords(p)
        cell = simple_space.getCell(coords)
        assert all(lb <= val <= ub for val, lb, ub in zip(p, cell.lowerBound, cell.upperBound)), \
            f"Point {p} not within cell bounds {cell.lowerBound}-{cell.upperBound}"


def test_large_space_performance(large_space):
    """Performance test for evaluating many points in a large 3D space.

    Args:
        large_space (DiscreteSpace): 3D discrete space fixture.

    Raises:
        AssertionError: If processing 10,000 points takes too long.
    """
    np.random.seed(42)
    points = np.random.uniform(0, 100, (10_000, 3))

    start_time = time.time()

    for p in points:
        coords = large_space.getCellCoords(p)
        assert np.all(coords >= 0) and np.all(coords < large_space.resolutions)

    elapsed = time.time() - start_time
    print(f"\nProcessed 10k points in {elapsed:.4f} seconds")
    assert elapsed < 3.0, "Performance too slow for 10,000 points"


def test_random_valid_points(simple_space):
    """Tests random points to ensure they fall within their assigned cell.

    Args:
        simple_space (DiscreteSpace): 2D discrete space fixture.
    """
    np.random.seed(0)
    for _ in range(1000):
        p = np.random.uniform(0, 10, 2)
        coords = simple_space.getCellCoords(p)
        cell = simple_space.getCell(coords)
        assert all(lb <= val <= ub for val, lb, ub in zip(p, cell.lowerBound, cell.upperBound))


def test_boundary_points(simple_space):
    """Tests points lying exactly on or near the boundaries of the space.

    Args:
        simple_space (DiscreteSpace): 2D discrete space fixture.
    """
    boundary_points = [
        [0, 0],
        [10, 10 - 1e-9],
        [10 - 1e-9, 0],
        [0, 10 - 1e-9],
    ]

    for p in boundary_points:
        coords = simple_space.getCellCoords(p)
        cell = simple_space.getCell(coords)
        assert all(lb <= val <= ub for val, lb, ub in zip(p, cell.lowerBound, cell.upperBound))
