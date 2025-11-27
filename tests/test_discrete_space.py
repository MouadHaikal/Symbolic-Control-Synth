import pytest
import numpy as np
import time
from symControl.space.discreteSpace import DiscreteSpace
from symControl.logger import gLogger

# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def simple_space():
    """
    Creates a small 2D discrete space for logic and error testing.
    """
    bounds = [(0, 10), (0, 10)]
    gLogger.info("Creating simple 2D discrete space")
    return DiscreteSpace("SmallGrid", 2, bounds, [5, 5])


@pytest.fixture
def large_space():
    """
    Creates a large 3D discrete space for performance testing.
    """
    bounds = [(0, 100), (0, 100), (0, 100)]
    gLogger.info("Creating large 3D discrete space")
    return DiscreteSpace("BigGrid", 3, bounds, [100, 100, 100])


# -----------------------
# Tests
# -----------------------

def test_out_of_bounds_point(simple_space):
    """
    Tests that points outside the space bounds raise ValueError.
    """
    gLogger.debug("Testing out-of-bounds points")
    with pytest.raises(ValueError):
        simple_space.getCellCoords([11, 5])
    with pytest.raises(ValueError):
        simple_space.getCellCoords([-0.1, 3])
    gLogger.info("Out-of-bounds test passed")


def test_invalid_dimensions(simple_space):
    """
    Tests that providing a point with wrong dimensions raises ValueError.
    """
    gLogger.debug("Testing invalid dimensions")
    with pytest.raises(ValueError):
        simple_space.getCellCoords([1, 2, 3])
    gLogger.info("Invalid dimensions test passed")


def test_point_within_cell(simple_space):
    """
    Tests that points are contained within the bounds of their assigned cell.
    """
    points = [
        [0.1, 0.1],
        [4.9, 4.9],
        [5.1, 5.1],
        [9.9, 9.9],
    ]
    gLogger.debug("Testing points within their cells")
    for p in points:
        coords = simple_space.getCellCoords(p)
        cell = simple_space.getCell(coords)
        for val, (lb, ub) in zip(p, cell.bounds):
            assert lb <= val <= ub, f"Point {p} not within cell bounds {cell.bounds}"
    gLogger.info("Point within cell test passed")


def test_large_space_performance(large_space):
    """
    Performance test for evaluating many points in a large 3D space.
    """
    np.random.seed(42)
    points = np.random.uniform(0, 100, (10_000, 3))
    gLogger.debug("Starting performance test for 10k points")

    start_time = time.time()
    for p in points:
        coords = large_space.getCellCoords(p)
        assert all(0 <= c < r for c, r in zip(coords, [100, 100, 100]))
    elapsed = time.time() - start_time
    gLogger.info(f"Processed 10k points in {elapsed:.4f} seconds")
    assert elapsed < 3.0, "Performance too slow for 10,000 points"


def test_random_valid_points(simple_space):
    """
    Tests random points to ensure they fall within their assigned cell.
    """
    np.random.seed(0)
    gLogger.debug("Testing random valid points")
    for _ in range(1000):
        p = np.random.uniform(0, 10, 2)
        coords = simple_space.getCellCoords(p)
        cell = simple_space.getCell(coords)
        for val, (lb, ub) in zip(p, cell.bounds):
            assert lb <= val <= ub
    gLogger.info("Random valid points test passed")


def test_boundary_points(simple_space):
    """
    Tests points lying exactly on or near the boundaries of the space.
    """
    boundary_points = [
        [0, 0],
        [10 - 1e-9, 10 - 1e-9],
        [10 - 1e-9, 0],
        [0, 10 - 1e-9],
    ]
    gLogger.debug("Testing boundary points")
    for p in boundary_points:
        coords = simple_space.getCellCoords(p)
        cell = simple_space.getCell(coords)
        for val, (lb, ub) in zip(p, cell.bounds):
            assert lb <= val <= ub
    gLogger.info("Boundary points test passed")
