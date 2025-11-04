import pytest
import numpy as np
from symControl.logger import gLogger
from symControl.space.discreteSpace import DiscreteSpace

@pytest.fixture
def ds():
    return DiscreteSpace("Grid", 2, [0, 0], [10, 10], [5, 5])

def test_discrete_space_logging_and_output(ds, caplog):
    points = [[1, 1], [4.9, 9.9], [5, 5]]

    expected_results = {
        (1, 1): {"coords": np.array([0, 0])},
        (4.9, 9.9): {"coords": np.array([2, 4])},
        (5, 5): {"coords": np.array([2, 2])},
    }

    caplog.set_level("CRITICAL", logger="symControl.logger")

    for p in points:
        try:
            coords = ds.getCellCoords(p)
            cell = ds.getCell(coords)
            gLogger.info(f"Point: {p} → Coords: {coords}")
            gLogger.info(f"Cell lowerBound: {cell.lowerBound}, upperBound: {cell.upperBound}")
        except RuntimeError as e:
            gLogger.error(f"Point {p} error: {e}")
            coords, cell = None, None

        key = tuple(p)
        if key in expected_results:
            assert np.array_equal(coords, expected_results[key]["coords"]), f"Coords mismatch for point {p}"

            assert all(lb <= val <= ub for val, lb, ub in zip(p, cell.lowerBound, cell.upperBound)), \
                f"Point {p} not inside cell bounds"

    
