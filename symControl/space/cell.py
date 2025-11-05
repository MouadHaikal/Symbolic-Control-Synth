from symControl.utils.validation import *


class Cell:
    """
    Represents a hyperrectangular cell in a multi-dimensional space.

    The cell is defined by its per-dimension bounds and provides its dimensionality, bounds,
    and geometric center. Validation checks ensure that the provided bounds are correct upon construction.

    Attributes:
        dimensions (int): Number of dimensions of the cell.
        bounds (Tuple[Tuple[float, float], ...]): Tuple of (lower, upper) bounds for each dimension.
        center (Tuple[float, ...]): Position of the center of the cell in each dimension.
    """
    __slots__ = ['dimensions', 'bounds', 'center']

    def __init__(self, dimensions: int, bounds: Sequence[Tuple[float, float]]):
        validateDimensions(bounds, dimensions)
        validateBounds(bounds)

        self.dimensions = dimensions
        self.bounds     = tuple(bounds)
        self.center     = tuple(
            (bounds[i][1] + bounds[i][0]) / 2
            for i in range(dimensions)
        )
