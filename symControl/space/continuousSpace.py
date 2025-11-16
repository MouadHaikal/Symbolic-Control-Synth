from symControl.utils.validation import *

class ContinuousSpace:
    """
    Represents a continuous multi-dimensional space defined by per-dimension bounds.

    This class validates the dimensionality and the validity of the bounds on initialization.
    Each dimension of the space is represented by a (lower, upper) tuple, ensuring that
    all lower bounds are less than or equal to the corresponding upper bounds.

    Attributes:
        dimensions (int): The number of dimensions in the space.
        bounds (Tuple[Tuple[float, float], ...]): A list of 2-sized tuples, each specifying (lower, upper) bounds for a dimension.
    """
    __slots__ = ['dimensions', 'bounds']

    def __init__(self, dimensions: int, bounds: Sequence[Tuple[float, float]]):
        validateDimensions(bounds, dimensions)
        validateBounds(bounds)

        self.dimensions = dimensions
        self.bounds     = tuple(bounds)
