from typing import Sequence, Tuple


def validateDimensions(input: Sequence, dimensions: int) -> None:
    """
    Checks if the length of the input matches the expected number of dimensions.

    Args:
        input (Sequence): The input sequence whose length will be checked.
        dimensions (int): The expected length of the input sequence.

    Raises:
        ValueError: If the length of the input does not equal the expected dimensions.
    """
    if len(input) != dimensions:
        raise ValueError(f"Dimension mismatch (input dim: {len(input)} != expected: {dimensions})")


def validateBounds(bounds: Sequence[Tuple[float, float]]) -> None:
    """
    Checks that each tuple in the bounds sequence has a valid lower and upper bound.

    Args:
        bounds (Sequence[Tuple[float, float]]): Sequence of (lower, upper) bound tuples.

    Raises:
        TypeError: If any element in bounds is not a 2-sized tuple.
        ValueError: If any lower bound is greater than its corresponding upper bound.
    """
    for i, bd in enumerate(bounds):
        if not isinstance(bd, tuple) or len(bd) != 2:
            raise TypeError(f"Bounds at index {i} is not a 2-sized tuple: {bd}")
        if bd[0] > bd[1]:
            raise ValueError(
                f"Lower bound ({bd[0]}) cannot be greater than upper bound ({bd[1]}) at index {i}"
            )


def validatePointBounds(point: Sequence[float], bounds: Sequence[Tuple[float, float]]) -> None:
    """
    Validates that each element of the input point lies within the respective bounds.

    Args:
        point (Sequence[float]): The input point to validate.
        bounds (Sequence[Tuple[float, float]]): A sequence where each item is a (lower, upper) tuple specifying the valid range for the respective dimension.

    Raises:
        ValueError: If any element in the point is out of its corresponding bounds.
    """
    for i, (low, high) in enumerate(bounds):
        if point[i] < low or point[i] > high:
            raise ValueError(f"Element at index {i} ({point[i]}) is out of bounds ({low}, {high})")

def validateRangeBounds(intervals: Sequence[Tuple[float, float]], bounds: Sequence[Tuple[float, float]]) -> None:
    """
    Validates that the interval given lies completely within the specified bounds of the space it belongs to.
    It is up to the developer to validate the dimensionality of the interval before calling this function, refer to validateDimensions.

    Args:
        intervals (Sequence[Tuple[float, float]]): The interval range to validate.
        bounds (Sequence[Tuple[float, float]]): The bounds to respect 

    Raises:
        ValueError: If any range is not contained within its respective bounds.
    """

    for dim in range(len(intervals)):
        if intervals[dim][0] < bounds[dim][0] or intervals[dim][1] > bounds[dim][1]:
            raise ValueError(f"Range at index {dim} isn't contained within parent space: {intervals[dim]} ⊄ {bounds[dim]}")
