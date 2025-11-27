from math import floor, prod

from symControl.utils.validation import *
from .continuousSpace import ContinuousSpace


class DiscreteSpace(ContinuousSpace):
    """
    Represents a discretized, multi-dimensional space, subdivided into grid cells of fixed resolution.

    Attributes:
        __resolutions (Tuple[int, ...]): The number of discrete cells per dimension.
        __cellSize (Tuple[float, ...]): The size of each cell per dimension.
    """

    __slots__ = ['__resolutions', '__cellSize']

    def __init__(self, dimensions: int, bounds: Sequence[Tuple[float, float]], resolutions: Sequence[int]):
        super().__init__(dimensions, bounds)
        validateDimensions(resolutions, dimensions)

        self.__resolutions = tuple(resolutions)
        self.__cellSize    = tuple(
            (bounds[i][1] - bounds[i][0]) / resolutions[i] 
            for i in range(dimensions)
        )


    def getCellCoords(self, point: Sequence[float]) -> Tuple[int, ...]:
        """
        Maps a continuous point to its corresponding discrete grid cell coordinates.

        Args:
            point (Sequence[float]): Coordinates of the point in continuous space.

        Returns:
            Tuple[int, ...]: Indices of the grid cell that contains the input point.

        Raises:
            ValueError: If the point lies outside the domain or its dimensionality does not match the grid..
        """
        validateDimensions(point, self.dimensions)
        validatePointBounds(point, self.bounds)

        normalized = [(point[i] - self.bounds[i][0]) / (self.bounds[i][1] - self.bounds[i][0]) for i in range(self.dimensions)]

        coords = tuple(
            min(floor(normalized[i] * self.__resolutions[i]), self.__resolutions[i] - 1) 
            for i in range(self.dimensions)
        )

        return coords


    @property
    def resolutions(self):
        return self.__resolutions

    @property
    def cellSize(self):
        return self.__cellSize

    @property 
    def cellCount(self):
        return prod(self.__resolutions)
