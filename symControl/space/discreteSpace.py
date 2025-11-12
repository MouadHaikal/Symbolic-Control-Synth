from .continuousSpace import ContinuousSpace
from .cell import Cell
from symControl.utils.validation import *

from math import floor


class DiscreteSpace(ContinuousSpace):
    """
    Represents a discretized, multi-dimensional space, subdivided into grid cells of fixed resolution.

    Inherits from ContinuousSpace, specifying a decomposition of each dimension into a finite number
    of cells. Provides methods to map between points in continuous space and discrete grid coordinates.

    Attributes:
        __resolutions (Tuple[int, ...]): The number of discrete cells per dimension.
        __cellSize (Tuple[float, ...]): The size of each cell per dimension.
    """
    __slots__ = ['__resolutions', '__cellSize']

    def __init__(self, name : str, dimensions: int, bounds: Sequence[Tuple[float, float]], resolutions: Sequence[int]):
        super().__init__(name, dimensions, bounds)
        validateDimensions(resolutions, dimensions)

        self.__resolutions = tuple(resolutions)
        self.__cellSize    = tuple(
            (bounds[i][1] - bounds[i][0]) / resolutions[i] 
            for i in range(dimensions)
        )


    def getCellCoords(self, point: Sequence[float] | Cell) -> Tuple[int, ...]:
        """
        Maps a continuous point to its corresponding discrete grid cell coordinates.

        Args:
            point (Sequence[float]) | Cell: If first type, it is point in the continuous space, otherwise a cell of that space.

        Returns:
            Tuple[int, ...]: The tuple of cell indices corresponding to the input point.

        Raises:
            ValueError: If the point is out of bounds or has the wrong dimensionality.
        """
        if isinstance(point, Cell):
            return self.getCellCoords(point.center)
        else:
            validateDimensions(point, self.dimensions)
            validatePointBounds(point, self.bounds)

            normalized = [(point[i] - self.bounds[i][0]) / (self.bounds[i][1] - self.bounds[i][0]) for i in range(self.dimensions)]

            coords = tuple(
                min(floor(normalized[i] * self.__resolutions[i]), self.__resolutions[i] - 1) 
                for i in range(self.dimensions)
            )

            return coords

    def getCell(self, coords: Sequence[int]) -> Cell:
        """
        Returns the cell object corresponding to the given cell coordinates.

        Args:
            coords (Sequence[int]): The discrete grid coordinates.

        Returns:
            Cell: The cell object defined by the computed bounds.

        Raises:
            ValueError: If the coordinates have the wrong dimensionality.
            ValueError: If the bounds of the cell lie outside the bounds of the state space
        """
        validateDimensions(coords, self.dimensions)

        cellBounds = [
            (
                self.bounds[i][0] + self.__cellSize[i] * coords[i],
                self.bounds[i][0] + self.__cellSize[i] * (coords[i] + 1)
            ) for i in range(self.dimensions)
        ]

        validateRangeBounds(cellBounds, self.bounds)

        return Cell(self.dimensions, cellBounds)

    def getCellSize(self) -> Tuple[float,...]:
        """
        Get the size of any cell in the space

        Args:
            Nothing

        Returns:
            Tuple[float,...]: self.__cellSize
        """
        return self.__cellSize;
