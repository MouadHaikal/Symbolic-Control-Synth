import numpy as np
from .continuousSpace import ContinuousSpace
from .cell import Cell

class DiscreteSpace(ContinuousSpace):
    __slots__ = ['resolutions', '__cellSize']

    def __init__(self, name : str, dimensions: int, lowerBounds, upperBounds, resolutions):
        super().__init__(name, dimensions, lowerBounds, upperBounds)

        self._validateInputDimensions(resolutions)
        self.resolutions = np.array(resolutions)
        self.__cellSize = (self.upperBounds - self.lowerBounds) / self.resolutions


    def getCellCoords(self, point) -> np.ndarray:
        point = np.array(point)
        self.__assertWithinBounds(point, 
                              self.lowerBounds, 
                              self.upperBounds)

        normX = (point - self.lowerBounds) / (self.upperBounds - self.lowerBounds)
        coords = np.floor(normX * self.resolutions).astype(int)

        # Avoid out of bound mapping
        coords = np.minimum(coords, self.resolutions - 1)
        return coords

    def getCell(self, coords) -> Cell:
        coords = np.array(coords)
        self.__assertWithinBounds(coords, 
                              np.zeros(self.dimensions), 
                              self.resolutions)


        lowerBound = self.lowerBounds + coords * self.__cellSize
        upperBound = lowerBound + self.__cellSize

        return Cell(lowerBound, upperBound)

        
    def __assertWithinBounds(self, vector, lowerBounds, upperBounds) -> bool:
        vector = np.array(vector)
        self._validateInputDimensions(vector)

        if not np.all((vector >= lowerBounds) & (vector < upperBounds)):
                raise RuntimeError("vector out of bounds")
        return True
