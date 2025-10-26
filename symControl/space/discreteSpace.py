import numpy as np
from .continuousSpace import ContinuousSpace
from .cell import Cell

class DiscreteSpace(ContinuousSpace):
    def __init__(self, name : str, dimensions: int, lowerBound, upperBound, resolutions):
        super().__init__(name, dimensions, lowerBound, upperBound)

        self.__validateInputDimensions(resolutions)
        self.resolutions = resolutions
        self.__cellSize = [(upperBound[i] - lowerBound[i]) / resolutions[i] for i in range(dimensions)]


    def getCellCoords(self, point) -> np.ndarray:
        self.__isWithinBounds(point, 
                              self.lowerBound, 
                              self.upperBound)

        normX =  np.array([point[i] / (self.upperBound[i] - self.lowerBound[i]) for i in range(self.dimensions)])
        coords = np.array([int(normX[i] * self.resolutions[i]) for i in range(self.dimensions)])

        # Avoid out of bound mapping
        for i in range(len(coords)):
            if coords[i] == self.resolutions[i]:
                coords[i] -= 1

        return coords

    def getCell(self, coords) -> Cell:
        self.__isWithinBounds(coords, 
                              np.zeros(self.dimensions), 
                              self.resolutions)


        lowerBound = [self.lowerBound[i] + coords[i] * self.__cellSize[i] for i in range(self.dimensions)]
        upperBound = [lowerBound[i] + self.__cellSize[i] for i in range(self.dimensions)]

        return Cell(lowerBound, upperBound)

        
    def __isWithinBounds(self, vector, lowerBounds, upperBounds) -> bool:
        self.__validateInputDimensions(vector)

        verdict = True
        for dimension in range(self.dimensions):
            if not (lowerBounds[dimension] <= vector[dimension] and 
                    vector[dimension] < upperBounds[dimension]):
                verdict = False
                break

        if not verdict:
            raise RuntimeError("vector out of bounds")

        return verdict
