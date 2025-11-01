import numpy as np
import sympy as sp

from logger import gLogger


class ContinuousSpace:
    def __init__(self, name : str, dimensions: int, lowerBounds, upperBounds):
        self.name = name
        self.dimensions = dimensions

        self.__validateInputDimensions(lowerBounds)
        self.lowerBounds = np.ndarray(lowerBounds)

        self.__validateInputDimensions(upperBounds)
        self.upperBounds = np.ndarray(upperBounds)
        
        self.__symbols = sp.symbols(f"{self.name}(1:{self.dimensions+1})")
        if not isinstance(self.__symbols, tuple):
            self.__symbols = (self.__symbols,)


    @property
    def labels(self) -> tuple:
        return self.__symbols


    def __validateInputDimensions(self, input):
        if self.dimensions != len(input):
            gLogger.error(f"Dimension mismatch (input:{len(input)} != expected:{self.dimensions})")
            # TODO: exit
