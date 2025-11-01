import numpy as np
import sympy as sp

class ContinuousSpace:
    def __init__(self, name: str, dimensions: int, lowerBound, upperBound):
        self.name = name
        self.dimensions = dimensions

        self.__validateInputDimensions(lowerBound)
        self.__validateInputDimensions(upperBound)

        lowerBound = np.array(lowerBound, dtype=np.float32)
        upperBound = np.array(upperBound, dtype=np.float32)
        self.__validateBounds(lowerBound, upperBound)

        self.lowerBound = lowerBound
        self.upperBound = upperBound
        
        labels = [f"{self.name}{i}" for i in range(1, self.dimensions + 1)]
        self.__symbols = sp.symbols(labels)
        if not isinstance(self.__symbols, tuple):
            self.__symbols = (self.__symbols,)

    @property
    def labels(self) -> tuple:
        return self.__symbols

    def __validateInputDimensions(self, input):
        if self.dimensions != len(input):
            raise RuntimeError(f"Dimension mismatch (input:{len(input)} != expected:{self.dimensions})")

    def __validateBounds(self, lowerBound, upperBound):
        for i in range(self.dimensions):
            if lowerBound[i] > upperBound[i]:
                raise RuntimeError(
                    f"Inconsistent bounds at dimension {i + 1}: "
                    f"lower bound {lowerBound[i]} cannot be greater than upper bound {upperBound[i]}"
                )
