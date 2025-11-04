import numpy as np
import sympy as sp


class ContinuousSpace:
    def __init__(self, name: str, dimensions: int, lowerBounds, upperBounds):
        self.name = name
        self.dimensions = dimensions

        self._validateInputDimensions(lowerBounds)
        self._validateInputDimensions(upperBounds)

        lowerBounds = np.array(lowerBounds, dtype=np.float64)
        upperBounds = np.array(upperBounds, dtype=np.float64)
        self._validateBounds(lowerBounds, upperBounds)

        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        

    def _validateInputDimensions(self, input):
        if self.dimensions != len(input):
            raise RuntimeError(f"Dimension mismatch (input:{len(input)} != expected:{self.dimensions})")

    def _validateBounds(self, lowerBound, upperBound):
        for i in range(self.dimensions):
            if lowerBound[i] > upperBound[i]:
                raise RuntimeError(
                    f"Inconsistent bounds at dimension {i + 1}: "
                    f"lower bound {lowerBound[i]} cannot be greater than upper bound {upperBound[i]}"
                )
            # gLogger.error(f"Dimension mismatch (input:{len(input)} != expected:{self.dimensions})")
            # TODO: exit
