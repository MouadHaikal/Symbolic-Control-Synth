import numpy as np
class Cell:
    __slots__ = ['lowerBound', 'upperBound', 'center']

    def __init__(self, lowerBound, upperBound):
        self.lowerBound = np.array(lowerBound)
        self.upperBound = np.array(upperBound)

        self.center = (self.upperBound + self.lowerBound) / 2
