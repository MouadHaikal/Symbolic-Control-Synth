class Cell:
    def __init__(self, lowerBound, upperBound):
        self.lowerBound = lowerBound
        self.upperBound = upperBound

        self.center = [(upperBound[i] + lowerBound[i]) / 2 for i in range(len(lowerBound))]
