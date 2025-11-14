from typing import Sequence, Tuple
from symControl.space.discreteSpace import DiscreteSpace
from symControl.space.continuousSpace import ContinuousSpace

class Automaton:
    def __init__(self, 
                 stateSpace: DiscreteSpace, 
                 controlSpace: DiscreteSpace, 
                 disturbanceSpace: ContinuousSpace, 
                 fAtPointCode: str) -> None: ...


def floodFill(lowerBound: Sequence[int], upperBound: Sequence[int]) -> Sequence[Tuple[int,...]]: ...
