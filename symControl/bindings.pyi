from symControl.space.discreteSpace import DiscreteSpace
from symControl.space.continuousSpace import ContinuousSpace

class Automaton:
    """
    Python binding for a C++ class modeling the automaton, which is constructed during initialization.
    """
    def __init__(self, 
                 stateSpace: DiscreteSpace, 
                 inputSpace: DiscreteSpace, 
                 disturbanceSpace: ContinuousSpace, 
                 isCooperative: bool,
                 maxDisturbJac: tuple[tuple[float, ...], ...],
                 buildAutomatonCode: str) -> None: ...

    def applySecuritySpec(self,
                          pyObstacleLowerBoundCoords: tuple[int, ...],
                          pyObstacleUpperBoundCoords: tuple[int, ...]) -> None: ...

    def getController(self,
                      pyStartStateCoords:       tuple[int, ...],
                      pyTargetLowerBoundCoords: tuple[int, ...],
                      pyTargetUpperBoundCoords: tuple[int, ...],
                      pathCount:                int) -> list[list[int]]: ...
                      
