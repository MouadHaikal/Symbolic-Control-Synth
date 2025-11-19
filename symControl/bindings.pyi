from symControl.space.discreteSpace import DiscreteSpace
from symControl.space.continuousSpace import ContinuousSpace

class Automaton:
    def __init__(self, 
                 stateSpace: DiscreteSpace, 
                 inputSpace: DiscreteSpace, 
                 disturbanceSpace: ContinuousSpace, 
                 isCooperative: bool,
                 maxDisturbJac: tuple[tuple[float, ...], ...] | None,
                 buildAutomatonCode: str) -> None: ...
