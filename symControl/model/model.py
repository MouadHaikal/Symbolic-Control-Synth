from typing import Set
from symControl.space.continuousSpace import ContinuousSpace
from symControl.space.discreteSpace import DiscreteSpace
from symControl.model.transitionFunction import TransitionFunction
from symControl.utils.validation import *
from symControl.utils.constants import *

class Model:
    """
    Represents the symbolic model obtained from a continuous model.

    This class encapsulates the state space, control space, and disturbance space as instances of the DiscreteSpace class, together with a symbolic transition function represented by a TransitionFunction instance.
    It currently provides a method for mapping a subset of the state space to the corresponding set of reachable states in the discrete domain.

    Attributes:
        stateSpace (DiscreteSpace): Represation of the space the model can operate in. 
        currentState (Cell): the current position in the stateSpace, 
                             by default it is mapped to the point with coordinates (0,...) as a start state during initalization.
        transitionFunction (TransitionFunction): the symbolic equations that govern the behavior of the model
    """

    __slots__ = ['stateSpace', 'currentState', 'transitionFunction']
    
    def __init__(self,
                 stateSpace: DiscreteSpace,
                 controlSpace: DiscreteSpace,
                 disturbanceSpace: ContinuousSpace,
                 timeStep: float,
                 equations: Sequence[str]
    ):
        self.stateSpace = stateSpace

        self.transitionFunction = TransitionFunction(stateSpace, 
                                                     controlSpace, 
                                                     disturbanceSpace, 
                                                     timeStep, 
                                                     equations)

        # Initialization of the currentState: when first created the model is put at position (0,...)
        startPos = [0 for _ in range(stateSpace.dimensions)]
        self.currentState = self.stateSpace.getCell(startPos)

    def changeCurrentState(self, coords: Sequence[float]) -> None:
        """
        Change the current state of the model after initialization.

        Takes the new coordinates, generates the new cell coordinates and create a cell object for it.

        Args:
            coords (Sequence[float]): the new coordinates where to put the model.

        Returns:
            None
        """
        validateDimensions(coords, self.stateSpace.dimensions)
        validatePointBounds(coords, self.stateSpace.bounds)

        cellCoords = self.stateSpace.getCellCoords(coords)
        self.currentState = self.stateSpace.getCell(cellCoords)
