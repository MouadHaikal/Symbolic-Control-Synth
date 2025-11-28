from symControl.utils.validation import *
from symControl.utils.constants import *
from symControl.space.continuousSpace import ContinuousSpace
from symControl.space.discreteSpace import DiscreteSpace
from symControl.model.transitionFunction import TransitionFunction


class Model:
    """
    Represents a dynamical system model defined over discrete and continuous spaces,
    with symbolic transition functions governing its behavior.

    Attributes:
        stateSpace (DiscreteSpace): Defines the discrete state space in which the model operates.
        transitionFunction (TransitionFunction): Symbolic representation of the system's transition dynamics.
    """

    __slots__ = ['stateSpace', 'transitionFunction']

    
    def __init__(self,
                 stateSpace: DiscreteSpace,
                 inputSpace: DiscreteSpace,
                 disturbanceSpace: ContinuousSpace,
                 timeStep: float,
                 equations: Sequence[str]
    ):
        self.stateSpace = stateSpace

        self.transitionFunction = TransitionFunction(stateSpace, 
                                                     inputSpace, 
                                                     disturbanceSpace, 
                                                     timeStep, 
                                                     equations
        )
