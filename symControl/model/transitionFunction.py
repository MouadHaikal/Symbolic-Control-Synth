import sympy as sp
import numpy as np

from scipy.optimize import minimize

from symControl.space.continuousSpace import ContinuousSpace
from symControl.space.discreteSpace import DiscreteSpace
from symControl.utils.validation import *
from symControl.utils.constants import *

 
class TransitionFunction:
    """
    Represents the transition dynamics for a controlled dynamical system with disturbance.

    Encapsulates the symbolic equations governing state evolution given
    discrete state, control, and disturbance spaces. Provides methods for
    numerical evaluation of transitions and for verifying cooperative system structure.

    Attributes:
        symbolContext (dict): Dictionary mapping system variable names to symbolic representations.
        timeStep (float): Integration time step value (TAU).
        dimensions (dict): Number of dimensions for state, control, and disturbance spaces.
        symbolBounds (dict): Bounds for each symbol variable in the system.
        equations (Tuple[sympy.Expr, ...]): Tuple of parsed transition equations.
        isCooperative (bool): Whether the system satisfies the cooperative property.
        stateJac (): TODO
        disturbJacUpper (): TODO
        _lambdaFunctions (List[callable]): List of lambda functions for evaluating equations numerically.
    """
    __slots__ = ['symbolContext', 'timeStep', 'symbolBounds', 'dimensions', 'equations', 'isCooperative', 'stateJac', 'stateJacGrad', 'disturbJacUpper', '_lambdaFunctions']

    def __init__(self, 
                 stateSpace:       DiscreteSpace, 
                 controlSpace:     DiscreteSpace, 
                 disturbanceSpace: ContinuousSpace,
                 timeStep:         float,
                 equations:        Sequence[str]
    ):
        validateDimensions(equations, stateSpace.dimensions)

        self.symbolContext = {
            **{f"{STATE}{i+1}": sp.Symbol(f"{STATE}{i+1}") 
                for i in range(stateSpace.dimensions)},

            **{f"{INPUT}{i+1}": sp.Symbol(f"{INPUT}{i+1}") 
                for i in range(controlSpace.dimensions)},

            **{f"{DISTURBANCE}{i+1}": sp.Symbol(f"{DISTURBANCE}{i+1}") 
                for i in range(disturbanceSpace.dimensions)},

            TAU: sp.Symbol(TAU, positive=True)
        }

        self.timeStep = timeStep  # Value of the TAU symbol

        self.dimensions = {
            STATE: stateSpace.dimensions,
            INPUT: controlSpace.dimensions,
            DISTURBANCE: disturbanceSpace.dimensions,
        }

        self.symbolBounds = {
            **{f"{STATE}{i+1}": stateSpace.bounds[i] for i in range(stateSpace.dimensions)},
            **{f"{INPUT}{i+1}": controlSpace.bounds[i] for i in range(controlSpace.dimensions)},
            **{f"{DISTURBANCE}{i+1}": disturbanceSpace.bounds[i] for i in range(disturbanceSpace.dimensions)}
        }

        self.equations = tuple(sp.parse_expr(eq, local_dict=self.symbolContext) for eq in equations)
        self.equations = sp.Matrix([expr.subs({self.symbolContext[TAU]: self.timeStep}).simplify() for expr in self.equations])

        self.isCooperative, self.stateJac, self.stateJacGrad, self.disturbJacUpper = self.__cooperativeCheck();

        self._lambdaFunctions = [sp.lambdify(self.symbolContext.keys(), eq, modules="numpy") for eq in self.equations]


    def evaluate(self, state: Sequence[float], control: Sequence[float], disturbance: Sequence[float]) -> Tuple[float, ...]:
        """
        Evaluates the system's transition equations at the given state, control, and disturbance inputs.

        Concatenates all input vectors and time step, then applies each symbolic transition equation
        to compute the next system state numerically using precompiled lambda functions.

        Args:
            state (Sequence[float]): Current state vector.
            control (Sequence[float]): Control input vector.
            disturbance (Sequence[float]): Disturbance input vector.

        Returns:
            Tuple[float, ...]: The numerically evaluated next state as a tuple of floats.

        Raises:
            ValueError: If the state, control, or disturbance vectors do not have the expected dimensions.
        """
        validateDimensions(state, self.dimensions[STATE])
        validateDimensions(control, self.dimensions[INPUT])
        validateDimensions(disturbance, self.dimensions[DISTURBANCE])

        inputContext = np.concatenate((state, control, disturbance))
        inputContext = np.append(inputContext, self.timeStep)
        
        return tuple(func(*inputContext) for func in self._lambdaFunctions)


    # to be implemented
    # TODO: evaluate(self, state: ContinuousSpace, control: Sequence[float], distrubance: Sequence[float]) -> ContinuousSpace

    def __cooperativeCheck(self):
        """
        Checks whether the transition equations satisfy the cooperative system property.

        Computes the Jacobian matrices of all transition functions with respect to state
        and disturbance variables, then minimizes each partial derivative over allowed bounds.
        Returns True only if all minimum partial derivatives are non-negative, indicating
        cooperation with respect to state and disturbance.

        Returns:
            bool: True if the system is cooperative, False otherwise.
        """
        stateJac = sp.Matrix([
            [
                sp.diff(eq, self.symbolContext[f"{STATE}{i+1}"])
                    for i in range(self.dimensions[STATE])
            ] 
            for eq in self.equations
        ])

        disturbJac = sp.Matrix([
            [
                sp.diff(eq, self.symbolContext[f"{DISTURBANCE}{i+1}"]) 
                    for i in range(self.dimensions[DISTURBANCE])
            ] 
            for eq in self.equations
        ])


        def minVal(expr):
            if expr.is_number:
                return expr.evalf()

            vars = sorted(
                [v for v in expr.free_symbols],
                key=lambda x: x.name
            )

            lmbd = sp.lambdify(vars, expr, "numpy")

            def func(x):
                return lmbd(*x)

            varBounds = [self.symbolBounds[var.name] for var in vars]
            x0        = [(bd[1] + bd[0]) / 2 for bd in varBounds]

            result = minimize(func, x0, method="L-BFGS-B", bounds=varBounds)

            return result.fun

        def maxVal(expr):
            return -minVal(-expr)



        isCooperative = (
            all(minVal(stateJac[i,j]) >= 0.0 
                for i in range(stateJac.rows) 
                for j in range(stateJac.cols)
            ) 
        and 
            all(minVal(disturbJac[i,j]) >= 0.0 
                for i in range(disturbJac.rows) 
                for j in range(disturbJac.cols)
            )
        )

        if isCooperative:
            return True, None, None, None


        disturbJacUpper = tuple(
            tuple( 
                float(maxVal(sp.Abs(disturbJac[i,j])))
                for j in range(disturbJac.cols)
            )
            for i in range(disturbJac.rows)
        )


        stateJacGrad = tuple(
            tuple(
                {
                    STATE: tuple(sp.diff(stateJac[i,j], self.symbolContext[f"{STATE}{k+1}"]) 
                        for k in range(self.dimensions[STATE])
                    ),

                    INPUT: tuple(sp.diff(stateJac[i,j], self.symbolContext[f"{INPUT}{k+1}"])
                        for k in range(self.dimensions[INPUT])      
                    ),

                    DISTURBANCE: tuple(sp.diff(stateJac[i,j], self.symbolContext[f"{DISTURBANCE}{k+1}"])
                        for k in range(self.dimensions[DISTURBANCE])
                    )
                } for j in range(stateJac.cols)
            ) for i in range(stateJac.rows)
        )

        print("stateJac:")
        print(stateJac)
        print("disturbJac:")
        print(disturbJac)
        print("disturbJacUpper:")
        print(disturbJacUpper)
        print("stateJacGrad:")
        print(stateJacGrad)


        return isCooperative, stateJac, stateJacGrad, disturbJacUpper
