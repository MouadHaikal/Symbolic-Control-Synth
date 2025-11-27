import sympy as sp
from scipy.optimize import minimize

from symControl.space.continuousSpace import ContinuousSpace
from symControl.space.discreteSpace import DiscreteSpace
from symControl.utils.validation import *
from symControl.utils.constants import *

 
class TransitionFunction:
    """
    Represents the transition dynamics for a controlled dynamical system with disturbance.

    Attributes:
        symbolContext (dict): Dictionary mapping system variable names to symbolic expressions.
        symbolBounds (dict): Bounds for each symbol variable in the system.
        dimensions (dict): Number of dimensions for state, control, and disturbance spaces.
        equations (Tuple[sympy.Expr, ...]): Parsed and time-step substituted symbolic transition equations.
        isCooperative (bool): Indicates if the system satisfies cooperative system properties.
        stateJac (sympy.Matrix): Jacobian matrix of the transition equations with respect to state variables.
        statejacGrad (tuple): Gradients of state Jacobian elements with respect to state, input, and disturbance variables.
        disturbJacUpper (tuple): Upper bound of the absolute disturbance Jacobian entries.
    """

    __slots__ = ['symbolContext', 'symbolBounds', 'dimensions', 'equations', 'isCooperative', 'stateJac', 'stateJacGrad', 'disturbJacUpper']

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

            TAU: sp.Symbol(TAU, positive=True),
        }

        self.symbolBounds = {
            **{f"{STATE}{i+1}": stateSpace.bounds[i] for i in range(stateSpace.dimensions)},
            **{f"{INPUT}{i+1}": controlSpace.bounds[i] for i in range(controlSpace.dimensions)},
            **{f"{DISTURBANCE}{i+1}": disturbanceSpace.bounds[i] for i in range(disturbanceSpace.dimensions)}
        }

        self.dimensions = {
            STATE: stateSpace.dimensions,
            INPUT: controlSpace.dimensions,
            DISTURBANCE: disturbanceSpace.dimensions,
        }

        parsedEquations = tuple(sp.parse_expr(eq, local_dict=self.symbolContext) for eq in equations)
        self.equations = sp.Matrix([expr.subs({self.symbolContext[TAU]: timeStep}).simplify() for expr in parsedEquations])

        self.isCooperative, self.stateJac, self.stateJacGrad, self.disturbJacUpper = self.__cooperativityAnalysis();


    def __cooperativityAnalysis(self):
        """
        Analyzes the transition equations to assess cooperativity by evaluating Jacobians and their partial derivatives,
        and if non-cooperative, computes additional bounds and gradients related to disturbance and state Jacobians.

        Returns:
            bool: True if the system is cooperative, False otherwise.
            sympy.Matrix | None: Jacobian matrix with respect to state variables, or None if cooperative.
            Tuple | None: Gradients of state Jacobian entries relative to state, input, and disturbance variables, or None if cooperative.
            Tuple: Upper bounds on absolute disturbance Jacobian entries, or empty tuple if cooperative.
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
            return True, None, None, tuple()


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


        return isCooperative, stateJac, stateJacGrad, disturbJacUpper
