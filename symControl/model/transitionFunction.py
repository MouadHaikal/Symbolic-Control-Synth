import sympy as sp
import numpy as np
from symControl.space.discreteSpace import DiscreteSpace

SYMBOLS = {
    'tau': sp.Symbol('tau'),
    'cos': sp.cos,
    'sin': sp.sin,
    'tan': sp.tan,
    'exp': sp.exp,
    'sqrt': sp.sqrt,
    'log': sp.log,
    'pi': sp.pi,
    'e': sp.E
}

class EquationParser:
    _instance = None

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = cls(SYMBOLS)
        return cls._instance


    def __init__(self, global_symbols: dict=SYMBOLS):
        self.global_symbols = global_symbols        
    
    def parse_equation(self, equation_str: str, local_symbols: dict) -> sp.Expr:
        """
        Parses a single equation string into a SymPy expression.
        Combines global symbols (like math functions) and local symbols (state, control, disturbance).
        """
        all_symbols = {**self.global_symbols, **local_symbols}

        try:
            return sp.parse_expr(equation_str, local_dict=all_symbols)
        except Exception as e:
            raise ValueError(f"Error parsing equation '{equation_str}': {e}")


class TransitionFunction:
    def __init__(self, stateSpace: DiscreteSpace, 
                 controlSpace: DiscreteSpace, 
                 disturbanceSpace: DiscreteSpace,
                 timeStep: float,
                 equations: list[str]):
        """
        """

        self.timeStep = timeStep # tau is used to conform with our symbols

        self.symbols = {
            "x": self.createSpaceSymbols("x", stateSpace.dimensions),
            "u": self.createSpaceSymbols("u", controlSpace.dimensions),
            "w": self.createSpaceSymbols("w", disturbanceSpace.dimensions),
        }

        self.dimensions = {
            "x": stateSpace.dimensions,
            "u": controlSpace.dimensions,
            "w": disturbanceSpace.dimensions,
        }

        self.bounds = {
            stateSpace.name: (stateSpace.lowerBounds, stateSpace.upperBounds),
            disturbanceSpace.name: (disturbanceSpace.lowerBounds, disturbanceSpace.upperBounds),
        }

        self.checkDimension(equations, stateSpace.dimensions)

        # self.equations is now a list of sympy expressions
        self.equations = self.parseEquations(equations)


    def checkDimension(self, input, dimension):
        if len(input) != dimension:
            raise ValueError(f"Number of equations ({len(input)}) must dimensions ({dimension}).")


    def createSpaceSymbols(self, name, dimensions) -> tuple:
        symbols = sp.symbols(f"{name}(1:{dimensions+1})")
        if not isinstance(symbols, tuple):
            symbols = (symbols,)
        return symbols

    def parseEquations(self, equations: list[str]):
        local_symbols_for_parser = {}
        for key, sym_tuple in self.symbols.items():
            for i, sym in enumerate(sym_tuple):
                local_symbols_for_parser[f"{key}{i+1}"] = sym
        
        equationParser = EquationParser.getInstance()

        parsed_equations = []
        for eq in equations:
            parsed_equations.append(equationParser.parse_equation(eq, 
                                                                  local_symbols=local_symbols_for_parser))

        # to enable efficient computations we can convert them to lamba functions which are backed by the computation efficieny of numpy
        self._lambda_functions = self.lambdify_functions(parsed_equations)

        return parsed_equations

    def lambdify_functions(self, parsed_equations):
        """
        It belongs to the transitionFunction class because it is not the responsibility of the parser to lambdify the functions
        Creates lambdified functions for each parsed equation, allowing efficient numerical evaluation.
        """
        all_variables = []
        # Order matters for lambdify, so we'll collect them in a consistent way
        all_variables.extend(self.symbols["x"])
        all_variables.extend(self.symbols["u"])
        all_variables.extend(self.symbols["w"])
        all_variables.append(SYMBOLS['tau'])
        
        lambdified_funcs = []
        for eq in parsed_equations:
            lambdified_funcs.append(sp.lambdify(all_variables, eq, modules='numpy'))
        return lambdified_funcs

    def evaluate(self, current_state, control_input, disturbance_input):

        self.checkDimension(current_state, self.dimensions["x"])
        self.checkDimension(control_input, self.dimensions["u"])
        self.checkDimension(disturbance_input, self.dimensions["w"])

        all_inputs = np.concatenate((current_state, control_input, disturbance_input))
        all_inputs = np.append(all_inputs, self.timeStep)
        
        results = [func(*all_inputs) for func in self._lambda_functions]
        return results

