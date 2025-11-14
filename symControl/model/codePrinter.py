from symControl import model, utils
from symControl.utils.constants import *
from symControl.utils.cudaCodeTemplates import *
from symControl.model.model import Model
from sympy.printing.c import C11CodePrinter
import sympy as sp

class CodePrinter(C11CodePrinter):
    __slots__ = ['model']

    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    def _print_Symbol(self, expr):
        def mapSymbolName(symName: str) -> str:
            if symName[0] == STATE:
                return f"state[{int(symName[1:]) - 1}]"
            elif symName[0] == CONTROL:
                return f"input[{int(symName[1:]) - 1}]"
            elif symName[0] == DISTURBANCE:
                return f"disturbance[{int(symName[1:]) - 1}]"
            else:
                raise ValueError(f"Unknown symbol name: {symName}")

        if expr.name in self.model.transitionFunction.symbolContext.keys():
            return self._print(mapSymbolName(expr.name))

        return super()._print_Symbol(expr)


    def printFAtPoint(self):
        nextState = sp.MatrixSymbol("nextState", self.model.stateSpace.dimensions, 1)
        return codeTemplate_f_at_Point.format(
            code=self.doprint(self.model.transitionFunction.equations, assign_to=nextState)
        )
