from typing import override
from symControl.utils.constants import *
from symControl.utils.cudaCodeTemplates import *
from symControl.model.model import Model
from sympy.printing.c import C11CodePrinter
import sympy as sp


class CodePrinter(C11CodePrinter):
    __slots__ = ['model', '__exprId', '__stateVar', '__inputVar', '__disturbanceVar']

    def __init__(self, model: Model):
        super().__init__()
        self.model            = model

        self.__exprIdx        = 0
        self.__stateVar       = "state"
        self.__inputVar       = "input"
        self.__disturbanceVar = "disturbance"



    @override
    def _print_Symbol(self, expr):
        if expr.name in self.model.transitionFunction.symbolContext.keys():
            return self._print(self.__mapSymbolName(expr.name))

        return super()._print_Symbol(expr)


    def printCode(self):
        if self.model.transitionFunction.isCooperative:
            return self.__printCoopCode()
        else:
            return self.__printNonCoopCode()



    
    def __mapSymbolName(self, symName: str) -> str:
        if symName[0] == STATE:
            return f"{self.__stateVar}[{self.model.transitionFunction.dimensions[STATE] * self.__exprIdx + 
                    int(symName[1:]) - 1}]"

        elif symName[0] == INPUT:
            return f"{self.__inputVar}[{self.model.transitionFunction.dimensions[INPUT] * self.__exprIdx + 
                    int(symName[1:]) - 1}]"

        elif symName[0] == DISTURBANCE:
            return f"{self.__disturbanceVar}[{self.model.transitionFunction.dimensions[DISTURBANCE] * self.__exprIdx + 
                    int(symName[1:]) - 1}]"

        else:
            raise ValueError(f"Unknown symbol name: {symName}")

    def __printCoopCode(self):
        self.__stateVar = "state"
        self.__inputVar = "input"
        self.__disturbanceVar = "disturbance"

        fAtPointOut = sp.MatrixSymbol("fAtPointOut", self.model.stateSpace.dimensions, 1)
        self.__exprIdx = 0
        fAtPointCode = self.doprint(self.model.transitionFunction.equations, assign_to=fAtPointOut)

        return coopCodeTemplate.format(
            fAtPointCode=fAtPointCode
        )

    def __printNonCoopCode(self):
        self.__stateVar = "states"
        self.__inputVar = "inputs"
        self.__disturbanceVar = "disturbances"


        fAtPointOut = sp.MatrixSymbol("fAtPointOut", self.model.stateSpace.dimensions, 1)

        self.__exprIdx = 0
        fAtPointCode = self.doprint(self.model.transitionFunction.equations, assign_to=fAtPointOut)


        nx = self.model.transitionFunction.dimensions[STATE];
        nu = self.model.transitionFunction.dimensions[INPUT];
        nw = self.model.transitionFunction.dimensions[DISTURBANCE];


        stateJacAtPointCode = ""

        for i in range(nx):
            for j in range(nx):
                self.__exprIdx = i*nx + j
                stateJacAtPointCode += f"{self.doprint(
                    self.model.transitionFunction.stateJac[i,j], 
                    assign_to=f'stateJacAtPointOut[{self.__exprIdx}]'
                )}\n"


        stateJacGradAtPointCode = ""
        
        for i in range(nx):
            for j in range(nx):
                self.__exprIdx = i*nx + j

                for k in range(nx):
                    stateJacGradAtPointCode += f"{self.doprint(
                        self.model.transitionFunction.stateJacGrad[i][j][STATE][k],
                        assign_to=f"outStates[{self.__exprIdx*nx + k}]"
                    )}\n"

                for k in range(nu):
                    stateJacGradAtPointCode += f"{self.doprint(
                        self.model.transitionFunction.stateJacGrad[i][j][INPUT][k],
                        assign_to=f"outInputs[{self.__exprIdx*nu + k}]"
                    )}\n"

                for k in range(nw):
                    stateJacGradAtPointCode += f"{self.doprint(
                        self.model.transitionFunction.stateJacGrad[i][j][DISTURBANCE][k],
                        assign_to=f"outDisturbances[{self.__exprIdx*nw + k}]"
                    )}\n"
                

        
        return nonCoopCodeTemplate.format(
            fAtPointCode=fAtPointCode,
            stateJacAtPointCode=stateJacAtPointCode,
            stateJacGradAtPointCode=stateJacGradAtPointCode
        )

