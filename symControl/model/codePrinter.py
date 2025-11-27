import sympy as sp
from sympy.printing.c import C11CodePrinter
from typing import override

from symControl.model.model import Model
from symControl.utils.cudaCodeTemplates import *
from symControl.utils.constants import *


class CodePrinter(C11CodePrinter):
    """
    This printer customizes symbol printing by mapping model variable names to indexed variables
    and generates C code according to whether the model's transition function is cooperative or not.

    Attributes:
        model (Model): The model containing transition functions and symbol context.
        __exprIdx (int): Internal index tracking the current expression being printed.
        __stateVar (str): Variable name for state variables used in generated code.
        __inputVar (str): Variable name for input variables used in generated code.
        __disturbanceVar (str): Variable name for disturbance variables used in generated code.
    """

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
        """
        Overrides symbol printing to map symbols found in the model's transition context 
        to appropriately indexed variable names.
        
        Args:
            expr: A symbolic expression representing a symbol.
        
        Returns:
            str: The printed representation of the symbol.
        """
        if expr.name in self.model.transitionFunction.symbolContext.keys():
            return self._print(self.__mapSymbolName(expr.name))

        return super()._print_Symbol(expr)


    def printCode(self):
        """
        Generates C code based on whether the model's transition function is cooperative.

        Returns:
            str: The generated C code as a string.
        """
        if self.model.transitionFunction.isCooperative:
            return self.__printCoopCode()
        else:
            return self.__printNonCoopCode()


    def __mapSymbolName(self, symName: str) -> str:
        """
        Maps a symbol name to its corresponding indexed variable name.
        
        Args:
            symName (str): The symbol name, prefixed by type identifier (STATE, INPUT, or DISTURBANCE).
        
        Returns:
            str: The mapped variable name with an appropriate index for code generation.
        
        Raises:
            ValueError: If the symbol name does not start with a recognized prefix.
        """
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
        """
        Generates C code for cooperative transition functions using the model's equations.
        
        Returns:
            str: Formatted C code for cooperative models.
        """
        self.__stateVar       = "state"
        self.__inputVar       = "input"
        self.__disturbanceVar = "disturbance"

        self.__exprIdx = 0

        fAtPointOut = sp.MatrixSymbol("fAtPointOut", self.model.stateSpace.dimensions, 1)
        fAtPointCode = self.doprint(self.model.transitionFunction.equations, assign_to=fAtPointOut)

        return coopCodeTemplate.format(
            fAtPointCode=fAtPointCode
        )

    def __printNonCoopCode(self):
        """
        Generates C code for non-cooperative transition functions, including evaluation of the
        state Jacobian and its gradients.
        
        Returns:
            str: Formatted C code for non-cooperative models.
        """
        self.__stateVar       = "state"
        self.__inputVar       = "input"
        self.__disturbanceVar = "disturbance"

        self.__exprIdx = 0

        fAtPointOut = sp.MatrixSymbol("fAtPointOut", self.model.stateSpace.dimensions, 1)
        fAtPointCode = self.doprint(self.model.transitionFunction.equations, assign_to=fAtPointOut)


        self.__stateVar       = "states"
        self.__inputVar       = "inputs"
        self.__disturbanceVar = "disturbances"

        self.__exprIdx = 0

        stateDim   = self.model.transitionFunction.dimensions[STATE];
        inputDim   = self.model.transitionFunction.dimensions[INPUT];
        disturbDim = self.model.transitionFunction.dimensions[DISTURBANCE];

        stateJacAtPointCode = ""
        for i in range(stateDim):
            for j in range(stateDim):
                self.__exprIdx = i*stateDim + j

                stateJacAtPointCode += f"{self.doprint(
                    self.model.transitionFunction.stateJac[i,j], 
                    assign_to=f'stateJacAtPointOut[{self.__exprIdx}]'
                )}\n"


        stateJacGradAtPointCode = ""
        for i in range(stateDim):
            for j in range(stateDim):
                self.__exprIdx = i*stateDim + j

                for k in range(stateDim):
                    stateJacGradAtPointCode += f"{self.doprint(
                        self.model.transitionFunction.stateJacGrad[i][j][STATE][k],
                        assign_to=f"outStates[{self.__exprIdx*stateDim + k}]"
                    )}\n"

                for k in range(inputDim):
                    stateJacGradAtPointCode += f"{self.doprint(
                        self.model.transitionFunction.stateJacGrad[i][j][INPUT][k],
                        assign_to=f"outInputs[{self.__exprIdx*inputDim + k}]"
                    )}\n"

                for k in range(disturbDim):
                    stateJacGradAtPointCode += f"{self.doprint(
                        self.model.transitionFunction.stateJacGrad[i][j][DISTURBANCE][k],
                        assign_to=f"outDisturbances[{self.__exprIdx*disturbDim + k}]"
                    )}\n"

        
        return nonCoopCodeTemplate.format(
            fAtPointCode=fAtPointCode,
            stateJacAtPointCode=stateJacAtPointCode,
            stateJacGradAtPointCode=stateJacGradAtPointCode
        )

