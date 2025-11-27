from symControl.space.discreteSpace import DiscreteSpace
from symControl.space.continuousSpace import ContinuousSpace
from symControl.model.model import Model
from symControl.model.codePrinter import CodePrinter
from symControl.bindings import Automaton


stateSpace       = DiscreteSpace(2, [(0, 1), (0, 1)], [10, 10])
inputSpace       = DiscreteSpace(2, [(-1, 1), (-1, 1)], [3, 5])
disturbanceSpace = ContinuousSpace(1, [(0.0, 0.05)])


model = Model(
    stateSpace=stateSpace,            
    controlSpace=inputSpace,        
    disturbanceSpace=disturbanceSpace,
    timeStep=1,
    equations=[
        "(x1**3)/3 + tau * (u1**2 + w1)",
        "sin(x2) + tau * (u2 + w1**2)", 
    ]
)

if model.transitionFunction.isCooperative:
    print("====================== COOPERATIVE =====================")
else:
    print("====================== NON COOPERATIVE =====================")

print(model.transitionFunction.equations)
printer = CodePrinter(model)

# print(printer.printCode())

automaton = Automaton(
    stateSpace,
    inputSpace,
    disturbanceSpace,
    model.transitionFunction.isCooperative,
    model.transitionFunction.disturbJacUpper,
    printer.printCode()
)

# print(model.transitionFunction.symbolContext)
