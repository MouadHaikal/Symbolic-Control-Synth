from symControl.space.discreteSpace import DiscreteSpace
from symControl.space.continuousSpace import ContinuousSpace
from symControl.model.model import Model
from symControl.model.codePrinter import CodePrinter
from symControl.bindings import Automaton


stateSpace       = DiscreteSpace(2, [(0, 1), (0, 1)], [4, 4])
inputSpace       = DiscreteSpace(2, [(-1, 1), (-1, 1)], [2, 2])
disturbanceSpace = ContinuousSpace(1, [(-0.1, 0.1)])


model = Model(
    stateSpace=stateSpace,            
    controlSpace=inputSpace,        
    disturbanceSpace=disturbanceSpace,
    timeStep=0.1,
    equations=[
        "x1 + tau * (u1 + w1)",
        "x2 + tau * (u2 + w1)", 
    ]
)

if model.transitionFunction.isCooperative:
    print("====================== COOPERATIVE =====================")
else:
    print("====================== NON COOPERATIVE =====================")

printer = CodePrinter(model)
# print("__")
# print(printer.printCode())
# print("__")
automaton = Automaton(
    stateSpace,
    inputSpace,
    disturbanceSpace,
    printer.printCode()
)

# print(model.transitionFunction.symbolContext)
