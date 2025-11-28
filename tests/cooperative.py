from symControl.space.discreteSpace import DiscreteSpace
from symControl.space.continuousSpace import ContinuousSpace
from symControl.model.model import Model
from symControl.model.codePrinter import CodePrinter
from symControl.bindings import Automaton


stateSpace       = DiscreteSpace(2, [(0, 300), (0, 300)], [300, 300])
inputSpace       = DiscreteSpace(2, [(-1, 1), (-1, 1)], [2, 2])
disturbanceSpace = ContinuousSpace(2, [(0.5, 0.7), (0.6, 0.7)])


model = Model(
    stateSpace=stateSpace,            
    inputSpace=inputSpace,        
    disturbanceSpace=disturbanceSpace,
    timeStep=1,
    equations=[
        "x1 + tau * (u1 + w1)",
        "x2 + tau * (u2 + w2)", 
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

# automaton.applySecuritySpec((3, 3), (3, 3))

controller = automaton.getController(0, (10,10), (30,30))
print(controller)

# print(model.transitionFunction.symbolContext)
