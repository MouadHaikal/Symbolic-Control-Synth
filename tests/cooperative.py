from symControl.space.discreteSpace import DiscreteSpace
from symControl.model.model import Model
from symControl.model.codePrinter import CodePrinter
from symControl.bindings import Automaton


stateSpace       = DiscreteSpace("State", 2, [(0, 1), (0, 1)], [10, 10])
controlSpace     = DiscreteSpace("Control", 1, [(-1, 1)], [10])
disturbanceSpace = DiscreteSpace("Disturbance", 2, [(-0.1, 0.1), (-0.1, 0.1)], [10, 10])


model = Model(
    stateSpace=stateSpace,            
    controlSpace=controlSpace,        
    disturbanceSpace=disturbanceSpace,
    timeStep=0.1,
    equations=[
        "x1 + tau * (u1 + w1)",
        "x2 + tau * (u1 + w2)"
    ]
)

printer = CodePrinter(model)
automata = Automaton(
    stateSpace,
    controlSpace,
    disturbanceSpace,
    printer.printFAtPoint()
)

print(model.transitionFunction.symbolContext)

