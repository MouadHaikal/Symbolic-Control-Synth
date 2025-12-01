from symControl.space.discreteSpace import DiscreteSpace;
from symControl.space.continuousSpace import ContinuousSpace;
from symControl.model.model import Model;
from symControl.model.codePrinter import CodePrinter;
from symControl.bindings import Automaton;


state = DiscreteSpace(3, ((0, 10), (0, 10), (-3.14, 3.14)), (100, 100, 30));
input = DiscreteSpace(2, ((0.25, 1), (-1, 1)), (3, 5));
disturbance = ContinuousSpace(3, ((0, 0.05), (0, 0.05), (0, 0.05)));


equations = [
    "x1 + tau * (u1 * cos(x3) + w1)",
    "x2 + tau * (u1 * sin(x3) + w2)",
    "(x3 + tau * (u2 + w3)) % (2 * 3.14)",
]

model = Model(state, input, disturbance, 1, equations)
printer = CodePrinter(model)


automata = Automaton(state, 
                     input, 
                     disturbance,
                     model.transitionFunction.isCooperative,
                     model.transitionFunction.disturbJacUpper,
                     printer.printCode()
)

obstacle1 = [
    state.getCellCoords([6.0, 6.0, -3.14]),
    state.getCellCoords([8.0, 7.5, 3.14]),
]
obstacle2 = [
    state.getCellCoords([6.0, 1.2, -3.14]),
    state.getCellCoords([8.5, 2.4, 3.14]),
]

automata.applySecuritySpec(obstacle1[0], obstacle1[1])
automata.applySecuritySpec(obstacle2[0], obstacle2[1])

startState = state.getCellCoords([6.8, 9.0, 0.0])

target = [
    state.getCellCoords([2.0, 0.5, -3.14]),
    state.getCellCoords([3.0, 1.5, 3.14]),
]

paths = automata.getController(startState, target[0], target[1], 50)
print("Paths: ", paths)


