from symControl.space.discreteSpace import DiscreteSpace;
from symControl.space.continuousSpace import ContinuousSpace;
from symControl.model.transitionFunction import TransitionFunction;
from symControl.model.model import Model;
from symControl.model.codePrinter import CodePrinter;
from symControl.bindings import Automaton;

state = DiscreteSpace(3, ((0, 10), (0, 10), (-3.14, 3.14)), (100, 100, 30));
input = DiscreteSpace(2, ((0.25, 1), (-1, 1)), (3, 5));
disturbance = ContinuousSpace(3, ((0, 0.05), (0, 0.05), (0, 0.05)));

equations = [
    "x1 + tau * (u1 * cos(x3) + w1)",
    "x2 + tau * (u1 * sin(x3) + w2)",
    "(x3 + tau * (u2 + w3)) % (2 * 6.14)",
]

model = Model(state, input, disturbance, 1, equations)
printer = CodePrinter(model)

codeToCompute = printer.printCode()
print(codeToCompute)

automata = Automaton(state, 
                     input, 
                     disturbance,
                     model.transitionFunction.isCooperative,
                     model.transitionFunction.disturbJacUpper,
                     codeToCompute)

unsafeCoords = [
    state.getCellCoords([3, 3, -3.14]),
    state.getCellCoords([7, 7, +3.14]),
]

automata.applySecuritySpec(unsafeCoords[0], unsafeCoords[1])

startState = state.getCellCoords([0.5, 4.5, 0])

targetCoords = [
    state.getCellCoords([4, 8.5, -3.14]),
    state.getCellCoords([5, 9.5, 3.14]),
]

paths = automata.getController(startState, targetCoords[0], targetCoords[1], 1)
print(paths)


