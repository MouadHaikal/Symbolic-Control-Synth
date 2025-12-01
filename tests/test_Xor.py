from symControl.space.discreteSpace import DiscreteSpace
from symControl.space.continuousSpace import ContinuousSpace
from symControl.model.model import Model
from symControl.model.codePrinter import CodePrinter
from symControl.bindings import Automaton

# Define spaces
state = DiscreteSpace(3, ((0, 10), (0, 10), (-3.14, 3.14)), (60, 60, 30))
input = DiscreteSpace(2, ((0.25, 1), (-1, 1)), (3, 5))
disturbance = ContinuousSpace(3, ((0, 0.05), (0, 0.05), (0, 0.05)))

# Define equations
equations = [
    "x1 + tau * (u1 * cos(x3) + w1)",
    "x2 + tau * (u1 * sin(x3) + w2)",
    "(x3 + tau * (u2 + w3)) % (2 * 6.14)",
]

# Create model and generate code
model = Model(state, input, disturbance, 1, equations)
printer = CodePrinter(model)
codeToCompute = printer.printCode()

# Initialize Automaton
automata = Automaton(state, 
                     input, 
                     disturbance,
                     model.transitionFunction.isCooperative,
                     model.transitionFunction.disturbJacUpper,
                     codeToCompute)

# Define Regions based on the provided table
# R4: [3, 7] x [3, 7] x [-pi, pi] (Obstacle)
obstacleCoords = [
    state.getCellCoords([3, 3, -3.14]),
    state.getCellCoords([7, 7, 3.14]),
]
automata.applySecuritySpec(obstacleCoords[0], obstacleCoords[1])

print("Security Spec applied succesfully")

# R1: [4, 5] x [8.5, 9.5] x [-pi, pi] (First XOR region)
r1Coords = [
    state.getCellCoords([4, 8.5, -3.14]),
    state.getCellCoords([5, 9.5, 3.14]),
]

# R2: [8.5, 9.5] x [2, 3] x [-pi, pi] (Second XOR region)
r2Coords = [
    state.getCellCoords([8.5, 2, -3.14]),
    state.getCellCoords([9.5, 3, 3.14]),
]

# R3: [2, 3] x [0.5, 1.5] x [-pi, pi] (Target region)
targetCoords = [
    state.getCellCoords([2, 0.5, -3.14]),
    state.getCellCoords([3, 1.5, 3.14]),
]

# Start State: (0.5, 4.5, 0)
startState = state.getCellCoords([0.5, 4.5, 0])

# Get XOR Controller
# Arguments: startState, R1_lower, R1_upper, R2_lower, R2_upper, Target_lower, Target_upper, pathCount
paths = automata.getXORController(
    startState,
    r1Coords[0], r1Coords[1],
    r2Coords[0], r2Coords[1],
    targetCoords[0], targetCoords[1],
    1
)

print(paths)
