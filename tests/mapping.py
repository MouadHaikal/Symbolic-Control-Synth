from symControl.model.model import Model
from symControl.space.continuousSpace import ContinuousSpace
from symControl.space.discreteSpace import DiscreteSpace


stateSpace       = DiscreteSpace("State", 2, [(0, 1), (0, 1)], [10, 10])
controlSpace     = DiscreteSpace("Control", 1, [(-1, 1)], [10])
disturbanceSpace = DiscreteSpace("Disturbance", 2, [(-0.1, 0.1), (-0.1, 0.1)], [10, 10])

equations=[
        "x1 + tau * (u1 + w1)",
        "x2 + tau * (u2 + w2)"
]

model = Model(stateSpace, controlSpace, disturbanceSpace, 1.0, equations)

space = ContinuousSpace("testing", 2, [(0.1, 0.25), (0.1, 0.25)])
print(model.getNextStates(space))
