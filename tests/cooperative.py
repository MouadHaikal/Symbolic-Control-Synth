from symControl.model.transitionFunction import TransitionFunction
from symControl.space.discreteSpace import DiscreteSpace


stateSpace       = DiscreteSpace("State", 2, [(0, 1), (0, 1)], [10, 10])
controlSpace     = DiscreteSpace("Control", 1, [(-1, 1)], [10])
disturbanceSpace = DiscreteSpace("Disturbance", 2, [(-0.1, 0.1), (-0.1, 0.1)], [10, 10])

transFunc = TransitionFunction(
    stateSpace=stateSpace,            
    controlSpace=controlSpace,        
    disturbanceSpace=disturbanceSpace,
    timeStep=0.1,
    equations=[
        "x1 + tau * (u1 + w1)",
        "x2 + tau * (u2 + w2)"
    ]
)

print(f"Model: {transFunc.equations}")
print(f"isCooperative: {transFunc.isCooperative}")
