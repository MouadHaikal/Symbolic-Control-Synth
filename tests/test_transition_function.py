import pytest
import numpy as np
import sympy as sp
from symControl.space.discreteSpace import DiscreteSpace
from symControl.model.transitionFunction import TransitionFunction
from symControl.logger import gLogger

# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def simple_spaces():
    """
    Fixture providing a simple 2D state, 1D control, 1D disturbance space with resolutions matching bounds.
    """
    gLogger.info("Creating simple state, control, disturbance spaces")
    state_bounds = [(0, 1), (0, 1)]
    control_bounds = [(0, 1)]
    disturbance_bounds = [(0, 1)]
    
    state_res = [10, 10]
    control_res = [10]
    disturbance_res = [10]
    
    state = DiscreteSpace("State", 2, state_bounds, state_res)
    control = DiscreteSpace("Control", 1, control_bounds, control_res)
    disturbance = DiscreteSpace("Disturbance", 1, disturbance_bounds, disturbance_res)
    return state, control, disturbance


# -----------------------
# Tests
# -----------------------

def test_initialization_and_parsing(simple_spaces):
    """
    Test initialization of TransitionFunction and correct equation parsing.
    """
    gLogger.debug("Testing initialization and parsing")
    state, control, disturb = simple_spaces
    tf = TransitionFunction(state, control, disturb, 0.1, ["x1 + tau*u1", "x2 + tau*w1"])

    # Check dimensions
    assert tf.dimensions["x"] == 2
    assert tf.dimensions["u"] == 1
    assert tf.dimensions["w"] == 1

    # Check symbols exist
    for s in ["x1", "x2", "u1", "w1", "tau"]:
        assert s in tf.symbolContext

    # Equations parsed
    assert all(isinstance(eq, sp.Expr) for eq in tf.equations)
    gLogger.info("Initialization and parsing test passed")


def test_evaluate_linear_system(simple_spaces):
    """
    Test numerical evaluation of simple linear transition equations.
    """
    gLogger.debug("Testing linear system evaluation")
    state, control, disturb = simple_spaces
    tf = TransitionFunction(state, control, disturb, 0.5, ["x1 + tau*u1", "x2 + tau*w1"])

    result = tf.evaluate([1, 2], [3], [4])
    expected = (1 + 0.5 * 3, 2 + 0.5 * 4)
    assert np.allclose(result, expected)
    gLogger.info("Linear system evaluation test passed")


def test_invalid_state_dimension(simple_spaces):
    """
    Ensure that providing wrong state dimension raises ValueError.
    """
    gLogger.debug("Testing invalid state dimension")
    state, control, disturb = simple_spaces
    tf = TransitionFunction(state, control, disturb, 0.1, ["x1 + u1", "x2 + w1"])

    with pytest.raises(ValueError):
        tf.evaluate([1], [0], [0])
    gLogger.info("Invalid state dimension test passed")


def test_invalid_control_dimension(simple_spaces):
    """
    Ensure that providing wrong control dimension raises ValueError.
    """
    gLogger.debug("Testing invalid control dimension")
    state, control, disturb = simple_spaces
    tf = TransitionFunction(state, control, disturb, 0.1, ["x1 + u1", "x2 + w1"])

    with pytest.raises(ValueError):
        tf.evaluate([1, 2], [0, 1], [0])
    gLogger.info("Invalid control dimension test passed")


def test_invalid_disturbance_dimension(simple_spaces):
    """
    Ensure that providing wrong disturbance dimension raises ValueError.
    """
    gLogger.debug("Testing invalid disturbance dimension")
    state, control, disturb = simple_spaces
    tf = TransitionFunction(state, control, disturb, 0.1, ["x1 + u1", "x2 + w1"])

    with pytest.raises(ValueError):
        tf.evaluate([1, 2], [0], [0, 1])
    gLogger.info("Invalid disturbance dimension test passed")


def test_cooperative_system(simple_spaces):
    """
    Check that a cooperative system is correctly identified.
    """
    gLogger.debug("Testing cooperative system detection")
    state, control, disturb = simple_spaces
    tf = TransitionFunction(
        state, control, disturb, 0.1, ["x1 + tau*(u1 + w1)", "x2 + tau*(u1 + w1)"]
    )
    assert tf.isCooperative is True
    gLogger.info("Cooperative system test passed")


def test_non_cooperative_system(simple_spaces):
    """
    Check that a non-cooperative system is correctly identified.
    """
    gLogger.debug("Testing non-cooperative system detection")
    state, control, disturb = simple_spaces
    tf = TransitionFunction(
        state, control, disturb, 0.1, ["x1 - tau*u1", "x2 - tau*w1"]
    )
    assert tf.isCooperative is False
    gLogger.info("Non-cooperative system test passed")


def test_nonlinear_equations(simple_spaces):
    """
    Test evaluation of nonlinear symbolic equations.
    """
    gLogger.debug("Testing nonlinear system evaluation")
    state, control, disturb = simple_spaces
    tf = TransitionFunction(
        state,
        control,
        disturb,
        0.1,
        ["x1 + tau*(x1*u1 + w1**2)", "x2 + tau*(sin(x2) + u1)"],  
    )

    result = tf.evaluate([1, np.pi / 2], [2], [1])
    expected_x1 = 1 + 0.1 * (1*2 + 1**2)
    expected_x2 = np.pi / 2 + 0.1 * (np.sin(np.pi/2) + 2)
    assert np.allclose(result, (expected_x1, expected_x2), atol=1e-3)
    gLogger.info("Nonlinear equations test passed")


def test_constant_equation(simple_spaces):
    """
    Check that constant equations are evaluated correctly.
    """
    gLogger.debug("Testing constant equations")
    state, control, disturb = simple_spaces
    tf = TransitionFunction(state, control, disturb, 0.1, ["5", "10"])

    result = tf.evaluate([0, 0], [0], [0])
    assert result == (5, 10)
    gLogger.info("Constant equation test passed")


def test_single_state_system():
    """
    Test a 1D system evaluation.
    """
    gLogger.debug("Testing single-state 1D system")
    state = DiscreteSpace("State", 1, [(0, 1)], [10])
    control = DiscreteSpace("Control", 1, [(0, 1)], [10])
    disturb = DiscreteSpace("Disturbance", 1, [(0, 1)], [10])

    tf = TransitionFunction(state, control, disturb, 1.0, ["x1 + tau*u1 + tau*w1"])
    result = tf.evaluate([1], [2], [3])
    assert np.isclose(result[0], 1 + 1*(2 + 3))
    gLogger.info("Single-state system test passed")
