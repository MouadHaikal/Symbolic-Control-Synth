import pytest
import numpy as np
import sympy as sp
from symControl.space.discreteSpace import DiscreteSpace
from symControl.model.transitionFunction import TransitionFunction

# ============================================================
# Basic tests for TransitionFunction
# ============================================================

class TestTransitionFunctionBasic:
    """Basic tests for TransitionFunction, checking parsing, evaluation, and numerical correctness."""

    # -------------------------------
    # Fixtures
    # -------------------------------
    @pytest.fixture
    def state_space(self):
        """Provides a 2D state space for testing.

        Returns:
            DiscreteSpace: A state space with 2 dimensions, lower bounds [0,0], upper bounds [10,10],
                           and grid [5,5].
        """
        return DiscreteSpace("State", 2, [0, 0], [10, 10], [5, 5])

    @pytest.fixture
    def control_space(self):
        """Provides a 1D control space for testing.

        Returns:
            DiscreteSpace: A control space with 1 dimension, lower bound [0], upper bound [1], grid [2].
        """
        return DiscreteSpace("Control", 1, [0], [1], [2])

    @pytest.fixture
    def disturbance_space(self):
        """Provides a 1D disturbance space for testing.

        Returns:
            DiscreteSpace: A disturbance space with 1 dimension, lower bound [0], upper bound [1], grid [2].
        """
        return DiscreteSpace("Disturbance", 1, [0], [1], [2])

    # -------------------------------
    # Tests
    # -------------------------------
    def test_invalid_number_of_equations(self, state_space, control_space, disturbance_space):
        """Test that ValueError is raised if number of equations != state dimensions.

        Args:
            state_space (DiscreteSpace): The state space fixture.
            control_space (DiscreteSpace): The control space fixture.
            disturbance_space (DiscreteSpace): The disturbance space fixture.
        """
        equations = ["x1 + u1"]
        with pytest.raises(ValueError, match="Number of equations"):
            TransitionFunction(state_space, control_space, disturbance_space, 0.1, equations)

    def test_equation_parsing_correctness(self, state_space, control_space, disturbance_space):
        """Test that equations are correctly parsed into SymPy expressions.

        Args:
            state_space (DiscreteSpace): The state space fixture.
            control_space (DiscreteSpace): The control space fixture.
            disturbance_space (DiscreteSpace): The disturbance space fixture.
        """
        equations = ["x1 + u1 + w1 * tau", "x2 + u1"]
        tf = TransitionFunction(state_space, control_space, disturbance_space, 0.1, equations)
        assert all(isinstance(eq, sp.Expr) for eq in tf.equations), \
            "All parsed equations should be SymPy expressions"

    def test_lambdified_evaluation(self, state_space, control_space, disturbance_space):
        """Evaluate lambdified equations numerically and check correctness.

        Args:
            state_space (DiscreteSpace): The state space fixture.
            control_space (DiscreteSpace): The control space fixture.
            disturbance_space (DiscreteSpace): The disturbance space fixture.
        """
        equations = ["x1 + u1 + w1 * tau", "x2 + u1"]
        tau = 0.1
        tf = TransitionFunction(state_space, control_space, disturbance_space, tau, equations)

        current_state = np.array([1.0, 2.0])
        control_input = np.array([0.5])
        disturbance_input = np.array([0.1])

        results = tf.evaluate(current_state, control_input, disturbance_input)

        # Check type and length
        assert len(results) == 2
        assert all(isinstance(r, (float, np.floating)) for r in results)

        # Expected numerical values
        expected = np.array([
            current_state[0] + control_input[0] + disturbance_input[0]*tau,
            current_state[1] + control_input[0]
        ])
        np.testing.assert_allclose(results, expected, rtol=1e-6, atol=1e-9)


# ============================================================
# Advanced tests for TransitionFunction
# ============================================================

class TestTransitionFunctionAdvanced:
    """Advanced tests: multiple variables, boundary values, and performance."""

    # -------------------------------
    # Fixtures
    # -------------------------------
    @pytest.fixture
    def state_space(self):
        """Provides a 3D state space for testing.

        Returns:
            DiscreteSpace: A state space with 3 dimensions, lower bounds [0,0,0], upper bounds [10,10,10],
                           and grid [5,5,5].
        """
        return DiscreteSpace("State", 3, [0, 0, 0], [10, 10, 10], [5, 5, 5])

    @pytest.fixture
    def control_space(self):
        """Provides a 2D control space for testing.

        Returns:
            DiscreteSpace: A control space with 2 dimensions, lower bounds [0,0], upper bounds [1,1], grid [2,2].
        """
        return DiscreteSpace("Control", 2, [0, 0], [1, 1], [2, 2])

    @pytest.fixture
    def disturbance_space(self):
        """Provides a 2D disturbance space for testing.

        Returns:
            DiscreteSpace: A disturbance space with 2 dimensions, lower bounds [0,0], upper bounds [1,1], grid [2,2].
        """
        return DiscreteSpace("Disturbance", 2, [0, 0], [1, 1], [2, 2])

    # -------------------------------
    # Tests
    # -------------------------------
    def test_multiple_equations_many_variables(self, state_space, control_space, disturbance_space):
        """Test parsing and evaluation of multiple symbolic equations.

        Args:
            state_space (DiscreteSpace): The state space fixture.
            control_space (DiscreteSpace): The control space fixture.
            disturbance_space (DiscreteSpace): The disturbance space fixture.
        """
        equations = [
            "x1 + x2 * x3 + u1 - w1 * tau",
            "x2**2 + u2 + w2 * tau",
            "x3 + sin(u1) + w1"
        ]
        tau = 0.05
        tf = TransitionFunction(state_space, control_space, disturbance_space, tau, equations)

        current_state = np.array([1.0, 2.0, 3.0])
        control_input = np.array([0.1, 0.2])
        disturbance_input = np.array([0.05, 0.1])

        results = tf.evaluate(current_state, control_input, disturbance_input)

        # Type and finiteness check
        assert len(results) == 3
        assert all(np.isfinite(r) for r in results)

        # Expected numerical results
        expected = np.array([
            current_state[0] + current_state[1]*current_state[2] + control_input[0] - disturbance_input[0]*tau,
            current_state[1]**2 + control_input[1] + disturbance_input[1]*tau,
            current_state[2] + np.sin(control_input[0]) + disturbance_input[0]
        ])
        np.testing.assert_allclose(results, expected, rtol=1e-6, atol=1e-9)

    def test_boundary_inputs(self, state_space, control_space, disturbance_space):
        """Test evaluation at boundary values of state, control, and disturbance.

        Args:
            state_space (DiscreteSpace): The state space fixture.
            control_space (DiscreteSpace): The control space fixture.
            disturbance_space (DiscreteSpace): The disturbance space fixture.
        """
        equations = ["x1 + u1 + w1 * tau", "x2 + u2 + w2 * tau", "x3 + sin(u1) + w1"]
        tau = 0.1
        tf = TransitionFunction(state_space, control_space, disturbance_space, tau, equations)

        lower_state = state_space.lowerBounds
        upper_state = state_space.upperBounds
        lower_control = control_space.lowerBounds
        upper_control = control_space.upperBounds
        lower_disturbance = disturbance_space.lowerBounds
        upper_disturbance = disturbance_space.upperBounds

        for state in [lower_state, upper_state]:
            for control in [lower_control, upper_control]:
                for disturbance in [lower_disturbance, upper_disturbance]:
                    results = tf.evaluate(state, control, disturbance)
                    assert all(np.isfinite(r) for r in results)

                    expected = np.array([
                        state[0] + control[0] + disturbance[0]*tau,
                        state[1] + control[1] + disturbance[1]*tau,
                        state[2] + np.sin(control[0]) + disturbance[0]
                    ])
                    np.testing.assert_allclose(results, expected, rtol=1e-6, atol=1e-9)

    def test_performance_many_evaluations(self, state_space, control_space, disturbance_space):
        """Test performance by evaluating TransitionFunction many times and verify correctness for a subset.

        Args:
            state_space (DiscreteSpace): The state space fixture.
            control_space (DiscreteSpace): The control space fixture.
            disturbance_space (DiscreteSpace): The disturbance space fixture.
        """
        equations = ["x1 + x2 * x3 + u1 - w1 * tau",
                     "x2**2 + u2 + w2 * tau",
                     "x3 + sin(u1) + w1"]
        tau = 0.05
        tf = TransitionFunction(state_space, control_space, disturbance_space, tau, equations)

        np.random.seed(42)
        n = 10_000
        states = np.random.uniform(0, 10, (n, 3))
        controls = np.random.uniform(0, 1, (n, 2))
        disturbances = np.random.uniform(0, 1, (n, 2))

        # Check finiteness for all
        for i in range(n):
            results = tf.evaluate(states[i], controls[i], disturbances[i])
            assert all(np.isfinite(r) for r in results)

        # Spot check first 5 for numerical correctness
        for i in range(5):
            expected = np.array([
                states[i][0] + states[i][1]*states[i][2] + controls[i][0] - disturbances[i][0]*tau,
                states[i][1]**2 + controls[i][1] + disturbances[i][1]*tau,
                states[i][2] + np.sin(controls[i][0]) + disturbances[i][0]
            ])
            np.testing.assert_allclose(tf.evaluate(states[i], controls[i], disturbances[i]), expected,
                                       rtol=1e-6, atol=1e-9)
