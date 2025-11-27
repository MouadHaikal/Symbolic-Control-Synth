# gui/app.py (updated)
from typing import Sequence, Tuple
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit,
    QDoubleSpinBox, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt

from symControl.space.discreteSpace import DiscreteSpace
from symControl.model.model import Model   # adjust import path if necessary
from gui.components import SpaceInputWidget
from gui.grid_widget import GridWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Symbolic Control Synth GUI")
        self.setGeometry(100, 100, 1000, 700)

        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)

        #  Tabs with Space Inputs
        self.space_tabs = QTabWidget()

        self.state_widget = SpaceInputWidget("State Space")
        self.control_widget = SpaceInputWidget("Control Space")
        self.dist_widget = SpaceInputWidget("Disturbance Space")

        self.space_tabs.addTab(self.state_widget, "State Space")
        self.space_tabs.addTab(self.control_widget, "Control Space")
        self.space_tabs.addTab(self.dist_widget, "Disturbance Space")

        main_layout.addWidget(self.space_tabs)

        # ---------------- Equation Input ----------------
        self.equation_label = QLabel("Equations (one per line):")
        self.equation_text = QTextEdit()
        main_layout.addWidget(self.equation_label)
        main_layout.addWidget(self.equation_text)

        # Buttons
        self.validate_button = QPushButton("Validate & Create Model")
        self.generate_grid_button = QPushButton("Generate Grid")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.generate_grid_button)
        button_layout.addWidget(self.validate_button)
        main_layout.addLayout(button_layout)

        # Simulation Controls group (appears after grid is created)
        self.sim_group = QGroupBox("Simulation Controls")
        sim_layout = QGridLayout()
        self.sim_group.setLayout(sim_layout)

        sim_layout.addWidget(QLabel("Control u1:"), 0, 0)
        self.u1_spin = QDoubleSpinBox(); self.u1_spin.setRange(-1000, 1000); self.u1_spin.setSingleStep(0.1)
        sim_layout.addWidget(self.u1_spin, 0, 1)

        sim_layout.addWidget(QLabel("Control u2:"), 0, 2)
        self.u2_spin = QDoubleSpinBox(); self.u2_spin.setRange(-1000, 1000); self.u2_spin.setSingleStep(0.1)
        sim_layout.addWidget(self.u2_spin, 0, 3)

        sim_layout.addWidget(QLabel("Dist w1:"), 1, 0)
        self.w1_spin = QDoubleSpinBox(); self.w1_spin.setRange(-1000, 1000); self.w1_spin.setSingleStep(0.1)
        sim_layout.addWidget(self.w1_spin, 1, 1)

        sim_layout.addWidget(QLabel("Dist w2:"), 1, 2)
        self.w2_spin = QDoubleSpinBox(); self.w2_spin.setRange(-1000, 1000); self.w2_spin.setSingleStep(0.1)
        sim_layout.addWidget(self.w2_spin, 1, 3)

        self.step_button = QPushButton("Step")
        self.reset_button = QPushButton("Reset")
        sim_layout.addWidget(self.step_button, 2, 1)
        sim_layout.addWidget(self.reset_button, 2, 2)

        self.sim_group.setVisible(False)
        main_layout.addWidget(self.sim_group)

        # Feedback
        self.feedback_label = QLabel("Output / Feedback:")
        self.feedback_text = QTextEdit()
        self.feedback_text.setReadOnly(True)
        main_layout.addWidget(self.feedback_label)
        main_layout.addWidget(self.feedback_text, stretch=1)

        # Grid container
        self.grid_widget = None

        # Internal model reference
        self.model = None

        # Connect buttons
        self.generate_grid_button.clicked.connect(self.on_generate_grid)
        self.validate_button.clicked.connect(self.on_validate_and_create_model)
        self.step_button.clicked.connect(self.on_step)
        self.reset_button.clicked.connect(self.on_reset)

    # =========================================================
    # Button functionality
    def on_generate_grid(self):
        def getBoundsTuple(lowerBounds, upperBounds):
            return [(lowerBounds[i], upperBounds[i]) for i in range(len(lowerBounds))]

        try:
            # Read space info
            state_data = self.state_widget.read_values()
            control_data = self.control_widget.read_values()
            dist_data = self.dist_widget.read_values()

            # Create DiscreteSpace objects
            state_space = DiscreteSpace(
                state_data["name"],
                state_data["dimensions"],
                getBoundsTuple(state_data["lowerBounds"], state_data["upperBounds"]),
                state_data["resolutions"]
            )
            control_space = DiscreteSpace(
                control_data["name"],
                control_data["dimensions"],
                getBoundsTuple(control_data["lowerBounds"], control_data["upperBounds"]),
                control_data["resolutions"]
            )
            dist_space = DiscreteSpace(
                dist_data["name"],
                dist_data["dimensions"],
                getBoundsTuple(dist_data["lowerBounds"], dist_data["upperBounds"]),
                dist_data["resolutions"]
            )

            # Save for later (so model creation works, error should be around here)
            self._last_state_space = state_space
            self._last_control_data = control_data
            self._last_dist_data = dist_data

            # Create and show the grid widget (only for 2D)
            if self.grid_widget is not None:
                self.grid_widget.setParent(None)  # remove old one if exists

            if state_space.dimensions == 2:
                self.grid_widget = GridWidget(state_space)
                self.central_widget.layout().addWidget(self.grid_widget)
            else:
                self.feedback_text.setText("Grid visualization only supports 2D spaces.")
                self.grid_widget = None

            # Feedback text
            self.feedback_text.setText(
                "Grid generation successful!\n\n"
                f"State Space:\n{state_data}\n\n"
                f"Control Space:\n{control_data}\n\n"
                f"Disturbance Space:\n{dist_data}"
            )

        except Exception as e:
            self.feedback_text.setText(f" Error:\n{str(e)}")


    def on_validate_and_create_model(self):
        """
        Take the equations from the text box, create DiscreteSpace objects for control
        and disturbance, then construct a Model. Show simulation controls if successful.
        """
        try:
            if not hasattr(self, "_last_state_space"):
                raise RuntimeError("Generate the grid first (press 'Generate Grid').")

            equations = [line.strip() for line in self.equation_text.toPlainText().splitlines() if line.strip()]
            if len(equations) == 0:
                raise ValueError("Enter at least one equation (one per state dimension).")

            state_space = self._last_state_space

            # create control and disturbance DiscreteSpace now from saved data
            cdata = self._last_control_data
            ddata = self._last_dist_data

            control_space = DiscreteSpace(cdata["name"], cdata["dimensions"], cdata["bounds"], cdata["resolutions"])
            dist_space = DiscreteSpace(ddata["name"], ddata["dimensions"], ddata["bounds"], ddata["resolutions"])

            # time step (tau) — choose a small default or allow user input; using 0.1 here
            tau = 0.1

            # Create the Model
            self.model = Model(state_space, control_space, dist_space, tau, equations)

            # highlight initial cell
            coords = state_space.getCellCoords(self.model.currentState)  # supports Cell input
            self.grid_widget.highlight_cell(coords)

            self.feedback_text.append("Model created successfully. Simulation controls visible.")
            self.sim_group.setVisible(True)

        except Exception as e:
            self.feedback_text.setText(f"Error while creating model:\n{str(e)}")

    def on_step(self):
        """ Evaluate one transition using values from spinboxes and move the model. """
        try:
            if self.model is None:
                raise RuntimeError("Create the model first (Validate & Create Model).")

            # collect current continuous state (center of current cell)
            current_cell = self.model.currentState
            state = list(current_cell.center)

            # controls and disturbances as sequences
            control = [self.u1_spin.value(), self.u2_spin.value()]  # assumes 2D control
            disturbance = [self.w1_spin.value(), self.w2_spin.value()]  # assumes 2D disturbance

            # Evaluate
            next_state = self.model.transitionFunction.evaluate(state, control, disturbance)

            # Change model's current state (changeCurrentState validates bounds)
            self.model.changeCurrentState(next_state)

            # get discrete coords for the new state (from the model's stateSpace)
            coords = self.model.stateSpace.getCellCoords(self.model.currentState)
            self.grid_widget.highlight_cell(coords)

            self.feedback_text.append(f"Stepped -> new continuous state: {next_state} | cell: {coords}")

        except Exception as e:
            self.feedback_text.setText(f" Error during step:\n{str(e)}")

    def on_reset(self):
        if self.model is None:
            return
        # reset model to (0,0) cell (Model already sets this at construction) by re-creating it
        try:
            state_space = self.model.stateSpace
            control_space = self.model.transitionFunction  # not used here
            # create a fresh model with same parameters (simpler to re-create)
            eqs = [str(e) for e in self.equation_text.toPlainText().splitlines() if e.strip()]
            # we need control and disturbance spaces re-created from previous saved values
            cdata = self._last_control_data
            ddata = self._last_dist_data
            control_space = DiscreteSpace(cdata["name"], cdata["dimensions"], cdata["bounds"], cdata["resolutions"])
            dist_space = DiscreteSpace(ddata["name"], ddata["dimensions"], ddata["bounds"], ddata["resolutions"])
            tau = self.model.timeStep if hasattr(self.model, "timeStep") else 0.1

            self.model = Model(state_space, control_space, dist_space, tau, eqs)
            coords = state_space.getCellCoords(self.model.currentState)
            if self.grid_widget:
                self.grid_widget.highlight_cell(coords)

            self.feedback_text.append("Simulation reset to initial cell.")

        except Exception as e:
            self.feedback_text.setText(f" Error during reset:\n{str(e)}")
