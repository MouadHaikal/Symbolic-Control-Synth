# gui/app.py
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit
)
from symControl.space.discreteSpace import DiscreteSpace
from gui.components import SpaceInputWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Symbolic Control Synth GUI")
        self.setGeometry(100, 100, 800, 600)

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
        self.validate_button = QPushButton("Validate Equations")
        self.generate_grid_button = QPushButton("Generate Grid")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.generate_grid_button)
        button_layout.addWidget(self.validate_button)
        main_layout.addLayout(button_layout)

        # Feedback
        self.feedback_label = QLabel("Output / Feedback:")
        self.feedback_text = QTextEdit()
        self.feedback_text.setReadOnly(True)
        main_layout.addWidget(self.feedback_label)
        main_layout.addWidget(self.feedback_text)

        # Connect button
        self.generate_grid_button.clicked.connect(self.on_generate_grid)

    # =========================================================
    # Button functionality
    def on_generate_grid(self):
        try:
            # Read space info
            state_data = self.state_widget.read_values()
            control_data = self.control_widget.read_values()
            dist_data = self.dist_widget.read_values()

            # Create DiscreteSpace objects
            state_space = DiscreteSpace(
                state_data["name"],
                state_data["dimensions"],
                state_data["lowerBounds"],
                state_data["upperBounds"],
                state_data["resolutions"]
            )

            control_space = DiscreteSpace(
                control_data["name"],
                control_data["dimensions"],
                control_data["lowerBounds"],
                control_data["upperBounds"],
                control_data["resolutions"]
            )

            dist_space = DiscreteSpace(
                dist_data["name"],
                dist_data["dimensions"],
                dist_data["lowerBounds"],
                dist_data["upperBounds"],
                dist_data["resolutions"]
            )

            self.feedback_text.setText(
                "✅ Grid generation successful!\n\n"
                f"State Space:\n{state_data}\n\n"
                f"Control Space:\n{control_data}\n\n"
                f"Disturbance Space:\n{dist_data}"
            )

        except Exception as e:
            self.feedback_text.setText(f"❌ Error:\n{str(e)}")
