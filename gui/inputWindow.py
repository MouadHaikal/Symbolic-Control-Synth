import sys
import os
import json
import csv
import ast
from typing import Sequence, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout,
    QLineEdit, QSpinBox, QLabel, QPushButton, QGroupBox, QFileDialog, QPlainTextEdit
)
from gui.gridWindow import GridWindow
from symControl.space.discreteSpace import DiscreteSpace
from symControl.model.model import Model


class InputWindow(QMainWindow):
    """GUI window for defining and exporting robot model information."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Model Input")
        self.setGeometry(100, 100, 600, 550)

        self._initUI()

    def _initUI(self):
        """
        Initialize UI components.
        """
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout()
        centralWidget.setLayout(layout)

        # Create space groups
        self.stateGroup = self._createSpaceGroup("State Space", "e.g., [(0,10),(0,10)]", "e.g., [10,10]")
        self.controlGroup = self._createSpaceGroup("Control Space", "e.g., [(-1,1),(0,2)]", "e.g., [10,10]")
        self.disturbGroup = self._createSpaceGroup("Disturbance Space", "e.g., [(-0.5,0.5),(0,1)]", "e.g., [10,10]")

        layout.addWidget(self.stateGroup["group"])
        layout.addWidget(self.controlGroup["group"])
        layout.addWidget(self.disturbGroup["group"])

        # Equations input
        self.equationsInput = QPlainTextEdit()
        self.equationsInput.setPlaceholderText(
            "Enter one equation per line.\n"
            "Use symbols x, u, w starting from 1 (x1, x2, ..., u1, u2, ..., w1, w2, ...),\n"
            "depending on the dimensions of state, control, and disturbance spaces.\n"
        )
        layout.addWidget(QLabel("Equations:"))
        layout.addWidget(self.equationsInput)

        # Time step
        self.timeStep = QLineEdit()
        self.timeStep.setPlaceholderText("e.g., 0.1")
        layout.addWidget(QLabel("Time Step (TAU):"))
        layout.addWidget(self.timeStep)

        # Buttons
        self.submitBtn = QPushButton("Submit Info")
        self.saveBtn = QPushButton("Save Input")
        self.loadBtn = QPushButton("Load Input")
        layout.addWidget(self.submitBtn)
        layout.addWidget(self.saveBtn)
        layout.addWidget(self.loadBtn)

        # Connect buttons
        self.submitBtn.clicked.connect(self.submitData)
        self.saveBtn.clicked.connect(self.saveInput)
        self.loadBtn.clicked.connect(self.loadInput)

    def _createSpaceGroup(self, title: str, boundsPlaceholder: str, resPlaceholder: str):
        """
        Helper to create space input groups (state, control, disturbance).
        """
        group = QGroupBox(title)
        layout = QFormLayout()

        dim = QSpinBox()
        dim.setMinimum(1)
        bounds = QLineEdit()
        bounds.setPlaceholderText(boundsPlaceholder)
        resolutions = QLineEdit()
        resolutions.setPlaceholderText(resPlaceholder)

        layout.addRow("Dimensions:", dim)
        layout.addRow("Bounds:", bounds)
        layout.addRow("Resolution:", resolutions)
        group.setLayout(layout)

        return {"group": group, "dim": dim, "bounds": bounds, "res": resolutions}

    def submitData(self):
        """
        Collect user inputs and build the model.
        """
        def parseSpace(group):
            bounds = ast.literal_eval(group["bounds"].text())
            resolutions = ast.literal_eval(group["res"].text())
            dim = len(bounds)
            return DiscreteSpace(dim, bounds, resolutions)

        discreteStateSpace = parseSpace(self.stateGroup)
        discreteControlSpace = parseSpace(self.controlGroup)
        discreteDisturbanceSpace = parseSpace(self.disturbGroup)

        equationsList = [line.strip() for line in self.equationsInput.toPlainText().splitlines() if line.strip()]

        try:
            tau = float(self.timeStep.text())
        except ValueError:
            tau = 0.1

        model = Model(
            stateSpace=discreteStateSpace,
            controlSpace=discreteControlSpace,
            disturbanceSpace=discreteDisturbanceSpace,
            timeStep=tau,
            equations=equationsList
        )

        self.gridWindow = GridWindow(model)
        self.setCentralWidget(self.gridWindow)

    def saveInput(self):
        """
        Save input data to JSON
        """
        def getGroupData(group):
            return {
                "dimensions": group["dim"].value(),
                "bounds": group["bounds"].text(),
                "resolutions": group["res"].text()
            }

        data = {
            "state": getGroupData(self.stateGroup),
            "control": getGroupData(self.controlGroup),
            "disturbance": getGroupData(self.disturbGroup),
            "equations": self.equationsInput.toPlainText(),
            "tau": self.timeStep.text()
        }

        filePath, _ = QFileDialog.getSaveFileName(
            self, "Save Input", "", "JSON Files (*.json);;CSV Files (*.csv)"
        )
        if not filePath:
            return

        if filePath.endswith(".json"):
            with open(filePath, "w") as f:
                json.dump(data, f, indent=4)
        elif filePath.endswith(".csv"):
            with open(filePath, "w", newline='') as f:
                writer = csv.writer(f)
                for key, value in data.items():
                    writer.writerow([key, value])

    def loadInput(self):
        """
        Load input data from JSON or CSV.
        """
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Load Input", "", "JSON Files (*.json);;CSV Files (*.csv)"
        )
        if not filePath:
            return

        if filePath.endswith(".json"):
            with open(filePath, "r") as f:
                data = json.load(f)
        elif filePath.endswith(".csv"):
            data = {}
            with open(filePath, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 2:
                        data[row[0]] = row[1]

        def setGroupData(group, dataDict):
            group["dim"].setValue(int(dataDict["dimensions"]))
            group["bounds"].setText(dataDict["bounds"])
            group["res"].setText(dataDict["resolutions"])

        setGroupData(self.stateGroup, data["state"])
        setGroupData(self.controlGroup, data["control"])
        setGroupData(self.disturbGroup, data["disturbance"])

        self.equationsInput.setPlainText(data["equations"])
        self.timeStep.setText(data["tau"])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InputWindow()
    window.show()
    sys.exit(app.exec())
