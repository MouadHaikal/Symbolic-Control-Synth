from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLineEdit, QPushButton, QScrollArea, QSpinBox, QWidget, 
    QVBoxLayout, QLabel, QDoubleSpinBox, QAbstractSpinBox, QMessageBox
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation

from symControl.space.discreteSpace import DiscreteSpace
from symControl.space.continuousSpace import ContinuousSpace
from symControl.model.model import Model
from symControl.model.codePrinter import CodePrinter
from symControl.bindings import Automaton



class HSeparator(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)


class SpaceForm(QGroupBox):
    def __init__(self, name: str, symbol: str, isDiscrete: bool, minDim = 1) -> None:
        super().__init__()

        self.setLayout(QVBoxLayout())

        header = QWidget()
        dimensions = QWidget()
        lowerBound = QWidget()
        upperBound = QWidget()

        self.layout().addWidget(header)
        self.layout().addWidget(dimensions)
        self.layout().addWidget(lowerBound)
        self.layout().addWidget(upperBound)


        # === Header ===
        header.setLayout(QVBoxLayout())

        title = QLabel(f"{name} ({symbol})")

        header.layout().addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        header.layout().addWidget(HSeparator())


        # === Dimensions ===
        dimensions.setLayout(QHBoxLayout())

        dimensionsLabel = QLabel("Dimensions :")
        self.dimensionsInput = QSpinBox()
        self.dimensionsInput.setMinimum(minDim)
        self.dimensionsInput.setMaximum(9)
        self.dimensionsInput.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        dimensions.layout().addWidget(dimensionsLabel, stretch=0)
        dimensions.layout().addWidget(self.dimensionsInput, stretch=1)


        # === Lower bound ===
        lowerBound.setLayout(QHBoxLayout())

        lowerBoundLabel = QLabel("Lower Bound :")
        self.lowerBoundInput = PointInput(self.dimensionsInput)

        lowerBound.layout().addWidget(lowerBoundLabel, stretch=0)
        lowerBound.layout().addWidget(self.lowerBoundInput, stretch=1)


        # === Upper bound ===
        upperBound.setLayout(QHBoxLayout())
            
        upperBoundLabel = QLabel(f"Upper Bound :")
        self.upperBoundInput = PointInput(self.dimensionsInput)

        upperBound.layout().addWidget(upperBoundLabel, stretch=0)
        upperBound.layout().addWidget(self.upperBoundInput, stretch=1)



        if isDiscrete:
            # === Resolution ===
            resolution = QWidget()
            self.layout().addWidget(resolution)

            resolution.setLayout(QHBoxLayout())

            resolutionLabel = QLabel(f"Resolution :")
            self.resolutionInput = PointInput(self.dimensionsInput, isInteger=True)

            resolution.layout().addWidget(resolutionLabel, stretch=0)
            resolution.layout().addWidget(self.resolutionInput, stretch=1)



        # ========== Styling ==========
        self.setMinimumSize(300, 170)
        self.setMaximumWidth(550)
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(10,0,10,0)
        
        header.setFixedHeight(45)

        dimensions.layout().setContentsMargins(0,0,0,0)
        dimensions.layout().setSpacing(0)

        lowerBound.layout().setContentsMargins(0,0,0,0)
        lowerBound.layout().setSpacing(0)

        upperBound.layout().setContentsMargins(0,0,0,0)
        upperBound.layout().setSpacing(0)


        dimensionsLabel.setFixedWidth(120)
        lowerBoundLabel.setFixedWidth(120)
        upperBoundLabel.setFixedWidth(120)

        if isDiscrete:
            resolution.layout().setContentsMargins(0,0,0,0)
            resolution.layout().setSpacing(0)
            resolutionLabel.setFixedWidth(120)

    def getValue(self) -> dict:
        """Return dict with space configuration"""
        result = {
            "dimensions": self.dimensionsInput.value(),
            "lowerBound": self.lowerBoundInput.getValue(),
            "upperBound": self.upperBoundInput.getValue(),
        }
        if hasattr(self, 'resolutionInput'):  # discrete only
            result["resolution"] = self.resolutionInput.getValue()
        return result

    def setValue(self, data: dict) -> None:
        """Load space configuration from dict"""
        self.dimensionsInput.setValue(data.get("dimensions", 1))
        self.lowerBoundInput.setValue(data.get("lowerBound", []))
        self.upperBoundInput.setValue(data.get("upperBound", []))
        if "resolution" in data:
            self.resolutionInput.setValue(data.get("resolution", []))


class PointInput(QWidget):
    def __init__(self, dimensionsInputBox: QSpinBox, isInteger: bool = False) -> None:
        super().__init__()
        self.dimensionsInputBox = dimensionsInputBox 
        self.isInteger = isInteger

        self.setLayout(QHBoxLayout())

        
        self.container = QWidget()  # Holds dynamic spinboxes
        self.container.setLayout(QHBoxLayout())


        prefixLabel = QLabel("(")
        suffixLabel = QLabel(")")
        self.layout().addWidget(prefixLabel)
        self.layout().addWidget(self.container)
        self.layout().addWidget(suffixLabel)
        

        self.spinBoxes = []

        dimensionsInputBox.valueChanged.connect(self.__rebuild)
        self.__rebuild()


        
        # ========== Styling ==========
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0,0,0,0)

        self.container.layout().setSpacing(0)
        self.container.layout().setContentsMargins(0,0,0,0)

        prefixLabel.setFixedWidth(15)
        prefixLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        suffixLabel.setFixedWidth(15)
        suffixLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def __rebuild(self):
        while self.container.layout().count():
            item = self.container.layout().takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self.spinBoxes.clear()

        
        dim = self.dimensionsInputBox.value()

        for i in range(dim):
            if self.isInteger:
                spinBox = QSpinBox()
                spinBox.setMinimum(1)
                spinBox.setMaximum(100000)
            else:
                spinBox = QDoubleSpinBox()
                spinBox.setDecimals(3)
                spinBox.setMinimum(-1e6)
                spinBox.setMaximum(1e6)
            spinBox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

            self.spinBoxes.append(spinBox)
            self.container.layout().addWidget(spinBox)

            if i < dim - 1:
                comma = QLabel(",")
                comma.setFixedWidth(15)
                self.container.layout().addWidget(comma)

    def getValue(self) -> list:
        """Return list of float values from spinboxes"""
        return [sb.value() for sb in self.spinBoxes]

    def setValue(self, values: list) -> None:
        """Set spinbox values from list"""
        for i, sb in enumerate(self.spinBoxes):
            if i < len(values):
                sb.setValue(values[i])


class EquationForm(QGroupBox):
    valueChanged = pyqtSignal()

    def __init__(self, dimensionsInputBox: QSpinBox, symbol: str) -> None:
        super().__init__()
        self.dimensionsInputBox = dimensionsInputBox 
        self.symbol = symbol

        self.setLayout(QHBoxLayout())

        self.container = QWidget()
        self.container.setLayout(QVBoxLayout())

        self.layout().addWidget(self.container)

        self.lineEdits = []

        dimensionsInputBox.valueChanged.connect(self.__rebuild)
        self.__rebuild()


        # ========== Styling ==========
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0,0,0,0)

        self.container.setMaximumWidth(500)
        self.container.layout().setSpacing(0)
        self.container.layout().setContentsMargins(0,0,0,0)

    def __rebuild(self):
        while self.container.layout().count():
            item = self.container.layout().takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self.lineEdits.clear()

        
        dim = self.dimensionsInputBox.value()


        for i in range(dim):
            lineContainer = QWidget()
            lineContainer.setLayout(QHBoxLayout())

            lineLabel = QLabel(f"{self.symbol}<sub>{i+1}</sub><sup>+</sup> = ")
            lineLabel.setFixedWidth(40)
            lineEdit = QLineEdit()
            lineEdit.textChanged.connect(self.valueChanged.emit)

            lineContainer.layout().addWidget(lineLabel)
            lineContainer.layout().addWidget(lineEdit)
            lineContainer.layout().setContentsMargins(0,3,0,3)

            self.lineEdits.append(lineEdit)
            self.container.layout().addWidget(lineContainer)

        self.adjustSize()

    def getValue(self) -> list:
        return [edit.text() for edit in self.lineEdits] 

    def setValue(self, equations: list) -> None:
        for i, edit in enumerate(self.lineEdits):
            if i < len(equations):
                edit.setText(equations[i])


class BuildWorker(QThread):
    """Worker thread that does the build computation."""
    finished = pyqtSignal(bool)  # emitted when done (returns isCooperative)
    failed = pyqtSignal(str)  # emitted on error

    def __init__(self, config: dict, app):
        super().__init__()
        self.config = config
        self.app = app

    def run(self) -> None:
        try:
            self.__run()
        except Exception as e:
            self.failed.emit(str(e))


    def __run(self) -> None:
        stateSpaceBounds = [
            tuple([
                self.config["stateSpace"]["lowerBound"][i], 
                self.config["stateSpace"]["upperBound"][i]
            ])
            for i in range(self.config["stateSpace"]["dimensions"])
        ]

        inputSpaceBounds = [
            tuple([
                self.config["inputSpace"]["lowerBound"][i], 
                self.config["inputSpace"]["upperBound"][i]
            ])
            for i in range(self.config["inputSpace"]["dimensions"])
        ]

        disturbanceSpaceBounds = [
            tuple([
                self.config["disturbanceSpace"]["lowerBound"][i], 
                self.config["disturbanceSpace"]["upperBound"][i]
            ])
            for i in range(self.config["disturbanceSpace"]["dimensions"])
        ]

        
        
        self.app.stateSpace = DiscreteSpace(
            self.config["stateSpace"]["dimensions"],
            stateSpaceBounds,
            self.config["stateSpace"]["resolution"]
        )

        inputSpace = DiscreteSpace(
            self.config["inputSpace"]["dimensions"],
            inputSpaceBounds,
            self.config["inputSpace"]["resolution"]
        )

        disturbanceSpace = ContinuousSpace(
            self.config["disturbanceSpace"]["dimensions"],
            disturbanceSpaceBounds,
        )


        model = Model(
            self.app.stateSpace,            
            inputSpace,        
            disturbanceSpace,
            self.config["timeStep"],
            self.config["equations"]
        )


        self.app.automaton = Automaton(
            self.app.stateSpace,
            inputSpace,
            disturbanceSpace,
            model.transitionFunction.isCooperative,
            model.transitionFunction.disturbJacUpper,
            CodePrinter(model).printCode()
        )

        self.finished.emit(model.transitionFunction.isCooperative)


class SpaceView(QWidget):
    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0,0,0,0)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)

        self.canvas = FigureCanvas(self.fig)
        self.layout().addWidget(self.canvas)
        

    def reset(self, spaceConfig: dict) -> None:
        self.ax.clear()

        xBounds = tuple([
            spaceConfig["lowerBound"][0],
            spaceConfig["upperBound"][0]
        ])

        yBounds = tuple([
            spaceConfig["lowerBound"][1],
            spaceConfig["upperBound"][1]
        ])

        self.ax.set_xlim(*xBounds)
        self.ax.set_ylim(*yBounds)

        self.fig.subplots_adjust(left=0.09, right=0.95, top=0.95, bottom=0.09)

        self.ax.set_xlabel(f"{self.symbol}$_1$")
        self.ax.set_ylabel(f"{self.symbol}$_2$")
        self.ax.grid(True, alpha=0.2)
        self.canvas.draw()


    def drawRegions(self, specConfig: dict) -> None:
        self.__drawPoint(specConfig["startPoint"], "blue", "Start")
        self.__drawRegion(specConfig["target"], "green")
        
        for obstacle in specConfig["obstacles"]:
            self.__drawRegion(obstacle, "red")

        self.canvas.draw()

    def drawPath(self, points: list[tuple[int, ...]], color = "blue") -> None:
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]

        self.ax.plot(xs, ys, color=color, linewidth=0.25, zorder=10)
        
        self.canvas.draw()
        


    def __drawRegion(self, bounds: dict, color: str) -> None:
        x = bounds["lowerBound"][0]
        y = bounds["lowerBound"][1]
        width  = bounds["upperBound"][0] - x
        height = bounds["upperBound"][1] - y

        rect = Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor=color, alpha=0.4)
        self.ax.add_patch(rect)

    def __drawPoint(self, point: list, color: str, label: str) -> None:
        x, y = point[0], point[1]
        self.ax.scatter(x, y, s=20, c="blue", marker='o', zorder=5)
        
        self.ax.annotate(label, (x, y),
                        xytext=(0, -15), textcoords='offset pixels',
                        ha='center', va='top', fontsize=10,
                        color= color, weight='bold')
        

class SpecificationForm(QWidget):
    drawSignal = pyqtSignal(dict)
    getConrtollerSignal = pyqtSignal(dict)
    viewSimulationSignal = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setLayout(QVBoxLayout())
        # addStretch

        self.obstacles = [] # obstacle RegionInput objects

        # ========== Styling ==========
        self.layout().setSpacing(20)
        self.layout().setContentsMargins(0,0,0,0)
        

    def reset(self, spaceConfig) -> None:
        while self.layout().count():
            child = self.layout().takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.obstacles.clear()


        self.topContainer = QGroupBox()
        self.bottomContainer = QGroupBox()
        self.finalButton = QPushButton("Apply")
        self.finalButton.clicked.connect(self.__onApplyClicked)
        self.layout().addWidget(self.topContainer)
        self.layout().addWidget(self.bottomContainer)
        self.layout().addWidget(self.finalButton)


        self.mockDimInput = QSpinBox()
        self.mockDimInput.setValue(spaceConfig["dimensions"])

        # =================== Top container ===================
        self.topContainer.setLayout(QVBoxLayout())

        startPoint = QWidget()
        self.targetRegion = RegionInput("Target", self.mockDimInput)

        self.topContainer.layout().addWidget(startPoint)
        self.topContainer.layout().addWidget(self.targetRegion)
        self.topContainer.layout().addWidget(HSeparator())


        # === Start Point ===
        startPoint.setLayout(QHBoxLayout())

        startPointLabel = QLabel("Start point :")
        self.startPointInput = PointInput(self.mockDimInput)

        startPoint.layout().addWidget(startPointLabel, stretch=0)
        startPoint.layout().addWidget(self.startPointInput, stretch=1)



        # =================== Bottom container ===================
        self.bottomContainer.setLayout(QVBoxLayout())

        obstacleButtons = QWidget()
        self.obstacleScrollArea = QScrollArea()

        self.bottomContainer.layout().addWidget(obstacleButtons)
        self.bottomContainer.layout().addWidget(self.obstacleScrollArea)

        # Obstacle buttons
        addObstacleButton = QPushButton("Add Obstacle")
        addObstacleButton.clicked.connect(self.__onAddObstacleClicked)
        resetObstaclesButton = QPushButton("Reset")
        resetObstaclesButton.clicked.connect(self.__onResetObstaclesClicked)

        obstacleButtons.setLayout(QHBoxLayout())
        obstacleButtons.layout().addWidget(addObstacleButton)
        obstacleButtons.layout().addWidget(resetObstaclesButton)


        # Scroll area
        self.obstacleContainer = QWidget()

        self.obstacleScrollArea.setWidgetResizable(True)
        self.obstacleScrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.obstacleScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.obstacleScrollArea.setWidget(self.obstacleContainer)


        self.obstacleContainer.setLayout(QVBoxLayout())
        self.obstacleContainer.layout().addStretch()

        self.obstacleContainer.layout().setContentsMargins(0,0,0,0)


    def __onAddObstacleClicked(self) -> None:
        obstacle = RegionInput(f"Obstacle {len(self.obstacles)+1}", self.mockDimInput)
        self.obstacles.append(obstacle)


        # Remove stretch (last item)
        self.obstacleContainer.layout().takeAt(self.obstacleContainer.layout().count() - 1)

        self.obstacleContainer.layout().addWidget(obstacle)
        self.obstacleContainer.layout().addStretch()


        # Scroll to bottom
        self.obstacleContainer.updateGeometry()
        self.obstacleScrollArea.updateGeometry()


        QTimer.singleShot(0, lambda: self.obstacleScrollArea.verticalScrollBar().setValue(
            self.obstacleScrollArea.verticalScrollBar().maximum()
        ))

    def __onResetObstaclesClicked(self) -> None:
        if not len(self.obstacles):
            return 

        reply = QMessageBox.question(
            self,
            "Confirm Reset",
            "Reset all obstacles?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            return


        for obstacle in self.obstacles:
              obstacle.deleteLater()
          
        self.obstacles.clear()

        while self.obstacleContainer.layout().count() > 1:
            child = self.obstacleContainer.layout().takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def __onApplyClicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Confirm Apply",
            "Apply and lock specifications?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            return

        self.topContainer.setDisabled(True)
        self.bottomContainer.setDisabled(True)
        self.finalButton.setText("Get Controller")
        self.finalButton.clicked.disconnect()
        self.finalButton.clicked.connect(self.__onGetControllerClicked)

        specConfig = {
            "startPoint": self.startPointInput.getValue(),
            "target": self.targetRegion.getValue(),
            "obstacles": [
                obstacle.getValue()
                for obstacle in self.obstacles
            ]
        }

        self.drawSignal.emit(specConfig)

    def __onGetControllerClicked(self) -> None:
        self.finalButton.setDisabled(True)

        specConfig = {
            "startPoint": self.startPointInput.getValue(),
            "target": self.targetRegion.getValue(),
            "obstacles": [
                obstacle.getValue()
                for obstacle in self.obstacles
            ]
        }

        self.getConrtollerSignal.emit(specConfig)

        self.finalButton.setText("View Simulation")
        self.finalButton.clicked.disconnect()
        self.finalButton.clicked.connect(self.__onViewSimulationClicked)
        self.finalButton.setEnabled(True)

    def __onViewSimulationClicked(self) -> None:
        self.finalButton.setDisabled(True)
        self.viewSimulationSignal.emit()

        
class RegionInput(QGroupBox):
    def __init__(self, name: str, dimensionsInputBox: QSpinBox) -> None:
        super().__init__()
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(QLabel(name))

        lowerBound = QWidget()
        upperBound = QWidget()

        self.layout().addWidget(lowerBound)
        self.layout().addWidget(upperBound)

        lowerBoundLabel = QLabel("Lower bound :")
        upperBoundLabel = QLabel("Upper bound :")

        self.lowerBoundInput = PointInput(dimensionsInputBox)
        self.upperBoundInput = PointInput(dimensionsInputBox)

        lowerBound.setLayout(QHBoxLayout())
        upperBound.setLayout(QHBoxLayout())

        lowerBound.layout().addWidget(lowerBoundLabel, stretch=0)
        upperBound.layout().addWidget(upperBoundLabel, stretch=0)

        lowerBound.layout().addWidget(self.lowerBoundInput, stretch=1)
        upperBound.layout().addWidget(self.upperBoundInput, stretch=1)



        # ========== Styling ==========

        lowerBound.layout().setSpacing(0)
        lowerBound.layout().setContentsMargins(0,0,0,0)

        upperBound.layout().setSpacing(0)
        upperBound.layout().setContentsMargins(0,0,0,0)


        self.layout().setSpacing(0)
        self.setFixedHeight(120)


    def getValue(self) -> dict:
        return {
            "lowerBound": self.lowerBoundInput.getValue(),
            "upperBound": self.upperBoundInput.getValue()
        }


class GetControllerWorker(QThread):
    finished = pyqtSignal(list)  # emitted when done - returns paths
    failed = pyqtSignal(str)  # emitted on error

    def __init__(self, specConfig: dict, app):
        super().__init__()
        self.specConfig = specConfig
        self.app = app


    def run(self) -> None:
        try:
            self.__run()
        except Exception as e:
            self.failed.emit(str(e))


    def __run(self) -> None:
        for obstacle in self.specConfig["obstacles"]:
            self.app.automaton.applySecuritySpec(
                self.app.stateSpace.getCellCoords(obstacle["lowerBound"]),
                self.app.stateSpace.getCellCoords(obstacle["upperBound"])
            )

        paths = self.app.automaton.getController(
            self.app.stateSpace.getCellCoords(self.specConfig["startPoint"]),
            self.app.stateSpace.getCellCoords(self.specConfig["target"]["lowerBound"]),
            self.app.stateSpace.getCellCoords(self.specConfig["target"]["upperBound"]),
            20
        )

        self.finished.emit(paths)

