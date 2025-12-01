from gui.robotSimulation import simulateRobot
from gui.utils import BuildWorker, GetControllerWorker
from gui.window import Window

import sys
from PyQt6.QtWidgets import QApplication, QStatusBar, QLabel
from PyQt6.QtGui import QIcon, QFont, QFontDatabase
from PyQt6.QtCore import Qt

from symControl.bindings import Automaton



class App(QApplication):
    def __init__(self, argv: list[str]) -> None:
        super().__init__(argv)


    def run(self) -> None:
        self.automaton = None
        self.stateSpace = None
        self.specConfig = dict()
        self.simulationPath = []

        # Build main window
        self.window = Window()
        self.window.setWindowTitle("Symbolic Control Synth")
        self.window.setWindowIcon(QIcon("resources/icon.png"))
        self.window.setGeometry(700, 300, 600, 550)
        self.window.setStatusBar(QStatusBar())
        self.statusBarPermanentLabel = QLabel("")
        self.window.statusBar().addPermanentWidget(self.statusBarPermanentLabel)

        self.window.buildSignal.connect(self.__onBuildSignaled)
        self.window.specForm.getConrtollerSignal.connect(self.__onGetControllerSignaled)
        self.window.specForm.viewSimulationSignal.connect(self.__onViewSimulationSignaled)

        self.window.show()

        # Set font
        QFontDatabase.addApplicationFont("resources/Rubik.ttf")
        self.setFont(QFont("Rubik", pointSize=12, weight=900))

        # Run
        sys.exit(self.exec())

    def __onGetControllerSignaled(self, specConfig: dict) -> None:
        self.window.statusBar().showMessage("Applying specifications...")

        self.specConfig = specConfig

        self.__setEnableWindow(False)

        self.getControllerWorker = GetControllerWorker(specConfig, self)
        self.getControllerWorker.finished.connect(self.__onGetControllerFinished)
        self.getControllerWorker.failed.connect(self.__onGetControllerFailed)
        self.getControllerWorker.start()

    def __onViewSimulationSignaled(self) -> None:
        if self.simulationPath[0] == -1:
            return

        start = self.stateSpace.getCellCoords(self.specConfig["startPoint"])[:2]

        obstacles = [
            [
                self.stateSpace.getCellCoords(obstacle["lowerBound"])[:2],
                self.stateSpace.getCellCoords(obstacle["upperBound"])[:2]
            ]
            for obstacle in self.specConfig["obstacles"]
        ]

        target = [
            self.stateSpace.getCellCoords(self.specConfig["target"]["lowerBound"])[:2],
            self.stateSpace.getCellCoords(self.specConfig["target"]["upperBound"])[:2]
        ]

        path = [
            self.stateSpace.getCellCenter(pt)[:2]
                for pt in self.simulationPath
        ]

        simulateRobot(start, obstacles, target, path)

        


    def __onBuildSignaled(self, config: dict) -> None:
        self.statusBarPermanentLabel.setText("")
        self.window.statusBar().showMessage("Building automaton...")

        self.__setEnableWindow(False)

        self.buildWorker = BuildWorker(config, self)
        self.buildWorker.finished.connect(self.__onBuildFinished)
        self.buildWorker.failed.connect(self.__onBuildFailed)
        self.buildWorker.start()


    def __onBuildFinished(self, isCooperative: bool) -> None:
        self.__setEnableWindow(True)
        self.window.statusBar().showMessage("Build complete!", 3000)

        if isCooperative:
            self.statusBarPermanentLabel.setText("Cooperative")
        else:
            self.statusBarPermanentLabel.setText("Non-Cooperative")

        self.buildWorker.quit()
        self.buildWorker.wait()

        self.window.centralTabWidget.setTabEnabled(1, True)
        self.window.resetSpecTab()
        self.window.centralTabWidget.setCurrentIndex(1)

    def __onBuildFailed(self, error: str) -> None:
        self.__setEnableWindow(True)
        self.window.statusBar().showMessage(f"Build failed: {error}")
        self.buildWorker.quit()
        self.buildWorker.wait()


    def __onGetControllerFinished(self, paths: list[list[int]]) -> None:
        self.__setEnableWindow(True)
        self.window.statusBar().showMessage("Controller synthesis complete!", 3000)

        self.getControllerWorker.quit()
        self.getControllerWorker.wait()

        self.simulationPath = paths[0]

        if paths[0][0] == -1:
            self.window.statusBar().showMessage("No path found!")
            return

        # Draw paths
        for path in paths:
            self.window.stateSpaceView.drawPath([
                self.stateSpace.getCellCenter(point) for point in path
            ])


    def __onGetControllerFailed(self, error: str) -> None:
        self.__setEnableWindow(True)
        self.window.statusBar().showMessage(f"Controller synthesis failed: {error}")
        self.getControllerWorker.quit()
        self.getControllerWorker.wait()


    def __setEnableWindow(self, enable: bool) -> None:
        self.window.setEnabled(enable)

        if enable:
            QApplication.restoreOverrideCursor()
        else:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

