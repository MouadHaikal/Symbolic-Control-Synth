from gui.utils import BuildWorker
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

        # Build main window
        self.window = Window()
        self.window.setWindowTitle("Symbolic Control Synth")
        self.window.setWindowIcon(QIcon("resources/icon.png"))
        self.window.setGeometry(700, 300, 600, 550)
        self.window.setStatusBar(QStatusBar())
        self.statusBarPermanentLabel = QLabel("")
        self.window.statusBar().addPermanentWidget(self.statusBarPermanentLabel)

        self.window.buildSignal.connect(self.__onBuildSignaled)

        self.window.show()

        # Set font
        QFontDatabase.addApplicationFont("resources/Rubik.ttf")
        self.setFont(QFont("Rubik", pointSize=12, weight=900))

        # Run
        sys.exit(self.exec())


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


    def __setEnableWindow(self, enable: bool) -> None:
        self.window.setEnabled(enable)

        if enable:
            QApplication.restoreOverrideCursor()
        else:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

