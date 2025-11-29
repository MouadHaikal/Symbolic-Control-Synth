from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QMainWindow, QPushButton, QWidget, QVBoxLayout, QLabel, QTabWidget, QMessageBox, QFileDialog
)
import json

from utils import *
from symControl.utils.constants import *



class Window(QMainWindow):
    buildSignal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        # Set main tab layout
        self.centralTabWidget = QTabWidget()
        self.setCentralWidget(self.centralTabWidget)
        
        # Initialize tabs
        self.__initConfigTab()
        self.__initSpecTab()


    def resetSpecTab(self) -> None:
        stateSpaceConfig = self.stateSpaceForm.getValue()
        self.stateSpaceView.reset(stateSpaceConfig)
        self.specForm.reset(stateSpaceConfig)

    def __initConfigTab(self) -> None:
        configTab = QWidget()
        self.centralTabWidget.addTab(configTab, "Configuration")

        # === Global layout ===
        configTab.setLayout(QVBoxLayout())

        spaceConfig  = QWidget()
        modelConfig  = QWidget()
        optionConfig = QWidget()

        configTab.layout().addWidget(spaceConfig, stretch=6)
        configTab.layout().addWidget(HSeparator())
        configTab.layout().addWidget(modelConfig, stretch=4)
        configTab.layout().addWidget(HSeparator())
        configTab.layout().addWidget(optionConfig, stretch=1)


        # === Space config layout ===
        spaceConfig.setLayout(QVBoxLayout())
        
        spaceConfigTitle = QLabel("Space Definition")
        spaceConfigContent = QWidget()

        spaceConfig.layout().addWidget(spaceConfigTitle, stretch=1, alignment=Qt.AlignmentFlag.AlignLeft)
        spaceConfig.layout().addWidget(spaceConfigContent, stretch=5)


        # Content
        spaceConfigContent.setLayout(QHBoxLayout())
        spaceConfigContent.layout().setSpacing(10)

        self.stateSpaceForm   = SpaceForm("State", STATE, isDiscrete=True)
        self.inputSpaceForm   = SpaceForm("Input", INPUT, isDiscrete=True)
        self.disturbSpaceForm = SpaceForm("Disturbance", DISTURBANCE, isDiscrete=False)

        spaceConfigContent.layout().addWidget(self.stateSpaceForm)
        spaceConfigContent.layout().addWidget(self.inputSpaceForm)
        spaceConfigContent.layout().addWidget(self.disturbSpaceForm)


        # === Model config layout ===
        modelConfig.setLayout(QVBoxLayout())
        
        modelConfigTitle = QLabel("Dynamic Model")
        modelConfigContent = QWidget()

        modelConfig.layout().addWidget(modelConfigTitle, stretch=1, alignment=Qt.AlignmentFlag.AlignLeft)
        modelConfig.layout().addWidget(modelConfigContent, stretch=5)


        # Content
        modelConfigContent.setLayout(QHBoxLayout())
        modelConfigContent.layout().setSpacing(10)

        self.equationForm = EquationForm(self.stateSpaceForm.dimensionsInput, STATE)
        self.equationForm.valueChanged.connect(self.__onEquationFormChanged)
        timeStepForm = QWidget()
        
        modelConfigContent.layout().addWidget(self.equationForm)
        modelConfigContent.layout().addWidget(timeStepForm)


        timeStepForm.setLayout(QHBoxLayout())

        timeStepLabel = QLabel(f"{TAU} = ")
        self.timeStepInput = QDoubleSpinBox()
        self.timeStepInput.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.timeStepInput.setDecimals(3)
        self.timeStepInput.setValue(1.0)

        timeStepForm.layout().addWidget(timeStepLabel)
        timeStepForm.layout().addWidget(self.timeStepInput)



        # === Option config layout ===
        optionConfig.setLayout(QHBoxLayout())

        loadButton = QPushButton("Load")
        loadButton.clicked.connect(self.__onLoadClicked)
        resetButton = QPushButton("Reset")
        resetButton.clicked.connect(self.__onResetClicked)
        saveButton = QPushButton("Save")
        saveButton.clicked.connect(self.__onSaveClicked)
        self.buildButton = QPushButton("Build")
        self.buildButton.setDisabled(True)
        self.buildButton.clicked.connect(self.__onBuildClicked)
        
        optionConfig.layout().addWidget(loadButton)
        optionConfig.layout().addWidget(resetButton)
        optionConfig.layout().addWidget(saveButton)
        optionConfig.layout().addWidget(self.buildButton)




        # ========== Styling ==========
        spaceConfig.setMaximumHeight(350)
        spaceConfigTitle.setStyleSheet("font-size: 20px; color: gray; font-family: Rubik; font-weight: 900;")
        spaceConfigTitle.setFixedHeight(30)
        modelConfigTitle.setStyleSheet("font-size: 20px; color: gray; font-family: Rubik; font-weight: 900;")
        modelConfigTitle.setFixedHeight(30)
        modelConfigContent.layout().setSpacing(50)
        timeStepForm.setFixedWidth(200)
        timeStepLabel.setFixedWidth(40)
        optionConfig.layout().setSpacing(50)

    def __initSpecTab(self) -> None:
        specTab = QWidget();
        self.centralTabWidget.addTab(specTab, "Specifications")
        self.centralTabWidget.setTabEnabled(1, False)

        # === Global layout ===
        specTab.setLayout(QHBoxLayout())

        self.stateSpaceView = SpaceView(STATE)
        self.specForm = SpecificationForm()

        specTab.layout().addWidget(self.stateSpaceView, stretch=2)
        specTab.layout().addWidget(self.specForm, stretch=1)
        

    def __onEquationFormChanged(self) -> None:
        isFilled = all(eq.strip() for eq in self.equationForm.getValue())
        self.buildButton.setEnabled(isFilled) 

        if isFilled:
            self.statusBar().showMessage("Ready to build automaton")
        else:
            self.statusBar().showMessage("Cannot build: not all inputs are filled")


    def __onLoadClicked(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "JSON Files (*.json)"
        )
        if not filepath:
            return
        
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            self.stateSpaceForm.setValue(config.get("stateSpace", {}))
            self.inputSpaceForm.setValue(config.get("inputSpace", {}))
            self.disturbSpaceForm.setValue(config.get("disturbanceSpace", {}))
            self.equationForm.setValue(config.get("equations", []))
            self.timeStepInput.setValue(config.get("timeStep", 0.0))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")

    def __onResetClicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Confirm Reset",
            "Reset all configurations to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            return
        
        self.stateSpaceForm.dimensionsInput.setValue(1)
        self.inputSpaceForm.dimensionsInput.setValue(1)
        self.disturbSpaceForm.dimensionsInput.setValue(1)
        
        self.timeStepInput.setValue(1.0)

    def __onSaveClicked(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Configuration", 
            "", 
            "JSON Files (*.json)"
        )
        if not filepath:
            return

        if not filepath.endswith('.json'):
            filepath += '.json'
        
        config = {
            "stateSpace": self.stateSpaceForm.getValue(),
            "inputSpace": self.inputSpaceForm.getValue(),
            "disturbanceSpace": self.disturbSpaceForm.getValue(),
            "equations": self.equationForm.getValue(),
            "timeStep": self.timeStepInput.value(),
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4)
            QMessageBox.information(self, "Success", f"Configuration saved to {filepath}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

    def __onBuildClicked(self) -> None:
        config = {
            "stateSpace": self.stateSpaceForm.getValue(),
            "inputSpace": self.inputSpaceForm.getValue(),
            "disturbanceSpace": self.disturbSpaceForm.getValue(),
            "equations": self.equationForm.getValue(),
            "timeStep": self.timeStepInput.value(),
        }
        
        self.buildSignal.emit(config)

