# test.py
import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import QTimer

# Import the GridWidget from grid_widget.py
from grid_widget import GridWidget

class MockDiscreteSpace:
    def __init__(self, resolutions):
        self.dimensions = len(resolutions)
        self._DiscreteSpace__resolutions = resolutions

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Create a 5x5 grid space
        space = MockDiscreteSpace([5, 5])
        
        # Create the grid widget
        self.grid = GridWidget(space)
        self.setCentralWidget(self.grid)
        self.setWindowTitle("Grid Test")
        self.setGeometry(100, 100, 400, 400)
        
        # Test highlighting
        self.grid.highlight_cell((2, 3))  # Highlight cell at row 2, column 3

def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()