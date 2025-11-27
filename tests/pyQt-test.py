# main.py
import sys
import os


print("--- DEBUG INFO ---")
print(f"Current Working Directory: {os.getcwd()}")
print("Python's Search Path (sys.path):")
for path in sys.path:
    print(f"  - {path}")
print("--------------------")
# --------------------------------

from PyQt6.QtWidgets import QApplication
from gui.app import MainWindow

from PyQt6.QtWidgets import QApplication
from gui.app import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
