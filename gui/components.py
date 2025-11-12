# gui/components.py
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QLabel, QHBoxLayout
)
from PyQt6.QtCore import Qt

class SpaceInputWidget(QWidget):
    """
    A self-contained widget used in each tab (state/control/disturbance)
    to collect inputs required to build a DiscreteSpace:
      - name
      - dimensions
      - lower bounds
      - upper bounds
      - resolutions
    """
    def __init__(self, space_name: str):
        super().__init__()
        self.space_name = space_name
        self.layout = QFormLayout()
        self.setLayout(self.layout)
        
        # --- Name ---
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText(space_name)
        self.layout.addRow(QLabel("Space Name:"), self.name_input)
        # --- Dimensions ---

        self.dim_input = QLineEdit()
        self.dim_input.setPlaceholderText("e.g. 2")
        self.layout.addRow(QLabel("Dimensions:"), self.dim_input)
        # --- Lower / Upper Bounds + Resolutions --
        # -
        self.lower_input = QLineEdit()
        self.lower_input.setPlaceholderText("e.g. 0, -2")
        self.layout.addRow(QLabel("Lower Bounds:"), self.lower_input)
        self.upper_input = QLineEdit()
        self.upper_input.setPlaceholderText("e.g. 10, 2")
        self.layout.addRow(QLabel("Upper Bounds:"), self.upper_input)

        self.res_input = QLineEdit()
        self.res_input.setPlaceholderText("e.g. 20, 20")
        self.layout.addRow(QLabel("Resolution:"), self.res_input)

    # -------------------------------------------------------------
    # Extracts the values entered by the user
    # -------------------------------------------------------------
    def read_values(self):
        """
        Returns:
            dict with keys: name, dimensions, lowerBounds, upperBounds, resolutions
        Throws ValueError if conversion fails.
        """
        name = self.name_input.text().strip()
        if not name:
            raise ValueError(f"{self.space_name}: name cannot be empty")

        try:
            dim = int(self.dim_input.text().strip())
        except:
            raise ValueError(f"{self.space_name}: dimensions must be an integer")

        def parse_list(text, expected_len):
            parts = text.split(",")
            if len(parts) != expected_len:
                raise ValueError(
                    f"{self.space_name}: expected {expected_len} values but got {len(parts)}"
                )
            return [float(x.strip()) for x in parts]

        # Validate list lengths based on dimension
        lower = parse_list(self.lower_input.text(), dim)
        upper = parse_list(self.upper_input.text(), dim)
        res = [int(x) for x in parse_list(self.res_input.text(), dim)]

        return {
            "name": name,
            "dimensions": dim,
            "lowerBounds": lower,
            "upperBounds": upper,
            "resolutions": res
        }
