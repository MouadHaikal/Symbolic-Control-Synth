from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush
from PyQt6.QtCore import QRect, Qt

class GridWidget(QWidget):
    """
    Simple 2D grid widget that draws the discrete cells and can highlight one cell.
    Expects a DiscreteSpace with 2 dimensions.
    """

    def __init__(self, discrete_space, parent=None):
        super().__init__(parent)
        self.space = discrete_space
        if self.space.dimensions != 2:
            raise ValueError("GridWidget only supports 2D DiscreteSpace")

        # resolutions
        self.res_x = self.space._DiscreteSpace__resolutions[0]  # note: private slot access
        self.res_y = self.space._DiscreteSpace__resolutions[1]

        # highlighted cell coords (tuple (i, j)) or None
        self.highlight = None

        # small padding
        self.margin = 8

    def sizeHint(self):
        return super().sizeHint()

    def highlight_cell(self, coords):
        """ coords is a tuple (i, j) with 0 <= i < res_x, 0 <= j < res_y """
        self.highlight = tuple(int(c) for c in coords)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect().adjusted(self.margin, self.margin, -self.margin, -self.margin)
        width = rect.width()
        height = rect.height()

        # cell size in pixels
        cell_w = width / self.res_x
        cell_h = height / self.res_y

        # draw grid
        pen = QPen(Qt.GlobalColor.black)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for i in range(self.res_x):
            for j in range(self.res_y):
                x = rect.left() + i * cell_w
                # note: j=0 at bottom visually? we'll draw j=0 at top to match typical row 0 = top
                y = rect.top() + j * cell_h
                cell_rect = QRect(int(x), int(y), int(cell_w + 0.5), int(cell_h + 0.5))

                # fill if highlighted
                if self.highlight is not None and (i, j) == self.highlight:
                    painter.setBrush(QBrush(QColor(100, 180, 255, 180)))
                else:
                    painter.setBrush(Qt.BrushStyle.NoBrush)

                painter.drawRect(cell_rect)





    

