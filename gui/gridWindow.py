from PyQt6.QtWidgets import (
    QWidget, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QVBoxLayout, QLineEdit, QLabel, QCheckBox, QTextEdit, QPushButton
)
from PyQt6.QtGui import QBrush, QColor, QPen, QPainterPath
from PyQt6.QtCore import Qt


class GridWindow(QWidget):
    """
    GUI window for visualizing robot grid, start/goal, obstacles, and paths.
    """

    def __init__(self, model, automaton, pixelScale=40):
        super().__init__()
        self.model = model
        self.automaton = automaton
        self.pixelScale = pixelScale
        self.bounds = model.stateSpace.bounds
        self.rows, self.cols = model.stateSpace.resolutions
        cellSizeY, cellSizeX = model.stateSpace.cellSize
        self.cellSizeY = cellSizeY * pixelScale
        self.cellSizeX = cellSizeX * pixelScale

        # Precompute coordinate → index scale factors
        (xmin, xmax), (ymin, ymax) = self.bounds
        y_span = ymax - ymin if (ymax - ymin) != 0 else 1.0
        x_span = xmax - xmin if (xmax - xmin) != 0 else 1.0

        self._ymin = ymin
        self._xmin = xmin
        self.yScale = y_span / self.rows
        self.xScale = x_span / self.cols

        self.startState = None
        self.goal = None
        self.obstacles = []
        self.pathItem = None

        self._initUI()
        self.createGrid()

    # ======================== UI SETUP ========================

    def _initUI(self):
        """
        Initialize layouts, inputs, and buttons.
        """
        mainLayout = QVBoxLayout()
        self.setLayout(mainLayout)

        # Graphics view
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        mainLayout.addWidget(self.view, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.view.setFixedSize(int(self.cols * self.cellSizeX + 10),
                               int(self.rows * self.cellSizeY + 10))

        # Inputs and buttons
        self._createInputs(mainLayout)

    def _createInputs(self, parentLayout):
        """
        Create start/goal/obstacle inputs and buttons.
        """
        layout = QVBoxLayout()
        parentLayout.addLayout(layout)

        self.gridToggle = self._addCheckBox(layout, "Show Grid", False, self.updateGridPen)

        self.startInput = self._addLineEdit(layout, "Start (x,y):", "e.g. 2.5, 1.3")
        self.goalInput = self._addLineEdit(layout, "Goal [(ymin,ymax),(xmin,xmax)]:", "[(2,4),(3,5)]")
        self.obsInput = self._addTextEdit(layout, "Obstacles [[(ymin,ymax),(xmin,xmax)], ...]:", 
                                          "[[(1,2),(1,3)], [(4,6),(0,2)]]", 60)

        self.addButton = self._addButton(layout, "Add / Refresh", self.addElements)
        self.testButton = self._addButton(layout, "Test Path", self.getPath)

    def _addLineEdit(self, layout, label, placeholder=""):
        layout.addWidget(QLabel(label))
        le = QLineEdit()
        le.setPlaceholderText(placeholder)
        layout.addWidget(le)
        return le

    def _addTextEdit(self, layout, label, placeholder="", height=50):
        layout.addWidget(QLabel(label))
        te = QTextEdit()
        te.setPlaceholderText(placeholder)
        te.setFixedHeight(height)
        layout.addWidget(te)
        return te

    def _addButton(self, layout, text, callback):
        btn = QPushButton(text)
        btn.clicked.connect(callback)
        layout.addWidget(btn)
        return btn

    def _addCheckBox(self, layout, text, default, callback):
        cb = QCheckBox(text)
        cb.setChecked(default)
        cb.stateChanged.connect(callback)
        layout.addWidget(cb)
        return cb

    # ======================== GRID ========================

    def createGrid(self):
        """
        Create the white grid with optional grid lines.
        """
        self.scene.clear()
        self.gridItems = []
        pen = QPen(QColor(200, 200, 200)) if self.gridToggle.isChecked() else QPen(Qt.PenStyle.NoPen)

        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                self.cellSizeX = round(self.cellSizeX)
                self.cellSizeY = round(self.cellSizeY)
                rect = QGraphicsRectItem(c * self.cellSizeX, r * self.cellSizeY,
                                         self.cellSizeX, self.cellSizeY)
                rect.setBrush(QBrush(Qt.GlobalColor.white))
                rect.setPen(pen)
                self.scene.addItem(rect)
                row.append(rect)
            self.gridItems.append(row)

    def updateGridPen(self):
        """
        Toggle grid lines on/off.
        """
        pen = QPen(QColor(200, 200, 200)) if self.gridToggle.isChecked() else QPen(Qt.PenStyle.NoPen)
        for row in self.gridItems:
            for cell in row:
                cell.setPen(pen)

    def clearGridColors(self):
        """
        Reset cell backgrounds without deleting grid items.
        """
        for row in self.gridItems:
            for cell in row:
                cell.setBrush(QBrush(Qt.GlobalColor.white))


    # ======================== HELPERS ========================

    def mapPointToGrid(self, point):
        """
        Map continuous (x,y) to (row,col) using precomputed scale factors.
        """
        x, y = point
        row = int((y - self._ymin) * self.yScale)
        col = int((x - self._xmin) * self.xScale)
        row = max(0, min(self.rows - 1, row))
        col = max(0, min(self.cols - 1, col))
        return row, col

    def colorRegion(self, region, color):
        """
        Color a continuous rectangular region on the grid.
        """
        try:
            (ymin, ymax), (xmin, xmax) = region
        except:
            print("Invalid region:", region)
            return

        rmin = max(0, int((ymin - self._ymin) * self.yScale))
        rmax = min(self.rows - 1, int((ymax - self._ymin) * self.yScale))
        cmin = max(0, int((xmin - self._xmin) * self.xScale))
        cmax = min(self.cols - 1, int((xmax - self._xmin) * self.xScale))


        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                self.gridItems[r][c].setBrush(QBrush(color))

    # ======================== ELEMENTS ========================

    def addElements(self):
        """
        Refresh grid and draw start, goal, and obstacles.
        """
        self.clearGridColors()
        self.obstacles = self._parseInput(self.obsInput.toPlainText(), default=[])
        for obs in self.obstacles:
            self.colorRegion(obs, Qt.GlobalColor.red)

        self.goal = self._parseInput(self.goalInput.text(), default=None)
        if self.goal:
            self.colorRegion(self.goal, Qt.GlobalColor.green)

        self.startState = self._parsePoint(self.startInput.text())
        if self.startState:
            r, c = self.mapPointToGrid(self.startState)
            self.gridItems[r][c].setBrush(QBrush(Qt.GlobalColor.blue))

        if self.pathItem:
            self.scene.removeItem(self.pathItem)
            self.pathItem = None

    def _parseInput(self, text, default=None):
        """
        Safely parse Python literals (list, tuple, numbers) using ast.literal_eval.
        """
        import ast
        if not text:
            return default
        try:
            return ast.literal_eval(text)
        except Exception:
            return default

    def _parsePoint(self, text):
        try:
            return tuple(map(float, text.split(",")))
        except:
            return None

    # ======================== PATH ========================

    def showPath(self, cell_path):
        """
        Draw a blue polyline across discrete cells.
        """
        if self.pathItem:
            self.scene.removeItem(self.pathItem)
            self.pathItem = None
        if not cell_path or len(cell_path) < 2:
            return

        path = QPainterPath()
        r0, c0 = cell_path[0]
        path.moveTo(c0 * self.cellSizeX + self.cellSizeX / 2,
                    r0 * self.cellSizeY + self.cellSizeY / 2)
        for r, c in cell_path[1:]:
            path.lineTo(c * self.cellSizeX + self.cellSizeX / 2,
                        r * self.cellSizeY + self.cellSizeY / 2)

        self.pathItem = self.scene.addPath(path, QPen(Qt.GlobalColor.blue, 4))
        self.pathItem.setZValue(20)

    def getPath(self):
        
        for obs in self.obstacles:
            self.automaton.applySecuritySpec(obs[0], obs[1])
        controller = self.automaton.getController(
            self.continuousToIndex(self.startState[0], self.startState[1]), 
            tuple([self.goal[1][0], self.goal[0][0]]), 
            tuple([self.goal[1][1], self.goal[0][1]])
        )

        for i in range(0, len(controller)):
            controller[i] = self.indexToState(controller[i])
        self.showPath(controller)
    
    def coordsToIndex(self, x: int, y: int) -> int:
        cols = self.model.stateSpace.resolutions[0]
        return y * cols + x


    def continuousToIndex(self, xReal: float, yReal: float) -> int:
        xs_bounds = self.model.stateSpace.bounds[0]
        ys_bounds = self.model.stateSpace.bounds[1]

        xsRes = self.model.stateSpace.resolutions[0]
        ysRes = self.model.stateSpace.resolutions[1]

        # Compute cell sizes
        dx = (xs_bounds[1] - xs_bounds[0]) / xsRes
        dy = (ys_bounds[1] - ys_bounds[0]) / ysRes

        ix = int((xReal - xs_bounds[0]) / dx)
        iy = int((yReal - ys_bounds[0]) / dy)

        return self.coordsToIndex(ix, iy)

    def indexToState(self, index: int) -> list:
        cols = self.model.stateSpace.resolutions[0]

        x = index % cols   
        y = index // cols  

        state = (x, y)  

        return state

