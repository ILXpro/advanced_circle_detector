import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT, FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

class ResultsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Circle Detection Results")
        self.setGeometry(300, 300, 400, 300)
        
        layout = QVBoxLayout()
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)
        
        copy_btn = QPushButton("Copy Results")
        copy_btn.clicked.connect(self.copyResults)
        layout.addWidget(copy_btn)
        
        self.setLayout(layout)

    def updateResults(self, circles_data):
        text = "Circle Detection Results:\n\n"
        for i, (x, y, r, intensity_drop) in enumerate(circles_data, 1):
            text += f"Circle {i}:\n"
            text += f"Center: ({x:.2f}, {y:.2f})\n"
            text += f"Radius: {r:.2f} pixels\n"
            text += f"Intensity Drop: {intensity_drop:.1f}%\n\n"
        self.text_area.setText(text)

    def copyResults(self):
        QApplication.clipboard().setText(self.text_area.toPlainText())

class ImageCanvas(QGraphicsView):
    [Previous ImageCanvas implementation remains the same]

class CircleDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Advanced Circle Detector'
        self.image = None
        self.processed_image = None
        self.brush_mode = False
        self.results_window = ResultsWindow(self)
        self.initUI()

    def initUI(self):
        [Previous UI implementation with added brush toggle button]
        
        self.brush_toggle = QPushButton("Toggle Brush Mode")
        self.brush_toggle.setCheckable(True)
        self.brush_toggle.clicked.connect(self.toggleBrushMode)
        control_layout.addWidget(self.brush_toggle)

    def toggleBrushMode(self):
        self.brush_mode = self.brush_toggle.isChecked()
        if not self.brush_mode:
            self.image_canvas.clearOverlay()

    def getIntensityProfile(self, im
