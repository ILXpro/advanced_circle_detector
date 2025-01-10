import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QFileDialog, QPushButton, QLabel, QSlider, QLineEdit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer with Manual Segmentation")
        self.setGeometry(100, 100, 800, 600)

        # Central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Matplotlib plot area with navigation toolbar
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)

        # Create a toolbar layout
        toolbar_layout = QtWidgets.QToolBar(self)

        # Load Image Button
        load_image_button = QPushButton("Load Image")
        load_image_button.setIcon(QtGui.QIcon("icons/load.png"))  # Set your icon path here
        load_image_button.setToolTip("Click to load an image")
        load_image_button.clicked.connect(self.load_image)

        # Brush Activation Button
        self.brush_active = False  # Track whether the brush is active
        brush_button = QPushButton("Toggle Brush")
        brush_button.setToolTip("Activate or deactivate the brush tool")
        brush_button.clicked.connect(self.toggle_brush)

        # Brush Size Slider
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 50)  # Set range for brush size
        self.brush_size_slider.setValue(10)  # Default brush size
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)

        # Analyze Button
        analyze_button = QPushButton("Analyze Segmentation")
        analyze_button.setToolTip("Analyze segmented area for circles")
        analyze_button.clicked.connect(self.analyze_segmented_area)

        # Input Box for Maximum Circle Detection
        self.max_circles_input = QLineEdit()
        self.max_circles_input.setPlaceholderText("Max Circles (0 for no limit)")
        
        # Add buttons and slider to the toolbar layout
        toolbar_layout.addWidget(load_image_button)  # Add load button to the toolbar
        toolbar_layout.addWidget(brush_button)  # Add brush toggle button to the toolbar
        toolbar_layout.addWidget(QLabel("Brush Size:"))  # Label for brush size slider
        toolbar_layout.addWidget(self.brush_size_slider)  # Add brush size slider to the toolbar
        toolbar_layout.addWidget(QLabel("Max Circles:"))  # Label for max circles input
        toolbar_layout.addWidget(self.max_circles_input)  # Add max circles input to the toolbar
        toolbar_layout.addWidget(analyze_button)  # Add analyze button to the toolbar

        # Add navigation toolbar for zooming and panning
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Add both the buttons and navigation toolbar to the layout
        self.layout.addWidget(toolbar_layout)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # Brush settings
        self.brush_size = 10  # Default brush size
        self.drawing = False  # Drawing state
        self.image_path = None  # Loaded image path

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        
        if file_name:
            self.image_path = file_name  # Store the path of the loaded image
            
            img_data = plt.imread(file_name)  # Load the image data using Matplotlib
            
            # Resize image for performance (optional)
            img_data_resized = cv2.resize(img_data, (800, 600))  # Resize to fit window
            
            ax = self.figure.add_subplot(111)  # Create a subplot for the image
            ax.clear()
            ax.imshow(img_data_resized)
            ax.axis('off')  # Hide axes
            
            self.canvas.draw()  # Refresh the canvas to show the new plot

            # Initialize drawing canvas for segmentation mask (black mask)
            self.mask = np.zeros(img_data_resized.shape[:2], dtype=np.uint8)  # Create a mask for segmentation

    def update_brush_size(self):
        """Update brush size based on slider value."""
        self.brush_size = self.brush_size_slider.value()

    def toggle_brush(self):
        """Toggle brush activation state."""
        self.brush_active = not self.brush_active

    def analyze_segmented_area(self):
         """Analyze segmented area for circles."""
         if hasattr(self, 'image_path'):
            image = cv2.imread(self.image_path)

            blurred_masked_image = cv2.GaussianBlur(image, (5, 5), 0)
            edges = cv2.Canny(blurred_masked_image, 100, 200)

            max_circles_str = self.max_circles_input.text()
            max_circles = int(max_circles_str) if max_circles_str.isdigit() else 0

            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                       param1=50, param2=30,
                                       minRadius=0, maxRadius=0)

            if circles is not None:
                circles = np.uint16(np.around(circles))  # Convert float to int
                
                result_image = image.copy()
                detected_count = min(len(circles[0]), max_circles) if max_circles > 0 else len(circles[0])
                
                for i in circles[0][:detected_count]:
                    cv2.circle(result_image, (i[0], i[1]), i[2], (0, 255, 0), 4)   # Draw outer circle in green.
                    cv2.circle(result_image, (i[0], i[1]), 2, (0, 0, 255), 3)      # Draw center of circle in red.

                ax = self.figure.add_subplot(111)
                ax.clear()
                ax.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)) 
                ax.axis('off')  

                size_info = f"Detected {detected_count} circles."
                print(size_info)  
                
                self.canvas.draw()  

    def mousePressEvent(self, event):
         if event.button() == Qt.LeftButton and self.brush_active:
             self.drawing = True
            
    def mouseMoveEvent(self, event):
         if self.drawing and self.brush_active:
             painter = QtGui.QPainter(self.mask)
             painter.setPen(QtGui.QPen(QtGui.QColor(255), self.brush_size))
             painter.drawPoint(event.pos())

             ax = self.figure.add_subplot(111)
             ax.imshow(cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)) 
             
             circle_patch = plt.Circle(event.pos(), radius=self.brush_size / 2,
                                       color='yellow', alpha=0.5)
             ax.add_patch(circle_patch)

             # Refresh canvas to show brush overlay.
             self.canvas.draw()

    def mouseReleaseEvent(self, event):
         if event.button() == Qt.LeftButton:
             self.drawing = False

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
