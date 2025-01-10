import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QFileDialog, QPushButton, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class CircleDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Advanced Circle Detector'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Image label
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.loadImage)
        self.layout.addWidget(self.load_button)

        # Detect circles button
        self.detect_button = QPushButton("Detect Circles")
        self.detect_button.clicked.connect(self.detectCircles)
        self.layout.addWidget(self.detect_button)

        # Status label
        self.status_label = QLabel()
        self.layout.addWidget(self.status_label)

        self.image = None
        self.original_image_path = ""

    def loadImage(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            self.original_image_path = file_name
            self.image = cv2.imread(file_name, cv2.IMREAD_COLOR)
            if self.image is not None:
                self.displayImage(self.image)
                self.status_label.setText("Image loaded.")
            else:
                QMessageBox.critical(self, "Error", "Unable to load image.")
        else:
            self.status_label.setText("No image selected.")

    def displayImage(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def detectCircles(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 50, 150)

        # Use HoughCircles to detect circles
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=15, maxRadius=150)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            circles = sorted(circles, key=lambda x: x[2], reverse=True)[:3]  # Keep only the three largest circles
            for (x, y, r) in circles:
                cv2.circle(self.image, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(self.image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            self.displayImage(self.image)
            self.status_label.setText("Circles detected.")
        else:
            self.status_label.setText("No circles detected.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CircleDetectorApp()
    ex.show()
    sys.exit(app.exec_())