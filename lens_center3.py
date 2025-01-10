import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QPushButton, QLabel, QSlider, QMessageBox, QScrollArea
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

class PaintApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.brush_size = 5  # Default brush size as a percentage of the image size
        self.drawing = False
        self.brush_mode = True  # Brush mode is on by default
        self.image_path = None
        self.image = None
        self.scaled_image = None
        self.mask = None
        self.lines_drawn = []
        self.scale_factor = 1.0
        self.zoom_rect = None
        self.selecting_zoom_area = False
        self.initUI()  # Initialize the UI after setting attributes

    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle("Simple Paint Program")
        self.setGeometry(100, 100, 800, 600)  # Standard window size

        # Central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Scroll Area for Canvas
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.canvas_label = QLabel(self)
        self.canvas_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll_area.setWidget(self.canvas_label)
        
        # Initialize toolbars
        self.initToolbar1()
        self.initToolbar2()
        self.initToolbar3()

        # Add widgets to layout
        self.main_layout.addLayout(self.toolbar_layout1)
        self.main_layout.addLayout(self.toolbar_layout2)
        self.main_layout.addLayout(self.toolbar_layout3)
        self.main_layout.addWidget(self.scroll_area)

        # Set brush button initial state
        self.update_brush_button()

    def initToolbar1(self):
        """Initialize the first toolbar (Loading and Analyzing the Image)."""
        self.toolbar_layout1 = QHBoxLayout()

        # Load image button
        load_image_button = QPushButton("Load Image")
        load_image_button.setToolTip("Click to load an image")
        load_image_button.clicked.connect(self.load_image)

        # Search for circles button
        search_circles_button = QPushButton("Search for Circles")
        search_circles_button.setToolTip("Search for circles in the painted area")
        search_circles_button.clicked.connect(self.search_for_circles)

        # Add buttons to the toolbar
        self.toolbar_layout1.addWidget(load_image_button)
        self.toolbar_layout1.addWidget(search_circles_button)

    def initToolbar2(self):
        """Initialize the second toolbar (Brush Activation/Deactivation, Drawing Lines, and Line Removal)."""
        self.toolbar_layout2 = QHBoxLayout()

        # Toggle brush mode button
        self.toggle_brush_button = QPushButton()
        self.toggle_brush_button.setToolTip("Enable/Disable brush mode")
        self.toggle_brush_button.clicked.connect(self.toggle_brush_mode)

        # Brush size slider
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 50)
        self.brush_size_slider.setValue(5)  # Default to 5%
        self.brush_size_slider.setToolTip("Adjust Brush Size")
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)

        # Undo last line button
        undo_button = QPushButton("Undo Last Line")
        undo_button.setToolTip("Undo the last drawn line")
        undo_button.clicked.connect(self.undo_last_line)

        # Clear all lines button
        clear_button = QPushButton("Clear All Lines")
        clear_button.setToolTip("Clear all drawn lines")
        clear_button.clicked.connect(self.clear_all_lines)

        # Add buttons and slider to the toolbar
        self.toolbar_layout2.addWidget(self.toggle_brush_button)
        self.toolbar_layout2.addWidget(QLabel("Brush Size (% of image):"))
        self.toolbar_layout2.addWidget(self.brush_size_slider)
        self.toolbar_layout2.addWidget(undo_button)
        self.toolbar_layout2.addWidget(clear_button)

    def initToolbar3(self):
        """Initialize the third toolbar (Working with the Image: Adding Area and Image Zoom/Zoom Out)."""
        self.toolbar_layout3 = QHBoxLayout()

        # Zoom in button
        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.setToolTip("Zoom in on the image")
        zoom_in_button.clicked.connect(self.zoom_in)

        # Zoom out button
        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.setToolTip("Zoom out of the image")
        zoom_out_button.clicked.connect(self.zoom_out)

        # Select zoom area button
        select_zoom_area_button = QPushButton("Select Zoom Area")
        select_zoom_area_button.setToolTip("Select an area to zoom in")
        select_zoom_area_button.clicked.connect(self.select_zoom_area)

        # Add buttons to the toolbar
        self.toolbar_layout3.addWidget(zoom_in_button)
        self.toolbar_layout3.addWidget(zoom_out_button)
        self.toolbar_layout3.addWidget(select_zoom_area_button)

    def load_image(self):
        """Load an image from file."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)

        if file_name:
            self.image_path = file_name
            self.image = cv2.imread(file_name, cv2.IMREAD_COLOR)  # Load image in color
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            self.scaled_image = self.resize_image_to_fit()
            self.canvas_image = self.scaled_image.copy()
            self.mask = np.zeros(self.scaled_image.shape[:2], dtype=np.uint8)
            self.display_image(self.scaled_image)
            self.update_brush_size()  # Update the brush size based on the loaded image
            self.lines_drawn = []  # Reset the lines drawn

    def resize_image_to_fit(self):
        """Resize the image to fit within the window while preserving aspect ratio."""
        max_width = self.canvas_label.width()
        max_height = self.canvas_label.height()
        height, width, _ = self.image.shape

        # Calculate the scaling factor while maintaining aspect ratio
        scale_factor = min(max_width / width, max_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image
        return cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def display_image(self, image):
        """Display the image on the canvas."""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.canvas_label.setPixmap(pixmap)
        self.canvas_label.setFixedSize(width, height)

    def update_brush_size(self):
        """Update the brush size based on the slider value."""
        if self.scaled_image is not None:
            max_side = max(self.scaled_image.shape[:2])
            self.brush_size = int((self.brush_size_slider.value() / 100) * max_side)

    def toggle_brush_mode(self):
        """Toggle the brush mode on and off."""
        self.brush_mode = not self.brush_mode
        self.update_brush_button()

    def update_brush_button(self):
        """Update the brush button text and appearance based on the brush mode."""
        if self.brush_mode:
            self.toggle_brush_button.setText("Brush On")
        else:
            self.toggle_brush_button.setText("Brush Off")

    def mousePressEvent(self, event):
        """Handle mouse press event for drawing with the brush or selecting zoom area."""
        if event.button() == Qt.LeftButton and self.scaled_image is not None:
            if self.brush_mode:
                self.drawing = True
                self.last_point = self.canvas_label.mapFromParent(event.pos())
            elif self.selecting_zoom_area:
                self.zoom_rect = QRect(event.pos(), QSize())
                self.selecting_zoom_area = False

    def mouseMoveEvent(self, event):
        """Handle mouse move event for drawing with the brush or updating zoom area."""
        if self.drawing and self.scaled_image is not None and self.brush_mode:
            current_point = self.canvas_label.mapFromParent(event.pos())
            self.draw_line(self.last_point, current_point)
            self.last_point = current_point
        elif self.zoom_rect is not None:
            self.zoom_rect.setBottomRight(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release event to stop drawing or perform zoom."""
        if event.button() == Qt.LeftButton and self.scaled_image is not None:
            if self.drawing:
                self.drawing = False
                self.lines_drawn.append((self.last_point, self.canvas_label.mapFromParent(event.pos())))  # Store the line points
            elif self.zoom_rect is not None:
                self.zoom_in_on_selected_area()

    def draw_line(self, start_point, end_point):
        """Draw a line on the image and update the mask."""
        painter = QPainter(self.canvas_label.pixmap())
        painter.setPen(QPen(QColor(255, 0, 0, 127), self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(start_point, end_point)
        painter.end()

        # Update the mask
        cv2.line(self.mask, (start_point.x(), start_point.y()), (end_point.x(), end_point.y()), 255, self.brush_size)

        self.update()

    def undo_last_line(self):
        """Undo the last drawn line."""
        if self.lines_drawn:
            self.lines_drawn.pop()
            self.reset_canvas()
            self.redraw_lines()

    def clear_all_lines(self):
        """Clear all drawn lines."""
        self.lines_drawn.clear()
        self.reset_canvas()

    def reset_canvas(self):
        """Reset the canvas to the original image without lines."""
        self.canvas_label.setPixmap(QPixmap.fromImage(QImage(self.scaled_image.data, self.scaled_image.shape[1], self.scaled_image.shape[0], 3 * self.scaled_image.shape[1], QImage.Format_RGB888)))
        self.mask = np.zeros(self.scaled_image.shape[:2], dtype=np.uint8)

    def redraw_lines(self):
        """Redraw all stored lines on the canvas."""
        for line in self.lines_drawn:
            self.draw_line(line[0], line[1])

    def search_for_circles(self):
        """Search for circles in the painted area and ensure the number of circles matches the number of lines."""
        if self.scaled_image is not None and np.any(self.mask):
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(self.scaled_image, cv2.COLOR_RGB2GRAY)

            # Use Canny edge detection
            edges = cv2.Canny(gray_image, 100, 200)

            # Apply the mask to the edges
            masked_edges = cv2.bitwise_and(edges, edges, mask=self.mask)

            # Detect contours
            contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours that are approximately circular
            circles = []
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                if len(approx) > 8:  # Check if contour is approximately circular
                    circles.append(cv2.minEnclosingCircle(contour))

            # Draw the detected circles on the image
            for (x, y), radius in circles:
                cv2.circle(self.scaled_image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(self.scaled_image, (int(x), int(y)), 2, (0, 0, 255), 3)

            self.display_image(self.scaled_image)
            print(f"Number of circles found: {len(circles)}")
            QMessageBox.information(self, "Circles Found", f"Number of circles found: {len(circles)}")
        else:
            circles = []
            print("Please paint an area to search for circles.")
            QMessageBox.information(self, "No Mask", "Please paint an area to search for circles.")
        
        # Output the number of lines drawn
        print(f"Number of lines drawn: {len(self.lines_drawn)}")
        
        # Compare the number of circles and lines
        if len(circles) == len(self.lines_drawn):
            print("The number of detected circles matches the number of drawn lines.")
        else:
            print("The number of detected circles does not match the number of drawn lines.")

    def zoom_in(self):
        """Zoom in on the image."""
        self.scale_image(1.2)

    def zoom_out(self):
        """Zoom out of the image."""
        self.scale_image(0.8)

    def select_zoom_area(self):
        """Enable selection of zoom area."""
        self.selecting_zoom_area = True
        QMessageBox.information(self, "Zoom Area", "Click and drag to select the area to zoom in.")

    def zoom_in_on_selected_area(self):
        """Zoom in on the selected area."""
        if self.zoom_rect is not None and not self.zoom_rect.isEmpty():
            self.scale_factor = 1.0
            cropped_image = self.canvas_label.pixmap().copy(self.zoom_rect)
            self.canvas_label.setPixmap(cropped_image.scaled(self.canvas_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.zoom_rect = None

    def scale_image(self, factor):
        """Scale the image by the given factor."""
        self.scale_factor *= factor
        new_size = self.canvas_label.pixmap().size() * self.scale_factor
        self.canvas_label.resize(new_size)
        self.canvas_label.setPixmap(self.canvas_label.pixmap().scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.scroll_area.horizontalScrollBar().setValue((self.canvas_label.width() - self.scroll_area.viewport().width()) // 2)
        self.scroll_area.verticalScrollBar().setValue((self.canvas_label.height() - self.scroll_area.viewport().height()) // 2)

    def wheelEvent(self, event):
        """Handle the wheel event for zooming using the touchpad."""
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    viewer = PaintApp()
    viewer.show()
    sys.exit(app.exec_())