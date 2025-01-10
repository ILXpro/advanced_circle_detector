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
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize
import math

MAX_IMAGE_SIZE = 800  # Maximum size for initial processing
MIN_POINTS_FOR_CIRCLE = 5  # Minimum points needed for circle fitting

class ResultsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lens Analysis Results")
        self.setGeometry(300, 300, 500, 400)
        layout = QVBoxLayout()
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)
        export_layout = QHBoxLayout()
        copy_btn = QPushButton("Copy Results")
        save_btn = QPushButton("Save to File")
        copy_btn.clicked.connect(self.copyResults)
        save_btn.clicked.connect(self.saveResults)
        export_layout.addWidget(copy_btn)
        export_layout.addWidget(save_btn)
        layout.addLayout(export_layout)
        self.setLayout(layout)

    def updateResults(self, circles_data):
        text = "Lens Analysis Results:\n\n"
        for i, (x, y, r, intensity_drop, concentricity) in enumerate(circles_data, 1):
            text += f"Circle {i}:\n"
            text += f"Center (x, y): ({x:.3f}, {y:.3f}) pixels\n"
            text += f"Radius: {r:.3f} pixels\n"
            text += f"Intensity Drop: {intensity_drop:.2f}%\n"
            text += f"Concentricity: {concentricity:.3f} pixels\n\n"
        self.text_area.setText(text)

    def copyResults(self):
        QApplication.clipboard().setText(self.text_area.toPlainText())

    def saveResults(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Results", "", 
                                                "Text Files (*.txt);;All Files (*)")
        if filename:
            with open(filename, 'w') as f:
                f.write(self.text_area.toPlainText())

class ImageCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.drawing = False
        self.last_point = None
        self.brush_size = 10
        self.brush_color = QColor(0, 255, 0, 127)
        self.image_item = None
        self.overlay_item = None
        self.scale_factor = 1.0
        self.zoom_mode = False
        self.points = []  # Store drawn points for curve fitting

    def toggleZoomMode(self, enabled):
        self.zoom_mode = enabled
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if self.zoom_mode:
            if event.button() == Qt.LeftButton:
                self.scale(1.2, 1.2)
            elif event.button() == Qt.RightButton:
                self.scale(0.8, 0.8)
        else:
            if event.button() == Qt.LeftButton:
                self.drawing = True
                point = self.mapToScene(event.pos())
                self.last_point = point
                self.points.append((point.x(), point.y()))

    def mouseMoveEvent(self, event):
        if not self.zoom_mode and self.drawing and self.last_point and self.overlay_item:
            new_point = self.mapToScene(event.pos())
            painter = QPainter(self.overlay_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(self.brush_color, self.brush_size/self.scale_factor, 
                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, new_point)
            painter.end()
            self.overlay_item.setPixmap(self.overlay_pixmap)
            self.last_point = new_point
            self.points.append((new_point.x(), new_point.y()))
            self.scene.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_point = None

    def wheelEvent(self, event):
        if self.zoom_mode:
            factor = 1.2 if event.angleDelta().y() > 0 else 0.8
            self.scale(factor, factor)
            self.scale_factor *= factor

    def fitInView(self, rect, flags=Qt.KeepAspectRatio):
        viewport_rect = self.viewport().rect()
        scene_rect = self.transform().mapRect(rect)
        factor = min(viewport_rect.width() / scene_rect.width(),
                    viewport_rect.height() / scene_rect.height())
        self.scale(factor, factor)
        self.scale_factor = factor

    def setImage(self, pixmap):
        if self.image_item:
            self.scene.removeItem(self.image_item)
        self.image_item = self.scene.addPixmap(pixmap)
        rect = pixmap.rect()
        self.scene.setSceneRect(QRectF(rect.x(), rect.y(), rect.width(), rect.height()))
        self.fitInView(self.scene.sceneRect())
        self.clearOverlay()

    def clearOverlay(self):
        if self.overlay_item:
            self.scene.removeItem(self.overlay_item)
        size = self.scene.sceneRect().size().toSize()
        self.overlay_pixmap = QPixmap(size)
        self.overlay_pixmap.fill(Qt.transparent)
        self.overlay_item = self.scene.addPixmap(self.overlay_pixmap)
        self.points = []  # Clear stored points

class CircleDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Advanced Lens Analyzer'
        self.image = None
        self.original_image = None  # Store original image
        self.processed_image = None
        self.results_window = ResultsWindow(self)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Viewer setup
        viewer_widget = QWidget()
        viewer_layout = QVBoxLayout(viewer_widget)
        
        fig = Figure(figsize=(1, 1))
        canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(canvas, self)
        viewer_layout.addWidget(self.toolbar)
        
        self.image_canvas = ImageCanvas()
        viewer_layout.addWidget(self.image_canvas)
        
        layout.addWidget(viewer_widget, stretch=2)

        # Control panel setup
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Detection controls
        detect_group = QGroupBox("Detection Controls")
        detect_layout = QVBoxLayout()
        
        open_btn = QPushButton("Open Image")
        open_btn.clicked.connect(self.openImage)
        detect_layout.addWidget(open_btn)
        
        self.auto_btn = QPushButton("Auto Detect")
        self.auto_btn.clicked.connect(self.autoDetect)
        detect_layout.addWidget(self.auto_btn)
        
        manual_btn = QPushButton("Detect in ROI")
        manual_btn.clicked.connect(self.detectInROI)
        detect_layout.addWidget(manual_btn)
        
        detect_group.setLayout(detect_layout)
        control_layout.addWidget(detect_group)

        # Mode toggles
        mode_group = QGroupBox("View Modes")
        mode_layout = QVBoxLayout()
        
        self.brush_toggle = QPushButton("Brush Mode")
        self.brush_toggle.setCheckable(True)
        self.brush_toggle.clicked.connect(self.toggleBrushMode)
        mode_layout.addWidget(self.brush_toggle)
        
        self.zoom_toggle = QPushButton("Zoom Mode")
        self.zoom_toggle.setCheckable(True)
        self.zoom_toggle.clicked.connect(self.toggleZoomMode)
        mode_layout.addWidget(self.zoom_toggle)
        
        mode_group.setLayout(mode_layout)
        control_layout.addWidget(mode_group)

        # Brush settings
        brush_group = QGroupBox("Brush Settings")
        brush_layout = QVBoxLayout()
        
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 50)
        self.brush_size_slider.setValue(10)
        self.brush_size_slider.valueChanged.connect(self.updateBrushSize)
        brush_layout.addWidget(QLabel("Brush Size:"))
        brush_layout.addWidget(self.brush_size_slider)
        
        clear_btn = QPushButton("Clear Brush")
        clear_btn.clicked.connect(self.clearBrush)
        brush_layout.addWidget(clear_btn)
        
        brush_group.setLayout(brush_layout)
        control_layout.addWidget(brush_group)

        # Image adjustments
        adjust_group = QGroupBox("Image Adjustments")
        adjust_layout = QVBoxLayout()
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.updateImage)
        adjust_layout.addWidget(QLabel("Brightness:"))
        adjust_layout.addWidget(self.brightness_slider)
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.updateImage)
        adjust_layout.addWidget(QLabel("Contrast:"))
        adjust_layout.addWidget(self.contrast_slider)
        
        adjust_group.setLayout(adjust_layout)
        control_layout.addWidget(adjust_group)

        control_layout.addStretch()
        layout.addWidget(control_panel)

    def resizeImage(self, image):
        height, width = image.shape[:2]
        if max(height, width) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    def openImage(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                "Image Files (*.png *.jpg *.bmp)")
        if filename:
            self.original_image = cv2.imread(filename)
            if self.original_image is not None:
                self.image = self.resizeImage(self.original_image.copy())
                self.processed_image = self.image.copy()
                self.updateDisplay()
                self.auto_btn.setEnabled(True)

    def updateDisplay(self):
        if self.processed_image is not None:
            height, width = self.processed_image.shape[:2]
            bytesPerLine = 3 * width
            qImg = QImage(self.processed_image.data, width, height, bytesPerLine,
                         QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            self.image_canvas.setImage(pixmap)

    def toggleBrushMode(self):
        enabled = self.brush_toggle.isChecked()
        self.image_canvas.zoom_mode = False
        self.zoom_toggle.setChecked(False)
        if not enabled:
            self.image_canvas.clearOverlay()

    def toggleZoomMode(self):
        enabled = self.zoom_toggle.isChecked()
        self.image_canvas.zoom_mode = enabled
        self.brush_toggle.setChecked(False)
        self.image_canvas.toggleZoomMode(enabled)

    def updateBrushSize(self, value):
        self.image_canvas.brush_size = value

    def clearBrush(self):
        self.image_canvas.clearOverlay()

    def fitCircleToPoints(self, points):
        if len(points) < MIN_POINTS_FOR_CIRCLE:
            return None, None, None

        points = np.array(points)
        
        def circle_error(params):
            xc, yc, r = params
            distances = np.sqrt((points[:,0] - xc)**2 + (points[:,1] - yc)**2)
            return np.sum((distances - r)**2)

        # Initial guess using bounding box
        x_m = np.mean(points[:,0])
        y_m = np.mean(points[:,1])
        r_guess = np.mean(np.sqrt((points[:,0] - x_m)**2 + (points[:,1] - y_m)**2))
        
        result = minimize(circle_error, [x_m, y_m, r_guess], method='Nelder-Mead')
        if result.success:
            return result.x[0], result.x[1], result.x[2]
        return None, None, None

    def autoDetect(self):
    if self.image is None:
        return

    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    height, width = gray.shape
    
    # Define relative circle sizes based on image dimensions
    min_radius = min(height, width) // 8
    max_radius = min(height, width) // 2
    
    circle_params = [
        {'radius_range': (max_radius//2, max_radius),
         'color': (255, 0, 0)},  # Large circle
        {'radius_range': (max_radius//3, max_radius//2),
         'color': (0, 255, 0)},  # Medium circle
        {'radius_range': (min_radius, max_radius//3),
         'color': (0, 0, 255)}   # Small circle
    ]



    def autoDetect(self):
        if self.image is None:
            return

        # Work with resized image for initial detection
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        height, width = gray.shape
        image_area = height * width

        # Parameters for three circles
        circle_params = [
            {'radius_range': (int(np.sqrt(0.3 * image_area / np.pi)), 
                            int(np.sqrt(0.6 * image_area / np.pi))),
             'color': (255, 0, 0)},  # Large circle
            {'radius_range': (int(np.sqrt(0.1 * image_area / np.pi)), 
                            int(np.sqrt(0.2 * image_area / np.pi))),
             'color': (0, 255, 0)},  # Medium circle
            {'radius_range': (int(np.sqrt(0.03 * image_area / np.pi)), 
                            int(np.sqrt(0.06 * image_area / np.pi))),
             'color': (0, 0, 255)}   # Small circle
        ]

        result = self.image.copy()
        circles_data = []
        previous_center = None

        # Process each circle size
        for i, params in enumerate(circle_params):
            edges = cv2.Canny(blurred, 50, 150)
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=width//4,  # Reduced from width//2 for better detection
                param1=50,
                param2=30,
                minRadius=params['radius_range'][0],
                maxRadius=params['radius_range'][1]
            )
            
            if circles is not None:
                # Get the strongest circle
                circle = circles[0][0]
                
                # Refine detection on original image
                scale_factor = self.original_image.shape[1] / self.image.shape[1]
                refined_x = int(circle[0] * scale_factor)
                refined_y = int(circle[1] * scale_factor)
                refined_r = int(circle[2] * scale_factor)
                
                # Get refined edge points using original image
                gray_orig = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                profile_points, intensities = self.getIntensityProfile(
                    gray_orig, 
                    (refined_x, refined_y), 
                    refined_r
                )
                
                if len(profile_points) > 10:
                    weights = np.abs(np.gradient(intensities))
                    (xc, yc, r), fit_error = self.fitCircle(profile_points, weights)
                    
                    # Scale back to display image coordinates
                    xc /= scale_factor
                    yc /= scale_factor
                    r /= scale_factor
                    
                    intensity_drop = self.calculateIntensityDrop(gray, (xc, yc), r)
                    
                    concentricity = 0.0
                    if previous_center is not None:
                        concentricity = np.sqrt((xc - previous_center[0])**2 + 
                                              (yc - previous_center[1])**2)
                    
                    circles_data.append((xc, yc, r, intensity_drop, concentricity))
                    previous_center = [xc, yc]
                    
                    # Draw detected circle
                    cv2.circle(result, (int(xc), int(yc)), int(r), params['color'], 2)
                    cv2.putText(result, f"{i+1}", (int(xc)-10, int(yc)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, params['color'], 2)

        self.processed_image = result
        self.updateDisplay()
        
        if circles_data:
            self.results_window.updateResults(circles_data)
            self.results_window.show()


    def detectInROI(self):
        if self.image is None or not self.image_canvas.points:
            return

        # Get drawn points
        points = np.array(self.image_canvas.points)
        
        # Fit circle to drawn points
        xc, yc, r = self.fitCircleToPoints(points)
        
        if xc is not None:
            result = self.image.copy()
            
            # Scale to original image for precise measurements
            scale_factor = self.original_image.shape[1] / self.image.shape[1]
            orig_xc = xc * scale_factor
            orig_yc = yc * scale_factor
            orig_r = r * scale_factor
            
            # Get intensity profile from original image
            gray_orig = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            profile_points, intensities = self.getIntensityProfile(
                gray_orig,
                (orig_xc, orig_yc),
                orig_r
            )
            
            # Calculate intensity drop
            intensity_drop = self.calculateIntensityDrop(
                cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY),
                (xc, yc),
                r
            )
            
            # Draw detected circle
            cv2.circle(result, (int(xc), int(yc)), int(r), (0, 255, 0), 2)
            cv2.circle(result, (int(xc), int(yc)), 2, (0, 255, 0), 3)
            
            circles_data = [(xc, yc, r, intensity_drop, 0.0)]
            
            self.processed_image = result
            self.updateDisplay()
            
            self.results_window.updateResults(circles_data)
            self.results_window.show()

    def getIntensityProfile(self, img, center, radius, num_points=360):
        x, y = center
        points = []
        intensities = []
        
        angles = np.linspace(0, 2*np.pi, num_points)
        for angle in angles:
            # Sample multiple points along each radius
            r_points = np.linspace(0.8*radius, 1.2*radius, 20)
            for r in r_points:
                px = x + r * np.cos(angle)
                py = y + r * np.sin(angle)
                
                if 0 <= px < img.shape[1]-1 and 0 <= py < img.shape[0]-1:
                    # Bilinear interpolation
                    x0, y0 = int(px), int(py)
                    x1, y1 = x0 + 1, y0 + 1
                    fx, fy = px - x0, py - y0
                    
                    intensity = (1-fx)*(1-fy)*img[y0, x0] + \
                               fx*(1-fy)*img[y0, x1] + \
                               (1-fx)*fy*img[y1, x0] + \
                               fx*fy*img[y1, x1]
                    
                    points.append([px, py])
                    intensities.append(intensity)
        
        return np.array(points), np.array(intensities)


    def calculateIntensityDrop(self, img, center, radius):
        x, y = int(center[0]), int(center[1])
        r = int(radius)
        
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            center_intensity = float(img[y, x])
            edge_intensities = []
            
            angles = np.linspace(0, 2*np.pi, 36)
            for angle in angles:
                edge_x = int(x + r * np.cos(angle))
                edge_y = int(y + r * np.sin(angle))
                if 0 <= edge_x < img.shape[1] and 0 <= edge_y < img.shape[0]:
                    edge_intensities.append(float(img[edge_y, edge_x]))
            
            if edge_intensities:
                avg_edge_intensity = np.mean(edge_intensities)
                intensity_drop = ((center_intensity - avg_edge_intensity) / 
                                center_intensity * 100)
                return intensity_drop
        
        return 0.0

    def fitCircle(self, points, weights):
        def circle_residuals(params, points, weights):
            xc, yc, r = params
            distances = np.sqrt((points[:,0] - xc)**2 + (points[:,1] - yc)**2) - r
            return distances * weights
        
        # Initial guess using weighted mean
        weights = np.array(weights) / np.sum(weights)
        center_guess = np.average(points, weights=weights, axis=0)
        radius_guess = np.average(np.sqrt(np.sum((points - center_guess)**2, axis=1)), 
                                weights=weights)
        
        params_guess = [center_guess[0], center_guess[1], radius_guess]
        result = least_squares(circle_residuals, params_guess, 
                             args=(points, weights), method='lm')
        
        return result.x, result.cost

    def updateImage(self):
        if self.image is None:
            return
            
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value() / 100.0
        
        self.processed_image = cv2.convertScaleAbs(self.image, 
                                                 alpha=contrast, 
                                                 beta=brightness)
        self.updateDisplay()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CircleDetector()
    ex.show()
    sys.exit(app.exec_())
