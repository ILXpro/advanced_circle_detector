import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT, FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

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

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = self.mapToScene(event.pos())

    def mouseMoveEvent(self, event):
        if self.drawing and self.last_point and self.overlay_item:
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
            self.scene.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_point = None

    def wheelEvent(self, event):
        factor = 1.1
        if event.angleDelta().y() < 0:
            factor = 0.9
        self.scale(factor, factor)
        self.scale_factor *= factor

class CircleDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Advanced Circle Detector'
        self.image = None
        self.processed_image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        viewer_widget = QWidget()
        viewer_layout = QVBoxLayout(viewer_widget)
        
        fig = Figure(figsize=(1, 1))
        canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(canvas, self)
        viewer_layout.addWidget(self.toolbar)
        
        self.image_canvas = ImageCanvas()
        viewer_layout.addWidget(self.image_canvas)
        
        layout.addWidget(viewer_widget, stretch=2)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
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

        adjust_group = QGroupBox("Image Adjustments")
        adjust_layout = QVBoxLayout()
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.updateImage)
        adjust_layout.addWidget(QLabel("Brightness:"))
        adjust_layout.addWidget(self.brightness_slider)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(1, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.updateImage)
        adjust_layout.addWidget(QLabel("Contrast:"))
        adjust_layout.addWidget(self.contrast_slider)

        adjust_group.setLayout(adjust_layout)
        control_layout.addWidget(adjust_group)

        control_layout.addStretch()
        layout.addWidget(control_panel)

    def clearBrush(self):
        self.image_canvas.clearOverlay()

    def openImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            self.image = cv2.imread(fileName)
            max_dim = 1000
            height, width = self.image.shape[:2]
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                self.image = cv2.resize(self.image, None, fx=scale, fy=scale)
            self.processed_image = self.image.copy()
            self.updateDisplay()

    def updateDisplay(self):
        if self.processed_image is None:
            return
            
        height, width = self.processed_image.shape[:2]
        bytes_per_line = 3 * width
        
        image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_canvas.setImage(QPixmap.fromImage(q_img))

    def updateBrushSize(self):
        self.image_canvas.brush_size = self.brush_size_slider.value()

    def findBestCircle(self, img, center, radius, search_range=10):
        height, width = img.shape[:2]
        y, x = center
        best_score = 0
        best_circle = (x, y, radius)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                new_x = x + dx
                new_y = y + dy
                
                if (new_x - radius < 0 or new_x + radius >= width or 
                    new_y - radius < 0 or new_y + radius >= height):
                    continue
                
                mask = np.zeros_like(gray)
                cv2.circle(mask, (new_x, new_y), radius, 255, 1)
                
                edge_score = np.sum(edges * mask) / 255
                gradient_score = np.sum(gradient_magnitude * mask) / np.sum(mask)
                total_score = edge_score * gradient_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_circle = (new_x, new_y, radius)
        
        return best_circle, best_score

    def autoDetect(self):
        if self.image is None:
            return

        # Initial detection on reduced image
        max_dim = 500
        height, width = self.image.shape[:2]
        scale = max_dim / max(height, width)
        if scale < 1:
            small_img = cv2.resize(self.image, None, fx=scale, fy=scale)
        else:
            small_img = self.image.copy()
            scale = 1

        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=int(min(gray.shape) / 2)
        )

        result = self.image.copy()
        if circles is not None:
            # Scale back to original image size
            circles = np.uint16(np.around(circles[0] / scale))
            
            # Find common center
            centers = np.array([[x, y] for x, y, _ in circles])
            if len(centers) > 1:
                kmeans = KMeans(n_clusters=1, random_state=42)
                kmeans.fit(centers)
                common_center = kmeans.cluster_centers_[0].astype(int)
                
                # Sort circles by radius
                circles = sorted(circles, key=lambda x: x[2], reverse=True)
                
                refined_circles = []
                for x, y, r in circles[:3]:  # Process top 3 circles
                    refined_circle, score = self.findBestCircle(
                        self.image,
                        (common_center[1], common_center[0]),
                        r
                    )
                    if score > 0:
                        refined_circles.append(refined_circle)
                
                # Draw refined circles
                for x, y, r in refined_circles:
                    cv2.circle(result, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(result, (x, y), 2, (0, 255, 0), 3)
                    print(f"Circle found at ({x}, {y}) with radius {r} pixels")

        self.processed_image = result
        self.updateDisplay()

    def detectInROI(self):
        if self.image is None or self.image_canvas.overlay_item is None:
            return

        overlay_pixmap = self.image_canvas.overlay_pixmap
        mask = overlay_pixmap.toImage()
        buffer = mask.bits().asstring(mask.byteCount())
        mask = np.frombuffer(buffer, dtype=np.uint8).reshape(
            (mask.height(), mask.width(), 4))[:,:,3]
        
        mask = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]))
        mask = (mask > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return

        x, y, w, h = cv2.boundingRect(contours[0])
        roi = self.image[max(0, y-20):min(self.image.shape[0], y+h+20),
                        max(0, x-20):min(self.image.shape[1], x+w+20)]
        roi_mask = mask[max(0, y-20):min(self.image.shape[0], y+h+20),
                       max(0, x-20):min(self.image.shape[1], x+w+20)]

        if roi.size == 0:
            return

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        
        mean = np.mean(blurred_roi)
        edges_roi = cv2.Canny(blurred_roi, mean * 0.66, mean * 1.33)
        edges_roi[roi_mask == 0] = 0

        circles = cv2.HoughCircles(
            edges_roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=min(roi.shape) // 2
        )

        result = self.image.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            for circle in circles:
                x_orig = circle[0] + max(0, x-20)
                y_orig = circle[1] + max(0, y-20)
                
                refined_circle, score = self.findBestCircle(
                    self.image,
                    (y_orig, x_orig),
                    circle[2]
                )
                
                x_ref, y_ref, r_ref = refined_circle
                cv2.circle(result, (x_ref, y_ref), r_ref, (0, 255, 0), 2)
                cv2.circle(result, (x_ref, y_ref), 2, (0, 255, 0), 3)
                print(f"Circle found at ({x_ref}, {y_ref}) with radius {r_ref} pixels")

        self.processed_image = result
        self.updateDisplay()

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
