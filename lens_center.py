import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tkinterdnd2 import TkinterDnD, DND_FILES  # Import tkinterdnd2 for drag-and-drop

class CircleDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Circle Detector with Keyboard Shortcuts")

        # Create GUI elements
        self.label = tk.Label(root, text="Drag and drop or select an image to detect circles")
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select Image (Press 1)", command=self.load_image)
        self.select_button.pack(pady=5)

        self.manual_segment_button = tk.Button(root, text="Manual Segmentation (Press 2)", command=self.start_manual_segmentation, state=tk.DISABLED)
        self.manual_segment_button.pack(pady=5)

        self.analyze_button = tk.Button(root, text="Analyze Circles (Press 3)", command=self.analyze_circles, state=tk.DISABLED)
        self.analyze_button.pack(pady=5)

        self.result_text = tk.Text(root, height=15, width=60, state=tk.DISABLED)
        self.result_text.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # Enable drag-and-drop for the image label
        self.image_label.drop_target_register(DND_FILES)  # Register the label as a drop target
        self.image_label.dnd_bind('<<Drop>>', self.handle_drop)  # Bind the drop event

        # Bind keyboard shortcuts
        self.root.bind('1', lambda event: self.load_image())
        self.root.bind('2', lambda event: self.start_manual_segmentation())
        self.root.bind('3', lambda event: self.analyze_circles())

        self.image_path = None
        self.image = None  # Store the original image
        self.manual_circle = None  # Store the manually segmented circle

    def load_image(self):
        """Open a file dialog to select an image."""
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if self.image_path:
            self.display_image(self.image_path)

    def handle_drop(self, event):
        """Handle the drag-and-drop event."""
        self.image_path = event.data.strip("{}")  # Remove curly braces from the dropped file path
        self.display_image(self.image_path)

    def display_image(self, image_path):
        """Display the selected or dropped image in the GUI."""
        try:
            # Open the image and resize it for display
            self.image = Image.open(image_path)
            self.image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Enable the manual segmentation and analyze buttons
            self.manual_segment_button.config(state=tk.NORMAL)
            self.analyze_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the image: {e}")

    def start_manual_segmentation(self):
        """Start manual segmentation by allowing the user to click on the image."""
        if not self.image_path:
            messagebox.showerror("Error", "No image selected!")
            return

        # Convert the image to a format that OpenCV can process
        self.image_cv = cv2.imread(self.image_path)

        # Create a window for manual segmentation
        cv2.namedWindow("Manual Segmentation")
        cv2.setMouseCallback("Manual Segmentation", self.draw_circle)

        # Display the image and wait for user input
        while True:
            cv2.imshow("Manual Segmentation", self.image_cv)
            key = cv2.waitKey(1) & 0xFF

            # Close the segmentation window with 'Esc'
            if key == 27:  # 'Esc' key
                self.manual_circle = None  # Reset the manual circle
                cv2.destroyAllWindows()
                break

            # Delete the manually drawn circle with 'D' or 'Backspace'
            if key == ord('d') or key == 8:  # 'D' or 'Backspace' key
                self.manual_circle = None  # Reset the manual circle
                self.image_cv = cv2.imread(self.image_path)  # Reload the original image

    def draw_circle(self, event, x, y, flags, param):
        """Handle mouse events for manual segmentation."""
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
            if self.manual_circle is None:
                self.manual_circle = (x, y, 0)  # Start drawing the circle
            else:
                # Calculate the radius based on the distance between the two clicks
                radius = int(np.sqrt((x - self.manual_circle[0]) ** 2 + (y - self.manual_circle[1]) ** 2))
                self.manual_circle = (self.manual_circle[0], self.manual_circle[1], radius)

                # Draw the circle on the image
                cv2.circle(self.image_cv, (self.manual_circle[0], self.manual_circle[1]), self.manual_circle[2], (0, 255, 0), 2)
                cv2.circle(self.image_cv, (self.manual_circle[0], self.manual_circle[1]), 2, (0, 0, 255), 3)  # Center point

                # Save the manually segmented circle
                self.manual_circle = (self.manual_circle[0], self.manual_circle[1], self.manual_circle[2])

    def analyze_circles(self):
        """Analyze the image to detect circles and calculate their properties."""
        if not self.image_path:
            messagebox.showerror("Error", "No image selected!")
            return

        try:
            # Load the image
            image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if image is None:
                messagebox.showerror("Error", "Could not load the image!")
                return

            # If a manual circle is defined, use it; otherwise, detect circles automatically
            if self.manual_circle:
                circles = np.array([[self.manual_circle]])
            else:
                # Convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply GaussianBlur to reduce noise and improve circle detection
                blurred = cv2.GaussianBlur(gray, (9, 9), 2)

                # Use Hough Circle Transform to detect circles
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                           param1=100, param2=30, minRadius=10, maxRadius=0)

                if circles is None:
                    messagebox.showinfo("Result", "No circles detected.")
                    return

                # Convert the (x, y, r) coordinates to integers
                circles = np.uint16(np.around(circles))

                # Sort circles by radius in descending order to find the 3 major circles
                circles = sorted(circles[0], key=lambda x: x[2], reverse=True)[:3]

            # Analyze the position of the other circles relative to the largest circle
            results = []
            for i, circle in enumerate(circles):
                x, y, radius = circle
                ellipticity_data = self.check_ellipticity(gray, x, y, radius)

                # Calculate area and perimeter
                area = round(np.pi * radius**2, 2)
                perimeter = round(2 * np.pi * radius, 2)

                results.append({
                    "Circle": f"Circle {i+1}",
                    "Center Position": (x, y),
                    "Radius": radius,
                    "Area": area,
                    "Perimeter": perimeter,
                    "Major Axis": ellipticity_data['major_axis'],
                    "Minor Axis": ellipticity_data['minor_axis'],
                })

            # Draw the circles on the image
            for circle in circles:
                x, y, radius = circle
                cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(image, (x, y), 2, (0, 0, 255), 3)  # Center point

            # Save the result image
            result_image_path = "detected_circles.jpg"
            cv2.imwrite(result_image_path, image)

            # Display the result image in the GUI
            result_image = Image.open(result_image_path)
            result_image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(result_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Display the results in the text box
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            for result in results:
                self.result_text.insert(tk.END, f"{result['Circle']}:\n")
                self.result_text.insert(tk.END, f"  Center Position = {result['Center Position']}\n")
                self.result_text.insert(tk.END, f"  Radius = {result['Radius']}\n")
                self.result_text.insert(tk.END, f"  Area = {result['Area']} pixels²\n")
                self.result_text.insert(tk.END, f"  Perimeter = {result['Perimeter']} pixels\n")
                self.result_text.insert(tk.END, f"  Major Axis = {result['Major Axis']}\n")
                self.result_text.insert(tk.END, f"  Minor Axis = {result['Minor Axis']}\n\n")
            self.result_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {e}")

    def check_ellipticity(self, gray_image, x, y, radius):
        """Check the ellipticity of a circle by fitting an ellipse to the region."""
        mask = np.zeros_like(gray_image, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        roi = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return {'major_axis': 'Not Detected', 'minor_axis': 'Not Detected'}

        try:
            ellipse = cv2.fitEllipse(contours[0])
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            return {'major_axis': round(major_axis, 2), 'minor_axis': round(minor_axis, 2)}
        except:
            return {'major_axis': 'Calculation Error', 'minor_axis': 'Calculation Error'}

# Run the application
if __name__ == "__main__":
    root = TkinterDnD.Tk()  # Use TkinterDnD for drag-and-drop support
    app = CircleDetectorApp(root)
    root.mainloop()