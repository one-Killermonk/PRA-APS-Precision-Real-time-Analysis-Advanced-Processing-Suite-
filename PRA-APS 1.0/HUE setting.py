# hsv_tuner.py
import sys
import cv2
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui

class HSVColorTuner(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSV Color Range Tuner for Welding")
        self.resize(1000, 700)
        
        self.image_path = None
        self.original_image = None
        
        # Default HSV ranges for welding (hot metal)
        self.h_min, self.h_max = 0, 30
        self.s_min, self.s_max = 50, 255
        self.v_min, self.v_max = 50, 255
        
        # Store slider references
        self.sliders = {}
        self.labels = {}
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Left panel - controls
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        # Load image button
        btn_load = QtWidgets.QPushButton("Load Image")
        btn_load.clicked.connect(self.load_image)
        left_layout.addWidget(btn_load)
        
        # HSV controls
        left_layout.addWidget(QtWidgets.QLabel("HSV Range Controls:"))
        
        # Create sliders for each HSV component
        self.create_hsv_control(left_layout, "H Min", 0, 180, self.h_min)
        self.create_hsv_control(left_layout, "H Max", 0, 180, self.h_max)
        self.create_hsv_control(left_layout, "S Min", 0, 255, self.s_min)
        self.create_hsv_control(left_layout, "S Max", 0, 255, self.s_max)
        self.create_hsv_control(left_layout, "V Min", 0, 255, self.v_min)
        self.create_hsv_control(left_layout, "V Max", 0, 255, self.v_max)
        
        # Current values display
        self.values_label = QtWidgets.QLabel(self.get_values_text())
        self.values_label.setStyleSheet("background: #000000; padding: 10px; border-radius: 5px; font-family: monospace;")
        self.values_label.setWordWrap(True)
        left_layout.addWidget(self.values_label)
        
        # Recommended presets
        left_layout.addWidget(QtWidgets.QLabel("Presets:"))
        presets_layout = QtWidgets.QGridLayout()
        
        btn_hot_metal = QtWidgets.QPushButton("Hot Metal")
        btn_hot_metal.clicked.connect(self.set_hot_metal_preset)
        presets_layout.addWidget(btn_hot_metal, 0, 0)
        
        btn_cool_metal = QtWidgets.QPushButton("Cool Metal")
        btn_cool_metal.clicked.connect(self.set_cool_metal_preset)
        presets_layout.addWidget(btn_cool_metal, 0, 1)
        
        btn_arc_glare = QtWidgets.QPushButton("Arc/Glare")
        btn_arc_glare.clicked.connect(self.set_arc_glare_preset)
        presets_layout.addWidget(btn_arc_glare, 1, 0)
        
        btn_reset = QtWidgets.QPushButton("Reset")
        btn_reset.clicked.connect(self.set_reset_preset)
        presets_layout.addWidget(btn_reset, 1, 1)
        
        left_layout.addLayout(presets_layout)
        
        # Copy to clipboard button
        btn_copy = QtWidgets.QPushButton("Copy Values to Clipboard")
        btn_copy.clicked.connect(self.copy_to_clipboard)
        left_layout.addWidget(btn_copy)
        
        left_layout.addStretch()
        
        # Right panel - image display
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #e0e0e0; border: 1px solid #ccc; min-height: 400px;")
        self.image_label.setText("Load an image to begin tuning HSV values")
        self.image_label.setMinimumSize(600, 400)
        
        right_layout.addWidget(self.image_label)
        
        layout.addWidget(left_panel)
        layout.addWidget(right_panel, 1)
        
    def create_hsv_control(self, layout, name, min_val, max_val, init_val):
        """Create a slider with label for HSV control"""
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        label = QtWidgets.QLabel(f"{name}:")
        label.setFixedWidth(50)
        
        # Value label
        value_label = QtWidgets.QLabel(str(init_val))
        value_label.setFixedWidth(40)
        value_label.setAlignment(QtCore.Qt.AlignRight)
        
        # Slider
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        slider.valueChanged.connect(lambda value, lbl=value_label, n=name: self.on_slider_change(value, lbl, n))
        
        container_layout.addWidget(label)
        container_layout.addWidget(slider)
        container_layout.addWidget(value_label)
        
        layout.addWidget(container)
        
        # Store references
        self.sliders[name] = slider
        self.labels[name] = value_label
    
    def on_slider_change(self, value, label, slider_name):
        """Handle slider value changes"""
        label.setText(str(value))
        
        # Update the appropriate HSV value
        if slider_name == "H Min":
            self.h_min = value
        elif slider_name == "H Max":
            self.h_max = value
        elif slider_name == "S Min":
            self.s_min = value
        elif slider_name == "S Max":
            self.s_max = value
        elif slider_name == "V Min":
            self.v_min = value
        elif slider_name == "V Max":
            self.v_max = value
            
        self.values_label.setText(self.get_values_text())
        self.update_image_display()
    
    def get_values_text(self):
        return (f"Current HSV Range:\n"
                f"H: {self.h_min} - {self.h_max}\n"
                f"S: {self.s_min} - {self.s_max}\n"
                f"V: {self.v_min} - {self.v_max}\n\n"
                f"Code format:\n"
                f"lower = np.array([{self.h_min}, {self.s_min}, {self.v_min}])\n"
                f"upper = np.array([{self.h_max}, {self.s_max}, {self.v_max}])")
    
    def set_hot_metal_preset(self):
        self.set_preset_values(0, 30, 50, 255, 50, 255)
    
    def set_cool_metal_preset(self):
        self.set_preset_values(0, 180, 0, 50, 0, 100)
    
    def set_arc_glare_preset(self):
        self.set_preset_values(0, 180, 0, 50, 200, 255)
    
    def set_reset_preset(self):
        self.set_preset_values(0, 180, 0, 255, 0, 255)
    
    def set_preset_values(self, h_min, h_max, s_min, s_max, v_min, v_max):
        """Set all slider values for a preset"""
        self.sliders["H Min"].setValue(h_min)
        self.sliders["H Max"].setValue(h_max)
        self.sliders["S Min"].setValue(s_min)
        self.sliders["S Max"].setValue(s_max)
        self.sliders["V Min"].setValue(v_min)
        self.sliders["V Max"].setValue(v_max)
    
    def copy_to_clipboard(self):
        clipboard = QtWidgets.QApplication.clipboard()
        code = (f"# HSV Range for welding detection\n"
                f"h_min = {self.h_min}\n"
                f"h_max = {self.h_max}\n"
                f"s_min = {self.s_min}\n"
                f"s_max = {self.s_max}\n"
                f"v_min = {self.v_min}\n"
                f"v_max = {self.v_max}\n\n"
                f"lower = np.array([h_min, s_min, v_min])\n"
                f"upper = np.array([h_max, s_max, v_max])")
        clipboard.setText(code)
        QtWidgets.QMessageBox.information(self, "Copied", "HSV values copied to clipboard!")
    
    def load_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.update_image_display()
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Could not load image!")
    
    def update_image_display(self):
        if self.original_image is None:
            return
            
        # Convert to HSV and apply mask
        hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        
        # Create mask based on HSV range
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        mask = cv2.inRange(hsv_image, lower, upper)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Create result image with mask overlay
        result = self.original_image.copy()
        
        # Create colored overlay (green for detected areas)
        overlay = result.copy()
        overlay[mask > 0] = [0, 255, 0]  # Green for detected areas
        
        # Blend original with overlay
        alpha = 0.5
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
        
        # Add text with percentage of detected area
        total_pixels = mask.size
        detected_pixels = cv2.countNonZero(mask)
        detection_percentage = (detected_pixels / total_pixels) * 100
        
        cv2.putText(result, f"Detection: {detection_percentage:.1f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f"H: {self.h_min}-{self.h_max}, S: {self.s_min}-{self.s_max}, V: {self.v_min}-{self.v_max}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert to QImage for display
        height, width, channel = result.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(result.data, width, height, bytes_per_line, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(q_img)
        
        # Scale if necessary
        max_width = 800
        max_height = 600
        if width > max_width or height > max_height:
            pixmap = pixmap.scaled(max_width, max_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        
        self.image_label.setPixmap(pixmap)

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Apply some styling
    app.setStyle("Fusion")
    
    window = HSVColorTuner()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()