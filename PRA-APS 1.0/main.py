# main.py
import sys, os, uuid, json, traceback
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from manager import Manager

# ---------- helper overlay blending ----------
def blend_overlay(base_bgr, mask, color_bgr=(0, 255, 0), alpha=0.45):
    """
    Blend color over base_bgr where mask==255. mask is HxW uint8 (0/255).
    """
    if mask is None:
        return base_bgr
    # ensure mask has same HxW
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]
    color_arr = np.full_like(base_bgr, color_bgr, dtype=np.uint8).astype(np.float32)
    base_f = base_bgr.astype(np.float32)
    out = base_f * (1.0 - alpha * mask_f) + color_arr * (alpha * mask_f)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# ---------- VideoWidget with drawing ----------
class VideoWidget(QtWidgets.QLabel):
    shapeAdded = QtCore.Signal(dict)   # emit shape dict when added

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setScaledContents(True)
        self._pix = None
        self.current_frame = None  # BGR numpy
        # drawing state
        self.shapes = []   # list of dicts
        self.draw_mode = None  # 'box' or 'circle' when user clicks buttons
        self.drawing = False
        self.current = None  # temporary shape during drawing: {type, rect}
        self.moving = False
        self.move_index = None
        self.move_offset = (0,0)
        # overlay info set externally
        self.overlay_colors = {}   # id -> 'g' or 'r'
        self.overlay_percent = {}  # id -> percent

    # display a full BGR frame
    def setFrame(self, frame_bgr):
        if frame_bgr is None:
            return
        self.current_frame = frame_bgr.copy()
        # convert to RGB QImage
        try:
            rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self._pix = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(self._pix)
        self.update()

    # map widget coords to original image coords
    def widget_to_image(self, wx, wy):
        if self._pix is None or self.current_frame is None:
            return 0,0
        pw, ph = self.width(), self.height()
        iw, ih = self._pix.width(), self._pix.height()
        # compute scale and offsets for aspect fit
        scale_x = iw / pw
        scale_y = ih / ph
        # since setScaledContents=True, QLabel scales pixmap to label size avoiding margins => use direct ratio
        img_x = int(wx * scale_x)
        img_y = int(wy * scale_y)
        img_x = max(0, min(iw-1, img_x))
        img_y = max(0, min(ih-1, img_y))
        return img_x, img_y

    # map image coords to widget coords (for painting)
    def image_to_widget(self, ix, iy):
        if self._pix is None:
            return ix, iy
        pw, ph = self.width(), self.height()
        iw, ih = self._pix.width(), self._pix.height()
        wx = int(ix * (pw / iw))
        wy = int(iy * (ph / ih))
        return wx, wy

    # mouse events for drawing / moving shapes
    def mousePressEvent(self, ev):
        if self._pix is None or self.current_frame is None:
            return
        wx, wy = ev.x(), ev.y()
        ix, iy = self.widget_to_image(wx, wy)
        if self.draw_mode in ("box", "circle"):
            self.drawing = True
            sid = str(uuid.uuid4())
            if self.draw_mode == "box":
                self.current = {"id": sid, "type": "box", "x": ix, "y": iy, "width": 1, "height": 1}
            else:
                self.current = {"id": sid, "type": "circle", "x": ix, "y": iy, "radius": 1}
            self.update()
            return
        # else check for move selection
        # iterate shapes top-down
        for idx in range(len(self.shapes)-1, -1, -1):
            s = self.shapes[idx]
            if s["type"] == "box":
                x,y,w,h = s["x"], s["y"], s["width"], s["height"]
                if x <= ix <= x+w and y <= iy <= y+h:
                    self.moving = True
                    self.move_index = idx
                    self.move_offset = (ix - x, iy - y)
                    return
            else:
                cx,cy,r = s["x"], s["y"], s["radius"]
                if (ix-cx)**2 + (iy-cy)**2 <= r*r:
                    self.moving = True
                    self.move_index = idx
                    self.move_offset = (ix - s["x"], iy - s["y"])
                    return

    def mouseMoveEvent(self, ev):
        if self._pix is None or self.current_frame is None:
            return
        wx, wy = ev.x(), ev.y()
        ix, iy = self.widget_to_image(wx, wy)
        if self.drawing and self.current:
            if self.current["type"] == "box":
                x0, y0 = self.current["x"], self.current["y"]
                self.current["width"] = max(1, ix - x0)
                self.current["height"] = max(1, iy - y0)
            else:
                cx, cy = self.current["x"], self.current["y"]
                self.current["radius"] = int(((ix-cx)**2 + (iy-cy)**2)**0.5)
            self.update()
            return
        if self.moving and self.move_index is not None:
            idx = self.move_index
            offx, offy = self.move_offset
            s = self.shapes[idx]
            s["x"] = int(ix - offx)
            s["y"] = int(iy - offy)
            self.update()
            return

    def mouseReleaseEvent(self, ev):
        if self.drawing and self.current:
            # finalize shape (normalize negative width/height)
            if self.current["type"] == "box":
                x = int(self.current["x"]); y = int(self.current["y"])
                w = int(self.current["width"]); h = int(self.current["height"])
                if w < 0:
                    x = x + w; w = abs(w)
                if h < 0:
                    y = y + h; h = abs(h)
                self.current["x"], self.current["y"], self.current["width"], self.current["height"] = x, y, w, h
            self.shapes.append(self.current)
            self.shapeAdded.emit(self.current)
            self.current = None
            self.drawing = False
            self.update()
            return
        if self.moving:
            self.moving = False
            self.move_index = None
            self.update()
            return

    # painting overlays
    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._pix is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # draw shapes
        for s in self.shapes:
            sid = s.get("id")
            color = QtGui.QColor(0,200,255) if self.overlay_colors.get(sid,'g')=='g' else QtGui.QColor(220,30,30)
            pen = QtGui.QPen(color, 3)
            painter.setPen(pen)
            if s["type"] == "box":
                x,y,w,h = s["x"], s["y"], s["width"], s["height"]
                rx, ry = self.image_to_widget(x,y)
                rw = int(w * (self.width()/self._pix.width()))
                rh = int(h * (self.height()/self._pix.height()))
                painter.drawRect(rx, ry, rw, rh)
                pct = self.overlay_percent.get(sid)
                if pct is not None:
                    painter.setPen(QtGui.QPen(QtGui.QColor(255,255,255),1))
                    painter.drawText(rx+6, ry+16, f"{pct:.1f}%")
            else:
                cx,cy,r = s["x"], s["y"], s["radius"]
                rcx, rcy = self.image_to_widget(cx, cy)
                rr = int(r * (self.width()/self._pix.width()))
                painter.drawEllipse(QtCore.QPointF(rcx, rcy), rr, rr)
                pct = self.overlay_percent.get(sid)
                if pct is not None:
                    painter.setPen(QtGui.QPen(QtGui.QColor(255,255,255),1))
                    painter.drawText(rcx-rr, rcy-rr-6, f"{pct:.1f}%")
        # draw current (during drawing)
        if self.current:
            pen = QtGui.QPen(QtGui.QColor(0,180,220), 2, QtCore.Qt.DashLine)
            painter.setPen(pen)
            if self.current["type"] == "box":
                x,y,w,h = self.current["x"], self.current["y"], self.current["width"], self.current["height"]
                rx, ry = self.image_to_widget(x,y)
                rw = int(w * (self.width()/self._pix.width()))
                rh = int(h * (self.height()/self._pix.height()))
                painter.drawRect(rx, ry, rw, rh)
            else:
                cx,cy,r = self.current["x"], self.current["y"], self.current["radius"]
                rcx, rcy = self.image_to_widget(cx, cy)
                rr = int(r * (self.width()/self._pix.width()))
                painter.drawEllipse(QtCore.QPointF(rcx, rcy), rr, rr)
        painter.end()

# ---------- MainWindow ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PRA-APS 1.0 - HSV Color Detection")
        self.resize(1200, 800)
        # theme (bright + blue outlines)
        self.setStyleSheet("""
            QWidget { background: #f6fbff; color: #07203a; }
            QPushButton { background: #e6f3ff; border: 1px solid #4aa3ff; padding: 6px; border-radius:6px; }
            QPushButton:hover { background: #d0eaff; }
            QTabWidget::pane { border: 1px solid #cfeeff; }
            QLabel { color: #07203a; }
            QSpinBox { background: white; }
        """)
        self.manager = Manager()
        self.cap = None
        self.cam_index = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.grab_frame)
        self.timer.start(30)
        self.init_ui()
        self.open_camera(self.cam_index)

    def init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)

        # left: tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tab_shapes = QtWidgets.QWidget()
        self.tab_inspect = QtWidgets.QWidget()
        self.tab_settings = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_shapes, "Shapes")
        self.tabs.addTab(self.tab_inspect, "Inspection")
        self.tabs.addTab(self.tab_settings, "Settings")
        self.tabs.currentChanged.connect(self.on_tab_change)
        h.addWidget(self.tabs, 1)

        # right: video
        right_v = QtWidgets.QVBoxLayout()
        self.video_widget = VideoWidget()
        right_v.addWidget(self.video_widget, 8)
        self.result_label = QtWidgets.QLabel("Result: -")
        self.result_label.setFixedHeight(40)
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        right_v.addWidget(self.result_label)
        h.addLayout(right_v, 1)

        self.build_shapes_tab()
        self.build_inspect_tab()
        self.build_settings_tab()

        # load shapes into widget
        shapes = self.manager.load_shapes()
        if shapes:
            self.video_widget.shapes = shapes

    # ---------- camera ----------
    def open_camera(self, idx):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        # use CAP_DSHOW on Windows to reduce latency; okay if platform ignores it
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        # request HD
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # warm up
        for _ in range(2):
            try:
                self.cap.read()
            except Exception:
                pass

    def grab_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        # display live frame. If inspection tab active and user clicked Inspect, show overlay; else show raw/live
        self.current_frame = frame.copy()
        if self.tabs.currentWidget() is self.tab_inspect and getattr(self, "inspecting", False):
            # draw overlays during inspection (live)
            self.do_inspect_and_display(self.current_frame)
        else:
            self.video_widget.setFrame(self.current_frame)

    # ---------- Shapes tab ----------
    def build_shapes_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_shapes)
        hbtn = QtWidgets.QHBoxLayout()
        btn_box = QtWidgets.QPushButton("Box")
        btn_circle = QtWidgets.QPushButton("Circle")
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_save = QtWidgets.QPushButton("Save")
        hbtn.addWidget(btn_box); hbtn.addWidget(btn_circle); hbtn.addWidget(btn_clear); hbtn.addWidget(btn_save)
        layout.addLayout(hbtn)
        layout.addWidget(QtWidgets.QLabel("Click Box/Circle and draw directly on live video. Then Save to persist shapes."))

        btn_box.clicked.connect(lambda: self.set_draw_mode("box"))
        btn_circle.clicked.connect(lambda: self.set_draw_mode("circle"))
        btn_clear.clicked.connect(self.clear_shapes)
        btn_save.clicked.connect(self.save_shapes)

    def set_draw_mode(self, mode):
        self.video_widget.draw_mode = mode

    def clear_shapes(self):
        self.video_widget.shapes = []
        self.manager.save_shapes([])
        self.video_widget.update()

    def save_shapes(self):
        # normalize shapes and save through manager
        shapes = []
        for s in self.video_widget.shapes:
            if s["type"] == "box":
                shapes.append({
                    "id": s["id"],
                    "type": "box",
                    "x": int(s["x"]),
                    "y": int(s["y"]),
                    "width": int(s["width"]),
                    "height": int(s["height"])
                })
            else:
                shapes.append({
                    "id": s["id"],
                    "type": "circle",
                    "x": int(s["x"]),
                    "y": int(s["y"]),
                    "radius": int(s["radius"])
                })
        self.manager.save_shapes(shapes)
        QtWidgets.QMessageBox.information(self, "Saved", "Shapes saved to shapes.json")
        self.video_widget.shapes = shapes

    # ---------- Inspection tab ----------
    def build_inspect_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_inspect)
        h = QtWidgets.QHBoxLayout()
        self.btn_inspect = QtWidgets.QPushButton("Toggle Inspect (Start/Stop)")
        self.btn_run = QtWidgets.QPushButton("Inspect Once")
        h.addWidget(self.btn_inspect); h.addWidget(self.btn_run)
        layout.addLayout(h)
        layout.addWidget(QtWidgets.QLabel("When inspecting, pixels inside shapes are tested vs HSV range from Settings."))
        self.btn_inspect.clicked.connect(self.toggle_inspect)
        self.btn_run.clicked.connect(self.inspect_once)
        self.inspecting = False

    def toggle_inspect(self):
        self.inspecting = not self.inspecting
        if self.inspecting:
            self.btn_inspect.setText("Stop Inspect")
        else:
            self.btn_inspect.setText("Start Inspect")
            # show raw frame again
            if hasattr(self, "current_frame") and self.current_frame is not None:
                self.video_widget.setFrame(self.current_frame)

    def inspect_once(self):
        if not hasattr(self, "current_frame") or self.current_frame is None:
            return
        self.do_inspect_and_display(self.current_frame, record_history=True)

    def do_inspect_and_display(self, frame, record_history=False):
        shapes = self.manager.load_shapes()
        settings = self.manager.load_settings()
        threshold = settings.get("threshold_pct", 90)
        alpha = settings.get("overlay_alpha", 0.45)
        overlay_frame = frame.copy()
        per_shape = {}
        overall_ok = True

        for s in shapes:
            sid = s.get("id")
            if s.get("type") == "box":
                x = max(0, int(s["x"])); y = max(0, int(s["y"]))
                w_ = max(1, int(s["width"])); h_ = max(1, int(s["height"]))
                x2 = min(frame.shape[1], x + w_); y2 = min(frame.shape[0], y + h_)
                crop = frame[y:y2, x:x2].copy()
                # CHANGED: Use HSV method instead of RGB
                mask, pct = self.manager.compute_mask_and_percent_hsv(crop, settings=settings)
                if mask is None:
                    per_shape[sid] = {"percent": 0.0, "ok": False}
                    overall_ok = False
                    continue
                # create green and red blends and composite
                green = blend_overlay(crop, mask, (0,200,0), alpha=alpha)
                inv_mask = (mask==0).astype(np.uint8)*255
                red = blend_overlay(crop, inv_mask, (0,0,200), alpha=alpha*0.8)
                combined = np.where(mask[:,:,None]==255, green, red)
                overlay_frame[y:y2, x:x2] = combined
                ok = pct >= float(threshold)
                color = (0,200,0) if ok else (0,0,200)
                cv2.rectangle(overlay_frame, (x,y), (x2,y2), color, 3)
                cv2.putText(overlay_frame, f"{pct:.1f}%", (x+6, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                per_shape[sid] = {"percent": pct, "ok": ok}
                if not ok:
                    overall_ok = False
            else:
                cx = int(s.get("x")); cy = int(s.get("y")); r = int(s.get("radius"))
                x0 = max(0, cx-r); y0 = max(0, cy-r); x1 = min(frame.shape[1], cx+r); y1 = min(frame.shape[0], cy+r)
                crop = frame[y0:y1, x0:x1].copy()
                if crop is None or crop.size == 0:
                    per_shape[sid] = {"percent": 0.0, "ok": False}
                    overall_ok = False
                    continue
                # CHANGED: Use HSV method instead of RGB
                mask, pct = self.manager.compute_mask_and_percent_hsv(crop, settings=settings)
                # enforce circle mask area
                h_c, w_c = crop.shape[:2]
                Y, X = np.ogrid[:h_c, :w_c]
                cx_rel = w_c//2; cy_rel = h_c//2
                circle_mask = ((X - cx_rel)**2 + (Y - cy_rel)**2) <= (r**2)
                if mask is None:
                    per_shape[sid] = {"percent": 0.0, "ok": False}
                    overall_ok = False
                    continue
                mask = (mask > 0).astype(np.uint8)*255
                mask[~circle_mask] = 0
                green = blend_overlay(crop, mask, (0,200,0), alpha=alpha)
                inv_mask = (mask==0).astype(np.uint8)*255
                red = blend_overlay(crop, inv_mask, (0,0,200), alpha=alpha*0.8)
                combined = np.where(mask[:,:,None]==255, green, red)
                overlay_frame[y0:y1, x0:x1] = combined
                ok = pct >= float(threshold)
                color = (0,200,0) if ok else (0,0,200)
                cv2.circle(overlay_frame, (cx,cy), r, color, 3)
                cv2.putText(overlay_frame, f"{pct:.1f}%", (cx-r, cy-r-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                per_shape[sid] = {"percent": pct, "ok": ok}
                if not ok:
                    overall_ok = False

        # update widget overlays info & display
        shapes_for_widget = shapes if shapes else []
        colors = {}
        percents = {}
        for s in shapes_for_widget:
            sid = s.get("id")
            colors[sid] = 'g' if per_shape.get(sid, {}).get("ok", False) else 'r'
            percents[sid] = per_shape.get(sid, {}).get("percent", 0.0)
        self.video_widget.shapes = shapes_for_widget
        self.video_widget.overlay_colors = colors
        self.video_widget.overlay_percent = percents
        self.video_widget.setFrame(overlay_frame)
        self.result_label.setText("Result: OK" if overall_ok else "Result: NG")

        if record_history:
            self.manager.record_history(per_shape, overall_ok)

    # ---------- Settings tab (UPDATED FOR HSV) ----------
    def build_settings_tab(self):
        layout = QtWidgets.QFormLayout(self.tab_settings)
        s = self.manager.load_settings()
        
        # HSV Min values
        self.spin_min_h = QtWidgets.QSpinBox(); self.spin_min_h.setRange(0,180); self.spin_min_h.setValue(int(s.get("h_min",0)))
        self.spin_min_s = QtWidgets.QSpinBox(); self.spin_min_s.setRange(0,255); self.spin_min_s.setValue(int(s.get("s_min",0)))
        self.spin_min_v = QtWidgets.QSpinBox(); self.spin_min_v.setRange(0,255); self.spin_min_v.setValue(int(s.get("v_min",0)))
        hmin = QtWidgets.QHBoxLayout(); hmin.addWidget(QtWidgets.QLabel("H min")); hmin.addWidget(self.spin_min_h)
        hmin.addWidget(QtWidgets.QLabel("S min")); hmin.addWidget(self.spin_min_s); hmin.addWidget(QtWidgets.QLabel("V min")); hmin.addWidget(self.spin_min_v)
        
        # HSV Max values
        self.spin_max_h = QtWidgets.QSpinBox(); self.spin_max_h.setRange(0,180); self.spin_max_h.setValue(int(s.get("h_max",180)))
        self.spin_max_s = QtWidgets.QSpinBox(); self.spin_max_s.setRange(0,255); self.spin_max_s.setValue(int(s.get("s_max",255)))
        self.spin_max_v = QtWidgets.QSpinBox(); self.spin_max_v.setRange(0,255); self.spin_max_v.setValue(int(s.get("v_max",255)))
        hmax = QtWidgets.QHBoxLayout(); hmax.addWidget(QtWidgets.QLabel("H max")); hmax.addWidget(self.spin_max_h)
        hmax.addWidget(QtWidgets.QLabel("S max")); hmax.addWidget(self.spin_max_s); hmax.addWidget(QtWidgets.QLabel("V max")); hmax.addWidget(self.spin_max_v)
        
        # threshold
        self.spin_thresh = QtWidgets.QSpinBox(); self.spin_thresh.setRange(1,100); self.spin_thresh.setValue(int(s.get("threshold_pct",90)))
        
        # save button
        btn_save = QtWidgets.QPushButton("Save Settings")
        btn_save.clicked.connect(self.save_settings_from_ui)
        
        # Recommended values for welding
        info_label = QtWidgets.QLabel("Recommended for welding:\n"
                                     "Hot Metal: H: 0-30, S: 50-255, V: 50-255\n"
                                     "Cool Metal: H: 0-180, S: 0-50, V: 0-100\n"
                                     "Arc/Glare: H: 0-180, S: 0-50, V: 200-255")
        info_label.setStyleSheet("background: #e8f4ff; padding: 8px; border-radius: 6px;")
        
        # layout rows
        layout.addRow("HSV Min Values", hmin)
        layout.addRow("HSV Max Values", hmax)
        layout.addRow("Threshold (%)", self.spin_thresh)
        layout.addRow(btn_save)
        layout.addRow(info_label)

    def save_settings_from_ui(self):
        new = {
            "h_min": int(self.spin_min_h.value()),
            "s_min": int(self.spin_min_s.value()),
            "v_min": int(self.spin_min_v.value()),
            "h_max": int(self.spin_max_h.value()),
            "s_max": int(self.spin_max_s.value()),
            "v_max": int(self.spin_max_v.value()),
            "threshold_pct": int(self.spin_thresh.value()),
            "overlay_alpha": self.manager.settings.get("overlay_alpha", 0.45),
            "morph_kernel": self.manager.settings.get("morph_kernel", 3),
            "min_blob_area": self.manager.settings.get("min_blob_area", 10)
        }
        self.manager.save_settings(new)
        QtWidgets.QMessageBox.information(self, "Saved", "HSV Settings saved to settings.json")

    # ---------- misc ----------
    def on_tab_change(self, idx):
        # when switching to Inspect tab, load shapes to widget
        if self.tabs.widget(idx) is self.tab_inspect:
            shapes = self.manager.load_shapes()
            if shapes:
                self.video_widget.shapes = shapes

    def closeEvent(self, ev):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        ev.accept()

# ---------- main ----------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()