# manager.py
import os
import json
import time
import cv2
import numpy as np

SHAPES_PATH = "shapes.json"
SETTINGS_PATH = "settings.json"
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

DEFAULT_SETTINGS = {
    "h_min": 0, "s_min": 0, "v_min": 0,
    "h_max": 180, "s_max": 255, "v_max": 255,
    "threshold_pct": 90,
    "overlay_alpha": 0.45,
    "morph_kernel": 3,
    "min_blob_area": 10
}

class Manager:
    def __init__(self):
        self.settings = self.load_settings()
        # shapes is a list of shape dicts
        self.shapes = self.load_shapes()

    # ---------- settings ----------
    def load_settings(self):
        if os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, "r") as f:
                    s = json.load(f)
                for k, v in DEFAULT_SETTINGS.items():
                    if k not in s:
                        s[k] = v
                return s
            except Exception:
                pass
        # fallback default
        self.save_settings(DEFAULT_SETTINGS.copy())
        return DEFAULT_SETTINGS.copy()

    def save_settings(self, settings: dict):
        os.makedirs(os.path.dirname(SETTINGS_PATH) or ".", exist_ok=True)
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
        self.settings = settings.copy()
        return SETTINGS_PATH

    # ---------- shapes ----------
    def load_shapes(self):
        if not os.path.exists(SHAPES_PATH):
            return []
        try:
            with open(SHAPES_PATH, "r") as f:
                data = json.load(f)
            return data.get("shapes", [])
        except Exception:
            return []

    def save_shapes(self, shapes: list):
        os.makedirs(os.path.dirname(SHAPES_PATH) or ".", exist_ok=True)
        meta = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "shapes": shapes}
        with open(SHAPES_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        self.shapes = shapes
        return SHAPES_PATH

    # ---------- core pixel-check (NOW USING HSV) ----------
    def compute_mask_and_percent_hsv(self, crop_bgr, settings=None):
        """
        crop_bgr: HxWx3 BGR uint8
        returns (mask_uint8 (0/255), percent_ok float 0..100)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None, 0.0
        
        # Convert BGR to HSV
        crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        
        s = settings if settings is not None else self.settings
        hmin = int(s.get("h_min", 0)); smin = int(s.get("s_min", 0)); vmin = int(s.get("v_min", 0))
        hmax = int(s.get("h_max", 180)); smax = int(s.get("s_max", 255)); vmax = int(s.get("v_max", 255))
        
        lower = np.array([hmin, smin, vmin], dtype=np.uint8)
        upper = np.array([hmax, smax, vmax], dtype=np.uint8)
        mask = cv2.inRange(crop_hsv, lower, upper)

        # cleaning
        k = int(s.get("morph_kernel", 3))
        if k > 1:
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # remove small blobs
        min_area = int(s.get("min_blob_area", 10))
        if min_area > 1:
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cleaned = np.zeros_like(mask)
            for ctr in contours:
                if cv2.contourArea(ctr) >= min_area:
                    cv2.drawContours(cleaned, [ctr], -1, 255, -1)
            mask = cleaned

        total = mask.size
        if total == 0:
            return mask, 0.0
        ok = int(cv2.countNonZero(mask))
        pct = (ok / total) * 100.0
        return mask, float(pct)

    # ---------- history ----------
    def record_history(self, per_shape_dict, overall_ok):
        os.makedirs(HISTORY_DIR, exist_ok=True)
        existing = [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]
        sn = len(existing) + 1
        percents = [v.get("percent", 0.0) for v in per_shape_dict.values()] if per_shape_dict else [0.0]
        overall_percent = sum(percents) / len(percents) if percents else 0.0
        rec = {
            "sn": sn,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall": "OK" if overall_ok else "NG",
            "overall_percent": float(overall_percent),
            "details": {k: {"percent": v["percent"], "ok": v["ok"]} for k, v in per_shape_dict.items()}
        }
        path = os.path.join(HISTORY_DIR, f"{sn}.json")
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        return path