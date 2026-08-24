import os
import cv2
import numpy as np

class VisionDetector:
    def __init__(self, model_path="vision/yolov8n_pothole.pt"):
        self.model = None
        try:
            from ultralytics import YOLO
            target_path = model_path if os.path.exists(model_path) else "yolov8n.pt"
            self.model = YOLO(target_path)
            print(f">>> YOLO Model Loaded: {target_path}", flush=True)
        except Exception as e:
            print(f"YOLO load notice: {e}", flush=True)

    def process_frame(self, raw_buffer, width, height):
        if raw_buffer is None:
            return np.zeros((height, width, 3), dtype=np.uint8), 0.0

        img = np.frombuffer(raw_buffer, np.uint8).reshape((height, width, 4))
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        annotated = bgr.copy()
        cam_score = 0.0

        if self.model:
            results = self.model.predict(source=bgr, conf=0.25, verbose=False)
            annotated = results[0].plot()
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    bx = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    norm_area = ((bx[2] - bx[0]) * (bx[3] - bx[1])) / (width * height)
                    score = (0.50 * conf) + (0.50 * min(norm_area * 4.5, 1.0))
                    if score > cam_score:
                        cam_score = score

        return annotated, cam_score
