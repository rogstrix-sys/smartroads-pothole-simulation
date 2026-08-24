import os
import time
import csv
import math
import cv2
import numpy as np
from vehicle import Driver
from ultralytics import YOLO

# 1. Initialize Car Driver API
driver = Driver()
timestep = int(driver.getBasicTimeStep())

# 2. Initialize Attached Sensors
camera = driver.getDevice("camera")
if camera:
    camera.enable(timestep)
    cam_w = camera.getWidth()
    cam_h = camera.getHeight()
else:
    cam_w, cam_h = 640, 480

accelerometer = driver.getDevice("accelerometer")
if accelerometer:
    accelerometer.enable(timestep)

gps = driver.getDevice("gps")
if gps:
    gps.enable(timestep)

# 3. Vehicle Drive Settings (Cruising at 30 km/h)
driver.setCruisingSpeed(30.0)
driver.setSteeringAngle(0.0)
driver.setHazardFlashers(False)
driver.setDippedBeams(True)

# 4. YOLO Model Setup
model_path = "vision/yolov8n_pothole.pt" if os.path.exists("vision/yolov8n_pothole.pt") else "yolov8n.pt"
model = YOLO(model_path)

os.makedirs("logs/snapshots", exist_ok=True)
csv_file = "logs/detections.csv"

# Initialize CSV log with headers
if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Latitude", "Longitude", "Delta_Az", "Severity", "Snapshot_Path"])

# City Reference Coordinates (Jaipur baseline)
ORIGIN_LAT = 26.912400
ORIGIN_LON = 75.787300

frame_counter = 0
last_logged_pos = -999.0

print("=== SmartRoads Vehicle Controller Active ===")

while driver.step() != -1:
    frame_counter += 1
    
    # Read Telemetry Sensors
    pos = gps.getValues() if gps else [0.0, 0.0, 0.0]
    acc = accelerometer.getValues() if accelerometer else [0.0, 0.0, 9.81]
    
    # Vertical vibration shock calculation (Delta Az)
    delta_az = abs(acc[1] - 9.81) if abs(acc[1] - 9.81) > abs(acc[2] - 9.81) else abs(acc[2] - 9.81)

    mock_lat = ORIGIN_LAT + (pos[0] * 0.000009)
    mock_lon = ORIGIN_LON + (pos[2] * 0.000009)

    # Process camera every 2nd step to maintain smooth real-time framerate
    if camera and frame_counter % 2 == 0:
        raw_image = camera.getImage()
        if raw_image:
            img = np.frombuffer(raw_image, np.uint8).reshape((cam_h, cam_w, 4))
            bgr_frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # YOLO Inference
            results = model.predict(source=bgr_frame, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()

            # Multi-Sensor Fusion Trigger (Visual Defect OR Shock Spike)
            has_visual_defect = results[0].boxes is not None and len(results[0].boxes.xyxy) > 0
            has_shock_impact = delta_az > 1.2

            # Spatial deduplication: Prevent logging duplicate events within 5 meters
            if (has_shock_impact or has_visual_defect) and (abs(pos[0] - last_logged_pos) > 5.0):
                last_logged_pos = pos[0]
                
                # Severity Classification
                if delta_az > 4.0:
                    severity = "CRITICAL"
                elif delta_az > 2.0 or (has_visual_defect and not has_shock_impact):
                    severity = "MEDIUM"
                else:
                    severity = "LOW"
                    
                snapshot_name = f"logs/snapshots/defect_{int(time.time()*1000)}.jpg"
                cv2.imwrite(snapshot_name, annotated_frame)

                with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        time.strftime("%H:%M:%S"),
                        f"{mock_lat:.6f}",
                        f"{mock_lon:.6f}",
                        f"{delta_az:.2f}",
                        severity,
                        snapshot_name
                    ])
                print(f"🔥 [{severity}] Defect Logged at GPS: ({mock_lat:.5f}, {mock_lon:.5f}) | ΔAz={delta_az:.2f} m/s²")

            cv2.imshow("SmartRoads AI Vision Stream", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
