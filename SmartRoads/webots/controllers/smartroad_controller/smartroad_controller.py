import os
import sys
import time
import csv
import cv2
import numpy as np
from controller import Supervisor

print(">>> Python Controller Starting...", flush=True)

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
if camera:
    camera.enable(timestep)

accelerometer = robot.getDevice("accelerometer")
if accelerometer:
    accelerometer.enable(timestep)

gps = robot.getDevice("gps")
if gps:
    gps.enable(timestep)

# Connect to root CAR_CONTAINER node
car_node = robot.getFromDef("CAR_CONTAINER")
if car_node is None:
    car_node = robot.getSelf()

trans_field = car_node.getField("translation")

# Setup YOLO & Logs
model_path = "vision/yolov8n_pothole.pt" if os.path.exists("vision/yolov8n_pothole.pt") else "yolov8n.pt"
try:
    from ultralytics import YOLO
    model = YOLO(model_path)
except Exception as e:
    print(f"YOLO Warning: {e}", flush=True)
    model = None

os.makedirs("logs/snapshots", exist_ok=True)
csv_file = "logs/detections.csv"

if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Latitude", "Longitude", "Delta_Az", "Severity", "Snapshot_Path"])

ORIGIN_LAT = 26.912400
ORIGIN_LON = 75.787300

frame_counter = 0
last_logged_x = -999.0
speed_mps = 8.0  # Cruise speed (~28 km/h)

print(">>> SUCCESS: SmartRoads Vehicle is Now Driving Forward!", flush=True)

while robot.step(timestep) != -1:
    frame_counter += 1
    
    # Kinematic propulsion
    curr = trans_field.getSFVec3f()
    new_x = curr[0] + (speed_mps * (timestep / 1000.0))
    trans_field.setSFVec3f([new_x, 1.75, 0.35])

    pos = gps.getValues() if gps else [new_x, 1.75, 0.35]
    
    is_at_pothole = (abs(new_x - 20.0) < 0.6) or (abs(new_x - 50.0) < 0.6)
    delta_az = 3.8 if is_at_pothole else 0.1

    mock_lat = ORIGIN_LAT + (pos[0] * 0.000009)
    mock_lon = ORIGIN_LON + (pos[1] * 0.000009)

    if camera and frame_counter % 2 == 0 and model is not None:
        raw_image = camera.getImage()
        if raw_image:
            w, h = camera.getWidth(), camera.getHeight()
            img = np.frombuffer(raw_image, np.uint8).reshape((h, w, 4))
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            results = model.predict(source=bgr, conf=0.25, verbose=False)
            annotated = results[0].plot()

            has_visual = results[0].boxes is not None and len(results[0].boxes.xyxy) > 0

            if (delta_az > 1.0 or has_visual) and (new_x - last_logged_x > 5.0):
                last_logged_x = new_x
                severity = "CRITICAL" if (delta_az > 3.0 or new_x > 45) else "MEDIUM"
                snap_name = f"logs/snapshots/defect_{int(time.time()*1000)}.jpg"
                cv2.imwrite(snap_name, annotated)

                with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([time.strftime("%H:%M:%S"), f"{mock_lat:.6f}", f"{mock_lon:.6f}", f"{delta_az:.2f}", severity, snap_name])
                print(f"🔥 [{severity}] Pothole Logged at X={new_x:.1f}m | GPS: ({mock_lat:.5f}, {mock_lon:.5f})", flush=True)

            cv2.imshow("Driver AI Vision Stream", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
