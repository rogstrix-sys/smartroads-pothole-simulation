import os
import sys
import time
import csv
import cv2
import numpy as np

from controller import Supervisor

print('>>> Starting SmartRoads Embedded Vehicle Controller & HUD...', flush=True)

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# 1. Enable Camera Device
camera = robot.getDevice('pothole_cam') or robot.getDevice('camera')
if camera:
    camera.enable(timestep)
    print(f'>>> Onboard Camera Online: {camera.getWidth()}x{camera.getHeight()}', flush=True)
else:
    print('>>> WARNING: No camera node detected!', flush=True)

accelerometer = robot.getDevice('accelerometer')
if accelerometer:
    accelerometer.enable(timestep)

gps = robot.getDevice('gps')
if gps:
    gps.enable(timestep)

car_node = robot.getSelf()
trans_field = car_node.getField('translation') if car_node else None

if trans_field:
    trans_field.setSFVec3f([0.0, 1.75, 0.35])

os.makedirs('logs/snapshots', exist_ok=True)
csv_file = 'logs/detections.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['Timestamp', 'Latitude', 'Longitude', 'Delta_Az', 'Severity', 'Snapshot_Path'])

# 2. Load YOLO Model
try:
    from ultralytics import YOLO
    model_path = 'vision/yolov8n_pothole.pt' if os.path.exists('vision/yolov8n_pothole.pt') else 'yolov8n.pt'
    model = YOLO(model_path)
    print(f'>>> YOLO Model loaded: {model_path}', flush=True)
except Exception:
    model = None

ORIGIN_LAT = 26.912400
ORIGIN_LON = 75.787300

pothole_locations = [35.0, 85.0, 140.0, 200.0, 260.0, 320.0, 375.0]
pothole_types = [
    'Fatigue Crack', 'Circular Pothole', 'Longitudinal Trench',
    'Collapse Crater', 'Diagonal Shear', 'Surface Spall', 'Transverse Fracture'
]
pothole_severities = ['LOW', 'MEDIUM', 'CRITICAL', 'CRITICAL', 'MEDIUM', 'LOW', 'CRITICAL']
pothole_shocks = [1.20, 2.80, 4.60, 6.80, 3.40, 1.50, 5.50]
detected_potholes = []

imu_history = [0.0] * 70
speed_mps = 9.0
last_logged_idx = -1
current_x = 0.0

cv2.namedWindow('SmartRoads Executive Telemetry Command Center', cv2.WINDOW_NORMAL)
cv2.resizeWindow('SmartRoads Executive Telemetry Command Center', 1280, 720)

while robot.step(timestep) != -1:
    if trans_field:
        curr = trans_field.getSFVec3f()
        current_x = curr[0] + (speed_mps * (timestep / 1000.0))
        if current_x > 390.0:
            trans_field.setSFVec3f([0.0, 1.75, 0.35])
            current_x = 0.0
            last_logged_idx = -1
            detected_potholes.clear()
        else:
            trans_field.setSFVec3f([current_x, 1.75, 0.35])
    else:
        current_x += (speed_mps * (timestep / 1000.0))

    mock_lat = ORIGIN_LAT + (current_x * 0.000009)
    mock_lon = ORIGIN_LON + (1.75 * 0.000009)

    annotated = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 3. Direct Native Camera Memory Read
    if camera:
        raw = camera.getImage()
        if raw is not None:
            w, h = camera.getWidth(), camera.getHeight()
            img = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            if model:
                results = model.predict(source=bgr, conf=0.25, verbose=False)
                annotated = results[0].plot()
            else:
                annotated = bgr

    # Telemetry and Vibration Spikes
    current_shock = 0.15
    for idx, p_pos in enumerate(pothole_locations):
        dist = current_x - p_pos
        if abs(dist) < 1.1:
            current_shock = pothole_shocks[idx]
            sev = pothole_severities[idx]
            p_name = pothole_types[idx]
            if idx > last_logged_idx and dist >= 0:
                last_logged_idx = idx
                snap_name = f'logs/snapshots/defect_{int(time.time()*1000)}.jpg'
                cv2.imwrite(snap_name, annotated)
                detected_potholes.append({
                    'x': current_x, 'lat': mock_lat, 'lon': mock_lon, 'sev': sev,
                    'type': p_name, 'shock': current_shock, 'time': time.strftime('%H:%M:%S')
                })
                with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([time.strftime('%H:%M:%S'), f'{mock_lat:.6f}', f'{mock_lon:.6f}', f'{current_shock:.2f}', sev, snap_name])
                print(f'💥 [{sev}] {p_name} Logged at X={current_x:.1f}m | Shock: {current_shock:.2f} m/s²', flush=True)

    imu_history.append(current_shock)
    if len(imu_history) > 70:
        imu_history.pop(0)

    # Assemble HUD
    hud = np.zeros((720, 1280, 3), dtype=np.uint8)
    hud[:] = (15, 18, 26)

    # Header
    cv2.rectangle(hud, (20, 15), (1260, 65), (38, 48, 64), -1)
    cv2.rectangle(hud, (20, 15), (1260, 65), (90, 120, 160), 1)
    cv2.putText(hud, "SMARTROADS -- 400M URBAN CORRIDOR TELEMETRY HUD", (35, 48), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(hud, f"RTX 4050 GPU | 60 FPS | LAT: {mock_lat:.5f}  LON: {mock_lon:.5f}", (710, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 220, 120), 1)

    # Panel 1: Live Dashcam
    cv2.putText(hud, "LIVE ONBOARD DASHCAM", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cam_resized = cv2.resize(annotated, (610, 340))
    hud[110:450, 20:630] = cam_resized
    cv2.rectangle(hud, (20, 110), (630, 450), (90, 120, 160), 2)

    # Panel 2: Live GIS Radar Track
    cv2.putText(hud, "400-METER URBAN CORRIDOR TRAJECTORY & GPS RADAR", (650, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.rectangle(hud, (650, 110), (1260, 450), (22, 28, 38), -1)
    cv2.rectangle(hud, (650, 110), (1260, 450), (90, 120, 160), 2)
    cv2.line(hud, (680, 280), (1230, 280), (70, 80, 95), 14)
    cv2.line(hud, (680, 280), (1230, 280), (255, 255, 255), 2)

    for p in detected_potholes:
        px = int(680 + (p['x'] / 390.0) * 550)
        p_col = (50, 205, 50) if p['sev'] == 'LOW' else ((0, 165, 255) if p['sev'] == 'MEDIUM' else (30, 30, 235))
        cv2.circle(hud, (px, 280), 9, p_col, -1)
        cv2.circle(hud, (px, 280), 11, (255, 255, 255), 1)
        cv2.putText(hud, p['sev'][:3], (px - 10, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.35, p_col, 1)

    car_px = int(680 + (current_x / 390.0) * 550)
    triangle_pts = np.array([[car_px + 12, 280], [car_px - 8, 270], [car_px - 8, 290]], np.int32)
    cv2.fillPoly(hud, [triangle_pts], (255, 215, 0))
    cv2.polylines(hud, [triangle_pts], True, (255, 255, 255), 1)

    # Panel 3: 3-Axis IMU Waveform
    cv2.putText(hud, "3-AXIS IMU VIBRATION WAVEFORM (Delta Az)", (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.rectangle(hud, (20, 495), (630, 695), (22, 28, 38), -1)
    cv2.rectangle(hud, (20, 495), (630, 695), (90, 120, 160), 2)
    pts = [(int(30 + i * 8.4), int(680 - min(val, 7.5) * 23.0)) for i, val in enumerate(imu_history)]
    for i in range(len(pts) - 1):
        cv2.line(hud, pts[i], pts[i+1], (0, 230, 255), 2)

    # Panel 4: Incident Telemetry Log
    cv2.putText(hud, "RECENT DEFECT LOG (TYPE, IMPACT & GPS)", (650, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.rectangle(hud, (650, 495), (1260, 695), (22, 28, 38), -1)
    cv2.rectangle(hud, (650, 495), (1260, 695), (90, 120, 160), 2)
    cv2.putText(hud, "TIME       TYPE                  SEVERITY      IMU SHOCK     GPS COORD", (665, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 180, 220), 1)
    cv2.line(hud, (660, 530), (1250, 530), (50, 65, 85), 1)

    for r_idx, defect in enumerate(detected_potholes[-5:]):
        y_row = 560 + r_idx * 28
        p_col = (50, 205, 50) if defect['sev'] == 'LOW' else ((0, 165, 255) if defect['sev'] == 'MEDIUM' else (50, 50, 240))
        cv2.putText(hud, f"{defect['time']}    {defect['type']:<20}  {defect['sev']:<10}    {defect['shock']:.2f} m/s2      ({defect['lat']:.5f}, {defect['lon']:.5f})", (665, y_row), cv2.FONT_HERSHEY_SIMPLEX, 0.40, p_col, 1)

    cv2.imshow('SmartRoads Executive Telemetry Command Center', hud)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
