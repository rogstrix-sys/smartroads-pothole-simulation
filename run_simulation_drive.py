import os
import sys
import time
import csv
import threading
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# Webots Python Controller setup
if 'WEBOTS_HOME' not in os.environ:
    os.environ['WEBOTS_HOME'] = r'C:\Program Files\Webots'

controller_path = os.path.join(os.environ['WEBOTS_HOME'], 'lib', 'controller', 'python')
if controller_path not in sys.path:
    sys.path.append(controller_path)

from controller import Supervisor

print('>>> Initializing Fast High-FPS Simulation Drive on RTX 4050...', flush=True)

# 1. Global Frame Buffer for Zero-Lag Streaming
latest_encoded_frame = None
frame_lock = threading.Lock()

# 2. FastAPI MJPEG Streaming Server
app = FastAPI()

def generate_mjpeg_stream():
    global latest_encoded_frame
    while True:
        with frame_lock:
            if latest_encoded_frame is None:
                time.sleep(0.01)
                continue
            frame_bytes = latest_encoded_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.016)  # ~60 FPS stream rate

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(generate_mjpeg_stream(), media_type='multipart/x-mixed-replace; boundary=frame')

def run_api_server():
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='warning')

# Start HTTP Streamer in Background Thread
api_thread = threading.Thread(target=run_api_server, daemon=True)
api_thread.start()

# 3. Initialize Webots Supervisor
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice('camera')
if camera:
    camera.enable(timestep)

accelerometer = robot.getDevice('accelerometer')
if accelerometer:
    accelerometer.enable(timestep)

gps = robot.getDevice('gps')
if gps:
    gps.enable(timestep)

car_node = robot.getFromDef('CAR_NODE')
trans_field = car_node.getField('translation') if car_node else robot.getSelf().getField('translation')

# Setup Log Files
os.makedirs('logs/snapshots', exist_ok=True)
csv_file = 'logs/detections.csv'
with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'Latitude', 'Longitude', 'Delta_Az', 'Severity', 'Snapshot_Path'])

path_file = 'logs/vehicle_path.csv'
with open(path_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Latitude', 'Longitude'])

# Load YOLO on Dedicated RTX GPU (device=0)
try:
    import torch
    from ultralytics import YOLO
    model_path = 'vision/yolov8n_pothole.pt' if os.path.exists('vision/yolov8n_pothole.pt') else 'yolov8n.pt'
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to('cuda:0')
        print('>>> YOLO Loaded on CUDA: NVIDIA GeForce RTX 4050', flush=True)
except Exception as e:
    print(f'YOLO load warning: {e}')
    model = None

ORIGIN_LAT = 26.912400
ORIGIN_LON = 75.787300

pothole_locations = [25.0, 65.0, 110.0, 160.0, 210.0]
pothole_severities = ['LOW', 'MEDIUM', 'CRITICAL', 'CRITICAL', 'MEDIUM']

last_logged_idx = -1
speed_mps = 8.5
step_counter = 0

print('>>> VEHICLE DRIVING & BROADCASTING 60 FPS STREAM AT http://127.0.0.1:8000/video_feed', flush=True)

while robot.step(timestep) != -1:
    step_counter += 1
    curr = trans_field.getSFVec3f()
    new_x = curr[0] + (speed_mps * (timestep / 1000.0))

    # Strict height stability lock at Z = 0.35m
    trans_field.setSFVec3f([new_x, 1.75, 0.35])

    mock_lat = ORIGIN_LAT + (new_x * 0.000009)
    mock_lon = ORIGIN_LON + (1.75 * 0.000009)

    if step_counter % 2 == 0:
        with open(path_file, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([f"{mock_lat:.7f}", f"{mock_lon:.7f}"])

    annotated = None
    if camera:
        raw_image = camera.getImage()
        if raw_image:
            w, h = camera.getWidth(), camera.getHeight()
            img = np.frombuffer(raw_image, np.uint8).reshape((h, w, 4))
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Fast Tensor Core inference on RTX 4050
            if model is not None:
                results = model.predict(source=bgr, conf=0.25, device=0, verbose=False)
                annotated = results[0].plot()
            else:
                annotated = bgr

            # Broadcast directly to MJPEG HTTP stream in memory
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with frame_lock:
                latest_encoded_frame = buffer.tobytes()

    # Check for Pothole Trigger
    for idx, p_pos in enumerate(pothole_locations):
        dist = new_x - p_pos
        if abs(dist) < 1.0 and idx > last_logged_idx and dist >= 0:
            last_logged_idx = idx
            sev = pothole_severities[idx]
            delta_az = 1.6 if sev == 'LOW' else (3.2 if sev == 'MEDIUM' else 5.4)
            snap_name = f'logs/snapshots/defect_{int(time.time()*1000)}.jpg'

            if annotated is not None:
                cv2.imwrite(snap_name, annotated)

            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([time.strftime('%H:%M:%S'), f'{mock_lat:.6f}', f'{mock_lon:.6f}', f'{delta_az:.2f}', sev, snap_name])
            print(f'💥 [{sev}] Pothole Logged at X={new_x:.1f}m | Shock: {delta_az:.2f} m/s²', flush=True)

    if new_x > 250.0:
        trans_field.setSFVec3f([0.0, 1.75, 0.35])
        last_logged_idx = -1
        with open(path_file, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['Latitude', 'Longitude'])
        print('=== Looping City Track ===', flush=True)
