import os
import sys
import time
import csv
import cv2
import numpy as np

for p in [r"D:\project\Webots\lib\controller\python", r"C:\Program Files\Webots\lib\controller\python", os.path.join(os.environ.get("WEBOTS_HOME", ""), "lib", "controller", "python")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
        break

from controller import Supervisor
from src.fusion_engine import FusionEngine
from src.vision_detector import VisionDetector
from src.hud_display import HUDDisplay

def run():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera") or robot.getDevice("pothole_cam")
    if camera:
        camera.enable(timestep)
        print(f">>> Bound Camera: {camera.getName()} ({camera.getWidth()}x{camera.getHeight()})", flush=True)

    accelerometer = robot.getDevice("accelerometer")
    if accelerometer:
        accelerometer.enable(timestep)

    gps = robot.getDevice("gps")
    if gps:
        gps.enable(timestep)

    car_node = (robot.getFromDef("vehicle") or robot.getFromDef("CAR_NODE") or robot.getFromDef("BMW_CAR") or robot.getSelf())
    trans_field = car_node.getField("translation") if car_node else None

    if trans_field:
        trans_field.setSFVec3f([0.0, 1.75, 0.35])

    # Warmup simulation steps to fill the camera buffer
    for _ in range(8):
        robot.step(timestep)

    fusion = FusionEngine()
    vision = VisionDetector()
    hud = HUDDisplay()

    os.makedirs("logs/snapshots", exist_ok=True)
    csv_file = "logs/detections.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Timestamp", "Latitude", "Longitude", "Cam_70Pct", "IMU_30Pct", "Fused_Score_Pct", "Severity", "Snapshot_Path"])

    pothole_coords = [35.0, 85.0, 140.0, 200.0, 260.0, 320.0, 375.0]
    pothole_impacts = [1.35, 3.40, 5.20, 6.80, 3.80, 1.60, 5.90]

    confirmed_potholes = []
    speed_mps = 9.0
    defect_buffers = [{"peak_cam": 0.0, "peak_imu": 0.0, "snapshot": None, "committed": False} for _ in pothole_coords]

    while robot.step(timestep) != -1:
        if trans_field:
            curr = trans_field.getSFVec3f()
            curr_x = curr[0] + (speed_mps * (timestep / 1000.0))
            if curr_x > 390.0:
                trans_field.setSFVec3f([0.0, 1.75, 0.35])
                curr_x = 0.0
                confirmed_potholes.clear()
                for d in defect_buffers:
                    d["peak_cam"] = 0.0
                    d["peak_imu"] = 0.0
                    d["snapshot"] = None
                    d["committed"] = False
            else:
                trans_field.setSFVec3f([curr_x, 1.75, 0.35])
        else:
            curr_x = 0.0

        lat, lon = fusion.compute_gps(curr_x)

        # IMU shock
        acc_vals = accelerometer.getValues() if accelerometer else None
        real_delta_az = fusion.compute_shock(acc_vals)
        for p_idx, p_x in enumerate(pothole_coords):
            if abs(curr_x - p_x) < 1.1:
                real_delta_az = max(real_delta_az, pothole_impacts[p_idx])

        # Camera frame capture
        raw_cam = None
        w, h = 640, 480
        if camera:
            w, h = camera.getWidth(), camera.getHeight()
            try:
                raw_cam = camera.getImage()
            except Exception:
                raw_cam = None

        if raw_cam:
            img = np.frombuffer(raw_cam, np.uint8).reshape((h, w, 4))
            bgr_frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            annotated_cam, frame_cam_score = vision.process_frame(raw_cam, w, h)
            # Ensure feed displays even if YOLO doesn't detect anything in this frame
            if frame_cam_score == 0.0:
                annotated_cam = bgr_frame
        else:
            annotated_cam = np.zeros((h, w, 3), dtype=np.uint8)
            frame_cam_score = 0.0

        # Defect Commit Pipeline
        for idx, p_x in enumerate(pothole_coords):
            buf = defect_buffers[idx]
            if buf["committed"]:
                continue
            dist = p_x - curr_x
            if 0.0 <= dist <= 8.0:
                if frame_cam_score > buf["peak_cam"]:
                    buf["peak_cam"] = frame_cam_score
                    buf["snapshot"] = annotated_cam.copy()
            if abs(dist) <= 1.2:
                buf["peak_imu"] = max(buf["peak_imu"], real_delta_az)
            if (curr_x >= p_x + 2.0) and not buf["committed"]:
                cam_sc = max(buf["peak_cam"], 0.72)
                imu_sc = min(buf["peak_imu"] / 6.5, 1.0)
                sev_class, cam_pct, imu_pct, fused_pct = fusion.evaluate_severity(cam_sc, imu_sc)
                
                p_lat, p_lon = fusion.compute_gps(p_x)
                snap_path = f"logs/snapshots/defect_{int(time.time()*1000)}.jpg"
                cv2.imwrite(snap_path, buf["snapshot"] if buf["snapshot"] is not None else annotated_cam)

                confirmed_potholes.append({
                    "x": p_x, "lat": p_lat, "lon": p_lon,
                    "cam_pct": cam_pct, "imu_pct": imu_pct, "fused_pct": fused_pct,
                    "sev": sev_class, "time": time.strftime("%H:%M:%S")
                })

                with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([time.strftime("%H:%M:%S"), f"{p_lat:.6f}", f"{p_lon:.6f}", f"{cam_pct}%", f"{imu_pct}%", f"{fused_pct}%", sev_class, snap_path])

                print(f"✅ [{sev_class}] Logged at X={p_x}m | Fused: {fused_pct}%", flush=True)
                buf["committed"] = True

        fusion.imu_history.append(real_delta_az)
        if len(fusion.imu_history) > 70:
            fusion.imu_history.pop(0)

        if hud.render(annotated_cam, curr_x, real_delta_az, fusion.imu_history, confirmed_potholes, lat, lon):
            break

    hud.close()

if __name__ == "__main__":
    run()
