import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class HUDDisplay:
    def __init__(self, window_name="Pothole Detection HUD"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        font_path = "C:/Windows/Fonts/arialbd.ttf" if os.path.exists("C:/Windows/Fonts/arialbd.ttf") else None
        font_reg = "C:/Windows/Fonts/arial.ttf" if os.path.exists("C:/Windows/Fonts/arial.ttf") else None
        self.f_title = ImageFont.truetype(font_path, 22) if font_path else ImageFont.load_default()
        self.f_sub = ImageFont.truetype(font_reg, 16) if font_reg else ImageFont.load_default()
        self.f_panel = ImageFont.truetype(font_path, 16) if font_path else ImageFont.load_default()
        self.f_head = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
        self.f_row = ImageFont.truetype(font_reg, 14) if font_reg else ImageFont.load_default()

    def render(self, camera_frame, curr_x, real_delta_az, imu_history, confirmed_potholes, lat, lon):
        hud = np.zeros((720, 1280, 3), dtype=np.uint8)
        hud[:] = (18, 22, 30)

        # Header panel
        cv2.rectangle(hud, (20, 15), (1260, 65), (32, 40, 54), -1)
        cv2.rectangle(hud, (20, 15), (1260, 65), (80, 110, 150), 2)

        # Camera frame
        cam_resized = cv2.resize(camera_frame, (610, 340))
        hud[110:450, 20:630] = cam_resized
        cv2.rectangle(hud, (20, 110), (630, 450), (80, 110, 150), 2)

        # Radar track
        cv2.rectangle(hud, (650, 110), (1260, 450), (22, 28, 38), -1)
        cv2.rectangle(hud, (650, 110), (1260, 450), (80, 110, 150), 2)
        cv2.line(hud, (680, 280), (1230, 280), (65, 75, 90), 12)
        cv2.line(hud, (680, 280), (1230, 280), (220, 220, 220), 2)

        for p in confirmed_potholes:
            px = int(680 + (p["x"] / 390.0) * 550)
            p_col = (50, 205, 50) if p["sev"] == "LOW" else ((0, 165, 255) if p["sev"] == "MEDIUM" else (40, 40, 235))
            cv2.circle(hud, (px, 280), 9, p_col, -1)
            cv2.circle(hud, (px, 280), 11, (255, 255, 255), 1)

        car_px = int(680 + (curr_x / 390.0) * 550)
        triangle_pts = np.array([[car_px + 12, 280], [car_px - 8, 270], [car_px - 8, 290]], np.int32)
        cv2.fillPoly(hud, [triangle_pts], (255, 215, 0))

        # IMU Graph
        cv2.rectangle(hud, (20, 495), (630, 695), (22, 28, 38), -1)
        cv2.rectangle(hud, (20, 495), (630, 695), (80, 110, 150), 2)
        pts = [(int(30 + i * 8.4), int(680 - min(val, 7.5) * 23.0)) for i, val in enumerate(imu_history)]
        for i in range(len(pts) - 1):
            cv2.line(hud, pts[i], pts[i+1], (0, 230, 255), 2, cv2.LINE_AA)

        # Incident table
        cv2.rectangle(hud, (650, 495), (1260, 695), (22, 28, 38), -1)
        cv2.rectangle(hud, (650, 495), (1260, 695), (80, 110, 150), 2)
        cv2.line(hud, (660, 530), (1250, 530), (55, 70, 90), 1)

        img_pil = Image.fromarray(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        draw.text((35, 27), "POTHOLE DETECTION HUD", font=self.f_title, fill=(255, 255, 255))
        draw.text((780, 31), f"70% CAM + 30% IMU | LAT: {lat:.5f}  LON: {lon:.5f}", font=self.f_sub, fill=(120, 235, 120))
        draw.text((20, 85), "LIVE ONBOARD DASHCAM (YOLOv8 DETECTION)", font=self.f_panel, fill=(210, 210, 210))
        draw.text((650, 85), "400-METER URBAN CORRIDOR TRAJECTORY & GPS RADAR", font=self.f_panel, fill=(210, 210, 210))
        draw.text((20, 470), f"3-AXIS IMU SUSPENSION RESPONSE (\u0394Az: {real_delta_az:.2f} m/s\u00b2)", font=self.f_panel, fill=(210, 210, 210))
        draw.text((650, 470), "POST-PASS SENSOR FUSION DEFECT LOG", font=self.f_panel, fill=(210, 210, 210))

        draw.text((665, 508), "TIME", font=self.f_head, fill=(160, 190, 220))
        draw.text((745, 508), "SEVERITY", font=self.f_head, fill=(160, 190, 220))
        draw.text((840, 508), "CAM (70%)", font=self.f_head, fill=(160, 190, 220))
        draw.text((935, 508), "IMU (30%)", font=self.f_head, fill=(160, 190, 220))
        draw.text((1030, 508), "FUSED", font=self.f_head, fill=(160, 190, 220))
        draw.text((1105, 508), "GPS COORDINATES", font=self.f_head, fill=(160, 190, 220))

        for r_idx, defect in enumerate(confirmed_potholes[-5:]):
            y_row = 544 + r_idx * 28
            col = (50, 205, 50) if defect["sev"] == "LOW" else ((255, 165, 0) if defect["sev"] == "MEDIUM" else (235, 50, 50))
            draw.text((665, y_row), f"{defect['time']}", font=self.f_row, fill=(230, 230, 230))
            draw.text((745, y_row), f"{defect['sev']:<8}", font=self.f_head, fill=col)
            draw.text((855, y_row), f"{defect['cam_pct']:>3}%", font=self.f_row, fill=(230, 230, 230))
            draw.text((950, y_row), f"{defect['imu_pct']:>3}%", font=self.f_row, fill=(230, 230, 230))
            draw.text((1040, y_row), f"{defect['fused_pct']:>3}%", font=self.f_head, fill=col)
            draw.text((1105, y_row), f"({defect['lat']:.5f}, {defect['lon']:.5f})", font=self.f_row, fill=(190, 215, 235))

        hud_final = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, hud_final)
        return cv2.waitKey(1) & 0xFF == ord('q')

    def close(self):
        cv2.destroyAllWindows()
