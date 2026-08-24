import math

class FusionEngine:
    def __init__(self, origin_lat=26.912400, origin_lon=75.787300):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.imu_history = [0.0] * 70

    def compute_gps(self, x_offset, y_offset=1.75):
        lat = self.origin_lat + (x_offset * 0.000009)
        lon = self.origin_lon + (y_offset * 0.000009)
        return lat, lon

    def compute_shock(self, acc_values):
        if not acc_values or math.isnan(acc_values[1]) or math.isnan(acc_values[2]):
            return 0.15
        val_z = abs(acc_values[2] - 9.81)
        val_y = abs(acc_values[1] - 9.81)
        return max(val_z, val_y)

    def evaluate_severity(self, cam_score, imu_score):
        fused_score = (0.70 * cam_score) + (0.30 * imu_score)
        fused_pct = int(fused_score * 100.0)
        cam_pct = int(cam_score * 100.0)
        imu_pct = int(imu_score * 100.0)

        if fused_pct >= 62 or (cam_score > 0.60 and imu_score > 0.55):
            sev = "CRITICAL"
        elif fused_pct >= 40 or cam_score > 0.45 or imu_score > 0.35:
            sev = "MEDIUM"
        else:
            sev = "LOW"

        return sev, cam_pct, imu_pct, fused_pct
