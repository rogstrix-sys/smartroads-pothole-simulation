# SmartRoads: Digital Twin for Real-Time Pothole & Road Surface Damage Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Webots](https://img.shields.io/badge/Simulator-Cyberbotics%20Webots-red.svg)](https://cyberbotics.com/)
[![YOLOv8](https://img.shields.io/badge/AI%20Vision-Ultralytics%20YOLOv8-green.svg)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SmartRoads is a high-fidelity **Digital Twin and Multi-Sensor Fusion Framework** designed to simulate an autonomous municipal road-survey vehicle. By combining **real-time Computer Vision (YOLOv8)** with **chassis physical suspension telemetry (3-Axis IMU)**, the system detects road damage and validates defect severity while rejecting visual false alarms (e.g., shadows, oil stains, manhole covers).

---

## 🌟 Key Highlights

* **3D Digital Twin Simulation:** Implemented in Cyberbotics Webots with realistic vehicle dynamics, suspension models, and calibrated road surface meshes.
* **Real-Time AI Vision:** Custom-weighted YOLOv8 executing at 90–140+ FPS on CUDA to detect cracks and potholes from a downward-facing camera feed.
* **Multi-Sensor Confirmation Pipeline:** Evaluates visual bounding boxes and confirms structural impact using vertical IMU spikes ($\Delta a_z = |a_z - 9.81|\text{ m/s}^2$).
* **Peak-Hold & Spatial Post-Pass Release Gate:** Solves perspective scaling and eliminates multi-frame duplicate logging by committing a single consolidated event once the vehicle completely passes the defect (+1.8m).
* **Dual Telemetry Interface:** Includes a 4-panel live OpenCV HUD with real-time IMU waveforms and an interactive Streamlit GIS mapping dashboard.

---

## 📐 System Architecture & Workflow

\\\
[ 3D Webots Simulation ] ──> Raw BGRA Feed & IMU Telemetry (32ms)
            │
            ▼
[ Computer Vision Pipeline ] ──> YOLOv8 Object Detection & Peak Visual Scoring
            │
            ▼
[ Wheel Contact Stage ] ──> Accelerometer Impact Shock Sampling (Δa_z = |a_z - 9.81|)
            │
            ▼
[ Post-Pass Release Gate ] ──> Unified Fusion Equation & Deduplication
            │
            ├─► Serialization: logs/detections.csv & logs/snapshots/*.jpg
            ├─► Desktop HUD: 4-Panel Real-Time Telemetry Interface
            └─► Web Dashboard: Streamlit + Folium Interactive GIS Map
\\\

### Sensor Fusion Scoring Formulation

streamlit-folium\text{Severity Score} = 0.70 \cdot (\text{Peak Visual Score}) + 0.30 \cdot \left(\frac{\min(\Delta a_z, 6.5)}{6.5}\right)streamlit-folium

* **Low / Small ($< 38\%$):** Superficial surface cracks with negligible vertical suspension displacement.
* **Medium (\% - 61\%$):** Moderate asphalt depression requiring scheduled municipal repair.
* **Critical ($\ge 62\%$):** Severe cratering with high vertical G-force shock requiring emergency response.

---

## 📊 Performance Benchmarks

| Metric / Component | Specification / Value |
| :--- | :--- |
| **Vision Backbone** | YOLOv8n (Road Defect Custom Weights) |
| **Vision Benchmark** | 66.28% mAP@50 (Precision: 74.39%, Recall: 62.46%) |
| **False Positive Rejection** | **~100%** (via physical IMU shock verification) |
| **Inference Latency** | 90–140+ FPS (NVIDIA RTX 4050 Laptop GPU) |
| **Deduplication Rate** | 100% (Single commit per physical defect via Post-Pass Gate) |

---

## 📁 Repository Structure

\\\	ext
smartroads-pothole-simulation/
├── SmartRoads/
│   └── webots/
│       ├── SmartRoads.wbt        # 3D World file (road corridor, sensors, vehicle)
│       └── textures/             # Organic transparent alpha pothole decals
├── dashboard/
│   └── app.py                    # Streamlit GIS mapping & audit log dashboard
├── logs/
│   ├── detections.csv            # Auto-generated geo-tagged detection records
│   └── snapshots/                # Region-of-interest cropped defect thumbnails
├── vision/
│   └── yolov8n_pothole.pt        # Trained YOLOv8 road defect weights
├── hud_dashboard.py              # Central synchronized dual-sensor fusion controller & HUD
├── requirements.txt              # Project dependencies
└── README.md
\\\

---

## 🚀 Getting Started

### 1. Launch 3D Physics Simulation
1. Open **Cyberbotics Webots**.
2. Load the world file: \SmartRoads/webots/SmartRoads.wbt\.
3. Click the **Play ($\blacktriangleright$)** button to initialize the physics engine.

### 2. Start Telemetry HUD & Vision Controller
\\\ash
python hud_dashboard.py
\\\
*Press \q\ in the HUD window to exit.*

### 3. Run Live Streamlit GIS Dashboard
In a second terminal:
\\\ash
streamlit run dashboard/app.py
\\\
Access the real-time Leaflet map and incident gallery at \http://localhost:8501\.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
