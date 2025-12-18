# AI-Based Time & Motion Analysis using YOLOv8 and MediaPipe

## Overview
This project implements an **automated industrial time and motion study system** by integrating two AI models into a single real-time pipeline:

1. **YOLOv8 (Object Detection)** – Detects and tracks the work zone (board).
2. **MediaPipe Hands (Pose Estimation)** – Tracks detailed hand landmarks and motion dynamics.

By fusing these models, the system classifies **Therbligs** (Operation, Transport, Hold, Delay) and computes **cycle-time metrics** automatically, eliminating manual stopwatch-based studies.

---

## Key Features
- Real-time hand tracking (Left & Right hands)
- Custom-trained YOLOv8 model for work-zone detection
- Velocity-based motion analysis
- Finger-count–based Therbligs classification
- Robust state-machine decision logic
- Automatic CSV export (Power BI / Excel compatible)
- GPU acceleration (CUDA) when available

---

## Dual-Model System Architecture

### Model 1: YOLOv8 – Work-Zone Detection
YOLOv8 is trained on a custom Roboflow dataset to detect the **board/work area**.

**Training Script**
```python
from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    model = YOLO("yolov8s.pt")
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        name="flow_process_chart_v1"
    )
```

**Output Model**
```
runs/detect/flow_process_chart_v1/weights/best.pt
```

---

### Model 2: MediaPipe Hands – Hand Motion Tracking
MediaPipe provides **21 landmarks per hand**, enabling:
- Precise hand position tracking
- Velocity estimation
- Finger-tip location detection

---

## Model Fusion Logic (YOLO + MediaPipe)
1. YOLO detects and stabilizes the board region
2. MediaPipe tracks hand landmarks per frame
3. Hand velocity determines motion vs stationary
4. Finger tips inside board determine intent
5. A rule-based state machine classifies Therbligs

---

## Therbligs Classification Rules

| Condition | Classification |
|--------|---------------|
| Stationary ≥ 2.0 s | Delay |
| ≥ 2 fingers inside board ≥ 0.25 s | Operation |
| 1 finger inside board ≥ 0.25 s | Hold |
| Default | Transport |

Priority Order:
1. Delay
2. Operation
3. Hold
4. Transport

---

## Data Export
Session results are automatically exported to:

```
Process_Data_Log.csv
```

**Schema**
- Session_ID
- Timestamp
- Hand
- Process
- Duration_Seconds

---

## Applications
- Automated Time & Motion Study
- Industrial Work Measurement
- Ergonomics Analysis
- Smart Manufacturing Systems

---

## Tech Stack
- Python
- YOLOv8 (Ultralytics)
- MediaPipe
- OpenCV
- PyTorch
- NumPy

---

## Author
Industrial Engineering × AI project for modernizing classical work measurement using computer vision.
