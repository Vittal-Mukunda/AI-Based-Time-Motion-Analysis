# AI-Based Time & Motion Analysis using YOLOv8 and MediaPipe

## 🎯 Overview

This project implements an **automated industrial time and motion study system** with a stunning **HUD** interface. It integrates two AI models into a single real-time pipeline:

1. **YOLOv8 (Object Detection)** – Detects and tracks the work zone (board).
2. **MediaPipe Hands (Pose Estimation)** – Tracks detailed hand landmarks and motion dynamics.

By fusing these models, the system classifies **Therbligs** (Operation, Transport, Delay) and computes **cycle-time metrics** automatically, eliminating manual stopwatch-based studies.

---

## ✨ Key Features

### Core AI Capabilities
-  Real-time hand tracking (Left & Right hands)
-  Custom-trained YOLOv8 model for work-zone detection
-  Velocity-based motion analysis
-  Finger-count–based Therbligs classification
-  Robust state-machine decision logic
-  Automatic CSV export (Power BI / Excel compatible)
-  GPU acceleration (CUDA) when available

###  Visual Enhancements (NEW!)
- **Pulsing Board Zone** - Animated glowing borders with cyan/magenta gradient
- **Motion Trails** - Tron-like fading light trails following hand movement
- **Glassmorphism HUD Panels** - Floating semi-transparent status displays
- **Efficiency Gauge** - Real-time productivity percentage display
- **Scan Lines** - Retro-futuristic holographic overlay
- **Premium Dashboard** - Dark theme analytics with gradient charts

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `H` | Toggle help overlay |
| `T` | Toggle motion trails |
| `S` | Toggle scan lines |
| `P` | Pause / Resume |
| `R` | Reset all counters |
| `Q` | Quit application |

---

##  Dual-Model System Architecture

### Model 1: YOLOv8 – Work-Zone Detection (Environmental Observer)
YOLOv8 is trained on a custom Roboflow dataset to detect the **board/work area**.

### Model 2: MediaPipe Hands – Hand Motion Tracking (Dexterity Tracker)
MediaPipe provides **21 landmarks per hand**, enabling:
- Precise hand position tracking
- Velocity estimation
- Finger-tip location detection

---

##  Model Fusion Logic

1. YOLO detects and stabilizes the board region
2. MediaPipe tracks hand landmarks per frame
3. Hand velocity determines motion vs stationary
4. Finger tips inside board determine intent
5. A rule-based state machine classifies Therbligs

---

##  Therbligs Classification Rules

| Condition | Classification | Color |
|-----------|---------------|-------|
| Stationary ≥ 1.0s outside zone | **Delay** | 🔴 Red |
| ≥ 1 finger inside board ≥ 0.1s | **Operation** | 🟢 Green |
| Hand moving (velocity > threshold) | **Transport** | 🟡 Yellow/Cyan |

---

##  Analytics Dashboard

The system provides a real-time matplotlib dashboard displaying:

- ** Operation Balance** - Left vs Right hand usage comparison
- ** Effort Distribution** - Operation/Transport/Delay breakdown
- ** Duration Chart** - Time spent in each state per hand
- ** Motion Trajectory** - 2D path visualization

---

##  Data Export

Session results are automatically exported to:
```
Master_Raw_Data.csv
```

**Schema:**
- Row Number, Timestamp, Session_ID, Frame_ID
- Hand_Side, Current_State, Velocity_Px
- Fingers_In_Zone, In_Board_Zone_Bool, Pos_X, Pos_Y

---

##  Usage

```python
# Run the main application
python Therbligs_Detect.py
```

Ensure you have updated the paths in the script:
```python
MODEL_PATH = r"path/to/your/best.pt"
VIDEO_PATH = r"path/to/your/video.mp4"
```

---

##  Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core language |
| YOLOv8 (Ultralytics) | Object detection |
| MediaPipe | Hand tracking |
| OpenCV | Video processing & HUD |
| PyTorch | GPU acceleration |
| NumPy | Mathematical operations |
| Matplotlib | Analytics dashboard |

---

##  Project Structure

```
AI-Based-Time-Motion-Analysis/
├── Therbligs_Detect.py     # Main application
├── hud_effects.py          # Visual effects module (NEW!)
├── Model_Training.py       # YOLO training script
├── Master_Raw_Data.csv     # Output data file
├── README.md               # This file
└── WSD PowerBI.pbix        # Power BI dashboard
```

---

## 🎯 Applications

- ✅ Automated Time & Motion Study
- ✅ Industrial Work Measurement
- ✅ Ergonomics Analysis
- ✅ Smart Manufacturing Systems
- ✅ Lean Manufacturing Optimization

---

## 👤 Author

Industrial Engineering × AI project for modernizing classical work measurement using computer vision.

---

*Featuring the "Dual-Brain" architecture with Iron Man-inspired visual feedback system.*
