<div align="center">

# Kinetic Analytics V2.0

### AI-Based Real-Time Time and Motion Study System

*Integrating YOLOv8 Object Detection and MediaPipe Hand Tracking for Automated Industrial Work Measurement*

---

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2-61DAFB?logo=react)](https://react.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00)](https://ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-00897B)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Methodology](#3-methodology)
   - 3.1 [Work Zone Detection — YOLOv8](#31-work-zone-detection--yolov8)
   - 3.2 [Hand Motion Tracking — MediaPipe Hands](#32-hand-motion-tracking--mediapipe-hands)
   - 3.3 [Signal Conditioning](#33-signal-conditioning)
   - 3.4 [Process State Classification](#34-process-state-classification)
   - 3.5 [Data Logging](#35-data-logging)
4. [Technology Stack](#4-technology-stack)
5. [Repository Structure](#5-repository-structure)
6. [Prerequisites](#6-prerequisites)
7. [Installation & Setup](#7-installation--setup)
   - 7.1 [Clone the Repository](#71-clone-the-repository)
   - 7.2 [Backend Setup](#72-backend-setup)
   - 7.3 [Frontend Setup](#73-frontend-setup)
   - 7.4 [Model Training (Optional)](#74-model-training-optional)
8. [Configuration](#8-configuration)
9. [Running the Application](#9-running-the-application)
10. [Data Output](#10-data-output)
11. [Analytics Dashboard](#11-analytics-dashboard)
12. [Keyboard Shortcuts (Legacy Mode)](#12-keyboard-shortcuts-legacy-mode)
13. [Testing](#13-testing)
14. [Troubleshooting](#14-troubleshooting)
15. [Applications](#15-applications)

---

## 1. Abstract

Classical time and motion studies require trained industrial engineers to observe workers, operate stopwatches, and manually categorise each movement according to the Therblig classification system. This process is labour-intensive, prone to observer bias, and difficult to scale.

**Kinetic Analytics V2.0** addresses these limitations through a dual-model computer vision pipeline that operates in real time:

1. A custom-trained **YOLOv8** model continuously locates the designated work zone (board) in each video frame.
2. **MediaPipe Hands** extracts 21 skeletal landmarks per hand at up to 30 Hz.
3. A rule-based **state machine** fuses both data streams to classify each hand's activity as *Operation*, *Transport*, or *Delay* — the three fundamental Therblig categories.

All measurements are logged to a structured CSV file compatible with Power BI and Excel, while a **React-based web dashboard** streams live video overlays, real-time analytics charts, and derived ergonomic indices (path complexity, fatigue index, motion symmetry) via WebSocket at 30 Hz.

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        VIDEO SOURCE                         │
│              (MP4 file  or  Webcam index)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │  Raw frames (1280×720)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND  (Python / FastAPI)               │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  BoardDetector│    │  HandTracker │    │ StateMachine │  │
│  │  (YOLOv8)    │──▶│  (MediaPipe) │──▶│  (Rule-based)│  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         └───────────────────┴───────────────────┘           │
│                             │                               │
│               ┌─────────────┴──────────────┐               │
│               │                            │               │
│        ┌──────▼──────┐            ┌────────▼───────┐       │
│        │  DataLogger │            │   Broadcaster  │       │
│        │  (CSV/fsync)│            │  (WebSocket)   │       │
│        └─────────────┘            └────────┬───────┘       │
└─────────────────────────────────────────────┼───────────────┘
                                              │  JSON @ 30 Hz
                                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FRONTEND  (React / Vite)                  │
│                                                             │
│  ┌────────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │  CanvasOverlay │  │ DigitalTwin   │  │  Analytics    │  │
│  │  (Live Feed)   │  │  View         │  │  Dashboard    │  │
│  └────────────────┘  └───────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

The backend exposes three HTTP endpoints and one persistent WebSocket:

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Health check — returns status, client count, FPS |
| `/stats` | `GET` | Full session statistics JSON |
| `/reset` | `POST` | Reset all counters; broadcasts `RESET_CONFIRMED` event |
| `/ws` | `WebSocket` | Real-time 30 Hz data stream |

---

## 3. Methodology

### 3.1 Work Zone Detection — YOLOv8

A YOLOv8s model is fine-tuned on a custom Roboflow dataset of industrial boards and work surfaces. At inference time the model receives each resized frame (1280×720) and returns an axis-aligned bounding box `[x₁, y₁, x₂, y₂]` for the highest-confidence detection of class 0 (board).

**Parameters:**

| Parameter | Value |
|---|---|
| Confidence threshold | 0.25 |
| IoU threshold | 0.45 |
| Inference device | CUDA (GPU) if available; CPU fallback |
| Input resolution | 1280 × 720 px |

### 3.2 Hand Motion Tracking — MediaPipe Hands

MediaPipe Hands is initialised to track up to two hands simultaneously. For each detected hand it outputs 21 normalised 2D landmarks. The system computes:

- **Raw centre**: arithmetic mean of all landmark pixel coordinates.
- **Bounding box**: min/max landmark extents with a 20-pixel margin.
- **Velocity**: Euclidean displacement of the filtered centre position between consecutive frames, stored in a rolling deque of length `VELOCITY_SMOOTHING_FRAMES` (default 5).

### 3.3 Signal Conditioning

Two filters are applied sequentially before any kinematic calculation:

**Teleportation Filter**
Any single-frame displacement greater than 300 px is treated as a detection artefact (e.g., the model briefly mislabelling the opposite hand). The raw position is discarded and replaced with the last accepted position. The threshold was empirically selected to exceed the maximum realistic hand speed at 30 Hz while remaining sensitive to true rapid movements.

**Exponential Moving Average (EMA)**
Valid positions are smoothed with EMA (α = 0.7):

```
smoothed_t = α × raw_t + (1 − α) × smoothed_{t−1}
```

A higher α (closer to 1.0) tracks the signal more aggressively; a lower α increases lag but rejects more noise. The value 0.7 was selected to balance responsiveness and noise suppression for hand speeds observed in typical assembly tasks.

### 3.4 Process State Classification

A rule-based finite state machine classifies each hand independently into one of three states on every frame:

```
          ┌───────────────────────────────────────────┐
          │             TRANSPORT (default)            │
          │    Hand moving (velocity > threshold)      │
          └──────────┬─────────────────────┬──────────┘
                     │                     │
      stationary ≥   │                     │  ≥1 fingertip inside
  DELAY_INACTIVITY_  │                     │  board zone for ≥
      TIME (0.5 s)   │                     │  ZONE_STABILITY_TIME
                     ▼                     ▼         (0.15 s)
              ┌──────────┐         ┌──────────────┐
              │  DELAY   │         │  OPERATION   │
              └──────────┘         └──────────────┘
```

**State rules (in priority order):**

| Priority | Condition | Assigned State |
|---|---|---|
| 1 (highest) | Stationary for > `DELAY_INACTIVITY_TIME` (0.5 s) outside zone | **Delay** |
| 2 | ≥ 1 fingertip inside board zone for > `ZONE_STABILITY_TIME` (0.15 s) | **Operation** |
| 3 (default) | Neither above condition met | **Transport** |

> **Bug fix (v2.0.1):** The original condition `if state.stationary_start_time` silently failed when the video started at timestamp 0.0 (Python treats `0.0` as falsy). This has been corrected to `if state.stationary_start_time is not None`, ensuring proper Delay and Operation transitions from the first frame.

### 3.5 Data Logging

The `RobustDataLogger` writes structured rows to `Master_Raw_Data.csv` at a rate of approximately 15 Hz (every `LOG_INTERVAL = 0.066 s`). The file is opened in append mode so multiple sessions accumulate without overwriting prior data.

**Crash safety:** `os.fsync()` is called every 10 rows to guarantee data is committed to disk even if the process is terminated unexpectedly.

**Session markers** are inserted as comment lines:

```
# SESSION START: 2026-03-29 14:22:01 (Session ID: 20260329_142201)
...data rows...
# SESSION END:   2026-03-29 14:35:47
```

**CSV schema:**

| Column | Type | Description |
|---|---|---|
| `Row_Number` | int | Auto-incrementing row ID (resumes across sessions) |
| `Timestamp` | float | Video time in seconds (3 d.p.) |
| `Session_ID` | str | `YYYYMMDD_HHMMSS` session identifier |
| `Frame_ID` | int | Zero-based video frame counter |
| `Hand_Side` | str | `Left` or `Right` |
| `Current_State` | str | `Operation`, `Transport`, or `Delay` |
| `Velocity_Px` | float | Hand velocity in pixels per frame |
| `Fingers_In_Zone` | int | Count of fingertips inside board zone (0–5) |
| `In_Board_Zone_Bool` | str | `True` if any finger in zone |
| `Pos_X` | int | Filtered hand centre X coordinate (pixels) |
| `Pos_Y` | int | Filtered hand centre Y coordinate (pixels) |

---

## 4. Technology Stack

### Backend

| Technology | Version | Role |
|---|---|---|
| Python | ≥ 3.9 | Core language |
| FastAPI | ≥ 0.110 | REST API + WebSocket server |
| Uvicorn | ≥ 0.29 | ASGI server |
| Ultralytics YOLOv8 | ≥ 8.2 | Work zone detection |
| MediaPipe | ≥ 0.10 | Hand landmark tracking |
| OpenCV | ≥ 4.9 | Video I/O and JPEG encoding |
| PyTorch | ≥ 2.2 | GPU inference backend |
| NumPy | ≥ 1.26 | Numerical operations |
| python-dotenv | ≥ 1.0 | Environment variable management |

### Frontend

| Technology | Version | Role |
|---|---|---|
| React | 18.2 | UI framework |
| Vite | ≥ 5.0 | Build tool and dev server |
| Recharts | ≥ 3.7 | 6-chart analytics dashboard |
| Tailwind CSS | ≥ 3.4 | Utility-first styling |
| WebSocket API | Native | Real-time data streaming |

### External Tools

| Tool | Purpose |
|---|---|
| Roboflow | Dataset labelling and export for YOLOv8 training |
| Microsoft Power BI | Post-session CSV analytics (`.pbix` file included) |

---

## 5. Repository Structure

```
AI-Based-Time-Motion-Analysis/
│
├── backend/                        # Python FastAPI backend
│   ├── core/
│   │   ├── __init__.py             # Module exports
│   │   ├── config.py               # Centralised configuration dataclass
│   │   ├── detector.py             # YOLOv8 BoardDetector + VideoCapture
│   │   ├── tracker.py              # MediaPipe HandTracker (teleport filter + EMA)
│   │   ├── state_machine.py        # Therblig process state machine
│   │   └── logger.py               # Crash-safe CSV data logger
│   ├── services/
│   │   ├── __init__.py
│   │   └── broadcaster.py          # WebSocket connection manager + payload builder
│   ├── main.py                     # FastAPI application entry point
│   ├── requirements.txt            # Python dependencies
│   └── .env.example                # Environment variable template
│
├── frontend/                       # React + Vite frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── AnalyticsCommandCenter.jsx   # 6-chart analytics dashboard
│   │   │   ├── CanvasOverlay.jsx            # Live video feed with overlays
│   │   │   ├── DigitalTwinView.jsx          # Skeleton-only motion visualisation
│   │   │   ├── Header.jsx                   # Status bar (FPS, connection)
│   │   │   ├── Sidebar.jsx                  # Per-hand real-time metrics
│   │   │   └── TabManager.jsx              # Tab navigation
│   │   ├── context/
│   │   │   └── DashboardContext.jsx         # Ref-based chart data aggregation
│   │   ├── hooks/
│   │   │   └── useWebSocket.js              # Auto-reconnecting WebSocket hook
│   │   ├── App.jsx                          # Root component
│   │   ├── main.jsx                         # React entry point
│   │   └── index.css                        # Global styles
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── tests/                          # Backend unit tests (pytest)
│   ├── __init__.py
│   ├── test_config.py              # Configuration defaults
│   ├── test_broadcaster.py         # Payload builder correctness
│   ├── test_logger.py              # CSV logging and resume logic
│   └── test_state_machine.py       # State transition correctness
│
├── Therbligs_Detect.py             # Legacy standalone app (HUD, matplotlib)
├── Model_Training.py               # YOLOv8 training script
├── Master_Raw_Data.csv             # Accumulated session data output
├── WSD PowerBI.pbix                # Power BI dashboard template
├── .gitignore
└── README.md
```

---

## 6. Prerequisites

### Hardware

| Requirement | Minimum | Recommended |
|---|---|---|
| CPU | Quad-core 2.5 GHz | 8-core 3.5 GHz+ |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU inference) | NVIDIA GPU with CUDA 11.8+ (VRAM ≥ 4 GB) |
| Storage | 2 GB free | 5 GB free |
| Camera | Any USB webcam | 1080p 30 fps webcam or pre-recorded MP4 |

### Software

| Software | Version | Installation |
|---|---|---|
| Python | ≥ 3.9 (3.12 tested) | https://www.python.org/downloads/ |
| Node.js | ≥ 18 LTS | https://nodejs.org/ |
| npm | ≥ 9 | Bundled with Node.js |
| Git | Any recent | https://git-scm.com/ |
| CUDA Toolkit *(optional)* | 11.8 or 12.x | https://developer.nvidia.com/cuda-downloads |

---

## 7. Installation & Setup

### 7.1 Clone the Repository

```bash
git clone https://github.com/<your-username>/AI-Based-Time-Motion-Analysis.git
cd AI-Based-Time-Motion-Analysis
```

---

### 7.2 Backend Setup

#### Step 1 — Create and activate a virtual environment

**Windows (Command Prompt / PowerShell):**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 2 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

> **GPU acceleration (optional):** If you have a CUDA-capable GPU, replace the PyTorch installation with the CUDA-enabled build *before* installing requirements:
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install -r backend/requirements.txt
> ```

#### Step 3 — Configure environment variables

Copy the example file and fill in your paths:

```bash
cp backend/.env.example backend/.env
```

Then open `backend/.env` in a text editor and set:

```dotenv
# Full path to your trained YOLOv8 weights file
MODEL_PATH=C:\Users\you\runs\detect\board_detector\weights\best.pt

# Full path to your video file, OR an integer for a live webcam
VIDEO_PATH=C:\Users\you\Videos\WSD_Video.mp4
# VIDEO_PATH=0    # to use the first connected webcam
```

> **Important:** The `backend/.env` file is listed in `.gitignore` and will **not** be committed. Never hard-code your personal paths into the source files.

---

### 7.3 Frontend Setup

#### Step 1 — Install Node.js dependencies

```bash
cd frontend
npm install
```

#### Step 2 — (Optional) Verify the WebSocket URL

The frontend connects to `ws://localhost:8080/ws` by default (see `frontend/src/hooks/useWebSocket.js`). If you run the backend on a different host or port, update that URL.

---

### 7.4 Model Training (Optional)

If you need to train or re-train the YOLOv8 board-detection model on your own dataset:

1. Label your images using [Roboflow](https://roboflow.com) and export as **YOLOv8 format**.
2. Update the `dataset_path` variable in `Model_Training.py` to point to the exported `data.yaml`.
3. Run the training script:

```bash
python Model_Training.py
```

The trained weights will be saved to:
```
runs/detect/My_Custom_Training/board_detector_gpu/weights/best.pt
```

Set `MODEL_PATH` in `backend/.env` to this path.

**Training configuration (defaults in `Model_Training.py`):**

| Parameter | Value |
|---|---|
| Base model | `yolov8s.pt` |
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Device | GPU 0 |

---

## 8. Configuration

All backend tuneable parameters are centralised in `backend/core/config.py`. Values can be changed directly in that file without touching any other module.

| Parameter | Default | Description |
|---|---|---|
| `FRAME_WIDTH` | `1280` | Processing frame width (px) |
| `FRAME_HEIGHT` | `720` | Processing frame height (px) |
| `YOLO_CONF_THRESHOLD` | `0.25` | Minimum YOLO detection confidence |
| `YOLO_IOU_THRESHOLD` | `0.45` | YOLO non-maximum suppression IoU |
| `MP_MIN_DETECTION_CONF` | `0.5` | MediaPipe initial detection confidence |
| `MP_MIN_TRACKING_CONF` | `0.5` | MediaPipe per-frame tracking confidence |
| `VELOCITY_THRESHOLD` | `3.0` | px/frame threshold for motion detection |
| `VELOCITY_SMOOTHING_FRAMES` | `5` | Rolling window size for velocity EMA |
| `ZONE_STABILITY_TIME` | `0.15` | Seconds in zone before classifying as Operation |
| `DELAY_INACTIVITY_TIME` | `0.5` | Seconds stationary before classifying as Delay |
| `LOG_INTERVAL` | `0.066` | Minimum seconds between CSV log rows (~15 Hz) |
| `WS_BROADCAST_INTERVAL` | `0.033` | Minimum seconds between WebSocket broadcasts (~30 Hz) |
| `IDLE_ALERT_THRESHOLD` | `5.0` | Seconds of Delay before an alert is raised |

---

## 9. Running the Application

Both services must run simultaneously. Open two terminal windows (both with the virtual environment activated for the backend terminal).

### Terminal 1 — Start the Backend

```bash
cd backend
python main.py
```

Expected output:
```
============================================================
  KINETIC ANALYTICS V2.0 - Backend Server
  WebSocket endpoint: ws://127.0.0.1:8080/ws
============================================================
[VIDEO] Opened source at 25.0 FPS
[DETECTOR] Loading YOLO on cuda...
[LOGGER] Created new file: .../Master_Raw_Data.csv
[LOOP] Processing at 25.0 FPS
```

### Terminal 2 — Start the Frontend

```bash
cd frontend
npm run dev
```

Expected output:
```
  VITE v5.x.x  ready in 312 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.x.x:5173/
```

### Access the Dashboard

Open your browser and navigate to:
```
http://localhost:5173
```

The status indicator in the header will turn **green** once the frontend successfully connects to the backend WebSocket.

---

## 10. Data Output

Session data is automatically appended to `Master_Raw_Data.csv` in the project root. The file accumulates across sessions using sequential row numbering.

### Example rows

```csv
Row_Number,Timestamp,Session_ID,Frame_ID,Hand_Side,Current_State,Velocity_Px,Fingers_In_Zone,In_Board_Zone_Bool,Pos_X,Pos_Y
1,0.040,20260329_142201,1,Right,Transport,6.2,0,False,623,388
2,0.040,20260329_142201,1,Left,Transport,4.1,0,False,341,402
3,0.106,20260329_142201,4,Right,Operation,1.3,2,True,512,310
```

### Power BI Integration

The included `WSD PowerBI.pbix` file connects directly to `Master_Raw_Data.csv`. After opening the dashboard in Power BI Desktop, click **Refresh** to load the latest session data.

---

## 11. Analytics Dashboard

The React frontend provides three views, switchable via the tab bar:

### Live Operations

Real-time video feed (1280 × 720) with the following overlays rendered at 60 fps via `requestAnimationFrame`:

- **Board zone**: Green glowing border with corner bracket indicators.
- **Hand skeletons**: Cyan (right hand) and magenta (left hand) landmark connections with 21 keypoints.
- **State labels**: Per-hand state badge with velocity readout.
- **Sidebar**: Per-hand time breakdown (Operation / Transport / Delay) as animated progress bars, updated at 4 Hz.

### Digital Twin

Skeleton-only rendering on a dark grid — useful for reviewing motion patterns without the video background. Identical landmark data to the live view.

### Analytics Suite (6 charts)

| Chart | Metric | Description |
|---|---|---|
| Path Complexity | RMS deviation from straight-line path | Values > 1.0 indicate inefficient trajectories |
| Velocity Profile | px/sec over time (both hands) | Identifies high-effort transport phases |
| Motion Symmetry | Left vs Right velocity scatter | Diagonal alignment = balanced bilateral work |
| Time Breakdown | Operation / Transport / Delay (donut) | Session-level Therblig distribution |
| Fatigue Index | Composite trend over time | Rising index indicates sustained high effort |
| Motion Path Trace | 2D spaghetti diagram (300 points/hand) | Spatial coverage and workspace clustering |

The **RESET** button clears all chart history and zeroes backend counters simultaneously via `POST /reset`.

---

## 12. Keyboard Shortcuts (Legacy Mode)

The legacy standalone application (`Therbligs_Detect.py`) supports the following keyboard controls when run directly:

| Key | Action |
|---|---|
| `H` | Toggle help overlay |
| `T` | Toggle motion trails |
| `S` | Toggle scan lines effect |
| `P` | Pause / Resume |
| `R` | Reset all counters |
| `Q` | Quit application |

Run the legacy app with:
```bash
python Therbligs_Detect.py
```
> Note: Update `MODEL_PATH` and `VIDEO_PATH` at the top of the script before running.

---

## 13. Testing

The test suite covers the backend logic modules without requiring a GPU, video file, or YOLO model.

### Run all tests

```bash
# From the project root (virtual environment must be active)
python -m pytest tests/ -v
```

### Expected output

```
tests/test_broadcaster.py::test_payload_keys                              PASSED
tests/test_broadcaster.py::test_payload_board_zone_none                   PASSED
...
tests/test_state_machine.py::TestStateMachineTransitions::test_finger_in_zone_long_enough_becomes_operation  PASSED
tests/test_state_machine.py::TestStateMachineTransitions::test_stationary_long_enough_becomes_delay          PASSED
...
======================== 31 passed in 6.20s ========================
```

### Test coverage by module

| Test file | Module tested | Scenarios |
|---|---|---|
| `test_config.py` | `core/config.py` | Default values, singleton, rate ordering |
| `test_broadcaster.py` | `services/broadcaster.py` | Payload keys, board zone, hand visibility, rounding |
| `test_logger.py` | `core/logger.py` | File creation, headers, row writing, throttle, resume |
| `test_state_machine.py` | `core/state_machine.py` | Zone containment, Transport/Delay/Operation transitions |

---

## 14. Troubleshooting

### Backend will not start

**`MODEL_PATH is not set`**
→ Create `backend/.env` from `backend/.env.example` and set `MODEL_PATH` to your `best.pt` file path.

**`Could not open video source`**
→ Verify `VIDEO_PATH` in `backend/.env` points to a valid file. Use `VIDEO_PATH=0` to fall back to the first webcam.

**YOLO model not found**
→ Check the path has no trailing spaces and uses the correct directory separators for your OS. Windows paths with backslashes must either be double-escaped (`\\`) or use forward slashes.

### Frontend shows "Disconnected"

→ Confirm the backend is running and listening on port 8080.
→ Check for firewall rules blocking local connections on port 8080.
→ Verify `useWebSocket.js` uses `ws://localhost:8080/ws`.

### Slow performance / low FPS

→ Ensure PyTorch is using the GPU: check startup log for `[DETECTOR] Loading YOLO on cuda...` (not `cpu`).
→ Reduce `FRAME_WIDTH` / `FRAME_HEIGHT` in `config.py` (e.g., `960 × 540`).
→ Lower JPEG quality in `main.py` (currently 50 — reduce to 30 for lower bandwidth).

### CSV file not created

→ Confirm the Python process has write permission to the project root directory.
→ Check the backend terminal for `[LOGGER]` messages indicating the file path in use.

### MediaPipe detects wrong hand as "Left" or "Right"

→ This is expected when using a mirrored camera feed. MediaPipe's handedness labels are from the subject's perspective. For analysis consistency, ensure the camera orientation matches your convention or post-process the labels.

---

## 15. Applications

- Automated time and motion studies in manufacturing and assembly environments
- Industrial ergonomics and fatigue assessment
- Lean manufacturing process optimisation (identifying non-value-added Delay time)
- Human factors research and workload analysis
- Smart factory digital-twin integration
- Rehabilitation monitoring and occupational therapy assessment

---

<div align="center">

*Kinetic Analytics V2.0 — Industrial Engineering × Computer Vision*

</div>
