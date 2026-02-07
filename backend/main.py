"""
Kinetic Analytics V2.0 - FastAPI Backend
Real-time hand tracking with WebSocket broadcasting
"""

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
import cv2
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core import (
    config, HandTracker, ProcessStateMachine, 
    BoardDetector, VideoCapture, RobustDataLogger
)
from services import ConnectionManager, create_broadcast_payload


# ============================================================================
# CONFIGURATION
# ============================================================================

# Use raw strings (r"...") to handle Windows backslashes correctly
MODEL_PATH = r"C:\Users\vitta\OneDrive\Desktop\Python 3.10\runs\detect\My_Custom_Training\board_detector_gpu\weights\best.pt"
VIDEO_PATH = r"C:\Users\vitta\OneDrive\Desktop\WSD Video.mp4"

# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    """Application state container."""
    def __init__(self):
        self.is_running = False
        self.video: VideoCapture = None
        self.detector: BoardDetector = None
        self.tracker: HandTracker = None
        self.logic: ProcessStateMachine = None
        self.logger: RobustDataLogger = None
        self.manager = ConnectionManager()
        
        self.counters: Dict[str, float] = {
            "Left_Operation": 0.0, "Left_Transport": 0.0, "Left_Delay": 0.0,
            "Right_Operation": 0.0, "Right_Transport": 0.0, "Right_Delay": 0.0
        }
        self.fps_history = deque(maxlen=30)
        self.current_fps = 0.0
        self.alerts = []

app_state = AppState()


# ============================================================================
# PROCESSING LOOP
# ============================================================================

async def processing_loop():
    """Main video processing loop running in background."""
    print("[LOOP] Starting processing loop...")
    
    try:
        app_state.video = VideoCapture(VIDEO_PATH)
        app_state.detector = BoardDetector(MODEL_PATH)
        app_state.tracker = HandTracker()
        app_state.logic = ProcessStateMachine(app_state.tracker.hand_states)
        app_state.logger = RobustDataLogger()
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return
    
    frame_id = 0
    last_frame_time = time.time()
    last_broadcast_time = 0.0
    
    print(f"[LOOP] Processing at {app_state.video.fps:.1f} FPS")
    app_state.is_running = True
    
    try:
        while app_state.is_running and app_state.video.is_opened:
            success, frame = app_state.video.read()
            if not success:
                print("[INFO] End of video stream, restarting...")
                app_state.video.release()
                app_state.video = VideoCapture(VIDEO_PATH)
                frame_id = 0
                continue
            
            frame_id += 1
            current_video_time = frame_id * app_state.video.frame_duration
            
            # === DETECTION ===
            board_box = app_state.detector.detect(frame)
            detected_hands = app_state.tracker.process_frame(frame, frame_id)
            
            # === STATE MACHINE ===
            hands_info = {}
            for lbl, data in detected_hands.items():
                process = app_state.logic.update(
                    lbl, data['landmarks_raw'], board_box, 
                    current_video_time, config.FRAME_WIDTH, config.FRAME_HEIGHT
                )
                app_state.logger.update_state_timers(current_video_time, lbl, process)
                app_state.counters[f"{lbl}_{process}"] += app_state.video.frame_duration
                
                hands_info[lbl] = {
                    'velocity': app_state.tracker.calculate_velocity(lbl),
                    'fingers': app_state.tracker.hand_states[lbl].finger_count_in_board,
                    'state': process,
                    'pos': data['center'],
                    'landmarks': data['landmarks']  # Already JSON-serializable
                }
            
            # === LOGGING ===
            app_state.logger.log_frame(frame_id, current_video_time, hands_info)
            
            # === FPS CALCULATION ===
            now = time.time()
            frame_time = now - last_frame_time
            last_frame_time = now
            if frame_time > 0:
                app_state.fps_history.append(1.0 / frame_time)
                app_state.current_fps = float(np.mean(list(app_state.fps_history)))
            
            # === BROADCAST (Rate limited) ===
            if (now - last_broadcast_time) >= config.WS_BROADCAST_INTERVAL:
                if app_state.manager.client_count > 0:
                    # Encode frame to JPEG base64
                    try:
                        # Resize for bandwidth optimization if needed, using 50% quality for speed
                        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                        img_str = base64.b64encode(buffer).decode('utf-8')
                    except Exception as e:
                        print(f"[ERROR] Encoding frame: {e}")
                        img_str = None

                    payload = create_broadcast_payload(
                        frame_id=frame_id,
                        video_time=current_video_time,
                        fps=app_state.current_fps,
                        board_box=board_box,
                        hands_info=hands_info,
                        counters=app_state.counters,
                        alerts=app_state.alerts,
                        image=img_str
                    )
                    await app_state.manager.broadcast(payload)
                    app_state.alerts = []  # Clear after broadcast
                last_broadcast_time = now
            
            # Yield control to allow other async tasks
            await asyncio.sleep(0.001)
    
    except asyncio.CancelledError:
        print("[LOOP] Processing loop cancelled")
    except Exception as e:
        print(f"[ERROR] Processing loop error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[CLEANUP] Releasing resources...")
        if app_state.video:
            app_state.video.release()
        if app_state.tracker:
            app_state.tracker.close()
        if app_state.logger:
            app_state.logger.close()
            print(f"[CLEANUP] CSV logger closed. Total rows logged: {app_state.logger.current_row_index}")
        app_state.is_running = False


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

processing_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global processing_task
    print("=" * 60)
    print("  KINETIC ANALYTICS V2.0 - Backend Server")
    print("  WebSocket endpoint: ws://127.0.0.1:8080/ws")
    print("=" * 60)
    
    # Start processing loop in background
    processing_task = asyncio.create_task(processing_loop())
    
    yield
    
    # Shutdown
    print("[SHUTDOWN] Stopping processing loop...")
    app_state.is_running = False
    if processing_task:
        processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            pass
    print("[SHUTDOWN] Server stopped.")

app = FastAPI(
    title="Kinetic Analytics V2.0",
    description="Real-time hand tracking WebSocket API",
    version="2.0.0",
    lifespan=lifespan
)

# ============================================================================
# CORS FIX (CRITICAL FOR FRONTEND CONNECTION)
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <--- FIXED: Allow ALL origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "clients": app_state.manager.client_count,
        "fps": round(app_state.current_fps, 1)
    }


@app.get("/stats")
async def get_stats():
    """Get current statistics."""
    return {
        "is_running": app_state.is_running,
        "fps": round(app_state.current_fps, 1),
        "counters": app_state.counters,
        "clients": app_state.manager.client_count
    }


@app.post("/reset")
async def reset_counters():
    """Reset all counters and session variables."""
    # Zero out all counters
    app_state.counters = {k: 0.0 for k in app_state.counters}
    
    # Reset FPS history
    app_state.fps_history.clear()
    app_state.current_fps = 0.0
    
    # Clear alerts
    app_state.alerts = []
    
    # Broadcast RESET_CONFIRMED event so frontend clears its local arrays
    if app_state.manager.client_count > 0:
        reset_payload = {
            "event": "RESET_CONFIRMED",
            "counters": app_state.counters,
            "message": "Session reset complete"
        }
        await app_state.manager.broadcast(reset_payload)
    
    print("[RESET] All session variables reset, RESET_CONFIRMED broadcasted")
    return {"status": "reset", "event": "RESET_CONFIRMED"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming."""
    await app_state.manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Handle client commands if needed
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await app_state.manager.disconnect(websocket)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    # Host 0.0.0.0 is crucial for local network visibility
    uvicorn.run(app, host="0.0.0.0", port=8080)