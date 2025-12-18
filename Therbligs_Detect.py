import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from multiprocessing import freeze_support
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import torch 
import csv # Added for export
import os  # Added to check file existence
from datetime import datetime # Added for timestamping

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Central configuration for all detection parameters"""
    # Camera
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    
    # YOLO
    YOLO_CONF_THRESHOLD: float = 0.15  
    YOLO_IOU_THRESHOLD: float = 0.45   
    
    # MediaPipe
    MP_MIN_DETECTION_CONF: float = 0.6
    MP_MIN_TRACKING_CONF: float = 0.6
    MP_MAX_HANDS: int = 2
    
    # Hand Velocity
    VELOCITY_THRESHOLD: float = 5.0   # Pixels per frame to define "Movement"
    VELOCITY_SMOOTHING_FRAMES: int = 3 
    
    # Zone & Overlap
    OVERLAP_THRESHOLD: float = 0.15 
    HAND_BOX_PADDING: int = 20
    
    # Board Tracking
    BOARD_CONFIRM_THRESHOLD: float = 6.0 
    BOARD_POSITION_SMOOTHING: int = 5     
    BOARD_CONFIDENCE_DECAY: float = 0.02  
    
    # --- NEW STRICT RULES CONSTANTS ---
    # Stability: Time required inside board with correct finger count
    ZONE_STABILITY_TIME: float = 0.25      
    
    # Delay: Time required stationary to trigger DELAY
    DELAY_INACTIVITY_TIME: float = 2.0     
        
    # UI
    BUTTON_RECT: List[int] = field(default_factory=lambda: [20, 20, 150, 50])
    WINDOW_NAME: str = 'Finger-Based Process Detection'

config = Config()

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class HandState:
    """Tracks state history and timers"""
    label: str  
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=config.VELOCITY_SMOOTHING_FRAMES))
    
    # Process State
    current_process: str = "Transport"
    
    # Timers
    stationary_start_time: Optional[float] = None 
    board_entry_time: Optional[float] = None 
    
    # Logic Tracking
    target_state_candidate: Optional[str] = None # "Hold" or "Operation"
    finger_count_in_board: int = 0
    
    # Metadata
    last_seen_frame: int = 0
    confidence: float = 1.0

@dataclass
class BoardTracker:
    detection_history: deque = field(default_factory=lambda: deque(maxlen=config.BOARD_POSITION_SMOOTHING))
    confirmed_box: Optional[List[int]] = None
    confidence_score: float = 0.0 
    last_update_frame: int = 0
    is_confirmed: bool = False
    
    def update(self, box: Optional[List[int]], confidence: float, frame_id: int):
        if box is not None and confidence > 0:
            self.confidence_score += confidence
            self.detection_history.append(box)
            self.last_update_frame = frame_id
            
            if self.confidence_score >= config.BOARD_CONFIRM_THRESHOLD and not self.is_confirmed:
                self.is_confirmed = True
                self.confirmed_box = self._get_smoothed_box()
            elif self.is_confirmed:
                self.confirmed_box = self._get_smoothed_box()
        else:
            frames_since_detection = frame_id - self.last_update_frame
            if frames_since_detection > 0:
                decay = config.BOARD_CONFIDENCE_DECAY * frames_since_detection
                self.confidence_score = max(0.0, self.confidence_score - decay)
            if self.confidence_score < config.BOARD_CONFIRM_THRESHOLD * 0.3:
                self.is_confirmed = False
    
    def _get_smoothed_box(self) -> Optional[List[int]]:
        if not self.detection_history: return self.confirmed_box
        boxes = np.array(list(self.detection_history))
        return boxes.mean(axis=0).astype(int).tolist()
    
    def get_box(self) -> Optional[List[int]]:
        return self.confirmed_box if self.is_confirmed else self._get_smoothed_box()

# ============================================================================
# GEOMETRY & FINGER COUNTING
# ============================================================================

class GeometryEngine:
    @staticmethod
    def count_fingers_in_box(landmarks, box: List[int], img_w: int, img_h: int) -> int:
        """
        Counts how many finger TIPS are strictly inside the bounding box.
        Landmark indices: 4 (Thumb), 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
        """
        if not box: return 0
        
        tips_indices = [4, 8, 12, 16, 20]
        count = 0
        
        x_min, y_min, x_max, y_max = box
        
        for idx in tips_indices:
            lm = landmarks.landmark[idx]
            px, py = lm.x * img_w, lm.y * img_h
            
            if x_min <= px <= x_max and y_min <= py <= y_max:
                count += 1
                
        return count

# ============================================================================
# HAND TRACKER
# ============================================================================

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # Note: MediaPipe Python usually runs on CPU on Windows
        self.hands_model = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MP_MAX_HANDS,
            min_detection_confidence=config.MP_MIN_DETECTION_CONF,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONF
        )
        self.hand_states: Dict[str, HandState] = {
            "Left": HandState(label="Left"),
            "Right": HandState(label="Right")
        }
    
    def get_hand_bbox(self, landmarks, frame_shape) -> List[int]:
        h, w = frame_shape[:2]
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]
        pad = config.HAND_BOX_PADDING
        return [max(0, int(min(x_coords)) - pad), max(0, int(min(y_coords)) - pad),
                min(w, int(max(x_coords)) + pad), min(h, int(max(y_coords)) + pad)]
    
    def calculate_smoothed_velocity(self, hand_label: str) -> float:
        state = self.hand_states[hand_label]
        if len(state.velocity_history) < 2: return 0.0
        return np.mean(list(state.velocity_history))
    
    def detect_hands(self, frame: np.ndarray, frame_id: int) -> Dict[str, Tuple[List[int], any]]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands_model.process(frame_rgb)
        
        detected_hands = {}
        hands_seen = set()
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_info = results.multi_handedness[hand_idx].classification[0]
                label = hand_info.label
                hands_seen.add(label)
                state = self.hand_states[label]
                state.last_seen_frame = frame_id
                state.confidence = hand_info.score
                
                hand_box = self.get_hand_bbox(hand_landmarks, frame.shape)
                
                # Update position/velocity
                wrist = hand_landmarks.landmark[0]
                center = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
                state.position_history.append(center)
                
                if len(state.position_history) >= 2:
                    prev = state.position_history[-2]
                    vel = math.hypot(center[0]-prev[0], center[1]-prev[1])
                    state.velocity_history.append(vel)
                
                detected_hands[label] = (hand_box, hand_landmarks)
        
        # Cleanup
        for label in ["Left", "Right"]:
            if label not in hands_seen and frame_id - self.hand_states[label].last_seen_frame > 5:
                self.hand_states[label].position_history.clear()
                self.hand_states[label].velocity_history.clear()
                self.hand_states[label].board_entry_time = None
                self.hand_states[label].stationary_start_time = None
                
        return detected_hands
    
    def close(self):
        self.hands_model.close()

# ============================================================================
# NEW STATE MACHINE (STRICT RULES)
# ============================================================================

class ProcessStateMachine:
    def __init__(self, hand_tracker: HandTracker):
        self.hand_tracker = hand_tracker
        self.geometry = GeometryEngine()
    
    def update_hand_state(self, hand_label: str, hand_landmarks, 
                          board_box: Optional[List[int]], current_time: float, 
                          img_w: int, img_h: int) -> str:
        
        state = self.hand_tracker.hand_states[hand_label]
        
        # 1. MOVEMENT ANALYSIS
        velocity = self.hand_tracker.calculate_smoothed_velocity(hand_label)
        is_moving = velocity > config.VELOCITY_THRESHOLD
        
        # Delay Timer Logic
        if is_moving:
            state.stationary_start_time = None
        elif state.stationary_start_time is None:
            state.stationary_start_time = current_time
            
        # 2. FINGER COUNTING IN BOARD
        # Only relevant if board exists
        fingers_in_board = 0
        if board_box:
            fingers_in_board = self.geometry.count_fingers_in_box(
                hand_landmarks, board_box, img_w, img_h
            )
        state.finger_count_in_board = fingers_in_board
        
        # 3. IDENTIFY POTENTIAL STATE (CANDIDATE)
        # Based *strictly* on finger count inside board
        candidate = None
        
        # --- MODIFIED: 1 finger for Hold, 2+ for Op ---
        if fingers_in_board >= 2:
            candidate = "Operation"
        elif fingers_in_board == 1:
            candidate = "Hold"
        
        # 4. ZONE TIMER LOGIC
        # If candidate matches what we are waiting for, check timer.
        # If candidate changes, reset timer.
        if candidate is not None:
            if state.target_state_candidate != candidate:
                # New potential state detected, start timer
                state.target_state_candidate = candidate
                state.board_entry_time = current_time
        else:
            # Not in a valid configuration (e.g., 0 or 1 finger, or outside board)
            state.target_state_candidate = None
            state.board_entry_time = None
            
        # 5. PRIORITY RESOLUTION (The Decision Tree)
        
        # Priority 1: DELAY (Must be stationary for >= 2.0s)
        if state.stationary_start_time and (current_time - state.stationary_start_time >= config.DELAY_INACTIVITY_TIME):
            new_process = "Delay"
            
        # Priority 2 & 3: OPERATION or HOLD (Must be in zone for >= 0.25s)
        elif state.board_entry_time and (current_time - state.board_entry_time >= config.ZONE_STABILITY_TIME):
            # If timer passed, confirm the candidate
            new_process = state.target_state_candidate
            
        # Priority 4: TRANSPORT (Default / Fallback)
        else:
            new_process = "Transport"
            
        # Final State Assignment
        state.current_process = new_process
        return new_process

# ============================================================================
# MAIN SYSTEM
# ============================================================================

class DualBrainDetectionSystem:
    def __init__(self, model_path: str):
        print("INITIALIZING...")
        
        # --- GPU SETUP ---
        # Check if CUDA is available and assign device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Hardware Acceleration: Using {self.device.upper()}")
        
        # Load YOLO model to the specific device
        self.yolo_model = YOLO(model_path).to(self.device)
        
        self.hand_tracker = HandTracker()
        self.board_tracker = BoardTracker()
        self.state_machine = ProcessStateMachine(self.hand_tracker)
        
        # NOTE: UPDATE THIS PATH TO YOUR VIDEO
        self.cap = cv2.VideoCapture(r"C:\Users\vitta\Downloads\Test_1.mp4")
        self.cap.set(3, config.FRAME_WIDTH)
        self.cap.set(4, config.FRAME_HEIGHT)
        
        self.app_state = {"timer_running": False, "start_time": 0.0, "pause_time": 0.0, "paused": False}
        self.process_counters = {f"{h}_{p}": 0.0 for h in ["Left", "Right"] for p in ["Operation", "Transport", "Hold", "Delay"]}
        
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_NAME, 1280, 720)
        cv2.setMouseCallback(config.WINDOW_NAME, self._on_mouse_click, self.app_state)
        print("SYSTEM READY")

    def _on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bx, by, bw, bh = config.BUTTON_RECT
            if bx < x < bx + bw and by < y < by + bh:
                if not param["timer_running"] and not param["paused"]:
                    param["timer_running"] = True
                    param["start_time"] = time.time()

    def _detect_board(self, frame: np.ndarray) -> Optional[List[int]]:
        # YOLO inference using the device set in __init__
        results = self.yolo_model.predict(frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False, device=self.device)
        board_box, board_conf = None, 0.0
        
        if results and results[0].boxes:
            for i, cls in enumerate(results[0].boxes.cls):
                if self.yolo_model.names[int(cls)] == 'board':
                    conf = float(results[0].boxes.conf[i])
                    if conf > board_conf:
                        board_conf = conf
                        # Transfer to CPU for numpy conversion
                        board_box = results[0].boxes.xyxy[i].cpu().numpy().astype(int).tolist()
        
        self.board_tracker.update(board_box, board_conf, self.frame_count)
        return self.board_tracker.get_box()

    def _export_data(self):
        """Exports data to CSV in a PowerBI friendly format (Append Mode)"""
        filename = "Process_Data_Log.csv"
        file_exists = os.path.isfile(filename)
        
        # Generate a unique Session ID for this run
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        data_rows = []
        
        # Transform the counters dictionary into a Tidy Data list
        # From: {"Left_Operation": 5.0, ...}
        # To: [{"Hand": "Left", "Process": "Operation", "Duration": 5.0}, ...]
        for key, duration in self.process_counters.items():
            if duration > 0.0:
                hand, process = key.split('_')
                data_rows.append({
                    "Session_ID": session_id,
                    "Timestamp": timestamp,
                    "Hand": hand,
                    "Process": process,
                    "Duration_Seconds": round(duration, 2)
                })
        
        if data_rows:
            try:
                with open(filename, mode='a', newline='') as f:
                    headers = ["Session_ID", "Timestamp", "Hand", "Process", "Duration_Seconds"]
                    writer = csv.DictWriter(f, fieldnames=headers)
                    
                    if not file_exists:
                        writer.writeheader() # Write headers only if file is new
                        
                    writer.writerows(data_rows)
                print(f"SUCCESS: Data exported to {filename}")
            except Exception as e:
                print(f"ERROR: Could not export data. Is the Excel file open? {e}")
        else:
            print("No data recorded to export.")

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: break
            
            self.frame_count += 1
            curr_time = time.time()
            dt = curr_time - self.last_frame_time
            self.last_frame_time = curr_time
            
            elapsed = 0.0
            if self.app_state["timer_running"]: elapsed = curr_time - self.app_state["start_time"]
            elif self.app_state["paused"]: elapsed = self.app_state["pause_time"]
            
            # Detect
            board_box = self._detect_board(frame)
            hands = self.hand_tracker.detect_hands(frame, self.frame_count)
            
            # Update Logic
            for lbl, (hbox, landmarks) in hands.items():
                self.state_machine.update_hand_state(
                    lbl, landmarks, board_box, curr_time, frame.shape[1], frame.shape[0]
                )
                
                # Counters
                if self.app_state["timer_running"]:
                    state = self.hand_tracker.hand_states[lbl]
                    self.process_counters[f"{lbl}_{state.current_process}"] += dt

            # Draw
            self._draw_ui(frame, elapsed, board_box, hands)
            cv2.imshow(config.WINDOW_NAME, frame)
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('r'): self.process_counters = {k:0.0 for k in self.process_counters}

        # --- EXPORT DATA AUTOMATICALLY ---
        self._export_data()

        # --- SUMMARY SCREEN ---
        summary_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(summary_img, "SESSION SUMMARY", (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        y_pos = 200
        for key, value in self.process_counters.items():
            if value > 0.1:
                text = f"{key}: {value:.2f} seconds"
                cv2.putText(summary_img, text, (400, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y_pos += 50
        
        cv2.putText(summary_img, "Data Exported to 'Process_Data_Log.csv'", (350, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(summary_img, "Press ANY KEY to Exit", (480, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow(config.WINDOW_NAME, summary_img)
        cv2.waitKey(0)

        self.cap.release()
        cv2.destroyAllWindows()
        self.hand_tracker.close()

    def _draw_ui(self, frame, elapsed, board_box, hands):
        # Draw Board
        if board_box:
            col = (0, 255, 0) if self.board_tracker.is_confirmed else (0, 165, 255)
            cv2.rectangle(frame, (board_box[0], board_box[1]), (board_box[2], board_box[3]), col, 2)
            cv2.putText(frame, "BOARD ZONE", (board_box[0], board_box[1]-10), 0, 0.6, col, 2)

        # Draw Hands
        for lbl, (hbox, _) in hands.items():
            st = self.hand_tracker.hand_states[lbl]
            # Color coding: Delay=Gray, Hold=Cyan, Op=Green, Trans=Red
            c = (128,128,128) if st.current_process=="Delay" else \
                (255,255,0) if st.current_process=="Hold" else \
                (0,255,0) if st.current_process=="Operation" else (0,0,255)
            
            cv2.rectangle(frame, (hbox[0], hbox[1]), (hbox[2], hbox[3]), c, 2)
            
            info = f"{lbl}: {st.current_process}" 
            cv2.putText(frame, info, (hbox[0], hbox[1]-10), 0, 0.6, c, 2)
            
            # Debug Timer Bar
            if st.target_state_candidate and st.board_entry_time:
                dur = time.time() - st.board_entry_time
                prog = min(dur/config.ZONE_STABILITY_TIME, 1.0)
                bw = hbox[2]-hbox[0]
                cv2.rectangle(frame, (hbox[0], hbox[1]-30), (hbox[0]+int(bw*prog), hbox[1]-25), (255,255,255), -1)

        # UI Stats
        y = 100
        for k, v in self.process_counters.items():
            if v > 0.1:
                cv2.putText(frame, f"{k}: {v:.1f}s", (20, y), 0, 0.6, (0,0,0), 1)
                y += 20
        
        # Start Button
        bx, by, bw, bh = config.BUTTON_RECT
        col = (0,0,255) if self.app_state["timer_running"] else (0,255,0)
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), col, -1)
        cv2.putText(frame, f"Time: {elapsed:.1f}s", (bx+bw+10, by+35), 0, 0.8, (0,0,0), 2)

if __name__ == '__main__':
    freeze_support()
    try:
        # Ensure your path is correct
        DualBrainDetectionSystem(r"runs\detect\flow_process_chart_v1\weights\best.pt").run()
    except Exception as e:
        print(f"An error occurred: {e}")
