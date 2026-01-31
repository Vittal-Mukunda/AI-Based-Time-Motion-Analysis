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
import csv 
import os 
from datetime import datetime 
import matplotlib
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Central configuration"""
    # Camera
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    
    # Dashboard
    GRAPH_UPDATE_FREQ: int = 3  # Update graphs every N frames
    
    # YOLO
    YOLO_CONF_THRESHOLD: float = 0.15  
    YOLO_IOU_THRESHOLD: float = 0.45   
    
    # MediaPipe
    MP_MIN_DETECTION_CONF: float = 0.6
    MP_MIN_TRACKING_CONF: float = 0.6
    MP_MAX_HANDS: int = 2
    
    # Logic
    VELOCITY_THRESHOLD: float = 5.0 
    VELOCITY_SMOOTHING_FRAMES: int = 3 
    ZONE_STABILITY_TIME: float = 0.15 
    DELAY_INACTIVITY_TIME: float = 2.0     
        
    # UI
    WINDOW_NAME: str = 'Video Feed (S: Record CSV | R: Reset Graphs | Q: Quit)'

config = Config()

# ============================================================================
# DATA LOGGING (POWER BI OPTIMIZED)
# ============================================================================

class FrameLogger:
    """Logs every frame's data for high-resolution Power BI analysis"""
    def __init__(self, filename="Live_Process_Data.csv"):
        self.filename = filename
        
        # 1. GENERATE UNIQUE SESSION ID
        # This ensures Power BI can distinguish this "run" from previous ones
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.headers = [
            "Timestamp", "Session_ID", "Frame_ID", 
            "Hand_Side", "Current_State", "Velocity_Px", 
            "Fingers_In_Zone", "In_Board_Zone_Bool", 
            "Pos_X", "Pos_Y"
        ]
        
        # 2. SMART HEADERS
        # Only write headers if the file DOES NOT exist. 
        # If it exists, we skip this so we can append data directly.
        if not os.path.isfile(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            print(f"Created new log file: {self.filename}")
        else:
            print(f"Appending to existing log file: {self.filename}")

    def log(self, frame_id, hand_data):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        rows = []
        for hand, data in hand_data.items():
            rows.append([
                timestamp, self.session_id, frame_id,
                hand, data['state'], round(data['velocity'], 2),
                data['fingers'], data['in_zone'],
                data['pos'][0], data['pos'][1]
            ])
        
        # 3. APPEND MODE ('a')
        # Opens file in append mode, writes rows, and closes immediately to ensure data safety.
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

# ============================================================================
# DASHBOARD (4 CUSTOM GRAPHS)
# ============================================================================

class Dashboard:
    def __init__(self):
        plt.ion()
        plt.style.use('dark_background')
        
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 6))
        self.fig.canvas.manager.set_window_title("Workstation Analytics (Live)")
        self.fig.tight_layout(pad=3.0)
        
        # Motion History for Envelope Graph
        self.path_history_L = deque(maxlen=50)
        self.path_history_R = deque(maxlen=50)

    def update_graph(self, hands_info, counters):
        # Data Prep
        states = ['Operation', 'Transport', 'Delay']
        state_colors = ['#00ff00', '#ff0000', '#888888'] # Green, Red, Gray
        
        # --- 1. Top-Left: Hand Utilization Pie (Operation Only) ---
        ax1 = self.axs[0, 0]
        ax1.clear()
        ax1.set_title("Hand Utilization (Operation Only)", fontsize=10, color='cyan')
        
        left_op_time = counters.get("Left_Operation", 0)
        right_op_time = counters.get("Right_Operation", 0)
        total_op_time = left_op_time + right_op_time
        
        if total_op_time > 0.1:
            ax1.pie([left_op_time, right_op_time], labels=['Left', 'Right'], 
                    autopct='%1.1f%%', colors=['cyan', 'magenta'], startangle=90, textprops={'fontsize': 8})
        else:
            ax1.text(0.5, 0.5, "No Operations Yet", ha='center', va='center', color='gray')

        # --- 2. Top-Right: Process Distribution Pie ---
        ax2 = self.axs[0, 1]
        ax2.clear()
        ax2.set_title("Process Breakdown (%)", fontsize=10, color='white')
        
        # Aggregate totals (Left + Right)
        total_ops = sum([counters.get(f"{h}_Operation", 0) for h in ['Left', 'Right']])
        total_trans = sum([counters.get(f"{h}_Transport", 0) for h in ['Left', 'Right']])
        total_delay = sum([counters.get(f"{h}_Delay", 0) for h in ['Left', 'Right']])
        
        totals = [total_ops, total_trans, total_delay]
        
        if sum(totals) > 0.1:
            ax2.pie(totals, labels=states, autopct='%1.1f%%', colors=state_colors, startangle=90, textprops={'fontsize': 8})
        else:
            ax2.text(0.5, 0.5, "Collecting Data...", ha='center', va='center', color='gray')

        # --- 3. Bottom-Left: Process Duration Bar ---
        ax3 = self.axs[1, 0]
        ax3.clear()
        ax3.set_title("Process Duration (Seconds)", fontsize=10, color='yellow')
        
        bars = ax3.bar(states, totals, color=state_colors)
        ax3.set_ylabel("Seconds")
        ax3.grid(axis='y', alpha=0.2)
        
        # Prevent auto-scale jitter by setting min height
        if max(totals) < 10:
            ax3.set_ylim(0, 10)

        for rect in bars:
            height = rect.get_height()
            if height > 0:
                ax3.text(rect.get_x() + rect.get_width()/2.0, height, f'{height:.1f}s', ha='center', va='bottom')

        # --- 4. Bottom-Right: Workstation Envelope (Motion Path) ---
        if 'Left' in hands_info: self.path_history_L.append(hands_info['Left']['pos'])
        if 'Right' in hands_info: self.path_history_R.append(hands_info['Right']['pos'])
        
        ax4 = self.axs[1, 1]
        ax4.clear()
        ax4.set_title("Workstation Envelope (Hand Motion)", fontsize=10, color='white')
        
        # Invert Y axis to match image coordinates
        ax4.set_xlim(0, config.FRAME_WIDTH)
        ax4.set_ylim(config.FRAME_HEIGHT, 0) 
        
        if self.path_history_L:
            lx, ly = zip(*self.path_history_L)
            ax4.plot(lx, ly, color='cyan', label='Left', linewidth=1)
            ax4.scatter(lx[-1], ly[-1], color='cyan', s=30) 
            
        if self.path_history_R:
            rx, ry = zip(*self.path_history_R)
            ax4.plot(rx, ry, color='magenta', label='Right', linewidth=1)
            ax4.scatter(rx[-1], ry[-1], color='magenta', s=30)
            
        ax4.legend(loc='upper right', fontsize=6)
        ax4.grid(True, alpha=0.1)

        # --- Refresh ---
        plt.draw()
        plt.pause(0.001)

# ============================================================================
# LOGIC & TRACKING
# ============================================================================

@dataclass
class HandState:
    label: str  
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=config.VELOCITY_SMOOTHING_FRAMES))
    current_process: str = "Transport"
    stationary_start_time: Optional[float] = None 
    board_entry_time: Optional[float] = None 
    target_state_candidate: Optional[str] = None 
    finger_count_in_board: int = 0
    last_seen_frame: int = 0

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands_model = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MP_MAX_HANDS,
            min_detection_confidence=config.MP_MIN_DETECTION_CONF,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONF
        )
        self.hand_states = {"Left": HandState("Left"), "Right": HandState("Right")}
    
    def calculate_velocity(self, hand_label: str) -> float:
        state = self.hand_states[hand_label]
        if len(state.velocity_history) < 2: return 0.0
        return np.mean(list(state.velocity_history))
    
    def process_frame(self, frame, frame_id):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_model.process(frame_rgb)
        detected = {}
        
        if results.multi_hand_landmarks:
            for i, landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                label = results.multi_handedness[i].classification[0].label
                h, w, _ = frame.shape
                x_vals = [lm.x * w for lm in landmarks.landmark]
                y_vals = [lm.y * h for lm in landmarks.landmark]
                
                box = [int(min(x_vals)-20), int(min(y_vals)-20), int(max(x_vals)+20), int(max(y_vals)+20)]
                center = (int(np.mean(x_vals)), int(np.mean(y_vals)))
                
                state = self.hand_states[label]
                state.last_seen_frame = frame_id
                state.position_history.append(center)
                
                if len(state.position_history) >= 2:
                    dist = math.hypot(center[0]-state.position_history[-2][0], center[1]-state.position_history[-2][1])
                    state.velocity_history.append(dist)
                    
                detected[label] = {'box': box, 'landmarks': landmarks, 'center': center}
                
        return detected

class ProcessStateMachine:
    def __init__(self, hand_states):
        self.hand_states = hand_states
    
    def count_fingers_in_zone(self, landmarks, box, w, h):
        if not box: return 0
        tips = [4, 8, 12, 16, 20]
        bx1, by1, bx2, by2 = box
        count = 0
        for idx in tips:
            lx, ly = landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h
            if bx1 < lx < bx2 and by1 < ly < by2:
                count += 1
        return count

    def update(self, hand_lbl, landmarks, board_box, time_now, w, h):
        state = self.hand_states[hand_lbl]
        velocity = np.mean(list(state.velocity_history)) if state.velocity_history else 0
        is_moving = velocity > config.VELOCITY_THRESHOLD
        fingers_in = self.count_fingers_in_zone(landmarks, board_box, w, h)
        state.finger_count_in_board = fingers_in
        
        candidate = "Operation" if fingers_in >= 1 else None
        
        if is_moving: state.stationary_start_time = None
        elif state.stationary_start_time is None: state.stationary_start_time = time_now
            
        if candidate != state.target_state_candidate:
            state.target_state_candidate = candidate
            state.board_entry_time = time_now if candidate else None
            
        new_process = "Transport"
        if state.stationary_start_time and (time_now - state.stationary_start_time > config.DELAY_INACTIVITY_TIME):
            new_process = "Delay"
        elif state.target_state_candidate == "Operation" and state.board_entry_time:
            if (time_now - state.board_entry_time > config.ZONE_STABILITY_TIME):
                new_process = "Operation"
        
        state.current_process = new_process
        return new_process

# ============================================================================
# MAIN SYSTEM
# ============================================================================

class DualBrainDetectionSystem:
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading AI on {self.device}...")
        
        self.yolo = YOLO(model_path).to(self.device)
        self.tracker = HandTracker()
        self.logic = ProcessStateMachine(self.tracker.hand_states)
        # Use the raw string (r"...") to handle backslashes correctly
        csv_path = r"C:\Users\vitta\OneDrive\Desktop\Flowprocess\Live_Process_Data.csv"
        self.logger = FrameLogger(filename=csv_path)
        
        # NOTE: Update video path here
        self.cap = cv2.VideoCapture(r"C:\Users\vitta\OneDrive\Desktop\Flowprocess\wsd.mp4")
        
        self.counters = {f"{h}_{p}": 0.0 for h in ["Left", "Right"] for p in ["Operation", "Transport", "Delay"]}
        self.app_state = {"timer_on": True, "start_time": time.time()}
        
        print("Initializing Dashboard...")
        self.dashboard = Dashboard()

    def run(self):
        frame_id = 0
        last_time = time.time()
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: break
            
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            frame_id += 1
            curr_time = time.time()
            dt = curr_time - last_time
            last_time = curr_time
            
            # 1. Detect Board
            board_box = None
            yolo_res = self.yolo.predict(frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False, device=self.device)
            if yolo_res[0].boxes:
                for box in yolo_res[0].boxes:
                    if self.yolo.names[int(box.cls)] == 'board':
                        board_box = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        break 
            
            # 2. Track Hands
            detected_hands = self.tracker.process_frame(frame, frame_id)
            
            # 3. Logic & Data Prep
            hands_info_for_graph = {}
            for lbl, data in detected_hands.items():
                process = self.logic.update(lbl, data['landmarks'], board_box, curr_time, config.FRAME_WIDTH, config.FRAME_HEIGHT)
                vel = self.tracker.calculate_velocity(lbl)
                fingers = self.tracker.hand_states[lbl].finger_count_in_board
                
                # Update Counters (ALWAYS)
                key = f"{lbl}_{process}"
                if key in self.counters: 
                    self.counters[key] += dt
                
                hands_info_for_graph[lbl] = {
                    'velocity': vel, 
                    'fingers': fingers, 
                    'state': process, 
                    'in_zone': fingers >= 1,
                    'pos': data['center']
                }
            
            # 4. Logging (Only if Recording is ON)
            if self.app_state["timer_on"]:
                self.logger.log(frame_id, hands_info_for_graph)
            
            # 5. Update Graphs
            if frame_id % config.GRAPH_UPDATE_FREQ == 0:
                self.dashboard.update_graph(hands_info_for_graph, self.counters)
            
            # 6. Draw UI
            self._draw_overlays(frame, board_box, detected_hands)
            cv2.imshow(config.WINDOW_NAME, frame)
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('s'): 
                self.app_state["timer_on"] = not self.app_state["timer_on"]
                print(f"Recording CSV: {self.app_state['timer_on']}")
            elif k == ord('r'):
                # Reset Counters
                self.counters = {k: 0.0 for k in self.counters}
                self.dashboard.path_history_L.clear()
                self.dashboard.path_history_R.clear()
                print("Graphs Reset.")

        self.cap.release()
        cv2.destroyAllWindows()
        plt.close('all')
        print(f"Data saved to {self.logger.filename}s")

    def _draw_overlays(self, frame, board_box, hands):
        if board_box:
            cv2.rectangle(frame, (board_box[0], board_box[1]), (board_box[2], board_box[3]), (0, 255, 255), 2)
            cv2.putText(frame, "BOARD ZONE", (board_box[0], board_box[1]-10), 0, 0.7, (0, 255, 255), 2)
            
        for lbl, data in hands.items():
            state = self.tracker.hand_states[lbl]
            proc = state.current_process
            vel = self.tracker.calculate_velocity(lbl)
            c = (0,255,0) if proc=="Operation" else (255,255,0) if proc=="Hold" else (100,100,100) if proc=="Delay" else (0,0,255)
            
            hbox = data['box']
            cv2.rectangle(frame, (hbox[0], hbox[1]), (hbox[2], hbox[3]), c, 2)
            text = f"{lbl} | {proc} | {int(vel)}px/s"
            cv2.putText(frame, text, (hbox[0], hbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            cv2.putText(frame, f"Fingers: {state.finger_count_in_board}", (hbox[0]+5, hbox[3]-25), 0, 0.5, (255,255,255), 1)

        status = "REC" if self.app_state["timer_on"] else "LIVE"
        col = (0,0,255) if self.app_state["timer_on"] else (0,255,0)
        cv2.circle(frame, (30, 30), 15, col, -1)
        cv2.putText(frame, f"{status}", (55, 40), 0, 0.8, (255,255,255), 2)

if __name__ == '__main__':
    freeze_support()
    model_path = r"runs\detect\flow_process_chart_v1\weights\best.pt"
    try:
        DualBrainDetectionSystem(model_path).run()
    except Exception as e:
        print(f"Error: {e}")
