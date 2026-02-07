import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from multiprocessing import freeze_support
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import torch 
import csv 
import os 
from datetime import datetime 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import threading
import queue

# Try to import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARNING] psutil not installed. CPU/Memory monitoring disabled.")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    GRAPH_UPDATE_FREQ: int = 3
    YOLO_CONF_THRESHOLD: float = 0.25 
    YOLO_IOU_THRESHOLD: float = 0.45   
    MP_MIN_DETECTION_CONF: float = 0.5
    MP_MIN_TRACKING_CONF: float = 0.5
    MP_MAX_HANDS: int = 2
    VELOCITY_THRESHOLD: float = 10.0 
    VELOCITY_SMOOTHING_FRAMES: int = 5
    ZONE_STABILITY_TIME: float = 0.1  
    DELAY_INACTIVITY_TIME: float = 1.0     
    LOG_INTERVAL: float = 0.05  
    WINDOW_NAME: str = 'Kinetic Analytics V2.0 | [Q] Exit [R] Reset'
    
    # === NEW UI CONFIG ===
    MASTER_WIDTH: int = 1920
    MASTER_HEIGHT: int = 1080
    SIDEBAR_WIDTH_RATIO: float = 0.18
    HEADER_HEIGHT_RATIO: float = 0.06
    PROGRESS_BAR_HEIGHT: int = 8
    TARGET_CYCLE_TIME: float = 60.0  # 1-minute target

    # Cyber-Web Color Palette (BGR format for OpenCV)
    BG_COLOR: tuple = (26, 26, 26)        # #1a1a1a
    ACCENT_COLOR: tuple = (255, 242, 0)   # #00f2ff (BGR)
    TEXT_COLOR: tuple = (224, 224, 224)   # #e0e0e0
    HEADER_BG: tuple = (20, 20, 20)
    SIDEBAR_BG: tuple = (30, 30, 30)
    
    # State Colors (BGR)
    OP_COLOR: tuple = (0, 255, 0)         # Green
    TRANS_COLOR: tuple = (255, 204, 0)    # Cyan
    DELAY_COLOR: tuple = (102, 102, 102)  # Grey
    
    # Alert Settings
    IDLE_ALERT_THRESHOLD: float = 5.0  # seconds

config = Config()

# ============================================================================
# UTILITIES FOR VISUALS (CYBER-WEB STYLE)
# ============================================================================

def draw_rounded_rect(img, pt1, pt2, color, thickness=1, radius=10, fill=False):
    """Draw a rectangle with simulated rounded corners using lines."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Clamp radius
    radius = min(radius, abs(x2-x1)//2, abs(y2-y1)//2)
    
    if fill:
        # Fill the main rectangle area
        overlay = img.copy()
        cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
        # Corner circles
        cv2.circle(overlay, (x1+radius, y1+radius), radius, color, -1)
        cv2.circle(overlay, (x2-radius, y1+radius), radius, color, -1)
        cv2.circle(overlay, (x1+radius, y2-radius), radius, color, -1)
        cv2.circle(overlay, (x2-radius, y2-radius), radius, color, -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    else:
        # Top line
        cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
        # Bottom line
        cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
        # Left line
        cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
        # Right line
        cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
        # Corners (arcs simulated with small lines)
        cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_hud_panel(img, x, y, w, h, title, value, color=(0, 255, 0), scale=1.0):
    """Draw a semi-transparent HUD panel with title and value."""
    # Clamp to image bounds
    ih, iw = img.shape[:2]
    x = max(0, min(x, iw - w))
    y = max(0, min(y, ih - h))
    x2 = min(x + w, iw)
    y2 = min(y + h, ih)
    
    if x2 <= x or y2 <= y:
        return
    
    # Semi-transparent background
    sub_img = img[y:y2, x:x2]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 30
    res = cv2.addWeighted(sub_img, 0.7, white_rect, 0.3, 1.0)
    img[y:y2, x:x2] = res
    
    # Border with accent color
    cv2.rectangle(img, (x, y), (x2-1, y2-1), color, 1)
    
    # Title
    font_scale = 0.4 * scale
    cv2.putText(img, title, (x+5, y+int(15*scale)), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (200, 200, 200), 1, cv2.LINE_AA)
    # Value
    cv2.putText(img, str(value), (x+5, y+int(35*scale)), cv2.FONT_HERSHEY_DUPLEX, 
                0.6 * scale, color, 1, cv2.LINE_AA)

def draw_dynamic_rect(img, box, color, alpha=0.2):
    """Fills a rectangle with transparency and corner brackets."""
    x1, y1, x2, y2 = box
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Corner brackets for cleaner look
    l = 20  # length of bracket
    t = 2   # thickness
    cv2.line(img, (x1, y1), (x1+l, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1+l), color, t)
    cv2.line(img, (x2, y1), (x2-l, y1), color, t)
    cv2.line(img, (x2, y1), (x2, y1+l), color, t)
    cv2.line(img, (x1, y2), (x1+l, y2), color, t)
    cv2.line(img, (x1, y2), (x1, y2-l), color, t)
    cv2.line(img, (x2, y2), (x2-l, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2-l), color, t)

def draw_toast_notification(img, message, position, color=(0, 200, 255), scale=1.0):
    """Draw a toast-style notification popup."""
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5 * scale
    thickness = 1
    
    # Calculate text size
    (text_w, text_h), baseline = cv2.getTextSize(message, font, font_scale, thickness)
    
    padding = int(10 * scale)
    box_w = text_w + padding * 2
    box_h = text_h + padding * 2 + baseline
    
    # Position adjustment to keep in bounds
    ih, iw = img.shape[:2]
    x = min(x, iw - box_w - 10)
    y = min(y, ih - box_h - 10)
    
    # Draw rounded background
    draw_rounded_rect(img, (x, y), (x + box_w, y + box_h), color, fill=True, radius=8)
    draw_rounded_rect(img, (x, y), (x + box_w, y + box_h), color, thickness=2, radius=8)
    
    # Draw text
    cv2.putText(img, message, (x + padding, y + padding + text_h), font, 
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_progress_bar(img, x, y, width, height, progress, color=(255, 242, 0)):
    """Draw a horizontal progress bar."""
    # Background
    cv2.rectangle(img, (x, y), (x + width, y + height), (50, 50, 50), -1)
    
    # Progress fill
    fill_width = int(width * min(progress, 1.0))
    if fill_width > 0:
        # Gradient effect
        for i in range(fill_width):
            alpha = 0.5 + 0.5 * (i / max(fill_width, 1))
            c = tuple(int(v * alpha) for v in color)
            cv2.line(img, (x + i, y), (x + i, y + height), c, 1)
    
    # Border
    cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)

# ============================================================================
# ROBUST DATA LOGGING (Power BI Compatible)
# ============================================================================

class RobustDataLogger:
    def __init__(self):
        self.file_path = r"C:\Users\vitta\OneDrive\Desktop\Python 3.10\AI-Based-Time-Motion-Analysis\Master_Raw_Data.csv"
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_log_time = -1.0
        
        self.headers = [
            "Row_Number", "Timestamp", "Session_ID", "Frame_ID", "Hand_Side", 
            "Current_State", "Velocity_Px", "Fingers_In_Zone", 
            "In_Board_Zone_Bool", "Pos_X", "Pos_Y"
        ]
        
        self.current_row_index = 0
        self._initialize_file()
        self.last_state = {"Left": None, "Right": None}
        self.state_start_time = {"Left": 0.0, "Right": 0.0}

    def _initialize_file(self):
        if not os.path.exists(self.file_path):
            try:
                with open(self.file_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
                print(f"[LOGGER] Created new file: {self.file_path}")
                self.current_row_index = 0
            except Exception as e:
                print(f"[LOGGER] Error creating file: {e}")
        else:
            try:
                if os.path.getsize(self.file_path) > 0:
                    with open(self.file_path, mode='r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            last_line = lines[-1].strip()
                            if last_line:
                                last_row_data = last_line.split(',')
                                self.current_row_index = int(last_row_data[0])
                    print(f"[LOGGER] Resuming from Row Number: {self.current_row_index}")
                else:
                    with open(self.file_path, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(self.headers)
            except Exception as e:
                print(f"[LOGGER] Error reading last row: {e}. Starting from 0.")
                self.current_row_index = 0

    def log_frame(self, frame_id, video_time, hand_data):
        if (video_time - self.last_log_time) < config.LOG_INTERVAL:
            return
        if not hand_data:
            return 

        rows_to_write = []
        for hand, data in hand_data.items():
            self.current_row_index += 1
            in_zone_str = "True" if data['fingers'] > 0 else "False"
            row = [
                self.current_row_index,
                f"{video_time:.3f}",
                self.session_id,
                frame_id,
                hand,
                data['state'],
                f"{data['velocity']:.1f}",
                data['fingers'],
                in_zone_str,
                data['pos'][0],
                data['pos'][1]
            ]
            rows_to_write.append(row)
        
        try:
            with open(self.file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows_to_write)
            self.last_log_time = video_time
        except Exception as e:
            print(f"[CSV ERROR] Could not write to file: {e}")

    def update_state_timers(self, video_time, hand, new_state):
        old_state = self.last_state[hand]
        if old_state is not None and old_state != new_state:
            self.state_start_time[hand] = video_time
        elif old_state is None:
            self.state_start_time[hand] = video_time
        self.last_state[hand] = new_state

# ============================================================================
# DASHBOARD (THREADED NON-BLOCKING)
# ============================================================================

class Dashboard:
    def __init__(self):
        plt.ion()
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#0f0f0f'
        plt.rcParams['figure.facecolor'] = '#0f0f0f'
        plt.rcParams['grid.color'] = '#444444'
        plt.rcParams['text.color'] = '#e0e0e0'
        plt.rcParams['axes.labelcolor'] = '#e0e0e0'
        plt.rcParams['xtick.color'] = '#e0e0e0'
        plt.rcParams['ytick.color'] = '#e0e0e0'
        
        self.fig = plt.figure(figsize=(12, 7))
        gs = GridSpec(2, 3, figure=self.fig)
        
        try:
            self.fig.canvas.manager.set_window_title("Workstation Analytics (Live)")
        except: pass
        
        self.path_history_L = deque(maxlen=50)
        self.path_history_R = deque(maxlen=50)

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        self.ax4 = self.fig.add_subplot(gs[1, :])
        
        self.ax4.set_title("Real-Time Motion Trajectory", fontsize=10, color='#00ffcc', pad=10)
        self.ax4.set_xlim(0, config.FRAME_WIDTH)
        self.ax4.set_ylim(config.FRAME_HEIGHT, 0)
        self.ax4.grid(True, alpha=0.1)
        
        self.line_L, = self.ax4.plot([], [], color='#00ffff', linewidth=2, label='Left Hand')
        self.line_R, = self.ax4.plot([], [], color='#ff00ff', linewidth=2, label='Right Hand')
        self.ax4.legend(loc='upper right', frameon=False, fontsize='small')

        self.states = ['Operation', 'Transport', 'Delay']
        x = np.arange(len(self.states))
        width = 0.35
        self.bars_L = self.ax3.bar(x - width/2, [0,0,0], width, label='Left', color='#00ffff', alpha=0.8)
        self.bars_R = self.ax3.bar(x + width/2, [0,0,0], width, label='Right', color='#ff00ff', alpha=0.8)
        self.ax3.set_xticks(x)
        self.ax3.set_xticklabels(self.states)
        self.ax3.set_title("Activity Duration (s)", fontsize=10, color='#ffcc00')
        self.ax3.spines['top'].set_visible(False)
        self.ax3.spines['right'].set_visible(False)
        self.ax3.legend(loc='upper right', fontsize='small')
        
        # Draw initial figure
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def update_graph(self, hands_info, counters):
        """Direct non-blocking update (no threading)."""
        if not plt.fignum_exists(self.fig.number): 
            return

        try:
            # 1. Operation Split Pie
            self.ax1.clear()
            self.ax1.set_title("L/R Balance", fontsize=10, color='white')
            l_op = counters.get("Left_Operation", 0)
            r_op = counters.get("Right_Operation", 0)
            if (l_op + r_op) > 0.1:
                wedges, _, _ = self.ax1.pie([l_op, r_op], labels=['L', 'R'], autopct='%1.0f%%', 
                                          colors=['#00ffff', '#ff00ff'], startangle=90,
                                          wedgeprops={'edgecolor': 'black'})
                for w in wedges: w.set_alpha(0.8)
            
            # 2. Effort Distribution Pie
            self.ax2.clear()
            self.ax2.set_title("State Dist.", fontsize=10, color='white')
            total_vals = [sum([counters.get(f"{h}_{s}", 0) for h in ['Left', 'Right']]) for s in self.states]
            if sum(total_vals) > 0.1:
                colors = ['#00ff00', '#0099ff', '#666666']
                wedges, _, _ = self.ax2.pie(total_vals, labels=['Op', 'Tr', 'De'], autopct='%1.0f%%', 
                                          colors=colors, startangle=90,
                                          wedgeprops={'edgecolor': 'black'})
                for w in wedges: w.set_alpha(0.8)

            # 3. Bar Chart Logic
            l_vals = [counters.get(f"Left_{s}", 0) for s in self.states]
            r_vals = [counters.get(f"Right_{s}", 0) for s in self.states]
            
            max_val = max(max(l_vals, default=0), max(r_vals, default=0))
            if max_val > self.ax3.get_ylim()[1]:
                self.ax3.set_ylim(0, max_val * 1.2)

            for bar, h in zip(self.bars_L, l_vals): bar.set_height(h)
            for bar, h in zip(self.bars_R, r_vals): bar.set_height(h)

            # 4. Motion Path Logic
            if 'Left' in hands_info: self.path_history_L.append(hands_info['Left']['pos'])
            if 'Right' in hands_info: self.path_history_R.append(hands_info['Right']['pos'])
            
            if self.path_history_L:
                lx, ly = zip(*self.path_history_L)
                self.line_L.set_data(lx, ly)
            if self.path_history_R:
                rx, ry = zip(*self.path_history_R)
                self.line_R.set_data(rx, ry)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            pass  # Silently ignore matplotlib errors
    
    def stop(self):
        """Cleanup method."""
        pass

# ============================================================================
# LOGIC & TRACKING (UNCHANGED CORE)
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
                if results.multi_handedness:
                    label = results.multi_handedness[i].classification[0].label
                else:
                    label = "Right"
                
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

    def update(self, hand_lbl, landmarks, board_box, video_time, w, h):
        state = self.hand_states[hand_lbl]
        velocity = np.mean(list(state.velocity_history)) if state.velocity_history else 0
        is_moving = velocity > config.VELOCITY_THRESHOLD
        
        fingers_in = self.count_fingers_in_zone(landmarks, board_box, w, h)
        state.finger_count_in_board = fingers_in
        candidate = "Operation" if fingers_in >= 1 else None
        
        if is_moving: state.stationary_start_time = None
        elif state.stationary_start_time is None: state.stationary_start_time = video_time
            
        if candidate != state.target_state_candidate:
            state.target_state_candidate = candidate
            state.board_entry_time = video_time if candidate else None
            
        new_process = "Transport"
        if state.stationary_start_time and (video_time - state.stationary_start_time > config.DELAY_INACTIVITY_TIME):
            new_process = "Delay"
        elif state.target_state_candidate == "Operation" and state.board_entry_time:
            if (video_time - state.board_entry_time > config.ZONE_STABILITY_TIME):
                new_process = "Operation"
        
        state.current_process = new_process
        return new_process

# ============================================================================
# MAIN SYSTEM (REFACTORED UI)
# ============================================================================

class DualBrainDetectionSystem:
    def __init__(self, model_path: str, video_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INIT] Loading AI on {self.device}...")
        
        self.yolo = YOLO(model_path).to(self.device)
        self.tracker = HandTracker()
        self.logic = ProcessStateMachine(self.tracker.hand_states)
        self.logger = RobustDataLogger()
        
        if not os.path.exists(video_path):
            print(f"[WARNING] Video path not found: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        
        self.counters = {f"{h}_{p}": 0.0 for h in ["Left", "Right"] for p in ["Operation", "Transport", "Delay"]}
        self.dashboard = Dashboard()
        self.mp_draw = mp.solutions.drawing_utils
        
        # === NEW UI COMPONENTS ===
        self.session_start_time = time.time()
        self.heatmap_accumulator = None  # Initialized on first frame
        self.delay_start_time = {"Left": None, "Right": None}
        self.show_idle_alert = False
        self.idle_alert_message = ""
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.current_fps = 0.0

    def _get_system_stats(self):
        """Get CPU and memory usage."""
        cpu_percent = 0.0
        mem_mb = 0.0
        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            except:
                pass
        return cpu_percent, mem_mb

    def _update_heatmap(self, hands_info):
        """Update heatmap accumulator with hand positions."""
        if self.heatmap_accumulator is None:
            return
        
        decay = 0.995  # Slow decay for smoother effect
        self.heatmap_accumulator = (self.heatmap_accumulator * decay).astype(np.float32)
        
        for lbl, data in hands_info.items():
            cx, cy = data['pos']
            # Scale position to heatmap coordinates
            hh, hw = self.heatmap_accumulator.shape[:2]
            hx = int(cx * hw / config.FRAME_WIDTH)
            hy = int(cy * hh / config.FRAME_HEIGHT)
            
            # Add heat at hand position (gaussian-like)
            radius = 30
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    px, py = hx + dx, hy + dy
                    if 0 <= px < hw and 0 <= py < hh:
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < radius:
                            intensity = (1 - dist/radius) * 5
                            self.heatmap_accumulator[py, px] = min(255, self.heatmap_accumulator[py, px] + intensity)

    def _check_idle_alerts(self, hands_info, video_time):
        """Check for prolonged delay state and trigger alerts."""
        any_delay = False
        for lbl, data in hands_info.items():
            if data['state'] == "Delay":
                if self.delay_start_time[lbl] is None:
                    self.delay_start_time[lbl] = video_time
                elif (video_time - self.delay_start_time[lbl]) > config.IDLE_ALERT_THRESHOLD:
                    any_delay = True
                    duration = video_time - self.delay_start_time[lbl]
                    self.idle_alert_message = f"⚠ {lbl} HAND IDLE: {duration:.1f}s"
            else:
                self.delay_start_time[lbl] = None
        
        # Also check if no hands detected for extended time
        if not hands_info:
            self.show_idle_alert = True
            self.idle_alert_message = "⚠ NO HANDS DETECTED"
        else:
            self.show_idle_alert = any_delay

    def _draw_web_interface(self, frame, board_box, hands, vid_time, hands_info):
        """
        Create a modern web dashboard-style interface.
        Returns the complete UI canvas.
        """
        # Calculate responsive dimensions
        master_w = config.MASTER_WIDTH
        master_h = config.MASTER_HEIGHT
        
        sidebar_w = int(master_w * config.SIDEBAR_WIDTH_RATIO)
        header_h = int(master_h * config.HEADER_HEIGHT_RATIO)
        progress_h = config.PROGRESS_BAR_HEIGHT
        
        content_w = master_w - sidebar_w
        content_h = master_h - header_h - progress_h
        
        # Scale factor for responsive elements
        scale = master_h / 1080.0
        
        # Create master canvas
        canvas = np.zeros((master_h, master_w, 3), dtype=np.uint8)
        canvas[:] = config.BG_COLOR
        
        # ===================== HEADER =====================
        cv2.rectangle(canvas, (0, 0), (master_w, header_h), config.HEADER_BG, -1)
        cv2.line(canvas, (0, header_h), (master_w, header_h), config.ACCENT_COLOR, 1)
        
        # Header content
        header_y = int(header_h * 0.7)
        font_header = cv2.FONT_HERSHEY_DUPLEX
        font_scale_header = 0.6 * scale
        
        # FPS
        cv2.putText(canvas, f"FPS: {self.current_fps:.1f}", (20, header_y), 
                    font_header, font_scale_header, config.ACCENT_COLOR, 1, cv2.LINE_AA)
        
        # CPU & Memory
        cpu, mem = self._get_system_stats()
        cv2.putText(canvas, f"CPU: {cpu:.0f}%", (int(150*scale), header_y), 
                    font_header, font_scale_header, config.TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"MEM: {mem:.0f}MB", (int(280*scale), header_y), 
                    font_header, font_scale_header, config.TEXT_COLOR, 1, cv2.LINE_AA)
        
        # Recording indicator
        pulse = abs(math.sin(time.time() * 3))
        rec_color = (0, 0, int(255 * pulse))
        cv2.circle(canvas, (int(450*scale), int(header_h/2)), int(8*scale), rec_color, -1)
        cv2.putText(canvas, f"REC: {vid_time:.1f}s", (int(470*scale), header_y), 
                    font_header, font_scale_header, (0, 100, 255), 1, cv2.LINE_AA)
        
        # Session ID (right side)
        session_text = f"SESSION: {self.logger.session_id}"
        text_size = cv2.getTextSize(session_text, font_header, font_scale_header * 0.8, 1)[0]
        cv2.putText(canvas, session_text, (master_w - text_size[0] - 20, header_y), 
                    font_header, font_scale_header * 0.8, (150, 150, 150), 1, cv2.LINE_AA)
        
        # ===================== SIDEBAR =====================
        cv2.rectangle(canvas, (0, header_h), (sidebar_w, master_h - progress_h), config.SIDEBAR_BG, -1)
        cv2.line(canvas, (sidebar_w, header_h), (sidebar_w, master_h - progress_h), config.ACCENT_COLOR, 1)
        
        sidebar_y = header_h + int(20 * scale)
        font_sidebar = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_sidebar = 0.5 * scale
        line_height = int(35 * scale)
        
        # Title
        cv2.putText(canvas, "ANALYTICS", (int(15*scale), sidebar_y), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7*scale, config.ACCENT_COLOR, 1, cv2.LINE_AA)
        sidebar_y += int(40 * scale)
        
        # Divider
        cv2.line(canvas, (10, sidebar_y), (sidebar_w-10, sidebar_y), (60, 60, 60), 1)
        sidebar_y += int(20 * scale)
        
        # Hand Stats (with colored indicators)
        for hand in ["Left", "Right"]:
            state = self.tracker.hand_states[hand]
            proc = state.current_process
            
            if proc == "Operation": 
                color = config.OP_COLOR
            elif proc == "Transport": 
                color = config.TRANS_COLOR
            else: 
                color = config.DELAY_COLOR
            
            # Hand label with status dot
            cv2.circle(canvas, (int(20*scale), sidebar_y - int(5*scale)), int(6*scale), color, -1)
            cv2.putText(canvas, f"{hand.upper()} HAND", (int(35*scale), sidebar_y), 
                        font_sidebar, font_scale_sidebar, config.TEXT_COLOR, 1, cv2.LINE_AA)
            sidebar_y += line_height
            
            # Current state
            cv2.putText(canvas, f"  State: {proc}", (int(15*scale), sidebar_y), 
                        font_sidebar, font_scale_sidebar * 0.9, color, 1, cv2.LINE_AA)
            sidebar_y += line_height
            
            # Velocity
            vel = self.tracker.calculate_velocity(hand)
            cv2.putText(canvas, f"  Vel: {vel:.1f} px/f", (int(15*scale), sidebar_y), 
                        font_sidebar, font_scale_sidebar * 0.9, (180, 180, 180), 1, cv2.LINE_AA)
            sidebar_y += line_height
            
            # Fingers in zone
            fingers = state.finger_count_in_board
            cv2.putText(canvas, f"  Fingers: {fingers}", (int(15*scale), sidebar_y), 
                        font_sidebar, font_scale_sidebar * 0.9, (180, 180, 180), 1, cv2.LINE_AA)
            sidebar_y += int(45 * scale)
        
        # Time counters section
        cv2.line(canvas, (10, sidebar_y), (sidebar_w-10, sidebar_y), (60, 60, 60), 1)
        sidebar_y += int(20 * scale)
        cv2.putText(canvas, "TIME COUNTERS", (int(15*scale), sidebar_y), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5*scale, config.ACCENT_COLOR, 1, cv2.LINE_AA)
        sidebar_y += int(30 * scale)
        
        # Display counters
        for state_name, color in [("Operation", config.OP_COLOR), ("Transport", config.TRANS_COLOR), ("Delay", config.DELAY_COLOR)]:
            total = self.counters.get(f"Left_{state_name}", 0) + self.counters.get(f"Right_{state_name}", 0)
            cv2.putText(canvas, f"{state_name}: {total:.1f}s", (int(15*scale), sidebar_y), 
                        font_sidebar, font_scale_sidebar * 0.9, color, 1, cv2.LINE_AA)
            sidebar_y += line_height
        
        # ===================== MAIN CONTENT (Camera Feed) =====================
        content_x = sidebar_w
        content_y = header_h
        
        # Resize frame to fit content area
        aspect_ratio = config.FRAME_WIDTH / config.FRAME_HEIGHT
        display_h = content_h
        display_w = int(display_h * aspect_ratio)
        
        if display_w > content_w:
            display_w = content_w
            display_h = int(display_w / aspect_ratio)
        
        # Center the frame
        frame_x = content_x + (content_w - display_w) // 2
        frame_y = content_y + (content_h - display_h) // 2
        
        # No heatmap overlay - use clean frame directly
        
        # Draw board zone on frame (solid color, no flashing)
        if board_box:
            zone_color = config.ACCENT_COLOR  # Solid cyan color
            draw_dynamic_rect(frame, board_box, zone_color)
            cv2.putText(frame, "ACTIVE ZONE", (board_box[0], board_box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1, cv2.LINE_AA)
        
        # Draw hand landmarks and trails on frame
        for lbl, data in hands.items():
            state = self.tracker.hand_states[lbl]
            proc = state.current_process
            
            if proc == "Operation": c = config.OP_COLOR
            elif proc == "Transport": c = config.TRANS_COLOR
            else: c = config.DELAY_COLOR
            
            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame, data['landmarks'], 
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
            
            # Motion trail
            hist = list(state.position_history)[-20:]
            for i in range(1, len(hist)):
                thickness = int(np.sqrt(20 / float(i + 1)) * 2.5)
                cv2.line(frame, hist[i-1], hist[i], c, thickness)
            
            # Floating label near hand
            hx, hy = data['center']
            label_text = f"{lbl[0]}: {proc}"
            cv2.putText(frame, label_text, (hx - 30, hy - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
        
        # Resize and place frame on canvas
        frame_resized = cv2.resize(frame, (display_w, display_h))
        canvas[frame_y:frame_y+display_h, frame_x:frame_x+display_w] = frame_resized
        
        # Frame border
        cv2.rectangle(canvas, (frame_x-2, frame_y-2), (frame_x+display_w+2, frame_y+display_h+2), 
                      config.ACCENT_COLOR, 2)
        
        # ===================== PROGRESS BAR =====================
        progress_y = master_h - progress_h
        elapsed = time.time() - self.session_start_time
        progress = elapsed / config.TARGET_CYCLE_TIME
        draw_progress_bar(canvas, 0, progress_y, master_w, progress_h, progress, config.ACCENT_COLOR)
        
        # ===================== TOAST NOTIFICATION =====================
        if self.show_idle_alert:
            toast_x = master_w - int(300 * scale)
            toast_y = master_h - int(80 * scale)
            draw_toast_notification(canvas, self.idle_alert_message, (toast_x, toast_y), 
                                   color=(0, 100, 255), scale=scale)
        
        return canvas

    def run(self):
        """Main video processing loop with error handling."""
        if not self.cap.isOpened():
            print("[ERROR] Could not open video source.")
            return

        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0 or np.isnan(video_fps): 
            video_fps = 30.0
        frame_duration = 1.0 / video_fps
        print(f"[SYNC] Processing at {video_fps} FPS (Video Time Mode)")

        frame_id = 0
        
        # Create resizable window
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_NAME, config.MASTER_WIDTH, config.MASTER_HEIGHT)
        
        try:
            while self.cap.isOpened():
                loop_start = time.time()
                success, frame = self.cap.read()
                if not success:
                    print("[INFO] End of video stream.")
                    break
                
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                frame_id += 1
                current_video_time = frame_id * frame_duration
                
                # === DETECTION ===
                board_box = None
                yolo_res = self.yolo.predict(frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False, device=self.device)
                if yolo_res[0].boxes:
                    for box in yolo_res[0].boxes:
                        if int(box.cls) == 0: 
                            board_box = box.xyxy[0].cpu().numpy().astype(int).tolist()
                            break 
                
                detected_hands = self.tracker.process_frame(frame, frame_id)
                
                hands_info_for_graph = {}
                for lbl, data in detected_hands.items():
                    process = self.logic.update(lbl, data['landmarks'], board_box, current_video_time, 
                                               config.FRAME_WIDTH, config.FRAME_HEIGHT)
                    self.logger.update_state_timers(current_video_time, lbl, process)
                    self.counters[f"{lbl}_{process}"] += frame_duration
                    
                    hands_info_for_graph[lbl] = {
                        'velocity': self.tracker.calculate_velocity(lbl), 
                        'fingers': self.tracker.hand_states[lbl].finger_count_in_board, 
                        'state': process, 
                        'pos': data['center'],
                        'landmarks': data['landmarks']
                    }
                
                # === FEATURE CHECKS ===
                self._check_idle_alerts(hands_info_for_graph, current_video_time)
                
                # === LOGGING ===
                self.logger.log_frame(frame_id, current_video_time, hands_info_for_graph)
                
                # === DASHBOARD (Non-blocking) ===
                if frame_id % config.GRAPH_UPDATE_FREQ == 0:
                    self.dashboard.update_graph(hands_info_for_graph, self.counters)
                
                # === FPS CALCULATION ===
                frame_time = time.time() - self.last_frame_time
                self.last_frame_time = time.time()
                if frame_time > 0:
                    self.fps_history.append(1.0 / frame_time)
                    self.current_fps = np.mean(list(self.fps_history))
                
                # === UI RENDERING ===
                ui_canvas = self._draw_web_interface(frame, board_box, detected_hands, 
                                                     current_video_time, hands_info_for_graph)
                cv2.imshow(config.WINDOW_NAME, ui_canvas)
                
                # === INPUT HANDLING ===
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'): 
                    break
                elif k == ord('r'):
                    self.counters = {k: 0.0 for k in self.counters}
                    self.session_start_time = time.time()
                    self.heatmap_accumulator = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH), dtype=np.float32)
                    print("[RESET] Counters and heatmap cleared.")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        except Exception as e:
            print(f"[ERROR] Exception in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # === GUARANTEED CLEANUP ===
            print("[CLEANUP] Releasing resources...")
            self.cap.release()
            cv2.destroyAllWindows()
            self.dashboard.stop()
            plt.close('all')
            print(f"[CLEANUP] Session data saved. Total rows: {self.logger.current_row_index}")
            print("[CLEANUP] Application closed successfully.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    freeze_support()
    
    MODEL_PATH = r"C:\Users\vitta\OneDrive\Desktop\Python 3.10\runs\detect\My_Custom_Training\board_detector_gpu\weights\best.pt"
    VIDEO_PATH = r"C:\Users\vitta\OneDrive\Desktop\Linkedin tut.mp4"
    
    print("=" * 60)
    print("  KINETIC ANALYTICS V2.0 - Modern Web Dashboard UI")
    print("  Press [Q] to Exit | [R] to Reset Counters")
    print("=" * 60)
    
    try:
        app = DualBrainDetectionSystem(MODEL_PATH, VIDEO_PATH)
        app.run()
    except Exception as e:
        print(f"[CRITICAL] Startup error: {e}")
        import traceback
        traceback.print_exc()