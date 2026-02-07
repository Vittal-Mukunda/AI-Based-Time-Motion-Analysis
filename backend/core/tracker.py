# Hand Tracking Module
# Extracts hand landmarks using MediaPipe
# Includes Teleportation Filter and EMA Smoothing

import cv2
import numpy as np
import math
import mediapipe as mp
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Dict, Any, Tuple

from .config import config

# Teleportation Filter Constants
TELEPORT_THRESHOLD = 300  # pixels - discard if distance > this in single frame
EMA_ALPHA = 0.7  # Exponential Moving Average smoothing factor


@dataclass
class HandState:
    """State tracking for a single hand."""
    label: str
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=config.VELOCITY_SMOOTHING_FRAMES))
    current_process: str = "Transport"
    stationary_start_time: Optional[float] = None
    board_entry_time: Optional[float] = None
    target_state_candidate: Optional[str] = None
    finger_count_in_board: int = 0
    last_seen_frame: int = 0
    # Smoothed position for EMA
    smoothed_position: Optional[Tuple[float, float]] = None
    last_valid_position: Optional[Tuple[int, int]] = None


class HandTracker:
    """MediaPipe-based hand tracking with Teleportation Filter."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands_model = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.MP_MIN_DETECTION_CONF,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONF
        )
        self.hand_states = {"Left": HandState("Left"), "Right": HandState("Right")}
    
    def _apply_teleport_filter(self, state: HandState, raw_center: Tuple[int, int]) -> Tuple[int, int]:
        """
        Apply teleportation filter and EMA smoothing.
        
        - If distance from last valid position > TELEPORT_THRESHOLD, reject as noise
        - Apply EMA smoothing to valid movements
        """
        if state.last_valid_position is None:
            # First detection - accept and initialize
            state.last_valid_position = raw_center
            state.smoothed_position = (float(raw_center[0]), float(raw_center[1]))
            return raw_center
        
        # Calculate distance from last valid position
        dx = raw_center[0] - state.last_valid_position[0]
        dy = raw_center[1] - state.last_valid_position[1]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > TELEPORT_THRESHOLD:
            # Teleportation detected! Use last valid position
            print(f"[TELEPORT FILTER] {state.label} hand teleported {distance:.0f}px - REJECTED")
            return state.last_valid_position
        
        # Valid movement - apply EMA smoothing
        if state.smoothed_position is None:
            state.smoothed_position = (float(raw_center[0]), float(raw_center[1]))
        else:
            # EMA: smoothed = alpha * new + (1-alpha) * old
            smoothed_x = EMA_ALPHA * raw_center[0] + (1 - EMA_ALPHA) * state.smoothed_position[0]
            smoothed_y = EMA_ALPHA * raw_center[1] + (1 - EMA_ALPHA) * state.smoothed_position[1]
            state.smoothed_position = (smoothed_x, smoothed_y)
        
        # Update last valid position
        filtered_center = (int(state.smoothed_position[0]), int(state.smoothed_position[1]))
        state.last_valid_position = filtered_center
        
        return filtered_center
    
    def calculate_velocity(self, hand_label: str) -> float:
        """Get smoothed velocity for a hand."""
        state = self.hand_states[hand_label]
        return float(np.mean(list(state.velocity_history))) if state.velocity_history else 0.0
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """
        Process a frame and extract hand data.
        Returns dict with hand info including landmarks as normalized coordinates.
        Applies teleportation filter and EMA smoothing to positions.
        """
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
                
                # Raw center before filtering
                raw_center = (int(np.mean(x_vals)), int(np.mean(y_vals)))
                
                state = self.hand_states[label]
                state.last_seen_frame = frame_id
                
                # Apply teleportation filter and EMA smoothing
                filtered_center = self._apply_teleport_filter(state, raw_center)
                
                # Use filtered center for position history
                state.position_history.append(filtered_center)
                
                if len(state.position_history) >= 2:
                    prev = state.position_history[-2]
                    dist = math.hypot(filtered_center[0] - prev[0], filtered_center[1] - prev[1])
                    state.velocity_history.append(dist)
                
                # Calculate bounding box from landmarks
                box = [
                    int(min(x_vals) - 20), 
                    int(min(y_vals) - 20), 
                    int(max(x_vals) + 20), 
                    int(max(y_vals) + 20)
                ]
                
                # Convert landmarks to list of [x, y] normalized coordinates
                landmarks_list = [[lm.x, lm.y] for lm in landmarks.landmark]
                
                detected[label] = {
                    'box': box,
                    'landmarks': landmarks_list,  # Now a simple list, not MediaPipe object
                    'landmarks_raw': landmarks,   # Keep raw for state machine
                    'center': filtered_center,    # Use filtered center
                    'visible': True
                }
        
        return detected
    
    def close(self):
        """Release MediaPipe resources."""
        self.hands_model.close()
