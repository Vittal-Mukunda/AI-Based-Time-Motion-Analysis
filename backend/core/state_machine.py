# Process State Machine
# Determines hand state: Operation, Transport, or Delay

import numpy as np
from typing import Optional, Dict, Any

from .config import config
from .tracker import HandState


class ProcessStateMachine:
    """
    State machine for determining hand process state.
    States: Operation (working in zone), Transport (moving), Delay (idle)
    """
    
    def __init__(self, hand_states: Dict[str, HandState]):
        self.hand_states = hand_states
    
    def count_fingers_in_zone(self, landmarks_raw, box: Optional[list], w: int, h: int) -> int:
        """Count how many fingertips are inside the board zone."""
        if box is None:
            return 0
        
        x1, y1, x2, y2 = box
        fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        count = 0
        
        for idx in fingertip_indices:
            lm = landmarks_raw.landmark[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            if x1 <= px <= x2 and y1 <= py <= y2:
                count += 1
        
        return count
    
    def update(self, hand_lbl: str, landmarks_raw, board_box: Optional[list], 
               video_time: float, w: int, h: int) -> str:
        """
        Update state machine and return current process state.
        
        Args:
            hand_lbl: "Left" or "Right"
            landmarks_raw: MediaPipe landmarks object
            board_box: [x1, y1, x2, y2] or None
            video_time: Current video timestamp
            w, h: Frame dimensions
        
        Returns:
            State string: "Operation", "Transport", or "Delay"
        """
        state = self.hand_states[hand_lbl]
        velocity = np.mean(list(state.velocity_history)) if state.velocity_history else 0
        is_moving = velocity > config.VELOCITY_THRESHOLD
        
        fingers_in = self.count_fingers_in_zone(landmarks_raw, board_box, w, h)
        state.finger_count_in_board = fingers_in
        candidate = "Operation" if fingers_in >= 1 else None
        
        # Update stationary timer
        if is_moving:
            state.stationary_start_time = None
        elif state.stationary_start_time is None:
            state.stationary_start_time = video_time
        
        # Update zone entry tracking
        if candidate != state.target_state_candidate:
            state.target_state_candidate = candidate
            state.board_entry_time = video_time if candidate else None
        
        # Determine final state
        new_process = "Transport"
        
        if state.stationary_start_time and (video_time - state.stationary_start_time > config.DELAY_INACTIVITY_TIME):
            new_process = "Delay"
        elif state.target_state_candidate == "Operation" and state.board_entry_time:
            if (video_time - state.board_entry_time > config.ZONE_STABILITY_TIME):
                new_process = "Operation"
        
        state.current_process = new_process
        return new_process
