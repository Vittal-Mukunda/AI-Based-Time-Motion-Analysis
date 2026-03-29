# Robust Data Logger
# CSV logging compatible with Power BI
# Includes Session Headers and Forced Disk Sync

import os
import csv
from datetime import datetime
from typing import Dict, Any, Optional

from .config import config


class RobustDataLogger:
    """CSV logger for hand tracking data with robust disk persistence."""
    
    def __init__(self, file_path: Optional[str] = None):
        if file_path is None:
            # Default: project root / Master_Raw_Data.csv
            _project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            file_path = os.path.join(_project_root, "Master_Raw_Data.csv")

        self.file_path = file_path
        _dir = os.path.dirname(self.file_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_log_time = -1.0
        self.rows_since_flush = 0  # Counter for flush every 10 rows
        
        self.headers = [
            "Row_Number", "Timestamp", "Session_ID", "Frame_ID", "Hand_Side",
            "Current_State", "Velocity_Px", "Fingers_In_Zone",
            "In_Board_Zone_Bool", "Pos_X", "Pos_Y"
        ]
        
        self.current_row_index = 0
        self._initialize_file()
        self.last_state = {"Left": None, "Right": None}
        self.state_start_time = {"Left": 0.0, "Right": 0.0}
        
        # Open file handle in append mode for continuous writing
        self._file_handle = None
        self._csv_writer = None
        self._open_file()
    
    def _initialize_file(self):
        """Initialize or resume from existing CSV file."""
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
                        # Find last data row (skip comments)
                        for line in reversed(lines):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                try:
                                    last_row_data = line.split(',')
                                    self.current_row_index = int(last_row_data[0])
                                    break
                                except (ValueError, IndexError):
                                    continue
                    print(f"[LOGGER] Resuming from Row Number: {self.current_row_index}")
                else:
                    with open(self.file_path, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(self.headers)
            except Exception as e:
                print(f"[LOGGER] Error reading last row: {e}. Starting from 0.")
                self.current_row_index = 0
    
    def _open_file(self):
        """Open file in append mode and write session header."""
        try:
            self._file_handle = open(self.file_path, 'a', newline='')
            self._csv_writer = csv.writer(self._file_handle)
            
            # Write session start header
            session_header = f"# SESSION START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Session ID: {self.session_id})"
            self._file_handle.write(session_header + '\n')
            self._file_handle.flush()
            os.fsync(self._file_handle.fileno())
            
            print(f"[LOGGER] File opened in append mode. Session: {self.session_id}")
        except Exception as e:
            print(f"[LOGGER] Error opening file: {e}")
    
    def _force_flush(self):
        """Force flush to disk with fsync for crash safety."""
        if self._file_handle:
            try:
                self._file_handle.flush()
                os.fsync(self._file_handle.fileno())
            except Exception as e:
                print(f"[LOGGER] Flush error: {e}")
    
    def close(self):
        """Close the file handle with final flush."""
        if self._file_handle:
            try:
                # Write session end marker
                session_end = f"# SESSION END: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                self._file_handle.write(session_end + '\n')
                self._force_flush()
                self._file_handle.close()
            except Exception as e:
                print(f"[LOGGER] Error closing file: {e}")
            finally:
                self._file_handle = None
                self._csv_writer = None
    
    def log_frame(self, frame_id: int, video_time: float, hand_data: Dict[str, Any]):
        """Log hand data for a frame."""
        if (video_time - self.last_log_time) < config.LOG_INTERVAL:
            return
        if not hand_data:
            return
        if not self._csv_writer:
            return
        
        rows_to_write = []
        for hand, data in hand_data.items():
            self.current_row_index += 1
            in_zone_str = "True" if data.get('fingers', 0) > 0 else "False"
            pos = data.get('pos', (0, 0))
            row = [
                self.current_row_index,
                f"{video_time:.3f}",
                self.session_id,
                frame_id,
                hand,
                data.get('state', 'Unknown'),
                f"{data.get('velocity', 0):.1f}",
                data.get('fingers', 0),
                in_zone_str,
                pos[0],
                pos[1]
            ]
            rows_to_write.append(row)
        
        try:
            self._csv_writer.writerows(rows_to_write)
            self.last_log_time = video_time
            
            # Force flush + fsync every 10 rows for crash safety
            self.rows_since_flush += len(rows_to_write)
            if self.rows_since_flush >= 10:
                self._force_flush()
                self.rows_since_flush = 0
        except Exception as e:
            print(f"[CSV ERROR] Could not write to file: {e}")
    
    def update_state_timers(self, video_time: float, hand: str, new_state: str):
        """Track state transitions for timing analysis."""
        old_state = self.last_state[hand]
        if old_state is not None and old_state != new_state:
            self.state_start_time[hand] = video_time
        elif old_state is None:
            self.state_start_time[hand] = video_time
        self.last_state[hand] = new_state
