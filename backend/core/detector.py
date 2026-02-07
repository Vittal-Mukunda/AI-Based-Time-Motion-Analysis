# YOLO Board Detection and Video Capture

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Optional, Tuple

from .config import config


class BoardDetector:
    """YOLO-based board zone detection."""
    
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DETECTOR] Loading YOLO on {self.device}...")
        self.model = YOLO(model_path)
    
    def detect(self, frame: np.ndarray) -> Optional[list]:
        """
        Detect board zone in frame.
        
        Returns:
            [x1, y1, x2, y2] bounding box or None
        """
        results = self.model.predict(
            frame, 
            conf=config.YOLO_CONF_THRESHOLD, 
            verbose=False, 
            device=self.device
        )
        
        if results[0].boxes:
            for box in results[0].boxes:
                if int(box.cls) == 0:  # Assuming class 0 is the board
                    return box.xyxy[0].cpu().numpy().astype(int).tolist()
        
        return None


class VideoCapture:
    """Video file or camera capture wrapper."""
    
    def __init__(self, source: str):
        """
        Initialize video capture.
        
        Args:
            source: File path or camera index (as string, e.g., "0")
        """
        # Try to parse as camera index
        try:
            source_int = int(source)
            self.cap = cv2.VideoCapture(source_int)
            self.is_camera = True
        except ValueError:
            self.cap = cv2.VideoCapture(source)
            self.is_camera = False
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or np.isnan(self.fps):
            self.fps = 30.0
        
        self.frame_duration = 1.0 / self.fps
        print(f"[VIDEO] Opened source at {self.fps:.1f} FPS")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read and resize a frame."""
        success, frame = self.cap.read()
        if success:
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        return success, frame
    
    def release(self):
        """Release video capture."""
        self.cap.release()
    
    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()
