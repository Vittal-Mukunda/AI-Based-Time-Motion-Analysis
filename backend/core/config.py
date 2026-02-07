# Kinetic Analytics V2.0 - Backend

from dataclasses import dataclass, field
from typing import Optional
from collections import deque

@dataclass
class Config:
    """Application configuration."""
    # Frame settings
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    
    # YOLO settings
    YOLO_CONF_THRESHOLD: float = 0.25
    YOLO_IOU_THRESHOLD: float = 0.45
    
    # MediaPipe settings
    MP_MIN_DETECTION_CONF: float = 0.5
    MP_MIN_TRACKING_CONF: float = 0.5
    
    # State machine thresholds
    VELOCITY_THRESHOLD: float = 3.0
    VELOCITY_SMOOTHING_FRAMES: int = 5
    ZONE_STABILITY_TIME: float = 0.15
    DELAY_INACTIVITY_TIME: float = 0.5
    
    # Logging
    LOG_INTERVAL: float = 0.066  # ~15 Hz
    
    # WebSocket
    WS_BROADCAST_INTERVAL: float = 0.033  # ~30 Hz
    
    # Alerts
    IDLE_ALERT_THRESHOLD: float = 5.0

config = Config()
