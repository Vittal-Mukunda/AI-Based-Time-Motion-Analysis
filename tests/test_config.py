"""Tests for backend/core/config.py"""

import sys
import os

# Allow importing backend package without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from core.config import Config, config


def test_config_defaults():
    """Config should expose correct default values."""
    c = Config()
    assert c.FRAME_WIDTH == 1280
    assert c.FRAME_HEIGHT == 720
    assert 0 < c.YOLO_CONF_THRESHOLD < 1
    assert 0 < c.YOLO_IOU_THRESHOLD < 1
    assert 0 < c.MP_MIN_DETECTION_CONF <= 1
    assert 0 < c.MP_MIN_TRACKING_CONF <= 1
    assert c.VELOCITY_THRESHOLD > 0
    assert c.VELOCITY_SMOOTHING_FRAMES > 0
    assert c.ZONE_STABILITY_TIME > 0
    assert c.DELAY_INACTIVITY_TIME > 0
    assert c.LOG_INTERVAL > 0
    assert c.WS_BROADCAST_INTERVAL > 0
    assert c.IDLE_ALERT_THRESHOLD > 0


def test_config_singleton():
    """Module-level config object should be a Config instance."""
    assert isinstance(config, Config)


def test_config_broadcast_faster_than_log():
    """WebSocket broadcast rate should be >= logging rate."""
    c = Config()
    assert c.WS_BROADCAST_INTERVAL <= c.LOG_INTERVAL
