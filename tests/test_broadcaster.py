"""Tests for backend/services/broadcaster.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.broadcaster import create_broadcast_payload


def _make_payload(**overrides):
    defaults = dict(
        frame_id=1,
        video_time=0.033,
        fps=30.0,
        board_box=None,
        hands_info={},
        counters={
            "Left_Operation": 0.0, "Left_Transport": 1.5, "Left_Delay": 0.5,
            "Right_Operation": 0.0, "Right_Transport": 1.5, "Right_Delay": 0.5,
        },
        alerts=[],
        image=None,
    )
    defaults.update(overrides)
    return create_broadcast_payload(**defaults)


def test_payload_keys():
    """Payload should contain all required top-level keys."""
    p = _make_payload()
    assert "frame_id" in p
    assert "video_time" in p
    assert "fps" in p
    assert "board_zone" in p
    assert "hands" in p
    assert "counters" in p
    assert "alerts" in p
    assert "image" in p


def test_payload_board_zone_none():
    """board_zone should be None when no detection."""
    p = _make_payload(board_box=None)
    assert p["board_zone"] is None


def test_payload_board_zone_present():
    """board_zone should have x1/y1/x2/y2 when detection provided."""
    p = _make_payload(board_box=[10, 20, 300, 400])
    bz = p["board_zone"]
    assert bz is not None
    assert bz == {"x1": 10, "y1": 20, "x2": 300, "y2": 400}


def test_payload_both_hands_present():
    """Both Left and Right hands should appear even if neither was detected."""
    p = _make_payload(hands_info={})
    assert "Left" in p["hands"]
    assert "Right" in p["hands"]


def test_payload_invisible_hand_defaults():
    """Undetected hands should have visible=False and safe default values."""
    p = _make_payload(hands_info={})
    for side in ("Left", "Right"):
        h = p["hands"][side]
        assert h["visible"] is False
        assert h["velocity"] == 0
        assert h["fingers_in_zone"] == 0
        assert h["center"] == [0, 0]
        assert h["landmarks"] == []


def test_payload_visible_hand():
    """Detected hand should be marked visible with the correct data."""
    hands_info = {
        "Right": {
            "state": "Operation",
            "velocity": 5.2,
            "fingers": 3,
            "pos": (640, 360),
            "landmarks": [[0.5, 0.5]] * 21,
        }
    }
    p = _make_payload(hands_info=hands_info)
    h = p["hands"]["Right"]
    assert h["visible"] is True
    assert h["state"] == "Operation"
    assert h["velocity"] == 5.2
    assert h["fingers_in_zone"] == 3
    assert h["center"] == [640, 360]
    assert len(h["landmarks"]) == 21


def test_payload_counters_rounded():
    """Counter values should be rounded to 2 decimal places."""
    counters = {"Left_Operation": 1.23456, "Left_Transport": 0.0, "Left_Delay": 0.0,
                "Right_Operation": 0.0, "Right_Transport": 0.0, "Right_Delay": 9.99999}
    p = _make_payload(counters=counters)
    assert p["counters"]["Left_Operation"] == 1.23
    assert p["counters"]["Right_Delay"] == 10.0


def test_payload_fps_rounded():
    """FPS should be rounded to 1 decimal place."""
    p = _make_payload(fps=29.9876)
    assert p["fps"] == 30.0


def test_payload_video_time_rounded():
    """video_time should be rounded to 3 decimal places."""
    p = _make_payload(video_time=1.2345678)
    assert p["video_time"] == 1.235


def test_payload_image_passthrough():
    """Base64 image string should pass through unchanged."""
    fake_b64 = "abc123=="
    p = _make_payload(image=fake_b64)
    assert p["image"] == fake_b64
