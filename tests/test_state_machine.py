"""Tests for backend/core/state_machine.py

These tests exercise the state-machine logic without requiring a real video
feed, YOLO model, or MediaPipe installation.  A lightweight stub for the
MediaPipe landmark object is used throughout.
"""

import sys
import os
from collections import deque
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from core.config import Config
from core.tracker import HandState
from core.state_machine import ProcessStateMachine


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / stubs
# ─────────────────────────────────────────────────────────────────────────────

def _make_landmark(x: float, y: float) -> SimpleNamespace:
    """Minimal MediaPipe landmark stub."""
    return SimpleNamespace(x=x, y=y)


def _make_landmarks(finger_positions: dict) -> SimpleNamespace:
    """
    Build a stub MediaPipe NormalizedLandmarkList with 21 points.
    finger_positions: {index: (x, y)} for override positions; rest default to (0, 0).
    """
    points = [_make_landmark(0.0, 0.0) for _ in range(21)]
    for idx, (x, y) in finger_positions.items():
        points[idx] = _make_landmark(x, y)
    return SimpleNamespace(landmark=points)


def _make_states() -> dict:
    return {"Left": HandState("Left"), "Right": HandState("Right")}


def _fsm(hand_states=None):
    if hand_states is None:
        hand_states = _make_states()
    return ProcessStateMachine(hand_states), hand_states


# ─────────────────────────────────────────────────────────────────────────────
# count_fingers_in_zone
# ─────────────────────────────────────────────────────────────────────────────

class TestCountFingersInZone:
    """Unit tests for the zone-containment helper."""

    def setup_method(self):
        self.fsm, _ = _fsm()
        # Board zone occupies the centre quarter of a 640×480 frame
        self.box = [160, 120, 480, 360]
        self.W, self.H = 640, 480

    def test_no_fingers_when_no_box(self):
        lm = _make_landmarks({4: (0.5, 0.5)})  # thumb inside if box existed
        assert self.fsm.count_fingers_in_zone(lm, None, self.W, self.H) == 0

    def test_no_fingers_outside_zone(self):
        # All fingertips at (0, 0) — top-left corner, outside the box
        lm = _make_landmarks({})
        assert self.fsm.count_fingers_in_zone(lm, self.box, self.W, self.H) == 0

    def test_one_finger_inside(self):
        # Index fingertip (8) at exactly the centre of the box
        cx = (self.box[0] + self.box[2]) / 2 / self.W
        cy = (self.box[1] + self.box[3]) / 2 / self.H
        lm = _make_landmarks({8: (cx, cy)})
        assert self.fsm.count_fingers_in_zone(lm, self.box, self.W, self.H) == 1

    def test_all_five_fingers_inside(self):
        cx = (self.box[0] + self.box[2]) / 2 / self.W
        cy = (self.box[1] + self.box[3]) / 2 / self.H
        lm = _make_landmarks({4: (cx, cy), 8: (cx, cy), 12: (cx, cy),
                               16: (cx, cy), 20: (cx, cy)})
        assert self.fsm.count_fingers_in_zone(lm, self.box, self.W, self.H) == 5

    def test_boundary_pixels_are_inside(self):
        # Exactly on the box edges should count as inside
        x1, y1, x2, y2 = self.box
        lm = _make_landmarks({4: (x1 / self.W, y1 / self.H)})
        assert self.fsm.count_fingers_in_zone(lm, self.box, self.W, self.H) == 1


# ─────────────────────────────────────────────────────────────────────────────
# update() — state transitions
# ─────────────────────────────────────────────────────────────────────────────

class TestStateMachineTransitions:
    """Integration-level tests for the full update() state-transition logic."""

    def setup_method(self):
        self.cfg = Config()
        self.states = _make_states()
        self.fsm = ProcessStateMachine(self.states)
        self.W, self.H = self.cfg.FRAME_WIDTH, self.cfg.FRAME_HEIGHT
        # Board occupies the centre of the frame
        self.box = [
            self.W // 4, self.H // 4,
            3 * self.W // 4, 3 * self.H // 4,
        ]

    def _update(self, hand: str, lm, t: float, box=None):
        return self.fsm.update(
            hand, lm, box if box is not None else self.box,
            t, self.W, self.H
        )

    # ── Transport ────────────────────────────────────────────────────────────

    def test_initial_state_is_transport(self):
        lm = _make_landmarks({})  # all outside zone
        result = self._update("Right", lm, 0.0)
        assert result == "Transport"

    def test_moving_hand_stays_transport(self):
        """High velocity → Transport even after several frames."""
        lm = _make_landmarks({})
        state = self.states["Right"]
        # Fill velocity history above threshold
        for _ in range(self.cfg.VELOCITY_SMOOTHING_FRAMES):
            state.velocity_history.append(self.cfg.VELOCITY_THRESHOLD * 5)

        result = self._update("Right", lm, 1.0)
        assert result == "Transport"

    # ── Delay ─────────────────────────────────────────────────────────────────

    def test_stationary_long_enough_becomes_delay(self):
        """Hand stationary outside zone for > DELAY_INACTIVITY_TIME → Delay."""
        lm = _make_landmarks({})  # no fingers in zone
        state = self.states["Left"]
        # Velocity is 0 (empty history) → stationary

        t0 = 0.0
        self._update("Left", lm, t0)  # starts stationary timer

        t1 = t0 + self.cfg.DELAY_INACTIVITY_TIME + 0.1
        result = self._update("Left", lm, t1)
        assert result == "Delay"

    # ── Operation ─────────────────────────────────────────────────────────────

    def test_finger_in_zone_long_enough_becomes_operation(self):
        """Finger inside zone for > ZONE_STABILITY_TIME → Operation."""
        cx = (self.box[0] + self.box[2]) / 2 / self.W
        cy = (self.box[1] + self.box[3]) / 2 / self.H
        lm = _make_landmarks({8: (cx, cy)})  # index finger inside zone

        t0 = 0.0
        self._update("Right", lm, t0)  # starts board_entry_time

        t1 = t0 + self.cfg.ZONE_STABILITY_TIME + 0.1
        result = self._update("Right", lm, t1)
        assert result == "Operation"

    def test_finger_in_zone_too_briefly_stays_transport(self):
        """Finger inside zone but for less than ZONE_STABILITY_TIME → not Operation."""
        cx = (self.box[0] + self.box[2]) / 2 / self.W
        cy = (self.box[1] + self.box[3]) / 2 / self.H
        lm = _make_landmarks({8: (cx, cy)})

        t0 = 0.0
        self._update("Right", lm, t0)

        t1 = t0 + self.cfg.ZONE_STABILITY_TIME * 0.5  # not long enough
        result = self._update("Right", lm, t1)
        assert result != "Operation"

    # ── State resets ─────────────────────────────────────────────────────────

    def test_motion_resets_delay_timer(self):
        """Resumed motion should clear the stationary timer, preventing Delay."""
        lm = _make_landmarks({})
        state = self.states["Left"]

        # Build up stationary time
        t0 = 0.0
        self._update("Left", lm, t0)

        # Now add velocity → moving
        for _ in range(self.cfg.VELOCITY_SMOOTHING_FRAMES):
            state.velocity_history.append(self.cfg.VELOCITY_THRESHOLD * 10)

        t1 = t0 + self.cfg.DELAY_INACTIVITY_TIME + 1.0
        result = self._update("Left", lm, t1)
        # Even though enough time has passed, moving hand should not be Delay
        assert result == "Transport"

    def test_finger_count_stored_on_state(self):
        """count_fingers_in_zone result should be persisted on the HandState."""
        cx = (self.box[0] + self.box[2]) / 2 / self.W
        cy = (self.box[1] + self.box[3]) / 2 / self.H
        lm = _make_landmarks({4: (cx, cy), 8: (cx, cy)})  # 2 fingers inside

        self._update("Left", lm, 0.0)
        assert self.states["Left"].finger_count_in_board == 2
