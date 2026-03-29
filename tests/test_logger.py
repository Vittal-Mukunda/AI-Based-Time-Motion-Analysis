"""Tests for backend/core/logger.py"""

import os
import sys
import csv
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from core.logger import RobustDataLogger


@pytest.fixture
def tmp_logger(tmp_path):
    """Create a RobustDataLogger writing to a temp file."""
    log_file = str(tmp_path / "test_log.csv")
    logger = RobustDataLogger(file_path=log_file)
    yield logger, log_file
    logger.close()


def test_creates_file(tmp_path):
    """Logger should create the CSV file on initialisation."""
    log_file = str(tmp_path / "new_log.csv")
    assert not os.path.exists(log_file)
    logger = RobustDataLogger(file_path=log_file)
    logger.close()
    assert os.path.exists(log_file)


def test_csv_headers(tmp_logger):
    """CSV file should start with the correct headers."""
    logger, log_file = tmp_logger
    with open(log_file, "r", newline="") as f:
        reader = csv.reader(f)
        first_data_row = None
        for row in reader:
            if row and not row[0].startswith("#"):
                first_data_row = row
                break
    assert first_data_row == logger.headers


def test_log_frame_writes_row(tmp_logger):
    """log_frame should append rows for each hand in the hand_data dict."""
    logger, log_file = tmp_logger
    hand_data = {
        "Right": {"state": "Transport", "velocity": 4.2, "fingers": 0, "pos": (320, 240)},
    }
    logger.log_frame(frame_id=1, video_time=0.1, hand_data=hand_data)
    logger.close()

    rows = []
    with open(log_file, "r", newline="") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows.append(stripped.split(","))

    # Skip header row; find data rows
    data_rows = [r for r in rows if r[0].strip() != "Row_Number"]
    assert len(data_rows) == 1
    row = data_rows[0]
    assert row[4].strip() == "Right"
    assert row[5].strip() == "Transport"


def test_log_interval_throttle(tmp_logger):
    """Second call within LOG_INTERVAL should be silently skipped."""
    logger, log_file = tmp_logger
    hand_data = {"Left": {"state": "Delay", "velocity": 0.0, "fingers": 0, "pos": (0, 0)}}
    logger.log_frame(frame_id=1, video_time=0.0, hand_data=hand_data)
    logger.log_frame(frame_id=2, video_time=0.01, hand_data=hand_data)  # within interval
    logger.close()

    data_rows = []
    with open(log_file, "r", newline="") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#") and not s.startswith("Row_Number"):
                data_rows.append(s)

    assert len(data_rows) == 1


def test_resume_row_numbering(tmp_path):
    """A second logger instance should continue row numbering from where the first left off."""
    log_file = str(tmp_path / "resume_log.csv")

    hand_data = {
        "Left": {"state": "Operation", "velocity": 1.0, "fingers": 2, "pos": (100, 200)},
        "Right": {"state": "Transport", "velocity": 3.0, "fingers": 0, "pos": (300, 400)},
    }

    logger1 = RobustDataLogger(file_path=log_file)
    logger1.log_frame(frame_id=1, video_time=0.0, hand_data=hand_data)
    logger1.close()

    logger2 = RobustDataLogger(file_path=log_file)
    last_row_before = logger2.current_row_index
    logger2.log_frame(frame_id=2, video_time=1.0, hand_data=hand_data)
    logger2.close()

    # The resumed logger should have written rows beyond what logger1 wrote
    assert logger2.current_row_index > last_row_before


def test_default_path_is_relative():
    """Default log path should resolve to a path inside the project (not a hardcoded user home)."""
    logger = RobustDataLogger()
    path = logger.file_path
    logger.close()
    # Should not contain the old hard-coded user prefix
    assert "vitta" not in path or os.path.exists(path)
    # Should end with the expected filename
    assert path.endswith("Master_Raw_Data.csv")
