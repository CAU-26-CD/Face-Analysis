"""Demo fast-path (DEMO_FAST_MODE) — see app/services/face_analysis/demo_config.py."""

import numpy as np
import pytest

from app.services.face_analysis.demo_config import demo_settings
from app.services.face_analysis.models import VideoFrame
from app.services.face_analysis.video_reader import VideoFrameReader


@pytest.fixture(autouse=True)
def _clear_demo_cache():
    demo_settings.cache_clear()
    yield
    demo_settings.cache_clear()


def test_max_frames_caps_yielded_frames(monkeypatch):
    reader = VideoFrameReader(frame_interval_seconds=0.1, max_frames=3)

    def _fake_read_frames(_path):
        for i in range(100):
            yield VideoFrame(
                timestamp_seconds=i * 0.1,
                frame_index=i,
                frame=np.zeros((8, 8, 3), dtype=np.uint8),
            )

    monkeypatch.setattr(reader, "_read_frames", _fake_read_frames)

    frames = list(reader.read_frames("ignored.mp4"))

    assert len(frames) == 3
    assert [f.frame_index for f in frames] == [0, 1, 2]


def test_max_frames_none_is_unbounded(monkeypatch):
    reader = VideoFrameReader(frame_interval_seconds=0.1)

    def _fake_read_frames(_path):
        for i in range(10):
            yield VideoFrame(
                timestamp_seconds=float(i),
                frame_index=i,
                frame=np.zeros((8, 8, 3), dtype=np.uint8),
            )

    monkeypatch.setattr(reader, "_read_frames", _fake_read_frames)

    assert len(list(reader.read_frames("ignored.mp4"))) == 10


def test_max_frames_rejects_non_positive():
    with pytest.raises(ValueError):
        VideoFrameReader(max_frames=0)


def test_demo_settings_defaults_on(monkeypatch):
    monkeypatch.delenv("DEMO_FAST_MODE", raising=False)
    demo_settings.cache_clear()
    settings = demo_settings()
    assert settings.enabled is True
    assert settings.max_frames >= 1
    assert settings.frame_interval_seconds > 0


def test_demo_settings_can_be_disabled(monkeypatch):
    monkeypatch.setenv("DEMO_FAST_MODE", "0")
    demo_settings.cache_clear()
    assert demo_settings().enabled is False


def test_demo_settings_env_overrides(monkeypatch):
    monkeypatch.setenv("DEMO_FAST_MODE", "1")
    monkeypatch.setenv("DEMO_MAX_FRAMES", "4")
    monkeypatch.setenv("DEMO_FRAME_INTERVAL_SECONDS", "2.5")
    monkeypatch.setenv("DEMO_YOLO_IMGSZ", "256")
    monkeypatch.setenv("DEMO_FACE_DET_SIZE", "224")
    demo_settings.cache_clear()

    settings = demo_settings()

    assert settings.max_frames == 4
    assert settings.frame_interval_seconds == 2.5
    assert settings.yolo_imgsz == 256
    assert settings.face_det_size == 224
