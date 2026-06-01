import numpy as np
import pytest

from app.services.face_analysis.models import PersonDetection, VideoFrame
from app.services.face_analysis.person_tracker import PersonTracker


def _frame(frame_index: int, fps: float = 10.0) -> VideoFrame:
    return VideoFrame(
        timestamp_seconds=frame_index / fps,
        frame_index=frame_index,
        frame=np.zeros((720, 1280, 3), dtype=np.uint8),
    )


def _person_at(frame_index: int, x: float, y: float, w: float = 80.0, h: float = 200.0, fps: float = 10.0) -> PersonDetection:
    return PersonDetection(
        timestamp_seconds=frame_index / fps,
        frame_index=frame_index,
        bbox=(x, y, x + w, y + h),
        confidence=0.9,
    )


def _run_sequence(
    tracker: PersonTracker,
    detections_by_frame: list[list[PersonDetection]],
) -> list[list[str]]:
    """Run the tracker over a list of per-frame detections, return track IDs per frame."""
    track_ids: list[list[str]] = []
    for frame_index, detections in enumerate(detections_by_frame):
        tracked = tracker.update(_frame(frame_index), detections)
        track_ids.append([t.track_id for t in tracked])
    return track_ids


def test_single_moving_person_keeps_same_track_id_across_frames():
    tracker = PersonTracker(with_reid=False)
    detections_by_frame = [
        [_person_at(i, x=100.0 + 5.0 * i, y=200.0)] for i in range(15)
    ]
    track_ids = _run_sequence(tracker, detections_by_frame)
    assigned = [ids[0] for ids in track_ids if ids]
    assert assigned, "expected the tracker to confirm the track within ~15 frames"
    assert len(set(assigned)) == 1, f"track id should remain stable, got {set(assigned)}"


def test_two_people_get_distinct_track_ids():
    tracker = PersonTracker(with_reid=False)
    detections_by_frame = [
        [
            _person_at(i, x=100.0 + 2.0 * i, y=200.0),
            _person_at(i, x=800.0 - 2.0 * i, y=200.0),
        ]
        for i in range(15)
    ]
    track_ids = _run_sequence(tracker, detections_by_frame)
    confirmed_frames = [ids for ids in track_ids if len(ids) == 2]
    assert confirmed_frames, "expected ByteTrack to activate both tracks"
    last = confirmed_frames[-1]
    assert last[0] != last[1], "two people in the same frame must get distinct ids"


def test_short_occlusion_within_track_buffer_recovers_same_id():
    tracker = PersonTracker(track_buffer=50, with_reid=False)
    sequence = []
    for i in range(0, 15):
        sequence.append([_person_at(i, x=100.0 + 5.0 * i, y=200.0)])
    for _ in range(5):
        sequence.append([])
    for i in range(20, 30):
        sequence.append([_person_at(i, x=100.0 + 5.0 * i, y=200.0)])

    track_ids = _run_sequence(tracker, sequence)
    before = [ids[0] for ids in track_ids[:15] if ids]
    after = [ids[0] for ids in track_ids[20:] if ids]
    assert before, "track should activate before occlusion"
    assert after, "track should re-activate after occlusion"
    assert before[-1] == after[-1], "same person within track_buffer should keep same id"


def test_reset_drops_internal_state():
    tracker = PersonTracker(with_reid=False)
    _run_sequence(tracker, [[_person_at(i, x=100.0 + 5.0 * i, y=200.0)] for i in range(15)])
    assert tracker._tracker is not None
    tracker.reset()
    assert tracker._tracker is None


def test_empty_detections_returns_empty():
    tracker = PersonTracker(with_reid=False)
    assert tracker.update(_frame(0), []) == []


def test_constructor_rejects_invalid_args():
    with pytest.raises(ValueError):
        PersonTracker(track_buffer=0)
    with pytest.raises(ValueError):
        PersonTracker(track_high_thresh=1.5)
