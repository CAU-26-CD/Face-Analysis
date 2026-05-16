import pytest

from app.services.face_analysis.exemplars import quality_score, select_top_k_diverse
from app.services.face_analysis.models import FaceDetection


def _detection(t: float, bbox: tuple[float, float, float, float], confidence: float) -> FaceDetection:
    return FaceDetection(
        timestamp_seconds=t,
        frame_index=int(t * 10),
        bbox=bbox,
        embedding=[0.0] * 512,
        confidence=confidence,
    )


def test_quality_score_rewards_higher_confidence_and_bigger_bbox():
    small = _detection(0.0, (0, 0, 20, 20), 0.9)
    big = _detection(0.0, (0, 0, 100, 100), 0.9)
    assert quality_score(big) > quality_score(small)

    low_conf = _detection(0.0, (0, 0, 100, 100), 0.5)
    high_conf = _detection(0.0, (0, 0, 100, 100), 0.95)
    assert quality_score(high_conf) > quality_score(low_conf)


def test_select_top_k_returns_diverse_picks_across_time():
    detections = []
    for second in range(10):
        detections.append(_detection(float(second), (0, 0, 100, 100), 0.9))
    picks = select_top_k_diverse(detections, 3)
    assert len(picks) == 3
    timestamps = [p.timestamp_seconds for p in picks]
    assert timestamps == sorted(timestamps)
    assert max(timestamps) - min(timestamps) >= 5.0


def test_select_top_k_prefers_higher_quality_within_a_bucket():
    detections = [
        _detection(0.0, (0, 0, 100, 100), 0.6),
        _detection(0.1, (0, 0, 100, 100), 0.95),
        _detection(0.2, (0, 0, 100, 100), 0.7),
    ]
    picks = select_top_k_diverse(detections, 1)
    assert picks[0].confidence == pytest.approx(0.95)


def test_select_top_k_with_fewer_detections_than_k_returns_all_available():
    detections = [
        _detection(0.0, (0, 0, 100, 100), 0.9),
        _detection(5.0, (0, 0, 100, 100), 0.9),
    ]
    picks = select_top_k_diverse(detections, 5)
    assert len(picks) == 2


def test_select_top_k_handles_empty_input():
    assert select_top_k_diverse([], 3) == []


def test_select_top_k_rejects_invalid_k():
    with pytest.raises(ValueError):
        select_top_k_diverse([], 0)


def test_select_top_k_handles_single_timestamp_burst():
    detections = [
        _detection(2.5, (0, 0, 100, 100), 0.7),
        _detection(2.5, (0, 0, 100, 100), 0.95),
        _detection(2.5, (0, 0, 100, 100), 0.8),
    ]
    picks = select_top_k_diverse(detections, 3)
    assert len(picks) >= 1
    assert any(p.confidence == pytest.approx(0.95) for p in picks)
