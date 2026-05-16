import pytest

from app.services.face_analysis.face_person_associator import FacePersonAssociator
from app.services.face_analysis.models import (
    FaceDetection,
    PersonDetection,
    TrackedPerson,
)


def _face(bbox: tuple[float, float, float, float], confidence: float = 0.9) -> FaceDetection:
    return FaceDetection(
        timestamp_seconds=1.0,
        frame_index=10,
        bbox=bbox,
        embedding=[0.0] * 512,
        confidence=confidence,
    )


def _person(track_id: str, bbox: tuple[float, float, float, float]) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        detection=PersonDetection(
            timestamp_seconds=1.0,
            frame_index=10,
            bbox=bbox,
            confidence=0.95,
        ),
    )


def test_face_fully_inside_person_binds_to_that_person():
    associator = FacePersonAssociator()
    face = _face((110, 60, 150, 110))
    person = _person("person_1", (100, 50, 200, 350))

    results = associator.associate([face], [person])

    assert len(results) == 1
    assert results[0].person_track_id == "person_1"
    assert results[0].detection is face


def test_face_with_no_matching_person_returns_none_track_id():
    associator = FacePersonAssociator()
    face = _face((500, 500, 540, 550))
    person = _person("person_1", (0, 0, 100, 200))

    results = associator.associate([face], [person])

    assert results[0].person_track_id is None


def test_face_returned_when_no_persons_present():
    associator = FacePersonAssociator()
    face = _face((100, 100, 140, 150))

    results = associator.associate([face], [])

    assert len(results) == 1
    assert results[0].person_track_id is None


def test_face_inside_two_overlapping_persons_picks_higher_iou():
    associator = FacePersonAssociator()
    face = _face((180, 60, 220, 110))
    loose = _person("person_loose", (100, 50, 400, 400))
    tight = _person("person_tight", (170, 50, 230, 130))

    results = associator.associate([face], [loose, tight])

    assert results[0].person_track_id == "person_tight"


def test_each_face_resolved_independently():
    associator = FacePersonAssociator()
    face_a = _face((110, 60, 150, 110))
    face_b = _face((310, 60, 350, 110))
    person_a = _person("person_a", (100, 50, 200, 350))
    person_b = _person("person_b", (300, 50, 400, 350))

    results = associator.associate([face_a, face_b], [person_b, person_a])

    by_face = {id(r.detection): r.person_track_id for r in results}
    assert by_face[id(face_a)] == "person_a"
    assert by_face[id(face_b)] == "person_b"


def test_face_center_outside_person_bbox_is_rejected_even_with_overlap():
    associator = FacePersonAssociator()
    face_outside = _face((50, 60, 110, 110))
    person = _person("person_1", (100, 50, 200, 350))

    results = associator.associate([face_outside], [person])
    assert results[0].person_track_id is None


def test_min_face_overlap_gates_low_containment():
    associator = FacePersonAssociator(min_face_overlap=0.9)
    face = _face((140, 60, 260, 110))
    person = _person("person_1", (100, 50, 200, 350))

    results = associator.associate([face], [person])
    assert results[0].person_track_id is None


def test_constructor_rejects_invalid_min_face_overlap():
    with pytest.raises(ValueError):
        FacePersonAssociator(min_face_overlap=1.1)
    with pytest.raises(ValueError):
        FacePersonAssociator(min_face_overlap=-0.1)


def test_empty_faces_returns_empty_results():
    associator = FacePersonAssociator()
    person = _person("person_1", (0, 0, 100, 100))
    assert associator.associate([], [person]) == []


def test_crowd_scene_two_faces_inside_separate_persons():
    associator = FacePersonAssociator()
    face_a = _face((120, 60, 160, 110))
    face_b = _face((220, 60, 260, 110))
    person_a = _person("person_a", (100, 50, 200, 350))
    person_b = _person("person_b", (200, 50, 300, 350))

    results = associator.associate([face_a, face_b], [person_a, person_b])

    assert [r.person_track_id for r in results] == ["person_a", "person_b"]


def test_zero_area_face_returns_none():
    associator = FacePersonAssociator()
    degenerate = _face((100, 100, 100, 100))
    person = _person("person_1", (0, 0, 500, 500))

    results = associator.associate([degenerate], [person])
    assert results[0].person_track_id is None
