"""Integration test for the rewired analyzer pipeline.

Uses fake detectors/trackers/associators so the test runs without YOLO or
InsightFace weights. The goal is to verify the orchestration: persons get
buffered by track_id, faces get attached via the associator, PersonTracklets
get built, and the clusterer/matcher receive the right inputs.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from app.services.face_analysis.analyzer import FaceVideoAnalyzer
from app.services.face_analysis.actor_matcher import ActorMatcher
from app.services.face_analysis.face_person_associator import FacePersonAssociator
from app.services.face_analysis.models import (
    FaceDetection,
    PersonDetection,
    TrackedPerson,
    VideoFrame,
)
from app.services.face_analysis.timeline import AppearanceTimelineBuilder
from app.services.face_analysis.tracklet_clusterer import TrackletClusterer


@dataclass
class _FrameScript:
    timestamp: float
    frame_index: int
    persons: list[tuple[str, tuple[float, float, float, float]]]
    faces: list[tuple[tuple[float, float, float, float], list[float], float]]


class _ScriptedFrameReader:
    def __init__(self, frames: list[_FrameScript]):
        self.frames = frames

    def read_frames(self, _video_path: Path):
        for f in self.frames:
            yield VideoFrame(
                timestamp_seconds=f.timestamp,
                frame_index=f.frame_index,
                frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            )


class _ScriptedPersonDetector:
    def __init__(self, frames: list[_FrameScript]):
        self.by_index = {f.frame_index: f for f in frames}

    def detect(self, video_frame: VideoFrame) -> list[PersonDetection]:
        script = self.by_index.get(video_frame.frame_index)
        if script is None:
            return []
        return [
            PersonDetection(
                timestamp_seconds=video_frame.timestamp_seconds,
                frame_index=video_frame.frame_index,
                bbox=bbox,
                confidence=0.9,
            )
            for _track_id, bbox in script.persons
        ]


class _ScriptedPersonTracker:
    """Returns track ids exactly as the script dictates — bypasses real
    ByteTrack so the test isolates orchestration, not tracking math."""

    def __init__(self, frames: list[_FrameScript]):
        self.by_index = {f.frame_index: f for f in frames}

    def update(self, video_frame, _detections):
        script = self.by_index.get(video_frame.frame_index)
        if script is None:
            return []
        return [
            TrackedPerson(
                track_id=track_id,
                detection=PersonDetection(
                    timestamp_seconds=video_frame.timestamp_seconds,
                    frame_index=video_frame.frame_index,
                    bbox=bbox,
                    confidence=0.9,
                ),
            )
            for track_id, bbox in script.persons
        ]

    def reset(self) -> None:
        pass


class _ScriptedFaceDetector:
    def __init__(self, frames: list[_FrameScript]):
        self.by_index = {f.frame_index: f for f in frames}

    def detect(self, video_frame: VideoFrame) -> list[FaceDetection]:
        script = self.by_index.get(video_frame.frame_index)
        if script is None:
            return []
        return [
            FaceDetection(
                timestamp_seconds=video_frame.timestamp_seconds,
                frame_index=video_frame.frame_index,
                bbox=bbox,
                embedding=list(embedding),
                confidence=confidence,
            )
            for bbox, embedding, confidence in script.faces
        ]


def _make_analyzer(frames: list[_FrameScript]) -> FaceVideoAnalyzer:
    return FaceVideoAnalyzer(
        frame_reader=_ScriptedFrameReader(frames),
        person_detector=_ScriptedPersonDetector(frames),
        person_tracker=_ScriptedPersonTracker(frames),
        face_detector=_ScriptedFaceDetector(frames),
        face_associator=FacePersonAssociator(),
        tracklet_clusterer=TrackletClusterer(similarity_threshold=0.5),
        actor_matcher=ActorMatcher(),
        timeline_builder=AppearanceTimelineBuilder(max_gap_seconds=2.0),
    )


def test_pipeline_groups_face_observations_under_one_person_track():
    embedding = [1.0, 0.0, 0.0]
    frames = [
        _FrameScript(
            timestamp=i * 0.1,
            frame_index=i,
            persons=[("person_1", (100, 50, 300, 500))],
            faces=[((140, 60, 220, 160), embedding, 0.95)],
        )
        for i in range(30)
    ]

    result = _make_analyzer(frames).analyze(Path("dummy.mp4"))

    assert len(result.new_candidates) == 1
    candidate = result.new_candidates[0]
    assert candidate.start_seconds == 0.0
    assert candidate.end_seconds == pytest.approx(2.9)
    assert candidate.detection_count == 30


def test_pipeline_keeps_two_people_in_separate_clusters():
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0]
    frames = [
        _FrameScript(
            timestamp=i * 0.1,
            frame_index=i,
            persons=[
                ("person_a", (100, 50, 300, 500)),
                ("person_b", (500, 50, 700, 500)),
            ],
            faces=[
                ((140, 60, 220, 160), emb_a, 0.95),
                ((540, 60, 620, 160), emb_b, 0.95),
            ],
        )
        for i in range(20)
    ]

    result = _make_analyzer(frames).analyze(Path("dummy.mp4"))

    assert len(result.new_candidates) == 2
    embeddings = sorted([c.embedding for c in result.new_candidates], key=lambda e: e[0])
    assert embeddings[0][1] == 1.0  # person_b (emb_b: [0,1,0])
    assert embeddings[1][0] == 1.0  # person_a (emb_a: [1,0,0])


def test_pipeline_drops_unassociated_face_from_identity_but_still_tracks_person():
    embedding = [1.0, 0.0, 0.0]
    frames = [
        _FrameScript(
            timestamp=i * 0.1,
            frame_index=i,
            persons=[("person_1", (100, 50, 300, 500))],
            faces=[((900, 900, 940, 950), embedding, 0.95)],
        )
        for i in range(10)
    ]

    result = _make_analyzer(frames).analyze(Path("dummy.mp4"))

    assert result.new_candidates == []
    assert result.appearances == []


def test_pipeline_appearance_intervals_use_person_observations():
    embedding = [1.0, 0.0, 0.0]
    early = [
        _FrameScript(
            timestamp=i * 0.1,
            frame_index=i,
            persons=[("person_1", (100, 50, 300, 500))],
            faces=[((140, 60, 220, 160), embedding, 0.95)],
        )
        for i in range(10)
    ]
    gap = [
        _FrameScript(
            timestamp=2.0 + i * 0.1,
            frame_index=20 + i,
            persons=[],
            faces=[],
        )
        for i in range(50)
    ]
    late = [
        _FrameScript(
            timestamp=7.0 + i * 0.1,
            frame_index=70 + i,
            persons=[("person_1", (100, 50, 300, 500))],
            faces=[((140, 60, 220, 160), embedding, 0.95)],
        )
        for i in range(10)
    ]

    result = _make_analyzer(early + gap + late).analyze(Path("dummy.mp4"))

    appearances = result.appearances
    assert len(appearances) == 2
    assert appearances[0].start_seconds == 0.0
    assert appearances[1].start_seconds == 7.0
