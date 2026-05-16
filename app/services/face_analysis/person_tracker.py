from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from app.services.face_analysis.models import (
    PersonDetection,
    TrackedPerson,
    VideoFrame,
)


@dataclass
class _ByteTrackInputs:
    """Minimal adapter that mimics the parts of an Ultralytics ``Boxes`` object
    that BYTETracker actually reads (``xywh``, ``conf``, ``cls``, ``len()``,
    boolean-mask indexing).
    """

    xywh: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return int(self.xywh.shape[0])

    def __getitem__(self, mask) -> "_ByteTrackInputs":
        return _ByteTrackInputs(
            xywh=self.xywh[mask],
            conf=self.conf[mask],
            cls=self.cls[mask],
        )


class PersonTracker:
    """Wraps Ultralytics' ByteTrack to assign stable track IDs to person bboxes.

    Tracking is bbox-only (IoU + Kalman motion), which means a person whose
    face turns to profile or is occluded by hair still keeps the same
    track_id as long as their body bbox is detected. Face identity is layered
    on top of this in later stages.
    """

    def __init__(
        self,
        track_buffer: int = 100,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
    ):
        if track_buffer < 1:
            raise ValueError("track_buffer must be >= 1")
        for name, value in (
            ("track_high_thresh", track_high_thresh),
            ("track_low_thresh", track_low_thresh),
            ("new_track_thresh", new_track_thresh),
            ("match_thresh", match_thresh),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        self._args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )
        self._tracker: Any = None

    def update(
        self,
        video_frame: VideoFrame,
        detections: list[PersonDetection],
    ) -> list[TrackedPerson]:
        tracker = self._get_tracker()

        if not detections:
            empty = _ByteTrackInputs(
                xywh=np.zeros((0, 4), dtype=np.float32),
                conf=np.zeros((0,), dtype=np.float32),
                cls=np.zeros((0,), dtype=np.float32),
            )
            tracker.update(empty)
            return []

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        xywh = np.column_stack(
            [
                (xyxy[:, 0] + xyxy[:, 2]) / 2.0,
                (xyxy[:, 1] + xyxy[:, 3]) / 2.0,
                xyxy[:, 2] - xyxy[:, 0],
                xyxy[:, 3] - xyxy[:, 1],
            ]
        ).astype(np.float32)
        conf = np.array([d.confidence for d in detections], dtype=np.float32)
        cls = np.zeros(len(detections), dtype=np.float32)

        inputs = _ByteTrackInputs(xywh=xywh, conf=conf, cls=cls)
        tracks = tracker.update(inputs)
        if tracks is None or len(tracks) == 0:
            return []

        tracked: list[TrackedPerson] = []
        for row in tracks:
            x1, y1, x2, y2 = (float(v) for v in row[0:4])
            track_id = int(row[4])
            score = float(row[5])
            tracked.append(
                TrackedPerson(
                    track_id=f"person_{track_id}",
                    detection=PersonDetection(
                        timestamp_seconds=float(video_frame.timestamp_seconds),
                        frame_index=int(video_frame.frame_index),
                        bbox=(x1, y1, x2, y2),
                        confidence=score,
                    ),
                )
            )
        return tracked

    def reset(self) -> None:
        """Drop tracker state so the next ``update`` starts a fresh video."""
        self._tracker = None

    def _get_tracker(self):
        if self._tracker is not None:
            return self._tracker
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._tracker = BYTETracker(self._args)
        return self._tracker
