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
class _BoxTrackInputs:
    """Minimal adapter that mimics the parts of an Ultralytics ``Boxes`` object
    that BYTETracker / BOTSORT actually read (``xywh``, ``xyxy``, ``conf``,
    ``cls``, ``len()``, boolean-mask indexing). Both trackers consume the same
    detection container; BOTSORT additionally reaches into ``xyxy`` from its
    GMC code path, so we expose that too.
    """

    xywh: np.ndarray
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return int(self.xywh.shape[0])

    def __getitem__(self, mask) -> "_BoxTrackInputs":
        return _BoxTrackInputs(
            xywh=self.xywh[mask],
            xyxy=self.xyxy[mask],
            conf=self.conf[mask],
            cls=self.cls[mask],
        )


class PersonTracker:
    """Wraps Ultralytics' BoT-SORT to assign stable track IDs to person bboxes.

    BoT-SORT extends ByteTrack with two pieces that matter for our use case:

    - **GMC (global motion compensation).** Estimates per-frame camera warp
      via sparse optical flow and applies it to existing Kalman predictions,
      so a panning shot doesn't blow up the track-detection IoU.
    - **ReID (appearance features).** A lightweight YOLO encoder turns each
      person crop into a feature vector. Track-to-detection cost combines
      IoU with cosine similarity on these features, so a person whose bbox
      drifts during fast motion or who briefly leaves and re-enters frame
      can still be re-associated to their existing track. This is the
      "follow the actor with a box" signal — it works whether the face is
      visible or not.

    Tracking is still bbox-driven; ReID is a tie-breaker that prevents
    spurious ID flips. Per-tracklet face identity is layered on top for
    cross-video actor matching.
    """

    def __init__(
        self,
        track_buffer: int = 100,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
        with_reid: bool = True,
        reid_model: str = "yolo11n-cls.pt",
        gmc_method: str = "sparseOptFlow",
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
    ):
        if track_buffer < 1:
            raise ValueError("track_buffer must be >= 1")
        for name, value in (
            ("track_high_thresh", track_high_thresh),
            ("track_low_thresh", track_low_thresh),
            ("new_track_thresh", new_track_thresh),
            ("match_thresh", match_thresh),
            ("proximity_thresh", proximity_thresh),
            ("appearance_thresh", appearance_thresh),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        self._args = SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
            gmc_method=gmc_method,
            proximity_thresh=proximity_thresh,
            appearance_thresh=appearance_thresh,
            with_reid=with_reid,
            model=reid_model,
        )
        self._tracker: Any = None

    def update(
        self,
        video_frame: VideoFrame,
        detections: list[PersonDetection],
    ) -> list[TrackedPerson]:
        tracker = self._get_tracker()
        frame = video_frame.frame

        if not detections:
            empty = _BoxTrackInputs(
                xywh=np.zeros((0, 4), dtype=np.float32),
                xyxy=np.zeros((0, 4), dtype=np.float32),
                conf=np.zeros((0,), dtype=np.float32),
                cls=np.zeros((0,), dtype=np.float32),
            )
            tracker.update(empty, img=frame)
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

        inputs = _BoxTrackInputs(xywh=xywh, xyxy=xyxy, conf=conf, cls=cls)
        tracks = tracker.update(inputs, img=frame)
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
        from ultralytics.trackers.bot_sort import BOTSORT

        self._tracker = BOTSORT(self._args)
        return self._tracker
