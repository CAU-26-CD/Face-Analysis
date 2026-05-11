from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.face_analysis.analyzer import FaceVideoAnalyzer
from app.services.face_analysis.video_reader import VideoFrameReader


def main() -> None:
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        video_path = PROJECT_ROOT / "uploads" / "session_7.webm"

    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    frame_reader = VideoFrameReader(frame_interval_seconds=5.0)
    analyzer = FaceVideoAnalyzer(frame_reader=frame_reader)

    for checked_frames, video_frame in enumerate(
        analyzer.read_sampled_frames(video_path),
        start=1,
    ):
        detections = analyzer.detector.detect(video_frame)
        print(
            "frame",
            checked_frames,
            "timestamp",
            video_frame.timestamp_seconds,
            "detections",
            len(detections),
        )

        for detection in detections:
            print(
                " ",
                detection.confidence,
                detection.bbox,
                "embedding_size",
                len(detection.embedding),
            )

        if checked_frames >= 5:
            break


if __name__ == "__main__":
    main()
