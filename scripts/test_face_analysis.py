from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.face_analysis.video_reader import VideoFrameReader


def main() -> None:
    video_path = PROJECT_ROOT / "uploads" / "session_7.webm"
    reader = VideoFrameReader(frame_interval_seconds=1.0)

    for video_frame in reader.read_frames(video_path):
        print(
            video_frame.timestamp_seconds,
            video_frame.frame_index,
            video_frame.frame.shape,
        )


if __name__ == "__main__":
    main()
