from pathlib import Path
import argparse
from dataclasses import asdict
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.face_analysis.analyzer import FaceVideoAnalyzer
from app.services.face_analysis.clustering import FaceClusterer
from app.services.face_analysis.timeline import AppearanceTimelineBuilder
from app.services.face_analysis.video_reader import VideoFrameReader


def main() -> None:
    args = parse_args()
    video_path = args.video_path

    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    analyzer = FaceVideoAnalyzer(
        frame_reader=VideoFrameReader(frame_interval_seconds=args.frame_interval),
        clusterer=FaceClusterer(similarity_threshold=args.similarity_threshold),
        timeline_builder=AppearanceTimelineBuilder(max_gap_seconds=args.max_gap),
    )

    def progress(
        sampled_frame_count: int,
        timestamp_seconds: float,
        detection_count: int,
    ) -> None:
        if sampled_frame_count == 1 or sampled_frame_count % 10 == 0:
            print(
                "progress",
                f"frames={sampled_frame_count}",
                f"timestamp={timestamp_seconds:.1f}s",
                f"detections={detection_count}",
                file=sys.stderr,
            )

    result = analyzer.analyze(video_path, progress_callback=progress)

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a video and print face appearance timeline JSON.",
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "uploads" / "session_7.webm",
        help="Video file path. Defaults to uploads/session_7.webm.",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=5.0,
        help="Seconds between sampled frames. Lower is slower and more precise.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.45,
        help="Cosine similarity threshold for assigning a face to an existing person.",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=6.0,
        help="Maximum seconds between detections to merge them into one appearance.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
