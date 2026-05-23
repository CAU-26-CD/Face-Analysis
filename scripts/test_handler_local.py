"""Smoke-test the RunPod handler without RunPod.

Two modes:

  1) Warm-up only (no args)
       python -m scripts.test_handler_local
     Imports ``app.handler`` so the module-level analyzer warm-up runs.
     Confirms YOLO + InsightFace load without error. Does NOT invoke a job.

  2) Full job (with args)
       python -m scripts.test_handler_local <s3_key> <s3_url> <callback_url>
     Invokes ``handler({"input": ...})`` end-to-end. Requires the same env
     the production worker needs (ANALYZER_SECRET, AWS_*, S3_BUCKET_NAME).

For purely local (no S3, no callback) sanity checks, use
``scripts/manual_e2e_local.py`` instead — it bypasses worker.py entirely.
"""

import sys


def main() -> int:
    # Importing the module runs _build_warm_analyzer() at module load.
    from app.handler import handler

    if len(sys.argv) == 1:
        print("handler imported and analyzer warmed successfully.")
        return 0

    if len(sys.argv) < 4:
        print(
            "usage: python -m scripts.test_handler_local "
            "<s3_key> <s3_url> <callback_url>",
            file=sys.stderr,
        )
        return 2

    s3_key, s3_url, callback_url = sys.argv[1], sys.argv[2], sys.argv[3]

    job = {
        "input": {
            "video_id": 999999,
            "session_id": 1,
            "s3_key": s3_key,
            "s3_url": s3_url,
            "callback_url": callback_url,
            "known_actors": [],
        }
    }

    result = handler(job)
    print("handler returned:", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
