# Face Analyzer

Standalone FastAPI service for analyzing faces in videos stored on S3.

This repository does not connect to the BE database. The BE server owns upload,
video metadata, `analysis_status`, and persistence. This analyzer only receives
an S3 video reference, runs OpenCV / insightFace / onnxruntime analysis, and
POSTs the result back to the BE callback URL.

## API

### POST `/analyze`

Starts a background analysis job and returns immediately.

Request:

```json
{
  "video_id": 1,
  "session_id": 7,
  "s3_key": "videos/7/abc123.webm",
  "s3_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/videos/7/abc123.webm",
  "callback_url": "https://BE_SERVER/api/v1/videos/analysis-callback"
}
```

Immediate response:

```json
{
  "video_id": 1,
  "analysis_status": "accepted"
}
```

Success callback:

```json
{
  "video_id": 1,
  "analysis_status": "done",
  "analysis_result": {
    "video_path": "...",
    "appearances": [
      {
        "person_id": "person_1",
        "start_seconds": 15.0,
        "end_seconds": 20.0,
        "detection_count": 3
      }
    ]
  }
}
```

Failure callback:

```json
{
  "video_id": 1,
  "analysis_status": "failed",
  "error_message": "..."
}
```

Every callback includes:

```http
X-Analyzer-Secret: {ANALYZER_SECRET}
```

## Environment

```bash
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=your-video-bucket
ANALYZER_SECRET=change-me
```

`CALLBACK_SECRET` is also accepted as a fallback for `ANALYZER_SECRET`.

If `S3_BUCKET_NAME` is set, the analyzer downloads with boto3 using `s3_key`.
If it is not set, it falls back to downloading `s3_url` directly.

## Run Locally

```bash
poetry install
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Docs: `http://127.0.0.1:8000/docs`

## Docker

```bash
docker build -t face-analyzer .
docker run --rm -p 8000:8000 --env-file .env face-analyzer
```

## BE Contract

The BE server should:

1. Upload the video to S3 and store `s3_key` / `s3_url`.
2. Call analyzer `POST /analyze`.
3. Receive the callback and validate `X-Analyzer-Secret`.
4. Update `videos.analysis_status` and `videos.analysis_result`.
