# Face Analyzer

S3에 업로드된 공연/리허설 영상을 받아 인물 등장 구간을 분석하고, 결과를 BE callback URL로 돌려주는 얼굴/인물 분석 워커입니다.

이 저장소는 BE 데이터베이스에 직접 연결하지 않습니다. BE가 영상 메타데이터, 프로젝트별 배우 갤러리, `analysis_status`, 결과 저장을 담당하고, 이 analyzer는 입력으로 받은 S3 영상과 기존 배우 얼굴 템플릿만 사용해 분석한 뒤 callback payload를 POST합니다.

## What It Does

```text
S3 video download
  -> VideoFrameReader              (0.3초 간격 프레임 샘플링)
  -> PersonDetector                (Ultralytics YOLO11n, COCO person class)
  -> PersonTracker                 (ByteTrack 기반 person bbox tracking)
  -> InsightFaceDetector           (face bbox + 512-d ArcFace embedding)
  -> FacePersonAssociator          (face -> person_track_id 연결)
  -> PersonTracklet exemplars      (track별 고품질/다양한 얼굴 임베딩 선별)
  -> TrackletClusterer             (끊긴 tracklet을 같은 인물 cluster로 병합)
  -> ActorMatcher                  (BE가 넘긴 known_actors 갤러리와 매칭)
  -> AppearanceTimelineBuilder     (등장 구간 생성)
  -> thumbnail upload to S3
  -> callback to BE
```

tracking 기준은 얼굴 bbox가 아니라 사람 bbox입니다. 얼굴이 옆모습이 되거나 잠깐 가려져도 몸 bbox가 계속 잡히면 같은 track으로 유지되고, 얼굴 임베딩은 그 track 위에 identity 정보를 얹는 방식으로 사용됩니다.

## Runtime Modes

이 프로젝트는 두 가지 방식으로 실행할 수 있습니다.

- **FastAPI service**: 로컬 개발 또는 일반 서버 배포용. `POST /analyze`를 받고 background task로 분석을 시작합니다.
- **RunPod Serverless worker**: GPU 서버리스 배포용. Docker 이미지의 기본 CMD는 `python -u -m app.handler`이며, RunPod job의 `input`을 분석 요청 payload로 사용합니다.

RunPod 환경에서는 module load 시점에 YOLO/InsightFace analyzer를 한 번 warm-up합니다. warm worker는 모델을 재사용하므로 cold start 이후 job 처리 시간이 줄어듭니다.

## API

### Health

```http
GET /health
```

```json
{ "status": "ok" }
```

### FastAPI: `POST /analyze`

요청을 수락하고 즉시 `202 Accepted`를 반환합니다. 실제 분석 결과는 `callback_url`로 전달됩니다.

```json
{
  "video_id": 1,
  "session_id": 7,
  "s3_key": "42/7/video.webm",
  "s3_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/42/7/video.webm",
  "callback_url": "https://be.example.com/api/v1/videos/analysis-callback",
  "known_actors": [
    {
      "actor_id": 10,
      "face_templates": [
        [0.12, -0.03, 0.44]
      ]
    }
  ],
  "thumbnail_dir": "42/7/"
}
```

응답:

```json
{
  "video_id": 1,
  "analysis_status": "accepted"
}
```

`known_actors`는 선택값이며 기본값은 `[]`입니다. 각 actor는 하나 이상의 512-d ArcFace embedding을 `face_templates`에 담아 전달합니다. analyzer는 여러 템플릿 중 가장 잘 맞는 값을 기준으로 기존 배우와 새 cluster를 매칭합니다.

`thumbnail_dir`도 선택값입니다. 전달하면 analyzer가 `{thumbnail_dir}/thumb-{idx}.jpg` 형태로 cluster 대표 썸네일을 S3에 업로드합니다. 생략하면 legacy 경로인 `{S3_THUMBNAIL_PREFIX 또는 thumbnails}/{video_id}/{idx}.jpg`를 사용합니다.

### RunPod Serverless Input

RunPod job은 같은 payload를 `input` 아래에 넣어 호출합니다.

```json
{
  "input": {
    "video_id": 1,
    "session_id": 7,
    "s3_key": "42/7/video.webm",
    "s3_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/42/7/video.webm",
    "callback_url": "https://be.example.com/api/v1/videos/analysis-callback",
    "known_actors": [],
    "thumbnail_dir": "42/7/"
  }
}
```

RunPod handler의 반환값은 RunPod dashboard/log 확인용입니다. 실제 분석 결과는 기존 BE callback contract 그대로 `callback_url`에 POST됩니다.

```json
{
  "video_id": 1,
  "delivered_via": "callback_url"
}
```

## Callback Contract

성공 시:

```json
{
  "video_id": 1,
  "analysis_status": "done",
  "matched": [
    {
      "actor_id": 10,
      "thumbnail_s3_key": "42/7/thumb-0.jpg",
      "similarity": 0.82,
      "new_exemplars": [
        [0.01, 0.02, 0.03]
      ]
    }
  ],
  "new_candidates": [
    {
      "temp_index": 0,
      "thumbnail_s3_key": "42/7/thumb-1.jpg",
      "face_embeddings": [
        [0.04, 0.21, 0.33]
      ],
      "detection_count": 8,
      "start_seconds": 30.0,
      "end_seconds": 42.0,
      "suggested_actor_id": null,
      "suggested_similarity": null
    }
  ],
  "analysis_result": {
    "video_path": "/tmp/face-analyzer-xxx/input.webm",
    "appearances": [
      {
        "person_id": "actor:10",
        "start_seconds": 15.0,
        "end_seconds": 20.0,
        "detection_count": 3
      },
      {
        "person_id": "new:0",
        "start_seconds": 30.0,
        "end_seconds": 42.0,
        "detection_count": 8
      }
    ]
  }
}
```

실패 시:

```json
{
  "video_id": 1,
  "analysis_status": "failed",
  "error_message": "..."
}
```

모든 callback 요청에는 다음 헤더가 포함됩니다.

```http
X-Analyzer-Secret: {ANALYZER_SECRET}
```

`ANALYZER_SECRET`이 없으면 callback을 보내지 않습니다. `CALLBACK_SECRET`도 fallback으로 지원합니다.

## Environment Variables

필수:

```bash
ANALYZER_SECRET=change-me
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=your-video-bucket
```

선택:

```bash
# GPU/accelerator selection
FACE_ANALYZER_DEVICE=cuda
FACE_ANALYZER_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider

# S3 download tuning
S3_DOWNLOAD_MAX_CONCURRENCY=16
S3_DOWNLOAD_CHUNKSIZE_MB=16
S3_USE_ACCELERATE=false

# Thumbnail upload destination
S3_THUMBNAIL_BUCKET=your-video-bucket
S3_THUMBNAIL_PREFIX=thumbnails

# Logging
LOG_LEVEL=INFO

# Demo fast mode (branch demo/fast-face-analysis only)
# Trades accuracy for a ~1-2s end-to-end analysis. Defaults to ON on that
# branch; set DEMO_FAST_MODE=0 to restore the full pipeline.
DEMO_FAST_MODE=1
DEMO_FRAME_INTERVAL_SECONDS=1.0   # seconds between sampled frames
DEMO_MAX_FRAMES=6                 # hard cap on sampled frames total
DEMO_YOLO_IMGSZ=320               # YOLO person-detector input size
DEMO_FACE_DET_SIZE=320            # InsightFace square det_size
DEMO_SKIP_THUMBNAILS=0            # 1 = drop thumbnail crop+upload entirely
```

> **Demo fast mode:** when `DEMO_FAST_MODE` is on, the analyzer samples only
> a handful of frames and shrinks both detector inputs so a full run finishes
> in ~1-2s (warm worker). Identity matching and per-cluster thumbnails still
> run, so the matching screen renders normally — only coverage is thinner.
> Set `DEMO_SKIP_THUMBNAILS=1` to also drop the thumbnail step (matching
> screen then has no face images).

`S3_BUCKET_NAME`이 설정되어 있으면 boto3로 `s3_key`를 다운로드합니다. 설정되어 있지 않으면 `s3_url`을 직접 HTTP GET으로 다운로드합니다. thumbnail upload는 `S3_THUMBNAIL_BUCKET` 또는 `S3_BUCKET_NAME`이 있을 때만 수행됩니다.

## Local Development

Poetry:

```bash
poetry install
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API 문서:

```text
http://127.0.0.1:8000/docs
```

## Docker

Dockerfile은 RunPod GPU worker 배포를 기준으로 작성되어 있습니다.

- base image: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
- Python: 3.12
- CUDA torch: PyTorch cu124 wheel
- video decode fallback용 `ffmpeg` 포함
- `insightface` buffalo_l 모델 prefetch
- YOLO `yolo11n.pt` weight prefetch
- 기본 실행 명령: `python -u -m app.handler`

이미지 빌드:

```bash
docker build -t face-analyzer:latest .
```

RunPod 또는 GPU host에서 serverless handler 형태로 실행:

```bash
docker run --rm --gpus all --env-file .env face-analyzer:latest
```

같은 이미지로 FastAPI 서버를 로컬에서 띄우려면 CMD를 override합니다.

```bash
docker run --rm --gpus all -p 8000:8000 --env-file .env \
  face-analyzer:latest \
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

CPU만 있는 환경에서도 일부 테스트나 API shell은 돌릴 수 있지만, 실제 영상 분석은 YOLO + InsightFace 때문에 GPU 사용을 권장합니다.

## RunPod Serverless Deployment

1. Docker 이미지를 빌드합니다.
2. 이미지를 registry에 push합니다.
3. RunPod Serverless endpoint template에서 해당 image를 지정합니다.
4. 환경 변수에 `ANALYZER_SECRET`, AWS credentials, `S3_BUCKET_NAME`, `AWS_REGION`, GPU provider 설정을 넣습니다.
5. BE에서 RunPod endpoint로 job을 submit할 때 기존 `/analyze` payload를 `input` 아래에 넣습니다.

권장 GPU 설정:

```bash
FACE_ANALYZER_DEVICE=cuda
FACE_ANALYZER_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
```

RunPod webhook은 사용하지 않습니다. 이 worker는 BE가 넘긴 `callback_url`로 직접 POST하고, `X-Analyzer-Secret` 헤더를 붙여 BE가 검증할 수 있게 합니다.

## Test

Unit test:

```bash
poetry run pytest tests/ -v
```

RunPod handler warm-up smoke test:

```bash
poetry run python -m scripts.test_handler_local
```

실제 S3 영상과 callback URL로 handler를 직접 호출:

```bash
set -a
source .env
set +a

poetry run python -m scripts.test_handler_local \
  "42/7/video.webm" \
  "https://bucket.s3.ap-northeast-2.amazonaws.com/42/7/video.webm" \
  "https://be.example.com/api/v1/videos/analysis-callback"
```

FastAPI end-to-end 테스트는 mock callback receiver를 하나 띄운 뒤 `/analyze`를 호출하면 됩니다.

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": 1,
    "session_id": 7,
    "s3_key": "42/7/video.webm",
    "s3_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/42/7/video.webm",
    "callback_url": "http://localhost:9000/cb",
    "known_actors": [],
    "thumbnail_dir": "42/7/"
  }'
```

## BE Integration Notes

BE는 다음 흐름으로 연동하면 됩니다.

1. 영상을 S3에 업로드하고 `s3_key`, `s3_url`을 저장합니다.
2. 같은 프로젝트에 저장된 actor들의 `actor_id`, `face_templates`를 모아 analyzer에 전달합니다.
3. callback 수신 시 `X-Analyzer-Secret`을 검증합니다.
4. `matched[]`는 기존 actor와 이번 video를 연결합니다.
5. `matched[].new_exemplars`는 기존 actor gallery에 추가할 수 있는 새 각도/표정 exemplar입니다.
6. `new_candidates[]`는 새 actor 후보입니다. BE가 actor row를 만들고 `temp_index -> actor_id` 매핑을 생성합니다.
7. `analysis_result.appearances[].person_id`의 `new:{temp_index}`를 실제 `actor:{actor_id}`로 치환한 뒤 저장합니다.
8. 재분석 시 기존 video-actor 연결과 이전 분석 결과를 정리한 뒤 새 callback 결과로 교체합니다.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| callback이 오지 않음 | `ANALYZER_SECRET` 또는 `CALLBACK_SECRET`이 설정되어 있는지 확인 |
| `NoCredentialsError` | AWS credentials 또는 `.env` load 누락 |
| S3 `403` | bucket 권한, IAM policy, region 확인 |
| S3 `404` | `s3_key` 확인. analyzer는 NFC/NFD Unicode normalization fallback을 한 번 시도함 |
| `422 Unprocessable Entity` | API payload schema 확인. `known_actors[].face_templates`는 list of list |
| RunPod job은 성공인데 BE에 결과 없음 | RunPod return은 dashboard용이며 실제 결과는 `callback_url`로 감. callback URL/secret/log 확인 |
| GPU OOM | batch/video 해상도/동시 worker 수 조정. job 종료마다 CUDA cache release는 수행됨 |

