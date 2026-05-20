from fastapi.testclient import TestClient

import app.main as main_module


def test_analyze_accepts_request_and_schedules_background_task(monkeypatch):
    calls = []

    def fake_run_analysis_job(payload: dict) -> None:
        calls.append(payload)

    monkeypatch.setattr(main_module, "run_analysis_job", fake_run_analysis_job)
    client = TestClient(main_module.app)

    response = client.post(
        "/analyze",
        json={
            "video_id": 1,
            "session_id": 7,
            "s3_key": "videos/7/abc123.webm",
            "s3_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/videos/7/abc123.webm",
            "callback_url": "https://be.example.com/api/v1/videos/analysis-callback",
        },
    )

    assert response.status_code == 202
    assert response.json() == {
        "video_id": 1,
        "analysis_status": "accepted",
    }
    assert calls == [
        {
            "video_id": 1,
            "session_id": 7,
            "s3_key": "videos/7/abc123.webm",
            "s3_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/videos/7/abc123.webm",
            "callback_url": "https://be.example.com/api/v1/videos/analysis-callback",
            "known_actors": [],
            "thumbnail_dir": None,
        }
    ]


def test_analyze_forwards_known_actors(monkeypatch):
    calls = []

    def fake_run_analysis_job(payload: dict) -> None:
        calls.append(payload)

    monkeypatch.setattr(main_module, "run_analysis_job", fake_run_analysis_job)
    client = TestClient(main_module.app)

    response = client.post(
        "/analyze",
        json={
            "video_id": 2,
            "session_id": 7,
            "s3_key": "videos/7/v2.webm",
            "s3_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/videos/7/v2.webm",
            "callback_url": "https://be.example.com/api/v1/videos/analysis-callback",
            "known_actors": [
                {"actor_id": 1, "face_templates": [[0.1, 0.2, 0.3]]},
                {
                    "actor_id": 2,
                    "face_templates": [
                        [-0.1, 0.0, 0.5],
                        [-0.05, 0.1, 0.4],
                    ],
                },
            ],
        },
    )

    assert response.status_code == 202
    assert calls[0]["known_actors"] == [
        {"actor_id": 1, "face_templates": [[0.1, 0.2, 0.3]]},
        {
            "actor_id": 2,
            "face_templates": [
                [-0.1, 0.0, 0.5],
                [-0.05, 0.1, 0.4],
            ],
        },
    ]


def test_analyze_forwards_thumbnail_dir(monkeypatch):
    calls = []

    def fake_run_analysis_job(payload: dict) -> None:
        calls.append(payload)

    monkeypatch.setattr(main_module, "run_analysis_job", fake_run_analysis_job)
    client = TestClient(main_module.app)

    response = client.post(
        "/analyze",
        json={
            "video_id": 3,
            "session_id": 7,
            "s3_key": "42/7/video.webm",
            "s3_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/42/7/video.webm",
            "callback_url": "https://be.example.com/api/v1/videos/analysis-callback",
            "thumbnail_dir": "42/7/",
        },
    )

    assert response.status_code == 202
    assert calls[0]["thumbnail_dir"] == "42/7/"


def test_analyze_defaults_thumbnail_dir_to_none(monkeypatch):
    calls = []

    def fake_run_analysis_job(payload: dict) -> None:
        calls.append(payload)

    monkeypatch.setattr(main_module, "run_analysis_job", fake_run_analysis_job)
    client = TestClient(main_module.app)

    response = client.post(
        "/analyze",
        json={
            "video_id": 4,
            "session_id": 7,
            "s3_key": "videos/7/legacy.webm",
            "s3_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/videos/7/legacy.webm",
            "callback_url": "https://be.example.com/api/v1/videos/analysis-callback",
        },
    )

    assert response.status_code == 202
    assert calls[0]["thumbnail_dir"] is None
