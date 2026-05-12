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
        }
    ]
