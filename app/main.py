from fastapi import BackgroundTasks, FastAPI, status
from pydantic import BaseModel, Field, HttpUrl

from app.services.face_analysis.worker import run_analysis_job


class AnalyzeRequest(BaseModel):
    video_id: int = Field(..., ge=1)
    session_id: int = Field(..., ge=1)
    s3_key: str = Field(..., min_length=1)
    s3_url: HttpUrl
    callback_url: HttpUrl


class AnalyzeAcceptedResponse(BaseModel):
    video_id: int
    analysis_status: str


app = FastAPI(
    title="Face Analyzer API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/analyze",
    response_model=AnalyzeAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> AnalyzeAcceptedResponse:
    background_tasks.add_task(run_analysis_job, request.model_dump(mode="json"))
    return AnalyzeAcceptedResponse(
        video_id=request.video_id,
        analysis_status="accepted",
    )
