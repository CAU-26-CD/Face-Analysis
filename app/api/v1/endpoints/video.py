import os, aiofiles
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.deps import get_current_user
from app.core.config import settings
from app.models.models import User, Video, Session
from app.schemas.schemas import VideoOut

router = APIRouter(
    prefix="/projects/{project_id}/sessions/{session_id}/video",
    tags=["video"],
)

ALLOWED_MIME = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "application/octet-stream"}


@router.get("", response_model=VideoOut)
async def get_video(
    project_id: int,
    session_id: int,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Video).where(
            Video.session_id == session_id,
            Video.project_id == project_id,
        )
    )
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@router.post("/start", response_model=VideoOut, status_code=201)
async def start_recording(
    project_id: int,
    session_id: int,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """녹화 시작 — record_started_at 기록, session in_progress=True"""
    sess_result = await db.execute(select(Session).where(Session.session_id == session_id))
    session = sess_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    video = Video(
        session_id=session_id,
        project_id=project_id,
        record_started_at=datetime.utcnow(),
    )
    session.in_progress = True
    db.add(video)
    await db.flush()
    return video


@router.post("/upload", response_model=VideoOut)
async def upload_video(
    project_id: int,
    session_id: int,
    file: UploadFile = File(...),
    # _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """녹화 종료 후 영상 파일 업로드"""
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail="지원하지 않는 영상 형식입니다 (mp4/mov/avi)")

    # 파일 크기 제한 확인
    max_bytes = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"영상 파일이 {settings.MAX_VIDEO_SIZE_MB}MB를 초과합니다")

    # 저장 경로
    upload_dir = os.path.join(settings.UPLOAD_DIR, str(project_id), str(session_id))
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Video 레코드 업데이트
    result = await db.execute(
        select(Video).where(Video.session_id == session_id, Video.project_id == project_id)
    )
    video = result.scalar_one_or_none()

    if video:
        video.file_path = file_path
        video.record_end_at = datetime.utcnow()
    else:
        video = Video(
            session_id=session_id,
            project_id=project_id,
            file_path=file_path,
            record_end_at=datetime.utcnow(),
        )
        db.add(video)

    # 세션 in_progress 종료
    sess_result = await db.execute(select(Session).where(Session.session_id == session_id))
    session = sess_result.scalar_one_or_none()
    if session:
        session.in_progress = False

    await db.flush()
    return video
