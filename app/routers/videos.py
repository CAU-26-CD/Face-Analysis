from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.deps import get_db
from app.models import RecordingSession

## 야매 프론트용 코드

router = APIRouter(
    prefix="/sessions",
    tags=["videos"]
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/{session_id}/video")
async def upload_video(
    session_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    session = (
        db.query(RecordingSession)
        .filter(RecordingSession.id == session_id)
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    extension = ".webm"
    if file.filename and "." in file.filename:
        extension = "." + file.filename.split(".")[-1]

    save_path = UPLOAD_DIR / f"session_{session_id}{extension}"

    with open(save_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    return {
        "message": "uploaded",
        "filename": save_path.name,
        "path": str(save_path),
    }