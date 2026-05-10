from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.deps import get_db
from app.models import RecordingSession
from app.schemas import SessionCreate, SessionResponse

## 세션 만들고, 녹화 시작/종료까지

router = APIRouter(
    prefix="/sessions",
    tags=["sessions"]
)


@router.post("", response_model=SessionResponse)
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    session = RecordingSession(title=payload.title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("", response_model=list[SessionResponse])
def read_sessions(db: Session = Depends(get_db)):
    return db.query(RecordingSession).all()


@router.get("/{session_id}", response_model=SessionResponse)
def read_session(session_id: int, db: Session = Depends(get_db)):
    session = (
        db.query(RecordingSession)
        .filter(RecordingSession.id == session_id)
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.post("/{session_id}/start-recording", response_model=SessionResponse)
def start_recording(session_id: int, db: Session = Depends(get_db)):
    session = (
        db.query(RecordingSession)
        .filter(RecordingSession.id == session_id)
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.recording_started_at is not None:
        raise HTTPException(status_code=400, detail="Recording already started")

    session.recording_started_at = datetime.utcnow()
    db.commit()
    db.refresh(session)

    return session


@router.post("/{session_id}/end-recording", response_model=SessionResponse)
def end_recording(session_id: int, db: Session = Depends(get_db)):
    session = (
        db.query(RecordingSession)
        .filter(RecordingSession.id == session_id)
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.recording_started_at is None:
        raise HTTPException(status_code=400, detail="Recording has not started yet")

    if session.recording_ended_at is not None:
        raise HTTPException(status_code=400, detail="Recording already ended")

    session.recording_ended_at = datetime.utcnow()
    db.commit()
    db.refresh(session)

    return session