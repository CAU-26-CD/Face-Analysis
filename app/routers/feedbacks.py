from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.deps import get_db
from app.models import Feedback, RecordingSession, User
from app.schemas import FeedbackCreate, FeedbackResponse

## 피드백 저장시 : 
#- 해당 세션이 존재하는지 확인
#- 녹화시작했는지 확인
#- 현재시간 - 녹화시작시간

router = APIRouter(
    prefix="/sessions/{session_id}/feedbacks",
    tags=["feedbacks"]
)


@router.post("", response_model=FeedbackResponse)
def create_feedback(
    session_id: int,
    payload: FeedbackCreate,
    db: Session = Depends(get_db)
):
    session = (
        db.query(RecordingSession)
        .filter(RecordingSession.id == session_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.recording_started_at is None:
        raise HTTPException(status_code=400, detail="Recording has not started yet")

    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    now = datetime.utcnow()
    offset_seconds = int((now - session.recording_started_at).total_seconds())

    feedback = Feedback(
        session_id=session.id,
        user_id=payload.user_id,
        content=payload.content,
        actor_name=payload.actor_name,
        category=payload.category,
        created_at=now,
        video_offset_seconds=offset_seconds
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return feedback


@router.get("", response_model=list[FeedbackResponse])
def read_feedbacks(session_id: int, db: Session = Depends(get_db)):
    session = (
        db.query(RecordingSession)
        .filter(RecordingSession.id == session_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    feedbacks = (
        db.query(Feedback)
        .filter(Feedback.session_id == session_id)
        .order_by(Feedback.video_offset_seconds.asc())
        .all()
    )

    return feedbacks