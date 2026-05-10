from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)

    feedbacks = relationship("Feedback", back_populates="user")


class RecordingSession(Base):
    __tablename__ = "recording_sessions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 영상 녹화 시작 기준 시각
    recording_started_at = Column(DateTime, nullable=True)
    recording_ended_at = Column(DateTime, nullable=True)

    feedbacks = relationship("Feedback", back_populates="session")


class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(Integer, ForeignKey("recording_sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    content = Column(Text, nullable=False)
    actor_name = Column(String, nullable=True)
    category = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 녹화 시작 후 몇 초 뒤에 남겨졌는지
    video_offset_seconds = Column(Integer, nullable=False)

    session = relationship("RecordingSession", back_populates="feedbacks")
    user = relationship("User", back_populates="feedbacks")