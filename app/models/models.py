"""
SQLAlchemy ORM models — ERD 기반
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Integer, String, Boolean, DateTime, ForeignKey, Text, func
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


# ──────────────────────────────────────────────
# User (사용자)
# ──────────────────────────────────────────────
class User(Base):
    __tablename__ = "user"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("project.project_id"), nullable=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False)

    feedbacks: Mapped[list["Feedback"]] = relationship("Feedback", back_populates="author")


# ──────────────────────────────────────────────
# Project (공연)
# ──────────────────────────────────────────────
class Project(Base):
    __tablename__ = "project"

    project_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(4), unique=True, nullable=False)      # 조인 코드 4자
    p_name: Mapped[str] = mapped_column(String(15), nullable=False)
    p_detail: Mapped[Optional[str]] = mapped_column(String(20))
    p_created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    is_liked: Mapped[bool] = mapped_column(Boolean, default=False)

    sessions: Mapped[list["Session"]] = relationship("Session", back_populates="project")
    blueprints: Mapped[list["Blueprint"]] = relationship("Blueprint", back_populates="project")
    videos: Mapped[list["Video"]] = relationship("Video", back_populates="project")
    actors: Mapped[list["Actor"]] = relationship("Actor", back_populates="project")
    feedbacks: Mapped[list["Feedback"]] = relationship("Feedback", back_populates="project")


# ──────────────────────────────────────────────
# Session (연습 회차)
# ──────────────────────────────────────────────
class Session(Base):
    __tablename__ = "session"

    session_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("project.project_id"), nullable=False)
    s_name: Mapped[str] = mapped_column(String(10), nullable=False)               # 워크스페이스 이름
    s_category: Mapped[Optional[str]] = mapped_column(String(1))                  # 런쓰루, 워크쓰루 등
    s_detail: Mapped[Optional[str]] = mapped_column(String(20))
    in_progress: Mapped[bool] = mapped_column(Boolean, default=False)             # 현재 진행중인지
    s_created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    project: Mapped["Project"] = relationship("Project", back_populates="sessions")
    video: Mapped[Optional["Video"]] = relationship("Video", back_populates="session", uselist=False)
    actors: Mapped[list["Actor"]] = relationship("Actor", back_populates="session")
    feedbacks: Mapped[list["Feedback"]] = relationship("Feedback", back_populates="session")


# ──────────────────────────────────────────────
# Blueprint (도면)
# ──────────────────────────────────────────────
class Blueprint(Base):
    __tablename__ = "blueprint"

    blueprint_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("project.project_id"), nullable=False)
    b_version: Mapped[Optional[str]] = mapped_column(String(50))
    data_json: Mapped[Optional[str]] = mapped_column(Text)                         # JSON 직렬화 도면 데이터
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    project: Mapped["Project"] = relationship("Project", back_populates="blueprints")


# ──────────────────────────────────────────────
# Video (세션 별 영상)
# ──────────────────────────────────────────────
class Video(Base):
    __tablename__ = "video"

    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("session.session_id"), primary_key=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("project.project_id"), nullable=False)
    record_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    record_end_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    file_path: Mapped[Optional[str]] = mapped_column(String(512))                  # 저장 경로 or URL

    session: Mapped["Session"] = relationship("Session", back_populates="video")
    project: Mapped["Project"] = relationship("Project", back_populates="videos")


# ──────────────────────────────────────────────
# Actor (배우 태그 — 세션당 최대 8개)
# ──────────────────────────────────────────────
class Actor(Base):
    __tablename__ = "actor"

    actor_id: Mapped[str] = mapped_column(String(50), primary_key=True)           # actor_Tag
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("session.session_id"), nullable=False)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("project.project_id"), nullable=False)
    a_name: Mapped[str] = mapped_column(String(50), unique=False, nullable=False)  # 실제 배우 이름

    session: Mapped["Session"] = relationship("Session", back_populates="actors")
    project: Mapped["Project"] = relationship("Project", back_populates="actors")
    feedbacks: Mapped[list["Feedback"]] = relationship("Feedback", back_populates="actor")
    f_categories: Mapped[list["FCategory"]] = relationship("FCategory", back_populates="actor")


# ──────────────────────────────────────────────
# Feedback (피드백)
# ──────────────────────────────────────────────
class Feedback(Base):
    __tablename__ = "feedback"

    feedback_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("actor.actor_id"), nullable=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("session.session_id"), nullable=False)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("project.project_id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("user.user_id"), nullable=False)  # 작성자
    content: Mapped[str] = mapped_column(Text, nullable=False)                     # 피드백 내용
    timestamp_sec: Mapped[Optional[int]] = mapped_column(Integer)                  # 영상 타임스탬프 (초 단위)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    author: Mapped["User"] = relationship("User", back_populates="feedbacks")
    project: Mapped["Project"] = relationship("Project", back_populates="feedbacks")
    session: Mapped["Session"] = relationship("Session", back_populates="feedbacks")
    actor: Mapped[Optional["Actor"]] = relationship("Actor", back_populates="feedbacks")
    categories: Mapped[list["FCategory"]] = relationship("FCategory", back_populates="feedback")


# ──────────────────────────────────────────────
# FCategory (피드백 카테고리 태그)
# ──────────────────────────────────────────────
class FCategory(Base):
    __tablename__ = "f_category"

    category_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feedback_id: Mapped[int] = mapped_column(Integer, ForeignKey("feedback.feedback_id"), nullable=False)
    actor_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("actor.actor_id"), nullable=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("session.session_id"), nullable=False)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("project.project_id"), nullable=False)
    f_name: Mapped[str] = mapped_column(String(50), nullable=False)                # 카테고리명 (중복 가능)

    feedback: Mapped["Feedback"] = relationship("Feedback", back_populates="categories")
    actor: Mapped[Optional["Actor"]] = relationship("Actor", back_populates="f_categories")
