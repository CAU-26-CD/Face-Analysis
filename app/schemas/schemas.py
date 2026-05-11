"""
Pydantic Schemas — Request / Response
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


# ─── Auth ───────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# ─── User ───────────────────────────────────────────────
class UserOut(BaseModel):
    user_id: int
    email: str

    model_config = {"from_attributes": True}


# ─── Project ────────────────────────────────────────────
class ProjectCreate(BaseModel):
    p_name: str = Field(max_length=15)
    p_detail: Optional[str] = Field(default=None, max_length=20)


class ProjectJoin(BaseModel):
    code: str = Field(min_length=4, max_length=4)


class ProjectOut(BaseModel):
    project_id: int
    code: str
    p_name: str
    p_detail: Optional[str]
    p_created_at: datetime
    is_liked: bool

    model_config = {"from_attributes": True}


# ─── Session ────────────────────────────────────────────
class SessionCreate(BaseModel):
    s_name: str = Field(max_length=10)
    s_category: Optional[str] = Field(default=None, max_length=1)
    s_detail: Optional[str] = Field(default=None, max_length=20)


class SessionUpdate(BaseModel):
    in_progress: Optional[bool] = None
    s_name: Optional[str] = None
    s_detail: Optional[str] = None


class SessionOut(BaseModel):
    session_id: int
    project_id: int
    s_name: str
    s_category: Optional[str]
    s_detail: Optional[str]
    in_progress: bool
    s_created_at: datetime

    model_config = {"from_attributes": True}


# ─── Actor ──────────────────────────────────────────────
class ActorCreate(BaseModel):
    actor_id: str = Field(max_length=50, description="태그 ID (예: A, B, 오지원 등)")
    a_name: str = Field(max_length=50)


class ActorOut(BaseModel):
    actor_id: str
    session_id: int
    project_id: int
    a_name: str

    model_config = {"from_attributes": True}


# ─── Feedback ───────────────────────────────────────────
class FeedbackCreate(BaseModel):
    content: str
    timestamp_sec: Optional[int] = Field(default=None, ge=0, description="영상 타임스탬프 (초)")
    actor_id: Optional[str] = None
    categories: list[str] = Field(default_factory=list, description="카테고리 이름 목록")


class FeedbackOut(BaseModel):
    feedback_id: int
    session_id: int
    project_id: int
    user_id: int
    actor_id: Optional[str]
    content: str
    timestamp_sec: Optional[int]
    created_at: datetime
    categories: list[str] = []

    model_config = {"from_attributes": True}


# ─── FCategory ──────────────────────────────────────────
class FCategoryOut(BaseModel):
    category_id: int
    feedback_id: int
    f_name: str

    model_config = {"from_attributes": True}


# ─── Video ──────────────────────────────────────────────
class VideoOut(BaseModel):
    session_id: int
    project_id: int
    record_started_at: Optional[datetime]
    record_end_at: Optional[datetime]
    file_path: Optional[str]

    model_config = {"from_attributes": True}


# ─── Report ─────────────────────────────────────────────
class FeedbackReportItem(BaseModel):
    actor_id: Optional[str]
    actor_name: Optional[str]
    total_count: int
    categories: dict[str, int]          # category_name -> count
    top_timestamps: list[int]           # 피드백 많은 구간 상위 타임스탬프 목록


class SessionReportOut(BaseModel):
    session_id: int
    total_feedbacks: int
    actor_reports: list[FeedbackReportItem]
