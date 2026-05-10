from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class UserCreate(BaseModel):
    name: str


class UserResponse(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


class SessionCreate(BaseModel):
    title: str


class SessionResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    recording_started_at: Optional[datetime]
    recording_ended_at: Optional[datetime]

    class Config:
        from_attributes = True


class FeedbackCreate(BaseModel):
    user_id: int
    content: str
    actor_name: Optional[str] = None
    category: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    session_id: int
    user_id: int
    content: str
    actor_name: Optional[str]
    category: Optional[str]
    created_at: datetime
    video_offset_seconds: int

    class Config:
        from_attributes = True