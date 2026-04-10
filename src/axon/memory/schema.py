from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Session(SQLModel, table=True):
    __tablename__ = "sessions"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    title: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ActionLog(SQLModel, table=True):
    __tablename__ = "action_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="sessions.id", index=True)
    role: str = ""
    content: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FileChange(SQLModel, table=True):
    __tablename__ = "file_changes"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="sessions.id", index=True)
    filepath: str = ""
    action: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
