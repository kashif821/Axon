from __future__ import annotations

from pathlib import Path
from typing import Annotated

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import select, SQLModel

from axon.config.settings import settings
from axon.memory.schema import ActionLog, FileChange, Session, Summary

db_path = Path(settings.axon_db_path)
db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.axon_db_path}",
    echo=False,
    future=True,
)

async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db() -> None:
    import aiosqlite
    from datetime import datetime

    db_dir = Path(settings.axon_db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(settings.axon_db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS action_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                content TEXT,
                files TEXT,
                summary_type TEXT,
                created_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS file_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                filepath TEXT,
                event_type TEXT,
                lines_added INTEGER DEFAULT 0,
                lines_removed INTEGER DEFAULT 0,
                timestamp TEXT
            )
        """)
        await db.commit()


async def create_session(title: str) -> Session:
    session = Session(title=title)

    async with async_session_factory() as session_obj:
        session_obj.add(session)
        await session_obj.commit()
        await session_obj.refresh(session)
        return session


async def log_action(
    session_id: str,
    role: str,
    content: str,
) -> ActionLog:
    action_log = ActionLog(session_id=session_id, role=role, content=content)

    async with async_session_factory() as session_obj:
        session_obj.add(action_log)
        await session_obj.commit()
        await session_obj.refresh(action_log)
        return action_log


async def get_session_history(session_id: uuid.UUID) -> list[ActionLog]:
    async with async_session_factory() as session_obj:
        statement = (
            select(ActionLog)
            .where(ActionLog.session_id == session_id)
            .order_by(ActionLog.timestamp)
        )
        result = await session_obj.execute(statement)
        return list(result.scalars().all())


async def get_all_sessions() -> list[Session]:
    async with async_session_factory() as session_obj:
        statement = select(Session).order_by(Session.updated_at.desc())
        result = await session_obj.execute(statement)
        return list(result.scalars().all())


async def log_file_change(
    session_id: uuid.UUID,
    filepath: str,
    action: str,
) -> FileChange:
    file_change = FileChange(session_id=session_id, filepath=filepath, action=action)

    async with async_session_factory() as session_obj:
        session_obj.add(file_change)
        await session_obj.commit()
        await session_obj.refresh(file_change)
        return file_change


async def get_session(session_id: uuid.UUID) -> Session | None:
    async with async_session_factory() as session_obj:
        statement = select(Session).where(Session.id == session_id)
        result = await session_obj.execute(statement)
        return result.scalar_one_or_none()


async def get_recent_actions(limit: int = 20) -> list[ActionLog]:
    async with async_session_factory() as session_obj:
        statement = select(ActionLog).order_by(ActionLog.timestamp.desc()).limit(limit)
        result = await session_obj.execute(statement)
        return list(result.scalars().all())


async def get_recent_file_changes(limit: int = 20) -> list[FileChange]:
    async with async_session_factory() as session_obj:
        statement = (
            select(FileChange).order_by(FileChange.timestamp.desc()).limit(limit)
        )
        result = await session_obj.execute(statement)
        return list(result.scalars().all())


async def save_summary(
    content: str, files: list[str], summary_type: str = "brain"
) -> Summary:
    import json

    summary = Summary(
        content=content,
        files=json.dumps(files),
        type=summary_type,
    )

    async with async_session_factory() as session_obj:
        session_obj.add(summary)
        await session_obj.commit()
        await session_obj.refresh(summary)
        return summary


async def get_recent_summaries(limit: int = 10) -> list[Summary]:
    async with async_session_factory() as session_obj:
        statement = select(Summary).order_by(Summary.timestamp.desc()).limit(limit)
        result = await session_obj.execute(statement)
        return list(result.scalars().all())
