from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import select, SQLModel

from axon.config.settings import settings
from axon.memory.schema import ActionLog, FileChange, Session


engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.axon_db_path}",
    echo=False,
    future=True,
)

async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db() -> None:
    db_dir = Path(settings.axon_db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def create_session(title: str) -> Session:
    session = Session(title=title)

    async with async_session_factory() as session_obj:
        session_obj.add(session)
        await session_obj.commit()
        await session_obj.refresh(session)
        return session


async def log_action(
    session_id: uuid.UUID,
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
