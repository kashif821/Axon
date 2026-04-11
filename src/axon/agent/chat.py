from __future__ import annotations

import uuid
from typing import AsyncIterator

from axon.llm.base import ChatMessage, MessageRole
from axon.llm.providers import get_llm_provider
from axon.memory.store import (
    create_session,
    get_session,
    get_session_history,
    log_action,
)


DEFAULT_SYSTEM_PROMPT = """You are Axon, an intelligent coding assistant. You help users with software development tasks including:
- Writing and refactoring code
- Debugging and fixing issues
- Explaining technical concepts
- Planning and architecture decisions

Be helpful, concise, and precise in your responses."""


async def stream_response(
    messages: list[ChatMessage],
    model: str | None = None,
) -> AsyncIterator[str]:
    provider = get_llm_provider()
    full_response = ""

    try:
        async for chunk in provider.stream(messages, model=model):
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                yield delta
    except Exception as e:
        raise


async def run_chat(
    model: str | None = None,
    session_id: str | None = None,
    confirm_write: bool = True,
) -> dict:
    from axon.cli.commands.build import ask_confirmation

    provider = get_llm_provider()
    current_session = None
    messages: list[ChatMessage] = []

    if session_id:
        try:
            session_uuid = uuid.UUID(session_id)
            db_session = await get_session(session_uuid)
            if db_session:
                current_session = db_session
                history = await get_session_history(session_uuid)
                for log in history:
                    role = MessageRole(log.role)
                    messages.append(ChatMessage(role=role, content=log.content))
        except ValueError:
            pass

    tree = _get_directory_tree()
    system_prompt = (
        DEFAULT_SYSTEM_PROMPT + "\n\nCurrent Working Directory Structure:\n" + tree
    )

    messages.insert(0, ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))

    return {
        "provider": provider,
        "messages": messages,
        "current_session": current_session,
        "model": model,
        "confirm_write": confirm_write,
        "system_prompt": system_prompt,
    }


def _get_directory_tree() -> str:
    from axon.agent.utils import get_directory_tree

    try:
        return get_directory_tree(max_depth=2)
    except Exception:
        return "[Unable to read directory structure]"
