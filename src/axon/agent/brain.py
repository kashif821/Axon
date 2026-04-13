import asyncio
import logging
import threading
from datetime import datetime

from axon.llm.base import ChatMessage, MessageRole
from axon.llm.providers import get_llm_provider
from axon.memory.store import get_recent_actions, save_summary
from axon.utils.console import console
from axon.watcher.idle import reset_activity

logger = logging.getLogger(__name__)


async def _heartbeat_callback(recent_changes: list) -> None:
    if not recent_changes:
        return

    files_list = [c["path"] for c in recent_changes]

    recent_actions = await get_recent_actions(limit=5)
    actions_text = (
        "\n".join(f"- {a.role}: {a.content[:100]}" for a in recent_actions)
        if recent_actions
        else "No recent actions"
    )

    prompt = f"""Summarize what the developer was working on.

Recent file changes:
{chr(10).join(f"- {f}" for f in files_list)}

Recent actions from memory:
{actions_text}

Be concise. Max 3 sentences."""

    try:
        provider = get_llm_provider()
        response = await provider.chat(
            messages=[
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=prompt,
                )
            ],
            temperature=0.5,
            max_tokens=200,
        )
        summary_text = response.choices[0].message.content or "No summary generated"

        await save_summary(
            content=summary_text,
            files=files_list,
            summary_type="brain",
        )

        from axon.watcher.monitor import clear_changes
        clear_changes()
        reset_activity()
        console.print("[bold magenta]🧠 Brain updated memory[/bold magenta]")

    except Exception as e:
        logger.error(f"Error in brain callback: {e}")


def _start_heartbeat(main_loop: asyncio.AbstractEventLoop) -> None:
    from axon.watcher.monitor import start_watching
    from axon.watcher.idle import start_idle_monitor

    start_watching(".")
    start_idle_monitor(_heartbeat_callback, main_loop=main_loop)
    logger.info("Brain heartbeat started")


def _stop_heartbeat() -> None:
    from axon.watcher.monitor import stop_watching
    from axon.watcher.idle import stop_idle_monitor

    stop_idle_monitor()
    stop_watching()
    logger.info("Brain heartbeat stopped")


async def run_brain_mode() -> None:
    from axon.memory.store import init_db
    await init_db()

    main_loop = asyncio.get_event_loop()
    _start_heartbeat(main_loop)

    console.print("[bold magenta]🧠 Coding brain active[/bold magenta]")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopping brain...[/dim]")
        _stop_heartbeat()
