from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from axon.llm.base import ChatMessage, MessageRole
from axon.llm.providers import get_llm_provider
from axon.memory.store import get_recent_actions, get_recent_file_changes


console = Console()

MEMORY_ASSISTANT_PROMPT = """You are Axon's memory assistant. Based on the developer's recent activity, answer their question concisely. Focus on what they actually did, what files they changed, and what's next."""


async def run_ask(question: str, model: Optional[str] = None) -> None:
    provider = get_llm_provider()

    recent_actions = await get_recent_actions(limit=20)
    recent_files = await get_recent_file_changes(limit=20)

    if not recent_actions and not recent_files:
        console.print(
            Panel(
                "No memory found. Run [bold]axon chat[/bold] or [bold]axon build[/bold] to start tracking your activity.",
                border_style="yellow",
                title="[bold]Memory Assistant[/bold]",
            )
        )
        return

    context_parts = []

    if recent_actions:
        context_parts.append("## Recent Actions:")
        for action in reversed(recent_actions[-10:]):
            role_label = "You" if action.role == "user" else "Axon"
            content_preview = (
                action.content[:200] + "..."
                if len(action.content) > 200
                else action.content
            )
            context_parts.append(f"- [{role_label}] {content_preview}")

    if recent_files:
        context_parts.append("\n## Files Modified:")
        for f in recent_files[:10]:
            context_parts.append(f"- {f.filepath} ({f.action})")

    memory_context = "\n".join(context_parts)

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=MEMORY_ASSISTANT_PROMPT),
        ChatMessage(
            role=MessageRole.USER,
            content=f"## Memory Context:\n{memory_context}\n\n## Question:\n{question}",
        ),
    ]

    full_response = ""
    try:
        async for chunk in provider.stream(messages, model=model):
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                console.print(delta, end="")

        console.print()

    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Error:[/bold red] {str(e)}",
                border_style="red",
                title="[bold]Error[/bold]",
            )
        )
