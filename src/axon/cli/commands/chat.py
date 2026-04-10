from __future__ import annotations

import asyncio
from typing import AsyncIterator

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

from axon.llm.base import ChatMessage, MessageRole
from axon.llm.providers import LLMConfigurationError, LLMError, get_llm_provider


console = Console()
_history = InMemoryHistory()


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
    except LLMError as e:
        raise


async def run_chat_loop(model: str | None = None) -> None:
    messages: list[ChatMessage] = []
    session = PromptSession(history=_history)

    _print_welcome()

    while True:
        try:
            user_input = await session.prompt_async(
                "\n[bold blue]>[/bold blue] ",
                auto_suggest=AutoSuggestFromHistory(),
            )

            if not user_input.strip():
                continue

            user_input = user_input.strip()

            if user_input.lower() in ("/exit", "/quit", "/q"):
                _print_exit_message()
                break

            if user_input.lower() in ("/clear", "/reset"):
                messages = []
                console.print("\n[dim]Conversation cleared.[/dim]\n")
                continue

            if user_input.lower() in ("/help", "/h"):
                _print_help()
                continue

            messages.append(ChatMessage(role=MessageRole.USER, content=user_input))

            console.print()

            try:
                full_response = ""
                with Live(
                    console=console,
                    refresh_per_second=15,
                    transient=False,
                ) as live:
                    async for chunk in stream_response(messages, model=model):
                        full_response += chunk
                        live.update(
                            Panel(
                                Markdown(full_response),
                                border_style="green",
                                title="[bold]Axon[/bold]",
                            )
                        )

                messages.append(
                    ChatMessage(role=MessageRole.ASSISTANT, content=full_response)
                )

                console.print()

            except LLMConfigurationError as e:
                console.print(
                    Panel(
                        f"[bold red]Configuration Error[/bold red]\n\n{e}",
                        border_style="red",
                        title="Error",
                    )
                )
                messages.pop()
                console.print(
                    "[dim]Tip: Check your .env file and ensure you have a valid API key.[/dim]\n"
                )

            except LLMError as e:
                console.print(
                    Panel(
                        f"[bold red]LLM Error[/bold red]\n\n{e}",
                        border_style="red",
                        title="Error",
                    )
                )
                messages.pop()

        except KeyboardInterrupt:
            console.print("\n[dim]Use /exit or /quit to exit.[/dim]\n")
            continue
        except EOFError:
            _print_exit_message()
            break


def _print_welcome() -> None:
    welcome_text = """
# Welcome to Axon Chat

An open-source, model-agnostic CLI coding agent.

**Commands:**
- `/exit`, `/quit` - Exit the chat
- `/clear` - Clear conversation history
- `/help` - Show this help message
"""
    console.print(Markdown(welcome_text))


def _print_exit_message() -> None:
    console.print("\n[dim]Goodbye![/dim]\n")


def _print_help() -> None:
    help_text = """
## Available Commands

| Command | Description |
|---------|-------------|
| `/exit`, `/quit` | Exit the chat |
| `/clear` | Clear conversation history |
| `/help` | Show this help message |

Type your message and press Enter to chat with the AI.
"""
    console.print(Markdown(help_text))
