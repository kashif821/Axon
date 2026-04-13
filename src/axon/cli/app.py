from __future__ import annotations

from typing import Optional

import typer

from axon.cli.commands.chat import run_chat_loop
from axon.utils.console import console

app = typer.Typer(
    name="axon",
    help="An open-source, model-agnostic CLI coding agent.",
    add_completion=False,
)


@app.command()
def chat(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (e.g., gpt-4, claude-3, groq/llama-3)",
    ),
    session: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Resume a previous session by its UUID",
    ),
) -> None:
    """Start an interactive chat session with Axon."""
    import asyncio

    asyncio.run(run_chat_loop(model=model, session_id=session))


@app.command()
def ask(
    question: str = typer.Argument(
        ..., help="The question to ask Axon about your recent activity"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use",
    ),
) -> None:
    """Ask Axon about your recent activity."""
    import asyncio

    from axon.cli.commands.ask import run_ask

    asyncio.run(run_ask(question=question, model=model))


@app.command()
def brain(
    subcommand: str = typer.Argument("status", help="Subcommand: status"),
) -> None:
    """Manage Axon's memory brain."""
    import asyncio

    from axon.cli.commands.brain import manage_brain

    asyncio.run(manage_brain(subcommand=subcommand))


@app.command()
def build(
    task: str = typer.Argument(..., help="Task description for building"),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (e.g., gpt-4, nvidia_nim/moonshotai/kimi-k2-thinking)",
    ),
) -> None:
    """Build code based on a task description."""
    import asyncio

    from axon.cli.commands.build import run_build

    asyncio.run(run_build(task, model=model))


@app.command()
def plan(
    task: str = typer.Argument(..., help="Task to plan"),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (e.g., gpt-4, nvidia_nim/moonshotai/kimi-k2-thinking)",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        "-e",
        help="Execute the plan using the Builder Agent after generating it",
    ),
) -> None:
    """Plan steps for a complex task."""
    import asyncio

    from axon.cli.commands.plan import stream_plan

    asyncio.run(stream_plan(task, model=model, execute=execute))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
