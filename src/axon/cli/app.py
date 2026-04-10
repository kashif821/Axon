from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from axon.cli.commands.chat import run_chat_loop

app = typer.Typer(
    name="axon",
    help="An open-source, model-agnostic CLI coding agent.",
    add_completion=False,
)

console = Console()


@app.command()
def chat(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (e.g., gpt-4, claude-3, groq/llama-3)",
    ),
) -> None:
    """Start an interactive chat session with Axon."""
    import asyncio

    asyncio.run(run_chat_loop(model=model))


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="The question or task to ask Axon"),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use",
    ),
) -> None:
    """Ask Axon a single question without interactive mode."""
    console.print("[dim]ask command not yet implemented[/dim]")


@app.command()
def brain(
    subcommand: str = typer.Argument(None, help="Subcommand: status, reset, sync"),
) -> None:
    """Manage Axon's memory brain."""
    console.print("[dim]brain command not yet implemented[/dim]")


@app.command()
def build(
    task: str = typer.Argument(..., help="Task description for building"),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (e.g., gpt-4, claude-3, gemini/gemini-2.5-flash)",
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
        help="Model to use (e.g., gpt-4, claude-3, gemini/gemini-2.5-flash)",
    ),
) -> None:
    """Plan steps for a complex task."""
    import asyncio

    from axon.cli.commands.plan import stream_plan

    asyncio.run(stream_plan(task, model=model))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
