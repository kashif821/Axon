from __future__ import annotations

from typing import Optional

import typer
import yaml
from rich.table import Table

from axon.cli.commands.chat import run_chat_loop
from axon.config.loader import get_config, get_environment_keys
from axon.utils.console import console

app = typer.Typer(
    name="axon",
    help="An open-source, model-agnostic CLI coding agent.",
    add_completion=False,
)


@app.command()
def init() -> None:
    """Initialize axon.yaml in current directory."""
    from pathlib import Path

    from axon.config.loader import DEFAULT_CONFIG

    yaml_path = Path.cwd() / "axon.yaml"

    if yaml_path.exists():
        console.print("[yellow]axon.yaml already exists in current directory[/yellow]")
        return

    with open(yaml_path, "w") as f:
        yaml.safe_dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Created {yaml_path}[/green]")
    console.print("")
    console.print("[bold]Setup instructions:[/bold]")
    console.print("1. Add your API keys to a .env file:")
    console.print("   OPENAI_API_KEY=your_key_here")
    console.print("   ANTHROPIC_API_KEY=your_key_here")
    console.print("   GEMINI_API_KEY=your_key_here")
    console.print("   GROQ_API_KEY=your_key_here")
    console.print("2. Run [bold]axon chat[/bold] to start an interactive session")


@app.command()
def models() -> None:
    """Show configured models and API key status."""
    config = get_config()
    env_keys = get_environment_keys()

    table = Table(
        title="[bold cyan]🔌 Configured Models[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        expand=True,
    )

    table.add_column("Mode", style="white", width=12)
    table.add_column("Model", style="cyan", min_width=35)
    table.add_column("API Key", style="green")

    modes = config.modes
    modes["default"] = config.default_model
    modes["chat"] = modes.get("chat", config.default_model)

    for mode, model in modes.items():
        provider = model.split("/")[0] if "/" in model else model
        has_key = env_keys.get(provider, False)
        status = (
            "[green]✅ key found[/green]" if has_key else "[red]❌ key missing[/red]"
        )
        table.add_row(mode, model, status)

    console.print(table)


@app.command()
def chat(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (overrides config)",
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

    from axon.memory.store import init_db
    from axon.cli.commands.chat import run_chat_loop
    from axon.config.loader import get_config, get_environment_keys
    from axon.utils.console import console

    asyncio.run(init_db())

    config = get_config()
    final_model = config.merge_cli(model=model)
    asyncio.run(run_chat_loop(model=final_model, session_id=session))


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

    from axon.memory.store import init_db
    from axon.cli.commands.ask import run_ask
    from axon.config.loader import get_config

    asyncio.run(init_db())
    config = get_config()
    final_model = config.merge_cli(model=model)
    asyncio.run(run_ask(question=question, model=final_model))


@app.command()
def brain(
    subcommand: str = typer.Argument("status", help="Subcommand: status, start"),
) -> None:
    """Manage Axon's memory brain."""
    import asyncio

    from axon.memory.store import init_db
    from axon.cli.commands.brain import manage_brain

    asyncio.run(init_db())
    asyncio.run(manage_brain(subcommand=subcommand))


@app.command()
def build(
    task: str = typer.Argument(..., help="Task description for building"),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (overrides config)",
    ),
) -> None:
    """Build code based on a task description."""
    import asyncio

    from axon.memory.store import init_db
    from axon.cli.commands.build import run_build
    from axon.config.loader import get_config

    asyncio.run(init_db())
    config = get_config()
    final_model = config.merge_cli(model=model)
    asyncio.run(run_build(task, model=final_model))


@app.command()
def plan(
    task: str = typer.Argument(..., help="Task to plan"),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (overrides config)",
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

    from axon.memory.store import init_db
    from axon.cli.commands.plan import stream_plan
    from axon.config.loader import get_config

    asyncio.run(init_db())
    config = get_config()
    final_model = config.merge_cli(model=model)
    asyncio.run(stream_plan(task, model=final_model, execute=execute))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
