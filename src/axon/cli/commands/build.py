from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from axon.agent.builder import build_task
from axon.llm.providers import LLMConfigurationError, LLMError


console = Console()


def ask_confirmation(filepath: str) -> bool:
    return Confirm.ask(
        f"[bold yellow]Allow Axon to write to {filepath}?[/bold yellow]",
        default=False,
    )


async def run_build(task: str, model: Optional[str] = None) -> None:
    try:
        written_files = await build_task(task, model=model)

        if written_files:
            console.print(
                Panel(
                    f"[bold green]Successfully created {len(written_files)} file(s):[/bold green]\n"
                    + "\n".join(f"  - {f}" for f in written_files),
                    border_style="green",
                    title="[bold]Build Complete[/bold]",
                )
            )
        else:
            console.print(
                Panel(
                    "[dim]No files were written.[/dim]",
                    border_style="yellow",
                    title="[bold]Build Complete[/bold]",
                )
            )

    except LLMConfigurationError as e:
        console.print(
            Panel(
                f"[bold red]Configuration Error[/bold red]\n\n{e}",
                border_style="red",
                title="Error",
            )
        )
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
