from __future__ import annotations

from typing import Optional

from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm

from axon.agent.builder import build_task
from axon.llm.providers import LLMConfigurationError, LLMError
from axon.utils.console import console, create_axon_layout, update_layout_content


def ask_confirmation(filepath: str) -> bool:
    return Confirm.ask(
        f"[bold yellow]Allow Axon to write to {filepath}?[/bold yellow]",
        default=False,
    )


async def run_build(task: str, model: Optional[str] = None) -> None:
    layout = create_axon_layout(
        current_model=model or "openai/moonshotai/kimi-k2-thinking"
    )
    modified_files: list[str] = []
    tokens_used = 0
    output_buffer = ""

    def update_output(new_content: str) -> None:
        nonlocal output_buffer
        output_buffer += new_content + "\n"
        layout["main"].update(
            Panel(
                output_buffer,
                title="[bold]Axon Builder[/bold]",
                border_style="blue",
                padding=(1, 1),
            )
        )

    try:
        with Live(layout, console=console, refresh_per_second=4, transient=False):
            written_files = await build_task(
                task, model=model, output_callback=update_output
            )
            modified_files = written_files

            if written_files:
                update_output(
                    f"[bold green]Successfully created {len(written_files)} file(s):[/bold green]\n\n"
                    + "\n".join(f"  - {f}" for f in written_files)
                    + "\n\n[dim]Build complete![/dim]"
                )
            else:
                update_output(
                    "[dim]No files were written.[/dim]\n\n[dim]Build complete![/dim]"
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
