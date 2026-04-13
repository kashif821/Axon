from __future__ import annotations

from typing import AsyncIterator

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from axon.agent.planner import generate_plan
from axon.llm.providers import LLMConfigurationError, LLMError
from axon.utils.console import console


async def stream_plan(
    task: str, model: str | None = None, execute: bool = False
) -> None:
    full_response = ""

    try:
        with Live(
            console=console,
            refresh_per_second=15,
            transient=False,
        ) as live:
            async for chunk in generate_plan(task, model=model):
                full_response += chunk
                live.update(
                    Panel(
                        Markdown(full_response),
                        border_style="cyan",
                        title="[bold]Axon Plan[/bold]",
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

    if execute and full_response:
        from axon.cli.commands.build import run_build

        console.print(
            "\n[bold purple]⚙️ Handing blueprint over to the Builder Agent...[/bold purple]\n"
        )
        await run_build(
            task=f"Please execute the following architectural plan:\n\n{full_response}",
            model=model,
        )
