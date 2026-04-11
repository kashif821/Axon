from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from axon.memory.store import get_all_sessions


console = Console()


async def manage_brain(subcommand: str = "status") -> None:
    if subcommand == "status":
        await _show_status()
    else:
        console.print(
            Panel(
                f"[bold red]Unknown subcommand:[/bold red] {subcommand}\n\n"
                "Available subcommands: status",
                border_style="red",
                title="[bold]Error[/bold]",
            )
        )


async def _show_status() -> None:
    sessions = await get_all_sessions()

    if not sessions:
        console.print(
            Panel(
                "[dim]No sessions found. Start a new chat with [bold]axon chat[/bold] to create your first session.[/dim]",
                border_style="cyan",
                title="[bold]Axon Brain Status[/bold]",
            )
        )
        return

    table = Table(
        title="[bold cyan]Axon Brain - Sessions[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        expand=True,
    )

    table.add_column("Session ID", style="dim", width=38)
    table.add_column("Title", style="white", min_width=30)
    table.add_column("Last Updated", style="green", width=20)

    for session in sessions:
        session_id = str(session.id)
        table.add_row(
            session_id,
            session.title or "[dim]Untitled[/dim]",
            session.updated_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
