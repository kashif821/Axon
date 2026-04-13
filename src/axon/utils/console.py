from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "thought": "italic magenta",
    }
)

console = Console(theme=custom_theme)


def create_axon_layout(
    current_model: str = "openai/moonshotai/kimi-k2-thinking",
    tokens_used: int = 0,
    modified_files: list[str] | None = None,
) -> Layout:
    modified_files = modified_files or []

    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    layout["body"].split_row(
        Layout(name="main", ratio=3),
        Layout(name="sidebar", ratio=1),
    )

    header_text = Text.assemble(
        ("  Axon Autonomous Agent ", "bold cyan"),
        ("v0.1.0", "dim"),
    )
    layout["header"].update(
        Panel(
            header_text,
            style="on cyan",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    layout["main"].update(
        Panel(
            "[dim]Initializing builder...[/dim]\n",
            title="[bold]Axon Builder[/bold]",
            border_style="blue",
            padding=(1, 1),
        )
    )

    sidebar_table = Table(title="[bold]Context[/bold]", show_header=False, box=None)
    sidebar_table.add_column("key", style="cyan")
    sidebar_table.add_column("value", style="white")
    sidebar_table.add_row("Model", current_model)
    sidebar_table.add_row("Tokens", str(tokens_used))
    sidebar_table.add_row("Files Modified", str(len(modified_files)))

    for f in modified_files[:10]:
        sidebar_table.add_row("  ", f"[dim]{f}[/dim]")

    layout["sidebar"].update(
        Panel(
            sidebar_table,
            title="[bold]Info[/bold]",
            border_style="green",
            padding=(1, 1),
        )
    )

    footer_text = Text.assemble(
        ("  ctrl+c", "bold yellow"),
        (" to interrupt  |  ", "dim"),
        ("ctrl+d", "bold yellow"),
        (" to exit", "dim"),
    )
    layout["footer"].update(
        Panel(
            footer_text,
            style="on yellow",
            border_style="yellow",
            padding=(0, 1),
        )
    )

    return layout


def update_layout_content(
    layout: Layout,
    content: str,
    modified_files: list[str] | None = None,
    tokens_used: int = 0,
) -> None:
    layout["main"].update(
        Panel(
            content,
            title="[bold]Axon Builder[/bold]",
            border_style="blue",
            padding=(1, 1),
        )
    )

    if modified_files is not None:
        sidebar_table = Table(show_header=False, box=None)
        sidebar_table.add_column("key", style="cyan")
        sidebar_table.add_column("value", style="white")
        sidebar_table.add_row("Tokens", str(tokens_used))
        sidebar_table.add_row("Files Modified", str(len(modified_files)))

        for f in modified_files[:10]:
            sidebar_table.add_row("  ", f"[dim]{f}[/dim]")

        layout["sidebar"].update(
            Panel(
                sidebar_table,
                title="[bold]Info[/bold]",
                border_style="green",
                padding=(1, 1),
            )
        )


__all__ = ["console", "custom_theme", "create_axon_layout", "update_layout_content"]
