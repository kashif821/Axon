from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from axon.llm.base import ChatMessage, MessageRole
from axon.llm.providers import get_llm_provider

console = Console()

WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file at the specified path. Creates parent directories if they don't exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "The path to the file to write (relative or absolute)",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["filepath", "content"],
        },
    },
}

BUILDER_SYSTEM_PROMPT = """You are a Senior Software Developer building production-ready code. Your task is to implement software based on user requests.

When given a task, you MUST use the write_file tool to create or modify files. Do NOT output raw code directly.

Follow these steps:
1. Understand the task requirements
2. Plan the file structure
3. Write each file using the write_file tool
4. Only write complete, working code

Important rules:
- Use the write_file tool for EVERY file you create
- Write complete, production-ready code (no placeholders or TODO comments in code)
- Always include proper imports, error handling, and type hints
- Create a README.md explaining the project if it's a new codebase
- Ensure all dependencies are listed in requirements.txt or pyproject.toml
"""


async def build_task(
    task: str,
    model: str | None = None,
    confirm_write: bool = True,
) -> list[str]:
    provider = get_llm_provider()
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=BUILDER_SYSTEM_PROMPT),
        ChatMessage(
            role=MessageRole.USER,
            content=f"Please build the following:\n\n{task}",
        ),
    ]

    tools = [WRITE_FILE_TOOL]
    written_files: list[str] = []

    while True:
        with console.status("[bold cyan]Axon is thinking...[/bold cyan]"):
            response = await provider.chat(messages, model=model, tools=tools)

        choice = response.choices[0]
        assistant_message = choice.message

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                func = tool_call.function

                if func.name == "write_file":
                    args = json.loads(func.arguments)
                    filepath = args["filepath"]
                    content = args["content"]

                    if confirm_write:
                        from axon.cli.commands.build import ask_confirmation

                        allowed = ask_confirmation(filepath)
                    else:
                        allowed = True

                    if allowed:
                        try:
                            path = Path(filepath)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(content)
                            written_files.append(filepath)
                            console.print(
                                f"[bold green]File written successfully:[/bold green] {filepath}"
                            )
                        except Exception as e:
                            console.print(
                                f"[bold red]Error writing file:[/bold red] {e}"
                            )
                    else:
                        console.print(
                            f"[bold yellow]User denied permission to write:[/bold yellow] {filepath}"
                        )
            break
        else:
            messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=assistant_message.content,
                )
            )
            if assistant_message.content:
                console.print(
                    Panel(
                        assistant_message.content,
                        border_style="yellow",
                        title="[bold]Axon Response[/bold]",
                    )
                )
            break

    return written_files
