from __future__ import annotations

import json
import os
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from axon.llm.base import ChatMessage, MessageRole
from axon.llm.providers import get_llm_provider
from axon.agent.utils import get_directory_tree

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

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file from the filesystem.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "The path to the file to read (relative or absolute)",
                },
            },
            "required": ["filepath"],
        },
    },
}

LIST_DIRECTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "list_directory",
        "description": "List the contents of a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the directory to list (default: current directory)",
                    "default": ".",
                },
            },
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

    try:
        tree = get_directory_tree(max_depth=2)
    except Exception:
        tree = "[Unable to read directory structure]"

    system_prompt = (
        BUILDER_SYSTEM_PROMPT + "\n\nCurrent Working Directory Structure:\n" + tree
    )

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(
            role=MessageRole.USER,
            content=f"Please build the following:\n\n{task}",
        ),
    ]

    tools = [WRITE_FILE_TOOL, READ_FILE_TOOL, LIST_DIRECTORY_TOOL]
    written_files: list[str] = []

    while True:
        with console.status("[bold cyan]Axon is thinking...[/bold cyan]"):
            try:
                response = await provider.chat(messages, model=model, tools=tools)
            except Exception as e:
                console.print(
                    "[bold red]API Error:[/bold red] The AI provider is currently overloaded or rate-limited. Please wait a moment and try again."
                )
                return written_files

        choice = response.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", []) or []
        content = getattr(choice.message, "content", None)

        if content and content.strip():
            console.print(
                Panel(
                    Markdown(content),
                    title="[bold]Axon[/bold]",
                    border_style="blue",
                )
            )

        if tool_calls:
            assistant_msg_dict = {
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls,
            }
            messages.append(assistant_msg_dict)

            last_tool_name = None
            for tc in tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else tc.id
                func = tc.get("function") if isinstance(tc, dict) else tc.function
                tc_name = func.get("name") if isinstance(func, dict) else func.name
                last_tool_name = tc_name

                func_arguments = (
                    func.get("arguments") if isinstance(func, dict) else func.arguments
                )
                args_str = func_arguments if func_arguments else "{}"
                try:
                    args = json.loads(args_str) or {}
                except Exception:
                    args = {}

                if tc_name == "read_file":
                    filepath = args.get("filepath")
                    if not filepath:
                        tool_result = "Error: filepath parameter is required"
                    else:
                        try:
                            file_content = Path(filepath).read_text()
                            tool_result = file_content
                        except FileNotFoundError:
                            tool_result = f"File not found: {filepath}"
                        except Exception as e:
                            tool_result = f"Error reading file: {e}"

                    messages.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=tool_result,
                            tool_call_id=tc_id,
                            name=tc_name,
                        )
                    )

                elif tc_name == "list_directory":
                    path = args.get("path", ".")
                    try:
                        contents = os.listdir(path)
                        tool_result = ", ".join(sorted(contents))
                    except FileNotFoundError:
                        tool_result = f"Directory not found: {path}"
                    except Exception as e:
                        tool_result = f"Error listing directory: {e}"

                    messages.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=tool_result,
                            tool_call_id=tc_id,
                            name=tc_name,
                        )
                    )

                elif tc_name == "write_file":
                    filepath = args.get("filepath")
                    file_content = args.get("content")

                    if not filepath:
                        console.print(
                            f"[bold red]Error writing file:[/bold red] filepath parameter is required"
                        )
                    elif file_content is None:
                        console.print(
                            f"[bold red]Error writing file:[/bold red] content parameter is required"
                        )
                    elif confirm_write:
                        from axon.cli.commands.build import ask_confirmation

                        allowed = ask_confirmation(filepath)
                        if allowed:
                            try:
                                path = Path(filepath)
                                path.parent.mkdir(parents=True, exist_ok=True)
                                path.write_text(file_content)
                                written_files.append(filepath)
                                console.print(
                                    f"[bold green]File written successfully:[/bold green] {filepath}"
                                )
                            except Exception as e:
                                console.print(
                                    f"[bold red]Error writing file:[/bold red] {e}"
                                )
                    else:
                        try:
                            path = Path(filepath)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(file_content)
                            written_files.append(filepath)
                            console.print(
                                f"[bold green]File written successfully:[/bold green] {filepath}"
                            )
                        except Exception as e:
                            console.print(
                                f"[bold red]Error writing file:[/bold red] {e}"
                            )
                    break

            if last_tool_name != "write_file":
                continue

        if not tool_calls:
            break

    return written_files
