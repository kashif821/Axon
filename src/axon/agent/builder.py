from __future__ import annotations

import ast
import json
import os
import re
import traceback
import urllib.request
from pathlib import Path
from typing import Callable

from rich.markdown import Markdown
from rich.panel import Panel

from axon.llm.base import ChatMessage, MessageRole
from axon.llm.providers import get_llm_provider
from axon.agent.utils import get_directory_tree
from axon.utils.console import console


def validate_python_code(code: str) -> str | None:
    try:
        ast.parse(code)
        return None
    except SyntaxError:
        return traceback.format_exc()


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

PATCH_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "patch_file",
        "description": "Surgically replace a specific block of text in an existing file. Use this instead of write_file for small changes to large files.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "The path to the file to patch (relative or absolute)",
                },
                "search": {
                    "type": "string",
                    "description": "The exact string to search for. Must match the file exactly, including whitespace and indentation.",
                },
                "replace": {
                    "type": "string",
                    "description": "The new string to replace the search string with.",
                },
            },
            "required": ["filepath", "search", "replace"],
        },
    },
}

FETCH_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch the text content of a URL. Use this to read documentation or API references before writing code.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
            },
            "required": ["url"],
        },
    },
}

BROWSE_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "browse_url",
        "description": (
            "Browse a URL using a real Chrome browser. "
            "Token-efficient: text mode uses ~800 tokens "
            "vs 10,000 for raw HTML. Use this for most "
            "web browsing. Supports reading, clicking, "
            "and filling forms on real websites."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to navigate to"},
                "mode": {
                    "type": "string",
                    "enum": ["text", "snapshot", "screenshot"],
                    "default": "text",
                    "description": (
                        "text: extract page text (~800 tokens), "
                        "snapshot: get clickable elements, "
                        "screenshot: take screenshot"
                    ),
                },
                "click_ref": {
                    "type": "string",
                    "description": "Element ref to click (e.g. e5)",
                },
                "fill_ref": {"type": "string", "description": "Element ref to fill"},
                "fill_value": {
                    "type": "string",
                    "description": "Value to type into element",
                },
            },
            "required": ["url"],
        },
    },
}


async def handle_browse_url(
    url: str, action: str = "text", ref: str = None, value: str = None
) -> str:
    import httpx

    PINCHTAB_BASE = "http://localhost:9867"

    try:
        async with httpx.AsyncClient() as client:
            health = await client.get(f"{PINCHTAB_BASE}/health", timeout=2)
            if health.status_code != 200:
                raise Exception("not running")
    except Exception:
        return (
            "PinchTab is not running. Start it with: pinchtab daemon start\nThen retry."
        )

    async with httpx.AsyncClient() as client:
        if action == "text":
            await client.post(
                f"{PINCHTAB_BASE}/navigate", json={"url": url}, timeout=30
            )
            response = await client.get(f"{PINCHTAB_BASE}/text", timeout=30)
            return response.text[:8000]

        elif action == "snapshot":
            await client.post(
                f"{PINCHTAB_BASE}/navigate", json={"url": url}, timeout=30
            )
            response = await client.get(
                f"{PINCHTAB_BASE}/snapshot?filter=interactive&format=compact",
                timeout=30,
            )
            return response.text

        elif action == "click" and ref:
            response = await client.post(
                f"{PINCHTAB_BASE}/action",
                json={"kind": "click", "ref": ref},
                timeout=15,
            )
            return f"Clicked {ref}"

        elif action == "fill" and ref and value:
            response = await client.post(
                f"{PINCHTAB_BASE}/action",
                json={"kind": "fill", "ref": ref, "value": value},
                timeout=15,
            )
            return f"Filled {ref} with value"

        return "Unknown action"


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
    output_callback: Callable[[str], None] | None = None,
) -> list[str]:
    provider = get_llm_provider()

    def _emit(text: str) -> None:
        if output_callback:
            output_callback(text)
        else:
            console.print(text)

    try:
        tree = get_directory_tree(max_depth=2)
    except Exception:
        tree = "[Unable to read directory structure]"

    system_prompt = (
        BUILDER_SYSTEM_PROMPT
        + "\n\nCurrent Working Directory Structure:\n"
        + tree
        + """

SELF-CORRECTION PROTOCOL:
If you run a shell command and it results in an error or a traceback, DO NOT immediately stop and ask the user for help.
You must autonomously read the error, use the `patch_file` tool to fix the bug in the code, and then use `run_shell_command` to test it again.
Only report back to the user once the command runs successfully, or if you have tried multiple times and are completely stuck.
"""
    )

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(
            role=MessageRole.USER,
            content=f"Please build the following:\n\n{task}",
        ),
    ]

    tools = [
        WRITE_FILE_TOOL,
        READ_FILE_TOOL,
        LIST_DIRECTORY_TOOL,
        PATCH_FILE_TOOL,
        FETCH_URL_TOOL,
        BROWSE_URL_TOOL,
    ]
    written_files: list[str] = []
    iteration_count = 0

    while True:
        iteration_count += 1

        if iteration_count > 15:
            _emit("[bold red]Axon hit the maximum iteration limit (15).[/bold red]")
            return written_files

        _emit("[bold cyan]Thinking...[/bold cyan]")
        try:
            response = await provider.chat(messages, model=model, tools=tools)
        except Exception as e:
            _emit(f"[bold red]API Error Details:[/bold red] {str(e)}")
            return written_files

        choice = response.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", []) or []
        content = getattr(choice.message, "content", None)

        if content and content.strip():
            _emit(content)

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

                elif tc_name == "fetch_url":
                    url = args.get("url")

                    if not url:
                        tool_result = "Error: url parameter is required"
                    else:
                        try:
                            request = urllib.request.Request(
                                url, headers={"User-Agent": "Mozilla/5.0"}
                            )
                            with urllib.request.urlopen(
                                request, timeout=10
                            ) as response:
                                html = response.read().decode("utf-8", errors="ignore")

                            text = re.sub(r"<[^>]+>", " ", html)
                            text = re.sub(r"\s+", " ", text).strip()
                            text = text[:8000]

                            if not text:
                                tool_result = "Error: Fetched content is empty"
                            else:
                                tool_result = text
                        except urllib.error.URLError as e:
                            tool_result = f"Error fetching URL: {e}"
                        except Exception as e:
                            tool_result = f"Error fetching URL: {e}"

                    messages.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=tool_result,
                            tool_call_id=tc_id,
                            name=tc_name,
                        )
                    )

                elif tc_name == "browse_url":
                    from axon.tools.browser import (
                        check_pinchtab,
                        browse_text,
                        browse_snapshot,
                        browse_screenshot,
                        browse_click,
                        browse_fill,
                    )

                    url = args.get("url")
                    mode = args.get("mode", "text")
                    click_ref = args.get("click_ref")
                    fill_ref = args.get("fill_ref")
                    fill_value = args.get("fill_value")

                    if not url:
                        tool_result = "Error: url parameter is required"
                    elif not await check_pinchtab():
                        tool_result = (
                            "PinchTab browser is offline. "
                            "Start with: pinchtab daemon start"
                        )
                    elif click_ref:
                        tool_result = await browse_click(click_ref)
                    elif fill_ref and fill_value:
                        tool_result = await browse_fill(fill_ref, fill_value)
                    elif mode == "snapshot":
                        tool_result = await browse_snapshot(url)
                    elif mode == "screenshot":
                        tool_result = await browse_screenshot()
                    else:
                        tool_result = await browse_text(url)

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
                        _emit(
                            f"[bold red]Error writing file:[/bold red] filepath parameter is required"
                        )
                        tool_result = "Error: filepath parameter is required"
                    elif file_content is None:
                        _emit(
                            f"[bold red]Error writing file:[/bold red] content parameter is required"
                        )
                        tool_result = "Error: content parameter is required"
                    elif filepath.endswith(".py"):
                        syntax_error = validate_python_code(file_content)
                        if syntax_error:
                            tool_result = f"SyntaxError detected before writing:\n{syntax_error}Please fix the syntax and try again."
                            _emit(
                                f"[bold red]Syntax Error detected:[/bold red] {filepath}"
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
                                    _emit(
                                        f"[bold green]File written successfully:[/bold green] {filepath}"
                                    )
                                    tool_result = (
                                        f"File written successfully: {filepath}"
                                    )
                                except Exception as e:
                                    tool_result = f"Error writing file: {e}"
                                    _emit(
                                        f"[bold red]Error writing file:[/bold red] {e}"
                                    )
                            else:
                                tool_result = "User denied permission to write file."
                        else:
                            try:
                                path = Path(filepath)
                                path.parent.mkdir(parents=True, exist_ok=True)
                                path.write_text(file_content)
                                written_files.append(filepath)
                                _emit(
                                    f"[bold green]File written successfully:[/bold green] {filepath}"
                                )
                                tool_result = f"File written successfully: {filepath}"
                            except Exception as e:
                                tool_result = f"Error writing file: {e}"
                                _emit(f"[bold red]Error writing file:[/bold red] {e}")
                    elif confirm_write:
                        from axon.cli.commands.build import ask_confirmation

                        allowed = ask_confirmation(filepath)
                        if allowed:
                            try:
                                path = Path(filepath)
                                path.parent.mkdir(parents=True, exist_ok=True)
                                path.write_text(file_content)
                                written_files.append(filepath)
                                _emit(
                                    f"[bold green]File written successfully:[/bold green] {filepath}"
                                )
                                tool_result = f"File written successfully: {filepath}"
                            except Exception as e:
                                tool_result = f"Error writing file: {e}"
                                _emit(f"[bold red]Error writing file:[/bold red] {e}")
                        else:
                            tool_result = "User denied permission to write file."
                    else:
                        try:
                            path = Path(filepath)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(file_content)
                            written_files.append(filepath)
                            _emit(
                                f"[bold green]File written successfully:[/bold green] {filepath}"
                            )
                            tool_result = f"File written successfully: {filepath}"
                        except Exception as e:
                            tool_result = f"Error writing file: {e}"
                            _emit(f"[bold red]Error writing file:[/bold red] {e}")
                    break

                elif tc_name == "patch_file":
                    filepath = args.get("filepath")
                    search = args.get("search")
                    replace = args.get("replace")

                    if not filepath:
                        _emit(
                            f"[bold red]Error patching file:[/bold red] filepath parameter is required"
                        )
                        tool_result = "Error: filepath parameter is required"
                    elif not search:
                        _emit(
                            f"[bold red]Error patching file:[/bold red] search parameter is required"
                        )
                        tool_result = "Error: search parameter is required"
                    elif replace is None:
                        _emit(
                            f"[bold red]Error patching file:[/bold red] replace parameter is required"
                        )
                        tool_result = "Error: replace parameter is required"
                    else:
                        try:
                            current_content = Path(filepath).read_text()

                            if search not in current_content:
                                tool_result = "Error: Search string not found in file. Ensure exact whitespace and indentation match."
                                _emit(
                                    f"[bold yellow]Search string not found in:[/bold yellow] {filepath}"
                                )
                            else:
                                new_content = current_content.replace(search, replace)

                                if filepath.endswith(".py"):
                                    syntax_error = validate_python_code(new_content)
                                    if syntax_error:
                                        tool_result = f"SyntaxError detected before patching:\n{syntax_error}Please fix the syntax and try again."
                                        _emit(
                                            f"[bold red]Syntax Error detected:[/bold red] {filepath}"
                                        )
                                    else:
                                        _emit(
                                            f"[bold red]⚠️ Axon wants to patch:[/bold red] [yellow]{filepath}[/yellow]"
                                        )
                                        allow = (
                                            input("Allow this patch? [y/N]: ")
                                            .strip()
                                            .lower()
                                        )

                                        if allow == "y":
                                            Path(filepath).write_text(new_content)
                                            tool_result = "File patched successfully."
                                            _emit(
                                                f"[bold green]File patched successfully:[/bold green] {filepath}"
                                            )
                                        else:
                                            tool_result = (
                                                "User denied permission to patch."
                                            )
                                            _emit("[dim]Patch denied by user.[/dim]")
                                else:
                                    _emit(
                                        f"[bold red]⚠️ Axon wants to patch:[/bold red] [yellow]{filepath}[/yellow]"
                                    )
                                    allow = (
                                        input("Allow this patch? [y/N]: ")
                                        .strip()
                                        .lower()
                                    )

                                    if allow == "y":
                                        Path(filepath).write_text(new_content)
                                        tool_result = "File patched successfully."
                                        _emit(
                                            f"[bold green]File patched successfully:[/bold green] {filepath}"
                                        )
                                    else:
                                        tool_result = "User denied permission to patch."
                                        _emit("[dim]Patch denied by user.[/dim]")
                        except FileNotFoundError:
                            tool_result = f"Error: File not found: {filepath}"
                            _emit(
                                f"[bold red]Error patching file:[/bold red] {filepath} not found"
                            )
                        except Exception as e:
                            tool_result = f"Error patching file: {e}"
                            _emit(f"[bold red]Error patching file:[/bold red] {e}")

                    messages.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=tool_result,
                            tool_call_id=tc_id,
                            name=tc_name,
                        )
                    )

                    if tool_result == "File patched successfully.":
                        break

            if last_tool_name not in ("write_file", "patch_file"):
                continue

        if not tool_calls:
            break

    return written_files
