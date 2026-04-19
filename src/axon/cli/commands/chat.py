from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv

from textual.app import App, ComposeResult
from textual.command import Provider, Hit, Hits
from textual import work
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Button, Label, Select, RichLog, Static, Footer
from textual.reactive import reactive
from textual.binding import Binding

from axon.llm.base import ChatMessage, MessageRole
from axon.llm.providers import get_llm_provider
from axon.memory.store import log_action


def get_available_providers() -> dict:
    import os
    import litellm

    provider_keys = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "nvidia_nim": "NVIDIA_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "together_ai": "TOGETHER_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "ollama": None,
    }
    available = {}
    for provider, key in provider_keys.items():
        if key is None or os.environ.get(key):
            models = litellm.models_by_provider.get(provider, [])
            if models:
                available[provider] = models
    return available


def get_yaml_models() -> list:
    try:
        from axon.config.loader import get_config

        config = get_config()
        if hasattr(config, "modes"):
            return list(config.modes.__dict__.values())
    except Exception:
        pass
    return []


def build_fallback_chain(primary_model: str, config) -> list:
    """Build a fallback chain sorted by cost (free/cheap first)."""
    chain = [primary_model]

    try:
        providers = getattr(config, "providers", {})
        for provider, models in providers.items():
            if not isinstance(models, dict):
                continue
            for model_id, info in models.items():
                if not isinstance(info, dict):
                    continue
                if model_id == primary_model:
                    continue
                cost = info.get("cost_in_1m", 999)
                chain.append((model_id, cost))

        # Sort by cost ascending (free/cheap first)
        if len(chain) > 1:
            chain[1:] = sorted(chain[1:], key=lambda x: x[1])
            return [chain[0]] + [m[0] for m in chain[1:]]
    except Exception:
        pass

    # Fallback to hardcoded list if config parsing fails
    return [primary_model, "groq/llama3-8b-8192", "openai/gpt-4o-mini"]


def get_all_available_models() -> dict:
    import os

    models = {}

    # Source A: yaml config
    yaml_models = get_yaml_models()
    for m in yaml_models:
        if m:
            provider = m.split("/")[0] if "/" in m else "yaml"
            if provider not in models:
                models[provider] = []
            if m not in models[provider]:
                models[provider].append(m)

    # Source B & C: LiteLLM providers with API keys
    litellm_providers = get_available_providers()
    for provider, provider_models in litellm_providers.items():
        if provider not in models:
            models[provider] = []
        for m in provider_models:
            if m not in models[provider]:
                models[provider].append(m)

    return models


def validate_model(model: str) -> bool:
    try:
        import litellm

        info = litellm.get_model_info(model)
        return info is not None
    except Exception:
        return False


def extract_thinking(chunk) -> str | None:
    """Extract reasoning/thinking content from LiteLLM chunk."""
    if hasattr(chunk, "reasoning_content"):
        return chunk.reasoning_content

    choices = getattr(chunk, "choices", [])
    if choices:
        delta = getattr(choices[0], "delta", None)
        if delta:
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                return rc
            content = getattr(delta, "content", "") or ""
            if content.startswith("<think>"):
                return content[7:]
    return None


TOOL_ICONS = {
    "write_file": "✎",
    "read_file": "◉",
    "patch_file": "⚡",
    "run_shell_command": "⚙",
    "fetch_url": "🌐",
    "list_directory": "📁",
    "search": "🔍",
    "grep": "📋",
}


def format_tool_call(name: str, args: dict) -> str:
    """Format tool call for display in chat."""
    icon = TOOL_ICONS.get(name, "→")

    if name == "write_file":
        return f"{icon} Writing {args.get('path', '')}"
    if name == "read_file":
        return f"{icon} Reading {args.get('path', '')}"
    if name == "run_shell_command":
        cmd = args.get("command", "")
        return f"{icon} Running {cmd[:50]}{'...' if len(cmd) > 50 else ''}"
    if name == "fetch_url":
        return f"{icon} Fetching {args.get('url', '')}"
    if name == "list_directory":
        return f"{icon} Listing {args.get('path', '.')}"
    if name == "search":
        return f"{icon} Searching {args.get('query', '')}"
    if name == "grep":
        return f"{icon} Grep {args.get('pattern', '')}"

    return f"{icon} {name}: {str(args)[:40]}"


def get_recent_file_changes() -> list:
    """Get recent file changes from the watcher."""
    try:
        from axon.watcher.monitor import get_recent_changes

        return get_recent_changes()
    except Exception:
        return []


def get_brain_status() -> str:
    """Get current brain status for sidebar display."""
    changes = get_recent_file_changes()
    if changes:
        return f"🧠 Changes detected ({len(changes)} files)"
    return "🧠 Observing"


def get_context_window(model: str) -> int:
    try:
        import litellm

        info = litellm.get_model_info(model)
        return info.get("max_tokens", 128_000)
    except Exception:
        return 128_000


def get_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    try:
        import litellm

        cost = litellm.completion_cost(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return cost or 0.0
    except Exception:
        return 0.0


def is_git_repo(path: str) -> bool:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=path,
        capture_output=True,
    )
    return result.returncode == 0


def get_file_stats(project_root: str) -> list:
    import subprocess

    result = subprocess.run(
        ["git", "diff", "--numstat"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=5,
    )
    files = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            added, removed, path = parts
            files.append(
                {
                    "path": path,
                    "added": int(added) if added.isdigit() else 0,
                    "removed": int(removed) if removed.isdigit() else 0,
                }
            )
    return files


_file_cache = {}


def get_line_diff(filepath: str) -> tuple:
    import difflib

    try:
        with open(filepath) as f:
            current = f.readlines()
        previous = _file_cache.get(filepath, [])
        _file_cache[filepath] = current

        added = sum(
            1
            for l in difflib.unified_diff(previous, current, lineterm="")
            if l.startswith("+") and not l.startswith("+++")
        )
        removed = sum(
            1
            for l in difflib.unified_diff(previous, current, lineterm="")
            if l.startswith("-") and not l.startswith("---")
        )
        return added, removed
    except Exception:
        return 0, 0


try:
    import aiosqlite
except ImportError:
    aiosqlite = None


class ModelCommandProvider(Provider):
    """Provides dynamic commands to switch LLM models based on available providers."""

    async def search(self, query: str) -> Hits:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        matcher = self.matcher(query)
        available_models = {}

        # Source A: yaml config models
        yaml_models = get_yaml_models()
        for m in yaml_models:
            if m:
                parts = m.split("/")
                provider = parts[0] if len(parts) > 1 else "yaml"
                available_models[m] = f"{parts[-1]} (via {provider})"

        # Source B & C: LiteLLM providers with API keys
        litellm_providers = get_available_providers()
        for provider, models in litellm_providers.items():
            for m in models:
                if m not in available_models:
                    pretty = m.split("/")[-1] if "/" in m else m
                    available_models[m] = f"{pretty} (via {provider})"

        if not available_models:
            available_models["gemini/gemini-2.5-flash"] = "gemini-2.5-flash (Fallback)"

        # Add custom model entry
        available_models["__custom__"] = "Enter custom model..."

        # Add session management commands
        available_models["__export_session__"] = "Export Session to Markdown"
        available_models["__new_session__"] = "Create New Session"
        available_models["__list_sessions__"] = "List All Sessions"

        for raw_id, pretty_name in available_models.items():
            display = f"Set Model: {pretty_name}"

            # Handle special commands
            if raw_id.startswith("__") and raw_id.endswith("__"):
                if raw_id == "__export_session__":
                    action = lambda: self.app.action_export_session()
                elif raw_id == "__new_session__":
                    action = lambda: self.app.action_new_session()
                elif raw_id == "__list_sessions__":
                    action = lambda: self.app.action_show_session_list()
                elif raw_id == "__custom__":
                    action = lambda: self.app.action_prompt_custom_model()
                else:
                    action = lambda: None
            else:
                action = lambda m=raw_id: self.app.switch_active_model(m)

            if not query:
                yield Hit(
                    1.0,
                    display,
                    action,
                    help=f"Provider: {raw_id.split('/')[0]}"
                    if "/" in raw_id
                    else "Custom",
                )
            else:
                score = matcher.match(display)
                if score > 0:
                    yield Hit(
                        score,
                        matcher.highlight(display),
                        action,
                        help=f"Provider: {raw_id.split('/')[0]}"
                        if "/" in raw_id
                        else "Custom",
                    )
                    yield Hit(
                        score,
                        matcher.highlight(display),
                        action,
                        help=f"Provider: {raw_id.split('/')[0]}",
                    )


class APIKeyModal(ModalScreen):
    """A beautiful dialog to update API keys directly from the UI."""

    CSS = """
    APIKeyModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #key-dialog {
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    #key-dialog Horizontal {
        height: auto;
        align: right middle;
        margin-top: 1;
        gap: 1;
    }
    """

    def compose(self):
        with Vertical(id="key-dialog"):
            yield Label(
                "[bold]Configure API Keys[/bold]\nSelect a provider and paste your new key.",
                classes="box",
            )

            yield Select(
                [
                    ("Google Gemini", "GEMINI_API_KEY"),
                    ("OpenAI", "OPENAI_API_KEY"),
                    ("Anthropic", "ANTHROPIC_API_KEY"),
                    ("Groq", "GROQ_API_KEY"),
                    ("NVIDIA NIM", "NVIDIA_NIM_API_KEY"),
                    ("Together AI", "TOGETHER_API_KEY"),
                ],
                prompt="Select Provider",
                id="provider-select",
            )

            yield Input(
                placeholder="Paste your sk-... key here", password=True, id="key-input"
            )

            with Horizontal():
                yield Button("Cancel", variant="error", id="cancel-btn")
                yield Button("Save Key", variant="success", id="save-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.dismiss()
        elif event.button.id == "save-btn":
            provider_env = self.query_one("#provider-select", Select).value
            new_key = self.query_one("#key-input", Input).value

            if provider_env and provider_env != Select.BLANK and new_key:
                self.save_to_env(str(provider_env), new_key)
                self.app.notify(f"Key saved for {provider_env}!", title="Success")
                self.dismiss()
            else:
                self.app.notify(
                    "Please select a provider and enter a key.", severity="error"
                )

    def save_to_env(self, key_name: str, key_value: str) -> None:
        import os

        try:
            from dotenv import set_key

            env_path = os.path.join(os.getcwd(), ".env")
            if not os.path.exists(env_path):
                open(env_path, "a").close()
            set_key(env_path, key_name, key_value)
        except ImportError:
            self.app.notify("Please run: pip install python-dotenv", severity="error")


class AxonApp(App):
    CSS = """
    Screen { background: #0d0d0d; }
    #main-layout { height: 100%; width: 100%; }
    #left-column { width: 1fr; padding: 0 1; }
    #chat-log { height: 1fr; }
    
    /* Header bar with session title and brain status */
    #header-bar {
        height: auto;
        dock: top;
        background: #1a1a1a;
        border-bottom: solid #333333;
        padding: 0 1;
    }
    #header-title {
        color: #3b82f6;
    }
    #header-brain {
        color: #888888;
    }
    
    /* The blue-bordered input block */
    #input-box {
        height: auto;
        border-left: outer #3b82f6;
        padding-left: 1;
        margin-bottom: 1;
    }
    #chat-input { border: none; width: 100%; background: transparent; padding: 0; }
    #status-line { color: #888888; margin-top: 1; }
    
    /* The Sidebar */
    #sidebar { width: 35; border-left: none; padding: 1 2; background: #111111; }
    #sidebar-modified { height: 2fr; overflow-y: auto; }
    #sidebar-workspace { height: auto; dock: bottom; border-top: solid #333333; padding: 0 1; }
    
    /* Thinking display */
    #thinking-display {
        color: #555555;
        text-style: italic;
        margin-bottom: 1;
    }
    
    /* Session List Overlay */
    #session-overlay {
        width: 60%;
        height: 60%;
        background: #1a1a1a;
        border: solid #3b82f6;
        padding: 1;
    }
    #session-list { height: 1fr; }
    #session-search { margin-bottom: 1; }
    """

    COMMANDS = App.COMMANDS | {ModelCommandProvider}

    BINDINGS = [
        Binding("tab", "cycle_mode", "Switch Mode", priority=True),
        Binding("ctrl+t", "cycle_variant", "Variants", priority=True),
        Binding("ctrl+p", "command_palette", "Commands", priority=True),
        Binding("ctrl+k", "open_keys", "API Keys", priority=True),
        Binding("ctrl+x l", "show_session_list", "Sessions", priority=True),
        Binding("ctrl+x n", "new_session", "New Session", priority=True),
        Binding("ctrl+r", "rename_session", "Rename", priority=True),
    ]

    current_mode = reactive("chat")
    token_variant = reactive("high")
    session_title = reactive("New Session")
    brain_status = reactive("🧠 Observing")
    current_session_id = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = "gemini/gemini-2.5-flash"
        self.current_session_id = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self._awaiting_rename = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-layout"):
            with Vertical(id="left-column"):
                yield Static(
                    "[bold]Axon[/bold] [cyan]│[/cyan] [session_title] [dim]│[/dim] [model] [dim]│[/dim] [brain_status]",
                    markup=True,
                    id="header-bar",
                )
                yield RichLog(id="chat-log", markup=True, wrap=True)
                yield Static("", id="live-stream", markup=True)
                yield Static("", id="thinking-display", markup=True)
                with Vertical(id="input-box"):
                    yield Input(id="chat-input", placeholder="Ask anything...")
                    yield Static("", id="status-line", markup=True)

            with Vertical(id="sidebar"):
                yield Static(
                    "[bold]Context[/bold]\n0 tokens\nInitializing...\n",
                    markup=True,
                    id="sidebar-context",
                )
                yield Static(
                    "[bold]Modified Files[/bold]\nLoading...\n",
                    markup=True,
                    id="sidebar-modified",
                )
                yield Static(
                    "[bold]Workspace[/bold]\nLoading...\n",
                    markup=True,
                    id="sidebar-workspace",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat-input").focus()
        self.update_status_line()

        # Start the real-time polling loop (runs every 2 seconds)
        self.set_interval(2.0, self.fetch_sidebar_data)

        # Do an immediate initial fetch
        self.run_worker(self.fetch_sidebar_data())

    async def fetch_sidebar_data(self) -> None:
        """Polls SQLite and the local filesystem to update the sidebar dynamically."""
        import os, subprocess, aiosqlite

        cwd = os.getcwd()

        # 1. DYNAMIC WORKSPACE
        try:
            self.query_one("#sidebar-workspace", Static).update(
                f"[bold]Workspace[/bold]\n[dim]{cwd}[/dim]"
            )
        except Exception:
            pass

        # 2. DYNAMIC MODIFIED FILES (Git or File Watcher)
        try:
            if is_git_repo(cwd):
                stats = get_file_stats(cwd)
                if stats:
                    lines = [
                        f"• {f['path']} (+{f['added']}/-{f['removed']})"
                        for f in stats[:3]
                    ]
                    mod_text = "\n".join(lines)
                    if len(stats) > 3:
                        mod_text += f"\n[dim]+{len(stats) - 3} more...[/dim]"
                else:
                    mod_text = "[dim]Working tree clean[/dim]"
            else:
                # Non-git repo: scan for changed files via line diff
                import glob

                py_files = glob.glob("**/*.py", recursive=True)[:5]
                if py_files:
                    changed = []
                    for fp in py_files:
                        added, removed = get_line_diff(fp)
                        if added or removed:
                            changed.append(f"• {fp} (+{added}/-{removed})")
                    if changed:
                        mod_text = "\n".join(changed[:3])
                    else:
                        mod_text = "[dim]No changes detected[/dim]"
                else:
                    mod_text = "[dim]No Python files[/dim]"

            self.query_one("#sidebar-modified", Static).update(
                f"[bold]Modified Files[/bold]\n{mod_text}"
            )
        except Exception:
            try:
                self.query_one("#sidebar-modified", Static).update(
                    "[bold]Modified Files[/bold]\n[dim]Unavailable[/dim]"
                )
            except Exception:
                pass

        # 3. SQLITE MEMORY (Tokens & Sessions & Agent)
        db_path = os.path.expanduser("~/.axon/memory.sqlite")
        if not os.path.exists(db_path):
            return

        try:
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row

                # Ensure tables exist
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS action_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp TEXT
                    )
                """)
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT,
                        files TEXT,
                        type TEXT,
                        timestamp TEXT
                    )
                """)

                async with db.execute(
                    "SELECT COUNT(*) as count FROM sessions"
                ) as cursor:
                    row = await cursor.fetchone()
                    session_count = row["count"] if row else 0

                tokens = 0
                session_id = getattr(self, "current_session_id", None)
                if session_id:
                    async with db.execute(
                        "SELECT IFNULL(SUM(LENGTH(content)), 0) as total FROM action_logs WHERE session_id = ?",
                        (session_id,),
                    ) as cursor:
                        row = await cursor.fetchone()
                        tokens = (row["total"] // 4) if row else 0

                # Get context window dynamically
                context_window = get_context_window(
                    getattr(self, "model", "gemini/gemini-2.5-flash")
                )

                total_tokens = self.total_prompt_tokens + self.total_completion_tokens
                pct = (total_tokens / context_window * 100) if context_window > 0 else 0

                try:
                    self.query_one("#sidebar-context", Static).update(
                        f"[bold]Context[/bold]\n{total_tokens:,} tokens\n{pct:.1f}% used\n${self.total_cost:.4f} spent\n{session_count} sessions\n"
                    )
                except Exception:
                    pass

                # Idle Agent - use brain status
                brain_status = get_brain_status()
                try:
                    self.query_one("#sidebar-agent", Static).update(
                        f"[bold]Idle Agent[/bold]\n{brain_status}\n"
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def action_cycle_mode(self) -> None:
        modes = ["chat", "plan", "build"]
        idx = modes.index(self.current_mode)
        self.current_mode = modes[(idx + 1) % len(modes)]

    def action_cycle_variant(self) -> None:
        variants = ["normal", "high", "max"]
        idx = variants.index(self.token_variant)
        self.token_variant = variants[(idx + 1) % len(variants)]

    def action_show_session_list(self) -> None:
        """Show session list overlay."""
        from datetime import datetime
        import aiosqlite
        import os

        db_path = os.path.expanduser("~/.axon/memory.sqlite")
        if not os.path.exists(db_path):
            self.notify("No sessions found", title="Sessions")
            return

        async def load_sessions():
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row

                # Ensure action_logs table exists
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS action_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp TEXT
                    )
                """)

                async with db.execute(
                    "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC"
                ) as cursor:
                    rows = await cursor.fetchall()
                    sessions = []
                    for r in rows:
                        async with db.execute(
                            "SELECT COUNT(*) as cnt FROM action_logs WHERE session_id = ?",
                            (r["id"],),
                        ) as cnt_cursor:
                            cnt_row = await cnt_cursor.fetchone()
                            msg_count = cnt_row["cnt"] if cnt_row else 0
                        sessions.append(
                            {
                                "id": r["id"],
                                "title": r["title"],
                                "created_at": r["created_at"],
                                "msg_count": msg_count,
                            }
                        )
                    return sessions

        async def show_overlay():
            sessions = await load_sessions()
            if not sessions:
                self.notify("No sessions found", title="Sessions")
                return

            session_text = "\n".join(
                f"{i + 1}. {s['title']} | {s['created_at'][:10]} | {s['msg_count']} msgs"
                for i, s in enumerate(sessions)
            )
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write(
                f"\n[bold cyan]Sessions:[/bold cyan]\n{session_text}\n[dim]Type session number to load, or /delete to remove[/dim]"
            )

        self.run_worker(show_overlay())

    def action_new_session(self) -> None:
        """Create a new session."""
        import uuid
        from datetime import datetime
        import aiosqlite
        import os

        db_path = os.path.expanduser("~/.axon/memory.sqlite")
        new_id = uuid.uuid4().hex
        now = datetime.now()
        title = f"Session {now.strftime('%Y-%m-%d %H:%M')}"

        async def create_session():
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                await db.execute(
                    "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT)"
                )
                await db.execute(
                    "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (new_id, title, now.isoformat(), now.isoformat()),
                )
                await db.commit()

        async def finalize():
            await create_session()
            self.current_session_id = new_id
            self.session_title = title
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_cost = 0.0
            self.update_header()
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write(
                "\n[bold green]New session started: {}[/bold green]".format(title)
            )
            self.notify(f"New session: {title}", title="Session Created")

        self.run_worker(finalize())

    def action_rename_session(self) -> None:
        """Rename current session."""
        if not self.current_session_id:
            self.notify("No active session", title="Rename")
            return

        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write("\n[bold cyan]Enter new title for session:[/bold cyan]")

        async def get_current_title():
            import aiosqlite
            import os

            db_path = os.path.expanduser("~/.axon/memory.sqlite")
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT title FROM sessions WHERE id = ?",
                    (self.current_session_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                    return row["title"] if row else "Untitled"

        async def rename(new_title: str):
            import aiosqlite
            import os

            db_path = os.path.expanduser("~/.axon/memory.sqlite")
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                await db.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    (new_title, self.current_session_id),
                )
                await db.commit()
            self.notify(f"Renamed to: {new_title}", title="Session Renamed")

        async def prompt_and_rename():
            current_title = await get_current_title()
            chat_log.write(
                f"[dim]Current: {current_title} - Type new name and press Enter[/dim]"
            )

        self.run_worker(prompt_and_rename())

    def action_export_session(self) -> None:
        """Export current session to markdown file."""
        if not self.current_session_id:
            self.notify("No active session to export", title="Export")
            return

        import aiosqlite, os
        from datetime import datetime

        db_path = os.path.expanduser("~/.axon/memory.sqlite")
        export_dir = os.path.expanduser("~/.axon/exports")
        os.makedirs(export_dir, exist_ok=True)

        async def export():
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row

                # Ensure action_logs table exists
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS action_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp TEXT
                    )
                """)

                async with db.execute(
                    "SELECT title, created_at FROM sessions WHERE id = ?",
                    (self.current_session_id,),
                ) as cursor:
                    session = await cursor.fetchone()
                    if not session:
                        return

                async with db.execute(
                    "SELECT role, content FROM action_logs WHERE session_id = ? ORDER BY timestamp",
                    (self.current_session_id,),
                ) as cursor:
                    logs = await cursor.fetchall()

                date_str = session["created_at"][:10]
                filename = f"session_{self.current_session_id[:8]}_{date_str}.md"
                filepath = os.path.join(export_dir, filename)

                model_name = getattr(self, "model", "unknown")

                md_content = f"""# Session: {session["title"]}
Date: {session["created_at"]}
Model: {model_name}

## Conversation
"""
                for log in logs:
                    role = "You" if log["role"] == "user" else "Axon"
                    md_content += f"**{role}:** {log['content']}\n\n"

                with open(filepath, "w") as f:
                    f.write(md_content)

                self.notify(f"Exported to {filepath}", title="Export Complete")

        self.run_worker(export())

    def switch_active_model(self, new_model: str) -> None:
        """Callback triggered by the Command Palette to change the active model."""
        self.model = new_model

        parts = new_model.split("/")
        ui_model = parts[-1] if len(parts) > 1 else new_model
        provider = parts[0].replace("_nim", "") if len(parts) > 1 else "local"

        # Save to config
        try:
            from axon.config.loader import get_config

            config = get_config()
            config.default_model = new_model
            config.save()
        except Exception:
            pass

        self.notify(
            f"✓ Switched to {ui_model}",
            title=f"Provider: {provider.capitalize()}",
            severity="information",
        )
        self.update_status_line()
        self.update_header()

    def action_switch_model(self, new_model_id: str, provider_name: str) -> None:
        """Hot-swaps the active model and updates the UI."""
        from textual.widgets import Static

        self.model = new_model_id

        # Update the status line immediately!
        try:
            self.update_status_line()
        except Exception:
            pass

        # Force a sidebar fetch so the context math updates for the new model
        self.run_worker(self.fetch_sidebar_data())
        self.notify(f"Brain shifted to {new_model_id}", title="Model Switched")

    def action_prompt_custom_model(self) -> None:
        """Prompt user to enter a custom model string."""
        self._awaiting_custom_model = True
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(
            "\n[bold cyan]Enter custom model (e.g., provider/model):[/bold cyan]"
        )
        chat_log.write("[dim]Example: openai/gpt-4o or anthropic/claude-3-opus[/dim]")

    def watch_current_mode(self, old: str, new: str) -> None:
        self.update_status_line()

    def watch_token_variant(self, old: str, new: str) -> None:
        self.update_status_line()

    def watch_session_title(self, old: str, new: str) -> None:
        self.update_header()

    def watch_brain_status(self, old: str, new: str) -> None:
        self.update_header()

    def update_header(self) -> None:
        """Update the header bar with session title, model, and brain status."""
        try:
            raw_model = getattr(self, "model", "gemini/gemini-2.5-flash")
            parts = raw_model.split("/")
            ui_model = parts[-1] if len(parts) > 1 else raw_model

            header_text = f"[bold]Axon[/bold] [cyan]│[/cyan] {self.session_title} [dim]│[/dim] {ui_model} [dim]│[/dim] {self.brain_status}"
            self.query_one("#header-bar", Static).update(header_text)
        except Exception:
            pass

    def update_status_line(self) -> None:
        mode_colors = {"chat": "cyan", "plan": "magenta", "build": "blue"}
        variant_colors = {"normal": "green", "high": "orange3", "max": "red"}

        m_color = mode_colors.get(self.current_mode, "white")
        v_color = variant_colors.get(self.token_variant, "white")

        raw_model = getattr(self, "model", "gemini/gemini-2.5-flash")
        parts = raw_model.split("/")
        ui_model_name = parts[-1] if len(parts) > 1 else raw_model
        provider_name = parts[0].replace("_nim", "") if len(parts) > 1 else "local"

        status_text = f"[bold {m_color}]{self.current_mode.capitalize()}[/]  {ui_model_name} [dim]via {provider_name}[/dim] · [bold {v_color}]{self.token_variant}[/]"

        try:
            self.query_one("#status-line", Static).update(status_text)
        except Exception:
            pass

    def action_command_palette(self) -> None:
        """Handle ctrl+p shortcut safely without recursion."""
        try:
            super().action_command_palette()
        except AttributeError:
            self.notify("Command palette coming soon!", title="Axon")

    def action_open_keys(self) -> None:
        """Opens the API Key configuration modal."""
        self.push_screen(APIKeyModal())

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return
        event.input.value = ""
        chat_log = self.query_one("#chat-log", RichLog)

        # Handle slash commands
        if user_text.startswith("/"):
            parts = user_text.strip().split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if cmd == "/rename":
                await self.handle_rename(args, chat_log)
            elif cmd == "/sessions":
                await self.handle_list_sessions(chat_log)
            elif cmd == "/switch":
                await self.handle_switch_session(args, chat_log)
            elif cmd == "/read":
                await self.handle_read_file(args, chat_log)
            elif cmd == "/clear":
                chat_log.clear()
                chat_log.write("[dim]Chat cleared[/dim]")
            elif cmd == "/model":
                await self.handle_switch_model(args, chat_log)
            elif cmd == "/help":
                await self.handle_help(chat_log)
            elif cmd == "/export":
                self.action_export_session()
            elif cmd == "/brain":
                await self.handle_brain_status(chat_log)
            elif cmd == "/files":
                await self.handle_list_files(chat_log)
            else:
                chat_log.write(
                    f"[red]Unknown command: {cmd}. Type /help for available commands[/red]"
                )
            return

        # Handle rename confirmation
        if hasattr(self, "_awaiting_rename") and self._awaiting_rename:
            self._awaiting_rename = False
            new_title = user_text.strip()
            if new_title:
                await self.do_rename(new_title, chat_log)
            return

        # Handle custom model input
        if hasattr(self, "_awaiting_custom_model") and self._awaiting_custom_model:
            self._awaiting_custom_model = False
            custom_model = user_text.strip()
            if custom_model and validate_model(custom_model):
                self.switch_active_model(custom_model)
                chat_log.write(f"[green]Custom model set: {custom_model}[/green]")
            elif custom_model:
                chat_log.write(f"[red]Invalid model: {custom_model}[/red]")
            return

            # AUTO-CREATE SESSION IF IT DOESN'T EXIST
            try:
                idx = int(user_text.split()[1]) - 1
                import aiosqlite, os

                db_path = os.path.expanduser("~/.axon/memory.sqlite")
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    async with db.execute(
                        "SELECT id, title FROM sessions ORDER BY created_at DESC LIMIT 1 OFFSET ?",
                        (idx,),
                    ) as cursor:
                        row = await cursor.fetchone()
                        if row:
                            self.current_session_id = row["id"]
                            self.total_prompt_tokens = 0
                            self.total_completion_tokens = 0
                            self.total_cost = 0.0
                            chat_log.write(
                                f"\n[bold green]Loaded session: {row['title']}[/bold green]"
                            )
                        else:
                            chat_log.write("[yellow]Session not found[/yellow]")
            except (IndexError, ValueError):
                chat_log.write("[yellow]Usage: /load <number>[/yellow]")
            return

        if user_text.lower() == "/delete":
            if not self.current_session_id:
                chat_log.write("[yellow]No active session to delete[/yellow]")
                return
            import aiosqlite, os

            db_path = os.path.expanduser("~/.axon/memory.sqlite")
            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    # Ensure action_logs table exists
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS action_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT,
                            role TEXT,
                            content TEXT,
                            timestamp TEXT
                        )
                    """)
                    await db.execute(
                        "DELETE FROM action_logs WHERE session_id = ?",
                        (self.current_session_id,),
                    )
                    await db.execute(
                        "DELETE FROM sessions WHERE id = ?", (self.current_session_id,)
                    )
                    await db.commit()
                self.current_session_id = None
                self.total_prompt_tokens = 0
                self.total_completion_tokens = 0
                self.total_cost = 0.0
                chat_log.write("[bold red]Session deleted.[/bold red]")
            except Exception as e:
                chat_log.write(f"[red]Delete failed: {e}[/red]")
            return

        # Handle custom model input (after prompting with action_prompt_custom_model)
        if hasattr(self, "_awaiting_custom_model") and self._awaiting_custom_model:
            self._awaiting_custom_model = False
            custom_model = user_text.strip()
            if custom_model and validate_model(custom_model):
                self.switch_active_model(custom_model)
                chat_log.write(f"[green]Custom model set: {custom_model}[/green]")
            elif custom_model:
                chat_log.write(
                    f"[red]Invalid model: {custom_model}. Cannot validate with LiteLLM.[/red]"
                )
            return

        # AUTO-CREATE SESSION IF IT DOESN'T EXIST
        if not getattr(self, "current_session_id", None):
            import uuid, os, aiosqlite
            from datetime import datetime

            db_path = os.path.expanduser("~/.axon/memory.sqlite")
            self.current_session_id = uuid.uuid4().hex
            now = datetime.now().isoformat()

            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS sessions (
                            id TEXT PRIMARY KEY,
                            title TEXT,
                            created_at TEXT,
                            updated_at TEXT
                        )
                    """)

                    await db.execute(
                        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        (self.current_session_id, user_text[:30] + "...", now, now),
                    )
                    await db.commit()
                    self.session_title = user_text[:30] + "..."
                    self.update_header()
            except Exception as e:
                chat_log.write(
                    f"[red]Memory Error: Could not create session: {e}[/red]"
                )

        self.stream_llm_response(user_text)

    # ===== Slash Command Handlers =====

    async def handle_rename(self, args: str, chat_log: RichLog) -> None:
        """Handle /rename command."""
        if not self.current_session_id:
            chat_log.write("[yellow]No active session to rename[/yellow]")
            return
        if not args.strip():
            current = self.session_title
            chat_log.write(
                f"[dim]Current: {current} - Type /rename <new name> to rename[/dim]"
            )
            self._awaiting_rename = True
            return
        await self.do_rename(args.strip(), chat_log)

    async def do_rename(self, new_title: str, chat_log: RichLog) -> None:
        """Actually rename the session in DB."""
        import aiosqlite, os
        from datetime import datetime

        db_path = os.path.expanduser("~/.axon/memory.sqlite")
        try:
            async with aiosqlite.connect(db_path) as db:
                await db.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (new_title, datetime.now().isoformat(), self.current_session_id),
                )
                await db.commit()
            self.session_title = new_title
            self.update_header()
            chat_log.write(f"[green]✓ Renamed to: {new_title}[/green]")
            self.notify(f"✓ Renamed to: {new_title}", title="Session Renamed")
        except Exception as e:
            chat_log.write(f"[red]Rename failed: {e}[/red]")

    async def handle_list_sessions(self, chat_log: RichLog) -> None:
        """Handle /sessions command."""
        import aiosqlite, os

        db_path = os.path.expanduser("~/.axon/memory.sqlite")
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC LIMIT 20"
            ) as cursor:
                rows = await cursor.fetchall()
                if not rows:
                    chat_log.write("[dim]No sessions found[/dim]")
                    return
                chat_log.write(
                    f"[bold magenta]Sessions ({len(rows)} total):[/bold magenta]"
                )
                for i, r in enumerate(rows):
                    date = r["created_at"][:10] if r["created_at"] else "?"
                    chat_log.write(f"{i + 1}. [{date}] {r['title']}")

    async def handle_switch_session(self, args: str, chat_log: RichLog) -> None:
        """Handle /switch command."""
        if not args.strip():
            chat_log.write("[yellow]Usage: /switch <number>[/yellow]")
            return
        try:
            idx = int(args.strip()) - 1
            import aiosqlite, os

            db_path = os.path.expanduser("~/.axon/memory.sqlite")
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT id, title FROM sessions ORDER BY created_at DESC LIMIT 1 OFFSET ?",
                    (idx,),
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        self.current_session_id = row["id"]
                        self.session_title = row["title"]
                        self.total_prompt_tokens = 0
                        self.total_completion_tokens = 0
                        self.total_cost = 0.0
                        self.update_header()
                        chat_log.clear()
                        chat_log.write(
                            f"[green]✓ Switched to session: {row['title']}[/green]"
                        )
                    else:
                        chat_log.write("[yellow]Session not found[/yellow]")
        except ValueError:
            chat_log.write("[yellow]Usage: /switch <number>[/yellow]")

    async def handle_read_file(self, args: str, chat_log: RichLog) -> None:
        """Handle /read command."""
        if not args.strip():
            chat_log.write("[yellow]Usage: /read <filepath>[/yellow]")
            return
        try:
            import os

            filepath = os.path.join(os.getcwd(), args.strip())
            if os.path.isfile(filepath):
                with open(filepath, "r") as f:
                    content = f.read()
                chat_log.write(
                    f"[bold cyan]File: {args.strip()}[/bold cyan]\n{content[:2000]}"
                )
            else:
                chat_log.write(f"[red]File not found: {args.strip()}[/red]")
        except Exception as e:
            chat_log.write(f"[red]Error reading file: {e}[/red]")

    async def handle_switch_model(self, args: str, chat_log: RichLog) -> None:
        """Handle /model command."""
        if not args.strip():
            chat_log.write(f"[dim]Current model: {self.model}[/dim]")
            return
        if validate_model(args.strip()):
            self.switch_active_model(args.strip())
            chat_log.write(f"[green]✓ Model: {args.strip()}[/green]")
        else:
            chat_log.write(f"[red]Invalid model: {args.strip()}[/red]")

    async def handle_help(self, chat_log: RichLog) -> None:
        """Handle /help command."""
        help_text = """[bold cyan]Available Commands:[/bold cyan]

[cyan]/rename[/cyan] <new name> - Rename current session
[cyan]/sessions[/cyan]              - List all sessions
[cyan]/switch[/cyan] <number>     - Switch to another session
[cyan]/read[/cyan] <filepath>      - Read a file
[cyan]/clear[/cyan]                - Clear chat display
[cyan]/model[/cyan] <model>        - Switch model
[cyan]/export[/cyan]               - Export session to markdown
[cyan]/brain[/cyan]                - Show brain status
[cyan]/files[/cyan]                - List modified files
[cyan]/help[/cyan]                - Show this help"""
        chat_log.write(help_text)

    async def handle_brain_status(self, chat_log: RichLog) -> None:
        """Handle /brain command."""
        changes = get_recent_file_changes()
        if changes:
            chat_log.write(f"[bold]🧠 Active ({len(changes)} files changed):[/bold]")
            for c in changes[:5]:
                chat_log.write(f"  • {c.get('path', 'unknown')}")
        else:
            chat_log.write("[dim]🧠 Observing workspace...[/dim]")

    async def handle_list_files(self, chat_log: RichLog) -> None:
        """Handle /files command."""
        changes = get_recent_file_changes()
        if not changes:
            chat_log.write("[dim]No modified files[/dim]")
            return
        chat_log.write(f"[bold]Modified Files ({len(changes)} total):[/bold]")
        for c in changes[:8]:
            path = c.get("path", "unknown")
            added = c.get("added", 0)
            removed = c.get("removed", 0)
            chat_log.write(f"  • {path} (+{added}/-{removed})")

    @work(thread=True)
    async def stream_llm_response(self, prompt: str) -> None:
        import litellm

        chat_log = self.query_one("#chat-log", RichLog)
        live_stream = self.query_one("#live-stream", Static)

        # 1. Print the user's message
        self.call_from_thread(
            chat_log.write,
            f"\n[bold cyan]You ({self.current_mode}):[/bold cyan] {prompt}",
        )
        self.call_from_thread(live_stream.update, "...")

        # 2. The Fallback Array (Primary -> Fast Free -> Cheap Reliable)
        from axon.config.loader import get_config

        config = get_config()
        primary_model = getattr(self, "model", config.default_model)
        models_to_try = build_fallback_chain(primary_model, config)

        # Get system prompt based on mode
        if self.current_mode == "build":
            sys_prompt = "You are a senior developer. Output ONLY valid code inside a single markdown code block. Do not explain the code."
        elif self.current_mode == "plan":
            sys_prompt = "You are an architect. Output ONLY concise bullet points."
        else:
            sys_prompt = "You are a concise AI. Answer briefly."

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        # 3. The Waterfall Routing Loop
        for current_model in models_to_try:
            try:
                if current_model != models_to_try[0]:
                    self.call_from_thread(
                        chat_log.write,
                        f"[yellow]⚠️ Shifting brain to {current_model}...[/yellow]",
                    )

                variant_caps = {"normal": 300, "high": 1500, "max": 6000}
                token_cap = variant_caps.get(self.token_variant, 300)

                response = await litellm.acompletion(
                    model=current_model,
                    messages=messages,
                    stream=True,
                    max_tokens=token_cap,
                )

                full_response = ""
                async for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    self.call_from_thread(
                        live_stream.update,
                        f"[bold green]Axon ({self.token_variant}):[/bold green]\n{full_response}",
                    )

                # Success! Write to persistent log and clear stream widget
                self.call_from_thread(live_stream.update, "")

                # Format the header based on the model that actually succeeded
                provider_color = (
                    "green" if current_model == models_to_try[0] else "yellow"
                )
                self.call_from_thread(
                    chat_log.write,
                    f"[bold {provider_color}]Axon ({current_model}):[/bold {provider_color}]\n{full_response}",
                )

                # Handle build mode - write code to file
                if self.current_mode == "build" and full_response:
                    import re

                    code_blocks = re.findall(
                        r"```.*?\n(.*?)```", full_response, re.DOTALL
                    )
                    if code_blocks:
                        filename = "axon_build_output.py"
                        if "name" in prompt.lower():
                            words = prompt.split()
                            for i, w in enumerate(words):
                                if "name" in w and i + 1 < len(words):
                                    filename = words[i + 1].strip("'\".,")
                        import os

                        filepath = os.path.join(os.getcwd(), filename)
                        with open(filepath, "w") as f:
                            f.write(code_blocks[0].strip())
                        self.call_from_thread(
                            chat_log.write,
                            f"[bold blue]System: File successfully written to {filepath}[/bold blue]",
                        )

                # Extract token usage and update totals
                if hasattr(response, "usage") and response.usage:
                    prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                    completion_tokens = (
                        getattr(response.usage, "completion_tokens", 0) or 0
                    )
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    cost = get_cost(current_model, prompt_tokens, completion_tokens)
                    self.total_cost += cost

                # Save to SQLite
                session_id = getattr(self, "current_session_id", None)
                if session_id:
                    import aiosqlite
                    from datetime import datetime

                    db_path = os.path.expanduser("~/.axon/memory.sqlite")

                    async def save_logs():
                        async with aiosqlite.connect(db_path) as db:
                            now = datetime.now().isoformat()
                            await db.execute(
                                "INSERT INTO action_logs (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                                (str(session_id), "user", prompt, now),
                            )
                            await db.execute(
                                "INSERT INTO action_logs (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                                (str(session_id), "assistant", full_response, now),
                            )
                            await db.commit()

                    try:
                        self.run_worker(save_logs(), exclusive=False)
                    except Exception as e:
                        self.log(f"Failed to save logs: {e}")

                return  # Break out of the loop, we successfully generated text!

            except litellm.exceptions.RateLimitError:
                self.call_from_thread(
                    chat_log.write,
                    f"[dim red]Rate limit hit on {current_model}. Retrying...[/dim red]",
                )
                continue  # Loop to the next model

            except Exception as e:
                error_str = str(e)
                if "rate" in error_str.lower() or "429" in error_str:
                    self.call_from_thread(
                        chat_log.write,
                        f"[dim red]Rate limit on {current_model}. Trying next...[/dim red]",
                    )
                    continue
                self.call_from_thread(
                    chat_log.write,
                    f"[dim red]API Error ({current_model}): {error_str[:50]}...[/dim red]",
                )
                continue  # Loop to the next model

        # 4. Total Failure State
        self.call_from_thread(live_stream.update, "")
        self.call_from_thread(
            chat_log.write,
            "[bold red]❌ All fallback models failed. Please check your .env API keys or internet connection.[/bold red]",
        )


async def run_chat_loop(model: str | None = None, session_id: str | None = None):
    app = AxonApp()
    if model:
        app.model = model
    else:
        from axon.config.loader import get_config

        app.model = get_config().default_model
    app.current_session_id = session_id
    await app.run_async()
