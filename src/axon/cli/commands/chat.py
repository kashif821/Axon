from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv

from textual.app import App, ComposeResult
from textual.command import Provider, Hit, Hits
from textual import work
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, RichLog, Static, Footer
from textual.reactive import reactive
from textual.binding import Binding

from axon.llm.base import ChatMessage, MessageRole
from axon.llm.providers import get_llm_provider
from axon.memory.store import log_action

try:
    import aiosqlite
except ImportError:
    aiosqlite = None


class ModelCommandProvider(Provider):
    """Provides dynamic commands to switch LLM models based on available .env keys."""

    async def search(self, query: str) -> Hits:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        import os

        matcher = self.matcher(query)
        available_models = {}

        # Dynamically scan the environment for ANY variable ending in _MODELS
        for env_key, env_value in os.environ.items():
            if env_key.endswith("_MODELS") and env_value:
                # Split the comma-separated list from the .env file
                model_strings = [m.strip() for m in env_value.split(",") if m.strip()]

                for raw_id in model_strings:
                    # Auto-generate a pretty name (e.g., "gemini-2.5-flash (via gemini)")
                    parts = raw_id.split("/")
                    pretty_name = parts[-1] if len(parts) > 1 else raw_id
                    provider = (
                        parts[0].replace("_nim", "") if len(parts) > 1 else "local"
                    )

                    available_models[raw_id] = f"{pretty_name} (via {provider})"

        if not available_models:
            available_models["gemini/gemini-2.5-flash"] = "gemini-2.5-flash (Fallback)"

        for raw_id, pretty_name in available_models.items():
            display = f"Set Model: {pretty_name}"

            if not query:
                yield Hit(
                    1.0,
                    display,
                    lambda m=raw_id: self.app.switch_active_model(m),
                    help=f"Provider: {raw_id.split('/')[0]}",
                )
            else:
                score = matcher.match(display)
                if score > 0:
                    yield Hit(
                        score,
                        matcher.highlight(display),
                        lambda m=raw_id: self.app.switch_active_model(m),
                        help=f"Provider: {raw_id.split('/')[0]}",
                    )


class AxonApp(App):
    CSS = """
    Screen { background: #0d0d0d; }
    #main-layout { height: 100%; width: 100%; }
    #left-column { width: 1fr; padding: 0 1; }
    #chat-log { height: 1fr; }
    
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
    """

    COMMANDS = App.COMMANDS | {ModelCommandProvider}

    BINDINGS = [
        Binding("tab", "cycle_mode", "Switch Mode", priority=True),
        Binding("ctrl+t", "cycle_variant", "Variants", priority=True),
        Binding("ctrl+p", "command_palette", "Commands", priority=True),
    ]

    current_mode = reactive("chat")
    token_variant = reactive("high")
    current_session_id = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = "gemini/gemini-2.5-flash"
        self.current_session_id = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-layout"):
            with Vertical(id="left-column"):
                yield RichLog(id="chat-log", markup=True, wrap=True)
                yield Static("", id="live-stream", markup=True)
                with Vertical(id="input-box"):
                    yield Input(id="chat-input", placeholder="Ask anything...")
                    yield Static("", id="status-line", markup=True)

            with Vertical(id="sidebar"):
                yield Static(
                    "Axon CLI: Shell Ready\n",
                    markup=True,
                    id="sidebar-header",
                )
                yield Static(
                    "[bold]Context[/bold]\n0 tokens\nInitializing...\n",
                    markup=True,
                    id="sidebar-context",
                )
                yield Static(
                    "[bold]LSP[/bold]\nStarting up...\n",
                    markup=True,
                    id="sidebar-lsp",
                )
                yield Static(
                    "[bold]Modified Files[/bold]\nLoading...\n",
                    markup=True,
                    id="sidebar-files",
                )
                yield Static(
                    "[bold]Idle Agent[/bold]\nObserving...\n",
                    markup=True,
                    id="sidebar-agent",
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
        """Polls SQLite and updates the sidebar UI in real-time."""
        import os, aiosqlite

        cwd = os.getcwd()
        try:
            self.query_one("#sidebar-workspace", Static).update(
                f"[bold]Workspace[/bold]\n[dim]{cwd}[/dim]"
            )
        except Exception:
            pass

        db_path = os.path.expanduser("~/.axon/memory.sqlite")
        if not os.path.exists(db_path):
            return

        try:
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT COUNT(*) as count FROM sessions"
                ) as cursor:
                    row = await cursor.fetchone()
                    session_count = row["count"] if row else 0

                tokens = 0
                session_id = getattr(self, "current_session_id", None)
                if session_id:
                    async with db.execute(
                        "SELECT SUM(LENGTH(content)) as total FROM action_logs WHERE session_id = ?",
                        (session_id,),
                    ) as cursor:
                        row = await cursor.fetchone()
                        tokens = (row["total"] // 4) if row and row["total"] else 0

                self.query_one("#sidebar-context", Static).update(
                    f"[bold]Context[/bold]\n{tokens:,} tokens\n{session_count} sessions\n"
                )

                async with db.execute(
                    "SELECT content FROM summaries ORDER BY timestamp DESC LIMIT 1"
                ) as cursor:
                    row = await cursor.fetchone()
                    agent_text = (
                        row["content"][:50] + "..." if row else "Observing workspace..."
                    )
                    self.query_one("#sidebar-agent", Static).update(
                        f"[bold]Idle Agent[/bold]\n{agent_text}\n"
                    )
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

    def switch_active_model(self, new_model: str) -> None:
        """Callback triggered by the Command Palette to change the active model."""
        self.model = new_model

        parts = new_model.split("/")
        ui_model = parts[-1] if len(parts) > 1 else new_model
        provider = parts[0].replace("_nim", "") if len(parts) > 1 else "local"

        self.notify(
            f"Routed to {ui_model}",
            title=f"Provider: {provider.capitalize()}",
            severity="information",
        )
        self.update_status_line()

    def watch_current_mode(self, old: str, new: str) -> None:
        self.update_status_line()

    def watch_token_variant(self, old: str, new: str) -> None:
        self.update_status_line()

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

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return
        event.input.value = ""
        chat_log = self.query_one("#chat-log", RichLog)

        if user_text.lower() == "/sessions":
            import os, aiosqlite

            db_path = os.path.expanduser("~/.axon/memory.sqlite")
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT id, title FROM sessions ORDER BY updated_at DESC LIMIT 5"
                ) as cursor:
                    rows = await cursor.fetchall()
                    chat_log.write("\n[bold magenta]Recent Sessions:[/bold magenta]")
                    for r in rows:
                        chat_log.write(f" ID: {r['id'][:6]}... | {r['title']}")
            return

        if user_text.lower().startswith("/load "):
            chat_log.write(
                "\n[bold yellow]System: Session loading coming soon.[/bold yellow]"
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
                    await db.execute(
                        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        (self.current_session_id, user_text[:30] + "...", now, now),
                    )
                    await db.commit()
            except Exception as e:
                chat_log.write(
                    f"[red]Memory Error: Could not create session: {e}[/red]"
                )

        self.stream_llm_response(user_text)

    @work(thread=True)
    async def stream_llm_response(self, prompt: str) -> None:
        import re, os, asyncio

        chat_log = self.query_one("#chat-log", RichLog)
        live_stream = self.query_one("#live-stream", Static)
        self.call_from_thread(
            chat_log.write,
            f"\n[bold cyan]You ({self.current_mode}):[/bold cyan] {prompt}",
        )

        variant_caps = {"normal": 300, "high": 1500, "max": 6000}
        token_cap = variant_caps.get(self.token_variant, 300)

        if self.current_mode == "build":
            sys_prompt = "You are a senior developer. Output ONLY valid code inside a single markdown code block. Do not explain the code."
        elif self.current_mode == "plan":
            sys_prompt = "You are an architect. Output ONLY concise bullet points."
        else:
            sys_prompt = "You are a concise AI. Answer briefly."

        try:
            from axon.llm.providers import get_llm_provider
            from axon.llm.base import ChatMessage, MessageRole

            provider = get_llm_provider(
                model=getattr(self, "model", "gemini/gemini-2.5-flash")
            )
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt),
                ChatMessage(role=MessageRole.USER, content=prompt),
            ]
            buffer = ""

            async for chunk in provider.stream(
                messages,
                model=getattr(self, "model", "gemini/gemini-2.5-flash"),
                max_tokens=token_cap,
            ):
                if delta := chunk.choices[0].delta.content:
                    buffer += delta
                    self.call_from_thread(
                        live_stream.update,
                        f"[bold green]Axon ({self.token_variant}):[/bold green]\n{buffer}",
                    )

            if buffer:
                self.call_from_thread(
                    chat_log.write, f"[bold green]Axon:[/bold green]\n{buffer}"
                )
                self.call_from_thread(live_stream.update, "")

                if self.current_mode == "build":
                    code_blocks = re.findall(r"```.*?\n(.*?)```", buffer, re.DOTALL)
                    if code_blocks:
                        filename = "axon_build_output.py"
                        if "name" in prompt.lower():
                            words = prompt.split()
                            for i, w in enumerate(words):
                                if "name" in w and i + 1 < len(words):
                                    filename = words[i + 1].strip("'\".,")
                        filepath = os.path.join(os.getcwd(), filename)
                        with open(filepath, "w") as f:
                            f.write(code_blocks[0].strip())
                        self.call_from_thread(
                            chat_log.write,
                            f"[bold blue]System: File successfully written to {filepath}[/bold blue]",
                        )

                # BULLETPROOF SQLITE LOGGING
                session_id = getattr(self, "current_session_id", None)
                if session_id:
                    import aiosqlite
                    from datetime import datetime

                    db_path = os.path.expanduser("~/.axon/memory.sqlite")

                    try:

                        async def save_logs():
                            async with aiosqlite.connect(db_path) as db:
                                now = datetime.now().isoformat()
                                await db.execute(
                                    "INSERT INTO action_logs (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                                    (session_id, "user", prompt, now),
                                )
                                await db.execute(
                                    "INSERT INTO action_logs (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                                    (session_id, "assistant", buffer, now),
                                )
                                await db.commit()

                        # Schedule the async function on the app's event loop
                        asyncio.get_event_loop().run_until_complete(save_logs())
                    except Exception:
                        pass

        except Exception as e:
            self.call_from_thread(live_stream.update, "")
            error_str = str(e)
            if (
                "429" in error_str
                or "RateLimitError" in error_str
                or "RESOURCE_EXHAUSTED" in error_str
            ):
                clean_error = "\n[bold orange3]System: API Rate Limit Reached. Please wait ~60 seconds before sending another prompt.[/bold orange3]"
                self.call_from_thread(chat_log.write, clean_error)
            else:
                self.call_from_thread(
                    chat_log.write,
                    f"\n[bold red]Error:[/bold red] {error_str[:200]}...",
                )


async def run_chat_loop(model: str | None = None, session_id: str | None = None):
    app = AxonApp()
    if model:
        app.model = model
    else:
        from axon.config.loader import get_config

        app.model = get_config().default_model
    app.current_session = session_id
    await app.run_async()
