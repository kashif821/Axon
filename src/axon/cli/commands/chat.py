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
        load_dotenv()
        matcher = self.matcher(query)

        # Build model list with beautiful display names
        models = []

        # Gemini models
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            models.extend(
                [
                    ("gemini/gemini-2.5-flash", "Google Gemini 2.5 Flash"),
                    ("gemini/gemini-2.5-pro", "Google Gemini 2.5 Pro"),
                    ("gemini/gemini-1.5-pro", "Google Gemini 1.5 Pro"),
                ]
            )

        # Groq models
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            models.extend(
                [
                    ("groq/llama-3.1-70b-versatile", "Groq Llama 3.1 70B"),
                    ("groq/llama3-70b-8192", "Groq Llama 3 70B"),
                    ("groq/mixtral-8x7b-32768", "Groq Mixtral 8x7B"),
                ]
            )

        # Nvidia NIM
        nvidia_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")
        if nvidia_key:
            models.extend(
                [
                    (
                        "nvidia_nim/meta/llama-3.1-70b-instruct",
                        "Nvidia NIM Llama 3.1 70B",
                    ),
                    ("nvidia_nim/nv-llama-3.1-70b-instruct", "Nvidia Llama 3.1 70B"),
                ]
            )

        # Moonshot / Kimi
        moonshot_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
        if moonshot_key:
            models.extend(
                [
                    ("moonshot/moonshot-v1-8k", "Moonshot Kimi K1.5"),
                    ("moonshot/moonshot-v1-128k", "Moonshot Kimi 128K"),
                ]
            )

        # OpenAI (always available if key set)
        if os.getenv("OPENAI_API_KEY"):
            models.extend(
                [
                    ("openai/gpt-4o", "OpenAI GPT-4o"),
                    ("openai/gpt-4o-mini", "OpenAI GPT-4o Mini"),
                ]
            )

        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            models.extend(
                [
                    ("anthropic/claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
                    ("anthropic/claude-3-opus-20240229", "Claude 3 Opus"),
                ]
            )

        if not models:
            models = [("gemini/gemini-2.5-flash", "Google Gemini 2.5 Flash (Demo)")]

        for model_id, display_name in models:
            score = matcher.match(display_name)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(display_name),
                    lambda m=model_id: self.app.switch_active_model(m),
                    help=f"Provider: {model_id.split('/')[0].replace('_nim', ' NIM')}",
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
    model = None
    current_session = None

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
        self.query_one("#chat-log", RichLog).write(
            "System: UI Initialized. Connecting to memory..."
        )

        # Trigger the database fetch
        self.fetch_sidebar_data()

        # Real-time sidebar polling every 5 seconds
        self.set_interval(5, self.fetch_sidebar_data)

    @work
    async def fetch_sidebar_data(self) -> None:
        import os, aiosqlite

        db_path = os.path.expanduser("~/.axon/memory.sqlite")
        cwd = os.getcwd()

        try:
            self.call_from_thread(
                self.query_one("#sidebar-workspace", Static).update,
                f"[bold]Workspace[/bold]\n[dim]{cwd}[/dim]",
            )

            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row

                # Count Total Sessions
                async with db.execute(
                    "SELECT COUNT(*) as count FROM sessions"
                ) as cursor:
                    row = await cursor.fetchone()
                    session_count = row["count"] if row else 0

                # Current Session Tokens
                tokens = 0
                if getattr(self, "current_session", None):
                    async with db.execute(
                        "SELECT SUM(LENGTH(content)) as total FROM action_logs WHERE session_id = ?",
                        (self.current_session,),
                    ) as cursor:
                        row = await cursor.fetchone()
                        tokens = (row["total"] // 4) if row and row["total"] else 0

                self.call_from_thread(
                    self.query_one("#sidebar-context", Static).update,
                    f"[bold]Context[/bold]\n{tokens:,} tokens\n{session_count} sessions\n",
                )

                # Idle Agent
                async with db.execute(
                    "SELECT content FROM summaries ORDER BY timestamp DESC LIMIT 1"
                ) as cursor:
                    row = await cursor.fetchone()
                    agent_text = (
                        row["content"][:50] + "..." if row else "Observing workspace..."
                    )
                    self.call_from_thread(
                        self.query_one("#sidebar-agent", Static).update,
                        f"[bold]Idle Agent[/bold]\n{agent_text}\n",
                    )
        except Exception:
            pass

        except Exception as e:
            # Silently fail if DB is locked or missing
            try:
                self.query_one("#sidebar-context", Static).update(
                    f"[bold]Context[/bold]\nMemory unavailable"
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
        """Fired when the user presses Enter."""
        user_text = event.value.strip()
        if not user_text:
            return
        event.input.value = ""
        chat_log = self.query_one("#chat-log", RichLog)

        # Handle local slash commands
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
                "\n[bold yellow]System: Session loading coming soon via ID lookup.[/bold yellow]"
            )
            return

        if user_text.lower() == "/delete":
            db_path = os.path.expanduser("~/.axon/memory.sqlite")
            try:
                async with aiosqlite.connect(db_path) as db:
                    await db.execute(
                        "DELETE FROM action_logs WHERE session_id = ?",
                        (self.current_session,),
                    )
                    await db.commit()
                chat_log.write("[bold red]System: Session deleted.[/bold red]")
            except Exception:
                chat_log.write("[bold red]System: No session to delete.[/bold red]")
            return

        # Fire the background worker
        self.stream_llm_response(user_text)

    @work(thread=True)
    async def stream_llm_response(self, prompt: str) -> None:
        import re, os

        chat_log = self.query_one("#chat-log", RichLog)
        live_stream = self.query_one("#live-stream", Static)
        self.call_from_thread(
            chat_log.write,
            f"\n[bold cyan]You ({self.current_mode}):[/bold cyan] {prompt}",
        )

        # 1. STRICT VARIANT LIMITS (Controls API Usage)
        variant_caps = {"normal": 300, "high": 1500, "max": 6000}
        token_cap = variant_caps.get(self.token_variant, 300)

        # 2. MODE PROMPTS (Controls Output Style)
        if self.current_mode == "build":
            sys_prompt = "You are a senior developer. Output ONLY valid code inside a single markdown code block. Do not explain the code."
        elif self.current_mode == "plan":
            sys_prompt = "You are an architect. Output ONLY concise bullet points."
        else:
            sys_prompt = "You are a concise AI. Answer briefly."

        try:
            provider = get_llm_provider(model=self.model)
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt),
                ChatMessage(role=MessageRole.USER, content=prompt),
            ]
            buffer = ""

            async for chunk in provider.stream(
                messages, model=self.model, max_tokens=token_cap
            ):
                if delta := chunk.choices[0].delta.content:
                    buffer += delta
                    self.call_from_thread(
                        live_stream.update,
                        f"[bold green]Axon ({self.token_variant} tokens):[/bold green]\n{buffer}",
                    )

            if buffer:
                self.call_from_thread(
                    chat_log.write, f"[bold green]Axon:[/bold green]\n{buffer}"
                )
                self.call_from_thread(live_stream.update, "")

                # 3. THE BUILDER: Physically create the file
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

            self.call_from_thread(self.fetch_sidebar_data)
        except Exception as e:
            self.call_from_thread(live_stream.update, "")
            error_str = str(e)

            # Catch ugly litellm rate limit errors
            if (
                "429" in error_str
                or "RateLimitError" in error_str
                or "RESOURCE_EXHAUSTED" in error_str
            ):
                clean_error = "\n[bold orange3]System: API Rate Limit Reached. Please wait ~60 seconds before sending another prompt.[/bold orange3]"
                self.call_from_thread(chat_log.write, clean_error)
            else:
                # Fallback for other errors (shortened to keep UI clean)
                self.call_from_thread(
                    chat_log.write,
                    f"\n[bold red]API Error:[/bold red] {error_str[:200]}...",
                )

            # Log to SQLite
            if hasattr(self, "current_session") and self.current_session:
                asyncio.run_coroutine_threadsafe(
                    log_action(self.current_session, "user", prompt), self.app._loop
                )
                asyncio.run_coroutine_threadsafe(
                    log_action(
                        self.current_session, "assistant", "Response generated."
                    ),
                    self.app._loop,
                )

            self.call_from_thread(self.fetch_sidebar_data)

        except Exception as e:
            self.call_from_thread(live_stream.update, "")
            self.call_from_thread(
                chat_log.write, f"\n[bold red]API Error:[/bold red] {str(e)}"
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
