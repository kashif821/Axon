import os
import logging
from datetime import datetime
from threading import Thread, Lock
from typing import Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import pathspec

logger = logging.getLogger(__name__)

_observer: Optional[Observer] = None
_event_handler: Optional[FileSystemEventHandler] = None
_changes: list = []
_changes_lock = Lock()
_pathspec_matcher: Optional[pathspec.PathSpec] = None
_last_seen: dict = {}
_DEBOUNCE_SECONDS = 1.0


DEFAULT_IGNORES = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    "dist",
    "build",
}


def _load_gitignore(root_path: str) -> pathspec.PathSpec:
    gitignore_path = os.path.join(root_path, ".gitignore")
    patterns = []

    for name in DEFAULT_IGNORES:
        patterns.append(f"{name}/")
        patterns.append(f"**/{name}/")

    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
                        if not line.endswith("/"):
                            patterns.append(f"{line}/")
        except Exception as e:
            logger.warning(f"Failed to load .gitignore: {e}")

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _normalize_event_type(event: FileSystemEvent) -> str:
    if event.event_type == "modified":
        return "modified"
    elif event.event_type == "created":
        return "created"
    elif event.event_type == "deleted":
        return "deleted"
    elif event.event_type == "moved":
        return "modified"
    return "modified"


class _ChangeHandler(FileSystemEventHandler):
    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)

    def _should_ignore(self, path: str) -> bool:
        if _pathspec_matcher is None:
            return False

        rel_path = os.path.relpath(path, self.root_path)
        if _pathspec_matcher.match_file(rel_path):
            return True

        for ignore_dir in DEFAULT_IGNORES:
            if ignore_dir in rel_path.split(os.sep):
                return True

        return False

    def on_any_event(self, event: FileSystemEvent):
        if event.is_directory:
            return

        if self._should_ignore(event.src_path):
            return

        filepath = event.src_path
        now = datetime.now()

        with _changes_lock:
            last_time = _last_seen.get(filepath)
            if last_time is not None:
                if (now - last_time).total_seconds() < _DEBOUNCE_SECONDS:
                    return

            _last_seen[filepath] = now

            _changes.append(
                {
                    "path": filepath,
                    "event": _normalize_event_type(event),
                    "timestamp": now.isoformat(),
                }
            )
            logger.debug(f"File change detected: {filepath} ({event.event_type})")


def start_watching(path: str) -> None:
    global _observer, _event_handler, _pathspec_matcher

    watch_path = os.path.abspath(path)

    if not os.path.exists(watch_path):
        raise ValueError(f"Path does not exist: {watch_path}")

    _pathspec_matcher = _load_gitignore(watch_path)

    _event_handler = _ChangeHandler(watch_path)
    _observer = Observer()
    _observer.schedule(_event_handler, watch_path, recursive=True)
    _observer.start()

    logger.info(f"Started watching: {watch_path}")


def stop_watching() -> None:
    global _observer, _event_handler

    if _observer is not None:
        _observer.stop()
        _observer.join()
        _observer = None
        _event_handler = None
        logger.info("Stopped watching")


def get_recent_changes() -> list:
    with _changes_lock:
        return list(_changes)


def clear_changes() -> None:
    with _changes_lock:
        _changes.clear()
        _last_seen.clear()
