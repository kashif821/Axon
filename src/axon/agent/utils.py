from __future__ import annotations

import os
from pathlib import Path


def get_directory_tree(
    path: str = ".",
    max_depth: int = 2,
    ignore_dirs: list[str] | None = None,
) -> str:
    if ignore_dirs is None:
        ignore_dirs = [
            ".git",
            "__pycache__",
            "node_modules",
            "venv",
            ".venv",
            ".env",
            ".axon",
        ]

    def _build_tree(current_path: Path, prefix: str = "", depth: int = 0) -> list[str]:
        if depth >= max_depth:
            return []

        lines = []
        try:
            entries = sorted(
                current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name)
            )
        except PermissionError:
            return [f"{prefix}[Permission Denied]"]
        except Exception:
            return [f"{prefix}[Error Reading]"]

        for entry in entries:
            if entry.name in ignore_dirs:
                continue

            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                extension = "├── " if entry != entries[-1] else "└── "
                lines.extend(
                    _build_tree(
                        entry, prefix + extension.replace("─", "   "), depth + 1
                    )
                )
            else:
                lines.append(f"{prefix}{entry.name}")

        return lines

    tree_lines = [f"{path}/"]
    tree_lines.extend(_build_tree(Path(path)))
    return "\n".join(tree_lines)
