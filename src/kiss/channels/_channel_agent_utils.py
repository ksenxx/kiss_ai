"""Shared helpers for channel agent backends and local config persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_NON_TOOL_METHODS = frozenset(
    {
        "connect",
        "find_channel",
        "find_user",
        "join_channel",
        "poll_messages",
        "send_message",
        "wait_for_reply",
        "is_from_bot",
        "strip_bot_mention",
        "disconnect",
        "get_tool_methods",
    }
)


class ToolMethodBackend:
    """Mixin that exposes public backend methods as agent tools.

    Public methods are discovered dynamically and filtered to exclude
    channel protocol and infrastructure methods.
    """

    def get_tool_methods(self) -> list:
        """Return the backend's public tool methods.

        Returns:
            List of bound callable methods intended for LLM tool use.
        """
        return [
            getattr(self, name)
            for name in sorted(dir(self))
            if not name.startswith("_")
            and name not in _NON_TOOL_METHODS
            and callable(getattr(self, name))
        ]


def load_json_config(path: Path, required_keys: tuple[str, ...]) -> dict[str, str] | None:
    """Load a JSON config file containing string values.

    Args:
        path: Config file path.
        required_keys: Keys that must be present and non-empty.

    Returns:
        Loaded string dictionary, or ``None`` if the file is missing,
        malformed, not a dict, or lacks a required key.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    if any(not data.get(key) for key in required_keys):
        return None
    result: dict[str, str] = {}
    for key, value in data.items():
        result[str(key)] = "" if value is None else str(value)
    return result


def save_json_config(path: Path, data: dict[str, str]) -> None:
    """Save a JSON config file with restricted permissions.

    Args:
        path: Config file path.
        data: String dictionary to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    if sys.platform != "win32":
        path.chmod(0o600)


def clear_json_config(path: Path) -> None:
    """Delete a JSON config file if it exists.

    Args:
        path: Config file path.
    """
    if path.exists():
        path.unlink()
