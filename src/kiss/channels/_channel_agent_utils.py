"""Shared helpers for channel agent backends and local config persistence."""

from __future__ import annotations

import json
import sys
import time as _time
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


class ChannelConfig:
    """Encapsulates the 4-function config persistence pattern used by channel agents.

    Replaces the repeated ``_config_path`` / ``_load_config`` / ``_save_config`` /
    ``_clear_config`` boilerplate in each channel agent module.

    Args:
        channel_dir: Directory for this channel (e.g. ``~/.kiss/channels/discord``).
        required_keys: Keys that must be present and non-empty for a valid config.
    """

    def __init__(self, channel_dir: Path, required_keys: tuple[str, ...]) -> None:
        self.path = channel_dir / "config.json"
        self.required_keys = required_keys

    def load(self) -> dict[str, str] | None:
        """Load the config, returning ``None`` if missing or invalid.

        Returns:
            Loaded string dictionary, or ``None``.
        """
        return load_json_config(self.path, self.required_keys)

    def save(self, data: dict[str, str]) -> None:
        """Save *data* to the config file with restricted permissions.

        Args:
            data: String dictionary to persist.
        """
        save_json_config(self.path, data)

    def clear(self) -> None:
        """Delete the config file if it exists."""
        clear_json_config(self.path)


class BaseChannelAgent:
    """Mixin for channel agent classes that provides a standard ``_get_tools()``
    implementation combining auth tools with backend tools.

    Subclasses must set ``self._backend`` (a ``ToolMethodBackend`` instance)
    and override :meth:`_is_authenticated` and :meth:`_get_auth_tools`.

    Use this mixin **before** ``StatefulSorcarAgent`` in the MRO::

        class SlackAgent(BaseChannelAgent, StatefulSorcarAgent): ...
    """

    _backend: Any

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated and ready for use.

        Subclasses must override this.
        """
        return False

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions.

        Subclasses must override this.
        """
        return []

    def _get_tools(self) -> list:
        """Assemble the full tool list: super tools + auth tools + backend tools.

        Returns:
            Combined list of tool callables.
        """
        tools: list = super()._get_tools()  # type: ignore[misc]
        tools.extend(self._get_auth_tools())
        if self._is_authenticated():
            tools.extend(self._backend.get_tool_methods())
        return tools


def channel_main(
    agent_cls: type,
    cli_name: str,
    *,
    channel_name: str = "",
    make_backend: Callable[..., Any] | None = None,
    extra_usage: str = "",
) -> None:
    """Standard CLI entry point shared by all channel agents.

    Handles argument parsing and either one-shot poll mode (when
    ``--channel`` is given) or interactive mode (when ``-t`` is given).
    Each channel agent's ``main()`` delegates to this function.

    Args:
        agent_cls: The channel Agent class to instantiate (e.g. ``SlackAgent``).
        cli_name: CLI command name for the usage message (e.g. ``"kiss-slack"``).
        channel_name: Human-readable channel name (e.g. ``"Slack"``).
            Used in status messages and agent naming.
        make_backend: Factory that creates and configures a backend for
            poll mode.  May accept a ``workspace`` keyword argument; if
            so, the ``--workspace`` CLI value is forwarded.  Should call
            ``sys.exit(1)`` if required config is missing.
            Pass ``None`` to disable poll mode.
        extra_usage: Additional usage flags to append to the usage line
            (e.g. ``"[--list-workspaces]"``).
    """
    import inspect

    from kiss.agents.sorcar.cli_helpers import (
        _apply_chat_args,
        _build_chat_arg_parser,
        _build_run_kwargs,
        _print_recent_chats,
        _print_run_stats,
    )

    if len(sys.argv) <= 1:  # pragma: no branch
        parts = [f"Usage: {cli_name} [-m MODEL] [-e ENDPOINT] [-b BUDGET]"]
        parts.append("[-w WORK_DIR] [-t TASK] [-f FILE] [-n] [--chat-id ID] [-l]")
        parts.append("[--workspace WS]")
        if make_backend is not None:
            parts.append("[--channel CH]")
        if extra_usage:
            parts.append(extra_usage)
        print(" ".join(parts))
        sys.exit(1)

    parser = _build_chat_arg_parser()
    parser.add_argument(
        "--workspace",
        default="default",
        help="Workspace identifier for multi-workspace token management (default: 'default')",
    )
    if make_backend is not None:
        parser.add_argument("--channel", default="", help="Channel/chat to monitor for messages")
        parser.add_argument(
            "--allow-users",
            default="",
            help="Comma-separated usernames or user IDs to allow",
        )
    args = parser.parse_args()

    if args.list_chat_id:
        _print_recent_chats()
        sys.exit(0)

    workspace: str = args.workspace

    channel: str = getattr(args, "channel", "")
    if make_backend is not None and channel:
        from kiss.channels.background_agent import ChannelPoller

        sig = inspect.signature(make_backend)
        if "workspace" in sig.parameters:
            backend = make_backend(workspace=workspace)
        else:
            backend = make_backend()
        allow_users_raw = [u.strip() for u in args.allow_users.split(",") if u.strip()]
        allow_users: list[str] | None = None
        if allow_users_raw:
            allow_users = []
            for raw in allow_users_raw:
                resolved = backend.find_user(raw)
                if resolved:
                    if resolved != raw:
                        print(f"  Resolved user {raw!r} -> {resolved}")
                    allow_users.append(resolved)
                else:
                    allow_users.append(raw)
            allow_users = allow_users or None
        poller = ChannelPoller(
            backend=backend,
            channel_name=channel,
            agent_name=f"{channel_name} Background Agent",
            extra_tools=backend.get_tool_methods(),
            model_name=args.model_name,
            max_budget=args.max_budget,
            work_dir=args.work_dir or str(Path.home() / ".kiss" / "channel_work"),
            allow_users=allow_users,
        )
        print(f"Checking {channel_name} channel for pending messages...")
        count = poller.run_once()
        print(f"Processed {count} message(s).")
        return

    sig = inspect.signature(agent_cls)
    if "workspace" in sig.parameters:
        agent = agent_cls(workspace=workspace)
    else:
        agent = agent_cls()
    run_kwargs = _build_run_kwargs(args)
    _apply_chat_args(agent, args, task=run_kwargs.get("prompt_template", ""))

    start_time = _time.time()
    agent.run(**run_kwargs)
    elapsed = _time.time() - start_time

    _print_run_stats(agent, elapsed)
