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
    make_daemon_backend: Callable[[], Any] | None = None,
    daemon_poll_interval: float = 3.0,
) -> None:
    """Standard CLI entry point shared by all channel agents.

    Handles argument parsing, daemon mode, and interactive (one-shot) mode.
    Each channel agent's ``main()`` delegates to this function.

    Args:
        agent_cls: The channel Agent class to instantiate (e.g. ``SlackAgent``).
        cli_name: CLI command name for the usage message (e.g. ``"kiss-slack"``).
        channel_name: Human-readable channel name (e.g. ``"Slack"``).
            Used in daemon messages and agent naming.
        make_daemon_backend: Factory that creates and configures a
            backend for daemon mode. Should call ``sys.exit(1)`` if
            required config is missing. Pass ``None`` to disable daemon mode.
        daemon_poll_interval: Message poll interval for daemon mode in seconds.
    """
    from kiss.agents.sorcar.sorcar_agent import (
        _build_arg_parser,
        _resolve_task,
        cli_ask_user_question,
        cli_wait_for_user,
    )

    if len(sys.argv) <= 1:  # pragma: no branch
        parts = [f"Usage: {cli_name} [-m MODEL] [-e ENDPOINT] [-b BUDGET]"]
        parts.append("[-w WORK_DIR] [-t TASK] [-f FILE] [-n]")
        if make_daemon_backend is not None:
            parts.append("[--daemon]")
        print(" ".join(parts))
        sys.exit(1)

    parser = _build_arg_parser()
    parser.add_argument(
        "-n", "--new", action="store_true", help="Start a new chat session"
    )
    if make_daemon_backend is not None:
        parser.add_argument(
            "--daemon", action="store_true", help="Run as background daemon"
        )
        parser.add_argument(
            "--daemon-channel", default="", help="Channel/chat to monitor"
        )
        parser.add_argument(
            "--allow-users", default="",
            help="Comma-separated user IDs to allow",
        )
    args = parser.parse_args()

    if make_daemon_backend is not None and getattr(args, "daemon", False):
        from kiss.channels.background_agent import ChannelDaemon

        backend = make_daemon_backend()
        allow_users = (
            [u.strip() for u in args.allow_users.split(",") if u.strip()] or None
        )
        daemon = ChannelDaemon(
            backend=backend,
            channel_name=args.daemon_channel,
            agent_name=f"{channel_name} Background Agent",
            extra_tools=backend.get_tool_methods(),
            model_name=args.model_name,
            max_budget=args.max_budget,
            work_dir=args.work_dir or str(Path.home() / ".kiss" / "daemon_work"),
            poll_interval=daemon_poll_interval,
            allow_users=allow_users,
        )
        print(f"Starting {channel_name} daemon... (Ctrl+C to stop)")
        try:
            daemon.run()
        except KeyboardInterrupt:
            print("Daemon stopped.")
        return

    agent = agent_cls()
    task_description = _resolve_task(args)
    work_dir = args.work_dir or str(Path(".").resolve())
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    if args.new:  # pragma: no branch
        agent.new_chat()
    else:
        agent.resume_chat(task_description)

    model_config: dict[str, Any] = {}
    if args.endpoint:  # pragma: no branch
        model_config["base_url"] = args.endpoint

    run_kwargs: dict[str, Any] = {
        "prompt_template": task_description,
        "model_name": args.model_name,
        "max_budget": args.max_budget,
        "model_config": model_config,
        "work_dir": work_dir,
        "verbose": args.verbose,
        "wait_for_user_callback": cli_wait_for_user,
        "ask_user_question_callback": cli_ask_user_question,
    }

    start_time = _time.time()
    agent.run(**run_kwargs)
    elapsed = _time.time() - start_time

    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")
