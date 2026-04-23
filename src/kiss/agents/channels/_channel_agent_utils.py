"""Shared helpers for channel agent backends and local config persistence."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time as _time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

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
        "poll_thread_messages",
    }
)


class ToolMethodBackend:
    """Mixin that exposes public backend methods as agent tools.

    Public methods are discovered dynamically and filtered to exclude
    channel protocol and infrastructure methods.

    Provides sensible defaults for all infrastructure methods so that
    channel backends only need to override methods with non-trivial
    behaviour (e.g. Slack's ``find_channel`` which queries the API).
    """

    _connection_info: str = ""

    @property
    def connection_info(self) -> str:
        """Human-readable connection status string."""
        return self._connection_info

    def find_channel(self, name: str) -> str | None:
        """Return *name* as the channel ID.

        Override for platforms that resolve names via an API call.

        Args:
            name: Channel name or identifier.

        Returns:
            The channel identifier, or ``None`` if *name* is empty.
        """
        return name if name else None

    def find_user(self, username: str) -> str | None:
        """Return *username* as the user ID.

        Override for platforms that resolve usernames via an API call.

        Args:
            username: Username or identifier.

        Returns:
            The user identifier, or ``None`` if *username* is empty.
        """
        return username if username else None

    def join_channel(self, channel_id: str) -> None:
        """No-op.  Override for platforms that require joining a channel.

        Args:
            channel_id: Channel identifier.
        """

    def disconnect(self) -> None:
        """No-op.  Override for platforms that need connection cleanup."""

    def is_from_bot(self, msg: dict[str, Any]) -> bool:
        """Return ``False``.  Override for platforms that can identify bot messages.

        Args:
            msg: Message dict from :meth:`poll_messages`.

        Returns:
            Whether the message was sent by the bot itself.
        """
        return False

    def strip_bot_mention(self, text: str) -> str:
        """Return *text* unchanged.  Override for platforms with bot @-mentions.

        Args:
            text: Raw message text.

        Returns:
            Text with bot mentions removed.
        """
        return text

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

    Use this mixin **before** ``ChatSorcarAgent`` in the MRO::

        class SlackAgent(BaseChannelAgent, ChatSorcarAgent): ...
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


class ChannelRunner:
    """One-shot channel message runner.

    Connects to a backend, retrieves recent messages, filters to
    allowed users, skips messages the bot has already replied to, and
    runs a ChatSorcarAgent for each pending message.
    """

    def __init__(
        self,
        backend: Any,
        channel_name: str,
        agent_name: str,
        extra_tools: list | None = None,
        model_name: str = "",
        max_budget: float = 5.0,
        work_dir: str = "",
        allow_users: list[str] | None = None,
    ) -> None:
        self._backend = backend
        self._channel_name = channel_name
        self._agent_name = agent_name
        self._extra_tools = extra_tools or []
        self._model_name = model_name
        self._max_budget = max_budget
        self._work_dir = work_dir or str(Path.home() / ".kiss" / "channel_work")
        self._allow_users = set(allow_users) if allow_users else None
        self._poll_thread_fn = getattr(backend, "poll_thread_messages", None)

    def run_once(self) -> int:
        """Check for pending messages, process them, and exit.

        Connects to the backend, joins the configured channel, retrieves
        recent messages, filters to allowed users, skips messages the bot
        has already replied to, and runs a ChatSorcarAgent for each
        pending message.  Each message is processed synchronously.

        Returns:
            Number of messages processed.

        Raises:
            RuntimeError: If connection or channel lookup fails.
        """
        from kiss.core.base import Base

        Base.reset_global_budget()
        if not self._backend.connect():
            raise RuntimeError(f"Failed to connect: {self._backend.connection_info}")
        logger.info("Connected: %s", self._backend.connection_info)
        try:
            channel_id = ""
            if self._channel_name:
                channel_id = self._backend.find_channel(self._channel_name) or ""
                if not channel_id:
                    raise RuntimeError(f"Channel not found: {self._channel_name!r}")
                self._backend.join_channel(channel_id)
                logger.info("Joined channel: %s (%s)", self._channel_name, channel_id)

            messages, _ = self._backend.poll_messages(channel_id, "0", limit=50)

            processed = 0
            for msg in messages:
                if self._backend.is_from_bot(msg):
                    continue
                user_id = msg.get("user", "")
                if self._allow_users and user_id not in self._allow_users:
                    continue
                if self._has_bot_reply(channel_id, msg):
                    continue
                self._handle_message(channel_id, msg)
                processed += 1

            return processed
        finally:
            self._disconnect_backend()

    def _has_bot_reply(self, channel_id: str, msg: dict[str, Any]) -> bool:
        """Check if the bot has already replied to a message's thread.

        Uses ``poll_thread_messages`` if the backend supports it.
        Returns ``False`` when thread polling is unavailable or the
        message has no replies.

        Args:
            channel_id: Channel ID containing the message.
            msg: Message dict from poll_messages.

        Returns:
            True if the bot has already replied in the thread.
        """
        if self._poll_thread_fn is None:
            return False
        if msg.get("reply_count", 0) == 0:
            return False
        msg_ts = msg.get("ts", "")
        if not msg_ts:
            return False
        try:
            replies, _ = self._poll_thread_fn(channel_id, msg_ts, "0", limit=100)
            return any(self._backend.is_from_bot(r) for r in replies)
        except Exception:
            logger.debug("Error checking thread replies for %s", msg_ts, exc_info=True)
            return False

    def _handle_message(self, channel_id: str, msg: dict[str, Any]) -> None:
        """Run one agent task for an inbound message."""
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

        text = self._backend.strip_bot_mention(msg.get("text", ""))
        thread_ts = msg.get("thread_ts", msg.get("ts", ""))
        session_key = f"{channel_id}:{msg.get('ts', '')}"

        agent = ChatSorcarAgent(self._agent_name)
        agent.new_chat()

        tools = list(self._extra_tools)
        replied = threading.Event()

        def reply(message: str) -> str:
            """Send a reply to the current conversation.

            Args:
                message: Text to send as the bot's reply.

            Returns:
                JSON string with ok status.
            """
            replied.set()
            try:
                self._backend.send_message(channel_id, message, thread_ts)
                return json.dumps({"ok": True})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        tools.append(reply)

        Path(self._work_dir).mkdir(parents=True, exist_ok=True)
        try:
            result = agent.run(
                prompt_template=text,
                model_name=self._model_name,
                max_budget=self._max_budget,
                work_dir=self._work_dir,
                tools=tools,
                verbose=False,
            )
            if not replied.is_set():  # pragma: no branch
                result_yaml = yaml.safe_load(result)
                summary = (result_yaml.get("summary", "") if result_yaml else "") or result
                self._send_reply(channel_id, summary, thread_ts)
        except Exception as e:
            logger.error("Agent error for %s: %s", session_key, e, exc_info=True)
            self._send_reply(channel_id, f"Error processing your message: {e}", thread_ts)

    def _send_reply(self, channel_id: str, text: str, thread_ts: str) -> None:
        """Send a reply message, retrying once on transient failure.

        Args:
            channel_id: Channel to post to.
            text: Message text.
            thread_ts: Thread timestamp for threading.
        """
        for attempt in range(2):
            try:
                self._backend.send_message(channel_id, text, thread_ts)
                return
            except Exception:
                if attempt == 0:
                    logger.warning("Reply failed, retrying...", exc_info=True)
                    _time.sleep(1)
                else:
                    logger.error("Reply failed after retry", exc_info=True)

    def _disconnect_backend(self) -> None:
        """Best-effort backend cleanup hook."""
        try:
            self._backend.disconnect()
        except Exception:
            logger.warning("Backend disconnect failed", exc_info=True)


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
        _build_arg_parser,
        _build_run_kwargs,
        _print_recent_chats,
        _print_run_stats,
    )

    if len(sys.argv) <= 1:  # pragma: no branch
        parts = [f"Usage: {cli_name} [-m MODEL] [-e ENDPOINT] [-b BUDGET]"]
        parts.append("[-w PWD] [-t TASK] [-f FILE] [-n] [--chat-id ID] [-l]")
        parts.append("[--workspace WS]")
        if make_backend is not None:
            parts.append("[--channel CH]")
        if extra_usage:
            parts.append(extra_usage)
        print(" ".join(parts))
        sys.exit(1)

    parser = _build_arg_parser()
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
        runner = ChannelRunner(
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
        count = runner.run_once()
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
