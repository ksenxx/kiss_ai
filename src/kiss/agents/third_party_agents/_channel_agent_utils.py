# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared helpers for channel agent backends and local config persistence."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import secrets
import sys
import tempfile
import time as _time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml

from kiss.core.config import kiss_home

logger = logging.getLogger(__name__)

_DEFAULT_KISS_DIR = Path.home() / ".kiss"

SILENCE_TOKENS = frozenset({"[SILENT]", "NO_REPLY"})


def summary_for_reply(result: str) -> str | None:
    """Extract the automatic reply text from a task-result YAML string.

    Implements Hermes-style silence tokens: when the agent's summary is
    exactly ``[SILENT]`` or ``NO_REPLY`` (optionally wrapped in HTML
    tags by the daemon's HTML conversion), the automatic reply is
    suppressed.

    Args:
        result: YAML string with 'success' and 'summary' keys, as
            returned by ``run_agent_via_kiss_web``.

    Returns:
        The reply text, or ``None`` when the summary is a silence token.
    """
    try:
        result_yaml = yaml.safe_load(result)
    except yaml.YAMLError:
        result_yaml = None
    summary = (result_yaml.get("summary", "") if isinstance(result_yaml, dict) else "") or result
    if re.sub(r"<[^>]+>", "", summary).strip() in SILENCE_TOKENS:
        return None
    return summary

_NON_TOOL_METHODS = frozenset(
    {
        "connect",
        "find_channel",
        "find_user",
        "join_channel",
        "poll_messages",
        "send_message",
        "send_typing",
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

    def send_typing(self, channel_id: str, thread_ts: str = "") -> None:
        """No-op.  Override for platforms with typing indicators.

        Args:
            channel_id: Channel identifier.
            thread_ts: Optional thread timestamp the indicator applies to.
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
        data = json.loads(path.read_text(encoding="utf-8"))
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
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
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
        channel_dir: Directory for this channel (e.g. ``~/.kiss/third_party_agents/discord``).
        required_keys: Keys that must be present and non-empty for a valid config.
    """

    def __init__(self, channel_dir: Path, required_keys: tuple[str, ...]) -> None:
        self._channel_dir = channel_dir
        self.required_keys = required_keys
        try:
            self._kiss_relative_dir: Path | None = channel_dir.relative_to(_DEFAULT_KISS_DIR)
        except ValueError:
            self._kiss_relative_dir = None

    @property
    def path(self) -> Path:
        """Config file path, resolved lazily so ``KISS_HOME`` is honoured.

        Channel dirs under the default ``~/.kiss`` are rebased onto
        ``$KISS_HOME`` when that env var is set (the test suite points it
        at a fresh per-process temp dir, isolating config state between
        parallel test runs and protecting the user's real configs).
        """
        if self._kiss_relative_dir is not None:
            return kiss_home() / self._kiss_relative_dir / "config.json"
        return self._channel_dir / "config.json"

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


_BREAKER_FAILURE_LIMIT = 5
_BREAKER_PAUSE_SECONDS = 900.0
_MAX_THREADS_PER_TICK = 20
_THREAD_MAX_AGE_SECONDS = 7 * 24 * 3600.0


def _ts_float(value: Any) -> float:
    """Parse a message timestamp as a float, returning ``0.0`` on failure.

    Args:
        value: Timestamp string (or any value) to parse.

    Returns:
        The timestamp as a float, or ``0.0`` when unparseable.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _message_ts(msg: dict[str, Any]) -> float:
    """Sort key: a message's ``ts`` field as a float.

    Args:
        msg: Message dict from ``poll_messages``/``poll_thread_messages``.

    Returns:
        The message timestamp as a float, ``0.0`` when unparseable.
    """
    return _ts_float(msg.get("ts", ""))


def default_channel_state() -> dict[str, Any]:
    """Return a fresh channel-state dict with the canonical schema.

    Returns:
        A state dict with empty ``threads``/``ledger``/``approved_users``/
        ``pending_pairing`` collections and zeroed ``failures``/
        ``paused_until`` counters.
    """
    return {
        "threads": {},
        "ledger": [],
        "failures": 0,
        "paused_until": 0.0,
        "approved_users": [],
        "pending_pairing": {},
        "cursor": "0",
        "thread_rotation": 0,
    }


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Coerce *value* to a finite float, rejecting bools, NaN, and inf.

    Args:
        value: Candidate value from an untrusted state file.
        default: Value to return when *value* is not a finite number.

    Returns:
        ``float(value)`` when it is a finite int/float, else *default*.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _normalize_thread_entry(entry: Any) -> dict[str, Any] | None:
    """Validate one ``threads`` entry from an untrusted state file.

    Args:
        entry: Candidate thread-state value.

    Returns:
        A normalized ``{"chat_id", "last_reply_ts", "updated_at"}``
        dict, or ``None`` when *entry* is not a dict at all.
    """
    if not isinstance(entry, dict):
        return None
    chat_id = entry.get("chat_id", "")
    last_reply_ts = entry.get("last_reply_ts", "")
    return {
        "chat_id": chat_id if isinstance(chat_id, str) else "",
        "last_reply_ts": last_reply_ts if isinstance(last_reply_ts, str) else "",
        "updated_at": _finite_float(entry.get("updated_at", 0.0)),
    }


def _normalize_ledger_entry(entry: Any) -> dict[str, Any] | None:
    """Validate one ``ledger`` entry from an untrusted state file.

    Args:
        entry: Candidate ledger value.

    Returns:
        A normalized entry with string ``channel_id``/``thread_ts``/
        ``text`` and a finite float ``created``, or ``None`` when
        *entry* is not a dict with all three string keys.
    """
    if not isinstance(entry, dict):
        return None
    keys = ("channel_id", "thread_ts", "text")
    if any(not isinstance(entry.get(key), str) for key in keys):
        return None
    normalized = {key: entry[key] for key in keys}
    normalized["created"] = _finite_float(entry.get("created", 0.0))
    return normalized


def _normalize_state(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a loaded state dict field by field against the schema.

    Every canonical field and nested entry is checked for the expected
    type and value range; anything invalid (wrong type, NaN/inf floats,
    negative counters, malformed nested entries) is replaced with the
    safe default so a schema-invalid but JSON-valid file can never
    crash or corrupt a tick.

    Args:
        data: Raw dict parsed from the state file.

    Returns:
        A fully normalized state dict with the canonical schema.
    """
    state = default_channel_state()
    threads = data.get("threads")
    if isinstance(threads, dict):
        for key, entry in threads.items():
            normalized = _normalize_thread_entry(entry)
            if normalized is not None:
                state["threads"][str(key)] = normalized
    ledger = data.get("ledger")
    if isinstance(ledger, list):
        state["ledger"] = [
            e for e in (_normalize_ledger_entry(item) for item in ledger) if e is not None
        ]
    failures = data.get("failures")
    if isinstance(failures, int) and not isinstance(failures, bool) and failures >= 0:
        state["failures"] = failures
    state["paused_until"] = _finite_float(data.get("paused_until", 0.0))
    approved = data.get("approved_users")
    if isinstance(approved, list):
        state["approved_users"] = [u for u in approved if isinstance(u, str)]
    pending = data.get("pending_pairing")
    if isinstance(pending, dict):
        for user_id, info in pending.items():
            if isinstance(info, dict) and isinstance(info.get("code"), str):
                ts = _finite_float(info.get("ts", 0.0))
                state["pending_pairing"][str(user_id)] = {"code": info["code"], "ts": ts}
    cursor = data.get("cursor")
    if isinstance(cursor, str) and cursor:
        state["cursor"] = cursor
    rotation = data.get("thread_rotation")
    if isinstance(rotation, int) and not isinstance(rotation, bool) and rotation >= 0:
        state["thread_rotation"] = rotation
    return state


def load_channel_state(path: Path) -> dict[str, Any]:
    """Load persistent channel-runner state from *path*.

    Missing, unreadable, or malformed files yield a fresh default
    state; loaded dicts are validated and normalized field by field
    (see :func:`_normalize_state`) so every canonical key is always
    present with the expected type, even when the file is JSON-valid
    but schema-invalid.

    Args:
        path: State file path.

    Returns:
        The channel state dict.
    """
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to read channel state %s", path)
            data = None
        if isinstance(data, dict):
            return _normalize_state(data)
    return default_channel_state()


def save_channel_state(path: Path, state: dict[str, Any]) -> None:
    """Persist channel state atomically with restricted permissions.

    Writes to a uniquely named temporary sibling file (via
    :func:`tempfile.mkstemp`, which creates it ``0o600``), then renames
    it over *path* so readers never observe a partially written state
    and concurrent writers never clobber each other's temp file.

    Args:
        path: State file path.
        state: The channel state dict to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(state, indent=2))
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


@contextlib.contextmanager
def channel_state_lock(state_path: Path, blocking: bool) -> Iterator[Any | None]:
    """Acquire the per-channel ``flock`` guarding a state file.

    The same lock serializes the runner's tick (non-blocking: an
    overlapping tick skips) and the pairing admin's read-modify-write
    (blocking: the admin waits for a running tick to finish), so an
    approval can never be overwritten by a stale in-memory save.  On
    platforms without ``fcntl`` the lock file is opened but not locked.

    Args:
        state_path: The state file whose sibling ``.lock`` file to lock.
        blocking: Whether to wait for the lock (admin path) or give up
            immediately when it is held (tick path).

    Yields:
        The open lock file object while the lock is held, or ``None``
        when *blocking* is ``False`` and another process holds it.
    """
    lock_path = state_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fp = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-Unix platforms
            yield fp
            return
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(fp.fileno(), flags)
        except BlockingIOError:
            yield None
            return
        yield fp
    finally:
        fp.close()


def sanitize_state_component(name: str) -> str:
    """Sanitize a workspace/channel name for use in a state file name.

    Args:
        name: Raw component (workspace, channel, or agent name).

    Returns:
        *name* with every non-alphanumeric run replaced by ``_``, or
        ``"default"`` when the result would be empty.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", name) or "default"


def derive_state_path(
    agent_cls: type, agent_label: str, workspace: str, channel: str
) -> Path:
    """Derive the persistent state-file path for a channel poller.

    When the agent's defining module has a module-level ``_config``
    :class:`ChannelConfig` (the convention followed by the channel
    adapters), the state file lives next to that adapter's
    ``config.json``; otherwise it falls back to a per-agent directory
    under ``$KISS_HOME/third_party_agents/channel_state/``.

    The file name combines a readable (sanitized, truncated) prefix
    with a stable hash digest of the *raw* workspace and channel
    strings, so distinct ``(workspace, channel)`` pairs whose sanitized
    forms coincide (e.g. ``a-b``/``c/d`` vs ``a_b``/``c?d``) can never
    share a state file.

    Args:
        agent_cls: The channel agent class.
        agent_label: Human-readable agent name used for the fallback
            directory (e.g. ``"Slack"``).
        workspace: Workspace identifier.
        channel: Channel name being polled.

    Returns:
        The state file path.
    """
    module = sys.modules.get(agent_cls.__module__)
    cfg = getattr(module, "_config", None) if module is not None else None
    digest = hashlib.sha256(f"{workspace}\x00{channel}".encode()).hexdigest()[:10]
    file_name = (
        f"channel_state_{sanitize_state_component(workspace)[:24]}_"
        f"{sanitize_state_component(channel)[:24]}_{digest}.json"
    )
    if isinstance(cfg, ChannelConfig):
        return cfg.path.parent / file_name
    slug = sanitize_state_component(agent_label or agent_cls.__name__)
    return kiss_home() / "third_party_agents" / "channel_state" / slug / file_name


def resolve_channel_overrides(
    cfg: dict[str, str] | None,
    model_name: str | None,
    max_budget: float | None,
    default_model: str,
    default_budget: float,
) -> tuple[str, float]:
    """Apply per-channel model/budget overrides from an adapter config.

    ``None`` (or an empty model string) means the flag was genuinely
    omitted on the command line: only then does a ``channel_model_name``
    / ``channel_max_budget`` config value apply, falling back to the
    given defaults.  An explicitly passed value always wins — even when
    it happens to equal the default (bad config budgets that do not
    parse as a positive finite float are ignored).

    Args:
        cfg: The adapter's loaded config dict, or ``None``.
        model_name: Model name from the CLI, or ``None`` when ``-m``
            was omitted.
        max_budget: Budget from the CLI, or ``None`` when ``-b`` was
            omitted.
        default_model: Model to use when neither ``-m`` nor a config
            override supplies one.
        default_budget: Budget to use when neither ``-b`` nor a config
            override supplies one.

    Returns:
        The effective ``(model_name, max_budget)`` pair.
    """
    if not model_name:
        override_model = str((cfg or {}).get("channel_model_name", "") or "")
        model_name = override_model or default_model
    if max_budget is None:
        max_budget = default_budget
        raw = (cfg or {}).get("channel_max_budget", "")
        if raw:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
            if math.isfinite(value) and value > 0:
                max_budget = value
    return model_name, max_budget


LAUNCH_KWARG_NAMES = frozenset(
    {
        "model_name",
        "work_dir",
        "max_budget",
        "tools",
        "use_worktree",
        "model_config",
        "web_tools",
        "is_parallel",
        "append_basic_tools",
        "append_to_system_prompt",
        "append_to_prompt",
        "timeout",
        "sock_path",
    }
)


def filter_launch_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the kwargs understood by ``run_agent_via_kiss_web``.

    Channel agent ``run()`` shims historically accepted the wider
    ``SorcarAgent.run`` keyword surface (``system_prompt``,
    ``_skip_persistence``, ...).  Tasks now always execute on the
    kiss-web daemon through :func:`kiss.server.sorcar.run`, whose API
    supports only the launcher parameters; everything else is dropped.

    Args:
        kwargs: Arbitrary ``run()`` keyword arguments.

    Returns:
        The subset of *kwargs* accepted by
        :func:`~kiss.agents.third_party_agents._kiss_web_launcher.run_agent_via_kiss_web`.
    """
    return {k: v for k, v in kwargs.items() if k in LAUNCH_KWARG_NAMES}


def agent_tools_file(agent_cls: type) -> str:
    """Return the tools-file path for a channel agent class.

    The ``kiss.server.sorcar.run`` API takes extra agent tools as the
    path of a Python file whose top-level ``get_tools()`` returns the
    tool callables.  For channel agents that file is the agent's OWN
    defining module: each agent module defines a ``get_tools()`` that
    builds a fresh agent from the credentials persisted under
    ``~/.kiss`` and returns its authentication and backend tools.

    Args:
        agent_cls: The channel agent class (e.g. ``SlackAgent``).

    Returns:
        The absolute path of the module defining *agent_cls*, or ``""``
        when that module does not define a callable ``get_tools()``
        (e.g. ``BaseChannelAgent`` itself or test-local classes).
    """
    module = sys.modules.get(agent_cls.__module__)
    if module is None or not callable(getattr(module, "get_tools", None)):
        return ""
    return str(getattr(module, "__file__", "") or "")


class BaseChannelAgent:
    """Base class for channel agents.

    A channel agent is **not** an executable agent itself: every task
    is submitted to the kiss-web daemon through the public API
    :func:`kiss.server.sorcar.run` (via
    :func:`~kiss.agents.third_party_agents._kiss_web_launcher.run_agent_via_kiss_web`),
    and the daemon builds and executes its own chat agent with the
    standard tools (bash, file editing, browser automation).  The
    channel agent instance is the *carrier* of channel identity: the
    :attr:`tools_file` naming the module whose ``get_tools()`` the
    daemon calls to build the channel tools, the :attr:`workspace`
    those tools authenticate under, the :attr:`channel_system_prompt`
    guidance, and the run results the launcher writes back
    (:attr:`last_run_result`, :attr:`budget_used`,
    :attr:`total_tokens_used`, :attr:`total_steps`).

    Subclasses must set ``self._backend`` (a ``ToolMethodBackend``
    instance), override :meth:`_is_authenticated` and
    :meth:`_get_auth_tools`, and define a module-level ``get_tools()``
    in their own module::

        class SlackAgent(BaseChannelAgent): ...

        def get_tools() -> list:
            return SlackAgent()._get_tools()
    """

    _backend: Any

    channel_system_prompt: str = ""

    def __init__(self, name: str = "", workspace: str = "default") -> None:
        self.name = name
        self.workspace = workspace or "default"
        self.last_run_result: str = ""
        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0
        self.total_steps: int = 0

    @property
    def tools_file(self) -> str:
        """Path of the module whose ``get_tools()`` supplies this agent's tools.

        ``""`` when the agent's defining module has no ``get_tools()``
        (plain carriers such as ``KissWebChatAgent`` add no channel
        tools).
        """
        return agent_tools_file(type(self))

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
        """Assemble the channel tool list: auth tools + backend tools.

        The standard agent tools (bash, file editing, browser) are
        supplied by the daemon-built agent, not by this instance.

        Returns:
            Combined list of channel tool callables.
        """
        tools: list = list(self._get_auth_tools())
        if self._is_authenticated():
            tools.extend(self._backend.get_tool_methods())
        return tools

    def run(self, prompt_template: str = "", **kwargs: Any) -> str:
        """Run a task through :func:`kiss.server.sorcar.run`.

        Submits the task to the kiss-web daemon via
        :func:`~kiss.agents.third_party_agents._kiss_web_launcher.run_agent_via_kiss_web`,
        which supplies this agent's channel tools through the API's
        ``tools=`` file-path contract (:attr:`tools_file` — the agent
        module whose ``get_tools()`` the daemon calls), appends
        :attr:`channel_system_prompt` to the prompt, and records the
        YAML result in :attr:`last_run_result` along with the cost /
        token / step totals.  Keyword arguments outside the launcher's
        parameter surface are dropped (see :func:`filter_launch_kwargs`).

        Args:
            prompt_template: The task prompt.
            **kwargs: Launcher keyword arguments (``model_name``,
                ``work_dir``, ``max_budget``, ``tools``,
                ``use_worktree``, ``model_config``, ``web_tools``,
                ``is_parallel``, ``append_basic_tools``,
                ``append_to_system_prompt``, ``append_to_prompt``,
                ``timeout``, ``sock_path``).

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        from kiss.agents.third_party_agents._kiss_web_launcher import (
            run_agent_via_kiss_web,
        )

        return run_agent_via_kiss_web(
            self, prompt_template, **filter_launch_kwargs(kwargs)
        )


class ChannelRunner:
    """One-shot channel message runner.

    Connects to a backend, retrieves recent messages, filters to
    allowed users, skips messages the bot has already replied to, and
    runs a kiss-web daemon task for each pending message.

    When *state_path* is set, Hermes-gateway behaviours are enabled on
    top: a non-blocking per-channel tick lock (overlapping cron ticks
    exit immediately), a circuit breaker that pauses the channel after
    repeated transport failures, an at-least-once delivery ledger with
    redelivery of replies lost mid-send, per-thread daemon-chat
    continuity, thread-continuation processing of user follow-ups, and
    optional DM pairing of unknown senders.  With ``state_path=None``
    behaviour is exactly the stateless legacy behaviour.
    """

    def __init__(
        self,
        backend: Any,
        channel_name: str,
        agent_name: str,
        tools_file: str = "",
        model_name: str = "",
        max_budget: float = 5.0,
        work_dir: str = "",
        allow_users: list[str] | None = None,
        workspace: str = "default",
        state_path: Path | None = None,
        dm_pairing: bool = False,
        cli_name: str = "",
    ) -> None:
        self._backend = backend
        self._channel_name = channel_name
        self._agent_name = agent_name
        self._tools_file = tools_file
        self._model_name = model_name
        self._max_budget = max_budget
        self._work_dir = work_dir or str(kiss_home() / "channel_work")
        self._allow_users = set(allow_users) if allow_users else None
        self._workspace = workspace or "default"
        self._poll_thread_fn = getattr(backend, "poll_thread_messages", None)
        self._state_path = state_path
        self._dm_pairing = dm_pairing
        self._cli_name = cli_name
        self._state: dict[str, Any] | None = None

    def run_once(self) -> int:
        """Check for pending messages, process them, and exit.

        Connects to the backend, joins the configured channel, retrieves
        recent messages, filters to allowed users, skips messages the bot
        has already replied to, and runs a kiss-web daemon task for
        each pending message.  Each message is processed synchronously.

        With a ``state_path`` the tick additionally takes the channel
        tick lock (returning ``0`` immediately when another tick holds
        it), short-circuits while the circuit breaker has the channel
        paused, redelivers pending ledger replies, processes thread
        continuations, and resets the failure counter on success.

        Returns:
            Number of messages processed (including thread
            continuations).  ``0`` when the tick lock is held elsewhere
            or the channel is paused.

        Raises:
            RuntimeError: If connection or channel lookup fails.
        """
        if self._state_path is None:
            return self._run_tick()
        with channel_state_lock(self._state_path, blocking=False) as lock_fp:
            if lock_fp is None:
                logger.info(
                    "Another tick holds the lock for %s; skipping", self._state_path
                )
                return 0
            try:
                self._state = load_channel_state(self._state_path)
                paused_until = float(self._state.get("paused_until", 0.0) or 0.0)
                if paused_until > _time.time():
                    logger.warning(
                        "Channel paused until %s by the circuit breaker; skipping tick",
                        _time.ctime(paused_until),
                    )
                    return 0
                try:
                    processed = self._run_tick()
                except Exception:
                    self._record_transport_failure()
                    raise
                self._state["failures"] = 0
                self._save_state()
                return processed
            finally:
                self._state = None

    def _run_tick(self) -> int:
        """Connect, redeliver pending replies, and process new messages.

        With persistent state the poll starts from the persisted
        ``cursor`` (so platforms with cursor contracts such as
        Telegram's ``getUpdates`` offset never re-serve handled
        messages across cron ticks), stale thread entries are pruned,
        and the backend's returned cursor is stored back into the state
        only after every message *and every thread continuation* of the
        tick has been handled — a tick that fails mid-way, or whose
        continuation pass reported a thread-poll or launch failure,
        keeps the old cursor so nothing is lost and the failed
        follow-up is retried on the next tick.

        Returns:
            Number of messages processed (top-level messages plus
            thread continuations).

        Raises:
            RuntimeError: If connection or channel lookup fails.
        """
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

            self._redeliver_pending()
            self._prune_stale_threads()
            oldest = "0"
            if self._state is not None:
                oldest = str(self._state.get("cursor", "0") or "0")
            messages, new_cursor = self._backend.poll_messages(channel_id, oldest, limit=50)

            processed = 0
            allow = self._effective_allow()
            for msg in messages:
                if self._backend.is_from_bot(msg):
                    continue
                user_id = msg.get("user", "")
                if allow is not None and user_id not in allow:
                    self._maybe_send_pairing(channel_id, msg, user_id)
                    continue
                if self._has_bot_reply(channel_id, msg):
                    continue
                self._handle_message(channel_id, msg)
                processed += 1

            continued, continuations_ok = self._process_thread_continuations(channel_id)
            processed += continued
            if (
                continuations_ok
                and self._state is not None
                and isinstance(new_cursor, str)
                and new_cursor
            ):
                self._state["cursor"] = new_cursor
            return processed
        finally:
            self._disconnect_backend()

    def _save_state(self) -> None:
        """Persist the loaded state to the state file, if any.

        Every runner save happens while :meth:`run_once` holds the
        channel's tick lock (the state is only loaded inside the locked
        section), and the pairing admin takes the same lock in blocking
        mode, so a save can never overwrite a concurrent admin
        approval.
        """
        if self._state_path is not None and self._state is not None:
            save_channel_state(self._state_path, self._state)

    def _record_transport_failure(self) -> None:
        """Count a transport failure; pause the channel after too many.

        After ``_BREAKER_FAILURE_LIMIT`` consecutive failures the
        channel is paused for ``_BREAKER_PAUSE_SECONDS`` (any
        successful tick resets the counter).
        """
        if self._state is None:
            return
        failures = int(self._state.get("failures", 0) or 0) + 1
        self._state["failures"] = failures
        if failures >= _BREAKER_FAILURE_LIMIT:
            self._state["paused_until"] = _time.time() + _BREAKER_PAUSE_SECONDS
            logger.error(
                "Channel paused for %.0fs after %d consecutive transport "
                "failures; to resume, delete the 'paused_until' key in %s "
                "or wait for the pause to expire",
                _BREAKER_PAUSE_SECONDS,
                failures,
                self._state_path,
            )
        self._save_state()

    def _effective_allow(self) -> set[str] | None:
        """Return the effective allow set: allow list plus approved users.

        With DM pairing enabled the allow set is always closed: the
        union of the configured allow list (possibly empty) and the
        state's ``approved_users``, so with no static ``--allow-users``
        only approved senders pass and everyone else gets the pairing
        flow.  Without pairing, ``None`` (everyone allowed) is returned
        when no allow list is configured, matching the legacy
        behaviour.

        Returns:
            The effective allow set, or ``None`` when everyone is
            allowed.
        """
        approved: set[str] = set()
        if self._state is not None:
            approved = {str(u) for u in self._state.get("approved_users", [])}
        if self._dm_pairing:
            return (self._allow_users or set()) | approved
        if self._allow_users is None:
            return None
        return set(self._allow_users) | approved

    def _pairing_reply_text(self, code: str) -> str:
        """Build the pairing reply naming the approval CLI command.

        The suggested command is complete and shell-usable: it includes
        ``--channel`` (when the runner monitors a named channel) and
        ``--workspace`` (when not ``default``), e.g.
        ``kiss-telegram --channel mychan --approve ab12cd34``.

        Args:
            code: The one-time pairing code.

        Returns:
            The reply text sent to the unapproved sender.
        """
        parts = [self._cli_name or "the channel CLI"]
        if self._channel_name:
            parts.append(f"--channel {self._channel_name}")
        if self._workspace != "default":
            parts.append(f"--workspace {self._workspace}")
        parts.append(f"--approve {code}")
        return (
            "You're not authorized to use this bot yet. "
            f"Ask the admin to approve you with: {' '.join(parts)}"
        )

    def _maybe_send_pairing(
        self, channel_id: str, msg: dict[str, Any], user_id: str
    ) -> None:
        """Send a one-time pairing code to an unapproved sender.

        A sender with a pending code is skipped silently so repeated
        messages do not spam the thread.  No-op unless DM pairing is
        enabled and persistent state is loaded.

        Args:
            channel_id: Channel the message arrived in.
            msg: The unapproved sender's message dict.
            user_id: The sender's user identifier.
        """
        if not self._dm_pairing or self._state is None or not user_id:
            return
        pending = self._state.setdefault("pending_pairing", {})
        if user_id in pending:
            return
        code = secrets.token_hex(4)
        pending[user_id] = {"code": code, "ts": _time.time()}
        self._save_state()
        thread_ts = msg.get("thread_ts", msg.get("ts", ""))
        self._send_reply(channel_id, self._pairing_reply_text(code), thread_ts)

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
        if msg.get("reply_count", 0) == 0:
            return False
        return self._bot_replied_in_thread(channel_id, msg)

    def _bot_replied_in_thread(self, channel_id: str, msg: dict[str, Any]) -> bool:
        """Poll a message's thread and check whether the bot has posted there.

        Unlike :meth:`_has_bot_reply` there is no ``reply_count``
        shortcut: the *msg* dict is a snapshot from before the agent
        task ran, so its counters cannot reflect a reply the agent just
        posted with its channel tools.

        Args:
            channel_id: Channel ID containing the message.
            msg: Message dict from poll_messages.

        Returns:
            True if the bot has posted in the message's thread.  False
            when thread polling is unavailable or fails.
        """
        if self._poll_thread_fn is None:
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

    def _bot_posted_after(
        self, channel_id: str, thread_ts: str, snapshot_ts: float
    ) -> bool:
        """Check whether the bot posted in a thread after a snapshot time.

        Used by continuation runs to honor the prompt's promise that a
        reply the agent posts itself suppresses the automatic summary:
        the thread is re-polled after the task and any bot message
        newer than the pre-launch snapshot means the agent already
        answered.

        Args:
            channel_id: Channel ID containing the thread.
            thread_ts: The thread parent timestamp.
            snapshot_ts: Newest bot-message timestamp observed before
                the task was launched.

        Returns:
            True if a bot message newer than *snapshot_ts* exists.
            False when thread polling is unavailable or fails.
        """
        if self._poll_thread_fn is None:
            return False
        try:
            replies, _ = self._poll_thread_fn(channel_id, thread_ts, "0", limit=100)
        except Exception:
            logger.debug("Error re-polling thread %s", thread_ts, exc_info=True)
            return False
        return any(
            self._backend.is_from_bot(r) and _message_ts(r) > snapshot_ts
            for r in replies
        )

    def _prompt_context(self, channel_id: str, thread_ts: str) -> str:
        """Build the channel-context suffix appended to task prompts.

        Args:
            channel_id: Channel the message arrived in.
            thread_ts: Thread timestamp the reply will be posted to.

        Returns:
            The context suffix, or ``""`` when the runner has no tools
            file (no channel tools reach the daemon-built agent).
        """
        if not self._tools_file:
            return ""
        context = (
            f"\n\n[Channel context: you are answering a message in "
            f"channel {channel_id!r}, thread {thread_ts!r}."
        )
        if self._poll_thread_fn is not None:
            # Only thread-polling backends can detect the agent's
            # own reply and suppress the automatic summary.
            context += (
                " If you post a reply there yourself with the "
                "channel messaging tools, no automatic summary "
                "reply is sent.]"
            )
        else:
            context += (
                " A summary of this task is posted to the thread "
                "automatically when you finish, so do not post "
                "one yourself.]"
            )
        context += (
            " [If no reply is warranted, finish with a summary of "
            "exactly [SILENT] and no reply is sent.]"
        )
        return context

    def _stored_chat_id(self, thread_ts: str) -> str:
        """Return the daemon chat id recorded for a thread, or ``""``.

        Args:
            thread_ts: Thread timestamp.

        Returns:
            The stored chat id, or ``""`` when unknown.
        """
        if self._state is None:
            return ""
        entry = self._state.get("threads", {}).get(thread_ts)
        if not isinstance(entry, dict):
            return ""
        return str(entry.get("chat_id", "") or "")

    def _store_thread_state(
        self, thread_ts: str, chat_id: str, last_reply_ts: str
    ) -> None:
        """Record a thread's daemon chat id and last handled timestamp.

        Args:
            thread_ts: Thread timestamp keying the state entry.
            chat_id: Daemon chat id to resume for follow-ups.
            last_reply_ts: Timestamp of the newest user message handled.
        """
        if self._state is None or not thread_ts:
            return
        self._state.setdefault("threads", {})[thread_ts] = {
            "chat_id": chat_id,
            "last_reply_ts": last_reply_ts,
            "updated_at": _time.time(),
        }
        self._save_state()

    def _notify_typing(self, channel_id: str, thread_ts: str) -> None:
        """Best-effort typing indicator before launching an agent task.

        Args:
            channel_id: Channel to indicate typing in.
            thread_ts: Thread the indicator applies to.
        """
        try:
            self._backend.send_typing(channel_id, thread_ts)
        except Exception:
            logger.debug("send_typing failed", exc_info=True)

    def _launch_task(
        self, channel_id: str, thread_ts: str, prompt: str, last_reply_ts: str
    ) -> str:
        """Run one daemon task with per-thread chat continuity.

        Uses a :class:`~kiss.agents.third_party_agents._kiss_web_launcher.KissWebChatAgent`
        carrier: when the state has a chat id recorded for the thread
        it is resumed (so a threaded conversation continues the same
        daemon chat), and after the run the resulting chat id and
        *last_reply_ts* are stored back into the state.  A best-effort
        typing indicator is sent right before the launch.

        Args:
            channel_id: Channel the message arrived in.
            thread_ts: Thread timestamp keying the chat session.
            prompt: The full task prompt.
            last_reply_ts: Timestamp of the newest user message handled.

        Returns:
            YAML result string from the daemon run.
        """
        from kiss.agents.third_party_agents._kiss_web_launcher import (
            KissWebChatAgent,
            run_agent_via_kiss_web,
        )

        agent = KissWebChatAgent(self._agent_name)
        agent.workspace = self._workspace
        chat_id = self._stored_chat_id(thread_ts)
        if chat_id:
            agent.resume_chat_by_id(chat_id)
        self._notify_typing(channel_id, thread_ts)
        Path(self._work_dir).mkdir(parents=True, exist_ok=True)
        result = run_agent_via_kiss_web(
            agent,
            prompt,
            model_name=self._model_name,
            max_budget=self._max_budget,
            work_dir=self._work_dir,
            tools=self._tools_file or None,
        )
        self._store_thread_state(thread_ts, agent.chat_id, last_reply_ts)
        return result

    def _handle_message(self, channel_id: str, msg: dict[str, Any]) -> None:
        """Run one agent task for an inbound message.

        The agent is launched as a kiss-web registered agent via
        :func:`run_agent_via_kiss_web` (``_cmd_run``) so the task is
        live-visible and interactable from any connected remote
        webview while it runs.  The channel tools come from the
        runner's tools file (the agent module's ``get_tools()``, per
        the ``kiss.server.sorcar.run`` tools-file contract); after the
        run the task summary is posted to the message's thread unless
        the agent already replied there itself.  With persistent state
        the thread's daemon chat id is resumed and stored so later
        follow-ups continue the same chat.
        """
        text = self._backend.strip_bot_mention(msg.get("text", ""))
        thread_ts = msg.get("thread_ts", msg.get("ts", ""))
        session_key = f"{channel_id}:{msg.get('ts', '')}"

        prompt = text + self._prompt_context(channel_id, thread_ts)
        try:
            result = self._launch_task(
                channel_id, thread_ts, prompt, str(msg.get("ts", ""))
            )
            if not self._bot_replied_in_thread(channel_id, msg):
                summary = summary_for_reply(result)
                if summary is None:
                    logger.info("Silence token; no reply sent for %s", session_key)
                else:
                    self._send_reply(channel_id, summary, thread_ts)
        except Exception as e:
            logger.error("Agent error for %s: %s", session_key, e, exc_info=True)
            self._send_reply(channel_id, f"Error processing your message: {e}", thread_ts)

    def _select_thread_followups(
        self,
        replies: list[dict[str, Any]],
        thread_ts: str,
        last_reply_ts: str,
    ) -> list[dict[str, Any]]:
        """Select user follow-up messages that need a continuation run.

        A follow-up is a non-bot, allow-list-passing, non-empty reply
        (other than the thread parent) newer than *last_reply_ts* — the
        persisted marker of the newest user message already handled.
        Deduplication happens **only** on that marker: a follow-up that
        arrived while the previous task was still running (and thus is
        older than the bot's eventual answer) is still selected, never
        dropped.

        Args:
            replies: Messages from ``poll_thread_messages``.
            thread_ts: The thread parent timestamp (excluded).
            last_reply_ts: Timestamp of the newest user message already
                handled for this thread.

        Returns:
            The matching messages sorted by timestamp.
        """
        last = _ts_float(last_reply_ts)
        allow = self._effective_allow()
        selected: list[dict[str, Any]] = []
        for reply in replies:
            if self._backend.is_from_bot(reply):
                continue
            ts = str(reply.get("ts", ""))
            if not ts or ts == thread_ts:
                continue
            if allow is not None and reply.get("user", "") not in allow:
                continue
            if _ts_float(ts) <= last:
                continue
            if not str(reply.get("text", "")).strip():
                continue
            selected.append(reply)
        selected.sort(key=_message_ts)
        return selected

    def _prune_stale_threads(self) -> None:
        """Drop thread entries not updated for 7 days from the state.

        Runs on every tick that has persistent state, independent of
        whether the backend supports thread polling.  Staleness is
        judged by the entry's ``updated_at`` wall-clock stamp (written
        by :meth:`_store_thread_state`), never by platform message IDs,
        which are not comparable to Unix time.
        """
        if self._state is None:
            return
        threads = self._state.get("threads", {})
        now = _time.time()
        stale = [
            thread_ts
            for thread_ts, entry in threads.items()
            if now - _finite_float(entry.get("updated_at", 0.0)) > _THREAD_MAX_AGE_SECONDS
        ]
        if stale:
            for thread_ts in stale:
                del threads[thread_ts]
            self._save_state()

    def _threads_to_process(self) -> list[str]:
        """Return the thread timestamps to poll for continuations.

        At most 20 threads are polled per tick, ordered most recently
        updated first.  When more threads exist, a rotation offset
        persisted in the state advances by 20 each tick so older active
        threads are never starved forever.

        Returns:
            Thread timestamps to poll for continuations.
        """
        if self._state is None:
            return []
        threads = self._state.get("threads", {})

        def updated_at(thread_ts: str) -> float:
            entry = threads[thread_ts]
            return _finite_float(entry.get("updated_at", 0.0)) if isinstance(entry, dict) else 0.0

        ordered = sorted(threads, key=updated_at, reverse=True)
        if len(ordered) <= _MAX_THREADS_PER_TICK:
            return ordered
        offset = int(self._state.get("thread_rotation", 0) or 0) % len(ordered)
        selected = [
            ordered[(offset + i) % len(ordered)] for i in range(_MAX_THREADS_PER_TICK)
        ]
        self._state["thread_rotation"] = (offset + _MAX_THREADS_PER_TICK) % len(ordered)
        self._save_state()
        return selected

    def _process_thread_continuations(self, channel_id: str) -> tuple[int, bool]:
        """Run continuation tasks for new user replies in known threads.

        For every thread recorded in the state (capped per tick), polls
        the thread for user follow-ups newer than the bot's last reply,
        concatenates their texts into one follow-up prompt, resumes the
        thread's stored daemon chat, and replies in the thread through
        the ledger-backed :meth:`_send_reply` (honoring silence
        tokens).

        Retry semantics: a failed continuation launch posts a
        best-effort error reply to the thread but never advances the
        thread's ``last_reply_ts`` (:meth:`_launch_task` stores it only
        after a successful run), and :meth:`_select_thread_followups`
        deduplicates only on that marker — never on bot posts — so the
        same follow-up is selected and retried on the next tick despite
        the error reply.

        Args:
            channel_id: Channel the threads live in.

        Returns:
            A ``(processed, all_success)`` pair: the number of
            continuation tasks run, and whether every thread poll and
            continuation launch succeeded.  ``all_success`` is ``False``
            when any thread poll or launch failed; the caller then keeps
            the old poll cursor so the tick is retried.  These failures
            are deliberately not counted as circuit-breaker transport
            failures.
        """
        if self._state is None or self._poll_thread_fn is None:
            return 0, True
        processed = 0
        all_success = True
        for thread_ts in self._threads_to_process():
            entry = self._state.get("threads", {}).get(thread_ts)
            if not isinstance(entry, dict):
                continue
            try:
                replies, _ = self._poll_thread_fn(channel_id, thread_ts, "0", limit=100)
            except Exception:
                logger.debug("Thread poll failed for %s", thread_ts, exc_info=True)
                all_success = False
                continue
            followups = self._select_thread_followups(
                replies, thread_ts, str(entry.get("last_reply_ts", ""))
            )
            if not followups:
                continue
            text = "\n\n".join(
                self._backend.strip_bot_mention(str(m.get("text", "")))
                for m in followups
            )
            last_ts = str(followups[-1].get("ts", ""))
            prompt = text + self._prompt_context(channel_id, thread_ts)
            bot_snapshot = max(
                (_message_ts(r) for r in replies if self._backend.is_from_bot(r)),
                default=0.0,
            )
            try:
                result = self._launch_task(channel_id, thread_ts, prompt, last_ts)
            except Exception as e:
                logger.error(
                    "Continuation error for %s: %s", thread_ts, e, exc_info=True
                )
                self._send_reply(
                    channel_id, f"Error processing your message: {e}", thread_ts
                )
                all_success = False
                continue
            summary = summary_for_reply(result)
            if summary is None:
                logger.info("Silence token; no reply sent for thread %s", thread_ts)
            elif self._bot_posted_after(channel_id, thread_ts, bot_snapshot):
                logger.info(
                    "Agent replied in thread %s itself; automatic summary suppressed",
                    thread_ts,
                )
            else:
                self._send_reply(channel_id, summary, thread_ts)
            processed += 1
        return processed, all_success

    def _send_reply(self, channel_id: str, text: str, thread_ts: str) -> None:
        """Send a reply message with at-least-once delivery semantics.

        When persistent state is loaded the reply is first recorded in
        the delivery ledger, then sent (retrying once on transient
        failure); on success the ledger entry is removed.  A reply that
        could not be sent stays in the ledger and is redelivered on the
        next tick with a ``(recovered reply)`` prefix.  Without state
        this is exactly the legacy retry-once send.

        Args:
            channel_id: Channel to post to.
            text: Message text.
            thread_ts: Thread timestamp for threading.
        """
        entry: dict[str, Any] | None = None
        if self._state is not None:
            entry = {
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "text": text,
                "created": _time.time(),
            }
            self._state.setdefault("ledger", []).append(entry)
            self._save_state()
        sent = self._send_with_retry(channel_id, text, thread_ts)
        if sent and entry is not None and self._state is not None:
            ledger = self._state.get("ledger", [])
            if entry in ledger:
                ledger.remove(entry)
            self._save_state()

    def _send_with_retry(self, channel_id: str, text: str, thread_ts: str) -> bool:
        """Send a message, retrying once on transient failure.

        Args:
            channel_id: Channel to post to.
            text: Message text.
            thread_ts: Thread timestamp for threading.

        Returns:
            Whether the send succeeded.
        """
        for attempt in range(2):
            try:
                self._backend.send_message(channel_id, text, thread_ts)
                return True
            except Exception:
                if attempt == 0:
                    logger.warning("Reply failed, retrying...", exc_info=True)
                    _time.sleep(1)
                else:
                    logger.error("Reply failed after retry", exc_info=True)
        return False

    def _redeliver_pending(self) -> None:
        """Redeliver ledger entries whose original send never succeeded.

        Each pending entry is resent with a ``(recovered reply)``
        prefix (Hermes-style honest at-least-once semantics) and
        removed from the ledger on success; entries that still fail
        stay for the next tick.
        """
        if self._state is None:
            return
        for entry in list(self._state.get("ledger", [])):
            text = "(recovered reply) " + str(entry.get("text", ""))
            sent = self._send_with_retry(
                str(entry.get("channel_id", "")), text, str(entry.get("thread_ts", ""))
            )
            if sent:
                ledger = self._state.get("ledger", [])
                if entry in ledger:
                    ledger.remove(entry)
                self._save_state()

    def _disconnect_backend(self) -> None:
        """Best-effort backend cleanup hook."""
        try:
            self._backend.disconnect()
        except Exception:
            logger.warning("Backend disconnect failed", exc_info=True)


def _handle_pairing_admin(
    agent_cls: type,
    agent_label: str,
    workspace: str,
    channel: str,
    approve_code: str,
    list_pending: bool,
) -> None:
    """Handle the ``--approve CODE`` / ``--list-pending`` admin flags.

    ``--list-pending`` prints every pending ``user_id:code`` pair for
    the channel's state and returns.  ``--approve`` moves the pending
    user whose code matches into ``approved_users``, prints the user
    id, and returns; when no pending code matches it prints an error
    and exits nonzero.  Both flags require ``--channel`` (the state is
    per-channel).

    The entire read-modify-write runs under the channel's state lock
    (blocking mode), so an approval issued while a tick is running
    waits for the tick to finish instead of being overwritten by the
    runner's stale in-memory save.

    Args:
        agent_cls: The channel agent class (for state-path derivation).
        agent_label: Human-readable channel name (e.g. ``"Slack"``).
        workspace: Workspace identifier.
        channel: Channel name whose state to operate on.
        approve_code: The pairing code to approve, or ``""``.
        list_pending: Whether to list pending pairing requests.

    Raises:
        SystemExit: Nonzero when ``--channel`` is missing or no pending
            request matches *approve_code*.
    """
    if not channel:
        print("Error: --approve and --list-pending require --channel")
        sys.exit(1)
    state_path = derive_state_path(agent_cls, agent_label, workspace, channel)
    with channel_state_lock(state_path, blocking=True):
        state = load_channel_state(state_path)
        pending = state.get("pending_pairing", {})
        if list_pending:
            for user_id, info in sorted(pending.items()):
                code = info.get("code", "") if isinstance(info, dict) else ""
                print(f"{user_id}:{code}")
            return
        for user_id, info in list(pending.items()):
            if isinstance(info, dict) and info.get("code") == approve_code:
                approved = state.setdefault("approved_users", [])
                if user_id not in approved:
                    approved.append(user_id)
                del pending[user_id]
                save_channel_state(state_path, state)
                print(f"Approved user: {user_id}")
                return
    print(f"Error: no pending pairing request with code {approve_code!r}")
    sys.exit(1)


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

    Poll mode persists Hermes-gateway state (thread chat continuity,
    delivery ledger, circuit breaker, pairing) in a per-channel state
    file next to the adapter's ``config.json`` (or under
    ``$KISS_HOME/third_party_agents/channel_state/`` for adapters
    without a module-level ``_config``), honors per-channel
    ``channel_model_name`` / ``channel_max_budget`` config overrides
    when ``-m`` / ``-b`` are not passed, and supports the pairing admin
    flags ``--pairing``, ``--approve CODE``, and ``--list-pending``.

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

    from kiss.agents.third_party_agents._channel_cli import (
        _build_arg_parser,
        _build_run_kwargs,
        _print_run_stats,
    )

    if len(sys.argv) <= 1:  # pragma: no branch
        parts = [f"Usage: {cli_name} [-m MODEL] [-e ENDPOINT] [-b BUDGET]"]
        parts.append("[-w PWD] [-t TASK] [-f FILE]")
        parts.append("[--workspace WS]")
        if make_backend is not None:
            parts.append("[--channel CH]")
            parts.append("[--pairing] [--approve CODE] [--list-pending]")
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
        parser.add_argument(
            "--pairing",
            action="store_true",
            default=False,
            help="Enable DM pairing: unapproved senders get a one-time approval code",
        )
        parser.add_argument(
            "--approve",
            default="",
            metavar="CODE",
            help="Approve the pending user whose pairing code is CODE (requires --channel)",
        )
        parser.add_argument(
            "--list-pending",
            action="store_true",
            default=False,
            help="List pending pairing requests for --channel and exit",
        )
    args = parser.parse_args()

    workspace: str = args.workspace

    channel: str = getattr(args, "channel", "")
    if make_backend is not None:
        approve_code = str(args.approve or "")
        list_pending = bool(args.list_pending)
        if approve_code or list_pending:
            _handle_pairing_admin(
                agent_cls,
                channel_name,
                workspace,
                channel,
                approve_code,
                list_pending,
            )
            return
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
        from kiss.core import config as core_config
        from kiss.core.models.model_info import get_default_model

        module = sys.modules.get(agent_cls.__module__)
        module_config = getattr(module, "_config", None) if module is not None else None
        cfg = module_config.load() if isinstance(module_config, ChannelConfig) else None
        model_name, max_budget = resolve_channel_overrides(
            cfg,
            args.model_name,
            args.max_budget,
            get_default_model(),
            core_config.DEFAULT_CONFIG.max_budget,
        )
        runner = ChannelRunner(
            backend=backend,
            channel_name=channel,
            agent_name=f"{channel_name} Background Agent",
            tools_file=agent_tools_file(agent_cls),
            model_name=model_name,
            max_budget=max_budget,
            work_dir=args.work_dir,
            allow_users=allow_users,
            workspace=workspace,
            state_path=derive_state_path(agent_cls, channel_name, workspace, channel),
            dm_pairing=getattr(args, "pairing", False),
            cli_name=cli_name,
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
    prompt = run_kwargs.pop("prompt_template", "")

    from kiss.agents.third_party_agents._kiss_web_launcher import (
        run_agent_via_kiss_web,
    )

    start_time = _time.time()
    run_agent_via_kiss_web(agent, prompt, **run_kwargs)
    elapsed = _time.time() - start_time

    _print_run_stats(agent, elapsed)
