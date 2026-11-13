# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Codex model implementation — uses the ``codex`` CLI as an LLM backend.

This lets you use OpenAI models through a ChatGPT subscription.  With
``codex_use_subscription`` on (the default) the CLI runs in
:func:`codex_cli_env` — the daemon's environment minus ``OPENAI_API_KEY``
/ ``CODEX_API_KEY``, plus the panel's ``CODEX_ACCESS_TOKEN`` — so a turn
bills the ChatGPT plan the user is logged in to (``codex login``) rather
than API tokens, and a CLI logged in with an API key is refused.  Off,
the CLI gets ``CODEX_API_KEY`` (the variable it actually reads) from the
``OPENAI_API_KEY`` the rest of KISS uses.

KISS drives Codex as a plain language model, not as a coding agent: the
model invokes ``codex exec --json`` with every native tool switched off
(:data:`CODEX_LOCKDOWN_ARGS` — read-only sandbox, no shell / exec / web
search / sub-agents / apps / user config with its MCP servers / hooks /
skills / AGENTS.md) and with KISS's system prompt, behind
:data:`CODEX_INSTRUCTIONS_PREAMBLE`, installed through
``model_instructions_file`` in place of Codex's built-in instructions.

KISS-level tools reach the model two ways.  Within a ``KISSAgent`` step
(:meth:`CodexModel.native_tools_exposed`) every tool but ``finish`` is
served to Codex as the ``kiss`` MCP server
(:mod:`kiss.core.models.codex_mcp_bridge`), so Codex calls them natively
during the turn — as many as it likes — and each call runs in the KISS
process through the agent's usual executor; the calls are replayed into
the conversation afterwards so the next, ephemeral turn sees them in its
transcript.  ``finish`` (and every tool, outside an agent step) uses the
text-based ``tool_calls`` protocol
(:meth:`CodexModel.generate_and_process_with_tools`), the same approach
:class:`kiss.core.models.claude_code_model.ClaudeCodeModel` and DeepSeek
R1 in :mod:`kiss.core.models.openai_compatible_model` use.  The JSONL
event stream on stdout carries the answer and the usage.
"""

import base64
import contextlib
import functools
import importlib.metadata
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import IO, TYPE_CHECKING, Any

from kiss.core.kiss_error import KISSError
from kiss.core.locked import locked
from kiss.core.models.codex_mcp_bridge import ToolBridge, toml_value
from kiss.core.models.model import (
    Attachment,
    CLITextModel,
    ThinkingCallback,
    TokenCallback,
    _iter_jsonl,
    environ_without,
    render_tool_call,
    text_call_id,
)
from kiss.core.processes import (
    kill_process_tree,
    popen_process_group,
    run_process_group,
)

if TYPE_CHECKING:
    import tiktoken

logger = logging.getLogger(__name__)


# ``codex exec`` flags that turn the coding agent into a bare model: the
# user's ``config.toml`` (MCP servers, hooks, providers) is not loaded
# (``--ignore-user-config``; auth.json still is), no command execution
# (``shell_tool`` / ``unified_exec``), a read-only sandbox so the
# ``apply_patch`` tool some models carry is rejected, no sub-agents
# (``agents.enabled``), no web search, apps, plugins, hooks, memories,
# image or browser tools, no skill descriptions
# (``skills.max_context_tokens`` at its minimum) and no AGENTS.md
# (``project_doc_max_bytes``).  Verified against codex-cli 0.163.
# ``code_mode_host`` stays on: 0.163 routes MCP tool calls through it, so
# switching it off would cut the ``kiss`` MCP server off too ("code-mode
# host is disabled").  With the shell tools off, the host's JavaScript
# cell reaches nothing but the same tools (no ``require``, ``fetch`` or
# ``process``), so it widens nothing.
# Known residual: the CLI has no switch for skill discovery (only per-skill
# ``[[skills.config]]`` entries), so a ``$name`` in the conversation that
# matches a skill installed on the machine (``~/.agents/skills``,
# ``.agents/skills`` up the repo, ``/etc/codex/skills``) still appends
# that SKILL.md to the request as extra text; it has no tools to act on.
CODEX_LOCKDOWN_ARGS: tuple[str, ...] = (
    "--ephemeral",
    "--ignore-user-config",
    "--sandbox", "read-only",
    "-c", "agents.enabled=false",
    "-c", 'web_search="disabled"',
    "-c", "project_doc_max_bytes=0",
    "-c", "skills.max_context_tokens=1",
    "--disable", "shell_tool",
    "--disable", "unified_exec",
    "--disable", "apps",
    "--disable", "plugins",
    "--disable", "view_image",
    "--disable", "sleep_tool",
    "--disable", "skill_search",
    "--disable", "image_generation",
    "--disable", "computer_use",
    "--disable", "browser_use",
    "--disable", "goals",
    "--disable", "hooks",
    "--disable", "tool_suggest",
    "--disable", "memories",
)

# Opens the instructions file every turn.  The model still sees Codex's
# ``<environment_context>`` (cwd, approval policy, the read-only sandbox):
# without this it refuses to write files through KISS's tools, believing
# the sandbox applies to them.
CODEX_INSTRUCTIONS_PREAMBLE = (
    "You are a model inside the KISS agent framework. Codex's own tools are "
    "disabled: the <environment_context> block (cwd, approval_policy, "
    "sandbox_mode such as read-only, network_access) describes only those "
    "disabled built-in tools and does NOT apply to the tools of the `kiss` "
    "MCP server or to the tools described in this prompt. Those tools are "
    "executed by KISS outside of Codex with full read and write access to "
    "the file system, so use them freely for every file, shell and web "
    "operation.\n\n"
)

# Added to the preamble while the KISS tools are served over MCP: the
# text-based protocol below then describes ``finish`` alone.
CODEX_MCP_TOOLS_NOTE = (
    "Call the `kiss` MCP server's tools directly, as native tools, as often "
    "as the task needs within one turn. Only the tools listed under "
    "'Available Tools' below (`finish`) are called with the JSON "
    "`tool_calls` block described there; emit that block at the end of your "
    "answer when the task is complete.\n\n"
)

# How long Codex waits for one call to a KISS tool.  A KISS tool can run
# for hours (``run_parallel``, a build under ``Bash``); the agent's Stop
# button, not this timeout, is the way to end such a call early.
CODEX_MCP_TOOL_TIMEOUT_SECONDS = 24 * 60 * 60

# A reasoning summary line that is only a bold markdown title.  OpenAI's
# backend sends Codex models' summaries as such titles alone
# (``**Running baseline tests**``); the Thinking panel shows plain text.
_BOLD_TITLE_LINE = re.compile(r"^\*\*([^*\n]+)\*\*$", re.MULTILINE)

# Credentials ``codex exec`` prefers over the ChatGPT login: with
# ``CODEX_API_KEY`` in its environment a turn bills API tokens even when
# the user is logged in to a plan (codex-rs/login/src/auth/manager.rs
# ``load_auth``); older CLIs read ``OPENAI_API_KEY`` the same way.
_API_KEY_ENV_VARS: tuple[str, ...] = ("OPENAI_API_KEY", "CODEX_API_KEY")

SUBSCRIPTION_HINT = (
    "Codex is set to bill your ChatGPT subscription (Settings > Codex): sign "
    "in with ChatGPT from that panel (or run `codex login` on this machine), "
    "paste a Codex access token there, or turn the switch off to use "
    "OPENAI_API_KEY."
)
API_KEY_LOGIN_ERROR = (
    "The codex CLI is logged in with an API key (`codex login --with-api-key`), "
    "which would bill per token. Run `codex logout` and sign in with ChatGPT, "
    "or turn the Codex subscription switch off."
)


def codex_api_key_login() -> bool:
    """Report whether the ``codex`` CLI's saved login is an API key.

    Reads ``auth.json`` under ``CODEX_HOME`` (default ``~/.codex``), the
    file ``codex login`` writes: ``{"auth_mode": "apikey", "OPENAI_API_KEY":
    ...}`` after ``codex login --with-api-key``, ``"auth_mode": "chatgpt"``
    with ``tokens`` after a ChatGPT sign-in.  A file without ``auth_mode``
    (older CLIs) counts as an API-key login whenever it holds a key: the
    CLI picks a saved key over saved tokens.  A missing or unreadable file
    is not an API-key login.  Cheap enough to run before every turn, so a
    ``codex logout`` mid-session takes effect at once.

    Returns:
        ``True`` when a turn with the subscription switch on (which strips
        the key variables) would still bill the saved API key per token.
    """
    home = os.environ.get("CODEX_HOME") or os.path.join(os.path.expanduser("~"), ".codex")
    try:
        with open(os.path.join(home, "auth.json"), encoding="utf-8") as f:
            auth = json.load(f)
    except (OSError, ValueError):
        return False
    if not isinstance(auth, dict):
        return False
    mode = auth.get("auth_mode")
    if mode is not None:
        return bool(mode == "apikey")
    return bool(auth.get("OPENAI_API_KEY"))


# Stderr lines of ``codex login status`` -> the panel's ``authMethod``.
_LOGIN_STATUS_METHODS: tuple[tuple[str, str], ...] = (
    ("Logged in using ChatGPT", "chatgpt"),
    ("Logged in using an API key", "api_key"),
    ("Logged in using personal access token", "access_token"),
    ("Logged in using access token", "access_token"),
    ("Logged in using workload identity", "workload_identity"),
    ("Logged in using Amazon Bedrock", "bedrock"),
)

_ANSI_ESCAPES = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_DEVICE_URL = re.compile(r"https://\S+")
_DEVICE_CODE = re.compile(r"^[A-Z0-9]{4,}-[A-Z0-9]{4,}$")


def codex_cli_env() -> dict[str, str]:
    """Return the environment a ``codex/`` turn runs the CLI in.

    With ``DEFAULT_CONFIG.codex_use_subscription`` on: a copy of
    ``os.environ`` without :data:`_API_KEY_ENV_VARS`, so the CLI falls
    through to the subscription credentials — ``CODEX_ACCESS_TOKEN`` (set
    here from ``DEFAULT_CONFIG`` when the settings panel holds one) or
    the ``codex login`` entry in ``~/.codex/auth.json``.  Off: the
    daemon's environment with ``CODEX_API_KEY`` — the only key variable
    this CLI reads — defaulted from ``OPENAI_API_KEY`` (environment or
    settings panel), so the API key the rest of KISS uses bills the turn.

    Returns:
        The child environment.
    """
    from kiss.core.config import DEFAULT_CONFIG

    return _codex_cli_env_for(bool(DEFAULT_CONFIG.codex_use_subscription))


def _codex_cli_env_for(use_subscription: bool) -> dict[str, str]:
    """:func:`codex_cli_env` for one reading of the subscription switch.

    The caller reads the switch exactly once: a settings save on the
    server thread can flip it at any time, and an environment built from
    both states would keep the key variables and add the token (the CLI
    then bills the key).
    """
    from kiss.core.config import DEFAULT_CONFIG

    env = environ_without(_API_KEY_ENV_VARS if use_subscription else ())
    if use_subscription:
        if DEFAULT_CONFIG.CODEX_ACCESS_TOKEN:
            env["CODEX_ACCESS_TOKEN"] = DEFAULT_CONFIG.CODEX_ACCESS_TOKEN
    elif not env.get("CODEX_API_KEY"):
        api_key = env.get("OPENAI_API_KEY") or DEFAULT_CONFIG.OPENAI_API_KEY
        if api_key:
            env["CODEX_API_KEY"] = api_key
    return env


def codex_auth_status(timeout: float = 30.0) -> dict[str, Any]:
    """Report how the ``codex`` CLI would authenticate a ``codex/`` turn.

    Runs ``codex login status`` in :func:`codex_cli_env` and
    maps its one-line answer (printed on stderr) to ``loggedIn`` and
    ``authMethod`` — ``chatgpt``, ``api_key``, ``access_token``,
    ``workload_identity``, ``bedrock`` or ``none`` — plus ``installed``
    (the CLI was found), ``useSubscription`` (the switch), ``hasToken``
    (a ``CODEX_ACCESS_TOKEN`` is configured), ``detail`` (the CLI's own
    line) and, when the CLI could not be run or answered something
    unknown, ``error`` with the reason.  Never raises: the settings
    panel shows whatever this returns.

    Args:
        timeout: Seconds to wait for the CLI.

    Returns:
        The status dict.
    """
    from kiss.core.config import DEFAULT_CONFIG

    status: dict[str, Any] = {
        "installed": False,
        "loggedIn": False,
        "authMethod": "none",
        "useSubscription": bool(DEFAULT_CONFIG.codex_use_subscription),
        "hasToken": bool(DEFAULT_CONFIG.CODEX_ACCESS_TOKEN),
    }
    path = find_codex_executable()
    if path is None:
        status["error"] = "Codex CLI ('codex') not found on PATH."
        return status
    status["installed"] = True
    try:
        proc = run_process_group([path, "login", "status"], timeout, env=codex_cli_env())
    except subprocess.TimeoutExpired:
        status["error"] = f"`codex login status` timed out after {timeout:g}s"
        return status
    except OSError as e:
        status["error"] = f"`codex login status` failed: {e}"
        return status
    detail = (proc.stderr.strip() or proc.stdout.strip())[:300]
    status["detail"] = detail
    for prefix, method in _LOGIN_STATUS_METHODS:
        if detail.startswith(prefix):
            status["loggedIn"] = True
            status["authMethod"] = method
            return status
    if not detail.startswith("Not logged in"):
        status["error"] = (
            f"`codex login status` (exit {proc.returncode}) answered: {detail or '(nothing)'}"
        )
    return status


# The device login whose CLI is still alive (polling for the user to
# enter its code): the prompt it printed and the process, or ``None``.
# ``_device_login_lock`` is held for the whole start of a login, so
# calls serialise: the one made while another is waiting for its code
# gets that code.
_device_login_lock = threading.Lock()
_device_login: "tuple[dict[str, Any], subprocess.Popen[str]] | None" = None


def codex_device_login(timeout: float = 60.0) -> dict[str, Any]:
    """Start ``codex login --device-auth`` and return what the user must do.

    The CLI prints a verification URL and a one-time code, then keeps
    polling (for up to 15 minutes) until the user has entered the code
    in a browser, at which point it stores the ChatGPT login in
    ``~/.codex/auth.json``.  This returns as soon as the URL and code
    appear — ``{"url": ..., "code": ...}`` — and leaves the CLI polling,
    reaped by a daemon thread when it exits; the settings panel polls
    :func:`codex_auth_status` to learn that the login completed.  On
    failure it returns ``{"error": reason}`` instead.  Runs in
    :func:`codex_cli_env` so no API key is involved.

    One login at a time: while the CLI of an earlier call is still
    alive, its code is the one the user can enter, so a call returns
    that same prompt instead of starting a second CLI (another
    surface's click, a retry); calls serialise, so one made while the
    earlier CLI is still printing gets its code too.

    Args:
        timeout: Seconds to wait for the URL and code to appear.

    Returns:
        ``{"url", "code"}`` or ``{"error"}``.
    """
    global _device_login
    with _device_login_lock:
        if _device_login is not None:
            prompt, proc = _device_login
            if proc.poll() is None:
                return dict(prompt)
            _device_login = None
        path = find_codex_executable()
        if path is None:
            return {"error": "Codex CLI ('codex') not found on PATH."}
        try:
            proc = _start_codex(path, "login", "--device-auth")
        except OSError as e:
            return {"error": f"`codex login --device-auth` failed to start: {e}"}
        prompt = {}
        # Both pipes are read on their own threads.  Stderr: the CLI's
        # log lines go there, and a child that fills the pipe (64 KiB)
        # before it prints the code would otherwise block and never
        # print it.  Stdout: a read of the pipe returns only once every
        # holder of its write end is gone, so a descendant that escaped
        # the timeout's group kill into its own session would hold this
        # thread (and the login lock) for as long as it lived; the wait
        # below takes lines from a queue instead, with the deadline, and
        # leaves the pipe to its reader.
        assert proc.stdout is not None and proc.stderr is not None
        stderr_lines: list[str] = []
        drain = threading.Thread(
            target=_read_lines,
            args=(proc.stderr, stderr_lines),
            name="codex-device-login-stderr",
            daemon=True,
        )
        try:
            drain.start()
            stdout_lines: queue.Queue[str | None] = queue.Queue()
            threading.Thread(
                target=_queue_lines,
                args=(proc.stdout, stdout_lines),
                name="codex-device-login-stdout",
                daemon=True,
            ).start()
            deadline = time.monotonic() + timeout
            expired = False
            while "url" not in prompt or "code" not in prompt:
                # The deadline is checked before every line, not only once
                # the queue runs dry: a CLI chattier than the timeout allows
                # reading would otherwise keep the login lock past it.
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    expired = True
                    break
                try:
                    raw = stdout_lines.get(timeout=remaining)
                except queue.Empty:
                    expired = True
                    break
                if raw is None:
                    break
                line = _ANSI_ESCAPES.sub("", raw).strip()
                url = _DEVICE_URL.search(line)
                if url and "url" not in prompt:
                    prompt["url"] = url.group(0)
                elif _DEVICE_CODE.match(line):
                    prompt["code"] = line
            if expired or "url" not in prompt or "code" not in prompt:
                kill_process_tree(proc)
                proc.wait()
                drain.join(timeout=_STDERR_DRAIN_GRACE_SECONDS)
                within = f" within {timeout:g}s" if expired else ""
                return {
                    "error": f"`codex login --device-auth` printed no login code{within}: "
                    + ("".join(stderr_lines).strip()[:300] or "(no output)")
                }
            _device_login = (prompt, proc)
            threading.Thread(
                target=_finish_device_login,
                args=(proc, drain, stderr_lines),
                name="codex-device-login",
                daemon=True,
            ).start()
        except RuntimeError as e:
            # A reader or the finisher could not start (no thread left):
            # the CLI would be nobody's to read or reap, so it ends here.
            _device_login = None
            kill_process_tree(proc)
            proc.wait()
            return {"error": f"`codex login --device-auth` could not be watched: {e}"}
        return dict(prompt)


#: How long the device-login threads wait for the stderr drain after the
#: CLI exited: a grandchild that escaped the kill may still hold the pipe.
_STDERR_DRAIN_GRACE_SECONDS = 5.0


def _read_lines(stream: IO[str], lines: list[str]) -> None:
    """Append every line of *stream* to *lines* until EOF."""
    for line in stream:
        lines.append(line)


def _start_codex(path: str, *args: str) -> "subprocess.Popen[str]":
    """Start ``codex *args`` in :func:`codex_cli_env` with piped output.

    The child leads its own process group: the npm ``codex`` command is a
    Node wrapper around the native binary, so killing the wrapper alone
    would leave the binary (and the pipes) alive — see :func:`kill_process_tree`.
    """
    return popen_process_group(
        [path, *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=codex_cli_env(),
    )


def _finish_device_login(
    proc: "subprocess.Popen[str]", drain: threading.Thread, stderr_lines: list[str]
) -> None:
    """Wait for the device-login CLI to exit, log a failure and forget its prompt.

    Its stdout is read to the end meanwhile by the thread
    :func:`codex_device_login` started, so the CLI cannot block on a
    full pipe while it polls for the login; *drain* collects its stderr
    into *stderr_lines*.  A login started meanwhile (it saw this CLI
    exited) has stored its own prompt, which is left alone.
    """
    global _device_login
    proc.wait()
    with _device_login_lock:
        if _device_login is not None and _device_login[1] is proc:
            _device_login = None
    drain.join(timeout=_STDERR_DRAIN_GRACE_SECONDS)
    if proc.returncode != 0:
        logger.warning(
            "codex login --device-auth exited %s: %s",
            proc.returncode,
            "".join(stderr_lines).strip(),
        )


_UI_CANDIDATE_PATHS: tuple[str, ...] = (
    "/Applications/Codex.app/Contents/Resources/codex",
    "~/Applications/Codex.app/Contents/Resources/codex",
    "~/AppData/Local/Programs/codex/resources/codex.exe",
    "~/AppData/Local/Programs/Codex/resources/codex.exe",
    "/opt/Codex/resources/codex",
    "~/.local/share/Codex/resources/codex",
)


def _find_in_candidate_paths(candidate_paths: Iterable[str]) -> str | None:
    """Return the first executable file found among *candidate_paths*.

    User-home references (``~``) are expanded.  Only files that exist and
    are executable are considered.

    Args:
        candidate_paths: Paths to check, in priority order.

    Returns:
        The expanded absolute path of the first match, or ``None``.
    """
    for candidate in candidate_paths:
        expanded = os.path.expanduser(candidate)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded
    return None


def find_codex_executable() -> str | None:
    """Locate a usable ``codex`` binary, searching ``PATH`` then the UI app.

    The standalone ``codex`` CLI (typically installed via Homebrew or npm)
    is preferred when available on ``PATH``.  If it isn't, but the user
    has the Codex desktop UI installed, the binary bundled with the app
    (e.g. ``/Applications/Codex.app/Contents/Resources/codex``) is used as
    a fallback.

    Returns:
        Absolute path to the ``codex`` binary, or ``None`` if neither the
        CLI nor a known Codex UI installation is present.
    """
    return shutil.which("codex") or _find_in_candidate_paths(_UI_CANDIDATE_PATHS)


def _find_codex_cli() -> str:
    """Locate the ``codex`` executable, raising if neither CLI nor UI is found.

    Returns:
        Absolute path to the ``codex`` binary.

    Raises:
        KISSError: If neither the standalone ``codex`` CLI is on ``PATH``
            nor the Codex UI app is installed in a known location.
    """
    path = find_codex_executable()
    if path is None:
        raise KISSError(
            "Codex binary not found on PATH or in any Codex UI installation. "
            "Install the Codex CLI (https://github.com/openai/codex) or the "
            "Codex desktop app."
        )
    return path


#: How long a resolved ``codex/default`` model is reused, and how soon a
#: failed resolution is retried (seconds).
_DEFAULT_MODEL_TTL_SECONDS = 3600.0
_DEFAULT_MODEL_RETRY_SECONDS = 60.0
#: ``(cli path, CODEX_HOME)`` -> ``(expiry on time.monotonic(), model or None)``.
_default_model_cache: dict[tuple[str, str], tuple[float, str | None]] = {}
# Reentrant and taken only through ``locked``: the writer is the task
# thread, where a Stop injected right after a plain ``with``'s
# ``__enter__`` would leave it held for every later ``codex/default`` turn.
_default_model_lock = threading.RLock()


def codex_default_model(env: dict[str, str], timeout: float = 15.0) -> str | None:
    """Return the model ``codex exec`` runs when given no ``-m`` (``codex/default``).

    ``codex exec --json`` never names the model it ran, so the CLI's
    default is asked of ``codex app-server`` (JSON-RPC, one JSON object
    per line on stdio), which answers without a model request: after
    ``initialize`` / ``initialized``, ``model/list`` lists the presets
    and flags the default with ``isDefault`` (``gpt-6.1-sol`` on CLI
    0.163.0).  That flag is the CLI's built-in default: a ``model`` in
    ``config.toml`` does not move it, matching ``codex exec
    --ignore-user-config`` (:data:`CODEX_LOCKDOWN_ARGS`).  Notifications
    and other lines on the server's stdout are skipped.

    The answer is cached per ``(cli path, CODEX_HOME)`` for an hour, a
    failure for a minute.

    Args:
        env: The turn's child environment (:meth:`CodexModel._prepare_turn_env`),
            so the same login and ``CODEX_HOME`` decide.
        timeout: Seconds the whole exchange may take.

    Returns:
        The default model's id, or ``None`` when the CLI is missing,
        fails, answers garbage, names no default or misses the deadline.
    """
    path = find_codex_executable()
    if path is None:
        return None
    key = (path, env.get("CODEX_HOME", ""))
    cached = _default_model_cache.get(key)
    if cached is not None and cached[0] > time.monotonic():
        return cached[1]
    return _record_default_model(key, _query_default_model(path, env, timeout))


@locked(_default_model_lock)
def _record_default_model(key: tuple[str, str], model: str | None) -> str | None:
    """Cache *model* as *key*'s answer and return it; see :func:`codex_default_model`.

    Args:
        key: ``(cli path, CODEX_HOME)`` the answer was resolved for.
        model: The resolved default, or ``None`` for a failed resolution.

    Returns:
        *model*, or the answer a concurrent turn cached meanwhile when
        *model* is a failure: a late failure must not replace it.
    """
    now = time.monotonic()
    current = _default_model_cache.get(key)
    if model is None and current is not None and current[1] and current[0] > now:
        return current[1]
    ttl = _DEFAULT_MODEL_TTL_SECONDS if model else _DEFAULT_MODEL_RETRY_SECONDS
    _default_model_cache[key] = (now + ttl, model)
    return model


def _query_default_model(path: str, env: dict[str, str], timeout: float) -> str | None:
    """Ask ``path app-server`` for its default model; see :func:`codex_default_model`.

    The server runs until it is killed, which happens in every case.
    """
    deadline = time.monotonic() + timeout
    # Made before the spawn: no call (where a Stop is raised) may run
    # between the spawn and the ``try`` whose ``finally`` kills the server.
    lines: queue.Queue[str | None] = queue.Queue()
    initialize = {"clientInfo": {"name": "kiss", "title": "kiss", "version": "0"}}
    try:
        proc = popen_process_group(
            [path, "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except OSError:
        return None
    assert proc.stdin is not None and proc.stdout is not None
    try:
        # Started inside the try: a reader that cannot start (no thread
        # left) must not leave the server running.
        threading.Thread(
            target=_queue_lines, args=(proc.stdout, lines), name="codex-app-server", daemon=True
        ).start()
        _send_rpc(proc.stdin, {"id": 1, "method": "initialize", "params": initialize})
        if _await_rpc_result(lines, 1, deadline) is None:
            return None
        _send_rpc(proc.stdin, {"method": "initialized", "params": {}})
        _send_rpc(proc.stdin, {"id": 2, "method": "model/list", "params": {}})
        listing = _await_rpc_result(lines, 2, deadline)
    except OSError:  # the server died: a broken pipe
        return None
    finally:
        kill_process_tree(proc)
        proc.wait()
        with contextlib.suppress(OSError):
            proc.stdin.close()
    data = listing.get("data") if isinstance(listing, dict) else None
    for entry in data if isinstance(data, list) else []:
        if isinstance(entry, dict) and entry.get("isDefault") is True:
            model = entry.get("id")
            return model if isinstance(model, str) and model else None
    return None


def _queue_lines(stream: IO[str], lines: "queue.Queue[str | None]") -> None:
    """Put every line of *stream* on *lines*, then ``None`` at EOF, and close it."""
    with stream:
        for line in stream:
            lines.put(line)
    lines.put(None)


def _send_rpc(stdin: IO[str], message: dict[str, Any]) -> None:
    """Write one JSON-RPC 2.0 *message* line to the app server."""
    stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
    stdin.flush()


def _await_rpc_result(lines: "queue.Queue[str | None]", rpc_id: int, deadline: float) -> Any:
    """Return the ``result`` of the response to request *rpc_id*.

    Lines that are not that response (notifications, log noise, other
    ids) are skipped.

    Returns:
        The result, or ``None`` on an error response, EOF or the deadline.
    """
    while True:
        # Checked on every line, so a backlog of notifications cannot
        # outlast the deadline, and a reply after it is ignored.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty:
            return None
        if line is None:
            return None
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict) and message.get("id") == rpc_id:
            return message.get("result")


# GPT-5's tokenizer is tiktoken's ``o200k_base``.  ``tiktoken.get_encoding``
# downloads its ranks, with no timeout, when tiktoken's cache lacks them, so
# they are read from the copy litellm (a dependency of harbor) installs under
# tiktoken's cache key.  The pattern is the one
# ``tiktoken_ext.openai_public.o200k_base`` builds inline.
_O200K_RANKS_FILE = "litellm/litellm_core_utils/tokenizers/fb374d419588a4632f3f557e76b4b70aebbca790"
_O200K_PATTERN = "|".join(
    [
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


@functools.cache
def _gpt5_encoding() -> "tiktoken.Encoding | None":
    """Build GPT-5's tokenizer from litellm's copy of its ranks, once per process.

    Returns:
        The ``o200k_base`` encoding, or ``None`` when tiktoken, litellm or
        litellm's copy of the ranks is missing or unreadable.
    """
    try:
        import tiktoken

        data = importlib.metadata.distribution("litellm").locate_file(_O200K_RANKS_FILE)
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in map(bytes.split, data.read_bytes().splitlines())
        }
        return tiktoken.Encoding(
            "o200k_base", pat_str=_O200K_PATTERN, mergeable_ranks=ranks, special_tokens={}
        )
    except (ImportError, OSError, ValueError) as error:
        logger.warning("No GPT-5 tokenizer (%s): Codex turns are sized by their bytes.", error)
        return None


# tiktoken's regex gives up with a PanicException on a whitespace run of about
# a million characters (its backtracking stack holds a million entries), so the
# text is counted in slices far shorter, at a token or so where a slice cuts.
_TOKENIZER_SLICE_CHARS = 100_000


def _gpt5_token_count(text: str) -> int:
    """Count the GPT-5 tokens of *text*; without the tokenizer, its UTF-8 bytes.

    The bytes bound the count from above: no byte-level BPE token is empty.
    """
    encoding = _gpt5_encoding()
    if encoding is None:
        # surrogatepass: a lone surrogate (a JSON "\ud800") must not fail the sizing.
        return len(text.encode(errors="surrogatepass"))
    return sum(
        len(encoding.encode_ordinary(text[start : start + _TOKENIZER_SLICE_CHARS]))
        for start in range(0, len(text), _TOKENIZER_SLICE_CHARS)
    )


@contextlib.contextmanager
def _image_files(images: list[Attachment]) -> Iterator[list[str]]:
    """Hold *images* in private temp files for the duration of the block.

    Codex reads an image's format from its bytes, so the names carry none.

    Args:
        images: The images to hand to ``codex exec -i``, in label order.

    Yields:
        One path per image, in the same order (no file without images).
    """
    if not images:
        yield []
        return
    with tempfile.TemporaryDirectory(prefix="kiss-codex-") as directory:
        paths = [os.path.join(directory, f"image-{n}") for n in range(1, len(images) + 1)]
        for path, image in zip(paths, images):
            with open(path, "wb") as f:
                f.write(image.data)
        yield paths


class CodexModel(CLITextModel):
    """A model that delegates to the OpenAI Codex CLI for LLM completions.

    Model names use the ``codex/`` prefix.  The part after the prefix is
    passed as the ``-m`` flag to the ``codex`` CLI (e.g.
    ``codex/gpt-5-codex`` → ``-m gpt-5-codex``); an effort alias is
    stripped first and selects the effort instead
    (``codex/gpt-6-astra-xhigh`` → ``-m gpt-6-astra -c
    model_reasoning_effort="xhigh"``, see :meth:`_effort_level`).  The
    special name ``codex/default`` invokes the CLI without ``-m``, letting
    Codex use the model configured for the active ChatGPT account.

    Every turn runs ``codex exec`` locked down to a bare model
    (:data:`CODEX_LOCKDOWN_ARGS`) with KISS's ``system_instruction`` as
    its instructions file, in :func:`codex_cli_env`.  Within a
    :meth:`native_tools_exposed` block Codex calls the tools itself over
    MCP; the tools not served there (and every tool outside such a block)
    use text-based prompting: their descriptions are injected into the
    system prompt and the model's text output is parsed for JSON
    ``tool_calls`` blocks.  Embeddings are not available.
    """

    _cli_model_name = "CodexModel"
    _cli_logger = logger
    _attachment_types = ("image/",)
    serves_tools_natively = True

    def __init__(
        self,
        model_name: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
    ):
        """Initialize a CodexModel instance.

        Args:
            model_name: Full model name including ``codex/`` prefix
                (e.g. ``codex/gpt-5-codex``, ``codex/default``).
            model_config: Optional configuration. Recognised keys:
                - ``system_instruction`` (str): Replaces Codex's built-in
                  instructions for the turn (``model_instructions_file``).
                - ``timeout`` (int): Subprocess timeout in seconds (default 300).
                - ``reasoning_effort`` (str): The effort level to run at,
                  passed as ``model_reasoning_effort``; outranks the
                  level of an alias (see :meth:`_effort_level`).
            token_callback: Optional callback invoked with each streamed
                text token.
            thinking_callback: Optional callback invoked with ``True`` when
                a thinking block starts and ``False`` when it ends.
        """
        super().__init__(
            model_name,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
        from kiss.core.models.model_info import _strip_thinking_alias

        self._cli_model = _strip_thinking_alias(model_name).removeprefix("codex/")
        # The MCP bridge serving the agent's tools, set for the duration
        # of a ``native_tools_exposed`` block.
        self._tool_bridge: ToolBridge | None = None
        # The current turn's child environment, the subscription switch
        # it was built from and whether an API key pays for it, all
        # decided at the top of generate (_prepare_turn_env); the catalog
        # model that prices it (see extract_cost_from_response), set in
        # generate; and
        # whether extract_cost_from_response has logged that this
        # API-billed model has no catalog price.
        self._turn_env: dict[str, str] = {}
        self._use_subscription = False
        self._api_billed = False
        self._priced_model = self._cli_model
        self._warned_unpriced = False

    @contextlib.contextmanager
    def native_tools_exposed(
        self,
        tools_schema: list[dict[str, Any]],
        execute: Callable[[dict[str, Any]], tuple[str, str]],
    ) -> Iterator[None]:
        """Serve *tools_schema* to Codex as the ``kiss`` MCP server within the block.

        Every ``codex exec`` turn run inside the block registers the
        bridge (:meth:`ToolBridge.codex_args`), so Codex calls the tools
        natively through *execute*; the calls it made are replayed into
        the conversation when the turn ends (:meth:`generate`) — and once
        more after the bridge has drained, for a call that was still
        running when a turn failed — and the text-based tools prompt of
        :meth:`generate_and_process_with_tools` then describes only the
        tools *not* served here.

        Args:
            tools_schema: The tools in OpenAI function-calling layout.
            execute: Runs one ``{"name", "arguments"}`` call and returns
                ``(name, result_text)``; called on a bridge thread.
        """
        bridge = ToolBridge(tools_schema, functools.partial(self._execute_for_codex, execute))
        try:
            with bridge:
                self._tool_bridge = bridge
                try:
                    yield
                finally:
                    self._tool_bridge = None
        finally:
            self._replay_served_calls(bridge)

    def _execute_for_codex(
        self, execute: Callable[[dict[str, Any]], tuple[str, str]], call: dict[str, Any]
    ) -> tuple[str, str]:
        """Run *call* through *execute* and return the result as text Codex can take.

        The result is transformed as every tool result is
        (:meth:`tool_result_text_and_attachments`): a binary payload the
        tool embedded in it is lifted out — the CLI has no way to show
        it, and the raw bytes would only bloat the turn — an image or a
        PDF is replaced by a vision model's description, and the usage
        line is appended.  What cannot be described (an audio clip) is
        noted in the text instead, since the replayed conversation holds
        the text only.

        This runs on the bridge's thread, which records exceptions only:
        a Stop landing in the description (the ``KeyboardInterrupt`` of
        the task's stop event, bound to this thread by *execute*) is
        reported as the error that ends the turn, as *execute* reports
        one landing in the tool.

        Args:
            execute: The agent's tool executor.
            call: ``{"name": ..., "arguments": {...}}``.

        Returns:
            ``(name, result_text)``.

        Raises:
            Exception: What *execute* raised, or what the description's
                limit check raised, or a ``KISSError`` naming the Stop
                that interrupted the description.
        """
        name, result = execute(call)
        try:
            text, attachments = self.tool_result_text_and_attachments({"result": result})
        except Exception:
            raise
        except BaseException as e:
            raise KISSError(f"Tool {name} stopped: {e}") from e
        if attachments:
            text += (
                f"\n\n[{len(attachments)} attachment(s) in this tool result could "
                "not be shown to this model]"
            )
        return name, text

    def _replay_served_calls(self, bridge: ToolBridge) -> None:
        """Append the tools Codex called over MCP through *bridge* to the conversation.

        The next, ephemeral turn only knows what its transcript says, so
        each call and its result precede the answer there — also when
        the turn failed and is retried: the calls did run.  The messages
        take the ordinary tool-call shape (an assistant message with
        ``tool_calls``, a ``tool`` message answering it), so the
        conversation stays valid for an API model a later model switch
        hands it to.  The result is recorded as it was served: it was
        transformed once, in :meth:`_execute_for_codex`.

        Args:
            bridge: The bridge whose answered calls are taken.
        """
        for call in bridge.take_served():
            call_id = text_call_id(call.name, call.arguments)
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": render_tool_call(call.name, call.arguments),
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    ],
                }
            )
            self.conversation.append(
                {"role": "tool", "tool_call_id": call_id, "content": call.result}
            )

    def _tool_call_in_flight(self) -> bool:
        """Whether Codex is silent because a bridged KISS tool is still running."""
        return self._tool_bridge is not None and self._tool_bridge.calls_in_flight > 0

    def _tool_call_error(self) -> Exception | None:
        """The fatal error a bridged KISS tool raised (a limit hit), ending the turn."""
        return self._tool_bridge.error if self._tool_bridge is not None else None

    def _build_cli_args(self, instructions_file: str, image_files: Iterable[str] = ()) -> list[str]:
        """Build the ``codex exec`` CLI argument list.

        Args:
            instructions_file: Path of the file holding
                :data:`CODEX_INSTRUCTIONS_PREAMBLE` and KISS's system
                instruction, passed as ``model_instructions_file`` so it
                replaces Codex's built-in prompt.
            image_files: Paths of the images the user attached, each
                passed with ``-i``: Codex sends the N-th ahead of the
                prompt as ``[Image #N]``, the label the prompt uses.

        Returns:
            List of CLI arguments.
        """
        cli = _find_codex_cli()
        args = [cli, "exec", "--json", "--skip-git-repo-check", *CODEX_LOCKDOWN_ARGS]
        # TOML, not JSON: json.dumps writes a non-BMP character (an emoji
        # in TMPDIR) as a surrogate pair, which TOML rejects.
        args.extend(["-c", f"model_instructions_file={toml_value(instructions_file)}"])
        # Without a summary setting (``--ignore-user-config`` drops the
        # user's) the CLI emits no ``reasoning`` items, so no thinking shows.
        args.extend(["-c", 'model_reasoning_summary="detailed"'])
        effort = self._effort_level()
        if effort:
            args.extend(["-c", f"model_reasoning_effort={toml_value(effort)}"])
        if self._tool_bridge is not None:
            args.extend(self._tool_bridge.codex_args(CODEX_MCP_TOOL_TIMEOUT_SECONDS))
        if self._cli_model and self._cli_model != "default":
            args.extend(["-m", self._cli_model])
        for image_file in image_files:
            args.extend(["-i", image_file])
        return args

    def _effort_level(self) -> str | None:
        """Return the reasoning effort this turn runs at, or ``None`` for the CLI default.

        An explicit ``reasoning_effort`` in ``model_config`` (the framework's
        key, which ``KISSAgent._turn_effort`` sets for one turn) wins.  Else
        the level of the model name: the catalog entry's ``thinking`` level
        when the name has one, and ``xhigh`` for the ``-xhigh`` alias the
        catalog does not list (``codex/gpt-6-astra-xhigh``), which
        :func:`~kiss.core.models.model_info._strip_thinking_alias` maps to
        its base model for every lookup.  Read at each launch so a per-turn
        override is honoured.

        Returns:
            The level for ``model_reasoning_effort``, or ``None`` when
            nothing selects one.
        """
        from kiss.core.models.model_info import MODEL_INFO, _strip_thinking_alias

        explicit = self.model_config.get("reasoning_effort")
        if explicit is not None:
            return str(explicit)
        info = MODEL_INFO.get(self.model_name)
        if info is not None:
            return info.thinking
        if _strip_thinking_alias(self.model_name) != self.model_name:
            return "xhigh"
        return None

    def _cli_env(self) -> dict[str, str] | None:
        """The CLI's environment: the one :meth:`_prepare_turn_env` built.

        While the tools are served over MCP it also holds the bridge's
        secret (:meth:`ToolBridge.codex_env`), which Codex forwards to
        the proxy it spawns.
        """
        if self._tool_bridge is None:
            return self._turn_env
        return {**self._turn_env, **self._tool_bridge.codex_env()}

    def generate(self) -> tuple[str, Any]:
        """Generate a response using the Codex CLI.

        The Codex CLI emits newline-delimited JSON events to stdout.  Each
        ``item.completed`` event of type ``agent_message`` carries the full
        assistant text for that item; the token callback is invoked once
        per such item.  The terminal ``turn.completed`` event provides the
        usage information and is what makes the turn a finished one.

        Returns:
            tuple[str, Any]: (generated_text, parsed_response).  The
            response is the dict :meth:`_parse_stream_events` builds.

        Raises:
            KISSError: If the CLI could not be started, exited with a
                failure status, ended without ``turn.completed``, or
                emitted a ``turn.failed`` / ``error``
                event — or, with the subscription switch on, is logged
                in with an API key (:data:`API_KEY_LOGIN_ERROR`).  With
                the switch on, the message of the last two (a failure
                the CLI reported as an event, the refused login) ends
                with :data:`SUBSCRIPTION_HINT`.
            TimeoutError: If the CLI produced no complete turn before the
                deadline.  Retryable, unlike ``KISSError``.
            KeyboardInterrupt: If the user stopped the task mid-turn.
        """
        self._rejected_response = None
        self._turn_env = self._prepare_turn_env()
        self._priced_model = self._cli_model
        if self._cli_model == "default" and self._api_billed:
            self._priced_model = codex_default_model(self._turn_env) or self._cli_model
        prompt, images = self._prompt_and_attachments()
        preamble = CODEX_INSTRUCTIONS_PREAMBLE
        if self._tool_bridge is not None:
            preamble += CODEX_MCP_TOOLS_NOTE
        instructions = preamble + (self.system_instruction_text() or "")
        fd, instructions_file = tempfile.mkstemp(prefix="kiss-codex-", suffix=".md")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(instructions)
            with (
                _image_files(list(images.values())) as image_files,
                self._cli_turn(
                    self._build_cli_args(instructions_file, image_files), "Codex CLI"
                ) as proc,
            ):
                proc.send_prompt(prompt)
                content, result_json, error_message = self._parse_stream_events(proc.lines())
                if error_message is not None:
                    stderr = proc.stderr_text().strip()
                    raise self._with_subscription_hint(
                        KISSError(
                            f"Codex CLI failed: {error_message}"
                            + (f"\nstderr: {stderr}" if stderr else "")
                        )
                    )
                proc.raise_for_exit()
                if "usage" not in result_json:
                    # Stdout ended before ``turn.completed``: the text so
                    # far may be cut short.  The same plain ``KISSError``
                    # the API transports raise for a stream without its
                    # terminal event.
                    raise KISSError(
                        "Codex CLI ended without a turn.completed event, so its "
                        "answer may be incomplete."
                    )
        finally:
            os.unlink(instructions_file)
            if self._tool_bridge is not None:
                self._replay_served_calls(self._tool_bridge)

        # The turn completed: its usage is returned in result_json, so the
        # partial-usage slot must not bill it a second time.
        self._rejected_response = None
        self.conversation.append({"role": "assistant", "content": content})
        return content, result_json

    def _prepare_turn_env(self) -> dict[str, str]:
        """Build the turn's child environment and decide its billing, once per turn.

        One reading of the subscription switch shapes the environment
        (:func:`codex_cli_env`), the verdict and the hint a failure of
        this turn carries (:meth:`_with_subscription_hint`), so a
        settings flip during the call cannot make them disagree.  The billing is
        decided here for good: a ``CODEX_API_KEY`` in the child
        environment makes the turn API spend, whatever the switch says
        afterwards.  So does a key saved by ``codex login --with-api-key``
        (:func:`codex_api_key_login`) when the child has neither a key
        variable nor a ``CODEX_ACCESS_TOKEN``, which the CLI reads first.

        Returns:
            The child environment.

        Raises:
            KISSError: When the switch is on but the turn would bill a
                saved API key: ``codex exec`` would spend it per token
                although the panel promises subscription billing.  A
                stored ``codex login --with-api-key`` is the only way to
                hit this; a configured ``CODEX_ACCESS_TOKEN`` makes the
                saved key moot, and a missing login is left to the turn
                itself, whose failure carries :data:`SUBSCRIPTION_HINT`.
        """
        from kiss.core.config import DEFAULT_CONFIG

        self._use_subscription = bool(DEFAULT_CONFIG.codex_use_subscription)
        env = _codex_cli_env_for(self._use_subscription)
        self._api_billed = bool(env.get("CODEX_API_KEY")) or (
            not env.get("CODEX_ACCESS_TOKEN") and codex_api_key_login()
        )
        if self._use_subscription and self._api_billed:
            raise KISSError(f"{API_KEY_LOGIN_ERROR}\n{SUBSCRIPTION_HINT}")
        return env

    def _with_subscription_hint(self, error: KISSError) -> KISSError:
        """Append :data:`SUBSCRIPTION_HINT` to *error* when the turn ran with the switch on."""
        if not self._use_subscription:
            return error
        # ``args[0]``: ``str(error)`` would prefix "KISS Error:" a second time.
        return KISSError(f"{error.args[0]}\n{SUBSCRIPTION_HINT}")

    def _parse_stream_events(self, lines: Iterable[str]) -> tuple[str, dict[str, Any], str | None]:
        """Parse the JSONL event stream from ``codex exec --json``.

        Recognised events:
            - ``thread.started``: records ``thread_id`` in the result.
            - ``item.completed`` with ``item.type == "agent_message"``:
              text is appended to *content* and forwarded via the token
              callback.
            - ``item.completed`` with ``item.type == "agent_reasoning"``:
              text is forwarded via the token callback wrapped in a
              thinking start/end pair.
            - ``turn.completed``: ``usage`` is recorded in the result,
              which is how :meth:`generate` knows the turn finished.
            - ``error`` / ``turn.failed``: collected into *error_message*.

        Other events (``turn.started``, the start-up notices the locked
        down CLI emits as ``item.type == "error"`` items) are ignored, as
        is a line whose JSON is not an object (progress noise such as
        ``42`` or ``"starting"``).

        Args:
            lines: An iterable of JSON strings (one event per line).

        Returns:
            Tuple of ``(content, result_json, error_message)``.  The
            *error_message* is ``None`` on success.
        """
        content = ""
        # What prices the turn (see extract_cost_from_response), frozen
        # here so the partial-usage response prices the same way: whether
        # an API key paid, decided at launch, the API requests the turn
        # made and, when known, the catalog model the key pays for (only
        # when it is not the name after ``codex/``).  The CLI reports no request count;
        # every tool item (command, file change, MCP call, ...) cost at
        # least one more request after the first, so 1 + tool items
        # bounds it from above, which is all the long-context rule needs.
        result_json: dict[str, Any] = {"api_billed": self._api_billed, "requests": 1}
        if self._priced_model != self._cli_model:
            result_json["priced_model"] = self._priced_model
        error_message: str | None = None

        for event in _iter_jsonl(lines):
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")

            if event_type == "thread.started":
                result_json["thread_id"] = event.get("thread_id", "")
            elif event_type == "item.completed":
                item = event.get("item", {})
                item_type = item.get("type")
                text = item.get("text", "")
                if item_type == "agent_message" and text:
                    content += text
                    self._invoke_token_callback(text)
                elif item_type in ("reasoning", "agent_reasoning") and text:
                    # ``reasoning`` since codex-cli 0.163.0 (ThreadItem's
                    # snake_case tag); ``agent_reasoning`` is the older name.
                    self._emit_as_thinking(_BOLD_TITLE_LINE.sub(r"\1", text))
                elif item_type not in ("agent_message", "agent_reasoning", "reasoning", "error"):
                    result_json["requests"] += 1
            elif event_type == "turn.completed":
                result_json["usage"] = event.get("usage", {})
                # Billed spend from here on: a failure, timeout or Stop
                # before generate() returns still bills it through
                # take_partial_usage_response().
                self._rejected_response = result_json
            elif event_type in ("error", "turn.failed"):
                err = event.get("error") or {}
                error_message = (
                    event.get("message")
                    or (err.get("message") if isinstance(err, dict) else None)
                    or str(event)
                )

        return content, result_json, error_message

    def _text_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> dict[str, Callable[..., Any]]:
        """Leave the tools served over MCP out of the text-based prompt.

        Within a :meth:`native_tools_exposed` block Codex calls those
        natively during the turn (``system_instruction`` reaches it as
        the instructions file), so the prompt describes only the rest
        (``finish``).

        Args:
            function_map: Every tool the framework offers for the turn.
        """
        served = self._tool_bridge.tool_names if self._tool_bridge is not None else frozenset()
        return {name: fn for name, fn in function_map.items() if name not in served}

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int]:
        """Extract token counts from the Codex CLI response dict.

        Codex reports ``input_tokens`` (total prompt tokens, including
        cached and cache-written), ``cached_input_tokens`` (subset that
        was cache-served), ``cache_write_input_tokens`` (subset written
        to the prompt cache, mirrored from the Responses API's
        ``input_tokens_details.cache_write_tokens``; absent on older
        CLIs) and ``output_tokens`` (which already includes
        ``reasoning_output_tokens``).  KISS expects ``input_tokens`` to
        exclude both cache buckets, so we subtract them.

        Args:
            response: The dict returned by :meth:`generate`.

        Returns:
            (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).
        """
        if not isinstance(response, dict):
            return 0, 0, 0, 0
        usage = response.get("usage") or {}
        total_input = usage.get("input_tokens") or 0
        cache_read = usage.get("cached_input_tokens") or 0
        cache_write = usage.get("cache_write_input_tokens") or 0
        non_cached_input = max(total_input - cache_read - cache_write, 0)
        return (
            non_cached_input,
            usage.get("output_tokens") or 0,
            cache_read,
            cache_write,
        )

    def context_tokens_from_response(self, response: Any) -> int:
        """Estimate the context the model works in after the turn.

        ``turn.completed`` sums the usage of every request the turn made,
        and a turn that calls the bridged tools makes several over one
        growing transcript.  With no per-request usage from the CLI, the
        last and largest request is known only within bounds: at least
        the average over ``requests`` (an upper bound on their number,
        see :meth:`_parse_stream_events`) and at most the whole sum
        (exactly the sum for a one-request turn).  Within them the size
        of what the next turn sends decides (the instructions and the
        transcript), so one huge tool result, or several calls made by
        one request, count in full.  That size is counted with GPT-5's
        tokenizer (:func:`_gpt5_token_count`): no ratio of characters or
        bytes to tokens holds for prose and for token-dense text such as
        JSON, CJK, emoji or base64 alike.

        Args:
            response: The dict returned by :meth:`generate`.

        Returns:
            The estimated context size in tokens; 0 without usage.
        """
        total = sum(self.extract_input_output_token_counts_from_response(response))
        if not total or response.get("requests", 1) <= 1:
            return total
        text = CODEX_INSTRUCTIONS_PREAMBLE + (self.system_instruction_text() or "")
        sent = _gpt5_token_count(text + self._build_prompt())
        return min(total, max(total // int(response["requests"]), sent))

    def extract_cost_from_response(self, response: Any) -> float | None:
        """Price the turn at the OpenAI model's rate when the API key pays.

        A turn the ChatGPT plan paid for (the switch was on at launch, or
        no API key was available) keeps the ``codex/*`` catalog price of
        zero (``None`` here).  A turn a ``CODEX_API_KEY`` paid for is
        billed by OpenAI at standard API rates, so its usage is priced
        through :func:`calculate_cost` under the model the response's
        ``priced_model`` names: the one after ``codex/`` (``codex/gpt-5.5``
        -> ``gpt-5.5``), or for ``codex/default`` the CLI's default model
        (:func:`codex_default_model`).  ``codex/default`` whose default
        could not be resolved, and names the catalog cannot price, stay
        at the catalog entry, with a warning once per model instance.

        The long-context tier is a per-request rule, and the CLI reports
        only the sum over the turn's requests, so it applies only to a
        single-request turn (see ``single_request``).  A multi-request
        turn bills at the short-context rates: the sum cannot tell which
        requests crossed the line, and the CLI may compact the transcript
        mid-turn, so even a long first request does not make the later
        ones long.

        Args:
            response: The dict returned by :meth:`generate`.

        Returns:
            The turn's USD cost, or ``None`` to bill the catalog entry.
        """
        from kiss.core.models.model_info import calculate_cost, is_known_model

        if not isinstance(response, dict) or not response.get("api_billed"):
            return None
        model = response.get("priced_model") or self._cli_model
        if not is_known_model(model):
            if not self._warned_unpriced:
                self._warned_unpriced = True
                logger.warning(
                    "%s bills the OpenAI API (codex_use_subscription is off) but "
                    "MODEL_INFO has no price for '%s': its cost is reported as $0.",
                    self.model_name,
                    model,
                )
            return None
        counts = self.extract_input_output_token_counts_from_response(response)
        single_request = response.get("requests", 1) <= 1
        return calculate_cost(model, *counts, single_request=single_request)
