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

import contextlib
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from kiss.core.kiss_error import KISSError
from kiss.core.models.codex_mcp_bridge import ServedCall, ToolBridge
from kiss.core.models.model import (
    CLITextModel,
    ThinkingCallback,
    TokenCallback,
    _iter_jsonl,
    environ_without,
    parse_binary_attachments,
)
from kiss.core.processes import kill_process_tree, popen_process_group, run_process_group

logger = logging.getLogger(__name__)


def _served_call_text(call: ServedCall) -> str:
    """Render a tool call Codex made over MCP as the assistant turn that made it.

    Args:
        call: The call as the bridge answered it.

    Returns:
        The call as a Python-style invocation, the way the agent's
        trajectory records every tool call.
    """
    arguments = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
    return f"```python\n{call.name}({arguments})\n```"


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


def codex_device_login(timeout: float = 60.0, fresh: bool = False) -> dict[str, Any]:
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
    earlier CLI is still printing gets its code too.  A call with
    *fresh* ends the earlier CLI and starts a new one.

    Args:
        timeout: Seconds to wait for the URL and code to appear.
        fresh: Start a new CLI even while an earlier one is still alive.

    Returns:
        ``{"url", "code"}`` or ``{"error"}``.
    """
    global _device_login
    with _device_login_lock:
        if _device_login is not None:
            prompt, proc = _device_login
            if proc.poll() is None:
                if not fresh:
                    return dict(prompt)
                kill_process_tree(proc)
            _device_login = None
        path = find_codex_executable()
        if path is None:
            return {"error": "Codex CLI ('codex') not found on PATH."}
        try:
            proc = _start_codex(path, "login", "--device-auth")
        except OSError as e:
            return {"error": f"`codex login --device-auth` failed to start: {e}"}
        prompt = {}
        expired = threading.Event()
        timer = threading.Timer(timeout, _kill_expired, args=(proc, expired))
        timer.start()
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = _ANSI_ESCAPES.sub("", raw).strip()
                url = _DEVICE_URL.search(line)
                if url and "url" not in prompt:
                    prompt["url"] = url.group(0)
                elif _DEVICE_CODE.match(line):
                    prompt["code"] = line
                if "url" in prompt and "code" in prompt:
                    break
        finally:
            # cancel() does not say whether the timer fired first; once
            # it is joined, ``expired`` is exact, so a CLI the timer
            # killed just as its code arrived is never published.
            timer.cancel()
            timer.join()
        if expired.is_set() or "url" not in prompt or "code" not in prompt:
            kill_process_tree(proc)
            _, stderr = proc.communicate()
            within = f" within {timeout:g}s" if expired.is_set() else ""
            return {
                "error": f"`codex login --device-auth` printed no login code{within}: "
                + (stderr.strip()[:300] or "(no output)")
            }
        _device_login = (prompt, proc)
        threading.Thread(
            target=_finish_device_login, args=(proc,), name="codex-device-login", daemon=True
        ).start()
        return dict(prompt)


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


def _kill_expired(proc: "subprocess.Popen[str]", expired: threading.Event) -> None:
    """Timer callback: record that the login wait expired, then kill the CLI."""
    expired.set()
    kill_process_tree(proc)


def _finish_device_login(proc: "subprocess.Popen[str]") -> None:
    """Wait for the device-login CLI to exit, log a failure and forget its prompt."""
    global _device_login
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        logger.warning("codex login --device-auth exited %s: %s", proc.returncode, stderr.strip())
    with _device_login_lock:
        if _device_login is not None and _device_login[1] is proc:
            _device_login = None


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


class CodexModel(CLITextModel):
    """A model that delegates to the OpenAI Codex CLI for LLM completions.

    Model names use the ``codex/`` prefix.  The part after the prefix is
    passed as the ``-m`` flag to the ``codex`` CLI (e.g.
    ``codex/gpt-5-codex`` → ``-m gpt-5-codex``).  The special name
    ``codex/default`` invokes the CLI without ``-m``, letting Codex use
    the model configured for the active ChatGPT account.

    Every turn runs ``codex exec`` locked down to a bare model
    (:data:`CODEX_LOCKDOWN_ARGS`) with KISS's ``system_instruction`` as
    its instructions file, in :func:`codex_cli_env`.  Tool
    calling is supported via text-based prompting: tool descriptions
    are injected into the system prompt and the model's text output is
    parsed for JSON ``tool_calls`` blocks.  Embeddings are not available.
    """

    _cli_model_name = "CodexModel"
    _cli_logger = logger
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
        self._cli_model = (
            model_name[len("codex/") :] if model_name.startswith("codex/") else model_name
        )
        # The MCP bridge serving the agent's tools, set for the duration
        # of a ``native_tools_exposed`` block.
        self._tool_bridge: ToolBridge | None = None
        # The current turn's child environment and whether an API key
        # pays for it, both decided at the top of generate
        # (_prepare_turn_env), and whether extract_cost_from_response
        # has logged that this API-billed model has no catalog price.
        self._turn_env: dict[str, str] = {}
        self._api_billed = False
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

        A binary payload a tool embedded in its result (a screenshot) is
        lifted out: the CLI has no way to show it, and the raw bytes
        would only bloat the turn.  The conversation gets the usual
        notice when the call is replayed.

        Args:
            execute: The agent's tool executor.
            call: ``{"name": ..., "arguments": {...}}``.

        Returns:
            ``(name, result_text)``.
        """
        name, result = execute(call)
        text, attachments = parse_binary_attachments(result)
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
        hands it to.

        Args:
            bridge: The bridge whose answered calls are taken.
        """
        for call in bridge.take_served():
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": _served_call_text(call),
                    "tool_calls": [
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    ],
                }
            )
            self.add_function_results_to_conversation_and_return(
                [(call.name, {"result": call.result})]
            )

    def _tool_call_in_flight(self) -> bool:
        """Whether Codex is silent because a bridged KISS tool is still running."""
        return self._tool_bridge is not None and self._tool_bridge.calls_in_flight > 0

    def _tool_call_error(self) -> Exception | None:
        """The fatal error a bridged KISS tool raised (a limit hit), ending the turn."""
        return self._tool_bridge.error if self._tool_bridge is not None else None

    def _build_cli_args(self, instructions_file: str) -> list[str]:
        """Build the ``codex exec`` CLI argument list.

        Args:
            instructions_file: Path of the file holding
                :data:`CODEX_INSTRUCTIONS_PREAMBLE` and KISS's system
                instruction, passed as ``model_instructions_file`` so it
                replaces Codex's built-in prompt.

        Returns:
            List of CLI arguments.
        """
        cli = _find_codex_cli()
        args = [cli, "exec", "--json", "--skip-git-repo-check", *CODEX_LOCKDOWN_ARGS]
        args.extend(["-c", f"model_instructions_file={json.dumps(instructions_file)}"])
        if self._tool_bridge is not None:
            args.extend(self._tool_bridge.codex_args(CODEX_MCP_TOOL_TIMEOUT_SECONDS))
        if self._cli_model and self._cli_model != "default":
            args.extend(["-m", self._cli_model])
        return args

    def _cli_env(self) -> dict[str, str] | None:
        """The CLI's environment: the one :meth:`_prepare_turn_env` built."""
        return self._turn_env

    def generate(self) -> tuple[str, Any]:
        """Generate a response using the Codex CLI.

        The Codex CLI emits newline-delimited JSON events to stdout.  Each
        ``item.completed`` event of type ``agent_message`` carries the full
        assistant text for that item; the token callback is invoked once
        per such item.  The terminal ``turn.completed`` event provides the
        usage information.

        Returns:
            tuple[str, Any]: (generated_text, parsed_response).  The
            response is a dict ``{"usage": {...}, "thread_id": "..."}``.

        Raises:
            KISSError: If the CLI could not be started, exited with a
                failure status, or emitted a ``turn.failed`` / ``error``
                event — or, with the subscription switch on, is logged
                in with an API key (:data:`API_KEY_LOGIN_ERROR`).  When
                the switch is on, the
                message ends with :data:`SUBSCRIPTION_HINT`.
            TimeoutError: If the CLI produced no complete turn before the
                deadline.  Retryable, unlike ``KISSError``.
            KeyboardInterrupt: If the user stopped the task mid-turn.
        """
        self._turn_env = self._prepare_turn_env()
        prompt = self._build_prompt()
        preamble = CODEX_INSTRUCTIONS_PREAMBLE
        if self._tool_bridge is not None:
            preamble += CODEX_MCP_TOOLS_NOTE
        fd, instructions_file = tempfile.mkstemp(prefix="kiss-codex-", suffix=".md")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(preamble + (self.system_instruction_text() or ""))
        try:
            args = self._build_cli_args(instructions_file)
            with self._cli_turn(args, "Codex CLI") as proc:
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
        finally:
            os.unlink(instructions_file)
            if self._tool_bridge is not None:
                self._replay_served_calls(self._tool_bridge)

        self.conversation.append({"role": "assistant", "content": content})
        return content, result_json

    def _prepare_turn_env(self) -> dict[str, str]:
        """Build the turn's child environment and decide its billing, once per turn.

        One reading of the subscription switch shapes both the
        environment (:func:`codex_cli_env`) and the verdict, so a settings
        flip during the call cannot make them disagree.  The billing is
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

        use_subscription = bool(DEFAULT_CONFIG.codex_use_subscription)
        env = _codex_cli_env_for(use_subscription)
        self._api_billed = bool(env.get("CODEX_API_KEY")) or (
            not env.get("CODEX_ACCESS_TOKEN") and codex_api_key_login()
        )
        if use_subscription and self._api_billed:
            raise KISSError(f"{API_KEY_LOGIN_ERROR}\n{SUBSCRIPTION_HINT}")
        return env

    @staticmethod
    def _with_subscription_hint(error: KISSError) -> KISSError:
        """Append :data:`SUBSCRIPTION_HINT` to *error* when the switch is on."""
        from kiss.core.config import DEFAULT_CONFIG

        if not DEFAULT_CONFIG.codex_use_subscription:
            return error
        return KISSError(f"{error}\n{SUBSCRIPTION_HINT}")

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
            - ``turn.completed``: ``usage`` is recorded in the result.
            - ``error`` / ``turn.failed``: collected into *error_message*.

        Other events (``turn.started``, the start-up notices the locked
        down CLI emits as ``item.type == "error"`` items) are ignored.

        Args:
            lines: An iterable of JSON strings (one event per line).

        Returns:
            Tuple of ``(content, result_json, error_message)``.  The
            *error_message* is ``None`` on success.
        """
        content = ""
        # What prices the turn (see extract_cost_from_response): whether
        # an API key paid, decided at launch, and the API requests the
        # turn made.  The CLI reports no request count; every tool item
        # (command, file change, MCP call, ...) cost at least one more
        # request after the first, so 1 + tool items bounds it from
        # above, which is all the long-context rule needs.
        result_json: dict[str, Any] = {"api_billed": self._api_billed, "requests": 1}
        error_message: str | None = None

        for event in _iter_jsonl(lines):
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
                elif item_type == "agent_reasoning" and text:
                    self._emit_as_thinking(text)
                elif item_type not in ("agent_message", "agent_reasoning", "reasoning", "error"):
                    result_json["requests"] += 1
            elif event_type == "turn.completed":
                result_json["usage"] = event.get("usage", {})
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

    def extract_cost_from_response(self, response: Any) -> float | None:
        """Price the turn at the OpenAI model's rate when the API key pays.

        A turn the ChatGPT plan paid for (the switch was on at launch, or
        no API key was available) keeps the ``codex/*`` catalog price of
        zero (``None`` here).  A turn a ``CODEX_API_KEY`` paid for is
        billed by OpenAI at standard API rates, so its usage is priced
        through :func:`calculate_cost` under the model named after
        ``codex/`` (``codex/gpt-5.5`` -> ``gpt-5.5``); the per-request
        long-context tier applies only to a single-request turn (see
        ``single_request``).  ``codex/default`` and other names the
        catalog cannot price stay at the catalog entry, with a warning
        once per model instance.

        Args:
            response: The dict returned by :meth:`generate`.

        Returns:
            The turn's USD cost, or ``None`` to bill the catalog entry.
        """
        from kiss.core.models.model_info import calculate_cost, is_known_model

        if not isinstance(response, dict) or not response.get("api_billed"):
            return None
        if not is_known_model(self._cli_model):
            if not self._warned_unpriced:
                self._warned_unpriced = True
                logger.warning(
                    "%s bills the OpenAI API (codex_use_subscription is off) but "
                    "MODEL_INFO has no price for '%s': its cost is reported as $0.",
                    self.model_name,
                    self._cli_model,
                )
            return None
        counts = self.extract_input_output_token_counts_from_response(response)
        single_request = response.get("requests", 1) <= 1
        return calculate_cost(self._cli_model, *counts, single_request=single_request)
