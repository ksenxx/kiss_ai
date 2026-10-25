# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Claude Code model implementation — uses the ``claude`` CLI as an LLM backend.

This lets you use Claude models through a Claude Code subscription at
subsidized per-token pricing.  With ``claude_code_use_subscription`` on
(the default) the CLI runs in :func:`claude_subscription_env` — the
daemon's environment minus ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN``,
plus the panel's ``CLAUDE_CODE_OAUTH_TOKEN`` — so a turn bills the plan the
user is logged in to rather than API tokens.

KISS drives Claude Code as a plain language model, not as a coding agent:
the model invokes ``claude --print`` with ``--tools ""`` (no native
tools), ``--safe-mode`` (no CLAUDE.md, skills, plugins, hooks or MCP
servers from the user's settings), ``--strict-mcp-config`` and
``--system-prompt`` carrying KISS's system prompt in place of Claude
Code's own, and consumes the stream-json event stream emitted on stdout.
KISS-level tools reach the model through the text-based ``tool_calls``
protocol: tool descriptions are injected into that system prompt and the
model's text output is parsed for tool-call JSON — the same approach used
by :class:`kiss.core.models.codex_model.CodexModel` and for DeepSeek R1
in :mod:`kiss.core.models.openai_compatible_model`.

The subprocess supervision both CLI-backed adapters share
(:class:`~kiss.core.models.model._CLIProcess`,
:class:`~kiss.core.models.model._ToolCallFilteredStream`) lives beside
:class:`~kiss.core.models.model.CLITextModel` in
:mod:`kiss.core.models.model`, which this module and
:mod:`kiss.core.models.codex_model` both import it from.
"""

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from typing import Any

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import (
    PRICE_MODIFIER_FIELDS,
    cache_creation_tokens,
    price_modifier_factor,
)
from kiss.core.models.model import (
    CLITextModel,
    ThinkingCallback,
    TokenCallback,
    _iter_balanced_json_objects,
    _iter_jsonl,
    _iter_tool_calls_lists,
    environ_without,
)
from kiss.core.processes import run_process_group

logger = logging.getLogger(__name__)


def _find_claude_cli() -> str:
    """Locate the ``claude`` executable on PATH.

    Returns:
        Absolute path to the ``claude`` binary.

    Raises:
        KISSError: If the ``claude`` CLI is not installed.
    """
    path = shutil.which("claude")
    if path is None:
        raise KISSError(
            "Claude Code CLI ('claude') not found on PATH. "
            "Install it from https://docs.anthropic.com/en/docs/claude-code"
        )
    return path


# Credentials the ``claude`` CLI prefers over the subscription login: with
# either in its environment a ``claude -p`` turn bills API tokens even when
# the user is logged in to a Claude plan (code.claude.com/docs/en/authentication).
_API_KEY_ENV_VARS: tuple[str, ...] = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

SUBSCRIPTION_HINT = (
    "Claude Code is set to bill your Claude subscription (Settings > Claude "
    "Code): run `claude login` on this machine, paste a `claude setup-token` "
    "token into that panel, or turn the switch off to use ANTHROPIC_API_KEY."
)


def claude_subscription_env() -> dict[str, str] | None:
    """Return the environment a ``cc/`` turn runs the CLI in.

    ``None`` (inherit the daemon's environment) when
    ``DEFAULT_CONFIG.claude_code_use_subscription`` is off.  Otherwise a
    copy of ``os.environ`` without :data:`_API_KEY_ENV_VARS`, so the CLI
    falls through to the subscription credentials — ``CLAUDE_CODE_OAUTH_TOKEN``
    (set here from ``DEFAULT_CONFIG`` when the settings panel holds one)
    or the ``claude login`` keychain entry.

    Returns:
        The child environment, or ``None`` to inherit.
    """
    from kiss.core.config import DEFAULT_CONFIG

    if not DEFAULT_CONFIG.claude_code_use_subscription:
        return None
    env = environ_without(_API_KEY_ENV_VARS)
    token = DEFAULT_CONFIG.CLAUDE_CODE_OAUTH_TOKEN
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    return env


def claude_auth_status(timeout: float = 20.0) -> dict[str, Any]:
    """Report how the ``claude`` CLI would authenticate a ``cc/`` turn.

    Runs ``claude auth status`` in :func:`claude_subscription_env` and
    returns its JSON (``loggedIn``, ``authMethod`` — ``claude.ai``,
    ``oauth_token``, ``api_key``, ``none`` …, ``subscriptionType``,
    ``email`` when logged in) plus ``installed`` (the CLI is on PATH),
    ``useSubscription`` (the switch), ``hasToken`` (a
    ``CLAUDE_CODE_OAUTH_TOKEN`` is configured) and, when the CLI could not
    be run or answered no JSON, ``error`` with the reason.  Never raises:
    the settings panel shows whatever this returns.

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
        "useSubscription": bool(DEFAULT_CONFIG.claude_code_use_subscription),
        "hasToken": bool(DEFAULT_CONFIG.CLAUDE_CODE_OAUTH_TOKEN),
    }
    path = shutil.which("claude")
    if path is None:
        status["error"] = "Claude Code CLI ('claude') not found on PATH."
        return status
    status["installed"] = True
    try:
        proc = run_process_group([path, "auth", "status"], timeout, env=claude_subscription_env())
    except (OSError, subprocess.SubprocessError) as e:
        status["error"] = f"`claude auth status` failed: {e}"
        return status
    try:
        parsed = json.loads(proc.stdout)
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict):
        detail = (proc.stderr or proc.stdout).strip()
        status["error"] = (
            f"`claude auth status` (exit {proc.returncode}) returned no JSON: {detail[:300]}"
        )
        return status
    status.update(parsed)
    status["loggedIn"] = bool(parsed.get("loggedIn"))
    status["authMethod"] = str(parsed.get("authMethod") or "none")
    return status


def _iter_stream_json_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield parsed stream-json events, unwrapping ``stream_event`` wrappers.

    Blank lines and lines that are not valid JSON are skipped.

    Args:
        lines: An iterable of JSON strings (one event per line).

    Yields:
        The parsed event dicts, with ``stream_event`` wrappers replaced by
        their inner event.
    """
    for event in _iter_jsonl(lines):
        if event.get("type") == "stream_event":
            event = event.get("event", {})
        yield event


def _find_consecutive_tool_calls_end(content: str) -> int:
    """Return the end position of the last consecutive ``tool_calls`` JSON block.

    Scans *content* for balanced JSON objects containing ``tool_calls``
    lists.  Consecutive blocks (separated only by whitespace) are all
    included.  The sequence stops when non-whitespace text appears between
    blocks or a non-tool-calls JSON object is encountered after the first
    tool_calls block.

    This lets the parser collect **all** tool calls the model outputs
    back-to-back, rather than only the first block.  Blocks separated by
    hallucinated text (e.g. ``(no output)``) are *not* consecutive, so
    only the first group is captured.

    Args:
        content: The accumulated text to scan.

    Returns:
        End position (exclusive) of the last consecutive tool_calls JSON
        object, or ``-1`` if none found.
    """
    last_tc_end = -1
    for start, end, parsed in _iter_balanced_json_objects(content):
        if _iter_tool_calls_lists(parsed):
            if last_tc_end == -1:
                last_tc_end = end
            else:
                between = content[last_tc_end:start]
                if between.strip():
                    break
                last_tc_end = end
        elif last_tc_end != -1:
            break
    return last_tc_end


class _ObjectSpans:
    """Balanced ``{...}`` spans of streamed text, fed one delta at a time.

    Follows the candidate rule of
    :func:`~kiss.core.models.model._iter_balanced_json_objects`: every
    ``{`` outside a JSON string opens a candidate object and the ``}``
    matching it closes one, at any depth.  That scanner retries from the
    next ``{`` when an earlier candidate never closes, so a stray ``{``
    in prose must not hide the tool-call object after it; here the
    stray stays open on the stack while the inner object still closes.
    A candidate whose first non-blank character is neither ``"`` nor
    ``}`` cannot be a JSON object and is dropped at once, so prose
    quotes after a prose brace do not poison the string state.  Quotes
    outside every candidate are ignored, as in that scanner.

    The streaming parser checks only the spans :meth:`feed` reports and
    rescans the whole text only when one of them is a ``tool_calls``
    object: rescanning per delta made a large streamed tool call (a
    ``Write`` of a brace-heavy file) cost minutes of CPU after the CLI
    had finished.
    """

    def __init__(self) -> None:
        self.pos = 0
        self.starts: list[int] = []
        self.fresh = False
        self.in_string = False
        self.escaped = False

    def feed(self, text: str) -> list[tuple[int, int]]:
        """Consume *text*.

        Args:
            text: The next delta of the streamed text.

        Returns:
            ``(start, end_exclusive)`` offsets into the whole text of
            every object that closed inside *text*, innermost first.
        """
        closed: list[tuple[int, int]] = []
        for ch in text:
            if self.fresh and not ch.isspace():
                self.fresh = False
                if ch not in '"}':
                    self.starts.pop()
            if not self.starts:
                if ch == "{":
                    self.starts.append(self.pos)
                    self.fresh = True
            elif self.in_string:
                if self.escaped:
                    self.escaped = False
                elif ch == "\\":
                    self.escaped = True
                elif ch == '"':
                    self.in_string = False
            elif ch == '"':
                self.in_string = True
            elif ch == "{":
                self.starts.append(self.pos)
                self.fresh = True
            elif ch == "}":
                closed.append((self.starts.pop(), self.pos + 1))
            self.pos += 1
        return closed


def _is_tool_calls_object(text: str) -> bool:
    """Return whether *text* is a JSON object carrying a ``tool_calls`` list.

    Args:
        text: One balanced ``{...}`` span of the streamed content.
    """
    try:
        parsed = json.loads(text, strict=False)
    except json.JSONDecodeError:
        return False
    return bool(_iter_tool_calls_lists(parsed))


def _merge_usage(target: dict[str, Any], usage: dict[str, Any], add: bool) -> None:
    """Fold one ``usage`` dict into *target* in place.

    Numeric fields are summed when *add* is true and replaced otherwise;
    the nested ``cache_creation`` dict is merged field by field the same
    way; the price modifiers the API reports (``speed``,
    ``inference_geo``: :data:`~kiss.core.models.anthropic_model.PRICE_MODIFIER_FIELDS`)
    are kept; should the messages of one total disagree (the CLI sends
    every request of a turn with the same ``speed`` and the workspace's
    inference geo, so in practice they do not) the priced value
    (``"fast"``, ``"us"``) wins, so the one estimate made from the summed
    counters never under-bills.  Other non-numeric fields are ignored and
    fields *usage* omits keep their value.  A finished message is summed into the running total.
    A ``message_delta`` snapshot replaces: its counters are *cumulative*
    for the message (https://platform.claude.com/docs/en/build-with-claude/streaming
    #event-types), so each supersedes the previous one while fields a
    snapshot omits (older models report only ``output_tokens``) keep what
    ``message_start`` reported.

    Args:
        target: The running total or the open message's usage, updated in place.
        usage: One ``usage`` dict from the stream.
        add: Sum the counters instead of replacing them.
    """
    for key, value in usage.items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value if add else value
        elif key == "cache_creation" and isinstance(value, dict):
            _merge_usage(target.setdefault("cache_creation", {}), value, add)
        elif key in PRICE_MODIFIER_FIELDS and isinstance(value, str):
            if not add or key not in target or value == PRICE_MODIFIER_FIELDS[key]:
                target[key] = value


class ClaudeCodeModel(CLITextModel):
    """A model that delegates to the Claude Code CLI for LLM completions.

    Model names use the ``cc/`` prefix.  The part after the prefix is passed
    as the ``--model`` flag to the ``claude`` CLI (e.g. ``cc/opus`` →
    ``--model opus``).

    The CLI runs as a bare model (no native tools, KISS's system prompt
    in place of its own; see :meth:`_build_cli_args`), mirroring
    :class:`~kiss.core.models.codex_model.CodexModel`.

    KISS-level tool calling is supported via text-based prompting: tool
    descriptions are injected into the system prompt and the model's text
    output is parsed for JSON ``tool_calls`` blocks.  Embeddings are not
    available.
    """

    _cli_model_name = "ClaudeCodeModel"
    _cli_logger = logger

    def __init__(
        self,
        model_name: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
    ):
        """Initialize a ClaudeCodeModel instance.

        Args:
            model_name: Full model name including ``cc/`` prefix (e.g. ``cc/opus``).
            model_config: Optional configuration. Recognised keys:
                - ``system_instruction`` (str): Passed to the CLI as
                  ``--system-prompt``, replacing Claude Code's own system
                  prompt (see :meth:`_build_cli_args`).
                - ``timeout`` (int): Subprocess timeout in seconds (default 300).
            token_callback: Optional callback invoked with each streamed text token.
            thinking_callback: Optional callback invoked with ``True`` when a
                thinking block starts and ``False`` when it ends.
        """
        super().__init__(
            model_name,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
        self._cli_model = model_name[3:] if model_name.startswith("cc/") else model_name
        self._last_thinking_content: str = ""
        self._stopped_for_tool_calls: bool = False
        # Usage of the finished assistant messages of the current
        # generate() call, kept on the instance so a run that raises
        # mid-stream can still be billed (see take_partial_usage_response),
        # plus the cumulative usage of the message still streaming, plus
        # each finished message's own usage (one per API request) for the
        # per-request long-context tier (see extract_cost_from_response).
        self._partial_usage: dict[str, Any] = {}
        self._open_message_usage: dict[str, Any] = {}
        self._request_usages: list[dict[str, Any]] = []
        # What prices the current turn (see extract_cost_from_response):
        # whether the subscription switch was on and whether an API key
        # pays for it, both decided when the CLI is launched (_cli_env);
        # the API model id the CLI resolved ``--model`` to (``cc/opus``
        # is an alias) and the number of API requests it made, both read
        # from the stream.
        self._use_subscription: bool = False
        self._api_billed: bool = False
        self._api_model: str = ""
        self._request_count: int = 0

    def _build_cli_args(self) -> list[str]:
        """Build the ``claude`` CLI argument list.

        Always uses ``stream-json`` output format so that tokens can be
        streamed incrementally and the process can be terminated early
        (e.g. when a KISS ``tool_calls`` block is detected).

        The CLI runs as a bare model: ``--tools ""`` removes its native
        tools, ``--safe-mode`` and ``--strict-mcp-config`` (with no MCP
        config) drop the user's CLAUDE.md, skills, plugins, hooks and MCP
        servers, and ``--system-prompt`` replaces Claude Code's system
        prompt with KISS's ``system_instruction`` — delivered as genuine
        system text because Claude treats a "system prompt" arriving in
        the user turn as a prompt injection and refuses the plain-text
        ``tool_calls`` protocol.

        Returns:
            List of CLI arguments.
        """
        cli = _find_claude_cli()
        args = [
            cli,
            "--print",
            "--safe-mode",
            "--strict-mcp-config",
            "--tools",
            "",
            "--no-session-persistence",
            "--model",
            self._cli_model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]
        system_instruction = self.system_instruction_text()
        if system_instruction:
            args += ["--system-prompt", system_instruction]
        return args

    def generate(self) -> tuple[str, Any]:
        """Generate a response using the Claude Code CLI.

        Always uses streaming so tokens are delivered incrementally; text
        from every assistant message is accumulated.

        On a tool-bearing turn — one wrapped in
        :class:`~kiss.core.models.model._ToolCallFilteredStream`, which both
        CLI adapters install — the response is truncated at the end of the
        first run of complete ``tool_calls`` JSON blocks and parsing stops,
        so a reasoning model that keeps going cannot hallucinate its own
        tool results into the content.  The CLI is killed at that
        instant; usage for such a turn is aggregated from the per-message
        ``message_delta`` events seen before the stop.

        Returns:
            tuple[str, Any]: (generated_text, parsed_json_response).

        Raises:
            KISSError: If the CLI could not be started or exited with a
                failure status.
            TimeoutError: If the CLI produced no complete turn before the
                deadline.  Retryable, unlike ``KISSError``.
            KeyboardInterrupt: If the user stopped the task mid-turn.
        """
        prompt = self._build_prompt()
        args = self._build_cli_args()
        self._stopped_for_tool_calls = False
        self._partial_usage = {}
        self._open_message_usage = {}
        self._request_usages = []
        self._api_model = ""
        self._request_count = 0
        stop_on_tool_calls = self._tool_bearing_turn

        with self._cli_turn(args, "Claude Code CLI") as proc:
            proc.send_prompt(prompt)
            content, result_json = self._parse_stream_events(
                proc.lines(), stop_on_tool_calls=stop_on_tool_calls
            )
            if not self._stopped_for_tool_calls:
                try:
                    proc.raise_for_exit()
                except KISSError as e:
                    # In -p mode the CLI reports an auth failure as its
                    # *result* on stdout (stderr is empty): surface it,
                    # with the way out when the subscription switch is on.
                    raise self._with_auth_hint(e, result_json) from None

        # The turn completed: its usage is in result_json, so the partial
        # snapshot must not be billed a second time by a later
        # take_partial_usage_response() call.
        self._partial_usage = {}
        self._open_message_usage = {}
        self.conversation.append({"role": "assistant", "content": content})
        return content, result_json

    def _cli_env(self) -> dict[str, str] | None:
        """The child environment: see :func:`claude_subscription_env`.

        Called once per launch, so this is where the turn's billing is
        decided for good: an inherited environment that holds an API
        key variable makes the turn API spend, whatever the switch says
        afterwards.  The same reading of the switch decides whether a
        failure of this turn carries :data:`SUBSCRIPTION_HINT`.
        """
        env = claude_subscription_env()
        self._use_subscription = env is not None
        self._api_billed = env is None and any(os.environ.get(v) for v in _API_KEY_ENV_VARS)
        return env

    def _with_auth_hint(self, error: KISSError, result_json: dict[str, Any]) -> KISSError:
        """Return *error* enriched with the CLI's own failure text.

        The ``result`` event of a failed turn carries ``is_error`` and
        the message the CLI would print (``Not logged in · Please run
        /login``, ``Invalid API key`` …).  That message is appended, and
        when the turn was launched with the subscription switch on so is
        :data:`SUBSCRIPTION_HINT`.

        Args:
            error: The exit-status error from ``raise_for_exit``.
            result_json: The turn's final ``result`` event (may be empty).

        Returns:
            *error* itself when the turn carried no error text, else a
            new ``KISSError``.
        """
        detail = result_json.get("result") if result_json.get("is_error") else ""
        if not isinstance(detail, str) or not detail.strip():
            return error
        message = f"{error}\n{detail.strip()}"
        if self._use_subscription:
            message = f"{message}\n{SUBSCRIPTION_HINT}"
        return KISSError(message)

    def _close_open_message(self) -> None:
        """Fold the message still streaming into the per-call usage aggregate.

        Called at ``message_stop`` and, for a stream cut short before it,
        when the usage is handed out, so an interrupted message is billed
        from its last cumulative snapshot exactly once.
        """
        if self._open_message_usage:
            _merge_usage(self._partial_usage, self._open_message_usage, add=True)
            self._request_usages.append(self._open_message_usage)
            self._open_message_usage = {}

    def take_partial_usage_response(self) -> Any:
        """Return (and consume) usage seen before a failed generation.

        The Claude CLI reports each assistant message's usage via
        ``message_start`` (input and cache counts) and cumulative
        ``message_delta`` snapshots, folded per message into
        ``self._partial_usage`` as the stream is parsed.  When
        ``generate()`` raises mid-stream (stall timeout, parse error), that
        accumulated usage is real spend the terminal ``result`` event never
        got to report; this hands it to the caller in the same shape as a
        normal response.

        Returns:
            ``{"usage": {...}}`` when partial usage was observed, else
            the completed response the base class staged (a turn whose
            post-generation cleanup raised), else ``None``.  The stored
            value is cleared so it is never counted twice.
        """
        self._close_open_message()
        if not self._partial_usage:
            return super().take_partial_usage_response()
        usage = self._partial_usage
        self._partial_usage = {}
        return self._stamp_billing({"usage": usage})

    def _parse_stream_events(
        self,
        lines: Iterable[str],
        stop_on_tool_calls: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Parse stream-json events from a Claude Code run.

        Iterates over newline-delimited JSON events from the Claude CLI.
        Thinking blocks are streamed via the thinking callback, and text
        blocks via the token callback; text from every assistant message
        is accumulated.

        Also handles ``content_block_start`` / ``content_block_delta`` /
        ``content_block_stop`` events emitted by the CLI with
        ``--include-partial-messages``.

        A ``result`` event is used as the authoritative final content.

        When *stop_on_tool_calls* is ``True``, the parser watches for
        complete ``{"tool_calls": [...]}`` JSON blocks in the accumulated
        text content.  As soon as one is found the content is truncated to
        the end of that first block and parsing stops, preventing reasoning
        models from hallucinating tool results in an unbounded stream.

        Args:
            lines: An iterable of JSON strings (one event per line).
            stop_on_tool_calls: When ``True``, stop as soon as a complete
                ``tool_calls`` JSON block is detected in the text content.

        Returns:
            Tuple of ``(content, result_json)`` where *content* is the text
            from the first assistant message and *result_json* is the parsed
            ``result`` event dict (or ``{}`` if none was received).
        """
        content = ""
        thinking_content = ""
        result_json: dict[str, Any] = {}
        saw_content_block = False
        seen_tool_calls_hint = False
        found_tool_calls = False
        last_tc_end = -1
        object_spans = _ObjectSpans()
        # Usage is kept on the instance (``_partial_usage`` for finished
        # messages, ``_open_message_usage`` for the one streaming) so a
        # stream that raises mid-parse still leaves its observed usage
        # billable (a run timing out otherwise erased all of its known
        # spend).

        events = _iter_stream_json_events(lines)
        for event in events:
            event_type = event.get("type")
            self._note_api_model(event)

            if event_type == "assistant":
                msg = event.get("message", {})
                # Once content_block events stream the text, the final
                # assistant message only repeats it.
                for block in [] if saw_content_block else msg.get("content", []):
                    block_type = block.get("type")
                    if block_type == "thinking":
                        thinking_text = block.get("thinking", "")
                        if thinking_text:
                            thinking_content += thinking_text
                            self._emit_as_thinking(thinking_text)
                    elif block_type == "text":
                        text = block.get("text", "")
                        if text:
                            content += text
                            object_spans.feed(text)
                            self._invoke_token_callback(text)
                if stop_on_tool_calls and content:
                    tc_end = _find_consecutive_tool_calls_end(content)
                    if tc_end > 0:
                        content = content[:tc_end]
                        self._stopped_for_tool_calls = True
                        break
            elif event_type == "content_block_start":
                saw_content_block = True
            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                delta_type = delta.get("type", "")
                if delta_type == "thinking_delta":
                    thinking_text = delta.get("thinking", "")
                    if thinking_text:
                        thinking_content += thinking_text
                        self._open_thinking_if_closed()
                        self._invoke_token_callback(thinking_text)
                elif delta_type == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        content += text
                        closed_spans = object_spans.feed(text)
                        self._invoke_token_callback(text)
                        if stop_on_tool_calls:
                            # The key may straddle two deltas: look at the
                            # new text plus the tail it extends.
                            if not seen_tool_calls_hint:
                                window = content[-(len(text) + len("tool_calls")) :]
                                seen_tool_calls_hint = "tool_calls" in window
                            # Outermost span first: inside valid JSON only
                            # the outer object carries ``tool_calls``.
                            if seen_tool_calls_hint and any(
                                _is_tool_calls_object(content[start:end])
                                for start, end in reversed(closed_spans)
                            ):
                                tc_end = _find_consecutive_tool_calls_end(content)
                                if tc_end > 0:
                                    last_tc_end = tc_end
                                    found_tool_calls = True
                            if found_tool_calls:
                                trailing = content[last_tc_end:].strip()
                                if trailing and not trailing.startswith("{"):
                                    content = content[:last_tc_end]
                                    self._stopped_for_tool_calls = True
                                    break
            elif event_type == "content_block_stop":
                # A no-op unless a thinking_delta opened the bracket, which
                # only happens inside the thinking block that stops here.
                self._close_thinking_if_open()
            elif event_type == "message_start":
                # A new assistant message: its input and cache counts are
                # known now; output_tokens is a placeholder the cumulative
                # message_delta snapshots replace.
                self._close_open_message()
                self._request_count += 1
                usage = event.get("message", {}).get("usage")
                if isinstance(usage, dict):
                    _merge_usage(self._open_message_usage, usage, add=False)
            elif event_type == "message_delta":
                usage = event.get("usage")
                if isinstance(usage, dict):
                    _merge_usage(self._open_message_usage, usage, add=False)
                # A stop_reason marks the message's final snapshot, so the
                # message is complete even if its message_stop never
                # arrives (or the stream carries no message_start/stop).
                if (event.get("delta") or {}).get("stop_reason"):
                    self._close_open_message()
            elif event_type == "message_stop":
                self._close_open_message()
            elif event_type == "result":
                result_json = event
                if not found_tool_calls and not content.strip():
                    # Fallback only: the terminal ``result`` carries just
                    # the last assistant message, so it never replaces
                    # text already streamed.
                    content = event.get("result", content)

        if found_tool_calls and not self._stopped_for_tool_calls:
            content = content[:last_tc_end]
            self._stopped_for_tool_calls = True

        if self._stopped_for_tool_calls and not result_json:
            # The turn ended at the KISS tool_calls block, before the
            # terminal ``result`` event, and the child is killed at once;
            # the per-message usage folded above (including the message
            # cut short) keeps the token accounting.
            self._close_open_message()
            if self._partial_usage:
                result_json = {"usage": self._partial_usage}
        if result_json:
            self._stamp_billing(result_json)

        self._last_thinking_content = thinking_content
        return content, result_json

    def _stamp_billing(self, response: dict[str, Any]) -> dict[str, Any]:
        """Record on *response* what prices it, as of this turn.

        ``api_billed`` (an API key paid, decided at launch), ``model``
        (the API model id the stream named, if any) and ``requests``
        (the API requests the CLI made in the turn, counted from
        ``message_start`` and finished messages; at least 1).  The price modifiers the API
        reported per message (``speed``, ``inference_geo``, folded into
        the per-message total by :func:`_merge_usage`) are copied into a
        ``usage`` that lacks them: the CLI's terminal ``result`` sums only
        the counters.  ``request_usages`` lists each finished message's
        own usage, so the per-request long-context tier can be priced.
        A later settings change or turn cannot reprice a response that
        carries these.

        Args:
            response: A response dict with the turn's ``usage``.

        Returns:
            The same dict.
        """
        self._close_open_message()
        response["api_billed"] = self._api_billed
        if self._api_model:
            response["model"] = self._api_model
        # A completed message_delta snapshot without a message_start is a
        # request too.
        response["requests"] = max(1, self._request_count, len(self._request_usages))
        usage = response.get("usage")
        if isinstance(usage, dict):
            for field in PRICE_MODIFIER_FIELDS:
                if field not in usage and field in self._partial_usage:
                    usage[field] = self._partial_usage[field]
        response["request_usages"] = list(self._request_usages)
        return response

    def _note_api_model(self, event: dict[str, Any]) -> None:
        """Remember the API model id a stream event names.

        The CLI resolves ``--model opus`` to a concrete id and reports it
        in the ``system``/``init`` event (``model``), in every
        ``message_start`` and ``assistant`` event (``message.model``) and
        in the terminal ``result``'s ``modelUsage`` map (keyed by id).
        The last one seen wins; the turn's response is stamped with it.
        A ``modelUsage`` naming several models (the CLI ran a helper
        model, e.g. Haiku, beside the main one) does not say which is
        the main one, so it does not replace the stream's model.

        Args:
            event: One parsed ``stream-json`` event.
        """
        event_type = event.get("type")
        model: Any = None
        if event_type == "system":
            model = event.get("model")
        elif event_type in ("assistant", "message_start"):
            model = (event.get("message") or {}).get("model")
        elif event_type == "result":
            model_usage = event.get("modelUsage")
            if isinstance(model_usage, dict) and len(model_usage) == 1:
                model = next(iter(model_usage))
        if isinstance(model, str) and model:
            self._api_model = model

    def extract_cost_from_response(self, response: Any) -> float | None:
        """Price the turn at the API model's rate when the API key pays.

        A turn the Claude subscription paid for (the switch was on at
        launch, or no API key was in the environment) keeps the ``cc/*``
        catalog price of zero (``None`` here).  A turn an inherited
        ``ANTHROPIC_API_KEY`` paid for is API spend at the resolved
        model's list price, cache tiers included, so its usage is priced
        through :func:`calculate_cost` under the API model id the stream
        reported (falling back to the name after ``cc/``); the
        per-request long-context tier is priced per message of a
        multi-request turn (``request_usages``), and the fast-mode and US-only
        inference modifiers the usage reports scale the whole turn
        (:func:`~kiss.core.models.anthropic_model.price_modifier_factor`).
        A model id the catalog does not
        know (new snapshot) is billed at the CLI's own ``total_cost_usd``
        estimate when the result carries one; otherwise
        :func:`calculate_cost` raises.  So is a turn whose ``modelUsage``
        names several models: the CLI priced each model's share at its
        own rate, which one catalog price cannot.

        Args:
            response: The dict returned by :meth:`generate` or
                :meth:`take_partial_usage_response`.

        Returns:
            The turn's USD cost, or ``None`` to bill the catalog entry.
        """
        from kiss.core.models.model_info import (
            _CLAUDE_DATE_SUFFIX_RE,
            calculate_cost,
            is_known_model,
        )

        if not isinstance(response, dict) or not response.get("api_billed"):
            return None
        counts = self.extract_input_output_token_counts_from_response(response)
        model = response.get("model") or self._cli_model
        if not is_known_model(model):
            # A dated snapshot id (``claude-sonnet-4-5-20250929``) prices
            # as its undated catalog entry.
            model = _CLAUDE_DATE_SUFFIX_RE.sub("", model)
        reported = response.get("total_cost_usd")
        model_usage = response.get("modelUsage")
        several = isinstance(model_usage, dict) and len(model_usage) > 1
        if (several or not is_known_model(model)) and isinstance(reported, int | float):
            return float(reported)
        single_request = response.get("requests", 1) <= 1
        cost = calculate_cost(model, *counts, single_request=single_request)
        if not single_request:
            # The summed counters bill at short-context rates; add the
            # long-context premium of each request that crossed the
            # threshold on its own.
            for usage in response.get("request_usages") or []:
                request = self.extract_input_output_token_counts_from_response({"usage": usage})
                cost += calculate_cost(model, *request) - calculate_cost(
                    model, *request, single_request=False
                )
        return cost * price_modifier_factor(response.get("usage") or {}, model)

    def _tool_call_text(self, content: str) -> str:
        """Parse the turn's thinking for ``tool_calls`` as well as its text.

        A reasoning model sometimes emits the block inside its thinking
        and then stops, so the thinking content of the turn just finished
        is appended to the text that is scanned.

        Args:
            content: The turn's assistant text.
        """
        return "\n".join([content, self._last_thinking_content])

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int, int]:
        """Extract token counts from the Claude Code CLI JSON response.

        Args:
            response: The parsed JSON response from the CLI.

        Returns:
            (input_tokens, output_tokens, cache_read_tokens,
            cache_write_5m_tokens, cache_write_1h_tokens).  The last element
            is the Anthropic one-hour cache-write token count.
        """
        if not isinstance(response, dict):
            return 0, 0, 0, 0, 0
        usage = response.get("usage") or {}
        cache_write_5m, cache_write_1h = cache_creation_tokens(usage)
        return (
            usage.get("input_tokens") or 0,
            usage.get("output_tokens") or 0,
            usage.get("cache_read_input_tokens") or 0,
            cache_write_5m,
            cache_write_1h,
        )
