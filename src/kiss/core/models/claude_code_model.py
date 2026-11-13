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
``--system-prompt-file`` carrying KISS's system prompt in place of Claude
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

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from typing import Any

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import (
    PRICE_MODIFIER_FIELDS,
    _attachments_to_blocks,
    _effort_level_for,
    cache_creation_tokens,
    price_modifier_factor,
)
from kiss.core.models.model import (
    Attachment,
    CLITextModel,
    ThinkingCallback,
    TokenCallback,
    _inside_invoke,
    _iter_balanced_json_objects,
    _iter_jsonl,
    _iter_tool_calls_lists,
    _parse_text_based_tool_calls,
    _xml_tool_call_end,
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

# The one setting that outranks ``--effort`` in the CLI's resolution order
# (code.claude.com/docs/en/model-config, "Adjust effort level"): left out
# of the child environment when KISS sends the flag, so the chosen level is
# the one that runs.
_EFFORT_ENV_VAR = "CLAUDE_CODE_EFFORT_LEVEL"

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

    Blank lines, lines that are not valid JSON and lines whose JSON is
    not an object (progress noise such as ``42`` or ``"starting"``) are
    skipped.

    Args:
        lines: An iterable of JSON strings (one event per line).

    Yields:
        The parsed event dicts, with ``stream_event`` wrappers replaced by
        their inner event.
    """
    for event in _iter_jsonl(lines):
        if isinstance(event, dict) and event.get("type") == "stream_event":
            event = event.get("event")
        if isinstance(event, dict):
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


# What may follow an object that closed inside a candidate ``{``: after
# a value, after a ``,`` (a key or a value starts), after a ``-`` and
# inside a number; a literal is spelled out letter by letter.  ``NaN``
# and ``Infinity`` are the decoder's (``json.loads(..., strict=False)``
# reads them), so a candidate holding them stays valid.
_AFTER_VALUE = ",}]"
_JSON_ITEM_START = '"{[-0123456789tfnNI'
_AFTER_MINUS = "0123456789I"
_IN_NUMBER = "0123456789.eE+-,}]"
_JSON_LITERALS = {"t": "rue", "f": "alse", "n": "ull", "N": "aN", "I": "nfinity"}


class _ObjectSpans:
    """Balanced ``{...}`` spans of streamed text, fed one delta at a time.

    An incremental :func:`~kiss.core.models.model._brace_matches`: every
    unescaped ``"`` toggles the string parity, each parity matches the
    braces it sees outside its strings on its own stack, and the span a
    ``}`` closes is reported for the parity that sees it.  A candidate
    ``{`` that a whole-text scan sees inside a prose string (``printf("{")``
    before the tool-call block) is therefore matched on the other stack
    and never hides the object after it, exactly as the whole-text
    scanner of :func:`~kiss.core.models.model._iter_balanced_json_objects`
    pairs it.  A candidate whose first non-blank character is neither
    ``"`` nor ``}`` cannot be a JSON object and is dropped from its stack
    at once (the braces of a valid object all pass that rule, so its own
    pairing is unchanged), which spares a decode per ``{...}`` of a
    brace-heavy file streamed inside a ``Write`` argument.  Likewise the
    candidates still open when an object closes inside them are dropped
    when the text after it is not JSON: after a value only ``,``, ``]``
    or ``}`` may come, after a ``,`` only a key or a value may start, and
    a ``]`` leaves the text after a value again.  So the prose that
    follows a block (``An example {"example": {...}`` or
    ``{"example": [{...}],`` and then the imagined output) ends them as
    candidates (:meth:`is_open`), whichever separators JSON allows come
    first.  Not knowing whether a ``,`` sits in an array or an object,
    the check accepts any value start there: a string, object or array
    ends the check, a literal or number is read to its end (``, no
    output`` breaks at the ``o``, ``, false output`` at the ``o`` after
    the literal) and leaves the text after a value again.

    The streaming parser checks only the spans :meth:`feed` reports and
    rescans the whole text only when one of them is a ``tool_calls``
    object: rescanning per delta made a large streamed tool call (a
    ``Write`` of a brace-heavy file) cost minutes of CPU after the CLI
    had finished.  A span that closed while candidates of its stack
    were still open (``enclosed`` maps its start to them, outermost
    first) is their data for as long as one of them may still parse
    (:func:`_may_still_parse`), exactly as the whole-text scanner
    swallows it once the candidate does: judging it on its own would
    promote an envelope-shaped argument into a call.  A candidate that
    can no longer parse (``Pass {"cmd" and then run:`` before the
    block) hides nothing, as the whole-text scanner skips it.
    """

    def __init__(self) -> None:
        self.pos = 0
        self.stacks: tuple[list[int], list[int]] = ([], [])
        self.fresh: list[int] | None = None
        self.closed_under: list[int] | None = None
        self.allowed = ""
        self.literal = ""
        self.in_string = False
        self.escaped = False
        self.enclosed: dict[int, tuple[int, ...]] = {}

    def is_open(self, start: int) -> bool:
        """Return whether the candidate ``{`` at offset *start* is still open on its stack.

        Args:
            start: The offset of the ``{`` in the whole text.
        """
        return any(start in stack for stack in self.stacks)

    def _continues_candidates(self, ch: str) -> bool:
        """Advance the expectation past *ch*; return whether ``closed_under`` may still parse.

        Called for every character after an object closed inside open
        candidates, until a string, object or array starts (which ends
        the check) or the text stops being JSON (which ends them).

        Args:
            ch: The next character of the streamed text.
        """
        if self.literal:
            if ch != self.literal[0]:
                return False
            self.literal = self.literal[1:]
            if not self.literal:
                self.allowed = _AFTER_VALUE
        elif ch.isspace():
            if self.allowed == _AFTER_MINUS:
                return False
            if self.allowed == _IN_NUMBER:
                self.allowed = _AFTER_VALUE
        elif ch not in self.allowed:
            return False
        elif ch == ",":
            self.allowed = _JSON_ITEM_START
        elif ch == "]":
            self.allowed = _AFTER_VALUE
        elif ch in '"{[}':
            self.closed_under = None
        elif self.allowed != _IN_NUMBER:
            if ch in _JSON_LITERALS:
                self.literal = _JSON_LITERALS[ch]
            else:
                self.allowed = _AFTER_MINUS if ch == "-" else _IN_NUMBER
        return True

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
            if self.fresh is not None and not ch.isspace():
                if ch not in '"}':
                    self.fresh.pop()
                self.fresh = None
            if self.closed_under is not None and not self._continues_candidates(ch):
                self.closed_under.clear()
                self.closed_under = None
            if ch == "\\" and not self.escaped:
                self.escaped = True
            else:
                stack = self.stacks[self.in_string]
                if ch == '"':
                    if not self.escaped:
                        self.in_string = not self.in_string
                elif ch == "{":
                    stack.append(self.pos)
                    self.fresh = stack
                elif ch == "}" and stack:
                    start = stack.pop()
                    if stack:
                        self.enclosed[start] = tuple(stack)
                        self.closed_under = stack
                        self.allowed = _AFTER_VALUE
                        self.literal = ""
                    closed.append((start, self.pos + 1))
                self.escaped = False
            self.pos += 1
        return closed


_CALL_CLOSERS = ("```", "</function_calls>")
"""What may close a tool call: a Markdown fence, Claude's native call wrapper."""


def _is_call_epilogue(tail: str) -> bool:
    """Return whether *tail*, the text streamed after a complete tool call, may still end the turn.

    White space and the closers of the call (:data:`_CALL_CLOSERS`: the
    Markdown fence around it, the ``<function_calls>`` wrapper Claude
    puts around its native calls), the last one possibly still
    streaming.  Anything else is the model writing on past the call.

    Args:
        tail: The text after the call's end.
    """
    rest = tail.strip()
    while rest:
        closer = next((c for c in _CALL_CLOSERS if rest.startswith(c)), None)
        if closer is None:
            return any(c.startswith(rest) for c in _CALL_CLOSERS)
        rest = rest[len(closer):].strip()
    return True


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


def _may_still_parse(content: str, candidates: tuple[int, ...], end: int) -> bool:
    """Return whether one of *candidates* is still a valid JSON prefix at *end*.

    The whole-text scanner (:func:`~kiss.core.models.model._iter_balanced_json_objects`)
    lets a candidate swallow the objects inside it only when it parses,
    and a candidate that fails to decode before an inner object never
    will: the decoder reads the same prefix however the text goes on.
    So the innermost candidate is tried first and the text is cut at
    *end* (just past an inner object's ``}``, outside any string on the
    candidates' parity): a decode that fails at the cut wants more text
    and may still parse, one that fails earlier is broken for good.

    Args:
        content: The streamed text so far.
        candidates: Offsets of the ``{`` still open on the inner
            object's stack when it closed, outermost first.
        end: The offset just past the inner object.
    """
    for start in reversed(candidates):
        try:
            json.loads(content[start:end], strict=False)
        except json.JSONDecodeError as exc:
            if exc.pos >= end - start:
                return True
            continue
        return True
    return False


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


def _fill_missing_usage(usage: dict[str, Any], streamed: dict[str, Any]) -> None:
    """Give *usage* each field of *streamed* that it lacks, in place.

    A field is lacking when it is absent or ``null``; the nested
    ``cache_creation`` split is filled field by field the same way.

    Args:
        usage: The ``usage`` of a terminal ``result`` event.
        streamed: The total :func:`_merge_usage` summed from the stream.
    """
    for key, value in streamed.items():
        if usage.get(key) is None:
            usage[key] = value
        elif key == "cache_creation" and isinstance(usage[key], dict):
            _fill_missing_usage(usage[key], value)


@contextlib.contextmanager
def _system_prompt_file(system_instruction: str | None) -> Iterator[str | None]:
    """Hold *system_instruction* in a private temp file for the duration of the block.

    Args:
        system_instruction: The turn's system instruction, if any.

    Yields:
        The file's path, or ``None`` (and no file) without an instruction.
    """
    if not system_instruction:
        yield None
        return
    fd, path = tempfile.mkstemp(prefix="kiss-claude-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(system_instruction)
        yield path
    finally:
        os.unlink(path)


def _cli_input(prompt: str, attachments: dict[str, Attachment]) -> tuple[list[str], str]:
    """Return the input flags and the stdin text that carry one turn to the CLI.

    A prompt alone is plain text.  With attachments the turn is one
    ``--input-format stream-json`` line: a user message holding each
    attachment's label and its ``image`` / ``document`` block, then the
    prompt.  The CLI forwards a block to the API, with a note of its own
    in place of an image it cannot decode.

    Args:
        prompt: The prompt text.
        attachments: The images and PDFs to hand over, by label.

    Returns:
        The arguments to add to the command line, and the text for stdin.
    """
    if not attachments:
        return [], prompt
    content: list[dict[str, Any]] = []
    for label, attachment in attachments.items():
        content.append({"type": "text", "text": label})
        content.extend(_attachments_to_blocks([attachment]))
    content.append({"type": "text", "text": prompt})
    message = {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }
    return ["--input-format", "stream-json"], json.dumps(message) + "\n"


class ClaudeCodeModel(CLITextModel):
    """A model that delegates to the Claude Code CLI for LLM completions.

    Model names use the ``cc/`` prefix.  The part after the prefix is passed
    as the ``--model`` flag to the ``claude`` CLI (e.g. ``cc/opus`` →
    ``--model opus``); a generated ``-{level}`` alias is stripped first and
    selects the effort instead (``cc/claude-opus-5-5-xhigh`` → ``--model
    claude-opus-5-5 --effort xhigh``, see :meth:`_effort_level`).

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
    _attachment_types = ("image/", "application/pdf")

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
                  ``--system-prompt-file``, replacing Claude Code's own system
                  prompt (see :meth:`_build_cli_args`).
                - ``timeout`` (int): Subprocess timeout in seconds (default 300).
                - ``reasoning_effort`` (str): The effort level to run at,
                  passed as ``--effort``; outranks the catalog's level
                  (see :meth:`_effort_level`).
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
        from kiss.core.models.model_info import _strip_thinking_alias

        self._cli_model = _strip_thinking_alias(model_name).removeprefix("cc/")
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

    def _build_cli_args(self, system_prompt_file: str | None = None) -> list[str]:
        """Build the ``claude`` CLI argument list.

        Always uses ``stream-json`` output format so that tokens can be
        streamed incrementally and the process can be terminated early
        (e.g. when a KISS ``tool_calls`` block is detected).

        The CLI runs as a bare model: ``--tools ""`` removes its native
        tools, ``--safe-mode`` and ``--strict-mcp-config`` (with no MCP
        config) drop the user's CLAUDE.md, skills, plugins, hooks and MCP
        servers, and ``--system-prompt-file`` replaces Claude Code's system
        prompt with KISS's ``system_instruction`` — delivered as genuine
        system text because Claude treats a "system prompt" arriving in
        the user turn as a prompt injection and refuses the plain-text
        ``tool_calls`` protocol.  A file, because the prompt does not fit
        on a command line: Sorcar's alone exceeds the 32,767 characters
        Windows allows.  ``--thinking-display summarized`` keeps the
        thinking text: in ``--print`` stream-json mode the CLI otherwise
        asks the API to omit it, so thinking blocks stream empty.
        ``--effort`` is added when :meth:`_effort_level` selects a level;
        otherwise the CLI runs the model at its own default effort.

        Args:
            system_prompt_file: The file holding the system instruction
                (:func:`_system_prompt_file`), or ``None`` to keep Claude
                Code's own system prompt.

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
            "--thinking-display",
            "summarized",
        ]
        effort = self._effort_level()
        if effort is not None:
            args += ["--effort", effort]
        if system_prompt_file is not None:
            args += ["--system-prompt-file", system_prompt_file]
        return args

    def _effort_level(self) -> str | None:
        """Return the effort level this turn runs at, or ``None`` for the CLI default.

        Resolved by :func:`~kiss.core.models.anthropic_model._effort_level_for`
        exactly as the direct Anthropic adapter resolves
        ``output_config.effort``: an explicit ``reasoning_effort`` in
        ``model_config`` (the key ``KISSAgent._turn_effort`` sets for one
        turn) wins, else the catalog entry's ``thinking`` level (a
        generated ``cc/claude-opus-5-5-xhigh`` alias carries ``xhigh``, a
        base ``cc/claude-opus-5-5`` the level the catalog runs it at).  The
        CLI accepts ``low`` … ``max`` and runs a level the model lacks at
        the highest one below it.  Read at each launch so a per-turn
        override is honoured.

        Returns:
            The level for ``--effort``, or ``None`` when nothing selects one.
        """
        return _effort_level_for(self.model_name, self.model_config)

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
            KISSError: If the CLI could not be started, exited with a
                failure status, ended without its ``result`` event or
                reported an error in it (the last three do not apply to
                a turn cut at its tool call).
            TimeoutError: If the CLI produced no complete turn before the
                deadline.  Retryable, unlike ``KISSError``.
            KeyboardInterrupt: If the user stopped the task mid-turn.
        """
        input_args, stdin_text = _cli_input(*self._prompt_and_attachments())
        self._stopped_for_tool_calls = False
        self._rejected_response = None
        self._partial_usage = {}
        self._open_message_usage = {}
        self._request_usages = []
        self._api_model = ""
        self._request_count = 0
        stop_on_tool_calls = self._tool_bearing_turn

        with (
            _system_prompt_file(self.system_instruction_text()) as system_prompt_file,
            self._cli_turn(
                self._build_cli_args(system_prompt_file) + input_args, "Claude Code CLI"
            ) as proc,
        ):
            proc.send_prompt(stdin_text)
            content, result_json = self._parse_stream_events(
                proc.lines(), stop_on_tool_calls=stop_on_tool_calls
            )
            if not self._stopped_for_tool_calls:
                try:
                    proc.raise_for_exit()
                    # A clean exit (or a CLI still winding down) finished
                    # the turn only with its terminal ``result``: without
                    # one the text so far may be cut short (the same plain
                    # ``KISSError`` the API transports raise for a stream
                    # without its terminal event), and one that reports
                    # an error carries the CLI's message, not an answer.
                    if result_json.get("type") != "result":
                        raise KISSError(
                            "Claude Code CLI ended without a result event, so its "
                            "answer may be incomplete."
                        )
                    if result_json.get("is_error"):
                        raise KISSError(
                            "Claude Code CLI reported an error "
                            f"(result subtype {result_json.get('subtype')!r})."
                        )
                except KISSError as e:
                    if result_json.get("type") == "result":
                        # Billed like a turn that returned: what the
                        # result reports, which _stamp_billing completed
                        # from the stream's messages (so not those again).
                        self._partial_usage = {}
                        self._rejected_response = result_json
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

    def _cli_env(self) -> dict[str, str]:
        """The child environment: see :func:`claude_subscription_env`.

        Called once per launch, so this is where the turn's billing is
        decided for good: an environment that holds an API key variable
        makes the turn API spend, whatever the switch says afterwards.
        With the switch off it is a copy of the daemon's environment
        taken here, and the billing is read from that same copy, so a
        key saved or deleted during the launch cannot split the two.
        The same reading of the switch decides whether a failure of
        this turn carries :data:`SUBSCRIPTION_HINT`.

        A turn launched with ``--effort`` runs without
        :data:`_EFFORT_ENV_VAR`, which would otherwise outrank the flag.
        """
        env = claude_subscription_env()
        self._use_subscription = env is not None
        env = environ_without(()) if env is None else env
        self._api_billed = any(env.get(v) for v in _API_KEY_ENV_VARS)
        if self._effort_level() is not None:
            env.pop(_EFFORT_ENV_VAR, None)
        return env

    def _with_auth_hint(self, error: KISSError, result_json: dict[str, Any]) -> KISSError:
        """Return *error* enriched with the CLI's own failure text.

        The ``result`` event of a failed turn carries ``is_error`` and
        the message the CLI would print (``Not logged in · Please run
        /login``, ``Invalid API key`` …).  That message is appended, and
        when the turn was launched with the subscription switch on so is
        :data:`SUBSCRIPTION_HINT`.

        Args:
            error: What :meth:`generate` found wrong with the turn (its
                exit status, or its ``result``).
            result_json: The turn's final ``result`` event (may be empty).

        Returns:
            *error* itself when the turn carried no error text, else a
            new ``KISSError``.
        """
        detail = result_json.get("result") if result_json.get("is_error") else ""
        if not isinstance(detail, str) or not detail.strip():
            return error
        # ``args[0]``: ``str(error)`` would prefix "KISS Error:" a second time.
        message = f"{error.args[0]}\n{detail.strip()}"
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
            the turn's ``result`` when one was staged (a turn that failed
            after it, or whose post-generation cleanup raised), else
            ``None``.  The stored value is cleared so it is never counted
            twice.
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

        The ``result`` event ends the turn; its text is the content only
        of a turn that streamed none.

        When *stop_on_tool_calls* is ``True``, the parser watches for
        complete ``{"tool_calls": [...]}`` JSON blocks (or a native
        ``<invoke>`` block) in the accumulated text content.  Once the call
        is complete the content is truncated to its end and the rest of the
        stream is read only for the message's final usage
        (``message_delta``, ``result``): the first new content after the
        call (the model imagining its result) stops the parse, preventing
        reasoning models from hallucinating tool results in an unbounded
        stream.

        Args:
            lines: An iterable of JSON strings (one event per line).
            stop_on_tool_calls: When ``True``, stop as soon as a complete
                ``tool_calls`` JSON block is detected in the text content.

        Returns:
            Tuple of ``(content, result_json)`` where *content* is the text
            of every assistant message (cut at the end of the tool call
            when one stopped the parse) and *result_json* is the parsed
            ``result`` event dict, its ``usage`` completed from the
            stream's messages (:meth:`_stamp_billing`); without one,
            ``{"usage": ...}`` summed from the stream's messages (``{}``
            when they reported none).
        """
        content = ""
        thinking_content = ""
        result_json: dict[str, Any] = {}
        saw_content_block = False
        seen_tool_calls_hint = False
        found_tool_calls = False
        last_tc_end = -1
        object_spans = _ObjectSpans()
        # A ``tool_calls`` object closed inside a still-open candidate
        # that may yet parse, to be judged with the whole text once a
        # span closes outside any candidate (the enclosing one, or a
        # later one when the enclosing one did not parse), and the open
        # candidates it closed under: a sibling closing under the same
        # ones is that candidate's data too and is not re-judged, so an
        # argument holding thousands of envelope-shaped records costs
        # one decode of the candidate, not one per record.
        pending_enclosed = False
        pending_under: tuple[int, ...] = ()
        # Set once the content is cut at a complete tool call: the CLI
        # sends the message's final usage (``message_delta``, ``result``)
        # after the call's text, so the stream is read on for it and
        # ends at the first new content (``drain_tail`` holds the text
        # streamed after the cut).
        draining = False
        drain_tail = ""
        # Usage is kept on the instance (``_partial_usage`` for finished
        # messages, ``_open_message_usage`` for the one streaming) so a
        # stream that raises mid-parse still leaves its observed usage
        # billable (a run timing out otherwise erased all of its known
        # spend).

        events = _iter_stream_json_events(lines)
        for event in events:
            event_type = event.get("type")
            self._note_api_model(event)
            if draining and (
                event_type in ("content_block_start", "message_start")
                or (event_type == "assistant" and not saw_content_block)
            ):
                # New content after the call: the model imagining its result.
                break

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
                if stop_on_tool_calls and content and not draining:
                    # The deltas already located the blocks when they streamed.
                    tc_end = _xml_tool_call_end(content)
                    if tc_end <= 0:
                        tc_end = (
                            last_tc_end
                            if found_tool_calls
                            else _find_consecutive_tool_calls_end(content)
                        )
                    if tc_end > 0:
                        # The text block is complete (the CLI sends this
                        # message before the block's stop): read on for
                        # the final usage.
                        content = content[:tc_end]
                        self._stopped_for_tool_calls = True
                        draining = True
            elif event_type == "content_block_start":
                saw_content_block = True
            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                delta_type = delta.get("type", "")
                if draining:
                    # Past the call: only white space (or the closing
                    # tag of Claude's call wrapper) may follow it.
                    drain_tail += delta.get("text", "") if delta_type == "text_delta" else "\0"
                    if not _is_call_epilogue(drain_tail):
                        break
                    continue
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
                            # the outer object carries ``tool_calls``.  A
                            # span still inside an open candidate that may
                            # yet parse is not judged on its own (see
                            # ``_ObjectSpans``): the rescan of an incomplete
                            # outer object would take an envelope-shaped
                            # argument for the call and cut the content
                            # there.  One inside candidates that can no
                            # longer parse (a prose brace) is judged now.
                            rescan = False
                            spans = reversed(closed_spans) if seen_tool_calls_hint else ()
                            for start, end in spans:
                                enclosing = object_spans.enclosed.get(start)
                                if enclosing is None:
                                    rescan = pending_enclosed or _is_tool_calls_object(
                                        content[start:end]
                                    )
                                elif enclosing != pending_under and _is_tool_calls_object(
                                    content[start:end]
                                ):
                                    pending_enclosed = _may_still_parse(content, enclosing, end)
                                    pending_under = enclosing
                                    rescan = not pending_enclosed
                                if rescan:
                                    break
                            # Prose after the pending object ended its
                            # candidates, the outermost included (see
                            # _ObjectSpans; a candidate that closed instead
                            # was judged above): it is the call.
                            if (
                                pending_enclosed
                                and pending_under
                                and not object_spans.is_open(pending_under[0])
                            ):
                                rescan = True
                            if rescan:
                                pending_enclosed = False
                                pending_under = ()
                                tc_end = _find_consecutive_tool_calls_end(content)
                                if tc_end > 0:
                                    last_tc_end = tc_end
                                    found_tool_calls = True
                            # Claude's native call syntax (see
                            # _parse_text_based_tool_calls): the first
                            # complete block is the call unless a JSON call
                            # came before it; what follows it is the model
                            # imagining the result.
                            if "</invoke>" in content[-(len(text) + len("</invoke>")) :]:
                                xml_end = _xml_tool_call_end(content)
                                if xml_end > 0:
                                    drain_tail = content[xml_end:]
                                    content = content[:xml_end]
                                    self._stopped_for_tool_calls = True
                                    if not _is_call_epilogue(drain_tail):
                                        break
                                    # The call is complete: read on for
                                    # the message's final usage.
                                    draining = True
                                    continue
                            if found_tool_calls and not _inside_invoke(content):
                                trailing = content[last_tc_end:].strip()
                                if trailing and not trailing.startswith("{"):
                                    drain_tail = content[last_tc_end:]
                                    content = content[:last_tc_end]
                                    self._stopped_for_tool_calls = True
                                    if not _is_call_epilogue(drain_tail):
                                        break
                                    # The call's closing fence: read on
                                    # for the message's final usage only.
                                    draining = True
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
                    # text already streamed.  It may hold no text at all
                    # (an error subtype, a JSON ``null``).
                    result_text = event.get("result")
                    if isinstance(result_text, str):
                        content = result_text

        if found_tool_calls and not self._stopped_for_tool_calls:
            content = content[:last_tc_end]
            self._stopped_for_tool_calls = True

        # The terminal ``result`` does not always report the turn's usage:
        # the turn ended at the KISS tool_calls block and the child was
        # killed at once, the stream ended without one (the CLI exited, or
        # the line did not decode), or its ``usage`` is missing or partial.
        # The per-message usage folded above (including a message cut
        # short) then keeps the token accounting (see _stamp_billing).
        self._close_open_message()
        if result_json or self._partial_usage:
            self._stamp_billing(result_json)

        self._last_thinking_content = thinking_content
        return content, result_json

    def _stamp_billing(self, response: dict[str, Any]) -> dict[str, Any]:
        """Record on *response* what prices it, as of this turn.

        ``api_billed`` (an API key paid, decided at launch), ``model``
        (the API model id the stream named, if any) and ``requests``
        (the API requests the CLI made in the turn, counted from
        ``message_start`` and finished messages; at least 1).  Each field
        the stream's messages reported (the counters and the price
        modifiers ``speed`` and ``inference_geo``, summed per message by
        :func:`_merge_usage`) is copied into a ``usage`` that lacks it,
        so a ``result`` whose usage is missing, empty or partial still
        bills what was observed.  A field the ``result`` does report
        stands: it sums the counters and names only the LAST message's
        modifiers (its usage reducer copies ``speed`` / ``inference_geo``
        from the newest message), which say nothing about the earlier
        messages.
        ``request_usages`` lists each finished message's own usage, so
        the per-request long-context tier and modifiers can be priced.
        A later settings change or turn cannot reprice a response that
        carries these.

        Args:
            response: The turn's terminal ``result`` event, or a response
                dict holding what ``usage`` is known of the turn.

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
        if self._partial_usage:
            usage = response.get("usage")
            if not isinstance(usage, dict):
                usage = response["usage"] = {}
            _fill_missing_usage(usage, self._partial_usage)
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
        per-request long-context tier and the fast-mode and US-only
        inference modifiers
        (:func:`~kiss.core.models.anthropic_model.price_modifier_factor`)
        are priced per message of a multi-request turn
        (``request_usages``), each message with the modifiers its own
        usage reports.  A model id the catalog does not
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
        turn_usage = response.get("usage") or {}
        request_usages = [
            usage for usage in response.get("request_usages") or [] if isinstance(usage, dict)
        ]
        if single_request or not request_usages:
            return calculate_cost(
                model, *counts, single_request=single_request
            ) * price_modifier_factor(turn_usage, model)
        # A multi-request turn: each listed message is priced on its own,
        # with its own long-context tier and its own speed / inference-geo
        # modifiers (fast mode falls back to standard speed AND standard
        # pricing when its rate limit is hit, so one turn can mix the
        # two: https://code.claude.com/docs/en/fast-mode#handle-rate-limits).
        # A message whose usage lacks a modifier field inherits the
        # turn's, which the CLI's result copies from the LAST message and
        # the stream's merge takes from the priced one; the turn's never
        # outranks a message's own.  Whatever the summed counters hold
        # beyond the listed messages (a message the stream never closed)
        # bills at short-context rates with the turn's modifiers.
        cost = 0.0
        listed = [0] * len(counts)
        for usage in request_usages:
            request = self.extract_input_output_token_counts_from_response({"usage": usage})
            modifiers = {
                field: usage.get(field, turn_usage.get(field)) for field in PRICE_MODIFIER_FIELDS
            }
            cost += calculate_cost(model, *request) * price_modifier_factor(modifiers, model)
            listed = [have + more for have, more in zip(listed, request, strict=True)]
        rest = [max(0, total - have) for total, have in zip(counts, listed, strict=True)]
        if any(rest):
            cost += calculate_cost(model, *rest, single_request=False) * price_modifier_factor(
                turn_usage, model
            )
        return cost

    def context_tokens_from_response(self, response: Any) -> int | None:
        """Return the size of the turn's largest API request.

        The turn's ``usage`` sums every request the CLI made; the context
        the model works in is one request's prompt plus its output, and
        the largest request holds the latest state of the conversation.

        Args:
            response: The dict returned by :meth:`generate` or
                :meth:`take_partial_usage_response`.

        Returns:
            The largest per-request token total in ``request_usages``, or
            ``None`` when the response lists no request.
        """
        usages = response.get("request_usages") if isinstance(response, dict) else None
        return max(
            (
                sum(self.extract_input_output_token_counts_from_response({"usage": usage}))
                for usage in usages or []
            ),
            default=None,
        )

    def _parse_tool_calls(self, content: str) -> list[dict[str, Any]]:
        """Return the turn's text calls, or its thinking's when the text calls no tool.

        A reasoning model sometimes emits the ``tool_calls`` block inside
        its thinking and then stops, so the thinking of the turn just
        finished is parsed when the text carries no call.  A call in the
        text wins: a call drafted while reasoning and then revised must
        not run beside the one the model emitted.

        Args:
            content: The turn's assistant text.
        """
        return _parse_text_based_tool_calls(content) or _parse_text_based_tool_calls(
            self._last_thinking_content
        )

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
