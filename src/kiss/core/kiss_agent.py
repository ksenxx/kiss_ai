# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Core KISS agent implementation with native function calling support."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, NamedTuple

from kiss.core import stop_signal
from kiss.core.base import Base
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.context_compaction import (
    ARGUMENT_PREVIEW_CHARS,
    CHARS_PER_TOKEN,
    COMPACTION_MIN_STEP_GAP,
    CompactionTable,
    Replacement,
    apply_compaction,
    apply_turn_removal,
    clear_thresholds,
    compaction_target_tokens,
    dropped_chars,
    expand_stubs,
    plan_compaction,
    plan_turn_removal,
    reasoning_tokens,
    should_compact,
)
from kiss.core.event_sink import bind_printer
from kiss.core.kiss_error import (
    BudgetExceededError,
    ContentFilterError,
    ContextWindowExceededError,
    KISSError,
    ModelRefusalError,
    OutputCapError,
    StepLimitExceededError,
)
from kiss.core.llm_reply import LLMReply, zero_usage
from kiss.core.locked import locked_call
from kiss.core.models.model import (
    Attachment,
    Model,
    hand_off_conversation,
    parse_binary_attachments,
    plain_io,
    render_tool_call,
    strip_system_cache_break,
)
from kiss.core.models.model_families import fallback_of_another_family, same_family
from kiss.core.models.model_info import (
    MODEL_INFO,
    calculate_cost,
    declared_fallback,
    get_cache_discount,
    get_cache_write_ratio,
    get_fallback_model,
    get_max_context_length,
    is_known_model,
    model,
)
from kiss.core.models.stream_abort import is_stall_error, stop_error
from kiss.core.printer import truncate_result
from kiss.core.prompt_cache_keepalive import (
    PromptCacheKeepAlive,
    is_long_running_call,
    keep_alive_max_pings,
)
from kiss.core.secret_redaction import redact_secrets
from kiss.core.tool_interrupt import (
    USER_INTERRUPTED_MESSAGE,
    ToolCallInterrupted,
    ToolCallToken,
    drain_pending_interrupt,
    end_tool_call,
    interrupt_tool_call,
    new_tool_call,
    raise_if_interrupted,
    register_tool_call,
    unregister_tool_call,
)
from kiss.core.tool_verdict import (
    ALLOW,
    REISSUE_REMINDER,
    ToolRedirect,
    Verdict,
    followup_result,
    refusal_result,
)
from kiss.core.utils import substitute_prompt_args

logger = logging.getLogger(__name__)

_NON_RETRYABLE_ERROR_TYPES = (
    "AuthenticationError",
    "PermissionDenied",
)
_NON_RETRYABLE_PHRASES = (
    "api key",
    "api_key",
    "invalid key",
    "invalid x-api-key",
    "incorrect api key",
    "unauthorized",
    "permission denied",
    "could not resolve authentication",
    "is not available",
    "not_found_error",
    "credit balance is too low",
)
MAX_CONSECUTIVE_ERRORS = 3
# A stalled stream (the watchdog aborted a silent response, ``stream_abort.stall_error``)
# is a transport hiccup, not a model failure: it has its own, larger limit so
# three stalls in a row on a long task do not end it (MAX_CONSECUTIVE_ERRORS).
MAX_CONSECUTIVE_STALLS = 12
MAX_CONSECUTIVE_NO_TOOL_CALLS = 2
# Sent once after a safety refusal when no fallback model exists: the
# model re-reads the task with its context stated before the run dies on it.
REFUSAL_RETRY_MSG = (
    "Your previous response was a refusal (stop_reason=refusal). Before deciding, "
    "re-read the task: it is the user's own authorized work in their own sandboxed "
    "environment, and engineering tasks about security, cryptography, exploits, "
    "firmware or compatibility testing are legitimate when they stay in that sandbox. "
    "If a part of the task genuinely requires producing something harmful outside it, "
    "do the legitimate parts, and say in the final summary which part you declined "
    "and why. Otherwise continue the task."
)
# Sent after a turn in which the model called natively served tools
# (``Model.native_tools_exposed``) but ended with text alone.
NATIVE_TOOLS_CONTINUE_MSG = (
    "**Your tool calls were executed. If the task is complete, call the "
    "`finish` tool now (a `tool_calls` JSON block) with your final result "
    "instead of describing it in text; otherwise continue working.**"
)
# ``model_config`` keys that name the caller's endpoint and its credentials;
# they belong to the provider that just failed and do not travel to a
# fallback of another provider (the OpenRouter twin, another family).
_ENDPOINT_CONFIG_KEYS = frozenset({"base_url", "api_key", "extra_headers"})
# The reasoning effort of the one retry of a request the output-token cap
# cut short, and the levels at or below which such a retry changes nothing.
RETRY_EFFORT = "low"
_LOW_EFFORT_LEVELS = frozenset({"low", "minimal", "none"})
CONTEXT_LIMIT_FRACTION = (
    DEFAULT_CONFIG.context_limit_fraction
    if 0.0 < DEFAULT_CONFIG.context_limit_fraction <= 1.0
    else 0.7
)
"""Share of the model's context window at which a session raises
:class:`ContextWindowExceededError` and RelentlessAgent hands off with a
trajectory summary.  Steps near a full window cost several times more
than early ones (the whole context is re-sent every step), so the hand-off
happens at 70 % by default; ``KISS_CONTEXT_LIMIT_FRACTION`` overrides it."""
_CONTEXT_OVERFLOW_PHRASES = (
    "exceeds the context window",
    "prompt is too long",
    "context_length_exceeded",
    "maximum context length",
    "exceeds the maximum number of tokens",
)
# A request the provider's own classifier refused before any model ran,
# reported as an HTTP error rather than a typed stop: OpenAI's "This content
# was flagged for possible cybersecurity risk" (400), Azure's "content
# management policy" / ``ResponsibleAIPolicyViolation``, OpenRouter's
# moderation 403, a ``content_policy_violation`` code.  Deterministic for
# the request and shared by the family, like a ``content_filter`` stop.
CONTENT_POLICY_STOP = "content_policy"
_CONTENT_POLICY_PHRASES = (
    "content was flagged",
    "flagged for possible",
    "content_policy",
    "content policy",
    "content management policy",
    "content_filter",
    "content filter",
    "usage policy",
    "usage policies",
    "responsibleaipolicyviolation",
)


class _UsageTotals(NamedTuple):
    """One immutable ``(budget_used, total_tokens_used, step_count)`` triple.

    A :class:`KISSAgent` publishes its whole usage state as ONE of
    these records with a single ``STORE_ATTR`` — atomic under both
    thread interleaving and asynchronously injected exceptions
    (``PyThreadState_SetAsyncExc``, the server's stop watchdog).  The
    legacy scalar counters are properties derived from the current
    record, so no observer — a concurrent live-usage monitor, the
    parent :class:`RelentlessAgent`'s session bank in its
    ``except BaseException`` recovery path — can ever see the tokens
    of a model response without its cost (or vice versa): a stop that
    lands anywhere inside a response-accounting update either leaves
    the previous complete triple or the next complete triple.

    ``helper_mark`` is the model's cumulative helper spend
    (``Model.helper_usage``) already included in the totals, so
    ``KISSAgent._bill_helper_usage`` bills each helper dollar exactly
    once with the same single store, wherever a stop lands.
    """

    budget_used: float
    total_tokens_used: int
    step_count: int
    helper_mark: tuple[object, float, int] = (None, 0.0, 0)


_ZERO_USAGE = _UsageTotals(0.0, 0, 0)


def _helper_usage_of(model: object) -> tuple[object, float, int] | None:
    """Return *model*'s cumulative helper spend (``Model.helper_usage``).

    ``None`` for no model or a duck-typed stand-in without a real record.
    """
    usage = getattr(model, "helper_usage", None)
    record = usage() if callable(usage) else None
    if isinstance(record, tuple) and len(record) == 3:
        return record
    return None


class NativeTurn:
    """The record of one turn's natively run tool calls.

    Codex calls KISS tools over the MCP bridge of
    :meth:`KISSAgent._native_tools_exposed`, on the bridge's threads,
    while the agent's thread waits for the turn to end.  Each turn gets
    a fresh record from :meth:`KISSAgent._execute_step`, and the bridge's
    handlers write to the record of the turn they were made for: a
    handler that outlives its turn (a Stop abandoned a tool deaf to the
    interrupt) can then only ever write to a record the agent has
    already replaced, never to the running turn's.

    Attributes:
        calls: Tool calls served during the turn.
        error: The fatal error a call raised (a limit hit, a sub-agent's
            budget exceeded), kept for the agent thread to raise once the
            turn ends; every further call of the turn is refused.
        ended: The run-ending verdict (:func:`~kiss.core.tool_verdict.end_run`)
            the ``tool_call_hook`` gave a call of the turn, kept for the
            agent thread to end the run with once the turn ends; every
            further call of the turn is refused.
        closed: Set once the turn's bridge has closed; a call that
            returns after that leaves the agent's conversation alone.
        printer: The agent's printer when the turn began: the sink of
            every event of the turn's calls (``tool_call``,
            ``tool_result``), so a call that returns after a Stop
            closed the turn and a reset gave the agent the next run's
            printer still reports to the run it was made in.
        lock: Held to close the turn, to admit a call once its hook
            has spoken (:meth:`KISSAgent._admit_call`), to publish a
            call's result (:meth:`KISSAgent._publish_native_call`; a
            call nested in one of the turn's calls settles under it too,
            :meth:`KISSAgent._settle_native_call`) and for each step a
            call takes on the agent around its other hooks
            (:meth:`KISSAgent._turn_step`), so the check of
            ``closed`` and the writes it guards are one step: a handler
            past the check when the Stop closes the turn publishes
            first, never after, when the agent may be running another
            turn or another run.  The hooks themselves run outside it
            (they may block, and the Stop must still close the turn), so
            the closing wait covers in-memory writes only.
        filing: The keys of the helper calls that calls of the turn are
            handing to the ``helper_charge_hook`` right now, outside
            ``lock`` (:meth:`KISSAgent._take_helper_call`): a call the
            Stop abandoned may still be in that hook when the run ends.
        running: The calls of the turn whose tools are running, by call
            id (:meth:`KISSAgent._tools_for_call`): closing the turn
            interrupts them.  The Stop that ends a turn interrupts the
            tools it finds running and then gives up; a call whose hook
            allowed it only after that starts its tool unseen.
    """

    def __init__(self, printer: Printer | None = None) -> None:
        self.calls = 0
        self.error: Exception | None = None
        self.ended: Verdict | None = None
        self.closed = threading.Event()
        self.printer = printer
        self.lock = threading.RLock()
        self.filing: set[str] = set()
        self.running: dict[int, ToolCallToken] = {}


class _EmptyModelResponseError(KISSError):
    """Raised when a model repeatedly returns no text and no tool calls.

    This is a recoverable model/adapter failure when the active model has a
    registered fallback: unlike normal prompt/validation ``KISSError``s, the
    agent loop can swap to the fallback model and continue the same task.
    """


def _failure_reason(error: KISSError) -> str:
    """Describe what the model returned, for the fail-over warning.

    Args:
        error: A :class:`ContentFilterError`, :class:`ModelRefusalError`,
            :class:`OutputCapError` or :class:`_EmptyModelResponseError`.

    Returns:
        A phrase naming the provider's stop reason, e.g. ``a content
        filter stop (stop_reason="content_filter")``.
    """
    if isinstance(error, ContentFilterError) and error.stop_reason == CONTENT_POLICY_STOP:
        return "a content policy refusal of the request"
    if isinstance(error, ContentFilterError):
        return f'a content filter stop (stop_reason="{error.stop_reason}")'
    if isinstance(error, ModelRefusalError):
        return f'a safety refusal (stop_reason="{error.stop_reason}")'
    if isinstance(error, OutputCapError):
        return f'an output cap cut-off (stop_reason="{error.stop_reason}")'
    return "repeated empty responses"


ROUTINE_TOOLS = frozenset(
    {
        "summary",
        "memory_search",
        "memory_pull",
        "memory_read",
        "memory_list",
        "memory_write",
        "memory_refresh",
    }
)
"""Tools after which the next model turn is routine: the model acknowledges a
summary or a memory operation rather than plans, debugs or reviews."""
WAIT_TOOLS = frozenset({"bash_job", "agent_job"})
_MISSING = object()


def is_routine_turn(name: str, args: dict[str, Any], result: str, refused: bool) -> bool:
    """Return whether the model turn after this tool call is a routine one.

    Routine turns are the one after a refused call, a
    :data:`ROUTINE_TOOLS` call, a ``bash_job`` / ``agent_job`` ``wait``
    that ended with the job still running (the model only waits again),
    or a ``Read`` whose window was unchanged and already in the context.
    Planning, debugging and review turns are never routine: a wait that
    returned a finished job's result
    (a test run, a reviewer's findings) is one of those.

    Args:
        name: The tool that was called.
        args: Its arguments.
        result: Its result text (or the refusal text).
        refused: Whether a hook or guard refused the call.
    """
    if refused or name in ROUTINE_TOOLS:
        return True
    if name in WAIT_TOOLS and str(args.get("action", "")) in ("wait", "wait_any"):
        # Both tools open with the job's status line (``Job <id>: running
        # (pid …)``, ``The … task is still running after …``, ``None of …
        # finished within …; they are still running``); the log tail
        # after it may say anything.
        status_line = result.lstrip().split("\n", 1)[0]
        return "running (pid" in status_line or "still running" in status_line
    return name == "Read" and "already in your context" in result


REASONING_TURN_NOTE = (
    "Harness note at step {step}: a new reasoning turn starts here. The reasoning of "
    "turns before the last {keep} is cleared from the context, so keep what you still "
    "need from it in your files or in your next message."
)
"""The user note that starts a reasoning turn every ``Config.reasoning_turn_steps`` steps."""

UNANSWERED_CALL_NOTE = (
    "No result: the harness failed during this step ({error}) before this call "
    "completed. The call may not have run: check its effect before you repeat it."
)
"""The result of a tool call left without one by a step that failed after the model's reply."""

ATTACHMENTS_LEFT_OUT_NOTE = (
    "[Harness: handing this step's results to you failed ({error}), so the {count} "
    "attachment(s) of this result were left out; its text is above.]"
)
"""Closes a tool result handed over as text after the hand-over of its step's results failed."""


def _as_text(result: dict[str, Any], error: Exception) -> dict[str, Any]:
    """Return a tool *result* without its binary attachments, saying so when it had any.

    Args:
        result: ``{"result": text}`` of one tool call.
        error: Why the result is handed over as text.
    """
    text, attachments = parse_binary_attachments(str(result["result"]), transcode=False)
    if attachments:
        text += "\n\n" + ATTACHMENTS_LEFT_OUT_NOTE.format(
            error=f"{type(error).__name__}: {error}", count=len(attachments)
        )
    return {"result": text}


def _call_args(function_call: dict[str, Any]) -> dict[str, Any]:
    """Return the arguments dict of a function call, or {} if absent/malformed."""
    raw_args = function_call.get("arguments")
    return raw_args if isinstance(raw_args, dict) else {}


def _call_in_context[T](
    context: contextvars.Context, call: Callable[..., T], /, *args: Any, **kwargs: Any
) -> T:
    """Call ``call(*args, **kwargs)`` under a copy of *context*, on whatever thread this runs.

    A thread the framework starts (a bridge's request handler) begins
    with an empty context, so the agent thread's context variables (a
    :func:`~kiss.core.base.collecting_trajectories` scope) are copied
    in; a fresh copy per call, since a context may be entered by one
    thread at a time.
    """
    return context.copy().run(call, *args, **kwargs)


def _unless_closed[T](turn: NativeTurn, step: Callable[..., T], /, *args: Any) -> T | None:
    """Call ``step(*args)`` unless *turn* is closed (``None`` then); called under ``turn.lock``."""
    return None if turn.closed.is_set() else step(*args)


def unknown_argument_hint(
    sig: inspect.Signature | None,
    args: dict[str, Any],
    tool: Any = None,
) -> str:
    """Return a hint for the arguments of a tool call that its signature does not take.

    A tool that knows its own vocabulary says so through an
    ``unknown_arguments_hint`` attribute, a function from the unknown
    ``{name: value}`` pairs to the hint text (``run_agent`` and
    ``run_parallel`` tell an options key from a renamed, removed or
    misspelled one).  Otherwise a tool with an ``options`` parameter
    takes its run settings as one JSON object, so a setting passed as
    a keyword (``use_memory=False``) is pointed there; the agent
    cannot know the accepted keys, so that hint is conditional: the
    tool itself rejects a key that is not a setting.

    Args:
        sig: The tool's signature, ``None`` when unknown.
        args: The arguments of the call.
        tool: The tool callable, for its ``unknown_arguments_hint``.

    Returns:
        The hint (starting with a newline), or ``""`` when every
        argument is a parameter or the tool takes no ``options``.
    """
    if sig is None:
        return ""
    unknown = {name: args[name] for name in args if name not in sig.parameters}
    if not unknown:
        return ""
    own_hint = getattr(tool, "unknown_arguments_hint", None)
    if callable(own_hint):
        return "\n" + str(own_hint(unknown))
    if "options" not in sig.parameters:
        return ""
    example = json.dumps(unknown)
    return (
        f"\n{', '.join(unknown)} is not an argument; if it is a run setting, pass it in the "
        f"`options` JSON object, e.g. options='{example}' (the accepted keys are listed "
        f"under `options`)."
    )


def _argument_excerpts(args: dict[str, Any]) -> str:
    """Render the arguments of a call as ``str(args)`` does, a long one by its first characters.

    What the result of a failed call shows of the call.  The model wrote
    it, so the first :data:`ARGUMENT_PREVIEW_CHARS` characters of an
    argument (what a compaction stub keeps of one) are enough to
    recognise it; the whole would put the argument into the context a
    second time.

    Args:
        args: The arguments of the call.

    Returns:
        ``{name: value, ...}`` with the ``repr`` of each value; one longer
        than the preview is cut to it and followed by ``…[N more chars]``.
    """
    shown = []
    for name, value in args.items():
        text = repr(value)
        if len(text) > ARGUMENT_PREVIEW_CHARS:
            more = len(text) - ARGUMENT_PREVIEW_CHARS
            text = f"{text[:ARGUMENT_PREVIEW_CHARS]} …[{more:,} more chars]"
        shown.append(f"{name!r}: {text}")
    return "{" + ", ".join(shown) + "}"


def _is_retryable_error(e: Exception) -> bool:
    error_type = type(e).__name__
    if any(pattern in error_type for pattern in _NON_RETRYABLE_ERROR_TYPES):
        return False
    error_msg = str(e).lower()
    if any(phrase in error_msg for phrase in _NON_RETRYABLE_PHRASES):
        return False
    return True


def _is_context_overflow_error(e: Exception) -> bool:
    """Return True when the provider rejected a request for exceeding the context window."""
    error_msg = str(e).lower()
    return any(phrase in error_msg for phrase in _CONTEXT_OVERFLOW_PHRASES)


def is_content_policy_error(e: Exception) -> bool:
    """Return whether *e* is a provider error refusing the request on its content policy.

    Such a refusal (:data:`_CONTENT_POLICY_PHRASES`) arrives as a plain
    HTTP error of the SDK, not as a typed stop, yet retrying it on the
    same family is as pointless as retrying a ``content_filter`` stop:
    the 2026-11-01 audit lost three sub-agents to "3 consecutive
    errors" on OpenAI's cybersecurity flag before failing over by hand.

    Args:
        e: The exception the model call raised.
    """
    error_msg = str(e).lower()
    return any(phrase in error_msg for phrase in _CONTENT_POLICY_PHRASES)


def _with_usage_footer(text: str, usage_info: str) -> str:
    """Return *text* with the turn's usage line appended as a fenced ``text`` block."""
    return text + "\n```text\n" + usage_info + "\n```\n"


def stop_reason_of(response: Any) -> str | None:
    """Why the provider ended *response*: its stop reason or finish reason.

    Anthropic carries ``stop_reason``; chat completions ``choices[0].finish_reason``; Gemini
    ``candidates[0].finish_reason``, recorded by its wire name
    (``STOP``, ``MAX_TOKENS``); the Responses API ``status``
    (``completed``) or ``incomplete_details.reason``
    (``max_output_tokens``).  ``None`` when the response says nothing.
    """
    reason = getattr(response, "stop_reason", None)
    if isinstance(reason, str):
        return reason
    for field in ("choices", "candidates"):
        entries = getattr(response, field, None)
        if isinstance(entries, list) and entries:
            reason = getattr(entries[0], "finish_reason", None)
            reason = getattr(reason, "value", reason)  # Gemini's FinishReason enum
            if isinstance(reason, str):
                return reason
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    if isinstance(reason, str):
        return reason
    status = getattr(response, "status", None)
    return status if isinstance(status, str) else None


if TYPE_CHECKING:  # pragma: no cover
    from kiss.core.printer import Printer


class KISSAgent(Base):
    """A KISS agent using native function calling."""

    # Class-level default so the derived counter properties below work
    # from the first ``Base.__init__`` store onwards (each setter then
    # publishes an instance-level replacement snapshot in one
    # ``STORE_ATTR``).
    _usage_totals: _UsageTotals = _ZERO_USAGE

    @property
    def budget_used(self) -> float:
        """Cumulative USD spend, derived from the atomic usage snapshot."""
        return self._usage_totals.budget_used

    @budget_used.setter
    def budget_used(  # pyright: ignore[reportIncompatibleVariableOverride]
        self,
        value: float,
    ) -> None:
        # Publishing a whole replacement record keeps the triple
        # coherent for concurrent readers; single-field writers are
        # same-thread only (this agent's run loop, pre-run zeroing).
        self._usage_totals = self._usage_totals._replace(budget_used=float(value))

    @property
    def total_tokens_used(self) -> int:
        """Cumulative tokens, derived from the atomic usage snapshot."""
        return self._usage_totals.total_tokens_used

    @total_tokens_used.setter
    def total_tokens_used(  # pyright: ignore[reportIncompatibleVariableOverride]
        self,
        value: int,
    ) -> None:
        self._usage_totals = self._usage_totals._replace(total_tokens_used=int(value))

    @property
    def step_count(self) -> int:
        """Completed agentic steps, derived from the atomic usage snapshot."""
        return self._usage_totals.step_count

    @step_count.setter
    def step_count(  # pyright: ignore[reportIncompatibleVariableOverride]
        self,
        value: int,
    ) -> None:
        self._usage_totals = self._usage_totals._replace(step_count=int(value))

    def usage_snapshot(self) -> tuple[float, int, int]:
        """Return one coherent ``(budget_used, total_tokens_used, step_count)``.

        Reads the immutable snapshot attribute ONCE, so the triple can
        never mix dimensions from two different accounting states —
        unlike three separate property reads, between which a
        concurrent response-accounting publish could land.  Concurrent
        readers (``sorcar_agent._executor_usage`` polled by the live
        usage monitor, the parent's session bank) rely on this.
        """
        totals = self._usage_totals
        return (totals.budget_used, totals.total_tokens_used, totals.step_count)

    def reset_usage(self) -> None:
        """Zero this agent's usage counters before a run.

        The ``(budget_used, total_tokens_used, step_count)`` triple is
        replaced with ONE store, so a concurrent :meth:`usage_snapshot`
        reads either the previous totals or all zeros, never a mix.
        Called at construction, by :meth:`_reset` and by the task runner
        before a run (``task_runner._zero_usage_counters``).  Helper
        spend the model already holds belongs to an earlier run, so the
        mark starts at it.
        """
        mark = _helper_usage_of(getattr(self, "model", None)) or _ZERO_USAGE.helper_mark
        self._usage_totals = _ZERO_USAGE._replace(helper_mark=mark)
        self.tool_calls_made = 0
        """Tool calls other than ``finish`` the model issued during the
        current run (blocked ones included: the model still acted).
        ``RelentlessAgent`` reads it after a sub-session ends — a
        continuation session that only called ``finish`` made no
        progress."""
        self._refused_call: str | None = None
        """The name of the latest refused tool call, until the model
        calls that tool again or reads the re-issue reminder
        (:meth:`_guarded_execute_tool`); ``None`` when nothing is owed."""
        self._redirect_ended: Verdict | None = None
        """The run-ending verdict a hook gave the call a running tool
        redirected to (:meth:`_execute_redirected`), carried out to
        :meth:`_guarded_execute_tool` as the redirecting call's own."""
        self._finish_failed = False
        """Whether the latest ``finish`` call :meth:`_execute_tool` ran
        raised (arguments that do not fit its signature): its error
        result is a tool result the model corrects at its next step,
        not the task's result (:meth:`_execute_step`)."""
        self._keeping_cache_warm = False
        """Whether a tool call is running under
        :meth:`_execute_tool_keeping_cache_warm`: a call nested in it (a
        :class:`ToolRedirect` follow-up) starts no second ping thread."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.pre_step_hook: Callable[..., None] | None = None
        self.tool_call_guard: Callable[[str, dict[str, Any]], str | Verdict | None] | None = None
        self.llm_call_hook: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None
        self.tool_call_hook: Callable[[str, dict[str, Any]], Verdict] | None = None
        self.llm_response_hook: Callable[[LLMReply], Verdict] | None = None
        self.tool_result_hook: Callable[[str, dict[str, Any], str, bool], None] | None = None
        """Called after every tool call the loop or a redirect made, with the
        tool's name, its arguments, the result text the model reads and
        whether the call was refused (the text is then the refusal).  A
        call a hook refused and a call made inside another (a redirected
        ``Edit`` and its follow-up) are reported too, each with its own
        result, so a reader needs no pairing of a turn's calls with its
        results.  ``SorcarAgent`` installs it for its coding gate."""
        self.helper_charge_hook: Callable[[str, float, int, str], None] | None = None
        """Called with the model, USD, tokens and unique key of each paid
        call the model's helpers made for this agent (a vision bridge
        description, ``Model.helper_calls``) once it is billed, so the
        task's split of spend by model files it under the model that made
        it; it may see a key twice and must record it once.  ``SorcarAgent``
        installs it (through ``RelentlessAgent`` for its sessions)."""
        self._helper_calls_filed: frozenset[str] = frozenset()
        self.reset_usage()
        self.budget_check_hook: Callable[[], None] | None = None
        self.budget_status_hook: Callable[[], tuple[float, float]] | None = None
        """Returns ``(spent, max)`` in USD of the task this agent runs a
        session of, for the ``Budget:`` part of the usage line the model
        sees; ``None`` shows this agent's own spend and cap.
        ``RelentlessAgent`` installs it so the line counts earlier
        sessions and sub-agent spend, as its budget check does."""
        self.context_reset_hook: Callable[[], None] | None = None
        """Called after old tool outputs were compacted out of the
        conversation, so tools that assume the model still sees an
        earlier output (the Read tool's dedupe) can forget it."""
        self._served_turn = threading.local()
        """``turn``: the native turn of the call a bridge thread served
        (:meth:`_execute_tool_on_helper_thread`), for the limit check of
        what the model does with the call's result on that thread
        (:meth:`_check_limits_for_helper`); unset on the agent's own.
        Kept across runs: a call the turn's end abandoned may return
        during the next one."""
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        """Clear the per-run counters, caches and flags.

        Called at construction and by :meth:`_reset` before every run,
        so nothing of a previous run leaks into the next one.
        """
        self.context_tokens_used = 0
        # Tokens and USD cost of the latest model call (``input_tokens``,
        # ``output_tokens``, ``cache_read``, ``cache_write``, ``cost``),
        # as recorded by ``_update_tokens_and_budget_from_response``;
        # reported by the ``llm_call`` event.  ``None`` before the first
        # call.
        self.last_call_usage: dict[str, Any] | None = None
        self._cached_tools_schema: list[dict[str, Any]] | None = None
        self._llm_hook_conversation_index = 0
        # The run's ``compaction`` option (:meth:`run`); a step driven
        # without a run (a test's ``_execute_step``) compacts as the
        # global setting says.
        self.compaction = True
        self._next_compaction_at = DEFAULT_CONFIG.compaction_start_tokens
        self._last_compaction_step = -COMPACTION_MIN_STEP_GAP
        # Reasoning the replies left in the context (:func:`reasoning_tokens`)
        # since an edit last removed it, and the newest reply's share of it,
        # which a model may keep through an edit.
        self._reasoning_tokens = 0
        self._newest_reasoning_tokens = 0
        # The originals of the stubs this run's compactions made: a call
        # whose argument is a stub gets the original back before it runs.
        self._compaction_table = CompactionTable()
        # Wall-clock start of the last request that read or wrote the
        # provider's prompt cache (a model call or a keep-alive ping).
        self._prompt_cache_touched_at = 0.0
        # Whether the next model call follows a routine tool call (see
        # ``_turn_effort``), and what the last call was marked as.
        self._routine_next_turn = False
        self._routine_call = False
        self._routine_effort = ""
        # The effort of the next call when it retries a request the
        # output-token cap cut short (``_retry_at_low_effort``), and
        # whether the current request already had that retry.
        self._retry_effort = ""
        self._low_effort_retried = False
        # Characters of the last step's tool results (footer "+N tokens").
        self._last_results_chars = 0
        # ``_reset`` sets it from the run's model config; a step driven
        # without a run (a test's ``_execute_step``) reads the default.
        self._plain_io = False
        self._fallback_used = False
        self._refusal_retried = False
        self._reset_progress_trackers()

    def _reset(
        self,
        model_name: str,
        is_agentic: bool,
        max_steps: int | None,
        max_budget: float | None,
        model_config: dict[str, Any] | None,
        printer: Printer | None = None,
        verbose: bool | None = None,
        print_prompts: bool = True,
        compaction: bool = True,
    ) -> None:
        self.model_name = model_name
        self.print_prompts = print_prompts
        self.set_printer(printer, verbose=verbose)
        # Per-run state is reset BEFORE the model is built.  ``model()``
        # raises for an unknown model name, and ``run()`` saves the
        # trajectory from its ``finally``: with the previous run's
        # ``run_start_timestamp`` and messages still in place, that save
        # would land on the previous run's file and overwrite it.
        self.is_agentic = is_agentic
        self.max_steps = max_steps if max_steps is not None else 10000
        self.max_budget = max_budget if max_budget is not None else 10.0
        self.function_map: dict[str, Callable[..., Any]] = {}
        self.messages: list[dict[str, Any]] = []
        self.reset_usage()
        self._reset_run_state()
        self.compaction = compaction
        # ``run_start_timestamp`` is the real wall clock: the saved record
        # pairs it with ``run_end_timestamp``.  The trajectory FILENAME is
        # keyed by (name, id, _trajectory_stamp) in whole seconds; two runs
        # of one instance started in the same second would share a path
        # and the later save would destroy the earlier record, so the
        # stamp strictly increases per run while keeping the same format.
        self.run_start_timestamp = int(time.time())
        self._trajectory_stamp = max(self.run_start_timestamp, self._trajectory_stamp + 1)
        # The agent's own copy: ``Model.__init__`` keeps the dict it is
        # given and ``_turn_effort`` edits it for the length of a turn, so
        # a dict shared by several agents (or kept by the caller) must
        # not be the one the model reads.
        if model_config is not None:
            model_config = dict(model_config)
        self._model_config: dict[str, Any] | None = model_config
        # Under ``plain_io`` the model reads no usage line; the printer and
        # the trajectory still get it.
        self._plain_io = plain_io(model_config)

        token_callback = self.printer.token_callback if self.printer else None
        thinking_callback = self.printer.thinking_callback if self.printer else None

        existing = getattr(self, "model", None)
        if (  # pragma: no branch
            existing is not None
            and existing.model_name == self.model_name
            and existing.model_config == (model_config or {})
        ):
            existing.reset_conversation()
            # Through the hook, not the two attributes: a transport that
            # owns a sub-model with its own callbacks re-binds it there.
            existing.rebind_callbacks(token_callback, thinking_callback)
            self.model = existing
        else:
            self.model = model(
                model_name,
                model_config=model_config,
                token_callback=token_callback,
                thinking_callback=thinking_callback,
            )
        # A helper the model runs on this agent's behalf (a vision bridge
        # description) checks the limits before each paid call, so a batch
        # of them stops at the cap; hand_off_conversation carries the hook
        # to a replacement model.
        self.model.limit_check = self._check_limits_for_helper

    def _reset_progress_trackers(self) -> None:
        """Clear the text-only-turn counter and last text.

        Called at construction, on every run reset, and after a mid-run
        fallback model swap, so stale counts — and stale status text that
        an implicit finish would otherwise report — never leak from the
        previous model or run into the new one.
        """
        self._consecutive_no_tool_calls = 0
        self._last_response_text = ""
        # The record of the current step's natively run tool calls (see
        # NativeTurn); _execute_step replaces it every step.
        self._native_turn = NativeTurn(self.printer)

    def _set_prompt(
        self,
        prompt_template: str,
        arguments: dict[str, str] | None = None,
        attachments: list[Attachment] | None = None,
    ) -> None:
        """Sets the prompt for the agent.

        Args:
            prompt_template: The template string for the prompt with placeholders.
            arguments: Optional dictionary of arguments to substitute into the template.
            attachments: Optional list of file attachments (images, PDFs) to include.
        """
        assert self.model is not None
        self.arguments = dict(arguments) if arguments is not None else {}
        self.prompt_template = prompt_template
        full_prompt = substitute_prompt_args(self.prompt_template, self.arguments)

        self._add_message("user", full_prompt)
        self.model.initialize(full_prompt, attachments=attachments)
        if self.printer and self.print_prompts:
            self.printer.print(full_prompt, type="prompt")

    def run(
        self,
        model_name: str,
        prompt_template: str,
        arguments: dict[str, str] | None = None,
        system_prompt: str = "",
        tools: list[Callable[..., Any]] | None = None,
        is_agentic: bool = True,
        max_steps: int | None = None,
        max_budget: float | None = None,
        model_config: dict[str, Any] | None = None,
        printer: Printer | None = None,
        verbose: bool | None = None,
        attachments: list[Attachment] | None = None,
        print_prompts: bool = True,
        llm_call_hook: (Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None) = None,
        tool_call_hook: Callable[[str, dict[str, Any]], Verdict] | None = None,
        llm_response_hook: Callable[[LLMReply], Verdict] | None = None,
        compaction: bool = True,
    ) -> str:
        """
        Runs the agent's main ReAct loop to solve the task.

        Args:
            model_name (str): The name of the model to use for the agent.
            prompt_template (str): The prompt template for the agent.
            arguments (dict[str, str] | None): The arguments to be substituted into the prompt
                template. Default is None.
            system_prompt (str): Optional system prompt to provide to the model.
                Default is empty string (no system prompt).
            tools (list[Callable[..., Any]] | None): The tools to use for the agent.
                If None, no tools are provided (only the built-in finish tool is added).
            is_agentic (bool): Whether the agent is agentic. Default is True.
            max_steps (int): The maximum number of steps to take.
                Default is 10000.
            max_budget (float): The maximum budget to spend.
                Default is 10.0.
            model_config (dict[str, Any] | None): The model configuration to use for the agent.
                Default is None.
            printer (Printer | None): Optional printer for streaming output.
                Default is None.
            verbose (bool | None): Whether to print output to console.
                Default is None (verbose enabled).
            attachments (list[Attachment] | None): Optional file attachments (images, PDFs)
                to include in the initial prompt. Default is None.
            print_prompts (bool): Whether to print the system prompt and task
                prompt to the printer. Internal helper agents (e.g. the
                summarizer in RelentlessAgent) pass False so their internal
                prompts never surface as user-visible "prompt" events in a
                shared printer's event stream. Default is True.
            llm_call_hook (Callable | None): Optional hook called before every
                ``generate_and_process_with_tools`` LLM call with the list of
                new messages (those added to the conversation since the
                previous LLM call) about to be sent to the LLM. Its return
                value — a possibly modified list of messages — replaces those
                new messages in the conversation before the call is made.
                After a turn in which the model called tools natively
                (Codex over MCP) the new messages include those calls and
                their results, and the turn's answer. Default is None (no
                hook).
            tool_call_hook (Callable | None): Optional hook called before every
                tool call with the tool's name and its arguments dict; it
                returns a :class:`~kiss.core.tool_verdict.Verdict`:
                ``ALLOW`` lets the tool execute as usual, ``refuse(text)``
                suppresses the execution and the model reads
                :func:`~kiss.core.tool_verdict.refusal_result` of *text*
                ("REFUSED: the X call did NOT run and had no effect. ..."),
                then a reminder to re-issue the call on the next allowed
                call of another tool. The hook runs before (and its
                rejection takes precedence over) the framework's
                :attr:`tool_call_guard`; allowing does not override a guard
                block. An implicit finish (text-only turns) also consults the
                hook (with ``("finish", {})``) and is suppressed when the hook
                refuses. :func:`~kiss.core.tool_verdict.end_run` ends the
                run instead: the call does not run, no later call of the
                turn runs, and the verdict's text becomes the run's result
                in the registered ``finish`` tool's contract. Default is
                None (no hook).
            llm_response_hook (Callable | None): Optional hook called after
                every billed model call of the agentic loop with an
                :class:`~kiss.core.llm_reply.LLMReply` (the reply's text,
                its tool calls, its token usage and cost, the agent's
                totals), before any tool call of the reply runs; a tool
                Codex ran natively during the turn has run already. It
                returns a :class:`~kiss.core.tool_verdict.Verdict`:
                ``ALLOW`` lets the reply be acted on, any other verdict
                (``end_run(result)``) ends the run with the verdict's text
                as the result and none of the reply's tool calls run.
                Default is None (no hook).
            compaction (bool): Whether this run compacts its context:
                old, large tool outputs and arguments are replaced with
                short stubs once the context grows past the configured
                thresholds (:meth:`_maybe_compact_conversation`). False
                never compacts. True (the default) compacts while the
                global Compaction setting
                (``DEFAULT_CONFIG.tool_output_compaction``) is on; off,
                no run compacts.

        Returns:
            str: The result of the agent's task.
        """
        self.llm_call_hook = llm_call_hook
        self.tool_call_hook = tool_call_hook
        self.llm_response_hook = llm_response_hook
        try:
            if system_prompt:
                model_config = dict(model_config) if model_config else {}
                model_config.setdefault("system_instruction", system_prompt)
            self._reset(
                model_name,
                is_agentic,
                max_steps,
                max_budget,
                model_config,
                printer,
                verbose,
                print_prompts=print_prompts,
                compaction=compaction,
            )

            if not self.is_agentic and tools is not None:
                raise KISSError(
                    f"Tools cannot be provided for a non-agentic agent "
                    f"{self.name} with id {self.id}."
                )
            self._setup_tools(tools)
            if system_prompt and self.printer and self.print_prompts:
                self.printer.print(strip_system_cache_break(system_prompt), type="system_prompt")
            self._set_prompt(prompt_template, arguments, attachments=attachments)

            if not self.is_agentic:
                return self._run_non_agentic()

            # Records made by tools and hooks during the run (a gate stage)
            # reach this run's event log through the thread's printer.
            previous_printer = bind_printer(self.printer)
            try:
                return self._run_agentic_loop()
            finally:
                bind_printer(previous_printer)

        finally:
            # Spend the model still holds (a description made in the last
            # step before the step limit, or before an error ended the
            # run) joins the totals before they are saved.
            self._bill_helper_usage()
            self._save()

    def _setup_tools(self, tools: list[Callable[..., Any]] | None) -> None:
        """Setup tools for agentic mode.

        Adds finish tool if not present, and web tools if enabled in config.
        Pre-builds and caches the tool schema so it is not rebuilt on every LLM call.

        Args:
            tools: Optional list of callable tools to make available to the agent.
        """
        if not self.is_agentic:
            return

        tools = list(tools or [])
        tool_names = {getattr(tool, "__name__", None) for tool in tools}

        if "finish" not in tool_names:
            tools.append(self.finish)

        self._add_functions(tools)
        self._cached_tools_schema = self.model._build_openai_tools_schema(self.function_map)

    def _print_result(self, result: str) -> None:
        """Emit *result* as the run's terminal ``result`` printer event."""
        if result and self.printer:
            self.printer.print(
                result,
                type="result",
                step_count=self.step_count,
                total_tokens=self.total_tokens_used,
                cost=f"${self.budget_used:.4f}",
                cost_usd=self.budget_used,
            )

    def _run_non_agentic(self) -> str:
        """Run a single generation without tools, with billing and transcript.

        The generation is a paid request like any step of the agentic
        loop, so it is under the same limits (:meth:`_check_limits`: a
        spent budget, a parent's ``budget_check_hook``) and made only
        when they allow it.

        Returns:
            str: The generated response text from the model.

        Raises:
            KISSError: When a limit is already exceeded.
        """
        start_timestamp = int(time.time())
        self.step_count += 1
        self._check_limits()
        response_text, _response = self._billed_model_call(self.model.generate)
        self._add_message(
            "model",
            _with_usage_footer(response_text, self._get_usage_info_string()),
            start_timestamp,
        )
        self._print_result(response_text)
        return str(response_text)

    def _try_switch_to_fallback(
        self, reason: str = "a non-retryable error", other_family: bool = False
    ) -> str | None:
        """Swap ``self.model`` to a fallback model, if one is available.

        Consulted by :meth:`_run_agentic_loop` after a recoverable
        model-level failure.  A non-retryable provider error (model gated
        / deprecated, credit balance too low, etc.) is the provider's
        problem, so the fallback is
        :func:`kiss.core.models.model_info.get_fallback_model`: the one
        declared in the catalog, else the model's OpenRouter twin when
        ``OPENROUTER_API_KEY`` is configured.  A safety refusal, a
        content-filter stop, an output-cap cut-off or repeated empty
        turns (``other_family=True``) are shared by every model of the
        family, so the fallback is
        :func:`kiss.core.models.model_families.fallback_of_another_family`:
        a model of another family when one is configured, the declared
        fallback or twin only when nothing else is.  This method:

        1. Guards against repeated swaps within a single run (only one
           fallback is allowed per :meth:`run` invocation).
        1. Rebuilds the model via :func:`kiss.core.models.model_info.model`
           using the ``model_config`` originally passed to :meth:`run`
           and the printer's streaming callbacks.  The fallback declared
           in the catalog keeps the caller's endpoint overrides
           (``base_url``, ``api_key``, ``extra_headers``: it lives where
           the user pointed); any other fallback is a different
           provider, so those keys are dropped and it routes with its
           own configured key instead of re-using the endpoint and
           credentials that just failed.
        1. Copies the primary model's conversation history onto the
           new model so no context is lost.
        1. Rebuilds :attr:`_cached_tools_schema` so
           :meth:`_execute_step` calls the fallback provider's schema
           (Anthropic-vs-OpenAI-vs-Gemini all differ).
        1. Emits a visible ``system_prompt`` warning naming the old and
           the new model and the stop reason, and saying so when the
           same family was the only choice.

        Args:
            reason: What the model returned, named in the warning.
            other_family: Whether the failure is one the whole family
                shares, so a model of another family is preferred.

        Returns:
            The new model name on a successful swap, or ``None`` when no
            fallback is available, the fallback equals the current
            model, or the one-shot guard has already been consumed.
        """
        if self._fallback_used:
            return None
        if other_family and is_known_model(self.model_name):
            new_name = fallback_of_another_family(self.model_name)
        else:
            # A model outside the catalog has no family to fail over from.
            new_name = get_fallback_model(self.model_name)
        if not new_name or new_name == self.model_name:
            return None
        fallback_config = self._model_config
        if fallback_config and new_name != declared_fallback(self.model_name):
            fallback_config = {
                k: v for k, v in fallback_config.items() if k not in _ENDPOINT_CONFIG_KEYS
            } or None
        token_cb = self.printer.token_callback if self.printer else None
        thinking_cb = self.printer.thinking_callback if self.printer else None
        new_model = model(
            new_name,
            model_config=fallback_config,
            token_callback=token_cb,
            thinking_callback=thinking_cb,
        )
        new_model.initialize("")
        old_name = self.model_name
        self.switch_model(new_model, new_name)
        self._fallback_used = True
        note = (
            "; no other model family is configured"
            if other_family and same_family(new_name, old_name)
            else ""
        )
        if self.printer:
            self.printer.print(
                f"Switched model from {old_name} to {new_name} after {reason}{note}",
                type="system_prompt",
            )
        return new_name

    def switch_model(self, new_model: Model, new_name: str) -> None:
        """Continue the run on the initialized *new_model*, named *new_name*.

        Shared by the fallback swap and Sorcar's ``set_model`` tool.  The
        live conversation is handed off (:func:`hand_off_conversation`)
        and the ``llm_call_hook`` boundary moves with it, so the messages
        appended since the last LLM call (the tool results of the
        switching step) stay new to the hook although the hand-off may
        change the message count.  Gemini keeps its thought signatures
        in a side map keyed by tool-call id, which ``initialize()``
        cleared on the new model: a Gemini-to-Gemini switch carries them
        over, or a signature-enforcing model would reject the next
        request.  The cached tools schema is rebuilt for the new model.

        Args:
            new_model: The initialized model that continues the conversation.
            new_name: Its model name.
        """
        self._llm_hook_conversation_index = hand_off_conversation(
            self.model, new_model, self._llm_hook_conversation_index
        )
        if getattr(self.model, "base_url", None) != getattr(new_model, "base_url", None):
            # The hand-off converted the conversation, which drops its reasoning.
            self._reasoning_tokens = self._newest_reasoning_tokens = 0
        old_signatures = getattr(self.model, "_thought_signatures", None)
        new_signatures = getattr(new_model, "_thought_signatures", None)
        if isinstance(old_signatures, dict) and isinstance(new_signatures, dict):
            new_signatures.update(old_signatures)
        self.model = new_model
        self.model_name = new_name
        if self.function_map:
            self._cached_tools_schema = new_model._build_openai_tools_schema(self.function_map)

    def _retry_at_low_effort(self, error: OutputCapError) -> bool:
        """Arrange one retry, at reasoning effort ``low``, of a request the cap cut short.

        The retry re-sends the same conversation (the adapter appended
        nothing for the cut-off turn) with :data:`RETRY_EFFORT` applied
        for that one call by :meth:`_turn_effort`.  No retry is arranged
        when the model takes no effort parameter (no catalog ``thinking``
        level), already runs at a low level (an explicit
        ``reasoning_effort``, else Anthropic's native
        ``output_config.effort``, else Gemini's ``thinking_config``
        level, else the catalog level), or this
        request had its retry already: the caller then fails over to
        another family.

        Args:
            error: The cut-off the adapter reported.

        Returns:
            Whether the next call retries the request at low effort.
        """
        info = MODEL_INFO.get(self.model.model_name)
        if self._low_effort_retried or info is None or not info.thinking:
            return False
        config = self.model.model_config
        output_config = config.get("output_config")
        gemini_level = config.get("thinking_config")
        if isinstance(gemini_level, dict):
            gemini_level = gemini_level.get("thinking_level")
        else:  # a Gemini ``ThinkingConfig`` (or None): its level is an enum
            gemini_level = getattr(
                getattr(gemini_level, "thinking_level", None), "value", None
            )
        effort = str(
            config.get("reasoning_effort")
            or (output_config.get("effort") if isinstance(output_config, dict) else None)
            or gemini_level
            or info.thinking
        ).lower()
        if effort in _LOW_EFFORT_LEVELS:
            return False
        self._low_effort_retried = True
        self._retry_effort = RETRY_EFFORT
        if self.printer:
            self.printer.print(
                f"Retrying the request at reasoning effort {RETRY_EFFORT} after "
                f"{error.stop_reason}",
                type="system_prompt",
            )
        return True

    def _recover_from_family_failure(
        self, error: _EmptyModelResponseError | ModelRefusalError | OutputCapError
    ) -> bool:
        """Arrange the next step after a failure the whole model family shares.

        A refusal, a content filter, an output cap or repeated empty
        turns are deterministic for the request, so the step is retried
        on a model of another family when one is configured
        (:meth:`_try_switch_to_fallback`); failing that, a refusal gets
        one retry with the task's context stated (:data:`REFUSAL_RETRY_MSG`),
        and a second refusal is final.

        Args:
            error: The typed failure of the step.

        Returns:
            Whether the loop continues (a model was switched or the
            retry message was added); ``False`` means *error* ends the run.
        """
        new_name = self._try_switch_to_fallback(reason=_failure_reason(error), other_family=True)
        if new_name is not None:
            self._low_effort_retried = False
            self._reset_progress_trackers()
            return True
        if isinstance(error, ModelRefusalError) and not self._refusal_retried:
            self._refusal_retried = True
            self.model.add_message_to_conversation("user", REFUSAL_RETRY_MSG)
            self._add_message("user", REFUSAL_RETRY_MSG)
            return True
        return False

    def _run_agentic_loop(self) -> str:
        consecutive_errors = 0
        consecutive_stalls = 0
        # The step bound lives here and nowhere else: driving the loop
        # off ``step_count`` (rather than a fixed ``range``) keeps it
        # correct even if steps are ever counted from outside the loop,
        # and leaves exactly one message for "out of steps".
        while self.step_count < self.max_steps:
            self.step_count += 1  # pyright: ignore[reportIncompatibleVariableOverride]
            self._check_limits()
            try:
                result = self._execute_step()
                consecutive_errors = consecutive_stalls = 0
                self._low_effort_retried = False
                if result is not None:
                    self._bill_helper_usage()  # the result event carries the final totals
                    self._print_result(result)
                    return result
            except KISSError as e:  # pragma: no cover – requires model to fail mid-step
                logger.debug("Exception caught", exc_info=True)
                if isinstance(e, OutputCapError) and self._retry_at_low_effort(e):
                    continue
                if isinstance(
                    e, _EmptyModelResponseError | ModelRefusalError | OutputCapError
                ) and self._recover_from_family_failure(e):
                    consecutive_errors = 0
                    continue
                raise
            except Exception as e:
                logger.debug("Exception caught", exc_info=True)
                if is_content_policy_error(e):
                    # The provider refused the request itself (HTTP 400
                    # "content was flagged ..."): the same request gets the
                    # same answer from the whole family, so it is a
                    # content filter, not an outage to retry three times.
                    refused = ContentFilterError(
                        f"Content policy refusal from model: {e}", stop_reason=CONTENT_POLICY_STOP
                    )
                    if self._recover_from_family_failure(refused):
                        consecutive_errors = 0
                        continue
                    raise refused from e
                if not _is_retryable_error(e):
                    new_name = self._try_switch_to_fallback(reason="a non-retryable error")
                    if new_name is None:
                        raise KISSError(f"Non-retryable error from model: {e}") from e
                    consecutive_errors = 0
                    self._reset_progress_trackers()
                    continue
                if is_stall_error(e):
                    consecutive_stalls += 1
                    if consecutive_stalls >= MAX_CONSECUTIVE_STALLS:
                        raise KISSError(
                            f"Agent {self.name} failed with {consecutive_stalls} "
                            f"consecutive stalled streams. Last error: {e}"
                        ) from e
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        raise KISSError(
                            f"Agent {self.name} failed with {consecutive_errors} "
                            f"consecutive errors. Last error: {e}"
                        ) from e
                content = f"Failed to get response from Model: {e}.\nPlease try again.\n"
                self.model.add_message_to_conversation("user", content)
                self._add_message("user", content)

        raise StepLimitExceededError(f"Agent {self.name} exceeded {self.max_steps} steps.")

    def _execute_step(self) -> str | None:
        """Execute a single step in the ReAct loop.

        An agent a tool runs inline (the merge-conflict resolver under
        ``commit_and_merge``) runs under that tool's call: a Stop
        accepted on its panel (:mod:`kiss.core.tool_interrupt`) is
        checked before the model request and once the step's tool
        results are recorded, so the enclosing call ends as interrupted
        instead of this agent requesting on, or returning a result, after
        the Stop reached its running tool.

        Returns:
            str | None: The result string if the task is finished, None otherwise.

        Raises:
            ToolCallInterrupted: When the call this agent runs inside
                was interrupted.
        """
        start_timestamp = int(time.time())
        logger.info(
            "Step %d/%d start: agent=%s budget=$%.4f/%s tokens=%d",
            self.step_count,
            self.max_steps,
            self.name,
            self.budget_used,
            f"${self.max_budget:.2f}",
            self.total_tokens_used,
        )
        raise_if_interrupted()

        if self.pre_step_hook is not None:
            self.pre_step_hook(self.model)
        self._maybe_compact_conversation()
        if self.llm_call_hook is not None:
            hook_start = self._llm_hook_conversation_index
            modified_messages = self.llm_call_hook(list(self.model.conversation[hook_start:]))
            self.model.conversation[hook_start:] = list(modified_messages)
        # Advance the boundary BEFORE the call so a raising call (retryable
        # provider error, refusal, fallback swap) never re-presents
        # already-hooked messages to the hook on the next attempt ...
        self._llm_hook_conversation_index = len(self.model.conversation)
        self._prompt_cache_touched_at = time.time()
        self._native_turn = turn = NativeTurn(self.printer)
        self.model.last_context_edits = []
        try:
            function_calls, response_text, _response = self._billed_model_call(
                self._generate_with_tools
            )
        except Exception as e:
            if turn.error is not None:
                raise turn.error
            if turn.ended is not None:
                # The run ended on a natively run tool call before the
                # turn failed (a stream that broke, a Codex process that
                # died): the verdict is the run's result, what came after
                # it is not — within the limits, as on a turn that
                # returned: the failed turn's usage is billed above.
                logger.warning(
                    "Step %d: the turn failed after a hook ended the run: %s", self.step_count, e
                )
                self._add_message(
                    "model", f"[the turn failed after the run ended: {e}]", start_timestamp
                )
                self._check_limits()
                return self._ended_run_result(turn.ended)
            raise
        # ... and again AFTER it returns, so the assistant turn the call
        # appended is not treated as a "new" message on the next call —
        # unless tools ran natively during it: their calls and results
        # were appended by the model and are new to the hook.
        if not turn.calls:
            self._llm_hook_conversation_index = len(self.model.conversation)
        self._newest_reasoning_tokens = reasoning_tokens(
            (self.last_call_usage or {}).get("output_tokens", 0), function_calls, response_text
        )
        self._reasoning_tokens += self._newest_reasoning_tokens
        function_results: list[tuple[str, dict[str, Any]]] = []
        try:
            result = self._act_on_reply(
                turn, function_calls, response_text, start_timestamp, function_results
            )
        except KISSError:
            raise
        except Exception as e:
            # The loop retries, and the reply is in the model's
            # conversation: its tool calls are answered first.
            if function_calls:
                self._answer_unanswered_calls(
                    function_calls, function_results, response_text, start_timestamp, e
                )
            raise
        if result is None and function_calls:
            self._hand_over_results(function_results)
        return result

    def _act_on_reply(
        self,
        turn: NativeTurn,
        function_calls: list[dict[str, Any]],
        response_text: str,
        start_timestamp: int,
        function_results: list[tuple[str, dict[str, Any]]],
    ) -> str | None:
        """Act on the model's reply to a step: ask the hooks, run its tool calls, record it.

        Args:
            turn: The step's record of natively run tool calls.
            function_calls: The tool calls of the reply.
            response_text: The text of the reply.
            start_timestamp: When the step began, for the trajectory.
            function_results: Filled with ``(tool name, {"result": text})``
                of each call once it is answered, in the calls' order, so
                the caller holds the results a failure leaves behind.

        Returns:
            The run's result when the step ended the run (``finish``, a
            run-ending verdict, an implicit finish), else ``None``:
            every call is then answered in *function_results*, for the
            caller to give the model, or the reply made none and the
            model was told how to go on.

        Raises:
            KISSError: A limit was hit (the step is recorded first when
                a tool ran), or the model keeps answering with nothing.
            ToolCallInterrupted: When the call this agent runs inside
                was interrupted.
        """
        if response_text and response_text.strip():
            self._last_response_text = response_text
        usage_info = self._get_usage_info_string()
        self.model.set_usage_info_for_messages("" if self._plain_io else usage_info)
        if self.printer:
            self.printer.print(
                usage_info,
                type="usage_info",
                total_tokens=self.total_tokens_used,
                cost=f"${self.budget_used:.4f}",
                cost_usd=self.budget_used,
                total_steps=self.step_count,
                cache_read=(self.last_call_usage or {}).get("cache_read", 0),
                model=self.model.model_name,
            )
        if self.model.last_context_edits:
            # What the provider cleared from this request's context before
            # the model read it (the keep-K policy, applied server-side).
            if self.printer:
                self.printer.print(
                    "",
                    type="context_edits",
                    step=self.step_count,
                    edits=self.model.last_context_edits,
                    cleared_input_tokens=sum(
                        int(edit.get("cleared_input_tokens") or 0)
                        for edit in self.model.last_context_edits
                    ),
                )
            cleared_results = any(
                int(edit.get("cleared_tool_uses") or 0) > 0
                for edit in self.model.last_context_edits
            )
            if cleared_results and self.context_reset_hook is not None:
                # Old tool results left the model's context: the Read
                # dedupe may no longer answer with "unchanged since".
                self.context_reset_hook()

        # A fatal error a natively run tool raised (budget exhausted, a
        # limit hit) ends the step now that the turn is billed and in the
        # trajectory; the ordinary tool loop below does the same.
        if turn.error is not None:
            raise turn.error
        if turn.calls or any(fc["name"] != "finish" for fc in function_calls):
            self._check_limits()
        # A run-ending verdict on a natively run tool call (recorded by
        # :meth:`_execute_tool_on_helper_thread`), or on this reply (the
        # response hook), ends the run now; none of the reply's tool calls
        # run.  The reply itself is in the trajectory either way, and the
        # response hook sees every reply, this one included: the earlier
        # verdict keeps precedence over the hook's.
        ended = turn.ended
        if self.llm_response_hook is not None:
            verdict = self.llm_response_hook(
                LLMReply(
                    response_text or "",
                    [{"name": fc["name"], "arguments": _call_args(fc)} for fc in function_calls],
                    zero_usage() | (self.last_call_usage or {}),
                    self.step_count,
                    self.budget_used,
                    self.total_tokens_used,
                )
            )
            if ended is None and not verdict.allowed:
                ended = verdict
        if ended is not None:
            self._add_message(
                "model", _with_usage_footer(response_text, usage_info), start_timestamp
            )
            return self._ended_run_result(ended)

        if not function_calls:
            self._add_message(
                "model", _with_usage_footer(response_text, usage_info), start_timestamp
            )
            if turn.calls:
                # The model worked through natively served tools and ended
                # the turn with text: not a text-only turn, so no implicit
                # finish — it is told to go on, or to finish through the
                # text protocol.
                self._consecutive_no_tool_calls = 0
                self._add_message("user", NATIVE_TOOLS_CONTINUE_MSG)
                self.model.add_message_to_conversation("user", NATIVE_TOOLS_CONTINUE_MSG)
                return None
            self._consecutive_no_tool_calls += 1
            refusal: Verdict | None = None
            if self._consecutive_no_tool_calls >= MAX_CONSECUTIVE_NO_TOOL_CALLS:
                if not response_text or not response_text.strip():
                    raise _EmptyModelResponseError(
                        f"Agent {self.name} aborted: model "
                        f"{self.model_name} returned "
                        f"{self._consecutive_no_tool_calls} consecutive "
                        f"empty responses (no text and no tool calls) at "
                        f"step {self.step_count}. This often indicates a "
                        f"streaming or reasoning-block parsing issue in "
                        f"the model adapter. Try a different model or "
                        f"restart the task."
                    )
                # The implicit finish stands in for a ``finish`` call the
                # model never made, so it faces the same vetoes; a refusal
                # is what the model reads next, since a one-shot veto
                # (Sorcar's unmerged-worktree gate) is spent on it.
                refusal = self._tool_call_refusal("finish", {})
                if refusal is not None and refusal.ends_run:
                    return self._ended_run_result(refusal)
                if refusal is None:
                    logger.info(
                        "Implicit finish: agent=%s step=%d text-only response "
                        "for %d consecutive turns",
                        self.name,
                        self.step_count,
                        self._consecutive_no_tool_calls,
                    )
                    return self._implicit_finish_result(
                        f"The model replied with text but no tool call for "
                        f"{self._consecutive_no_tool_calls} consecutive turns "
                        f"without calling finish."
                    )
            retry_msg = (refusal.text if refusal is not None else "") or (
                "**Your response MUST have at least one function call. "
                "Your response has 0 function calls. If you have completed "
                "the task, call the `finish` tool now with your final "
                "result instead of describing it in text.**"
            )
            self._add_message("user", retry_msg)
            self.model.add_message_to_conversation("user", retry_msg)
            return None

        self._consecutive_no_tool_calls = 0
        call_reprs = []
        finish_result: str | None = None
        # A budget/context limit hit after a tool ran is raised only
        # AFTER the step is recorded in ``messages``: the tool DID run
        # (a Write landed, a command executed), and a partial result
        # built from the trajectory must not omit the last step.
        limit_error: KISSError | None = None

        for fc in function_calls:
            try:
                name, response_str, blocked = self._guarded_execute_tool(fc)
            except BudgetExceededError as e:
                # The tool itself overran the budget (a sub-agent's
                # spend): the step still happened, so it is recorded
                # like a limit hit after the tool ran.
                call_reprs.append(render_tool_call(fc["name"], _call_args(fc)))
                function_results.append((fc["name"], {"result": str(e)}))
                limit_error = e
                break
            self._mark_routine_turn(fc, response_str, blocked is not None)
            call_reprs.append(render_tool_call(fc["name"], _call_args(fc)))
            function_results.append((name, {"result": response_str}))
            if blocked is not None and blocked.ends_run:
                # The hook ended the run on this call: the calls after it
                # in the turn do not run.
                finish_result = self._ended_run_result(blocked)
                break
            if name == "finish" and blocked is None and not self._finish_failed:
                # The task is done: the calls after ``finish`` in the
                # turn do not run (a second finish would replace this
                # result, a mutating call would act after the outcome).
                # A finish that raised (``_finish_failed``) is an
                # ordinary tool error the model corrects.
                finish_result = response_str
                break
            try:
                self._check_limits()
            except KISSError as e:
                limit_error = e
                break

        self._record_tool_step(
            response_text, call_reprs, usage_info, function_results, start_timestamp
        )
        if limit_error is not None:
            raise limit_error
        # The step is in the trajectory; the enclosing call, if
        # interrupted while the tools ran, ends before this agent's
        # result or next request.
        raise_if_interrupted()

        if finish_result is not None:
            logger.info(
                "finish() called: agent=%s step=%d budget=$%.4f tokens=%d result=%r",
                self.name,
                self.step_count,
                self.budget_used,
                self.total_tokens_used,
                finish_result[:200] if len(finish_result) > 200 else finish_result,
            )
            return finish_result
        return None

    def _record_tool_step(
        self,
        response_text: str,
        call_reprs: list[str],
        usage_info: str,
        function_results: list[tuple[str, dict[str, Any]]],
        start_timestamp: int,
    ) -> None:
        """Add a step's reply with its tool calls, then their results, to the trajectory.

        The results message is the ``[name]: text`` paragraphs of the
        results, and a result may itself hold such a paragraph (a file
        that quotes a log): its ``result_lengths`` key, the characters
        of each result in order, says where each one ends.

        Args:
            response_text: The text of the reply.
            call_reprs: The rendered tool calls of the reply that got a result.
            usage_info: The usage line under the reply.
            function_results: ``(tool name, {"result": text})`` of each of those calls.
            start_timestamp: When the step began.
        """
        tool_call_timestamp = int(time.time())
        self._add_message(
            "model",
            _with_usage_footer(response_text + "\n" + "\n".join(call_reprs), usage_info),
            start_timestamp,
        )
        paragraphs = [(name, f"{result['result']}") for name, result in function_results]
        self._add_message(
            "user",
            "\n\n".join(f"[{name}]: {text}" for name, text in paragraphs),
            tool_call_timestamp,
        )
        self.messages[-1]["result_lengths"] = [len(text) for _, text in paragraphs]

    def _answer_unanswered_calls(
        self,
        function_calls: list[dict[str, Any]],
        function_results: list[tuple[str, dict[str, Any]]],
        response_text: str,
        start_timestamp: int,
        error: Exception,
    ) -> None:
        """Settle the step *error* ended with every tool call of its reply answered.

        A hook, the printer or a limit check raised after the model's
        reply entered its conversation.  The loop retries such a step,
        and a request whose tool calls have no results is one the
        provider rejects: retried as it stood, the step failed again
        until the run gave up.  The calls that were answered keep their
        results (the tools ran, the model need not repeat them); each
        of the others reads why it has none.  The step is recorded and
        its results given to the model like those of a step that
        completed.

        Args:
            function_calls: The tool calls of the reply, at least one.
            function_results: The results of the first of them, in
                order; completed here.
            response_text: The text of the reply.
            start_timestamp: When the step began.
            error: What ended the step.
        """
        note = UNANSWERED_CALL_NOTE.format(error=f"{type(error).__name__}: {error}")
        function_results.extend(
            (fc["name"], {"result": note}) for fc in function_calls[len(function_results) :]
        )
        self._record_tool_step(
            response_text,
            [render_tool_call(fc["name"], _call_args(fc)) for fc in function_calls],
            self._get_usage_info_string(),
            function_results,
            start_timestamp,
        )
        self._hand_over_results(function_results)

    def _hand_over_results(self, function_results: list[tuple[str, dict[str, Any]]]) -> None:
        """Give the model the results of its reply's tool calls, each exactly once.

        The adapter adds the results whole or not at all.  It can raise
        on one: a picture is described for a model that cannot see, and
        the limit check before the description runs the budget hook.
        The loop retries such a step, and a request with a tool call
        unanswered is one the provider rejects, so the results are
        handed over once more as text (:func:`_as_text`), which needs
        no description, before the error goes on to the loop.  A
        ``KISSError`` (the budget is spent) gets the same hand-over
        though it ends the run: text costs no model call.

        Args:
            function_results: ``(tool name, {"result": text})`` of every
                call of the reply, in the calls' order.
        """
        # Re-read after the tools ran: a tool may have spent (a sub-agent
        # whose cost was folded into the task), and the line appended to
        # these results is what the model reads next.
        # What this step's tool results add to the context (about 4
        # characters per token), shown in the footer so an oversized
        # output is noticed by the model that asked for it.
        self._last_results_chars = sum(len(str(r["result"])) for _, r in function_results)
        if not self._plain_io:
            self.model.set_usage_info_for_messages(self._get_usage_info_string())
        try:
            self.model.add_function_results_to_conversation_and_return(function_results)
        except Exception as e:
            self.model.add_function_results_to_conversation_and_return(
                [(name, _as_text(result, e)) for name, result in function_results]
            )
            raise
        turn_steps = DEFAULT_CONFIG.reasoning_turn_steps
        if (
            turn_steps > 0
            and self.step_count % turn_steps == 0
            and self.model.keeps_reasoning_turns()
        ):
            # A message holding only tool results continues the provider's
            # reasoning turn; this note starts a new one, so the keep-K
            # policy measures steps (see Model.keeps_reasoning_turns).
            note = REASONING_TURN_NOTE.format(
                step=self.step_count, keep=DEFAULT_CONFIG.reasoning_keep_turns
            )
            self._add_message("user", note)
            self.model.add_message_to_conversation("user", note)

    @contextlib.contextmanager
    def _native_tools_exposed(self) -> Iterator[None]:
        """Serve the tools natively to the model within the block, ``finish`` excluded.

        See :meth:`Model.native_tools_exposed`: a no-op for every model
        but Codex, which takes them as an MCP server and calls them on
        the bridge's threads through
        :meth:`_execute_tool_on_helper_thread`.  ``finish`` stays on the
        model's ordinary tool protocol so that a turn ends when the task
        does and nothing runs after it; so does any tool marked
        ``runs_between_turns = True`` (Sorcar's ``set_model``, which
        replaces this agent's model and must not do so under a running
        turn).  The tool set and its executor are built only for a
        model that serves tools natively (:attr:`Model.serves_tools_natively`).
        """
        if not self.model.serves_tools_natively:
            yield
            return
        schema = [
            tool
            for tool in self._cached_tools_schema or []
            if tool["function"]["name"] != "finish"
            and not getattr(
                self.function_map.get(tool["function"]["name"]), "runs_between_turns", False
            )
        ]
        turn = self._native_turn
        execute = functools.partial(
            _call_in_context,
            contextvars.copy_context(),
            self._execute_tool_on_helper_thread,
            stop_event=stop_signal.get_thread_stop_event(),
            printer_binding=self.printer.thread_binding() if self.printer else None,
            turn=turn,
        )
        try:
            with self.model.native_tools_exposed(schema, execute):
                yield
        finally:
            # Under the turn's lock: a handler publishing a call's
            # result finishes first (see NativeTurn.lock).  The daemon's
            # forced Stop injects KeyboardInterrupt twice; the bridge's
            # wait above takes the first, and the second can land in
            # this wait for the lock; so can the per-panel Stop's
            # ToolCallInterrupted when the agent runs inline in a tool.
            # Either is held and re-raised once the turn is closed, as
            # the bridge's _close holds its own: an open turn would let
            # that handler publish into the agent's next run.  The wait
            # is bounded by the publication, which is in-memory work
            # (the hooks run outside the lock).  The tools still running
            # for the turn are then interrupted (NativeTurn.running),
            # outside the lock; both steps can be repeated.
            forced_stop: BaseException | None = None
            while True:
                try:
                    locked_call(turn.lock, turn.closed.set)
                    for call in list(turn.running.values()):
                        interrupt_tool_call(call.thread_ident, call_id=call.call_id)
                    break
                except (KeyboardInterrupt, ToolCallInterrupted) as e:
                    forced_stop = e
            if forced_stop is not None:
                raise forced_stop

    def _guarded_execute_tool(
        self, function_call: dict[str, Any], turn: NativeTurn | None = None
    ) -> tuple[str, str, Verdict | None]:
        """Execute one tool call the way the agent loop does: hook, guard, then tool.

        The :attr:`tool_call_hook` is called before EVERY tool call (its
        contract), so it runs first; a refusal is the result the model
        sees.  Allowing means "no objection", not "must execute": the
        framework's :attr:`tool_call_guard` may still block the call.  A
        long-running call keeps the prompt cache warm meanwhile.  The
        :attr:`tool_result_hook` then reads the result (a refusal
        included, :meth:`_call_result_hook`) before it is settled
        (:meth:`_settle_tool_result`): the note of an allowing verdict
        (:func:`~kiss.core.tool_verdict.allow_with_note`) and a pending
        re-issue reminder are appended to it.  A call made inside a
        natively served call (*turn* given: the call a tool of the turn
        redirected to, or its follow-up) is settled under ``turn.lock``,
        and not at all once the turn is closed
        (:meth:`_settle_native_call`): the hook and the refusal
        bookkeeping belong to the run the call was made in, not to the
        one the agent may be running by then.  The served call itself
        is run by :meth:`_execute_tool_on_helper_thread`, which settles
        and records it under the same lock.

        Args:
            function_call: The tool call dict with ``name`` and
                ``arguments``.
            turn: The turn a call nested in a natively served call
                belongs to; ``None`` for a call of the agent loop.

        Returns:
            tuple[str, str, Verdict | None]: (function_name, function_response_string,
            blocked) where *blocked* is the refusing verdict when the call
            was not executed (its ``ends_run`` says whether the run ends
            on it), else ``None``.
        """
        name, response, verdict, owed, call = self._execute_tool_through_hooks(function_call, turn)
        self._call_result_hook(call, name, response, verdict, turn)
        if turn is None:
            return self._settle_tool_result(name, response, verdict, owed)
        return locked_call(turn.lock, self._settle_native_call, turn, name, response, verdict, owed)

    def _execute_tool_through_hooks(
        self, function_call: dict[str, Any], turn: NativeTurn | None
    ) -> tuple[str, str, Verdict, str | None, dict[str, Any]]:
        """Run the hook, the guard, then the tool: :meth:`_guarded_execute_tool` unsettled.

        Args:
            function_call: The tool call dict with ``name`` and
                ``arguments``.
            turn: The native turn the call belongs to, for the calls a
                tool of the turn redirects to; ``None`` for a call of
                the agent loop.

        Returns:
            tuple[str, str, Verdict, str | None, dict[str, Any]]: The
            tool's name and result (a refusal included), the hook's or
            guard's verdict on the call, :attr:`_refused_call` as it was
            before the call -- what :meth:`_settle_tool_result` takes --
            and the call as it was made: *function_call* with the
            arguments the hook, the guard and the tool saw, which is
            what the ``tool_result_hook`` must be given
            (:meth:`_call_result_hook`) so that it books the result
            under the arguments that ran.
        """
        name = function_call["name"]
        args = _call_args(function_call)
        # A compaction stub the model passed as an argument is a reference,
        # not content: the tool runs on the original (a new call dict, so
        # the conversation's own record of the call stays as sent), or the
        # call is refused before anything is written, run or matched.
        args, restored, refusal = expand_stubs(args, self._compaction_table)
        if refusal is not None:
            verdict = Verdict(False, refusal)
        else:
            if restored:
                function_call = {**function_call, "arguments": args}
            verdict = self._tool_call_verdict(name, args, turn)
        admitted, owed = (
            self._admit_call(None, name, verdict)
            if turn is None
            else locked_call(turn.lock, self._admit_call, turn, name, verdict)
        )
        if not admitted:
            return name, refusal_result(name, "The turn has ended."), verdict, None, function_call
        if not verdict.allowed:
            name, response = self._execute_tool(function_call, blocked=verdict.text, turn=turn)
        elif turn is None and is_long_running_call(name, args):
            name, response = self._execute_tool_keeping_cache_warm(function_call)
        else:
            name, response = self._execute_tool(function_call, turn=turn)
        if restored and verdict.allowed:
            response = f"{response}\n[{'; '.join(restored)}]"
        return name, response, verdict, owed, function_call

    def _admit_call(
        self, turn: NativeTurn | None, name: str, verdict: Verdict
    ) -> tuple[bool, str | None]:
        """Book *verdict* on calling *name* and admit the call, unless *turn* has closed.

        The call is counted here (:attr:`tool_calls_made`, ``finish``
        excepted), refused or not.  Called under ``turn.lock`` for a
        natively served call, so the
        check of ``closed`` and the bookkeeping are one step with the
        closing of the turn (see :class:`NativeTurn`): the
        :attr:`tool_call_hook` of a call the turn's end abandoned may
        return long after, when the agent is running another run, whose
        pending refusal and redirect verdict it must leave alone and
        whose tools it must not run.

        Args:
            turn: The native turn the call belongs to; ``None`` for a
                call of the agent loop.
            name: The tool's name.
            verdict: The hook's or guard's verdict on the call.

        Returns:
            tuple[bool, str | None]: Whether the call is admitted
            (``False`` once the turn is closed: nothing was booked and
            the tool must not run), and :attr:`_refused_call` as it was
            before the call.
        """
        if turn is not None and turn.closed.is_set():
            return False, None
        if name != "finish":
            self.tool_calls_made += 1
        owed = self._refused_call
        if not verdict.allowed:
            # A refused finish is owed no re-issue: the gate's verdict says
            # what to do next, and a reminder to "re-issue it now" sent
            # the model straight back into the gate, a wasted round each
            # time.
            self._refused_call = None if name == "finish" else name
        else:
            self._redirect_ended = None
        return True, owed

    def _turn_step[T](
        self, turn: NativeTurn | None, step: Callable[..., T], *args: Any
    ) -> T | None:
        """Take *step*, a read or write of this agent's state, as one step of *turn*.

        What a natively served call does between its hooks: the hooks
        may block, so they are called outside ``turn.lock`` (the Stop
        closes the turn under it), and the agent is touched only in
        steps taken under the lock while the turn is open.  A call the
        turn's end abandoned then reads no hook of the agent's next run
        and writes nothing to it.

        Args:
            turn: The native turn the call belongs to; ``None`` for a
                call of the agent loop, whose *step* is simply called.
            step: The step.
            *args: Its arguments.

        Returns:
            What *step* returned; ``None`` when it did not run because
            the turn is closed.
        """
        if turn is None:
            return step(*args)
        return locked_call(turn.lock, _unless_closed, turn, step, *args)

    def _call_result_hook(
        self,
        function_call: dict[str, Any],
        name: str,
        response: str,
        verdict: Verdict,
        turn: NativeTurn | None,
    ) -> None:
        """Give the :attr:`tool_result_hook` the result of a call before it is settled.

        The hook reads the tool's result as it came back (a refusal
        included), before the note of an allowing verdict and a pending
        re-issue reminder are appended to it (:meth:`_settle_tool_result`),
        and whether the call was refused: by the hook or guard, or by
        one on a call it redirected to (:attr:`_redirect_ended`).  For a
        natively served call the hook and the refusal are read in one
        step of its *turn* (:meth:`_turn_step`), so a result the turn's
        end abandoned never reaches the hook of the agent's next run,
        and the hook is called outside ``turn.lock``: it may block
        (Sorcar's gate runs git in it) and the Stop closes the turn
        under that lock.

        Args:
            function_call: The call as it was made: ``name`` and the
                ``arguments`` the hook, the guard and the tool saw (a
                compaction stub expanded, :meth:`_execute_tool_through_hooks`).
            name: The tool's name, as :meth:`_execute_tool` returned it.
            response: The tool's result (a refusal included).
            verdict: The hook's or guard's verdict on the call.
            turn: The native turn the call belongs to; ``None`` for a
                call of the agent loop.
        """
        hook, refused = self._turn_step(turn, self._result_hook_for, verdict) or (None, False)
        if hook is not None:
            hook(name, _call_args(function_call), response, refused)

    def _result_hook_for(
        self, verdict: Verdict
    ) -> tuple[Callable[[str, dict[str, Any], str, bool], None] | None, bool]:
        """Return the :attr:`tool_result_hook` and whether a call with *verdict* was refused."""
        return self.tool_result_hook, not verdict.allowed or self._redirect_ended is not None

    def _settle_tool_result(
        self,
        name: str,
        response: str,
        verdict: Verdict,
        owed: str | None,
    ) -> tuple[str, str, Verdict | None]:
        """Settle the result of a call :meth:`_guarded_execute_tool` ran.

        The :attr:`tool_result_hook` has read the result already
        (:meth:`_call_result_hook`).

        Args:
            name: The tool's name, as :meth:`_execute_tool` returned it.
            response: The tool's result (a refusal included).
            verdict: The :attr:`tool_call_hook`'s or
                :attr:`tool_call_guard`'s verdict on the call.
            owed: :attr:`_refused_call` as it was before the call.

        Returns:
            tuple[str, str, Verdict | None]: As :meth:`_guarded_execute_tool`.
        """
        if verdict.allowed:
            # A call this one redirected to (ToolRedirect) was refused or
            # settled the refusal inside: the model owes the re-issue at
            # its next step, not in this result, as for a call of its own.
            settled = self._refused_call != owed
            # A hook ended the run on a call this one redirected to (or
            # on its follow-up): the run ends on this call, whose result
            # (the note, the nested refusal) is what the turn records.
            blocked = self._redirect_ended
        else:
            settled = True
            blocked = verdict
        if verdict.allowed and verdict.text and name != "finish":
            response += verdict.text
        if not settled:
            response = self._with_reissue_reminder(name, response)
        return name, response, blocked

    def _with_reissue_reminder(self, name: str, response: str) -> str:
        """Return *response*, the result of an allowed call of *name*, settling a pending refusal.

        A refused call (:attr:`_refused_call`) is owed a re-issue.  The
        next allowed call settles it: a call of the refused tool itself
        is the re-issue, so its result is returned as is; a call of any
        other tool (the ``summary`` or ``create_worktree`` the refusal
        asked for) gets :data:`~kiss.core.tool_verdict.REISSUE_REMINDER`
        appended, so the model reads, at the very step it chooses what
        to do next, that the refused call still has to be made.  An
        allowed ``finish`` ends the run and its result is the task's
        result, so it carries no reminder; a refused ``finish`` is never
        recorded as owed (:meth:`_guarded_execute_tool`).
        """
        refused = self._refused_call
        self._refused_call = None
        if refused is None or refused == name or name == "finish":
            return response
        return response + REISSUE_REMINDER.format(name=refused)

    def _execute_tool_on_helper_thread(
        self,
        function_call: dict[str, Any],
        stop_event: threading.Event | None,
        printer_binding: Any,
        turn: NativeTurn,
    ) -> tuple[str, str]:
        """Execute a tool call that arrived on a thread other than the agent's.

        Codex calls KISS tools over MCP, and each call reaches the
        bridge of :meth:`_native_tools_exposed` on a fresh server thread
        while the agent's thread waits for the turn to end.  The stop
        event (``kiss.core.stop_signal``), the printer's task routing
        and the event sink's printer (``kiss.core.event_sink``, what a
        tool or hook publishes a gate stage through) are thread-local,
        so they are re-bound here before the call takes the normal
        hook-guard-tool path (:meth:`_execute_tool_through_hooks`, as
        :meth:`_guarded_execute_tool`) — otherwise the tool's events
        would reach no tab, its records would be dropped and the task's
        Stop would not reach the tool.  The agent thread's context
        variables (a :func:`~kiss.core.base.collecting_trajectories`
        scope) come along the same way: the bridge calls this under a
        copy of that context (:func:`_call_in_context`), so a helper
        agent the tool creates belongs to the run like one created on
        the agent's thread.  The call is recorded in the
        trajectory as it happens, since the step's own record comes
        later; the model's conversation is the model's to keep.  The
        limits are checked after every call, as in the ordinary tool
        loop, outside ``turn.lock`` since the check calls hooks
        (:meth:`_check_limits`); a fatal error (a limit hit, a
        sub-agent's budget exceeded) is kept on *turn* for
        :meth:`_execute_step` to raise once the turn ends, and every
        further call of that turn is refused.  A
        run-ending verdict of the ``tool_call_hook`` is kept on *turn*
        the same way, for :meth:`_execute_step` to end the run with;
        every further call of the turn gets a refused result.

        A call the turn's end abandoned (a Stop waited out the tool, see
        :meth:`~kiss.core.models.codex_mcp_bridge.ToolBridge._wait_for_calls`)
        may still return later, while the agent thread is past the turn
        and perhaps saving: ``turn.closed`` tells it to leave the
        conversation alone (:meth:`_publish_native_call`, which checks
        it and publishes under ``turn.lock``, the lock the turn is
        closed under, so the two cannot interleave), and what it counts
        or raises lands on its own, already replaced *turn*, never on
        the running one (a check of ``closed`` before publishing would
        race with the replacement).  *turn* stays with the thread
        (:attr:`_served_turn`): Codex has the pictures of the result
        described on it once this returns, under the limit check of
        the turn (:meth:`_check_limits_for_helper`).
        A ``KeyboardInterrupt`` (the printer raises one under a Stop)
        must not escape either: this is the bridge's thread, not the
        task's, and the bridge records only exceptions.

        Args:
            function_call: The tool call dict with ``name`` and
                ``arguments``.
            stop_event: The agent thread's stop event.
            printer_binding: The agent thread's ``Printer.thread_binding()``.
            turn: The record of the turn the call belongs to.

        Returns:
            tuple[str, str]: (function_name, function_response_string).

        Raises:
            Exception: What the tool or the limit check raised, or a
                ``KISSError`` naming the earlier fatal error or the Stop
                that interrupted the call.
        """
        stop_signal.set_thread_stop_event(stop_event)
        self._served_turn.turn = turn
        if turn.printer is not None:
            turn.printer.adopt_thread_binding(printer_binding)
        if turn.error is not None:
            raise KISSError(f"The run is ending: {turn.error}")
        if turn.ended is not None:
            # A plain refused result, not an error: the bridge would report
            # an error as fatal and the run must end with the verdict.
            name = function_call["name"]
            return name, refusal_result(name, f"The run has ended: {turn.ended.text}")
        previous_printer = bind_printer(turn.printer)
        try:
            name, response, verdict, owed, call = self._execute_tool_through_hooks(
                function_call, turn
            )
            self._call_result_hook(call, name, response, verdict, turn)
            name, response, blocked = locked_call(
                turn.lock,
                self._publish_native_call,
                turn,
                function_call,
                name,
                response,
                verdict,
                owed,
            )
            if blocked is None or not blocked.ends_run:
                self._check_limits(turn)
        except Exception as e:
            turn.error = e
            raise
        except BaseException as e:
            turn.error = KISSError(f"Tool {function_call['name']} stopped: {e}")
            raise turn.error from e
        finally:
            bind_printer(previous_printer)
        return name, response

    def _settle_native_call(
        self,
        turn: NativeTurn,
        name: str,
        response: str,
        verdict: Verdict,
        owed: str | None,
    ) -> tuple[str, str, Verdict | None]:
        """Settle a call made inside one of *turn*'s calls, unless *turn* has closed.

        Called under ``turn.lock``, as :meth:`_publish_native_call` is
        and for the same reason: a nested call the turn's end abandoned
        (along with the call that redirected to it) never settles a
        refusal once the turn is closed.  Nothing is recorded here: the
        redirecting call's result carries the nested one and is
        recorded with it.

        Args:
            turn: The record of the turn the call belongs to.
            name: The tool's name, as :meth:`_execute_tool` returned it.
            response: The tool's result (a refusal included).
            verdict: The hook's or guard's verdict on the call.
            owed: :attr:`_refused_call` as it was before the call.

        Returns:
            tuple[str, str, Verdict | None]: As :meth:`_guarded_execute_tool`;
            the unsettled result when the turn is closed.
        """
        if turn.closed.is_set():
            return name, response, None
        return self._settle_tool_result(name, response, verdict, owed)

    def _publish_native_call(
        self,
        turn: NativeTurn,
        function_call: dict[str, Any],
        name: str,
        response: str,
        verdict: Verdict,
        owed: str | None,
    ) -> tuple[str, str, Verdict | None]:
        """Settle and record a call :meth:`_execute_tool_on_helper_thread` ran, if *turn* is open.

        Called under ``turn.lock`` (see :class:`NativeTurn`): the check
        of ``closed`` and what it guards — the settling of the result
        (:meth:`_settle_tool_result`: the refusal bookkeeping), the
        routine-turn note, the trajectory — are one step with respect
        to the closing of the turn, so a call the turn's end abandoned
        never writes to the agent once the turn is closed (the
        ``tool_result_hook`` ran before and the limit check comes
        after, both outside the lock: :meth:`_turn_step`).

        Args:
            turn: The record of the turn the call belongs to.
            function_call: The tool call dict with ``name`` and
                ``arguments``.
            name: The tool's name, as :meth:`_execute_tool` returned it.
            response: The tool's result (a refusal included).
            verdict: The hook's or guard's verdict on the call.
            owed: :attr:`_refused_call` as it was before the call.

        Returns:
            tuple[str, str, Verdict | None]: As :meth:`_guarded_execute_tool`;
            the unsettled result when the turn is closed.
        """
        if turn.closed.is_set():
            return name, response, None
        name, response, blocked = self._settle_tool_result(name, response, verdict, owed)
        self._mark_routine_turn(function_call, response, blocked is not None)
        turn.calls += 1
        self._add_message(
            "model", render_tool_call(function_call["name"], _call_args(function_call))
        )
        self._add_message("user", f"[{name}]: {response}")
        if blocked is not None and blocked.ends_run:
            turn.ended = blocked
        return name, response, blocked

    def _mark_routine_turn(self, function_call: dict[str, Any], result: str, refused: bool) -> None:
        """Note whether the next model call follows a routine tool call.

        See :func:`is_routine_turn`.

        Args:
            function_call: The tool call dict with ``name`` and ``arguments``.
            result: The tool's result text (or the refusal text).
            refused: Whether the call was refused by a hook or guard.
        """
        name = function_call["name"]
        if name == "finish":
            return
        # A guard inside the tool (a refused foreground sleep, a refused
        # rewrite) reports itself the way a hook refusal does.
        refused = refused or result.startswith("REFUSED:")
        self._routine_next_turn = is_routine_turn(name, _call_args(function_call), result, refused)

    def _execute_tool_keeping_cache_warm(self, function_call: dict[str, Any]) -> tuple[str, str]:
        """Run a tool call that may outlive the prompt cache, pinging the cache meanwhile.

        Every ping is a billed request; its cost is added to the budget
        here, on the agent's thread (the usage totals have a single
        writer), once the tool has returned, whether normally or by an
        interrupt or limit error.  Until then the keep-alive counts the
        pings itself and sends none the budget does not allow
        (:meth:`_budget_allows_ping`).  Only for a call of the agent loop: a
        natively served one runs on the bridge's thread, may return after
        its turn has ended, and its model has no cache to ping.

        Args:
            function_call: The tool call dict from the model response.

        Returns:
            The ``(name, result)`` pair of :meth:`_execute_tool`.
        """
        if self._keeping_cache_warm:
            # Nested in a call already kept warm (a ToolRedirect's
            # follow-up): its pings cover this call, and the model object
            # takes one ping thread at a time.
            return self._execute_tool(function_call)
        keep_alive = PromptCacheKeepAlive(
            self.model,
            self.function_map,
            self._cached_tools_schema,
            self._prompt_cache_touched_at,
            max_pings=keep_alive_max_pings(self.model_name),
            may_ping=self._budget_allows_ping,
        )
        self._keeping_cache_warm = True
        try:
            with keep_alive:
                return self._execute_tool(function_call)
        finally:
            self._keeping_cache_warm = False
            # A Stop landing on ``__exit__``'s entry skips its join: stop
            # the pings again here (idempotent) before reading them.
            keep_alive.stop()
            # The step footer reports the step's generation call, so its
            # usage is restored once the pings have been billed.
            generation_usage = self.last_call_usage
            for ping in keep_alive.responses:
                # A keep-alive ping is a billed model call too; its
                # duration is not measured (0).
                self.last_call_usage = None
                self._update_tokens_and_budget_from_response(ping)
                self._print_llm_call(time.time())
            self.last_call_usage = generation_usage
            if keep_alive.last_ping_at is not None:
                self._prompt_cache_touched_at = keep_alive.last_ping_at
                logger.info(
                    "Prompt cache kept warm with %d ping(s) during %s: agent=%s step=%d",
                    len(keep_alive.responses),
                    function_call["name"],
                    self.name,
                    self.step_count,
                )

    def _handoff_context_tokens(self) -> float | None:
        """Return the context size at which this agent hands off, or ``None`` if unknown."""
        try:
            return CONTEXT_LIMIT_FRACTION * get_max_context_length(self.model.model_name)
        except KISSError:
            return None

    def _compaction_target(self) -> int | None:
        """Return the steady-state context target in tokens, ``None`` when the target is off."""
        fraction = DEFAULT_CONFIG.compaction_target_fraction
        if not 0.0 < fraction <= 1.0:
            return None
        try:
            window: int | None = get_max_context_length(self.model.model_name)
        except KISSError:
            window = None
        return compaction_target_tokens(window, DEFAULT_CONFIG.compaction_start_tokens, fraction)

    def _maybe_compact_conversation(self) -> None:
        """Compact old tool outputs and arguments once the context is past its threshold.

        Runs before a model call when the global Compaction setting
        (``DEFAULT_CONFIG.tool_output_compaction``) and the run's
        ``compaction`` option (:attr:`compaction`) are both on and the
        last response's context size reached
        ``_next_compaction_at`` (100k tokens, then every further 100k of
        growth) or the steady-state target (:meth:`_compaction_target`,
        past which every step is a candidate), and at least
        :data:`~kiss.core.context_compaction.COMPACTION_MIN_STEP_GAP`
        steps passed since the last compaction.  The compaction is a
        full prompt-cache miss, so it is applied only when
        :func:`~kiss.core.context_compaction.should_compact` finds the
        drop worth it; a skipped compaction is re-planned on every later
        step, since more results age out of the kept window.  Every
        compaction applied is published as a ``compaction`` event with
        the tokens it removed, so the event log shows what the context
        would have cost.  A model that asks for it
        (:meth:`Model.clears_tool_uses_in_bulk`) is cleared by the rule of
        :meth:`_maybe_clear_tool_uses` instead of these thresholds.
        """
        if not (DEFAULT_CONFIG.tool_output_compaction and self.compaction):
            return
        if self.model.clears_tool_uses_in_bulk():
            self._maybe_clear_tool_uses()
            return
        target = self._compaction_target()
        above_target = target is not None and self.context_tokens_used >= target
        if self.context_tokens_used < self._next_compaction_at and not above_target:
            return
        if self.step_count - self._last_compaction_step < COMPACTION_MIN_STEP_GAP:
            return
        whole_turns = self.model.compacts_whole_turns()
        removal = plan_turn_removal(self.model.conversation) if whole_turns else None
        plan = [] if whole_turns else plan_compaction(
            self.model.conversation, table=self._compaction_table
        )
        dropped_tokens = (removal.dropped if removal else dropped_chars(plan)) // CHARS_PER_TOKEN
        if not should_compact(
            self.context_tokens_used,
            dropped_tokens,
            self._handoff_context_tokens(),
            target,
            get_cache_discount(self.model.model_name),
        ):
            logger.debug(
                "Skipped compaction dropping ~%d of %d context tokens: agent=%s step=%d",
                dropped_tokens,
                self.context_tokens_used,
                self.name,
                self.step_count,
            )
            return
        self._next_compaction_at = self.context_tokens_used + DEFAULT_CONFIG.compaction_step_tokens
        self._last_compaction_step = self.step_count
        if removal is not None:
            # The removed items all precede the messages the llm_call_hook
            # has not seen yet (the newest turn is kept), so its boundary
            # moves back by what was removed, the note put in their place
            # counted as seen.
            removed = apply_turn_removal(self.model.conversation, removal)
            self._llm_hook_conversation_index = max(
                removal.start + 1, self._llm_hook_conversation_index - removed + 1
            )
            logger.info(
                "Removed the first %d turns from the context (~%d of %d tokens): agent=%s step=%d",
                removal.turns, dropped_tokens, self.context_tokens_used, self.name, self.step_count,
            )
            if self.printer:
                self.printer.print(
                    "", type="compaction", step=self.step_count,
                    context_tokens=self.context_tokens_used, dropped_tokens=dropped_tokens,
                    results=0, arguments=0, turns=removal.turns, target_tokens=target,
                )
            if self.context_reset_hook is not None:
                self.context_reset_hook()
            return
        self._apply_stubs(plan, dropped_tokens, target)

    def _maybe_clear_tool_uses(self) -> None:
        """Stub every tool call but the newest, and drop the reasoning, past the trigger.

        The rule of a model whose requests read the conversation from a
        prompt cache (:meth:`Model.clears_tool_uses_in_bulk`): an edit of
        an earlier message makes the provider write everything after it
        again, so nothing is edited until the last response's context
        reached the trigger.  Then the results and arguments of every
        tool call but the last ``Config.context_edit_keep_tool_uses`` are
        stubbed and the reasoning of the earlier replies is removed
        (:meth:`Model.drop_stale_reasoning`) in one edit of the stored
        conversation, when the two together remove at least the minimum
        and :data:`~kiss.core.context_compaction.COMPACTION_MIN_STEP_GAP`
        steps passed since the last edit.  The trigger and the minimum are
        ``Config.context_edit_trigger_tokens`` and
        ``Config.context_edit_clear_at_least`` at the model's cache prices
        (:func:`~kiss.core.context_compaction.clear_thresholds`).  Between
        two such edits a request is the one before it with new messages at
        its end, which is what the provider's cache needs to be read.
        """
        config = DEFAULT_CONFIG
        trigger, at_least = clear_thresholds(
            config.context_edit_trigger_tokens,
            config.context_edit_clear_at_least,
            get_cache_write_ratio(self.model.model_name),
            self._handoff_context_tokens(),
        )
        if self.context_tokens_used < trigger:
            return
        if self.step_count - self._last_compaction_step < COMPACTION_MIN_STEP_GAP:
            return
        plan = plan_compaction(
            self.model.conversation,
            keep_recent=config.context_edit_keep_tool_uses,
            table=self._compaction_table,
        )
        dropped_tokens = (
            dropped_chars(plan) // CHARS_PER_TOKEN
            + self._reasoning_tokens - self._kept_reasoning_tokens()
        )
        if dropped_tokens < max(1, at_least):
            return
        self._last_compaction_step = self.step_count
        self._apply_stubs(plan, dropped_tokens, None)

    def _kept_reasoning_tokens(self) -> int:
        """Return the reasoning an edit leaves in the conversation: the newest reply's, on a
        model that needs it (:meth:`Model.keeps_newest_reasoning`)."""
        return self._newest_reasoning_tokens if self.model.keeps_newest_reasoning() else 0

    def _apply_stubs(
        self, plan: list[Replacement], dropped_tokens: int, target: int | None
    ) -> None:
        """Stub the planned tool outputs and arguments in the stored conversation.

        The model then removes the reasoning the edit invalidated
        (:meth:`Model.drop_stale_reasoning`), the edit is published as a
        ``compaction`` event and the context-reset hook runs.

        Args:
            plan: The result of ``plan_compaction``.
            dropped_tokens: The tokens the plan removes.
            target: The steady-state target the event reports, ``None``
                when the edit did not come from it.
        """
        results = sum(replacement.kind == "result" for replacement in plan)
        compacted = apply_compaction(plan)
        reasoning_blocks = self.model.drop_stale_reasoning()
        self._reasoning_tokens = self._kept_reasoning_tokens()
        logger.info(
            "Compacted %d old tool outputs and %d arguments (~%d of %d context tokens), "
            "removed %d reasoning blocks: agent=%s step=%d",
            results,
            compacted - results,
            dropped_tokens,
            self.context_tokens_used,
            reasoning_blocks,
            self.name,
            self.step_count,
        )
        if self.printer:
            self.printer.print(
                "",
                type="compaction",
                step=self.step_count,
                context_tokens=self.context_tokens_used,
                dropped_tokens=dropped_tokens,
                results=results,
                arguments=compacted - results,
                reasoning_blocks=reasoning_blocks,
                target_tokens=target,
            )
        if self.context_reset_hook is not None:
            self.context_reset_hook()

    def _tool_call_verdict(
        self, name: str, args: dict[str, Any], turn: NativeTurn | None = None
    ) -> Verdict:
        """Return the verdict of the vetoes on calling *name* with *args*.

        The caller's :attr:`tool_call_hook` is consulted first (it sees
        every call); only when it allows does the framework's
        :attr:`tool_call_guard` get its say (Sorcar blocks ``finish``
        while a user follow-up is queued or a worktree is unmerged).
        Both keep state of their run (Sorcar's guard its coding gate,
        the root SEA's hook the summary it is owed) and must not be
        asked about a call that is not its run's.  So they are read in
        one step of *turn* (:meth:`_turn_step`): a call whose turn is
        closed asks neither.  And the hook may block past the Stop
        that closes *turn* (the bridge abandons the call) and past the
        reset for the next run: the guard is not asked once the turn
        closed while the hook ran.  Either way the call is refused
        when :meth:`_admit_call` sees the turn closed.

        Args:
            name: The tool's name.
            args: The call's arguments.
            turn: The native turn the call belongs to; ``None`` for a
                call of the agent loop.

        Returns:
            Verdict: The hook's refusing verdict (a
            :func:`~kiss.core.tool_verdict.end_run` one ends the run),
            the guard's refusal text as a refusing verdict, the guard's
            own verdict when it returns one (an allowing one with a note
            the call's result ends with), or the hook's allowing verdict
            when neither objects (its ``text``, if any, is such a note)
            or the turn closed while the hook ran; ``ALLOW`` when the
            turn was closed already.
        """
        hook, guard = self._turn_step(turn, self._vetoes) or (None, None)
        verdict = ALLOW
        if hook is not None:
            verdict = hook(name, args)
            if not verdict.allowed:
                return verdict
        if guard is None or (turn is not None and turn.closed.is_set()):
            return verdict
        text = guard(name, args)
        if isinstance(text, Verdict):
            return text
        return verdict if text is None else Verdict(False, text)

    def _vetoes(
        self,
    ) -> tuple[
        Callable[[str, dict[str, Any]], Verdict] | None,
        Callable[[str, dict[str, Any]], str | Verdict | None] | None,
    ]:
        """Return the :attr:`tool_call_hook` and the :attr:`tool_call_guard`."""
        return self.tool_call_hook, self.tool_call_guard

    def _tool_call_refusal(self, name: str, args: dict[str, Any]) -> Verdict | None:
        """Return the refusing verdict on calling *name* with *args*, or ``None`` when allowed.

        :meth:`_tool_call_verdict` without the allowing verdict's note,
        for the implicit-finish probe.
        """
        verdict = self._tool_call_verdict(name, args)
        return None if verdict.allowed else verdict

    def _ended_run_result(self, verdict: Verdict) -> str:
        """Return the run's result for a hook's run-ending *verdict*.

        The verdict's text is the result, in the registered ``finish``
        tool's contract (:meth:`_result_in_finish_contract`) with the
        verdict's ``success``, so the caller reads a hook-ended run like
        a run the model finished.

        Args:
            verdict: The :func:`~kiss.core.tool_verdict.end_run` verdict.
        """
        logger.info(
            "Run ended by a hook: agent=%s step=%d success=%s result=%r",
            self.name,
            self.step_count,
            verdict.success,
            verdict.text[:200],
        )
        return self._result_in_finish_contract(verdict.text, verdict.text, verdict.success)

    def _result_in_finish_contract(self, summary: str, result: str, success: bool) -> str:
        """Return a result the run did not get from a ``finish`` call, in ``finish``'s contract.

        Callers like RelentlessAgent register the structured
        :func:`kiss.core.utils.finish` and parse the result as YAML, so
        returning raw text would silently drop the success/is_continue
        metadata.  The result is terminal (``is_continue=False``), so
        RelentlessAgent does not resume the run.

        Args:
            summary: The ``summary_in_html`` of a
                ``finish(success, is_continue, summary_in_html)`` contract.
            result: The ``result`` of a ``finish(result)`` contract, and
                the result itself when the registered ``finish`` has
                neither parameter (or there is none).
            success: The ``success`` of a structured contract.
        """
        finish_fn = self.function_map.get("finish")
        params: set[str] = set()
        if finish_fn is not None:
            try:
                params = set(inspect.signature(finish_fn).parameters)
            except (TypeError, ValueError):  # pragma: no cover — exotic callable
                params = set()
        if "summary_in_html" in params:
            assert finish_fn is not None
            return str(finish_fn(success=success, is_continue=False, summary_in_html=summary))
        if "result" in params:
            assert finish_fn is not None
            return str(finish_fn(result=result))
        return result

    def _implicit_finish_result(self, explanation: str) -> str:
        """Build the result for an implicit finish.

        Preserves the registered ``finish`` tool's output contract: callers
        like RelentlessAgent register the structured
        :func:`kiss.core.utils.finish` and parse the result as YAML, so
        returning raw status text would silently drop the
        success/is_continue metadata.  The text-only net is terminal
        (``success=True, is_continue=False`` — the text IS the answer, so
        RelentlessAgent must not resume a model that only ever talks).
        *explanation* and the model's last status text form the summary.
        A ``finish(result)`` contract gets the model's last status text
        (or *explanation*) as its ``result``, through the registered
        callable so a wrapper like ``SummarizerFinish.finish`` still
        applies its structure.

        Args:
            explanation: Why the run is being finished implicitly (which
                net fired and for how many turns).

        Returns:
            str: The implicit result in the registered finish contract.
        """
        text = self._last_response_text.strip()
        summary = explanation + (f" Last status from the model: {text}" if text else "")
        return self._result_in_finish_contract(summary, text or explanation, True)

    def _print_tool_call(
        self, name: str, args: dict[str, Any], call_id: int, printer: Printer | None
    ) -> None:
        """Emit the ``tool_call`` event that opens the UI's panel for one call.

        Args:
            name: The tool's name.
            args: The call's arguments.
            call_id: The id the panel's Stop button sends back.
            printer: The call's event sink (see :meth:`_execute_tool`).
        """
        if printer:
            printer.print(name, type="tool_call", tool_input=args, call_id=call_id)

    def _execute_tool(
        self,
        function_call: dict[str, Any],
        blocked: str | None = None,
        turn: NativeTurn | None = None,
    ) -> tuple[str, str]:
        """Execute a single tool call.

        Args:
            function_call: The tool call dict with ``name`` and
                ``arguments``.
            blocked: When not ``None``, the :attr:`tool_call_hook`'s or
                :attr:`tool_call_guard`'s refusal text — the tool is NOT
                executed and :func:`~kiss.core.tool_verdict.refusal_result`
                of it (which opens by saying the call did not run) is
                returned as an error result instead.
            turn: The native turn the call belongs to, as do the calls
                the tool redirects to (:meth:`_execute_redirected`);
                ``None`` for a call of the agent loop.  Once it has
                closed the tool is not run and the call is refused
                (:meth:`_tools_for_call`).

        Returns:
            tuple[str, str]: (function_name, function_response_string).
        """
        function_name = function_call["name"]
        function_args = _call_args(function_call)
        # A natively served call reports to the printer of the turn it
        # belongs to (NativeTurn.printer), not to self.printer: a call the
        # Stop abandoned returns after a reset gave the agent the next
        # run's printer, and its result is not that run's to show.
        printer = self.printer if turn is None else turn.printer
        # The call's token exists before anything can interrupt it, so
        # the ``finally`` below always has it; ``finish`` is never
        # interruptible (its result IS the task's result).
        token = new_tool_call(function_name)
        # Which response of which agent the call came in; an agent built
        # without ``__init__`` has neither an id nor a step.
        token.response = (
            getattr(self, "id", None),
            getattr(self, "step_count", None) if turn is None else None,
        )

        if blocked is not None:
            self._print_tool_call(function_name, function_args, token.call_id, printer)
            refused = redact_secrets(refusal_result(function_name, blocked))
            if printer:
                printer.print(
                    refused,
                    type="tool_result",
                    tool_name=function_name,
                    tool_input=function_args,
                    is_error=True,
                )
            return function_name, refused

        is_error = False
        interrupted = False
        delivered = False
        # The call is registered — inside the try — so the UI's
        # per-panel Stop button can interrupt it (kiss.core.tool_interrupt):
        # the interrupt lands as ToolCallInterrupted, raised by the tool
        # itself at a safe point, injected into it, or — when the tool
        # returned first — drained by end_tool_call, so it can never
        # escape into the agent loop.  Registered BEFORE the ``tool_call``
        # event goes out: the event carries the ``call_id`` the panel's
        # Stop button sends back, and a click that arrived between the
        # two used to find no running call and be rejected.
        try:
            if function_name != "finish":
                register_tool_call(token)
            function_map = self._turn_step(turn, self._tools_for_call, token, turn)
            if function_map is None:
                # The turn closed after the call was admitted: the tools
                # are the next run's by now.
                return function_name, refusal_result(function_name, "The turn has ended.")
            self._print_tool_call(function_name, function_args, token.call_id, printer)
            if token.interrupted:
                # The call started interrupted: a Stop accepted on the
                # call it is nested in landed before it was registered
                # (register_tool_call).  No watchdog need cover it, and
                # the tool may never look at its event, so it is not
                # run; an injection that raced this check is drained.
                raise_if_interrupted()
            try:
                if function_name not in function_map:  # pragma: no cover
                    raise KISSError(f"Function {function_name} is not a registered tool")
                function_response = str(function_map[function_name](**function_args))
            except ToolRedirect as redirect:
                function_response = self._execute_redirected(redirect, turn)
            except BudgetExceededError:
                raise
            except (Exception, SystemExit) as e:
                logger.debug("Exception caught", exc_info=True)
                fn = function_map.get(function_name)
                sig = inspect.signature(fn) if fn else None
                sig_str = f"\nExpected signature: {function_name}{sig}" if sig else ""
                # The error and the hint repeat arguments too (``float`` quotes the
                # text it could not convert, the hint spells out the ``options`` to
                # pass), so the result as a whole is cut as a printer cuts one,
                # after its secrets were replaced: a cut inside one would keep a part.
                function_response = truncate_result(redact_secrets(
                    f"Failed to call {function_name} with {_argument_excerpts(function_args)}: "
                    f"{e}{sig_str}{unknown_argument_hint(sig, function_args, fn)}\n"
                ))
                is_error = True
            end_tool_call(token)
        except ToolCallInterrupted as exc:
            delivered = True
            if isinstance(exc.__context__, KeyboardInterrupt | BudgetExceededError):
                # The interrupt landed while a task Stop or a budget
                # overrun was already unwinding through the tool: that
                # outcome is the one that matters, and it must not be
                # downgraded to a continued task.
                raise exc.__context__ from None
            logger.info("Tool call %s interrupted by the user", function_name)
            function_response = USER_INTERRUPTED_MESSAGE
            is_error = False
            interrupted = True
        finally:
            # The store comes first, before any call that could give a
            # pending injection a landing spot: from here on the
            # watchdog never injects into the unwinding below.
            token.closing = True
            try:
                # Waits out an injection in progress, so ``injected`` is
                # final; an injection that happened before the store
                # (a task Stop or budget overrun raised by the tool
                # while the watchdog fired) is pending on this thread
                # and lands here or in the drain, not later up the stack.
                unregister_tool_call(token)
                if token.injected and not delivered:
                    drain_pending_interrupt()
            except ToolCallInterrupted:
                # The Stop / budget error already unwinding wins; the
                # landing may have pre-empted the registry pop.
                unregister_tool_call(token)
            if turn is not None:
                turn.running.pop(token.call_id, None)

        # What the tool printed is stored with the event and sent to the model's
        # provider: a secret of the key store goes there as its variable's name.
        function_response = redact_secrets(function_response)
        if printer:
            printer.print(
                function_response,
                type="tool_result",
                tool_name=function_name,
                tool_input=function_args,
                is_error=is_error,
                interrupted=interrupted,
            )
        if function_name == "finish":
            self._finish_failed = is_error

        return function_name, function_response

    def _tools_for_call(
        self, token: ToolCallToken, turn: NativeTurn | None
    ) -> dict[str, Callable[..., Any]]:
        """Return the tools to run *token*'s call with; a step of its *turn* (:meth:`_turn_step`).

        A natively served call is entered in ``turn.running`` here, in
        one step with the check that its turn is open.  The end of the
        turn then either precedes the step, and the call is refused (the
        agent's tools may be the next run's by then), or follows it and
        interrupts the call (:meth:`_native_tools_exposed`).

        Args:
            token: The call's token.
            turn: The native turn the call belongs to; ``None`` for a
                call of the agent loop.
        """
        if turn is not None:
            turn.running[token.call_id] = token
        return self.function_map

    def _execute_redirected(self, redirect: ToolRedirect, turn: NativeTurn | None) -> str:
        """Make the call a tool raised *redirect* for, as if the model had issued it.

        The call goes through :meth:`_guarded_execute_tool`, so the
        :attr:`tool_call_hook`, the :attr:`tool_call_guard`, the
        ``tool_call``/``tool_result`` events and the refusal bookkeeping
        all apply to it, and so does the :attr:`ToolRedirect.followup`
        call (the rest of a compound command), made the same way once
        the redirected call went through: not refused, not interrupted,
        and its result opening with neither ``Error`` (the tools'
        convention for a call that failed) nor ``Failed to call`` (the
        tool raised).  Only the original call counts in
        :attr:`tool_calls_made`.

        A run-ending verdict on either call is kept on
        :attr:`_redirect_ended`, for :meth:`_guarded_execute_tool` to
        return as the redirecting call's own.  A Stop accepted on the
        redirecting call (the panel the user sees; nested calls run
        under it, see :mod:`kiss.core.tool_interrupt`) is checked before
        the redirected call and after each nested call, so the chain
        ends as interrupted instead of running on.

        The calls of a natively served redirecting call belong to its
        *turn*: they settle only while it is open, and so is the count
        restored and the verdict kept (:meth:`_end_redirect`) -- the
        agent's counters are the next run's once the turn is closed.

        Args:
            redirect: The call to make, the note its result opens with
                and the follow-up call, if any.
            turn: The native turn the redirecting call belongs to;
                ``None`` for a call of the agent loop.

        Returns:
            The note, a newline, and the redirected call's result (the
            refusal text when a hook or guard stopped it); with a
            follow-up, then the line naming it and its result, or the
            line saying it was not made.

        Raises:
            ToolCallInterrupted: When the redirecting call was interrupted.
        """
        counted = self.tool_calls_made
        call = {"name": redirect.name, "arguments": redirect.arguments}
        ended: Verdict | None = None
        try:
            raise_if_interrupted()
            _, response, blocked = self._guarded_execute_tool(call, turn)
            raise_if_interrupted()
            result = f"{redirect.note}\n{response}"
            if redirect.followup is not None:
                name, arguments = redirect.followup
                after = None
                if blocked is None and not response.startswith(
                    ("Error", "Failed to call", USER_INTERRUPTED_MESSAGE)
                ):
                    _, after, blocked = self._guarded_execute_tool(
                        {"name": name, "arguments": arguments}, turn
                    )
                    raise_if_interrupted()
                result += "\n" + followup_result(name, arguments, after, redirect.name)
            if blocked is not None and blocked.ends_run:
                ended = blocked
        finally:
            if turn is None:
                self._end_redirect(turn, counted, ended)
            else:
                locked_call(turn.lock, self._end_redirect, turn, counted, ended)
        return result

    def _end_redirect(self, turn: NativeTurn | None, counted: int, ended: Verdict | None) -> None:
        """Close the books of a :class:`ToolRedirect`'s calls, unless their *turn* has closed.

        Only the redirecting call counts: :attr:`tool_calls_made` goes
        back to *counted*, its value before the redirected calls.  A
        run-ending verdict *ended* on one of them is kept on
        :attr:`_redirect_ended` for the redirecting call's settlement.
        Called under ``turn.lock`` for a natively served call, and not
        applied once the turn is closed: the counters then belong to
        the run the agent was reset for.

        Args:
            turn: The native turn the redirecting call belongs to;
                ``None`` for a call of the agent loop.
            counted: :attr:`tool_calls_made` before the redirected calls.
            ended: The run-ending verdict a redirected call got, if any.
        """
        if turn is not None and turn.closed.is_set():
            return
        self.tool_calls_made = counted
        if ended is not None:
            self._redirect_ended = ended

    def _check_limits(self, turn: NativeTurn | None = None) -> None:
        """Check budget and context limits, raise KISSError if exceeded.

        The step limit is deliberately *not* checked here: it is owned by
        ``_run_agentic_loop``, the only place that advances
        ``step_count``.  Checking it in both places gave one condition
        two differently worded errors, and the copy here could never
        fire because the loop stops first.

        The check that follows a natively served call (*turn* given)
        runs on the bridge's thread, outside ``turn.lock``: it reads and
        writes the agent in steps of the turn (:meth:`_turn_step`) and
        calls the :attr:`helper_charge_hook` and the
        :attr:`budget_check_hook` between them, so a hook that blocks
        does not keep the Stop from closing the turn; the rest of the
        check is then skipped.

        Args:
            turn: The native turn of the served call the check follows;
                ``None`` on the agent's thread.

        Raises:
            KISSError: If the agent's budget or context limit is exceeded.
        """
        self._bill_helper_usage(turn)
        budget_hook = self._turn_step(turn, self._check_budget)
        if budget_hook is not None:
            budget_hook()
        self._turn_step(turn, self._check_context)

    def _check_limits_for_helper(self) -> None:
        """Check the limits before the model pays a helper on this agent's behalf.

        The model's ``limit_check``, asked before each description of
        a tool result's picture.  On the agent's thread it is
        :meth:`_check_limits`.  Codex has the result of a natively
        served call described on the bridge's thread, once the executor
        has returned (``CodexModel._execute_for_codex``): there it is
        the check of that call's turn (:attr:`_served_turn`), taken in
        steps of the turn like the check that follows the call.  A
        call the turn's end abandoned so asks no hook of the agent's
        next run, and gets no description: nobody reads it, and the
        next run would be billed for it.

        Raises:
            KISSError: If the agent's budget or context limit is
                exceeded, or the served call's turn has ended.
        """
        turn: NativeTurn | None = getattr(self._served_turn, "turn", None)
        self._check_limits(turn)
        if turn is not None and turn.closed.is_set():
            raise KISSError("The turn has ended.")

    def _check_budget(self) -> Callable[[], None] | None:
        """Raise if the budget is spent; else return the :attr:`budget_check_hook` to ask next.

        Raises:
            BudgetExceededError: If the agent's budget is exceeded.
        """
        if self.budget_used >= self.max_budget:
            raise BudgetExceededError(f"Agent {self.name} budget exceeded.")
        return self.budget_check_hook

    def _check_context(self) -> None:
        """Raise if the conversation has reached the context limit.

        Raises:
            ContextWindowExceededError: If the context limit is reached.
        """
        handoff_tokens = self._handoff_context_tokens()
        if handoff_tokens is not None and self.context_tokens_used >= handoff_tokens:
            raise ContextWindowExceededError(
                f"Agent {self.name} conversation reached "
                f"{self.context_tokens_used:,} of "
                f"{get_max_context_length(self.model.model_name):,} context tokens "
                f"(limit {CONTEXT_LIMIT_FRACTION:.0%})."
            )

    def _bill_helper_usage(self, turn: NativeTurn | None = None) -> None:
        """Add what the model's helpers spent on this agent's behalf to its totals.

        A model that cannot see has a vision model describe the
        attachments of a tool result (``Model.tool_result_text_and_attachments``);
        those calls happen outside ``_billed_model_call``, so the model keeps
        their cumulative cost and tokens (``Model.helper_usage``) for this
        fold, which precedes every limit check so the budget cap covers
        them, the result event, and every exit of ``run`` (the step
        limit, an error) so the saved totals include them.  They are no
        steps of this agent.  The part above the totals' ``helper_mark``
        is billed and the mark advanced in ONE totals store, and the
        model's record is never consumed, so a stop landing anywhere
        bills each helper dollar exactly once.  A record of another
        epoch (the model was reset or replaced since the mark) is
        unbilled in full.  An agent checked before ``run`` set its
        model, and a duck-typed model stand-in, have nothing to fold.
        Then each helper call is filed under its model
        (:meth:`_file_helper_calls`).

        Args:
            turn: As for :meth:`_check_limits`.
        """
        self._turn_step(turn, self._fold_helper_usage)
        self._file_helper_calls(turn)

    def _fold_helper_usage(self) -> None:
        """Bill the model's helper spend above the totals' ``helper_mark`` and advance the mark."""
        record = _helper_usage_of(getattr(self, "model", None))
        totals = self._usage_totals
        if record is not None and record != totals.helper_mark:
            epoch, cost, tokens = record
            mark_epoch, billed_cost, billed_tokens = totals.helper_mark
            if mark_epoch is epoch:
                cost, tokens = cost - billed_cost, tokens - billed_tokens
            self._usage_totals = totals._replace(
                budget_used=totals.budget_used + max(cost, 0.0),
                total_tokens_used=totals.total_tokens_used + max(tokens, 0),
                helper_mark=record,
            )

    def _file_helper_calls(self, turn: NativeTurn | None = None) -> None:
        """Hand each helper call not yet filed to :attr:`helper_charge_hook`.

        The helper calls (``Model.helper_calls``) emit no ``llm_call`` of
        this agent's, so without the hook the task's split of its spend
        by model would file them under this agent's model.  A call is
        marked filed after the hook returns, so a stop in between hands
        it over again: the hook must record a key once.  A hook that
        fails is logged and the call marked filed (its spend is billed
        either way).  Without a hook, or a model that keeps no record,
        nothing happens.  The hook and the calls are read once for the
        whole filing, and a call that a natively served call is handing
        over is left to it (:meth:`_take_helper_call`).

        Args:
            turn: As for :meth:`_check_limits`; no call is handed over
                or marked once it is closed.
        """
        batch = self._turn_step(turn, self._helper_calls_to_file)
        if batch is None:
            return
        hook, calls = batch
        for key, helper, cost, tokens in calls:
            if not self._turn_step(turn, self._take_helper_call, key, turn):
                continue
            try:
                hook(helper, cost, tokens, key)
            except Exception:
                logger.warning("helper call %s on %s not filed", key, helper, exc_info=True)
            finally:
                if turn is not None:
                    turn.filing.discard(key)
            self._turn_step(turn, self._mark_helper_call_filed, key)

    def _helper_calls_to_file(
        self,
    ) -> tuple[Callable[..., None], tuple[tuple[str, str, float, int], ...]] | None:
        """Return the :attr:`helper_charge_hook` and the model's helper calls, if it has both."""
        hook = self.helper_charge_hook
        calls: Callable[[], tuple[tuple[str, str, float, int], ...]] | None = getattr(
            getattr(self, "model", None), "helper_calls", None
        )
        return (hook, calls()) if hook is not None and callable(calls) else None

    def _take_helper_call(self, key: str, turn: NativeTurn | None) -> bool:
        """Return whether helper call *key* is to be handed to the hook now.

        Not when it is filed, nor while a natively served call is
        handing it over (``NativeTurn.filing``; a served call, *turn*
        given, enters it there): the last filing of ``run`` would wait
        in a hook still running in a call the Stop abandoned.
        """
        filing = (turn or self._native_turn).filing
        if key in self._helper_calls_filed or key in filing:
            return False
        if turn is not None:
            filing.add(key)
        return True

    def _mark_helper_call_filed(self, key: str) -> None:
        """Mark helper call *key* as filed."""
        self._helper_calls_filed = self._helper_calls_filed | {key}

    def _add_functions(self, tools: list[Callable[..., Any]]) -> None:
        """Adds callable tools to the agent's function map.

        Args:
            tools: List of callable functions to register as tools.

        Raises:
            KISSError: If a tool with the same name is already registered.
        """
        for tool in tools:
            if tool.__name__ in self.function_map:
                error_msg = (
                    f"Tool {tool.__name__} already registered for agent "
                    f"{self.name} with id {self.id}."
                )
                raise KISSError(error_msg)
            self.function_map[tool.__name__] = tool

    @contextlib.contextmanager
    def _turn_effort(self) -> Iterator[None]:
        """Lower the model's reasoning effort for one turn, when asked to.

        Two turns ask for it.  ``DEFAULT_CONFIG.routine_turn_effort``
        names the effort of a turn that follows a refused call, a
        ``summary`` or memory tool, a ``bash_job`` / ``agent_job`` wait or
        an unchanged ``Read`` (:func:`is_routine_turn`); empty (the
        default) changes nothing.  :attr:`_retry_effort` names the effort
        of the retry of a request the output-token cap cut short
        (:meth:`_retry_at_low_effort`) and is consumed here.  The
        override is applied only to a model whose catalog entry carries
        a ``thinking`` level (one that accepts an effort parameter),
        through the ``reasoning_effort`` key every adapter reads when it
        builds the request, and restored afterwards.  ``_routine_call``
        and ``_routine_effort`` record the decision for the ``llm_call``
        event so ``cost_report`` can compare routine and other calls.
        """
        self._routine_call = self._routine_next_turn
        self._routine_next_turn = False
        level = self._retry_effort or (
            DEFAULT_CONFIG.routine_turn_effort if self._routine_call else ""
        )
        self._retry_effort = ""
        info = MODEL_INFO.get(self.model.model_name)
        if not (level and info is not None and info.thinking):
            self._routine_effort = ""
            yield
            return
        config = self.model.model_config
        previous = config.get("reasoning_effort", _MISSING)
        config["reasoning_effort"] = level
        self._routine_effort = level
        try:
            yield
        finally:
            if previous is _MISSING:
                config.pop("reasoning_effort", None)
            else:
                config["reasoning_effort"] = previous

    def _generate_with_tools(self) -> tuple[list[dict[str, Any]], str, Any]:
        """One tool-calling model turn with the native tools exposed.

        Returns:
            ``(function_calls, response_text, response)`` as
            ``Model.generate_and_process_with_tools`` returns them.
        """
        with self._native_tools_exposed(), self._turn_effort():
            turn: tuple[list[dict[str, Any]], str, Any] = (
                self.model.generate_and_process_with_tools(
                    self.function_map, tools_schema=self._cached_tools_schema
                )
            )
            return turn

    def _billed_model_call[T: tuple[Any, ...]](self, call: Callable[[], T]) -> T:
        """Run one model request and account its usage whatever happens.

        *call* returns a tuple whose LAST element is the provider's raw
        response (``Model.generate`` and
        ``Model.generate_and_process_with_tools`` both do); on return it
        is billed and its ``llm_call`` event emitted.  A call that
        raises is billed for what the provider charged anyway: a
        truncated, incomplete or refused response, or a timed-out
        whole-task run, raises an ``Exception`` after the partial usage
        is billed and the event emitted, and a Stop (``KeyboardInterrupt``)
        does the same; neither lets an accounting failure replace it.
        A Stop already requested of this thread is raised before the
        request is made: a stopped task pays for no further model call,
        whether or not a printer or a tool is there to notice the stop.

        Args:
            call: The zero-argument model request.

        Returns:
            Whatever *call* returned.

        Raises:
            KeyboardInterrupt: The user pressed Stop.
            ContextWindowExceededError: The provider rejected the request
                for exceeding the model's context window.
        """
        if stop_signal.stop_requested():
            raise stop_error()
        started = time.time()
        self.last_call_usage = None
        try:
            result = call()
        except Exception as e:
            self._bill_failed_call(started, e)
            if _is_context_overflow_error(e):
                raise ContextWindowExceededError(
                    f"Agent {self.name} exceeded the model's context window: {e}"
                ) from e
            raise
        except BaseException as e:
            self._bill_failed_call(started, e)
            raise
        self._update_tokens_and_budget_from_response(result[-1])
        self._print_llm_call(started)
        return result

    def _bill_failed_call(self, started: float, error: BaseException | None = None) -> None:
        """Account a generation call that raised (a provider error or a Stop).

        The model may have been billed for a call that then failed: a
        complete response the adapter rejected (truncated, incomplete,
        refused) or usage observed mid-stream (Claude Code's per-message
        deltas).  :meth:`Model.take_partial_usage_response` hands it over
        exactly once (duck-typed model stand-ins may predate the hook);
        it is billed to this run (not to the adapter's next call) and its
        ``llm_call`` event emitted, so the per-call records add up to the
        task's cost.  Nothing raised here may replace the call's own
        exception: an unpriced model's ``KISSError`` would turn a
        retryable provider error into a terminal one, and a printer
        failure would hide the real cause; both are logged and swallowed.

        Args:
            started: ``time.time()`` taken right before the call.
            error: What the call raised.  A typed cut-off or refusal
                (``OutputCapError``, ``ModelRefusalError``) names the
                provider's ``stop_reason``, which fills the record when
                the billed partial response does not carry it (a streamed
                usage chunk has no ``finish_reason``).
        """
        try:
            take_partial = getattr(self.model, "take_partial_usage_response", None)
            partial = take_partial() if callable(take_partial) else None
            if partial is not None:
                self._update_tokens_and_budget_from_response(partial)
            usage = self.last_call_usage
            stop_reason = getattr(error, "stop_reason", None)
            if usage is not None and stop_reason and not usage.get("stop_reason"):
                usage["stop_reason"] = stop_reason
            self._print_llm_call(started)
        except Exception:
            logger.warning("Billing a failed model call raised", exc_info=True)

    def _update_tokens_and_budget_from_response(self, response: Any) -> None:
        """Updates token counter and budget from API response.

        The tokens and the cost of one response are committed together
        as ONE immutable :class:`_UsageTotals` snapshot stored with a
        single ``STORE_ATTR``.  Separate ``total_tokens_used`` /
        ``budget_used`` stores allowed an asynchronously injected stop
        (``PyThreadState_SetAsyncExc``) to land between them, and the
        parent's ``except BaseException`` recovery bank then recorded a
        permanently torn source triple — tokens without their cost
        (round-4 review, finding 1).
        """
        try:
            usage, call_tokens = self._usage_of_response(response)
            # A CLI turn of several API requests bills their sum, but the
            # context the model works in is one request's size: the CLI
            # adapters report it; any other model's call is one request.
            context_of = getattr(self.model, "context_tokens_from_response", None)
            context_tokens = context_of(response) if callable(context_of) else None
            if not isinstance(context_tokens, int):
                context_tokens = call_tokens
            if context_tokens > 0:
                self.context_tokens_used = context_tokens
            self.last_call_usage = usage
            # ONE store publishes tokens and cost together (see the
            # docstring): a stop injected anywhere in this method now
            # leaves either the previous complete triple (the response
            # is simply unaccounted, exactly as if the stop had landed
            # before the call) or the next complete triple.
            totals = self._usage_totals
            self._usage_totals = totals._replace(
                budget_used=totals.budget_used + usage["cost"],
                total_tokens_used=totals.total_tokens_used + call_tokens,
            )
        except KISSError:
            raise
        except Exception as e:  # pragma: no cover
            logger.debug("Exception caught", exc_info=True)
            logger.error("Error updating tokens and budget from response: %s", e, exc_info=True)

    def _usage_of_response(self, response: Any) -> tuple[dict[str, Any], int]:
        """Return the usage record of one model *response* and its token count.

        The record is what :attr:`last_call_usage` holds for a call;
        its ``cost`` is in USD.  Nothing of the agent is written, so the
        keep-alive's thread prices its pings with it too
        (:meth:`_budget_allows_ping`).
        """
        counts = self.model.extract_input_output_token_counts_from_response(response)
        # Adapters return 4 to 9 counts; the missing trailing ones are 0.
        (
            input_tokens,
            output_tokens,
            text_cache_read,
            cache_write,
            cache_write_1h,
            audio_input,
            audio_output,
            audio_cache_read,
            image_output,
        ) = (*counts, 0, 0, 0, 0, 0)[:9]
        cache_read = text_cache_read + audio_cache_read
        call_tokens = (
            input_tokens
            + output_tokens
            + cache_read
            + cache_write
            + cache_write_1h
            + audio_input
            + audio_output
            + image_output
        )
        # A provider that states what it charged for the call
        # (OpenRouter's ``usage.cost``) is authoritative: the catalog
        # rate is only an estimate of the routed upstream's price.
        # Duck-typed model stand-ins may predate the hook.
        extract_cost = getattr(self.model, "extract_cost_from_response", None)
        reported = extract_cost(response) if callable(extract_cost) else None
        cost = float(reported) if isinstance(reported, int | float) else None
        if cost is None:
            cost = calculate_cost(
                self.model.model_name,
                input_tokens,
                output_tokens,
                text_cache_read,
                cache_write,
                cache_write_1h,
                num_audio_input_tokens=audio_input,
                num_audio_output_tokens=audio_output,
                num_audio_cache_read_tokens=audio_cache_read,
                num_image_output_tokens=image_output,
            )
            # The catalog holds Standard-tier rates; a provider that
            # served a differently priced tier (OpenAI Flex/Fast,
            # Anthropic fast mode or US-only inference) says so in
            # the response.
            tier_multiplier = getattr(self.model, "cost_multiplier_for_response", None)
            factor = tier_multiplier(response) if callable(tier_multiplier) else 1.0
            if isinstance(factor, int | float):
                cost *= float(factor)
        usage = {
            "input_tokens": input_tokens + audio_input,
            "output_tokens": output_tokens + audio_output + image_output,
            "cache_read": cache_read,
            "cache_write": cache_write + cache_write_1h,
            "cost": cost,
            "stop_reason": stop_reason_of(response),
        }
        return usage, call_tokens

    def _budget_allows_ping(self, pings: list[Any]) -> bool:
        """Return whether the budget allows the keep-alive one more ping after *pings*.

        Asked on the keep-alive's thread before each ping, so it only
        reads.  *pings* are the responses of the pings sent so far: they
        are billed on the agent's thread once the tool has returned
        (:meth:`_execute_tool_keeping_cache_warm`), and until then no
        limit check sees them.  Their cost is counted here, against the
        agent's budget and against the task's
        (:attr:`budget_status_hook`), and the :attr:`budget_check_hook`
        is asked: a ping the check after the tool call would end the run
        on keeps a cache warm that nothing reads.  A check hook that
        comes without the task's numbers cannot be shown the unbilled
        pings, so it is asked for the first ping only and none follows
        an unbilled one.  Nothing is raised on that thread: a budget
        that cannot be read allows no ping either.

        Args:
            pings: The keep-alive's responses so far, not billed yet.
        """
        try:
            unbilled = sum(self._usage_of_response(ping)[0]["cost"] for ping in pings)
            budgets = [(self.budget_used, self.max_budget)]
            if self.budget_status_hook is not None:
                budgets.append(self.budget_status_hook())
            elif pings and self.budget_check_hook is not None:
                return False
            if any(spent + unbilled >= cap for spent, cap in budgets):
                return False
            if self.budget_check_hook is not None:
                self.budget_check_hook()
        except Exception:
            # The hook's refusal: the limit check after the tool call raises it again.
            logger.debug("Exception caught", exc_info=True)
            return False
        return True

    def _print_llm_call(self, started: float) -> None:
        """Emit an ``llm_call`` event for the model call that just returned.

        One event per call, carrying the model, the call's own tokens and
        USD cost (:attr:`last_call_usage`) and its wall-clock duration —
        the per-call price record the autorouter's cost tools read back
        from the events table.  Nothing is emitted when the call was not
        billed (no usage in the response) or there is no printer.  A
        printer that raises is logged, not raised: the event records a
        call already made and billed, and the reply of that call is in
        the model's conversation (or the tool a keep-alive ping covered
        has returned), so ending the step here would lose the reply's
        tool calls and have the loop retry with them unanswered.

        Args:
            started: ``time.time()`` taken right before the call.
        """
        usage = self.last_call_usage
        if self.printer is None or usage is None:
            return
        try:
            self.printer.print(
                "",
                type="llm_call",
                model=self.model.model_name,
                step=self.step_count,
                duration_ms=int((time.time() - started) * 1000),
                routine=self._routine_call,
                effort=self._routine_effort,
                **usage,
            )
        except Exception:
            logger.warning("Printing the llm_call event of a model call raised", exc_info=True)

    def _get_usage_info_string(self) -> str:
        """Returns a compact single-line usage information string."""
        try:
            max_tokens = get_max_context_length(self.model.model_name)
            if self.budget_status_hook is not None:
                spent, cap = self.budget_status_hook()
            else:
                spent, cap = self.budget_used, self.max_budget
            result_tokens = self._last_results_chars // CHARS_PER_TOKEN
            added = (
                f"This step's tool output: ~{result_tokens:,} tokens, "
                if self._last_results_chars > 0
                else ""
            )
            usage = self.last_call_usage or {}
            last_cost = float(usage.get("cost") or 0.0)
            step_cost = f"This step's model call: ${last_cost:.4f}, " if last_cost > 0 else ""
            # The output size and stop reason make an exhausted output cap
            # visible in the trajectory instead of inferred from totals.
            output = (
                f"This step's output: {int(usage.get('output_tokens') or 0):,} tokens"
                f" (stop: {usage.get('stop_reason') or 'unknown'}), "
                if usage
                else ""
            )
            return (
                f"Steps: {self.step_count}/{self.max_steps}, "
                f"Context: {self.context_tokens_used:,}/{max_tokens:,} tokens, "
                f"{added}{step_cost}{output}"
                f"Total tokens: {self.total_tokens_used:,}, "
                f"Budget: ${spent:.4f}/${cap:.2f}, "
            )
        except Exception:  # pragma: no cover
            logger.debug("Exception caught", exc_info=True)
            return f"Steps: {self.step_count}/{self.max_steps}"

    def finish(self, result: str) -> str:
        """
        The agent must call this function with the final answer to the task.

        Args:
            result (str): The result generated by the agent.

        Returns:
            Returns the result of the agent's task.
        """
        return result
