# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Core KISS agent implementation with native function calling support."""

from __future__ import annotations

import contextlib
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
    CHARS_PER_TOKEN,
    COMPACTION_MIN_STEP_GAP,
    apply_compaction,
    dropped_chars,
    plan_compaction,
    should_compact,
)
from kiss.core.kiss_error import (
    BudgetExceededError,
    ContextWindowExceededError,
    KISSError,
    ModelRefusalError,
)
from kiss.core.models.model import Attachment, strip_system_cache_break
from kiss.core.models.model_info import (
    MODEL_INFO,
    calculate_cost,
    declared_fallback,
    get_fallback_model,
    get_max_context_length,
    model,
)
from kiss.core.models.stream_abort import is_stall_error, stop_error
from kiss.core.prompt_cache_keepalive import (
    PromptCacheKeepAlive,
    is_long_running_call,
    keep_alive_max_pings,
)
from kiss.core.tool_interrupt import (
    USER_INTERRUPTED_MESSAGE,
    ToolCallInterrupted,
    drain_pending_interrupt,
    end_tool_call,
    new_tool_call,
    register_tool_call,
    unregister_tool_call,
)
from kiss.core.tool_verdict import REISSUE_REMINDER, Verdict, refusal_result
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
# they belong to the provider that just failed and do not travel to the
# implicit OpenRouter-twin fallback.
_ENDPOINT_CONFIG_KEYS = frozenset({"base_url", "api_key", "extra_headers"})
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
    """

    budget_used: float
    total_tokens_used: int
    step_count: int


_ZERO_USAGE = _UsageTotals(0.0, 0, 0)


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
        closed: Set once the turn's bridge has closed; a call that
            returns after that leaves the agent's conversation alone.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.error: Exception | None = None
        self.closed = threading.Event()


class _EmptyModelResponseError(KISSError):
    """Raised when a model repeatedly returns no text and no tool calls.

    This is a recoverable model/adapter failure when the active model has a
    registered fallback: unlike normal prompt/validation ``KISSError``s, the
    agent loop can swap to the fallback model and continue the same task.
    """


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

    Routine turns (P14 of the 2026-10-24 trajectory audit) are the one
    after a refused call, a :data:`ROUTINE_TOOLS` call, a ``bash_job`` /
    ``agent_job`` ``wait`` that ended with the job still running (the
    model only waits again), or a ``Read`` whose window was unchanged
    and already in the context.  Planning, debugging and review turns
    are never routine: a wait that returned a finished job's result
    (a test run, a reviewer's findings) is one of those.

    Args:
        name: The tool that was called.
        args: Its arguments.
        result: Its result text (or the refusal text).
        refused: Whether a hook or guard refused the call.
    """
    if refused or name in ROUTINE_TOOLS:
        return True
    if name in WAIT_TOOLS and str(args.get("action", "")) == "wait":
        # Both tools open with the job's status line (``Job <id>: running
        # (pid …)``, ``The … task is still running after …``); the log
        # tail after it may say anything.
        status_line = result.lstrip().split("\n", 1)[0]
        return "running (pid" in status_line or "still running" in status_line
    return name == "Read" and "already in your context" in result


def _call_args(function_call: dict[str, Any]) -> dict[str, Any]:
    """Return the arguments dict of a function call, or {} if absent/malformed."""
    raw_args = function_call.get("arguments")
    return raw_args if isinstance(raw_args, dict) else {}


def _call_repr(function_call: dict[str, Any]) -> str:
    """Render a tool call as the Python-style invocation the trajectory records."""
    args_str = ", ".join(f"{k}={v!r}" for k, v in _call_args(function_call).items())
    return f"```python\n{function_call['name']}({args_str})\n```"


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


def _with_usage_footer(text: str, usage_info: str) -> str:
    """Return *text* with the turn's usage line appended as a fenced ``text`` block."""
    return text + "\n```text\n" + usage_info + "\n```\n"


def stop_reason_of(response: Any) -> str | None:
    """Why the provider ended *response*: Anthropic's ``stop_reason``, OpenAI's finish reason.

    Chat completions carry ``choices[0].finish_reason``; the Responses
    API ``status`` (``completed``) or ``incomplete_details.reason``
    (``max_output_tokens``).  ``None`` when the response says nothing.
    """
    reason = getattr(response, "stop_reason", None)
    if isinstance(reason, str):
        return reason
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        reason = getattr(choices[0], "finish_reason", None)
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
        usage monitor and by abandoned-child reclaims, the parent's
        session bank) rely on this.
        """
        totals = self._usage_totals
        return (totals.budget_used, totals.total_tokens_used, totals.step_count)

    def reset_usage(self) -> None:
        """Zero this agent's usage counters before a run.

        The ``(budget_used, total_tokens_used, step_count)`` triple is
        replaced with ONE store, so a concurrent :meth:`usage_snapshot`
        reads either the previous totals or all zeros, never a mix.
        Called by :meth:`_reset` and by the task runner before a run
        (``task_runner._zero_usage_counters``).
        """
        self._usage_totals = _ZERO_USAGE
        self.tool_calls_made = 0
        self._refused_call: str | None = None

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.pre_step_hook: Callable[..., None] | None = None
        self.tool_call_guard: Callable[[str, dict[str, Any]], str | None] | None = None
        self.llm_call_hook: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None
        self.tool_call_hook: Callable[[str, dict[str, Any]], Verdict] | None = None
        self.tool_calls_made = 0
        """Tool calls other than ``finish`` the model issued during the
        current run (blocked ones included: the model still acted).
        ``RelentlessAgent`` reads it after a sub-session ends — a
        continuation session that only called ``finish`` made no
        progress."""
        self._refused_call = None
        """The name of the latest refused tool call, until the model
        calls that tool again or reads the re-issue reminder
        (:meth:`_guarded_execute_tool`); ``None`` when nothing is owed."""
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
        self._next_compaction_at = DEFAULT_CONFIG.compaction_start_tokens
        self._last_compaction_step = -COMPACTION_MIN_STEP_GAP
        # Wall-clock start of the last request that read or wrote the
        # provider's prompt cache (a model call or a keep-alive ping).
        self._prompt_cache_touched_at = 0.0
        # Whether the next model call follows a routine tool call (see
        # ``_turn_effort``), and what the last call was marked as.
        self._routine_next_turn = False
        self._routine_call = False
        self._routine_effort = ""
        # Characters of the last step's tool results (footer "+N tokens").
        self._last_results_chars = 0
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
    ) -> None:
        self.model_name = model_name
        self.print_prompts = print_prompts
        self.verbose = verbose if verbose is not None else True
        self.set_printer(printer, verbose=self.verbose)
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
        self._native_turn = NativeTurn()

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
                refuses. Default is None (no hook).

        Returns:
            str: The result of the agent's task.
        """
        self.llm_call_hook = llm_call_hook
        self.tool_call_hook = tool_call_hook
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

            return self._run_agentic_loop()

        finally:
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

        Returns:
            str: The generated response text from the model.
        """
        start_timestamp = int(time.time())
        self.step_count += 1
        response_text, _response = self._billed_model_call(self.model.generate)
        self._add_message(
            "model",
            _with_usage_footer(response_text, self._get_usage_info_string()),
            start_timestamp,
        )
        self._print_result(response_text)
        return str(response_text)

    def _try_switch_to_fallback(self, reason: str = "a non-retryable error") -> str | None:
        """Swap ``self.model`` to the registered fallback model, if any.

        Consulted by :meth:`_run_agentic_loop` after a recoverable
        model-level failure: a non-retryable provider error (model gated /
        deprecated, credit balance too low, etc.) or repeated empty turns
        from a provider adapter.  If
        :func:`kiss.core.models.model_info.get_fallback_model` names a
        fallback for the current model (one declared in the catalog, or
        the model's OpenRouter twin when ``OPENROUTER_API_KEY`` is
        configured), this method:

        1. Guards against repeated swaps within a single run (only one
           fallback is allowed per :meth:`run` invocation).
        1. Rebuilds the model via :func:`kiss.core.models.model_info.model`
           using the ``model_config`` originally passed to :meth:`run`
           and the printer's streaming callbacks.  A fallback declared
           in the catalog keeps the caller's endpoint overrides
           (``base_url``, ``api_key``, ``extra_headers``: it lives where
           the user pointed); the implicit OpenRouter twin is a different
           provider, so those keys are dropped and the twin routes
           through OpenRouter with the configured OpenRouter key instead
           of re-using the endpoint and credentials that just failed.
        1. Copies the primary model's conversation history onto the
           new model so no context is lost.
        1. Rebuilds :attr:`_cached_tools_schema` so
           :meth:`_execute_step` calls the fallback provider's schema
           (Anthropic-vs-OpenAI-vs-Gemini all differ).
        1. Emits a visible ``system_prompt`` event announcing the swap.

        Returns:
            The new model name on a successful swap, or ``None`` when no
            fallback is available, the fallback equals the current
            model, or the one-shot guard has already been consumed.
        """
        if self._fallback_used:
            return None
        new_name = get_fallback_model(self.model_name)
        if not new_name or new_name == self.model_name:
            return None
        fallback_config = self._model_config
        if fallback_config and declared_fallback(self.model_name) is None:
            fallback_config = {
                k: v for k, v in fallback_config.items() if k not in _ENDPOINT_CONFIG_KEYS
            } or None
        old_conversation = list(self.model.conversation)
        token_cb = self.printer.token_callback if self.printer else None
        thinking_cb = self.printer.thinking_callback if self.printer else None
        new_model = model(
            new_name,
            model_config=fallback_config,
            token_callback=token_cb,
            thinking_callback=thinking_cb,
        )
        new_model.initialize("")
        new_model.conversation = old_conversation
        new_model.usage_info_for_messages = self.model.usage_info_for_messages
        self.model = new_model
        old_name = self.model_name
        self.model_name = new_name
        self._cached_tools_schema = self.model._build_openai_tools_schema(self.function_map)
        self._fallback_used = True
        if self.printer:
            self.printer.print(
                f"Model {old_name} returned {reason}; switching to fallback model: {new_name}",
                type="system_prompt",
            )
        return new_name

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
                if result is not None:
                    self._print_result(result)
                    return result
            except KISSError as e:  # pragma: no cover – requires model to fail mid-step
                logger.debug("Exception caught", exc_info=True)
                if isinstance(e, _EmptyModelResponseError | ModelRefusalError):
                    reason = (
                        'a safety refusal (stop_reason="refusal")'
                        if isinstance(e, ModelRefusalError)
                        else "repeated empty responses"
                    )
                    new_name = self._try_switch_to_fallback(reason=reason)
                    if new_name is not None:
                        consecutive_errors = 0
                        self._reset_progress_trackers()
                        continue
                    if isinstance(e, ModelRefusalError) and not self._refusal_retried:
                        # One retry with the task's context stated; a second
                        # refusal is final.
                        self._refusal_retried = True
                        self.model.add_message_to_conversation("user", REFUSAL_RETRY_MSG)
                        self._add_message("user", REFUSAL_RETRY_MSG)
                        continue
                raise
            except Exception as e:
                logger.debug("Exception caught", exc_info=True)
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

        raise KISSError(f"Agent {self.name} exceeded {self.max_steps} steps.")

    def _execute_step(self) -> str | None:
        """Execute a single step in the ReAct loop.

        Returns:
            str | None: The result string if the task is finished, None otherwise.
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
        self._native_turn = turn = NativeTurn()
        try:
            function_calls, response_text, _response = self._billed_model_call(
                self._generate_with_tools
            )
        except Exception:
            if turn.error is not None:
                raise turn.error
            raise
        # ... and again AFTER it returns, so the assistant turn the call
        # appended is not treated as a "new" message on the next call —
        # unless tools ran natively during it: their calls and results
        # were appended by the model and are new to the hook.
        if not turn.calls:
            self._llm_hook_conversation_index = len(self.model.conversation)
        if response_text and response_text.strip():
            self._last_response_text = response_text
        usage_info = self._get_usage_info_string()
        self.model.set_usage_info_for_messages(usage_info)
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

        # A fatal error a natively run tool raised (budget exhausted, a
        # limit hit) ends the step now that the turn is billed and in the
        # trajectory; the ordinary tool loop below does the same.
        if turn.error is not None:
            raise turn.error
        if turn.calls or any(fc["name"] != "finish" for fc in function_calls):
            self._check_limits()

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
                if self._implicit_finish_allowed():
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
            retry_msg = (
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
        function_results: list[tuple[str, dict[str, Any]]] = []
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
                call_reprs.append(_call_repr(fc))
                function_results.append((fc["name"], {"result": str(e)}))
                limit_error = e
                break
            self._mark_routine_turn(fc, response_str, blocked is not None)
            call_reprs.append(_call_repr(fc))
            function_results.append((name, {"result": response_str}))
            if name == "finish" and blocked is None:
                finish_result = response_str
            else:
                try:
                    self._check_limits()
                except KISSError as e:
                    limit_error = e
                    break

        tool_call_timestamp = int(time.time())
        self._add_message(
            "model",
            _with_usage_footer(response_text + "\n" + "\n".join(call_reprs), usage_info),
            start_timestamp,
        )
        self._add_message(
            "user",
            "\n\n".join(f"[{name}]: {result['result']}" for name, result in function_results),
            tool_call_timestamp,
        )
        if limit_error is not None:
            raise limit_error

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

        # Re-read after the tools ran: a tool may have spent (a sub-agent
        # whose cost was folded into the task), and the line appended to
        # these results is what the model reads next.
        # What this step's tool results add to the context (about 4
        # characters per token), shown in the footer so an oversized
        # output is noticed by the model that asked for it.
        self._last_results_chars = sum(len(str(r["result"])) for _, r in function_results)
        self.model.set_usage_info_for_messages(self._get_usage_info_string())
        self.model.add_function_results_to_conversation_and_return(function_results)
        return None

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
            self._execute_tool_on_helper_thread,
            stop_event=stop_signal.get_thread_stop_event(),
            printer_binding=self.printer.thread_binding() if self.printer else None,
            turn=turn,
        )
        try:
            with self.model.native_tools_exposed(schema, execute):
                yield
        finally:
            turn.closed.set()

    def _guarded_execute_tool(self, function_call: dict[str, Any]) -> tuple[str, str, str | None]:
        """Execute one tool call the way the agent loop does: hook, guard, then tool.

        The :attr:`tool_call_hook` is called before EVERY tool call (its
        contract), so it runs first; a refusal is the result the model
        sees.  Allowing means "no objection", not "must execute": the
        framework's :attr:`tool_call_guard` may still block the call.  A
        long-running call keeps the prompt cache warm meanwhile.

        Args:
            function_call: The tool call dict with ``name`` and
                ``arguments``.

        Returns:
            tuple[str, str, str | None]: (function_name, function_response_string,
            blocked) where *blocked* is the refusal text when the call was
            not executed, else ``None``.
        """
        name = function_call["name"]
        args = _call_args(function_call)
        if name != "finish":
            self.tool_calls_made += 1
        blocked: str | None = None
        hook = self.tool_call_hook
        if hook is not None:
            verdict = hook(name, args)
            blocked = None if verdict.allowed else verdict.text
        if blocked is None and self.tool_call_guard is not None:
            blocked = self.tool_call_guard(name, args)
        if blocked is not None:
            self._refused_call = name
            return (*self._execute_tool(function_call, blocked=blocked), blocked)
        if is_long_running_call(name, args):
            name, response = self._execute_tool_keeping_cache_warm(function_call)
        else:
            name, response = self._execute_tool(function_call)
        return name, self._with_reissue_reminder(name, response), None

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
        result, so it carries no reminder.
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
        while the agent's thread waits for the turn to end.  Both the
        stop event (``kiss.core.stop_signal``) and the printer's task
        routing are thread-local, so they are re-bound here before the
        call takes the normal :meth:`_guarded_execute_tool` path —
        otherwise the tool's events would reach no tab and the task's
        Stop would not reach the tool.  The call is recorded in the
        trajectory as it happens, since the step's own record comes
        later; the model's conversation is the model's to keep.  The
        limits are checked after every call, as in the ordinary tool
        loop; a fatal error (a limit hit, a sub-agent's budget exceeded)
        is kept on *turn* for :meth:`_execute_step` to raise once the
        turn ends, and every further call of that turn is refused.

        A call the turn's end abandoned (a Stop waited out the tool, see
        :meth:`~kiss.core.models.codex_mcp_bridge.ToolBridge._wait_for_calls`)
        may still return later, while the agent thread is past the turn
        and perhaps saving: ``turn.closed`` tells it to leave the
        conversation alone, and what it counts or raises lands on its
        own, already replaced *turn*, never on the running one (a check
        of ``closed`` before publishing would race with the replacement).
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
        if self.printer is not None:
            self.printer.adopt_thread_binding(printer_binding)
        if turn.error is not None:
            raise KISSError(f"The run is ending: {turn.error}")
        try:
            name, response, blocked = self._guarded_execute_tool(function_call)
            if turn.closed.is_set():
                return name, response
            self._mark_routine_turn(function_call, response, blocked is not None)
            turn.calls += 1
            self._add_message("model", _call_repr(function_call))
            self._add_message("user", f"[{name}]: {response}")
            self._check_limits()
        except Exception as e:
            turn.error = e
            raise
        except BaseException as e:
            turn.error = KISSError(f"Tool {function_call['name']} stopped: {e}")
            raise turn.error from e
        return name, response

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
        interrupt or limit error.

        Args:
            function_call: The tool call dict from the model response.

        Returns:
            The ``(name, result)`` pair of :meth:`_execute_tool`.
        """
        keep_alive = PromptCacheKeepAlive(
            self.model,
            self.function_map,
            self._cached_tools_schema,
            self._prompt_cache_touched_at,
            max_pings=keep_alive_max_pings(self.model_name),
        )
        try:
            with keep_alive:
                return self._execute_tool(function_call, blocked=None)
        finally:
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

    def _maybe_compact_conversation(self) -> None:
        """Compact old tool outputs once the context crosses the next threshold.

        Runs before a model call when ``DEFAULT_CONFIG.tool_output_compaction``
        is on and the last response's context size reached
        ``_next_compaction_at`` (100k tokens, then every further 100k of
        growth) and at least :data:`~kiss.core.context_compaction.COMPACTION_MIN_STEP_GAP`
        steps passed since the last compaction.  The compaction is a
        full prompt-cache miss, so it is applied only when
        :func:`~kiss.core.context_compaction.should_compact` finds the
        drop worth it; a skipped compaction is re-planned on every later
        step, since more results age out of the kept window.
        """
        if not DEFAULT_CONFIG.tool_output_compaction:
            return
        if self.context_tokens_used < self._next_compaction_at:
            return
        if self.step_count - self._last_compaction_step < COMPACTION_MIN_STEP_GAP:
            return
        plan = plan_compaction(self.model.conversation)
        dropped_tokens = dropped_chars(plan) // CHARS_PER_TOKEN
        if not should_compact(
            self.context_tokens_used, dropped_tokens, self._handoff_context_tokens()
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
        compacted = apply_compaction(plan)
        logger.info(
            "Compacted %d old tool outputs (~%d of %d context tokens): agent=%s step=%d",
            compacted,
            dropped_tokens,
            self.context_tokens_used,
            self.name,
            self.step_count,
        )
        if self.context_reset_hook is not None:
            self.context_reset_hook()

    def _implicit_finish_allowed(self) -> bool:
        """Return whether an implicit finish may end the run right now.

        The text-only-turn implicit finish stands in for a ``finish`` call
        the model never made, so it faces the same two vetoes a real
        ``finish`` call would: the
        framework's :attr:`tool_call_guard` (Sorcar blocks ``finish``
        while a user follow-up is queued, so finishing anyway would drop
        the follow-up) and the caller's :attr:`tool_call_hook` (a
        refusal suppresses the finish).  Both are consulted with
        ``("finish", {})`` in the order a real call uses: the hook runs
        first and, when it rejects, the guard is not consulted at all.

        Returns:
            bool: ``True`` when neither the hook nor the guard objects.
        """
        if self.tool_call_hook is not None and not self.tool_call_hook("finish", {}).allowed:
            return False
        return self.tool_call_guard is None or self.tool_call_guard("finish", {}) is None

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
        finish_fn = self.function_map.get("finish")
        params: set[str] = set()
        if finish_fn is not None:
            try:
                params = set(inspect.signature(finish_fn).parameters)
            except (TypeError, ValueError):  # pragma: no cover — exotic callable
                params = set()
        if "summary_in_html" in params:
            summary = explanation + (f" Last status from the model: {text}" if text else "")
            assert finish_fn is not None
            return str(finish_fn(success=True, is_continue=False, summary_in_html=summary))
        if "result" in params:
            assert finish_fn is not None
            return str(finish_fn(result=text or explanation))
        return text or explanation

    def _print_tool_call(self, name: str, args: dict[str, Any], call_id: int) -> None:
        """Emit the ``tool_call`` event that opens the UI's panel for one call.

        Args:
            name: The tool's name.
            args: The call's arguments.
            call_id: The id the panel's Stop button sends back.
        """
        if self.printer:
            self.printer.print(name, type="tool_call", tool_input=args, call_id=call_id)

    def _execute_tool(
        self,
        function_call: dict[str, Any],
        blocked: str | None = None,
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

        Returns:
            tuple[str, str]: (function_name, function_response_string).
        """
        function_name = function_call["name"]
        function_args = _call_args(function_call)
        # The call's token exists before anything can interrupt it, so
        # the ``finally`` below always has it; ``finish`` is never
        # interruptible (its result IS the task's result).
        token = new_tool_call(function_name)

        if blocked is not None:
            self._print_tool_call(function_name, function_args, token.call_id)
            refused = refusal_result(function_name, blocked)
            if self.printer:
                self.printer.print(
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
            self._print_tool_call(function_name, function_args, token.call_id)
            try:
                if function_name not in self.function_map:  # pragma: no cover
                    raise KISSError(f"Function {function_name} is not a registered tool")
                function_response = str(self.function_map[function_name](**function_args))
            except BudgetExceededError:
                raise
            except (Exception, SystemExit) as e:
                logger.debug("Exception caught", exc_info=True)
                fn = self.function_map.get(function_name)
                sig = inspect.signature(fn) if fn else None
                sig_str = f"\nExpected signature: {function_name}{sig}" if sig else ""
                function_response = (
                    f"Failed to call {function_name} with {function_args}: {e}{sig_str}"
                    f"{unknown_argument_hint(sig, function_args, fn)}\n"
                )
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

        if self.printer:
            self.printer.print(
                function_response,
                type="tool_result",
                tool_name=function_name,
                tool_input=function_args,
                is_error=is_error,
                interrupted=interrupted,
            )

        return function_name, function_response

    def _check_limits(self) -> None:
        """Check budget and context limits, raise KISSError if exceeded.

        The step limit is deliberately *not* checked here: it is owned by
        ``_run_agentic_loop``, the only place that advances
        ``step_count``.  Checking it in both places gave one condition
        two differently worded errors, and the copy here could never
        fire because the loop stops first.

        Raises:
            KISSError: If the agent's budget or context limit is exceeded.
        """
        if self.budget_used >= self.max_budget:
            raise BudgetExceededError(f"Agent {self.name} budget exceeded.")
        if self.budget_check_hook is not None:
            self.budget_check_hook()
        handoff_tokens = self._handoff_context_tokens()
        if handoff_tokens is not None and self.context_tokens_used >= handoff_tokens:
            raise ContextWindowExceededError(
                f"Agent {self.name} conversation reached "
                f"{self.context_tokens_used:,} of "
                f"{get_max_context_length(self.model.model_name):,} context tokens "
                f"(limit {CONTEXT_LIMIT_FRACTION:.0%})."
            )

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
        """Lower the model's reasoning effort for one routine turn, when configured.

        ``DEFAULT_CONFIG.routine_turn_effort`` names the effort of a
        turn that follows a refused call, a ``summary`` or memory tool,
        a ``bash_job`` / ``agent_job`` wait or an unchanged ``Read``
        (:func:`is_routine_turn`); empty (the default) changes nothing.
        The override is applied only to a model whose catalog entry
        carries a ``thinking`` level (one that accepts an effort
        parameter), through the ``reasoning_effort`` key every adapter
        reads when it builds the request, and restored afterwards.
        ``_routine_call`` records the decision for the ``llm_call``
        event so ``cost_report`` can compare routine and other calls.
        """
        self._routine_call = self._routine_next_turn
        self._routine_next_turn = False
        level = DEFAULT_CONFIG.routine_turn_effort
        info = MODEL_INFO.get(self.model.model_name)
        if not (level and self._routine_call and info is not None and info.thinking):
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
            self._bill_failed_call(started)
            if _is_context_overflow_error(e):
                raise ContextWindowExceededError(
                    f"Agent {self.name} exceeded the model's context window: {e}"
                ) from e
            raise
        except BaseException:
            self._bill_failed_call(started)
            raise
        self._update_tokens_and_budget_from_response(result[-1])
        self._print_llm_call(started)
        return result

    def _bill_failed_call(self, started: float) -> None:
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
        """
        try:
            take_partial = getattr(self.model, "take_partial_usage_response", None)
            partial = take_partial() if callable(take_partial) else None
            if partial is not None:
                self._update_tokens_and_budget_from_response(partial)
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
            usage = self.model.extract_input_output_token_counts_from_response(response)
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
            ) = (*usage, 0, 0, 0, 0, 0)[:9]
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
            if call_tokens > 0:
                self.context_tokens_used = call_tokens
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
            self.last_call_usage = {
                "input_tokens": input_tokens + audio_input,
                "output_tokens": output_tokens + audio_output + image_output,
                "cache_read": cache_read,
                "cache_write": cache_write + cache_write_1h,
                "cost": cost,
                "stop_reason": stop_reason_of(response),
            }
            # ONE store publishes tokens and cost together (see the
            # docstring): a stop injected anywhere in this method now
            # leaves either the previous complete triple (the response
            # is simply unaccounted, exactly as if the stop had landed
            # before the call) or the next complete triple.
            totals = self._usage_totals
            self._usage_totals = _UsageTotals(
                totals.budget_used + cost,
                totals.total_tokens_used + call_tokens,
                totals.step_count,
            )
        except KISSError:
            raise
        except Exception as e:  # pragma: no cover
            logger.debug("Exception caught", exc_info=True)
            logger.error("Error updating tokens and budget from response: %s", e, exc_info=True)

    def _print_llm_call(self, started: float) -> None:
        """Emit an ``llm_call`` event for the model call that just returned.

        One event per call, carrying the model, the call's own tokens and
        USD cost (:attr:`last_call_usage`) and its wall-clock duration —
        the per-call price record the autorouter's cost tools read back
        from the events table.  Nothing is emitted when the call was not
        billed (no usage in the response) or there is no printer.

        Args:
            started: ``time.time()`` taken right before the call.
        """
        usage = self.last_call_usage
        if self.printer is None or usage is None:
            return
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

    def _get_usage_info_string(self) -> str:
        """Returns a compact single-line usage information string."""
        try:
            max_tokens = get_max_context_length(self.model.model_name)
            if self.budget_status_hook is not None:
                spent, cap = self.budget_status_hook()
            else:
                spent, cap = self.budget_used, self.max_budget
            added = (
                f"This step's tool output: ~{self._last_results_chars // 4:,} tokens, "
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
