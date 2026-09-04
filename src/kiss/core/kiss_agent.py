# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Core KISS agent implementation with native function calling support."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from kiss.core.base import Base
from kiss.core.kiss_error import (
    BudgetExceededError,
    ContextWindowExceededError,
    KISSError,
    ModelRefusalError,
)
from kiss.core.models.model import Attachment
from kiss.core.models.model_info import calculate_cost, get_max_context_length, model
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
MAX_CONSECUTIVE_NO_TOOL_CALLS = 2
# A turn is "stagnant" when its tool calls AND their results are identical to
# the previous turn's — a done-but-not-finishing model padding status turns
# with a harmless verification call.  Legitimate polling is not stagnant
# because a changing world changes the results.  After
# STAGNANT_TURNS_REMINDER such turns the model is reminded to call finish;
# after STAGNANT_TURNS_FINISH the agent treats the run as an implicit finish.
STAGNANT_TURNS_REMINDER = 3
STAGNANT_TURNS_FINISH = 6
# Default stall timeout (seconds of output silence) for a run-to-completion
# model executing a whole task in one CLI invocation.  The per-turn default
# (300 s, see CLITextModel._cli_turn) is too short for a full agentic run,
# where a single long native command (a build, a test suite) can be silent
# for many minutes.  model_config["timeout"] still overrides it.
CLI_TASK_TIMEOUT_SECONDS = 3600
CONTEXT_LIMIT_FRACTION = 0.9
_CONTEXT_OVERFLOW_PHRASES = (
    "exceeds the context window",
    "prompt is too long",
    "context_length_exceeded",
    "maximum context length",
    "exceeds the maximum number of tokens",
)


class _EmptyModelResponseError(KISSError):
    """Raised when a model repeatedly returns no text and no tool calls.

    This is a recoverable model/adapter failure when the active model has a
    registered fallback: unlike normal prompt/validation ``KISSError``s, the
    agent loop can swap to the fallback model and continue the same task.
    """


def _call_args(function_call: dict[str, Any]) -> dict[str, Any]:
    """Return the arguments dict of a function call, or {} if absent/malformed."""
    raw_args = function_call.get("arguments")
    return raw_args if isinstance(raw_args, dict) else {}


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


if TYPE_CHECKING:  # pragma: no cover
    from kiss.core.printer import Printer


class KISSAgent(Base):
    """A KISS agent using native function calling."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.pre_step_hook: Callable[..., None] | None = None
        self.tool_call_guard: Callable[[str, dict[str, Any]], str | None] | None = None
        self.llm_call_hook: (
            Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None
        ) = None
        self.tool_call_hook: Callable[[str, dict[str, Any]], str] | None = None
        self.context_tokens_used = 0
        self.budget_check_hook: Callable[[], None] | None = None
        self._llm_hook_conversation_index = 0
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
        self._cached_tools_schema: list[dict[str, Any]] | None = None
        self.messages: list[dict[str, Any]] = []
        self.step_count = 0
        self.total_tokens_used = 0
        self.context_tokens_used = 0
        self._llm_hook_conversation_index = 0
        self.budget_used = 0.0
        # ``run_start_timestamp`` is the real wall clock: the saved record
        # pairs it with ``run_end_timestamp``.  The trajectory FILENAME is
        # keyed by (name, id, _trajectory_stamp) in whole seconds; two runs
        # of one instance started in the same second would share a path
        # and the later save would destroy the earlier record, so the
        # stamp strictly increases per run while keeping the same format.
        self.run_start_timestamp = int(time.time())
        self._trajectory_stamp = max(self.run_start_timestamp, self._trajectory_stamp + 1)
        self._reset_progress_trackers()
        self._model_config: dict[str, Any] | None = model_config
        self._fallback_used = False

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
        """Clear the text-only-turn and stagnant-turn counters and last text.

        Called at construction, on every run reset, and after a mid-run
        fallback model swap, so stale counts — and stale status text that
        an implicit finish would otherwise report — never leak from the
        previous model or run into the new one.
        """
        self._consecutive_no_tool_calls = 0
        self._stagnant_call_turns = 0
        self._last_turn_signature: tuple[Any, ...] | None = None
        self._last_response_text = ""

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
        llm_call_hook: (
            Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None
        ) = None,
        tool_call_hook: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> str:
        """
        Runs the agent's main ReAct loop to solve the task.

        Run-to-completion models (``cc/*``, ``codex/*`` — see
        ``Model.runs_task_to_completion``) skip the ReAct loop entirely:
        the whole task, with *system_prompt* appended after
        ``CLI_SYSTEM_PROMPT_HEADER``, is handed to the CLI agent in one
        ``generate()`` call and its final output is returned (wrapped in
        the registered ``finish`` contract).  *tools* are registered but
        never exposed to such a model; it uses its own native tools.

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
                Default is None (no hook).
            tool_call_hook (Callable | None): Optional hook called before every
                tool call with the tool's name and its arguments dict. If it
                returns the string ``"OK"``, the tool executes as usual; any
                other returned string suppresses the tool execution and is
                returned to the model as the tool's result instead. The hook
                runs before (and its rejection takes precedence over) the
                framework's :attr:`tool_call_guard`; an ``"OK"`` verdict does
                not override a guard block. An implicit finish (text-only
                turns or stagnant identical tool calls) also consults the
                hook (with ``("finish", {})``) and is suppressed unless the
                hook returns ``"OK"``. Default is None (no hook).

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
                self.printer.print(system_prompt, type="system_prompt")
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

    def _generate_once(self) -> str:
        """Run one model generation with token/budget accounting and transcript.

        Shared by :meth:`_run_non_agentic` and
        :meth:`_run_task_to_completion`, which differ only in how the
        generated text becomes the run's result.

        Returns:
            str: The generated response text from the model.
        """
        start_timestamp = int(time.time())
        self.step_count += 1

        try:
            response_text, response = self.model.generate()
        except Exception as e:
            # A run that failed mid-stream may still have observed
            # billable usage (e.g. Claude Code's per-message deltas);
            # account it before propagating so a timed-out whole-task run
            # does not erase its known spend.
            partial = self.model.take_partial_usage_response()
            if partial is not None:
                self._update_tokens_and_budget_from_response(partial)
            if _is_context_overflow_error(e):
                raise ContextWindowExceededError(
                    f"Agent {self.name} exceeded the model's context window: {e}"
                ) from e
            raise
        self._update_tokens_and_budget_from_response(response)
        usage_info_str = self._get_usage_info_string()
        self._add_message(
            "model", response_text + "\n```text\n" + usage_info_str + "\n```\n", start_timestamp
        )
        return str(response_text)

    def _print_result(self, result: str) -> None:
        """Emit *result* as the run's terminal ``result`` printer event."""
        if result and self.printer:
            self.printer.print(
                result,
                type="result",
                step_count=self.step_count,
                total_tokens=self.total_tokens_used,
                cost=f"${self.budget_used:.4f}",
            )

    def _run_non_agentic(self) -> str:
        """Run a single generation without tools.

        Returns:
            str: The generated response text from the model.
        """
        response_text = self._generate_once()
        self._print_result(response_text)
        return response_text

    def _run_task_to_completion(self) -> str:
        """Hand the whole task to a run-to-completion model in one shot.

        CLI-backed models (``cc/*``, ``codex/*``) are full coding agents
        with their own native tools, so instead of the turn-by-turn KISS
        tool loop the task is sent in a single ``generate()`` call — the
        system prompt rides inside the prompt, appended to the task after
        ``CLI_SYSTEM_PROMPT_HEADER`` (see ``CLITextModel._build_prompt``).
        KISS tools are not exposed to the CLI; its final message becomes
        the run's result, wrapped in the registered ``finish`` tool's
        output contract when that tool follows the structured
        ``summary_in_html`` signature (callers like RelentlessAgent and
        ChatSorcarAgent parse the result as YAML).

        Also reached when a tool (Sorcar's ``set_model``) swaps the live
        model to a CLI agent mid-run: the conversation so far is then
        flattened into the prompt as a ``[User]/[Assistant]/[Tool Result]``
        transcript (see ``CLITextModel._build_prompt``), so the CLI
        continues from the accumulated context.

        Returns:
            str: The task result in the registered finish contract.
        """
        if "timeout" not in self.model.model_config:
            # A copy, not setdefault: the model may hold the caller's own
            # config dict by reference, and callers rely on their configs
            # never being mutated.
            self.model.model_config = {
                **self.model.model_config,
                "timeout": CLI_TASK_TIMEOUT_SECONDS,
            }
        response_text = self._generate_once()
        result = self._wrap_in_finish_contract(response_text)
        self._print_result(result)
        return result

    def _wrap_in_finish_contract(self, text: str) -> str:
        """Return *text* in the registered ``finish`` tool's output contract.

        When the registered ``finish`` follows the structured
        :func:`kiss.core.utils.finish` signature (has a ``summary_in_html``
        parameter), *text* is wrapped as a successful, non-continuing
        result so YAML-parsing callers keep working; otherwise *text* is
        returned unchanged (the built-in ``finish(result)`` contract is
        plain text).

        Args:
            text: The model's final output for the task.

        Returns:
            str: *text* in the registered finish contract.
        """
        finish_fn, params = self._registered_finish_and_params()
        if "summary_in_html" in params:
            assert finish_fn is not None
            return str(finish_fn(success=True, is_continue=False, summary_in_html=text))
        return text

    def _registered_finish_and_params(
        self,
    ) -> tuple[Callable[..., Any] | None, set[str]]:
        """Return the registered ``finish`` tool and its parameter names.

        Returns:
            tuple: ``(finish_fn, params)`` where *finish_fn* is the
            registered ``finish`` callable (or ``None``) and *params* is
            the set of its parameter names (empty when absent or when the
            signature cannot be inspected).
        """
        finish_fn = self.function_map.get("finish")
        params: set[str] = set()
        if finish_fn is not None:
            try:
                params = set(inspect.signature(finish_fn).parameters)
            except (TypeError, ValueError):  # pragma: no cover — exotic callable
                params = set()
        return finish_fn, params

    def _try_switch_to_fallback(
        self, reason: str = "a non-retryable error"
    ) -> str | None:
        """Swap ``self.model`` to the registered fallback model, if any.

        Consulted by :meth:`_run_agentic_loop` after a recoverable
        model-level failure: a non-retryable provider error (model gated /
        deprecated, credit balance too low, etc.) or repeated empty turns
        from a provider adapter.  If ``MODEL_INFO`` registers a ``fallback``
        for the current model name, this method:

        1. Guards against repeated swaps within a single run (only one
           fallback is allowed per :meth:`run` invocation).
        1. Rebuilds the model via :func:`kiss.core.models.model_info.model`
           using the same ``model_config`` originally passed to
           :meth:`run` (preserving ``base_url``/``api_key`` overrides
           used by end-to-end tests) and the printer's streaming
           callbacks.
        1. Copies the primary model's conversation history onto the
           new model so no context is lost.
        1. Rebuilds :attr:`_cached_tools_schema` so
           :meth:`_execute_step` calls the fallback provider's schema
           (Anthropic-vs-OpenAI-vs-Gemini all differ).
        1. Emits a visible ``system_prompt`` event announcing the swap.

        Returns:
            The new model name on a successful swap, or ``None`` when no
            fallback is registered, the fallback equals the current
            model, or the one-shot guard has already been consumed.
        """
        from kiss.core.models.model_info import get_fallback_model
        if self._fallback_used:
            return None
        new_name = get_fallback_model(self.model_name)
        if not new_name or new_name == self.model_name:
            return None
        old_conversation = list(self.model.conversation)
        token_cb = self.printer.token_callback if self.printer else None
        thinking_cb = self.printer.thinking_callback if self.printer else None
        new_model = model(
            new_name,
            model_config=self._model_config,
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
                f"Model {old_name} returned {reason}; "
                f"switching to fallback model: {new_name}",
                type="system_prompt",
            )
        return new_name

    def _run_agentic_loop(self) -> str:
        consecutive_errors = 0
        # The step bound lives here and nowhere else: driving the loop
        # off ``step_count`` (rather than a fixed ``range``) keeps it
        # correct even if steps are ever counted from outside the loop,
        # and leaves exactly one message for "out of steps".
        while self.step_count < self.max_steps:
            # Checked every iteration, not just before the loop: a tool
            # (Sorcar's set_model) can swap the live model mid-run, and a
            # switch INTO a CLI agent must hand it the remaining task in
            # one shot rather than resume turn-by-turn tool prompting.
            if self.model.runs_task_to_completion:
                return self._run_task_to_completion()
            self.step_count += 1
            self._check_limits()
            try:
                result = self._execute_step()
                consecutive_errors = 0
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
                raise
            except Exception as e:
                logger.debug("Exception caught", exc_info=True)
                if _is_context_overflow_error(e):
                    raise ContextWindowExceededError(
                        f"Agent {self.name} exceeded the model's context window: {e}"
                    ) from e
                if not _is_retryable_error(e):
                    new_name = self._try_switch_to_fallback(
                        reason="a non-retryable error"
                    )
                    if new_name is None:
                        raise KISSError(f"Non-retryable error from model: {e}") from e
                    consecutive_errors = 0
                    self._reset_progress_trackers()
                    continue
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
        if self.llm_call_hook is not None:
            hook_start = self._llm_hook_conversation_index
            modified_messages = self.llm_call_hook(
                list(self.model.conversation[hook_start:])
            )
            self.model.conversation[hook_start:] = list(modified_messages)
        # Advance the boundary BEFORE the call so a raising call (retryable
        # provider error, refusal, fallback swap) never re-presents
        # already-hooked messages to the hook on the next attempt ...
        self._llm_hook_conversation_index = len(self.model.conversation)
        function_calls, response_text, response = self.model.generate_and_process_with_tools(
            self.function_map, tools_schema=self._cached_tools_schema
        )
        # ... and again AFTER it returns, so the assistant turn the call
        # appended is not treated as a "new" message on the next call.
        self._llm_hook_conversation_index = len(self.model.conversation)
        if response_text and response_text.strip():
            self._last_response_text = response_text
        self._update_tokens_and_budget_from_response(response)
        usage_info = self._get_usage_info_string()
        self.model.set_usage_info_for_messages(usage_info)
        if self.printer:
            self.printer.print(
                usage_info,
                type="usage_info",
                total_tokens=self.total_tokens_used,
                cost=f"${self.budget_used:.4f}",
                total_steps=self.step_count,
            )

        if function_calls and any(fc["name"] != "finish" for fc in function_calls):
            self._check_limits()

        if not function_calls:
            self._consecutive_no_tool_calls += 1
            self._add_message(
                "model", response_text + "\n```text\n" + usage_info + "\n```\n", start_timestamp
            )
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
                        f"without calling finish.",
                        success=True,
                        is_continue=False,
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
        turn_had_blocked_call = False

        for fc in function_calls:
            blocked: str | None = None
            # The hook is called before EVERY tool call (its contract), so it
            # runs first; a non-"OK" verdict is the result the model sees.
            # An "OK" verdict means "no objection", not "must execute": the
            # framework's tool_call_guard may still block the call.
            if self.tool_call_hook is not None:
                hook_verdict = self.tool_call_hook(fc["name"], _call_args(fc))
                if hook_verdict != "OK":
                    blocked = hook_verdict
            if blocked is None and self.tool_call_guard is not None:
                blocked = self.tool_call_guard(fc["name"], _call_args(fc))
            if blocked is not None:
                turn_had_blocked_call = True
            name, response_str = self._execute_tool(fc, blocked=blocked)
            args_str = ", ".join(f"{k}={v!r}" for k, v in _call_args(fc).items())
            call_reprs.append(f"```python\n{name}({args_str})\n```")
            function_results.append((name, {"result": response_str}))
            if name == "finish" and blocked is None:
                finish_result = response_str
            else:
                self._check_limits()

        if turn_had_blocked_call:
            # A guard rejected a call this turn: the guard is deliberately
            # steering the model (e.g. blocking finish until a pending user
            # message is handled), so such turns must never escalate to an
            # implicit finish that would bypass the guard.
            self._stagnant_call_turns = 0
            self._last_turn_signature = None
        else:
            turn_signature: tuple[Any, ...] = (
                tuple(
                    (fc["name"], repr(sorted(_call_args(fc).items()))) for fc in function_calls
                ),
                tuple(result["result"] for _, result in function_results),
            )
            if turn_signature == self._last_turn_signature:
                self._stagnant_call_turns += 1
            else:
                self._stagnant_call_turns = 1
                self._last_turn_signature = turn_signature

        model_content = (
            response_text + "\n" + "\n".join(call_reprs) + "\n```text\n" + usage_info + "\n```\n"
        )
        tool_call_timestamp = int(time.time())
        self._add_message("model", model_content, start_timestamp)
        self._add_message(
            "user",
            "\n\n".join(f"[{name}]: {result['result']}" for name, result in function_results),
            tool_call_timestamp,
        )

        if finish_result is not None:
            logger.info(
                "finish() called: agent=%s step=%d budget=$%.4f "
                "tokens=%d result=%r",
                self.name,
                self.step_count,
                self.budget_used,
                self.total_tokens_used,
                finish_result[:200] if len(finish_result) > 200 else finish_result,
            )
            return finish_result

        self.model.add_function_results_to_conversation_and_return(function_results)

        if self._stagnant_call_turns >= STAGNANT_TURNS_FINISH and self._implicit_finish_allowed():
            logger.info(
                "Implicit finish: agent=%s step=%d repeated identical tool "
                "call(s) with identical results for %d consecutive turns",
                self.name,
                self.step_count,
                self._stagnant_call_turns,
            )
            return self._implicit_finish_result(
                f"The session stalled: the model repeated the identical tool "
                f"call(s) with identical results for {self._stagnant_call_turns} "
                f"consecutive turns without calling finish.",
                success=False,
                is_continue=True,
            )
        if self._stagnant_call_turns >= STAGNANT_TURNS_REMINDER:
            reminder = (
                f"**You have repeated the identical tool call(s) with "
                f"identical results for {self._stagnant_call_turns} "
                "consecutive turns; this makes no progress. If the task is "
                "complete, call the `finish` tool now with your final "
                "result. If you are waiting on a long-running process, "
                "vary your action (e.g. sleep before checking again). "
                "Otherwise, take a different action.**"
            )
            self.model.add_message_to_conversation("user", reminder)
            self._add_message("user", reminder)
        return None

    def _implicit_finish_allowed(self) -> bool:
        """Return whether an implicit finish may end the run right now.

        Both implicit-finish nets — text-only turns and stagnant identical
        tool calls — stand in for a ``finish`` call the model never made,
        so they face the same two vetoes a real ``finish`` call would: the
        framework's :attr:`tool_call_guard` (Sorcar blocks ``finish``
        while a user follow-up is queued, so finishing anyway would drop
        the follow-up) and the caller's :attr:`tool_call_hook` (anything
        but ``"OK"`` suppresses the finish).  Both are consulted with
        ``("finish", {})`` in the order a real call uses: the hook runs
        first and, when it rejects, the guard is not consulted at all.

        Returns:
            bool: ``True`` when neither the hook nor the guard objects.
        """
        if self.tool_call_hook is not None and self.tool_call_hook("finish", {}) != "OK":
            return False
        return self.tool_call_guard is None or self.tool_call_guard("finish", {}) is None

    def _implicit_finish_result(self, explanation: str, *, success: bool, is_continue: bool) -> str:
        """Build the result for an implicit finish.

        Preserves the registered ``finish`` tool's output contract: callers
        like RelentlessAgent register the structured
        :func:`kiss.core.utils.finish` and parse the result as YAML, so
        returning raw status text would silently drop the
        success/is_continue metadata.  For that contract the caller states
        the outcome: the text-only net is terminal (``success=True,
        is_continue=False`` — the text IS the answer, so RelentlessAgent
        must not resume a model that only ever talks), the stagnation net
        is resumable (``success=False, is_continue=True``).  *explanation*
        and the model's last status text form the summary.  The built-in
        ``finish(result)`` contract (plain text) gets the model's last
        status text.

        Args:
            explanation: Why the run is being finished implicitly (which
                net fired and for how many turns).
            success: ``success`` value for the structured contract.
            is_continue: ``is_continue`` value for the structured contract.

        Returns:
            str: The implicit result in the registered finish contract.
        """
        text = self._last_response_text.strip()
        finish_fn, params = self._registered_finish_and_params()
        if "summary_in_html" in params:
            summary = explanation + (f" Last status from the model: {text}" if text else "")
            assert finish_fn is not None
            return str(
                finish_fn(success=success, is_continue=is_continue, summary_in_html=summary)
            )
        return text or explanation

    def _execute_tool(
        self,
        function_call: dict[str, Any],
        blocked: str | None = None,
    ) -> tuple[str, str]:
        """Execute a single tool call.

        Args:
            function_call: The tool call dict with ``name`` and
                ``arguments``.
            blocked: When not ``None``, the :attr:`tool_call_guard`
                rejection message — the tool is NOT executed and
                *blocked* is returned as an error result instead.

        Returns:
            tuple[str, str]: (function_name, function_response_string).
        """
        function_name = function_call["name"]
        function_args = _call_args(function_call)

        if self.printer:
            self.printer.print(function_name, type="tool_call", tool_input=function_args)

        if blocked is not None:
            if self.printer:
                self.printer.print(
                    blocked,
                    type="tool_result",
                    tool_name=function_name,
                    tool_input=function_args,
                    is_error=True,
                )
            return function_name, blocked

        is_error = False
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
                f"Failed to call {function_name} with {function_args}: {e}{sig_str}\n"
            )
            is_error = True

        if self.printer:
            self.printer.print(
                function_response,
                type="tool_result",
                tool_name=function_name,
                tool_input=function_args,
                is_error=is_error,
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
        try:
            max_context = get_max_context_length(self.model.model_name)
        except KISSError:
            max_context = None
        if max_context is not None and self.context_tokens_used >= (
            CONTEXT_LIMIT_FRACTION * max_context
        ):
            raise ContextWindowExceededError(
                f"Agent {self.name} conversation reached "
                f"{self.context_tokens_used:,} of {max_context:,} context tokens "
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

    def _update_tokens_and_budget_from_response(self, response: Any) -> None:
        """Updates token counter and budget from API response."""
        try:
            usage = self.model.extract_input_output_token_counts_from_response(response)
            audio_input = 0
            audio_output = 0
            if len(usage) == 4:
                input_tokens, output_tokens, cache_read, cache_write = usage
                cache_write_1h = 0
            elif len(usage) == 5:
                input_tokens, output_tokens, cache_read, cache_write, cache_write_1h = usage
            else:
                (
                    input_tokens,
                    output_tokens,
                    cache_read,
                    cache_write,
                    cache_write_1h,
                    audio_input,
                    audio_output,
                ) = usage
            call_tokens = (
                input_tokens
                + output_tokens
                + cache_read
                + cache_write
                + cache_write_1h
                + audio_input
                + audio_output
            )
            self.total_tokens_used += call_tokens
            if call_tokens > 0:
                self.context_tokens_used = call_tokens
            cost = calculate_cost(
                self.model.model_name,
                input_tokens,
                output_tokens,
                cache_read,
                cache_write,
                cache_write_1h,
                num_audio_input_tokens=audio_input,
                num_audio_output_tokens=audio_output,
            )
            self.budget_used += cost
        except KISSError:
            raise
        except Exception as e:  # pragma: no cover
            logger.debug("Exception caught", exc_info=True)
            logger.error(
                "Error updating tokens and budget from response: %s", e, exc_info=True
            )

    def _get_usage_info_string(self) -> str:
        """Returns a compact single-line usage information string."""
        try:
            max_tokens = get_max_context_length(self.model.model_name)
            return (
                f"Steps: {self.step_count}/{self.max_steps}, "
                f"Context: {self.context_tokens_used:,}/{max_tokens:,} tokens, "
                f"Total tokens: {self.total_tokens_used:,}, "
                f"Budget: ${self.budget_used:.4f}/${self.max_budget:.2f}, "
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
