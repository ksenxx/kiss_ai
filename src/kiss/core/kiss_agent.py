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
    "PermissionDeniedError",
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
        self.context_tokens_used = 0
        self.budget_check_hook: Callable[[], None] | None = None

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
        self.is_agentic = is_agentic
        self.max_steps = max_steps if max_steps is not None else 10000
        self.max_budget = max_budget if max_budget is not None else 10.0
        self.function_map: dict[str, Callable[..., Any]] = {}
        self._cached_tools_schema: list[dict[str, Any]] | None = None
        self.messages: list[dict[str, Any]] = []
        self.step_count = 0
        self.total_tokens_used = 0
        self.context_tokens_used = 0
        self.budget_used = 0.0
        self.run_start_timestamp = int(time.time())
        self._consecutive_no_tool_calls = 0
        self._model_config: dict[str, Any] | None = model_config
        self._fallback_used = False

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

        Returns:
            str: The result of the agent's task.
        """
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

    def _run_non_agentic(self) -> str:
        """Run a single generation without tools.

        Returns:
            str: The generated response text from the model.
        """
        start_timestamp = int(time.time())
        self.step_count = 1

        try:
            response_text, response = self.model.generate()
        except Exception as e:
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
        if response_text and self.printer:
            self.printer.print(
                response_text,
                type="result",
                step_count=self.step_count,
                total_tokens=self.total_tokens_used,
                cost=f"${self.budget_used:.4f}",
            )
        return str(response_text)

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
            self.step_count += 1
            self._check_limits()
            try:
                result = self._execute_step()
                consecutive_errors = 0
                if result is not None:
                    if self.printer:
                        cost = f"${self.budget_used:.4f}"
                        self.printer.print(
                            result,
                            type="result",
                            step_count=self.step_count,
                            total_tokens=self.total_tokens_used,
                            cost=cost,
                        )
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
                        self._consecutive_no_tool_calls = 0
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
                    self._consecutive_no_tool_calls = 0
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
        function_calls, response_text, response = self.model.generate_and_process_with_tools(
            self.function_map, tools_schema=self._cached_tools_schema
        )
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
                return str(response_text)
            retry_msg = (
                "**Your response MUST have at least one function call. "
                "Your response has 0 function calls.**"
            )
            self._add_message("user", retry_msg)
            self.model.add_message_to_conversation("user", retry_msg)
            return None

        self._consecutive_no_tool_calls = 0
        call_reprs = []
        function_results: list[tuple[str, dict[str, Any]]] = []
        finish_result: str | None = None

        for fc in function_calls:
            blocked: str | None = None
            if self.tool_call_guard is not None:
                blocked = self.tool_call_guard(fc["name"], _call_args(fc))
            name, response_str = self._execute_tool(fc, blocked=blocked)
            args_str = ", ".join(f"{k}={v!r}" for k, v in _call_args(fc).items())
            call_reprs.append(f"```python\n{name}({args_str})\n```")
            function_results.append((name, {"result": response_str}))
            if name == "finish" and blocked is None:
                finish_result = response_str
            else:
                self._check_limits()

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
        return None

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
