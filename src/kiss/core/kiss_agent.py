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
from kiss.core.kiss_error import KISSError
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
)
MAX_CONSECUTIVE_ERRORS = 3
MAX_CONSECUTIVE_NO_TOOL_CALLS = 2


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


if TYPE_CHECKING:  # pragma: no cover
    from kiss.core.printer import Printer


class KISSAgent(Base):
    """A KISS agent using native function calling."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        # Optional hook invoked at the top of every ``_execute_step``,
        # immediately before the next model call.  Receives the live
        # :class:`kiss.core.models.model.Model` so the hook can mutate
        # its conversation (e.g. ``add_message_to_conversation``) to
        # inject additional user-role context queued asynchronously by
        # an external source (e.g. the VS Code frontend's
        # ``appendUserMessage`` command — see
        # :meth:`kiss.agents.sorcar.sorcar_agent.SorcarAgent.run`).
        self.pre_step_hook: Callable[..., None] | None = None

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
        # When False, the system prompt and task prompt are NOT printed
        # to the printer.  Internal helper agents (e.g. the summarizer in
        # ``RelentlessAgent``) share the parent task's printer; printing
        # their internal prompts would surface confusing
        # ``type="prompt"`` events in the user-visible event stream.
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
            existing.token_callback = token_callback
            existing.thinking_callback = thinking_callback
            self.model = existing
        else:
            self.model = model(
                model_name,
                model_config=model_config,
                token_callback=token_callback,
                thinking_callback=thinking_callback,
            )
        self.is_agentic = is_agentic
        self.max_steps = max_steps if max_steps is not None else 100
        self.max_budget = max_budget if max_budget is not None else 10.0
        self.function_map: dict[str, Callable[..., Any]] = {}
        self._cached_tools_schema: list[dict[str, Any]] | None = None
        self.messages: list[dict[str, Any]] = []
        self.step_count = 0
        self.total_tokens_used = 0
        self.budget_used = 0.0
        self.run_start_timestamp = int(time.time())
        self._consecutive_no_tool_calls = 0

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
                Default is 100.
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

        tools = tools or []
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

        response_text, response = self.model.generate()
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

    def _run_agentic_loop(self) -> str:
        consecutive_errors = 0
        for _ in range(self.max_steps):
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
            except KISSError:  # pragma: no cover – requires model to raise KISSError mid-step
                logger.debug("Exception caught", exc_info=True)
                raise
            except Exception as e:  # pragma: no cover – requires live API error
                logger.debug("Exception caught", exc_info=True)
                if not _is_retryable_error(e):
                    raise KISSError(f"Non-retryable error from model: {e}") from e
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise KISSError(
                        f"Agent {self.name} failed with {consecutive_errors} "
                        f"consecutive errors. Last error: {e}"
                    ) from e
                content = f"Failed to get response from Model: {e}.\nPlease try again.\n"
                self.model.add_message_to_conversation("user", content)
                self._add_message("user", content)

        raise KISSError(  # pragma: no cover
            f"Agent {self.name} completed {self.max_steps} steps without finishing."
        )

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

        if not function_calls:
            self._consecutive_no_tool_calls += 1
            self._add_message(
                "model", response_text + "\n```text\n" + usage_info + "\n```\n", start_timestamp
            )
            if self._consecutive_no_tool_calls >= MAX_CONSECUTIVE_NO_TOOL_CALLS:
                # When the model returns text content but no tool calls,
                # treat it as an implicit finish (covered by
                # test_no_tool_call_loop.py).  But if the response is
                # empty/whitespace-only, returning it would surface to the
                # user as the literal string "(no result)" (substituted by
                # JsonPrinter._broadcast_result) and be persisted as
                # "No summary available" — a silent task death.  This was
                # the production failure mode for claude-fable-5 (task
                # history ids 3706/3707/3708/3710 in ~/.kiss/sorcar.db),
                # where the provider adapter dropped reasoning blocks and
                # left an empty assistant turn.  Raise a visible diagnostic
                # so RelentlessAgent can route it into a success=False
                # result the user can act on.
                if not response_text or not response_text.strip():
                    raise KISSError(
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
            name, response_str = self._execute_tool(fc)
            args_str = ", ".join(f"{k}={v!r}" for k, v in _call_args(fc).items())
            call_reprs.append(f"```python\n{name}({args_str})\n```")
            function_results.append((name, {"result": response_str}))
            if name == "finish":
                finish_result = response_str

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
    ) -> tuple[str, str]:
        """Execute a single tool call.

        Returns:
            tuple[str, str]: (function_name, function_response_string).
        """
        function_name = function_call["name"]
        function_args = _call_args(function_call)

        if self.printer:
            self.printer.print(function_name, type="tool_call", tool_input=function_args)

        is_error = False
        try:
            if function_name not in self.function_map:  # pragma: no cover
                raise KISSError(f"Function {function_name} is not a registered tool")
            function_response = str(self.function_map[function_name](**function_args))
        except (Exception, SystemExit) as e:
            logger.debug("Exception caught", exc_info=True)
            fn = self.function_map.get(function_name)
            sig = inspect.signature(fn) if fn else None
            sig_str = f"\nExpected signature: {function_name}{sig}" if sig else ""
            function_response = (
                f"Failed to call {function_name} with {function_args}: {e}{sig_str}\n"
            )
            # The tool invocation itself raised — mark the printed
            # ``tool_result`` as an error so the terminal surfaces the
            # red ``FAILED`` rule (and so the Read-specific
            # syntax-highlighting branch in ``ConsolePrinter`` cannot
            # misrender the "Failed to call …" diagnostic as Python
            # source).  Note: in-tool sentinels like "Error:" or
            # "(file is empty)" do NOT set this flag — those are
            # valid (non-raised) tool return values and are filtered
            # by the printer's own content-based guard.
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
        """Check budget and step limits, raise KISSError if exceeded.

        Raises:
            KISSError: If agent budget or step limit is exceeded.
        """
        if self.budget_used > self.max_budget:
            raise KISSError(f"Agent {self.name} budget exceeded.")
        if self.step_count > self.max_steps:
            raise KISSError(f"Agent {self.name} exceeded {self.max_steps} steps.")

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
            if len(usage) == 4:
                input_tokens, output_tokens, cache_read, cache_write = usage
                cache_write_1h = 0
            else:
                input_tokens, output_tokens, cache_read, cache_write, cache_write_1h = usage
            self.total_tokens_used += (
                input_tokens + output_tokens + cache_read + cache_write + cache_write_1h
            )
            cost = calculate_cost(
                self.model.model_name,
                input_tokens,
                output_tokens,
                cache_read,
                cache_write,
                cache_write_1h,
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
            capped_tokens = self.total_tokens_used % max_tokens
            return (
                f"Steps: {self.step_count}/{self.max_steps}, "
                f"Tokens: {capped_tokens:,}/{max_tokens:,}, "
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
