# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Core KISS agent implementation with native function calling support."""

from __future__ import annotations

import inspect
import time
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import calculate_cost, get_max_context_length, model

if TYPE_CHECKING:  # pragma: no cover
    from kiss.core.printer import Printer


class KISSAgent(Base):
    """A KISS agent using native function calling."""

    # Instance attributes initialized in _reset
    budget_used: float
    step_count: int
    total_tokens_used: int

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _reset(
        self,
        model_name: str,
        is_agentic: bool,
        max_steps: int | None,
        max_budget: float | None,
        model_config: dict[str, Any] | None,
        printer: Printer | None = None,
        verbose: bool | None = None,
    ) -> None:
        cfg = config_module.DEFAULT_CONFIG.agent
        self.model_name = model_name if model_name is not None else cfg.model_name
        self.verbose = verbose if verbose is not None else cfg.verbose
        self.set_printer(printer, verbose=self.verbose)
        token_callback = self.printer.token_callback if self.printer else None

        self.model = model(model_name, model_config=model_config, token_callback=token_callback)
        self.is_agentic = is_agentic
        self.max_steps = (
            max_steps
            if max_steps is not None
            else config_module.DEFAULT_CONFIG.agent.max_steps
        )
        self.max_budget = (
            max_budget
            if max_budget is not None
            else config_module.DEFAULT_CONFIG.agent.max_agent_budget
        )
        self.function_map: dict[str, Callable[..., Any]] = {}
        self.messages: list[dict[str, Any]] = []
        self.step_count = 0
        self.total_tokens_used = 0
        self.budget_used = 0.0
        self.run_start_timestamp = int(time.time())

    def _set_prompt(self, prompt_template: str, arguments: dict[str, str] | None = None) -> None:
        """Sets the prompt for the agent.

        Args:
            prompt_template: The template string for the prompt with placeholders.
            arguments: Optional dictionary of arguments to substitute into the template.
        """
        assert self.model is not None
        self.arguments = dict(arguments) if arguments is not None else {}
        self.prompt_template = prompt_template
        full_prompt = self.prompt_template.format(**self.arguments)

        self._add_message("user", full_prompt)
        self.model.initialize(full_prompt)
        if self.printer:
            self.printer.print(full_prompt, type="prompt")

    def run(
        self,
        model_name: str,
        prompt_template: str,
        arguments: dict[str, str] | None = None,
        tools: list[Callable[..., Any]] | None = None,
        is_agentic: bool = True,
        max_steps: int | None = None,
        max_budget: float | None = None,
        model_config: dict[str, Any] | None = None,
        printer: Printer | None = None,
        verbose: bool | None = None,
    ) -> str:
        """
        Runs the agent's main ReAct loop to solve the task.

        Args:
            model_name (str): The name of the model to use for the agent.
            prompt_template (str): The prompt template for the agent.
            arguments (dict[str, str] | None): The arguments to be substituted into the prompt
                template. Default is None.
            tools (list[Callable[..., Any]] | None): The tools to use for the agent.
                If None, no tools are provided (only the built-in finish tool is added).
            is_agentic (bool): Whether the agent is agentic. Default is True.
            max_steps (int): The maximum number of steps to take.
                Default is DEFAULT_CONFIG.agent.max_steps.
            max_budget (float): The maximum budget to spend.
                Default is DEFAULT_CONFIG.agent.max_agent_budget.
            model_config (dict[str, Any] | None): The model configuration to use for the agent.
                Default is None.
            printer (Printer | None): Optional printer for streaming output.
                Default is None.
            verbose (bool | None): Whether to print output to console.
                Default is None (uses config verbose setting).

        Returns:
            str: The result of the agent's task.
        """
        try:
            self._reset(
                model_name, is_agentic, max_steps, max_budget,
                model_config, printer, verbose,
            )

            if not self.is_agentic and tools is not None:
                raise KISSError(
                    f"Tools cannot be provided for a non-agentic agent "
                    f"{self.name} with id {self.id}."
                )
            self._setup_tools(tools)
            self._set_prompt(prompt_template, arguments)

            # Non-agentic mode: single generation, no tool loop
            if not self.is_agentic:
                return self._run_non_agentic()

            # Agentic mode: ReAct loop
            return self._run_agentic_loop()

        finally:
            self._save()

    def _setup_tools(self, tools: list[Callable[..., Any]] | None) -> None:
        """Setup tools for agentic mode.

        Adds finish tool if not present, and web tools if enabled in config.

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
            self.printer.print(response_text, type="result",
            step_count=self.step_count,
            total_tokens=self.total_tokens_used,
            cost=f"${self.budget_used:.4f}",
            )
        return response_text

    def _run_agentic_loop(self) -> str:
        for _ in range(self.max_steps):
            self.step_count += 1
            try:
                result = self._execute_step()
                if result is not None:
                    if self.printer:
                        cost = f"${self.budget_used:.4f}"
                        self.printer.print(
                            result, type="result",
                            step_count=self.step_count,
                            total_tokens=self.total_tokens_used,
                            cost=cost,
                        )
                    return result
            except (KISSError, RuntimeError) as e:  # pragma: no cover
                content = f"Failed to get response from Model: {e}.\nPlease try again.\n"
                self.model.add_message_to_conversation("user", content)
                self._add_message("user", content)

            self._check_limits()

        raise KISSError(  # pragma: no cover
            f"Agent {self.name} completed {self.max_steps} steps without finishing."
        )

    def _execute_step(self) -> str | None:
        """Execute a single step in the ReAct loop.

        Returns:
            str | None: The result string if the task is finished, None otherwise.
        """
        start_timestamp = int(time.time())

        function_calls, response_text, response = self.model.generate_and_process_with_tools(
            self.function_map
        )
        self._update_tokens_and_budget_from_response(response)
        usage_info = self._get_usage_info_string()
        self.model.set_usage_info_for_messages(usage_info)

        if not function_calls:  # pragma: no cover
            self._add_message(
                "model", response_text + "\n```text\n" + usage_info + "\n```\n", start_timestamp
            )
            self._add_message(
                "user",
                "**Your response MUST have at least one function call. "
                "Your response has 0 function calls.**",
            )
            return None

        if self.printer:
            self.printer.print(usage_info, type="usage_info")

        call_reprs = []
        function_results: list[tuple[str, dict[str, Any]]] = []
        finish_result: str | None = None

        for fc in function_calls:
            name, response_str = self._execute_tool(fc)
            raw_args = fc.get("arguments")
            func_args = raw_args if isinstance(raw_args, dict) else {}
            args_str = ", ".join(f"{k}={v!r}" for k, v in func_args.items())
            call_reprs.append(f"```python\n{name}({args_str})\n```")
            function_results.append((name, {"result": response_str}))
            if name == "finish":
                finish_result = response_str

        model_content = (
            response_text + "\n" + "\n".join(call_reprs)
            + "\n```text\n" + usage_info + "\n```\n"
        )
        tool_call_timestamp = int(time.time())
        self._add_message("model", model_content, start_timestamp)
        self._add_message(
            "user",
            "\n\n".join(f"[{name}]: {result['result']}" for name, result in function_results),
            tool_call_timestamp,
        )

        if finish_result is not None:
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
        raw_args = function_call.get("arguments")
        function_args = raw_args if isinstance(raw_args, dict) else {}

        if self.printer:
            self.printer.print(function_name, type="tool_call", tool_input=function_args)

        try:
            if function_name not in self.function_map:  # pragma: no cover
                raise KISSError(f"Function {function_name} is not a registered tool")
            function_response = str(self.function_map[function_name](**function_args))
        except Exception as e:
            fn = self.function_map.get(function_name)
            sig = inspect.signature(fn) if fn else None
            sig_str = f"\nExpected signature: {function_name}{sig}" if sig else ""
            function_response = (
                f"Failed to call {function_name} with "
                f"{function_args}: {e}{sig_str}\n"
            )

        if self.printer:
            self.printer.print(function_response, type="tool_result")

        return function_name, function_response

    def _check_limits(self) -> None:
        """Check budget and step limits, raise KISSError if exceeded.

        Raises:
            KISSError: If agent budget, global budget, or step limit is exceeded.
        """
        if self.budget_used > self.max_budget:
            raise KISSError(f"Agent {self.name} budget exceeded.")
        if Base.global_budget_used > config_module.DEFAULT_CONFIG.agent.global_max_budget:
            raise KISSError("Global budget exceeded.")
        if self.step_count >= self.max_steps:
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
            input_tokens, output_tokens, cache_read, cache_write = (
                self.model.extract_input_output_token_counts_from_response(response)
            )
            self.total_tokens_used += input_tokens + output_tokens + cache_read + cache_write
            cost = calculate_cost(
                self.model.model_name, input_tokens, output_tokens, cache_read, cache_write
            )
            self.budget_used += cost
            Base.global_budget_used += cost
        except Exception as e:  # pragma: no cover
            print(f"Error updating tokens and budget from response: {e} {traceback.format_exc()}")

    def _get_usage_info_string(self) -> str:
        """Returns a compact single-line usage information string."""
        try:
            max_tokens = get_max_context_length(self.model.model_name)
            global_max = config_module.DEFAULT_CONFIG.agent.global_max_budget
            return (
                f"Steps: {self.step_count}/{self.max_steps}, "
                f"Tokens: {self.total_tokens_used}/{max_tokens}, "
                f"Budget: ${self.budget_used:.4f}/${self.max_budget:.2f}, "
                f"Global Budget: ${Base.global_budget_used:.4f}/${global_max:.2f}"
            )
        except Exception:  # pragma: no cover
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
