# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Core KISS agent implementation with native function calling support."""

import time
import traceback
from collections.abc import Callable
from typing import Any

from kiss.core.base import Base
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.formatter import Formatter
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import calculate_cost, get_max_context_length, model
from kiss.core.simple_formatter import SimpleFormatter
from kiss.core.utils import search_web


class KISSAgent(Base):
    """A KISS agent using native function calling."""

    # Instance attributes initialized in _init_run_state
    budget_used: float
    step_count: int
    total_tokens_used: int

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _reset(self, model_name: str) -> None:
        """Resets the agent's state."""
        self._tool_map: dict[str, Callable[..., Any]] = {}
        self.function_map: list[str] = []
        self.model_name = model_name
        self.messages: list[dict[str, Any]] = []
        self.step_count = 0
        self.total_tokens_used = 0
        self.budget_used = 0.0
        self.run_start_timestamp = int(time.time())

    def _set_prompt(self, prompt_template: str, arguments: dict[str, str] | None = None) -> None:
        """Sets the prompt for the agent."""
        assert self.model is not None
        self.arguments = dict(arguments) if arguments is not None else {}
        self.prompt_template = prompt_template
        full_prompt = self.prompt_template.format(**self.arguments)

        self._add_message_with_formatter("user", full_prompt)
        self.model.initialize(full_prompt)

    def run(
        self,
        model_name: str,
        prompt_template: str,
        arguments: dict[str, str] | None = None,
        tools: list[Callable[..., Any]] | None = None,
        formatter: Formatter | None = None,
        is_agentic: bool = True,
        max_steps: int = DEFAULT_CONFIG.agent.max_steps,
        max_budget: float = DEFAULT_CONFIG.agent.max_agent_budget,
        model_config: dict[str, Any] | None = None,
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
            formatter (Formatter | None): The formatter to use for the agent. If None, the default
                formatter is used.
            is_agentic (bool): Whether the agent is agentic. Default is True.
            max_steps (int): The maximum number of steps to take.
                Default is DEFAULT_CONFIG.agent.max_steps.
            max_budget (float): The maximum budget to spend.
                Default is DEFAULT_CONFIG.agent.max_agent_budget.
            model_config (dict[str, Any] | None): The model configuration to use for the agent.
                Default is None.
        Returns:
            str: The result of the agent's task.
        """
        try:
            self._initialize_run(
                model_name, formatter, is_agentic, max_steps, max_budget, model_config
            )

            if not self.is_agentic and tools is not None:
                raise KISSError(
                    f"Tools cannot be provided for a non-agentic agent "
                    f"{self.name} with id {self.id}."
                )

            self._reset(model_name)
            self._setup_tools(tools)
            self.function_map = list(self._tool_map.keys())
            self._set_prompt(prompt_template, arguments)

            # Non-agentic mode: single generation, no tool loop
            if not self.is_agentic:
                return self._run_non_agentic()

            # Agentic mode: ReAct loop
            return self._run_agentic_loop()

        finally:
            self._save()

    def _initialize_run(
        self,
        model_name: str,
        formatter: Formatter | None,
        is_agentic: bool,
        max_steps: int,
        max_budget: float,
        model_config: dict[str, Any] | None,
    ) -> None:
        """Initialize run parameters."""
        self.model = model(model_name, model_config=model_config)
        self._formatter = formatter or SimpleFormatter()
        self.is_agentic = is_agentic
        self.max_steps = max_steps
        self.max_budget = max_budget

    def _setup_tools(self, tools: list[Callable[..., Any]] | None) -> None:
        """Setup tools for agentic mode."""
        if not self.is_agentic:
            return

        tools = tools or []
        tool_names = {getattr(tool, "__name__", None) for tool in tools}

        if "finish" not in tool_names:
            tools.append(self.finish)
        if DEFAULT_CONFIG.agent.use_web_search and "search_web" not in tool_names:
            tools.append(search_web)

        self._add_functions(tools)

    def _run_non_agentic(self) -> str:
        """Run a single generation without tools."""
        if DEFAULT_CONFIG.agent.verbose:
            self._formatter.print_status(f"Asking {self.model.model_name}...\n")
        start_timestamp = int(time.time())
        self.step_count = 1

        response_text, response = self.model.generate()
        self._update_tokens_and_budget_from_response(response)
        usage_info_str = self._get_usage_info_string()
        self._add_message_with_formatter(
            "model", response_text + "\n" + usage_info_str + "\n", start_timestamp
        )

        return response_text

    def _run_agentic_loop(self) -> str:
        """Run the main ReAct loop for agentic mode."""
        for _ in range(self.max_steps):
            self.step_count += 1
            try:
                result = self._execute_step()
                if result is not None:
                    return result
            except (KISSError, RuntimeError) as e:
                content = f"Failed to get response from Model: {e}.\nPlease try again.\n"
                self.model.add_message_to_conversation("user", content)
                self._add_message_with_formatter("model", content)

            self._check_limits()

        raise KISSError(f"Agent {self.name} completed {self.max_steps} steps without finishing.")

    def _execute_step(self) -> str | None:
        """Execute a single step in the ReAct loop. Returns result if finished, None otherwise."""
        if DEFAULT_CONFIG.agent.verbose:
            self._formatter.print_status(f"Asking {self.model.model_name}...\n")
        start_timestamp = int(time.time())

        function_calls, response_text, response = self.model.generate_and_process_with_tools(
            self._tool_map
        )
        self._update_tokens_and_budget_from_response(response)
        usage_info = self._get_usage_info_string()
        self.model.set_usage_info_for_messages(usage_info)

        if len(function_calls) != 1:
            self._add_message_with_formatter(
                "model", response_text + "\n" + usage_info, start_timestamp
            )
            # For Anthropic models, we must provide tool_result blocks for any tool_use blocks
            # in the assistant's response, even when rejecting multiple tool calls
            if function_calls:
                error_msg = (
                    f"**Your response MUST have exactly one function call. "
                    f"Your response has {len(function_calls)} function calls.**"
                )
                # Add tool_result for each tool_use to satisfy Anthropic API requirements
                # Include the tool_use_id to ensure correct matching
                function_results = [
                    (fc["name"], {"result": error_msg, "tool_use_id": fc.get("id")})
                    for fc in function_calls
                ]
                self.model.add_function_results_to_conversation_and_return(function_results)
                self._add_message_with_formatter("user", error_msg)
            else:
                self._add_message_with_formatter(
                    "user",
                    "**Your response MUST have exactly one function call. "
                    "Your response has 0 function calls.**",
                )
            return None

        return self._execute_tool(function_calls[0], response_text, usage_info, start_timestamp)

    def _execute_tool(
        self,
        function_call: dict[str, Any],
        response_text: str,
        usage_info: str,
        start_timestamp: int,
    ) -> str | None:
        """Execute a tool call. Returns result if finished, None otherwise."""
        function_name = function_call["name"]
        function_args = function_call.get("arguments", {})

        try:
            if function_name not in self._tool_map:
                raise KISSError(f"Function {function_name} is not a registered tool")

            args_str = ", ".join(f"{k}={v!r}" for k, v in function_args.items())
            call_repr = f"```python\n{function_name}({args_str})\n```"
            tool_call_timestamp = int(time.time())
            result_raw = self._tool_map[function_name](**function_args)
            function_response = str(result_raw)
        except Exception as e:
            args_str = ", ".join(f"{k}={v!r}" for k, v in function_args.items())
            call_repr = f"```python\n{function_name}({args_str})\n```"
            tool_call_timestamp = int(time.time())
            function_response = f"Failed to call {function_name} with {function_args}: {e}\n"

        model_content = response_text + "\n" + call_repr + "\n" + usage_info
        self._add_message_with_formatter("model", model_content, start_timestamp)
        user_content = f"Tools call(s) successful.\nResult(s):\n{function_response}"
        self._add_message_with_formatter("user", user_content, tool_call_timestamp)

        if function_name == "finish":
            return function_response

        self.model.add_function_results_to_conversation_and_return(
            [(function_name, {"result": function_response})]
        )
        return None

    def _check_limits(self) -> None:
        """Check budget and step limits, raise KISSError if exceeded."""
        if self.budget_used > self.max_budget:
            raise KISSError(f"Agent {self.name} budget exceeded.")
        if Base.global_budget_used > DEFAULT_CONFIG.agent.global_max_budget:
            raise KISSError("Global budget exceeded.")
        if self.step_count >= self.max_steps:
            raise KISSError(f"Agent {self.name} exceeded {self.max_steps} steps.")

    def _add_functions(self, tools: list[Callable[..., Any]]) -> None:
        """Adds callable tools to the agent's function map."""
        for tool in tools:
            if tool.__name__ in self._tool_map:
                error_msg = (
                    f"Tool {tool.__name__} already registered for agent "
                    f"{self.name} with id {self.id}."
                )
                raise KISSError(error_msg)
            self._tool_map[tool.__name__] = tool

    def _update_tokens_and_budget_from_response(self, response: Any) -> None:
        """Updates token counter and budget from API response."""
        try:
            input_tokens, output_tokens = (
                self.model.extract_input_output_token_counts_from_response(response)
            )
            self.total_tokens_used += input_tokens + output_tokens
            cost = calculate_cost(self.model.model_name, input_tokens, output_tokens)
            self.budget_used += cost
            Base.global_budget_used += cost
        except Exception as e:
            print(f"Error updating tokens and budget from response: {e} {traceback.format_exc()}")

    def _get_usage_info_string(self) -> str:
        """Returns the token usage and budget information string."""
        step_info = f"[Step {self.step_count}/{self.max_steps}]"
        try:
            max_tokens = get_max_context_length(self.model.model_name)
            token_info = f"[Token usage: {self.total_tokens_used}/{max_tokens}]"
            budget_info = f"[Agent budget usage: ${self.budget_used:.4f}/${self.max_budget:.2f}]"
            global_budget_info = (
                f"[Global budget usage: ${Base.global_budget_used:.4f}/"
                f"${DEFAULT_CONFIG.agent.global_max_budget:.2f}]"
            )
            return (
                "#### Usage Information\n"
                f"  - {token_info}\n"
                f"  - {budget_info}\n"
                f"  - {global_budget_info}\n"
                f"  - {step_info}\n"
            )
        except Exception:
            if DEFAULT_CONFIG.agent.verbose:
                self._formatter.print_error(f"Error getting usage info: {traceback.format_exc()}")
            return f"#### Usage Information\n  - {step_info}\n"

    def _add_message_with_formatter(
        self, role: str, content: str, timestamp: int | None = None
    ) -> None:
        """Add a message and print it using the formatter."""
        self._add_message(role, content, timestamp)
        self._formatter.print_message(self.messages[-1])

    def finish(self, result: str) -> str:
        """
        The agent must call this function with the final answer to the task.

        Args:
            result (str): The result generated by the agent.

        Returns:
            Returns the result of the agent's task.
        """
        return result
