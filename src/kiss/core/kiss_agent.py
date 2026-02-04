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
from kiss.core.useful_tools import fetch_url, search_web


class KISSAgent(Base):
    """A KISS agent using native function calling."""

    # Instance attributes initialized in _init_run_state
    budget_used: float
    step_count: int
    total_tokens_used: int

    def __init__(self, name: str) -> None:
        """Initialize a KISSAgent instance.

        Args:
            name: The name identifier for the agent.
        """
        super().__init__(name)

    def _reset(self, model_name: str) -> None:
        """Resets the agent's state.

        Args:
            model_name: The name of the model to use.
        """
        self._tool_map: dict[str, Callable[..., Any]] = {}
        self.function_map: list[str] = []
        self.model_name = model_name
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
        """Initialize run parameters.

        Args:
            model_name: The name of the model to use.
            formatter: Optional formatter for output display.
            is_agentic: Whether the agent operates in agentic mode with tools.
            max_steps: Maximum number of steps allowed in the ReAct loop.
            max_budget: Maximum budget in dollars for this run.
            model_config: Optional model-specific configuration dictionary.
        """
        self.model = model(model_name, model_config=model_config)
        self._formatter = formatter or SimpleFormatter()
        self.is_agentic = is_agentic
        self.max_steps = max_steps
        self.max_budget = max_budget

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
        if DEFAULT_CONFIG.agent.use_web and "search_web" not in tool_names:
            tools.append(fetch_url)
            tools.append(search_web)

        self._add_functions(tools)

    def _run_non_agentic(self) -> str:
        """Run a single generation without tools.

        Returns:
            str: The generated response text from the model.
        """
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
        """Run the main ReAct loop for agentic mode.

        Returns:
            str: The final result from the agent's task.

        Raises:
            KISSError: If the agent exceeds maximum steps without finishing.
        """
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
        """Execute a single step in the ReAct loop.

        Returns:
            str | None: The result string if the task is finished, None otherwise.
        """
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
            if function_calls:
                error_msg = (
                    f"**Your response MUST have exactly one function call. "
                    f"Your response has {len(function_calls)} function calls.**"
                )
                function_results = [
                    (fc["name"], {"result": error_msg})
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
        """Execute a tool call.

        Args:
            function_call: Dictionary containing 'name' and 'arguments' for the tool call.
            response_text: The model's response text preceding the tool call.
            usage_info: Token and budget usage information string.
            start_timestamp: Unix timestamp when the step started.

        Returns:
            str | None: The result string if the finish tool was called, None otherwise.
        """
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

        # max_tokens = get_max_context_length(self.model_name)
        # if len(function_response)/4 > 0.95*(max_tokens - self.total_tokens_used):
        #     function_response = (
        #         "**The tool response is too long. "
        #         "Please use head or tail commands to shorten the response.**"
        #     )

        model_content = response_text + "\n" + call_repr + "\n" + usage_info
        self._add_message_with_formatter("model", model_content, start_timestamp)
        self._add_message_with_formatter("user", function_response, tool_call_timestamp)

        if function_name == "finish":
            return function_response

        self.model.add_function_results_to_conversation_and_return(
            [(function_name, {"result": function_response})]
        )
        return None

    def _check_limits(self) -> None:
        """Check budget and step limits, raise KISSError if exceeded.

        Raises:
            KISSError: If agent budget, global budget, or step limit is exceeded.
        """
        if self.budget_used > self.max_budget:
            raise KISSError(f"Agent {self.name} budget exceeded.")
        if Base.global_budget_used > DEFAULT_CONFIG.agent.global_max_budget:
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
            if tool.__name__ in self._tool_map:
                error_msg = (
                    f"Tool {tool.__name__} already registered for agent "
                    f"{self.name} with id {self.id}."
                )
                raise KISSError(error_msg)
            self._tool_map[tool.__name__] = tool

    def _update_tokens_and_budget_from_response(self, response: Any) -> None:
        """Updates token counter and budget from API response.

        Args:
            response: The API response object containing usage information.
        """
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
        """Returns the token usage and budget information string.

        Returns:
            str: A formatted markdown string with usage information.
        """
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
        """Add a message and print it using the formatter.

        Args:
            role: The role of the message sender (e.g., 'user', 'model').
            content: The content of the message.
            timestamp: Optional Unix timestamp. If None, uses current time.
        """
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
