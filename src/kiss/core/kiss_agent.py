# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Core KISS agent implementation with native function calling support."""

import json
import random
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import yaml

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import calculate_cost, get_max_context_length, model
from kiss.core.simple_formatter import SimpleFormatter
from kiss.core.utils import config_to_dict, search_web


# Register YAML representer for multiline strings (module-level, runs once)
def _str_presenter(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(str, _str_presenter)


class KISSAgent:
    """
    A KISS agent using native function calling.
    """

    agent_counter: ClassVar[int] = 1
    global_budget_used: ClassVar[float] = 0.0
    artifact_subdir: ClassVar[str] = (
        f"{time.strftime('%S_%M_%H_%d_%m_%Y')}_{random.randint(0, 1000000)}"
    )

    def __init__(
        self,
        name: str,
    ):
        """Initializes a KISS Agent.

        Args:
            name: The name of the agent.
        """
        self.id = KISSAgent.agent_counter
        KISSAgent.agent_counter += 1
        self.name = name

    def _reset(self) -> None:
        """Resets the agent's state."""
        self.id_to_message: list[dict[str, Any]] = []
        self.message_ids_of_trajectory: list[int] = []
        self.function_calls_as_str: list[str] = []
        self.run_start_timestamp = int(time.time())
        self.function_map: dict[str, Callable[..., Any]] = {}
        self.total_tokens_used = 0
        self.budget_used = 0.0

    def _set_prompt(self, prompt_template: str, arguments: dict[str, str] | None = None) -> None:
        """Sets the prompt for the agent."""
        assert self.model is not None
        self.arguments = dict(arguments) if arguments is not None else {}
        self.prompt_template = prompt_template
        full_prompt = self.prompt_template.format(**self.arguments)

        self._update_model_usage_info()
        self._add_message("user", full_prompt, location=0)
        self.model.initialize(full_prompt)

    def run(
        self,
        model_name: str,
        prompt_template: str,
        arguments: dict[str, str] | None = None,
        tools: list[Callable[..., Any]] | None = None,
        formatter: Any = None,
        is_agentic: bool = True,
        max_steps: int = DEFAULT_CONFIG.agent.max_steps,
        max_budget: float = DEFAULT_CONFIG.agent.max_agent_budget,
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
        Returns:
            str: The result of the agent's task.
        """
        try:
            self.model = model(model_name)
            self.formatter = formatter if formatter else SimpleFormatter()
            self.is_agentic = is_agentic
            self.max_steps = max_steps
            self.step_count = 0
            self.max_budget = max_budget

            if not self.is_agentic and tools is not None:
                error_msg = (
                    f"Tools cannot be provided for a non-agentic agent "
                    f"{self.name} with id {self.id}."
                )
                raise KISSError(error_msg)

            self._reset()

            if self.is_agentic:
                tools = tools or []
                if not any(getattr(tool, "__name__", None) == "finish" for tool in tools):
                    tools.append(self.finish)
                has_search = any(
                    getattr(tool, "__name__", None) == "search_web" for tool in tools
                )
                if DEFAULT_CONFIG.agent.use_google_search and not has_search:
                    tools.append(search_web)
                self._add_functions(tools)
            self._set_prompt(prompt_template, arguments)

            for _ in range(self.max_steps):
                self.step_count += 1
                if self.step_count == 1:
                    trajectory_list = json.loads(self.get_trajectory())
                    self.formatter.print_messages(trajectory_list)
                self.function_calls_as_str = []
                try:
                    self.formatter.print_status(f"Asking {self.model.model_name}...\n")
                    if not self.is_agentic:
                        response_text: str
                        response_text, response = self.model.generate()
                        self._update_usage_from_response(response)
                        return response_text

                    function_calls, response_text, response = (
                        self.model.generate_and_process_with_tools(self.function_map)
                    )
                    self._update_usage_from_response(response)

                    if function_calls:
                        call_results = self._process_function_calls(function_calls)
                        function_calls_str = "\n".join(self.function_calls_as_str)
                        usage_info_str = self._get_usage_info_string()
                        model_content = (
                            response_text + "\n" + function_calls_str + "\n\n" + usage_info_str
                        )
                        self._add_and_print_message("model", model_content)
                        user_content = (
                            f"Tools call(s) successful.\nResult(s):\n"
                            f"{call_results['all_responses']}"
                        )
                        self._add_and_print_message("user", user_content)

                        if call_results["finish_response"] is not None:
                            return call_results["finish_response"]

                except (KISSError, RuntimeError) as e:
                    self._handle_model_error(e)

                if self.budget_used > self.max_budget:
                    raise KISSError(f"Agent {self.name} budget exceeded.")

                if KISSAgent.global_budget_used > DEFAULT_CONFIG.agent.global_max_budget:
                    raise KISSError("Global budget exceeded.")

                if self.step_count == self.max_steps:
                    raise KISSError(f"Agent {self.name} exceeded {self.max_steps} steps.")

            # This should not be reached, but handle edge case where loop completes without return
            raise KISSError(
                f"Agent {self.name} completed {self.max_steps} steps without finishing."
            )

        finally:
            self._save(DEFAULT_CONFIG.agent.artifact_dir)

    def get_trajectory(self) -> str:
        """Returns the trajectory of the agent in standard JSON format for visualization."""
        trajectory = []
        for node_id in self.message_ids_of_trajectory:
            message = self.id_to_message[node_id]
            if message["content"]:  # Ignore messages with empty content
                content = message["content"]
                role = message["role"]
                trajectory.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )
        return json.dumps(trajectory, indent=2)

    def _add_functions(self, tools: list[Callable[..., Any]]) -> None:
        """Adds callable tools to the agent's function map."""
        for tool in tools:
            if tool.__name__ in self.function_map:
                error_msg = (
                    f"Tool {tool.__name__} already registered for agent "
                    f"{self.name} with id {self.id}."
                )
                raise KISSError(error_msg)
            self.function_map[tool.__name__] = tool

    def _add_and_print_message(self, role: str, content: str, location: int = -1) -> int:
        """Adds a message and prints it if verbose."""
        message_id = self._add_message(role, content, location)
        if DEFAULT_CONFIG.agent.verbose:
            self.formatter.print_message(self.id_to_message[message_id])
        return message_id

    def _update_tokens_and_budget_from_response(self, response: Any) -> None:
        """Updates token counter and budget from API response.

        Args:
            response: The API response object.
        """
        try:
            input_tokens, output_tokens = (
                self.model.extract_input_output_token_counts_from_response(response)
            )
            self.total_tokens_used += input_tokens + output_tokens
            cost = calculate_cost(self.model.model_name, input_tokens, output_tokens)
            self.budget_used += cost
            KISSAgent.global_budget_used += cost
        except Exception as e:
            print(f"Error updating tokens and budget from response: {e} {traceback.format_exc()}")

    def _update_usage_from_response(self, response: Any) -> None:
        """Updates tokens, budget, and model usage info from response."""
        self._update_tokens_and_budget_from_response(response)
        self._update_model_usage_info()

    def _get_usage_info_string(self) -> str:
        """Returns the token usage and budget information string.

        Returns:
            str: Token and budget usage string in format
                "[Token usage: {used}/{max}] [Budget usage: ${spent}/${total}]",
                or empty string if unavailable.
        """
        step_info = f"[Step {self.step_count}/{self.max_steps}]"
        if self.model is None:
            return step_info
        try:
            max_tokens = get_max_context_length(self.model.model_name)
            total_budget = self.max_budget
            agent_budget_used = self.budget_used
            global_budget_used = KISSAgent.global_budget_used
            global_max_budget = DEFAULT_CONFIG.agent.global_max_budget
            token_info = f"[Token usage: {self.total_tokens_used}/{max_tokens}]"
            budget_info = f"[Agent budget usage: ${agent_budget_used:.4f}/${total_budget:.2f}]"
            global_budget_info = (
                f"[Global budget usage: ${global_budget_used:.4f}/${global_max_budget:.2f}]"
            )
            return_info = (
                "#### Usage Information\n"
                f"  - {token_info}\n"
                f"  - {budget_info}\n"
                f"  - {global_budget_info}\n"
                f"  - {step_info}\n"
            )
        except Exception:
            print(f"Error getting usage info: {traceback.format_exc()}")
            return_info = f"#### Usage Information\n  - {step_info}\n"

        return return_info

    def _update_model_usage_info(self) -> None:
        """Updates the model's usage info for messages."""
        usage_info = self._get_usage_info_string()
        if usage_info and self.model is not None:
            self.model.set_usage_info_for_messages(usage_info)

    def _add_message(self, role: str, content: str, location: int = -1) -> int:
        """Method to create and add a message to the history tree."""
        # Append token and budget usage info to all messages for id_to_message

        unique_id = len(self.id_to_message)
        message = {
            "unique_id": unique_id,
            "role": role,
            "content": content,
            "timestamp": int(time.time()),
        }
        self.id_to_message.append(message)
        if location != -1:
            self.message_ids_of_trajectory.insert(location, unique_id)
        else:
            self.message_ids_of_trajectory.append(unique_id)
        return unique_id

    def _build_state_dict(self) -> dict[str, Any]:
        """Builds the state dictionary for saving."""
        assert self.model is not None

        # Get budget information
        budget_used = self.budget_used
        total_budget = self.max_budget

        global_budget_used = KISSAgent.global_budget_used
        global_max_budget = DEFAULT_CONFIG.agent.global_max_budget

        # Get token information
        tokens_used = getattr(self, "total_tokens_used", 0)
        try:
            max_tokens = get_max_context_length(self.model.model_name)
        except Exception:
            max_tokens = None

        # Get step information
        step_count = getattr(self, "step_count", 0)
        max_steps = getattr(self, "max_steps", DEFAULT_CONFIG.agent.max_steps)

        return {
            "name": self.name,
            "id": self.id,
            "id_to_message": self.id_to_message,
            "message_ids_of_trajectory": self.message_ids_of_trajectory,
            "function_map": list(self.function_map.keys()),
            "run_start_timestamp": self.run_start_timestamp,
            "run_end_timestamp": int(time.time()),
            "config": config_to_dict(),
            "arguments": self.arguments if hasattr(self, "arguments") else {},
            "prompt_template": self.prompt_template if hasattr(self, "prompt_template") else "",
            "is_agentic": self.is_agentic,
            "model": self.model.model_name,
            "budget_used": budget_used,
            "total_budget": total_budget,
            "global_budget_used": global_budget_used,
            "global_max_budget": global_max_budget,
            "tokens_used": tokens_used,
            "max_tokens": max_tokens,
            "step_count": step_count,
            "max_steps": max_steps,
            "trajectory": self.get_trajectory(),
            "command": " ".join(sys.argv),
        }

    def _save(self, folder: str) -> None:
        """Save the agent's state to a file."""
        state = self._build_state_dict()
        folder_path = Path(folder) / KISSAgent.artifact_subdir
        folder_path.mkdir(parents=True, exist_ok=True)
        # Replace characters that could cause path issues (spaces and slashes)
        name_safe = self.name.replace(" ", "_").replace("/", "_")
        filename = folder_path / f"trajectory_{name_safe}_{self.id}_{self.run_start_timestamp}.yaml"
        with filename.open("w", encoding="utf-8") as f:
            yaml.dump(state, f, indent=2)

    def _process_function_calls(
        self, function_calls: list[dict[str, Any]]
    ) -> dict[str, str | None]:
        """Processes all function calls and returns finish result if called.

        Args:
            function_calls (list[dict[str, Any]]): List of function call dictionaries.

        Returns:
            dict[str, str | None]: Dictionary with 'all_responses' and 'finish_response' keys.
        """
        function_results = []
        function_response = ""
        for function_call in function_calls:
            this_function_response = self._execute_function_call(function_call)
            function_response += this_function_response

            if function_call["name"] == "finish":
                return {
                    "all_responses": function_response,
                    "finish_response": this_function_response,
                }

            function_results.append((function_call["name"], {"result": this_function_response}))

        if function_results:
            self._add_function_results_to_conversation(function_results)
        return {"all_responses": function_response, "finish_response": None}

    def _execute_function_call(self, function_call: dict[str, Any]) -> str:
        """Executes a single function call and returns the response.

        Args:
            function_call (dict[str, Any]): Dictionary with 'name' and 'arguments' keys.

        Returns:
            str: The raw result from the function call, or error message on failure.
        """

        try:
            function_name = function_call["name"]
            if function_name not in self.function_map:
                raise KISSError(f"Function {function_name} is not a registered tool")
            args_str = ", ".join(
                f"{k}={v!r}" for k, v in function_call.get("arguments", {}).items()
            )
            call_repr = f"```python\n{function_name}({args_str})\n```"
            self.function_calls_as_str.append(call_repr)

            ###########################################################################
            # The actual function call is here.
            ###########################################################################
            result_raw = self.function_map[function_name](**function_call["arguments"])

        except Exception as e:
            fn_name = function_call.get("name", "unknown")
            args_str = str(function_call.get("arguments", {}))
            error_msg = (
                f"Failed to call {fn_name} with {args_str}: {e}\n{traceback.format_exc()}"
            )
            return error_msg
        return str(result_raw)

    def _add_function_results_to_conversation(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Adds function results to conversation and trajectory."""
        assert self.model is not None
        self._update_model_usage_info()
        self.model.add_function_results_to_conversation_and_return(function_results)

    def _handle_model_error(self, error: Exception) -> None:
        """Handles errors from model generation."""
        content = f"Failed to get response from Model: {error}.\nPlease try again.\n"
        self.model.add_message_to_conversation("user", content)
        self._add_and_print_message("user", content)

    def finish(self, result: str) -> str:
        """
        The agent must call this function with the final answer to the task.

        Args:
            result (str): The result generated by the agent.

        Returns:
            Returns the result of the agent's task.
        """
        return result
