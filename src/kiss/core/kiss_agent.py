# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Core KISS agent implementation with native function calling support."""

import json
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


def _str_presenter(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(str, _str_presenter)


class KISSAgent:
    """
    A KISS agent using native function calling.
    """

    agent_counter: ClassVar[int] = 1
    global_budget_used: ClassVar[float] = 0.0

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
        self.messages: list[dict[str, Any]] = []
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
            self.model = model(model_name, model_config=model_config)
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
                if DEFAULT_CONFIG.agent.use_web_search and not has_search:
                    tools.append(search_web)
                self._add_functions(tools)
            self._set_prompt(prompt_template, arguments)

            for _ in range(self.max_steps):
                self.step_count += 1
                if self.step_count == 1:
                    trajectory_list = json.loads(self.get_trajectory())
                    self.formatter.print_messages(trajectory_list)
                try:
                    self.formatter.print_status(f"Asking {self.model.model_name}...\n")
                    if not self.is_agentic:
                        response_text: str
                        response_text, response = self.model.generate()
                        self._update_tokens_and_budget_from_response(response)
                        usage_info_str = self._get_usage_info_string()
                        model_content = (
                            response_text + "\n" + usage_info_str +"\n"
                        )
                        self._add_and_print_message("model", model_content)
                        return response_text

                    else:
                        function_calls, response_text, response = (
                            self.model.generate_and_process_with_tools(self.function_map)
                        )
                        self._update_tokens_and_budget_from_response(response)
                        usage_info = self._get_usage_info_string()
                        self.model.set_usage_info_for_messages(usage_info)


                        if len(function_calls) == 0 or len(function_calls) > 1:
                            usage_info_str = self._get_usage_info_string()
                            model_content = (
                                response_text + "\n" + usage_info_str
                            )
                            self._add_and_print_message("model", model_content)
                            user_content = (
                                f"**Your response MUST have exactly one function call. "
                                f"Your response has {len(function_calls)} function calls.**"
                            )
                            self._add_and_print_message("user", user_content)
                        else:
                            function_call = function_calls[0]
                            function_name = function_call["name"]
                            function_args = function_call.get("arguments", {})

                            try:
                                if function_name not in self.function_map:
                                    raise KISSError(
                                        f"Function {function_name} is not a registered tool"
                                    )
                                args_str = ", ".join(f"{k}={v!r}" for k, v in function_args.items())
                                call_repr = f"```python\n{function_name}({args_str})\n```"
                                result_raw = self.function_map[function_name](**function_args)
                                function_response = str(result_raw)
                            except Exception as e:
                                args_str = str(function_args)
                                function_response = (
                                    f"Failed to call {function_name} with {args_str}: {e}\n"
                                )

                            usage_info_str = self._get_usage_info_string()
                            model_content = (
                                response_text + "\n" + call_repr + "\n" + usage_info_str
                            )
                            self._add_and_print_message("model", model_content)

                            user_content = (
                                f"Tools call(s) successful.\nResult(s):\n"
                                f"{function_response}"
                            )
                            self._add_and_print_message("user", user_content)

                            if function_name == "finish":
                                return function_response

                            self.model.add_function_results_to_conversation_and_return(
                                [(function_name, {"result": function_response})]
                            )

                except (KISSError, RuntimeError) as e:
                    content = f"Failed to get response from Model: {e}.\nPlease try again.\n"
                    self.model.add_message_to_conversation("user", content)
                    self._add_and_print_message("model", content)

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
            self._save()

    def get_trajectory(self) -> str:
        """Returns the trajectory of the agent in standard JSON format for visualization."""
        trajectory = []
        for msg in self.messages:
            if msg["content"]:
                trajectory.append({"role": msg["role"], "content": msg["content"]})
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

    def _add_and_print_message(self, role: str, content: str, location: int = -1) -> None:
        """Adds a message and prints it if verbose."""
        self._add_message(role, content, location)
        if DEFAULT_CONFIG.agent.verbose:
            self.formatter.print_message(self.messages[-1])

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
            self.formatter.print_error(f"Error getting usage info: {traceback.format_exc()}")
            return_info = f"#### Usage Information\n  - {step_info}\n"

        return return_info

    def _add_message(self, role: str, content: str, location: int = -1) -> None:
        """Method to create and add a message to the history tree."""

        unique_id = len(self.messages)
        message = {
            "unique_id": unique_id,
            "role": role,
            "content": content,
            "timestamp": int(time.time()),
        }
        if location != -1:
            self.messages.insert(location, message)
        else:
            self.messages.append(message)

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
            "messages": self.messages,
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
            "command": " ".join(sys.argv),
        }

    def _save(self) -> None:
        """Save the agent's state to a file."""
        state = self._build_state_dict()
        folder_path = Path(DEFAULT_CONFIG.agent.artifact_dir) / "trajectories"
        folder_path.mkdir(parents=True, exist_ok=True)
        name_safe = self.name.replace(" ", "_").replace("/", "_")
        filename = folder_path / f"trajectory_{name_safe}_{self.id}_{self.run_start_timestamp}.yaml"
        with filename.open("w", encoding="utf-8") as f:
            yaml.dump(state, f, indent=2)

    def finish(self, result: str) -> str:
        """
        The agent must call this function with the final answer to the task.

        Args:
            result (str): The result generated by the agent.

        Returns:
            Returns the result of the agent's task.
        """
        return result
