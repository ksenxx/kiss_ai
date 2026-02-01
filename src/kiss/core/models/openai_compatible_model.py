# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""OpenAI-compatible model implementation for custom endpoints."""

import inspect
import json
import re
import uuid
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Model

# DeepSeek R1 reasoning models use <think>...</think> tags for chain-of-thought
# These models need text-based tool calling instead of native function calling
DEEPSEEK_REASONING_MODELS = {
    # OpenRouter DeepSeek R1 models
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1-turbo",
    "deepseek/deepseek-r1-distill-qwen-1.5b",
    "deepseek/deepseek-r1-distill-qwen-7b",
    "deepseek/deepseek-r1-distill-llama-8b",
    "deepseek/deepseek-r1-distill-qwen-14b",
    "deepseek/deepseek-r1-distill-qwen-32b",
    "deepseek/deepseek-r1-distill-llama-70b",
    # Together AI DeepSeek R1 models
    "DeepSeek-R1",
    "DeepSeek-R1-0528-tput",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Llama-70B",
}


def _extract_deepseek_reasoning(content: str) -> tuple[str, str]:
    """Extract reasoning and final answer from DeepSeek R1 response.

    DeepSeek R1 models wrap their reasoning in <think>...</think> tags.
    Returns (reasoning, final_answer) tuple.
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    match = think_pattern.search(content)
    if match:
        reasoning = match.group(1).strip()
        # Remove the think tags to get the final answer
        final_answer = think_pattern.sub("", content).strip()
        return reasoning, final_answer
    return "", content


def _build_text_based_tools_prompt(function_map: dict[str, Callable[..., Any]]) -> str:
    """Build a text-based tools description for models without native function calling.

    Returns a prompt section describing available tools and how to call them.
    """
    if not function_map:
        return ""

    tools_desc = []
    for func_name, func in function_map.items():
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or f"Function {func_name}"

        # Build parameter descriptions
        params = []
        for param_name, param in sig.parameters.items():
            param_type = param.annotation
            type_name = getattr(param_type, "__name__", str(param_type))
            if type_name == "_empty":
                type_name = "any"
            params.append(f"    - {param_name} ({type_name})")

        params_str = "\n".join(params) if params else "    (no parameters)"
        first_line = doc.split(chr(10))[0]
        tools_desc.append(f"- **{func_name}**: {first_line}\n  Parameters:\n{params_str}")

    return f"""
## Available Tools

To call a tool, output a JSON object in the following format:

```json
{{"tool_calls": [{{"name": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}]}}
```

You can call multiple tools at once by including multiple objects in the tool_calls array.

### Tools:
{chr(10).join(tools_desc)}

IMPORTANT: When you want to call a tool, output ONLY the JSON object with tool_calls.
Do not include any other text before or after the JSON.
When you have the final answer, call the `finish` tool with your result.
"""


def _parse_text_based_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse tool calls from text-based model output.

    Looks for JSON objects with tool_calls array in the content.
    Returns list of function call dictionaries.
    """
    function_calls: list[dict[str, Any]] = []

    # Try to find JSON in the content - look for tool_calls pattern
    # First try to find JSON code blocks
    json_patterns = [
        r"```json\s*(\{.*?\})\s*```",  # JSON in code blocks
        r"```\s*(\{.*?\})\s*```",  # JSON in generic code blocks
        r"(\{[^{}]*\"tool_calls\"[^{}]*\[[^\]]*\][^{}]*\})",  # Inline JSON with tool_calls
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if "tool_calls" in data and isinstance(data["tool_calls"], list):
                    for tc in data["tool_calls"]:
                        if "name" in tc:
                            function_calls.append(
                                {
                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                    "name": tc["name"],
                                    "arguments": tc.get("arguments", {}),
                                }
                            )
                    if function_calls:
                        return function_calls
            except json.JSONDecodeError:
                continue

    # Also try to parse the entire content as JSON (in case model outputs clean JSON)
    try:
        data = json.loads(content.strip())
        if "tool_calls" in data and isinstance(data["tool_calls"], list):
            for tc in data["tool_calls"]:
                if "name" in tc:
                    function_calls.append(
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "name": tc["name"],
                            "arguments": tc.get("arguments", {}),
                        }
                    )
    except json.JSONDecodeError:
        pass

    return function_calls


class OpenAICompatibleModel(Model):
    """A model that uses an OpenAI-compatible API with a custom base URL.

    This model can be used with any API that implements the OpenAI chat completions
    format, such as local LLM servers (Ollama, vLLM, LM Studio), or third-party
    providers that offer OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        model_config: dict[str, Any] | None = None,
    ):
        """Initialize an OpenAI-compatible model.

        Args:
            model_name: The name/identifier of the model to use.
            base_url: The base URL for the API endpoint (e.g., "http://localhost:11434/v1").
            api_key: API key for authentication.
            model_config: Optional dictionary of model configuration parameters.
        """
        super().__init__(model_name, model_config=model_config)
        self.base_url = base_url
        self.api_key = api_key
        # For OpenRouter, strip the "openrouter/" prefix from model name for API calls
        self._api_model_name = (
            model_name[len("openrouter/") :] if model_name.startswith("openrouter/") else model_name
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name}, base_url={self.base_url})"

    __repr__ = __str__

    def initialize(self, prompt: str) -> None:
        """Initializes the conversation with an initial user prompt."""
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.conversation = [{"role": "user", "content": prompt}]

    def _is_deepseek_reasoning_model(self) -> bool:
        """Check if this is a DeepSeek R1 reasoning model."""
        return self.model_name in DEEPSEEK_REASONING_MODELS

    def generate(self) -> tuple[str, Any]:
        """Generates content from prompt without tools."""
        kwargs = self.model_config.copy()
        kwargs.update(
            {
                "model": self._api_model_name,
                "messages": self.conversation,
            }
        )
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""

        # For DeepSeek R1 reasoning models, extract the final answer (strip <think> tags)
        if self._is_deepseek_reasoning_model():
            _, content = _extract_deepseek_reasoning(content)

        self.conversation.append({"role": "assistant", "content": content})
        return content, response

    def generate_and_process_with_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generates content with tools, processes the response, and adds it to conversation."""
        # Use text-based tool calling for DeepSeek R1 models
        if self._is_deepseek_reasoning_model():
            return self._generate_with_text_based_tools(function_map)

        # Standard OpenAI-style native function calling
        tools = self._build_openai_tools_schema(function_map)
        kwargs = self.model_config.copy()
        kwargs.update(
            {
                "model": self._api_model_name,
                "messages": self.conversation,
                "tools": tools or None,
            }
        )
        response = self.client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        content = message.content or ""
        function_calls: list[dict[str, Any]] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                function_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": arguments,
                    }
                )

            self.conversation.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )
        else:
            self.conversation.append({"role": "assistant", "content": content})

        return function_calls, content, response

    def _generate_with_text_based_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generate with text-based tool calling for models without native function calling.

        This method injects tool descriptions into the conversation and parses
        tool calls from the model's text output.
        """
        # Build tools prompt and inject it into the system/user context
        tools_prompt = _build_text_based_tools_prompt(function_map)

        # Create a modified conversation with tools prompt injected
        modified_conversation = list(self.conversation)
        if modified_conversation and modified_conversation[0]["role"] == "user":
            # Append tools prompt to the first user message
            modified_conversation[0] = {
                "role": "user",
                "content": modified_conversation[0]["content"] + "\n" + tools_prompt,
            }
        else:
            # Insert as system message
            modified_conversation.insert(0, {"role": "system", "content": tools_prompt})

        kwargs = self.model_config.copy()
        kwargs.update(
            {
                "model": self._api_model_name,
                "messages": modified_conversation,
            }
        )
        response = self.client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        content = message.content or ""

        # For DeepSeek R1, extract reasoning and final answer
        _, content_clean = _extract_deepseek_reasoning(content)

        # Parse tool calls from the text output
        function_calls = _parse_text_based_tool_calls(content_clean)

        if function_calls:
            # Store tool calls in conversation for proper result handling
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": fc["id"],
                            "type": "function",
                            "function": {
                                "name": fc["name"],
                                "arguments": json.dumps(fc["arguments"]),
                            },
                        }
                        for fc in function_calls
                    ],
                }
            )
        else:
            self.conversation.append({"role": "assistant", "content": content})

        return function_calls, content, response

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Adds function results to the conversation state."""
        # Find tool call IDs from the last assistant message
        tool_call_map: dict[str, str] = {}
        for msg in reversed(self.conversation):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_call_map = {tc["function"]["name"]: tc["id"] for tc in msg["tool_calls"]}
                break

        for func_name, result_dict in function_results:
            result_content = result_dict.get("result", str(result_dict))
            if self.usage_info_for_messages:
                result_content = f"{result_content}\n\n{self.usage_info_for_messages}"
            self.conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_map.get(func_name, f"call_{func_name}"),
                    "content": result_content,
                }
            )

    def add_message_to_conversation(self, role: str, content: str) -> None:
        """Adds a message to the conversation state."""
        if role == "user" and self.usage_info_for_messages:
            content = f"{content}\n\n{self.usage_info_for_messages}"
        self.conversation.append({"role": role, "content": content})

    def extract_input_output_token_counts_from_response(self, response: Any) -> tuple[int, int]:
        """Extracts input and output token counts from an API response."""
        if hasattr(response, "usage") and response.usage:
            return response.usage.prompt_tokens or 0, response.usage.completion_tokens or 0
        return 0, 0

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generates an embedding vector for the given text."""
        model_to_use = embedding_model or self.model_name
        try:
            response = self.client.embeddings.create(model=model_to_use, input=text)
            return list(response.data[0].embedding)
        except Exception as e:
            raise KISSError(f"Embedding generation failed for model {model_to_use}: {e}") from e
