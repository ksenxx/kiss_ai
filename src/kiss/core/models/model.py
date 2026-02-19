# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Abstract base class for LLM provider model implementations."""

import asyncio
import inspect
import threading
import types as types_module
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any, Union, get_args, get_origin

# Type alias for the async token streaming callback.
TokenCallback = Callable[[str], Coroutine[Any, Any, None]]


_callback_helper_loop: asyncio.AbstractEventLoop | None = None
_callback_helper_ready = threading.Event()
_callback_helper_lock = threading.Lock()


def _get_callback_loop() -> asyncio.AbstractEventLoop:
    global _callback_helper_loop
    with _callback_helper_lock:
        if _callback_helper_loop is not None and not _callback_helper_loop.is_closed():
            return _callback_helper_loop

        def run_loop() -> None:
            global _callback_helper_loop
            _callback_helper_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_callback_helper_loop)
            _callback_helper_ready.set()
            _callback_helper_loop.run_forever()

        _callback_helper_ready.clear()
        t = threading.Thread(target=run_loop, daemon=True)
        t.start()
    _callback_helper_ready.wait(timeout=5)
    if _callback_helper_loop is None or _callback_helper_loop.is_closed():
        raise RuntimeError("Callback helper loop failed to start")
    return _callback_helper_loop


class Model(ABC):
    """Abstract base class for LLM provider implementations."""

    def __init__(
        self,
        model_name: str,
        model_description: str = "",
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
    ):
        """Initialize a Model instance.

        Args:
            model_name: The name/identifier of the model.
            model_description: Optional description of the model.
            model_config: Optional dictionary of model configuration parameters.
            token_callback: Optional async callback invoked with each streamed text token.
        """
        self.model_name = model_name
        self.model_description = model_description
        self.model_config = model_config or {}
        self.token_callback = token_callback
        self.usage_info_for_messages: str = ""
        self.conversation: list[Any] = []
        self.client: Any = None
        self._callback_loop: asyncio.AbstractEventLoop | None = None

    def _invoke_token_callback(self, token: str) -> None:
        """Invoke the async token_callback synchronously, preserving order."""
        if self.token_callback is None:
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop and running_loop.is_running():
            helper_loop = _get_callback_loop()
            future = asyncio.run_coroutine_threadsafe(
                self.token_callback(token), helper_loop
            )
            future.result(timeout=30)
            return
        if self._callback_loop is None or self._callback_loop.is_closed():
            self._callback_loop = asyncio.new_event_loop()
        self._callback_loop.run_until_complete(self.token_callback(token))

    def close_callback_loop(self) -> None:
        """Close the reusable event loop if it exists."""
        if self._callback_loop is not None and not self._callback_loop.is_closed():
            self._callback_loop.close()
        self._callback_loop = None

    def __del__(self) -> None:
        self.close_callback_loop()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name})"

    __repr__ = __str__

    @abstractmethod
    def initialize(self, prompt: str) -> None:
        """Initializes the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt to start the conversation.
        """
        pass

    @abstractmethod
    def generate(self) -> tuple[str, Any]:
        """Generates content from prompt.

        Returns:
            tuple[str, Any]: A tuple of (generated_text, raw_response).
        """
        pass

    @abstractmethod
    def generate_and_process_with_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generates content with tools, processes the response, and adds it to conversation.

        Args:
            function_map: Dictionary mapping function names to callable functions.

        Returns:
            tuple[list[dict[str, Any]], str, Any]: A tuple of
                (function_calls, response_text, raw_response).
        """
        pass

    @abstractmethod
    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Adds function results to the conversation state.

        Args:
            function_results: List of tuples containing (function_name, result_dict).
        """
        pass

    @abstractmethod
    def add_message_to_conversation(self, role: str, content: str) -> None:
        """Adds a message to the conversation state.

        Args:
            role: The role of the message sender (e.g., 'user', 'assistant').
            content: The message content.
        """
        pass

    @abstractmethod
    def extract_input_output_token_counts_from_response(self, response: Any) -> tuple[int, int]:
        """Extracts input and output token counts from an API response.

        Args:
            response: The raw API response object.

        Returns:
            tuple[int, int]: A tuple of (input_tokens, output_tokens).
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generates an embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name to use for embedding generation.

        Returns:
            list[float]: The embedding vector as a list of floats.
        """
        pass

    def set_usage_info_for_messages(self, usage_info: str) -> None:
        """Sets token information to append to messages sent to the LLM.

        Args:
            usage_info: The usage information string to append.
        """
        self.usage_info_for_messages = usage_info

    # =========================================================================
    # Helper methods for building tool schemas (shared across implementations)
    # =========================================================================

    def _build_openai_tools_schema(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> list[dict[str, Any]]:
        """Builds the OpenAI-compatible tools schema from a function map.

        Args:
            function_map: Dictionary mapping function names to callable functions.

        Returns:
            list[dict[str, Any]]: A list of tool schemas in OpenAI format.
        """
        tools = []
        for func in function_map.values():
            tool_schema = self._function_to_openai_tool(func)
            tools.append(tool_schema)
        return tools

    def _function_to_openai_tool(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Converts a Python function to an OpenAI tool schema.

        Args:
            func: The Python function to convert.

        Returns:
            dict[str, Any]: The tool schema in OpenAI format.
        """
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        # Parse docstring for parameter descriptions
        param_descriptions = self._parse_docstring_params(doc)

        # Build parameters schema
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            param_type = param.annotation
            param_schema = self._python_type_to_json_schema(param_type)

            # Add description from docstring if available
            if param_name in param_descriptions:
                param_schema["description"] = param_descriptions[param_name]

            properties[param_name] = param_schema

            # Check if parameter is required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        # Get first line of docstring as function description
        description = doc.split("\n")[0] if doc else f"Function {func.__name__}"

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _parse_docstring_params(self, docstring: str) -> dict[str, str]:
        """Parses parameter descriptions from a docstring.

        Args:
            docstring: The docstring to parse.

        Returns:
            dict[str, str]: A dictionary mapping parameter names to descriptions.
        """
        param_descriptions: dict[str, str] = {}
        lines = docstring.split("\n")
        in_args_section = False

        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("args:"):
                in_args_section = True
                continue
            elif stripped.lower().startswith(("returns:", "raises:", "example:")):
                in_args_section = False
                continue

            if in_args_section and ":" in stripped:
                # Parse "param_name: description" or "param_name (type): description"
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    param_part = parts[0].strip()
                    desc_part = parts[1].strip()
                    # Handle "param_name (type)" format
                    if "(" in param_part:
                        param_name = param_part.split("(")[0].strip()
                    else:
                        param_name = param_part
                    param_descriptions[param_name] = desc_part

        return param_descriptions

    def _python_type_to_json_schema(self, python_type: Any) -> dict[str, Any]:
        """Converts a Python type annotation to a JSON schema type.

        Args:
            python_type: The Python type annotation to convert.

        Returns:
            dict[str, Any]: The JSON schema type definition.
        """
        if python_type is inspect.Parameter.empty:
            return {"type": "string"}

        origin = get_origin(python_type)
        args = get_args(python_type)

        # Handle Union types (including Optional which is Union[X, None])
        if origin is Union or origin is types_module.UnionType:
            # Filter out NoneType
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                return self._python_type_to_json_schema(non_none_args[0])
            # Multiple types - use anyOf
            return {"anyOf": [self._python_type_to_json_schema(a) for a in non_none_args]}

        # Handle list/List types
        if origin is list:
            if args:
                return {
                    "type": "array",
                    "items": self._python_type_to_json_schema(args[0]),
                }
            return {"type": "array"}

        # Handle dict/Dict types
        if origin is dict:
            return {"type": "object"}

        # Handle basic types
        type_mapping: dict[type, dict[str, str]] = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            type(None): {"type": "null"},
        }

        if python_type in type_mapping:
            return type_mapping[python_type]

        # Default to string for unknown types
        return {"type": "string"}
