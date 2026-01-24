# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Abstract base class for LLM provider model implementations."""

import inspect
import types as types_module
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Union, get_args, get_origin


class Model(ABC):
    """A model is a LLM provider."""

    def __init__(
        self,
        model_name: str,
        model_description: str = "",
        model_config: dict[str, Any] | None = None,
    ):
        self.model_name = model_name
        self.model_description = model_description
        self.model_config = model_config or {}
        self.usage_info_for_messages: str = ""
        self.conversation: list[Any] = []
        self.client: Any = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name})"

    __repr__ = __str__

    @abstractmethod
    def initialize(self, prompt: str) -> None:
        """Initializes the conversation with an initial user prompt."""
        pass

    @abstractmethod
    def generate(self) -> tuple[str, Any]:
        """Generates content from prompt."""
        pass

    @abstractmethod
    def generate_and_process_with_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generates content with tools, processes the response, and adds it to conversation."""
        pass

    @abstractmethod
    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Adds function results to the conversation state."""
        pass

    @abstractmethod
    def add_message_to_conversation(self, role: str, content: str) -> None:
        """Adds a message to the conversation state."""
        pass

    @abstractmethod
    def extract_input_output_token_counts_from_response(self, response: Any) -> tuple[int, int]:
        """Extracts input and output token counts from an API response."""
        pass

    @abstractmethod
    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generates an embedding vector for the given text."""
        pass

    def set_usage_info_for_messages(self, usage_info: str) -> None:
        """Sets token information to append to messages sent to the LLM."""
        self.usage_info_for_messages = usage_info

    # =========================================================================
    # Helper methods for building tool schemas (shared across implementations)
    # =========================================================================

    def _build_openai_tools_schema(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> list[dict[str, Any]]:
        """Builds the OpenAI-compatible tools schema from a function map."""
        tools = []
        for func in function_map.values():
            tool_schema = self._function_to_openai_tool(func)
            tools.append(tool_schema)
        return tools

    def _function_to_openai_tool(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Converts a Python function to an OpenAI tool schema."""
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
        """Parses parameter descriptions from a docstring."""
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
        """Converts a Python type annotation to a JSON schema type."""
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
