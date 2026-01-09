# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Abstract base class for LLM provider model implementations."""

import inspect
import json
import types as types_module
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Union, cast, get_args, get_origin


class Model(ABC):
    """A model is a LLM provider."""

    def __init__(self, model_name: str, model_description: str = ""):
        self.model_name = model_name
        self.model_description = model_description
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
    def generate_content_with_tools(self, function_map: dict[str, Callable[..., Any]]) -> Any:
        """Generates content using the API with tool use support."""
        pass

    @abstractmethod
    def add_model_response_to_conversation(self, response: Any) -> None:
        """Adds model response to the conversation state."""
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

    def generate_and_process_with_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generates content with tools, processes the response, and adds it to conversation."""
        response = self.generate_content_with_tools(function_map)
        function_calls = self._extract_function_calls_from_response(response)
        text = self._extract_text_from_response(response)
        self.add_model_response_to_conversation(response)
        return function_calls, text, response

    def set_usage_info_for_messages(self, usage_info: str) -> None:
        """Sets token information to append to messages sent to the LLM."""
        self.usage_info_for_messages = usage_info

    def _extract_function_calls_from_response(self, response: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _extract_text_from_response(self, response: Any) -> str:
        raise NotImplementedError

    def _get_initial_prompt_text(self) -> str:
        """Gets the text of the initial prompt from conversation."""
        if not self.conversation:
            return ""
        first = self.conversation[0]
        if isinstance(first, dict):
            return str(first.get("content", "") or "")
        return ""

    def _append_usage_info(self, content: str) -> str:
        """Appends usage info to content if available."""
        return content + self.usage_info_for_messages if self.usage_info_for_messages else content

    def _parse_function_args(self, args: Any) -> dict[str, Any]:
        """Parses function call arguments into a dictionary."""
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                return cast(dict[str, Any], json.loads(args))
            except json.JSONDecodeError:
                pass
        return {}

    def _build_function_schema(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> list[dict[str, Any]]:
        """Builds JSON schema for function parameters from function map."""
        schemas = []
        for tool in function_map.values():
            sig = inspect.signature(tool)
            doc = inspect.getdoc(tool) or ""
            props = {}
            required = []
            for name, param in sig.parameters.items():
                props[name] = {
                    "type": self._type_to_json(param.annotation),
                    "description": self._get_param_desc(doc, name) or f"Parameter {name}",
                }
                if param.default == inspect.Parameter.empty:
                    required.append(name)
            schema: dict[str, Any] = {"type": "object", "properties": props}
            if required:
                schema["required"] = required
            schemas.append(
                {
                    "name": tool.__name__,
                    "description": doc or f"Function {tool.__name__}",
                    "schema": schema,
                }
            )
        return schemas

    def _type_to_json(self, annotation: Any) -> str:
        """Converts Python type annotation to JSON schema type."""
        if annotation == inspect.Signature.empty:
            return "string"
        if isinstance(annotation, str):
            lower = annotation.lower()
            return (
                "integer"
                if "int" in lower
                else "number"
                if "float" in lower
                else "boolean"
                if "bool" in lower
                else "string"
            )
        origin = get_origin(annotation)
        if origin in (Union, types_module.UnionType):
            args = get_args(annotation)
            if len(args) == 2 and type(None) in args:
                return self._type_to_json(next(a for a in args if a is not type(None)))
        return {str: "string", int: "integer", float: "number", bool: "boolean"}.get(
            annotation, "string"
        )

    def _get_param_desc(self, docstring: str, param_name: str) -> str:
        """Extracts parameter description from docstring."""
        for line in docstring.split("\n"):
            if f"{param_name}:" in line:
                return line.split(":", 1)[1].strip()
        return ""


class DictConversationModel(Model):
    """Base class for models using dict-based conversation (OpenAI, Anthropic, Together)."""

    def add_message_to_conversation(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})

    def _get_initial_prompt_text(self) -> str:
        if not self.conversation:
            return ""
        return str(self.conversation[0].get("content", "") or "")
