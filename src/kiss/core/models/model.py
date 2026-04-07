# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Abstract base class for LLM provider model implementations.

Also contains shared text-based tool calling helpers used by models that
lack native function calling support (e.g. DeepSeek R1, Claude Code CLI).
"""

import base64
import dataclasses
import inspect
import json
import logging
import mimetypes
import os
import re
import types as types_module
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union, get_args, get_origin

logger = logging.getLogger(__name__)

# Type alias for the synchronous token streaming callback.
TokenCallback = Callable[[str], None]

SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "application/pdf",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/webm",
    "audio/flac",
    "audio/aac",
    "audio/mp4",
    "video/mp4",
    "video/webm",
    "video/ogg",
    "video/mpeg",
    "video/quicktime",
}


@dataclasses.dataclass
class Attachment:
    """A file attachment (image, document, audio, or video) to include in a prompt.

    Attributes:
        data: Raw file bytes.
        mime_type: MIME type string (e.g. "image/jpeg", "application/pdf",
            "audio/mpeg", "video/mp4").
    """

    data: bytes
    mime_type: str

    @staticmethod
    def from_file(path: str) -> "Attachment":
        """Create an Attachment from a file path.

        Args:
            path: Path to the file to attach.

        Returns:
            An Attachment with the file's bytes and detected MIME type.

        Raises:
            ValueError: If the MIME type is not supported.
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(path)
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:  # pragma: no cover – mimetypes knows all supported extensions
            suffix = file_path.suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".pdf": "application/pdf",
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".ogg": "audio/ogg",
                ".webm": "video/webm",
                ".flac": "audio/flac",
                ".aac": "audio/aac",
                ".m4a": "audio/mp4",
                ".mp4": "video/mp4",
                ".mpeg": "video/mpeg",
                ".mov": "video/quicktime",
            }
            mime_type = mime_map.get(suffix, "")
        if mime_type not in SUPPORTED_MIME_TYPES:
            raise ValueError(
                f"Unsupported MIME type '{mime_type}' for file '{path}'. "
                f"Supported: {sorted(SUPPORTED_MIME_TYPES)}"
            )
        return Attachment(data=file_path.read_bytes(), mime_type=mime_type)

    def to_base64(self) -> str:
        """Return the file data as a base64-encoded string."""
        return base64.b64encode(self.data).decode("ascii")

    def to_data_url(self) -> str:
        """Return a data: URL suitable for OpenAI image_url fields."""
        return f"data:{self.mime_type};base64,{self.to_base64()}"


# Mapping from audio MIME types to file extensions for the Whisper API.
_AUDIO_MIME_TO_EXT: dict[str, str] = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "audio/flac": ".flac",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
}


def transcribe_audio(data: bytes, mime_type: str, api_key: str | None = None) -> str:
    """Transcribe audio bytes to text using OpenAI's Whisper API.

    This is used as a fallback for model providers that do not support audio
    attachments natively (e.g. Anthropic).

    Args:
        data: Raw audio file bytes.
        mime_type: MIME type of the audio (e.g. ``"audio/mpeg"``).
        api_key: OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable when *None*.

    Returns:
        The transcribed text.

    Raises:
        ValueError: If no API key is available.
        RuntimeError: If the transcription API call fails.
    """
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OpenAI API key is required for audio transcription")

    ext = _AUDIO_MIME_TO_EXT.get(mime_type, ".mp3")
    client = OpenAI(api_key=key)
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(f"audio{ext}", data, mime_type),
            response_format="text",
        )
    except Exception as exc:
        raise RuntimeError(f"Audio transcription failed: {exc}") from exc

    return str(transcript).strip()


class Model(ABC):
    """Abstract base class for LLM provider implementations."""

    def __init__(
        self,
        model_name: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
    ):
        """Initialize a Model instance.

        Args:
            model_name: The name/identifier of the model.
            model_config: Optional dictionary of model configuration parameters.
            token_callback: Optional callback invoked with each streamed text token.
        """
        self.model_name = model_name
        self.model_config = model_config or {}
        self.token_callback = token_callback
        self.usage_info_for_messages: str = ""
        self.conversation: list[Any] = []
        self.client: Any = None

    def _invoke_token_callback(self, token: str) -> None:
        """Invoke the token callback synchronously."""
        if self.token_callback is not None:
            self.token_callback(token)

    def reset_conversation(self) -> None:
        """Reset conversation state for reuse across sub-sessions.

        Clears the conversation history and usage info while keeping the
        HTTP client and model configuration intact.
        """
        self.conversation = []
        self.usage_info_for_messages = ""

    def _replace_last_assistant_with_tool_calls(
        self, content: str, function_calls: list[dict[str, Any]]
    ) -> None:
        """Replace the last assistant message with one that includes tool calls.

        Used by text-based tool calling paths (ClaudeCodeModel, OpenAIModel)
        where ``generate()`` already appended a plain assistant message and it
        needs to be upgraded to include parsed tool call metadata.

        Args:
            content: The full text content of the assistant message.
            function_calls: Parsed tool calls, each with ``id``, ``name``,
                and ``arguments`` (dict).
        """
        self.conversation[-1] = {
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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name})"

    __repr__ = __str__

    @abstractmethod
    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Initializes the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt to start the conversation.
            attachments: Optional list of file attachments (images, PDFs, audio,
                video) to include. Provider support varies — unsupported types
                are skipped with a warning.
        """
        pass  # pragma: no cover

    @abstractmethod
    def generate(self) -> tuple[str, Any]:
        """Generates content from prompt.

        Returns:
            tuple[str, Any]: A tuple of (generated_text, raw_response).
        """
        pass  # pragma: no cover

    @abstractmethod
    def generate_and_process_with_tools(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generates content with tools, processes the response, and adds it to conversation.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Optional pre-built tool schema list. When provided,
                skips schema rebuilding from function_map (performance optimization).

        Returns:
            tuple[list[dict[str, Any]], str, Any]: A tuple of
                (function_calls, response_text, raw_response).
        """
        pass  # pragma: no cover

    def _find_tool_call_ids_from_last_assistant(self) -> list[tuple[str, str]]:
        """Find tool call (name, id) pairs from the last assistant message.

        Searches backwards through the conversation for the most recent
        assistant message containing tool calls and extracts their IDs.

        Returns:
            list[tuple[str, str]]: A list of (function_name, tool_call_id) tuples,
                or an empty list if no assistant message with tool calls is found.
        """
        for msg in reversed(self.conversation):
            if msg.get("role") == "assistant":
                # OpenAI-style: tool_calls list with function.name and id
                if msg.get("tool_calls"):
                    return [
                        (tc["function"]["name"], tc["id"]) for tc in msg["tool_calls"]
                    ]
                # Anthropic-style: content list with tool_use blocks
                content = msg.get("content")
                if isinstance(content, list):
                    ids = [
                        (b.get("name", ""), b.get("id", ""))
                        for b in content
                        if b.get("type") == "tool_use"
                    ]
                    if ids:
                        return ids
                break
        return []

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Adds function results to the conversation state.

        Matches results to tool calls by index from the last assistant message.

        Args:
            function_results: List of tuples containing (function_name, result_dict).
        """
        tool_calls = self._find_tool_call_ids_from_last_assistant()

        for i, (func_name, result_dict) in enumerate(function_results):
            result_content = result_dict.get("result", str(result_dict))
            if self.usage_info_for_messages:
                result_content = f"{result_content}\n\n{self.usage_info_for_messages}"

            if i < len(tool_calls):
                tool_call_id = tool_calls[i][1]
            else:
                tool_call_id = f"call_{func_name}_{i}"

            self.conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_content,
                }
            )

    def add_message_to_conversation(self, role: str, content: str) -> None:
        """Adds a message to the conversation state.

        Args:
            role: The role of the message sender (e.g., 'user', 'assistant').
            content: The message content.
        """
        if role == "user" and self.usage_info_for_messages:  # pragma: no branch
            content = f"{content}\n\n{self.usage_info_for_messages}"
        self.conversation.append({"role": role, "content": content})

    @abstractmethod
    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int]:
        """Extracts token counts from an API response.

        Args:
            response: The raw API response object.

        Returns:
            tuple[int, int, int, int]: (input_tokens, output_tokens,
                cache_read_tokens, cache_write_tokens).
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generates an embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name to use for embedding generation.

        Returns:
            list[float]: The embedding vector as a list of floats.
        """
        pass  # pragma: no cover

    def set_usage_info_for_messages(self, usage_info: str) -> None:
        """Sets token information to append to messages sent to the LLM.

        Args:
            usage_info: The usage information string to append.
        """
        self.usage_info_for_messages = usage_info

    # =========================================================================
    # Helper methods for building tool schemas (shared across implementations)
    # =========================================================================

    def _resolve_openai_tools_schema(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Return pre-built tools_schema or build one from function_map.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Optional pre-built tool schema list. When provided,
                returned as-is (skips schema rebuilding for performance).

        Returns:
            list[dict[str, Any]]: The resolved OpenAI-format tool schema list.
        """
        if tools_schema is not None:
            return tools_schema
        return self._build_openai_tools_schema(function_map)

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
                # split(":", 1) with ":" always gives 2 parts
                if len(parts) == 2:  # pragma: no branch
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


# =========================================================================
# Text-based tool calling helpers (shared by OpenAICompatibleModel,
# ClaudeCodeModel, and any future model without native function calling)
# =========================================================================


def _build_text_based_tools_prompt(function_map: dict[str, Callable[..., Any]]) -> str:
    """Build a text-based tools description for models without native function calling.

    Args:
        function_map: Dictionary mapping function names to callable functions.

    Returns:
        A formatted prompt string describing available tools and how to call them,
        or an empty string if no functions are provided.
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

    Args:
        content: The text content to parse for tool calls.

    Returns:
        A list of function call dictionaries, each containing 'id', 'name',
        and 'arguments' keys. Returns empty list if no valid tool calls found.
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
                logger.debug("Exception caught", exc_info=True)
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
        logger.debug("Exception caught", exc_info=True)
        pass

    return function_calls
