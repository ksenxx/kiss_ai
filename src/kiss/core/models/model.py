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

TokenCallback = Callable[[str], None]

ThinkingCallback = Callable[[bool], None]

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


# Tool results that include binary file contents (e.g. ``Read`` on a PNG)
# embed each file inside these sentinel markers so that model implementations
# can lift the payload into a real image/document content block instead of
# shipping kilobytes of base64 as plain text.
BINARY_ATTACHMENT_OPEN_RE = re.compile(
    r"<<KISS_BINARY_ATTACHMENT mime_type=([^>\s]+)>>"
)
BINARY_ATTACHMENT_CLOSE = "<</KISS_BINARY_ATTACHMENT>>"

# MIME types the ``Read`` tool is willing to embed inline in its return
# value.  Set equal to :data:`SUPPORTED_MIME_TYPES` so audio/video are
# also encoded as sentinel-wrapped base64; each model backend then decides
# whether it can actually ingest the bytes (e.g. OpenAI Chat Completions
# accepts ``input_audio``; Gemini accepts any ``inline_data`` MIME;
# Anthropic transcribes audio to text and drops video; text-CLI backends
# drop the bytes after lifting the placeholder text).
READ_TOOL_BINARY_MIME_TYPES = set(SUPPORTED_MIME_TYPES)


def encode_binary_attachment(mime_type: str, data: bytes) -> str:
    """Encode ``data`` for inline transport inside a tool result string.

    Args:
        mime_type: MIME type label (e.g. ``"image/png"``).
        data: Raw file bytes.

    Returns:
        A sentinel-wrapped base64 payload that
        :func:`parse_binary_attachments` will later decode back into an
        :class:`Attachment`.
    """
    payload = base64.b64encode(data).decode("ascii")
    return (
        f"<<KISS_BINARY_ATTACHMENT mime_type={mime_type}>>"
        f"{payload}{BINARY_ATTACHMENT_CLOSE}"
    )


def parse_binary_attachments(text: str) -> tuple[str, list[Attachment]]:
    """Split a tool result string into plain text + binary attachments.

    Scans ``text`` for ``<<KISS_BINARY_ATTACHMENT ...>>...<<...>>`` blocks
    produced by :func:`encode_binary_attachment` and returns the residual
    plain text (with each block replaced by a short
    ``[attached image/png, N bytes]`` placeholder) along with the decoded
    :class:`Attachment` list.

    Args:
        text: Raw tool result content possibly containing sentinel blocks.

    Returns:
        ``(plain_text, attachments)``.  If the input has no sentinels,
        returns ``(text, [])`` unchanged.
    """
    attachments: list[Attachment] = []
    out_parts: list[str] = []
    cursor = 0
    for match in BINARY_ATTACHMENT_OPEN_RE.finditer(text):
        open_start, open_end = match.span()
        close_idx = text.find(BINARY_ATTACHMENT_CLOSE, open_end)
        if close_idx == -1:  # pragma: no cover – defensive: malformed sentinel
            break
        mime_type = match.group(1)
        b64_payload = text[open_end:close_idx]
        try:
            data = base64.b64decode(b64_payload, validate=False)
        except Exception:  # pragma: no cover – validate=False is permissive
            logger.debug("Failed to decode binary attachment", exc_info=True)
            cursor = close_idx + len(BINARY_ATTACHMENT_CLOSE)
            continue
        out_parts.append(text[cursor:open_start])
        out_parts.append(f"[attached {mime_type}, {len(data)} bytes]")
        attachments.append(Attachment(data=data, mime_type=mime_type))
        cursor = close_idx + len(BINARY_ATTACHMENT_CLOSE)
    out_parts.append(text[cursor:])
    return "".join(out_parts), attachments


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
        thinking_callback: ThinkingCallback | None = None,
    ):
        """Initialize a Model instance.

        Args:
            model_name: The name/identifier of the model.
            model_config: Optional dictionary of model configuration parameters.
            token_callback: Optional callback invoked with each streamed text token.
            thinking_callback: Optional callback invoked with ``True`` when a
                thinking block starts and ``False`` when it ends.  Used by
                printers to switch between thinking and text display modes.
        """
        self.model_name = model_name
        self.model_config = model_config or {}
        self.token_callback = token_callback
        self.thinking_callback = thinking_callback
        self.usage_info_for_messages: str = ""
        self.conversation: list[Any] = []
        self.client: Any = None

    def _invoke_token_callback(self, token: str) -> None:
        """Invoke the token callback synchronously."""
        if self.token_callback is not None:
            self.token_callback(token)

    def _invoke_thinking_callback(self, is_start: bool) -> None:
        """Invoke the thinking callback synchronously.

        Args:
            is_start: ``True`` when a thinking block starts, ``False`` when it ends.
        """
        if self.thinking_callback is not None:
            self.thinking_callback(is_start)

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
                if msg.get("tool_calls"):
                    return [
                        (tc["function"]["name"], tc["id"]) for tc in msg["tool_calls"]
                    ]
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
            # Strip binary attachment payloads — the default OpenAI-style
            # ``role: tool`` message does not accept image content blocks,
            # so we drop the base64 bytes and keep only the placeholder
            # text so the conversation does not balloon to megabytes.
            result_content, _ = parse_binary_attachments(result_content)
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
    ) -> tuple[int, int, int, int] | tuple[int, int, int, int, int]:
        """Extracts token counts from an API response.

        Args:
            response: The raw API response object.

        Returns:
            A 4-tuple ``(input_tokens, output_tokens, cache_read_tokens,
            cache_write_tokens)`` or a 5-tuple whose last element is the
            Anthropic one-hour cache-write token count.
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

    def _estimate_conversation_tokens(self, msgs: list[Any] | None = None) -> int:
        """Rough estimate of conversation size in tokens (chars / 4).

        Args:
            msgs: Message list to estimate. Defaults to self.conversation.
        """
        total_chars = 0
        for msg in (msgs if msgs is not None else self.conversation):
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        for v in block.values():
                            if isinstance(v, str):
                                total_chars += len(v)
        return total_chars // 2

    def compact_conversation(self, max_context_tokens: int) -> None:
        """Drop old messages to keep conversation within half the context limit.

        Preserves the initial user/human messages and keeps the last N messages,
        decrementing N from 50 until the conversation fits within half the token
        limit. Falls back to keeping only the initial messages if nothing fits.
        """
        if self._estimate_conversation_tokens() <= int(max_context_tokens * 0.7):
            return

        keep_start = 0
        for i, msg in enumerate(self.conversation):
            if msg.get("role") in ("user", "human"):
                keep_start = i + 1
                break

        keep_recent = 30
        while keep_recent > 0:
            tail_start = max(keep_start, len(self.conversation) - keep_recent)
            candidate = self.conversation[:keep_start] + self.conversation[tail_start:]
            if self._estimate_conversation_tokens(candidate) <= max_context_tokens // 2:
                self.conversation = candidate
                return
            keep_recent -= 1

        self.conversation = self.conversation[:keep_start]

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

    # =========================================================================
    # Helper methods for building tool schemas (shared across implementations)
    # ========================================================================

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

        param_descriptions = self._parse_docstring_params(doc)

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            param_type = param.annotation
            param_schema = self._python_type_to_json_schema(param_type)

            if param_name in param_descriptions:
                param_schema["description"] = param_descriptions[param_name]

            properties[param_name] = param_schema

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

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
                parts = stripped.split(":", 1)
                if len(parts) == 2:  # pragma: no branch
                    param_part = parts[0].strip()
                    desc_part = parts[1].strip()
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

        if origin is Union or origin is types_module.UnionType:
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                return self._python_type_to_json_schema(non_none_args[0])
            return {"anyOf": [self._python_type_to_json_schema(a) for a in non_none_args]}

        if origin is list:
            if args:
                return {
                    "type": "array",
                    "items": self._python_type_to_json_schema(args[0]),
                }
            return {"type": "array"}

        if origin is dict:
            return {"type": "object"}

        type_mapping: dict[type, dict[str, str]] = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            type(None): {"type": "null"},
        }

        if python_type in type_mapping:
            return type_mapping[python_type]

        return {"type": "string"}


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

CRITICAL RULES:
1. When you want to call a tool, output ONLY the JSON object with tool_calls.
2. STOP IMMEDIATELY after outputting the tool_calls JSON. Do NOT continue generating.
3. Do NOT predict, simulate, or hallucinate what the tool results will be.
4. The system will execute the tools and provide actual results in the next message.
5. When you have the final answer, call the `finish` tool with your result.
"""


def _iter_balanced_json_objects(
    content: str,
) -> "list[tuple[int, int, Any]]":
    """Find every balanced ``{...}`` substring that parses as valid JSON.

    Scans *content* character by character.  At every ``{`` that is not
    inside a JSON string, walks forward tracking brace depth and string
    state (with ``\\`` escape handling) until the matching ``}`` is found,
    then attempts to parse the substring with :func:`json.loads`.  This
    correctly handles nested objects, arrays, and brackets/braces embedded
    inside JSON string values — cases the previous regex-based approach
    silently dropped.

    Args:
        content: The text to scan.

    Returns:
        A list of ``(start, end_exclusive, parsed)`` tuples in left-to-right
        order, one per successfully parsed top-level balanced object.
    """
    results: list[tuple[int, int, Any]] = []
    i, n = 0, len(content)
    while i < n:
        if content[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        in_str = False
        esc = False
        end = -1
        while j < n:
            ch = content[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
            j += 1
        if end == -1:
            # Unbalanced — no more complete objects can start here.
            break
        try:
            # ``strict=False`` permits raw control characters (e.g. literal
            # newlines, tabs) inside JSON string values.  Reasoning models
            # such as ``cc/opus`` routinely emit unescaped newlines inside
            # ``summary`` arguments, which strict JSON would reject.
            parsed = json.loads(content[i:end], strict=False)
        except json.JSONDecodeError:
            i += 1
            continue
        results.append((i, end, parsed))
        i = end
    return results


def _iter_tool_calls_lists(obj: Any) -> "list[list[Any]]":
    """Recursively collect every ``tool_calls`` list inside *obj*.

    Walks dicts and lists looking for any ``"tool_calls"`` key whose value
    is a list.  This supports both top-level ``{"tool_calls": [...]}``
    objects and tool_calls nested inside an outer wrapper such as
    ``{"outer": {"tool_calls": [...]}}``.

    Args:
        obj: A parsed JSON value (dict, list, scalar).

    Returns:
        Lists of tool-call entries in encounter order.
    """
    out: list[list[Any]] = []
    if isinstance(obj, dict):
        tcs = obj.get("tool_calls")
        if isinstance(tcs, list):
            out.append(tcs)
        for v in obj.values():
            out.extend(_iter_tool_calls_lists(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_iter_tool_calls_lists(item))
    return out


def _parse_text_based_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse tool calls from text-based model output.

    Uses a brace-balanced JSON scanner (see
    :func:`_iter_balanced_json_objects`) to find every valid JSON object in
    *content*, then collects every nested ``tool_calls`` list.  Duplicates
    (same ``name`` + same ``arguments``) are removed so a single logical
    call emitted multiple times collapses to one entry.

    This robustly handles reasoning models (e.g. ``cc/opus``) that:

    - emit bare ``{"tool_calls": [...]}`` JSON without code fences,
    - include arguments containing nested objects, arrays, or strings with
      ``[]``/``{}`` characters (e.g. markdown links inside a finish summary),
    - emit several distinct ``tool_calls`` blocks in one response.

    Args:
        content: The text content to parse for tool calls.

    Returns:
        A list of function call dictionaries, each containing ``id``,
        ``name``, and ``arguments`` keys.  Returns an empty list if no
        valid tool calls are found.
    """
    function_calls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for _start, _end, parsed in _iter_balanced_json_objects(content):
        for tcs in _iter_tool_calls_lists(parsed):
            for tc in tcs:
                if not (isinstance(tc, dict) and "name" in tc):
                    continue
                arguments = tc.get("arguments", {})
                try:
                    key = (tc["name"], json.dumps(arguments, sort_keys=True))
                except TypeError:
                    # Non-JSON-serializable args — fall back to repr for keying.
                    key = (tc["name"], repr(arguments))
                if key in seen:
                    continue
                seen.add(key)
                function_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": tc["name"],
                        "arguments": arguments,
                    }
                )

    return function_calls


# Matches an empty fenced code block left behind after the JSON inside it
# has been stripped — e.g. ``"```json\n\n```"`` or ``"```\n\n```"``.
_EMPTY_FENCE_PATTERN = re.compile(r"```(?:json)?\s*```", re.DOTALL)


def _strip_text_based_tool_calls(content: str) -> str:
    """Remove tool_calls JSON blocks from *content*, keeping surrounding text.

    Uses :func:`_iter_balanced_json_objects` to find every balanced JSON
    object and strips those that contain a ``tool_calls`` list.  Empty
    fenced code blocks left behind (e.g. ``"```json\\n\\n```"``) are also
    cleaned up so the visible Thoughts panel does not show stray fences.

    Args:
        content: The full model response text.

    Returns:
        The text with tool_calls JSON removed, stripped of leading/trailing
        whitespace.  Returns ``""`` if the entire content was a tool_calls
        wrapper.
    """
    spans = [
        (start, end)
        for start, end, parsed in _iter_balanced_json_objects(content)
        if _iter_tool_calls_lists(parsed)
    ]
    if not spans:
        return content.strip()
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(content[cursor:start])
        cursor = end
    parts.append(content[cursor:])
    return _EMPTY_FENCE_PATTERN.sub("", "".join(parts)).strip()
