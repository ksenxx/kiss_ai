# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Gemini model implementation for Google's GenAI models."""

import base64
import binascii
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import (
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    parse_binary_attachments,
    responses_items_to_chat_messages,
)

logger = logging.getLogger(__name__)


def _coerce_args_dict(args: Any) -> dict[str, Any]:
    """Coerce a tool-call ``arguments`` value into a dict.

    GeminiModel stores arguments as dicts, but a conversation handed off
    from an OpenAI-schema model (e.g. via the Sorcar ``set_model`` tool)
    stores them as JSON strings.  Unparseable values degrade to ``{}``.

    Args:
        args: The arguments value (dict, JSON string, or anything else).

    Returns:
        The arguments as a dict.
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            logger.debug("Exception caught", exc_info=True)
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _decode_base64(data: str) -> bytes | None:
    """Decode base64 text, returning ``None`` on invalid input."""
    try:
        return base64.b64decode(data)
    except (ValueError, binascii.Error):
        logger.debug("Exception caught", exc_info=True)
        return None


def _data_url_to_part(url: str, default_mime: str) -> types.Part | None:
    """Convert a base64 ``data:`` URL into a Gemini ``Part``.

    Args:
        url: The candidate data URL.
        default_mime: MIME type to use when the URL omits one.

    Returns:
        The equivalent Gemini ``Part``, or ``None`` when *url* is not a
        decodable base64 data URL.
    """
    if not (url.startswith("data:") and ";base64," in url):
        return None
    header, _, payload = url.partition(",")
    data = _decode_base64(payload)
    if data is None:
        return None
    media_type = header[len("data:"):].split(";", 1)[0] or default_mime
    return types.Part.from_bytes(data=data, mime_type=media_type)


def _media_block_to_part(block: dict[str, Any]) -> types.Part | None:
    """Convert a foreign media block/part into a Gemini ``Part``.

    Handles Anthropic ``image`` / ``document`` blocks (base64 or url
    source) and OpenAI ``image_url`` / ``file`` content parts (base64
    data URLs).  Such blocks enter the conversation when it is handed off
    from another provider's model (e.g. via the Sorcar ``set_model``
    tool).  Remote (non-data) URLs cannot be inlined and are dropped.

    Args:
        block: The foreign media block/part dict.

    Returns:
        The equivalent Gemini ``Part``, or ``None`` when the block cannot
        be represented (in which case it is dropped with a warning).
    """
    block_type = block.get("type")
    if block_type in ("image", "document"):
        source = block.get("source") or {}
        if source.get("type") == "base64":
            data = _decode_base64(source.get("data", ""))
            if data is not None:
                media_type = source.get("media_type", "application/octet-stream")
                return types.Part.from_bytes(data=data, mime_type=media_type)
    elif block_type == "image_url":
        url = (block.get("image_url") or {}).get("url", "")
        part = _data_url_to_part(url, "image/png")
        if part is not None:
            return part
    elif block_type == "file":
        file_data = (block.get("file") or {}).get("file_data", "")
        part = _data_url_to_part(file_data, "application/pdf")
        if part is not None:
            return part
    logger.warning("Dropping unconvertible %s block for Gemini.", block_type)
    return None


def _tool_result_response_dict(content: Any) -> dict[str, Any]:
    """Build a ``FunctionResponse.response`` dict from tool-result content.

    Args:
        content: A tool result payload — a string (possibly JSON) or a
            list of Anthropic nested blocks whose text is extracted.

    Returns:
        A JSON-serializable dict for ``FunctionResponse.response``.
    """
    if isinstance(content, list):
        content = "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError:
            logger.debug("Exception caught", exc_info=True)
            return {"result": content}
    return {"result": content}


class GeminiModel(Model):
    """A model that uses Google's GenAI API (Gemini)."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
    ):
        """Initialize a GeminiModel instance.

        Args:
            model_name: The name of the Gemini model to use.
            api_key: The Google API key for authentication.
            model_config: Optional dictionary of model configuration parameters.
            token_callback: Optional callback invoked with each streamed text token.
            thinking_callback: Optional callback invoked with ``True`` when a
                thinking block starts and ``False`` when it ends.
        """
        super().__init__(
            model_name,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
        self.api_key = api_key
        self._thought_signatures: dict[str, bytes] = {}
        self._in_thinking_stream: bool = False

    def reset_conversation(self) -> None:
        """Reset conversation state including thought signatures."""
        super().reset_conversation()
        self._thought_signatures = {}

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Initializes the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt to start the conversation.
            attachments: Optional list of file attachments (images, PDFs) to include.
        """
        self.client = genai.Client(api_key=self.api_key)
        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if attachments:
            msg["attachments"] = attachments
        self.conversation = [msg]
        self._thought_signatures = {}

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Add tool results to the conversation, lifting binary attachments.

        Gemini's ``FunctionResponse.response`` is a JSON dict and cannot
        carry raw bytes, so binary attachments produced by the ``Read``
        tool (e.g. a screenshot, audio, or video clip) are stripped from
        the tool message and re-attached as a follow-up ``user`` message
        whose ``attachments`` field is rendered via
        :meth:`_convert_conversation_to_gemini_contents` into
        :class:`google.genai.types.Part` instances using
        ``Part.from_bytes`` — which accepts any Gemini-supported MIME
        type (images, PDFs, audio, video).

        Args:
            function_results: List of ``(function_name, result_dict)`` tuples.
        """
        tool_calls = self._find_tool_call_ids_from_last_assistant()
        pending_attachments: list[Attachment] = []

        for i, (func_name, result_dict) in enumerate(function_results):
            result_content = result_dict.get("result", str(result_dict))
            result_content, attachments = parse_binary_attachments(result_content)
            if self.usage_info_for_messages:
                result_content = f"{result_content}\n\n{self.usage_info_for_messages}"

            tool_call_id = (
                tool_calls[i][1] if i < len(tool_calls) else f"call_{func_name}_{i}"
            )
            self.conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_content,
                }
            )
            pending_attachments.extend(attachments)

        if pending_attachments:
            self.conversation.append(
                {
                    "role": "user",
                    "content": "[attachments from previous tool result(s)]",
                    "attachments": pending_attachments,
                }
            )

    def _tool_call_id_to_name_map(self) -> dict[str, str]:
        """Map tool-call ids to function names across the whole conversation.

        Scans assistant messages for both OpenAI-style ``tool_calls``
        entries and Anthropic-style ``tool_use`` content blocks (present
        when the conversation was handed off from another provider's
        model, e.g. via the Sorcar ``set_model`` tool).

        Returns:
            dict[str, str]: Mapping of tool-call id to function name.
        """
        mapping: dict[str, str] = {}
        for msg in responses_items_to_chat_messages(self.conversation):
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if tc.get("id"):
                    mapping[tc["id"]] = fn.get("name", "unknown")
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if block.get("id"):
                            mapping[block["id"]] = block.get("name", "unknown")
        return mapping

    def _function_call_part(self, name: str, args: dict[str, Any], call_id: Any) -> types.Part:
        """Build a Gemini ``function_call`` part, re-attaching any thought signature.

        Args:
            name: The function name.
            args: The function arguments dict.
            call_id: The tool-call id used to look up a stored thought signature.

        Returns:
            types.Part: The function-call part.
        """
        thought_sig = self._thought_signatures.get(call_id) if call_id else None
        if thought_sig:
            return types.Part(
                function_call=types.FunctionCall(name=name, args=args),
                thought_signature=thought_sig,
            )
        return types.Part.from_function_call(name=name, args=args)

    def _function_response_part(
        self, name: str, content: Any, call_id: Any
    ) -> types.Part:
        """Build a Gemini ``function_response`` part, re-attaching any thought signature.

        Args:
            name: The function name the result belongs to.
            content: The tool result payload (string or nested block list).
            call_id: The tool-call id used to look up a stored thought signature.

        Returns:
            types.Part: The function-response part.
        """
        response_dict = _tool_result_response_dict(content)
        thought_sig = self._thought_signatures.get(call_id) if call_id else None
        if thought_sig:
            return types.Part(
                function_response=types.FunctionResponse(name=name, response=response_dict),
                thought_signature=thought_sig,
            )
        return types.Part.from_function_response(name=name, response=response_dict)

    def _block_list_to_parts(
        self, blocks: list[Any], id_to_name: dict[str, str]
    ) -> list[types.Part]:
        """Convert a foreign content-block list into Gemini parts.

        Handles Anthropic Messages blocks (``text`` / ``tool_use`` /
        ``tool_result`` / ``image`` / ``document``; ``thinking`` blocks are
        hidden provider state and are dropped) and OpenAI Chat Completions
        content parts (``text`` / ``image_url`` / ``file``).  Such block
        lists enter the conversation when it is handed off from another
        provider's model (e.g. via the Sorcar ``set_model`` tool).

        Args:
            blocks: The content-block list.
            id_to_name: Mapping of tool-call id to function name.

        Returns:
            list[types.Part]: The equivalent Gemini parts.
        """
        parts: list[types.Part] = []
        for block in blocks:
            if not isinstance(block, dict):
                text = str(block)
                if text.strip():
                    parts.append(types.Part.from_text(text=text))
                continue
            block_type = block.get("type")
            if block_type in ("thinking", "redacted_thinking"):
                continue
            if block_type == "text":
                text = block.get("text", "")
                if text.strip():
                    parts.append(types.Part.from_text(text=text))
            elif block_type == "tool_use":
                parts.append(
                    self._function_call_part(
                        block.get("name", "unknown"),
                        _coerce_args_dict(block.get("input")),
                        block.get("id"),
                    )
                )
            elif block_type == "tool_result":
                call_id = block.get("tool_use_id")
                parts.append(
                    self._function_response_part(
                        id_to_name.get(call_id or "", "unknown"),
                        block.get("content"),
                        call_id,
                    )
                )
            elif block_type in ("image", "document", "image_url", "file"):
                part = _media_block_to_part(block)
                if part is not None:
                    parts.append(part)
            else:
                logger.warning("Dropping unsupported %s block for Gemini.", block_type)
        return parts

    def _convert_conversation_to_gemini_contents(self) -> list[types.Content]:
        """Converts the internal conversation format to Gemini contents.

        Besides GeminiModel's native format, this also converts foreign
        formats that enter the conversation when it is handed off from
        another provider's model (e.g. via the Sorcar ``set_model`` tool):
        OpenAI ``tool_calls`` with JSON-string arguments, ``role="tool"``
        messages, ``role="system"`` messages (hoisted into
        ``system_instruction`` by :meth:`_build_config` and skipped here),
        and Anthropic content-block lists.

        Returns:
            list[types.Content]: The conversation in Gemini API format.
        """
        id_to_name = self._tool_call_id_to_name_map()
        contents = []
        for msg in responses_items_to_chat_messages(self.conversation):
            role = msg["role"]
            content = msg.get("content", "")

            parts: list[types.Part] = []

            if role == "user":
                gemini_role = "user"
                if isinstance(content, str):
                    for att in msg.get("attachments", []):
                        parts.append(types.Part.from_bytes(data=att.data, mime_type=att.mime_type))
                    parts.append(types.Part.from_text(text=content))
                elif isinstance(content, list):
                    parts.extend(self._block_list_to_parts(content, id_to_name))

            elif role == "assistant":
                gemini_role = "model"
                if isinstance(content, str) and content:
                    parts.append(types.Part.from_text(text=content))
                elif isinstance(content, list):
                    parts.extend(self._block_list_to_parts(content, id_to_name))

                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    parts.append(
                        self._function_call_part(
                            fn.get("name"),
                            _coerce_args_dict(fn.get("arguments")),
                            tc.get("id"),
                        )
                    )

            elif role == "tool":
                gemini_role = "user"
                tool_call_id = msg.get("tool_call_id")
                parts.append(
                    self._function_response_part(
                        id_to_name.get(tool_call_id or "", "unknown"),
                        content,
                        tool_call_id,
                    )
                )

            else:
                continue

            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))

        return contents

    @staticmethod
    def _parts_from_response(response: Any) -> list[Any]:
        """Extract parts from a Gemini response or chunk."""
        if response and response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:  # pragma: no branch
                return list(candidate.content.parts)
        return []

    def _parse_parts(self, parts: list[Any]) -> tuple[str, list[dict[str, Any]]]:
        """Build content and function calls from Gemini parts."""
        content = ""
        function_calls: list[dict[str, Any]] = []
        for part in parts:
            if part.text:
                content += part.text
            if part.function_call:
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                if part.thought_signature:
                    self._thought_signatures[call_id] = part.thought_signature
                function_calls.append(
                    {
                        "id": call_id,
                        "name": part.function_call.name,
                        "arguments": part.function_call.args,
                    }
                )
        return content, function_calls

    def _resolve_system_instruction(self) -> str | None:
        """Merge configured and conversation-level system instructions.

        OpenAI-style ``role="system"`` messages (present when the
        conversation was handed off from an OpenAI-schema model, e.g. via
        the Sorcar ``set_model`` tool) are hoisted into Gemini's
        ``system_instruction`` config parameter, since Gemini contents
        accept only ``user`` / ``model`` roles.  Duplicates of the
        configured ``system_instruction`` are skipped.

        Returns:
            str | None: The merged system instruction, or ``None``.
        """
        configured = self.model_config.get("system_instruction")
        system_texts: list[str] = [configured] if configured else []
        for msg in responses_items_to_chat_messages(self.conversation):
            if msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if isinstance(content, str) and content.strip() and content not in system_texts:
                system_texts.append(content)
        return "\n\n".join(system_texts) if system_texts else None

    def _build_config(self, tools: list[types.Tool] | None = None) -> types.GenerateContentConfig:
        thinking_config = self.model_config.get("thinking_config")
        if thinking_config is None:
            thinking_config = types.ThinkingConfig(include_thoughts=True)
        max_output_tokens = self.model_config.get("max_tokens")
        if max_output_tokens is None:
            max_output_tokens = self.model_config.get("max_completion_tokens")
        return types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=self.model_config.get("temperature"),
            top_p=self.model_config.get("top_p"),
            stop_sequences=self.model_config.get("stop"),
            thinking_config=thinking_config,
            tools=tools,  # type: ignore[arg-type]
            system_instruction=self._resolve_system_instruction(),
        )

    def _stream_parts(self, parts: list[Any]) -> None:
        """Stream parts, routing thinking tokens through the thinking callback.

        Tracks thinking state across calls so that multiple chunks of thinking
        parts produce a single ``thinking_callback(True)`` … ``thinking_callback(False)``
        boundary pair.

        Args:
            parts: Gemini response parts from a single streaming chunk.
        """
        for part in parts:
            if not part.text:
                continue
            is_thought = getattr(part, "thought", None) is True
            if is_thought:
                if not self._in_thinking_stream:
                    self._in_thinking_stream = True
                    self._invoke_thinking_callback(True)
            else:
                if self._in_thinking_stream:
                    self._in_thinking_stream = False
                    self._invoke_thinking_callback(False)
            self._invoke_token_callback(part.text)

    def _end_thinking_stream(self) -> None:
        """Close an open thinking block after streaming completes.

        Must be called after a streaming loop finishes to ensure the
        thinking panel is closed if the last streamed part was a thought.
        """
        if self._in_thinking_stream:
            self._in_thinking_stream = False
            self._invoke_thinking_callback(False)

    def generate(self) -> tuple[str, Any]:  # pragma: no cover – API call
        """Generates content from prompt without tools.

        Returns:
            tuple[str, Any]: A tuple of (generated_text, raw_response).
        """
        contents = self._convert_conversation_to_gemini_contents()
        config = self._build_config()

        if self.token_callback is not None:
            content = ""
            response = None
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name, contents=contents, config=config
            ):
                self._stream_parts(self._parts_from_response(chunk))
                if chunk.text:
                    content += chunk.text
                response = chunk
            self._end_thinking_stream()
            if response is None:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
                content = response.text or ""
        else:
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )
            content = response.text or ""

        self.conversation.append({"role": "assistant", "content": content})
        return content, response

    def generate_and_process_with_tools(  # pragma: no cover – API call
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generates content with tools, processes the response, and adds it to conversation.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Optional pre-built OpenAI-format tool schema list.

        Returns:
            tuple[list[dict[str, Any]], str, Any]: A tuple of
                (function_calls, response_text, raw_response).
        """

        source = self._resolve_openai_tools_schema(function_map, tools_schema)
        declarations = []
        for tool in source:
            fn = tool["function"]
            declarations.append(
                types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description"),
                    parameters=fn.get("parameters"),
                )
            )
        gemini_tools = [types.Tool(function_declarations=declarations)] if declarations else None

        contents = self._convert_conversation_to_gemini_contents()
        config = self._build_config(tools=gemini_tools)

        all_parts: list[Any] = []
        if self.token_callback is not None:
            response = None
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name, contents=contents, config=config
            ):
                response = chunk
                parts = self._parts_from_response(chunk)
                self._stream_parts(parts)
                all_parts.extend(parts)
            self._end_thinking_stream()
            if response is None:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
                all_parts = self._parts_from_response(response)
                self._stream_parts(all_parts)
        else:
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )
            all_parts = self._parts_from_response(response)

        content, function_calls = self._parse_parts(all_parts)

        # Omit the ``tool_calls`` key entirely when the model made no
        # calls: a ``tool_calls: None`` entry would be replayed verbatim
        # after a model hand-off and rejected by other providers (e.g.
        # the OpenAI Responses API fails with "Unknown parameter:
        # 'input[N].tool_calls'").
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if function_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": fc["id"],
                    "type": "function",
                    "function": {"name": fc["name"], "arguments": fc["arguments"]},
                }
                for fc in function_calls
            ]
        self.conversation.append(assistant_msg)

        return function_calls, content, response

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int]:
        """Extracts token counts from a Gemini API response.

        Returns:
            (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).
        """
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            prompt_tokens = um.prompt_token_count or 0
            output_tokens = um.candidates_token_count or 0
            thoughts_tokens = getattr(um, "thoughts_token_count", 0) or 0
            output_tokens += thoughts_tokens
            cached_tokens = getattr(um, "cached_content_token_count", 0) or 0
            return max(prompt_tokens - cached_tokens, 0), output_tokens, cached_tokens, 0
        return 0, 0, 0, 0

    def get_embedding(  # pragma: no cover – API call
        self, text: str, embedding_model: str | None = None,
    ) -> list[float]:
        """Generates an embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name. Defaults to "gemini-embedding-001".

        Returns:
            list[float]: The embedding vector as a list of floats.

        Raises:
            KISSError: If embedding generation fails.
        """
        model_to_use = embedding_model or "gemini-embedding-001"
        try:
            response = self.client.models.embed_content(model=model_to_use, contents=text)
            return list(response.embeddings[0].values)
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            raise KISSError(f"Embedding generation failed for model {model_to_use}: {e}") from e
