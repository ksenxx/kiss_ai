# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Gemini model implementation for Google's GenAI models."""

import logging
import uuid
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Attachment, Model, ThinkingCallback, TokenCallback

logger = logging.getLogger(__name__)

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

    def _convert_conversation_to_gemini_contents(self) -> list[types.Content]:
        """Converts the internal conversation format to Gemini contents.

        Returns:
            list[types.Content]: The conversation in Gemini API format.
        """
        contents = []
        for msg in self.conversation:
            role = msg["role"]
            content = msg.get("content", "")

            parts = []

            if role == "user":
                gemini_role = "user"
                if isinstance(content, str):
                    for att in msg.get("attachments", []):
                        parts.append(types.Part.from_bytes(data=att.data, mime_type=att.mime_type))
                    parts.append(types.Part.from_text(text=content))

            elif role == "assistant":
                gemini_role = "model"
                if isinstance(content, str) and content:
                    parts.append(types.Part.from_text(text=content))

                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        args = fn.get("arguments")
                        args = args if isinstance(args, dict) else {}
                        call_id = tc.get("id")
                        thought_sig = self._thought_signatures.get(call_id) if call_id else None
                        if thought_sig:
                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=fn.get("name"), args=args
                                    ),
                                    thought_signature=thought_sig,
                                )
                            )
                        else:
                            parts.append(
                                types.Part.from_function_call(name=fn.get("name"), args=args)
                            )

            elif role == "tool":
                gemini_role = "user"

                tool_call_id = msg.get("tool_call_id")
                func_name = "unknown"
                for prev_msg in reversed(self.conversation):
                    if prev_msg is msg:
                        continue
                    if prev_msg["role"] == "assistant" and prev_msg.get("tool_calls"):
                        for tc in prev_msg["tool_calls"]:
                            if tc["id"] == tool_call_id:
                                func_name = tc["function"]["name"]
                                break
                    if func_name != "unknown":
                        break

                import json

                try:
                    if isinstance(content, str):
                        response_dict = json.loads(content)
                    else:
                        response_dict = {"result": content}
                except json.JSONDecodeError:
                    logger.debug("Exception caught", exc_info=True)
                    response_dict = {"result": content}

                thought_sig = self._thought_signatures.get(tool_call_id)
                if thought_sig:
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=func_name, response=response_dict
                            ),
                            thought_signature=thought_sig,
                        )
                    )
                else:
                    parts.append(
                        types.Part.from_function_response(name=func_name, response=response_dict)
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

    def _build_config(self, tools: list[types.Tool] | None = None) -> types.GenerateContentConfig:
        thinking_config = self.model_config.get("thinking_config")
        if thinking_config is None:
            thinking_config = types.ThinkingConfig(include_thoughts=True)
        return types.GenerateContentConfig(
            max_output_tokens=self.model_config.get("max_tokens"),
            temperature=self.model_config.get("temperature"),
            top_p=self.model_config.get("top_p"),
            stop_sequences=self.model_config.get("stop"),
            thinking_config=thinking_config,
            tools=tools,  # type: ignore[arg-type]
            system_instruction=self.model_config.get("system_instruction"),
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

        self.conversation.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": fc["id"],
                        "type": "function",
                        "function": {"name": fc["name"], "arguments": fc["arguments"]},
                    }
                    for fc in function_calls
                ]
                if function_calls
                else None,
            }
        )

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
            return prompt_tokens - cached_tokens, output_tokens, cached_tokens, 0
        return 0, 0, 0, 0

    def get_embedding(  # pragma: no cover – API call
        self, text: str, embedding_model: str | None = None,
    ) -> list[float]:
        """Generates an embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name. Defaults to "text-embedding-004".

        Returns:
            list[float]: The embedding vector as a list of floats.

        Raises:
            KISSError: If embedding generation fails.
        """
        model_to_use = embedding_model or "text-embedding-004"
        try:
            response = self.client.models.embed_content(model=model_to_use, contents=text)
            return list(response.embeddings[0].values)
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            raise KISSError(f"Embedding generation failed for model {model_to_use}: {e}") from e
