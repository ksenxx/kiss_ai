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

import httpx
from google import genai
from google.genai import types

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import (
    FRAMEWORK_ONLY_CONFIG_KEYS,
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    responses_items_to_chat_messages,
)
from kiss.core.models.stream_abort import stall_error, stop_aware_events

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10.0

# The request parameters Gemini accepts, taken from the SDK's own config
# model.  ``GenerateContentConfig`` forbids unknown fields, so this is the
# authoritative set of ``model_config`` keys that can be forwarded.
_GEMINI_CONFIG_FIELDS = frozenset(types.GenerateContentConfig.model_fields)


class _ResponseTrackingHttpxClient(httpx.Client):
    """The transport ``GeminiModel`` gives ``genai.Client``.

    Two things the SDK cannot otherwise provide:

    * **A timeout.**  ``ApiClient._request_once`` passes an explicit
      ``timeout=`` to ``build_request``, which is ``None`` unless
      ``HttpOptions.timeout`` is set — and an explicit ``None`` in httpx
      means *no timeout at all*, overriding any client default.  Forcing
      the timeout here installs a real read deadline (the stall clock)
      while keeping a short connect deadline, matching
      :class:`~kiss.core.models.anthropic_model.AnthropicModel`.
    * **The streamed response.**  ``generate_content_stream`` hands back
      a bare generator, so :class:`~kiss.core.models.stream_abort.StreamAbortWatchdog`
      has nothing whose socket it can shut down.  Remembering the
      in-flight response here is what lets a Stop unblock a thread parked
      in ``recv()``.
    """

    def __init__(self, stall_timeout: float) -> None:
        """Build the transport.

        Args:
            stall_timeout: Seconds of byte-level silence tolerated before
                httpx raises ``ReadTimeout``.
        """
        super().__init__(follow_redirects=True)
        self._timeout = httpx.Timeout(stall_timeout, connect=_CONNECT_TIMEOUT)
        self.last_response: httpx.Response | None = None

    def build_request(self, *args: Any, **kwargs: Any) -> httpx.Request:
        """Build a request with this client's timeout, ignoring the caller's."""
        kwargs["timeout"] = self._timeout
        return super().build_request(*args, **kwargs)

    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        """Send *request*, remembering the response for the abort watchdog."""
        response = super().send(request, **kwargs)
        self.last_response = response
        return response


class _AbortableStream:
    """A Gemini stream in the shape :func:`stop_aware_events` expects.

    The watchdog needs to iterate the stream, reach the underlying
    ``httpx.Response`` (to half-close its socket) and close the stream.
    ``response`` is a property because the SDK generator is lazy: the
    request is not issued until the first ``next()``.
    """

    def __init__(self, chunks: Any, http_client: _ResponseTrackingHttpxClient) -> None:
        """Wrap the SDK's chunk generator.

        ``last_response`` is reset here because this stream's request has
        not been issued yet (the SDK generator is lazy): whatever the
        shared client still remembers belongs to a PREVIOUS request whose
        keep-alive connection may already be back in httpx's pool, and an
        abort must never shut that socket down.

        Args:
            chunks: The generator ``generate_content_stream`` returned.
            http_client: The transport that recorded the response.
        """
        self._chunks = chunks
        self._http_client = http_client
        self._response: httpx.Response | None = None
        self._closed = False
        http_client.last_response = None

    def __iter__(self) -> Any:
        """Iterate the SDK's chunks, snapshotting the in-flight response.

        The snapshot is taken at the first chunk so a watchdog abort that
        fires late — after this stream is dead and a retry has begun on
        the SAME shared client — still shuts down THIS request's socket,
        never the live retry's (``last_response`` is overwritten by every
        request on the shared client).
        """
        for chunk in self._chunks:
            if self._response is None:
                self._response = self._http_client.last_response
            yield chunk

    @property
    def response(self) -> httpx.Response | None:
        """This stream's httpx response, or ``None`` before the request.

        Cached on first read; after :meth:`close` only the cached value
        is ever returned, so a parked abort can no longer observe a later
        request through the shared client.
        """
        if self._response is None and not self._closed:
            self._response = self._http_client.last_response
        return self._response

    def close(self) -> None:
        """Close the SDK generator, freezing the response snapshot."""
        if self._response is None:
            self._response = self._http_client.last_response
        self._closed = True
        self._chunks.close()


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


def _is_thought(part: Any) -> bool:
    """Return whether *part* carries summarized reasoning rather than answer text.

    Args:
        part: A Gemini response part.

    Returns:
        ``True`` when the part is flagged ``thought``.
    """
    return getattr(part, "thought", None) is True


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
        self._http_client = _ResponseTrackingHttpxClient(self._stream_stall_timeout)

    def reset_conversation(self) -> None:
        """Reset conversation state including thought signatures.

        The signatures are keyed by the conversation the base class is
        clearing, so keeping them would replay a previous run's
        reasoning into the next one.  (The thinking bracket is cleared
        by the base implementation, for every adapter.)
        """
        super().reset_conversation()
        self._thought_signatures = {}

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Initializes the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt to start the conversation.
            attachments: Optional list of file attachments (images, PDFs) to include.
        """
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(httpx_client=self._http_client),
        )
        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if attachments:
            msg["attachments"] = attachments
        self.conversation = [msg]
        self._thought_signatures = {}

    def _chat_messages(self) -> list[dict[str, Any]]:
        """Return the conversation converted to Chat-Completions messages.

        The one place :func:`responses_items_to_chat_messages` runs for a
        request: :meth:`generate` and
        :meth:`generate_and_process_with_tools` convert once and pass the
        result to the three consumers (:meth:`_tool_call_id_to_name_map`,
        :meth:`_convert_conversation_to_gemini_contents`,
        :meth:`_resolve_system_instruction`) instead of each consumer
        re-running the full conversion.

        Returns:
            The conversation in Chat-Completions format.
        """
        return responses_items_to_chat_messages(self.conversation)

    def _tool_call_id_to_name_map(
        self, chat_messages: list[dict[str, Any]] | None = None
    ) -> dict[str, str]:
        """Map tool-call ids to function names across the whole conversation.

        Scans assistant messages for both OpenAI-style ``tool_calls``
        entries and Anthropic-style ``tool_use`` content blocks (present
        when the conversation was handed off from another provider's
        model, e.g. via the Sorcar ``set_model`` tool).

        Args:
            chat_messages: The pre-converted conversation, or ``None`` to
                convert ``self.conversation`` here.

        Returns:
            dict[str, str]: Mapping of tool-call id to function name.
        """
        mapping: dict[str, str] = {}
        for msg in (
            chat_messages if chat_messages is not None else self._chat_messages()
        ):
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

    def _convert_conversation_to_gemini_contents(
        self, chat_messages: list[dict[str, Any]] | None = None
    ) -> list[types.Content]:
        """Converts the internal conversation format to Gemini contents.

        Besides GeminiModel's native format, this also converts foreign
        formats that enter the conversation when it is handed off from
        another provider's model (e.g. via the Sorcar ``set_model`` tool):
        OpenAI ``tool_calls`` with JSON-string arguments, ``role="tool"``
        messages, ``role="system"`` messages (hoisted into
        ``system_instruction`` by :meth:`_build_config` and skipped here),
        and Anthropic content-block lists.

        A user message may also carry an ``attachments`` list, which becomes
        :class:`google.genai.types.Part` instances via ``Part.from_bytes``
        (any Gemini-supported MIME type: images, PDFs, audio, video).  That
        is how bytes a tool returned reach the model: Gemini's
        ``FunctionResponse.response`` is a JSON dict and cannot carry them,
        so
        :meth:`~kiss.core.models.model.Model._deliver_tool_result_attachments`
        appends them as a follow-up user message instead.

        Args:
            chat_messages: The pre-converted conversation, or ``None`` to
                convert ``self.conversation`` here.

        Returns:
            list[types.Content]: The conversation in Gemini API format.
        """
        if chat_messages is None:
            chat_messages = self._chat_messages()
        id_to_name = self._tool_call_id_to_name_map(chat_messages)
        contents = []
        for msg in chat_messages:
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
        """Build content and function calls from Gemini parts.

        Parts flagged ``thought=True`` carry summarized reasoning, which
        is shown live through the thinking callback but must never become
        assistant content: it would be printed twice, stored in the
        conversation and re-uploaded as prompt context on every later
        step.  This mirrors the SDK's own ``response.text`` property.
        """
        content = ""
        function_calls: list[dict[str, Any]] = []
        for part in parts:
            if part.text and not _is_thought(part):
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

    def _resolve_system_instruction(
        self, chat_messages: list[dict[str, Any]] | None = None
    ) -> str | None:
        """Merge configured and conversation-level system instructions.

        OpenAI-style ``role="system"`` messages (present when the
        conversation was handed off from an OpenAI-schema model, e.g. via
        the Sorcar ``set_model`` tool) are hoisted into Gemini's
        ``system_instruction`` config parameter, since Gemini contents
        accept only ``user`` / ``model`` roles.  Duplicates of the
        configured ``system_instruction`` are skipped.

        Args:
            chat_messages: The pre-converted conversation, or ``None`` to
                convert ``self.conversation`` here.

        Returns:
            str | None: The merged system instruction, or ``None``.
        """
        configured = self.model_config.get("system_instruction")
        system_texts: list[str] = [configured] if configured else []
        for msg in (
            chat_messages if chat_messages is not None else self._chat_messages()
        ):
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

    def _build_config(
        self,
        tools: list[types.Tool] | None = None,
        chat_messages: list[dict[str, Any]] | None = None,
    ) -> types.GenerateContentConfig:
        """Translate ``model_config`` into a Gemini generation config.

        Every ``model_config`` key that names a real
        :class:`~google.genai.types.GenerateContentConfig` field is
        forwarded — ``seed``, ``top_k``, ``presence_penalty``,
        ``response_mime_type`` and the rest included — after the portable
        aliases (``max_tokens`` / ``max_completion_tokens`` →
        ``max_output_tokens``, ``stop`` → ``stop_sequences``) are
        translated.  Keys Gemini has no field for are reported rather than
        dropped in silence: ``GenerateContentConfig`` forbids extras, so
        forwarding them blindly would raise a ``ValidationError``.

        Args:
            tools: Optional Gemini tool declarations for this request.
            chat_messages: The pre-converted conversation for
                :meth:`_resolve_system_instruction`, or ``None`` to
                convert ``self.conversation`` there.

        Returns:
            The config to pass to ``generate_content``.
        """
        params = {
            key: value
            for key, value in self.model_config.items()
            if key not in FRAMEWORK_ONLY_CONFIG_KEYS
        }
        max_output_tokens = params.pop("max_tokens", None)
        max_completion_tokens = params.pop("max_completion_tokens", None)
        if max_output_tokens is None:
            max_output_tokens = max_completion_tokens
        if max_output_tokens is not None:
            params.setdefault("max_output_tokens", max_output_tokens)
        stop = params.pop("stop", None)
        if stop is not None:
            params.setdefault("stop_sequences", stop)
        if params.get("thinking_config") is None:
            params["thinking_config"] = types.ThinkingConfig(include_thoughts=True)

        params = self._keep_supported_request_params(
            params, _GEMINI_CONFIG_FIELDS, "Gemini"
        )
        params["tools"] = tools
        params["system_instruction"] = self._resolve_system_instruction(chat_messages)
        return types.GenerateContentConfig(**params)

    def _stream_parts(self, parts: list[Any]) -> None:
        """Stream parts, routing thinking tokens through the thinking callback.

        Tracks thinking state across calls (on the base
        :attr:`Model._thinking_open` flag) so that multiple chunks of
        thinking parts produce a single ``thinking_callback(True)`` …
        ``thinking_callback(False)`` boundary pair.

        Args:
            parts: Gemini response parts from a single streaming chunk.
        """
        for part in parts:
            if not part.text:
                continue
            is_thought = _is_thought(part)
            if is_thought != self._thinking_open:
                self._invoke_thinking_callback(is_thought)
            self._invoke_token_callback(part.text)

    def _stream_turn(self, contents: list[types.Content], config: Any) -> tuple[list[Any], Any]:
        """Stream one turn, aborting on a user stop or a stalled connection.

        Args:
            contents: The conversation in Gemini API format.
            config: The generation config for this turn.

        Returns:
            The streamed parts, and the chunk to bill the turn against —
            the one carrying ``usage_metadata`` rather than whichever
            chunk happened to come last, since a terminal
            ``finishReason``-only chunk would otherwise report the step
            as free.  ``None`` when the stream produced nothing at all.

        The stall is bounded at both levels, because neither alone is
        enough:

        * **byte level** — the transport's ``httpx.Timeout`` raises
          ``ReadTimeout`` on a socket carrying nothing.
        * **event level** — the watchdog inside
          :func:`~kiss.core.models.stream_abort.stop_aware_events` aborts
          a stream that keeps *arriving* without ever yielding a chunk.
          ``ApiClient._iter_response_stream`` drops every blank line
          before yielding, so ordinary SSE keep-alives reset the
          byte-level clock while starving the agent indefinitely.

        Raises:
            KeyboardInterrupt: When the user stopped the task mid-stream.
            TimeoutError: When no chunk (or no byte) arrives for
                ``stream_stall_timeout`` seconds.
                ``KISSAgent._run_agentic_loop`` treats this as retryable
                and re-asks the model.
        """
        parts: list[Any] = []
        usage_chunk = None
        last_chunk = None
        stream = _AbortableStream(
            self.client.models.generate_content_stream(
                model=self.model_name, contents=contents, config=config
            ),
            self._http_client,
        )
        # `events` is closed in `finally`, mirroring the OpenAI
        # transports: the loop body can raise (a token callback
        # propagating Stop, most commonly), and an abandoned generator
        # runs its cleanup only when the traceback holding its frame is
        # released — until then a daemon watchdog thread stays alive and
        # armed over a connection that never returns to the pool.
        events = stop_aware_events(
            stream,
            stall_timeout=self._stream_stall_timeout,
            on_abort=self._close_thinking_if_open,
            name="gemini-stream-abort-watchdog",
        )
        try:
            for chunk in events:
                last_chunk = chunk
                if chunk.usage_metadata is not None:
                    usage_chunk = chunk
                chunk_parts = self._parts_from_response(chunk)
                self._stream_parts(chunk_parts)
                parts.extend(chunk_parts)
        except httpx.TimeoutException as e:
            # The byte-level clock fired first; report it in the same
            # words as the event-level watchdog would have.
            raise stall_error(self._stream_stall_timeout) from e
        finally:
            events.close()
            self._close_thinking_if_open()
        return parts, usage_chunk or last_chunk

    def _generate_parts(
        self, contents: list[types.Content], config: Any
    ) -> tuple[list[Any], Any]:
        """Run one turn and return its parts plus the response to bill it against.

        Streams when a token callback is bound, and falls back to the
        unary call when streaming is not wanted or the stream yielded no
        chunk at all.

        Args:
            contents: The conversation in Gemini API format.
            config: The generation config for this turn.

        Returns:
            The response parts and the raw response.
        """
        if self.token_callback is not None:
            parts, response = self._stream_turn(contents, config)
            if response is not None:
                return parts, response
        response = self.client.models.generate_content(
            model=self.model_name, contents=contents, config=config
        )
        parts = self._parts_from_response(response)
        if self.token_callback is not None:
            try:
                self._stream_parts(parts)
            finally:
                self._close_thinking_if_open()
        return parts, response

    def generate(self) -> tuple[str, Any]:
        """Generates content from prompt without tools.

        Returns:
            tuple[str, Any]: A tuple of (generated_text, raw_response).
        """
        chat_messages = self._chat_messages()
        contents = self._convert_conversation_to_gemini_contents(chat_messages)
        parts, response = self._generate_parts(
            contents, self._build_config(chat_messages=chat_messages)
        )
        content, _ = self._parse_parts(parts)
        self.conversation.append({"role": "assistant", "content": content})
        return content, response

    def generate_and_process_with_tools(
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

        chat_messages = self._chat_messages()
        contents = self._convert_conversation_to_gemini_contents(chat_messages)
        config = self._build_config(tools=gemini_tools, chat_messages=chat_messages)
        all_parts, response = self._generate_parts(contents, config)
        content, function_calls = self._parse_parts(all_parts)

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
            tool_use_tokens = getattr(um, "tool_use_prompt_token_count", 0) or 0
            input_tokens = max(prompt_tokens - cached_tokens, 0) + tool_use_tokens
            return input_tokens, output_tokens, cached_tokens, 0
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
