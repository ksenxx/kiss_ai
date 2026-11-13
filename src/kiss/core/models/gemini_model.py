# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Gemini model implementation for Google's GenAI models."""

import base64
import binascii
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, NamedTuple

import httpx
from google import genai
from google.genai import errors, types

from kiss.core import stop_signal
from kiss.core.kiss_error import ContentFilterError, KISSError, OutputCapError
from kiss.core.models.model import (
    FRAMEWORK_ONLY_CONFIG_KEYS,
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    _audio_format_to_mime,
    iter_assistant_tool_calls,
    merge_system_texts,
    parse_tool_arguments,
    pop_max_tokens_and_stop,
    responses_items_to_chat_messages,
    split_data_url,
    text_of_parts,
)
from kiss.core.models.model_info import MODEL_INFO, _strip_thinking_alias
from kiss.core.models.stream_abort import (
    CONNECT_TIMEOUT,
    RequestAbortWatchdog,
    make_pool_abortable,
    stall_error,
    stop_error,
    stop_or_stall_error,
)

logger = logging.getLogger(__name__)

# Portable effort levels above Gemini's top ``thinking_level``.
_GEMINI_TOP_LEVEL = {"xhigh": "high", "max": "high"}

# Whole-request deadline of an embeddings call, which answers in seconds:
# the read timeout only bounds the silence between two reads, so a peer
# that trickles its reply would hold the calling thread for good.
# ``model_config["timeout"]`` overrides it, as on the OpenAI transport.
_EMBEDDINGS_TIMEOUT = 60.0

# Deepest nesting of a JSON tool result that is sent parsed; a deeper one
# is sent as its text.  The SDK builds a request with one Python call per
# nesting level, so a result nested about a thousand deep raised
# ``RecursionError`` in every request of the conversation that held it.
_MAX_RESULT_NESTING = 100


# The request parameters Gemini accepts, taken from the SDK's own config
# model.  ``GenerateContentConfig`` forbids unknown fields, so this is the
# authoritative set of ``model_config`` keys that can be forwarded.
_GEMINI_CONFIG_FIELDS = frozenset(types.GenerateContentConfig.model_fields)

# The candidate ``finishReason`` values of a content filter stop (the
# ``FinishReason`` enum of the SDK): deterministic for the same request,
# like a blocked prompt, so the agentic loop fails over to another model
# family instead of retrying this one.
_FILTER_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    }
)

# The documented dummy thought signature for a function call Gemini never
# signed.  Gemini 3 and 4 models reject a request that continues a turn
# whose function calls carry no signature (400 INVALID_ARGUMENT), which
# is the first request after a conversation is handed off from another
# provider's model (Sorcar's set_model, the fallback swap) and the replay
# of a parallel batch, of which Gemini signs only the first call.  The
# dummy skips that validation: Gemini API docs, "Thought signatures"
# (generateContent), FAQ "How do I transfer history from a different
# model".
FOREIGN_CALL_SIGNATURE = b"skip_thought_signature_validator"


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
        self._timeout = httpx.Timeout(stall_timeout, connect=CONNECT_TIMEOUT)
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


class _EmbeddingClient(NamedTuple):
    """What one embedding at a time sends through: an SDK client over an abortable pool."""

    api_key: str  # the key ``sdk`` was built with
    pool: _ResponseTrackingHttpxClient
    sdk: genai.Client

    def close(self) -> None:
        """Close the SDK client, and the pool it leaves open."""
        self.sdk.close()
        self.pool.close()


_END_OF_STREAM = object()
"""Sentinel for a stream that ended before its first chunk."""


class _AbortableStream:
    """A Gemini stream in the shape :func:`stop_aware_events` expects.

    The watchdog needs to iterate the stream, reach the underlying
    ``httpx.Response`` (to half-close its socket) and close the stream.
    ``response`` is a property because the SDK generator is lazy: the
    request is not issued until the first ``next()``.
    """

    def __init__(
        self, chunks: Any, http_client: _ResponseTrackingHttpxClient, header_budget: float
    ) -> None:
        """Wrap the SDK's chunk generator.

        ``last_response`` is reset here because this stream's request has
        not been issued yet (the SDK generator is lazy): whatever the
        shared client still remembers belongs to a PREVIOUS request whose
        keep-alive connection may already be back in httpx's pool, and an
        abort must never shut that socket down.

        Args:
            chunks: The generator ``generate_content_stream`` returned.
            http_client: The transport that recorded the response.
            header_budget: Seconds the wait for the first chunk may take
                before the request is cut (see :meth:`__iter__`).
        """
        self._chunks = chunks
        self._http_client = http_client
        self._header_budget = header_budget
        self._response: httpx.Response | None = None
        self._closed = False
        http_client.last_response = None

    def __iter__(self) -> Any:
        """Iterate the SDK's chunks, snapshotting the in-flight response.

        The first chunk is awaited under a request watchdog: until the
        response headers are complete there is no response for the
        stream watchdog to half-close, and the byte-level clock only
        bounds the silence between two reads, so a peer trickling header
        bytes would hold the turn for good and past any Stop.  The
        watchdog cuts the client's pool at a Stop or at the header
        budget (the SDK's own retries run inside this wait).

        The snapshot is taken at the first chunk so a watchdog abort that
        fires late — after this stream is dead and a retry has begun on
        the SAME shared client — still shuts down THIS request's socket,
        never the live retry's (``last_response`` is overwritten by every
        request on the shared client).
        """
        with RequestAbortWatchdog(
            self._http_client, self._header_budget, "gemini-header-abort-watchdog"
        ):
            first = next(self._chunks, _END_OF_STREAM)
        if first is _END_OF_STREAM:
            return
        # The chunk came from the client's latest response, whatever an
        # earlier poll adopted (an attempt's error reply the SDK retried).
        self._response = self._http_client.last_response
        yield first
        yield from self._chunks

    @property
    def response(self) -> httpx.Response | None:
        """This stream's httpx response, or ``None`` before the request.

        Cached on first read; after :meth:`close` only the cached value
        is ever returned, so a parked abort can no longer observe a later
        request through the shared client.  A response the SDK already
        finished with (the error reply of an attempt it is retrying) is
        not this stream's: the watchdog keeps polling for the live one.
        """
        if self._response is None and not self._closed:
            latest = self._http_client.last_response
            if latest is not None and not latest.is_closed:
                self._response = latest
        return self._response

    def close(self) -> None:
        """Close the SDK generator, freezing the response snapshot."""
        if self._response is None:
            self._response = self._http_client.last_response
        self._closed = True
        self._chunks.close()


def _is_thought(part: Any) -> bool:
    """Return whether *part* carries summarized reasoning rather than answer text.

    Args:
        part: A Gemini response part.

    Returns:
        ``True`` when the part is flagged ``thought``.
    """
    return getattr(part, "thought", None) is True


def _reason_name(reason: Any) -> str:
    """Return the wire name of a ``FinishReason`` or ``BlockedReason`` (``"SAFETY"``).

    Args:
        reason: The SDK's enum member (``str(member)`` is ``FinishReason.SAFETY``).
    """
    return str(getattr(reason, "value", reason))


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
    split = split_data_url(url, default_mime)
    if split is None:
        return None
    media_type, payload = split
    data = _decode_base64(payload)
    if data is None:
        return None
    return types.Part.from_bytes(data=data, mime_type=media_type)


def _media_block_to_part(block: dict[str, Any]) -> types.Part | None:
    """Convert a foreign media block/part into a Gemini ``Part``.

    Handles Anthropic ``image`` / ``document`` blocks (base64 or url
    source) and OpenAI ``image_url`` / ``file`` content parts (base64
    data URLs) and ``input_audio`` parts (base64 data with the OpenAI
    ``format`` name, mapped back to its MIME type).  Such blocks enter
    the conversation when it is handed off from another provider's
    model (e.g. via the Sorcar ``set_model`` tool).  Remote (non-data)
    URLs cannot be inlined and are dropped.

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
    elif block_type == "input_audio":
        audio = block.get("input_audio") or {}
        data = _decode_base64(audio.get("data", ""))
        if data is not None:
            fmt = audio.get("format", "")
            mime_type = _audio_format_to_mime(fmt)
            return types.Part.from_bytes(data=data, mime_type=mime_type)
    logger.warning("Dropping unconvertible %s block for Gemini.", block_type)
    return None


def _nested_within(value: Any, limit: int) -> bool:
    """Return whether parsed JSON *value* nests its containers at most *limit* deep.

    Walked level by level rather than recursively: the depth is what must
    not reach the interpreter's stack.
    """
    containers = [value] if isinstance(value, (dict, list)) else []
    for _ in range(limit):
        containers = [
            child
            for container in containers
            for child in (container.values() if isinstance(container, dict) else container)
            if isinstance(child, (dict, list))
        ]
    return not containers


def _finite_json(value: Any) -> Any:
    """Return JSON *value* with each NaN and infinity spelled as a string.

    Python's decoder accepts ``NaN``, ``Infinity`` and a number that
    overflows to one, and the SDK writes them back as bare tokens.  Those
    are not JSON: Gemini answers 400 ``Invalid JSON payload`` to that
    request and to every later one of the conversation holding the value.
    """
    try:
        json.dumps(value, allow_nan=False)
    except ValueError:
        return json.loads(json.dumps(value), parse_constant=str)
    return value


def _tool_result_response_dict(content: Any) -> dict[str, Any]:
    """Build a ``FunctionResponse.response`` dict from tool-result content.

    Args:
        content: A tool result payload — a string (possibly JSON) or a
            list of Anthropic nested blocks whose text is extracted.

    Returns:
        A JSON-serializable dict for ``FunctionResponse.response``: a
        string that is JSON nested at most :data:`_MAX_RESULT_NESTING`
        deep parsed (under ``result`` unless it is an object, each NaN
        and infinity as its name), any other content under ``result`` as
        it is.
    """
    if isinstance(content, list):
        content = text_of_parts(content)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if _nested_within(parsed, _MAX_RESULT_NESTING):
                parsed = _finite_json(parsed)
                return parsed if isinstance(parsed, dict) else {"result": parsed}
        except (ValueError, RecursionError):  # not JSON, or too many digits or levels to decode
            logger.debug("Exception caught", exc_info=True)
    return {"result": content}


def _modality_token_count(
    details: Any, modality: types.MediaModality = types.MediaModality.AUDIO
) -> int:
    """Return the *modality* ``token_count`` in a Gemini per-modality token list.

    Args:
        details: ``usage_metadata.prompt_tokens_details``,
            ``cache_tokens_details`` or ``candidates_tokens_details``: a
            list of ``ModalityTokenCount``, or ``None`` when the response
            omits it.
        modality: The modality to count (AUDIO by default).
    """
    return sum(d.token_count or 0 for d in details or () if d.modality == modality)


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
        # Set once the API refused this model's thinking_config (a model
        # without thinking refuses any): later requests carry none.
        self._thinking_refused = False
        # The pool behind ``client``, kept so a stream abort can reach the
        # live response; built with the client, and closed when the client
        # is replaced, by _build_client.
        self._http_client: _ResponseTrackingHttpxClient
        # The embedding client no call is using, under the one key "idle":
        # ``dict.pop`` takes it and ``dict.setdefault`` puts it back, each
        # atomic, so no two calls share a client and at most one is kept.
        self._idle_embedding_client: dict[str, _EmbeddingClient] = {}

    def _new_client(self) -> tuple[_ResponseTrackingHttpxClient, genai.Client]:
        """Return a new abortable tracking pool and a ``genai.Client`` that sends through it.

        Abortable (:func:`make_pool_abortable`) so that a Stop or a deadline
        also ends a connection still in its TLS handshake: the pool does not
        list it yet, and the connect timeout alone would bound it.  The pool
        is closed when the client cannot be built (no API key): nothing else
        holds it.
        """
        pool = _ResponseTrackingHttpxClient(self._stream_stall_timeout)
        try:
            make_pool_abortable(pool)
            options = types.HttpOptions(httpx_client=pool)
            return pool, genai.Client(api_key=self.api_key, http_options=options)
        except BaseException:
            pool.close()
            raise

    def _build_client(self) -> genai.Client:
        """Return a new ``genai.Client`` over a fresh tracking pool; close the previous pool.

        Each client sends through a pool of its own, and
        ``genai.Client.close()`` leaves an httpx client it was handed open
        (the SDK leaves it to the caller), so :meth:`Model._reuse_client`
        closing the replaced client does not release that client's pool.
        It is closed here instead, once the replacement exists, rather than
        left with its keep-alive sockets to the garbage collector.
        """
        http_client, client = self._new_client()
        if self.client is not None:
            # The client being replaced was built here too: this is its pool.
            self._http_client.close()
        self._http_client = http_client
        return client

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
        # Built here rather than in __init__ because the Sorcar agent may
        # override ``api_key`` between construction and initialize() (a
        # task-specific key on a same-provider model switch); see
        # Model._reuse_client.
        self._reuse_client((self.api_key,), self._build_client)
        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if attachments:
            msg["attachments"] = attachments
        self.conversation = [msg]
        self._thought_signatures = {}

    def _chat_messages(self) -> list[dict[str, Any]]:
        """Return the conversation converted to Chat-Completions messages.

        The one place :func:`responses_items_to_chat_messages` runs for a
        request: :meth:`generate_and_process_with_tools` converts once and
        passes the result to :meth:`_convert_conversation_to_gemini_contents`
        (which feeds :meth:`_tool_call_id_to_name_map`) and
        :meth:`_build_config` instead of each consumer re-running the
        full conversion.

        Returns:
            The conversation in Chat-Completions format.
        """
        return responses_items_to_chat_messages(self.conversation)

    def _tool_call_id_to_name_map(self, chat_messages: list[dict[str, Any]]) -> dict[str, str]:
        """Map tool-call ids to function names across the whole conversation.

        Scans assistant messages for both OpenAI-style ``tool_calls``
        entries and Anthropic-style ``tool_use`` content blocks (present
        when the conversation was handed off from another provider's
        model, e.g. via the Sorcar ``set_model`` tool).

        Args:
            chat_messages: The conversation, pre-converted by
                :meth:`_chat_messages`.

        Returns:
            dict[str, str]: Mapping of tool-call id to function name.
        """
        mapping: dict[str, str] = {}
        for msg in chat_messages:
            if msg.get("role") == "assistant":
                for name, call_id in iter_assistant_tool_calls(msg):
                    if call_id:
                        mapping[call_id] = name or "unknown"
        return mapping

    def _function_call_part(self, name: str, args: dict[str, Any], call_id: Any) -> types.Part:
        """Build a Gemini ``function_call`` part, re-attaching its thought signature.

        A call without a stored signature (produced by another provider's
        model, or an unsigned later call of a parallel batch) carries
        :data:`FOREIGN_CALL_SIGNATURE`, or Gemini 3 and 4 would reject it.

        Args:
            name: The function name.
            args: The function arguments dict.
            call_id: The tool-call id used to look up a stored thought signature.

        Returns:
            types.Part: The function-call part.
        """
        return types.Part(
            function_call=types.FunctionCall(name=name, args=_finite_json(args)),
            thought_signature=self._thought_signatures.get(call_id) or FOREIGN_CALL_SIGNATURE,
        )

    def _function_response_part(self, name: str, content: Any, call_id: Any) -> types.Part:
        """Build a Gemini ``function_response`` part, re-attaching any thought signature.

        Args:
            name: The function name the result belongs to.
            content: The tool result payload (string or nested block list).
            call_id: The tool-call id used to look up a stored thought signature.

        Returns:
            types.Part: The function-response part.
        """
        return types.Part(
            function_response=types.FunctionResponse(
                name=name, response=_tool_result_response_dict(content)
            ),
            thought_signature=self._thought_signatures.get(call_id) if call_id else None,
        )

    def _block_list_to_parts(
        self, blocks: list[Any], id_to_name: dict[str, str]
    ) -> list[types.Part]:
        """Convert a foreign content-block list into Gemini parts.

        Handles Anthropic Messages blocks (``text`` / ``tool_use`` /
        ``tool_result``, with the ``image`` / ``document`` blocks nested in
        it / ``image`` / ``document``; ``thinking`` blocks are hidden
        provider state and are dropped) and OpenAI Chat Completions
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
                        parse_tool_arguments(block.get("input")),
                        block.get("id"),
                    )
                )
            elif block_type == "tool_result":
                call_id = block.get("tool_use_id")
                content = block.get("content")
                parts.append(
                    self._function_response_part(
                        id_to_name.get(call_id or "", "unknown"), content, call_id
                    )
                )
                # The bytes a tool returned, which Anthropic nests in the
                # result itself: the response dict above carries its text
                # only, so they become media parts of the same turn.
                for nested in content if isinstance(content, list) else []:
                    if isinstance(nested, dict) and nested.get("type") in ("image", "document"):
                        part = _media_block_to_part(nested)
                        if part is not None:
                            parts.append(part)
            elif block_type in ("image", "document", "image_url", "file", "input_audio"):
                part = _media_block_to_part(block)
                if part is not None:
                    parts.append(part)
            else:
                logger.warning("Dropping unsupported %s block for Gemini.", block_type)
        return parts

    def _convert_conversation_to_gemini_contents(
        self, chat_messages: list[dict[str, Any]]
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
            chat_messages: The conversation, pre-converted by
                :meth:`_chat_messages`.

        Returns:
            list[types.Content]: The conversation in Gemini API format.
        """
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
                            parse_tool_arguments(fn.get("arguments")),
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

    def _build_config(
        self,
        chat_messages: list[dict[str, Any]],
        tools: list[types.Tool] | None = None,
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
        ``thinking_config`` and ``reasoning_effort`` are merged by
        :meth:`_thinking_config`, and left out for a model that refused
        one.  OpenAI-style ``role="system"``
        messages (present when the conversation was handed off from an
        OpenAI-schema model, e.g. via the Sorcar ``set_model`` tool) are
        hoisted into Gemini's ``system_instruction``, since Gemini
        contents accept only ``user`` / ``model`` roles
        (:func:`~kiss.core.models.model.merge_system_texts`).

        Args:
            chat_messages: The conversation, pre-converted by
                :meth:`_chat_messages`, for the system instruction.
            tools: Optional Gemini tool declarations for this request.

        Returns:
            The config to pass to ``generate_content``.
        """
        params = {
            key: value
            for key, value in self.model_config.items()
            if key not in FRAMEWORK_ONLY_CONFIG_KEYS
        }
        max_output_tokens, stop_sequences = pop_max_tokens_and_stop(params)
        if max_output_tokens is not None:
            params.setdefault("max_output_tokens", max_output_tokens)
        if stop_sequences is not None:
            params.setdefault("stop_sequences", stop_sequences)
        thinking_config = self._thinking_config(
            params.get("thinking_config"), params.pop("reasoning_effort", None)
        )
        params["thinking_config"] = None if self._thinking_refused else thinking_config

        params = self._keep_supported_request_params(params, _GEMINI_CONFIG_FIELDS, "Gemini")
        params["tools"] = tools
        params["system_instruction"] = merge_system_texts(
            self.system_instruction_text(), chat_messages
        )
        return types.GenerateContentConfig(**params)

    def _thinking_config(
        self, configured: types.ThinkingConfig | dict[str, Any] | None, effort: str | None
    ) -> types.ThinkingConfig:
        """Return the request's thinking config, with thought summaries on.

        ``include_thoughts`` defaults to ``True`` (an explicit ``False`` is
        kept) so the UI shows the model's thinking even when the caller
        configures only a level or a budget.  ``thinking_level`` comes
        from ``reasoning_effort`` when given (it replaces any configured
        level or budget, so KISSAgent can lower the effort of a retry;
        ``xhigh`` and ``max``, which Gemini lacks, become ``high``);
        otherwise from the catalog ``thinking`` level of this model
        (e.g. ``high`` for ``gemini-4-argon-high``) when the caller set
        neither a level nor a budget, since Gemini rejects both at once.

        Args:
            configured: The ``thinking_config`` from ``model_config``.
            effort: The ``reasoning_effort`` from ``model_config``.

        Returns:
            The thinking config to send.
        """
        fields = types.ThinkingConfig.model_validate(configured or {}).model_dump(
            exclude_none=True
        )
        fields.setdefault("include_thoughts", True)
        if effort:
            fields.pop("thinking_budget", None)
            fields["thinking_level"] = _GEMINI_TOP_LEVEL.get(effort, effort)
        elif "thinking_level" not in fields and "thinking_budget" not in fields:
            info = MODEL_INFO.get(self.model_name)
            if info is not None and info.thinking:
                fields["thinking_level"] = info.thinking
        return types.ThinkingConfig(**fields)

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

    def _stream_turn(
        self, contents: list[types.Content], config: Any
    ) -> tuple[list[Any], Any, str | None]:
        """Stream one turn, aborting on a user stop or a stalled connection.

        Args:
            contents: The conversation in Gemini API format.
            config: The generation config for this turn.

        Returns:
            The streamed parts, the chunk to bill the turn against —
            the one carrying ``usage_metadata`` rather than whichever
            chunk happened to come last, since a terminal
            ``finishReason``-only chunk would otherwise report the step
            as free; ``None`` when the stream produced nothing at all —
            and the candidate's ``finishReason`` (``"STOP"``,
            ``"MAX_TOKENS"``), ``None`` for an empty stream.

        The read ends at the chunk that ends the turn (a blocked prompt,
        or usage sent with or after the finish reason), not at the end
        of the body: a gateway may hold a finished response open, and
        the stall watchdog would then cut a complete answer or a refusal
        as a stall, to be asked and billed again.

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
            ContentFilterError: When the prompt was blocked
                (``promptFeedback.blockReason``) or the candidate ended
                on a :data:`_FILTER_FINISH_REASONS` stop.
            KISSError: When the stream delivered chunks but ended before
                the one carrying the ``finishReason``: a gateway can
                close a body cleanly before the model is done, and on
                this API no finish reason means the model has not
                stopped generating, so the text and the tool calls may
                be incomplete.  The same plain ``KISSError`` the other
                transports raise for a stream without its terminal
                event.  A stream that delivered nothing at all is an
                empty turn instead (no second request, see
                ``test_empty_stream_is_an_empty_turn``).
        """
        parts: list[Any] = []
        usage_chunk = None
        last_chunk = None
        finish: types.FinishReason | None = None
        finish_reason: str | None = None
        block_reason: str | None = None
        # The wait for the headers is bounded as a whole (connection,
        # request, the SDK's retries): the byte-level clock cannot see
        # a peer that trickles them.  A cut after the budget is a stall.
        headers_due = time.monotonic() + CONNECT_TIMEOUT + 2 * self._stream_stall_timeout
        stream = _AbortableStream(
            self.client.models.generate_content_stream(
                model=_strip_thinking_alias(self.model_name), contents=contents, config=config
            ),
            self._http_client,
            headers_due - time.monotonic(),
        )
        try:
            with self._watched_events(stream, "gemini-stream-abort-watchdog") as events:
                for chunk in events:
                    last_chunk = chunk
                    if chunk.usage_metadata is not None:
                        # Billed already: kept for take_partial_usage_response
                        # if the stream fails before it ends.
                        usage_chunk = self._rejected_response = chunk
                    if chunk.prompt_feedback is not None and chunk.prompt_feedback.block_reason:
                        block_reason = _reason_name(chunk.prompt_feedback.block_reason)
                    if chunk.candidates and chunk.candidates[0].finish_reason:
                        finish = chunk.candidates[0].finish_reason
                        finish_reason = _reason_name(finish)
                    chunk_parts = self._parts_from_response(chunk)
                    self._stream_parts(chunk_parts)
                    parts.extend(chunk_parts)
                    if block_reason is not None or (
                        finish_reason is not None and chunk.usage_metadata is not None
                    ):
                        # The turn is over: a blocked prompt generates
                        # nothing, and usage sent with or after the finish
                        # reason bills the turn.  Usage from before it is a
                        # running count, so a bare finish reason reads on.
                        # Leaving the loop closes the stream.
                        break
        except httpx.TimeoutException as e:
            # The byte-level clock fired first; report it in the same
            # words as the event-level watchdog would have.
            raise stall_error(self._stream_stall_timeout) from e
        except Exception as e:
            if stream.response is None and (
                stop_signal.stop_requested() or time.monotonic() >= headers_due
            ):
                # The header watchdog cut a request whose headers never
                # completed: the user's Stop, or a stall reported like a
                # silent one.
                raise stop_or_stall_error(self._stream_stall_timeout) from e
            raise
        # A refused end leaves the usage seen so far in ``_rejected_response``.
        if block_reason is not None:
            raise ContentFilterError(
                f"Gemini blocked the prompt (block_reason={block_reason}); the same "
                "request will be blocked again.",
                stop_reason=block_reason,
            )
        if finish_reason is None:
            if last_chunk is not None:
                raise KISSError(
                    "Gemini stream ended without a finish reason, so its text and any "
                    "tool calls may be incomplete."
                )
        elif finish_reason in _FILTER_FINISH_REASONS:
            raise ContentFilterError(
                f"Gemini's content filter stopped the response (finish_reason={finish_reason}).",
                stop_reason=finish_reason,
            )
        self._rejected_response = None
        response = usage_chunk or last_chunk
        if response is not None and finish is not None:
            # The billed chunk may have come before the finishReason-only
            # one: it carries the reason too, for ``stop_reason_of``.
            candidates = response.candidates or [types.Candidate()]
            candidates[0].finish_reason = finish
            response.candidates = candidates
        return parts, response, finish_reason

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
        config = self._build_config(chat_messages, tools=gemini_tools)
        try:
            all_parts, response, finish_reason = self._stream_turn(contents, config)
        except errors.ClientError as e:
            if "Thinking is not enabled" not in str(e):
                raise
            # A model without thinking (gemini-2.5-flash-image) refuses any
            # thinking_config, the default include_thoughts included: ask
            # again without one, as every later request of this model does.
            self._thinking_refused = True
            config = self._build_config(chat_messages, tools=gemini_tools)
            all_parts, response, finish_reason = self._stream_turn(contents, config)
        content, function_calls = self._parse_parts(all_parts)
        if finish_reason == "MAX_TOKENS" and gemini_tools and not function_calls:
            # A tool-bearing turn with no completed call is unusable (a
            # tool-less caller may still read the cut-off text; a call
            # arrives whole, so one completed before the cap is kept).
            # The cut-off request was still billed.
            self._rejected_response = response
            raise OutputCapError(
                f"Model {self.model_name} hit its output-token cap "
                f'(finish_reason="MAX_TOKENS") before completing a tool call, so '
                f"its text is incomplete. Raise max_tokens or lower reasoning_effort.",
                stop_reason="MAX_TOKENS",
            )

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
    ) -> tuple[int, int, int, int] | tuple[int, int, int, int, int, int, int, int, int]:
        """Extracts token counts from a Gemini API response.

        Gemini prices audio input above text on several models (e.g.
        gemini-2.5-flash: $1.00 audio vs $0.30 text per 1M, cached
        $0.10 vs $0.03), so the AUDIO shares of the prompt
        (``prompt_tokens_details``) and of its cached part
        (``cache_tokens_details``) are split out for
        :func:`~kiss.core.models.model_info.calculate_cost` to bill at
        the model's audio rates.  Image models price generated images
        far above text output (gemini-3.1-flash-image: $60 vs $3 per
        1M), so the IMAGE share of ``candidates_tokens_details`` is
        split out too.

        Returns:
            ``(input_tokens, output_tokens, cache_read_tokens,
            cache_write_tokens)`` with text counts only, or, when the
            prompt holds audio or the output holds images, the 9-tuple
            that adds ``cache_write_1h_tokens`` (0),
            ``audio_input_tokens`` (uncached), ``audio_output_tokens``
            (0), ``audio_cache_read_tokens`` and
            ``image_output_tokens``.
        """
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            candidates_tokens = um.candidates_token_count or 0
            image_output = min(
                _modality_token_count(
                    getattr(um, "candidates_tokens_details", None),
                    types.MediaModality.IMAGE,
                ),
                candidates_tokens,
            )
            thoughts_tokens = getattr(um, "thoughts_token_count", 0) or 0
            output_tokens = candidates_tokens - image_output + thoughts_tokens
            prompt_tokens = um.prompt_token_count or 0
            cached_tokens = getattr(um, "cached_content_token_count", 0) or 0
            tool_use_tokens = getattr(um, "tool_use_prompt_token_count", 0) or 0
            cached_audio = min(
                _modality_token_count(getattr(um, "cache_tokens_details", None)),
                cached_tokens,
            )
            audio_tokens = max(
                _modality_token_count(getattr(um, "prompt_tokens_details", None)) - cached_audio,
                0,
            )
            input_tokens = max(prompt_tokens - cached_tokens - audio_tokens, 0) + tool_use_tokens
            text_cached = cached_tokens - cached_audio
            if audio_tokens or cached_audio or image_output:
                return (
                    input_tokens,
                    output_tokens,
                    text_cached,
                    0,
                    0,
                    audio_tokens,
                    0,
                    cached_audio,
                    image_output,
                )
            return input_tokens, output_tokens, cached_tokens, 0
        return 0, 0, 0, 0

    def get_embedding(
        self,
        text: str,
        embedding_model: str | None = None,
    ) -> list[float]:
        """Generates an embedding vector for the given text.

        The request ends at the calling thread's Stop and at
        :data:`_EMBEDDINGS_TIMEOUT`.  The abort shuts down every socket of
        the pool it watches, and one instance embeds for several threads
        at once (the memory tools), so a call sends through a client no
        other call is using: the instance's idle one, with the connection
        earlier calls left open, or a new one when none is idle.  The
        client is idle again afterwards, unless its request was cut or
        never sent, or another client took its place meanwhile.

        A thread of its own takes the client, sends through it and puts it
        away (:meth:`_run_embedding`), and this call waits for it.  The
        calling thread cannot own the client: a Stop is injected there at
        any check point, the cleanup's own included, and would leave the
        client neither idle nor closed.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name. Defaults to this instance's
                own model name, so a non-embedding model (a chat, TTS or
                transcription model) fails instead of silently embedding
                through ``gemini-embedding-001``.

        Returns:
            list[float]: The embedding vector as a list of floats.

        Raises:
            KISSError: If embedding generation fails.
            KeyboardInterrupt: The user pressed Stop: before the request,
                which is then not sent, or before its reply was back.
        """
        model_to_use = embedding_model or self.model_name
        if stop_signal.stop_requested():  # nothing is started: the idle client stays idle
            raise stop_error()
        deadline = float(self.model_config.get("timeout", _EMBEDDINGS_TIMEOUT))
        outcome: list[list[float] | BaseException] = []
        worker = threading.Thread(
            target=self._run_embedding,
            args=(outcome, stop_signal.get_thread_stop_event(), deadline, model_to_use, text),
            name="gemini-embedding",
            daemon=True,
        )
        try:
            worker.start()
            worker.join()
            result = outcome[0]
            if isinstance(result, BaseException):
                raise result
        except Exception as e:
            if stop_signal.stop_requested():  # the watchdog cut the request
                raise stop_error() from e
            logger.debug("Exception caught", exc_info=True)
            raise KISSError(f"Embedding generation failed for model {model_to_use}: {e}") from e
        if stop_signal.stop_requested():  # set too late for the watchdog to cut the reply
            raise stop_error()
        return result

    def _run_embedding(
        self,
        outcome: list[list[float] | BaseException],
        stop_event: threading.Event | None,
        deadline: float,
        model_to_use: str,
        text: str,
    ) -> None:
        """Embed *text* for :meth:`get_embedding`, on a thread that receives no injected Stop.

        Args:
            outcome: Receives the vector, or the exception that ended the request.
            stop_event: The calling thread's stop event: it ends this request too.
            deadline: Seconds the whole request may take.
            model_to_use: The embedding model.
            text: The text to embed.
        """
        stop_signal.set_thread_stop_event(stop_event)
        leased = self._idle_embedding_client.pop("idle", None)
        due = 0.0  # no request yet
        try:
            if leased is not None and leased.api_key != self.api_key:
                leased.close()  # built with a key since replaced
                leased = None
            if leased is None:  # nothing idle: the first call, one beside another, or after a cut
                pool, sdk = self._new_client()
                leased = _EmbeddingClient(self.api_key, pool, sdk)
            if stop_signal.stop_requested():  # pressed while the client was got ready: not sent
                raise stop_error()
            due = time.monotonic() + deadline
            with RequestAbortWatchdog(leased.pool, deadline, "gemini-embeddings-abort-watchdog"):
                response: Any = leased.sdk.models.embed_content(model=model_to_use, contents=text)
            outcome.append(list(response.embeddings[0].values))
        except BaseException as e:  # raised by get_embedding, in the calling thread
            outcome.append(e)
        finally:
            if leased is not None:
                # Kept unless the watchdog may have shut its sockets down, or no request started.
                cut = stop_signal.stop_requested() or time.monotonic() >= due
                if cut or self._idle_embedding_client.setdefault("idle", leased) is not leased:
                    leased.close()
