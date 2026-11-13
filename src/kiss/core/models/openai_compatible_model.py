# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""OpenAI-compatible model implementation for custom endpoints."""

import json
import logging
import re
import sys
import threading
import time
from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import urlparse

import httpx

# The OpenAI SDK takes ``sys.modules["httpx2"]`` as ready whenever it is
# there (``openai._httpx2``, on every client construction and request).
# Its first import running on another thread (the Anthropic SDK's, on the
# daemon's prewarm thread) failed an OpenAI task with "partially
# initialized module 'httpx2'"; imported with the SDK, it is whole before
# any client exists.
import httpx2  # noqa: F401
from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletionMessage

if TYPE_CHECKING:  # pragma: no cover – import cycle avoided at runtime
    from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2

from kiss.core import stop_signal
from kiss.core.kiss_error import ContentFilterError, KISSError, OutputCapError
from kiss.core.locked import locked
from kiss.core.models.model import (
    FRAMEWORK_ONLY_CONFIG_KEYS,
    TOOL_RESULT_ATTACHMENT_NOTE,
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    _audio_mime_to_format,
    _build_text_based_tools_prompt,
    _chat_tool_call_entry,
    _get_attr_or_key,
    _parse_text_based_tool_calls,
    accepted_request_params,
    checked_nonempty,
    normalize_content_blocks,
    parse_tool_arguments,
    responses_items_to_chat_messages,
    text_of_parts,
)
from kiss.core.models.stream_abort import (
    CONNECT_TIMEOUT,
    STALL_MARKER,
    make_pool_abortable,
    shutdown_pool_sockets,
    stop_error,
    stop_or_stall_error,
)

logger = logging.getLogger(__name__)

_StreamResult = TypeVar("_StreamResult")

# The request parameters a Chat Completions call accepts, taken from the
# SDK's own signature (keyword-only, no **kwargs).
_CHAT_REQUEST_PARAMS = accepted_request_params(Completions.create)

# Silent SDK-level retries on transport failures.  The SDK default (2)
# re-sends the identical request without any log or callback; when the
# provider had already received and processed the request but the
# response was lost in transit (``APIConnectionError: Connection
# error.``), every silent retry generates — and bills — the same answer
# again.  Combined with the agent loop's own visible retries
# (``MAX_CONSECUTIVE_ERRORS`` = 3), the default turned one logical turn
# into up to 9 identical upstream request/response pairs (observed as
# "repeated request and response" with ``openrouter/moonshotai`` models).
# One retry keeps resilience against one-off connect failures while
# bounding the duplication, matching ``anthropic_model._MAX_RETRIES``.
_MAX_RETRIES = 1
# The SDK's delay before that retry (its ``INITIAL_RETRY_DELAY``; a
# ``retry-after`` header may stretch it, and the whole-request deadline
# then cuts the retry, which the agentic loop makes up for).
_RETRY_BACKOFF_SECONDS = 0.5
# The client's read timeout: a non-streamed reasoning turn legitimately
# sends nothing for minutes.  ``model_config["timeout"]`` overrides it
# per request.
_CLIENT_TIMEOUT = 1800.0

# How often ``_await_call`` looks at the stop event while its worker
# thread waits on the provider; the stream watchdog's stop poll rate.
_BLOCKING_STOP_POLL_SECONDS = 0.1

# What a caller that gave a stream's opening call up stores as its
# response, so that the worker closes a stream that opens after all.
_ABANDONED = object()

# Chat Completions ``finish_reason`` values meaning the provider cut the
# completion short, with the cause shown to the user.  ``error`` is how a
# gateway (OpenRouter's normalized value) reports a provider that failed
# mid-generation; the Responses transport refuses the matching
# ``status="incomplete"`` / ``"failed"`` responses.
_CUT_SHORT_FINISH_REASONS = {
    "length": "the model hit its output-token budget",
    "content_filter": "the provider's content filter stopped the output",
    "error": "the provider failed mid-generation",
}


def _terminal_error(reason: str, output_cap_reason: str, message: str, text: str = "") -> KISSError:
    """Return the error for a completion the provider ended short of success.

    The one classification both transports share; each keeps its own
    message and guards (Chat Completions types every cut-short
    ``finish_reason``, the Responses transport only an ``incomplete``
    status).

    Args:
        reason: The provider's stop reason (Chat Completions
            ``finish_reason``, Responses ``incomplete_details.reason``),
            ``""`` when none applies.
        output_cap_reason: The *reason* that names the output-token
            budget on this transport (``"length"`` /
            ``"max_output_tokens"``).
        message: The error message.
        text: The text the completion held when it ended; the typed
            errors carry it.

    Returns:
        :class:`OutputCapError` (carrying *reason* as ``stop_reason``)
        when the output budget was hit, :class:`ContentFilterError` for
        ``content_filter`` -- both make the agentic loop fail over to
        another model family instead of retrying the same one -- else a
        plain :class:`KISSError`.
    """
    if reason == output_cap_reason:
        return OutputCapError(message, stop_reason=reason, text=text)
    if reason == "content_filter":
        return ContentFilterError(message, text=text)
    return KISSError(message)


def _well_formed(value: Any) -> Any:
    """Return a copy of request data whose text is well-formed Unicode.

    ``json.loads`` turns a ``"\\ud83d"`` escape (a page's truncated emoji,
    a model's own argument echoed back by its tool) into a lone
    surrogate, which no UTF-8 request body can carry.  A pair split into
    two code points is joined; a half without its partner becomes U+FFFD.

    Args:
        value: A request argument: text, or dicts and lists holding some.

    Returns:
        The repaired copy; a value of any other type is returned as is.
    """
    if isinstance(value, str):
        return value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    if isinstance(value, dict):
        return {_well_formed(key): _well_formed(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_well_formed(item) for item in value]
    return value


def _call_into(outcome: dict[str, Any], create: Callable[..., Any], kwargs: dict[str, Any]) -> None:
    """Worker-thread body of ``_await_call``: store ``create``'s result or error in *outcome*.

    A result whose caller stored :data:`_ABANDONED` meanwhile is a stream
    nobody will read: it is closed here.

    Args:
        outcome: Receives ``"response"`` on success, else ``"error"``.
        create: The SDK call to run.
        kwargs: Its keyword arguments.
    """
    try:
        try:
            response = create(**kwargs)
        except UnicodeEncodeError:
            # The SDK could not encode the request body (nothing was
            # sent): text of the conversation holds a lone surrogate, and
            # would fail every later request the same way.
            response = create(**_well_formed(kwargs))
        if outcome.setdefault("response", response) is _ABANDONED:
            response.close()
    except BaseException as error:  # noqa: BLE001 - re-raised on the requesting thread
        outcome["error"] = error


def _await_call(
    outcome: dict[str, Any],
    create: Callable[..., Any],
    kwargs: dict[str, Any],
    deadline: float,
    abort: Callable[[], object],
    name: str,
) -> None:
    """Run ``create(**kwargs)`` on a worker thread until it answers, a Stop or *deadline*.

    A thread blocked in the SDK call cannot notice a Stop, and shutting
    the request's socket down wakes only a call waiting on it, not one
    asleep before the SDK's own retry (a ``Retry-After`` of up to a
    minute).  So the call runs on a worker thread while this one watches
    the stop event (at the watchdog's poll rate) and the deadline.  A call
    given up is torn down with *abort*, which wakes a worker parked on
    the socket; one the abort cannot reach is abandoned (a daemon
    thread) until the SDK returns.

    Args:
        outcome: Receives ``"response"`` or ``"error"`` (:func:`_call_into`);
            still empty on return when the call was given up.
        create: The SDK call to run.
        kwargs: Its keyword arguments.
        deadline: The ``time.monotonic()`` instant to give the call up at.
        abort: Tears down the client of a call given up.
        name: The worker thread's name.
    """
    stop_event = stop_signal.get_thread_stop_event() or threading.Event()
    worker = threading.Thread(
        target=_call_into, args=(outcome, create, kwargs), name=name, daemon=True
    )
    try:
        worker.start()
        while worker.is_alive():
            if stop_event.is_set():
                # A reply landing this very moment is billed too: give
                # the worker one poll interval to store it.
                worker.join(_BLOCKING_STOP_POLL_SECONDS)
                break
            if time.monotonic() >= deadline:
                break
            worker.join(_BLOCKING_STOP_POLL_SECONDS)  # returns with the answer at once
    finally:
        # Also on a KeyboardInterrupt raised into this thread (the
        # runner's forced stop), even one landing inside start()
        # before the worker reports itself alive (is_alive() is
        # still False then, yet the worker runs): a worker that has
        # not answered is torn down, failing at once on the closed
        # client if it had not sent the request yet.
        if not outcome:
            abort()
            # The shutdown wakes the worker; one that already had the
            # whole reply in hand stores it now, so wait a poll
            # interval before the caller looks at the outcome: a reply
            # that landed between the check and the abort was billed and
            # must not be dropped.  A worker that has not started
            # yet cannot be joined and has nothing to store.
            if worker.is_alive():
                worker.join(_BLOCKING_STOP_POLL_SECONDS)


# Per-request timeout for embeddings calls.  The client's own timeout is
# the 1800 s a long generation may legitimately need, but an embeddings
# request answers in seconds, so a provider that accepts the connection
# and never responds must not park the calling task thread for half an
# hour.  ``model_config["timeout"]`` (the same key the chat and Responses
# requests already forward to the SDK) overrides it.
EMBEDDINGS_TIMEOUT = 60.0

# OpenAI processing tiers as a factor on the Standard rate the catalog
# holds (https://developers.openai.com/api/docs/pricing, tier switcher).
# Flex bills Batch rates (half of Standard) and Ultrafast six times
# (gpt-6-astra $60/$6/$75/$300 on $10/$1/$12.50/$50) for every model
# offering the tier; ``default``/``auto``/``scale`` bill Standard.  Fast
# (formerly Priority) is twice Standard on the GPT-5.x/GPT-6 line
# (gpt-6-astra $20/$2/$25/$100, gpt-5.4 $5/$0.50/$30, gpt-5.2
# $3.50/$0.35/$28, gpt-5 $2.50/$0.25/$20) but not uniformly so on older
# or smaller models; the exceptions below are the published Fast rate
# over the Standard rate, longest prefix first.
_OPENAI_FLEX_MULTIPLIER = 0.5
_OPENAI_ULTRAFAST_MULTIPLIER = 6.0
_OPENAI_FAST_MULTIPLIER = 2.0
_OPENAI_FAST_EXCEPTIONS = (
    ("gpt-5.5", 2.5),  # $12.50/$1.25/$75 on $5/$0.50/$30
    ("gpt-5-mini", 1.8),  # $0.45/$0.045/$3.60 on $0.25/$0.025/$2
    ("gpt-4.1-nano", 2.0),  # $0.20/$0.05/$0.80 on $0.10/$0.025/$0.40
    ("gpt-4.1", 1.75),  # $3.50/$0.875/$14 on $2/$0.50/$8; -mini $0.70/$0.175/$2.80
    ("gpt-4o-mini", 0.25 / 0.15),  # $0.25/$0.125/$1 on $0.15/$0.075/$0.60
    ("gpt-4o", 1.7),  # $4.25/$2.125/$17 on $2.50/$1.25/$10
    ("o4-mini", 2.0 / 1.1),  # $2/$0.50/$8 on $1.10/$0.275/$4.40
    ("o3", 1.75),  # $3.50/$0.875/$14 on $2/$0.50/$8
)


def _openai_fast_multiplier(model_name: str) -> float:
    """Return the Fast-tier factor over Standard for *model_name*.

    Args:
        model_name: The catalog model name (thinking aliases and
            provider prefixes are ignored).

    Returns:
        The published Fast/Standard ratio from
        :data:`_OPENAI_FAST_EXCEPTIONS`, else :data:`_OPENAI_FAST_MULTIPLIER`.
    """
    from kiss.core.models.model_info import (
        _openai_bare_name,
        _strip_provider_prefix,
        _strip_thinking_alias,
    )

    # ``openrouter/openai/gpt-5.5`` and ``gpt-5.5-high`` both bill as gpt-5.5.
    bare = _openai_bare_name(model_name) or _strip_thinking_alias(
        _strip_provider_prefix(model_name)
    )
    for prefix, factor in _OPENAI_FAST_EXCEPTIONS:
        if bare.startswith(prefix):
            return factor
    return _OPENAI_FAST_MULTIPLIER


def _usd_amount(value: Any) -> float | None:
    """Return *value* as a USD amount, or ``None`` when it is not one.

    A reported cost is added to the run's spend, where ``NaN`` makes
    every later budget comparison false and a negative figure refunds
    the run; an integer beyond the float range does not convert at all.

    Args:
        value: A cost field of a response's ``usage``.

    Returns:
        The amount when *value* is a finite, non-negative number (a
        boolean is not), else ``None``.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if 0 <= value <= sys.float_info.max else None


def _provider_model_name(model_name: str) -> str:
    """Return the upstream provider id for a KISS catalog ``model_name``.

    Two transformations are applied in order:

    * A trailing thinking-level alias suffix (``-xhigh``, or a
      marker-verified ``-high`` / ``-medium`` / ``-low``) is stripped so
      the synthetic alias maps back to its base model id.  The alias is
      resolved against the full catalog key BEFORE prefix removal so only
      an exact ``alias_of``-marked entry can rewrite the name (an
      unrelated catalog alias can never rewrite a similarly-named custom
      model).  ``MODEL_INFO`` carries the sibling entries purely so
      callers can select a ``reasoning_effort`` level by model name; the
      provider's HTTP endpoint only knows the base name.
    * An ``openrouter/`` routing prefix is removed (callers reach
      OpenRouter via the catalog key ``openrouter/<provider>/<id>`` but
      the OpenRouter API itself wants the bare ``<provider>/<id>``).

    Args:
        model_name: The catalog model name as passed in.

    Returns:
        The string to send as ``model=`` over the wire.
    """
    from kiss.core.models.model_info import _strip_thinking_alias

    base_name = _strip_thinking_alias(model_name)
    if base_name.startswith("openrouter/"):
        return base_name[len("openrouter/") :]
    return base_name


def _model_thinking_level(model_name: str) -> str | None:
    """Return the default ``reasoning_effort`` level for *model_name*, if any.

    The level lives on ``MODEL_INFO[model_name].thinking`` and is set
    per-model via the ``thinking`` key in ``MODEL_INFO.json`` /
    ``~/.kiss/MY_MODELS.json`` (e.g. ``thinking="xhigh"`` for the gpt-5.5
    family).  A ``MY_MODELS.json`` entry registered after the catalog was
    loaded is read from the file, as ``model_info._lookup_model_info``
    does, so a custom model behaves the same before and after a daemon
    restart.  Models in neither (e.g. custom endpoints with arbitrary
    model names) return ``None`` so we never send an unsupported
    ``reasoning_effort`` to such providers.

    Args:
        model_name: The full model name as passed to
            :class:`OpenAICompatibleModel`, including any ``openrouter/``
            prefix.

    Returns:
        The thinking level string (e.g. ``"xhigh"``) if the matching
        ``MODEL_INFO`` or ``MY_MODELS.json`` entry sets one, otherwise
        ``None``.
    """
    from kiss.core.models.model_info import MODEL_INFO, _read_my_models

    info = MODEL_INFO.get(model_name)
    if info is not None:
        return info.thinking
    thinking = (_read_my_models().get(model_name) or {}).get("thinking")
    return thinking if isinstance(thinking, str) else None


DEEPSEEK_REASONING_MODELS = {
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1-turbo",
    "deepseek/deepseek-r1-distill-qwen-1.5b",
    "deepseek/deepseek-r1-distill-qwen-7b",
    "deepseek/deepseek-r1-distill-llama-8b",
    "deepseek/deepseek-r1-distill-qwen-14b",
    "deepseek/deepseek-r1-distill-qwen-32b",
    "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-0528-tput",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}

_ADAPTIVE_TOOL_EFFORT_VERDICTS: dict[tuple[str, str], bool] = {}
# Reentrant for ``locked``: both takers run on task threads, where a
# Stop injection landing right after a plain ``with``'s ``__enter__``
# would leak the lock for every later model call of the daemon.
_ADAPTIVE_TOOL_EFFORT_LOCK = threading.RLock()


@locked(_ADAPTIVE_TOOL_EFFORT_LOCK)
def _tool_effort_verdict(key: tuple[str, str]) -> bool | None:
    """Return the learned ``tools`` + ``reasoning_effort`` verdict, if any.

    Args:
        key: The ``(base_url, api_model_name)`` the verdict was learned
            for.  The capability is a property of the *pair*: a
            reasoning model can reject the combination while another
            model on the same host accepts it.

    Returns:
        ``True``/``False`` when the pair has been probed, ``None``
        otherwise.
    """
    return _ADAPTIVE_TOOL_EFFORT_VERDICTS.get(key)


def _rejects_effort_value(error: BadRequestError) -> bool:
    """Return whether a ``reasoning_effort`` 400 refuses the level sent, not the parameter.

    ``Invalid value: 'xhigh'. Supported values are: ...`` (OpenAI's
    ``invalid_value`` code) is a verdict on one level, not on ``tools``
    + ``reasoning_effort``: caching it would silently strip every level
    from the pair's later requests.

    Args:
        error: The rejection whose message mentions ``reasoning_effort``.

    Returns:
        True for a value rejection.
    """
    if error.code in ("invalid_value", "unsupported_value"):
        return True
    return re.search(r"\b(?:invalid|unsupported) value\b", str(error), re.IGNORECASE) is not None


@locked(_ADAPTIVE_TOOL_EFFORT_LOCK)
def _record_tool_effort_verdict(key: tuple[str, str], accepted: bool) -> None:
    """Record what one ``(endpoint, model)`` did with the combination.

    Parallel subagents probe concurrently, so the check and the write
    straddle an HTTP round trip.  A rejection is definitive and always
    wins; a success is only recorded when nothing is known yet, so a
    probe that raced with a rejection can never erase it and the outcome
    no longer depends on which response happened to arrive last.

    Args:
        key: The ``(base_url, api_model_name)`` that was probed.
        accepted: Whether the endpoint accepted the combination.
    """
    if accepted:
        _ADAPTIVE_TOOL_EFFORT_VERDICTS.setdefault(key, True)
    else:
        _ADAPTIVE_TOOL_EFFORT_VERDICTS[key] = False


# The attachment formats OpenAI accepts as input.  Images: PNG, JPEG,
# WEBP and non-animated GIF ("Image input requirements",
# https://developers.openai.com/api/docs/guides/images-vision, which states
# the behaviour "is the same in both the Responses API and the Chat
# Completions API").  Audio: mp3 and wav, the only values the SDK's own
# ``input_audio.format`` literal allows — in the Chat Completions param
# (``chat_completion_content_part_input_audio_param.py``) as well as the
# Responses one (``response_input_audio_param.py``).
#
# Both transports apply this one set: whether a turn goes to Chat
# Completions or the Responses API is a routing decision the caller never
# made, so it must not change which attachments the model gets to see.
OPENAI_INPUT_IMAGE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
)
OPENAI_INPUT_AUDIO_FORMATS = frozenset({"mp3", "wav"})


def _extract_deepseek_reasoning(content: str) -> str:
    """Return a DeepSeek R1 response without its ``<think>...</think>`` blocks.

    The reasoning itself reaches the thinking callback through the
    streamed ``reasoning_content`` deltas, never through this scan.

    Args:
        content: The raw response content from a DeepSeek R1 model.

    Returns:
        The content outside the closed ``<think>`` blocks, stripped;
        *content* itself when it holds no closed block.
    """
    # A find() loop rather than ``<think>(.*?)</think>``: the lazy regex
    # rescans to the end of the text for every unclosed opener, which is
    # quadratic on model output that repeats the tag (and ``re`` holds
    # the GIL while it does so).
    answer_parts: list[str] = []
    cursor = 0
    while True:
        start = content.find("<think>", cursor)
        if start == -1:
            break
        end = content.find("</think>", start + len("<think>"))
        if end == -1:
            break
        answer_parts.append(content[cursor:start])
        cursor = end + len("</think>")
    if not answer_parts:
        return content
    answer_parts.append(content[cursor:])
    return "".join(answer_parts).strip()


def _delta_reasoning_text(delta: Any) -> str | None:
    """Extract reasoning text from a Chat Completions streaming delta.

    Providers expose reasoning on different delta fields:

    * ``reasoning_content`` — DeepSeek-native (and vLLM-style) servers.
    * ``reasoning`` — OpenRouter's normalized plain-string field (also
      newer vLLM versions).
    * ``reasoning_details`` — OpenRouter's structured list of
      ``reasoning.text`` / ``reasoning.summary`` entries.

    The plain string fields are preferred; ``reasoning_details`` is
    consulted ONLY when both string fields are empty because OpenRouter
    sends the same text in ``reasoning`` AND ``reasoning_details``
    simultaneously — reading both would double-emit every thinking token.

    Args:
        delta: A streaming chunk's ``choices[0].delta`` object.

    Returns:
        The reasoning text for this delta, or None when the delta carries
        no reasoning.
    """
    reasoning_content = getattr(delta, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content:
        return reasoning_content
    reasoning = getattr(delta, "reasoning", None)
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    details = getattr(delta, "reasoning_details", None)
    if not isinstance(details, list):
        return None
    texts: list[str] = []
    for entry in details:
        if isinstance(entry, dict):
            entry_type = entry.get("type")
            text = entry.get("text")
            summary = entry.get("summary")
        else:
            entry_type = getattr(entry, "type", None)
            text = getattr(entry, "text", None)
            summary = getattr(entry, "summary", None)
        value = text if entry_type == "reasoning.text" else summary
        if isinstance(value, str) and value:
            texts.append(value)
    return "".join(texts) or None


def _message_text(message: ChatCompletionMessage) -> str:
    """Return the text of a non-streamed chat completion *message*.

    A model that declines answers with ``refusal`` and null ``content``:
    the refusal is then the text of the turn, as the Responses transport
    reads a ``refusal`` part.
    """
    return message.content or message.refusal or ""


def _anthropic_media_block_to_openai_part(block: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an Anthropic ``image``/``document`` block to an OpenAI content part.

    Anthropic media blocks carry a ``source`` dict (``{"type": "base64",
    "media_type": ..., "data": ...}`` or ``{"type": "url", "url": ...}``).
    OpenAI Chat Completions instead uses ``image_url`` / ``file`` parts.
    Such blocks enter the conversation when it is handed off from an
    :class:`AnthropicModel` (e.g. via the Sorcar ``set_model`` tool).

    Args:
        block: The Anthropic media block dict.

    Returns:
        The equivalent OpenAI content-part dict, or ``None`` when the block
        cannot be represented (in which case it is dropped with a warning).
    """
    source = block.get("source") or {}
    url = ""
    if source.get("type") == "base64":
        media_type = source.get("media_type", "application/octet-stream")
        url = f"data:{media_type};base64,{source.get('data', '')}"
    elif source.get("type") == "url":
        url = source.get("url", "")
    if not url:
        logger.warning("Dropping unconvertible Anthropic %s block.", block.get("type"))
        return None
    if block.get("type") == "image":
        return {"type": "image_url", "image_url": {"url": url}}
    return {"type": "file", "file": {"file_data": url}}


def _stringify_tool_call_arguments(tool_calls: list[Any]) -> list[Any]:
    """Ensure every tool call's ``function.arguments`` is a JSON string.

    GeminiModel stores tool-call arguments as dicts; the OpenAI Chat
    Completions API requires a JSON string.  Such entries enter the
    conversation when it is handed off from a :class:`GeminiModel`
    (e.g. via the Sorcar ``set_model`` tool).

    Args:
        tool_calls: The ``tool_calls`` list of an assistant message.

    Returns:
        The list with dict arguments replaced by their JSON encoding.
    """
    result: list[Any] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                tc = {**tc, "function": {**fn, "arguments": json.dumps(args or {})}}
        result.append(tc)
    return result


def _tool_result_block_text(block: dict[str, Any]) -> str:
    """Extract the text payload of an Anthropic ``tool_result`` block.

    The block's ``content`` may be a plain string or a list of nested
    blocks; only the text of nested ``text`` blocks is kept because the
    OpenAI ``role="tool"`` message accepts string content only.

    Args:
        block: The Anthropic ``tool_result`` block dict.

    Returns:
        The concatenated text content of the block.
    """
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return text_of_parts(content)
    return "" if content is None else str(content)


def _tool_result_block_media(block: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert the media nested in an Anthropic ``tool_result`` block.

    Anthropic's result writer keeps the images/PDFs a tool produced as
    ``image`` / ``document`` blocks inside the ``tool_result``; the
    OpenAI ``role="tool"`` message is text-only, so after a hand-off they
    travel in a follow-up user message instead.

    Args:
        block: The Anthropic ``tool_result`` block dict.

    Returns:
        The OpenAI content parts of its nested media blocks, in order.
    """
    content = block.get("content")
    if not isinstance(content, list):
        return []
    parts = [
        _anthropic_media_block_to_openai_part(nested)
        for nested in content
        if isinstance(nested, dict) and nested.get("type") in ("image", "document")
    ]
    return [part for part in parts if part is not None]


def _merge_tools_prompt_into_content(content: Any, tools_prompt: str, part_type: str) -> Any:
    """Append *tools_prompt* to a user message content of either shape.

    The first user message's ``content`` is a plain string in the common
    case, but a list of content parts when the prompt was initialized with
    attachments (images/PDFs/audio).

    Args:
        content: The existing message content (str, list of parts, or
            ``None`` for a message without text; the Responses transport
            merges before its normaliser runs).
        tools_prompt: The text-based tools prompt to append.
        part_type: The text part type of the API in use: ``"text"`` for
            Chat Completions, ``"input_text"`` for Responses.

    Returns:
        The merged content in the same shape as the input.
    """
    if isinstance(content, list):
        return [*content, {"type": part_type, "text": tools_prompt}]
    return ("" if content is None else str(content)) + "\n" + tools_prompt


def _conversation_with_tools_prompt(
    messages: list[Any], tools_prompt: str, part_type: str, leading: dict[str, Any]
) -> list[Any]:
    """Return *messages* with *tools_prompt* carried by its first message.

    The prompt is merged into a leading user message; when the first
    message is anything else, *leading* (the prompt in the transport's
    own message shape) is inserted ahead of it.

    Args:
        messages: The conversation messages, left untouched.
        tools_prompt: The text-based tools prompt.
        part_type: The text part type of the API in use (see
            :func:`_merge_tools_prompt_into_content`).
        leading: The message to insert when there is no leading user
            message to merge into.

    Returns:
        A new message list.
    """
    first = messages[0] if messages else None
    if isinstance(first, dict) and first.get("role") == "user":
        merged = _merge_tools_prompt_into_content(first.get("content"), tools_prompt, part_type)
        return [{"role": "user", "content": merged}, *messages[1:]]
    return [leading, *messages]


def _function_call_entry(call_id: str, name: str, args_json: str) -> dict[str, Any]:
    """Return the ``{"id", "name", "arguments"}`` call entry handed to the agent loop.

    Args:
        call_id: The provider's call id.
        name: The tool name.
        args_json: The model's argument string; decoded here (``{}`` when
            it is not a JSON object).
    """
    return {"id": call_id, "name": name, "arguments": parse_tool_arguments(args_json)}


def _function_call_item(
    call_id: str, name: str, args_json: str, item_id: str = ""
) -> dict[str, Any]:
    """Return the Responses ``function_call`` input item that replays one call.

    Args:
        call_id: The provider's call id.
        name: The tool name.
        args_json: The call's arguments as a JSON string.
        item_id: The provider's output item id (``fc_…``), kept when known
            so a Responses provider recognises its own item on replay.
    """
    item = {"type": "function_call", "call_id": call_id, "name": name, "arguments": args_json}
    if item_id:
        item["id"] = item_id
    return item


def _streamed_slot(
    slots: dict[int, dict[str, str]], index: int, blank: dict[str, str]
) -> dict[str, str]:
    """Return the slot of *slots* at *index*, made the last entry of the dict.

    The last entry is thereby the call being streamed, whatever its
    index: a fragment that names no call continues it.

    Args:
        slots: Streamed tool-call slots by index.  Mutated in place.
        index: The index of the call an event belongs to.
        blank: The slot to insert when *index* holds none.

    Returns:
        The slot now at *index*.
    """
    slot = slots[index] = slots.pop(index, blank)
    return slot


def _indexless_slot(accum: dict[int, dict[str, str]], call_id: str | None) -> int:
    """Return the accumulator index of a ``tool_calls`` delta sent without ``index``.

    A provider that omits ``index`` still opens each call with its id.

    Args:
        accum: The per-index accumulator.
        call_id: The delta's ``id`` (empty or ``None`` on a continuation).

    Returns:
        The slot already holding *call_id*, the next free index for an
        id not seen yet, and for a delta without id the last entry (the
        call being streamed, see :func:`_streamed_slot`).
    """
    if not call_id:
        return next(reversed(accum), 0)
    for index, slot in accum.items():
        if slot["id"] == call_id:
            return index
    return max(accum, default=-1) + 1


def _accumulate_tool_call_deltas(
    accum: dict[int, dict[str, str]], tool_call_deltas: list[Any]
) -> None:
    """Merge streamed ``tool_calls`` deltas into the per-index accumulator.

    Args:
        accum: Mapping of tool-call index to its accumulated
            id / name / arguments strings.  Mutated in place.
        tool_call_deltas: The ``delta.tool_calls`` entries of one chunk.
    """
    for tc_delta in tool_call_deltas:
        index = tc_delta.index
        if index is None:
            index = _indexless_slot(accum, tc_delta.id)
        slot = _streamed_slot(accum, index, {"id": "", "name": "", "arguments": ""})
        if tc_delta.id:
            slot["id"] = tc_delta.id
        if tc_delta.function:
            if tc_delta.function.name:
                slot["name"] = tc_delta.function.name
            if tc_delta.function.arguments:
                slot["arguments"] += tc_delta.function.arguments


class OpenAICompatibleBase(Model):
    """Shared vendor-detection helpers for the OpenAI-compatible transports.

    Both :class:`OpenAICompatibleModel` (Chat Completions) and
    :class:`~kiss.core.models.openai_compatible_model2.OpenAICompatibleModel2`
    (Responses) route the same model names to the same vendors, so the
    DeepSeek-R1 / OpenRouter-Anthropic detection and the OpenRouter
    prompt-cache request marker live here once.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
    ):
        """Initialize the shared OpenAI-compatible state.

        Resolves the provider model name (thinking-alias and
        ``openrouter/`` prefix stripped) and applies the catalog's default
        thinking level: when ``MODEL_INFO`` declares one for this model
        and the caller asked for no effort (:meth:`_asks_for_effort`),
        ``reasoning_effort`` is defaulted on a copy of ``model_config``.

        Args:
            model_name: The KISS catalog model name.
            base_url: The base URL for the API endpoint.
            api_key: API key for authentication.
            model_config: Optional model parameters.
            token_callback: Optional callback for each streamed text token.
            thinking_callback: Optional callback bracketing thinking blocks.
        """
        super().__init__(
            model_name,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
        self.base_url = base_url
        self.api_key = api_key
        self._api_model_name = _provider_model_name(model_name)
        thinking_level = _model_thinking_level(self.model_name)
        if thinking_level is not None and not self._asks_for_effort():
            self.model_config = dict(self.model_config)
            self.model_config["reasoning_effort"] = thinking_level

    def _asks_for_effort(self) -> bool:
        """Whether ``model_config`` names a reasoning effort, in either spelling.

        ``reasoning_effort`` is the Chat Completions parameter and the
        framework's portable key, ``reasoning.effort`` the Responses
        API's own; each transport sends the one its API takes, and a
        level under the portable key outranks the other on both.
        """
        reasoning = self.model_config.get("reasoning")
        return "reasoning_effort" in self.model_config or (
            isinstance(reasoning, dict) and "effort" in reasoning
        )

    def __str__(self) -> str:
        """Return a debug string showing the class, model and endpoint."""
        return f"{self.__class__.__name__}(name={self.model_name}, base_url={self.base_url})"

    __repr__ = __str__

    def _ensure_client(self) -> None:
        """Build the SDK client once, so its connection pool is reused.

        Every request builder calls this first (the delegated tool-calling
        transport before every agent step), so the client is rebuilt only
        when ``base_url``, ``api_key`` or ``extra_headers`` changed between
        turns; see :meth:`Model._reuse_client`.
        """
        extra_headers = self.model_config.get("extra_headers") or {}
        inputs = (self.base_url, self.api_key, tuple(sorted(extra_headers.items())))
        self._reuse_client(inputs, self._build_client)

    def _build_client(self) -> OpenAI:
        """Return a new ``OpenAI`` client for the current endpoint, key and headers.

        Its pool can be aborted (:func:`make_pool_abortable`), so a Stop
        also reaches a request whose connection is still being set up
        (:meth:`_abort_pending_request`).
        """
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=_CLIENT_TIMEOUT,
            max_retries=_MAX_RETRIES,
            default_headers=self.model_config.get("extra_headers") or {},
        )
        make_pool_abortable(client._client)
        return client

    def _stream_request(
        self,
        create: Callable[..., Any],
        kwargs: dict[str, Any],
        consume: Callable[[Any], _StreamResult],
    ) -> _StreamResult:
        """Open a streamed request under the stall clocks and hand it to *consume*.

        Three clocks bound the wait for a silent provider: the SDK's
        connect/read timeout (``stream_stall_timeout`` between bytes,
        ``stream_abort.CONNECT_TIMEOUT`` to connect), the deadline on the
        headers :meth:`_open_stream` holds the opening call to, and, once
        the headers arrive, the abort watchdog the transport's *consume*
        runs the events under.  All raise the same retryable stall error
        (or the Stop error when the user stopped meanwhile).  The client's
        one silent retry (:data:`_MAX_RETRIES`, kept for a transient
        429/5xx or connect failure) also re-sends a request whose headers
        never came, so a provider that never answers is reported after two
        header timeouts.

        A Stop that landed after *consume* left its loop early (a
        ``break`` on the terminal event skips the post-loop check inside
        ``stop_aware_events``) is honoured here, once the usage *consume*
        recorded is in ``_rejected_response`` for the stopped run's bill.

        Args:
            create: The SDK call that opens the stream
                (``client.chat.completions.create`` or
                ``client.responses.create``).
            kwargs: The request kwargs; ``stream`` and ``timeout`` are set
                here.
            consume: Called with the open stream; its result is returned.

        Returns:
            Whatever *consume* returns.
        """
        kwargs["stream"] = True
        kwargs["timeout"] = httpx.Timeout(self._stream_stall_timeout, connect=CONNECT_TIMEOUT)
        try:
            result = consume(self._open_stream(create, kwargs))
        except (httpx.TimeoutException, APITimeoutError) as err:
            raise stop_or_stall_error(self._stream_stall_timeout) from err
        if stop_signal.stop_requested():
            raise stop_error()
        return result

    def _open_stream(self, create: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        """Call *create* (which opens the stream) under a Stop-aware deadline on its headers.

        The SDK's read timeout bounds the silence between two reads
        only, and until the headers are complete there is no response
        whose socket the stream watchdog could shut down: a peer
        trickling header bytes would hold the turn for good, past any
        Stop.  So the call runs on a worker thread (:func:`_await_call`)
        and is given up at the thread's Stop or at the request's
        deadline (:meth:`_request_deadline`: the read timeout for every
        attempt the SDK may make, so its one silent retry of a header
        timeout is kept).  The client is then retired
        (:meth:`_abort_pending_request`) for the next turn to build a
        fresh one, and the cut request is reported as the Stop, or as
        the stall the stream watchdog would have raised.  A stream that
        opens after the call was given up is closed by the worker; the
        one handed back runs under the event watchdog only.

        Args:
            create: The SDK call that opens the stream.
            kwargs: Its keyword arguments, ``timeout`` already set.

        Returns:
            The open stream.

        Raises:
            KeyboardInterrupt: The user pressed Stop before the headers came.
            TimeoutError: The headers were not complete by the deadline.
        """
        due = time.monotonic() + self._request_deadline(kwargs)
        outcome: dict[str, Any] = {}
        try:
            _await_call(
                outcome, create, kwargs, due, self._abort_pending_request, "model-stream-open"
            )
        except BaseException:
            # A forced Stop: nobody reads a stream that just opened, so it
            # goes down with its client.
            self._abort_pending_request()
            raise
        finally:
            # On every way out: a stream that opens from here on is the
            # worker's to close.
            stream = outcome.setdefault("response", _ABANDONED)
        if stream is not _ABANDONED:
            return stream
        err = outcome.get("error")
        if err is None or stop_signal.stop_requested() or time.monotonic() >= due:
            self._abort_pending_request()
            if err is None or isinstance(err, APIConnectionError):
                raise stop_or_stall_error(self._stream_stall_timeout) from err
        raise err

    def _blocking_request(self, create: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        """Issue one non-streamed request, aborting it as soon as the user presses Stop.

        The callback-less turn keeps the client's long timeout on purpose:
        a non-streamed reasoning turn legitimately sends nothing for
        minutes, so no stall clock applies.  But a thread blocked in the
        SDK call cannot notice a Stop, and no stream watchdog can reach a
        request that has no response yet.  So the call runs on a worker
        thread while this one watches the stop event
        (:func:`_await_call`).  On a Stop the pending socket is shut down, which
        wakes the worker, and the client is closed and dropped so the
        SDK's retry of the woken call fails at once instead of re-sending
        the stopped request and the next turn builds a fresh pool.  A
        connection still being set up is shut down the moment it is up,
        before the request is written on it (the pool's
        ``AbortableNetworkBackend``).  If the socket could not be reached
        otherwise the worker is simply abandoned (a daemon thread) until
        the client timeout.

        The request as a whole has a deadline too: its read timeout over
        every SDK attempt (:meth:`_request_deadline`), since that timeout
        bounds the silence between two reads only and a peer sending a
        byte before each would hold the call for good.  Past it the
        request is torn down the same way and reported as a retryable
        stall (:func:`~kiss.core.models.stream_abort.stall_error`).

        A reply that arrived was billed, whatever the caller then makes
        of it (a Stop already set, a failed status, a tool call the agent
        rejects), so it is recorded in ``_rejected_response`` (as the
        streamed path records its usage chunk) before anything can raise.
        Outside a stoppable task (no stop event bound) the deadline alone
        bounds the call.

        Args:
            create: The SDK call (``client.chat.completions.create``,
                ``client.responses.create`` or an adaptive wrapper of one).
            kwargs: The request kwargs; a configured ``stream_options``
                is dropped here, since the request opens no stream.

        Returns:
            The SDK response object.

        Raises:
            KeyboardInterrupt: The user pressed Stop during the request.
            TimeoutError: No reply arrived by the request's deadline.
        """
        # OpenAI answers ``stream_options`` without ``stream`` with HTTP 400.
        kwargs.pop("stream_options", None)
        seconds = self._request_deadline(kwargs)
        deadline = time.monotonic() + seconds
        outcome: dict[str, Any] = {}
        try:
            _await_call(
                outcome,
                create,
                kwargs,
                deadline,
                self._abort_pending_request,
                "model-blocking-request",
            )
        finally:
            # Also on a KeyboardInterrupt raised into this thread: a
            # reply the worker already stored is still billed.
            if "response" in outcome:
                self._rejected_response = outcome["response"]
        if stop_signal.stop_requested():
            raise stop_error()
        if "response" not in outcome and time.monotonic() >= deadline:
            raise TimeoutError(
                f"{STALL_MARKER}: the request exceeded {seconds:.0f}s (model_config "
                f"'timeout' over every attempt). It was aborted instead of hanging; "
                f"it will be retried."
            )
        if "error" in outcome:
            raise outcome["error"]
        return outcome["response"]

    def _request_deadline(self, kwargs: dict[str, Any]) -> float:
        """Seconds a non-streamed request, or a stream's opening call, may take as a whole.

        Its read timeout (``kwargs["timeout"]``, else the client's) for
        every attempt the SDK may make, plus the backoff between them: a
        request that answered within its timeouts is never cut, a peer
        that keeps the read alive without answering is.

        Args:
            kwargs: The request kwargs.
        """
        timeout = kwargs.get("timeout", _CLIENT_TIMEOUT)
        read = timeout.read if isinstance(timeout, httpx.Timeout) else timeout
        if read is None:
            read = _CLIENT_TIMEOUT
        return float(read) * (_MAX_RETRIES + 1) + _RETRY_BACKOFF_SECONDS

    def _abort_pending_request(self) -> None:
        """Tear down the client whose blocking request the user stopped (or the deadline cut).

        Shuts the pool's sockets down (the only thing that wakes the
        worker parked in ``recv()``), then closes the client and drops it
        so :meth:`_ensure_client` builds a new one for the next turn.
        """
        client, self.client = self.client, None
        if client is None:
            return
        shutdown_pool_sockets(client._client)
        client.close()

    def _config_request_params(self) -> dict[str, Any]:
        """Return the ``model_config`` entries that may be request parameters.

        Framework-only keys (:data:`FRAMEWORK_ONLY_CONFIG_KEYS`) are left
        out, as is ``extra_headers``, which :meth:`_ensure_client` already
        installed as the client's default headers; the transport reshapes
        what it must and then keeps the subset its SDK call accepts via
        :meth:`_finish_request_kwargs`.

        Returns:
            A fresh dict of the forwardable config entries.
        """
        return {
            k: v
            for k, v in self.model_config.items()
            if k not in FRAMEWORK_ONLY_CONFIG_KEYS and k != "extra_headers"
        }

    def _finish_request_kwargs(
        self, kwargs: dict[str, Any], supported: frozenset[str], label: str
    ) -> dict[str, Any]:
        """Drop the parameters the SDK call rejects and add the cache marker.

        Args:
            kwargs: The assembled request kwargs.
            supported: The parameter names the SDK call accepts.
            label: The transport name used in the unsupported-key warning.

        Returns:
            The kwargs to send.
        """
        kwargs = self._keep_supported_request_params(kwargs, supported, label)
        self._apply_cache_control_for_openrouter_anthropic(kwargs)
        return kwargs

    # The content-part type carrying plain text in this transport's
    # messages: ``text`` for Chat Completions, ``input_text`` for Responses.
    _TEXT_PART_TYPE = "text"

    @classmethod
    def _text_part(cls, text: str) -> dict[str, str]:
        """Return *text* as this transport's text content part."""
        return {"type": cls._TEXT_PART_TYPE, "text": text}

    @classmethod
    def _user_message_with_attachments(cls, attachments: list[Attachment]) -> dict[str, Any] | None:
        """Build the user message carrying tool-result *attachments*.

        The attachment parts are followed by
        :data:`~kiss.core.models.model.TOOL_RESULT_ATTACHMENT_NOTE`.

        Args:
            attachments: The attachments to convert (unsupported formats
                are dropped with a warning).

        Returns:
            The user message, or ``None`` when no attachment survived the
            conversion so there is nothing to send.
        """
        parts = cls._attachments_to_content_parts(attachments)
        if not parts:
            return None
        parts.append(cls._text_part(TOOL_RESULT_ATTACHMENT_NOTE))
        return {"role": "user", "content": parts}

    def _create_embeddings(
        self, model_to_use: str, text_or_texts: str | list[str]
    ) -> list[list[float]]:
        """Send an embeddings request bounded by :data:`EMBEDDINGS_TIMEOUT`.

        The request names the model by its provider id
        (:func:`_provider_model_name`), as the chat and Responses requests
        do.  Its connect phase is capped at ``CONNECT_TIMEOUT``.  The
        timeout also bounds the whole request (:func:`_await_call`,
        which gives it up at the thread's Stop as well): the SDK's own
        timeout bounds the silence between two reads only, which a peer
        sending a byte before each read resets for as long as it likes,
        and its sleep before the one silent retry (:data:`_MAX_RETRIES`)
        follows the provider's ``Retry-After``; either would hold a
        memory-search worker that process exit joins.  A configured
        ``timeout`` of ``None`` (no SDK timeout) leaves the bound at the
        default.

        The request goes through a client of its own, closed with the
        call, not the shared ``self.client``: the cut shuts down every
        socket of the client's pool, and that pool
        (:func:`make_pool_abortable`) then refuses every later
        connection, so a shared client would be dead for good after one
        cut request, and the cut would reach the concurrent requests of
        the same instance (the memory tools embed a search's query while
        the index sync embeds its changed pages).

        Args:
            model_to_use: The embeddings model's catalog name.
            text_or_texts: One text or the batch to embed.

        Returns:
            The embedding vectors in input order.

        Raises:
            KISSError: If the request fails, times out, or returns a
                malformed response (a ``null`` embedding, a missing index,
                a vector count that does not match the input).
        """
        timeout = self.model_config.get("timeout", EMBEDDINGS_TIMEOUT)
        if isinstance(timeout, int | float):
            timeout = httpx.Timeout(timeout, connect=min(timeout, CONNECT_TIMEOUT))
        read = timeout.read if isinstance(timeout, httpx.Timeout) else None
        deadline = read if read is not None else EMBEDDINGS_TIMEOUT
        client = self._build_client()
        outcome: dict[str, Any] = {}
        try:
            _await_call(
                outcome,
                client.embeddings.create,
                {
                    "model": _provider_model_name(model_to_use),
                    "input": text_or_texts,
                    "timeout": timeout,
                },
                time.monotonic() + deadline,
                partial(shutdown_pool_sockets, client._client),
                "model-embeddings-request",
            )
            if "response" not in outcome:
                raise outcome.get("error") or TimeoutError(f"no reply within {deadline:g}s")
            ordered = sorted(outcome["response"].data, key=lambda item: item.index)
            expected = 1 if isinstance(text_or_texts, str) else len(text_or_texts)
            if [item.index for item in ordered] != list(range(expected)):
                raise ValueError(
                    f"expected the embeddings of indices 0..{expected - 1}, got indices "
                    f"{[item.index for item in ordered]}"
                )
            return [list(item.embedding) for item in ordered]
        except Exception as e:
            if stop_signal.stop_requested():  # the request was given up at the Stop
                raise stop_error() from e
            logger.debug("Exception caught", exc_info=True)
            raise KISSError(f"Embedding generation failed for model {model_to_use}: {e}") from e
        finally:
            client.close()

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name for embedding generation. Uses the
                model's name if not specified.

        Returns:
            A list of floating point numbers representing the embedding vector.

        Raises:
            KISSError: If the embedding generation fails.
        """
        return self._create_embeddings(embedding_model or self.model_name, text)[0]

    def get_embeddings(
        self, texts: list[str], embedding_model: str | None = None
    ) -> list[list[float]]:
        """Generate embedding vectors for several texts in one request.

        Args:
            texts: The texts to embed (at most 2048, 300k tokens in total).
            embedding_model: Optional model name for embedding generation. Uses the
                model's name if not specified.

        Returns:
            The embedding vectors, in the order of *texts*.

        Raises:
            KISSError: If the embedding generation fails.
        """
        if not texts:
            return []
        return self._create_embeddings(embedding_model or self.model_name, texts)

    def _generate_with_text_based_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generate one turn with text-based tool calling (the DeepSeek R1 fallback).

        The tool descriptions are injected into the conversation's first
        message and the model's text is parsed for tool-call blocks.  The
        transport supplies the request (:meth:`_request_with_tools_prompt`)
        and the conversation items it records for the turn
        (:meth:`_append_text_based_tool_turn`).

        Args:
            function_map: Dictionary mapping function names to callable functions.

        Returns:
            A tuple of (function_calls, content, response) where function_calls is a list
            of dictionaries containing parsed tool call information, content is the raw
            text response, and response is the raw API response object.
        """
        tools_prompt = _build_text_based_tools_prompt(function_map)
        content, response = self._request_with_tools_prompt(tools_prompt)
        content_clean = _extract_deepseek_reasoning(content)
        function_calls = _parse_text_based_tool_calls(content_clean)
        self._append_text_based_tool_turn(content_clean, function_calls)
        return function_calls, content, response

    @abstractmethod
    def _request_with_tools_prompt(self, tools_prompt: str) -> tuple[str, Any]:
        """Send the conversation with *tools_prompt* in its first message.

        Returns:
            A tuple of (content, response).
        """

    @abstractmethod
    def _append_text_based_tool_turn(
        self, content: str, function_calls: list[dict[str, Any]]
    ) -> None:
        """Record a text-based tool turn in this transport's conversation shape.

        Args:
            content: The response without the DeepSeek reasoning block.
            function_calls: The parsed tool calls (may be empty).
        """

    @staticmethod
    def _attachment_to_content_part(att: Attachment) -> dict[str, Any] | None:
        """Convert one :class:`Attachment` to this transport's content part.

        Each transport has its own part shape (``image_url`` /
        ``input_audio`` / ``file`` for Chat Completions, ``input_image`` /
        ``input_file`` for Responses), so the conversion is theirs; the
        filtering loop in :meth:`_attachments_to_content_parts` is shared.

        Args:
            att: The attachment to convert.

        Returns:
            The content-part dict, or ``None`` when the provider does not
            accept the attachment's format (a warning is logged).
        """
        raise NotImplementedError  # pragma: no cover – overridden by both transports

    @classmethod
    def _attachments_to_content_parts(cls, attachments: list[Attachment]) -> list[dict[str, Any]]:
        """Convert attachments to a list of content-part dicts.

        Unsupported MIME types are dropped (with a warning from
        :meth:`_attachment_to_content_part`).

        Args:
            attachments: The attachments to convert.

        Returns:
            A list of content-part dicts.  May be shorter than *attachments*
            if some MIME types were not supported.
        """
        parts: list[dict[str, Any]] = []
        for att in attachments:
            part = cls._attachment_to_content_part(att)
            if part is not None:
                parts.append(part)
        return parts

    def _is_deepseek_reasoning_model(self) -> bool:
        """Check if this is a DeepSeek R1 reasoning model.

        Uses ``_api_model_name`` (which strips the ``openrouter/`` routing
        prefix) so that models accessed via OpenRouter are matched correctly.

        Returns:
            True if the API model name is in DEEPSEEK_REASONING_MODELS.
        """
        return self._api_model_name in DEEPSEEK_REASONING_MODELS

    def _is_openrouter_anthropic(self) -> bool:
        """Check if this is an OpenRouter Anthropic model (Claude via OpenRouter).

        Matches both the direct form (``openrouter/anthropic/...``) and the alias
        form (``openrouter/~anthropic/claude-*-latest``), the same prefixes
        ``model_info`` prices with Anthropic cache rates.
        """
        from kiss.core.models.model_info import _ANTHROPIC_OPENROUTER_PREFIXES

        return self.model_name.startswith(_ANTHROPIC_OPENROUTER_PREFIXES)

    def _base_url_is_openrouter(self) -> bool:
        """Whether this adapter's endpoint host is OpenRouter (any ``openrouter.ai`` host)."""
        host = urlparse(self.base_url or "").hostname or ""
        return host == "openrouter.ai" or host.endswith(".openrouter.ai")

    def cost_multiplier_for_response(self, response: Any) -> float:
        """Return OpenAI's processing-tier factor for the tier that served *response*.

        OpenAI prices every token rate of a model by processing tier
        (https://developers.openai.com/api/docs/pricing, tier switcher):
        Flex at half the Standard rate, Fast (``service_tier`` ``fast``,
        or ``priority`` as GPT-5.6 and earlier still report it) at the
        model's published Fast ratio (see :func:`_openai_fast_multiplier`),
        and Ultrafast at six times.  The response's ``service_tier``
        names the tier actually used, so a Fast request downgraded under
        the ramp-rate limit comes back ``default`` and bills Standard.
        Endpoints without tiers never send the field and bill Standard.

        Args:
            response: A chat completion, Responses object, final stream
                chunk, or the dict form of any of them.

        Returns:
            The factor to apply to the Standard-rate catalog estimate.
        """
        tier = _get_attr_or_key(response, "service_tier")
        if tier == "flex":
            return _OPENAI_FLEX_MULTIPLIER
        if tier in ("fast", "priority"):
            return _openai_fast_multiplier(self.model_name)
        if tier == "ultrafast":
            return _OPENAI_ULTRAFAST_MULTIPLIER
        return 1.0

    def extract_cost_from_response(self, response: Any) -> float | None:
        """Return the USD amount OpenRouter reports it charged for *response*.

        OpenRouter attaches ``usage.cost`` (the total charged to the
        OpenRouter account, in USD), ``usage.is_byok`` and
        ``usage.cost_details.upstream_inference_cost`` (the upstream
        provider's charge) to every response and to the final usage chunk
        of a stream:
        https://openrouter.ai/docs/cookbook/administration/usage-accounting.
        Without BYOK the upstream charge is already inside ``cost`` (live
        responses report ``upstream_inference_cost == cost``); only under
        BYOK is it a separate bill on the user's own provider key, with
        ``cost`` being OpenRouter's fee.
        The same model id is billed at different rates depending on the
        upstream OpenRouter routes to, so this figure is the actual bill
        while the catalog rate is only the headline estimate.  Other
        OpenAI-compatible providers report no cost field.

        Args:
            response: A chat completion, Responses object, final stream
                chunk, or the dict form of any of them.

        Returns:
            ``cost`` (plus ``upstream_inference_cost`` when
            ``usage.is_byok`` is true) for a model served by OpenRouter
            (an ``openrouter/`` name or a custom endpoint whose base URL
            is openrouter.ai) whose response carries a ``usage.cost``
            that is an amount (:func:`_usd_amount`, as is the sum), else
            ``None`` so the agent falls back to ``calculate_cost``.
        """
        if not self.model_name.startswith("openrouter/") and not self._base_url_is_openrouter():
            return None
        usage = _get_attr_or_key(response, "usage")
        cost = _usd_amount(_get_attr_or_key(usage, "cost"))
        if cost is None or _get_attr_or_key(usage, "is_byok") is not True:
            return cost
        upstream = _get_attr_or_key(
            _get_attr_or_key(usage, "cost_details"), "upstream_inference_cost"
        )
        return _usd_amount(cost + (_usd_amount(upstream) or 0.0))

    def _apply_cache_control_for_openrouter_anthropic(self, kwargs: dict[str, Any]) -> None:
        """Add top-level cache_control for OpenRouter Anthropic prompt caching.

        Uses the same approach as AnthropicModel: a single top-level cache_control
        that lets OpenRouter automatically place the breakpoint at the last cacheable
        block and move it forward as the conversation grows.

        Args:
            kwargs: The request kwargs dict (``chat.completions.create`` or
                ``responses.create``).  Mutated in place.
        """
        if not self._is_openrouter_anthropic():
            return
        if not self.model_config.get("enable_cache", True):
            return
        existing = kwargs.get("extra_body")
        extra_body = dict(existing) if isinstance(existing, dict) else {}
        extra_body["cache_control"] = {"type": "ephemeral"}
        kwargs["extra_body"] = extra_body


class OpenAICompatibleModel(OpenAICompatibleBase):
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
        token_callback: TokenCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
    ):
        """Initialize an OpenAI-compatible model.

        Args:
            model_name: The name/identifier of the model to use.
            base_url: The base URL for the API endpoint (e.g., "http://localhost:11434/v1").
            api_key: API key for authentication.
            model_config: Optional dictionary of model configuration parameters.
            token_callback: Optional callback invoked with each streamed text token.
            thinking_callback: Optional callback invoked with ``True`` when a
                thinking block starts and ``False`` when it ends.
        """
        super().__init__(
            model_name,
            base_url,
            api_key,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
        self._responses_delegate: OpenAICompatibleModel2 | None = None
        self._delegate_raw_items: dict[str, list[dict[str, Any]]] = {}
        self.last_audio_data: str | None = None

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Initialize the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt to start the conversation.
            attachments: Optional list of file attachments (images, PDFs) to include.
        """
        self._ensure_client()
        self.conversation = []
        self._delegate_raw_items.clear()
        system_instruction = self.system_instruction_text()
        if system_instruction:
            self.conversation.append({"role": "system", "content": system_instruction})
        content: str | list[dict[str, Any]] = prompt
        if attachments:
            parts = self._attachments_to_content_parts(attachments)
            parts.append(self._text_part(prompt))
            content = parts
        self.conversation.append({"role": "user", "content": content})

    @staticmethod
    def _attachment_to_content_part(att: Attachment) -> dict[str, Any] | None:
        """Convert a single :class:`Attachment` to an OpenAI content-part dict.

        Only the formats OpenAI actually accepts are carried
        (:data:`OPENAI_INPUT_IMAGE_MIME_TYPES`,
        :data:`OPENAI_INPUT_AUDIO_FORMATS` and PDFs) — the same set
        :class:`~kiss.core.models.openai_compatible_model2.OpenAICompatibleModel2`
        applies, so a turn delegated to the Responses transport carries
        exactly the same attachments.  Anything else returns ``None`` with a
        warning naming the format, rather than being sent for the provider
        to reject or dropped in silence.

        Args:
            att: The attachment to convert.

        Returns:
            A content-part dict suitable for the ``content`` array of a
            chat-completions message, or ``None`` if OpenAI does not accept
            the format.
        """
        if att.mime_type == "application/pdf":
            return {"type": "file", "file": {"file_data": att.to_data_url()}}
        if att.mime_type in OPENAI_INPUT_IMAGE_MIME_TYPES:
            return {"type": "image_url", "image_url": {"url": att.to_data_url()}}
        if att.mime_type.startswith("audio/"):
            fmt = _audio_mime_to_format(att.mime_type)
            if fmt in OPENAI_INPUT_AUDIO_FORMATS:
                return {
                    "type": "input_audio",
                    "input_audio": {"data": att.to_base64(), "format": fmt},
                }
        logger.warning(
            "OpenAI does not accept %s attachments (images: %s; audio: %s; "
            "application/pdf); skipping.",
            att.mime_type,
            ", ".join(sorted(OPENAI_INPUT_IMAGE_MIME_TYPES)),
            ", ".join(sorted(OPENAI_INPUT_AUDIO_FORMATS)),
        )
        return None

    def _deliver_tool_result_attachments(self, attachments: list[Attachment]) -> None:
        """Re-attach tool-result bytes as Chat Completions content parts.

        The ``tool`` role only accepts string content, so binary payloads a
        tool produced (a PNG screenshot, an MP3) cannot live inside the tool
        message: they are carried by a follow-up ``user`` message holding
        ``image_url`` / ``file`` / ``input_audio`` parts.  Formats OpenAI
        does not accept are dropped by
        :meth:`_attachments_to_content_parts` with a warning, and when that
        leaves nothing to send no message is appended at all.

        Args:
            attachments: The attachments lifted out of the tool results.
        """
        message = self._user_message_with_attachments(attachments)
        if message is not None:
            self.conversation.append(message)

    @classmethod
    def _normalize_content_blocks(cls, content: Any) -> list[dict[str, Any]]:
        """Normalize content blocks to JSON-serializable dicts.

        See :func:`normalize_content_blocks`: Anthropic ``image`` /
        ``document`` blocks left by a conversation hand-off are converted,
        and ``thinking`` / ``redacted_thinking`` blocks are dropped because
        the OpenAI Chat Completions API has no such content-part type and
        rejects them with ``invalid_value``.

        Args:
            content: The content blocks from a response.

        Returns:
            list[dict[str, Any]]: Normalized content blocks as dictionaries.
        """
        return normalize_content_blocks(
            content,
            ("image", "document"),
            _anthropic_media_block_to_openai_part,
            keep_thinking=False,
        )

    def _normalize_conversation_for_api(
        self,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize all messages in a conversation before sending to the API.

        Ensures that all text content blocks are non-whitespace and that no
        messages contain only whitespace-only text blocks.  Also converts
        Anthropic Messages-format entries (assistant ``tool_use`` blocks,
        user ``tool_result`` blocks, ``thinking`` blocks) — which enter the
        conversation when it is handed off from an :class:`AnthropicModel`,
        e.g. via the Sorcar ``set_model`` tool — into the OpenAI Chat
        Completions equivalents (``tool_calls`` arrays, ``role="tool"``
        messages) so the API does not reject them with ``invalid_value``.
        OpenAI Responses-API items (handed off from an
        :class:`OpenAICompatibleModel2`) are converted first via
        :func:`responses_items_to_chat_messages`.

        Args:
            conversation: The conversation to normalize.

        Returns:
            list[dict[str, Any]]: The normalized conversation.
        """
        normalized: list[dict[str, Any]] = []
        for msg in responses_items_to_chat_messages(conversation):
            normalized.extend(self._normalize_message_for_api(msg))
        return normalized

    @classmethod
    def _normalize_message_for_api(cls, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize a single conversation message into OpenAI-format messages.

        A message may expand to zero messages (all content filtered out),
        one message (the common case), or several messages (an Anthropic
        user message carrying multiple ``tool_result`` blocks becomes one
        ``role="tool"`` message per block, followed by a user message
        carrying the images/PDFs nested in those results, if any).

        Args:
            msg: The conversation message to normalize.

        Returns:
            list[dict[str, Any]]: OpenAI Chat Completions-format messages.
        """
        msg_copy = msg.copy()
        if msg_copy.get("tool_calls"):
            msg_copy["tool_calls"] = _stringify_tool_call_arguments(msg_copy["tool_calls"])
        attachments = msg_copy.pop("attachments", None)
        if attachments:
            parts = cls._attachments_to_content_parts(attachments)
            prior = msg_copy.get("content")
            if isinstance(prior, str) and prior.strip():
                parts.append(cls._text_part(prior))
            elif isinstance(prior, list):
                parts.extend(prior)
            msg_copy["content"] = parts
        content = msg_copy.get("content")
        has_tool_calls = bool(msg_copy.get("tool_calls"))

        if isinstance(content, str):
            # A blank tool result still answers its tool call: dropping
            # it leaves the assistant's ``tool_calls`` unanswered, which
            # the next request is refused for.
            if content.strip() or has_tool_calls or msg_copy.get("role") == "tool":
                return [msg_copy]
            return []
        if not isinstance(content, list):
            return [msg_copy] if content is not None or has_tool_calls else []

        blocks = cls._normalize_content_blocks(content)
        tool_results = [b for b in blocks if b.get("type") == "tool_result"]
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        rest = [b for b in blocks if b.get("type") not in ("tool_result", "tool_use")]

        if tool_uses:
            text = text_of_parts(rest)
            tool_calls = list(msg_copy.get("tool_calls") or [])
            for b in tool_uses:
                tool_calls.append(
                    _chat_tool_call_entry(
                        b.get("id", ""), b.get("name", ""), json.dumps(b.get("input") or {})
                    )
                )
            return [
                {
                    "role": msg_copy.get("role", "assistant"),
                    "content": text,
                    "tool_calls": tool_calls,
                }
            ]
        if tool_results:
            converted = [
                {
                    "role": "tool",
                    "tool_call_id": b.get("tool_use_id", ""),
                    "content": _tool_result_block_text(b),
                }
                for b in tool_results
            ]
            media = [part for b in tool_results for part in _tool_result_block_media(b)]
            if media:
                rest = media + [cls._text_part(TOOL_RESULT_ATTACHMENT_NOTE)] + rest
            if rest:
                msg_copy["content"] = rest
                converted.append(msg_copy)
            return converted
        if not blocks:
            return [msg_copy] if has_tool_calls else []
        msg_copy["content"] = blocks
        return [msg_copy]

    @staticmethod
    def _build_tool_call_lists(
        entries: list[tuple[str, str, str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build function_calls and raw_tool_calls from (id, name, arguments_json) triples.

        Args:
            entries: List of (call_id, function_name, arguments_json_string) tuples.

        Returns:
            A tuple of (function_calls, raw_tool_calls) for conversation storage.
        """
        function_calls = [_function_call_entry(*entry) for entry in entries]
        raw_tool_calls = [_chat_tool_call_entry(*entry) for entry in entries]
        return function_calls, raw_tool_calls

    @staticmethod
    def _parse_tool_call_accum(
        accum: dict[int, dict[str, str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse accumulated streaming tool-call deltas into structured lists.

        Args:
            accum: Mapping of tool-call index to accumulated id/name/arguments strings.

        Returns:
            A tuple of (function_calls, raw_tool_calls) for conversation storage.
        """
        entries = [
            (accum[idx]["id"], accum[idx]["name"], accum[idx]["arguments"]) for idx in sorted(accum)
        ]
        return OpenAICompatibleModel._build_tool_call_lists(entries)

    @staticmethod
    def _parse_tool_calls_from_message(
        message: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract tool calls from a non-streamed OpenAI message.

        Args:
            message: The message object from a chat completion response.

        Returns:
            A tuple of (function_calls, raw_tool_calls) for conversation storage.
        """
        if not message.tool_calls:
            return [], []
        entries = [(tc.id, tc.function.name, tc.function.arguments) for tc in message.tool_calls]
        return OpenAICompatibleModel._build_tool_call_lists(entries)

    @staticmethod
    def _extract_output_audio(response: Any) -> Any | None:
        """Return the output-audio object of a chat completion, if any.

        Args:
            response: A chat-completions response (or final stream chunk).

        Returns:
            The ``message.audio`` object of the first choice — carrying
            ``data`` (base64 audio) and ``transcript`` — or ``None`` when
            the response has no audio output (the common text-only case,
            and always for streamed chunks, which carry deltas instead of
            a full message).
        """
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        return getattr(message, "audio", None)

    @staticmethod
    def _response_error_message(response: Any) -> str:
        """Return the gateway-level error message carried by a response or a choice.

        Several OpenAI-compatible gateways answer an upstream failure
        with ``200 {"choices": [], "error": {...}}`` rather than a 4xx,
        or end a choice with ``finish_reason="error"`` and an ``error``
        beside it, so the only diagnosis available is this extra field.

        Args:
            response: A Chat Completions response or one of its choices
                (SDK object or dict).

        Returns:
            The error message, or ``""`` when *response* carries none.
        """
        err = _get_attr_or_key(response, "error")
        if err is None:
            return ""
        return str(_get_attr_or_key(err, "message") or err)

    def _raise_for_finish_reason(self, choice: Any, response: Any, text: str = "") -> None:
        """Reject a completion the provider cut short instead of using its partial output.

        A :data:`_CUT_SHORT_FINISH_REASONS` value means the output stopped
        mid-sentence — and, on a tool-bearing turn, mid-JSON.  The
        partial ``arguments`` string then fails to parse and
        :meth:`_build_tool_call_lists` turns it into ``{}``, so the agent
        would call the tool with no arguments and be told the tool failed
        for a reason unrelated to the real cause.

        The cut-short completion was still billed: the transport already
        holds it in ``_rejected_response`` for
        :meth:`take_partial_usage_response`.

        Args:
            choice: The choice that carried the ``finish_reason`` (the
                response's first choice, or the stream chunk's), or
                ``None`` when no choice ended the completion.
            response: The response (or the stream's usage chunk) the
                choice came with; a gateway may put the upstream error
                there instead of on the choice.
            text: The text a stream delivered before it ended; a
                non-streamed choice carries its own.

        Raises:
            KISSError: When the provider cut the completion short:
                :class:`OutputCapError` for ``length``,
                :class:`ContentFilterError` for ``content_filter``, a
                plain :class:`KISSError` for a gateway's ``error``.
        """
        finish_reason: str = getattr(choice, "finish_reason", None) or ""
        cause = _CUT_SHORT_FINISH_REASONS.get(finish_reason)
        if cause is None:
            return
        error = self._response_error_message(choice) or self._response_error_message(response)
        advice = " Raise max_tokens or lower reasoning_effort." if finish_reason == "length" else ""
        message = (
            f"Chat Completions response was truncated (finish_reason={finish_reason!r}): "
            f"{cause}{f' ({error})' if error else ''}, so its text and any tool-call "
            f"arguments are incomplete.{advice}"
        )
        text = text or getattr(getattr(choice, "message", None), "content", None) or ""
        raise _terminal_error(finish_reason, "length", message, text)

    def _raise_for_failed_completion(self, response: Any) -> None:
        """Reject a non-streamed response that carries no usable choice.

        The counterpart of
        :meth:`~kiss.core.models.openai_compatible_model2.OpenAICompatibleModel2._raise_for_failed_response`
        for the Chat Completions transport.  *response* is billed either
        way; :meth:`_blocking_request` already recorded it in
        ``_rejected_response`` and the public generation methods clear
        the slot on success.

        Args:
            response: The response returned by ``chat.completions.create``.

        Raises:
            KISSError: When ``choices`` is empty (surfacing the gateway's
                own error message) or the single choice was cut short.
        """
        choices = getattr(response, "choices", None) or []
        if not choices:
            message = self._response_error_message(response)
            raise KISSError(
                "Chat Completions returned no choices" + (f": {message}" if message else "")
            )
        self._raise_for_finish_reason(choices[0], response)

    def _stream_chat_completion(
        self, kwargs: dict[str, Any], adaptive: bool
    ) -> tuple[str, dict[int, dict[str, str]], Any, Any]:
        """Stream one chat completion, driving the token/thinking callbacks.

        The single streaming loop behind both streamed entry points —
        :meth:`_stream_text` (tool-less) and
        :meth:`generate_and_process_with_tools` (the agentic path) — which
        differ only in whether the request goes through the adaptive
        ``reasoning_effort`` probe.

        The request is opened by the base :meth:`_stream_request`, whose
        per-request ``httpx.Timeout`` covers the wait for the headers;
        after they arrive :meth:`_consume_chat_stream` reads the chunks
        under the abort watchdog.

        Args:
            kwargs: Fully built request arguments; ``stream`` and
                ``stream_options.include_usage`` are set here.
            adaptive: Send through :meth:`_create_chat_completion_adaptive`
                (which may retry once without ``reasoning_effort``)
                instead of a plain create.

        Returns:
            ``(content, tool_call_accumulator, response, finish_choice)``;
            ``finish_choice`` is the choice that carried the
            ``finish_reason`` (:meth:`_consume_chat_stream` refuses a
            stream that ended without one).
        """
        # The usage chunk is the stream's billing record, so it is asked
        # for whatever else the caller's ``stream_options`` hold.
        configured = kwargs.get("stream_options")
        kwargs["stream_options"] = {
            **(configured if isinstance(configured, dict) else {}),
            "include_usage": True,
        }
        if adaptive:
            create: Callable[..., Any] = partial(self._create_chat_completion_adaptive, self.client)
            watchdog_name = "openai-tools-stream-abort-watchdog"
        else:
            create = self.client.chat.completions.create
            watchdog_name = "openai-stream-abort-watchdog"
        content, tool_calls_accum, response, finish_choice = self._stream_request(
            create, kwargs, partial(self._consume_chat_stream, watchdog_name=watchdog_name)
        )
        # Billed: kept for ``take_partial_usage_response`` until the public
        # generation method hands the response to its caller.
        self._rejected_response = response
        return content, tool_calls_accum, response, finish_choice

    def _consume_chat_stream(
        self, stream: Any, watchdog_name: str
    ) -> tuple[str, dict[int, dict[str, str]], Any, Any]:
        """Read one open chat-completion stream, driving the token/thinking callbacks.

        ``stop_aware_events`` (instead of a bare ``for chunk in ...``)
        aborts the socket on Stop or after ``stream_stall_timeout``
        seconds without an event; otherwise the thread would sit in
        ``recv()`` for the client's full 1800s timeout because the stop
        flag is only read when the agent emits something and an injected
        ``KeyboardInterrupt`` cannot reach a thread inside C code
        (``reports/stop_button_delay_2026-08-05.html``).

        Args:
            stream: The SDK's chunk iterator.
            watchdog_name: Thread name of the abort watchdog.

        Returns:
            ``(content, tool_call_accumulator, response, finish_choice)``;
            ``response`` is the last chunk read that carried usage, else
            the last chunk, ``finish_choice`` the choice that carried the
            ``finish_reason``.

        Raises:
            KISSError: When the stream ended before any choice carried a
                ``finish_reason``.
        """
        content = ""
        tool_calls_accum: dict[int, dict[str, str]] = {}
        response = None
        last_chunk = None
        finish_choice = None
        # The transport handler sits inside the watched block so that
        # the generator and the thinking bracket are closed only after
        # it ran: an error raised by that cleanup is not a transport
        # failure.
        with self._watched_events(stream, watchdog_name) as events:
            try:
                for chunk in events:
                    last_chunk = chunk
                    if chunk.usage is not None:
                        # Billed already: kept for
                        # take_partial_usage_response if the stream
                        # fails before it ends.  Recorded before the
                        # callbacks below run, since a provider may put
                        # the usage on a chunk that also carries content
                        # and a callback may raise (Stop) on it.
                        response = self._rejected_response = chunk
                    delta = None
                    if chunk.choices:
                        choice = chunk.choices[0]
                        if getattr(choice, "finish_reason", None):
                            finish_choice = choice
                        delta = choice.delta
                    if delta:
                        reasoning = _delta_reasoning_text(delta)
                        if reasoning:
                            self._open_thinking_if_closed()
                            self._invoke_token_callback(reasoning)
                        # A refusal streams in its own field, ``content`` null.
                        for text in (delta.content, delta.refusal):
                            if text:
                                self._close_thinking_if_open()
                                content += text
                                self._invoke_token_callback(text)
                        if delta.tool_calls:
                            self._close_thinking_if_open()
                            _accumulate_tool_call_deltas(tool_calls_accum, delta.tool_calls)
                    if finish_choice is not None and chunk.usage is not None:
                        # The answer and its billing are both complete;
                        # only ``[DONE]`` can follow.  Waiting for it let
                        # a connection that went silent here trip the
                        # stall watchdog, and the retry billed the
                        # finished turn again.  Leaving the loop closes
                        # the stream (the watched generator's cleanup),
                        # as v2 does on ``response.completed``.  The
                        # usage must be on this chunk: usage sent before
                        # the finish reason is a running count, and the
                        # final one may still follow.
                        break
            except (httpx.HTTPError, APIConnectionError) as err:
                # A transport failure AFTER ``finish_reason`` arrived
                # lost only the stream's tail (the usage chunk /
                # ``[DONE]``); the answer itself — text and tool-call
                # arguments — is complete.  Raising here would make the
                # agent loop re-send the whole conversation and the
                # provider regenerate (and bill) the same answer, which
                # the user sees as a repeated request and response.  A
                # failure BEFORE ``finish_reason`` means real content
                # was lost, so it still propagates for the agent-level
                # retry.  Timeouts are subclasses of these and belong to
                # ``_stream_request``.  Stops (``KeyboardInterrupt``) and
                # stalls (``TimeoutError``) raised by
                # ``stop_aware_events`` are not transport errors and are
                # never swallowed.
                if finish_choice is None or isinstance(
                    err, httpx.TimeoutException | APITimeoutError
                ):
                    raise
                logger.warning(
                    "Stream connection lost after finish_reason=%r; keeping "
                    "the complete response instead of retrying: %s",
                    finish_choice.finish_reason,
                    err,
                )
        if finish_choice is None:
            # The body ended (EOF or ``[DONE]``) before any choice carried
            # a ``finish_reason``: a gateway can close the stream cleanly
            # before the upstream model is done, so the partial text and
            # tool-call arguments must not become the turn.  The same
            # plain ``KISSError`` v2 raises for a stream without
            # ``response.completed``; usage already seen stays in
            # ``_rejected_response`` for ``take_partial_usage_response``.
            raise KISSError(
                "Chat Completions stream ended without a finish_reason, so its text "
                "and any tool-call arguments may be incomplete."
            )
        return (
            content,
            tool_calls_accum,
            response if response is not None else last_chunk,
            finish_choice,
        )

    def _stream_text(self, kwargs: dict[str, Any]) -> tuple[str, Any]:
        """Stream a chat completion, invoking the token callback for each text delta.

        When no callback is set, falls back to a normal (non-streaming) call.

        Args:
            kwargs: Keyword arguments for the OpenAI chat completions API.

        Returns:
            A tuple of (content, response).
        """
        if self.token_callback is None:
            response = self._blocking_request(self.client.chat.completions.create, kwargs)
            self._raise_for_failed_completion(response)
            return _message_text(response.choices[0].message), response

        content, _accum, response, finish_choice = self._stream_chat_completion(
            kwargs, adaptive=False
        )
        self._raise_for_finish_reason(finish_choice, response, content)
        return content, response

    def _build_chat_kwargs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Build the base kwargs dict for a Chat Completions request.

        Re-syncs the SDK client with the current endpoint inputs
        (:meth:`_ensure_client`), copies ``model_config`` (dropping keys
        that are not Chat Completions parameters, after sending a
        ``reasoning.effort`` that names a level as ``reasoning_effort``
        where that key names none: a null one is no setting, as on the
        Responses transport a tool turn may be delegated to),
        attaches the provider model name and ``messages``, and applies
        OpenRouter Anthropic cache control.

        Args:
            messages: The normalized messages to send.

        Returns:
            The kwargs dict for ``client.chat.completions.create(...)``.
        """
        self._ensure_client()
        kwargs = self._config_request_params()
        reasoning = kwargs.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
            if kwargs.get("reasoning_effort") is None:
                kwargs["reasoning_effort"] = reasoning["effort"]
            if len(reasoning) == 1:
                del kwargs["reasoning"]  # nothing left that this API would ignore
        kwargs.update({"model": self._api_model_name, "messages": messages})
        return self._finish_request_kwargs(kwargs, _CHAT_REQUEST_PARAMS, "OpenAI Chat Completions")

    def _normalized_conversation_checked(self) -> list[dict[str, Any]]:
        """Normalize the conversation for the API, raising when nothing remains.

        ``initialize()`` stores the configured ``system_instruction`` as the
        first conversation message, but a conversation handed over from a
        Responses / Anthropic / Gemini model (``set_model``, the fallback
        switch) carries none: those transports send the prompt out of band.
        The prompt is re-inserted here so every request carries it.

        Returns:
            The normalized conversation messages.

        Raises:
            KISSError: When every message was filtered out as whitespace-only.
        """
        normalized = checked_nonempty(self._normalize_conversation_for_api(self.conversation))
        system_instruction = self.system_instruction_text()
        if system_instruction and not any(
            message.get("role") == "system" for message in normalized
        ):
            normalized.insert(0, {"role": "system", "content": system_instruction})
        return normalized

    def generate(self) -> tuple[str, Any]:
        """Generate content from prompt without tools.

        Returns:
            A tuple of (content, response) where content is the generated text
            and response is the raw API response object.
        """
        kwargs = self._build_chat_kwargs(self._normalized_conversation_checked())

        self.last_audio_data = None
        content, response = self._stream_text(kwargs)
        audio = self._extract_output_audio(response)
        if audio is not None:
            self.last_audio_data = getattr(audio, "data", None)
            if not content:
                content = getattr(audio, "transcript", None) or ""

        if self._is_deepseek_reasoning_model():
            content = _extract_deepseek_reasoning(content)

        self.conversation.append({"role": "assistant", "content": content})
        self._rejected_response = None
        return content, response

    def generate_and_process_with_tools(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generate content with tools, process the response, and add it to conversation.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Optional pre-built tool schema list.

        Returns:
            A tuple of (function_calls, content, response) where function_calls is a list
            of dictionaries containing tool call information, content is the text response,
            and response is the raw API response object.
        """
        if self._is_deepseek_reasoning_model():
            return self._generate_with_text_based_tools(function_map)

        tools = self._resolve_openai_tools_schema(function_map, tools_schema)

        if tools and self._asks_for_effort() and self._should_delegate_to_responses():
            return self._generate_with_tools_via_responses(function_map, tools)

        kwargs = self._build_chat_kwargs(self._normalized_conversation_checked())
        if tools:
            kwargs["tools"] = tools
        else:
            # No tools is no key: ``"tools": null`` is outside the API's contract.
            kwargs.pop("tools", None)

        if self.token_callback is not None:
            (
                content,
                tool_calls_accum,
                response,
                finish_choice,
            ) = self._stream_chat_completion(kwargs, adaptive=True)
            self._raise_for_finish_reason(finish_choice, response, content)
            function_calls, raw_tool_calls = self._parse_tool_call_accum(tool_calls_accum)
        else:
            response = self._blocking_request(
                partial(self._create_chat_completion_adaptive, self.client), kwargs
            )
            self._raise_for_failed_completion(response)
            message = response.choices[0].message
            content = _message_text(message)
            function_calls, raw_tool_calls = self._parse_tool_calls_from_message(message)

        self._append_assistant_turn(content, function_calls, raw_tool_calls)
        self._rejected_response = None
        return function_calls, content, response

    def _append_assistant_turn(
        self,
        content: str,
        function_calls: list[dict[str, Any]],
        raw_tool_calls: list[dict[str, Any]],
    ) -> None:
        """Append the assistant turn, attaching ``tool_calls`` when calls were made.

        Args:
            content: The assistant text content.
            function_calls: Parsed function calls (may be empty).
            raw_tool_calls: Raw ``tool_calls`` entries to store when
                ``function_calls`` is non-empty.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if function_calls:
            msg["tool_calls"] = raw_tool_calls
        self.conversation.append(msg)

    def _should_delegate_to_responses(self) -> bool:
        """Decide whether tool-bearing requests should use ``/v1/responses``.

        The ``use_responses_api`` model-config flag forces the decision in
        either direction (``True`` = always delegate, ``False`` = never).
        When the flag is absent, the decision is the vendor's declared
        ``delegate_tools_to_responses`` capability from
        ``model_info.OPENAI_COMPATIBLE_PROVIDERS`` (currently only
        ``api.openai.com``, the provider that both implements
        ``/v1/responses`` and rejects ``tools`` + ``reasoning_effort`` on
        Chat Completions); unknown endpoints never delegate.

        Returns:
            True when the tool-calling transport should be the Responses API.
        """
        flag = self.model_config.get("use_responses_api")
        if flag is not None:
            return bool(flag)
        from kiss.core.models.model_info import openai_compatible_provider_for_base_url

        provider = openai_compatible_provider_for_base_url(self.base_url)
        return provider is not None and provider.delegate_tools_to_responses

    @property
    def _effort_verdict_key(self) -> tuple[str, str]:
        """Key for the adaptive ``tools`` + ``reasoning_effort`` verdict cache.

        Computed from the *current* ``base_url``: the endpoint can be
        reconfigured after construction (the same change that makes
        ``_ensure_client`` rebuild the SDK client), and a key captured at
        construction time would make the new endpoint inherit the old
        endpoint's cached verdict — or record its own probe result under
        the old endpoint's key.

        Returns:
            The ``(base_url, api_model_name)`` pair verdicts are cached
            under.
        """
        return (self.base_url, self._api_model_name)

    @staticmethod
    def _tools_reasoning_effort_capability(key: tuple[str, str]) -> bool | None:
        """Return whether the endpoint of *key* accepts ``tools`` + ``reasoning_effort``.

        The verdict is the vendor's declared
        ``tools_accept_reasoning_effort`` capability from
        ``model_info.OPENAI_COMPATIBLE_PROVIDERS`` when the endpoint is
        registered and verified; otherwise the cached adaptive verdict
        learned from the endpoint's actual responses, if any.

        Args:
            key: The ``(base_url, api_model_name)`` pair of the request
                (see :attr:`_effort_verdict_key`).

        Returns:
            True when the combination is known to be accepted, False when
            known to be rejected, None when unknown (send optimistically
            and let ``_create_chat_completion_adaptive`` learn the verdict).
        """
        from kiss.core.models.model_info import openai_compatible_provider_for_base_url

        provider = openai_compatible_provider_for_base_url(key[0])
        declared = provider.tools_accept_reasoning_effort if provider else None
        if declared is not None:
            return declared
        return _tool_effort_verdict(key)

    def _create_chat_completion_adaptive(self, client: OpenAI, **kwargs: Any) -> Any:
        """Create a chat completion, learning the effort capability if unknown.

        For endpoints whose ``tools`` + ``reasoning_effort`` capability is
        unknown (not declared in the vendor registry and not yet probed),
        the request is sent optimistically with the effort attached. If the
        endpoint rejects it with a 400 that mentions ``reasoning_effort``,
        the verdict is cached as False and the request is retried once
        without the effort; on success the verdict is cached as True.  An
        endpoint already known to reject the combination gets the request
        without the effort straight away; other known endpoints and
        requests without tools or effort pass straight through.

        Both sends go through *client*, bound by the caller when the turn
        began: a Stop meanwhile drops ``self.client`` (and a later turn
        rebuilds it), and the retry of a stopped turn must neither
        re-send its messages through the next turn's client nor at all.
        The verdict key is bound the same way, before the request: the
        reply of a stopped turn may be handled after the next turn moved
        the model to another endpoint, and it is a verdict on the
        endpoint it was sent to.

        Args:
            client: The SDK client the turn started with.
            kwargs: Fully-built Chat Completions request arguments
                (``reasoning_effort`` is left out of the request when the
                endpoint rejects it).

        Returns:
            The API response (or stream iterator when ``stream=True``).
        """
        key = self._effort_verdict_key
        capability = (
            self._tools_reasoning_effort_capability(key)
            if kwargs.get("tools") and "reasoning_effort" in kwargs
            else True
        )
        if capability is False:
            dropped = kwargs.pop("reasoning_effort")
            logger.debug(
                "Dropping reasoning_effort=%r because tools are attached and "
                "endpoint %s is known to reject the combination.",
                dropped,
                self.base_url,
            )
        unverified = capability is None
        try:
            response = client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if unverified and "reasoning_effort" in str(e):
                dropped = kwargs.pop("reasoning_effort")
                if _rejects_effort_value(e):
                    # The level, not the combination, is refused: no
                    # verdict, so the pair's other levels are still sent.
                    logger.warning(
                        "Endpoint %s rejected reasoning_effort=%r (%s); retrying without it.",
                        key[0],
                        dropped,
                        e,
                    )
                else:
                    _record_tool_effort_verdict(key, False)
                    logger.debug(
                        "Endpoint %s rejected tools + reasoning_effort=%r; "
                        "retrying without it and caching the verdict.",
                        key[0],
                        dropped,
                    )
                if client is not self.client:
                    raise  # the user stopped this turn meanwhile
                return client.chat.completions.create(**kwargs)
            raise
        if unverified:
            _record_tool_effort_verdict(key, True)
        return response

    @staticmethod
    def _chat_parts_to_responses_parts(
        parts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert Chat-Completions user content parts to Responses parts.

        ``text`` becomes ``input_text``, ``image_url`` becomes
        ``input_image``, ``file`` becomes ``input_file`` and ``input_audio``
        passes through.  Unknown part types are dropped with a warning.

        Args:
            parts: Chat-Completions content parts.

        Returns:
            Responses-API ``input_*`` content parts.
        """
        converted: list[dict[str, Any]] = []
        for part in parts:
            ptype = part.get("type")
            if ptype == "text":
                if str(part.get("text", "")).strip():
                    converted.append({"type": "input_text", "text": part["text"]})
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url:
                    converted.append({"type": "input_image", "image_url": url, "detail": "auto"})
            elif ptype == "file":
                file_info = part.get("file") or {}
                file_data = file_info.get("file_data", "")
                if file_data:
                    converted.append(
                        {
                            "type": "input_file",
                            "filename": file_info.get("filename", "attachment.pdf"),
                            "file_data": file_data,
                        }
                    )
            elif ptype == "input_audio":
                converted.append(part)
            else:
                logger.warning(
                    "Dropping unconvertible content part %r for the Responses API.",
                    ptype,
                )
        return converted

    def _chat_conversation_to_responses_input(self) -> list[dict[str, Any]]:
        """Convert the chat-format conversation to Responses ``input`` items.

        The conversation stays in OpenAI Chat Completions format as the
        single source of truth (so hand-off between models keeps working);
        this method derives the equivalent Responses-API ``input`` array
        on demand for the delegated tool-calling transport:

        * ``role="tool"`` messages become ``function_call_output`` items.
        * Assistant messages with ``tool_calls`` are replaced by the raw
          Responses output items cached for that turn (reasoning items
          included) when available, and otherwise reconstructed as
          ``function_call`` items.
        * Content-part lists are mapped to ``input_*`` parts.

        The conversation goes through :meth:`_normalized_conversation_checked`
        so a hand-off conversation without a system message gets the
        configured prompt as its first item: the delegate is built without
        ``system_instruction`` and sends no ``instructions``, so this item
        is the only copy of the prompt the request carries.

        Returns:
            The Responses-API ``input`` array.
        """
        items: list[dict[str, Any]] = []
        for msg in self._normalized_conversation_checked():
            tool_calls = msg.get("tool_calls") or []
            if msg.get("role") == "assistant" and tool_calls:
                cached = self._delegate_raw_items.get(tool_calls[0].get("id", ""))
                if cached is not None:
                    items.extend(cached)
                    continue
            items.extend(self._chat_message_to_responses_items(msg))
        return items

    @classmethod
    def _chat_message_to_responses_items(cls, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert one Chat-Completions message to Responses ``input`` items.

        * ``role="tool"`` messages become ``function_call_output`` items.
        * Assistant messages with ``tool_calls`` become an optional
          assistant text message plus one ``function_call`` item per call
          (dict arguments are JSON-encoded).
        * Content-part lists are mapped to ``input_*`` parts (assistant
          part lists collapse to their concatenated text).

        Args:
            msg: A Chat-Completions-format message.

        Returns:
            Responses-API ``input`` items (possibly empty).
        """
        role = msg.get("role")
        content = msg.get("content")
        items: list[dict[str, Any]] = []
        if role == "tool":
            output = content if isinstance(content, str) else json.dumps(content)
            return [
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": output,
                }
            ]
        if role == "assistant":
            if isinstance(content, list):
                text = text_of_parts(content)
            else:
                text = content if isinstance(content, str) else ""
            if text.strip():
                items.append({"role": "assistant", "content": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                args_json = args if isinstance(args, str) else json.dumps(args or {})
                items.append(_function_call_item(tc.get("id", ""), fn.get("name", ""), args_json))
            return items
        if isinstance(content, list):
            parts = cls._chat_parts_to_responses_parts(content)
            if parts:
                items.append({"role": role, "content": parts})
            return items
        if isinstance(content, str) and content.strip():
            items.append({"role": role, "content": content})
        return items

    def _generate_with_tools_via_responses(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Run one tool-bearing turn through the ``/v1/responses`` endpoint.

        Delegates the request to a lazily-created
        :class:`~kiss.core.models.openai_compatible_model2.OpenAICompatibleModel2`
        so ``reasoning_effort`` (including ``"xhigh"``) is preserved instead
        of being stripped.  The delegate is stateless across turns: its
        conversation is rebuilt from this model's chat-format conversation
        before every request, keeping the chat conversation the single
        source of truth (and hand-off compatible).  The raw Responses output
        items of the turn (reasoning items included) are cached per
        ``call_id`` so follow-up turns replay them verbatim.

        Args:
            function_map: Mapping of tool name to callable.
            tools: Chat-Completions-style tools schema (the delegate
                flattens it to the Responses shape).

        Returns:
            ``(function_calls, content, response)`` matching
            :meth:`generate_and_process_with_tools`.
        """
        import kiss.core.models as models

        # The lazy import, serialized by ``import_provider_module``.
        model2_cls = models.OpenAICompatibleModel2

        delegate_config = {
            k: v
            for k, v in self.model_config.items()
            if k not in ("system_instruction", "use_responses_api")
        }
        delegate = self._responses_delegate
        if delegate is None:
            delegate = model2_cls(
                model_name=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                model_config=delegate_config,
            )
            self._responses_delegate = delegate
        # The delegate is cached for its connection pool, so its
        # callbacks are synced every turn (the fresh one included):
        # KISSAgent._reset rebinds THIS model's callbacks to the new
        # run's printer without touching the delegate, which would
        # otherwise keep streaming every token and thinking bracket into
        # a finished task's printer.
        delegate.token_callback = self.token_callback
        delegate.thinking_callback = self.thinking_callback
        # The endpoint inputs have to be re-synced for the same reason:
        # this model's base_url / api_key / model_config (its
        # ``extra_headers``) can be reconfigured between turns (the very
        # change that rebuilds this model's own client), and a delegate
        # frozen at construction time would keep sending tool turns to
        # the OLD endpoint with the OLD credentials.  The delegate sends
        # through this model's own SDK client (same endpoint, credentials
        # and headers, one connection pool), rebuilt here exactly when
        # those inputs changed.
        delegate.base_url = self.base_url
        delegate.api_key = self.api_key
        delegate.model_config = delegate_config
        delegate._stream_stall_timeout = self._stream_stall_timeout
        self._ensure_client()
        delegate.client = self.client
        delegate._client_inputs = self._client_inputs
        delegate.reset_conversation()
        input_items = self._chat_conversation_to_responses_input()
        delegate.conversation = list(input_items)

        try:
            function_calls, content, response = delegate.generate_and_process_with_tools(
                function_map, tools
            )
        except BaseException:
            # A rejected delegated response was still billed; the agent
            # drains this model's hook, not the delegate's.  BaseException:
            # a Stop surfaces as KeyboardInterrupt and is billed too.
            rejected = delegate.take_partial_usage_response()
            if rejected is not None:
                self._rejected_response = rejected
            raise
        finally:
            # An abort of the delegate's pending request (a Stop, the
            # request deadline) closed the SHARED client
            # (_abort_pending_request) and dropped only the delegate's
            # reference; drop this model's too, or _reuse_client keeps the
            # closed pool and every later turn fails to send.  The turn
            # may still return normally: a reply that landed as the
            # deadline's abort ran is kept.
            if delegate.client is None:
                self.client = None

        # Everything the delegate appended is this turn's native output
        # (message / reasoning / function_call items); only its
        # add_function_results, never called here, appends anything else.
        new_items = list(delegate.conversation[len(input_items) :])
        # The ``tool_calls`` entries carry the model's own argument string
        # (the ``function_call`` item's ``arguments``), not a re-encoding
        # of the decoded dict: key order, number formatting and whitespace
        # replay exactly as written, and a string the decoder rejected
        # is not silently replaced by ``{}``.
        args_json_by_call_id = {
            item["call_id"]: item["arguments"]
            for item in new_items
            if item.get("type") == "function_call" and isinstance(item.get("arguments"), str)
        }
        raw_tool_calls = [
            _chat_tool_call_entry(
                fc["id"],
                fc["name"],
                args_json_by_call_id.get(fc["id"], json.dumps(fc["arguments"])),
            )
            for fc in function_calls
        ]
        if function_calls:
            self._delegate_raw_items[function_calls[0]["id"]] = new_items
        self._append_assistant_turn(content, function_calls, raw_tool_calls)
        self._prune_delegate_raw_items()
        return function_calls, content, response

    def _prune_delegate_raw_items(self) -> None:
        """Drop cached Responses items whose tool call left the conversation.

        The cache exists so a follow-up turn can replay a delegated
        turn's raw output items (reasoning included) verbatim, keyed by
        the turn's first call_id (the one
        :meth:`_chat_conversation_to_responses_input` looks up).  An entry
        whose call_id is no longer anywhere in the conversation can never
        be looked up again, so keeping it only
        grows the process's memory — one whole turn's payload per tool
        turn, for the lifetime of a model instance that Sorcar reuses
        across sub-sessions.
        """
        live = {
            tc.get("id", "")
            for msg in self.conversation
            if isinstance(msg, dict)
            for tc in (msg.get("tool_calls") or [])
        }
        for call_id in [k for k in self._delegate_raw_items if k not in live]:
            del self._delegate_raw_items[call_id]

    def reset_conversation(self) -> None:
        """Reset the conversation and the raw Responses items cached for it.

        The base implementation clears only the conversation, but this
        transport also caches raw Responses items per tool turn, dead
        weight once the conversation they describe is gone.  The
        Responses delegate needs no reset here: every delegated turn
        resets it before use (:meth:`_generate_with_tools_via_responses`).
        """
        super().reset_conversation()
        self._delegate_raw_items.clear()

    def _request_with_tools_prompt(self, tools_prompt: str) -> tuple[str, Any]:
        """Send the chat conversation with *tools_prompt* in its first message.

        Args:
            tools_prompt: The text-based tools prompt.

        Returns:
            A tuple of (content, response).
        """
        messages = _conversation_with_tools_prompt(
            self._normalized_conversation_checked(),
            tools_prompt,
            "text",
            {"role": "system", "content": tools_prompt},
        )
        return self._stream_text(self._build_chat_kwargs(messages))

    def _append_text_based_tool_turn(
        self, content: str, function_calls: list[dict[str, Any]]
    ) -> None:
        """Append the assistant message, with ``tool_calls`` when calls were parsed.

        Args:
            content: The response without the DeepSeek reasoning block.
            function_calls: The parsed tool calls (may be empty).
        """
        raw_tool_calls = [
            _chat_tool_call_entry(fc["id"], fc["name"], json.dumps(fc["arguments"]))
            for fc in function_calls
        ]
        self._append_assistant_turn(content, function_calls, raw_tool_calls)
        self._rejected_response = None

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> (
        tuple[int, int, int, int]
        | tuple[int, int, int, int, int]
        | tuple[int, int, int, int, int, int, int]
    ):
        """Extract token counts from an API response.

        Returns:
            (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).
            For OpenAI, cached_tokens is a subset of prompt_tokens; input_tokens
            is reported as (prompt_tokens - cached_tokens) so costs apply correctly.
            OpenRouter returns cache_write_tokens in prompt_tokens_details.
            OpenAI reasoning models may report reasoning tokens in
            completion_tokens_details.reasoning_tokens; those are counted as output
            tokens so Sorcar shows thinking-token usage.
            Responses-API responses (produced by the delegated tool-calling
            transport) report ``usage.input_tokens`` / ``usage.output_tokens``
            instead; those are routed to the delegate's extractor.
        """
        if self._responses_delegate is not None:
            usage_obj = _get_attr_or_key(response, "usage")
            if _get_attr_or_key(usage_obj, "input_tokens") is not None:
                return self._responses_delegate.extract_input_output_token_counts_from_response(
                    response
                )
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", None) or 0
            completion_tokens = getattr(usage, "completion_tokens", None) or 0
            cached_tokens = 0
            cache_write_tokens = 0
            audio_input_tokens = 0
            audio_output_tokens = 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached_tokens = getattr(details, "cached_tokens", 0) or 0
                cache_write_tokens = getattr(details, "cache_write_tokens", 0) or 0
                audio_input_tokens = getattr(details, "audio_tokens", 0) or 0
            if not cached_tokens:
                # Together reports the cached prompt tokens of some
                # (non-reasoning) models flat on ``usage.cached_tokens``,
                # with no details object; the SDK keeps the extra field.
                # Read only when the nested count is absent, never added
                # to it.  (https://docs.together.ai/docs/inference/openai-compatibility)
                flat_cached = _get_attr_or_key(usage, "cached_tokens")
                if isinstance(flat_cached, int | float) and not isinstance(flat_cached, bool):
                    cached_tokens = max(0, int(flat_cached))
            completion_details = getattr(usage, "completion_tokens_details", None)
            if completion_details is not None:
                audio_output_tokens = getattr(completion_details, "audio_tokens", 0) or 0
            text_input_tokens = max(
                0,
                prompt_tokens - cached_tokens - cache_write_tokens - audio_input_tokens,
            )
            text_output_tokens = max(0, completion_tokens - audio_output_tokens)
            if audio_input_tokens or audio_output_tokens:
                return (
                    text_input_tokens,
                    text_output_tokens,
                    cached_tokens,
                    cache_write_tokens,
                    0,
                    audio_input_tokens,
                    audio_output_tokens,
                )
            return (
                text_input_tokens,
                text_output_tokens,
                cached_tokens,
                cache_write_tokens,
            )
        return 0, 0, 0, 0
