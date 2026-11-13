# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Anthropic model implementation for Claude models."""

import base64
import contextlib
import functools
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

# The Anthropic SDK (>= 1.0) speaks HTTP through ``httpx2``, the
# maintained fork of httpx, and rejects ``httpx`` objects (Timeout, clients)
# at construction time; the exceptions it raises are httpx2's as well.
import httpx2
from anthropic import Anthropic, APITimeoutError, transform_schema
from anthropic.lib.bedrock import AnthropicBedrock
from anthropic.resources.messages import Messages
from pydantic import TypeAdapter

from kiss.core import stop_signal, user_config
from kiss.core.brand import HOME_DIR
from kiss.core.kiss_error import KISSError, ModelRefusalError, OutputCapError
from kiss.core.locked import locked_call
from kiss.core.models.model import (
    FRAMEWORK_ONLY_CONFIG_KEYS,
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    _audio_format_to_mime,
    _get_attr_or_key,
    accepted_request_params,
    checked_nonempty,
    import_provider_module,
    iter_assistant_tool_calls,
    merge_system_texts,
    normalize_content_blocks,
    parse_tool_arguments,
    pop_max_tokens_and_stop,
    responses_items_to_chat_messages,
    split_data_url,
    split_system_cache_break,
    strip_system_cache_break,
    text_of_parts,
    transcribe_audio,
)
from kiss.core.models.stream_abort import (
    CONNECT_TIMEOUT,
    RequestAbortWatchdog,
    make_pool_abortable,
    stop_error,
    stop_or_stall_error,
)

if TYPE_CHECKING:
    from kiss.core.models.model_info import ModelInfo

logger = logging.getLogger(__name__)

_MAX_RETRIES = 1

_RETRY_BACKOFF_SECONDS = 8.0
"""The SDK's longest delay before a retry (``MAX_RETRY_DELAY``).

The header deadline leaves room for every retry and its backoff, so a
silent peer is still reported by the SDK's own timeout after them.
"""

_WORKSPACE_ID_REQUIRED_MARKER = "anthropic-workspace-id is required"
"""The API's own words for the missing-workspace-id 400.

Matching this full phrase (not just the header name) keeps other errors
that merely mention the header — e.g. a malformed id — un-rewritten.
"""

_WORKSPACE_ID_HINT = (
    "Anthropic rejected the request: {error}\n"
    "The configured ANTHROPIC_API_KEY is an identity-linked API key, and the "
    "Anthropic API requires every request made with such a key to carry the "
    "id of the workspace it acts in. Find the workspace id (wrkspc_...) in "
    "the Anthropic Console under Settings > Workspaces, then set it as "
    "ANTHROPIC_WORKSPACE_ID: either in the settings panel of this install "
    "(API Keys section — applies from the next task on, no restart needed) "
    "or as a line\n"
    "    export ANTHROPIC_WORKSPACE_ID=wrkspc_...\n"
    f"in $KISS_HOME/api_keys.env (default ~/{HOME_DIR}/api_keys.env), followed by "
    "a daemon restart. To fix a remote install from the deploying machine "
    "instead, set it there and re-run ./rsorcar so the key store is shipped "
    "to the remote again."
)

# The request parameters an Anthropic message call accepts, taken from the
# SDK's own signature (keyword-only, no **kwargs), so an unsupported
# model_config key is reported instead of raising TypeError.
_ANTHROPIC_REQUEST_PARAMS = accepted_request_params(Messages.stream)

ANTHROPIC_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
"""The Anthropic ``output_config.effort`` scale, in ascending order.

Verified live on 2026-10-07 against every ``claude-*`` catalog entry
(https://platform.claude.com/docs/en/build-with-claude/effort): Opus 4.7 /
4.8 / 5 / 5.5, Sonnet 5 / 5.5 and Fable 5 / 5.1 accept all five levels;
Opus 4.6 and Sonnet 4.6 accept every level except ``xhigh``; Opus 4.5
accepts only ``low`` / ``medium`` / ``high``; Haiku 4.5 and Sonnet 4.5
reject the parameter.  Omitting ``effort`` is the vendor default
(``medium`` on Opus 5.5 and Haiku 5.5, ``high`` on Fable 5.1 and Sonnet
5.5 per the model table at
https://platform.claude.com/docs/en/models/haiku-5-5/overview, checked
2026-10-27); the adapter never omits it
for a catalog entry that carries a ``thinking`` level (see
:func:`_effort_level_for`)."""


def _effort_level_for(model_name: str, model_config: dict[str, Any]) -> str | None:
    """Return the ``output_config.effort`` level to send for *model_name*.

    Resolution order:

    1. An explicit ``reasoning_effort`` in ``model_config`` (the
       framework-wide spelling shared with the OpenAI adapters) wins: it
       is the key the agent sets when it lowers the effort for one turn
       (``KISSAgent._turn_effort``), so a caller's native setting must
       not outrank it.
    2. Otherwise the caller's native ``output_config.effort``.
    3. Otherwise the catalog entry's ``thinking`` level, when it is one
       of :data:`ANTHROPIC_EFFORT_LEVELS`: a generated ``-{level}`` alias
       carries its own level (``claude-opus-5-5-medium`` → ``"medium"``)
       and a base entry carries the level the catalog runs it at
       (``claude-opus-5-5`` → ``"high"``, the cap ``update_models.py``
       stores on a base; see ``_write_entry_with_thinking_split``).  The
       base level is sent explicitly rather than left to the vendor
       default because the two differ on Opus 5.5 and Haiku 5.5
       (``medium`` when omitted) and the request must match the catalog.
       Entries that reject the parameter (Haiku 4.5, Sonnet 4.5) carry no
       ``thinking`` and get none.

    Args:
        model_name: The catalog model name as passed to the adapter.
        model_config: The adapter's model configuration.

    Returns:
        The effort level string, or ``None`` when nothing selects one.
    """
    from kiss.core.models.model_info import MODEL_INFO

    explicit = model_config.get("reasoning_effort")
    if explicit is not None:
        return str(explicit)
    native = model_config.get("output_config")
    if isinstance(native, dict) and native.get("effort") is not None:
        return str(native["effort"])
    info = MODEL_INFO.get(model_name)
    if info is not None and info.thinking in ANTHROPIC_EFFORT_LEVELS:
        return str(info.thinking)
    return None


KEEP_ALIVE_TEXT = "Tool still running. Reply with the single word: ok."
"""Placeholder tool result of a prompt-cache keep-alive ping (see
:meth:`AnthropicModel.keep_prompt_cache_warm`)."""
KEEP_ALIVE_MAX_TOKENS = 256
"""Output cap of a keep-alive ping: bounds both its cost and how long the
agent thread may wait for an in-flight ping once the tool call ends.
Adaptive thinking may spend some of it before the reply; ``stop_reason ==
"max_tokens"`` is fine.  Raised to ``budget_tokens + 1`` when manual
extended thinking is configured, which the API requires."""
KEEP_ALIVE_TIMEOUT_SECONDS = 30.0
"""Bound of a keep-alive ping (a cache read plus a one-word reply): the
SDK's HTTP timeout between bytes and, through a
:class:`~kiss.core.models.stream_abort.RequestAbortWatchdog`, the whole
request's deadline; with retries off this bounds the wait for a ping in
flight."""


def _last_assistant_tool_use_ids(messages: list[Any]) -> list[str]:
    """Return the ``tool_use`` ids of the last assistant message in *messages*, or ``[]``."""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return [call_id for _name, call_id in iter_assistant_tool_calls(message) if call_id]
    return []


def cache_creation_tokens(usage: Any) -> tuple[int, int]:
    """Return the 5-minute and 1-hour cache-creation token counts.

    Anthropic reports cache writes split by TTL under ``cache_creation``
    and as the aggregate ``cache_creation_input_tokens``.  Writes the
    split does not cover — no split at all, or an aggregate the SDK
    updated from a streamed ``message_delta`` while the ``message_start``
    split stayed behind — share the TTL of the split when it shows only
    five-minute writes (a request's breakpoints all carry the same TTL,
    and KISS requests the five-minute default); otherwise they go to the
    one-hour bucket, the more expensive of the two, so an unknown TTL is
    never under-billed.  The Claude Code CLI re-emits exactly this shape as JSON, so
    :mod:`kiss.core.models.claude_code_model` shares this parser —
    otherwise a change to Anthropic's cache tiers would have to be made
    twice.

    Args:
        usage: The provider's usage record — an SDK object for the API
            transport, a decoded JSON dict for the CLI transport.

    Returns:
        ``(cache_write_5m_tokens, cache_write_1h_tokens)``.
    """
    five_m = one_h = 0
    cache_creation = _get_attr_or_key(usage, "cache_creation")
    if cache_creation is not None:
        five_m = _get_attr_or_key(cache_creation, "ephemeral_5m_input_tokens") or 0
        one_h = _get_attr_or_key(cache_creation, "ephemeral_1h_input_tokens") or 0
    total = _get_attr_or_key(usage, "cache_creation_input_tokens") or 0
    unsplit = max(0, total - five_m - one_h)
    if five_m and not one_h:
        return five_m + unsplit, 0
    return five_m, one_h + unsplit


_THINKING_FAMILIES = ("opus", "sonnet", "haiku", "fable")


def _parse_claude_version(model_name: str) -> tuple[str, int, int | None] | None:
    """Parse ``claude-<family>-<major>[-<minor>][-<date>]`` model names.

    The accepted shapes are exactly ``claude-<family>-<major>``,
    ``claude-<family>-<major>-<date>``, ``claude-<family>-<major>-<minor>``,
    and ``claude-<family>-<major>-<minor>-<date>``, where ``<family>`` is
    alphabetic, ``<major>`` and ``<minor>`` are non-zero-padded runs of
    digits, ``<minor>`` is not 8 digits long, and ``<date>`` is exactly
    8 digits and may only be the final segment.

    Args:
        model_name: The model name to parse (e.g. ``claude-opus-4-8``,
            ``claude-opus-5``, ``claude-haiku-4-5-20251001``).

    Returns:
        A ``(family, major, minor)`` tuple, where ``minor`` is ``None``
        when the name has no minor version (e.g. ``claude-opus-4`` or a
        dated snapshot of a bare major like ``claude-opus-4-20250514``,
        whose 8-digit date segment is not a minor version).  Returns
        ``None`` for every name outside the grammar above: non-Claude
        models, legacy ``claude-3-5-sonnet-*`` names whose family slot
        holds a digit, zero-padded or non-numeric majors or minors
        (``claude-opus-04``, ``claude-opus-x``, ``claude-opus-5-04``),
        empty or non-numeric trailing segments (``claude-opus-5-``,
        ``claude-opus-5-junk``), extra segments beyond
        ``<minor>-<date>`` (``claude-opus-5-1-2``,
        ``claude-opus-5-1-20260301-777``), and non-final 8-digit date
        segments (``claude-opus-4-20250514-1``,
        ``claude-opus-5-20260301-99``).
    """
    parts = model_name.split("-")
    if len(parts) < 3 or len(parts) > 5 or parts[0] != "claude":
        return None
    family = parts[1]
    if not family.isalpha():
        return None
    major_str = parts[2]
    if not major_str.isdigit() or (len(major_str) > 1 and major_str[0] == "0"):
        return None
    tail = parts[3:]
    if tail and len(tail[-1]) == 8 and tail[-1].isdigit():
        tail = tail[:-1]  # a trailing 8-digit segment is a date, not a minor
    if len(tail) > 1:
        return None
    minor: int | None = None
    if tail:
        minor_str = tail[0]
        if (
            not minor_str.isdigit()
            or len(minor_str) == 8
            or (len(minor_str) > 1 and minor_str[0] == "0")
        ):
            return None
        minor = int(minor_str)
    return family, int(major_str), minor


ClaudeProfile = tuple["ModelInfo | None", tuple[str, int, int | None] | None]


def _claude_profile(model_name: str) -> ClaudeProfile:
    """Resolve the catalog entry and parsed version the capability checks use.

    A generated effort alias (``claude-opus-5-5-medium``) is judged by its
    base id: the alias name itself falls outside the version grammar of
    :func:`_parse_claude_version` and has no catalog flags of its own.

    Args:
        model_name: The catalog model name (``claude-*``, possibly an
            effort alias).

    Returns:
        ``(info, version)`` where ``info`` is the ``MODEL_INFO`` entry of
        the base id (``None`` when uncatalogued) and ``version`` the
        ``(family, major, minor)`` tuple of :func:`_parse_claude_version`
        (``None`` when the name is outside the grammar).
    """
    from kiss.core.models.model_info import MODEL_INFO, _strip_thinking_alias

    model_name = _strip_thinking_alias(model_name)
    return MODEL_INFO.get(model_name), _parse_claude_version(model_name)


def _uses_adaptive_thinking(profile: ClaudeProfile) -> bool:
    """Return True if the Claude model requires ``thinking.type=adaptive``.

    ``profile`` is the ``(info, version)`` pair of :func:`_claude_profile`.
    Consultation order:

    1. ``info.adaptive_thinking`` when explicitly set (``True`` /
       ``False``) — the highest-priority source of truth.  This is loaded
       from ``MODEL_INFO.json`` at import time so a JSON edit reconfigures
       the adapter without a code change.
    2. A version-aware heuristic on the model name, parsed as
       ``claude-<family>-<major>[-<minor>]``:

       * every modern-family model (opus/sonnet/haiku/fable) with major
         version >= 5 uses adaptive thinking (``claude-opus-5``,
         ``claude-fable-5``, future ``claude-*-6`` ... included), since
         Anthropic rejects ``thinking.type=enabled`` for these models;
       * within the 4.x generation only Opus 4.6 and later use
         adaptive; older Opus 4.x (4, 4.1, 4.5) and all Sonnet/Haiku
         4.x still use ``enabled``. An 8-digit date segment (e.g.
         ``claude-opus-4-20250514``) is not treated as a minor version.
    """
    info, version = profile
    if info is not None and info.adaptive_thinking is not None:
        return info.adaptive_thinking
    if version is None:
        return False
    family, major, minor = version
    if family not in _THINKING_FAMILIES:
        return False
    if major >= 5:
        return True
    return family == "opus" and major == 4 and minor is not None and minor >= 6


def _checks_thinking_prefix(profile: ClaudeProfile) -> bool:
    """Return whether the model binds a thinking block to the messages before it.

    ``profile`` is the ``(info, version)`` pair of :func:`_claude_profile`.
    Claude 5.1 and later (Fable and Mythos 5.1 and 5.2, Opus, Sonnet and
    Haiku 5.5) sign each thinking block over the system prompt, the tools
    and the messages before it.  Sent back after any of those changed, the
    block fails the check: the API drops it or, on Claude Fable 5.2,
    answers 400 ("The block is bound to a different conversation").  A
    name outside the version grammar is taken for an earlier model, as the
    legacy ``claude-3`` names are.
    """
    _info, version = profile
    return version is not None and (version[1], version[2] or 0) >= (5, 1)


def _supports_extended_thinking(profile: ClaudeProfile) -> bool:
    """Return True if the Claude model should send the ``thinking`` param.

    ``profile`` is the ``(info, version)`` pair of :func:`_claude_profile`.
    Consultation order:

    1. ``info.extended_thinking`` when explicitly set —
       the highest-priority source of truth. Setting the flag to
       ``False`` lets ``MODEL_INFO.json`` opt a specific model out of
       extended thinking without a code change.
    2. A version-aware heuristic on the model name, parsed as
       ``claude-<family>-<major>``: every modern-family model
       (opus/sonnet/haiku/fable) with major version >= 4 supports
       extended thinking. Legacy ``claude-3*`` names (whose family slot
       holds a digit) and non-Claude names never match, so they stay
       non-thinking.

    The paper-analysed ``claude-fable-5`` failure — and the identical
    ``claude-opus-5`` regression after it — lived in the gap between an
    unset JSON flag and the previous hardcoded ``claude-*-4`` prefix
    allowlist: the adapter never sent the ``thinking`` param, so the
    model's reasoning stayed invisible (or came back encrypted-only and
    ``KISSAgent`` misread it as "empty response").
    """
    info, version = profile
    if info is not None and info.extended_thinking is not None:
        return info.extended_thinking
    if version is None:
        return False
    family, major, _minor = version
    return family in _THINKING_FAMILIES and major >= 4


def _openai_part_to_anthropic_block(part: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an OpenAI content part to the equivalent Anthropic block.

    OpenAI Chat Completions content parts (``image_url`` / ``file`` /
    ``input_audio``) enter the conversation when it is handed off from an
    OpenAI-schema model (e.g. via the Sorcar ``set_model`` tool).  The
    Anthropic Messages API instead expects ``image`` / ``document`` blocks;
    audio has no Anthropic equivalent and is transcribed via Whisper when
    possible.

    Args:
        part: The OpenAI content-part dict.

    Returns:
        The equivalent Anthropic block dict, or ``None`` when the part
        cannot be represented (in which case it is dropped with a warning).
    """
    part_type = part.get("type")
    if part_type == "image_url":
        url = (part.get("image_url") or {}).get("url", "")
        parsed = split_data_url(url)
        if parsed is not None:
            media_type, data = parsed
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        if url:
            return {"type": "image", "source": {"type": "url", "url": url}}
        logger.warning("Dropping unconvertible OpenAI image_url part.")
        return None
    if part_type == "file":
        file_data = (part.get("file") or {}).get("file_data", "")
        parsed = split_data_url(file_data)
        if parsed is not None:
            media_type, data = parsed
            return {
                "type": "document",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        logger.warning("Dropping unconvertible OpenAI file part.")
        return None
    if part_type == "input_audio":
        audio = part.get("input_audio") or {}
        fmt = audio.get("format", "")
        mime_type = _audio_format_to_mime(fmt)
        try:
            audio_bytes = base64.b64decode(audio.get("data", ""))
        except (ValueError, TypeError):
            logger.warning("Dropping undecodable OpenAI input_audio part.")
            return None
        # Transcribed (or dropped with a warning) exactly like an audio attachment.
        return next(iter(_attachments_to_blocks([Attachment(audio_bytes, mime_type)])), None)
    logger.warning("Dropping unconvertible OpenAI %s content part.", part_type)
    return None


def _tool_calls_to_tool_use_blocks(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI ``tool_calls`` entries into Anthropic ``tool_use`` blocks.

    Args:
        tool_calls: The ``tool_calls`` list of an OpenAI-format assistant
            message (dicts with ``id`` / ``function`` keys).

    Returns:
        The equivalent list of Anthropic ``tool_use`` block dicts.
    """
    blocks: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        call_id = tc.get("id", "")
        name = fn.get("name", "")
        arguments = fn.get("arguments", "")
        input_dict = parse_tool_arguments(arguments)
        blocks.append({"type": "tool_use", "id": call_id, "name": name, "input": input_dict})
    return blocks


# Whisper transcripts of the audio clips this process converted, keyed by
# (SHA-256 of the clip, MIME type): one short text per distinct clip.  A
# conversation keeps foreign audio raw (an ``attachments`` entry or an
# OpenAI ``input_audio`` part) and is converted again for every request
# and keep-alive ping, each of which would otherwise pay for one more
# Whisper call and could get a different transcript, changing the prompt
# bytes the prompt cache is keyed on.  The daemon lives for days and its
# conversations do not: only the clips used most recently are kept.
_TRANSCRIPT_CACHE_SIZE = 256
_TRANSCRIPTS: OrderedDict[tuple[bytes, str], str] = OrderedDict()
_TRANSCRIPTS_LOCK = threading.Lock()


def _transcript(att: Attachment) -> str:
    """Return the Whisper transcript of audio *att*, transcribing each clip once.

    The :data:`_TRANSCRIPT_CACHE_SIZE` clips used most recently are
    remembered; a failure is not, so the next request tries again; two
    threads converting the same new clip at once may both transcribe it.
    """
    key = (hashlib.sha256(att.data).digest(), att.mime_type)
    with _TRANSCRIPTS_LOCK:
        text = _TRANSCRIPTS.get(key)
        if text is not None:
            _TRANSCRIPTS.move_to_end(key)
            return text
    text = transcribe_audio(att.data, att.mime_type)
    with _TRANSCRIPTS_LOCK:
        _TRANSCRIPTS[key] = text
        if len(_TRANSCRIPTS) > _TRANSCRIPT_CACHE_SIZE:
            _TRANSCRIPTS.popitem(last=False)
    return text


def _attachments_to_blocks(attachments: list[Attachment]) -> list[dict[str, Any]]:
    """Convert :class:`Attachment` objects into Anthropic content blocks.

    Images become ``image`` blocks, PDFs become ``document`` blocks, and
    audio is transcribed to text via Whisper when possible, once per clip
    (:func:`_transcript`).  Unsupported MIME types (e.g. video) are
    dropped with a warning.

    Args:
        attachments: The attachments to convert.

    Returns:
        list[dict[str, Any]]: The equivalent Anthropic content blocks.
    """
    blocks: list[dict[str, Any]] = []
    for att in attachments:
        if att.mime_type.startswith("image/") or att.mime_type == "application/pdf":
            # Encoded only when sent: audio is transcribed, the rest dropped.
            block_type = "image" if att.mime_type.startswith("image/") else "document"
            source = {
                "type": "base64",
                "media_type": att.mime_type,
                "data": att.to_base64(),
            }
            blocks.append({"type": block_type, "source": source})
        elif att.mime_type.startswith("audio/"):
            try:
                text = _transcript(att)
                blocks.append({"type": "text", "text": f"[Audio transcription]\n{text}"})
            except Exception as exc:
                logger.warning(
                    "Anthropic does not support %s attachments and "
                    "automatic transcription failed (%s); skipping.",
                    att.mime_type,
                    exc,
                )
        else:
            logger.warning(
                "Anthropic does not support %s attachments; skipping.",
                att.mime_type,
            )
    return blocks


def _content_as_block_list(content: Any) -> list[dict[str, Any]]:
    """Return message content as a list of Anthropic content blocks.

    Args:
        content: A message ``content`` value (string or block list).

    Returns:
        The content as a block list, wrapping strings in a text block.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"
"""The beta header of the server-side context edits (``context_management``)."""


def context_edits(thinking: bool = True) -> list[dict[str, Any]]:
    """Return the server-side context edits of a request under the keep-K policy.

    The retained context holds the model's reasoning for the last
    ``Config.reasoning_keep_turns`` turns (``clear_thinking_20251015``; the
    agent starts a turn every ``Config.reasoning_turn_steps`` steps).  Empty
    when the setting is 0.  Old tool calls are not cleared by the server:
    the boundary of its ``clear_tool_uses`` edit moved by a result or two
    between requests, and each move re-wrote the cached prefix without
    freeing anything (312 times on 2026-11-11, 55 million tokens written
    again), so the agent clears them in its own conversation
    (:meth:`AnthropicModel.clears_tool_uses_in_bulk`).

    Args:
        thinking: Whether the request's thinking is enabled or adaptive;
            the API rejects the thinking edit on a request without it, so
            ``False`` leaves that edit out.
    """
    from kiss.core.config import DEFAULT_CONFIG

    if not thinking or DEFAULT_CONFIG.reasoning_keep_turns <= 0:
        return []
    return [{
        "type": "clear_thinking_20251015",
        "keep": {"type": "thinking_turns", "value": DEFAULT_CONFIG.reasoning_keep_turns},
    }]


def applied_context_edits(source: Any) -> list[dict[str, Any]]:
    """Return the ``context_management.applied_edits`` of a response or stream event as dicts.

    Args:
        source: A final message or a ``message_delta`` event (SDK object
            or dict); the field is a pydantic extra on the non-beta types.

    Returns:
        One dict per applied edit (``type``, ``cleared_input_tokens``,
        ``cleared_thinking_turns`` or ``cleared_tool_uses``); empty when the
        response reports none.
    """
    management = _get_attr_or_key(source, "context_management")
    edits = _get_attr_or_key(management, "applied_edits") if management is not None else None
    if not isinstance(edits, list):
        return []
    return [
        edit if isinstance(edit, dict) else edit.model_dump()
        for edit in edits
        if isinstance(edit, dict) or hasattr(edit, "model_dump")
    ]


def _add_beta(kwargs: dict[str, Any], token: str) -> None:
    """Add *token* to the request's ``anthropic-beta`` header unless it is already there.

    Bedrock's SDK client moves the header into the body's ``anthropic_beta``.

    Args:
        kwargs: Keyword arguments for the Anthropic API call, updated in place.
        token: The beta feature name.
    """
    existing_headers = kwargs.get("extra_headers") or {}
    beta_header = existing_headers.get("anthropic-beta", "")
    if token not in beta_header:
        merged_beta = f"{beta_header},{token}" if beta_header else token
        kwargs["extra_headers"] = {**existing_headers, "anthropic-beta": merged_beta}


def _check_streamed_tool_inputs(
    raw_tool_inputs: dict[int, str], message: Any, model_name: str
) -> None:
    """Raise when a completed ``tool_use`` block streamed invalid JSON.

    Fine-grained tool streaming sends tool input unvalidated, and the
    SDK's partial parser silently drops a malformed trailing value, so
    the raw text is checked strictly.  The block an output cap cut off
    is skipped: the caller drops it.

    Args:
        raw_tool_inputs: Concatenated ``input_json_delta`` text per block index.
        message: The final streamed message.
        model_name: The model, named in the error.

    Raises:
        ValueError: A completed tool input is not valid JSON (retryable).
    """
    cut_index = len(message.content) - 1 if message.stop_reason == "max_tokens" else -1
    for index, raw in raw_tool_inputs.items():
        if index == cut_index or not raw.strip():
            continue
        try:
            json.loads(raw)
        except ValueError as exc:
            raise ValueError(
                f"Anthropic model {model_name} streamed malformed JSON for a tool "
                f"call's arguments: {exc}"
            ) from exc


def _system_param(system_instruction: str, enable_cache: bool) -> str | list[dict[str, Any]]:
    """Shape the ``system`` request parameter from the merged instruction.

    With prompt caching on, the instruction is sent as text blocks: the
    part before :data:`~kiss.core.models.model.SYSTEM_CACHE_BREAK` (or
    the whole instruction when there is no marker) carries an explicit
    ``cache_control`` breakpoint, so ``tools`` plus that prefix form one
    cache entry that every request with the same prefix reads — the
    automatic top-level breakpoint alone writes an entry only at the
    end of the prompt, which differs for every task, so a task's first
    call never hit the cache.  The per-run tail after the marker follows
    as a plain block.  With caching off the marker is dropped and the
    instruction is sent as a single string.

    Args:
        system_instruction: The merged, non-empty system instruction.
        enable_cache: Whether prompt caching is on for this model.

    Returns:
        The ``system`` value: a string, or a list of text blocks.
    """
    prefix, tail = split_system_cache_break(system_instruction)
    if not enable_cache or not prefix.strip():
        return strip_system_cache_break(system_instruction)
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}
    ]
    if tail.strip():
        blocks.append({"type": "text", "text": tail})
    return blocks


def _well_formed(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the request *kwargs* with their text repaired to well-formed Unicode.

    ``json.loads`` turns a ``"\\ud83d"`` escape (a model's own tool
    argument, a tool result) into a lone surrogate, which no UTF-8
    request body can carry: the SDK raises ``UnicodeEncodeError`` while
    building the body, before anything is sent, and would for every
    later request, since the text stays in the conversation.  The repair
    is the OpenAI-compatible adapters' (a split pair joined, a lone half
    as U+FFFD), imported on first use so that a process using Claude
    alone does not load the OpenAI SDK.

    Args:
        kwargs: The ``messages`` request keyword arguments.

    Returns:
        The repaired copy.
    """
    openai_adapter = import_provider_module("kiss.core.models.openai_compatible_model")
    repaired: dict[str, Any] = openai_adapter._well_formed(kwargs)
    return repaired


def _has_cache_breakpoint(kwargs: dict[str, Any]) -> bool:
    """Tell whether the request *kwargs* would carry a prompt-cache breakpoint.

    Looks at the request as it goes on the wire: the SDK merges
    ``extra_body`` over the outer keyword arguments, so a ``system`` or
    ``messages`` supplied there replaces the outer one.  A breakpoint
    is a top-level ``cache_control`` (automatic caching) or a
    ``cache_control`` on a system block, a tool, or a content block of a
    message.

    Args:
        kwargs: The ``messages.create`` keyword arguments.

    Returns:
        ``True`` when at least one breakpoint is present.
    """
    extra_body = kwargs.get("extra_body")
    payload = {**kwargs, **extra_body} if isinstance(extra_body, dict) else kwargs
    return "cache_control" in payload or _count_cache_breakpoints(payload) > 0


def _count_cache_breakpoints(payload: dict[str, Any]) -> int:
    """Count the explicit ``cache_control`` markers in a request *payload*.

    Args:
        payload: The request fields (``system``, ``tools``, ``messages``).

    Returns:
        The number of system blocks, tools and message content blocks
        that carry a ``cache_control`` marker.
    """
    system = payload.get("system")
    blocks: list[Any] = list(system) if isinstance(system, list) else []
    blocks.extend(payload.get("tools") or [])
    for message in payload.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            blocks.extend(content)
    return sum(1 for block in blocks if isinstance(block, dict) and "cache_control" in block)


_BEDROCK_DEFAULT_REGION = "us-east-1"
# Request fields of the Anthropic API that Bedrock's InvokeModel rejects:
# automatic prompt caching and the priority service tier.
_BEDROCK_UNSUPPORTED_FIELDS = ("cache_control", "service_tier")
# Content blocks the API does not let carry a ``cache_control`` marker.
_UNCACHEABLE_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})
# Explicit breakpoints a request may carry; one more is a 400.
_MAX_CACHE_BREAKPOINTS = 4


def _with_conversation_cache_breakpoint(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the request messages with a ``cache_control`` breakpoint on the last block.

    Stands in for automatic prompt caching where the top-level
    ``cache_control`` is rejected (Bedrock's InvokeModel): the marker goes
    on the last cacheable block of the final message and moves forward
    with every turn, so the API reads the earlier turns from cache (it
    looks back about twenty blocks from a breakpoint for the longest
    cached prefix) and writes only the new blocks.  Without it only
    ``tools`` and the system prefix are cached and the whole conversation
    is billed at the full input rate on every turn.  Thinking blocks
    cannot carry the marker and are skipped; a final message that already
    carries one, or a request already at the breakpoint limit, is sent
    as is.  The final message and the marked block are copied, so the
    conversation the messages were built from keeps no marker and the
    next request re-sends the earlier turns byte for byte.

    Args:
        kwargs: The ``messages.create`` keyword arguments, with non-empty
            normalized ``messages``.

    Returns:
        The messages to send, sharing every unmarked element with the input.
    """
    messages: list[dict[str, Any]] = kwargs["messages"]
    last = messages[-1]
    blocks = _content_as_block_list(last["content"])
    if any("cache_control" in block for block in blocks):
        return messages
    if _count_cache_breakpoints(kwargs) >= _MAX_CACHE_BREAKPOINTS:
        return messages
    for index in range(len(blocks) - 1, -1, -1):
        if blocks[index].get("type") not in _UNCACHEABLE_BLOCK_TYPES:
            blocks[index] = {**blocks[index], "cache_control": {"type": "ephemeral"}}
            return [*messages[:-1], {**last, "content": blocks}]
    return messages


# Price modifiers the Anthropic API reports in the response ``usage``, as
# factors on the catalog's standard rates (they apply to every token
# category, cache tiers included, and stack with each other):
# https://platform.claude.com/docs/en/build-with-claude/fast-mode prices
# ``speed: "fast"`` at twice the standard rate (Opus 5.5 $8/$40 on $4/$20,
# Opus 5 and 4.8 $10/$50 on $5/$25) and reports the speed that served
# the call in ``usage.speed``;
# https://platform.claude.com/docs/en/manage-claude/data-residency prices
# US-only inference (``usage.inference_geo`` ``"us"``, which a workspace
# default can select without the request asking) at 1.1x on Claude 4.6
# and later, the only models that accept the parameter.  Neither applies
# on Bedrock, whose regional premium the catalog prices carry.
_FAST_MODE_PRICE_FACTOR = 2.0
_US_INFERENCE_GEO_PRICE_FACTOR = 1.1
# The usage field of each modifier and the value that raises the price.
PRICE_MODIFIER_FIELDS = {"speed": "fast", "inference_geo": "us"}


def price_modifier_factor(usage: Any, model_name: str) -> float:
    """Return the factor the speed and inference geo in *usage* apply to standard rates.

    ``speed`` ``"fast"`` applies :data:`_FAST_MODE_PRICE_FACTOR`;
    ``inference_geo`` ``"us"`` applies :data:`_US_INFERENCE_GEO_PRICE_FACTOR`
    on Claude 4.6 and later (earlier models bill standard whatever the
    field says).  The two stack.  The Claude Code CLI re-emits the API's
    usage as JSON, so :mod:`kiss.core.models.claude_code_model` shares
    this rule.

    Args:
        usage: The provider's usage record: an SDK object for the API
            transport, a decoded JSON dict for the CLI transport.
        model_name: The bare Anthropic model name the usage was billed
            under (no provider prefix; thinking aliases are tolerated).

    Returns:
        ``1.0`` unless *usage* reports a priced modifier.
    """
    factor = 1.0
    if _get_attr_or_key(usage, "speed") == PRICE_MODIFIER_FIELDS["speed"]:
        factor *= _FAST_MODE_PRICE_FACTOR
    if _get_attr_or_key(usage, "inference_geo") == PRICE_MODIFIER_FIELDS[
        "inference_geo"
    ] and _supports_inference_geo(model_name):
        factor *= _US_INFERENCE_GEO_PRICE_FACTOR
    return factor


def _supports_inference_geo(model_name: str) -> bool:
    """Return whether *model_name* is a Claude 4.6-or-later model.

    Args:
        model_name: The bare Anthropic model name (no provider prefix; a
            thinking alias such as ``claude-opus-5-5-high`` is stripped).

    Returns:
        ``True`` for Claude 4.6 and every later version (``claude-opus-4-6``,
        ``claude-sonnet-5``, ``claude-fable-5-1``, ...), ``False`` for
        earlier versions and names outside the ``claude-<family>-<major>``
        grammar.
    """
    from kiss.core.models.model_info import _strip_thinking_alias

    parsed = _parse_claude_version(_strip_thinking_alias(model_name))
    if parsed is None:
        return False
    _, major, minor = parsed
    return major > 4 or (major == 4 and (minor or 0) >= 6)


def _workspace_id() -> str:
    """Return the ``ANTHROPIC_WORKSPACE_ID`` an identity-linked key must send (``""`` if unset).

    Read from ``DEFAULT_CONFIG`` like the key it belongs to (the settings
    panel's key store, or the environment variable at startup).
    """
    from kiss.core.config import DEFAULT_CONFIG

    return DEFAULT_CONFIG.ANTHROPIC_WORKSPACE_ID.strip()


def _bedrock_credentials() -> tuple[str, str, str, str]:
    """Return the configured ``(access key, secret key, session token, region)``.

    Read from ``DEFAULT_CONFIG`` (the settings panel's key store, or the
    ``AWS_*`` environment variables); the region defaults to
    ``us-east-1`` when unset.
    """
    from kiss.core.config import DEFAULT_CONFIG

    return (
        DEFAULT_CONFIG.AWS_ACCESS_KEY_ID,
        DEFAULT_CONFIG.AWS_SECRET_ACCESS_KEY,
        DEFAULT_CONFIG.AWS_SESSION_TOKEN,
        DEFAULT_CONFIG.AWS_REGION or _BEDROCK_DEFAULT_REGION,
    )


def _key_pair_bedrock_client(
    credentials: Sequence[str], timeout: httpx2.Timeout
) -> AnthropicBedrock:
    """Build the Bedrock client that signs with *credentials*; run under the key store's lock.

    The constructor (and ``with_options``, which builds a copy through
    it) re-reads AWS_BEARER_TOKEN_BEDROCK from the environment when no
    api_key is passed and treats an EMPTY exported value as a token:
    it refuses the key pair next to it and sends ``Authorization:
    Bearer`` with nothing behind it.  An empty key is unset by KISS's
    own rules (``DEFAULT_CONFIG`` reads it as ""), so it is unset for
    the SDK too.  The check, the deletion and the constructor's read run
    under the key store's lock, which the settings panel holds while it
    exports a token it just saved: outside it, the empty value could be
    replaced by that token between the check and the deletion, and the
    new token deleted with it, or picked up by the constructor.
    """
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK") == "":
        del os.environ["AWS_BEARER_TOKEN_BEDROCK"]
    access_key, secret_key, session_token, region = credentials
    return AnthropicBedrock(
        aws_access_key=access_key or None,
        aws_secret_key=secret_key or None,
        aws_session_token=session_token or None,
        aws_region=region,
        timeout=timeout,
        max_retries=_MAX_RETRIES,
    )


def _bedrock_client(
    bearer_token: str, credentials: Sequence[str], timeout: httpx2.Timeout
) -> AnthropicBedrock:
    """Build the Anthropic SDK's Amazon Bedrock client.

    A bearer token (``AWS_BEARER_TOKEN_BEDROCK``) is the client's
    ``api_key``; the SDK refuses it together with an access key pair, so
    the pair is passed only when no token is set.  With neither, the
    client falls back to boto3's credential chain (``~/.aws``, an
    instance role).

    Args:
        bearer_token: The Bedrock API key (bearer token), or ``""``.
        credentials: The ``(access key, secret key, session token,
            region)`` the caller read (:func:`_bedrock_credentials`).
        timeout: The per-request httpx timeout.

    Returns:
        An ``anthropic.AnthropicBedrock`` instance (SigV4 signing with the
        key pair needs ``boto3``, which the SDK imports at request time).
    """
    if bearer_token:
        return AnthropicBedrock(
            api_key=bearer_token,
            aws_region=credentials[3],
            timeout=timeout,
            max_retries=_MAX_RETRIES,
        )
    return locked_call(user_config._config_lock, _key_pair_bedrock_client, credentials, timeout)


class AnthropicModel(Model):
    """A model that uses Anthropic's Messages API (Claude), directly or via Amazon Bedrock."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
        bedrock: bool = False,
    ):
        """Initialize an AnthropicModel instance.

        Args:
            model_name: The name of the Claude model to use; with
                ``bedrock=True`` a ``bedrock/<Bedrock model id>`` name.
            api_key: The Anthropic API key for authentication; with
                ``bedrock=True`` the Bedrock bearer token
                (``AWS_BEARER_TOKEN_BEDROCK``), empty to sign requests
                with the AWS access key pair instead.
            model_config: Optional dictionary of model configuration parameters.
            token_callback: Optional callback invoked with each streamed text token.
            thinking_callback: Optional callback invoked with ``True`` when a
                thinking block starts and ``False`` when it ends.
            bedrock: Route the requests through Amazon Bedrock (the SDK's
                ``AnthropicBedrock`` client) instead of the Anthropic API.
        """
        super().__init__(
            model_name,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
        self.api_key = api_key
        self.bedrock = bedrock
        # ``_api_model_name`` is the id sent as ``model=`` over the wire
        # and ``_catalog_model_name`` the ``claude-*`` catalog key the
        # capability checks (thinking, effort) and pricing use.  Direct: a
        # generated effort alias (``claude-opus-5-5-medium``, ``alias_of``
        # marker in the catalog) maps back to its base id, which is the
        # only name the Messages API knows (404 ``not_found_error``
        # otherwise).  Bedrock: the wire id is Bedrock's own
        # (``us.anthropic.claude-opus-4-6-v1:0``) and the catalog key is
        # the Claude model behind it.
        from kiss.core.models.model_info import (
            _strip_thinking_alias,
            bedrock_claude_name,
            bedrock_model_id,
        )

        if bedrock:
            self._api_model_name = bedrock_model_id(model_name)
            self._catalog_model_name = bedrock_claude_name(model_name) or model_name
        else:
            self._api_model_name = _strip_thinking_alias(model_name)
            self._catalog_model_name = model_name

    def _thinking_and_max_tokens(self) -> tuple[dict[str, Any] | None, int]:
        """Return the ``thinking`` parameter and the ``max_tokens`` of the requests.

        An explicit ``model_config["thinking"]`` goes out as is; a thinking
        model without one gets the default: adaptive thinking on the models
        that use it, otherwise a budget of up to 10000 tokens under
        ``max_tokens`` (none when fewer than 1024 remain).  ``max_tokens``
        is the configured one (``max_tokens`` or ``max_completion_tokens``),
        16384 by default, raised to the model's output cap when the default
        thinking is on.  ``None`` as the parameter means the request carries
        none.
        """
        max_tokens = self.model_config.get("max_tokens")
        if max_tokens is None:
            max_tokens = self.model_config.get("max_completion_tokens")
        user_set_max_tokens = max_tokens is not None
        if max_tokens is None:
            max_tokens = 16384
        if "thinking" in self.model_config:
            return self.model_config["thinking"], max_tokens
        profile = _claude_profile(self._catalog_model_name)
        if not _supports_extended_thinking(profile):
            return None, max_tokens
        if not user_set_max_tokens:
            version = profile[1]
            # Opus 4.6+ allows 65536 output tokens; Opus 4.5 (like
            # every Sonnet) is capped at 64000 and answers HTTP 400
            # "max_tokens: 65536 > 64000" otherwise (verified live
            # 2026-10-07).
            is_big_opus = (
                version is not None
                and version[0] == "opus"
                and (version[1], version[2] or 0) >= (4, 6)
            )
            max_tokens = 65536 if is_big_opus else 64000
        if _uses_adaptive_thinking(profile):
            return {"type": "adaptive", "display": "summarized"}, max_tokens
        budget = min(10000, max_tokens - 1)
        if budget >= 1024:
            return {"type": "enabled", "budget_tokens": budget}, max_tokens
        return None, max_tokens

    def _context_edits(self) -> list[dict[str, Any]]:
        """Return the server-side context edits the requests carry (:func:`context_edits`).

        The thinking edit only when the request's thinking is on (the API
        rejects it otherwise); nothing on Bedrock, which rejects the field.
        """
        if self.bedrock:
            return []
        thinking, _max_tokens = self._thinking_and_max_tokens()
        return context_edits((thinking or {}).get("type") in ("enabled", "adaptive"))

    def keeps_reasoning_turns(self) -> bool:
        """Return whether the requests clear the thinking of all but the last turns.

        True when the requests carry the thinking edit (:meth:`_context_edits`):
        on the Anthropic API, with ``Config.reasoning_keep_turns`` positive and
        the request's thinking on.  The agent then starts a new turn every
        ``Config.reasoning_turn_steps`` steps.
        """
        return bool(self._context_edits())

    def clears_tool_uses_in_bulk(self) -> bool:
        """Return whether the agent clears this model's old tool calls rarely and in bulk.

        True on the Anthropic API while ``Config.context_edit_trigger_tokens``
        is positive.  The requests read their prefix from the prompt cache,
        and an edit of an earlier message makes the provider write the rest
        of the conversation again, so the agent edits the stored conversation
        only once the context passes the trigger, and then all at once
        (:meth:`KISSAgent._maybe_clear_tool_uses`).  It does not ask the
        provider for its ``clear_tool_uses`` edit, whose boundary moved
        between requests (:func:`context_edits`).  Bedrock, which never had
        that edit, keeps the thresholds of the Compaction settings.
        """
        from kiss.core.config import DEFAULT_CONFIG

        return not self.bedrock and DEFAULT_CONFIG.context_edit_trigger_tokens > 0

    def keeps_newest_reasoning(self) -> bool:
        """Whether a clear leaves the newest reply's thinking: on a model before Claude 5.1.

        Such a model signs a thinking block over the block alone, and the
        API requires the blocks of the reply that the pending tool results
        answer.
        """
        return not _checks_thinking_prefix(_claude_profile(self._catalog_model_name))

    def drop_stale_reasoning(self) -> int:
        """Remove the old thinking blocks with the tool outputs an edit stubbed; return how many.

        The agent calls this after it stubbed old tool outputs and arguments
        in the conversation, so the reasoning goes in the same re-write of
        the prompt cache: kept, it is read again on every later call and
        raises the size the context falls back to after each clear (93k,
        100k, 139k, 163k in one session of 2026-11-11, which then handed
        off).  On a model that binds a thinking block to the messages
        before it (:func:`_checks_thinking_prefix`) every ``thinking`` and
        ``redacted_thinking`` block goes: the API accepts a conversation
        with all of them removed and the model reasons afresh on its next
        turn, whereas one stale block is a 400 on Claude Fable 5.2.  An
        earlier model keeps the blocks of its newest reply, which the API
        requires at the start of the assistant message that the pending
        tool results answer.
        """
        replies = [
            message for message in self.conversation
            if message.get("role") == "assistant" and isinstance(message.get("content"), list)
        ]
        if self.keeps_newest_reasoning():
            replies = replies[:-1]
        dropped = 0
        for message in replies:
            content = message["content"]
            kept = [
                block for block in content
                if not (isinstance(block, dict)
                        and block.get("type") in ("thinking", "redacted_thinking"))
            ]
            dropped += len(content) - len(kept)
            message["content"] = kept
        return dropped

    def _build_client(self, inputs: tuple[str, ...]) -> Anthropic | AnthropicBedrock:
        """Return a new SDK client for *inputs*, the settings :meth:`initialize` read.

        Its pool can be aborted (:func:`make_pool_abortable`), so the
        request watchdogs also end a request whose connection is still
        being set up: its endpoint's address not looked up yet, or its
        TCP or TLS handshake unanswered.

        Args:
            inputs: The API key, then the workspace id, or on Bedrock the
                AWS credentials and region (:func:`_bedrock_credentials`).
        """
        timeout = httpx2.Timeout(self._stream_stall_timeout, connect=CONNECT_TIMEOUT)
        api_key, *settings = inputs
        client: Anthropic | AnthropicBedrock
        if self.bedrock:
            client = _bedrock_client(api_key, settings, timeout)
        else:
            client = Anthropic(
                api_key=api_key,
                timeout=timeout,
                max_retries=_MAX_RETRIES,
                default_headers={"anthropic-workspace-id": settings[0]} if settings[0] else None,
            )
        make_pool_abortable(client._client)
        return client

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Initializes the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt to start the conversation.
            attachments: Optional list of file attachments (images, PDFs, audio,
                video) to include. Audio attachments are automatically
                transcribed to text via OpenAI Whisper when an ``OPENAI_API_KEY``
                is available; otherwise they are skipped with a warning.  Video
                attachments are always skipped.
        """
        # An identity-linked Anthropic API key is only valid together with
        # the id of the workspace the request acts in; the API rejects a
        # request without it (400: "anthropic-workspace-id is required when
        # authenticating with an identity-linked API key").  The id is the
        # ANTHROPIC_WORKSPACE_ID setting, which load_api_keys() imports
        # from ~/.kiss/api_keys.env (or the environment variable) at
        # daemon startup and ./rsorcar ships to remote installs.
        #
        # The client is built once and reused across runs; see
        # Model._reuse_client.  The settings are read once, and the client
        # built from that very reading: read again by the builder, a save
        # in between left one setting's client under the other's reuse key.
        # The key store's lock keeps a save from tearing the AWS fields.
        inputs = (
            (self.api_key, *locked_call(user_config._config_lock, _bedrock_credentials))
            if self.bedrock
            else (self.api_key, _workspace_id())
        )
        self._reuse_client(inputs, functools.partial(self._build_client, inputs))
        content: str | list[dict[str, Any]] = prompt
        if attachments:
            blocks = _attachments_to_blocks(attachments)
            blocks.append({"type": "text", "text": prompt})
            content = blocks
        self.conversation = [{"role": "user", "content": content}]

    def _normalize_content_blocks(self, content: Any) -> list[dict[str, Any]]:
        """Normalize Anthropic content blocks to JSON-serializable dicts.

        See :func:`normalize_content_blocks`: OpenAI media parts left by a
        conversation hand-off are converted, thinking blocks are kept.

        Args:
            content: The content blocks from an Anthropic response.

        Returns:
            list[dict[str, Any]]: Normalized content blocks as dictionaries.
        """
        return normalize_content_blocks(
            content,
            ("image_url", "file", "input_audio"),
            _openai_part_to_anthropic_block,
            keep_thinking=True,
        )

    def _build_anthropic_tools_schema(
        self,
        openai_schema: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert an OpenAI-format tools schema to Anthropic format.

        Args:
            openai_schema: Tool schema list in OpenAI format.

        Returns:
            list[dict[str, Any]]: A list of tool schemas in Anthropic format.
        """
        tools = []
        for tool in openai_schema:
            fn = tool.get("function", {})
            tools.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return tools

    def _normalize_conversation_for_api(
        self,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize all messages in a conversation before sending to the API.

        Ensures that all text content blocks are non-whitespace and that no
        messages contain only whitespace-only text blocks.  Also converts
        OpenAI Chat Completions-format entries (assistant ``tool_calls``
        arrays, ``role="tool"`` messages, ``image_url`` / ``file`` /
        ``input_audio`` content parts) — which enter the conversation when it
        is handed off from an OpenAI-schema model, e.g. via the Sorcar
        ``set_model`` tool — into the Anthropic Messages equivalents
        (``tool_use`` / ``tool_result`` / ``image`` / ``document`` blocks) so
        the API does not reject them.  OpenAI Responses-API items (handed
        off from an ``OpenAICompatibleModel2``) are converted first via
        :func:`responses_items_to_chat_messages`.  Consecutive user turns are
        merged so that ``tool_result`` blocks land in the message immediately
        following their ``tool_use`` turn, as the Anthropic API requires.

        Args:
            conversation: The conversation to normalize.

        Returns:
            list[dict[str, Any]]: The normalized conversation.
        """
        normalized: list[dict[str, Any]] = []
        for msg in responses_items_to_chat_messages(conversation):
            for converted in self._normalize_message_for_api(msg):
                prev = normalized[-1] if normalized else None
                if (
                    prev is not None
                    and prev.get("role") == "user"
                    and converted.get("role") == "user"
                ):
                    prev["content"] = _content_as_block_list(
                        prev["content"]
                    ) + _content_as_block_list(converted["content"])
                else:
                    normalized.append(converted)
        return normalized

    def _normalize_message_for_api(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize a single message into Anthropic Messages-format messages.

        A message may expand to zero messages (all content filtered out) or
        one message.  OpenAI Chat Completions-format entries — which enter
        the conversation when it is handed off from an OpenAI-schema model,
        e.g. via the Sorcar ``set_model`` tool — are converted to their
        Anthropic equivalents:

        * ``role="system"`` messages are dropped here;
          ``_build_create_kwargs`` hoists their text into the top-level
          ``system`` parameter (the Messages API rejects the "system" role).
        * ``role="tool"`` messages become user messages carrying a
          ``tool_result`` block.
        * assistant messages with ``tool_calls`` become assistant messages
          whose content is text + ``tool_use`` blocks.

        Args:
            msg: The conversation message to normalize.

        Returns:
            list[dict[str, Any]]: Anthropic Messages-format messages.
        """
        role = msg.get("role")
        if role == "system":
            return []
        if role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
            }
            content = msg.get("content")
            if isinstance(content, list):
                nested = self._normalize_content_blocks(content)
                if nested:
                    block["content"] = nested
            elif content is not None and str(content).strip():
                block["content"] = str(content)
            return [{"role": "user", "content": [block]}]

        msg_copy = msg.copy()
        attachments = msg_copy.pop("attachments", None)
        if attachments:
            att_blocks = _attachments_to_blocks(attachments)
            prior = msg_copy.get("content")
            if isinstance(prior, str):
                if prior.strip():
                    att_blocks.append({"type": "text", "text": prior})
            elif isinstance(prior, list):
                att_blocks.extend(prior)
            msg_copy["content"] = att_blocks
        content = msg_copy.get("content")
        tool_calls = msg_copy.pop("tool_calls", None)
        if tool_calls:
            blocks: list[dict[str, Any]] = []
            if isinstance(content, str):
                if content.strip():
                    blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                blocks.extend(self._normalize_content_blocks(content))
            blocks.extend(_tool_calls_to_tool_use_blocks(tool_calls))
            return [{"role": msg_copy.get("role", "assistant"), "content": blocks}]

        if isinstance(content, str):
            if content.strip():
                return [msg_copy]
            return []
        if isinstance(content, list):
            normalized_blocks = self._normalize_content_blocks(content)
            if normalized_blocks:
                msg_copy["content"] = normalized_blocks
                return [msg_copy]
            return []
        return []

    def _build_create_kwargs(self, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Build keyword arguments for the Anthropic API create call.

        Args:
            tools: Optional list of tool schemas to include.

        Returns:
            dict[str, Any]: The keyword arguments for the API call.
        """
        kwargs = {
            key: value
            for key, value in self.model_config.items()
            if key not in FRAMEWORK_ONLY_CONFIG_KEYS
        }
        # ``reasoning_effort`` is the framework-wide spelling (shared with
        # the OpenAI adapters); the Messages API wants it as
        # ``output_config.effort`` (precedence: see ``_effort_level_for``).
        kwargs.pop("reasoning_effort", None)
        effort_level = _effort_level_for(self._catalog_model_name, self.model_config)
        if effort_level is not None:
            output_config = kwargs.get("output_config")
            output_config = dict(output_config) if isinstance(output_config, dict) else {}
            output_config["effort"] = effort_level
            kwargs["output_config"] = output_config
        enable_cache = self.model_config.get("enable_cache", True)
        # The same hoisting Gemini applies (see Model docstring of
        # merge_system_texts): the Messages API rejects the "system" role,
        # so handed-off system messages ride in the ``system`` parameter.
        system_instruction = merge_system_texts(
            self.model_config.get("system_instruction"), self.conversation
        )

        _, stop_sequences = pop_max_tokens_and_stop(kwargs)
        if stop_sequences is not None:
            kwargs.setdefault("stop_sequences", stop_sequences)

        kwargs = self._keep_supported_request_params(
            kwargs,
            _ANTHROPIC_REQUEST_PARAMS,
            "Anthropic Messages",
        )

        thinking, max_tokens = self._thinking_and_max_tokens()
        if thinking is not None:
            kwargs["thinking"] = thinking

        if "thinking" in kwargs:
            _add_beta(kwargs, "interleaved-thinking-2025-05-14")
        # The keep-K reasoning policy, applied by the server.  Bedrock
        # rejects the field.
        edits = self._context_edits()
        if edits:
            _add_beta(kwargs, CONTEXT_MANAGEMENT_BETA)
            body: dict[str, Any] = dict(kwargs.get("extra_body") or {})
            body["context_management"] = {"edits": edits}
            kwargs["extra_body"] = body

        normalized_messages = checked_nonempty(
            self._normalize_conversation_for_api(self.conversation)
        )
        kwargs.update(
            {
                "model": self._api_model_name,
                "messages": normalized_messages,
                "max_tokens": max_tokens,
            }
        )
        if system_instruction:
            kwargs["system"] = _system_param(system_instruction, enable_cache)
        if tools:
            kwargs["tools"] = tools
            # Without fine-grained tool streaming the API buffers each tool
            # parameter until the model has finished writing it, so a large
            # argument (a long ``finish`` summary, a ``Write`` of a big
            # file) streams no event for minutes and trips the
            # ``stream_stall_timeout`` watchdog on every retry.  The header,
            # not the per-tool ``eager_input_streaming`` field, because
            # Bedrock's Claude Sonnet 4.5 rejects the field.
            _add_beta(kwargs, "fine-grained-tool-streaming-2025-05-14")
            if "tool_choice" not in kwargs and "thinking" not in kwargs:
                kwargs["tool_choice"] = {"type": "any"}

        # Top-level ``cache_control`` (automatic prompt caching) and
        # ``service_tier`` (priority tier) exist only on the Anthropic
        # API: Bedrock's InvokeModel rejects unknown request fields, and
        # caches through explicit breakpoints only: the one
        # ``_system_param`` sets and the moving one on the conversation's
        # last block, which stands in for automatic caching.  Both fields
        # are removed wherever model_config may carry them: as a kwarg
        # and inside ``extra_body``, which the SDK merges into the
        # request JSON verbatim.
        if enable_cache and not self.bedrock:
            kwargs.setdefault("cache_control", {"type": "ephemeral"})
        if self.bedrock:
            if enable_cache:
                kwargs["messages"] = _with_conversation_cache_breakpoint(kwargs)
            for field in _BEDROCK_UNSUPPORTED_FIELDS:
                kwargs.pop(field, None)
            extra_body = kwargs.get("extra_body")
            if isinstance(extra_body, dict):
                kwargs["extra_body"] = {
                    key: value
                    for key, value in extra_body.items()
                    if key not in _BEDROCK_UNSUPPORTED_FIELDS
                }

        return kwargs

    def _create_message(self, kwargs: dict[str, Any]) -> Any:
        """Create a message, streaming tokens to the callback when set.

        Aborts with a clear, retryable :class:`TimeoutError` when the
        stream stalls for :attr:`_stream_stall_timeout` seconds at either
        level:

        * **byte level** — the client's httpx2 read timeout raises
          ``httpx2.ReadTimeout`` mid-iteration (SDK does not retry it, and
          its message is often empty) or ``anthropic.APITimeoutError``
          when the response headers never arrive (SDK retries
          ``_MAX_RETRIES`` times first);
        * **event level** — the watchdog inside
          :func:`~kiss.core.models.stream_abort.stop_aware_events` aborts
          the response when no SSE event is yielded in time, catching wedged
          requests that keep the connection alive with ``ping`` events
          (which the SDK filters out before yielding).

        ``KISSAgent._run_agentic_loop`` treats ``TimeoutError`` as
        retryable and re-asks the model instead of hanging forever.  An
        open thinking bracket is closed on EVERY exit — stop, stall and
        any other transport failure alike, exactly as the OpenAI and
        Gemini transports do — because the agentic loop retries a
        non-``KISSError`` in the same run without resetting the printer,
        which would otherwise render the retry's answer as "thinking".

        Args:
            kwargs: Keyword arguments for the Anthropic API call.

        Returns:
            The raw Anthropic response message.

        Raises:
            TimeoutError: When the streaming connection delivers no data
                (or no events) for ``stream_stall_timeout`` seconds.
        """
        # Until the response headers are complete there is no response
        # for the stream watchdog to cut, and the byte-level clock only
        # bounds the silence between two reads: a peer trickling header
        # bytes (or 1xx replies) would hold this thread for good and
        # past any Stop.  So the request is sent under a request
        # watchdog that cuts the pool at a Stop or at this deadline.
        headers_due = time.monotonic() + (
            self._stream_stall_timeout * (_MAX_RETRIES + 1) + _RETRY_BACKOFF_SECONDS
        )
        stream = None
        try:
            with contextlib.ExitStack() as stack:
                with RequestAbortWatchdog(
                    self.client._client,
                    headers_due - time.monotonic(),
                    "anthropic-header-abort-watchdog",
                ):
                    try:
                        stream = stack.enter_context(self.client.messages.stream(**kwargs))
                    except UnicodeEncodeError:
                        # The body could not be encoded, so nothing was
                        # sent (see _well_formed).
                        stream = stack.enter_context(
                            self.client.messages.stream(**_well_formed(kwargs))
                        )
                events = stack.enter_context(
                    self._watched_events(stream, "anthropic-stream-abort-watchdog")
                )
                final: Any = None
                raw_tool_inputs: dict[int, str] = {}
                for event in events:
                    # The SDK accumulates message_start / message_delta
                    # usage here; kept for take_partial_usage_response
                    # if the stream fails before message_stop.
                    self._rejected_response = stream.current_message_snapshot
                    if event.type == "message_stop":
                        # The terminal event: the snapshot is the whole
                        # message, so nothing after it is read.  Reading
                        # on to the end of the body let a connection that
                        # went silent here trip the stall watchdog, and
                        # the retry billed the finished turn again.
                        # The OpenAI transports stop at their terminal
                        # event in the same way.  The price: a body not
                        # read to its end closes the connection instead
                        # of returning it to the pool, so the next
                        # request opens a new one.
                        final = stream.current_message_snapshot
                        break
                    if event.type == "message_delta":
                        # The server's context edits ride on the final delta.
                        self.last_context_edits = applied_context_edits(event)
                    elif (
                        event.type == "content_block_delta"
                        and event.delta.type == "input_json_delta"
                    ):
                        raw_tool_inputs[event.index] = (
                            raw_tool_inputs.get(event.index, "") + event.delta.partial_json
                        )
                    if self.token_callback is None:
                        continue
                    # A thinking_delta only ever arrives inside a thinking
                    # block, and the bracket helpers are idempotent on
                    # _thinking_open, so no per-block flag is needed.
                    if event.type == "content_block_delta":
                        delta = event.delta
                        delta_type = getattr(delta, "type", "")
                        if delta_type == "thinking_delta":
                            text = getattr(delta, "thinking", "")
                            if text:
                                self._open_thinking_if_closed()
                                self._invoke_token_callback(text)
                        elif delta_type == "text_delta":
                            self._invoke_token_callback(getattr(delta, "text", ""))
                    elif event.type == "content_block_stop":
                        self._close_thinking_if_open()
                if final is None:
                    # The body ended (EOF) before the message's terminal
                    # event: a gateway can close the stream cleanly before
                    # the model is done, and the SDK's snapshot then holds
                    # a partial parse (a tool input missing its open
                    # string, a cut number) with no stop_reason to tell.
                    # The same plain ``KISSError`` the OpenAI transports
                    # raise for a stream without its terminal event; the
                    # usage seen so far stays in ``_rejected_response``
                    # for ``take_partial_usage_response``.
                    raise KISSError(
                        "Anthropic stream ended without a message_stop event, so its "
                        "text and any tool-call arguments may be incomplete."
                    )
                _check_streamed_tool_inputs(raw_tool_inputs, final, self.model_name)
                self._rejected_response = None
                return final
        except (httpx2.TimeoutException, APITimeoutError, TimeoutError) as exc:
            # The byte-level clock (httpx2 / the SDK, which is the only
            # clock BEFORE the headers arrive) or the event-level one
            # (`stop_aware_events`' stall error).  A stop pressed while
            # the request was still silent has no watchdog to report it
            # and must not be retried as a stall.  A stall names the
            # model: it is usually diagnosed from a log line that does
            # not say which model was being asked.
            raise stop_or_stall_error(
                self._stream_stall_timeout, f"Anthropic model {self.model_name}"
            ) from exc
        except Exception as exc:
            if stop_signal.stop_requested():
                raise stop_error() from exc
            if stream is None and time.monotonic() >= headers_due:
                # The header watchdog cut a request whose headers never
                # completed: a stall, reported like a silent one.
                raise stop_or_stall_error(
                    self._stream_stall_timeout, f"Anthropic model {self.model_name}"
                ) from exc
            if _WORKSPACE_ID_REQUIRED_MARKER in str(exc):
                # The API's own message names the missing header but not
                # where KISS reads it from; replace it with an actionable
                # error.  Raising KISSError makes the agentic loop
                # (KISSAgent._run_agentic_loop's ``except KISSError``
                # branch) re-raise immediately, so the run fails with the
                # hint instead of retrying a request that can never
                # succeed.
                raise KISSError(_WORKSPACE_ID_HINT.format(error=exc)) from exc
            raise

    def _raise_on_refusal(self, response: Any) -> None:
        """Raise :class:`ModelRefusalError` when the model refused the request.

        Adaptive-thinking Claude models (fable-5 in production, task
        ``daa89a7e``/``c3cd9c95`` in ``~/.kiss/history.db``) can return
        ``stop_reason="refusal"`` with an EMPTY ``content`` list when their
        safety layer declines an otherwise benign prompt (observed on
        security-research text that opus-4-8 answers normally).  Without
        this check the empty turn propagated as ``("", [])``, KISSAgent
        burned a useless "MUST have at least one function call" retry (a
        refusal is deterministic for identical content), and the eventual
        fallback swap was misreported as "repeated empty responses" — a
        misleading adapter-bug diagnosis.

        Args:
            response: The raw Anthropic response message.

        Raises:
            ModelRefusalError: When ``response.stop_reason`` is ``"refusal"``.
        """
        if getattr(response, "stop_reason", None) == "refusal":
            # The refused request's input was still billed.
            self._rejected_response = response
            raise ModelRefusalError(
                f"Model {self.model_name} refused the request for safety "
                f'reasons (stop_reason="refusal", empty response). Retrying '
                f"the identical request will keep failing; rephrase the "
                f"prompt or use a different model."
            )

    def generate_and_process_with_tools(  # pragma: no cover – API call
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generates content with tools and processes the response.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Optional pre-built OpenAI-format tool schema list.

        Returns:
            tuple[list[dict[str, Any]], str, Any]: A tuple of
                (function_calls, response_text, raw_response).
        """
        resolved = self._resolve_openai_tools_schema(function_map, tools_schema)
        tools = self._build_anthropic_tools_schema(resolved)
        kwargs = self._build_create_kwargs(tools=tools or None)
        response = self._create_message(kwargs)
        self._raise_on_refusal(response)

        stop_reason = getattr(response, "stop_reason", None)
        raw_blocks = list(getattr(response, "content", None) or [])
        if (
            stop_reason == "max_tokens"
            and raw_blocks
            and _get_attr_or_key(raw_blocks[-1], "type") == "tool_use"
        ):
            # The cap stopped generation inside the last block, whose
            # ``input`` is a partial parse; every tool_use block before it
            # was completed before the model moved on.  Decided on the raw
            # content: normalization drops an empty or whitespace-only text
            # block, so a completed tool followed by one the cap landed in
            # would otherwise be taken for the cut-off block.
            raw_blocks.pop()
        blocks = self._normalize_content_blocks(raw_blocks)

        if (
            stop_reason == "max_tokens"
            and tools
            and not any(b.get("type") == "tool_use" for b in blocks)
        ):
            # A tool-bearing turn with no tool call left is unusable
            # (a tool-less caller may still read the cut-off text).
            # The cut-off request was still billed.
            self._rejected_response = response
            raise OutputCapError(
                f"Model {self.model_name} hit its output-token cap "
                f'(stop_reason="max_tokens") before completing a tool call, so '
                f"its text and any tool-call arguments are incomplete. Raise "
                f"max_tokens or lower reasoning_effort.",
                stop_reason="max_tokens",
            )

        content = text_of_parts(blocks)

        function_calls: list[dict[str, Any]] = []
        for b in blocks:
            if b.get("type") == "tool_use":
                function_calls.append(
                    {
                        "id": b.get("id", ""),
                        "name": b.get("name", ""),
                        "arguments": b.get("input", {}) or {},
                    }
                )

        if blocks:
            self.conversation.append({"role": "assistant", "content": blocks})
        return function_calls, content, response

    def keep_prompt_cache_warm(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Re-read the cached prefix so its 5-minute TTL restarts.

        Sends the request the last step sent, byte-for-byte in the parts
        the cache is keyed on (tools, system, thinking, ``tool_choice``,
        messages), plus one user message that answers every in-flight
        ``tool_use`` with a placeholder ``tool_result`` (the API rejects an
        assistant tool-use turn with nothing after it).  Only
        ``max_tokens`` shrinks, which is not part of the cached prefix.
        The reply is discarded: the conversation is not touched, so the
        real tool results are appended exactly as without the ping.

        Args:
            function_map: The agent's tools, as passed to
                :meth:`generate_and_process_with_tools`.
            tools_schema: The pre-built tool schema list, as passed to
                :meth:`generate_and_process_with_tools`.

        Returns:
            The raw Anthropic message of the ping, for cost accounting, or
            ``None`` when there is nothing to keep warm: prompt caching is
            disabled for this model (``model_config["enable_cache"]``
            false) or the request as it goes on the wire carries no
            cache breakpoint (an ``extra_body`` override replaced the
            field that held it).
        """
        if not self.model_config.get("enable_cache", True):
            return None
        resolved = self._resolve_openai_tools_schema(function_map, tools_schema)
        tools = self._build_anthropic_tools_schema(resolved)
        kwargs = self._build_create_kwargs(tools=tools or None)
        if not _has_cache_breakpoint(kwargs):
            return None
        pending = _last_assistant_tool_use_ids(kwargs["messages"])
        if pending:
            content: str | list[dict[str, Any]] = [
                {"type": "tool_result", "tool_use_id": call_id, "content": KEEP_ALIVE_TEXT}
                for call_id in pending
            ]
        else:
            content = KEEP_ALIVE_TEXT
        kwargs["messages"] = [*kwargs["messages"], {"role": "user", "content": content}]
        thinking = kwargs.get("thinking") or {}
        budget = thinking.get("budget_tokens", 0) if thinking.get("type") == "enabled" else 0
        kwargs["max_tokens"] = max(KEEP_ALIVE_MAX_TOKENS, budget + 1)
        # ``output_format`` (a type the SDK parses the reply into) is a
        # parameter of ``messages.stream`` only, which sends it as
        # ``output_config.format``.  ``messages.create`` has no such
        # parameter (TypeError on every ping), and the format is part of
        # what the cache is keyed on, so the ping sends the same field.
        output_format = kwargs.pop("output_format", None)
        if output_format is not None:
            schema = transform_schema(TypeAdapter(output_format).json_schema())
            kwargs["output_config"] = {
                **kwargs.get("output_config", {}),
                "format": {"type": "json_schema", "schema": schema},
            }
        # The ping's own bound replaces a per-request ``timeout`` the
        # model_config may carry (the SDK accepts that key, so it survives
        # into ``kwargs``): passed as a second keyword it would raise
        # TypeError on every ping and the cache would silently go cold.
        kwargs["timeout"] = KEEP_ALIVE_TIMEOUT_SECONDS
        # No SDK retries: a failed ping is just a cold cache, and the agent
        # thread waits for this request when the tool call ends.  The SDK
        # timeout only bounds the silence between bytes, so the watchdog
        # gives the whole request the same deadline and aborts it at once
        # when the task is stopped (the keep-alive thread carries the
        # task's stop event).  Nothing else sends through the client while
        # a ping is in flight: the agent thread is inside the tool call.
        create = self.client.with_options(max_retries=0).messages.create
        with RequestAbortWatchdog(
            self.client._client, KEEP_ALIVE_TIMEOUT_SECONDS, "anthropic-keepalive-abort-watchdog"
        ):
            try:
                return create(**kwargs)
            except UnicodeEncodeError:  # nothing was sent, as in _create_message
                return create(**_well_formed(kwargs))

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Add tool results to the conversation as ``tool_result`` blocks.

        Anthropic is the one provider whose tool result may carry the bytes
        directly, so binary payloads a tool produced become image blocks
        inside the ``tool_result`` instead of a follow-up user message.

        Args:
            function_results: List of (func_name, result_dict) tuples.
                result_dict can contain:
                - "result": The result content (a string, or any
                  JSON-encodable value)
                - "tool_use_id": Optional explicit tool_use_id to use
        """
        tool_call_ids = self._find_tool_call_ids_from_last_assistant()

        tool_results_blocks: list[dict[str, Any]] = []
        for i, (func_name, result_dict) in enumerate(function_results):
            text, attachments = self.tool_result_text_and_attachments(result_dict)
            tool_use_id = self.tool_result_call_id(
                result_dict, i, tool_call_ids, func_name, prefix="toolu"
            )
            content: str | list[dict[str, Any]] = text
            if attachments:
                blocks: list[dict[str, Any]] = []
                if text.strip():
                    blocks.append({"type": "text", "text": text})
                blocks.extend(_attachments_to_blocks(attachments))
                content = blocks or text
            tool_results_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                }
            )

        self.conversation.append({"role": "user", "content": tool_results_blocks})

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int, int]:
        """Extracts token counts from an Anthropic API response.

        Returns:
            (input_tokens, output_tokens, cache_read_tokens,
            cache_write_5m_tokens, cache_write_1h_tokens).
        """
        if hasattr(response, "usage") and response.usage:
            cache_write_5m, cache_write_1h = cache_creation_tokens(response.usage)
            return (
                getattr(response.usage, "input_tokens", 0) or 0,
                getattr(response.usage, "output_tokens", 0) or 0,
                getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                cache_write_5m,
                cache_write_1h,
            )
        return 0, 0, 0, 0, 0

    def cost_multiplier_for_response(self, response: Any) -> float:
        """Return the factor the speed and inference geo that served *response* apply.

        The Anthropic API reports both in the response ``usage``:
        ``speed`` is ``"fast"`` when fast mode served the call
        (:data:`_FAST_MODE_PRICE_FACTOR`), and ``inference_geo`` is
        ``"us"`` when inference was pinned to US infrastructure, by the
        request or by the workspace default
        (:data:`_US_INFERENCE_GEO_PRICE_FACTOR`, Claude 4.6 and later only;
        earlier models bill standard whatever the field says).  The two
        stack.  Bedrock responses carry neither modifier and its regional
        premium is already in the catalog prices, so they bill at 1.0.

        Args:
            response: The final message (or the usage-bearing snapshot of
                a turn that raised) returned by the SDK.

        Returns:
            The factor to apply to the standard-rate catalog estimate.
        """
        usage = getattr(response, "usage", None)
        if usage is None or self.bedrock:
            return 1.0
        return price_modifier_factor(usage, self._api_model_name)

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generates an embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name (not used by Anthropic).

        Raises:
            KISSError: Anthropic does not provide an embeddings API.
        """
        raise KISSError("Anthropic does not provide an embeddings API.")
