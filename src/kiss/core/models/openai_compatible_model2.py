# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""OpenAI-compatible model targeting the **Responses API** (``/v1/responses``).

This is a sibling of :class:`OpenAICompatibleModel` (Chat Completions).
The Responses API natively supports the combination of ``tools`` and
``reasoning.effort`` — including the ``"xhigh"`` level used by the gpt-5.5
family — that ``/v1/chat/completions`` rejects for GPT-5.x/o-series
reasoning models.  The v2 model exists so the agentic loop can keep
sending tools without having to strip ``reasoning_effort``.

The public surface mirrors v1 (``initialize``, ``generate``,
``generate_and_process_with_tools``,
``add_function_results_to_conversation_and_return``,
``extract_input_output_token_counts_from_response``, ``get_embedding``)
so callers can swap implementations without changing call sites.
"""

import base64
import json
import logging
from collections.abc import Callable
from typing import Any

from openai.resources.responses import Responses
from openai.types.responses import response_create_params

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import (
    FRAMEWORK_ONLY_CONFIG_KEYS,
    Attachment,
    ThinkingCallback,
    TokenCallback,
    _audio_mime_to_format,
    _build_text_based_tools_prompt,
    _parse_text_based_tool_calls,
    accepted_request_params,
)
from kiss.core.models.openai_compatible_model import (
    OPENAI_INPUT_AUDIO_FORMATS,
    OPENAI_INPUT_IMAGE_MIME_TYPES,
    OpenAICompatibleBase,
    OpenAICompatibleModel,
    _extract_deepseek_reasoning,
)
from kiss.core.models.stream_abort import stop_aware_events

logger = logging.getLogger(__name__)

# The request parameters a Responses call accepts, taken from the SDK's own
# signature (keyword-only, no **kwargs), and the sub-keys its
# ``stream_options`` declares (the Chat Completions shape differs).
_RESPONSES_REQUEST_PARAMS = accepted_request_params(Responses.create)
_RESPONSES_STREAM_OPTION_KEYS = frozenset(
    response_create_params.StreamOptions.__annotations__
)


class OpenAICompatibleModel2(OpenAICompatibleBase):
    """OpenAI-compatible model that targets the Responses API.

    The only behavioural difference vs :class:`OpenAICompatibleModel` is the
    transport: every text generation goes through ``/v1/responses`` and the
    parameters are reshaped accordingly (``reasoning.effort`` instead of
    ``reasoning_effort``, flat tool schemas, ``instructions`` instead of a
    ``system`` message, ``input_text``/``input_image``/``input_file``
    content parts, ``function_call_output`` items for tool results).
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
        """Initialize a Responses-API-targeting OpenAI-compatible model.

        Args:
            model_name: The model id (optionally prefixed by ``openrouter/``).
            base_url: API endpoint root, e.g. ``https://api.openai.com/v1``.
            api_key: Bearer token for the endpoint.
            model_config: Optional model parameters.  Recognised keys
                mirror v1: ``reasoning_effort``, ``system_instruction``,
                ``extra_headers``, ``enable_cache``, plus any other key
                forwarded to ``responses.create(...)``.
            token_callback: Called with each streamed text token.
            thinking_callback: Called with ``True`` when a reasoning block
                starts and ``False`` when it ends.
        """
        super().__init__(
            model_name,
            base_url,
            api_key,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
        self._pending_function_calls: list[dict[str, str]] = []
        self._last_stream_item_indexes: dict[str, int] = {}
        self._last_stream_message_output_index: int | None = None

    def initialize(
        self, prompt: str, attachments: list[Attachment] | None = None
    ) -> None:
        """Initialize the conversation with ``prompt`` and any ``attachments``.

        The ``system_instruction`` (if any) is captured separately and
        forwarded to the Responses API via the top-level ``instructions``
        argument — it is never injected as a message.

        Args:
            prompt: The initial user prompt.
            attachments: Optional list of image / PDF / audio attachments
                to attach to the initial user message.
        """
        self._ensure_client()
        self.conversation = []
        self._reset_turn_state()
        if attachments:
            parts = self._attachments_to_content_parts(attachments)
            parts.append({"type": "input_text", "text": prompt})
            self.conversation.append({"role": "user", "content": parts})
        else:
            self.conversation.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            )

    def _reset_stream_indexes(self) -> None:
        """Forget the item ordering learned from the previous streamed turn."""
        self._last_stream_item_indexes = {}
        self._last_stream_message_output_index = None

    def _reset_turn_state(self) -> None:
        """Drop every scrap of per-turn state left by the previous turn."""
        self._pending_function_calls = []
        self._reset_stream_indexes()

    def reset_conversation(self) -> None:
        """Reset the conversation together with all per-turn state.

        The base implementation clears only the conversation.  Leaving
        ``_pending_function_calls`` behind would then reject the next
        generation on a *fresh, empty* conversation — a permanently
        unusable model object — and leaving the stream indexes behind
        would order a later non-streamed turn's assistant message using
        the previous streamed turn's indexes.
        """
        super().reset_conversation()
        self._reset_turn_state()

    def _request_response(
        self, kwargs: dict[str, Any]
    ) -> tuple[str, list[dict[str, str]], Any]:
        """Issue one Responses request, streaming iff a token callback is set.

        The single place the transport talks to ``responses.create``, so
        the per-turn stream bookkeeping is reset for *every* turn — a
        non-streamed turn must never be ordered using the indexes of the
        streamed turn before it.

        Args:
            kwargs: The request kwargs; ``stream`` is set here when
                streaming.

        Returns:
            ``(content, tool_calls, response)``.
        """
        self._reset_stream_indexes()
        if self.token_callback is not None:
            kwargs["stream"] = True
            return self._consume_stream(self.client.responses.create(**kwargs))
        response = self.client.responses.create(**kwargs)
        self._raise_for_failed_response(response)
        content, tool_calls = self._parse_non_streaming(response)
        return content, tool_calls, response

    @staticmethod
    def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
        """Return ``obj.key`` or ``obj[key]``, falling back to ``default``.

        This helper supports both pydantic SDK model instances (which use
        attribute access) and plain ``dict``-shaped responses returned by
        some OpenAI-compatible gateways or recorded JSON payloads.

        Args:
            obj: Object or mapping to read from.
            key: Attribute or mapping key to look up.
            default: Value returned when ``key`` is absent.

        Returns:
            The looked-up value or ``default``.
        """
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _attachment_to_content_part(att: Attachment) -> dict[str, Any] | None:
        """Convert a single :class:`Attachment` to a Responses-API content part.

        Args:
            att: The attachment to convert.

        Returns:
            A Responses-API content part dict (``input_image`` /
            ``input_file``) or ``None`` if the MIME type is unsupported by
            the Responses API (e.g. video).
        """
        if att.mime_type.startswith("image/"):
            if att.mime_type not in OPENAI_INPUT_IMAGE_MIME_TYPES:
                logger.warning(
                    "OpenAI Responses API does not support %s image "
                    "attachments; dropping.",
                    att.mime_type,
                )
                return None
            return {
                "type": "input_image",
                "image_url": att.to_data_url(),
                "detail": "auto",
            }
        if att.mime_type == "application/pdf":
            return {
                "type": "input_file",
                "filename": "attachment.pdf",
                "file_data": att.to_data_url(),
            }
        if att.mime_type.startswith("audio/"):
            fmt = _audio_mime_to_format(att.mime_type)
            if fmt not in OPENAI_INPUT_AUDIO_FORMATS:
                logger.warning(
                    "OpenAI Responses API does not accept %s audio "
                    "(supported: %s); skipping attachment.",
                    att.mime_type,
                    sorted(OPENAI_INPUT_AUDIO_FORMATS),
                )
                return None
            return {
                "type": "input_audio",
                "input_audio": {"data": att.to_base64(), "format": fmt},
            }
        logger.warning(
            "OpenAI Responses API does not support %s attachments; skipping.",
            att.mime_type,
        )
        return None

    @classmethod
    def _attachments_to_content_parts(
        cls, attachments: list[Attachment]
    ) -> list[dict[str, Any]]:
        """Convert ``attachments`` to a list of Responses-API content parts.

        Unsupported MIME types are skipped (with a warning).

        Args:
            attachments: Attachments to convert.

        Returns:
            A list of content parts; possibly shorter than ``attachments``
            if some MIME types were unsupported.
        """
        parts: list[dict[str, Any]] = []
        for att in attachments:
            part = cls._attachment_to_content_part(att)
            if part is not None:
                parts.append(part)
        return parts

    def _consume_pending_call_id(
        self,
        func_name: str,
        i: int,
        trailing: list[tuple[str, str]],
        fallback_unanswered: list[dict[str, str]] | None = None,
    ) -> str:
        """Return the next call_id matching ``func_name`` from pending state.

        Prefers the FIFO ``self._pending_function_calls`` queue (which
        survives incremental result submissions).  Matches by name first
        — falls back to popping the head of the queue if no name matches
        — finally falls back to the contiguous trailing run, and last
        resort to a synthesised id.

        Args:
            func_name: Name of the function whose result is being added.
            i: Index of this result in the current
                ``function_results`` batch.
            trailing: Snapshot of the contiguous trailing
                ``function_call`` items.

        Returns:
            The model's original ``call_id`` if known, otherwise a
            synthesised fallback ``f"call_{func_name}_{i}"``.
        """
        if self._pending_function_calls:
            for j, pending in enumerate(self._pending_function_calls):
                if pending.get("name") == func_name:
                    call_id = self._pending_function_calls.pop(j)["call_id"]
                    if not call_id:
                        raise KISSError(
                            f"Pending function_call for {func_name!r} has "
                            "empty call_id"
                        )
                    return call_id
            pending_names = [
                p.get("name", "") for p in self._pending_function_calls
            ]
            raise KISSError(
                f"No pending function_call named {func_name!r}; "
                f"pending calls are {pending_names!r}"
            )
        if fallback_unanswered is not None:
            for j, pending in enumerate(fallback_unanswered):
                if pending.get("name") == func_name:
                    call_id = fallback_unanswered.pop(j).get("call_id", "")
                    if call_id:
                        return call_id
            if fallback_unanswered:
                fallback_names = [
                    p.get("name", "") for p in fallback_unanswered
                ]
                raise KISSError(
                    f"No unanswered function_call named {func_name!r}; "
                    f"unanswered calls are {fallback_names!r}"
                )
        if i < len(trailing) and trailing[i][1]:
            trailing_name, call_id = trailing[i]
            if trailing_name and trailing_name != func_name:
                raise KISSError(
                    f"Trailing function_call mismatch for result "
                    f"{func_name!r}; trailing call at index {i} is "
                    f"{trailing_name!r} with call_id {call_id!r}."
                )
            return call_id
        raise KISSError(
            f"No prior function_call found for tool result {func_name!r}; "
            "cannot append function_call_output without a matching call_id."
        )

    def _trailing_function_call_ids(self) -> list[tuple[str, str]]:
        """Return ``(name, call_id)`` pairs for the trailing function_call run.

        Walks the conversation from the end backwards, collecting every
        contiguous ``function_call`` item.  Stops at the first item that is
        not a ``function_call`` (typically the assistant text message or a
        user message).  Returns the calls in original (forward) order so
        callers can match them by index against ``function_results``.

        Returns:
            A list of ``(name, call_id)`` tuples, possibly empty.
        """
        collected: list[tuple[str, str]] = []
        for item in reversed(self.conversation):
            if isinstance(item, dict) and item.get("type") == "function_call":
                collected.append((item.get("name", ""), item.get("call_id", "")))
                continue
            break
        collected.reverse()
        return collected

    @staticmethod
    def _flatten_tools_schema(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Flatten Chat-Completions tool schemas to Responses-API shape.

        Chat Completions wraps each function tool as
        ``{"type":"function","function":{"name":...,...}}``.  The Responses
        API expects the function fields to live directly on the tool dict
        (``{"type":"function","name":...,"description":...,"parameters":...}``).

        Tools that already have the flat shape (no ``function`` key) are
        passed through unchanged.

        Args:
            tools: A list of tool schema dicts in either Chat-Completions
                or Responses-API shape.

        Returns:
            A new list of tool schema dicts in Responses-API shape.
        """
        flat: list[dict[str, Any]] = []
        for tool in tools:
            if (
                isinstance(tool, dict)
                and tool.get("type") == "function"
                and isinstance(tool.get("function"), dict)
            ):
                base = {k: v for k, v in tool.items() if k != "function"}
                base.update(tool["function"])
                base["type"] = "function"
                flat.append(base)
            else:
                flat.append(tool)
        return flat

    @staticmethod
    def _flatten_tool_choice(tool_choice: Any) -> Any:
        """Flatten a Chat-Completions ``tool_choice`` to Responses-API shape.

        Chat Completions targets a specific function with:
        ``{"type":"function","function":{"name":"foo"}}``.  The Responses
        API expects the flat form: ``{"type":"function","name":"foo"}``.

        Strings (``"auto"`` / ``"none"`` / ``"required"``) and any other
        already-flat shape are passed through unchanged.

        Args:
            tool_choice: Either a Chat-Completions or Responses-API
                ``tool_choice`` value.

        Returns:
            The same value, with the nested ``function`` key hoisted onto
            the top level when present.
        """
        if (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "function"
            and isinstance(tool_choice.get("function"), dict)
        ):
            out = {k: v for k, v in tool_choice.items() if k != "function"}
            out.update(tool_choice["function"])
            out["type"] = "function"
            return out
        return tool_choice

    @staticmethod
    def _normalize_input(conversation: list[Any]) -> list[Any]:
        """Drop empty/whitespace-only items from a Responses ``input`` array.

        Both Chat Completions and Responses APIs reject empty or
        whitespace-only text content with ``invalid_request_error``.
        This helper filters:

        * Whole messages whose string ``content`` is empty / whitespace.
        * ``input_text`` / ``output_text`` content parts whose ``text`` is
          empty / whitespace.
        * Messages whose entire content list collapses to nothing.

        Standalone non-message items (``function_call``,
        ``function_call_output``) are passed through verbatim.

        Args:
            conversation: The raw conversation array.

        Returns:
            A new list containing only items safe to send to the API.
        """
        out: list[Any] = []
        for item in conversation:
            if not isinstance(item, dict):
                out.append(item)
                continue
            if item.get("role") is None:
                if item.get("type") == "_kiss_pending_tool_result_attachment":
                    continue
                if item.get("type") == "message":
                    item = dict(item)
                    item["role"] = "assistant"
                else:
                    out.append(item)
                    continue
            content = item.get("content")
            if isinstance(content, str):
                if content.strip():
                    out.append(item)
                continue
            if isinstance(content, list):
                filtered: list[Any] = []
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype in ("input_text", "output_text"):
                            if not str(part.get("text", "")).strip():
                                continue
                        elif ptype == "refusal":
                            if not str(part.get("refusal", "")).strip():
                                continue
                    filtered.append(part)
                if filtered:
                    new_item = dict(item)
                    new_item["content"] = filtered
                    out.append(new_item)
                continue
            continue
        return out

    _NATIVE_INPUT_ITEM_TYPES = {
        "function_call",
        "function_call_output",
        "reasoning",
        "item_reference",
        "_kiss_pending_tool_result_attachment",
    }
    _NATIVE_INPUT_PART_TYPES = {
        "input_text",
        "output_text",
        "input_image",
        "input_file",
        "input_audio",
        "refusal",
    }

    @classmethod
    def _is_native_input_item(cls, item: dict[str, Any]) -> bool:
        """Return ``True`` when ``item`` is already a valid Responses input item.

        Native items are standalone Responses items (``function_call``,
        ``function_call_output``, ``reasoning``, ...), messages with plain
        string content, and messages whose content parts all use Responses
        part types (``input_text`` / ``output_text`` / ...).  Foreign items
        — Chat-Completions messages with ``tool_calls`` arrays or
        ``role="tool"``, Anthropic block lists, Gemini messages with an
        ``attachments`` field — enter the conversation when it is handed
        off from another provider's model (e.g. via the Sorcar
        ``set_model`` tool) and require conversion.

        Args:
            item: A conversation item dict.

        Returns:
            ``True`` when the item can be sent to the Responses API as-is.
        """
        itype = item.get("type")
        if itype in cls._NATIVE_INPUT_ITEM_TYPES:
            return True
        role = item.get("role")
        if role is None:
            return True
        if role == "tool" or "tool_calls" in item or item.get("attachments"):
            return False
        content = item.get("content")
        if isinstance(content, list):
            return all(
                not isinstance(part, dict)
                or part.get("type") in cls._NATIVE_INPUT_PART_TYPES
                for part in content
            )
        return True

    def _foreign_items_to_native_input(
        self, conversation: list[Any]
    ) -> list[Any]:
        """Convert handed-off foreign conversation items to Responses items.

        When a live conversation is handed off from another provider's
        model (``new_model.conversation = old_model.conversation``, e.g.
        via the Sorcar ``set_model`` tool), it contains OpenAI
        Chat-Completions messages (``tool_calls`` arrays, ``role="tool"``
        messages, ``text`` / ``image_url`` / ``file`` parts), Anthropic
        Messages block lists (``tool_use`` / ``tool_result`` / ``thinking``
        blocks) or Gemini messages (dict tool-call arguments,
        ``attachments`` fields).  Each foreign item is normalized to
        Chat-Completions format via
        :meth:`OpenAICompatibleModel._normalize_message_for_api` and then
        translated to Responses ``input`` items via
        :meth:`OpenAICompatibleModel._chat_message_to_responses_items`.
        Native Responses items pass through unchanged, so the conversion
        is idempotent.

        Args:
            conversation: The conversation, possibly containing foreign items.

        Returns:
            The conversation with every item in Responses input format.
        """
        out: list[Any] = []
        for item in conversation:
            if not isinstance(item, dict) or self._is_native_input_item(item):
                out.append(item)
                continue
            for chat_msg in OpenAICompatibleModel._normalize_message_for_api(item):
                out.extend(
                    OpenAICompatibleModel._chat_message_to_responses_items(chat_msg)
                )
        return out

    @staticmethod
    def _translate_response_format_for_responses(
        rf: dict[str, Any],
    ) -> dict[str, Any]:
        """Flatten Chat-Completions ``response_format`` for the Responses API.

        Chat-Completions uses ``{"type":"json_schema","json_schema":{...}}``
        but the Responses API expects the schema fields hoisted to the
        top level: ``{"type":"json_schema","name":"...","schema":{...}}``.
        Other ``response_format`` shapes (``{"type":"json_object"}`` and
        already-flat dicts) are passed through unchanged.
        """
        if (
            isinstance(rf, dict)
            and rf.get("type") == "json_schema"
            and isinstance(rf.get("json_schema"), dict)
        ):
            return {"type": "json_schema", **rf["json_schema"]}
        return rf

    @staticmethod
    def _is_valid_json(s: str) -> bool:
        """Return ``True`` if ``s`` decodes as valid JSON.

        Used by the streaming tool-call accumulator to decide whether a
        partial argument buffer assembled from ``arguments.delta`` events
        should be overridden by the full ``arguments`` payload carried in
        a later (or earlier-arriving) ``output_item.added`` event.

        Args:
            s: A candidate JSON string.

        Returns:
            ``True`` if ``json.loads(s)`` succeeds; ``False`` otherwise.
        """
        if not s:
            return False
        try:
            json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return False
        return True

    @staticmethod
    def _move_tool_slot(
        tool_calls: dict[int, dict[str, str]],
        item_to_idx: dict[str, int],
        args_from_delta: set[int],
        args_from_added: set[int],
        item_id: str,
        old_idx: int,
        new_idx: int,
    ) -> int:
        """Move a tool-call slot to a new output_index without losing siblings.

        When the destination is occupied by a DIFFERENT item_id (typically
        the result of an earlier provisional allocation), relocate that
        occupant to ``old_idx`` if free, otherwise to a fresh index past
        the current maximum.  This guarantees parallel streaming
        function_call items both survive when their real output_indexes
        arrive out of order.
        """
        if old_idx == new_idx:
            return new_idx
        moving_in_delta = old_idx in args_from_delta
        moving_in_added = old_idx in args_from_added
        occupant_in_delta = new_idx in args_from_delta
        occupant_in_added = new_idx in args_from_added

        moving = tool_calls.pop(old_idx)
        occupant = tool_calls.get(new_idx)
        occupant_relocated_to: int | None = None
        occupant_merged = False
        if occupant is not None:
            occupant_iid = str(occupant.get("item_id", "") or "")
            if occupant_iid and occupant_iid != item_id:
                relocated_idx = (
                    old_idx
                    if old_idx not in tool_calls
                    else (max(tool_calls, default=-1) + 1)
                )
                tool_calls[relocated_idx] = tool_calls.pop(new_idx)
                item_to_idx[occupant_iid] = relocated_idx
                occupant_relocated_to = relocated_idx
            else:
                for field_name in ("id", "name", "arguments", "item_id"):
                    if not occupant.get(field_name) and moving.get(field_name):
                        occupant[field_name] = moving[field_name]
                moving = occupant
                tool_calls.pop(new_idx, None)
                occupant_merged = True
        tool_calls[new_idx] = moving
        if item_id:
            item_to_idx[item_id] = new_idx

        args_from_delta.discard(old_idx)
        args_from_delta.discard(new_idx)
        args_from_added.discard(old_idx)
        args_from_added.discard(new_idx)
        if moving_in_delta:
            args_from_delta.add(new_idx)
        if moving_in_added:
            args_from_added.add(new_idx)
        if occupant_merged:
            if occupant_in_delta:
                args_from_delta.add(new_idx)
            if occupant_in_added:
                args_from_added.add(new_idx)
        elif occupant_relocated_to is not None:
            if occupant_in_delta:
                args_from_delta.add(occupant_relocated_to)
            if occupant_in_added:
                args_from_added.add(occupant_relocated_to)
        return new_idx

    @staticmethod
    def _stream_output_index(event: Any, default: int) -> int:
        """Read ``event.output_index`` defensively.

        Returns ``default`` (typically ``len(tool_calls)`` — i.e. allocate
        a fresh slot) when ``output_index`` is missing, ``None``, or
        non-integer.  Critically, this does NOT collapse ``output_index=0``
        to the default the way ``int(... or 0)`` would have.
        """
        raw = getattr(event, "output_index", None)
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _commit_final_text(
        key: tuple[int, int],
        final_text: str,
        text_part_buffers: dict[tuple[int, int], str],
        text_delta_seen: set[tuple[int, int]],
        allow_suffix: bool = True,
    ) -> str:
        """Record an authoritative final text value for a content part.

        Overrides the per-part buffer with ``final_text`` and returns the
        text (if any) that the caller must append to ``content`` and forward
        to the token callback: the full value for done-only parts (marking
        ``key`` as seen so the terminal merge doesn't duplicate it), or —
        when ``allow_suffix`` — the missing suffix when earlier deltas
        streamed a strict prefix of the final value.
        """
        old_text = text_part_buffers.get(key, "")
        text_part_buffers[key] = final_text
        if key not in text_delta_seen and final_text:
            text_delta_seen.add(key)
            return final_text
        if (
            allow_suffix
            and key in text_delta_seen
            and final_text.startswith(old_text)
        ):
            return final_text[len(old_text):]
        return ""

    def _slot_for_argument_event(
        self,
        event: Any,
        tool_calls: dict[int, dict[str, str]],
        item_to_idx: dict[str, int],
        args_from_delta: set[int],
        args_from_added: set[int],
    ) -> tuple[dict[str, str], int]:
        """Locate (or create) the tool-call slot for an arguments event.

        Resolves the ``(slot, idx)`` for a ``function_call_arguments.delta``
        / ``.done`` event: prefers the slot already mapped to the event's
        ``item_id``, re-keying it via :meth:`_move_tool_slot` when the event
        carries a real ``output_index`` that differs from the provisional
        one (the model's real output ordering must win — otherwise parallel
        tool calls return in the wrong order).  Unknown ``item_id`` values
        allocate a fresh slot at the event's ``output_index`` (or at the
        next free position).
        """
        item_id = str(getattr(event, "item_id", "") or "")
        idx_opt = item_to_idx.get(item_id)
        if idx_opt is None:
            idx = self._stream_output_index(event, len(tool_calls))
        elif getattr(event, "output_index", None) is not None:
            real_idx = self._stream_output_index(event, len(tool_calls))
            if real_idx != idx_opt:
                idx = self._move_tool_slot(
                    tool_calls,
                    item_to_idx,
                    args_from_delta,
                    args_from_added,
                    item_id,
                    idx_opt,
                    real_idx,
                )
            else:
                idx = idx_opt
        else:
            idx = idx_opt
        slot = tool_calls.setdefault(
            idx, {"id": "", "name": "", "arguments": "", "item_id": ""}
        )
        if item_id and not slot.get("item_id"):
            slot["item_id"] = item_id
            item_to_idx[item_id] = idx
        return slot, idx

    @staticmethod
    def _relocate_slot(
        tool_calls: dict[int, dict[str, str]],
        item_to_idx: dict[str, int],
        args_from_delta: set[int],
        args_from_added: set[int],
        from_idx: int,
    ) -> None:
        """Move the slot at ``from_idx`` to a fresh index past the maximum.

        Used when a newly-arrived function_call item claims an
        ``output_index`` already occupied by a DIFFERENT item: the occupant
        (typically a provisional allocation) is relocated so both parallel
        tool calls survive.  The occupant's ``item_id`` mapping and
        marker-set membership follow it to the new index.
        """
        free_idx = max(tool_calls) + 1
        relocated = tool_calls.pop(from_idx)
        tool_calls[free_idx] = relocated
        occupant_iid = str(relocated.get("item_id", "") or "")
        if occupant_iid:
            item_to_idx[occupant_iid] = free_idx
        if from_idx in args_from_delta:
            args_from_delta.discard(from_idx)
            args_from_delta.add(free_idx)
        if from_idx in args_from_added:
            args_from_added.discard(from_idx)
            args_from_added.add(free_idx)

    def _shape_responses_kwargs(
        self,
        *,
        input_items: list[Any],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Translate ``model_config`` + ``input_items`` into Responses kwargs.

        Single source of truth for parameter shaping shared by both
        :meth:`_build_request_kwargs` (normal path) and
        :meth:`_generate_with_text_based_tools` (DeepSeek fallback).

        Translations applied:

        * ``system_instruction`` → top-level ``instructions=``.
        * ``reasoning_effort`` → ``reasoning.effort`` (merged into a
          shallow-copy of any caller-supplied ``reasoning`` dict so the
          caller's config is never mutated).  Whenever an effort is sent,
          ``reasoning.summary`` defaults to ``"auto"`` — OpenAI returns
          reasoning items with EMPTY summaries (and emits zero
          ``response.reasoning_summary_text.delta`` stream events) unless
          the request explicitly opts in, so without this default no
          thinking tokens would ever be revealed.  A caller-supplied
          ``summary`` always wins.
        * ``max_tokens`` / ``max_completion_tokens`` → ``max_output_tokens``
          (``max_completion_tokens`` wins when both are set).
        * ``response_format`` → ``text.format`` (Responses-API shape).
        * Chat-Completions ``tool_choice`` (``{"type":"function",
          "function":{...}}``) → flat Responses shape.
        * ``enable_cache`` (v2-internal flag) is stripped.
        * OpenRouter Anthropic prompt-caching ``cache_control`` is added.

        Args:
            input_items: The raw conversation array.  Whitespace-only
                items are filtered; a :class:`KISSError` is raised if
                everything is empty.
            tools: Pre-flattened Responses-API tool list, or ``None``.

        Returns:
            A kwargs dict ready to pass to ``client.responses.create(...)``.

        Raises:
            KISSError: When the conversation is empty after normalisation.
        """
        kwargs = {
            key: value
            for key, value in self.model_config.items()
            if key not in FRAMEWORK_ONLY_CONFIG_KEYS
        }
        system_instruction = self.model_config.get("system_instruction")
        reasoning_effort = kwargs.pop("reasoning_effort", None)

        max_tokens = kwargs.pop("max_tokens", None)
        max_completion_tokens = kwargs.pop("max_completion_tokens", None)
        if "max_output_tokens" not in kwargs:
            if max_completion_tokens is not None:
                kwargs["max_output_tokens"] = max_completion_tokens
            elif max_tokens is not None:
                kwargs["max_output_tokens"] = max_tokens

        response_format = kwargs.pop("response_format", None)
        if response_format is not None:
            existing_text = kwargs.get("text")
            text_cfg = (
                dict(existing_text) if isinstance(existing_text, dict) else {}
            )
            text_cfg["format"] = self._translate_response_format_for_responses(
                response_format
            )
            kwargs["text"] = text_cfg

        kwargs["model"] = self._api_model_name
        normalized = self._normalize_input(input_items)
        if not normalized:
            raise KISSError(
                "Cannot generate response: all input items have empty or "
                "whitespace-only content. At least one item with "
                "non-whitespace content is required."
            )
        kwargs["input"] = normalized

        if system_instruction:
            kwargs["instructions"] = system_instruction
        if reasoning_effort is not None:
            existing = kwargs.get("reasoning")
            if isinstance(existing, dict):
                existing = dict(existing)
            else:
                existing = {}
            existing["effort"] = reasoning_effort
            kwargs["reasoning"] = existing
        reasoning_cfg = kwargs.get("reasoning")
        if (
            isinstance(reasoning_cfg, dict)
            and "effort" in reasoning_cfg
            and "summary" not in reasoning_cfg
        ):
            reasoning_cfg = dict(reasoning_cfg)
            reasoning_cfg["summary"] = "auto"
            kwargs["reasoning"] = reasoning_cfg
        kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        parallel_tool_calls = kwargs.pop("parallel_tool_calls", None)
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = self._flatten_tool_choice(tool_choice)
            if parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = parallel_tool_calls

        kwargs = self._keep_supported_request_params(
            kwargs,
            _RESPONSES_REQUEST_PARAMS,
            "OpenAI Responses",
        )
        self._sanitize_stream_options(kwargs)
        self._apply_cache_control_for_openrouter_anthropic(kwargs)
        return kwargs

    def _unanswered_function_calls_from_conversation_with_names(
        self,
    ) -> list[dict[str, str]]:
        """Return ``(name, call_id)`` pairs for every unanswered prior function_call.

        Walks the entire conversation (not just the trailing run) and
        collects every ``function_call`` item that has not yet received
        a matching ``function_call_output``.  Used as the fallback path
        in :meth:`add_function_results_to_conversation_and_return` when
        the in-memory pending queue has been lost (process restart /
        object reconstruction from a serialized conversation).

        Returns:
            A list of ``{"name": ..., "call_id": ...}`` dicts in
            declaration order.  Empty when every prior
            ``function_call`` has been answered.
        """
        outstanding: list[dict[str, str]] = []
        for item in self.conversation:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "function_call":
                call_id = str(item.get("call_id", "") or "")
                name = str(item.get("name", "") or "")
                if call_id:
                    outstanding.append({"name": name, "call_id": call_id})
            elif itype == "function_call_output":
                call_id = str(item.get("call_id", "") or "")
                for j, pending in enumerate(outstanding):
                    if pending["call_id"] == call_id:
                        outstanding.pop(j)
                        break
        return outstanding

    def _validate_function_call_conversation(self) -> list[str]:
        """Validate ``function_call`` / ``function_call_output`` pairing.

        Beyond detecting unanswered function_calls, also catches:

        * ``function_call_output`` items with no prior matching
          ``function_call`` (orphan outputs — restored/mutated
          conversations).
        * Duplicate ``function_call_output`` for the same ``call_id``.
        * ``function_call`` / ``function_call_output`` items missing a
          ``call_id``.

        These all violate the Responses-API conversation contract and
        would be rejected by the server.  Fail locally so callers see
        the problem immediately instead of paying for a wasted
        round-trip.

        Returns:
            ``call_id`` strings of unanswered prior function_calls, in
            declaration order.

        Raises:
            KISSError: If any of the conditions above is detected.
        """
        seen_calls: set[str] = set()
        answered_calls: set[str] = set()
        outstanding: list[str] = []
        for item in self.conversation:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if item.get("role") == "user" and outstanding:
                raise KISSError(
                    "user message appears before function_call_output "
                    f"items for pending call_ids: {outstanding!r}"
                )
            if itype == "function_call":
                call_id = str(item.get("call_id", "") or "")
                if not call_id:
                    raise KISSError("function_call item missing call_id")
                name = str(item.get("name", "") or "")
                if not name:
                    raise KISSError(
                        "function_call item missing name for call_id "
                        f"{call_id!r}"
                    )
                arguments = item.get("arguments", "")
                if not isinstance(arguments, str):
                    raise KISSError(
                        "function_call.arguments must be a string for "
                        f"call_id {call_id!r}"
                    )
                if call_id in seen_calls:
                    raise KISSError(
                        f"Duplicate function_call call_id: {call_id!r}"
                    )
                seen_calls.add(call_id)
                outstanding.append(call_id)
            elif itype == "function_call_output":
                call_id = str(item.get("call_id", "") or "")
                if not call_id:
                    raise KISSError(
                        "function_call_output item missing call_id"
                    )
                if "output" not in item:
                    raise KISSError(
                        "function_call_output missing output for call_id "
                        f"{call_id!r}"
                    )
                if not isinstance(item.get("output"), str):
                    raise KISSError(
                        "function_call_output.output must be a string for "
                        f"call_id {call_id!r}"
                    )
                if call_id not in seen_calls:
                    raise KISSError(
                        "function_call_output has no prior function_call: "
                        f"{call_id!r}"
                    )
                if call_id in answered_calls:
                    raise KISSError(
                        "Duplicate function_call_output for call_id: "
                        f"{call_id!r}"
                    )
                answered_calls.add(call_id)
                if call_id in outstanding:
                    outstanding.remove(call_id)
        return outstanding

    def _unanswered_function_calls_from_conversation(self) -> list[str]:
        """Return ``call_id`` of every prior ``function_call`` lacking an output.

        Thin id-only view over
        :meth:`_unanswered_function_calls_from_conversation_with_names`,
        which owns the single scanning loop.

        Returns:
            ``call_id`` strings of unanswered function_calls, in
            declaration order.
        """
        return [
            pending["call_id"]
            for pending in self._unanswered_function_calls_from_conversation_with_names()
        ]

    def _ensure_no_pending_function_calls(self) -> None:
        """Reject new generations while prior ``function_call`` outputs are pending.

        The Responses-API conversation contract requires every
        model-produced ``function_call`` to have a matching
        ``function_call_output`` before the next request.  Sending a
        partial conversation (e.g. with two ``function_call`` items but
        only one ``function_call_output``) is rejected by the API and
        wastes a round-trip.  Fail locally so the caller sees the
        problem immediately.

        Validates BOTH the in-memory ``_pending_function_calls`` queue
        and the conversation itself, since restored / reconstructed
        conversations may have prior ``function_call`` items without a
        populated pending queue.

        Raises:
            KISSError: When any ``function_call`` in the conversation or
                pending queue has not yet received a matching
                ``function_call_output``.
        """
        pending = [
            p.get("call_id") or p.get("name") or "<unknown>"
            for p in self._pending_function_calls
        ]
        outstanding = self._validate_function_call_conversation()
        if pending or outstanding:
            raise KISSError(
                "Cannot generate a new response while function_call outputs "
                f"are still pending for call_ids/names: {pending or outstanding!r}"
            )

    def _sanitize_stream_options(self, kwargs: dict[str, Any]) -> None:
        """Keep only the ``stream_options`` sub-keys the Responses API declares.

        ``stream_options`` exists on both transports but with different
        shapes: Chat Completions accepts ``include_usage`` (a config written
        for it would carry that), while the Responses API declares only
        ``include_obfuscation`` and reports the totals in its terminal
        ``response.completed`` event instead.  The same reporting rule as
        for top-level parameters applies to the nested dict, so a
        legitimate ``include_obfuscation`` survives and anything else is
        named in a warning.

        Args:
            kwargs: The request kwargs; mutated in place.
        """
        options = kwargs.get("stream_options")
        if not isinstance(options, dict):
            return
        kept = self._keep_supported_request_params(
            options, _RESPONSES_STREAM_OPTION_KEYS, "OpenAI Responses stream_options"
        )
        if kept:
            kwargs["stream_options"] = kept
        else:
            kwargs.pop("stream_options")

    def _build_request_kwargs(
        self, *, tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """Assemble the kwargs dict for ``client.responses.create(...)``.

        Thin wrapper around :meth:`_shape_responses_kwargs` that uses the
        current ``self.conversation`` as the input array.

        Args:
            tools: Pre-flattened Responses-API tool list, or ``None`` for a
                tool-less call.

        Returns:
            The kwargs dict.
        """
        self._ensure_no_pending_function_calls()
        return self._shape_responses_kwargs(
            input_items=self.conversation, tools=tools
        )

    @staticmethod
    def _reasoning_inner_index(event: Any) -> int:
        """Return the reasoning event's inner (summary or content) index.

        Avoids the ``or``-truthiness trap that mis-keys ``summary_index=0``
        as missing and falls back to ``content_index``.  Treats only
        ``None`` (or values that can't be coerced to ``int``) as absent.

        Args:
            event: A reasoning_text / reasoning_summary_text streaming event.

        Returns:
            The matched index as ``int``; ``0`` when absent or malformed.
        """
        for attr in ("summary_index", "content_index"):
            value = getattr(event, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0
        return 0

    @staticmethod
    def _raw_items_have_message_text(items: list[dict[str, Any]]) -> bool:
        """Return ``True`` iff ``items`` already carry an assistant text part.

        Used by the conversation-replay branches to decide whether
        streamed assistant text needs to be appended manually after
        replaying ``response.output`` items verbatim — some gateways
        emit text deltas during streaming but omit the corresponding
        ``message`` item from the terminal ``response.output``, which
        would silently drop the assistant text from the conversation.

        Args:
            items: The list returned by
                :meth:`_response_output_items_to_input_items`.

        Returns:
            ``True`` when at least one item carries non-empty
            ``output_text`` / ``refusal`` content or a non-empty
            assistant string content.
        """
        for item in items:
            if not isinstance(item, dict):
                continue
            if (
                item.get("type") != "message"
                and item.get("role") != "assistant"
            ):
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return True
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") not in ("output_text", "refusal"):
                        continue
                    text = str(
                        part.get("text") or part.get("refusal") or ""
                    )
                    if text.strip():
                        return True
        return False

    @staticmethod
    def _event_type(event: Any) -> str:
        """Return the SSE event-type string for ``event``.

        Args:
            event: An event object returned by the streaming SDK.

        Returns:
            The event's ``type`` attribute as a string (empty if missing).
        """
        return str(getattr(event, "type", "") or "")

    def _consume_stream(
        self,
        stream: Any,
    ) -> tuple[str, list[dict[str, str]], Any]:
        """Consume a Responses stream under an abort watchdog.

        ``stop_aware_events`` aborts the request the moment the user
        presses Stop or the provider goes silent for
        ``stream_stall_timeout`` seconds; a bare ``for event in stream``
        would hold the agent inside ``recv()`` until the client's own
        30-minute timeout expires
        (``reports/stop_button_delay_2026-08-05.html``).

        Closing the generator explicitly is what makes that safe:
        :meth:`_consume_stream_events` raises from *inside* the loop on
        ``response.failed`` / ``response.incomplete``, and an abandoned
        generator runs its cleanup only whenever the traceback holding
        its frame is released — until then a daemon watchdog thread stays
        alive and armed over a connection that never returns to the pool.

        Args:
            stream: The streaming iterator returned by the SDK.

        Returns:
            ``(content, tool_calls, response)`` — see
            :meth:`_consume_stream_events`.
        """
        events = stop_aware_events(
            stream,
            stall_timeout=self._stream_stall_timeout,
            on_abort=self._close_thinking_if_open,
            name="openai-responses-stream-abort-watchdog",
        )
        try:
            return self._consume_stream_events(events)
        finally:
            events.close()

    def _consume_stream_events(
        self,
        events: Any,
    ) -> tuple[str, list[dict[str, str]], Any]:
        """Drive callbacks from streaming events and collect final state.

        Walks every SSE event from the Responses streaming API:

        * ``response.output_text.delta`` — forwarded to ``token_callback``
          and accumulated into the returned ``content`` string.
        * ``response.reasoning_text.delta`` and
          ``response.reasoning_summary_text.delta`` — forwarded to
          ``token_callback`` while bracketed by ``thinking_callback(True)``
          / ``thinking_callback(False)``.
        * ``response.output_item.added`` carrying a ``function_call`` item —
          seeds a new tool-call accumulator with the ``call_id`` and ``name``.
        * ``response.function_call_arguments.delta`` — accumulates argument
          chunks for the matching tool-call.
        * ``response.completed`` — captures the final response object.

        Args:
            events: The stop/stall-aware iterator over the SDK stream.

        Returns:
            ``(content, tool_calls, response)`` where ``tool_calls`` is a
            list of ``{"id": call_id, "name": name, "arguments":
            args_json}`` dicts in declaration order, and ``response`` is
            the final response object from the terminal
            ``response.completed`` event (or the last event seen as a
            fallback).
        """
        content = ""
        tool_calls: dict[int, dict[str, str]] = {}
        args_from_delta: set[int] = set()
        args_from_added: set[int] = set()
        item_to_idx: dict[str, int] = {}
        text_part_buffers: dict[tuple[int, int], str] = {}
        added_message_buffers: dict[tuple[int, int], str] = {}
        text_delta_seen: set[tuple[int, int]] = set()
        reasoning_delta_seen: set[tuple[int, int]] = set()
        reasoning_part_buffers: dict[tuple[int, int], str] = {}
        response: Any = None
        saw_completed = False
        in_reasoning = False
        item_output_indexes: dict[str, int] = {}
        self._last_stream_item_indexes = item_output_indexes
        self._last_stream_message_output_index = None

        for event in events:
            etype = self._event_type(event)
            ev_item_id = str(getattr(event, "item_id", "") or "")
            if not ev_item_id:
                nested_item = getattr(event, "item", None)
                if nested_item is not None:
                    ev_item_id = str(getattr(nested_item, "id", "") or "")
            if ev_item_id:
                ev_output_idx = getattr(event, "output_index", None)
                if ev_output_idx is not None:
                    try:
                        item_output_indexes[ev_item_id] = int(ev_output_idx)
                    except (TypeError, ValueError):
                        pass

            if etype in (
                "response.output_text.delta",
                "response.refusal.delta",
            ):
                if in_reasoning:
                    in_reasoning = False
                    self._invoke_thinking_callback(False)
                output_index = self._stream_output_index(event, 0)
                content_index = int(getattr(event, "content_index", 0) or 0)
                text_delta_seen.add((output_index, content_index))
                delta = getattr(event, "delta", "") or ""
                key = (output_index, content_index)
                text_part_buffers[key] = text_part_buffers.get(key, "") + delta
                content += delta
                self._invoke_token_callback(delta)
            elif etype in (
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            ):
                if not in_reasoning:
                    in_reasoning = True
                    self._invoke_thinking_callback(True)
                output_index = self._stream_output_index(event, 0)
                inner_index = self._reasoning_inner_index(event)
                key = (output_index, inner_index)
                reasoning_delta_seen.add(key)
                delta = getattr(event, "delta", "") or ""
                reasoning_part_buffers[key] = (
                    reasoning_part_buffers.get(key, "") + delta
                )
                self._invoke_token_callback(delta)
            elif etype in (
                "response.reasoning_text.done",
                "response.reasoning_summary_text.done",
            ):
                output_index = self._stream_output_index(event, 0)
                inner_index = self._reasoning_inner_index(event)
                key = (output_index, inner_index)
                final_reasoning = str(getattr(event, "text", "") or "")
                old_reasoning = reasoning_part_buffers.get(key, "")
                if key not in reasoning_delta_seen and final_reasoning:
                    if not in_reasoning:
                        in_reasoning = True
                        self._invoke_thinking_callback(True)
                    self._invoke_token_callback(final_reasoning)
                elif (
                    key in reasoning_delta_seen
                    and final_reasoning.startswith(old_reasoning)
                ):
                    suffix = final_reasoning[len(old_reasoning):]
                    if suffix:
                        opened_here = False
                        if not in_reasoning:
                            in_reasoning = True
                            opened_here = True
                            self._invoke_thinking_callback(True)
                        self._invoke_token_callback(suffix)
                        if opened_here:
                            in_reasoning = False
                            self._invoke_thinking_callback(False)
                reasoning_part_buffers[key] = final_reasoning
                if in_reasoning:
                    in_reasoning = False
                    self._invoke_thinking_callback(False)
            elif etype in (
                "response.output_text.done",
                "response.refusal.done",
            ):
                if in_reasoning:
                    in_reasoning = False
                    self._invoke_thinking_callback(False)
                output_index = self._stream_output_index(event, 0)
                content_index = int(getattr(event, "content_index", 0) or 0)
                if etype == "response.output_text.done":
                    final_text = str(getattr(event, "text", "") or "")
                else:
                    final_text = str(getattr(event, "refusal", "") or "")
                emitted = self._commit_final_text(
                    (output_index, content_index),
                    final_text,
                    text_part_buffers,
                    text_delta_seen,
                )
                if emitted:
                    content += emitted
                    self._invoke_token_callback(emitted)
            elif etype == "response.content_part.done":
                if in_reasoning:
                    in_reasoning = False
                    self._invoke_thinking_callback(False)
                output_index = self._stream_output_index(event, 0)
                content_index = int(getattr(event, "content_index", 0) or 0)
                part = getattr(event, "part", None)
                ptype = getattr(part, "type", "") if part is not None else ""
                if ptype == "output_text":
                    final_text = str(getattr(part, "text", "") or "")
                elif ptype == "refusal":
                    final_text = str(getattr(part, "refusal", "") or "")
                else:
                    final_text = ""
                if ptype in ("output_text", "refusal"):
                    emitted = self._commit_final_text(
                        (output_index, content_index),
                        final_text,
                        text_part_buffers,
                        text_delta_seen,
                    )
                    if emitted:
                        content += emitted
                        self._invoke_token_callback(emitted)
            elif etype == "response.output_item.added":
                if in_reasoning:
                    in_reasoning = False
                    self._invoke_thinking_callback(False)
                item = getattr(event, "item", None)
                if item is not None and getattr(item, "type", "") == "message":
                    output_index = self._stream_output_index(event, 0)
                    parts = getattr(item, "content", None) or []
                    for content_index, part in enumerate(parts):
                        ptype = getattr(part, "type", "")
                        if ptype == "output_text":
                            final_text = str(getattr(part, "text", "") or "")
                        elif ptype == "refusal":
                            final_text = str(getattr(part, "refusal", "") or "")
                        else:
                            continue
                        key = (output_index, content_index)
                        added_message_buffers[key] = final_text
                elif item is not None and getattr(item, "type", "") == "function_call":
                    item_id = str(getattr(item, "id", "") or "")
                    call_id = str(getattr(item, "call_id", "") or "")
                    name = str(getattr(item, "name", "") or "")
                    args = str(getattr(item, "arguments", "") or "")
                    event_idx = self._stream_output_index(event, len(tool_calls))
                    has_real_index = (
                        getattr(event, "output_index", None) is not None
                    )
                    if item_id and item_id in item_to_idx:
                        old_idx = item_to_idx[item_id]
                        if has_real_index and event_idx != old_idx:
                            idx = self._move_tool_slot(
                                tool_calls,
                                item_to_idx,
                                args_from_delta,
                                args_from_added,
                                item_id,
                                old_idx,
                                event_idx,
                            )
                        else:
                            idx = old_idx
                    else:
                        idx = event_idx
                        occupant = tool_calls.get(event_idx)
                        if occupant is not None:
                            occupant_iid = str(
                                occupant.get("item_id", "") or ""
                            )
                            if (
                                occupant_iid
                                and item_id
                                and occupant_iid != item_id
                            ):
                                if not has_real_index:
                                    idx = max(tool_calls) + 1
                                    event_idx = idx
                        if has_real_index:
                            occupant = tool_calls.get(event_idx)
                            if occupant is not None:
                                occupant_iid = str(
                                    occupant.get("item_id", "") or ""
                                )
                                if occupant_iid and occupant_iid != item_id:
                                    self._relocate_slot(
                                        tool_calls,
                                        item_to_idx,
                                        args_from_delta,
                                        args_from_added,
                                        event_idx,
                                    )
                    slot = tool_calls.setdefault(
                        idx,
                        {"id": "", "name": "", "arguments": "", "item_id": ""},
                    )
                    if call_id and not slot.get("id"):
                        slot["id"] = call_id
                    if name and not slot.get("name"):
                        slot["name"] = name
                    if args:
                        existing_args = slot.get("arguments", "")
                        if not existing_args:
                            slot["arguments"] = args
                        elif idx in args_from_delta:
                            slot["arguments"] = args
                        args_from_added.add(idx)
                    if item_id:
                        slot["item_id"] = item_id
                        item_to_idx[item_id] = idx
            elif etype == "response.function_call_arguments.delta":
                if in_reasoning:
                    in_reasoning = False
                    self._invoke_thinking_callback(False)
                slot, idx = self._slot_for_argument_event(
                    event, tool_calls, item_to_idx, args_from_delta, args_from_added
                )
                arg_delta: str = str(getattr(event, "delta", "") or "")
                if idx in args_from_added and arg_delta:
                    existing_arg_str: str = str(slot.get("arguments", "") or "")
                    if (
                        existing_arg_str == arg_delta
                        or self._is_valid_json(existing_arg_str)
                    ):
                        arg_delta = ""
                slot["arguments"] += arg_delta
                if arg_delta:
                    args_from_delta.add(idx)
            elif etype == "response.function_call_arguments.done":
                if in_reasoning:
                    in_reasoning = False
                    self._invoke_thinking_callback(False)
                slot, _idx = self._slot_for_argument_event(
                    event, tool_calls, item_to_idx, args_from_delta, args_from_added
                )
                final_args = getattr(event, "arguments", None)
                if final_args is not None:
                    slot["arguments"] = str(final_args)
            elif etype in (
                "response.output_item.done",
                "response.output_item.completed",
            ):
                item = getattr(event, "item", None)
                if item is not None and getattr(item, "type", "") == "function_call":
                    if in_reasoning:
                        in_reasoning = False
                        self._invoke_thinking_callback(False)
                    item_id = str(getattr(item, "id", "") or "")
                    has_real_index = (
                        getattr(event, "output_index", None) is not None
                    )
                    real_idx = self._stream_output_index(event, len(tool_calls))
                    if item_id and item_id in item_to_idx:
                        old_idx = item_to_idx[item_id]
                        if has_real_index and real_idx != old_idx:
                            idx = self._move_tool_slot(
                                tool_calls,
                                item_to_idx,
                                args_from_delta,
                                args_from_added,
                                item_id,
                                old_idx,
                                real_idx,
                            )
                        else:
                            idx = old_idx
                    else:
                        idx = real_idx
                        if has_real_index:
                            occupant = tool_calls.get(real_idx)
                            if occupant is not None:
                                occupant_iid = str(
                                    occupant.get("item_id", "") or ""
                                )
                                if (
                                    occupant_iid
                                    and item_id
                                    and occupant_iid != item_id
                                ):
                                    self._relocate_slot(
                                        tool_calls,
                                        item_to_idx,
                                        args_from_delta,
                                        args_from_added,
                                        real_idx,
                                    )
                    slot = tool_calls.setdefault(
                        idx, {"id": "", "name": "", "arguments": "", "item_id": ""}
                    )
                    call_id = str(getattr(item, "call_id", "") or "")
                    name = str(getattr(item, "name", "") or "")
                    if call_id:
                        slot["id"] = call_id
                    if name:
                        slot["name"] = name
                    if hasattr(item, "arguments"):
                        raw_args = getattr(item, "arguments")
                        if raw_args is not None:
                            slot["arguments"] = str(raw_args)
                    if item_id:
                        slot["item_id"] = item_id
                        item_to_idx[item_id] = idx
                elif item is not None and getattr(item, "type", "") == "message":
                    if in_reasoning:
                        in_reasoning = False
                        self._invoke_thinking_callback(False)
                    output_index = self._stream_output_index(event, 0)
                    parts = getattr(item, "content", None) or []
                    for content_index, part in enumerate(parts):
                        ptype = getattr(part, "type", "")
                        if ptype == "output_text":
                            final_text = str(getattr(part, "text", "") or "")
                        elif ptype == "refusal":
                            final_text = str(getattr(part, "refusal", "") or "")
                        else:
                            continue
                        emitted = self._commit_final_text(
                            (output_index, content_index),
                            final_text,
                            text_part_buffers,
                            text_delta_seen,
                            allow_suffix=False,
                        )
                        if emitted:
                            content += emitted
                            self._invoke_token_callback(emitted)
            elif etype in ("response.failed", "error"):
                if in_reasoning:
                    in_reasoning = False
                    self._invoke_thinking_callback(False)
                resp = getattr(event, "response", None)
                err = getattr(resp, "error", None) if resp is not None else None
                if err is None:
                    err = getattr(event, "error", None)
                message = ""
                if err is not None:
                    message = str(getattr(err, "message", "") or "")
                    if not message and isinstance(err, dict):
                        message = str(err.get("message", "") or "")
                raise KISSError(
                    f"Responses API stream ended with {etype}"
                    + (f": {message}" if message else "")
                )
            elif etype == "response.incomplete":
                if in_reasoning:
                    in_reasoning = False
                    self._invoke_thinking_callback(False)
                resp = getattr(event, "response", None)
                details = (
                    getattr(resp, "incomplete_details", None)
                    if resp is not None
                    else None
                )
                reason = ""
                if details is not None:
                    reason = str(getattr(details, "reason", "") or "")
                    if not reason and isinstance(details, dict):
                        reason = str(details.get("reason", "") or "")
                raise KISSError(
                    "Responses API stream ended incomplete"
                    + (f": {reason}" if reason else "")
                )
            elif etype == "response.completed":
                saw_completed = True
                response = getattr(event, "response", None)

        if in_reasoning:
            self._invoke_thinking_callback(False)
        if not saw_completed:
            raise KISSError(
                "Responses API stream ended without a terminal "
                "response.completed event."
            )
        if response is None:
            raise KISSError(
                "Responses API stream completed without a response payload."
            )

        all_text_keys = (
            set(text_part_buffers.keys()) | set(added_message_buffers.keys())
        )
        if all_text_keys:
            self._last_stream_message_output_index = min(
                key[0] for key in all_text_keys
            )

        self._raise_for_failed_response(response)

        final_content, final_tool_calls = self._parse_non_streaming(response)
        if self._response_has_message_text(response):
            if final_content.startswith(content):
                suffix = final_content[len(content):]
                if suffix:
                    self._invoke_token_callback(suffix)
            content = final_content
        else:
            for key, added_text in added_message_buffers.items():
                if key in text_delta_seen:
                    continue
                if key in text_part_buffers:
                    continue
                text_part_buffers[key] = added_text
                if added_text:
                    content += added_text
                    self._invoke_token_callback(added_text)
                    text_delta_seen.add(key)
            if text_part_buffers:
                content = "".join(
                    text_part_buffers[k] for k in sorted(text_part_buffers)
                )
        call_id_to_idx: dict[str, int] = {
            str(slot.get("id", "")): idx
            for idx, slot in tool_calls.items()
            if str(slot.get("id", ""))
        }
        for final_tc in final_tool_calls:
            iid = final_tc.get("item_id", "")
            fcid = final_tc.get("id", "")
            fname = final_tc.get("name", "")
            if iid and iid in item_to_idx:
                idx = item_to_idx[iid]
            elif fcid and fcid in call_id_to_idx:
                idx = call_id_to_idx[fcid]
            else:
                try:
                    candidate_idx = int(
                        final_tc.get("output_index", "") or len(tool_calls)
                    )
                except (ValueError, TypeError):
                    candidate_idx = len(tool_calls)
                if not iid and not fcid and not fname:
                    if tool_calls:
                        continue
                    raise KISSError(
                        "Responses API returned identityless function_call "
                        "without id/call_id/name"
                    )
                idx = candidate_idx
            slot = tool_calls.setdefault(
                idx, {"id": "", "name": "", "arguments": "", "item_id": ""}
            )
            if final_tc.get("id"):
                slot["id"] = final_tc["id"]
                call_id_to_idx[final_tc["id"]] = idx
            if final_tc.get("name"):
                slot["name"] = final_tc["name"]
            if "arguments" in final_tc:
                slot["arguments"] = final_tc["arguments"]
            if not slot.get("item_id") and iid:
                slot["item_id"] = iid
                item_to_idx[iid] = idx

        ordered: list[dict[str, str]] = []
        for k in sorted(tool_calls):
            slot = tool_calls[k]
            slot.setdefault("output_index", str(k))
            ordered.append(slot)
        return content, ordered, response

    @classmethod
    def _response_output(cls, response: Any) -> list[Any]:
        """Return ``response.output`` for both SDK objects and plain dicts.

        Args:
            response: A Responses-API response object (SDK model or
                plain dict).

        Returns:
            The ``output`` list (or ``[]`` when absent).
        """
        return cls._get_attr_or_key(response, "output", []) or []

    @classmethod
    def _raise_for_failed_response(cls, response: Any) -> None:
        """Raise :class:`KISSError` for terminal ``failed`` / ``incomplete`` statuses.

        Args:
            response: A Responses-API response object (streaming-final or
                non-streaming) carrying a ``status``, optional ``error``,
                and optional ``incomplete_details``.

        Raises:
            KISSError: When ``response.status`` is either ``"failed"`` or
                ``"incomplete"``.  The Responses API can terminate a
                response in either state; both must be surfaced to the
                caller so partial / truncated ``output`` (e.g. a
                half-formed ``function_call.arguments`` string) is never
                silently treated as a successful generation.
        """
        status = cls._get_attr_or_key(response, "status")
        if status == "failed":
            err = cls._get_attr_or_key(response, "error")
            message = ""
            if err is not None:
                message = str(cls._get_attr_or_key(err, "message", "") or "")
            raise KISSError(
                "Responses API returned failed response"
                + (f": {message}" if message else "")
            )
        if status == "incomplete":
            details = cls._get_attr_or_key(response, "incomplete_details")
            reason = ""
            if details is not None:
                reason = str(cls._get_attr_or_key(details, "reason", "") or "")
            raise KISSError(
                "Responses API returned incomplete response"
                + (f": {reason}" if reason else "")
            )

    @classmethod
    def _response_has_message_text(cls, response: Any) -> bool:
        """Return True iff ``response.output`` contains a message/refusal part.

        Used by the streaming terminal-merge logic to decide whether the
        final completed-response text should override partial streamed
        deltas.  This must return ``True`` even when the message's text
        is empty so that filtered/refused finalizations correctly clear
        any speculative deltas.

        Args:
            response: A non-streaming Responses-API response object (or
                the terminal ``response.completed.response`` payload).

        Returns:
            ``True`` if at least one ``message`` item carries an
            ``output_text`` or ``refusal`` content part; ``False``
            otherwise.
        """
        for item in cls._response_output(response):
            if cls._get_attr_or_key(item, "type", "") != "message":
                continue
            for part in cls._get_attr_or_key(item, "content", []) or []:
                if cls._get_attr_or_key(part, "type", "") in (
                    "output_text",
                    "refusal",
                ):
                    return True
        return False

    @classmethod
    def _parse_non_streaming(
        cls,
        response: Any,
    ) -> tuple[str, list[dict[str, str]]]:
        """Extract text + tool calls from a non-streaming Responses object.

        Args:
            response: The object returned by
                ``client.responses.create(stream=False)``.

        Returns:
            ``(content, tool_calls)`` where ``content`` is the concatenated
            ``output_text`` from every assistant message and ``tool_calls``
            is a list of ``{"id","name","arguments"}`` dicts.
        """
        text_chunks: list[str] = []
        tool_calls: list[dict[str, str]] = []
        for output_index, item in enumerate(cls._response_output(response)):
            itype = cls._get_attr_or_key(item, "type", "")
            if itype == "message":
                for part in cls._get_attr_or_key(item, "content", []) or []:
                    ptype = cls._get_attr_or_key(part, "type", "")
                    if ptype == "output_text":
                        text_chunks.append(
                            cls._get_attr_or_key(part, "text", "") or ""
                        )
                    elif ptype == "refusal":
                        text_chunks.append(
                            cls._get_attr_or_key(part, "refusal", "") or ""
                        )
            elif itype == "function_call":
                tool_calls.append(
                    {
                        "id": str(
                            cls._get_attr_or_key(item, "call_id", "") or ""
                        ),
                        "name": str(
                            cls._get_attr_or_key(item, "name", "") or ""
                        ),
                        "arguments": str(
                            cls._get_attr_or_key(item, "arguments", "") or ""
                        ),
                        "item_id": str(
                            cls._get_attr_or_key(item, "id", "") or ""
                        ),
                        "output_index": str(output_index),
                    }
                )
        return "".join(text_chunks), tool_calls

    @classmethod
    def _response_output_items_to_input_items(
        cls,
        response: Any,
    ) -> list[dict[str, Any]]:
        """Serialize ``response.output`` items into Responses ``input`` items.

        The Responses API recommends replaying prior ``response.output``
        items verbatim as the next request's ``input`` array when not
        using ``previous_response_id``.  This preserves reasoning items,
        message items, function_call items, and any other future item
        type — in their original order — so reasoning models that emit
        ``reasoning``/``function_call`` pairs keep working across turns.

        Args:
            response: The raw Responses-API object (or the ``response``
                field of the terminating ``response.completed`` event).

        Returns:
            A list of dicts ready to append to ``self.conversation``.
            Empty when ``response.output`` is absent or empty.
        """
        items: list[dict[str, Any]] = []
        for item in cls._response_output(response):
            if hasattr(item, "model_dump"):
                try:
                    d = item.model_dump(exclude_none=True)
                except Exception:  # noqa: BLE001 - SDK can raise on partials
                    d = None
                    logger.debug(
                        "Failed to model_dump response.output item", exc_info=True
                    )
            elif isinstance(item, dict):
                d = dict(item)
            else:
                d = None
            if isinstance(d, dict):
                items.append(d)
        return items

    @staticmethod
    def _build_function_calls(
        tool_calls: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Turn raw ``{"id","name","arguments_json"}`` dicts into call entries.

        Args:
            tool_calls: Raw tool-call dicts collected from streaming or
                non-streaming responses.

        Returns:
            A list of ``{"id","name","arguments":<dict>}`` dicts where
            ``arguments`` is the JSON-decoded argument object (or ``{}``
            if decoding fails).
        """
        out: list[dict[str, Any]] = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.get("arguments") or "{}")
            except json.JSONDecodeError:
                logger.debug("Failed to decode tool-call args", exc_info=True)
                args = {}
            out.append(
                {"id": tc.get("id", ""), "name": tc.get("name", ""), "arguments": args}
            )
        return out

    def generate(self) -> tuple[str, Any]:
        """Generate a response with no tools.

        Returns:
            ``(content, response)`` where ``content`` is the assistant
            text and ``response`` is the raw Responses-API object (or the
            final ``response`` field of the terminating ``response.completed``
            event when streaming).
        """
        self.conversation = self._foreign_items_to_native_input(self.conversation)
        kwargs = self._build_request_kwargs(tools=None)
        content, _tc, response = self._request_response(kwargs)

        if self._is_deepseek_reasoning_model():
            _, content = _extract_deepseek_reasoning(content)
            if content.strip():
                self.conversation.append(
                    {"role": "assistant", "content": content}
                )
            return content, response

        raw_items = self._response_output_items_to_input_items(response)
        if raw_items:
            self.conversation.extend(raw_items)
            if content.strip() and not self._raw_items_have_message_text(
                raw_items
            ):
                self.conversation.append(
                    {"role": "assistant", "content": content}
                )
        else:
            self.conversation.append({"role": "assistant", "content": content})
        return content, response

    def generate_and_process_with_tools(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generate a response and parse any tool calls returned by the model.

        Unlike v1, ``reasoning_effort`` is NOT stripped when tools are
        attached — the Responses API supports the combination natively.

        Args:
            function_map: Mapping of tool name → callable.  Used for
                schema generation when ``tools_schema`` is not supplied
                and as the DeepSeek fallback's tool registry.
            tools_schema: Optional pre-built tool schema (Chat-Completions
                or Responses shape; flattened automatically).

        Returns:
            ``(function_calls, content, response)`` matching the v1 contract.
        """
        self.conversation = self._foreign_items_to_native_input(self.conversation)
        if self._is_deepseek_reasoning_model():
            return self._generate_with_text_based_tools(function_map)

        chat_tools = self._resolve_openai_tools_schema(function_map, tools_schema)
        responses_tools = self._flatten_tools_schema(chat_tools)
        kwargs = self._build_request_kwargs(tools=responses_tools)

        content, raw_tool_calls, response = self._request_response(kwargs)

        for tc in raw_tool_calls:
            if not tc.get("id"):
                raise KISSError(
                    f"Responses API returned function_call without call_id: {tc!r}"
                )
            if not tc.get("name"):
                raise KISSError(
                    f"Responses API returned function_call without name: {tc!r}"
                )

        function_calls = self._build_function_calls(raw_tool_calls)

        raw_items = self._response_output_items_to_input_items(response)
        if raw_items:
            tool_by_item_id = {
                tc.get("item_id"): tc
                for tc in raw_tool_calls
                if tc.get("item_id")
            }
            tool_by_call_id = {
                tc.get("id"): tc for tc in raw_tool_calls if tc.get("id")
            }
            used_patches: set[int] = set()
            stale_indexes: list[int] = []
            for raw_i, item in enumerate(raw_items):
                if (
                    not isinstance(item, dict)
                    or item.get("type") != "function_call"
                ):
                    continue
                patch: dict[str, str] | None = tool_by_item_id.get(
                    item.get("id")
                ) or tool_by_call_id.get(item.get("call_id"))
                if patch is None:
                    if not item.get("call_id") and not item.get("name"):
                        for tc_idx, tc in enumerate(raw_tool_calls):
                            if tc_idx in used_patches:
                                continue
                            patch = tc
                            used_patches.add(tc_idx)
                            break
                else:
                    for tc_idx, tc in enumerate(raw_tool_calls):
                        if tc is patch:
                            used_patches.add(tc_idx)
                            break
                if patch is None:
                    if not item.get("call_id") or not item.get("name"):
                        stale_indexes.append(raw_i)
                    continue
                if patch.get("id"):
                    item["call_id"] = patch["id"]
                if patch.get("name"):
                    item["name"] = patch["name"]
                if "arguments" in patch:
                    item["arguments"] = patch["arguments"]
                if patch.get("item_id") and not item.get("id"):
                    item["id"] = patch["item_id"]
                if not item.get("call_id") or not item.get("name"):
                    stale_indexes.append(raw_i)
            for raw_i in reversed(stale_indexes):
                del raw_items[raw_i]
            existing_call_ids = {
                item.get("call_id")
                for item in raw_items
                if isinstance(item, dict) and item.get("type") == "function_call"
            }
            existing_item_ids = {
                item.get("id")
                for item in raw_items
                if isinstance(item, dict) and item.get("type") == "function_call"
            }
            item_output_indexes = dict(self._last_stream_item_indexes)
            combined: list[tuple[int, int, dict[str, Any]]] = []
            used_orders: set[int] = set()
            unknown_raw: list[tuple[int, dict[str, Any]]] = []
            for fallback_i, item in enumerate(raw_items):
                item_id = str(item.get("id", "") or "")
                known_order = item_output_indexes.get(item_id)
                if known_order is None:
                    unknown_raw.append((fallback_i, item))
                else:
                    used_orders.add(known_order)
                    combined.append((known_order, fallback_i, item))
            stream_only_orders_list: list[tuple[int, int, dict[str, Any]]] = []
            for tc_i, tc in enumerate(raw_tool_calls):
                if (
                    tc.get("id") in existing_call_ids
                    or tc.get("item_id") in existing_item_ids
                ):
                    continue
                fc_item: dict[str, Any] = {
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", ""),
                }
                item_id = tc.get("item_id", "")
                if item_id:
                    fc_item["id"] = item_id
                resolved_order = item_output_indexes.get(item_id)
                if resolved_order is None:
                    try:
                        resolved_order = int(tc.get("output_index", "") or "")
                    except (ValueError, TypeError):
                        resolved_order = len(raw_items)
                stream_only_orders_list.append(
                    (resolved_order, len(raw_items) + tc_i, fc_item)
                )
                used_orders.add(resolved_order)
            next_order = 0
            for fallback_i, item in unknown_raw:
                while next_order in used_orders:
                    next_order += 1
                used_orders.add(next_order)
                combined.append((next_order, fallback_i, item))
                next_order += 1
            combined.extend(stream_only_orders_list)
            if content.strip() and not any(
                self._raw_items_have_message_text([entry[2]])
                for entry in combined
            ):
                msg_order = self._last_stream_message_output_index
                if msg_order is None:
                    msg_order = (
                        max(used_orders) + 1 if used_orders else len(raw_items)
                    )
                used_orders.add(msg_order)
                combined.append(
                    (
                        msg_order,
                        -1,
                        {"role": "assistant", "content": content},
                    )
                )
            combined.sort(key=lambda entry: (entry[0], entry[1]))
            merged_items: list[dict[str, Any]] = [
                item for _order, _tie, item in combined
            ]
            self.conversation.extend(merged_items)
        else:
            if content:
                self.conversation.append({"role": "assistant", "content": content})
            for tc in raw_tool_calls:
                fc_item = {
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", ""),
                }
                item_id = tc.get("item_id", "")
                if item_id:
                    fc_item["id"] = item_id
                self.conversation.append(fc_item)
        self._pending_function_calls = [
            {"name": tc.get("name", ""), "call_id": tc.get("id", "")}
            for tc in raw_tool_calls
        ]
        return function_calls, content, response

    def _generate_with_text_based_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Fallback for DeepSeek R1: parse tool calls from text output.

        Args:
            function_map: Mapping of tool name → callable used to render
                the tool-description prompt injected ahead of the user
                message.

        Returns:
            Same shape as :meth:`generate_and_process_with_tools`.
        """
        tools_prompt = _build_text_based_tools_prompt(function_map)

        modified = list(self.conversation)
        if modified:
            first = modified[0]
            if isinstance(first, dict) and first.get("role") == "user":
                parts = first.get("content")
                if isinstance(parts, list):
                    new_parts = [
                        dict(p) if isinstance(p, dict) else p for p in parts
                    ]
                    found = False
                    for p in new_parts:
                        if isinstance(p, dict) and p.get("type") == "input_text":
                            p["text"] = (p.get("text", "") or "") + "\n" + tools_prompt
                            found = True
                            break
                    if not found:
                        new_parts.append(
                            {"type": "input_text", "text": tools_prompt}
                        )
                    modified[0] = {"role": "user", "content": new_parts}
                elif isinstance(parts, str):
                    modified[0] = {
                        "role": "user",
                        "content": parts + "\n" + tools_prompt,
                    }
            else:
                modified.insert(
                    0,
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": tools_prompt}
                        ],
                    },
                )

        self._ensure_no_pending_function_calls()
        kwargs = self._shape_responses_kwargs(input_items=modified, tools=None)

        content, _tc, response = self._request_response(kwargs)

        _, content_clean = _extract_deepseek_reasoning(content)
        function_calls = _parse_text_based_tool_calls(content_clean)

        if content_clean.strip():
            self.conversation.append(
                {"role": "assistant", "content": content_clean}
            )
        for fc in function_calls:
            self.conversation.append(
                {
                    "type": "function_call",
                    "call_id": fc["id"],
                    "name": fc["name"],
                    "arguments": json.dumps(fc.get("arguments", {})),
                }
            )
        self._pending_function_calls = [
            {"name": fc["name"], "call_id": fc["id"]} for fc in function_calls
        ]
        return function_calls, content, response

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Append ``function_call_output`` items for each tool result.

        Binary attachments embedded in tool result strings (the
        ``<<KISS_BINARY_ATTACHMENT ...>>`` sentinel) are lifted out of the
        function-call-output text and re-attached to a follow-up user
        message as ``input_image`` / ``input_file`` content parts so the
        model can actually see the bytes.

        Args:
            function_results: ``(function_name, result_dict)`` tuples in
                the same order as the preceding function_call items in
                the conversation.
        """
        self.conversation = self._foreign_items_to_native_input(self.conversation)
        conv_snapshot_len = len(self.conversation)
        pending_snapshot = list(self._pending_function_calls)
        try:
            self._add_function_results_inner(function_results)
        except Exception:
            self.conversation = self.conversation[:conv_snapshot_len]
            self._pending_function_calls = pending_snapshot
            raise

    def _add_function_results_inner(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Inner mutating implementation guarded by atomic snapshots."""
        trailing = self._trailing_function_call_ids()
        fallback_unanswered = (
            self._unanswered_function_calls_from_conversation_with_names()
            if not self._pending_function_calls
            else None
        )
        for i, (func_name, result_dict) in enumerate(function_results):
            result_content, attachments = self.tool_result_text_and_attachments(
                result_dict
            )
            # The pending call is consumed either way, so the bookkeeping
            # that guards the next turn stays correct even when the caller
            # names the id itself.
            matched_call_id = self._consume_pending_call_id(
                func_name, i, trailing, fallback_unanswered
            )
            explicit_call_id = result_dict.get("tool_use_id")
            call_id = str(explicit_call_id) if explicit_call_id else matched_call_id
            self.conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result_content,
                }
            )
            for att in attachments:
                self.conversation.append(
                    {
                        "type": "_kiss_pending_tool_result_attachment",
                        "mime_type": att.mime_type,
                        "data": base64.b64encode(att.data).decode("ascii"),
                    }
                )

        sentinel_indexes = [
            i
            for i, item in enumerate(self.conversation)
            if isinstance(item, dict)
            and item.get("type") == "_kiss_pending_tool_result_attachment"
        ]

        if (
            not self._pending_function_calls
            and not self._unanswered_function_calls_from_conversation()
            and sentinel_indexes
        ):
            sentinel_attachments = [
                Attachment(
                    data=base64.b64decode(
                        str(self.conversation[i].get("data", "") or "")
                    ),
                    mime_type=str(
                        self.conversation[i].get("mime_type", "") or ""
                    ),
                )
                for i in sentinel_indexes
            ]
            for i in reversed(sentinel_indexes):
                del self.conversation[i]
            parts = self._attachments_to_content_parts(sentinel_attachments)
            if parts:
                parts.append(
                    {
                        "type": "input_text",
                        "text": "[attachments from previous tool result(s)]",
                    }
                )
                self.conversation.append({"role": "user", "content": parts})

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int]:
        """Extract Responses-API token counts.

        The Responses API reports usage as
        ``input_tokens`` / ``output_tokens`` (not ``prompt_tokens`` /
        ``completion_tokens``), with cached tokens under
        ``input_tokens_details.cached_tokens`` and reasoning tokens under
        ``output_tokens_details.reasoning_tokens``.  ``output_tokens``
        already counts reasoning tokens, so we return it as-is for the
        output count.

        Args:
            response: The raw response object.

        Returns:
            ``(input_tokens, output_tokens, cache_read_tokens,
            cache_write_tokens)``.  ``cache_write_tokens`` is 0 for
            api.openai.com (no cache write fees) but is passed through
            when a gateway (e.g. OpenRouter Anthropic passthrough)
            reports ``input_tokens_details.cache_write_tokens``.
        """
        usage = self._get_attr_or_key(response, "usage")
        if usage is None:
            return 0, 0, 0, 0
        input_tokens = int(self._get_attr_or_key(usage, "input_tokens", 0) or 0)
        output_tokens = int(
            self._get_attr_or_key(usage, "output_tokens", 0) or 0
        )
        cached_tokens = 0
        cache_write_tokens = 0
        details = self._get_attr_or_key(usage, "input_tokens_details")
        if details is not None:
            cached_tokens = int(
                self._get_attr_or_key(details, "cached_tokens", 0) or 0
            )
            cache_write_tokens = int(
                self._get_attr_or_key(details, "cache_write_tokens", 0) or 0
            )
        return (
            max(0, input_tokens - cached_tokens - cache_write_tokens),
            output_tokens,
            cached_tokens,
            cache_write_tokens,
        )
