# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: both OpenAI transports keep the same attachments.

``OpenAICompatibleModel`` (Chat Completions) and
``OpenAICompatibleModel2`` (Responses) serve the same providers, and which
one handles a turn is decided inside the framework — a tool-bearing turn
can be delegated to the Responses transport without the caller asking for
it.  If the two disagree about which attachment formats they carry, the
same photo or voice memo reaches the model on one turn and vanishes on the
next, for a reason no caller can see or control.

The accepted formats are OpenAI's, not a transport's:

* images — PNG, JPEG, WEBP and non-animated GIF ("Image input
  requirements", https://developers.openai.com/api/docs/guides/images-vision,
  which also states the behaviour "is the same in both the Responses API
  and the Chat Completions API");
* audio — ``mp3`` and ``wav``, the only values the SDK's own
  ``input_audio.format`` literal allows in *both*
  ``chat_completion_content_part_input_audio_param.py`` and
  ``response_input_audio_param.py``;
* PDFs, which both APIs accept (in their own content-part shapes).

Each attachment here carries unique bytes, so "did this attachment reach
the provider?" is answered by looking for its base64 payload in the JSON
body a real local HTTP server received from the real OpenAI SDK — a check
that works across the two different content-part shapes.  No mocks,
patches or doubles.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import pytest

from kiss.core.models.model import Attachment
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
)

_MODEL = "attachment-parity-under-test"


def _chat_reply(request: Request) -> Reply:
    """Answer a Chat Completions call with one short assistant message."""
    return Reply(
        json_body={
            "id": "chatcmpl-attachments",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
        }
    )


def _responses_reply(request: Request) -> Reply:
    """Answer a Responses call with one short assistant message."""
    return Reply(
        json_body={
            "id": "resp_attachments",
            "object": "response",
            "created_at": 0,
            "model": _MODEL,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "ok", "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 4,
            },
        }
    )


def _attachment(mime_type: str, marker: str) -> Attachment:
    """Return an attachment whose bytes identify it uniquely.

    Args:
        mime_type: The MIME type to label the bytes with.
        marker: Text embedded in the payload so the base64 of this
            attachment cannot be confused with another's.

    Returns:
        The attachment.
    """
    return Attachment(data=f"kiss-{marker}-payload".encode(), mime_type=mime_type)


def _was_sent(body: dict[str, Any], att: Attachment) -> bool:
    """Return whether *att*'s bytes appear anywhere in a request body.

    Args:
        body: The decoded JSON request body the provider received.
        att: The attachment to look for.

    Returns:
        ``True`` when the attachment's base64 payload is present, whatever
        content-part shape the transport used to carry it.
    """
    return base64.b64encode(att.data).decode("ascii") in json.dumps(body)


def chat_body_with(attachments: list[Attachment]) -> dict[str, Any]:
    """Run one real Chat Completions turn carrying *attachments*.

    Args:
        attachments: The attachments to initialize the conversation with.

    Returns:
        The JSON body the local endpoint received.
    """
    with ScriptedOpenAIServer(_chat_reply) as server:
        model = OpenAICompatibleModel(
            _MODEL, base_url=server.base_url, api_key="test-key"
        )
        model.initialize("What is in these files?", attachments=attachments)
        model.generate()
        return server.requests[-1].body


def responses_body_with(attachments: list[Attachment]) -> dict[str, Any]:
    """Run one real Responses turn carrying *attachments*.

    Args:
        attachments: The attachments to initialize the conversation with.

    Returns:
        The JSON body the local endpoint received.
    """
    with ScriptedOpenAIServer(_responses_reply) as server:
        model = OpenAICompatibleModel2(
            _MODEL, base_url=server.base_url, api_key="test-key"
        )
        model.initialize("What is in these files?", attachments=attachments)
        model.generate()
        return server.requests[-1].body


class TestBothTransportsMakeTheSameDecision:
    """The routing choice must not change which attachments are carried."""

    @pytest.mark.parametrize(
        ("mime_type", "expected"),
        [
            ("image/png", True),
            ("image/jpeg", True),
            ("image/webp", True),
            ("image/gif", True),
            ("application/pdf", True),
            ("audio/mpeg", True),
            ("audio/wav", True),
            ("image/heic", False),
            ("image/bmp", False),
            ("audio/ogg", False),
            ("audio/flac", False),
            ("video/mp4", False),
        ],
    )
    def test_transports_agree(self, mime_type: str, expected: bool) -> None:
        """Each format is either carried by both transports or by neither."""
        att = _attachment(mime_type, mime_type.replace("/", "-"))
        chat_sent = _was_sent(chat_body_with([att]), att)
        responses_sent = _was_sent(responses_body_with([att]), att)
        assert chat_sent == responses_sent, (
            f"{mime_type}: chat completions sent={chat_sent}, "
            f"responses sent={responses_sent}"
        )
        assert chat_sent is expected


class TestDroppedAttachmentsAreReported:
    """A dropped attachment must be announced, never silently discarded."""

    def test_chat_completions_reports_an_unsupported_image(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Chat Completions must name the image format it refuses."""
        att = _attachment("image/heic", "heic-image")
        with caplog.at_level(logging.WARNING):
            body = chat_body_with([att])
        assert not _was_sent(body, att)
        assert "image/heic" in caplog.text

    def test_chat_completions_reports_an_unsupported_audio_format(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Chat Completions must name the audio format it refuses."""
        att = _attachment("audio/ogg", "ogg-audio")
        with caplog.at_level(logging.WARNING):
            body = chat_body_with([att])
        assert not _was_sent(body, att)
        assert "audio/ogg" in caplog.text

    def test_responses_reports_an_unsupported_audio_format(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The Responses transport must name the audio format it refuses."""
        att = _attachment("audio/ogg", "ogg-audio")
        with caplog.at_level(logging.WARNING):
            body = responses_body_with([att])
        assert not _was_sent(body, att)
        assert "audio/ogg" in caplog.text


class TestSupportedAttachmentsKeepTheirTransportShape:
    """Agreeing on the decision must not flatten the two wire formats."""

    def test_chat_completions_uses_image_url_and_input_audio(self) -> None:
        """Chat Completions carries images as ``image_url`` parts."""
        image = _attachment("image/png", "png-shape")
        audio = _attachment("audio/wav", "wav-shape")
        body = chat_body_with([image, audio])
        parts = body["messages"][-1]["content"]
        kinds = [part["type"] for part in parts]
        assert "image_url" in kinds
        assert "input_audio" in kinds
        assert _was_sent(body, image) and _was_sent(body, audio)

    def test_responses_uses_input_image_and_input_audio(self) -> None:
        """The Responses transport carries images as ``input_image`` parts."""
        image = _attachment("image/png", "png-shape")
        audio = _attachment("audio/wav", "wav-shape")
        body = responses_body_with([image, audio])
        parts = body["input"][-1]["content"]
        kinds = [part["type"] for part in parts]
        assert "input_image" in kinds
        assert "input_audio" in kinds
        assert _was_sent(body, image) and _was_sent(body, audio)
