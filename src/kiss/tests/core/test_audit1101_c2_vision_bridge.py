# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The vision bridge's billing, budget, capability and audio findings of audit C2.

* F2: a Stop (``KeyboardInterrupt``) during a description call left what
  the helper had already been billed unreported: the calling model never
  received the spend, so the task's totals missed it.
* F3: every attachment of a tool result was described even after the
  first description had exhausted the agent's budget.
* F5: the text-only members of a vision family (``gpt-oss``, ``gemma-2``,
  ``gpt-audio``) were taken to see images, so the bytes went to an
  endpoint that accepts text only.
* F6: every attachment of a text-only model went through the bridge,
  audio included, so a clip a non-vision audio model (Voxtral) could
  hear was dropped with "only images and PDFs are described".

The vision endpoint is a real HTTP server speaking Anthropic SSE to the
real SDK; the text-only model is played by the scripted local chat
server; no code is patched.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core import config as config_module
from kiss.core import stop_signal
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import BudgetExceededError
from kiss.core.models.model import encode_binary_attachment
from kiss.core.models.model_families import model_supports_images
from kiss.core.models.model_info import calculate_cost
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.vision_bridge import vision_model
from kiss.tests.agents.sorcar.local_model_server import serve, tool_call_body
from kiss.tests.core.models.anthropic_sse_harness import sse
from kiss.tests.core.test_vision_bridge_2026_10_31 import (
    LOCAL_VISION_MODEL,
    TEXT_ONLY_MODEL,
    VISION_COST,
    VISION_INPUT_TOKENS,
    VISION_OUTPUT_TOKENS,
    _blank_every_provider_key,
    _vision_turn,
)

AUDIO_MODEL = "openrouter/mistralai/voxtral-small-24b-2507"
"""A catalog model that hears audio but cannot see images."""

STOPPED_TOKENS = VISION_INPUT_TOKENS + 1
STOPPED_COST = calculate_cost(LOCAL_VISION_MODEL, VISION_INPUT_TOKENS, 1)
"""What a description stopped right after ``message_start`` is billed (its usage so far)."""

DEADLINE = 20.0
"""Seconds to wait for the vision stream to stall, and for the stopped call to unwind."""

PNG = b"\x89PNG\r\n\x1a\n" + b"not-a-real-picture"
WAV = b"RIFF\x00\x00\x00\x00WAVEfmt "


def _result_with(*attachments: tuple[str, bytes]) -> dict[str, Any]:
    """Return a tool result carrying *attachments* as ``(mime, bytes)`` pairs."""
    return {"result": "\n".join(encode_binary_attachment(mime, data) for mime, data in attachments)}


def _stall_prefix() -> bytes:
    """The start of a streamed turn: ``message_start`` (with its usage) and an open text block."""
    message: dict[str, Any] = {
        "id": "msg_stalled",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": LOCAL_VISION_MODEL,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": VISION_INPUT_TOKENS, "output_tokens": 1},
    }
    block = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    return sse("message_start", {"type": "message_start", "message": message}) + sse(
        "content_block_start", block
    )


class _VisionHandler(BaseHTTPRequestHandler):
    """Answer the n-th request with the n-th scripted reply; ``None`` stalls after message_start."""

    protocol_version = "HTTP/1.1"
    replies: list[bytes | None]
    requests: list[dict[str, Any]]
    stalling: threading.Event
    release: threading.Event
    lock: threading.Lock

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        with self.lock:
            self.requests.append(json.loads(raw or b"{}"))
            reply = self.replies[min(len(self.requests), len(self.replies)) - 1]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        if reply is not None:
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)
            self.wfile.flush()
            return
        prefix = _stall_prefix()
        self.send_header("Content-Length", str(len(prefix) + 1_000_000))
        self.end_headers()
        self.wfile.write(prefix)
        self.wfile.flush()
        self.stalling.set()
        while not self.release.wait(timeout=0.1):
            try:
                self.wfile.write(sse("ping", {"type": "ping"}))
                self.wfile.flush()
            except OSError:
                break
        self.close_connection = True
        self.connection.close()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the default stderr access log."""


class _DaemonServer(ThreadingHTTPServer):
    daemon_threads = True


class VisionEndpoint:
    """A real Anthropic endpoint for the bridge, with per-request scripted replies."""

    def __init__(self, replies: list[bytes | None]) -> None:
        self.requests: list[dict[str, Any]] = []
        self.stalling = threading.Event()
        self.release = threading.Event()
        handler = type(
            "_Handler",
            (_VisionHandler,),
            {
                "replies": list(replies),
                "requests": self.requests,
                "stalling": self.stalling,
                "release": self.release,
                "lock": threading.Lock(),
            },
        )
        self._server = _DaemonServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}"

    def close(self) -> None:
        self.release.set()
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def described_turn() -> bytes:
    """One complete scripted description, at the scripted usage."""
    return b"".join(_vision_turn("A white PNG reading KISS 42."))


@contextlib.contextmanager
def _vision_endpoint(
    monkeypatch: pytest.MonkeyPatch, replies: list[bytes | None]
) -> Iterator[VisionEndpoint]:
    """Serve *replies* and make the server the bridge's vision endpoint.

    Only the Anthropic key is set, so the bridge picks
    ``claude-sonnet-5-5``, whose SDK client reads ``ANTHROPIC_BASE_URL``.
    """
    _blank_every_provider_key(monkeypatch)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "sk-ant-local-test")
    endpoint = VisionEndpoint(replies)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", endpoint.base_url)
    assert vision_model() == LOCAL_VISION_MODEL
    try:
        yield endpoint
    finally:
        endpoint.close()


def _text_only_adapter(model_name: str = TEXT_ONLY_MODEL) -> OpenAICompatibleModel:
    """An adapter of *model_name* that is never sent a chat request."""
    return OpenAICompatibleModel(
        model_name=model_name, api_key="dummy", base_url="http://127.0.0.1:9/v1"
    )


# --- F5: text-only members of a vision family -------------------------------


@pytest.mark.parametrize(
    "model_name",
    [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b-high",
        "openrouter/openai/gpt-oss-120b",
        "openrouter/openai/gpt-oss-safeguard-20b-low",
        "google/gemma-2-27b-it",
        "openrouter/google/gemma-2-27b-it",
        "gpt-audio",
        "openrouter/openai/gpt-audio-mini",
    ],
)
def test_text_only_members_of_a_vision_family_do_not_see(model_name: str) -> None:
    """GPT-OSS, Gemma 2 and gpt-audio accept no image, whatever the route or thinking alias."""
    assert model_supports_images(model_name) is False


@pytest.mark.parametrize(
    "model_name",
    ["gpt-6-luna", "openrouter/google/gemma-3-27b-it", "google/gemma-4-31B-it", "gpt-5"],
)
def test_the_seeing_members_still_see(model_name: str) -> None:
    """The exclusions are narrow: the families' vision models are unaffected."""
    assert model_supports_images(model_name) is True


def test_a_tool_image_for_gpt_oss_goes_through_the_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registered GPT-OSS model gets the description path, not the raw image part."""
    _blank_every_provider_key(monkeypatch)
    adapter = _text_only_adapter("openrouter/openai/gpt-oss-120b")
    assert not adapter.supports_images()
    text, attachments = adapter.tool_result_text_and_attachments(_result_with(("image/png", PNG)))
    assert attachments == []
    assert "[image/png attachment 1 could not be described: no vision model is configured" in text


# --- F6: audio stays for the adapter ------------------------------------------


def test_audio_for_a_non_vision_audio_model_stays_for_the_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Voxtral cannot see, but its clip is not the bridge's business: it reaches the adapter."""
    _blank_every_provider_key(monkeypatch)
    adapter = _text_only_adapter(AUDIO_MODEL)
    assert not adapter.supports_images()
    text, attachments = adapter.tool_result_text_and_attachments(_result_with(("audio/wav", WAV)))
    assert [(a.mime_type, a.data) for a in attachments] == [("audio/wav", WAV)]
    assert "could not be described" not in text
    # Through the adapter's own path the clip becomes an input_audio part.
    adapter.conversation = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "listen", "arguments": "{}"},
                }
            ],
        }
    ]
    adapter.add_function_results_to_conversation_and_return(
        [("listen", _result_with(("audio/wav", WAV)))]
    )
    parts = adapter.conversation[-1]["content"]
    assert adapter.conversation[-1]["role"] == "user"
    assert [p["type"] for p in parts][0] == "input_audio"
    assert parts[0]["input_audio"]["format"] == "wav"


def test_a_mixed_result_describes_the_image_and_keeps_the_audio(
    monkeypatch: pytest.MonkeyPatch, described_turn: bytes
) -> None:
    """Of an image and a clip, the image is described and the clip is kept."""
    with _vision_endpoint(monkeypatch, [described_turn]) as endpoint:
        adapter = _text_only_adapter(AUDIO_MODEL)
        text, attachments = adapter.tool_result_text_and_attachments(
            _result_with(("image/png", PNG), ("audio/wav", WAV))
        )
    assert len(endpoint.requests) == 1
    assert [a.mime_type for a in attachments] == ["audio/wav"]
    assert f"[Description of image/png attachment 1 by {LOCAL_VISION_MODEL}; " in text
    assert "KISS 42" in text
    assert "could not be described" not in text
    _epoch, cost, tokens = adapter.helper_usage()
    assert (cost, tokens) == (
        pytest.approx(VISION_COST),
        VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS,
    )


# --- F3: a batch stops at the budget cap --------------------------------------


def test_a_batch_of_descriptions_stops_at_the_budget_cap(
    monkeypatch: pytest.MonkeyPatch, described_turn: bytes
) -> None:
    """The third of three images is not described once the second crossed the cap.

    The cap sits above one chat call plus one description and below two
    descriptions: the first description is allowed, the second crosses
    the cap, and no third vision request may be made.  The run ends with
    the budget error before the next chat call, with both descriptions
    billed.
    """

    def read_images() -> str:
        """Return three screenshots in one result."""
        return "\n".join(encode_binary_attachment("image/png", PNG) for _ in range(3))

    script = [
        tool_call_body("read_images", {}, prompt_tokens=100),
        tool_call_body("finish", {"result": "done"}, prompt_tokens=100),
    ]
    chat_call_cost = calculate_cost(TEXT_ONLY_MODEL, 100, 100)
    cap = chat_call_cost + 1.5 * VISION_COST
    agent = KISSAgent("vision-bridge-batch-cap")
    with (
        _vision_endpoint(monkeypatch, [described_turn]) as endpoint,
        serve(script) as (url, requests),
    ):
        with pytest.raises(BudgetExceededError):
            agent.run(
                model_name=TEXT_ONLY_MODEL,
                prompt_template="Read the images and report what they show.",
                tools=[read_images],
                model_config={"base_url": url, "api_key": "local"},
                max_budget=cap,
                max_steps=5,
                verbose=False,
                print_prompts=False,
            )
    assert len(endpoint.requests) == 2, "a description was requested past the cap"
    assert len(requests) == 1, "the agent made another chat call past its cap"
    assert agent.budget_used == pytest.approx(chat_call_cost + 2 * VISION_COST)
    assert agent.total_tokens_used == 200 + 2 * (VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS)
    # The model's cumulative helper record is exactly what the totals above billed.
    _epoch, helper_cost, helper_tokens = agent.model.helper_usage()
    assert helper_cost == pytest.approx(2 * VISION_COST)
    assert helper_tokens == 2 * (VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS)


# --- F2: a Stop keeps the helper's bill ---------------------------------------


def _split_until_stopped(
    adapter: OpenAICompatibleModel, result: dict[str, Any], endpoint: VisionEndpoint
) -> BaseException | None:
    """Run the split on a worker thread and press Stop once the vision stream stalls.

    Returns:
        What the split raised, or ``None`` when it returned.
    """
    stop = threading.Event()
    outcome: list[BaseException | None] = []

    def target() -> None:
        stop_signal.set_thread_stop_event(stop)
        try:
            adapter.tool_result_text_and_attachments(result)
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome.append(exc)
        else:
            outcome.append(None)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    assert endpoint.stalling.wait(DEADLINE), "the vision stream never stalled"
    stop.set()
    worker.join(DEADLINE)
    assert not worker.is_alive(), f"the split was still running {DEADLINE}s after Stop"
    return outcome[0]


def test_a_stop_during_the_first_description_keeps_its_bill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopped after the provider reported usage, the call's spend reaches the calling model."""
    with _vision_endpoint(monkeypatch, [None]) as endpoint:
        adapter = _text_only_adapter()
        raised = _split_until_stopped(adapter, _result_with(("image/png", PNG)), endpoint)
    assert isinstance(raised, KeyboardInterrupt), repr(raised)
    assert len(endpoint.requests) == 1
    _epoch, cost, tokens = adapter.helper_usage()
    assert (cost, tokens) == (pytest.approx(STOPPED_COST), STOPPED_TOKENS)
    assert cost > 0


def test_a_stop_during_a_later_description_keeps_every_bill(
    monkeypatch: pytest.MonkeyPatch, described_turn: bytes
) -> None:
    """A completed description and the stopped one after it are both billed; no third is made."""
    with _vision_endpoint(monkeypatch, [described_turn, None]) as endpoint:
        adapter = _text_only_adapter()
        raised = _split_until_stopped(
            adapter,
            _result_with(("image/png", PNG), ("image/png", PNG), ("image/png", PNG)),
            endpoint,
        )
    assert isinstance(raised, KeyboardInterrupt), repr(raised)
    assert len(endpoint.requests) == 2
    _epoch, cost, tokens = adapter.helper_usage()
    assert cost == pytest.approx(VISION_COST + STOPPED_COST)
    assert tokens == VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS + STOPPED_TOKENS
