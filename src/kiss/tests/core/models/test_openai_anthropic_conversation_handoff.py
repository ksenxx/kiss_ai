# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for Anthropic → OpenAI conversation hand-off.

When the Sorcar ``set_model`` tool switches a live agent from an Anthropic
model (e.g. ``claude-*``) to an OpenAI model (e.g. ``gpt-5.5``), the raw
conversation is handed over: ``new_model.conversation = old_model.conversation``.
An Anthropic conversation stores assistant messages as content-block lists
containing ``thinking`` / ``tool_use`` blocks and user messages containing
``tool_result`` blocks.  Replaying those verbatim to the OpenAI Chat
Completions API fails with::

    Error code: 400 - {'error': {'message': "Invalid value: 'thinking'.
    Supported values are: 'text', 'image_url', 'input_audio', 'refusal',
    'audio', and 'file'.", ... 'param': 'messages[1].content[0].type', ...}}

These tests verify that ``OpenAICompatibleModel`` converts such hand-off
conversations to OpenAI format before every API call.
"""

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core.models.model_info import model
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.conftest import requires_openai_api_key

# An Anthropic-format conversation exactly as AnthropicModel stores it:
# assistant content is a block list with thinking + tool_use blocks, and the
# tool result comes back as a user message with a tool_result block.
_ANTHROPIC_STYLE_CONVERSATION: list[dict[str, Any]] = [
    {"role": "user", "content": "Use the add tool to compute 2 + 3, then report the sum."},
    {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user wants 2 + 3. I should call the add tool.",
                "signature": "sig-abc123",
            },
            {"type": "text", "text": "Let me add those numbers."},
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "add",
                "input": {"a": 2, "b": 3},
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_01", "content": "5"},
        ],
    },
]


def _add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First addend.
        b: Second addend.

    Returns:
        The sum of a and b.
    """
    return a + b


def _make_offline_model() -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        model_name="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
    )


class TestHandoffNormalization:
    """Offline behavior of the conversation normalizer on Anthropic input."""

    def test_thinking_blocks_are_dropped(self) -> None:
        m = _make_offline_model()
        normalized = m._normalize_conversation_for_api(_ANTHROPIC_STYLE_CONVERSATION)
        for msg in normalized:
            content = msg.get("content")
            if isinstance(content, list):
                assert all(
                    b.get("type") not in ("thinking", "redacted_thinking") for b in content
                )

    def test_tool_use_becomes_tool_calls(self) -> None:
        m = _make_offline_model()
        normalized = m._normalize_conversation_for_api(_ANTHROPIC_STYLE_CONVERSATION)
        assistant = normalized[1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "Let me add those numbers."
        assert assistant["tool_calls"] == [
            {
                "id": "toolu_01",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
            }
        ]

    def test_tool_result_becomes_tool_message(self) -> None:
        m = _make_offline_model()
        normalized = m._normalize_conversation_for_api(_ANTHROPIC_STYLE_CONVERSATION)
        tool_msg = normalized[2]
        assert tool_msg == {"role": "tool", "tool_call_id": "toolu_01", "content": "5"}

    def test_tool_result_with_nested_text_blocks(self) -> None:
        m = _make_offline_model()
        conv = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_02",
                        "content": [
                            {"type": "text", "text": "part1 "},
                            {"type": "text", "text": "part2"},
                        ],
                    }
                ],
            }
        ]
        normalized = m._normalize_conversation_for_api(conv)
        assert normalized == [
            {"role": "tool", "tool_call_id": "toolu_02", "content": "part1 part2"}
        ]

    def test_anthropic_image_block_becomes_image_url(self) -> None:
        m = _make_offline_model()
        conv = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGVsbG8=",
                        },
                    },
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        normalized = m._normalize_conversation_for_api(conv)
        assert normalized[0]["content"][0] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
        }

    def test_openai_style_messages_pass_through_unchanged(self) -> None:
        m = _make_offline_model()
        conv: list[dict[str, Any]] = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "5"},
            {"role": "assistant", "content": "the sum is 5"},
        ]
        assert m._normalize_conversation_for_api(conv) == conv


@requires_openai_api_key
class TestHandoffLive:
    """Live reproduction of the gpt-5.5 400 'Invalid value: thinking' bug."""

    def test_generate_with_tools_after_anthropic_handoff(self) -> None:
        """The exact failing scenario: tools call replaying an Anthropic history."""
        m = model("gpt-4.1-mini")
        assert isinstance(m, OpenAICompatibleModel)
        m.initialize("placeholder")
        m.conversation = [dict(msg) for msg in _ANTHROPIC_STYLE_CONVERSATION]
        function_calls, content, _ = m.generate_and_process_with_tools({"add": _add})
        assert "5" in content or function_calls

    def test_generate_without_tools_after_anthropic_handoff(self) -> None:
        """generate() must also survive thinking blocks in the history."""
        m = model("gpt-4.1-mini")
        assert isinstance(m, OpenAICompatibleModel)
        m.initialize("placeholder")
        m.conversation = [
            {"role": "user", "content": "Reply with exactly the word: ready"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "simple echo", "signature": "s"},
                    {"type": "text", "text": "ready"},
                ],
            },
            {"role": "user", "content": "Now reply with exactly the word: done"},
        ]
        content, _ = m.generate()
        assert "done" in content.lower()


class _CapturingHandler(BaseHTTPRequestHandler):
    """Captures the JSON body of every POST and returns a minimal response."""

    captured_bodies: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            self.__class__.captured_bodies.append(json.loads(raw))
        except json.JSONDecodeError:
            self.__class__.captured_bodies.append({})
        body = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-5.5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def capture_server() -> Generator[str]:
    _CapturingHandler.captured_bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()


def _assert_no_anthropic_blocks_on_wire(body: dict[str, Any]) -> None:
    """Assert no Anthropic-only block types survive in the wire messages."""
    for msg in body["messages"]:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                assert block.get("type") not in (
                    "thinking",
                    "redacted_thinking",
                    "tool_use",
                    "tool_result",
                ), f"Anthropic block leaked onto the wire: {block}"


class TestXhighAliasHandoffWire:
    """gpt-5.5-xhigh replaying an Anthropic-format history: exact wire payload.

    The ``-xhigh`` suffix is a KISS-internal alias, so the hand-off
    normalization must compose with the alias resolution: the request must
    carry ``model="gpt-5.5"`` AND a fully OpenAI-formatted message list with
    no ``thinking`` / ``tool_use`` / ``tool_result`` blocks.  A real local
    HTTP server captures the body — no mocks.
    """

    def test_tools_request_is_normalized_and_alias_resolved(
        self, capture_server: str
    ) -> None:
        """Tools path: alias → base id, Anthropic history → OpenAI messages."""
        m = model(
            "gpt-5.5-xhigh",
            model_config={"base_url": capture_server, "api_key": "test-key"},
        )
        assert isinstance(m, OpenAICompatibleModel)
        m.initialize("placeholder")
        m.conversation = [dict(msg) for msg in _ANTHROPIC_STYLE_CONVERSATION]
        m.generate_and_process_with_tools({"add": _add})
        body = _CapturingHandler.captured_bodies[-1]
        assert body["model"] == "gpt-5.5"
        _assert_no_anthropic_blocks_on_wire(body)
        assistant = body["messages"][1]
        assert assistant["tool_calls"][0]["function"]["name"] == "add"
        assert assistant["tool_calls"][0]["function"]["arguments"] == (
            '{"a": 2, "b": 3}'
        )
        tool_msg = body["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_01"
        assert tool_msg["content"] == "5"
        # The capture server is an unknown (unregistered) endpoint, so the
        # transport keeps reasoning_effort optimistically (adaptive probe).
        assert body["reasoning_effort"] == "xhigh"

    def test_generate_request_is_normalized_with_xhigh_effort(
        self, capture_server: str
    ) -> None:
        """No-tools path: thinking blocks dropped, xhigh effort still sent."""
        m = model(
            "gpt-5.5-xhigh",
            model_config={"base_url": capture_server, "api_key": "test-key"},
        )
        assert isinstance(m, OpenAICompatibleModel)
        m.initialize("placeholder")
        m.conversation = [
            {"role": "user", "content": "Reply with exactly the word: ready"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "simple echo", "signature": "s"},
                    {"type": "text", "text": "ready"},
                ],
            },
            {"role": "user", "content": "Now reply with exactly the word: done"},
        ]
        m.generate()
        body = _CapturingHandler.captured_bodies[-1]
        assert body["model"] == "gpt-5.5"
        assert body["reasoning_effort"] == "xhigh"
        _assert_no_anthropic_blocks_on_wire(body)
        assistant_msg = body["messages"][1]
        assert assistant_msg["role"] == "assistant"
        # Thinking block dropped; only the OpenAI-valid text part remains.
        assert assistant_msg["content"] == [{"type": "text", "text": "ready"}]


@requires_openai_api_key
class TestXhighHandoffLive:
    """Live replay of an Anthropic-format history through gpt-5.5-xhigh."""

    def test_generate_with_tools_after_anthropic_handoff(self) -> None:
        """The exact reported scenario, on the xhigh alias of gpt-5.5."""
        m = model("gpt-5.5-xhigh")
        assert isinstance(m, OpenAICompatibleModel)
        m.initialize("placeholder")
        m.conversation = [dict(msg) for msg in _ANTHROPIC_STYLE_CONVERSATION]
        function_calls, content, _ = m.generate_and_process_with_tools({"add": _add})
        assert "5" in content or function_calls

    def test_generate_without_tools_after_anthropic_handoff(self) -> None:
        """generate() on gpt-5.5-xhigh must survive thinking blocks too."""
        m = model("gpt-5.5-xhigh")
        assert isinstance(m, OpenAICompatibleModel)
        m.initialize("placeholder")
        m.conversation = [
            {"role": "user", "content": "Reply with exactly the word: ready"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "simple echo", "signature": "s"},
                    {"type": "text", "text": "ready"},
                ],
            },
            {"role": "user", "content": "Now reply with exactly the word: done"},
        ]
        content, _ = m.generate()
        assert "done" in content.lower()
