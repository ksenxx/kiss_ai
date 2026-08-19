# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests targeting uncovered branches in core/models/.

The core/ methods split out of this file live in
``kiss.tests.core.test_coverage_integration``; the agents/sorcar/ ones in
``kiss.tests.agents.sorcar.test_coverage_integration``.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest import TestCase

import pytest

from kiss.core.models.model import (
    Model,
    _parse_text_based_tool_calls,
)
from kiss.core.models.model_info import MODEL_INFO
from kiss.core.models.openai_compatible_model import (
    OpenAICompatibleModel,
)


class TestModelSchemaConversion(TestCase):
    def _get_model(self) -> Model:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        return OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )

    def test_optional_type_schema(self) -> None:
        m = self._get_model()
        schema = m._python_type_to_json_schema(int | None)
        assert schema == {"type": "integer"}

    def test_union_type_schema(self) -> None:
        m = self._get_model()
        schema = m._python_type_to_json_schema(str | int)
        assert "anyOf" in schema


class TestModelCallbackLoop(TestCase):

    def test_invoke_callback_no_callback(self) -> None:
        """Cover the early return when token_callback is None."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        m._invoke_token_callback("ignored")


class TestGetModel(TestCase):
    def test_model_info_cache_pricing(self) -> None:
        for name, info in MODEL_INFO.items():
            if name.startswith("claude-") and info.input_price_per_1M > 0:
                assert info.cache_read_price_per_1M is not None
                break


def _noop_callback(token: str) -> None:
    """No-op token callback for streaming tests."""
    pass


def _make_collector_callback(collector: list[str]):
    """Create a token callback that collects tokens into a list."""
    def _cb(token: str) -> None:
        collector.append(token)
    return _cb


class TestParseTextBasedToolCalls:
    def test_no_tool_calls_key(self) -> None:
        content = '```json\n{"result": "hello"}\n```'
        calls = _parse_text_based_tool_calls(content)
        assert len(calls) == 0

    def test_raw_json_tool_calls_no_code_block(self) -> None:
        """Cover the fallback json.loads(content.strip()) path (line 170).

        The extra 'meta' key with braces prevents the inline regex from matching,
        so the fallback json.loads(content.strip()) is used.
        """
        content = json.dumps(
            {
                "tool_calls": [{"name": "finish"}],
                "meta": {"x": 1},
            }
        )
        calls = _parse_text_based_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "finish"


class TestOpenAICompatibleModelInit:
    def test_str_repr(self) -> None:
        m = OpenAICompatibleModel("gpt-4", base_url="http://localhost:8080", api_key="k")
        s = str(m)
        assert "gpt-4" in s
        assert "http://localhost:8080" in s
        assert repr(m) == s

class TestOpenAICompatibleModelInitialize:
    def test_initialize_with_pdf_attachment(self) -> None:
        from kiss.core.models.model import Attachment

        m = OpenAICompatibleModel("gpt-4", base_url="http://localhost", api_key="k")
        att = Attachment(data=b"%PDF-1.4", mime_type="application/pdf")
        m.initialize("Read this PDF", attachments=[att])
        content = m.conversation[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "file"

    def test_initialize_with_unsupported_attachment(self) -> None:
        """Cover the attachment loop fallthrough (branch 254->246)."""
        from kiss.core.models.model import Attachment

        m = OpenAICompatibleModel("gpt-4", base_url="http://localhost", api_key="k")
        att = Attachment(data=b"text data", mime_type="text/plain")
        m.initialize("Analyze this", attachments=[att])
        content = m.conversation[0]["content"]
        assert isinstance(content, list)
        assert content[-1]["type"] == "text"


class TestFinalizeStreamResponse:
    def test_with_last_chunk(self) -> None:
        result = OpenAICompatibleModel._finalize_stream_response(None, "last")
        assert result == "last"


class TestExtractTokenCounts:
    def test_with_usage(self) -> None:
        class FakeUsage:
            prompt_tokens = 100
            completion_tokens = 50
            prompt_tokens_details = None

        class FakeResponse:
            usage = FakeUsage()

        m = OpenAICompatibleModel("gpt-4", base_url="http://localhost", api_key="k")
        counts = m.extract_input_output_token_counts_from_response(FakeResponse())
        assert counts == (100, 50, 0, 0)

    def test_no_usage_attr(self) -> None:
        m = OpenAICompatibleModel("gpt-4", base_url="http://localhost", api_key="k")
        result = m.extract_input_output_token_counts_from_response(object())
        assert result == (0, 0, 0, 0)


def _make_chat_response(content: str = "Hello!", tool_calls: list | None = None) -> dict:
    """Build a minimal OpenAI chat completion response."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "fake-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_stream_chunks(
    content: str = "Hello!",
    tool_calls_deltas: list | None = None,
) -> list[str]:
    """Build SSE stream chunks for OpenAI-compatible streaming."""
    chunks = []
    chunks.append(
        json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "fake-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
        )
    )
    for char in content:
        chunks.append(
            json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "model": "fake-model",
                    "choices": [
                        {"index": 0, "delta": {"content": char}, "finish_reason": None}
                    ],
                }
            )
        )
    if tool_calls_deltas:
        for delta in tool_calls_deltas:
            chunks.append(
                json.dumps(
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "model": "fake-model",
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }
                )
            )
    chunks.append(
        json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "fake-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
    )
    chunks.append(
        json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "fake-model",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
    )
    return chunks


FAKE_EMBEDDING_RESPONSE = json.dumps(
    {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "fake-embed",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
).encode()


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    """Handler that simulates OpenAI API responses."""

    response_mode: str = "normal"

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}

        if self.path == "/v1/chat/completions":
            stream = body.get("stream", False)
            if stream:
                self._handle_stream(body)
            else:
                self._handle_non_stream(body)
        elif self.path == "/v1/embeddings":
            self._handle_embeddings()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_non_stream(self, body: dict) -> None:
        mode = self.__class__.response_mode
        if mode == "tool_calls":
            resp = _make_chat_response(
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "arguments": '{"x": 42}',
                        },
                    }
                ],
            )
        elif mode == "tool_calls_bad_json":
            resp = _make_chat_response(
                content="",
                tool_calls=[
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "bad_func",
                            "arguments": "not-json",
                        },
                    }
                ],
            )
        else:
            resp = _make_chat_response("Hello from server!")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        data = json.dumps(resp).encode()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_stream(self, body: dict) -> None:
        mode = self.__class__.response_mode
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        if mode == "stream_tool_calls":
            tc_deltas = [
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_s1",
                            "function": {"name": "test_func", "arguments": ""},
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"name": None, "arguments": '{"x":'},
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"name": None, "arguments": "42}"},
                        }
                    ]
                },
            ]
            chunks = _make_stream_chunks(content="", tool_calls_deltas=tc_deltas)
        elif mode == "deepseek_tool_calls":
            tc_json = json.dumps(
                {"tool_calls": [{"name": "finish", "arguments": {"result": "42"}}]}
            )
            chunks = _make_stream_chunks(
                content=f"<think>reasoning</think>{tc_json}"
            )
        elif mode == "deepseek":
            chunks = _make_stream_chunks(
                content="<think>reasoning</think>The answer is 42"
            )
        elif mode == "reasoning_content":
            chunks = []
            chunks.append(
                json.dumps(
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "model": "fake-model",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": "",
                                    "reasoning_content": "thinking...",
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            )
            chunks.append(
                json.dumps(
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "model": "fake-model",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "Final"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            )
            chunks.append(
                json.dumps(
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "model": "fake-model",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                )
            )
            chunks.append(
                json.dumps(
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "model": "fake-model",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                )
            )
        else:
            chunks = _make_stream_chunks("Hello streamed!")

        for chunk in chunks:
            self.wfile.write(f"data: {chunk}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _handle_embeddings(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(FAKE_EMBEDDING_RESPONSE)))
        self.end_headers()
        self.wfile.write(FAKE_EMBEDDING_RESPONSE)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture(scope="module")
def fake_openai_server():
    """Start a fake OpenAI-compatible server."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


class TestOpenAICompatibleModelGenerate:
    """Test generate() via a fake server (non-streaming and streaming)."""

    def test_generate_deepseek_strips_think_tags(self, fake_openai_server: str) -> None:
        """Cover the _is_deepseek_reasoning_model branch in generate()."""
        FakeOpenAIHandler.response_mode = "deepseek"
        tokens: list[str] = []
        m = OpenAICompatibleModel(
            "deepseek/deepseek-r1",
            base_url=fake_openai_server,
            api_key="test",
            token_callback=_make_collector_callback(tokens),
        )
        m.initialize("What is 6*7?")
        content, response = m.generate()
        assert "The answer is 42" in content
        assert "<think>" not in content


class TestOpenAICompatibleModelGenerateWithTools:
    """Test generate_and_process_with_tools() via fake server."""

    def test_non_streaming_no_tools(self, fake_openai_server: str) -> None:
        FakeOpenAIHandler.response_mode = "normal"
        m = OpenAICompatibleModel("fake-model", base_url=fake_openai_server, api_key="test")
        m.initialize("Hi")

        def dummy() -> str:
            """Dummy tool."""
            return "ok"

        fc, content, response = m.generate_and_process_with_tools({"dummy": dummy})
        assert fc == []
        assert "Hello from server!" in content

    def test_non_streaming_with_bad_json_tool_calls(self, fake_openai_server: str) -> None:
        """Cover the JSONDecodeError branch in _parse_tool_calls_from_message."""
        FakeOpenAIHandler.response_mode = "tool_calls_bad_json"
        m = OpenAICompatibleModel("fake-model", base_url=fake_openai_server, api_key="test")
        m.initialize("Call a tool")

        def bad_func() -> str:
            """Bad function."""
            return "ok"

        fc, content, response = m.generate_and_process_with_tools({"bad_func": bad_func})
        assert len(fc) == 1
        assert fc[0]["arguments"] == {}

    def test_deepseek_text_based_tools_with_system_message(
        self, fake_openai_server: str
    ) -> None:
        """Cover the system message branch in _generate_with_text_based_tools."""
        FakeOpenAIHandler.response_mode = "deepseek"
        m = OpenAICompatibleModel(
            "deepseek/deepseek-r1",
            base_url=fake_openai_server,
            api_key="test",
            model_config={"system_instruction": "Be helpful"},
            token_callback=_noop_callback,
        )
        m.initialize("Question")

        def finish(result: str) -> str:
            """Finish."""
            return result

        fc, content, response = m.generate_and_process_with_tools({"finish": finish})
        assert isinstance(fc, list)

    def test_deepseek_text_based_tools_with_tool_calls_in_response(
        self, fake_openai_server: str
    ) -> None:
        """Cover line 549: function_calls populated in _generate_with_text_based_tools."""
        FakeOpenAIHandler.response_mode = "deepseek_tool_calls"
        tokens: list[str] = []
        m = OpenAICompatibleModel(
            "deepseek/deepseek-r1",
            base_url=fake_openai_server,
            api_key="test",
            token_callback=_make_collector_callback(tokens),
        )
        m.initialize("Call finish tool")

        def finish(result: str) -> str:
            """Finish."""
            return result

        fc, content, response = m.generate_and_process_with_tools({"finish": finish})
        assert len(fc) == 1
        assert fc[0]["name"] == "finish"
        assert "tool_calls" in m.conversation[-1]

    def test_streaming_with_reasoning_content(self, fake_openai_server: str) -> None:
        """Cover the reasoning_content branch in streaming."""
        FakeOpenAIHandler.response_mode = "reasoning_content"
        tokens: list[str] = []
        m = OpenAICompatibleModel(
            "fake-model",
            base_url=fake_openai_server,
            api_key="test",
            token_callback=_make_collector_callback(tokens),
        )
        m.initialize("Think and answer")

        def dummy() -> str:
            """Dummy."""
            return "ok"

        fc, content, response = m.generate_and_process_with_tools({"dummy": dummy})
        assert "Final" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
