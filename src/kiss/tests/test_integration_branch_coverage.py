"""Integration tests for 100% branch coverage of five target files.

Target files:
- kiss.core.relentless_agent
- kiss.core.models.openai_compatible_model
- kiss.core.utils
- kiss.agents.sorcar.web_use_tool
- kiss.agents.sorcar.useful_tools

No mocks, patches, or test doubles.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
)
from kiss.core.models.openai_compatible_model import (
    OpenAICompatibleModel,
    _build_text_based_tools_prompt,
    _parse_text_based_tool_calls,
)
from kiss.core.relentless_agent import (
    RelentlessAgent,
)
from kiss.core.relentless_agent import (
    finish as ra_finish,
)
from kiss.core.utils import (
    finish as utils_finish,
)
from kiss.core.utils import (
    get_config_value,
    is_subpath,
    read_project_file,
    resolve_path,
)


async def _noop_callback(token: str) -> None:
    """Async no-op token callback for streaming tests."""
    pass


def _make_collector_callback(collector: list[str]):
    """Create an async token callback that collects tokens into a list."""
    async def _cb(token: str) -> None:
        collector.append(token)
    return _cb


# ═══════════════════════════════════════════════════════════════════════
# relentless_agent.finish()
# ═══════════════════════════════════════════════════════════════════════


class TestRelentlessAgentFinish:
    """Cover all branches of the module-level finish() in relentless_agent."""

    def test_finish_string_true(self) -> None:
        """Cover isinstance(success, str) and isinstance(is_continue, str) branches."""
        result = ra_finish("true", "yes", "ok")  # type: ignore[arg-type]
        payload = yaml.safe_load(result)
        assert payload["success"] is True
        assert payload["is_continue"] is True

# ═══════════════════════════════════════════════════════════════════════
# relentless_agent._docker_bash
# ═══════════════════════════════════════════════════════════════════════


class TestRelentlessAgentDockerBash:
    def test_docker_bash_raises_without_manager(self) -> None:
        from kiss.core.kiss_error import KISSError

        agent = RelentlessAgent("test")
        agent.docker_manager = None
        with pytest.raises(KISSError, match="Docker manager not initialized"):
            agent._docker_bash("echo hi", "test")


# ═══════════════════════════════════════════════════════════════════════
# OpenAICompatibleModel standalone helpers
# ═══════════════════════════════════════════════════════════════════════


class TestBuildTextBasedToolsPrompt:
    def test_function_untyped_param(self) -> None:
        def untyped(x):  # type: ignore[no-untyped-def]
            """Untyped."""
            pass

        prompt = _build_text_based_tools_prompt({"untyped": untyped})
        assert "x (any)" in prompt


class TestParseTextBasedToolCalls:
    def test_no_tool_calls_key(self) -> None:
        content = '```json\n{"result": "hello"}\n```'
        calls = _parse_text_based_tool_calls(content)
        assert len(calls) == 0

    def test_tool_calls_not_list(self) -> None:
        content = '{"tool_calls": "not a list"}'
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

    def test_tool_call_without_name(self) -> None:
        content = '{"tool_calls": [{"arguments": {"x": 1}}]}'
        calls = _parse_text_based_tool_calls(content)
        assert len(calls) == 0

# ═══════════════════════════════════════════════════════════════════════
# OpenAICompatibleModel class methods (no API calls)
# ═══════════════════════════════════════════════════════════════════════


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
        # text/plain doesn't match image/ or application/pdf, so only text part
        assert content[-1]["type"] == "text"


class TestParseToolCallAccum:
    def test_invalid_json_arguments(self) -> None:
        accum = {
            0: {"id": "c1", "name": "test", "arguments": "not json"},
        }
        fc, raw = OpenAICompatibleModel._parse_tool_call_accum(accum)
        assert len(fc) == 1
        assert fc[0]["arguments"] == {}

class TestFinalizeStreamResponse:
    def test_with_last_chunk(self) -> None:
        result = OpenAICompatibleModel._finalize_stream_response(None, "last")
        assert result == "last"

    def test_raises_on_empty(self) -> None:
        from kiss.core.kiss_error import KISSError

        with pytest.raises(KISSError, match="empty"):
            OpenAICompatibleModel._finalize_stream_response(None, None)


class TestExtractTokenCounts:
    def test_with_usage(self) -> None:
        class FakeUsage:
            prompt_tokens = 100
            completion_tokens = 50
            prompt_tokens_details = None

        class FakeResponse:
            usage = FakeUsage()

        m = OpenAICompatibleModel("gpt-4", base_url="http://localhost", api_key="k")
        inp, out, cache_r, cache_w = m.extract_input_output_token_counts_from_response(
            FakeResponse()
        )
        assert inp == 100
        assert out == 50
        assert cache_r == 0
        assert cache_w == 0

    def test_no_usage_attr(self) -> None:
        m = OpenAICompatibleModel("gpt-4", base_url="http://localhost", api_key="k")
        result = m.extract_input_output_token_counts_from_response(object())
        assert result == (0, 0, 0, 0)


# ═══════════════════════════════════════════════════════════════════════
# OpenAICompatibleModel with a fake HTTP server (real API calls, no mocks)
# ═══════════════════════════════════════════════════════════════════════


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
    # First chunk with role
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
    # Content chunks
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
    # Tool call deltas if any
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
    # Final chunk with finish_reason
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
    # Usage chunk (no choices)
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

    # Class-level response configuration
    # Modes: normal, stream, tool_calls, stream_tool_calls,
    # deepseek, error, embedding
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
            # DeepSeek returning text-based tool calls
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
            # Simulate reasoning_content attribute on delta
            chunks = []
            # A chunk with reasoning_content
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
            # Finish
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
            # Usage
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

    def test_non_streaming_with_tool_calls(self, fake_openai_server: str) -> None:
        FakeOpenAIHandler.response_mode = "tool_calls"
        m = OpenAICompatibleModel("fake-model", base_url=fake_openai_server, api_key="test")
        m.initialize("Call a tool")

        def test_func(x: int) -> str:
            """Test function."""
            return str(x)

        fc, content, response = m.generate_and_process_with_tools({"test_func": test_func})
        assert len(fc) == 1
        assert fc[0]["name"] == "test_func"
        # conversation should have tool_calls
        assert "tool_calls" in m.conversation[-1]

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
        assert fc[0]["arguments"] == {}  # Bad JSON → empty dict

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
        # The first message is system, not user, which hits the else branch

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
        # conversation should have tool_calls
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


class TestOpenAICompatibleModelEmbedding:
    def test_get_embedding_failure(self) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel(
            "fake-model", base_url="http://localhost:1", api_key="test"
        )
        m.initialize("test")
        with pytest.raises(KISSError, match="Embedding generation failed"):
            m.get_embedding("Hello world")


# ═══════════════════════════════════════════════════════════════════════
# utils.py
# ═══════════════════════════════════════════════════════════════════════


class TestUtilsFunctions:
    def test_get_config_value_default(self) -> None:
        class Cfg:
            pass

        assert get_config_value(None, Cfg(), "missing", default="fallback") == "fallback"

    def test_get_config_value_raises(self) -> None:
        class Cfg:
            pass

        with pytest.raises(ValueError):
            get_config_value(None, Cfg(), "missing")

    def test_utils_finish(self) -> None:
        result = utils_finish(status="success", analysis="good", result="42")
        payload = yaml.safe_load(result)
        assert payload["status"] == "success"
        assert payload["result"] == "42"

    def test_read_project_file_success(self) -> None:
        """Cover the filesystem path (os.path.isfile) branch (lines 161-162)."""
        # Path relative to project_root (which is src/ for installed package)
        content = read_project_file("kiss/core/utils.py")
        assert "def resolve_path" in content

    def test_read_project_file_single_part(self) -> None:
        """Cover the len(rel_parts) <= 1 (no package) branch."""
        from kiss.core.kiss_error import KISSError

        with pytest.raises(KISSError, match="Could not find"):
            read_project_file("nonexistent_single_file.xyz")

    def test_resolve_path_relative(self) -> None:
        result = resolve_path("foo/bar.txt", "/base")
        assert result == Path("/base/foo/bar.txt").resolve()

    def test_resolve_path_absolute(self) -> None:
        result = resolve_path("/absolute/path.txt", "/base")
        assert result == Path("/absolute/path.txt").resolve()

    def test_is_subpath_false(self) -> None:
        assert is_subpath(Path("/a/b/c"), [Path("/d/e")]) is False


# ═══════════════════════════════════════════════════════════════════════
# useful_tools.py (covering timeout branches)
# ═══════════════════════════════════════════════════════════════════════


class TestUsefulToolsBashTimeout:
    """Cover the timeout branches in Bash (both streaming and non-streaming)."""

    @pytest.fixture
    def tmpdir(self) -> Generator[Path]:
        d = Path(tempfile.mkdtemp())
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_non_streaming_timeout(self, tmpdir: Path) -> None:
        """Cover lines 307-312: non-streaming Bash timeout."""
        ut = UsefulTools()
        result = ut.Bash("sleep 100", "long sleep", timeout_seconds=0.5)
        assert "timeout" in result.lower()

    def test_streaming_timeout(self, tmpdir: Path) -> None:
        """Cover line 362: streaming Bash timeout."""
        ut = UsefulTools(stream_callback=lambda _: None)
        result = ut.Bash("sleep 100", "long sleep", timeout_seconds=0.5)
        assert "timeout" in result.lower()


# ═══════════════════════════════════════════════════════════════════════
# relentless_agent.perform_task (integration via real KISSAgent + fake server)
# ═══════════════════════════════════════════════════════════════════════


class TestRelentlessAgentRun:
    """Integration tests for RelentlessAgent.run()."""

    @pytest.fixture
    def tmpdir(self) -> Generator[Path]:
        d = Path(tempfile.mkdtemp())
        yield d
        shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
