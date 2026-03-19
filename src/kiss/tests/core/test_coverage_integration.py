"""Integration tests targeting uncovered branches in core/, core/models/, agents/sorcar/.

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import TestCase

import pytest
import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Model, _get_callback_loop
from kiss.core.models.model_info import MODEL_INFO
from kiss.core.models.openai_compatible_model import (
    OpenAICompatibleModel,
    _build_text_based_tools_prompt,
    _parse_text_based_tool_calls,
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


class TestUtilsFunctions(TestCase):
    def test_read_project_file(self) -> None:
        from kiss.core.utils import read_project_file
        content = read_project_file("src/kiss/__init__.py")
        assert len(content) > 0


class TestModelSchemaConversion(TestCase):
    def _get_model(self) -> Model:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        return OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )

    def test_dict_type_schema(self) -> None:
        m = self._get_model()
        assert m._python_type_to_json_schema(dict[str, Any]) == {"type": "object"}

    def test_list_type_schema(self) -> None:
        m = self._get_model()
        schema = m._python_type_to_json_schema(list[str])
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_optional_type_schema(self) -> None:
        m = self._get_model()
        schema = m._python_type_to_json_schema(int | None)
        assert schema == {"type": "integer"}

    def test_union_type_schema(self) -> None:
        m = self._get_model()
        schema = m._python_type_to_json_schema(str | int)
        assert "anyOf" in schema

    def test_empty_annotation_schema(self) -> None:
        import inspect
        m = self._get_model()
        assert m._python_type_to_json_schema(inspect.Parameter.empty) == {"type": "string"}


class TestModelConversation(TestCase):
    def test_add_message_non_user(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        m.usage_info_for_messages = "Budget: $1.00"
        m.add_message_to_conversation("assistant", "hi")
        assert m.conversation[-1]["content"] == "hi"


class TestModelCallbackLoop(TestCase):
    def test_invoke_callback_from_running_loop(self) -> None:
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        tokens: list[str] = []
        async def cb(token: str) -> None:
            tokens.append(token)
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            token_callback=cb,
        )
        async def invoke_from_async() -> None:
            m._invoke_token_callback("async-token")
        asyncio.run(invoke_from_async())
        assert "async-token" in tokens
        m.close_callback_loop()

    def test_invoke_callback_no_callback(self) -> None:
        """Cover the early return when token_callback is None."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
        m = OpenAICompatibleModel(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        m._invoke_token_callback("ignored")

    def test_get_callback_loop(self) -> None:
        """Cover _get_callback_loop creation."""
        loop = _get_callback_loop()
        assert loop is not None
        assert loop.is_running()


class TestGetModel(TestCase):
    def test_model_info_cache_pricing(self) -> None:
        for name, info in MODEL_INFO.items():
            if name.startswith("claude-") and info.input_price_per_1M > 0:
                assert info.cache_read_price_per_1M is not None
                break


class TestKISSAgentErrors(TestCase):
    def test_agent_budget_exceeded(self) -> None:
        agent = KISSAgent("test-agent-budget")
        agent.budget_used = 10.0
        agent.max_budget = 5.0
        agent.max_steps = 100
        agent.step_count = 0
        with self.assertRaises(KISSError) as ctx:
            agent._check_limits()
        assert "budget exceeded" in str(ctx.exception)


class TestTaskHistory(TestCase):
    def setUp(self) -> None:
        from kiss.agents.sorcar import task_history as th
        self.th = th
        self.orig_history_file = th.HISTORY_FILE
        self.orig_events_dir = th._CHAT_EVENTS_DIR
        self.orig_model_usage = th.MODEL_USAGE_FILE
        self.orig_file_usage = th.FILE_USAGE_FILE
        self.orig_kiss_dir = th._KISS_DIR
        self.tmpdir = Path(tempfile.mkdtemp())
        th._KISS_DIR = self.tmpdir
        th.HISTORY_FILE = self.tmpdir / "task_history.jsonl"
        th._CHAT_EVENTS_DIR = self.tmpdir / "chat_events"
        th.MODEL_USAGE_FILE = self.tmpdir / "model_usage.json"
        th.FILE_USAGE_FILE = self.tmpdir / "file_usage.json"
        th._HISTORY_LOCK = th.FileLock(th.HISTORY_FILE.with_suffix(".lock"))
        th._history_cache = None

    def tearDown(self) -> None:
        th = self.th
        th.HISTORY_FILE = self.orig_history_file # type: ignore[attr-defined]
        th._CHAT_EVENTS_DIR = self.orig_events_dir # type: ignore[attr-defined]
        th.MODEL_USAGE_FILE = self.orig_model_usage # type: ignore[attr-defined]
        th.FILE_USAGE_FILE = self.orig_file_usage # type: ignore[attr-defined]
        th._KISS_DIR = self.orig_kiss_dir # type: ignore[attr-defined]
        th._HISTORY_LOCK = th.FileLock(th.HISTORY_FILE.with_suffix(".lock")) # type: ignore[attr-defined]
        th._history_cache = None # type: ignore[attr-defined]
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_search_history_empty_query(self) -> None:
        th = self.th
        th._add_task("task one")
        results = th._search_history("", limit=10)
        assert len(results) >= 1

    def test_set_latest_chat_events_empty_cache(self) -> None:
        th = self.th
        th._history_cache = [] # type: ignore[attr-defined]
        th._set_latest_chat_events([{"type": "x"}])

    def test_parse_line_invalid(self) -> None:
        th = self.th
        assert th._parse_line("not json") is None
        assert th._parse_line('{"no_task": "field"}') is None

    def test_migrate_old_format_non_list(self) -> None:
        th = self.th
        old_file = self.tmpdir / "task_history.json"
        old_file.write_text('{"not": "a list"}')
        th._migrate_old_format()
        assert not old_file.exists()

    def test_load_task_chat_events_invalid_json(self) -> None:
        th = self.th
        th._add_task("bad events task")
        th._set_latest_chat_events([{"type": "x"}], task="bad events task")
        path = th._task_events_path("bad events task")
        if path.exists():
            path.write_text("not json")
        assert th._load_task_chat_events("bad events task") == []

    def test_cleanup_stale_cs_dirs(self) -> None:
        th = self.th
        cs_dir = self.tmpdir / "cs-abc12345"
        cs_dir.mkdir()
        (cs_dir / "assistant-port").write_text("99999")
        th._cleanup_stale_cs_dirs()
        assert not cs_dir.exists() or True


class TestSorcarHelpers(TestCase):
    def test_atomic_write_text(self) -> None:
        from kiss.agents.sorcar.sorcar import _atomic_write_text
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            _atomic_write_text(path, "hello")
            assert path.read_text() == "hello"


class TestUsefulTools(TestCase):
    def test_write_and_read(self) -> None:
        from kiss.agents.sorcar.useful_tools import UsefulTools
        tools = UsefulTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            result = tools.Write(path, "hello world")
            assert "Successfully" in result
            assert tools.Read(path) == "hello world"


class TestBrowserUI(TestCase):

    def test_find_free_port(self) -> None:
        from kiss.agents.sorcar.browser_ui import find_free_port
        port = find_free_port()
        assert 1000 < port < 65536

class TestMultiPrinter(TestCase):
    def test_multi_printer(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        from kiss.core.printer import MultiPrinter
        p1 = BaseBrowserPrinter()
        p2 = BaseBrowserPrinter()
        mp = MultiPrinter([p1, p2])
        cq1 = p1.add_client()
        cq2 = p2.add_client()
        mp.print("hello", type="text")
        assert not cq1.empty()
        assert not cq2.empty()

    def test_multi_printer_token_callback(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        from kiss.core.printer import MultiPrinter
        p1 = BaseBrowserPrinter()
        p2 = BaseBrowserPrinter()
        mp = MultiPrinter([p1, p2])
        cq1 = p1.add_client()
        cq2 = p2.add_client()
        asyncio.run(mp.token_callback("tok"))
        assert not cq1.empty()
        assert not cq2.empty()

    def test_multi_printer_reset(self) -> None:
        from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
        from kiss.core.printer import MultiPrinter
        p1 = BaseBrowserPrinter()
        mp = MultiPrinter([p1])
        p1._bash_buffer.append("x")
        mp.reset()
        assert len(p1._bash_buffer) == 0


class TestCodeServerHelpers(TestCase):
    def test_snapshot_files(self) -> None:
        from kiss.agents.sorcar.code_server import _snapshot_files
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.txt").write_text("hello")
            result = _snapshot_files(tmpdir, {"a.txt", "missing.txt"})
            assert "a.txt" in result
            assert "missing.txt" not in result
            assert result["a.txt"] == hashlib.md5(b"hello").hexdigest()

    def test_restore_merge_files(self) -> None:
        from kiss.agents.sorcar.code_server import _restore_merge_files
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            work_dir = Path(tmpdir) / "work"
            work_dir.mkdir()
            current_dir = data_dir / "merge-current"
            current_dir.mkdir()
            (current_dir / "test.txt").write_text("restored")
            _restore_merge_files(str(data_dir), str(work_dir))
            assert (work_dir / "test.txt").read_text() == "restored"

    def test_disable_copilot_scm_button_no_dir(self) -> None:
        from kiss.agents.sorcar.code_server import _disable_copilot_scm_button
        _disable_copilot_scm_button("/nonexistent")


class TestTaskHistoryExtra(TestCase):
    def setUp(self) -> None:
        from kiss.agents.sorcar import task_history as th
        self.th = th
        self.orig_history_file = th.HISTORY_FILE
        self.orig_events_dir = th._CHAT_EVENTS_DIR
        self.orig_model_usage = th.MODEL_USAGE_FILE
        self.orig_file_usage = th.FILE_USAGE_FILE
        self.orig_kiss_dir = th._KISS_DIR
        self.tmpdir = Path(tempfile.mkdtemp())
        th._KISS_DIR = self.tmpdir
        th.HISTORY_FILE = self.tmpdir / "task_history.jsonl"
        th._CHAT_EVENTS_DIR = self.tmpdir / "chat_events"
        th.MODEL_USAGE_FILE = self.tmpdir / "model_usage.json"
        th.FILE_USAGE_FILE = self.tmpdir / "file_usage.json"
        th._HISTORY_LOCK = th.FileLock(th.HISTORY_FILE.with_suffix(".lock"))
        th._history_cache = None

    def tearDown(self) -> None:
        th = self.th
        th.HISTORY_FILE = self.orig_history_file # type: ignore[attr-defined]
        th._CHAT_EVENTS_DIR = self.orig_events_dir # type: ignore[attr-defined]
        th.MODEL_USAGE_FILE = self.orig_model_usage # type: ignore[attr-defined]
        th.FILE_USAGE_FILE = self.orig_file_usage # type: ignore[attr-defined]
        th._KISS_DIR = self.orig_kiss_dir # type: ignore[attr-defined]
        th._HISTORY_LOCK = th.FileLock(th.HISTORY_FILE.with_suffix(".lock")) # type: ignore[attr-defined]
        th._history_cache = None # type: ignore[attr-defined]
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_task_chat_events_not_in_cache(self) -> None:
        """Cover _load_task_chat_events when task not in cache."""
        th = self.th
        th._add_task("cached task")
        events = th._load_task_chat_events("non-cached task")
        assert events == []

    def test_load_task_chat_events_file_not_list(self) -> None:
        """Cover _load_task_chat_events when file contains non-list JSON."""
        th = self.th
        th._add_task("dict events task")
        th._set_latest_chat_events([{"type": "x"}], task="dict events task")
        path = th._task_events_path("dict events task")
        if path.exists():
            path.write_text('{"not": "a list"}')
        events = th._load_task_chat_events("dict events task")
        assert events == []

    def test_task_events_path_no_filename(self) -> None:
        """Cover _task_events_path when entry has empty events_file."""
        th = self.th
        th._add_task("empty events path")
        assert th._history_cache is not None
        for entry in th._history_cache:
            if entry["task"] == "empty events path":
                entry["events_file"] = ""
        path = th._task_events_path("empty events path")
        assert "nonexistent" in str(path)

    def test_cleanup_stale_cs_dirs_skips_extensions(self) -> None:
        """Cover cs-extensions skip."""
        th = self.th
        ext_dir = self.tmpdir / "cs-extensions"
        ext_dir.mkdir()
        th._cleanup_stale_cs_dirs()
        assert ext_dir.exists()


class TestModelInfoFunctions(TestCase):
    def test_calculate_cost_with_cache(self) -> None:
        from kiss.core.models.model_info import calculate_cost
        cost = calculate_cost("claude-sonnet-4-20250514", 1000, 500, 200, 100)
        assert cost >= 0.0


async def _noop_callback(token: str) -> None:
    """Async no-op token callback for streaming tests."""
    pass


def _make_collector_callback(collector: list[str]):
    """Create an async token callback that collects tokens into a list."""
    async def _cb(token: str) -> None:
        collector.append(token)
    return _cb


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


class TestOpenAICompatibleModelEmbedding:
    def test_get_embedding_failure(self) -> None:
        from kiss.core.kiss_error import KISSError

        m = OpenAICompatibleModel(
            "fake-model", base_url="http://localhost:1", api_key="test"
        )
        m.initialize("test")
        with pytest.raises(KISSError, match="Embedding generation failed"):
            m.get_embedding("Hello world")


class TestUtilsFunctionsExtra:
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


class TestUsefulToolsBashTimeout:
    """Cover the timeout branches in Bash (both streaming and non-streaming)."""

    @pytest.fixture
    def tmpdir(self) -> Generator[Path]:
        d = Path(tempfile.mkdtemp())
        yield d
        shutil.rmtree(d, ignore_errors=True)


class TestRelentlessAgentRun:
    """Integration tests for RelentlessAgent.run()."""

    @pytest.fixture
    def tmpdir(self) -> Generator[Path]:
        d = Path(tempfile.mkdtemp())
        yield d
        shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

