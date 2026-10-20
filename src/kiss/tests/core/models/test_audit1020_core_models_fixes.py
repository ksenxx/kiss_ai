# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Core model fixes from the 2026-10-20 audit (slice F-A1A2).

* A1-L5: ``get_embedding`` / ``get_embeddings`` work before ``initialize()``
  (they used to dereference the ``None`` client).
* A1-L4: one text-based tool-calling implementation serves both
  transports; every placement of the tools prompt (merged into a
  leading user message, inserted ahead of a non-user first message) is
  driven end to end against a scripted server.
* A1-U1: ``_CLIProcess.close()`` releases ``_stdin_lock`` through
  ``locked_call``, so a Stop injected on the task thread right after
  the acquire cannot leak the lock.  Hammered in a child process
  (:mod:`kiss.tests.lock_injection`); fails on the pre-fix ``with``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from kiss.core import locked as locked_module
from kiss.core.models import model as model_module
from kiss.core.models.model import _CLIProcess
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.tests.core.models.openai_sse_harness import Reply, Request, ScriptedOpenAIServer
from kiss.tests.lock_injection import (
    INJECTIONS,
    Contender,
    Hammered,
    ignore_thread_interrupt,
    inject,
    lock_is_free,
)

_MODEL = "audit1020-model-under-test"
# A catalog name on the text-based tools list (DeepSeek R1 has no native tools).
_TEXT_TOOLS_MODEL = "deepseek/deepseek-r1"
_TOOL_TEXT = '{"tool_calls": [{"name": "echo", "arguments": {"text": "hi"}}]}'
_ECHO_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
]


def _echo(text: str) -> str:
    """Echo *text* back.

    Args:
        text: The text to echo.

    Returns:
        The same text.
    """
    return text


def _embedding_responder(request: Request) -> Reply:
    """Answer /embeddings with a one-vector-per-input reply."""
    inputs = request.body["input"]
    count = len(inputs) if isinstance(inputs, list) else 1
    return Reply(
        json_body={
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": [float(i)]} for i in range(count)
            ],
            "model": "text-embedding-x",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }
    )


def _text_responder(request: Request, text: str = _TOOL_TEXT) -> Reply:
    """Answer a chat or Responses request with *text* as the assistant's plain text."""
    if request.path.endswith("/chat/completions"):
        body = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": _TEXT_TOOLS_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        }
    else:
        body = {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": _TEXT_TOOLS_MODEL,
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
        }
    return Reply(json_body=body)


def _text_of(message: dict[str, Any]) -> str:
    """Return the text of a chat (string) or Responses (parts) message."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content)


# ---------------------------------------------------------------------------
# A1-L5
# ---------------------------------------------------------------------------


class TestEmbeddingBeforeInitialize:
    """Both embedding entry points create the client themselves."""

    @pytest.mark.parametrize("cls", [OpenAICompatibleModel, OpenAICompatibleModel2])
    def test_get_embedding_without_initialize(self, cls: type) -> None:
        """A fresh model embeds one text without a prior initialize()."""
        with ScriptedOpenAIServer(_embedding_responder) as server:
            model = cls(_MODEL, base_url=server.base_url, api_key="k")
            assert model.get_embedding("hello", embedding_model="text-embedding-x") == [0.0]
            assert server.requests[-1].body["input"] == "hello"

    @pytest.mark.parametrize("cls", [OpenAICompatibleModel, OpenAICompatibleModel2])
    def test_get_embeddings_without_initialize(self, cls: type) -> None:
        """A fresh model embeds a batch without a prior initialize()."""
        with ScriptedOpenAIServer(_embedding_responder) as server:
            model = cls(_MODEL, base_url=server.base_url, api_key="k")
            vectors = model.get_embeddings(["a", "b"], embedding_model="text-embedding-x")
            assert vectors == [[0.0], [1.0]]
            assert server.requests[-1].body["input"] == ["a", "b"]


# ---------------------------------------------------------------------------
# A1-L4
# ---------------------------------------------------------------------------


class TestSharedTextBasedTools:
    """The shared fallback places the tools prompt and records the turn per transport."""

    @pytest.mark.parametrize("cls", [OpenAICompatibleModel, OpenAICompatibleModel2])
    def test_prompt_merged_into_leading_user_message(self, cls: type) -> None:
        """A conversation starting with the user carries the prompt in that message."""
        with ScriptedOpenAIServer(_text_responder) as server:
            model = cls(_TEXT_TOOLS_MODEL, base_url=server.base_url, api_key="k")
            model.initialize("please call echo")
            calls, content, _ = model.generate_and_process_with_tools(
                function_map={"echo": _echo}, tools_schema=_ECHO_SCHEMA
            )
            sent = server.requests[-1].body
            messages = sent.get("messages") or sent["input"]
            assert messages[0]["role"] == "user"
            first = _text_of(messages[0])
            assert first.startswith("please call echo") and "## Available Tools" in first
            assert [m["role"] for m in messages if "role" in m].count("user") == 1
        assert content == _TOOL_TEXT
        assert [(c["name"], c["arguments"]) for c in calls] == [("echo", {"text": "hi"})]
        # The stored conversation was not rewritten with the tools prompt.
        assert _text_of(model.conversation[0]) == "please call echo"

    def test_chat_prompt_inserted_as_system_message(self) -> None:
        """v1: a system instruction comes first, so the prompt gets its own system message."""
        with ScriptedOpenAIServer(_text_responder) as server:
            model = OpenAICompatibleModel(
                _TEXT_TOOLS_MODEL,
                base_url=server.base_url,
                api_key="k",
                model_config={"system_instruction": "be brief"},
            )
            model.initialize("please call echo")
            calls, _, _ = model.generate_and_process_with_tools(
                function_map={"echo": _echo}, tools_schema=_ECHO_SCHEMA
            )
            messages = server.requests[-1].body["messages"]
            assert [m["role"] for m in messages] == ["system", "system", "user"]
            assert "echo" in messages[0]["content"]
            assert messages[1]["content"] == "be brief"
            assert messages[2]["content"] == "please call echo"
        assert calls[0]["name"] == "echo"
        last = model.conversation[-1]
        assert last["role"] == "assistant" and last["tool_calls"][0]["function"]["name"] == "echo"

    def test_responses_prompt_inserted_as_user_item(self) -> None:
        """v2: a non-user first item gets the prompt as a leading user item."""
        with ScriptedOpenAIServer(_text_responder) as server:
            model = OpenAICompatibleModel2(_TEXT_TOOLS_MODEL, base_url=server.base_url, api_key="k")
            model.initialize("please call echo")
            model.conversation.insert(
                0, {"role": "assistant", "content": [{"type": "output_text", "text": "earlier"}]}
            )
            calls, _, _ = model.generate_and_process_with_tools(
                function_map={"echo": _echo}, tools_schema=_ECHO_SCHEMA
            )
            items = server.requests[-1].body["input"]
            assert [item["role"] for item in items] == ["user", "assistant", "user"]
            assert items[0]["content"][0]["type"] == "input_text"
            assert "echo" in items[0]["content"][0]["text"]
            assert _text_of(items[2]) == "please call echo"
        assert calls[0]["name"] == "echo"
        recorded = model.conversation[-1]
        assert recorded["type"] == "function_call" and recorded["name"] == "echo"
        assert json.loads(recorded["arguments"]) == {"text": "hi"}
        # The parsed call is pending until its output is appended.
        assert model._pending_function_calls == [{"name": "echo", "call_id": calls[0]["id"]}]

    @pytest.mark.parametrize("cls", [OpenAICompatibleModel, OpenAICompatibleModel2])
    def test_reasoning_only_reply_records_no_calls(self, cls: type) -> None:
        """A reply that is nothing but a <think> block yields no calls and no pending call."""

        def think_only(request: Request) -> Reply:
            return _text_responder(request, "<think>nothing to do</think>")

        with ScriptedOpenAIServer(think_only) as server:
            model = cls(_TEXT_TOOLS_MODEL, base_url=server.base_url, api_key="k")
            model.initialize("please call echo")
            before = len(model.conversation)
            calls, content, _ = model.generate_and_process_with_tools(
                function_map={"echo": _echo}, tools_schema=_ECHO_SCHEMA
            )
        assert calls == [] and content == "<think>nothing to do</think>"
        recorded = model.conversation[before:]
        # Chat keeps the raw assistant text; Responses drops an empty cleaned text.
        assert [m.get("role") for m in recorded] in ([], ["assistant"])
        assert not any(m.get("type") == "function_call" for m in recorded)
        assert not getattr(model, "_pending_function_calls", [])


# ---------------------------------------------------------------------------
# A1-U1
# ---------------------------------------------------------------------------


class CloseProcess:
    """One real ``_CLIProcess.close()`` of an already-exited child on the worker."""

    def __init__(self) -> None:
        self.proc = _CLIProcess([sys.executable, "-c", "pass"], "probe", 5.0)
        self.proc.wait_for_exit()
        self.closed = 0

    def __call__(self) -> None:
        self.proc.close()
        self.closed += 1


# The files of the frames an interrupt is allowed to land in: the model's
# own ``close()`` and the lock helper it calls.  ``close()`` also runs
# ``Event.set`` and ``Thread.join`` (plain ``with`` on stdlib locks, off
# the task-thread Stop path in the daemon) and ``Popen.poll``; an
# interrupt landing in those frames would only wedge the stdlib's locks.
_INJECT_IN = (model_module.__file__, locked_module.__file__)


def _worker_in_target(worker: threading.Thread) -> bool:
    """Whether *worker*'s innermost Python frame is in one of ``_INJECT_IN``."""
    frame = sys._current_frames().get(worker.ident or -1)
    return frame is not None and frame.f_code.co_filename in _INJECT_IN


def run_hammer() -> None:
    """Child-process body: hammer ``close()`` and check ``_stdin_lock`` stays free."""
    driver = CloseProcess()
    lock = driver.proc._stdin_lock
    previous_hook = threading.excepthook
    threading.excepthook = ignore_thread_interrupt
    contender = Contender(lock)
    contender.start()
    worker = Hammered(driver)
    worker.start()
    delivered = attempts = 0
    try:
        while delivered < INJECTIONS and attempts < 10 * INJECTIONS:
            attempts += 1
            time.sleep(0.0005)
            if not worker.is_alive():
                worker = Hammered(driver)
                worker.start()
            if worker.in_call and _worker_in_target(worker):
                delivered += inject(worker)
        worker.stop.set()
        worker.join(5.0)
        contender.stop.set()
        contender.join(5.0)
    finally:
        threading.excepthook = previous_hook
    assert not worker.is_alive(), "the worker hung on a lock"
    assert not contender.is_alive(), "the contender hung on a leaked lock"
    assert delivered > INJECTIONS // 2, "the interrupts did not reach the worker"
    assert driver.closed > 0, "the worker never completed a close"
    assert lock_is_free(lock), "an interrupt leaked _stdin_lock"
    # A later close on another thread still gets through the lock.
    taker = threading.Thread(target=driver.proc.close, daemon=True)
    taker.start()
    taker.join(10.0)
    assert not taker.is_alive(), "a later close hung on the leaked lock"


def test_cli_process_close_survives_injection() -> None:
    """No ``_stdin_lock`` site ``close()`` runs on the task thread leaks it under injection."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer()"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]


if __name__ == "__main__":
    pytest.main([__file__])
