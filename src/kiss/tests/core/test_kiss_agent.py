# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for KISSAgent: agentic mode, non-agentic mode, retry behavior, error handling.

Merged from: test_kiss_agent_agentic, test_kiss_agent_non_agentic,
test_kiss_agent_coverage, test_error_handling, test_kiss_agent_retry.
"""

from __future__ import annotations

import json
import unittest
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from typing import Any

import httpx
import httpx2  # the Anthropic SDK (>= 1.0) builds its errors on httpx2, not httpx
import pytest
from anthropic import AuthenticationError as AnthropicAuthError
from openai import AuthenticationError as OpenAIAuthError

from kiss.core.base import Base
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.llm_reply import LLMReply
from kiss.core.models.model import Model
from kiss.core.models.model_info import MODEL_INFO
from kiss.core.printer import Printer
from kiss.core.tool_verdict import ALLOW, Verdict
from kiss.tests.conftest import requires_gemini_api_key, simple_calculator
from kiss.tests.core import local_anthropic_server as anthropic_server
from kiss.tests.core.test_audit1101_c2_vision_bridge import _vision_endpoint
from kiss.tests.core.test_budget_enforcement_e2e import (
    _read_body,
    _send_json,
    _start_server,
    _tool_calls_response,
)
from kiss.tests.core.test_codex_mcp_bridge import snapshot
from kiss.tests.core.test_vision_bridge_2026_10_31 import TEXT_ONLY_MODEL, _vision_turn

# A routed name has no OpenRouter twin; pinning ``fallback`` to ``None`` as
# well makes "no fallback" hold even if ``~/.kiss/MY_MODELS.json`` declares
# one, so the loop raises instead of swapping models.
_NO_FALLBACK_MODEL = "openrouter/openai/gpt-4o-mini"


def _pin_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODEL_INFO[_NO_FALLBACK_MODEL], "fallback", None)

TEST_MODEL = "gemini-3-flash-preview"

_DUMMY_REQUEST = httpx.Request("GET", "https://api.example.com/")


def _openai_auth_error(msg: str = "Incorrect API key provided") -> OpenAIAuthError:
    return OpenAIAuthError(
        message=msg,
        response=httpx.Response(401, request=_DUMMY_REQUEST),
        body=None,
    )


def _anthropic_auth_error(msg: str = "invalid x-api-key") -> AnthropicAuthError:
    return AnthropicAuthError(
        message=msg,
        response=httpx2.Response(401, request=httpx2.Request("GET", "https://api.example.com/")),
        body=None,
    )


class _RetryableErrorModel:
    serves_tools_natively = False  # no native tool set is built for it
    native_tools_exposed = Model.native_tools_exposed  # the base no-op
    # The context policies of the base model: none applies.
    clears_tool_uses_in_bulk = Model.clears_tool_uses_in_bulk
    keeps_reasoning_turns = Model.keeps_reasoning_turns
    compacts_whole_turns = Model.compacts_whole_turns

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0
        self.model_name = "gpt-4o-mini"
        self.conversation: list[dict[str, Any]] = []

    def initialize(self, prompt: str, attachments: list[Any] | None = None) -> None:
        self.conversation.append({"role": "user", "content": prompt})

    def generate_and_process_with_tools(
        self, function_map: dict[str, Any], tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        self.calls += 1
        if self.calls <= self.failures:
            raise Exception("Internal server error")
        return ([{"name": "finish", "arguments": {"result": "ok"}}], "done", object())

    def add_message_to_conversation(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})

    def set_usage_info_for_messages(self, usage_info: str) -> None:
        self.last_usage = usage_info

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        self.function_results = function_results

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int]:
        return (0, 0, 0, 0)


class _NonRetryableErrorModel(_RetryableErrorModel):
    def __init__(self, failures: int) -> None:
        super().__init__(failures)
        self.model_name = _NO_FALLBACK_MODEL

    def generate_and_process_with_tools(
        self, function_map: dict[str, Any], tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        raise Exception("Unauthorized: invalid API key")


class _UnknownPricedModel(_RetryableErrorModel):
    def __init__(self) -> None:
        super().__init__(failures=0)
        self.model_name = "unknown-budget-model"

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int]:
        return (1000, 1000, 0, 0)


def _make_agent(model_obj: Any, max_steps: int = 5) -> KISSAgent:
    agent = KISSAgent("RetryTest")
    agent.model = model_obj
    agent.model_name = model_obj.model_name
    agent.printer = None
    agent.is_agentic = True
    agent.max_steps = max_steps
    agent.max_budget = 1.0
    agent.function_map = {"finish": agent.finish}
    agent.messages = []
    agent.step_count = 0
    agent.total_tokens_used = 0
    agent.budget_used = 0.0
    agent.run_start_timestamp = 0
    agent._cached_tools_schema = None
    agent._consecutive_no_tool_calls = 0
    agent._model_config = None
    agent._fallback_used = False
    return agent


@pytest.fixture(autouse=True)
def _restore_base_state() -> Iterator[None]:
    original_counter = Base.agent_counter
    yield
    Base.agent_counter = original_counter


@requires_gemini_api_key
class TestDuplicateToolRaises(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Error Test Agent")

    def test_duplicate_tool_raises_error(self) -> None:
        with self.assertRaises(KISSError) as context:
            self.agent.run(
                model_name=TEST_MODEL,
                prompt_template="Test prompt",
                tools=[simple_calculator, simple_calculator],
            )
        self.assertIn("already registered", str(context.exception))


@requires_gemini_api_key
class TestNonAgenticWithToolsRaises(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = KISSAgent("Non-Agentic Test Agent")

    def test_non_agentic_with_tools_raises_error(self) -> None:
        try:
            self.agent.run(
                model_name=TEST_MODEL,
                prompt_template="Test prompt",
                tools=[simple_calculator],
                is_agentic=False,
            )
            self.fail("Expected KISSError to be raised")
        except KISSError as e:
            self.assertIn("Tools cannot be provided", str(e))
        except AttributeError:
            pass


@requires_gemini_api_key
class TestNonAgenticGeneration(unittest.TestCase):
    def test_non_agentic_returns_response(self) -> None:
        agent = KISSAgent("NonAgentic")
        result = agent.run(
            model_name=TEST_MODEL,
            prompt_template="Reply with exactly: HELLO",
            is_agentic=False,
            verbose=False,
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


@requires_gemini_api_key
class TestSetupToolsWebBranch(unittest.TestCase):
    def test_custom_finish_tool_not_overridden(self) -> None:
        def finish(result: str) -> str:
            """Finish the task with the given result.

            Args:
                result: The final result.

            Returns:
                The result string.
            """
            return f"custom:{result}"

        agent = KISSAgent("CustomFinish")
        result = agent.run(
            model_name=TEST_MODEL,
            prompt_template="Call finish with result='hello'.",
            tools=[finish],
            is_agentic=True,
            max_steps=5,
            verbose=False,
        )
        self.assertIn("custom:", result)


class TestAgenticLoopAuthError(unittest.TestCase):
    """Test that auth errors fail fast instead of retrying until max_steps."""

    INVALID_KEY_CONFIG = {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-invalid-key-for-testing",
    }
    # ``gpt-4o-mini`` would fall back to its OpenRouter twin (with the
    # real key) whenever ``OPENROUTER_API_KEY`` is configured; the auth
    # error must be final for the step bound below to mean anything.
    MODEL = _NO_FALLBACK_MODEL

    def setUp(self) -> None:
        monkeypatch = pytest.MonkeyPatch()
        self.addCleanup(monkeypatch.undo)
        _pin_no_fallback(monkeypatch)

    def test_auth_error_raises_kiss_error_fast(self) -> None:
        agent = KISSAgent("Auth Error Test")

        def dummy_tool() -> str:
            """A tool. Call this tool."""
            return "ok"

        with self.assertRaises(KISSError) as ctx:
            agent.run(
                model_name=self.MODEL,
                prompt_template="Call dummy_tool then finish.",
                tools=[dummy_tool],
                is_agentic=True,
                max_steps=10,
                max_budget=1.0,
                verbose=False,
                model_config=self.INVALID_KEY_CONFIG,
            )
        self.assertIn("non-retryable", str(ctx.exception).lower())
        self.assertLessEqual(agent.step_count, 1)

    def test_non_agentic_auth_error_propagates(self) -> None:
        agent = KISSAgent("Non-Agentic Auth Error Test")
        with self.assertRaises(Exception):
            agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Say hello",
                is_agentic=False,
                verbose=False,
                model_config=self.INVALID_KEY_CONFIG,
            )


def test_run_agentic_loop_raises_immediately_for_non_retryable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_no_fallback(monkeypatch)
    agent = _make_agent(_NonRetryableErrorModel(failures=0))

    with pytest.raises(
        KISSError,
        match="Non-retryable error from model: Unauthorized: invalid API key",
    ):
        agent._run_agentic_loop()


def test_unknown_model_pricing_error_is_not_swallowed() -> None:
    agent = _make_agent(_UnknownPricedModel())

    with pytest.raises(KISSError, match="Cannot calculate budget for unknown model"):
        agent._run_agentic_loop()


class _ScriptedChat(BaseHTTPRequestHandler):
    """A Chat Completions endpoint that answers its n-th request with the n-th reply."""

    replies: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        self.requests.append(json.loads(_read_body(self)))
        _send_json(self, self.replies[len(self.requests) - 1])

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _BusyHooks:
    """Hooks whose store is busy on the first two replies and on the first ``echo("b")``."""

    def __init__(self) -> None:
        self.replies = 0
        self.failed_b = False

    def on_reply(self, reply: LLMReply) -> Verdict:
        self.replies += 1
        if self.replies <= 2:
            raise RuntimeError("the hook's database is locked")
        return ALLOW

    def on_call(self, name: str, args: dict[str, Any]) -> Verdict:
        if args == {"text": "b"} and not self.failed_b:
            self.failed_b = True
            raise RuntimeError("the hook's database is locked")
        return ALLOW


def _told(message: dict[str, Any]) -> str:
    """What *message* of a request tells the model: its role, or what a tool result says."""
    if message["role"] != "tool":
        return str(message["role"])
    busy = "RuntimeError: the hook's database is locked" in message["content"]
    said = "no result" if busy else message["content"].split("\n")[0]
    return f"{message['tool_call_id']}: {said}"


ECHOED: list[str] = []


def echo(text: str) -> str:
    """Echo *text*.

    Args:
        text: The text to echo.
    """
    ECHOED.append(text)
    return f"echo {text}"


def test_a_step_a_hook_failed_is_retried_with_every_tool_call_answered() -> None:
    """A hook that raises once the model's reply is in the conversation ends the step, and
    the loop retries.  The reply's tool calls then had no results: the retry was a request
    with ``call_1`` and ``call_2`` unanswered, which a provider rejects, so one busy hook
    cost the run.  Each call is answered before the retry: one that ran keeps its result
    (the model need not repeat it), the others read why they have none; a reply without
    calls gets nothing invented.  The trajectory holds such a step like one that completed."""
    ECHOED.clear()
    a_and_b = _tool_calls_response([("echo", '{"text": "a"}'), ("echo", '{"text": "b"}')], 10, 5)
    only_text = _tool_calls_response([], 10, 5)
    only_text["choices"][0] |= {
        "message": {"role": "assistant", "content": "Let me think."},
        "finish_reason": "stop",
    }
    _ScriptedChat.replies = [
        a_and_b,  # the response hook fails: neither call ran
        only_text,  # it fails again: no call to answer
        _tool_calls_response([("echo", '{"text": "c"}')], 10, 5),  # a step that completes
        a_and_b,  # the call hook fails on b: a ran
        _tool_calls_response([("finish", '{"result": "done"}')], 10, 5),
    ]
    _ScriptedChat.requests = []
    hooks = _BusyHooks()
    server, url = _start_server(_ScriptedChat)
    agent = KISSAgent("busy hooks")
    try:
        result = agent.run(
            model_name="gpt-4o-mini",
            prompt_template="Echo a and b.",
            tools=[echo],
            max_steps=6,
            max_budget=1.0,
            verbose=False,
            model_config={"base_url": url, "api_key": "k"},
            llm_response_hook=hooks.on_reply,
            tool_call_hook=hooks.on_call,
        )
    finally:
        server.shutdown()
    assert result == "done" and ECHOED == ["c", "a"]
    assert [_told(message) for message in _ScriptedChat.requests[-1]["messages"]] == [
        "user",
        *("assistant", "call_1: no result", "call_2: no result", "user"),
        *("assistant", "user"),
        *("assistant", "call_1: echo c"),
        *("assistant", "call_1: echo a", "call_2: no result", "user"),
    ]
    steps = [m["content"] for m in agent.messages if "[echo]: " in m["content"]]
    assert [step.count("the hook's database is locked") for step in steps] == [2, 0, 1]
    assert steps[2].startswith("[echo]: echo a\n\n[echo]: ")


class _BusyOncePrinter(Printer):
    """A printer whose event store is busy for the first ``llm_call`` event."""

    def __init__(self) -> None:
        self.types: list[str] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:  # noqa: A002
        self.types.append(type)
        if type == "llm_call" and self.types.count("llm_call") == 1:
            raise RuntimeError("the event store is locked")

    def token_callback(self, token: str) -> None:
        pass


def test_a_printer_that_fails_on_the_llm_call_event_does_not_cost_the_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``llm_call`` event is printed once the model's reply is in its conversation.  A
    printer that raised there ended the step before the agent had the reply's tool calls:
    the loop retried with ``toolu_1`` unanswered, a request the provider rejects.  The
    event records a call already made and billed, so its loss is logged and the step goes
    on: the call runs and is answered.  (A printer makes the agent stream, so the scripted
    Anthropic stand-in plays the model.)"""
    ECHOED.clear()
    script = [
        anthropic_server.tool_use_message("echo", {"text": "a"}, 100, "toolu_1"),
        anthropic_server.tool_use_message("finish", {"result": "done"}, 100, "toolu_2"),
    ]
    printer = _BusyOncePrinter()
    with anthropic_server.serve(script) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
        result = KISSAgent("busy printer").run(
            model_name=anthropic_server.MODEL,
            prompt_template="Echo a.",
            tools=[echo],
            max_steps=3,
            max_budget=1.0,
            printer=printer,
        )
    assert result == "done" and ECHOED == ["a"]
    answer = [r for r in requests if r.get("stream")][1]["messages"][2:]
    assert [m["role"] for m in answer] == ["user"], "a retry message followed the reply"
    assert [(b["type"], b["tool_use_id"]) for b in answer[0]["content"]] == [
        ("tool_result", "toolu_1")
    ]
    assert printer.types.count("llm_call") == 2


class _StoreBusyOnce:
    """A budget hook whose store is busy once: at the first check after a picture is described."""

    def __init__(self, described: list[dict[str, Any]]) -> None:
        self.described = described
        self.failed = False

    def check(self) -> None:
        if self.described and not self.failed:
            self.failed = True
            raise RuntimeError("the budget store is locked")


def test_results_whose_hand_over_failed_reach_the_model_once_as_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that cannot see has each picture of a tool result described as the results
    are handed to it, and each description asks the budget hook first.  The hook failed
    before the last picture: the results before it were in the conversation, the last was
    not, and the loop retried with a call unanswered, a request the provider rejects.
    The adapter now adds a batch whole or not at all, and the agent hands a batch that
    failed over once more as text: every call is answered exactly once, no tool runs and
    no description is bought a second time, a result without a picture reads as it did,
    and the pictures left out are named with the error."""
    ECHOED.clear()
    calls = [("echo", '{"text": "a"}'), ("snapshot", "{}"), ("snapshot", "{}")]
    _ScriptedChat.replies = [
        _tool_calls_response(calls, 10, 5),
        _tool_calls_response([("finish", '{"result": "done"}')], 10, 5),
    ]
    _ScriptedChat.requests = []
    server, url = _start_server(_ScriptedChat)
    agent = KISSAgent("busy budget store")
    try:
        with _vision_endpoint(monkeypatch, [b"".join(_vision_turn("A picture."))]) as endpoint:
            agent.budget_check_hook = _StoreBusyOnce(endpoint.requests).check
            result = agent.run(
                model_name=TEXT_ONLY_MODEL,
                prompt_template="Echo a and take two snapshots.",
                tools=[echo, snapshot],
                max_steps=3,
                max_budget=1.0,
                verbose=False,
                model_config={"base_url": url, "api_key": "k"},
            )
    finally:
        server.shutdown()
    assert result == "done" and ECHOED == ["a"]
    assert len(endpoint.requests) == 1, "the pictures were described again for the retry"
    told = _ScriptedChat.requests[-1]["messages"]
    assert [m.get("tool_call_id", m["role"]) for m in told] == [
        "user",
        *("assistant", "call_1", "call_2", "call_3", "user"),
    ]
    assert told[2]["content"].startswith("echo a")
    assert "Harness:" not in told[2]["content"] and "attachment(s)" not in told[2]["content"]
    for answer in told[3:5]:
        assert answer["content"].startswith("captured [attached image/png, 12 bytes]\n\n[")
        assert "1 attachment(s)" in answer["content"]
        assert "RuntimeError: the budget store is locked" in answer["content"]
        assert "KISS_BINARY_ATTACHMENT" not in answer["content"]
    assert told[5]["content"].startswith("Failed to get response from Model: the budget store")


def write(file_path: str, content: str) -> str:
    """Write *content* to *file_path*.

    Args:
        file_path: The file to write.
        content: Its new content.
    """
    return f"wrote {len(content)} characters to {file_path}"


def pause(seconds: str) -> str:
    """Pause.

    Args:
        seconds: How long, as a number.
    """
    return f"paused {float(seconds)} s"


def test_a_failed_call_is_answered_with_an_excerpt_of_its_arguments() -> None:
    """A call its tool raised on is answered with the call's arguments, and they came back
    whole.  ``write`` with ``path`` for ``file_path`` and 100,000 characters of content was
    answered with all of them (100,399 characters in a ``SorcarAgent`` run): the content was
    in the context twice before the corrected call added it a third time.  The model wrote
    the call, so each argument is shown by its first 120 characters and how many follow.
    The error repeats arguments as well (``float`` quotes the text it could not convert), as
    does the hint for an argument that is none, so the result as a whole is cut to its first
    and last 1,500 characters."""
    long_text = "A" * 100_000
    calls = [
        ("write", json.dumps({"path": "notes.txt", "content": long_text})),
        ("pause", json.dumps({"seconds": long_text})),
    ]
    _ScriptedChat.replies = [
        _tool_calls_response(calls, 10, 5),
        _tool_calls_response([("finish", '{"result": "done"}')], 10, 5),
    ]
    _ScriptedChat.requests = []
    server, url = _start_server(_ScriptedChat)
    try:
        result = KISSAgent("failed calls").run(
            model_name="gpt-4o-mini",
            prompt_template="Write the notes.",
            tools=[write, pause],
            max_steps=3,
            max_budget=1.0,
            verbose=False,
            model_config={"base_url": url, "api_key": "k"},
        )
    finally:
        server.shutdown()
    assert result == "done"
    wrong_name, not_a_number = (  # each result, with the usage line the agent puts under it
        message["content"]
        for message in _ScriptedChat.requests[-1]["messages"]
        if message["role"] == "tool"
    )
    excerpt = "'" + "A" * 119 + " …[99,882 more chars]"
    assert wrong_name.startswith(
        f"Failed to call write with {{'path': 'notes.txt', 'content': {excerpt}}}: write() got "
        "an unexpected keyword argument 'path'"
    )
    assert "\nExpected signature: write(file_path: 'str', content: 'str') -> 'str'\n" in wrong_name
    assert len(wrong_name) < 700
    whole = (
        f"Failed to call pause with {{'seconds': {excerpt}}}: could not convert string to "
        f"float: {long_text!r}\nExpected signature: pause(seconds: 'str') -> 'str'\n"
    )
    assert not_a_number.startswith(whole[:1500] + "\n... (truncated) ...\n" + whole[-1500:])
    assert len(not_a_number) < 3400
