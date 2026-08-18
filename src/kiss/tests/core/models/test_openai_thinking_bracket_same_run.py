# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a broken OpenAI stream must close its thinking bracket.

``_stream_chat_completion`` opens the thinking bracket the moment a
provider emits its first ``reasoning_content`` delta, and used to close
it only when the loop ran to exhaustion.  ``stop_aware_events`` calls
``on_abort`` for a user stop and for a stall, but deliberately re-raises
every *other* transport failure untouched, so a provider that simply
dropped the connection mid-reasoning left the bracket open.

``KISSAgent._run_agentic_loop`` classifies such a failure as retryable
and takes the next step **in the same run**, without
``reset_conversation()``.  The printer was therefore still in thinking
mode when the retry's ordinary answer text arrived, and rendered the
answer as reasoning — grey italics under a "Thinking" rule that never
closed.

No mocks, patches, fakes or test doubles: a real ``ThreadingHTTPServer``
speaks genuine Chat Completions SSE to the real ``openai`` SDK, the real
``OpenAICompatibleModel`` streams it, and the real ``ConsolePrinter``
renders it into a ``StringIO``.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from typing import Any

import pytest

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)

_MODEL = "gpt-bracket-under-test"


_STALL_TIMEOUT = 5.0


_DEADLINE = 30.0


_ANSWER = "Two plus two is four"


def _delta_chunk(delta: dict[str, Any]) -> bytes:
    """Render one Chat Completions chunk carrying *delta*."""
    return chat_chunk(
        {
            "id": "chatcmpl-bracket",
            "object": "chat.completion.chunk",
            "model": _MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )


_REASONING_THEN_CUT = [
    _delta_chunk({"role": "assistant", "reasoning_content": "Let me think"}),
    _delta_chunk({"content": "never arrives"}),
]


_PLAIN_ANSWER = [
    _delta_chunk({"role": "assistant", "content": _ANSWER}),
    chat_chunk(
        {
            "id": "chatcmpl-bracket",
            "object": "chat.completion.chunk",
            "model": _MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 5,
                "total_tokens": 12,
            },
        }
    ),
]


class _CutThenAnswerPolicy:
    """Drops the first stream mid-reasoning, then answers normally.

    This is the provider behaviour behind the defect: the failure is
    neither a user stop nor a stall, so the abort watchdog re-raises it
    without closing anything.

    Attributes:
        requests: How many requests have been served so far.
    """

    def __init__(self) -> None:
        self.requests = 0

    def __call__(self, request: Request) -> Reply:
        """Answer *request*: cut the first stream, complete every later one."""
        self.requests += 1
        if self.requests == 1:
            return Reply(sse_chunks=_REASONING_THEN_CUT, truncate_after=1)
        return Reply(sse_chunks=_PLAIN_ANSWER)


@pytest.fixture
def cut_then_answer() -> Generator[tuple[ScriptedOpenAIServer, _CutThenAnswerPolicy]]:
    """A real endpoint that crashes the first stream and serves the second."""
    policy = _CutThenAnswerPolicy()
    with ScriptedOpenAIServer(policy) as server:
        yield server, policy


def _run_bounded(*calls: Any) -> list[BaseException | None]:
    """Run *calls* in order on ONE daemon thread, bounded by :data:`_DEADLINE`.

    One thread, because that is what the agent is: consecutive turns of
    a run share a thread, and ``ConsolePrinter`` keeps its block state
    thread-local.  Splitting the turns across threads would hand the
    second one a clean slate and hide the very leak under test.  The
    thread is a daemon so a wedged turn fails the test instead of
    wedging the session.

    Args:
        *calls: The zero-argument turns to run in sequence.

    Returns:
        One entry per call: the exception it raised, or ``None``.
    """
    outcome: list[BaseException | None] = []

    def target() -> None:
        for call in calls:
            try:
                call()
            except BaseException as exc:  # noqa: BLE001 — reported to the test
                outcome.append(exc)
            else:
                outcome.append(None)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(_DEADLINE)
    if worker.is_alive():
        pytest.fail(f"turn still running after {_DEADLINE}s")
    return outcome


def _make_model(
    base_url: str, token_callback: Any, thinking_callback: Any
) -> OpenAICompatibleModel:
    """Build a real adapter pointed at the local endpoint."""
    return OpenAICompatibleModel(
        _MODEL,
        base_url=base_url,
        api_key="test-key",
        model_config={"stream_stall_timeout": _STALL_TIMEOUT},
        token_callback=token_callback,
        thinking_callback=thinking_callback,
    )


def test_a_dropped_stream_closes_the_thinking_bracket_it_opened(
    cut_then_answer: tuple[ScriptedOpenAIServer, _CutThenAnswerPolicy],
) -> None:
    """Every opened bracket must be closed before the failure propagates."""
    server, _policy = cut_then_answer
    thinking: list[bool] = []
    model = _make_model(server.base_url, lambda _t: None, thinking.append)
    model.initialize("Think, then answer.")

    [error] = _run_bounded(model.generate)

    assert error is not None, "the cut stream should have failed the turn"
    assert thinking == [True, False], (
        f"the thinking bracket was left unbalanced: {thinking}"
    )
    assert model._thinking_open is False


def test_a_clean_stream_still_reports_one_balanced_bracket(
    cut_then_answer: tuple[ScriptedOpenAIServer, _CutThenAnswerPolicy],
) -> None:
    """The unconditional close must not double-report a healthy turn."""
    server, policy = cut_then_answer
    policy.requests = 1  # skip the cut reply; serve the complete stream
    thinking: list[bool] = []
    model = _make_model(server.base_url, lambda _t: None, thinking.append)
    model.initialize("Answer plainly.")

    content, _response = model.generate()

    assert content == _ANSWER
    assert thinking == [], "a turn without reasoning must open no bracket"
    assert model._thinking_open is False
