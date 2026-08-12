# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: what a reused model instance must forget between runs.

``KISSAgent._reset`` keeps the adapter object alive across runs — it only
resets its conversation and points it at the new run's printer — so any
per-stream state the adapter leaves behind survives into a task that has
nothing to do with it.

Handoff finding **A2**: the thinking bracket was such a state.  A stream
that dies mid-reasoning for a reason that is neither a user stop nor a
stall (a provider that drops the connection) unwinds through
``stop_aware_events``' bare ``raise`` — ``on_abort`` is only called for
the stop and the stall — so ``Model._thinking_open`` stays ``True`` on
the instance.  ``Model.reset_conversation()`` cleared the conversation
and the usage info but not that flag, and the next run's *fresh* printer
then received a closing ``thinking_callback(False)`` for a thinking block
it never opened: a stray "end of thinking" rule in the console, and a
panel switched back to text mode that was never in thinking mode.

Handoff finding **C3**: ``_reset`` rebound ``token_callback`` /
``thinking_callback`` by assigning the two attributes, which reaches only
the attributes and misses whatever a subclass owns internally.  It now
calls ``Model.rebind_callbacks()`` instead, so a transport that delegates
turns to a cached sub-model has one place to override.

No mocks, patches, fakes or test doubles: a real ``ThreadingHTTPServer``
speaks genuine Chat Completions SSE to the real ``openai`` SDK, the real
``OpenAICompatibleModel`` streams it, the real ``KISSAgent._reset``
performs the reuse, and the real ``ConsolePrinter`` renders the second
run into a ``StringIO``.
"""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Generator
from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.print_to_console import ConsolePrinter
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)

_MODEL = "gpt-reuse-under-test"
_STALL_TIMEOUT = 2.0
_DEADLINE = 30.0
_MODEL_CONFIG: dict[str, Any] = {"stream_stall_timeout": _STALL_TIMEOUT}


def _delta_chunk(delta: dict[str, Any]) -> bytes:
    """Render one Chat Completions chunk carrying *delta*."""
    return chat_chunk(
        {
            "id": "chatcmpl-reuse",
            "object": "chat.completion.chunk",
            "model": _MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )


_REASONING_THEN_CUT = [
    _delta_chunk({"role": "assistant", "reasoning_content": "Let me think"}),
    _delta_chunk({"content": "never arrives"}),
]
_TEXT_THEN_QUIET = [_delta_chunk({"role": "assistant", "content": "Hello there"})]


class _ReusePolicy:
    """Serves run 1's dying stream, then run 2's stream that goes quiet.

    Attributes:
        mode: ``"cut"`` drops the connection right after the reasoning
            delta (a provider crashing mid-turn — neither a stop nor a
            stall), ``"quiet"`` streams one text delta and then holds the
            connection open until the stall timeout fires.
        release: Frees a held connection at teardown.
    """

    def __init__(self) -> None:
        self.mode = "cut"
        self.release = threading.Event()

    def __call__(self, request: Request) -> Reply:
        """Answer *request* according to the current mode."""
        if self.mode == "cut":
            return Reply(sse_chunks=_REASONING_THEN_CUT, truncate_after=1)
        return Reply(sse_chunks=_TEXT_THEN_QUIET, hold=self.release)


@pytest.fixture
def reuse_server() -> Generator[tuple[ScriptedOpenAIServer, _ReusePolicy]]:
    """A real OpenAI-compatible endpoint driven by :class:`_ReusePolicy`."""
    policy = _ReusePolicy()
    with ScriptedOpenAIServer(policy) as server:
        yield server, policy
        policy.release.set()


def _make_model(
    base_url: str,
    token_callback: Any = None,
    thinking_callback: Any = None,
) -> OpenAICompatibleModel:
    """Build a real adapter pointed at the local endpoint."""
    return OpenAICompatibleModel(
        _MODEL,
        base_url=base_url,
        api_key="test-key",
        model_config=dict(_MODEL_CONFIG),
        token_callback=token_callback,
        thinking_callback=thinking_callback,
    )


def _run_bounded(call: Any) -> BaseException | None:
    """Run *call* on a worker thread bounded by :data:`_DEADLINE`.

    Args:
        call: The zero-argument turn to run.

    Returns:
        The exception it raised, or ``None`` when it returned.
    """
    outcome: dict[str, BaseException] = {}

    def target() -> None:
        try:
            call()
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(_DEADLINE)
    if worker.is_alive():
        pytest.fail(f"turn still running after {_DEADLINE}s")
    return outcome.get("error")


def _leak_thinking_bracket(server: ScriptedOpenAIServer) -> OpenAICompatibleModel:
    """Run a turn that dies mid-reasoning and return the damaged adapter.

    Args:
        server: The scripted endpoint, in ``"cut"`` mode.

    Returns:
        The adapter whose thinking bracket is still open.
    """
    model = _make_model(
        server.base_url,
        token_callback=lambda _t: None,
        thinking_callback=lambda _s: None,
    )
    model.initialize("Think, then answer.")
    error = _run_bounded(model.generate)
    assert error is not None, "the cut stream should have failed the turn"
    assert model._thinking_open is True, (
        "test setup no longer reproduces the leak: the adapter closed its "
        "thinking bracket on a non-stop, non-stall stream failure"
    )
    return model


class TestThinkingBracketIsForgottenOnReset:
    """A run must not inherit the previous run's open thinking block."""

    def test_reset_conversation_clears_the_bracket(
        self, reuse_server: tuple[ScriptedOpenAIServer, _ReusePolicy]
    ) -> None:
        """The flag itself must be gone once the conversation is reset."""
        server, _policy = reuse_server
        model = _leak_thinking_bracket(server)

        model.reset_conversation()

        assert model._thinking_open is False

    def test_next_run_gets_no_spurious_thinking_end(
        self, reuse_server: tuple[ScriptedOpenAIServer, _ReusePolicy]
    ) -> None:
        """The next run's printer must not be told a block it never saw ended."""
        server, policy = reuse_server
        model = _leak_thinking_bracket(server)

        thinking: list[bool] = []
        model.reset_conversation()
        model.rebind_callbacks(lambda _t: None, thinking.append)

        policy.mode = "quiet"
        model.initialize("Answer plainly.")
        error = _run_bounded(model.generate)

        assert isinstance(error, TimeoutError), f"got {error!r}"
        assert thinking == [], (
            f"the reused adapter reported thinking boundaries {thinking} to a "
            f"printer that never saw a thinking block start"
        )

    def test_console_shows_no_stray_thinking_rule(
        self, reuse_server: tuple[ScriptedOpenAIServer, _ReusePolicy]
    ) -> None:
        """The user-visible symptom: a rule closing a block that never opened."""
        server, policy = reuse_server
        model = _leak_thinking_bracket(server)

        out = io.StringIO()
        printer = ConsolePrinter(file=out)
        agent = KISSAgent("thinking-bracket-reuse")
        agent.model = model
        agent._reset(
            _MODEL,
            is_agentic=False,
            max_steps=None,
            max_budget=None,
            model_config=dict(_MODEL_CONFIG),
            printer=printer,
            verbose=True,
        )
        assert agent.model is model, "the adapter instance was not reused"

        policy.mode = "quiet"
        model.initialize("Answer plainly.")
        _run_bounded(model.generate)

        assert "─" not in out.getvalue(), (
            f"a thinking rule was drawn for a block that never started:\n"
            f"{out.getvalue()!r}"
        )


class TestRebindCallbacksReachesTheNewPrinter:
    """``_reset`` must route a reused adapter's stream to the new printer."""

    def test_streamed_text_lands_in_the_new_printer(
        self, reuse_server: tuple[ScriptedOpenAIServer, _ReusePolicy]
    ) -> None:
        """Tokens must reach the printer bound by the *second* run."""
        server, policy = reuse_server
        first = io.StringIO()
        model = _make_model(server.base_url, token_callback=ConsolePrinter(
            file=first
        ).token_callback)
        model.initialize("First run.")

        second = io.StringIO()
        printer = ConsolePrinter(file=second)
        agent = KISSAgent("rebind-under-test")
        agent.model = model
        agent._reset(
            _MODEL,
            is_agentic=False,
            max_steps=None,
            max_budget=None,
            model_config=dict(_MODEL_CONFIG),
            printer=printer,
            verbose=True,
        )
        assert agent.model is model, "the adapter instance was not reused"

        policy.mode = "quiet"
        agent.model.initialize("Second run.")
        started = time.monotonic()
        _run_bounded(agent.model.generate)

        assert "Hello there" in second.getvalue()
        assert first.getvalue() == "", "the first run's printer was still bound"
        assert time.monotonic() - started < _DEADLINE
