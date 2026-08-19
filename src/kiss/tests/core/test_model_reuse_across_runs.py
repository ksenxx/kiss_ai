# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `reuse_server` module fixture is imported from
# the core/models twin of this file and injected via test parameters)
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
import time

from kiss.core.kiss_agent import KISSAgent
from kiss.core.print_to_console import ConsolePrinter
from kiss.tests.core.models.openai_sse_harness import (
    ScriptedOpenAIServer,
)
from kiss.tests.core.models.test_model_reuse_across_runs import (  # noqa: F401
    _DEADLINE,
    _MODEL,
    _MODEL_CONFIG,
    _REASONING_THEN_CUT,
    _STALL_TIMEOUT,
    _TEXT_THEN_QUIET,
    _delta_chunk,
    _make_model,
    _ReusePolicy,
    _run_bounded,
    _stream_dies_mid_reasoning,
    reuse_server,
)


class TestThinkingBracketIsForgottenOnReset:
    """A run must not inherit the previous run's open thinking block."""

    def test_console_shows_no_stray_thinking_rule(
        self, reuse_server: tuple[ScriptedOpenAIServer, _ReusePolicy]
    ) -> None:
        """The user-visible symptom: a rule closing a block that never opened."""
        server, policy = reuse_server
        model = _stream_dies_mid_reasoning(server)

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
