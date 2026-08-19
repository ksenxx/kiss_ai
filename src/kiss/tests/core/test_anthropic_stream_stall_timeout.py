# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `stall_server` module fixture is imported from
# the core/models twin of this file and injected via test parameters)
"""End-to-end tests: a stalled Anthropic stream must not hang the agent.

Bug reproduction ("task stuck in thinking", production failure at
2026-07-21 10:08 in ``~/.kiss/sorcar.db``, task
``f554c68446fa42af89c2fd3c7cc14f63``, model = ``claude-fable-5``):

* Step 1 of the task completed normally (Read ./SORCAR.md).  Step 2's
  provider request was issued at 10:08:16.710 ("Step 2/100 start" in
  ``~/.kiss/kiss-web-stderr.log``) and then NOTHING happened: no stream
  event, no thinking delta, no log line, no error — for 5.5 minutes,
  until the user stopped the task by hand.  Concurrent tasks in the same
  process kept streaming normally, so this was a per-request hang.
* Root cause: ``AnthropicModel.initialize`` built ``Anthropic(api_key=…)``
  with the SDK defaults — ``httpx.Timeout(connect=5, read=600)`` and 2
  silent retries — and ``_create_message`` iterated the stream with no
  stall detection.  A request that the API accepts but never answers (or
  a stream that dies mid-turn) therefore blocks the agent's step loop
  for 10–30 minutes with zero output.  ``KISSAgent``'s retry/fallback
  machinery only reacts to raised exceptions, so it never fired.

Fix under test (including the cross-model-review findings):

1. ``AnthropicModel.initialize`` builds the client with a bounded
   no-bytes-flowing timeout (``httpx.Timeout(stream_stall_timeout,
   connect=10)``; default 180s, overridable via
   ``model_config["stream_stall_timeout"]``) and ``max_retries=1`` so
   the SDK's silent pre-header retries are bounded too.
2. ``_create_message`` converts both ``httpx.TimeoutException``
   (mid-stream byte stall) and ``anthropic.APITimeoutError`` (headers
   never arrive; SDK raises it after its own bounded retries) into a
   clear, retryable ``TimeoutError``.
3. An event-level ``_StreamStallWatchdog`` closes the response when no
   SSE *event* is yielded within the stall window — catching wedged
   requests that keep the connection alive with ``ping`` events, which
   the SDK filters out before yielding (so they reset the byte-level
   timeout while the agent still sees nothing).
4. A stall that strikes after a thinking block started closes the
   thinking bracket (``thinking_callback(False)``) so the UI does not
   stay in "thinking" mode across the retry.
5. ``KISSAgent._run_agentic_loop`` treats the ``TimeoutError`` like any
   retryable model error: it retries, and after ``MAX_CONSECUTIVE_ERRORS``
   raises a visible ``KISSError`` instead of hanging forever.

Test strategy (no mocks, patches of code under test, or fakes of the
SDK): a local ``ThreadingHTTPServer`` speaks the real Anthropic SSE wire
format to the real ``anthropic`` SDK client.  Stall modes reproduce the
distinct production-relevant hangs: ``silent`` (200 + SSE headers, then
zero bytes), ``no_headers`` (request accepted, response never starts),
``ping_only`` (only keep-alive pings forever), and ``think_then_ping``
(a thinking block starts, then only pings).  The client is routed to the
local server via the SDK's own ``ANTHROPIC_BASE_URL`` environment
variable so the fixed ``initialize()`` code path (client construction
incl. timeout/retries) is exercised verbatim.  Every potentially-hanging
call runs on a daemon worker thread with a hard deadline, so on pre-fix
code the tests FAIL fast instead of hanging CI.
"""

from __future__ import annotations

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import MODEL_INFO, ModelInfo
from kiss.tests.core.models.test_anthropic_stream_stall_timeout import (  # noqa: F401
    _FAST_FAIL_BUDGET,
    _MODEL,
    _STALL_TIMEOUT,
    _STATE,
    _DaemonThreadingHTTPServer,
    _finish_tool_stream,
    _message_start,
    _run_bounded,
    _sse,
    _StallHandler,
    _StallState,
    _thinking_block_prefix,
    stall_server,
)


def _register_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the synthetic ``claude-*`` model so the model() factory
    routes it through the real ``AnthropicModel`` (no fallback, so the
    retry path — not the fallback path — is what recovers)."""
    monkeypatch.setitem(
        MODEL_INFO,
        _MODEL,
        ModelInfo(
            context_length=128_000,
            input_price_per_million=0.0,
            output_price_per_million=0.0,
            is_function_calling_supported=True,
            is_embedding_supported=False,
            is_generation_supported=True,
            fallback=None,
            extended_thinking=False,
        ),
    )


def _ensure_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the model() factory a non-empty Anthropic key on CI machines."""
    from kiss.core import config as config_module

    if not getattr(config_module.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", ""):
        monkeypatch.setattr(
            config_module.DEFAULT_CONFIG,
            "ANTHROPIC_API_KEY",
            "test-key",
            raising=False,
        )


class TestAgentSurvivesStalledStream:
    """KISSAgent must retry after a stall and never get stuck thinking."""

    def test_agent_recovers_when_stream_stalls_once(
        self, monkeypatch: pytest.MonkeyPatch, stall_server: str
    ) -> None:
        """One dead request (the production scenario), then a healthy API:
        the agent must retry and finish instead of hanging forever."""
        _register_model(monkeypatch)
        _ensure_api_key(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_BASE_URL", stall_server)
        _STATE.stall_first_n = 1
        agent = KISSAgent("test-stall-recovery")
        outcome = _run_bounded(
            lambda: agent.run(
                model_name=_MODEL,
                prompt_template="Update ./README.md based on the latest code.",
                max_steps=5,
                max_budget=1.0,
                model_config={"stream_stall_timeout": _STALL_TIMEOUT},
                verbose=False,
            ),
            deadline=2 * _FAST_FAIL_BUDGET,
        )
        assert outcome == ("ok", "recovered")
        assert _STATE.request_count == 2

    def test_agent_fails_visibly_when_api_stays_dead(
        self, monkeypatch: pytest.MonkeyPatch, stall_server: str
    ) -> None:
        """Every request stalls: the agent must surface a KISSError after
        its bounded retries — a visible failure the orchestrator can
        report — rather than the silent infinite "thinking" of the bug."""
        _register_model(monkeypatch)
        _ensure_api_key(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_BASE_URL", stall_server)
        agent = KISSAgent("test-stall-hard-failure")
        outcome = _run_bounded(
            lambda: agent.run(
                model_name=_MODEL,
                prompt_template="Update ./README.md based on the latest code.",
                max_steps=5,
                max_budget=1.0,
                model_config={"stream_stall_timeout": _STALL_TIMEOUT},
                verbose=False,
            ),
            deadline=4 * _FAST_FAIL_BUDGET,
        )
        assert isinstance(outcome, KISSError)
        msg = str(outcome)
        assert "consecutive errors" in msg
        assert "stalled" in msg
