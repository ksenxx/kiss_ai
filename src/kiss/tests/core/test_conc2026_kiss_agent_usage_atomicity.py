# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: one model response's tokens and cost are accounted atomically.

Round-4 review, finding 1.  ``KISSAgent`` used to commit a response's
usage as two separate attribute stores — ``total_tokens_used`` first,
``budget_used`` (the cost) later.  The server's stop watchdog injects
``KeyboardInterrupt`` asynchronously (``PyThreadState_SetAsyncExc``),
which can land between ANY two bytecodes; landing in that gap unwound
the update with the tokens committed and the cost absent, and the
parent ``RelentlessAgent``'s ``except BaseException`` recovery bank
then recorded the torn triple PERMANENTLY.  The same gap let the
concurrent pollers (``_executor_usage`` used by the live usage monitor
and by abandoned-child reclaims) observe a response's tokens without
its cost.

The fix publishes the whole ``(budget_used, total_tokens_used,
step_count)`` triple as ONE immutable ``_UsageTotals`` snapshot stored
with a single ``STORE_ATTR``; the legacy scalar counters are derived
properties and ``KISSAgent.usage_snapshot()`` reads the snapshot once.

Both tests drive a REAL ``KISSAgent.run`` over a local HTTP
chat-completions server (real request, real response parsing, real
``calculate_cost`` pricing) with a real injected ``KeyboardInterrupt``
or a real paused-reader interleaving at every opcode boundary of the
response-accounting path.  No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler
from typing import Any

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.core.kiss_agent import KISSAgent
from kiss.tests.core.test_budget_enforcement_e2e import (
    _CHEAP,
    _read_body,
    _send_json,
    _start_server,
    _tool_call_response,
)

_WAIT = 30.0

#: The whole per-response accounting path: a stop can land between any
#: two bytecodes of the update itself or of the snapshot construction.
_UPDATE_CODES = frozenset(
    {
        KISSAgent._update_tokens_and_budget_from_response.__code__,
    }
)


class _FinishHandler(BaseHTTPRequestHandler):
    """Local chat-completions server whose model finishes immediately."""

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        _send_json(
            self,
            _tool_call_response("finish", json.dumps({"result": "done"}), *_CHEAP),
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _run_once(url: str, name: str) -> KISSAgent:
    """Run one real single-step task against the local server."""
    agent = KISSAgent(name)
    agent.run(
        model_name="gpt-4o-mini",
        prompt_template="Finish immediately.",
        max_steps=3,
        max_budget=1.0,
        model_config={"base_url": url, "api_key": "test-key"},
        verbose=False,
        print_prompts=False,
    )
    return agent


def _swept_run(
    url: str,
    name: str,
    boundary: int,
    on_boundary: str,
    paused: threading.Event | None = None,
    resume: threading.Event | None = None,
    out: dict[str, Any] | None = None,
) -> tuple[KISSAgent, bool, bool]:
    """Run one real task, acting at the ``boundary``-th accounting opcode.

    ``on_boundary`` is ``"inject"`` (raise one real ``KeyboardInterrupt``
    — the exact delivery model of ``PyThreadState_SetAsyncExc``) or
    ``"pause"`` (park the run thread so a concurrent reader can observe
    the executor mid-update).

    Returns:
        ``(agent, acted, interrupted)`` — ``acted`` is False when the
        accounting path had fewer opcode events than *boundary* (the
        sweep is complete).
    """
    agent = KISSAgent(name)
    if out is not None:
        out["agent"] = agent
    seen = 0
    acted = False

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        nonlocal seen, acted
        if frame.f_code in _UPDATE_CODES:
            frame.f_trace_opcodes = True
            if event == "opcode" and not acted:
                if seen == boundary:
                    acted = True
                    if on_boundary == "inject":
                        raise KeyboardInterrupt("injected stop")
                    assert paused is not None and resume is not None
                    paused.set()
                    assert resume.wait(timeout=_WAIT)
                else:
                    seen += 1
        return tracer

    interrupted = False
    try:
        sys.settrace(tracer)
        agent.run(
            model_name="gpt-4o-mini",
            prompt_template="Finish immediately.",
            max_steps=3,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "test-key"},
            verbose=False,
            print_prompts=False,
        )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        sys.settrace(None)
    return agent, acted, interrupted


def test_injected_stop_never_tears_response_tokens_from_cost() -> None:
    """A stop at ANY accounting opcode leaves a coherent, bankable triple.

    Sweeps every opcode boundary of the real response-accounting path
    during a real single-response run.  At each boundary a real
    ``KeyboardInterrupt`` unwinds the run; the executor's snapshot must
    then be all-or-nothing (the response's tokens and cost both present
    or both absent — never the round-4 torn ``(0.0, tokens)`` state),
    and the parent's ``except BaseException`` recovery bank
    (``_accumulate_usage``) must record exactly that coherent triple.
    Both interruption classes (before and after the single atomic
    snapshot store) must occur across the sweep, which proves the sweep
    starts BEFORE the source store and ends after it.
    """
    srv, url = _start_server(_FinishHandler)
    try:
        full = _run_once(url, "baseline").usage_snapshot()
        assert full[0] > 0 and full[1] > 0 and full[2] >= 1, full

        boundary = 0
        saw_unaccounted = False
        saw_accounted = False
        while True:
            agent, acted, interrupted = _swept_run(
                url, f"swept-{boundary}", boundary, "inject",
            )
            if not acted:
                break
            assert interrupted, f"boundary {boundary}: injection was swallowed"
            budget, tokens, steps = agent.usage_snapshot()
            assert (budget, tokens) in {(0.0, 0), (full[0], full[1])}, (
                f"boundary {boundary}: torn source triple "
                f"({budget}, {tokens}, {steps}); full={full}"
            )
            saw_unaccounted = saw_unaccounted or tokens == 0
            saw_accounted = saw_accounted or tokens == full[1]

            # The production recovery path: perform_task's
            # except-BaseException bank of the interrupted executor.
            parent = RelentlessAgent(f"parent-{boundary}")
            parent._accumulate_usage(agent)
            assert parent.usage_snapshot() == (budget, tokens, steps), (
                f"boundary {boundary}: the recovery bank distorted the triple"
            )
            boundary += 1
        assert boundary > 0
        assert saw_unaccounted, "no injection landed before the snapshot store"
        assert saw_accounted, "no injection landed after the snapshot store"
    finally:
        srv.shutdown()


def test_concurrent_reader_never_sees_tokens_without_cost() -> None:
    """A reader polling mid-update sees all-or-nothing, at every boundary.

    The round-4 review also demonstrated the tear WITHOUT interruption:
    ``_executor_usage`` (polled by ``_LiveUsageMonitor``,
    unfinished-usage collection, and abandoned-child reclaims on other
    threads) observed ``(0.0, 100, 1)`` between the token store and the
    cost store.  Here the run thread is parked at EVERY opcode boundary
    of the real accounting path in turn, and a reader thread reads the
    executor through the production reader (``_executor_usage`` via one
    ``usage_snapshot()``): every observation must be one of the two
    legal states.
    """
    from kiss.agents.sorcar.sorcar_agent import _executor_usage

    srv, url = _start_server(_FinishHandler)
    try:
        full = _run_once(url, "reader-baseline").usage_snapshot()
        boundary = 0
        while True:
            paused = threading.Event()
            resume = threading.Event()
            holder = types.SimpleNamespace(_current_executor=None)
            outcome: dict[str, Any] = {}

            def drive(b: int = boundary) -> None:
                _, outcome["acted"], _ = _swept_run(
                    url, f"read-{b}", b, "pause",
                    paused=paused, resume=resume, out=outcome,
                )

            runner = threading.Thread(target=drive, daemon=True)
            runner.start()
            # The reader can only observe the executor while it exists;
            # poll until the run thread either parks or finishes.
            while not paused.wait(timeout=0.01):
                if not runner.is_alive():
                    break
            if paused.is_set():
                # The run thread is parked INSIDE the accounting
                # update, so this read — through the production
                # executor reader — is provably concurrent with it.
                holder._current_executor = outcome["agent"]
                budget, tokens, _steps = _executor_usage(holder)
                assert (budget, tokens) in {(0.0, 0), (full[0], full[1])}, (
                    f"boundary {boundary}: reader saw a torn triple "
                    f"({budget}, {tokens})"
                )
                resume.set()
            runner.join(timeout=_WAIT)
            assert not runner.is_alive()
            if not outcome.get("acted"):
                break
            agent = outcome["agent"]
            assert isinstance(agent, KISSAgent)
            assert agent.usage_snapshot() == full
            boundary += 1
        assert boundary > 0
    finally:
        srv.shutdown()
