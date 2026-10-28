# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regressions for the 2026-10-26 core-agent sweep (fixer group F1).

Each test reproduces one finding against real threads, processes and
sockets: a keep-alive ping thread that outlives a Stop landing on
``__exit__``, a tool-raised ``BudgetExceededError`` dropped from the
trajectory, a blocking ``flock`` that holds off a Stop, a watchdog that
signals a reaped leader, and a decisions request answered through a
Stop.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

import kiss.core.prompt_cache_keepalive as keepalive
from kiss.core import stop_signal
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.file_lock import lock_exclusive
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import BudgetExceededError
from kiss.core.models.decisions_model import DecisionsModel, noul
from kiss.core.models.model_info import calculate_cost
from kiss.core.processes import CappedOutput, _capped_timeout, popen_process_group
from kiss.core.prompt_cache_keepalive import PromptCacheKeepAlive
from kiss.tests.conftest import posix_only
from kiss.tests.core import local_anthropic_server as anthropic_server

# --- PromptCacheKeepAlive: a Stop landing on ``__exit__``'s entry ---------


class _CountingModel:
    """The one method the keep-alive thread calls, counting its pings."""

    def __init__(self) -> None:
        self.pings = 0

    def keep_prompt_cache_warm(
        self, function_map: dict[str, Any], tools_schema: Any
    ) -> dict[str, int]:
        """Record one ping and return a stand-in response for it."""
        self.pings += 1
        return {"ping": self.pings}


def _interrupt_on_exit_entry(frame: FrameType, event: str, arg: Any) -> None:
    """Raise ``KeyboardInterrupt`` at the first bytecode of ``__exit__``.

    That is where ``PyThreadState_SetAsyncExc`` delivers a task Stop
    whose previous check point was the body's last call: the ``with``
    statement has already handed control to ``__exit__`` but ``_stop``
    is not set yet, and the exception propagates out of the ``with``
    without a second ``__exit__`` call.
    """
    if event == "call" and frame.f_code is PromptCacheKeepAlive.__exit__.__code__:
        sys.setprofile(None)
        raise KeyboardInterrupt


def _pings_frozen(model: _CountingModel) -> bool:
    before = model.pings
    time.sleep(0.2)
    return model.pings == before


def slow_tool(timeout_seconds: int) -> str:
    """Sleep long enough for a keep-alive ping (standing in for a long build).

    Args:
        timeout_seconds: Declared timeout; only its size matters to the keep-alive.
    """
    time.sleep(0.3)
    return f"slept with timeout {timeout_seconds}"


def _pings_of(requests: list[dict[str, Any]]) -> int:
    return sum(1 for r in requests if not r.get("stream"))


def test_stop_on_exit_entry_is_repeated_by_the_agents_finally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_execute_tool_keeping_cache_warm``'s ``finally`` stops the pings again and bills them.

    The Stop lands on ``__exit__``'s entry inside the production caller
    (a real agent step against the local Anthropic server), so the
    ``with`` block never sets ``_stop``: the ping thread must still end
    with the tool, and the pings it made must land in the budget.
    """
    monkeypatch.setattr(keepalive, "KEEP_ALIVE_INTERVAL_SECONDS", 0.02)
    script = [
        anthropic_server.tool_use_message("slow_tool", {"timeout_seconds": 600}, 20_000, "toolu_1")
    ]
    with anthropic_server.serve(script) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
        agent = KISSAgent("keepalive-stop")
        sys.setprofile(_interrupt_on_exit_entry)
        try:
            with pytest.raises(KeyboardInterrupt):
                agent.run(
                    model_name=anthropic_server.MODEL,
                    prompt_template="Call slow_tool.",
                    tools=[slow_tool],
                    max_steps=3,
                    max_budget=50.0,
                    verbose=False,
                )
        finally:
            sys.setprofile(None)
        pings = _pings_of(requests)
        assert pings >= 1, "the tool ran long enough for a ping"
        time.sleep(0.2)
        assert _pings_of(requests) == pings, "the ping thread outlived the stopped tool"
    usage = anthropic_server.PING_USAGE
    ping_cost = calculate_cost(
        anthropic_server.MODEL,
        usage["input_tokens"],
        usage["output_tokens"],
        usage["cache_read_input_tokens"],
        usage["cache_creation_input_tokens"],
    )
    step_cost = calculate_cost(anthropic_server.MODEL, 20_000, 30)
    assert agent.budget_used == pytest.approx(step_cost + pings * ping_cost)


def test_ping_thread_quits_once_the_owner_thread_is_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without any ``stop()`` call, the thread still ends because the task's stop event is set."""
    monkeypatch.setattr(keepalive, "KEEP_ALIVE_INTERVAL_SECONDS", 0.02)
    model = _CountingModel()
    task_stop = threading.Event()
    stop_signal.set_thread_stop_event(task_stop)
    try:
        keep_alive = PromptCacheKeepAlive(model, {}, None, time.time(), max_pings=1000)
        sys.setprofile(_interrupt_on_exit_entry)
        try:
            with pytest.raises(KeyboardInterrupt), keep_alive:
                time.sleep(0.1)
                task_stop.set()  # what the Stop button does before injecting
        finally:
            sys.setprofile(None)
    finally:
        stop_signal.set_thread_stop_event(None)
    assert not keep_alive._stop.is_set(), "the interrupted __exit__ never set the flag"
    keep_alive._thread.join(2.0)
    assert not keep_alive._thread.is_alive(), "the ping thread outlived the stopped task"
    assert _pings_frozen(model)


# --- KISSAgent: a tool that raises BudgetExceededError ---------------------


def budget_tool(label: str) -> str:
    """Stand in for a tool whose sub-agents overspent the task's budget.

    Args:
        label: Echoed in the error so the trajectory can be checked.
    """
    raise BudgetExceededError(f"children spent the budget ({label})")


def test_a_tool_raised_budget_error_is_recorded_before_it_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The step that overran the budget lands in ``messages`` like a limit hit after a tool ran."""
    script = [
        anthropic_server.tool_use_message("budget_tool", {"label": "run-7"}, 20_000, "toolu_1"),
    ]
    with anthropic_server.serve(script) as (url, _requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
        agent = KISSAgent("budget-recorded")
        with pytest.raises(BudgetExceededError, match="run-7"):
            agent.run(
                model_name=anthropic_server.MODEL,
                prompt_template="Call budget_tool.",
                tools=[budget_tool],
                max_steps=3,
                max_budget=50.0,
                verbose=False,
            )
    roles = [m["role"] for m in agent.messages]
    assert roles == ["user", "model", "user"], roles
    assert "budget_tool(label='run-7')" in agent.messages[-2]["content"]
    assert agent.messages[-1]["content"] == (
        "[budget_tool]: KISS Error: children spent the budget (run-7)"
    )


# --- file_lock: a Stop while waiting for another process's lock -----------

_HOLDER = """
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)
fcntl.flock(fd, fcntl.LOCK_EX)
open(sys.argv[1] + ".held", "w").close()
time.sleep(60)
"""


@posix_only("the holder takes a POSIX flock")
def test_a_waiter_for_a_held_lock_can_be_interrupted(tmp_path: Path) -> None:
    """A ``KeyboardInterrupt`` injected into the waiting thread lands between two probes.

    A blocking ``flock`` would hold the exception off until the lock is
    granted, i.e. until the holder (a wedged daemon) goes away.
    """
    lock_path = tmp_path / "lock"
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(lock_path)])
    outcome: list[BaseException | bool] = []
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "lock.held").exists():
            assert time.monotonic() < deadline, "the holder never took the lock"
            time.sleep(0.01)
        fd = os.open(lock_path, os.O_RDWR)

        def wait_for_lock() -> None:
            try:
                outcome.append(lock_exclusive(fd))
            except BaseException as e:  # noqa: BLE001 — the interrupt is the result
                outcome.append(e)

        waiter = threading.Thread(target=wait_for_lock)
        waiter.start()
        time.sleep(0.2)  # the waiter is now blocked on the held lock
        assert waiter.ident is not None
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(waiter.ident), ctypes.py_object(KeyboardInterrupt)
        )
        waiter.join(2.0)
        assert not waiter.is_alive(), "the waiter is still blocked in the kernel"
        assert len(outcome) == 1 and isinstance(outcome[0], KeyboardInterrupt), outcome
        os.close(fd)
    finally:
        holder.kill()
        holder.wait(timeout=10)


# --- processes: the watchdog and a reaped leader ---------------------------


def test_timeout_fired_after_the_leader_was_reaped_is_ignored() -> None:
    """A reaped leader's pid may already belong to a stranger: no kill, no relabel."""
    proc = popen_process_group([sys.executable, "-c", "pass"])
    proc.wait(timeout=20)
    out = CappedOutput()
    _capped_timeout(proc, out)
    assert out.timed_out is False


# --- DecisionsModel: a Stop before or during the request -------------------


class _DecisionsHandler(BaseHTTPRequestHandler):
    """Answers every question ``True`` and sets the stop event of its server, if any."""

    def do_POST(self) -> None:  # noqa: N802 — http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)  # type: ignore[attr-defined]
        stop = self.server.stop_during_request  # type: ignore[attr-defined]
        if stop is not None:
            stop.set()
        answers = {name: {"value": True, "probability": 0.9} for name in body["questions"]}
        usage = {"input_tokens": 123, "output_tokens": 4, "cost": 0.0123}
        payload = json.dumps({"model": body["model"], "answers": answers, "usage": usage}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _serve_decisions(stop_during_request: threading.Event | None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DecisionsHandler)
    server.requests = []  # type: ignore[attr-defined]
    server.stop_during_request = stop_during_request  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _model_for(server: ThreadingHTTPServer) -> DecisionsModel:
    return DecisionsModel(
        "openrouter/~typesafe/jev-latest",
        base_url=f"http://127.0.0.1:{server.server_port}/api",
        api_key="k",
        model_config={"timeout": 5},
    )


def test_decide_refuses_to_ask_once_the_task_is_stopped() -> None:
    server = _serve_decisions(None)
    task_stop = threading.Event()
    stop_signal.set_thread_stop_event(task_stop)
    try:
        model = _model_for(server)
        assert model.decide("fine", {"ok": noul("Is it fine?")})["answers"]["ok"]["value"] is True
        task_stop.set()
        with pytest.raises(KeyboardInterrupt):
            model.decide("fine", {"ok": noul("Is it fine?")})
    finally:
        stop_signal.set_thread_stop_event(None)
        server.shutdown()
    assert len(server.requests) == 1, "the stopped call must not reach the endpoint"  # type: ignore[attr-defined]


def test_decide_keeps_a_completed_answer_when_stop_lands_during_the_request() -> None:
    """The provider charged for the answer: it is returned with its usage, not dropped."""
    task_stop = threading.Event()
    server = _serve_decisions(task_stop)
    stop_signal.set_thread_stop_event(task_stop)
    try:
        result = _model_for(server).decide("fine", {"ok": noul("Is it fine?")})
    finally:
        stop_signal.set_thread_stop_event(None)
        server.shutdown()
    assert task_stop.is_set()
    assert result["answers"]["ok"]["value"] is True
    assert result["usage"] == {"input_tokens": 123, "output_tokens": 4, "cost": 0.0123}
    assert len(server.requests) == 1  # type: ignore[attr-defined]
