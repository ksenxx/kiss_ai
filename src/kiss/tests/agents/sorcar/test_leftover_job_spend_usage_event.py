# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A leftover ``run_agent`` job's spend must reach the UI, not only the DB row.

Found by the 2026-10-05 cost audit.  ``SorcarAgent.run``'s ``finally``
stops every ``run_agent`` job the model neither waited for nor killed
(``kill_jobs_of``), and the stopped sub-task's failure ``result`` carries
what it spent, which the job thread folds into the parent's ledger
(``_attribute_dispatch_usage``).  That fold lands AFTER the run's
terminal ``result`` event: the persisted row (read from the agent's
totals once ``run`` returns) includes the spend, but no later event
carried the new total, so the chat header and the replayed history
showed the pre-fold cost.  The run must publish a ``usage_info`` with
the cumulative totals after the fold, as it already does for the
pre-run classifier's spend.

End to end: a real :class:`SorcarAgent` run against a local
chat-completions server whose model calls ``run_agent(wait="false")``
against a never-finishing local-WSS daemon stand-in and then finishes.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.agents.sorcar.agent_dispatch import make_agent_job_tool, make_run_agent_tool
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.json_printer import JsonPrinter
from kiss.tests.agents.sorcar.test_dispatch_timeout import _StopConfirmingDaemon
from kiss.tests.core.test_budget_enforcement_e2e import (
    _read_body,
    _send_json,
    _start_server,
    _tool_call_response,
)

_CHEAP = (10, 5)
_STOPPED_COST = "$1.0842"
# A minimal SEA for ``run_agent(agent=...)``: one ``BaseSea`` subclass per file.
_HELPER_SEA = (
    "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
    "class Sea(BaseSea):\n"
    "    def settings(self, settings):\n"
    "        return settings | {'model': 'm'}\n"
)


def _send_response(handler: BaseHTTPRequestHandler, resp: dict[str, Any], stream: bool) -> None:
    """Write *resp* as plain JSON or, when the request streamed, as SSE chunks."""
    if not stream:
        _send_json(handler, resp)
        return
    choice = resp["choices"][0]
    chunk = {"id": resp["id"], "object": "chat.completion.chunk", "model": resp["model"]}
    chunks = [
        {**chunk, "choices": [{
            "index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None,
        }]},
        {**chunk, "choices": [{
            "index": 0, "delta": {"tool_calls": choice["message"]["tool_calls"]},
            "finish_reason": choice["finish_reason"],
        }]},
        {**chunk, "choices": [], "usage": resp["usage"]},
    ]
    body = ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    handler.wfile.flush()


class _RunAgentThenFinishHandler(BaseHTTPRequestHandler):
    """Model script: ``run_agent(wait="false")`` once, then ``finish``."""

    requests = 0
    script = ""

    def do_POST(self) -> None:  # noqa: N802
        stream = bool(json.loads(_read_body(self) or "{}").get("stream", False))
        type(self).requests += 1
        if type(self).requests == 1:
            arguments = json.dumps({
                "task": "never finishes", "agent": type(self).script, "wait": "false",
            })
            _send_response(self, _tool_call_response("run_agent", arguments, *_CHEAP), stream)
            return
        arguments = json.dumps({
            "success": True, "is_continue": False, "summary_in_html": "<p>done</p>",
        })
        _send_response(self, _tool_call_response("finish", arguments, *_CHEAP), stream)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _CapturePrinter(JsonPrinter):
    """A real ``JsonPrinter`` whose broadcast events are collected."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:  # type: ignore[override]
        self.events.append(dict(event))


def _run(printer: _CapturePrinter, tmp_path: Path, model_url: str) -> SorcarAgent:
    """Run one ``SorcarAgent`` session against the scripted model.

    ``append_basic_tools=False`` keeps the run offline (no browser, no
    memory), so the real dispatch tools are passed explicitly, bound to
    the agent exactly as ``_get_tools`` binds them.
    """
    agent = SorcarAgent("leftover-job-parent")
    agent.run(
        tools=[make_run_agent_tool(str(tmp_path), agent), make_agent_job_tool(agent)],
        model_name="gpt-4o-mini",
        prompt_template="dispatch and finish",
        max_steps=4,
        max_budget=5.0,
        append_basic_tools=False,
        use_memory=False,
        web_tools=False,
        work_dir=str(tmp_path),
        printer=printer,
        verbose=False,
        model_config={"base_url": model_url, "api_key": "test-key"},
    )
    return agent


def test_leftover_job_spend_is_published_after_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The killed job's spend is in the agent's totals AND in a trailing event.

    The stopped sub-task's result carries $1.0842 / 4321 tokens / 7
    steps.  After the run: the agent's snapshot includes it (that is
    what the row stores), the last broadcast event is a ``usage_info``
    whose cost equals the snapshot, and it comes after the ``result``.
    """
    monkeypatch.setattr(cron_agent, "_daemon_endpoint_file", None)
    monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")
    daemon = _StopConfirmingDaemon(stopped_result={
        "type": "result", "taskId": "task-stopped-1", "success": False,
        "text": "Task stopped", "cost": _STOPPED_COST, "total_tokens": 4321,
        "step_count": 7,
    })
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    script = tmp_path / "helper.py"
    script.write_text(_HELPER_SEA)
    _RunAgentThenFinishHandler.requests = 0
    _RunAgentThenFinishHandler.script = str(script)
    srv, url = _start_server(_RunAgentThenFinishHandler)
    printer = _CapturePrinter()
    try:
        agent = _run(printer, tmp_path, url)
    finally:
        agent_dispatch.kill_jobs_of(None)
        srv.shutdown()
        daemon.close()
    assert _RunAgentThenFinishHandler.requests >= 2  # the live-jobs gate rejects one finish
    assert daemon.wait_for_command("stop"), "the leftover job was not stopped"
    budget, tokens, steps = agent.usage_snapshot()
    assert budget == pytest.approx(1.0842, abs=1e-3)
    assert tokens >= 4321 + 30
    assert steps >= 7 + 2

    types = [e.get("type") for e in printer.events]
    assert "result" in types, types
    last = printer.events[-1]
    assert last["type"] == "usage_info", types
    assert types.index("result") < len(types) - 1
    assert last["cost"] == f"${budget:.4f}"
    assert last["total_tokens"] == tokens
    assert last["total_steps"] == steps
    # Every earlier usage event predates the fold and so is below the total.
    earlier = [
        e for e in printer.events[:-1] if e.get("type") in ("usage_info", "result")
    ]
    assert earlier and all(
        float(str(e["cost"]).lstrip("$")) < budget for e in earlier if e.get("cost")
    ), earlier


def test_no_leftover_job_emits_no_extra_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run whose jobs all finished (or that had none) ends on its ``result``."""
    monkeypatch.setenv("KISS_DISABLE_TASK_CLASSIFIER", "1")

    class _FinishOnly(_RunAgentThenFinishHandler):
        requests = 1  # skip the dispatch reply

    srv, url = _start_server(_FinishOnly)
    printer = _CapturePrinter()
    try:
        agent = _run(printer, tmp_path, url)
    finally:
        srv.shutdown()
    assert agent.usage_snapshot()[0] > 0
    assert printer.events[-1]["type"] == "result", [e.get("type") for e in printer.events]
