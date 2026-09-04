# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for findings F8 (artifact dir) and F9 (step limit).

* **F8** — ``config.set_artifact_base_dir`` had no production caller and
  was the only runtime writer of the ``_artifact_dir`` global.  Because
  ``Base.get_trajectory_path`` resolves that global at *save* time, a
  swap during a run sent a trajectory to a different root than the one
  the agent started under.  The directory is now fixed for the process.
* **F9** — the ``max_steps`` bound was enforced twice: by
  ``for _ in range(self.max_steps)`` in ``_run_agentic_loop`` and again
  by ``step_count > self.max_steps`` inside ``_check_limits``.  The
  second branch was unreachable from the loop and raised a *differently
  worded* error, so callers matching on the message had two strings to
  handle for one condition.

The step-limit tests drive a real ``KISSAgent.run()`` against a real
local HTTP server speaking the OpenAI chat-completions protocol, so
there are no mocks, no test doubles, and no paid LLM calls.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from kiss.core.base import Base
from kiss.core.config import get_artifact_dir, get_jobs_root
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import BudgetExceededError, KISSError

_MODEL = "gpt-4o-mini"


def note(text: str) -> str:
    """Record a note and keep going.

    Args:
        text: The note to record.

    Returns:
        A confirmation string.
    """
    return f"noted: {text}"


class _NeverFinishesHandler(BaseHTTPRequestHandler):
    """Serve a tool call to ``note`` forever, so the agent never finishes."""

    def log_message(self, format: str, *args: object) -> None:
        """Silence the default stderr access log."""

    def do_POST(self) -> None:  # noqa: N802
        """Answer any chat-completions request with one ``note`` tool call."""
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        tool_call: dict[str, object] = {
            "id": "call_note",
            "type": "function",
            "function": {"name": "note", "arguments": '{"text": "still working"}'},
        }
        if body.get("stream"):
            self._send_stream(tool_call)
        else:
            self._send_json(tool_call)

    def _send_json(self, tool_call: dict[str, object]) -> None:
        payload = json.dumps({
            "id": "chatcmpl-steps",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_stream(self, tool_call: dict[str, object]) -> None:
        deltas = [
            {"role": "assistant", "content": ""},
            {"tool_calls": [{"index": 0, **tool_call}]},
        ]
        chunks: list[dict[str, object]] = [
            {
                "id": "chatcmpl-steps",
                "object": "chat.completion.chunk",
                "model": _MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            for delta in deltas
        ]
        chunks.append({
            "id": "chatcmpl-steps",
            "object": "chat.completion.chunk",
            "model": _MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        })
        chunks.append({
            "id": "chatcmpl-steps",
            "object": "chat.completion.chunk",
            "model": _MODEL,
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        })
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def never_finishing_model() -> Iterator[str]:
    """Run a real local OpenAI-compatible endpoint that never finishes."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NeverFinishesHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)


def _run_until_limit(base_url: str, max_steps: int) -> tuple[KISSAgent, KISSError]:
    agent = KISSAgent(f"stepper-{max_steps}")
    with pytest.raises(KISSError) as excinfo:
        agent.run(
            _MODEL,
            "Keep taking notes; never finish.",
            tools=[note],
            max_steps=max_steps,
            model_config={"base_url": base_url, "api_key": "local"},
            verbose=False,
        )
    return agent, excinfo.value


def test_a_trajectory_saved_late_lands_where_the_first_save_put_it() -> None:
    """F8: one run must produce one trajectory file, not two orphans.

    ``Base.get_trajectory_path`` resolves the artifact root at *save*
    time rather than at run start, so a root that moved mid-run would
    scatter a single agent's trajectory across two directories — the
    later save silently orphaning everything written before it, with no
    error anywhere.  Real agents, real saves, real files.
    """
    agent = Base("f8 lifetime")
    agent._add_message("user", "written-before")
    agent._save()
    first_path = agent.get_trajectory_path()
    assert first_path.is_file()

    # Everything a busy process does between two saves of the same run.
    _resolve_roots_concurrently()

    agent._add_message("user", "written-after")
    agent._save()

    assert agent.get_trajectory_path() == first_path
    saved = first_path.read_text(encoding="utf-8")
    assert "written-before" in saved, "the early messages were orphaned"
    assert "written-after" in saved
    # Scoped to THIS agent's name: a bare ``*_{agent.id}_*`` glob also
    # matches same-id trajectories left by earlier tests in the same
    # process (another module's fixture restores Base.agent_counter, so
    # ids repeat), failing this test on shared-process runs.
    siblings = sorted(
        first_path.parent.glob(f"trajectory_f8_lifetime_{agent.id}_*.yaml")
    )
    assert siblings == [first_path], (
        f"the run left its trajectory in more than one place: {siblings}"
    )
    assert first_path.parent.parent == Path(get_artifact_dir())
    assert Path(get_artifact_dir()).parent == get_jobs_root()


def _resolve_roots_concurrently() -> list[str]:
    """Resolve the artifact root from eight real threads at once.

    Returns:
        Each thread's resolved job directory.
    """
    resolved: list[str] = []
    lock = threading.Lock()

    def resolve() -> None:
        agent = Base("f8 concurrent")
        with lock:
            resolved.append(str(agent.get_trajectory_path().parent.parent))

    threads = [threading.Thread(target=resolve) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return resolved


def test_concurrent_agents_share_one_stable_artifact_root() -> None:
    """F8: parallel sub-agents all resolve the same jobs root."""
    resolved = _resolve_roots_concurrently()

    assert set(resolved) == {get_artifact_dir()}


def test_step_limit_raises_one_consistent_error(never_finishing_model: str) -> None:
    """F9: exhausting ``max_steps`` reports the single agreed message."""
    agent, error = _run_until_limit(never_finishing_model, max_steps=3)

    assert agent.step_count == 3
    assert str(error) == f"KISS Error: Agent {agent.name} exceeded 3 steps."


def test_step_limit_of_one_stops_after_a_single_step(
    never_finishing_model: str,
) -> None:
    """F9: the bound is checked before the step, so ``max_steps=1`` runs one step."""
    agent, error = _run_until_limit(never_finishing_model, max_steps=1)

    assert agent.step_count == 1
    assert str(error) == f"KISS Error: Agent {agent.name} exceeded 1 steps."


def test_budget_limit_still_has_its_own_error(never_finishing_model: str) -> None:
    """F9 guard: collapsing the step bound must not disturb the budget bound."""
    agent = KISSAgent("budget-stepper")
    with pytest.raises(BudgetExceededError) as excinfo:
        agent.run(
            _MODEL,
            "Keep taking notes; never finish.",
            tools=[note],
            max_steps=50,
            max_budget=0.0,
            model_config={"base_url": never_finishing_model, "api_key": "local"},
            verbose=False,
        )

    assert "budget exceeded" in str(excinfo.value)
    assert agent.step_count == 1
