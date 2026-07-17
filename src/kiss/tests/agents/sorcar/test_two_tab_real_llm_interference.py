# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Two-tab cross-interference tests with real LLM calls.

Drives two :class:`VSCodeServer` ``_cmd_run`` invocations concurrently
against a real local HTTP server that speaks the OpenAI chat-completion
protocol (so :class:`ChatSorcarAgent` runs unmodified — same printer,
broadcast, recording, persistence, and subscriber routing as in
production).  Captures every payload the printer would put on the wire
and asserts that the two tabs cannot pollute each other's UI, usage
totals, task-end events, or queued-answer routing.

Hypotheses (each verified by one test method):

* **H1 — no content event leaks globally.**  No display event must be
  broadcast with neither ``tabId`` nor a resolvable thread-local
  ``taskId``: any such event would land in whichever tab is active on
  the frontend, polluting a different running task's chat surface.

* **H2 — per-tab fan-out is strict.**  Every payload that carries a
  ``taskId`` must also carry the ``tabId`` of the tab that *owns*
  that task — never the peer tab's id.

* **H3 — per-tab usage totals do not double-count the peer.**  The
  ``result`` panel's ``total_tokens`` and ``cost`` for tab A must not
  include tab B's tokens (and vice versa).  Both tasks return the
  same per-call usage (10 prompt + 5 completion = 15 total) from the
  fake server, so any double-counting would surface as
  ``total_tokens > 15`` on either tab.

* **H4 — ``task_done`` carries the owning ``tabId``.**  The
  end-of-task lifecycle event must be tagged with the originating tab
  so that the frontend's "Running …" / "Done (…)" label flips on the
  right tab only.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as ps
import kiss.server.vscode_config as vc
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server.server import VSCodeServer
from kiss.server.web_server import WebPrinter

# Display events that belong to a single task's chat stream.  Any of
# these broadcast without a ``tabId`` while two tabs run a task would
# render into whichever tab is active on the frontend.
_CONTENT_EVENT_TYPES = frozenset({
    "result", "text_delta", "text_end", "thinking_start", "thinking_delta",
    "thinking_end", "tool_call", "tool_result", "system_output",
    "system_prompt", "prompt", "usage_info", "clear", "task_done",
    "task_error", "task_stopped", "task_interrupted",
    "followup_suggestion",
})


# ---------------------------------------------------------------------------
# Real fake OpenAI-compatible HTTP server
# ---------------------------------------------------------------------------

def _sse_chunks_for_tool_call(
    tool_name: str, arguments: str, model: str = "gpt-4o-mini",
) -> list[dict[str, Any]]:
    """Build the OpenAI streaming chunks that invoke *tool_name*.

    Mirrors what the OpenAI server would emit for a single-tool-call
    completion: a first chunk with the tool name and call id, a
    second chunk with the arguments payload, a third chunk with
    ``finish_reason="tool_calls"``, and a final chunk carrying the
    ``usage`` block (``stream_options.include_usage=True``).
    """
    cid = f"chatcmpl-{tool_name}"
    return [
        {
            "id": cid, "object": "chat.completion.chunk", "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "index": 0,
                        "id": f"call_{tool_name}",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": ""},
                    }],
                },
                "finish_reason": None,
            }],
        },
        {
            "id": cid, "object": "chat.completion.chunk", "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": arguments},
                    }],
                },
                "finish_reason": None,
            }],
        },
        {
            "id": cid, "object": "chat.completion.chunk", "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls",
            }],
        },
        {
            "id": cid, "object": "chat.completion.chunk", "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
    ]


def _write_sse_chunks(handler: BaseHTTPRequestHandler, chunks: list[dict[str, Any]]) -> None:
    """Write the chunks as a Server-Sent Events stream.

    Each chunk is emitted as ``data: <json>\\n\\n``, followed by a
    terminating ``data: [DONE]\\n\\n`` marker.  Sleeps briefly between
    chunks so the streaming parser actually consumes multiple chunks
    instead of seeing them as one read.
    """
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.end_headers()
    for chunk in chunks:
        payload = b"data: " + json.dumps(chunk).encode() + b"\n\n"
        # Chunked-transfer wrapper: <hex-len>\r\n<bytes>\r\n
        handler.wfile.write(f"{len(payload):x}\r\n".encode())
        handler.wfile.write(payload + b"\r\n")
        handler.wfile.flush()
        time.sleep(0.005)
    done = b"data: [DONE]\n\n"
    handler.wfile.write(f"{len(done):x}\r\n".encode())
    handler.wfile.write(done + b"\r\n")
    handler.wfile.write(b"0\r\n\r\n")  # zero-length chunk: end of body
    handler.wfile.flush()


class _FinishHandler(BaseHTTPRequestHandler):
    """SSE handler that streams a ``finish`` tool-call for every POST.

    Adds a short random sleep before responding so the two concurrent
    task threads' broadcasts interleave aggressively, surfacing any
    routing race that depends on lucky scheduling (per the testing
    rule of inserting ``time.sleep(<0.1s)`` near suspected racing
    statements).
    """

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        if cl:
            self.rfile.read(cl)
        time.sleep(random.uniform(0.0, 0.05))
        args = json.dumps({"success": "true", "summary": "done"})
        _write_sse_chunks(
            self, _sse_chunks_for_tool_call("finish", args),
        )

    def log_message(  # noqa: A002
        self, format: str, *args: object,
    ) -> None:
        """Suppress noisy access-log lines on stderr."""
        pass


def _start_fake_openai() -> tuple[ThreadingHTTPServer, str]:
    """Start a local fake OpenAI server and return ``(server, base_url)``."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FinishHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


def _start_fake_ask_then_finish() -> tuple[ThreadingHTTPServer, str]:
    """Start a fake OpenAI server that asks once, then finishes.

    The handler inspects each request body: if it already contains an
    answer marker (``ANSWER_A`` or ``ANSWER_B``), it streams the
    ``finish`` tool call; otherwise it streams an
    ``ask_user_question`` tool call whose question text embeds the
    originating tab tag (``A`` or ``B``).  This lets two concurrent
    agents each invoke ``ask_user_question`` exactly once and then
    conclude once their per-tab answer has been routed back.
    """

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            cl = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(cl) if cl else b""
            body_str = body_bytes.decode("utf-8", errors="replace")
            time.sleep(random.uniform(0.0, 0.04))
            if "ANSWER_A" in body_str or "ANSWER_B" in body_str:
                args = json.dumps({"success": "true", "summary": "done"})
                chunks = _sse_chunks_for_tool_call("finish", args)
            else:
                tag = "A" if "tab A prompt" in body_str else "B"
                args = json.dumps(
                    {"question": f"What is your answer for {tag}?"},
                )
                chunks = _sse_chunks_for_tool_call(
                    "ask_user_question", args,
                )
            _write_sse_chunks(self, chunks)

        def log_message(  # noqa: A002
            self, format: str, *args: object,
        ) -> None:
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


# ---------------------------------------------------------------------------
# Capturing printer
# ---------------------------------------------------------------------------

class _CapturingWebPrinter(WebPrinter):
    """Real :class:`WebPrinter` that records every wire payload.

    Overrides only :meth:`_send_to_ws_clients` so the full production
    ``broadcast`` routing (verbatim ``tabId`` events, thread-local
    ``taskId`` injection, global fallback, and per-subscriber fan-out)
    runs unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[dict[str, Any]] = []
        self._sent_lock = threading.Lock()

    def _send_to_ws_clients(self, data: str) -> None:
        """Record every JSON payload that would be sent to clients."""
        with self._sent_lock:
            self.sent.append(json.loads(data))


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _git_init(repo: Path) -> None:
    """Initialise *repo* as a clean git repository with one empty commit."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "--allow-empty",
         "-m", "init"],
        check=True,
    )


def _redirect_persistence(tmpdir: str) -> tuple:
    """Point the persistence module at a tmpdir; return saved-state tuple."""
    saved = (ps._DB_PATH, ps._db_conn, ps._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    ps._KISS_DIR = kiss_dir
    ps._DB_PATH = kiss_dir / "sorcar.db"
    ps._db_conn = None
    return saved


def _restore_persistence(saved: tuple) -> None:
    ps._DB_PATH, ps._db_conn, ps._KISS_DIR = saved


def _redirect_config(tmpdir: str) -> tuple:
    """Point ``vscode_config`` at a tmpdir and seed a minimal config."""
    saved = (vc.CONFIG_DIR, vc.CONFIG_PATH)
    cfg_dir = Path(tmpdir) / ".kiss-cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    vc.CONFIG_DIR = cfg_dir
    vc.CONFIG_PATH = cfg_dir / "config.json"
    return saved


def _restore_config(saved: tuple) -> None:
    vc.CONFIG_DIR, vc.CONFIG_PATH = saved


# ---------------------------------------------------------------------------
# Base fixture
# ---------------------------------------------------------------------------

class _TwoTabFixture(unittest.TestCase):
    """Common setup/teardown for the two-tab interference tests.

    Spins up:

    * a fake OpenAI HTTP server,
    * a redirected persistence DB (so the test does not write to the
      user's real ``~/.kiss/sorcar.db``),
    * a redirected vscode_config (so the fake endpoint and disabled
      auto-commit only apply to this test process),
    * two git-initialised work directories,
    * a :class:`VSCodeServer` with a capturing :class:`WebPrinter`.

    Sets ``OPENAI_API_KEY`` so ``get_available_models()`` reports
    ``gpt-4o-mini`` as available.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="kiss-2tab-")
        self.saved_persistence = _redirect_persistence(self.tmp)
        self.saved_config = _redirect_config(self.tmp)

        self.srv, self.url = _start_fake_openai()

        # Make ``get_available_models()`` accept ``gpt-4o-mini``.
        self._saved_openai_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"

        # Two distinct git repos so per-task pre-snapshots don't
        # serialise on each other's repo_lock more than necessary.
        self.work_a = Path(self.tmp) / "repoA"
        self.work_b = Path(self.tmp) / "repoB"
        self.work_a.mkdir()
        self.work_b.mkdir()
        _git_init(self.work_a)
        _git_init(self.work_b)

        # Seed the config so ``build_model_config`` returns our fake
        # endpoint and ``_run_task_inner`` reads a sensible budget.
        vc.save_config({
            "custom_endpoint": self.url,
            "custom_api_key": "test-key",
            "auto_commit_mode": False,
            "max_budget": 100,
            "use_web_browser": False,
        })

        self.printer = _CapturingWebPrinter()
        self.server = VSCodeServer(printer=self.printer)

    def tearDown(self) -> None:
        try:
            self.srv.shutdown()
        except Exception:
            pass
        if ps._db_conn is not None:
            try:
                ps._db_conn.close()
            except Exception:
                pass
            ps._db_conn = None
        _restore_persistence(self.saved_persistence)
        _restore_config(self.saved_config)
        # Reset the per-process tab registry so a later test starts
        # from a clean slate (the registry is a class-level dict).
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()
        if self._saved_openai_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._saved_openai_key
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers used by every test
    # ------------------------------------------------------------------

    def _run_two_tasks(
        self,
        *,
        prompt_a: str = "tab A prompt",
        prompt_b: str = "tab B prompt",
    ) -> tuple[threading.Thread, threading.Thread]:
        """Submit two ``run`` commands concurrently and return the threads.

        Uses ``_cmd_run`` (production entry point) so the full
        ``_run_task`` lifecycle runs unmodified — status broadcasts,
        agent run, post-task cleanup, ``task_done``, and printer
        cleanup all execute on the per-task worker thread.
        """
        cmd_a = {
            "tabId": "tabA",
            "prompt": prompt_a,
            "model": "gpt-4o-mini",
            "workDir": str(self.work_a),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
        }
        cmd_b = {
            "tabId": "tabB",
            "prompt": prompt_b,
            "model": "gpt-4o-mini",
            "workDir": str(self.work_b),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
        }

        # Stagger by a few milliseconds so both task threads are
        # concurrently inside ``ChatSorcarAgent.run`` (printer
        # thread-local task_id set, subscriber registered, recording
        # buffer active).  A random jitter in the fake server's
        # response time surfaces interleaving races.
        self.server._cmd_run(cmd_a)
        time.sleep(random.uniform(0.0, 0.02))
        self.server._cmd_run(cmd_b)

        ta = _RunningAgentState.running_agent_states["tabA"].task_thread
        tb = _RunningAgentState.running_agent_states["tabB"].task_thread
        assert ta is not None
        assert tb is not None
        # Wait for both task threads to finish.
        ta.join(timeout=30)
        tb.join(timeout=30)
        assert not ta.is_alive(), "tab A task hung"
        assert not tb.is_alive(), "tab B task hung"
        return ta, tb

    def _events_for_tab(self, tab_id: str) -> list[dict[str, Any]]:
        """Return every captured payload tagged with *tab_id*."""
        with self.printer._sent_lock:
            return [
                dict(e) for e in self.printer.sent if e.get("tabId") == tab_id
            ]

    def _events_typed(
        self, types: Iterable[str], tab_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter captured payloads by ``type`` (and optional ``tabId``)."""
        types = set(types)
        with self.printer._sent_lock:
            snap = [dict(e) for e in self.printer.sent]
        return [
            e for e in snap
            if e.get("type") in types
            and (tab_id is None or e.get("tabId") == tab_id)
        ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTwoTabRealLLMNoCrossTabLeak(_TwoTabFixture):
    """H1 — no content event may leak globally while two tabs run."""

    def test_no_untagged_content_event_during_two_concurrent_real_runs(
        self,
    ) -> None:
        """While two real LLM tasks run in parallel, every display event
        must carry a ``tabId``.

        A display event without ``tabId`` would render into whichever
        tab is currently active on the frontend — polluting the chat
        surface of the other concurrently-running task.
        """
        self._run_two_tasks()

        with self.printer._sent_lock:
            leaked = [
                dict(e) for e in self.printer.sent
                if e.get("type") in _CONTENT_EVENT_TYPES
                and "tabId" not in e
            ]
        self.assertFalse(
            leaked,
            "Two real LLM tasks running in parallel leaked content "
            "event(s) without a tabId.  Such events render into "
            "whichever tab is active on the frontend, polluting the "
            f"other tab's chat.  Leaked payloads: {leaked}",
        )


class TestTwoTabRealLLMPerTabFanout(_TwoTabFixture):
    """H2 — each task's stamped fan-out copies go only to its owning tab."""

    def test_task_id_only_ever_paired_with_owning_tab_id(self) -> None:
        """For every captured payload, the ``(taskId, tabId)`` pair must be
        consistent: a given ``taskId`` must always appear next to the
        same ``tabId`` (the tab that owns that task).

        Cross-tab fan-out would manifest as a payload whose ``taskId``
        belongs to tab A's run but whose ``tabId`` is tab B (or vice
        versa).  This test asserts the subscriber-based fan-out path
        in :meth:`WebPrinter.broadcast` does not mix the two streams.
        """
        self._run_two_tasks()

        with self.printer._sent_lock:
            pairs: dict[str, set[str]] = {}
            for e in self.printer.sent:
                tid = e.get("taskId")
                tab = e.get("tabId")
                if not tid or not tab:
                    continue
                pairs.setdefault(str(tid), set()).add(str(tab))

        # At minimum we expect each tab to have run one task with at
        # least one stamped fan-out copy.
        self.assertGreaterEqual(
            len(pairs), 2,
            "Expected at least two distinct task ids in the captured "
            "stream (one per concurrent run).  Got: "
            f"{pairs!r}.  Sent payload count: {len(self.printer.sent)}",
        )
        for tid, tabs in pairs.items():
            self.assertEqual(
                len(tabs), 1,
                "Cross-tab fan-out: task id "
                f"{tid!r} was stamped onto multiple tab ids {tabs!r}. "
                "WebPrinter.broadcast's subscriber fan-out should only "
                "deliver each task's events to that task's owning tab.",
            )


class TestTwoTabRealLLMTaskDoneRoutedToOwningTab(_TwoTabFixture):
    """H4 — ``task_done`` lifecycle events carry the owning ``tabId``."""

    def test_each_tab_receives_its_own_task_done(self) -> None:
        """Each tab must receive exactly one ``task_done`` event tagged
        with its own ``tabId``, and never with the peer's.

        The frontend uses ``task_done`` to flip the "Running …" header
        of the chat webview to "Done (Xm Ys)".  Mis-routing would
        either leave a running task labelled "Running …" forever or
        flip the wrong tab's label.
        """
        self._run_two_tasks()

        done_a = self._events_typed({"task_done"}, tab_id="tabA")
        done_b = self._events_typed({"task_done"}, tab_id="tabB")

        self.assertEqual(
            len(done_a), 1,
            f"Expected exactly one task_done event for tabA; got {done_a!r}",
        )
        self.assertEqual(
            len(done_b), 1,
            f"Expected exactly one task_done event for tabB; got {done_b!r}",
        )

        # No task_done may carry the peer's tabId.
        with self.printer._sent_lock:
            all_done = [
                e for e in self.printer.sent if e.get("type") == "task_done"
            ]
        for e in all_done:
            self.assertIn(
                e.get("tabId"), ("tabA", "tabB"),
                "task_done event carries an unexpected tabId: "
                f"{e!r}",
            )


class TestTwoTabRealLLMUsageNotDoubleCounted(_TwoTabFixture):
    """H3 — per-tab ``result`` totals don't include the peer's tokens."""

    def test_per_tab_result_totals_are_scoped(self) -> None:
        """The ``result`` payload broadcast at the end of each task
        must carry only that task's tokens / cost / steps — not the
        sum of both concurrent tasks'.

        The fake server reports ``total_tokens=15`` per call and the
        agent does exactly one LLM call (``finish`` on the first
        completion).  If the printer mixes up usage offsets between
        the two concurrently-running tasks, the ``result`` panel
        would report ``total_tokens >= 30`` on at least one tab.
        """
        self._run_two_tasks()

        results_a = self._events_typed({"result"}, tab_id="tabA")
        results_b = self._events_typed({"result"}, tab_id="tabB")

        # ``result`` is emitted by :meth:`ChatSorcarAgent.run` via
        # :meth:`JsonPrinter._broadcast_result` — exactly once per
        # task.  An additional ``result`` would indicate the no-model
        # fallback fired (which would also fail the H1 test).
        self.assertGreaterEqual(
            len(results_a), 1,
            f"Expected a result event on tabA; got {results_a!r}",
        )
        self.assertGreaterEqual(
            len(results_b), 1,
            f"Expected a result event on tabB; got {results_b!r}",
        )

        for tab_label, evs in (("tabA", results_a), ("tabB", results_b)):
            for r in evs:
                tokens = int(r.get("total_tokens", 0) or 0)
                self.assertLess(
                    tokens, 30,
                    f"{tab_label} result reports total_tokens={tokens}, "
                    "which is at least the sum of both concurrent "
                    "tasks' usage (15 each).  The printer's per-task "
                    "token offset was clobbered by the peer's run.",
                )


class TestTwoTabRealLLMAskUserAnswerRouting(unittest.TestCase):
    """H5 — concurrent ``ask_user_question`` answers don't cross tabs.

    Drives two real ``ChatSorcarAgent`` runs against a stateful fake
    OpenAI server that first asks a question and then finishes once
    its tab's answer has been delivered.  Submits one ``userAnswer``
    per tab and asserts each agent's *next* prompt to the fake server
    contains only its own answer — never the peer's.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="kiss-2tab-ask-")
        self.saved_persistence = _redirect_persistence(self.tmp)
        self.saved_config = _redirect_config(self.tmp)
        self.srv, self.url = _start_fake_ask_then_finish()
        self._saved_openai_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        self.work_a = Path(self.tmp) / "repoA"
        self.work_b = Path(self.tmp) / "repoB"
        self.work_a.mkdir()
        self.work_b.mkdir()
        _git_init(self.work_a)
        _git_init(self.work_b)
        vc.save_config({
            "custom_endpoint": self.url,
            "custom_api_key": "test-key",
            "auto_commit_mode": False,
            "max_budget": 100,
            "use_web_browser": False,
        })
        self.printer = _CapturingWebPrinter()
        self.server = VSCodeServer(printer=self.printer)

    def tearDown(self) -> None:
        try:
            self.srv.shutdown()
        except Exception:
            pass
        if ps._db_conn is not None:
            try:
                ps._db_conn.close()
            except Exception:
                pass
            ps._db_conn = None
        _restore_persistence(self.saved_persistence)
        _restore_config(self.saved_config)
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()
        if self._saved_openai_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._saved_openai_key
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _wait_for_askuser(
        self, tab_id: str, timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Block until an ``askUser`` event for *tab_id* is captured."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.printer._sent_lock:
                for e in self.printer.sent:
                    if (
                        e.get("type") == "askUser"
                        and e.get("tabId") == tab_id
                    ):
                        return dict(e)
            time.sleep(0.01)
        raise AssertionError(
            f"askUser for {tab_id!r} did not arrive within {timeout}s",
        )

    def test_each_tab_answer_routes_to_its_own_agent(self) -> None:
        """Submit ``ANSWER_A`` on tabA and ``ANSWER_B`` on tabB while
        both agents block in ``ask_user_question``.  Each agent's
        resumed prompt to the fake server must contain only its own
        answer — never the peer's.

        The ``askUser`` event itself is broadcast via the agent's
        thread-local ``taskId`` (set by :meth:`ChatSorcarAgent.run`)
        and fanned out to the owning tab's subscriber.  If the
        printer's per-task routing got crossed, an ``ANSWER_B`` could
        wake up tabA's agent (or vice versa) — corrupting the next
        model call's chat history.
        """
        cmd_a = {
            "tabId": "tabA",
            "prompt": "tab A prompt",
            "model": "gpt-4o-mini",
            "workDir": str(self.work_a),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
        }
        cmd_b = {
            "tabId": "tabB",
            "prompt": "tab B prompt",
            "model": "gpt-4o-mini",
            "workDir": str(self.work_b),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
        }
        self.server._cmd_run(cmd_a)
        time.sleep(random.uniform(0.0, 0.02))
        self.server._cmd_run(cmd_b)

        # Wait for both agents to surface their question via askUser.
        ask_a = self._wait_for_askuser("tabA")
        ask_b = self._wait_for_askuser("tabB")
        # Tab-routing sanity: the question text must include the right
        # tab tag (so a swap would have been visible here too).
        self.assertIn("A", ask_a.get("question", ""))
        self.assertIn("B", ask_b.get("question", ""))

        # Submit each tab's answer.  Stagger so the routing must
        # decide which queue each answer goes to without relying on
        # timing.
        self.server._cmd_user_answer({"tabId": "tabA", "answer": "ANSWER_A"})
        time.sleep(random.uniform(0.0, 0.02))
        self.server._cmd_user_answer({"tabId": "tabB", "answer": "ANSWER_B"})

        # Wait for both task threads to finish (second LLM call must
        # be a ``finish`` triggered by the answer arriving).
        ta = _RunningAgentState.running_agent_states["tabA"].task_thread
        tb = _RunningAgentState.running_agent_states["tabB"].task_thread
        assert ta is not None and tb is not None
        ta.join(timeout=30)
        tb.join(timeout=30)
        self.assertFalse(ta.is_alive(), "tabA task hung after answering")
        self.assertFalse(tb.is_alive(), "tabB task hung after answering")

        # Each tab must have completed with a tab-scoped task_done.
        done_a = [
            e for e in self.printer.sent
            if e.get("type") == "task_done" and e.get("tabId") == "tabA"
        ]
        done_b = [
            e for e in self.printer.sent
            if e.get("type") == "task_done" and e.get("tabId") == "tabB"
        ]
        self.assertEqual(
            len(done_a), 1,
            f"tabA task_done expected exactly once; got {done_a!r}",
        )
        self.assertEqual(
            len(done_b), 1,
            f"tabB task_done expected exactly once; got {done_b!r}",
        )

        # The tool_result events captured per tab should show each
        # agent received its own answer, not the peer's.  ``tool_result``
        # is broadcast via the agent's thread-local task_id, then
        # fanned out to the owning tab's subscriber.
        tr_a = [
            e for e in self.printer.sent
            if e.get("type") == "tool_result" and e.get("tabId") == "tabA"
        ]
        tr_b = [
            e for e in self.printer.sent
            if e.get("type") == "tool_result" and e.get("tabId") == "tabB"
        ]

        def _has_answer(events: list[dict[str, Any]], answer: str) -> bool:
            for e in events:
                txt = json.dumps(e)
                if answer in txt:
                    return True
            return False

        self.assertTrue(
            _has_answer(tr_a, "ANSWER_A"),
            "tabA's tool_result stream is missing its own ANSWER_A: "
            f"{tr_a!r}",
        )
        self.assertFalse(
            _has_answer(tr_a, "ANSWER_B"),
            "Cross-tab user-answer leak: tabA's tool_result stream "
            f"contains tabB's ANSWER_B.  Events: {tr_a!r}",
        )
        self.assertTrue(
            _has_answer(tr_b, "ANSWER_B"),
            "tabB's tool_result stream is missing its own ANSWER_B: "
            f"{tr_b!r}",
        )
        self.assertFalse(
            _has_answer(tr_b, "ANSWER_A"),
            "Cross-tab user-answer leak: tabB's tool_result stream "
            f"contains tabA's ANSWER_A.  Events: {tr_b!r}",
        )


if __name__ == "__main__":  # pragma: no cover — module run as script
    unittest.main()
