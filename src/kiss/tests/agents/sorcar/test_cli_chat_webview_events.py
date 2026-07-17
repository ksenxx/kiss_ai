# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: ``sorcar`` CLI must persist agent events to the chat DB.

The chat webview (VS Code extension or remote ``RemoteAccessServer``)
loads a session by reading its ``events`` table.  When ``sorcar`` is
launched from the command line, the default :class:`ConsolePrinter`
renders rich panels to the terminal but does NOT call
``_queue_chat_event`` for the live event stream.  As a result, after a
CLI task finishes, the only events ``_persist_replay_events_if_missing``
leaves in the ``events`` table are the synthesised ``prompt`` and
``result`` — every intermediate ``tool_call``, ``tool_result``,
``text_delta``, etc. is silently lost.  Re-opening the session in the
chat webview therefore renders a blank conversation with just the
prompt and the final result.

This test reproduces the bug by driving a ``ChatSorcarAgent`` through
the same code path the CLI uses (``_build_run_kwargs`` from
:mod:`kiss.agents.sorcar.cli_helpers`, then ``agent.run(**kwargs)``)
against a deterministic fake LLM that returns a single ``finish``
tool call.  After the task finishes, the test loads the persisted
event stream from the DB and asserts that it contains the live
``tool_call`` event for the ``finish`` invocation — the canonical
"agent did something" marker the chat webview needs to render the
tool-call panel.  Before the fix the persisted stream only contains
the synthesised ``prompt`` + ``result`` and the test fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.cli_helpers import _build_run_kwargs
from kiss.ui.cli.cli_printer import RecordingConsolePrinter


def _finish_response(summary: str = "done") -> dict[str, Any]:
    """Return a single-shot ``finish`` tool-call completion."""
    return {
        "id": "chatcmpl-fin",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_fin",
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": json.dumps(
                            {"success": "true", "summary": summary},
                        ),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        },
    }


class _Handler(BaseHTTPRequestHandler):
    """Fake OpenAI-compatible chat-completions endpoint.

    Always returns a one-shot ``finish`` tool call (streaming when the
    client sets ``stream: true``).  Drives the agent loop deterministically
    to exactly one step so the test focuses on whether per-event
    persistence happened, not on simulating a longer conversation.
    """

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(cl) if cl else b""
        try:
            req = json.loads(raw)
        except Exception:
            req = {}
        is_stream = req.get("stream", False)
        resp = _finish_response("cli-test-summary")

        if not is_stream:
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        choice = resp["choices"][0]
        msg = choice["message"]
        c1 = {
            "id": resp["id"],
            "object": "chat.completion.chunk",
            "model": resp["model"],
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": None},
                "finish_reason": None,
            }],
        }
        delta2: dict[str, Any] = {}
        if msg.get("tool_calls"):
            delta2["tool_calls"] = msg["tool_calls"]
        if msg.get("content"):
            delta2["content"] = msg["content"]
        c2 = {
            "id": resp["id"],
            "object": "chat.completion.chunk",
            "model": resp["model"],
            "choices": [{
                "index": 0,
                "delta": delta2,
                "finish_reason": choice["finish_reason"],
            }],
        }
        c3 = {
            "id": resp["id"],
            "object": "chat.completion.chunk",
            "model": resp["model"],
            "choices": [],
            "usage": resp["usage"],
        }
        body = "".join(
            f"data: {json.dumps(c)}\n\n" for c in (c1, c2, c3)
        ).encode() + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _start_server() -> tuple[ThreadingHTTPServer, str]:
    """Start the fake LLM on an ephemeral port; return (server, base_url)."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


def _redirect_persistence(tmpdir: str) -> tuple[Path, Any, Path]:
    """Point :mod:`kiss.agents.sorcar.persistence` at a temp directory.

    Returns the saved tuple so the caller can restore it after the test.
    """
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, Any, Path]) -> None:
    """Restore the persistence-module globals saved by ``_redirect_persistence``."""
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _cli_args(task: str, work_dir: str, base_url: str) -> argparse.Namespace:
    """Build an ``argparse.Namespace`` mirroring a real ``sorcar`` CLI run.

    All fields that :func:`_build_run_kwargs` reads must be present.
    """
    return argparse.Namespace(
        model_name="gpt-4o-mini",
        endpoint=base_url,
        header=["x-test:1"],
        max_budget=1.0,
        work_dir=work_dir,
        verbose=True,
        no_web=True,
        parallel=False,
        task=task,
        file=None,
        new=False,
        chat_id=None,
        list_chat_id=False,
        cleanup=False,
        use_chat=True,
        use_worktree=False,
    )


class TestCliPersistsEventsForChatWebview:
    """A CLI ``sorcar`` run must persist the live event stream.

    The chat webview reads ``events`` rows when the user opens a chat
    session.  If the CLI never persists ``tool_call`` / ``tool_result``
    / ``text_delta`` events, the webview shows a blank conversation
    bracketed by the synthesised prompt and result.  This test asserts
    that the persisted event stream includes the live ``tool_call``
    event for the model's ``finish`` invocation.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        self.srv, self.url = _start_server()

    def teardown_method(self) -> None:
        self.srv.shutdown()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_cli_task(self) -> str:
        """Drive ``ChatSorcarAgent`` through the CLI's run-kwargs path.

        Returns the persisted ``task_history`` row id so the caller can
        load its event stream from the DB.
        """
        args = _cli_args(
            task="Call finish with summary 'cli-test-summary'.",
            work_dir=self.tmpdir,
            base_url=self.url,
        )
        # The recording printer is injected by the UI-layer entry point
        # (``kiss.ui.cli.sorcar_cli.main``) exactly like this — the
        # sorcar-layer helper must not import the UI layer itself.
        run_kwargs = _build_run_kwargs(
            args, printer_factory=RecordingConsolePrinter,
        )
        # ``model_config`` is built from ``--endpoint`` / ``--header``
        # by ``_build_run_kwargs``; the fake server expects a non-empty
        # api_key so add it without touching the CLI helper itself.
        run_kwargs["model_config"]["api_key"] = "test-key"
        # CLI runs do not pre-subscribe a tab (the chat webview, if
        # any, subscribes on its own when the user opens the session).
        agent = ChatSorcarAgent("CLI Sorcar Agent")
        agent.run(**run_kwargs)
        assert agent._last_task_id is not None
        # Ensure asynchronously queued events are flushed to disk
        # before the test reads them.
        th._flush_chat_events()
        return str(agent._last_task_id)

    def test_finish_tool_call_event_is_persisted(self) -> None:
        """The persisted event stream must include the ``tool_call`` event
        for the model's ``finish(...)`` invocation.

        Before the fix the CLI installed a plain ``ConsolePrinter`` which
        never queued events, so only the synthesised ``prompt`` and
        ``result`` ended up in the DB — the chat webview rendered a
        blank conversation with no intermediate panels.  After the fix
        the CLI installs a recording-and-console printer, so the live
        event stream (including ``tool_call``) is persisted and the
        chat webview can faithfully replay the run.
        """
        task_id = self._run_cli_task()

        session = th._load_chat_events_by_task_id(task_id)
        assert session is not None, "Task row not found in DB"
        events: list[dict[str, Any]] = session["events"]  # type: ignore[assignment]
        event_types = [e.get("type") for e in events]

        # The user's prompt must be persisted (this works pre-fix too,
        # but only because ``_persist_replay_events_if_missing``
        # synthesises it after the run ends).
        assert "prompt" in event_types, (
            f"prompt event missing from persisted stream: {event_types}"
        )
        # The final result event must be present — likewise.
        assert "result" in event_types, (
            f"result event missing from persisted stream: {event_types}"
        )
        # The intermediate ``tool_call`` event for the model's
        # ``finish(...)`` invocation must also be persisted.  This is
        # the bug under test: a plain ConsolePrinter never queues this
        # event, and the fallback synthesiser only emits prompt +
        # result, so before the fix the assertion below fails.
        assert "tool_call" in event_types, (
            "tool_call event for finish() was NOT persisted — "
            f"the chat webview cannot render the live tool call. "
            f"persisted event types: {event_types}"
        )
        # The persisted tool_call must name the finish tool so the
        # chat webview renders it identically to a live VS Code run.
        finish_calls = [
            e for e in events
            if e.get("type") == "tool_call" and "finish" in str(e).lower()
        ]
        assert finish_calls, (
            "persisted tool_call exists but does not reference finish; "
            f"events: {events}"
        )
