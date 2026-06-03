# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test: the task runner persists the per-task ``result`` row from the
YAML returned by ``ChatSorcarAgent.run`` even though the printer
recording has already been torn down by the time control returns to
``_TaskRunnerMixin._run_task_inner``.

Spec
----
1. The user runs task T1 in a fresh chat tab.  The underlying agent
   (mocked here to avoid LLM calls) returns the standard
   ``finish()`` YAML payload — ``success: true`` plus a ``summary``
   string — but does NOT broadcast a ``type=result`` printer event
   itself.  In production that event IS broadcast (by
   :meth:`KISSAgent.perform_task` / :meth:`RelentlessAgent`) but it
   reaches the printer BEFORE ``ChatSorcarAgent.run``'s ``finally``
   block calls ``stop_recording`` — by the time control returns to
   the task runner, the recording has already been popped and the
   thread-local ``task_id`` cleared, so ``peek_recording`` returns
   an empty list.
2. The task runner therefore cannot rely on
   ``_extract_result_summary`` for the persisted result.  It must
   parse the YAML return value of ``tab.agent.run`` and persist the
   ``summary`` field as ``task_history.result``.
3. When the user runs a second task T2 in the same chat,
   :meth:`ChatSorcarAgent.build_chat_prompt` loads T1's result from
   the DB and surfaces it under "### Result 1".  Without the fix,
   the row contains the sentinel string ``"No summary available"``;
   with the fix it contains the real summary.

The test reproduces the production ordering exactly: the patched
``SorcarAgent.run`` returns the YAML *after* the recording has been
stopped by ``ChatSorcarAgent.run``'s ``finally``, so the only path to
recover the summary is to parse the YAML in the task runner.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.server import VSCodeServer


def _redirect_db(tmpdir: str) -> tuple:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _init_git_repo(tmpdir: str) -> None:
    subprocess.run(["git", "init", tmpdir], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmpdir,
                   capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir,
                   capture_output=True)
    Path(tmpdir, ".gitkeep").touch()
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir,
                   capture_output=True)


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]], threading.Lock]:
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    printer = server.printer

    def capture(event: dict[str, Any]) -> None:
        ev = printer._inject_task_id(event)
        with printer._lock:
            printer._record_event(ev)
        printer._persist_event(ev)
        with lock:
            events.append(ev)

    printer.broadcast = capture  # type: ignore[assignment]
    return server, events, lock


class _PromptCapture:
    """Captures ``prompt_template`` and returns a canned YAML result
    WITHOUT broadcasting a ``type=result`` printer event.

    This mirrors the real production ordering: the YAML reaches the
    task runner via the return value, but the matching ``result`` event
    is no longer accessible to ``peek_recording`` by the time the task
    runner reads back — because ``ChatSorcarAgent.run``'s ``finally``
    has already popped the recording buffer.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.summaries = [
            "remembered banana as the magic word",
            "second task done",
        ]

    def fake_run(self, agent_self: object, **kwargs: object) -> str:
        import yaml as _yaml
        self.prompts.append(str(kwargs.get("prompt_template", "")))
        summary = self.summaries.pop(0) if self.summaries else "done"
        # ``yaml.dump`` produces a string that ``yaml.safe_load``
        # round-trips, so the task runner's YAML parsing of the
        # return value recovers the exact summary string regardless
        # of punctuation it contains.
        return str(_yaml.dump({"success": True, "summary": summary}))


def _patch_parent_run(capture: _PromptCapture) -> Any:
    parent = cast(Any, SorcarAgent.__mro__[1])
    original = parent.run

    def _run_proxy(self_agent: object, **kwargs: object) -> str:
        return capture.fake_run(self_agent, **kwargs)

    parent.run = _run_proxy
    return original


def _unpatch_parent_run(original: Any) -> None:
    cast(Any, SorcarAgent.__mro__[1]).run = original


def _run_and_wait(server: VSCodeServer, tab_id: str, prompt: str,
                  work_dir: str) -> None:
    # ``autoCommit=True`` so the post-task flow short-circuits the
    # interactive merge review (which would otherwise set
    # ``tab.is_merging`` and block the second consecutive task in the
    # same tab).
    server._handle_command({
        "type": "run", "prompt": prompt,
        "model": "claude-opus-4-6", "workDir": work_dir,
        "tabId": tab_id, "autoCommit": True,
    })
    t = server._get_tab(tab_id).task_thread
    assert t is not None
    t.join(timeout=10)
    assert not t.is_alive()


class TestResultSummaryFromYaml(unittest.TestCase):
    """Without the fix, the second task's ``build_chat_prompt``
    preamble surfaces the sentinel string ``"No summary available"``
    as Result 1 — confirming that ``_extract_result_summary`` could
    not recover the summary after recording teardown.  The fix routes
    the persisted result through the YAML return value instead, so
    the real per-task summary appears.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)
        self.server, self.events, self.lock = _make_server()
        self.capture = _PromptCapture()
        self.original_run = _patch_parent_run(self.capture)

    def tearDown(self) -> None:
        _unpatch_parent_run(self.original_run)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_second_task_prompt_includes_real_first_summary(self) -> None:
        tab_id = "tab-chat"
        _run_and_wait(self.server, tab_id,
                      "remember the magic word: banana",
                      self.tmpdir)

        # The persisted ``result`` for the first task must be the YAML
        # summary string, NOT the "No summary available" sentinel.
        rows = th._load_history(limit=10)
        assert rows, "expected first task to be persisted"
        first_result = rows[0]["result"]
        assert first_result != "No summary available", (
            "task runner persisted the sentinel instead of the YAML "
            "summary — _extract_result_summary cannot see the result "
            "event after stop_recording pops the buffer"
        )
        assert "banana" in str(first_result), (
            f"expected the real summary ('...banana...') in "
            f"task_history.result, got {first_result!r}"
        )

        # Run a second task in the same tab/chat.
        _run_and_wait(self.server, tab_id, "what was the magic word?",
                      self.tmpdir)
        assert len(self.capture.prompts) >= 2

        second_prompt = self.capture.prompts[1]
        assert "Previous tasks and results" in second_prompt, (
            "second prompt should include the chat-history preamble"
        )
        assert "No summary available" not in second_prompt, (
            "second prompt must not surface the sentinel result row "
            "for the first task; build_chat_prompt is leaking the "
            "fallback string into the LLM context.  Prompt was:\n"
            + second_prompt
        )
        assert "banana" in second_prompt, (
            "second prompt should reference the real first-task "
            "summary (which contained 'banana') under '### Result 1'"
        )


if __name__ == "__main__":
    unittest.main()
