# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 0903: SorcarAgent.run's exception wrapping.

The worktree agent's ``run`` once carried two byte-identical
``try/except KISSError/except Exception`` blocks around its
``super().run`` delegation — one for the direct-execution fallback and
one for the worktree path.  The worktree refactor left a single path
(the agent's ``create_worktree`` tool makes the worktree on demand, so
``run`` never creates one up front), now ``SorcarAgent.run`` itself;
this test PINS the wrapping contract of that path against real behavior:

* a non-``KISSError`` exception from the underlying run is wrapped into
  a ``success: false`` YAML result — in a plain directory and in a git
  repository alike;
* the summary of that result is HTML like every other result's, the
  error (plain text) escaped — as returned, as the row stores it and as
  a daemon broadcasts it to the task's tab;
* a ``KISSError`` propagates unwrapped;
* a repository ``work_dir`` does not make ``run`` create a worktree or
  broadcast ``worktree_created``.

End-to-end: real agents, a real scratch git repo, a real server
printer.  The underlying run is made to fail by a ``tools_hook`` that
raises — a real caller-owned collaborator (the channel a SEA's
``tools()`` uses), called inside ``SorcarAgent.run`` once the toolset
is built — not a mock of the code under test.  (A failure in
``ChatSorcarAgent.run``'s own setup, before the inner run, propagates:
the daemon's task runner reports it as ``Task failed: ...``.)  The
daemon scenario is a real ``RemoteAccessServer`` given a work dir no
machine can create, with the stand-in model server as its endpoint.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.core.kiss_error import KISSError
from kiss.core.memoryfield.evaluate import is_failed_result
from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    LocalDaemonTestCase,
    StandInModelServer,
    finish_response,
)

# An error message is plain text; this one reads as markup at every turn.
_ERROR = "bad <b>tag</b> & '__init__' in \"<unicode string>\"\n  line 2"
_ERROR_SUMMARY = (
    "<p>Task failed with error: bad &lt;b&gt;tag&lt;/b&gt; &amp; '__init__' in "
    '"&lt;unicode string&gt;"<br/>  line 2</p>'
)


class _RaisingToolsHook:
    """A SEA-style ``tools`` hook that raises instead of returning the toolset."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __call__(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        raise self._exc


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-audit0903-run-wrapping-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def _failure_summary(result: str) -> str:
    """Parse *result* as YAML and return its failure summary."""
    payload = yaml.safe_load(result)
    assert payload["success"] is False
    summary: str = payload["summary"]
    return summary


def test_direct_path_wraps_exception_into_failure_yaml(
    env: IsolatedKissHome,
) -> None:
    """Non-repo work_dir → direct execution; a crash becomes YAML with an HTML summary."""
    agent = ChatSorcarAgent("audit0903-wrap-direct")
    with tempfile.TemporaryDirectory() as plain_dir:
        result = agent.run(
            prompt_template="crash please",
            work_dir=plain_dir,
            printer=CapturePrinter(),
            tools_hook=_RaisingToolsHook(RuntimeError(_ERROR)),
        )
    assert agent._wt is None
    # The shape of every result, as ``finish`` builds it.
    assert yaml.safe_load(result) == {
        "success": False, "is_continue": False, "summary": _ERROR_SUMMARY,
    }
    [row] = persistence._load_history()
    assert (row["result"], row["success"]) == (_ERROR_SUMMARY, 0)


def test_direct_path_propagates_kiss_error(env: IsolatedKissHome) -> None:
    """A KISSError from the underlying run is never wrapped."""
    agent = ChatSorcarAgent("audit0903-wrap-kisserror")
    with tempfile.TemporaryDirectory() as plain_dir:
        with pytest.raises(KISSError, match="kiss-boom"):
            agent.run(
                prompt_template="crash please",
                work_dir=plain_dir,
                printer=CapturePrinter(),
                tools_hook=_RaisingToolsHook(KISSError("kiss-boom")),
            )


def test_repo_work_dir_wraps_exception_without_a_worktree(
    env: IsolatedKissHome,
) -> None:
    """Repo work_dir → same wrapping; no worktree is created up front."""
    agent = ChatSorcarAgent("audit0903-wrap-worktree")
    printer = CapturePrinter()
    result = agent.run(
        prompt_template="crash please",
        work_dir=str(env.repo),
        printer=printer,
        tools_hook=_RaisingToolsHook(RuntimeError("boom-worktree")),
    )
    assert "Task failed with error: boom-worktree" in _failure_summary(result)
    assert printer.events_of_type("worktree_created") == []
    assert agent._wt is None
    assert not (env.repo / ".kiss-worktrees").exists()


def _finish_at_once(request: dict[str, Any]) -> dict[str, Any]:
    """The stand-in model's only answer: ``finish`` at the first step."""
    return finish_response("<p>done</p>")


class TestDaemonReportsTheFailureAsHtml(LocalDaemonTestCase):
    """A real daemon runs a task whose work dir cannot be created."""

    home_prefix = "kiss-audit0903-failure-html-"

    def setUp(self) -> None:
        # The daemon's model endpoint, started first so that it stops
        # only after the task thread is joined.  The run fails before
        # its first model call; were the agent loop to start, it would
        # finish successfully here and fail the test.
        model = StandInModelServer(_finish_at_once)
        self.addCleanup(model.stop)
        super().setUp()
        self.home.write_config(custom_endpoint=model.url, custom_api_key="k", max_budget=100)
        self.start_daemon()

    async def _run_task(self, work_dir: str) -> list[dict[str, Any]]:
        """Submit a task in *work_dir*; return the frames sent until its tab went idle."""
        reader, writer = await open_local_connection(self.server)
        try:
            submit = {
                "type": "submit", "prompt": "say hi", "tabId": "tab-1",
                "model": STANDIN_MODEL, "workDir": work_dir,
            }
            writer.write(json.dumps(submit).encode() + b"\n")
            await writer.drain()
            frames: list[dict[str, Any]] = []
            while True:
                frame = json.loads(await asyncio.wait_for(reader.readline(), timeout=60))
                frames.append(frame)
                if frame.get("type") == "status" and frame.get("tabId") == "tab-1":
                    if not frame["running"]:
                        return frames
        finally:
            writer.close()

    def test_the_failure_is_stored_and_broadcast_as_html(self) -> None:
        """The row, the tab's ``result`` event and the transcript hold one HTML summary.

        The directory sits under a plain file whose legal name contains
        an HTML metacharacter: the summary is one paragraph with the name
        escaped, so a reader of it as HTML (the result panel, a replay,
        a commit message) shows the name as it is.
        """
        blocker = Path(self.tmpdir) / "not-a-dir & unicode string"
        blocker.write_text("")
        work_dir = str(blocker / "child")
        frames = self.run_on_loop(self._run_task(work_dir), timeout=120)
        self.join_tasks()

        [row] = persistence._load_history()
        summary = str(row["result"])
        self.assertRegex(summary, r"\A<p>Task failed with error: [^<>]*</p>\Z")
        self.assertIn("not-a-dir &amp; unicode string", summary)
        self.assertEqual(row["success"], 0)
        # Classified by its text alone, as ``rsi7d`` does, it is still a failure.
        self.assertTrue(is_failed_result(summary))

        [broadcast] = [frame for frame in frames if frame.get("type") == "result"]
        self.assertEqual((broadcast.get("summary"), broadcast["success"]), (summary, False))
        stored = persistence._load_chat_events_by_task_id(str(row["id"]))
        assert stored is not None
        events = cast(list[dict[str, Any]], stored["events"])
        [replayed] = [event for event in events if event.get("type") == "result"]
        self.assertEqual(replayed.get("summary"), summary)
