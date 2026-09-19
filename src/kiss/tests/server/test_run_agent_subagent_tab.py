# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a ``run_agent`` dispatch behaves like a ``run_parallel`` sub-task.

A task dispatched through ``daemon_client.run(parent_task_id=…,
parent_tab_id=…)`` — the wire shape the ``run_agent`` tool uses on
behalf of a calling task — must inherit the ``run_parallel`` sub-agent
tab contract instead of opening a top-level tab of its own:

1. no shared-registry tab is created for the dispatch's ``api-…`` id;
2. the run self-broadcasts ``new_tab`` carrying the child's persisted
   task id and the calling task's ``parent_tab_id``, so every client
   viewing the parent opens a nested client-local sub-agent tab;
3. the child's history row nests under the parent task (persisted
   ``subagent`` extra), so parent replays restore the sub-agent tab;
4. a ``subagentDone`` is broadcast when the run ends, to the fan-out
   viewers subscribed via ``resumeSession`` and to the dispatch tab id.

Like the sibling suites, a real :class:`RemoteAccessServer` is served
on a temporary Unix-domain socket and only the LLM boundary
(``RelentlessAgent.run``) is replaced by a stub, so the daemon's full
pipeline — command dispatch, worker thread, agent wiring, broadcasts,
persistence — executes for real.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar import daemon_client
from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core import vscode_config
from kiss.server.web_server import RemoteAccessServer

PARENT_TAB_ID = "webtab-parent-1"
VIEWER_SUB_TAB_ID = "webtab-parent-1__sub_child"


def _init_repo(repo: str) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True,
            check=False,
        )

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test User")
    git("config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    git("add", "seed.txt")
    git("commit", "-q", "-m", "seed")


class RunAgentSubagentTabTest(unittest.TestCase):
    """Sub-agent dispatches get run_parallel tab semantics end to end."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="run_agent_subtab_")
        self.sock_path = str(Path(self.tmpdir) / "sorcar.sock")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.repo)

        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved_persistence = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None
        self._saved_config_override = (
            vars(vscode_config).get("CONFIG_DIR"),
            vars(vscode_config).get("CONFIG_PATH"),
        )
        vscode_config.CONFIG_DIR = kiss_dir
        vscode_config.CONFIG_PATH = kiss_dir / "config.json"

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            uds_path=self.sock_path, work_dir=self.repo,
        )
        self.server._printer._loop = self.loop
        self.server._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path,
            ),
            self.loop,
        ).result(timeout=5)

        self._viewer_writer: asyncio.StreamWriter | None = None
        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        from kiss.server import agent_state

        agent_state.agent_states.clear()

        async def _shutdown() -> None:
            if self._viewer_writer is not None:
                try:
                    self._viewer_writer.close()
                except Exception:
                    pass
            with self.server._printer._ws_lock:
                writers = list(self.server._printer._uds_writers)
            for writer in writers:
                try:
                    writer.close()
                except Exception:
                    pass
            self.uds_server.close()
            await self.uds_server.wait_closed()
            pending = [
                t for t in asyncio.all_tasks()
                if t is not asyncio.current_task()
            ]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(
                _shutdown(), self.loop,
            ).result(timeout=5)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()

        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_persistence
        saved_dir, saved_path = self._saved_config_override
        if saved_dir is None:
            if "CONFIG_DIR" in vars(vscode_config):
                delattr(vscode_config, "CONFIG_DIR")
        else:
            vscode_config.CONFIG_DIR = saved_dir
        if saved_path is None:
            if "CONFIG_PATH" in vars(vscode_config):
                delattr(vscode_config, "CONFIG_PATH")
        else:
            vscode_config.CONFIG_PATH = saved_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _open_viewer(self) -> list[dict[str, Any]]:
        """Open a webview-like UDS connection and drain its events."""

        async def _open() -> tuple[
            asyncio.StreamReader, asyncio.StreamWriter,
        ]:
            return await asyncio.open_unix_connection(
                self.sock_path, limit=16 * 1024 * 1024,
            )

        reader, writer = asyncio.run_coroutine_threadsafe(
            _open(), self.loop,
        ).result(timeout=5)
        self._viewer_writer = writer
        received: list[dict[str, Any]] = []

        async def _drain() -> None:
            while True:
                line = await reader.readline()
                if not line:
                    return
                try:
                    received.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        asyncio.run_coroutine_threadsafe(_drain(), self.loop)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self.server._printer._ws_lock:
                if self.server._printer._uds_writers:
                    return received
            time.sleep(0.02)
        raise AssertionError("viewer UDS connection never registered")

    def _send_from_viewer(self, cmd: dict[str, Any]) -> None:
        writer = self._viewer_writer
        assert writer is not None

        async def _send() -> None:
            writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
            await writer.drain()

        asyncio.run_coroutine_threadsafe(_send(), self.loop).result(
            timeout=5,
        )

    @staticmethod
    def _wait_for(
        predicate: Any, timeout: float = 15.0, what: str = "condition",
    ) -> Any:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = predicate()
            if found:
                return found
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for {what}")

    def _install_stub(
        self,
        registry_during_child: dict[str, Any],
        child_release: threading.Event,
        child_marker: str,
    ) -> None:
        """Stub the LLM boundary; the child run blocks on *child_release*."""

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            prompt = str(kwargs.get("prompt_template", ""))
            if child_marker in prompt:
                registry_during_child["tabs"] = [
                    dict(entry)
                    for entry in
                    self.server._vscode_server.tab_registry.snapshot()
                ]
                # What a NESTED spawn (the child calling run_parallel
                # or run_agent itself) would use as its parent tab —
                # before any client watches the child, and after the
                # viewer's resumeSession subscription.
                registry_during_child["nested_parent_before"] = (
                    self_agent._subagent_parent_tab_id()
                )
                child_release.wait(timeout=30)
                registry_during_child["nested_parent_after"] = (
                    self_agent._subagent_parent_tab_id()
                )
            self_agent.total_tokens_used = 5
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:  # pragma: no branch
                printer.print(
                    raw, type="result", step_count=1,
                    total_tokens=5, cost="$0.0010",
                )
            return raw

        self._parent_class.run = stub_run

    def test_parented_dispatch_gets_run_parallel_tab_semantics(self) -> None:
        """The full sub-agent tab contract, driven over the real daemon."""
        child_marker = "child subagent task xk7"
        child_release = threading.Event()
        registry_during_child: dict[str, Any] = {}
        self._install_stub(
            registry_during_child, child_release, child_marker,
        )
        received = self._open_viewer()

        parent = daemon_client.run(
            "parent seed task",
            work_dir=self.repo,
            use_worktree=False,
            auto_commit=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert parent.success is True
        assert parent.task_id

        child_out: dict[str, Any] = {}

        def dispatch_child() -> None:
            child_out["result"] = daemon_client.run(
                child_marker,
                work_dir=self.repo,
                use_worktree=False,
                auto_commit=False,
                parent_task_id=parent.task_id,
                parent_tab_id=PARENT_TAB_ID,
                sock_path=self.sock_path,
                timeout=60,
            )

        dispatcher = threading.Thread(target=dispatch_child, daemon=True)
        dispatcher.start()
        try:
            # 1. The run announces itself exactly like a run_parallel
            #    child: a ``new_tab`` broadcast naming the child's
            #    persisted task id and the CALLING task's tab id.
            new_tab = self._wait_for(
                lambda: next(
                    (
                        e for e in list(received)
                        if e.get("type") == "new_tab"
                        and e.get("parent_tab_id") == PARENT_TAB_ID
                    ),
                    None,
                ),
                what="new_tab broadcast with the parent tab id",
            )
            child_task_id = str(new_tab.get("task_id") or "")
            assert child_task_id, f"new_tab carries no task id: {new_tab!r}"
            assert child_task_id != parent.task_id

            # 2. No top-level registry tab for the dispatch: the
            #    snapshot taken inside the (still blocked) child run
            #    must not list its ``api-…`` tab.
            self._wait_for(
                lambda: "tabs" in registry_during_child,
                what="child run to reach the stub",
            )
            api_tabs = [
                entry for entry in registry_during_child["tabs"]
                if str(entry.get("tabId", "")).startswith("api-")
            ]
            assert api_tabs == [], (
                f"sub-agent dispatch must not register a top-level "
                f"tab, got {api_tabs!r}"
            )

            # 3. A client opening the sub-agent tab subscribes it to
            #    the child's stream (what media/main.js does on
            #    ``new_tab``): the fan-out then targets that tab.
            self._send_from_viewer({
                "type": "resumeSession",
                "taskId": child_task_id,
                "tabId": VIEWER_SUB_TAB_ID,
            })
            self._wait_for(
                lambda: VIEWER_SUB_TAB_ID
                in self.server._printer._fanout_targets(child_task_id),
                what="viewer sub-agent tab subscription",
            )
        finally:
            child_release.set()
        dispatcher.join(timeout=60)
        assert not dispatcher.is_alive(), "child dispatch never returned"
        child = child_out["result"]
        assert child.success is True
        assert child.task_id == child_task_id

        # 4. The completion signal of a run_parallel worker: one
        #    ``subagentDone`` per watching tab (the subscribed viewer
        #    tab and the dispatch's own api- id).  The two arrive as
        #    separate broadcasts, so wait until BOTH are in.
        def _done_tabs() -> set[str]:
            return {
                str(e.get("tab_id"))
                for e in list(received)
                if e.get("type") == "subagentDone"
            }

        self._wait_for(
            lambda: VIEWER_SUB_TAB_ID in _done_tabs()
            and any(t.startswith("api-") for t in _done_tabs()),
            what="subagentDone broadcasts for the viewer and api- tabs",
        )
        # Exactly one completion per watching tab: the normal-path
        # signal and its exception-path backstop must never both fire.
        time.sleep(0.3)
        done_events = [
            e for e in list(received) if e.get("type") == "subagentDone"
        ]
        done_ids = [str(e.get("tab_id")) for e in done_events]
        assert len(done_ids) == len(set(done_ids)), (
            f"subagentDone double-fired: {done_ids!r}"
        )

        # 4b. A nested spawn made by the child must be parented under
        #     the tab really watching the child — never the hidden
        #     api-… dispatch id once a real viewer is subscribed
        #     (before any viewer exists, the dispatch id is the only
        #     candidate left and the resolver falls back to it).
        assert str(
            registry_during_child["nested_parent_before"],
        ).startswith("api-")
        assert (
            registry_during_child["nested_parent_after"]
            == VIEWER_SUB_TAB_ID
        ), (
            "nested spawns must be parented under the watching tab, "
            f"got {registry_during_child['nested_parent_after']!r}"
        )

        # 5. The child's history row nests under the calling task,
        #    exactly like a run_parallel child's — this is what makes
        #    parent replays restore the sub-agent tab.
        nested = _persistence._load_subagent_rows_by_parent_task_id(
            parent.task_id,
        )
        nested_ids = {str(row.get("task_id")) for row in nested}
        assert child_task_id in nested_ids, (
            f"child row not nested under the parent task: {nested!r}"
        )
        top_level_ids = {
            str(row.get("id")) for row in _persistence._load_history()
        }
        assert child_task_id not in top_level_ids, (
            "child row must be hidden from top-level history"
        )
        assert parent.task_id in top_level_ids

        # 6. A parent replay (every webview ``ready`` triggers one per
        #    bound tab) re-announces the finished child with its row's
        #    start stamp: the webview attributes the child to the
        #    fan-out tool call that was running at that time, and only
        #    an owning, collapsed panel keeps the finished child's tab
        #    from reopening on every reconnect.
        child_row = next(
            row for row in nested if str(row.get("task_id")) == child_task_id
        )
        row_start_ts = int(
            json.loads(str(child_row.get("extra") or "{}")).get("startTs", 0),
        )
        assert row_start_ts > 0, f"child row has no startTs: {child_row!r}"
        self._send_from_viewer({
            "type": "resumeSession",
            "chatId": parent.chat_id,
            "taskId": parent.task_id,
            "tabId": PARENT_TAB_ID,
        })
        replayed = self._wait_for(
            lambda: next(
                (
                    e for e in list(received)
                    if e.get("type") == "openSubagentTab"
                    and e.get("tab_id") == f"{PARENT_TAB_ID}__sub_{child_task_id}"
                ),
                None,
            ),
            what="openSubagentTab re-announce of the finished child",
        )
        assert replayed.get("isDone") is True
        assert replayed.get("parent_tab_id") == PARENT_TAB_ID
        assert replayed.get("startTs") == row_start_ts, (
            f"re-announce must carry the row's startTs: {replayed!r}"
        )

    def test_multi_task_dispatch_completes_every_child_row(self) -> None:
        """A multi-``<task>`` dispatch completes each child row's tab.

        Every ``<task>`` subtask allocates its own child row and
        broadcasts its own ``new_tab``, so each row must also get a
        ``subagentDone`` — the intermediate one as its subtask ends,
        not only the last one at end of run.  (The exception-path
        backstop in the mandatory-cleanup ``finally`` stays untested
        by design: reaching it requires interrupting the persistence
        step mid-cleanup, which needs a fault injected into the
        database layer — a test double.)
        """
        child_marker = "multi task child mk3"
        child_release = threading.Event()
        registry_during_child: dict[str, Any] = {}
        self._install_stub(
            registry_during_child, child_release, child_marker,
        )
        child_release.set()
        received = self._open_viewer()

        parent = daemon_client.run(
            "parent seed task",
            work_dir=self.repo,
            use_worktree=False,
            auto_commit=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert parent.success is True

        child = daemon_client.run(
            f"<task>{child_marker} one</task>"
            f"<task>{child_marker} two</task>",
            work_dir=self.repo,
            use_worktree=False,
            auto_commit=False,
            parent_task_id=parent.task_id,
            parent_tab_id=PARENT_TAB_ID,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert child.success is True

        def _new_tab_task_ids() -> set[str]:
            return {
                str(e.get("task_id"))
                for e in list(received)
                if e.get("type") == "new_tab"
                and e.get("parent_tab_id") == PARENT_TAB_ID
            }

        def _done_count() -> int:
            return sum(
                1 for e in list(received)
                if e.get("type") == "subagentDone"
                and str(e.get("tab_id", "")).startswith("api-")
            )

        # One new_tab per subtask row, and one completion per row
        # (both target the dispatch's api- id here: no client tab ever
        # subscribed to either child row).
        self._wait_for(
            lambda: len(_new_tab_task_ids()) == 2 and _done_count() == 2,
            what="two new_tab broadcasts and two subagentDone signals",
        )
        # Both rows nest under the calling task.
        nested_ids = {
            str(row.get("task_id"))
            for row in _persistence._load_subagent_rows_by_parent_task_id(
                parent.task_id,
            )
        }
        assert _new_tab_task_ids() <= nested_ids

    def test_parentless_dispatch_keeps_top_level_tab(self) -> None:
        """Without a parent, the dispatch keeps its registry tab and
        broadcasts neither ``new_tab`` nor ``subagentDone``."""
        child_marker = "plain top-level task qz9"
        child_release = threading.Event()
        registry_during_child: dict[str, Any] = {}
        self._install_stub(
            registry_during_child, child_release, child_marker,
        )
        received = self._open_viewer()
        child_release.set()

        result = daemon_client.run(
            child_marker,
            work_dir=self.repo,
            use_worktree=False,
            auto_commit=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert result.success is True
        api_tabs = [
            entry for entry in registry_during_child["tabs"]
            if str(entry.get("tabId", "")).startswith("api-")
        ]
        assert len(api_tabs) == 1, (
            f"an ordinary run must keep its registry tab, got {api_tabs!r}"
        )
        # The terminal broadcasts have flushed by the time the client's
        # result event arrived (same ordering the subagentDone assertion
        # of the parented test relies on); give the viewer stream a
        # moment to drain before the negative check.
        time.sleep(0.5)
        leaked = [
            e for e in list(received)
            if e.get("type") in ("new_tab", "subagentDone")
        ]
        assert leaked == [], (
            f"parentless dispatch leaked sub-agent broadcasts: {leaked!r}"
        )

    def test_parent_reviewer_wire_field_marks_child_reviewer(self) -> None:
        """``parent_reviewer=True`` survives the daemon round trip.

        The dispatched child's reconstructed ``_subagent_info`` must
        carry the reviewer marker (see
        :mod:`kiss.agents.sorcar.fanout_guard`), so the child's own
        ``run_parallel`` refuses to spawn further reviewers.  A control
        dispatch without the flag stays unmarked.
        """
        reviewer_marker = "reviewer wire child zq9"
        control_marker = "control wire child zq9"
        # No parentReviewer flag, but the prompt itself is a review
        # task: the daemon must mark it from the EFFECTIVE prompt (the
        # path an agent script's prompt() override would take).
        worded_marker = "wire child zq9, inspect it for defects"
        recorded: dict[str, Any] = {}

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            prompt = str(kwargs.get("prompt_template", ""))
            for marker in (reviewer_marker, control_marker, worded_marker):
                if marker in prompt:
                    recorded[marker] = {
                        "info": dict(self_agent._subagent_info or {}),
                        "is_reviewer": self_agent._is_reviewer_subagent(),
                        "spawn_result": next(
                            t for t in self_agent._get_tools()
                            if getattr(t, "__name__", "") == "run_parallel"
                        )('["Review the diff for regressions"]')
                        if marker == reviewer_marker else "",
                    }
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:  # pragma: no branch
                printer.print(
                    raw, type="result", step_count=1,
                    total_tokens=1, cost="$0.0001",
                )
            return raw

        self._parent_class.run = stub_run
        self._open_viewer()

        parent = daemon_client.run(
            "parent seed task",
            work_dir=self.repo,
            use_worktree=False,
            auto_commit=False,
            sock_path=self.sock_path,
            timeout=60,
        )
        assert parent.success is True

        for marker, parent_reviewer in (
            (reviewer_marker, True),
            (control_marker, False),
            (worded_marker, False),
        ):
            child = daemon_client.run(
                marker,
                work_dir=self.repo,
                use_worktree=False,
                auto_commit=False,
                use_web_tools=False,
                parent_task_id=parent.task_id,
                parent_tab_id=PARENT_TAB_ID,
                parent_reviewer=parent_reviewer,
                sock_path=self.sock_path,
                timeout=60,
            )
            assert child.success is True

        reviewer = recorded[reviewer_marker]
        assert reviewer["info"]["reviewer"] is True
        assert reviewer["is_reviewer"] is True
        assert reviewer["spawn_result"].startswith(
            "Error: You are a reviewer sub-agent"
        )
        control = recorded[control_marker]
        assert control["info"]["reviewer"] is False
        assert control["is_reviewer"] is False
        worded = recorded[worded_marker]
        assert worded["info"]["reviewer"] is True
        assert worded["is_reviewer"] is True
