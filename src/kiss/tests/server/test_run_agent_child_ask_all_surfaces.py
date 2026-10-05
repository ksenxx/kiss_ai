# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Live E2E: run_agent tabs and questions survive collapse and multi-surface replay.

Uses a real configured model, daemon, run_agent dispatch and question callback,
with the shared webview running under jsdom. No model responses or production
functions are replaced. Set KISS_LIVE_TEST_MODEL to select another configured
model; without the selected model's credentials this test skips.

Legacy announcements using a different tab ID for a hand-closed task are not
emitted by the current daemon. The existing JS compatibility suite covers that
fallback; exercising it here would require injecting a fake daemon event.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import unittest
from typing import Any

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar import persistence as _persistence
from kiss.core import vscode_config
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.server.server import _subagent_is_done
from kiss.tests.server.test_run_agent_subagent_tab import DaemonLocalHarness
from kiss.tests.server.test_subagent_tabs_all_surfaces import _JSDOM_PKG, SurfaceBridge

QUESTION = "Which colour should the child use? (zq7)"
ANSWER = "teal zq7"
CHILD_TASK = (
    f"Immediately call ask_user_question with question={QUESTION!r}. "
    "Wait for the user's answer, then call finish with success=true, "
    "is_continue=false and the answer in summary_in_html. Do nothing else."
)
DISPATCH_ARGS = {
    "task": CHILD_TASK,
    "use_worktree": "false",
    "auto_commit": "false",
    "auto_classify": "false",
    "use_memory": "false",
    "use_web_tools": "false",
    "timeout": "180",
}
PARENT_PROMPT = (
    "This is a UI integration test. Immediately call run_agent exactly once "
    f"with these arguments: {json.dumps(DISPATCH_ARGS)}. "
    "After it returns call finish with success=true, is_continue=false and "
    "summary_in_html='<p>done</p>'. Do not call any other tools."
)
SURFACES = ("sidebar", "remote", "slow", "remote2", "editor")


class RunAgentChildAskAllSurfacesTest(DaemonLocalHarness):
    """Exercise live, delayed, reconnected and completed child views."""

    def setUp(self) -> None:
        """Start an isolated daemon with an actual configured model."""
        model = os.environ.get("KISS_LIVE_TEST_MODEL", "claude-fable-5-1")
        if model not in get_available_models():
            self.skipTest(f"live model {model} is not configured")
        if shutil.which("node") is None or not _JSDOM_PKG.is_file():
            self.skipTest("node and the extension's jsdom dependency are required")
        super().setUp()
        vscode_config.save_config({
            "last_model": model,
            "classify_tasks": False,
            "use_memory": False,
            "use_web_browser": False,
            "is_worktree": False,
            "auto_commit_mode": False,
        })
        self.server._vscode_server._refresh_default_model()
        # This is the daemon's normal dispatch destination, set at boot.
        self._saved_daemon_endpoint = cron_agent._daemon_endpoint_file
        cron_agent._daemon_endpoint_file = str(self.endpoint_file)
        self.bridge = SurfaceBridge(str(self.endpoint_file))

    def tearDown(self) -> None:
        """Stop remaining tasks before releasing the daemon and its database."""
        states = list(agent_state.agent_states.values())
        for state in states:
            self.server._vscode_server._handle_command({
                "type": "stop", "tabId": state.tab_id,
            })
        for state in states:
            if state.task_thread:
                state.task_thread.join(timeout=20)
        self.bridge.quit()
        cron_agent._daemon_endpoint_file = self._saved_daemon_endpoint
        super().tearDown()

    @staticmethod
    def _wait_for(predicate: Any, timeout: float = 120.0, what: str = "") -> Any:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.1)
        raise AssertionError(f"timed out waiting for {what}")

    def _wait_child_tab(self, name: str, child_tab: str, visible: bool = True) -> None:
        # Automatic helper agents can have their own tabs. Assert the target
        # child's lifecycle without requiring unrelated helpers to be absent.
        def matches() -> bool:
            present = any(tab["id"] == child_tab for tab in self.bridge.sub_tabs(name))
            return present == visible

        try:
            self._wait_for(
                matches, what=f"{name}: {child_tab} visible={visible}",
                timeout=15 if visible else 120,
            )
        except AssertionError as error:
            raise AssertionError(
                f"{error}; tabs={self.bridge.tabs(name)}; "
                f"events={self.bridge.events(name)[-20:]}"
            ) from error

    def _assert_question_shown(self, name: str, child_tab: str) -> None:
        # A new question brings its tab forward on every surface
        # (main.js focusAskingTab); when the user has since moved to another
        # tab, the asking tab carries the "Waiting for your answer" mark
        # instead, which is only drawn on a non-active tab.
        def shown() -> bool:
            ask = self.bridge.call("ask", name=name)
            return ask["activeTabId"] == child_tab or child_tab in ask["attention"]

        self._wait_for(shown, what=f"child question on {name}")
        if self.bridge.call("ask", name=name)["activeTabId"] != child_tab:
            assert self.bridge.call("activateTab", name=name, tabId=child_tab)["found"]

        def answering() -> dict[str, Any] | None:
            ask = self.bridge.call("ask", name=name)
            return ask if ask["answering"] else None

        ask = self._wait_for(answering, what=f"answer composer on {name}")
        assert ask["activeTabId"] == child_tab, (name, ask)
        assert QUESTION in ask["question"], (name, ask)

    def _count_replays(self, name: str, tab_id: str) -> int:
        return sum(
            1 for event in self.bridge.events(name)
            if isinstance(event, dict)
            and event["type"] == "task_events"
            and event.get("tabId") == tab_id
        )

    def test_child_tab_and_question_on_every_surface(self) -> None:
        """One answer completes the child and closes its tab on every surface."""
        self.bridge.open("sidebar")
        self.bridge.open("remote", ' class="remote-chat"')
        self.bridge.open("slow", ' class="remote-chat"')
        self.bridge.call("hold", name="slow")
        parent_tab = self.bridge.tabs("sidebar")[0]["id"]
        self.bridge.call("submit", name="sidebar", text=PARENT_PROMPT)

        def child_row() -> str | None:
            for row in _persistence._load_history():
                if row.get("task") != PARENT_PROMPT:
                    continue
                children = _persistence._load_subagent_rows_by_parent_task_id(str(row["id"]))
                for child in children:
                    if child["task"] == CHILD_TASK:
                        return str(child["task_id"])
            return None

        child_id = self._wait_for(child_row, what="run_agent child row")
        child_tab = f"{parent_tab}__sub_{child_id}"
        for name in ("sidebar", "remote"):
            self._wait_child_tab(name, child_tab)
            self._assert_question_shown(name, child_tab)

        # Collapsing a transcript panel must not hide a running task, including
        # a child blocked on a question.
        self.bridge.call("activateTab", name="sidebar", tabId=parent_tab)
        clicked = self.bridge.call("click", name="sidebar", selector=".tc-run-parallel .tc-h")
        assert clicked["found"], clicked
        for name in ("sidebar", "remote"):
            self._wait_child_tab(name, child_tab)
        # The user closing the child's tab on one surface closes it on every
        # surface (the child keeps running); expanding the parent's fan-out
        # panel reopens it everywhere, question included.
        self.bridge.call("closeTab", name="remote", tabId=child_tab)
        for name in ("sidebar", "remote"):
            self._wait_child_tab(name, child_tab, False)
        clicked = self.bridge.call("click", name="sidebar", selector=".tc-run-parallel .tc-h")
        assert clicked["found"], clicked
        for name in ("sidebar", "remote"):
            self._wait_child_tab(name, child_tab)
            self._assert_question_shown(name, child_tab)

        # A slow client's resume replays the child onto already-open surfaces.
        replays = {n: self._count_replays(n, child_tab) for n in ("sidebar", "remote")}
        self.bridge.call("release", name="slow")
        self._wait_child_tab("slow", child_tab)
        self._assert_question_shown("slow", child_tab)
        for name in ("sidebar", "remote"):
            def replayed() -> bool:
                return self._count_replays(name, child_tab) > replays[name]

            self._wait_for(replayed, what=f"slow client's replay on {name}")
            self._assert_question_shown(name, child_tab)

        self.bridge.open("remote2", ' class="remote-chat"')
        self.bridge.open(
            "editor", ' class="editor-tab-mode"'
            f' data-kiss-tab-id="{parent_tab}" data-kiss-in-registry="1"',
        )
        for name in SURFACES:
            self._wait_child_tab(name, child_tab)
            self._assert_question_shown(name, child_tab)

        # Reload one surface while the child is waiting; existing answer boxes
        # and pending question panels must remain intact on the others.
        self.bridge.call("close", name="remote2")
        self.bridge.open("remote2", ' class="remote-chat"')
        for name in SURFACES:
            self._wait_child_tab(name, child_tab)
            self._assert_question_shown(name, child_tab)

        assert self.bridge.call("answer", name="remote2", text=ANSWER)["answering"]
        for name in SURFACES:
            self._wait_child_tab(name, child_tab, False)
            ask = self.bridge.call("ask", name=name)
            assert not ask["answering"] and not ask["attention"], (name, ask)
            assert any(t["id"] == parent_tab for t in self.bridge.tabs(name))

        # A completion-summary helper may outlive this child. Its tab must
        # survive removal of the finished parent, including on a new client.
        descendants = [
            tab["id"] for tab in self.bridge.sub_tabs("sidebar")
            if tab["running"] and tab["id"].startswith(child_tab + "__sub_")
        ]
        self.bridge.open("epilogue", ' class="remote-chat"')
        for descendant in descendants:
            def descendant_visible_or_done() -> bool:
                return _subagent_is_done(descendant.rsplit("__sub_", 1)[-1]) or any(
                    tab["id"] == descendant for tab in self.bridge.sub_tabs("epilogue")
                )

            self._wait_for(descendant_visible_or_done, what="live descendant replay")

        def parent_finished() -> bool:
            return not any(
                tab["running"] for tab in self.bridge.tabs("sidebar")
                if tab["id"] == parent_tab
            )

        self._wait_for(parent_finished, what="parent completion")
        _persistence._flush_chat_events(child_id)
        child_session = _persistence._load_chat_events_by_task_id(child_id)
        assert child_session is not None
        child_events = child_session["events"]
        assert isinstance(child_events, list)
        assert any(
            event.get("type") == "tool_result" and ANSWER in str(event)
            for event in child_events
        ), "the real child did not receive the answer"
        # Finish automatic summary helpers and the new client's initial replay
        # before testing manual history, so their terminal events cannot race it.
        def replay_and_helpers_finished() -> bool:
            return self._count_replays("epilogue", parent_tab) > 0 and all(
                _subagent_is_done(state.task_id)
                for state in list(agent_state.agent_states.values())
            )

        self._wait_for(replay_and_helpers_finished, what="completion helpers and replay")
        # Finished history remains manually accessible. Closing it on one
        # surface must not let another surface's history replay revive it.
        for name in ("sidebar", "remote2"):
            self.bridge.call("activateTab", name=name, tabId=parent_tab)
            self.bridge.call(
                "click", name=name, selector=".tc-run-parallel:not(.collapsed) .tc-h",
            )
            assert self.bridge.call(
                "click", name=name, selector=".tc-run-parallel.collapsed .tc-h",
            )["found"]
            self._wait_child_tab(name, child_tab)
            if name == "remote2":
                self._wait_child_tab("sidebar", child_tab, False)
            self.bridge.call("closeTab", name=name, tabId=child_tab)
            self._wait_child_tab(name, child_tab, False)

        self.bridge.open("late", ' class="remote-chat"')

        def late_replayed() -> bool:
            return "task_events" in self.bridge.event_types("late")

        self._wait_for(late_replayed, what="completed history replay")
        self._wait_child_tab("late", child_tab, False)
        # The editor root is closable again after its child has completed.
        assert self.bridge.call("closeTab", name="editor", tabId=parent_tab)["found"]
        for name in (*SURFACES, "epilogue", "late"):
            self._wait_child_tab(name, child_tab, False)
            assert not self.bridge.call("tabs", name=name)["errors"], name


if __name__ == "__main__":
    unittest.main()
