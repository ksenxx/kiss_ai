# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for multi-client interactive-UI mirroring.

A chat can be open in several frontend tabs at once — the VS Code
window that launched the task, a browser on a phone, a second laptop.
Every blocking UI the daemon opens for that chat (the ask-user modal,
the merge/diff review, the auto-commit prompt, the worktree
merge/discard strip) must therefore appear in EVERY one of those tabs,
and must disappear from all of them as soon as one of them resolves it.

These tests drive the real backend: a real git repository, the real
``_prepare_and_start_merge`` / ``_finish_merge`` / autocommit code
paths, and commands entering through ``_handle_command`` exactly as the
transports deliver them.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.server.merge_flow as merge_flow_module
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server.server import VSCodeServer
from kiss.server.task_runner import _TaskRunnerMixin
from kiss.tests.agents.vscode._memory_printer import MemoryPrinter

TASK_ID = "task-77"
OWNER_TAB = "owner-tab"
VIEWER_TAB = "viewer-tab"


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command in *cwd* and return the completed process."""
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    """Create a git repo with one committed file so HEAD exists."""
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "README.md").write_text("# Hello\n\nSome content\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "initial commit")


class TestMultiClientUiBroadcast(unittest.TestCase):
    """Merge / auto-commit UIs open and close on every subscribed tab."""

    def setUp(self) -> None:
        """Stand up a real repo, a server and two subscribed tabs."""
        self.tmpdir = tempfile.mkdtemp()
        _init_repo(self.tmpdir)
        self.printer = MemoryPrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = self.tmpdir
        self.owner = self.server._get_tab(OWNER_TAB)
        self.owner.use_worktree = False
        self.owner.task_history_id = TASK_ID
        self.viewer = self.server._get_tab(VIEWER_TAB)
        self.printer.subscribe_tab(TASK_ID, OWNER_TAB)
        self.printer.subscribe_tab(TASK_ID, VIEWER_TAB)
        self._orig_gen = merge_flow_module.generate_commit_message_from_diff
        merge_flow_module.generate_commit_message_from_diff = (  # type: ignore[assignment]
            lambda diff_text, user_prompt=None, task_result=None: (
                "chore: auto-commit test"
            )
        )

    def tearDown(self) -> None:
        """Undo the module patch and drop all per-test global state."""
        merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _tabs_for(self, event_type: str) -> set[str]:
        """Return the tab ids an event of *event_type* was stamped with."""
        return {
            str(ev.get("tabId"))
            for ev in self.printer.emitted
            if ev.get("type") == event_type
        }

    def _start_merge(self) -> None:
        """Modify a tracked file and open the merge review on the owner tab."""
        repo = GitWorktreeOps.discover_repo(Path(self.tmpdir))
        pre_head_sha, pre_hunks, pre_untracked, pre_hashes = (
            _TaskRunnerMixin._capture_pre_snapshot(self.tmpdir, repo, OWNER_TAB)
        )
        Path(self.tmpdir, "README.md").write_text(
            "# Hello\n\nUpdated content by the agent\n",
        )
        started = self.server._prepare_and_start_merge(
            self.tmpdir,
            pre_hunks,
            pre_untracked,
            pre_hashes,
            base_ref=pre_head_sha or "HEAD",
            tab_id=OWNER_TAB,
        )
        self.assertTrue(started, "merge review failed to open")

    def test_merge_review_opens_on_every_subscribed_tab(self) -> None:
        """``merge_data`` / ``merge_started`` reach owner and viewer alike."""
        self._start_merge()
        self.assertEqual(self._tabs_for("merge_data"), {OWNER_TAB, VIEWER_TAB})
        self.assertEqual(
            self._tabs_for("merge_started"), {OWNER_TAB, VIEWER_TAB},
        )
        mirrored = [
            ev
            for ev in self.printer.emitted
            if ev.get("type") == "merge_data" and ev.get("tabId") == VIEWER_TAB
        ]
        self.assertEqual(len(mirrored), 1)
        self.assertEqual(mirrored[0]["mirrorOf"], OWNER_TAB)
        self.assertEqual(mirrored[0]["data"], self._owner_merge_data())

    def _owner_merge_data(self) -> dict[str, Any]:
        """Return the ``data`` payload of the owner's ``merge_data`` event."""
        for ev in self.printer.emitted:
            if ev.get("type") == "merge_data" and ev.get("tabId") == OWNER_TAB:
                data: dict[str, Any] = ev["data"]
                return data
        raise AssertionError("owner never received merge_data")

    def test_viewer_finishing_review_closes_it_everywhere(self) -> None:
        """``all-done`` from the viewer ends the OWNER's review for all tabs."""
        self._start_merge()
        self.assertTrue(self.owner.is_merging)
        self.server._handle_command({
            "type": "mergeAction",
            "action": "all-done",
            "tabId": VIEWER_TAB,
            "workDir": "",
        })
        self.assertFalse(self.owner.is_merging)
        self.assertEqual(self._tabs_for("merge_ended"), {OWNER_TAB, VIEWER_TAB})

    def test_autocommit_prompt_opens_on_every_subscribed_tab(self) -> None:
        """The post-merge auto-commit prompt reaches owner and viewer."""
        self._start_merge()
        self.server._handle_command({
            "type": "mergeAction",
            "action": "all-done",
            "tabId": OWNER_TAB,
            "workDir": self.tmpdir,
        })
        self.assertEqual(
            self._tabs_for("autocommit_prompt"), {OWNER_TAB, VIEWER_TAB},
        )

    def test_viewer_can_answer_the_autocommit_prompt(self) -> None:
        """A viewer's ``commit`` commits the owner's repo and closes all bars."""
        self._start_merge()
        self.server._handle_command({
            "type": "mergeAction",
            "action": "all-done",
            "tabId": OWNER_TAB,
            "workDir": self.tmpdir,
        })
        self.server._handle_command({
            "type": "autocommitAction",
            "action": "commit",
            "tabId": VIEWER_TAB,
            "workDir": "",
        })
        self.assertEqual(
            self._tabs_for("autocommit_done"), {OWNER_TAB, VIEWER_TAB},
        )
        done = next(
            ev
            for ev in self.printer.emitted
            if ev.get("type") == "autocommit_done"
        )
        self.assertTrue(done["committed"], done["message"])
        status = _git(self.tmpdir, "status", "--porcelain")
        self.assertEqual(status.stdout.strip(), "")

    def test_viewer_running_another_task_is_not_mirrored(self) -> None:
        """A co-subscribed tab busy with its own task keeps its own UI."""
        self.viewer.is_task_active = True
        self.viewer.agent = cast(WorktreeSorcarAgent, _OtherTaskAgent())
        self._start_merge()
        self.assertEqual(self._tabs_for("merge_data"), {OWNER_TAB})

    def test_client_joining_mid_review_is_caught_up(self) -> None:
        """A tab opened while the review is pending still gets the UI."""
        self._start_merge()
        self.server._get_tab("late-tab")
        self.printer.subscribe_tab(TASK_ID, "late-tab")
        self.assertEqual(
            self._tabs_for("merge_data"),
            {OWNER_TAB, VIEWER_TAB, "late-tab"},
        )
        self.assertEqual(
            self._tabs_for("merge_started"),
            {OWNER_TAB, VIEWER_TAB, "late-tab"},
        )
        self.server._handle_command({
            "type": "mergeAction",
            "action": "all-done",
            "tabId": OWNER_TAB,
            "workDir": self.tmpdir,
        })
        self.assertIn("late-tab", self._tabs_for("merge_ended"))

    def test_mirror_is_forgotten_once_nothing_is_on_screen(self) -> None:
        """A tab that watched a finished review is no longer a viewer."""
        self._start_merge()
        _git(self.tmpdir, "checkout", "--", "README.md")
        self.server._handle_command({
            "type": "mergeAction",
            "action": "all-done",
            "tabId": OWNER_TAB,
            "workDir": self.tmpdir,
        })
        self.assertEqual(self._tabs_for("merge_ended"), {OWNER_TAB, VIEWER_TAB})
        self.assertEqual(self.printer.ui_mirror_owner(VIEWER_TAB), VIEWER_TAB)

    def test_a_task_reports_its_unanswered_ui(self) -> None:
        """Resuming a chat can tell whether a UI is still waiting on it."""
        self.assertFalse(self.printer.has_ui_mirror_for_task(TASK_ID))
        self._start_merge()
        self.assertTrue(self.printer.has_ui_mirror_for_task(TASK_ID))
        self.assertFalse(self.printer.has_ui_mirror_for_task("other-task"))
        _git(self.tmpdir, "checkout", "--", "README.md")
        self.server._handle_command({
            "type": "mergeAction",
            "action": "all-done",
            "tabId": OWNER_TAB,
            "workDir": self.tmpdir,
        })
        self.assertFalse(self.printer.has_ui_mirror_for_task(TASK_ID))

    def test_closing_a_viewer_tab_stops_mirroring_to_it(self) -> None:
        """Disposing a viewer drops it from the owner's mirror set."""
        self._start_merge()
        self.server._close_tab(VIEWER_TAB)
        self.assertEqual(
            self.printer.ui_mirror_tabs(OWNER_TAB), [OWNER_TAB],
        )
        self.assertEqual(self.printer.ui_mirror_owner(VIEWER_TAB), VIEWER_TAB)
        self.assertTrue(self.owner.is_merging, "the owner keeps reviewing")

    def test_closing_the_owner_tab_closes_the_ui_everywhere(self) -> None:
        """Disposing the owner takes its prompt off the other clients.

        The owner's window is the one holding the repository, so a
        button left behind on the other screens could only act on the
        wrong folder.
        """
        self._start_merge()
        self.server._handle_command({
            "type": "mergeAction",
            "action": "all-done",
            "tabId": OWNER_TAB,
            "workDir": self.tmpdir,
        })
        self.assertEqual(
            self._tabs_for("autocommit_prompt"), {OWNER_TAB, VIEWER_TAB},
        )
        self.server._close_tab(OWNER_TAB)
        self.assertIn(VIEWER_TAB, self._tabs_for("autocommit_done"))
        self.assertEqual(
            self.printer.ui_mirror_owner(VIEWER_TAB, "autocommit_prompt"),
            VIEWER_TAB,
        )

    def test_reloading_the_owner_tab_keeps_the_review_open(self) -> None:
        """Re-subscribing a tab must not tear its own review down."""
        self._start_merge()
        self.printer.cleanup_tab(OWNER_TAB)
        self.assertEqual(self._tabs_for("merge_ended"), set())
        self.assertEqual(
            self.printer.ui_mirror_tabs(OWNER_TAB), [OWNER_TAB, VIEWER_TAB],
        )

    def test_viewers_survive_the_subscriber_set_expiring(self) -> None:
        """A slow review still closes everywhere after subscribers expire."""
        self._start_merge()
        self.printer.cleanup_task(TASK_ID, subscriber_linger_seconds=0)
        self.server._handle_command({
            "type": "mergeAction",
            "action": "all-done",
            "tabId": OWNER_TAB,
            "workDir": self.tmpdir,
        })
        self.assertEqual(self._tabs_for("merge_ended"), {OWNER_TAB, VIEWER_TAB})
        self.assertEqual(
            self._tabs_for("autocommit_prompt"), {OWNER_TAB, VIEWER_TAB},
        )

    def test_an_action_resolves_to_the_owner_showing_that_ui(self) -> None:
        """A tab mirroring two chats routes each action to the right owner."""
        self.printer.open_ui_mirror("other-owner", [VIEWER_TAB], "/elsewhere")
        self.printer.broadcast_tab_ui({
            "type": "autocommit_prompt",
            "tabId": "other-owner",
            "changedFiles": ["x"],
        })
        self._start_merge()
        self.assertEqual(
            self.printer.ui_mirror_owner(VIEWER_TAB, "merge_data"), OWNER_TAB,
        )
        self.assertEqual(
            self.printer.ui_mirror_owner(VIEWER_TAB, "autocommit_prompt"),
            "other-owner",
        )

    def test_worktree_strip_opens_and_closes_on_every_tab(self) -> None:
        """The worktree merge/discard strip mirrors like the other UIs."""
        self.printer.open_ui_mirror(OWNER_TAB, [VIEWER_TAB], self.tmpdir)
        self.printer.broadcast_tab_ui({
            "type": "worktree_done",
            "tabId": OWNER_TAB,
            "changedFiles": ["README.md"],
        })
        self.assertEqual(
            self._tabs_for("worktree_done"), {OWNER_TAB, VIEWER_TAB},
        )
        self.printer.broadcast_tab_ui({
            "type": "worktree_result",
            "tabId": self.printer.ui_mirror_owner(VIEWER_TAB, "worktree_done"),
            "success": True,
            "message": "merged",
        })
        self.assertEqual(
            self._tabs_for("worktree_result"), {OWNER_TAB, VIEWER_TAB},
        )


class _OtherTaskAgent:
    """Stand-in agent whose task id differs from the mirrored task."""

    _last_task_id = "some-other-task"


if __name__ == "__main__":
    unittest.main()
