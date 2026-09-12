# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A stale fan-out copy must not record an old task's worktree for a rebound tab.

Concurrency audit (F3, reviewer R3 finding 13): ``_fanout_stamped``
snapshots a task's subscriber tabs and then, per tab, records the
worktree directory carried by a ``worktree_created`` /
``worktree_done`` event in ``_tab_worktree_dirs`` — the fallback that
remote ``openFile`` / ``checkPaths`` resolve relative paths against.
A concurrent rebind of the same tab (``cleanup_tab`` followed by a
subscription to another task) could land between the snapshot and the
write, so the OLD task's worktree ended up recorded for the rebound
tab.  The write now re-checks the subscription under the printer lock,
and ``cleanup_tab`` drops the entry under that same lock after
unsubscribing.

The window between snapshot and write is a few bytecodes wide and
cannot be hit deterministically without test doubles; the stale
interleaving is therefore reproduced by calling the per-tab tracking
step exactly as the fan-out does, after the rebind has run.  The
healthy paths (subscribed fan-out copy, directly addressed event,
replay envelope, cleanup) run end-to-end through ``broadcast``.
"""

from __future__ import annotations

import unittest

from kiss.server.web_server import WebPrinter


class TestTabWorktreeDirRebind(unittest.TestCase):
    """The per-tab worktree fallback follows the tab's CURRENT task only."""

    def setUp(self) -> None:
        self.printer = WebPrinter()
        self.printer.subscribe_tab("task-old", "tab-1")

    def _created(self, wt_dir: str) -> dict[str, object]:
        return {
            "type": "worktree_created", "taskId": "task-old",
            "worktreeDir": wt_dir, "worktreeWorkDir": wt_dir + "/sub",
        }

    def test_subscribed_fanout_copy_is_recorded(self) -> None:
        self.printer.broadcast(self._created("/repo/.kiss-worktrees/old"))
        self.assertEqual(
            self.printer.worktree_dir_for_tab("tab-1"),
            "/repo/.kiss-worktrees/old/sub",
        )

    def test_cleanup_drops_entry_and_stale_copy_is_rejected(self) -> None:
        self.printer.broadcast(self._created("/repo/.kiss-worktrees/old"))
        # Rebind: the replay path drops the tab's state and subscribes
        # it to another task.
        self.printer.cleanup_tab("tab-1")
        self.assertEqual(self.printer.worktree_dir_for_tab("tab-1"), "")
        self.printer.subscribe_tab("task-new", "tab-1")
        # The stale fan-out copy of the OLD task, whose subscriber
        # snapshot still named tab-1, reaches its per-tab tracking step
        # only now: it must not resurrect the old worktree.
        self.printer._track_worktree_event(
            self._created("/repo/.kiss-worktrees/old"), "tab-1", "task-old",
        )
        self.assertEqual(self.printer.worktree_dir_for_tab("tab-1"), "")
        # A copy from the task the tab is NOW subscribed to is recorded.
        self.printer.broadcast({
            "type": "worktree_created", "taskId": "task-new",
            "worktreeDir": "/repo/.kiss-worktrees/new",
        })
        self.assertEqual(
            self.printer.worktree_dir_for_tab("tab-1"),
            "/repo/.kiss-worktrees/new",
        )

    def test_directly_addressed_events_always_record(self) -> None:
        # worktree_done is stamped with the tab id by the server and
        # takes the direct path; a replay envelope nests the events.
        self.printer.broadcast({
            "type": "worktree_done", "tabId": "tab-9",
            "worktreeDir": "/repo/.kiss-worktrees/done",
        })
        self.assertEqual(
            self.printer.worktree_dir_for_tab("tab-9"),
            "/repo/.kiss-worktrees/done",
        )
        self.printer.broadcast({
            "type": "task_events", "tabId": "tab-8",
            "events": [
                {"type": "prompt", "text": "x"},
                {
                    "type": "worktree_created",
                    "worktreeDir": "/repo/.kiss-worktrees/replayed",
                },
            ],
        })
        self.assertEqual(
            self.printer.worktree_dir_for_tab("tab-8"),
            "/repo/.kiss-worktrees/replayed",
        )
        self.printer.broadcast({
            "type": "worktree_result", "tabId": "tab-8", "success": True,
        })
        self.assertEqual(self.printer.worktree_dir_for_tab("tab-8"), "")


if __name__ == "__main__":
    unittest.main()
