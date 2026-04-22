"""Regression tests for a subset of tab-routing fixes in VSCode server.

Targets only the five bug sites fixed in this change:
  1. ``_handle_command`` unknown-command error — carries ``tabId`` from cmd.
  2. ``run()`` generic-exception error — carries ``tabId`` from parsed cmd.
  3. ``_start_merge_session`` — ``merge_data`` and ``merge_started`` carry tab.
  4. ``_broadcast_worktree_done`` — ``worktree_done`` carries tab.
  5. ``_handle_worktree_action`` — ``worktree_progress`` carries tab.
  6. ``_get_adjacent_task`` — ``adjacent_task_events`` carries tab.

No mocks: uses a real ``VSCodeServer`` with its ``printer.broadcast``
replaced by a capture-list helper.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.vscode.server import VSCodeServer


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]]]:
    """Build a VSCodeServer whose broadcasts are captured in a list."""
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


class TestUnknownCommandErrorRouted(unittest.TestCase):
    def test_unknown_command_error_carries_tab_id(self) -> None:
        server, events = _make_server()
        server._handle_command({"type": "bogusCmd", "tabId": "t-7"})
        err = [e for e in events if e.get("type") == "error"]
        assert len(err) == 1
        assert err[0].get("tabId") == "t-7"
        assert "Unknown command" in err[0]["text"]

    def test_unknown_command_without_tab_id_omits_field(self) -> None:
        server, events = _make_server()
        server._handle_command({"type": "bogusCmd"})
        err = [e for e in events if e.get("type") == "error"]
        assert len(err) == 1
        assert "tabId" not in err[0]


class TestRunGenericErrorRouted(unittest.TestCase):
    def test_generic_exception_carries_tab_id_from_cmd(self) -> None:
        """run() catches a handler exception and includes cmd.tabId on error."""
        server, events = _make_server()

        def boom(self: VSCodeServer, cmd: dict[str, Any]) -> None:
            raise RuntimeError("boom-x")

        server._HANDLERS = {**VSCodeServer._HANDLERS, "kaboom": boom}  # type: ignore[assignment]

        import io
        import sys
        orig_stdin = sys.stdin
        sys.stdin = io.StringIO(json.dumps({"type": "kaboom", "tabId": "t-42"}) + "\n")
        try:
            server.run()
        finally:
            sys.stdin = orig_stdin

        err = [e for e in events if e.get("type") == "error"]
        assert len(err) == 1
        assert err[0].get("tabId") == "t-42"
        assert err[0]["text"] == "boom-x"

    def test_invalid_json_error_has_no_tab_id(self) -> None:
        """Invalid JSON path cannot parse tabId; field must be absent."""
        server, events = _make_server()

        import io
        import sys
        orig_stdin = sys.stdin
        sys.stdin = io.StringIO("{not json\n")
        try:
            server.run()
        finally:
            sys.stdin = orig_stdin

        err = [e for e in events if e.get("type") == "error"]
        assert len(err) == 1
        assert "tabId" not in err[0]
        assert "Invalid JSON" in err[0]["text"]


class TestStartMergeSessionRouted(unittest.TestCase):
    def _write_merge_json(self, path: Path) -> None:
        payload = {
            "files": [{"path": "a.py", "hunks": [{"lines": ["+x"]}]}],
        }
        path.write_text(json.dumps(payload))

    def test_merge_data_and_merge_started_carry_tab_id(self) -> None:
        server, events = _make_server()
        with tempfile.TemporaryDirectory() as td:
            merge_path = Path(td) / "pending-merge.json"
            self._write_merge_json(merge_path)
            started = server._start_merge_session(str(merge_path), tab_id="t-9")
            assert started is True

        md = [e for e in events if e.get("type") == "merge_data"]
        ms = [e for e in events if e.get("type") == "merge_started"]
        assert len(md) == 1 and md[0].get("tabId") == "t-9"
        assert len(ms) == 1 and ms[0].get("tabId") == "t-9"

    def test_no_tab_id_omits_field(self) -> None:
        server, events = _make_server()
        if hasattr(server.printer._thread_local, "tab_id"):
            delattr(server.printer._thread_local, "tab_id")
        with tempfile.TemporaryDirectory() as td:
            merge_path = Path(td) / "pending-merge.json"
            self._write_merge_json(merge_path)
            started = server._start_merge_session(str(merge_path), tab_id="")
            assert started is True

        md = [e for e in events if e.get("type") == "merge_data"]
        ms = [e for e in events if e.get("type") == "merge_started"]
        assert len(md) == 1 and "tabId" not in md[0]
        assert len(ms) == 1 and "tabId" not in ms[0]


class TestBroadcastWorktreeDoneRouted(unittest.TestCase):
    def test_worktree_done_carries_tab_id(self) -> None:
        server, events = _make_server()
        server._get_tab("t-11")
        server._broadcast_worktree_done(changed=[], tab_id="t-11")

        wd = [e for e in events if e.get("type") == "worktree_done"]
        assert len(wd) == 1
        assert wd[0].get("tabId") == "t-11"

    def test_worktree_done_without_tab_omits_field(self) -> None:
        server, events = _make_server()
        server._get_tab("")
        server._broadcast_worktree_done(changed=[], tab_id="")

        wd = [e for e in events if e.get("type") == "worktree_done"]
        assert len(wd) == 1
        assert "tabId" not in wd[0]


class TestWorktreeProgressRouted(unittest.TestCase):
    def test_worktree_progress_carries_tab_id(self) -> None:
        server, events = _make_server()
        tab = server._get_tab("t-13")
        tab.use_worktree = True
        tab.agent._wt = object()  # type: ignore[assignment]

        def fake_merge() -> str:
            return "Successfully merged"

        tab.agent.merge = fake_merge  # type: ignore[assignment]

        result = server._handle_worktree_action("merge", tab_id="t-13")
        assert result["success"] is True

        wp = [e for e in events if e.get("type") == "worktree_progress"]
        assert len(wp) == 1
        assert wp[0].get("tabId") == "t-13"


class TestAdjacentTaskRouted(unittest.TestCase):
    def test_adjacent_task_events_carries_tab_id(self) -> None:
        server, events = _make_server()
        server._get_adjacent_task(
            chat_id="does-not-exist",
            task="",
            direction="prev",
            tab_id="t-17",
        )
        ate = [e for e in events if e.get("type") == "adjacent_task_events"]
        assert len(ate) == 1
        assert ate[0].get("tabId") == "t-17"

    def test_cmd_handler_propagates_tab_id(self) -> None:
        """`_cmd_get_adjacent_task` forwards cmd.tabId into the event."""
        server, events = _make_server()
        server._get_tab("t-19")
        server._cmd_get_adjacent_task({
            "type": "getAdjacentTask",
            "tabId": "t-19",
            "task": "",
            "direction": "prev",
        })
        ate = [e for e in events if e.get("type") == "adjacent_task_events"]
        assert len(ate) == 1
        assert ate[0].get("tabId") == "t-19"

    def test_no_tab_id_still_tags_event(self) -> None:
        """Empty tab_id still carries a (empty) tabId field (B4 fix).

        Previously, an empty tab_id caused the event to be emitted
        untagged, which reached every tab's frontend handler and
        overwrote whichever tab was active.  With the fix the event
        always carries a tabId so no tab mis-interprets it.
        """
        server, events = _make_server()
        server._get_adjacent_task(
            chat_id="does-not-exist",
            task="",
            direction="prev",
            tab_id="",
        )
        ate = [e for e in events if e.get("type") == "adjacent_task_events"]
        assert len(ate) == 1
        assert "tabId" in ate[0]
        assert ate[0]["tabId"] == ""


if __name__ == "__main__":
    unittest.main()
