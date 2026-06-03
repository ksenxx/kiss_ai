# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test for the web ``all-done`` autocommit work_dir path.

When the standalone ``kiss-web`` daemon (``RemoteAccessServer``) finishes
a browser-driven merge review, it synthesises a ``mergeAction``
``all-done`` command for the backend ``VSCodeServer``.  Because the
shared daemon may have been launched from a non-git folder, the command
must carry the tab's own repository directory (stamped into the
``merge_data`` payload by ``_start_merge_session`` and stored on the
per-tab :class:`_WebMergeState`) so the post-merge autocommit scan runs
against the correct repository.

This test drives the real ``_handle_web_merge_action`` ``accept-all``
branch (which emits the terminal ``all-done`` command) and asserts the
forwarded command carries the tab's ``workDir``.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from kiss.agents.vscode.web_server import RemoteAccessServer, _WebMergeState


def _build_merge_data(work_dir: Path) -> dict:
    """Create real files and a single-hunk single-file merge data dict."""
    current = work_dir / "f_current.txt"
    base = work_dir / "f_base.txt"
    target = work_dir / "f.txt"
    lines = "".join(f"line{i}\n" for i in range(10))
    current.write_text(lines)
    base.write_text(lines)
    target.write_text(lines)
    return {
        "files": [
            {
                "current": str(current),
                "base": str(base),
                "target": str(target),
                "hunks": [{"bs": 2, "bc": 1, "cs": 2, "cc": 1}],
            }
        ],
        "work_dir": str(work_dir),
    }


class TestWebAllDoneForwardsWorkDir(unittest.IsolatedAsyncioTestCase):
    """The web ``all-done`` command must carry the tab's ``workDir``."""

    async def test_all_done_carries_merge_state_work_dir(self) -> None:
        """Completing a web merge review forwards the merge state's
        ``work_dir`` on the synthesised ``all-done`` command."""
        repo = Path(tempfile.mkdtemp())
        # Daemon work_dir is a *different* folder than the tab's repo.
        daemon_dir = tempfile.mkdtemp()
        server = RemoteAccessServer(
            host="127.0.0.1", port=0, work_dir=daemon_dir,
        )
        loop = asyncio.get_running_loop()
        server._loop = loop

        captured: list[dict] = []

        def fake_handle(cmd: dict) -> None:
            captured.append(cmd)

        server._vscode_server._handle_command = fake_handle  # type: ignore[assignment]

        merge_data = _build_merge_data(repo)
        tab_id = "tab-wd"
        with server._merge_states_lock:
            server._merge_states[tab_id] = _WebMergeState(merge_data)

        await server._handle_web_merge_action({
            "type": "mergeAction",
            "action": "accept-all",
            "tabId": tab_id,
        })

        all_done = [
            c for c in captured
            if c.get("type") == "mergeAction" and c.get("action") == "all-done"
        ]
        assert all_done, captured
        assert all_done[0].get("workDir") == str(repo), all_done[0]

    async def test_web_merge_state_stores_work_dir(self) -> None:
        """``_WebMergeState`` records the ``work_dir`` from merge_data."""
        repo = Path(tempfile.mkdtemp())
        state = _WebMergeState(_build_merge_data(repo))
        assert state.work_dir == str(repo)


if __name__ == "__main__":
    unittest.main()
