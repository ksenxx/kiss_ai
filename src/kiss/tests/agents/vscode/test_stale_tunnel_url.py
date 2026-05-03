"""Integration tests for stale tunnel URL detection and re-broadcast.

Verifies that:
1. web_server.py broadcasts the new URL after tunnel restart
2. SorcarSidebarView.ts persistently watches remote-url.json for changes
3. SorcarSidebarView.ts only sends the URL when it actually changes
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_WEB_SERVER_PY = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "web_server.py"
)
_SIDEBAR_TS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "src"
    / "SorcarSidebarView.ts"
)


class TestWebServerBroadcastsNewUrlOnTunnelRestart(unittest.TestCase):
    """web_server.py must broadcast ``remote_url`` after tunnel restart."""

    def setUp(self) -> None:
        self.src = _WEB_SERVER_PY.read_text()

    def test_restart_tunnel_url_broadcasts_remote_url(self) -> None:
        """_restart_tunnel_url must broadcast the new URL to all clients."""
        match = re.search(
            r"async def _restart_tunnel_url\(self\).*?(?=\n    (?:async )?def |\nclass |\Z)",
            self.src,
            re.DOTALL,
        )
        assert match, "_restart_tunnel_url method not found in web_server.py"
        body = match.group(0)
        self.assertIn(
            'self._printer.broadcast',
            body,
            "_restart_tunnel_url must broadcast to clients after tunnel restart",
        )
        self.assertIn(
            '"remote_url"',
            body,
            "broadcast must include remote_url type",
        )

    def test_broadcast_uses_active_url(self) -> None:
        """The broadcast must use self._active_url, not just tunnel_url."""
        match = re.search(
            r"async def _restart_tunnel_url\(self\).*?(?=\n    (?:async )?def |\nclass |\Z)",
            self.src,
            re.DOTALL,
        )
        assert match
        body = match.group(0)
        self.assertIn("self._active_url", body)

    def test_broadcast_happens_after_save(self) -> None:
        """Broadcast must happen after _save_url_file, not before."""
        match = re.search(
            r"async def _restart_tunnel_url\(self\).*?(?=\n    (?:async )?def |\nclass |\Z)",
            self.src,
            re.DOTALL,
        )
        assert match
        body = match.group(0)
        save_pos = body.index("_save_url_file")
        broadcast_pos = body.index("self._printer.broadcast")
        self.assertGreater(
            broadcast_pos,
            save_pos,
            "broadcast must happen after _save_url_file",
        )


class TestSidebarWatchesUrlFile(unittest.TestCase):
    """SorcarSidebarView.ts must persistently watch remote-url.json."""

    def setUp(self) -> None:
        self.src = _SIDEBAR_TS.read_text()

    def test_has_last_sent_url_field(self) -> None:
        """Sidebar must track the last sent URL to avoid redundant sends."""
        self.assertIn(
            "_lastSentUrl",
            self.src,
            "SorcarSidebarView must have _lastSentUrl field",
        )

    def test_try_read_checks_last_sent_url(self) -> None:
        """_tryReadAndSendUrl must skip sending if URL hasn't changed."""
        match = re.search(
            r"private _tryReadAndSendUrl\(.*?\{(.*?)\n  \}",
            self.src,
            re.DOTALL,
        )
        assert match, "_tryReadAndSendUrl not found"
        body = match.group(1)
        self.assertIn(
            "_lastSentUrl",
            body,
            "_tryReadAndSendUrl must compare against _lastSentUrl",
        )
        self.assertIn(
            "!== this._lastSentUrl",
            body,
            "_tryReadAndSendUrl must only send when URL differs from last sent",
        )

    def test_watch_url_file_is_persistent(self) -> None:
        """_watchUrlFile must not have a remaining countdown or stop condition."""
        match = re.search(
            r"private _watchUrlFile\(.*?\{(.*?)\n  \}",
            self.src,
            re.DOTALL,
        )
        assert match, "_watchUrlFile not found"
        body = match.group(1)
        self.assertNotIn(
            "remaining",
            body,
            "_watchUrlFile must be persistent — no 'remaining' countdown",
        )
        self.assertIn("setInterval", body)

    def test_send_remote_url_always_starts_watcher(self) -> None:
        """_sendRemoteUrl must always start the file watcher."""
        match = re.search(
            r"private _sendRemoteUrl\(\).*?\{(.*?)\n  \}",
            self.src,
            re.DOTALL,
        )
        assert match, "_sendRemoteUrl not found"
        body = match.group(1)
        self.assertIn(
            "_watchUrlFile",
            body,
            "_sendRemoteUrl must always start the persistent watcher",
        )
        self.assertNotIn(
            "setTimeout",
            body,
            "_sendRemoteUrl must not use retry timeouts — watcher handles it",
        )

    def test_send_remote_url_has_no_retries_parameter(self) -> None:
        """_sendRemoteUrl must not accept a retries parameter."""
        match = re.search(r"private _sendRemoteUrl\((.*?)\)", self.src)
        assert match, "_sendRemoteUrl not found"
        params = match.group(1).strip()
        self.assertEqual(
            params,
            "",
            "_sendRemoteUrl must have no parameters (persistent watcher replaces retries)",
        )

    def test_dispose_clears_url_file_watcher(self) -> None:
        """dispose() must clear the URL file watcher timer."""
        match = re.search(
            r"public dispose\(\).*?\{(.*?)\n  \}",
            self.src,
            re.DOTALL,
        )
        assert match, "dispose not found"
        body = match.group(1)
        self.assertIn(
            "_urlFileWatchTimer",
            body,
            "dispose must clear the URL file watcher",
        )


if __name__ == "__main__":
    unittest.main()
