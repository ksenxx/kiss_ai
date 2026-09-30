# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""``ready`` answers with the tips and Inject-promptlet bootstrap.

The daemon owns ``MY_INJECTION.md``, the bundled promptlets, ``TIPS.md``
and the tips opt-out marker.  Both surfaces (the VS Code webview over
UDS and the remote webapp over WSS) paint from the ``tricksData`` and
``tipsData`` events that :meth:`RemoteAccessServer._handle_ready` sends
to the (re)connecting client, so the extension parses none of those
files itself.  A real :class:`RemoteAccessServer` is constructed and
driven directly.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kiss.server import web_server
from kiss.server.tips import tips_data
from kiss.tests.server.test_web_server_bugs import _RecordingEndpoint, _ServerTestBase


@contextlib.contextmanager
def _environ(**values: str) -> Iterator[None]:
    """Set environment variables for the block, then restore them."""
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestReadyBootstrapEvents(_ServerTestBase):
    """``_handle_ready`` sends ``tricksData`` and ``tipsData`` to the sender."""

    def _events(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.broadcasts if e.get("type") == kind]

    async def test_ready_sends_tricks_and_tips_to_the_connection(self) -> None:
        """One ``tricksData`` and one ``tipsData``, stamped for the
        (re)connecting connection only, carrying exactly what the page
        builder embeds for the browser."""
        with tempfile.TemporaryDirectory() as tmp:
            tips_file = Path(tmp) / "TIPS.md"
            tips_file.write_text("# Tip\n\nHello from the daemon.\n")
            injections = Path(tmp) / "INJECTIONS.md"
            injections.write_text("## Trick\n\nbundled one\n")
            home = Path(tmp) / "home"
            home.mkdir()
            (home / "MY_INJECTION.md").write_text("## Trick\n\nmine\n")
            with _environ(
                KISS_TIPS_PATH=str(tips_file),
                KISS_INJECTIONS_PATH=str(injections),
                KISS_HOME=str(home),
            ):
                await self.server._handle_ready(
                    {"type": "ready", "tabId": "t1", "connId": "c1", "restoredTabs": []},
                    _RecordingEndpoint(),
                )
                expected_tips = tips_data(web_server._read_version())  # type: ignore[attr-defined]

        tricks = self._events("tricksData")
        self.assertEqual(len(tricks), 1, self.broadcasts)
        self.assertEqual(tricks[0]["connId"], "c1")
        self.assertEqual(tricks[0]["tricks"], ["mine", "bundled one"])
        self.assertEqual(tricks[0]["userCount"], 1)

        tips = self._events("tipsData")
        self.assertEqual(len(tips), 1, self.broadcasts)
        self.assertEqual(tips[0]["connId"], "c1")
        self.assertEqual(tips[0]["tips"], ["Hello from the daemon."])
        self.assertIs(tips[0]["show"], True)
        self.assertEqual(tips[0]["version"], expected_tips["version"])

    async def test_opt_out_marker_turns_show_off(self) -> None:
        """``$KISS_HOME/TIPS_DISABLED`` (written by the ``tipsOptOut``
        API on either surface) keeps the tips closed on every surface."""
        with tempfile.TemporaryDirectory() as tmp:
            tips_file = Path(tmp) / "TIPS.md"
            tips_file.write_text("# Tip\n\nHello.\n")
            home = Path(tmp) / "home"
            home.mkdir()
            (home / "TIPS_DISABLED").write_text("2026-10-01T00:00:00\n")
            with _environ(KISS_TIPS_PATH=str(tips_file), KISS_HOME=str(home)):
                await self.server._handle_ready(
                    {"type": "ready", "tabId": "t1", "connId": "c2", "restoredTabs": []},
                    _RecordingEndpoint(),
                )
        tips = self._events("tipsData")
        self.assertEqual(len(tips), 1)
        self.assertEqual(tips[0]["tips"], ["Hello."])
        self.assertIs(tips[0]["show"], False)


class TestSubmitForwardsEditorContext(_ServerTestBase):
    """The browser ``submit`` carries the webview's editor context into ``run``."""

    async def test_active_file_reaches_the_run_command(self) -> None:
        """``activeFile`` (the file tab the user viewed last, sent by
        main.js on both surfaces) lands on the ``run`` the daemon
        builds, so the task's system prompt names it exactly as for a
        VS Code run; a submit without one carries ``None``."""
        await self.server._handle_submit(
            {"tabId": "t1", "prompt": "explain", "activeFile": "/ws/notes.md"},
        )
        await self.server._handle_submit({"tabId": "t2", "prompt": "hi"})
        runs = [c for c in self.run_cmds if c.get("type") == "run"]
        self.assertEqual([r["activeFile"] for r in runs], ["/ws/notes.md", None])
