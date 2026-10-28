# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a path-only prompt in a machine-bound tab opens the machine's file.

A chat tab bound to registered machine ``box`` forwards its ``submit``
there.  A prompt that is nothing but the path of a file on the machine
is answered by the machine's daemon (``promptOpened`` then the file's
``fileContent``) and relayed over the tab's stream.  The relayed
``fileContent`` MUST carry ``machine: "box"``, exactly like the reply
to a forwarded ``openFile``: the webapp binds the editor tab to the
machine named in that stamp, and its later ``saveFile`` / reload go
back to the machine.  Unstamped, the webapp took the file for one of
this daemon and saved it here, which answered "File not found" for a
path that exists only on the machine.

The test uses the same real local daemon + real remote daemon ("box")
pair as :mod:`kiss.tests.server.test_remote_tabs`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kiss.tests.server.test_remote_tabs import RemoteTabsHarness


class TestRemoteTabPathOnlyPrompt(RemoteTabsHarness):
    """A path-only prompt of a bound tab is served, stamped, by the machine."""

    async def test_the_relayed_file_is_stamped_with_the_machine(self) -> None:
        tab = "tab-path-only"
        # Bind the tab to the machine with a real run first: the
        # prompt that follows is forwarded along the same stream.
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "box")

        target = Path(self.remote.home) / "notes.md"
        # ``newline=""``: the daemon serves the file's bytes as they are,
        # so the text must not pick up ``\r\n`` on Windows.
        target.write_text("first draft\n", encoding="utf-8", newline="")
        await self._submit(tab, str(target), "box")
        opened = await self._until("promptOpened", tab)
        self.assertEqual(opened["tabId"], tab)
        served = await self._until("fileContent", tab)
        # ``fileContent.path`` is the resolved path (``/private/var`` for
        # a macOS ``/var`` temp dir), as for every ``openFile`` reply.
        self.assertEqual(served["path"], str(target.resolve()))
        self.assertEqual(served["content"], "first draft\n")
        self.assertTrue(served.get("version"), "an editable file carries a version stamp")
        self.assertEqual(
            served.get("machine"), "box",
            "the relayed file must name the machine it was read from",
        )
        # The tab is still bound: the prompt was a file, not a task,
        # and the tab every surface shows keeps its chat's title
        # rather than the path (the binding published the prompt as
        # the title before the machine could tell it was a file).
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "box")
        tabs = await self._ask({"type": "ready", "tabId": tab}, "tabs_state")
        entry = next(t for t in tabs["tabs"] if t["tabId"] == tab)
        self.assertEqual(entry["title"], "Say hello")

        # A save stamped with that machine (what the webapp sends for
        # a tab bound to it) is forwarded there and rewrites the file.
        saved = await self._ask({
            "type": "saveFile", "path": str(target), "content": "second draft\n",
            "tabId": tab, "token": "t1", "version": served["version"],
            "machine": "box",
        }, "fileSaved")
        self.assertTrue(saved["ok"], saved)
        self.assertEqual(saved.get("machine"), "box")
        self.assertEqual(target.read_text(encoding="utf-8"), "second draft\n")

    async def test_a_late_prompt_opened_does_not_undo_a_newer_title(self) -> None:
        # The bind publishes the prompt as the tab's title before the
        # machine can say whether it was a file.  The restore that
        # ``promptOpened`` triggers must be the pending submit's own
        # (the echoed prompt id) and must leave a title a later
        # publication set: the frame is replayed into the real stream
        # as a late arrival would be.
        tab = "tab-late-ack"
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        stream = self.server._server_api.remote_tabs._tabs[tab]
        await self._submit(tab, "Second task", "box")
        await self._run_until_idle(tab)
        self.assertEqual(await self._title_of(tab), "Second task")
        pending = stream.displaced_titles[tab]
        self.assertEqual(pending[1], "Say hello")
        # An earlier submit's answer: its id is not the pending one.
        stale = {"type": "promptOpened", "tabId": tab, "promptId": "earlier"}
        await asyncio.to_thread(stream._relay_frame, stale, json.dumps(stale))
        self.assertEqual(await self._title_of(tab), "Second task")
        self.assertIn(tab, stream.displaced_titles)
        # The pending submit's answer, after something republished the
        # tab: the newer title stands and the entry is consumed.
        self.server._agent_server._registry_update_tab(tab, title="Renamed")
        late = {"type": "promptOpened", "tabId": tab, "promptId": pending[0]}
        await asyncio.to_thread(stream._relay_frame, late, json.dumps(late))
        self.assertEqual(await self._title_of(tab), "Renamed")
        self.assertNotIn(tab, stream.displaced_titles)

    async def _title_of(self, tab: str) -> str:
        tabs = await self._ask({"type": "ready", "tabId": tab}, "tabs_state")
        return str(next(t for t in tabs["tabs"] if t["tabId"] == tab)["title"])

    async def test_a_tab_run_here_gets_an_unstamped_file(self) -> None:
        # The contrast: a tab of this daemon opens the file here and
        # the reply names no machine, so the webapp saves it here.
        tab = "tab-local-only"
        target = Path(self.tmpdir) / "local.md"
        target.write_text("local\n", encoding="utf-8", newline="")
        await self._submit(tab, str(target), "")
        await self._until("promptOpened", tab)
        served = await self._until("fileContent", tab)
        self.assertEqual(served["path"], str(target.resolve()))
        self.assertEqual(served["content"], "local\n")
        self.assertNotIn("machine", served)
