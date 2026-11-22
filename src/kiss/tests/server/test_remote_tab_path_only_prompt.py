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
from types import CodeType

from kiss.agents.sorcar.persistence import _record_steer_input
from kiss.server.remote_tabs import RemoteTab, RemoteTabs
from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.pause import Pause, Pauses
from kiss.tests.server.remote_tabs_probe import machine_of
from kiss.tests.server.test_remote_tabs import RemoteTabsHarness

# Where ``RemoteTabs._registry_title`` scans the registry snapshot it took:
# a thread held at its entry has read the tab's title and not used it yet.
_TITLE_SCAN = next(
    c for c in RemoteTabs._registry_title.__code__.co_consts if isinstance(c, CodeType)
)


def _held_before_its_send(prompt: str) -> Pause:
    """Holds the submit of *prompt* once it is published, before the step that sends it.

    Where its typed text is recorded: a submit drops the tab's pin and
    is sent as one step, which a submit held at the send itself would
    keep every other submit of the tab out of.
    """
    return Pause(_record_steer_input.__code__, lambda local: local["text"] == prompt)


class TestRemoteTabPathOnlyPrompt(RemoteTabsHarness):
    """A path-only prompt of a bound tab is served, stamped, by the machine."""

    async def test_the_relayed_file_is_stamped_with_the_machine(self) -> None:
        tab = "tab-path-only"
        # Bind the tab to the machine with a real run first: the
        # prompt that follows is forwarded along the same stream.
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "box")

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
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "box")
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
        # Nothing is pending any more: the answer repeated changes nothing.
        await asyncio.to_thread(stream._relay_frame, late, json.dumps(late))
        self.assertEqual(await self._title_of(tab), "Renamed")

    async def test_a_file_prompt_gives_back_the_title_it_replaced(self) -> None:
        # Two surfaces submit to the tab at once: the file prompt of one
        # has read the tab's title when the task prompt of the other is
        # published, and is published after it.  The title its
        # ``promptOpened`` gives back is the one it replaced, the
        # task's, not the one it read before.  The task prompt reaches
        # the machine after that answer: a tab running a task opens no
        # file.
        tab = "tab-two-surfaces"
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        target = Path(self.remote.home) / "notes.md"
        target.write_text("draft\n", encoding="utf-8", newline="")
        _, other = await open_local_connection(self.server)
        self.addCleanup(other.close)
        registry = self.server._agent_server.tab_registry
        title_read = Pause(_TITLE_SCAN, lambda local: True)
        task_sent = _held_before_its_send("Second task")
        with Pauses(title_read, task_sent):
            await self._submit(tab, str(target), "box")
            self.assertTrue(await asyncio.to_thread(title_read.paused.wait, 30))
            other.write(json.dumps({
                "type": "submit", "prompt": "Second task", "tabId": tab, "machine": "box",
                "model": self.model, "workDir": self.tmpdir,
            }).encode("utf-8") + b"\n")
            await other.drain()
            # The task prompt is published in the window if anything lets it in.
            for _ in range(40):
                if any(e["title"] == "Second task" for e in registry.snapshot()):
                    break
                await asyncio.sleep(0.05)
            title_read.release()
            await self._until("promptOpened", tab)
            task_sent.release()
        await self._until("task_done", tab)
        self.assertEqual(await self._title_of(tab), "Second task")

    async def _two_files(self, tab: str) -> tuple[Path, Path]:
        """Run a task in *tab* on the machine, which titles it, and write two files there."""
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        files = (Path(self.remote.home) / "first.md", Path(self.remote.home) / "second.md")
        for target in files:
            target.write_text("draft\n", encoding="utf-8", newline="")
        return files

    async def _published(self, tab: str, title: str) -> None:
        """Wait until the registry shows *tab* under *title*."""
        registry = self.server._agent_server.tab_registry
        for _ in range(600):
            if any(e["tabId"] == tab and e["title"] == title for e in registry.snapshot()):
                return
            await asyncio.sleep(0.05)
        raise AssertionError(f"{title} was not published")

    def _answer_held(self, tab: str) -> Pause:
        """Holds the relay of the first ``promptOpened`` the machine sends for *tab*."""
        stream = self.server._server_api.remote_tabs._tabs[tab]
        return Pause(
            RemoteTab._relay_frame.__code__,
            lambda local: local["self"] is stream and local["msg"].get("type") == "promptOpened",
        )

    async def test_two_file_prompts_in_a_row_give_back_the_title_before_both(self) -> None:
        # The second file prompt is published before the answer to the
        # first is relayed: the title it replaced is the first one's
        # path, which no task earned.  Each answer undoes its own
        # prompt, so the tab ends on the title it had before both.
        tab = "tab-two-files"
        first, second = await self._two_files(tab)
        with self._answer_held(tab) as answered:
            await self._submit(tab, str(first), "box")
            self.assertTrue(await asyncio.to_thread(answered.paused.wait, 30))
            await self._submit(tab, str(second), "box")
            await self._published(tab, str(second))
            answered.release()
            await self._until("promptOpened", tab)
            await self._until("promptOpened", tab)
        self.assertEqual(await self._title_of(tab), "Say hello")

    async def test_ten_file_prompts_in_a_row_give_back_the_title_before_them_all(self) -> None:
        # However many file prompts wait for their answers, none is
        # forgotten: forgotten, the first one handed its title to none,
        # and its path was the title the last one gave back.
        tab = "tab-ten-files"
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        files = [Path(self.remote.home) / f"note{n}.md" for n in range(10)]
        with self._answer_held(tab) as answered:
            for target in files:
                target.write_text("draft\n", encoding="utf-8", newline="")
                await self._submit(tab, str(target), "box")
                await self._published(tab, str(target))
            self.assertTrue(await asyncio.to_thread(answered.paused.wait, 30))
            answered.release()
            for _ in files:
                await self._until("promptOpened", tab)
        self.assertEqual(await self._title_of(tab), "Say hello")

    async def test_a_submit_that_is_no_file_prompt_keeps_no_earlier_one(self) -> None:
        # A submit is remembered beneath the next one until the machine
        # tells what it was.  The answers to a task name its prompt, by
        # an id of this daemon's when the surface's is too long for the
        # machine to echo: its title stands, and the submits before it
        # are forgotten.  A prompt that is no text is named in no
        # answer, and opens no file: nothing is kept beneath it.
        tab = "tab-no-file"
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        stream = self.server._server_api.remote_tabs._tabs[tab]
        await self._submit(tab, "Second task", "box", prompt_id="p" * 65)
        frames = await self._run_until_idle(tab)
        self.assertIn([stream.displaced_titles[tab][0]], [f.get("promptIds") for f in frames])
        self.assertEqual(stream.earlier_titles[tab], [])
        await self._send({
            "type": "submit", "prompt": 5, "tabId": tab, "machine": "box",
            "model": self.model, "workDir": self.tmpdir,
        })
        await self._run_until_idle(tab)
        self.assertNotIn(tab, stream.earlier_titles)

    async def test_file_prompts_answered_in_the_other_order_give_it_back_too(self) -> None:
        # Two surfaces: the first file prompt is published and held
        # before its send, the second is published on top of it, sent
        # and answered first.  Its answer puts back the first prompt's
        # path, which the first prompt's own answer then undoes.
        tab = "tab-crossed-files"
        first, second = await self._two_files(tab)
        _, other = await open_local_connection(self.server)
        self.addCleanup(other.close)
        first_sent = _held_before_its_send(str(first))
        with first_sent:
            await self._submit(tab, str(first), "box")
            self.assertTrue(await asyncio.to_thread(first_sent.paused.wait, 30))
            other.write(json.dumps({
                "type": "submit", "prompt": str(second), "tabId": tab, "machine": "box",
                "model": self.model, "workDir": self.tmpdir,
            }).encode("utf-8") + b"\n")
            await other.drain()
            await self._until("promptOpened", tab)
            first_sent.release()
            await self._until("promptOpened", tab)
        self.assertEqual(await self._title_of(tab), "Say hello")

    async def test_a_file_prompt_after_a_rename_does_not_inherit_an_older_title(self) -> None:
        # The tab is renamed between two file prompts, the first still
        # unanswered: the second replaced the new name, and the first
        # one's late answer hands it nothing older.
        tab = "tab-renamed-between"
        first, second = await self._two_files(tab)
        with self._answer_held(tab) as answered:
            await self._submit(tab, str(first), "box")
            self.assertTrue(await asyncio.to_thread(answered.paused.wait, 30))
            self.server._agent_server._registry_update_tab(tab, title="Renamed")
            await self._submit(tab, str(second), "box")
            await self._published(tab, str(second))
            answered.release()
            await self._until("promptOpened", tab)
            await self._until("promptOpened", tab)
        self.assertEqual(await self._title_of(tab), "Renamed")

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
