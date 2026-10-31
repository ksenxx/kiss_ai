# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Binding a chat tab to a remote machine: what the tab keeps, and the commands racing the bind.

* A tab that showed a local chat and is then bound to a machine holds
  no local chat any more: opening that chat from the CHATS panel (in a
  new tab, the bound tab shows the machine's chat) leaves the bound tab
  open, and a run of the chat there is not fanned out to it.
* The Merge/Discard bar of a bound tab acts on the machine whose task
  left the worktree: ``worktreeAction`` is forwarded down the tab's
  stream, and the machine's answer comes back stamped with the tab.
* A ``closeTab`` from another surface that lands after the bind
  registered the tab's stream but before it published the tab keeps
  the tab closed, and a local-chat ``resumeSession`` landing there
  keeps the chat it showed.
* A local-chat ``resumeSession`` into a tab whose bind is still
  connecting stands the bind down, as ``closeTab`` and ``newChat`` do;
  a resume naming a task the stream never relayed leaves a sub-agent
  tab on its stream.

Real daemons (this one, and remote ones answered by local scripted
model endpoints), real client connections; no mocks, patches or test
doubles.  A race window is held open with
:class:`~kiss.tests.server.pause.Pause`, which holds a daemon thread at
the entry of a real function while the other surface's command lands.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from kiss.core import user_config as vc
from kiss.server.remote_tabs import RemoteTab, RemoteTabs
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon
from kiss.tests.server.conftest import init_repo
from kiss.tests.server.parallel_agent_harness import (
    StandInModelServer,
    finish_response,
    tool_call_response,
)
from kiss.tests.server.pause import Pause
from kiss.tests.server.test_remote_tabs import RemoteTabsHarness


def _restore_config(saved: str | None) -> None:
    """Put this daemon's ``config.json`` back as it was before the test (or remove it)."""
    if saved is None:
        vc._config_path().unlink(missing_ok=True)
    else:
        vc._config_path().write_text(saved)


class _BindHarness(RemoteTabsHarness):
    """The remote-tabs harness, plus registry probes and a second surface."""

    def _row(self, tab_id: str) -> dict[str, str] | None:
        """*tab_id*'s entry in this daemon's shared tab registry (None when closed)."""
        registry = self.server._agent_server.tab_registry
        return next((e for e in registry.snapshot() if e["tabId"] == tab_id), None)

    async def _surface(self) -> tuple[LocalReader, LocalWriter]:
        """Connect another surface (a second window) to this daemon."""
        reader, writer = await open_local_connection(self.server)
        self.addCleanup(writer.close)
        return reader, writer

    @staticmethod
    async def _send_on(writer: LocalWriter, cmd: dict[str, Any]) -> None:
        """Send *cmd* from the surface whose writer is *writer*."""
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _poll(self, what: str, check: Any) -> None:
        """Wait (up to 60 s) until *check()* is true."""
        for _ in range(1200):
            if check():
                return
            await asyncio.sleep(0.05)
        raise AssertionError(f"timed out waiting for {what}")

    async def _bind_settled(self, tab_id: str) -> None:
        """Wait until no bind of *tab_id* is in flight any more."""
        remote_tabs = self.server._server_api.remote_tabs
        await self._poll("the bind to settle", lambda: tab_id not in remote_tabs._connecting)


class TestBoundTabLeavesItsLocalChat(_BindHarness):
    """A tab that showed a local chat, then bound to a machine."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        # This daemon runs tasks too (on the same fake model): the tab
        # shows a chat of its own before it is bound.
        config_path = vc._config_path()
        self.addCleanup(
            _restore_config, config_path.read_text() if config_path.exists() else None,
        )
        vc.save_config({
            "custom_endpoint": f"http://127.0.0.1:{self.model_port}/v1",
            "custom_api_key": "k",
            "max_budget": 100,
        })

    async def test_opening_the_old_local_chat_keeps_the_bound_tab(self) -> None:
        tab, other = "tab-was-local", "tab-reopens-the-chat"
        await self._submit(tab, "Say hello here", "")
        await self._run_until_idle(tab)
        row = self._row(tab)
        assert row is not None
        chat_id = row["chatId"]
        self.assertTrue(chat_id, "the local run bound the tab to its chat")

        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        remote_tabs = self.server._server_api.remote_tabs
        self.assertEqual(remote_tabs.machine_of(tab), "box")
        self.assertEqual(self._row(tab)["chatId"], "", "a machine-bound tab holds no local chat")  # type: ignore[index]

        # The CHATS panel opens the local chat in a new tab: the bound
        # tab shows the machine's chat now.
        await self._send({"type": "resumeSession", "chatId": chat_id, "tabId": other})
        await self._until("task_events", other)
        self.assertEqual(self._row(other)["chatId"], chat_id)  # type: ignore[index]
        self.assertIsNotNone(self._row(tab), "the bound tab is still open")
        self.assertEqual(remote_tabs.machine_of(tab), "box")

        # The chat runs on in the new tab; the bound tab is not a viewer of it.
        await self._submit(other, "Say hello again here", "")
        leaked: list[dict[str, Any]] = []
        for _ in range(500):
            msg = await self._next()
            if msg.get("tabId") == tab:
                leaked.append(msg)
            if (
                msg.get("tabId") == other
                and msg.get("type") == "status"
                and not msg.get("running")
                and not msg.get("replayId")
            ):
                break
        else:
            raise AssertionError("the run in the new tab never ended")
        self.assertEqual(leaked, [], "no frame of the local run reached the bound tab")
        self.assertIsNotNone(self._row(tab))


class TestBindRacesAnotherSurface(_BindHarness):
    """Another surface's command lands inside a bind of the same tab."""

    async def test_a_close_before_the_publication_keeps_the_tab_closed(self) -> None:
        tab = "tab-closed-while-binding"
        _, writer = await self._surface()
        await self._send({"type": "openTab", "tabId": tab})
        await self._poll("the open", lambda: self._row(tab) is not None)
        # Held right after the stream was registered, before the
        # registry publication of the bind.
        with Pause(RemoteTabs._registry_title.__code__, lambda a: a.get("tab_id") == tab) as hold:
            await self._submit(tab, "Say hello", "box")
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 60))
            await self._send_on(writer, {"type": "closeTab", "tabId": tab})
            await self._poll("the close", lambda: self._row(tab) is None)
            hold.release()
            await self._bind_settled(tab)
        self.assertIsNone(self._row(tab), "the closed tab stays closed")
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "")

    async def test_a_local_resume_before_the_publication_keeps_its_chat(self) -> None:
        tab = "tab-resumed-while-publishing"
        _, writer = await self._surface()
        with Pause(RemoteTabs._registry_title.__code__, lambda a: a.get("tab_id") == tab) as hold:
            await self._submit(tab, "Say hello", "box")
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 60))
            await self._send_on(
                writer, {"type": "resumeSession", "chatId": "chat-local", "tabId": tab},
            )
            await self._poll(
                "the replay", lambda: (self._row(tab) or {}).get("chatId") == "chat-local",
            )
            hold.release()
            await self._bind_settled(tab)
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "")
        self.assertEqual(self._row(tab)["chatId"], "chat-local")  # type: ignore[index]

    async def test_a_local_resume_while_the_bind_connects_stands_it_down(self) -> None:
        tab = "tab-resumed-while-connecting"
        _, writer = await self._surface()
        # Held inside the connect, before the stream is registered.
        with Pause(RemoteTab.__init__.__code__, lambda a: a.get("tab_id") == tab) as hold:
            await self._submit(tab, "Say hello", "box")
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 60))
            await self._send_on(
                writer, {"type": "resumeSession", "chatId": "chat-local", "tabId": tab},
            )
            await self._poll(
                "the replay", lambda: (self._row(tab) or {}).get("chatId") == "chat-local",
            )
            hold.release()
            await self._bind_settled(tab)
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "")
        self.assertEqual(self._row(tab)["chatId"], "chat-local")  # type: ignore[index]


class TestSubAgentTabResume(_BindHarness):
    """A resume into a sub-agent tab a bound tab's stream relays."""

    async def test_a_resume_of_a_task_the_stream_never_saw_leaves_the_tab_on_it(self) -> None:
        tab = "tab-fanned-out"
        spawned = await self._fan_out(tab)
        child = f"{tab}__sub_{spawned['task_id']}"
        remote_tabs = self.server._server_api.remote_tabs
        await self._send({"type": "resumeSession", "taskId": spawned["task_id"], "tabId": child})
        await self._poll("the join", lambda: remote_tabs.machine_of(child) == "box")
        # Only a resume of a task the stream relayed joins a tab to it;
        # one it never saw is answered here (nothing runs for it), and
        # the sub-agent tab, still running on the machine, keeps its stream.
        await self._send({"type": "resumeSession", "taskId": "task-not-relayed", "tabId": child})
        for _ in range(300):
            msg = await asyncio.wait_for(self._next(), 60)
            if msg.get("tabId") == child and msg.get("type") == "status" and not msg["running"]:
                break
        else:
            raise AssertionError("the resume was never answered")
        self.assertEqual(remote_tabs.machine_of(child), "box")
        self.assertEqual(remote_tabs.machine_of(tab), "box")


class TestBoundTabWorktreeBar(_BindHarness):
    """The Merge/Discard bar of a tab bound to a machine whose task left a worktree."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.repo = Path(self.tmpdir) / "repo"
        init_repo(self.repo)
        self.standin = StandInModelServer(self._isolate_edit_and_finish)
        self.addCleanup(self.standin.stop)
        self.wt_remote: RemoteDaemon = await asyncio.to_thread(
            start_remote_daemon, Path(self.tmpdir) / "wt-home", password="pw-wt",
            config={"custom_endpoint": self.standin.url, "custom_api_key": "k",
                    "max_budget": 100},
        )
        self.addCleanup(self.wt_remote.stop)
        await self._send({
            "type": "machineAction", "action": "add", "name": "wt-box",
            "address": self.wt_remote.url, "machinePassword": "pw-wt",
        })
        # The add of "box" may still echo a ``machines`` reply: wait for the one listing wt-box.
        for _ in range(20):
            added = await self._until("machines")
            if any(m["name"] == "wt-box" for m in added["machines"]):
                break
        else:
            raise AssertionError("wt-box was never listed")
        self.assertEqual(added["error"], "")

    def _worktrees(self) -> list[Path]:
        """The task worktrees of the test repository."""
        return sorted((self.repo / ".kiss-worktrees").glob("kiss_wt-*"))

    def _isolate_edit_and_finish(self, request: dict[str, Any]) -> dict[str, Any]:
        """Isolate the repository, change a file in its worktree, finish without merging."""
        done = sum(1 for m in request.get("messages", []) if m.get("role") == "tool")
        if not request.get("tools"):  # a task classifier's call, not the agent loop's
            return finish_response("classified")
        if done == 0:
            return tool_call_response("create_worktree", {"path": str(self.repo)})
        if done == 1:
            (worktree,) = self._worktrees()
            return tool_call_response(
                "Write", {"file_path": str(worktree / "new.txt"), "content": "machine\n"},
            )
        return finish_response("left unmerged")

    async def test_discard_acts_on_the_machine_that_owns_the_worktree(self) -> None:
        tab = "tab-bound-worktree"
        await self._submit(tab, "Edit the repository", "wt-box")
        await asyncio.wait_for(self._until("worktree_done", tab), 120)
        self.assertEqual(len(self._worktrees()), 1, "the machine's task left its worktree")

        await self._send({"type": "worktreeAction", "action": "discard", "tabId": tab})
        result = await asyncio.wait_for(self._until("worktree_result", tab), 60)
        self.assertTrue(result["success"], result)
        self.assertEqual(self._worktrees(), [], "the machine discarded its worktree")
