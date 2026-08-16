# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the one-tab-per-chat invariant.

INVARIANT: on every client, for a given chat id, at most ONE tab is
open.  Tabs are mirrored verbatim from the daemon's canonical tab
registry (``tabs_state``), so the invariant holds on every client if
and only if the registry never contains two tabs bound to the same
chat id.

These tests drive a real :class:`RemoteAccessServer` over real
``wss://`` connections (no mocks) and pin the invariant against every
path that binds a chat to a tab:

* ``resumeSession`` for a chat already shown in another tab must
  displace (close) the old tab — the newest bind wins;
* the same displacement must happen when the resumed chat has no
  recorded events yet (the no-result replay path);
* a ``run`` carrying an explicit ``chatId`` bound to another tab must
  displace that tab too;
* a displaced tab's server-side per-tab state must be dropped;
* a legacy client's ``restoredTabs`` carrying two tabs for one chat
  must be adopted as a single tab;
* a hand-edited / legacy ``tabs.json`` with duplicate chat bindings
  must be deduplicated on load;
* concurrent binds of one chat to many tabs must leave exactly one
  bound tab (registry-level atomicity).
"""

from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.server.tab_registry import TabRegistry
from kiss.tests.server.test_web_server_tab_mirroring import (
    TabMirroringBase,
    _tab_entry,
)


def _chat_tab_ids(snapshot: dict[str, Any], chat_id: str) -> list[str]:
    """Return the ids of every snapshot tab bound to *chat_id*."""
    return [
        t.get("tabId", "")
        for t in snapshot.get("tabs", [])
        if t.get("chatId") == chat_id
    ]


def _assert_unique_chats(
    case: unittest.TestCase, snapshot: dict[str, Any],
) -> None:
    """Fail when two snapshot tabs share a non-empty chat id."""
    seen: dict[str, str] = {}
    for t in snapshot.get("tabs", []):
        chat = t.get("chatId", "")
        if not chat:
            continue
        tab = t.get("tabId", "")
        case.assertNotIn(
            chat, seen,
            f"INVARIANT VIOLATED: chat {chat!r} is bound to two open "
            f"tabs ({seen.get(chat)!r} and {tab!r}) — every client "
            "will show two tabs for the same chat",
        )
        seen[chat] = tab


class TestOneTabPerChat(TabMirroringBase):
    """The registry must never bind one chat to two tabs."""

    async def _bind(
        self, ws: Any, tab_id: str, chat_id: str,
    ) -> dict[str, Any]:
        """Open *tab_id*, resume *chat_id* into it, return the
        ``tabs_state`` snapshot showing the binding."""
        await self._send(ws, {
            "type": "openTab", "tabId": tab_id, "title": "new chat",
        })
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws, present={tab_id}),
        )
        await self._send(ws, {
            "type": "resumeSession", "chatId": chat_id, "tabId": tab_id,
        })

        def _bound(ev: dict[str, Any]) -> bool:
            try:
                return _tab_entry(ev, tab_id).get("chatId") == chat_id
            except AssertionError:
                return False

        snap = await self._wait_for_event(ws, "tabs_state", pred=_bound)
        self.assertIsNotNone(
            snap, f"tab {tab_id!r} never got bound to chat {chat_id!r}",
        )
        assert snap is not None
        return snap

    async def test_resume_same_chat_in_second_tab_displaces_first(
        self,
    ) -> None:
        """Resuming a chat already shown in tab-1 into tab-2 must
        close tab-1 everywhere — never show the chat twice."""
        _task_id, chat_id = self._seed_chat(
            "Seeded task", [{"type": "prompt", "text": "seeded prompt"}],
        )
        ws_a = await self._connect_ok()
        ws_b = await self._connect_ok()
        await self._ready(ws_a)
        await self._ready(ws_b)

        await self._bind(ws_a, "tab-1", chat_id)
        await self._bind(ws_a, "tab-2", chat_id)

        snap = await self._wait_for_snapshot_with(
            ws_b, present={"tab-2"}, absent={"tab-1"},
        )
        self.assertIsNotNone(
            snap,
            "the old tab bound to the resumed chat was never closed — "
            "clients now show the same chat in two tabs",
        )
        assert snap is not None
        self.assertEqual(_chat_tab_ids(snap, chat_id), ["tab-2"])
        _assert_unique_chats(self, snap)

    async def test_resume_unrecorded_chat_twice_displaces_first(
        self,
    ) -> None:
        """The no-result replay path (chat without recorded events)
        must also displace the previous tab bound to the chat."""
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        chat_id = "chat-without-events"

        await self._bind(ws_a, "tab-n1", chat_id)
        snap = await self._bind(ws_a, "tab-n2", chat_id)

        self.assertNotIn(
            "tab-n1", [t.get("tabId") for t in snap.get("tabs", [])],
            "the no-result replay path bound one chat to two tabs",
        )
        self.assertEqual(_chat_tab_ids(snap, chat_id), ["tab-n2"])
        _assert_unique_chats(self, snap)

    async def test_run_with_explicit_chat_id_displaces_bound_tab(
        self,
    ) -> None:
        """A ``run`` continuing a chat from a fresh tab must displace
        the tab currently bound to that chat."""
        _task_id, chat_id = self._seed_chat(
            "Run task", [{"type": "prompt", "text": "run prompt"}],
        )
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._bind(ws_a, "tab-r1", chat_id)

        await self._send(ws_a, {
            "type": "run",
            "prompt": "Continue the chat elsewhere",
            "chatId": chat_id,
            "model": "definitely-not-a-real-model",
            "workDir": self.tmpdir,
            "tabId": "tab-r2",
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
        })
        snap = await self._wait_for_snapshot_with(
            ws_a, present={"tab-r2"}, absent={"tab-r1"},
        )
        self.assertIsNotNone(
            snap,
            "a run with an explicit chatId left the chat bound to two "
            "tabs",
        )
        assert snap is not None
        self.assertEqual(_chat_tab_ids(snap, chat_id), ["tab-r2"])
        _assert_unique_chats(self, snap)

    async def test_displaced_tab_server_state_is_dropped(self) -> None:
        """Displacement must clean up the old tab's per-tab server
        state exactly like an explicit ``closeTab`` would."""
        _task_id, chat_id = self._seed_chat(
            "State task", [{"type": "prompt", "text": "state prompt"}],
        )
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._bind(ws_a, "tab-s1", chat_id)
        snap = await self._bind(ws_a, "tab-s2", chat_id)
        self.assertNotIn(
            "tab-s1", [t.get("tabId") for t in snap.get("tabs", [])],
        )
        assert self.server is not None
        vs = self.server._vscode_server
        self.assertNotIn(
            "tab-s1", vs._tab_chat_views,
            "the displaced tab's chat-view mapping leaked",
        )
        self.assertEqual(vs._tab_chat_views.get("tab-s2"), chat_id)

    async def test_legacy_restored_tabs_with_duplicate_chat(self) -> None:
        """Two legacy tabs for one chat must be adopted as ONE tab."""
        _task_id, chat_id = self._seed_chat(
            "Legacy dup", [{"type": "prompt", "text": "legacy"}],
        )
        ws_a = await self._connect_ok()
        await self._send(ws_a, {
            "type": "ready", "tabId": "dup-1",
            "restoredTabs": [
                {"tabId": "dup-1", "chatId": chat_id, "title": "first"},
                {"tabId": "dup-2", "chatId": chat_id, "title": "second"},
            ],
        })
        snap = await self._wait_for_snapshot_with(
            ws_a, present={"dup-1"}, absent={"dup-2"},
        )
        self.assertIsNotNone(
            snap,
            "legacy restoredTabs seeded two tabs bound to one chat",
        )
        assert snap is not None
        self.assertEqual(_chat_tab_ids(snap, chat_id), ["dup-1"])
        _assert_unique_chats(self, snap)

    async def test_registry_on_disk_duplicates_are_deduped(self) -> None:
        """A legacy/hand-edited ``tabs.json`` with two tabs bound to
        the same chat must load as a single bound tab."""
        await self._stop_server()
        kiss_dir = Path(self.tmpdir) / ".kiss"
        (kiss_dir / "tabs.json").write_text(
            json.dumps({"tabs": [
                {"tabId": "disk-1", "chatId": "chat-x", "title": "one",
                 "workDir": "", "taskId": ""},
                {"tabId": "disk-2", "chatId": "chat-x", "title": "two",
                 "workDir": "", "taskId": ""},
                {"tabId": "disk-3", "chatId": "", "title": "unbound",
                 "workDir": "", "taskId": ""},
            ]}),
            encoding="utf-8",
        )
        await self._start_server()
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        snap = await self._wait_for_snapshot_with(
            ws_a, present={"disk-1", "disk-3"}, absent={"disk-2"},
        )
        self.assertIsNotNone(
            snap,
            "the registry loaded two tabs bound to the same chat "
            "from disk",
        )
        assert snap is not None
        self.assertEqual(_chat_tab_ids(snap, "chat-x"), ["disk-1"])
        _assert_unique_chats(self, snap)


class TestTabRegistryChatUniqueness(unittest.TestCase):
    """Registry-level invariant: at most one tab per chat id."""

    def setUp(self) -> None:
        import shutil
        import tempfile
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-chat-uniq-")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.path = Path(self.tmpdir) / "tabs.json"

    def _bound_tabs(self, reg: TabRegistry, chat_id: str) -> list[str]:
        return [
            e["tabId"] for e in reg.snapshot() if e["chatId"] == chat_id
        ]

    def test_update_tab_displaces_other_tab_bound_to_chat(self) -> None:
        reg = TabRegistry(self.path)
        reg.update_tab("t1", chat_id="c1", create=True)
        reg.update_tab("t2", chat_id="c1", create=True)
        self.assertEqual(self._bound_tabs(reg, "c1"), ["t2"])
        self.assertEqual([e["tabId"] for e in reg.snapshot()], ["t2"])

    def test_rebinding_same_tab_is_not_a_displacement(self) -> None:
        reg = TabRegistry(self.path)
        reg.update_tab("t1", chat_id="c1", create=True)
        reg.update_tab("t1", chat_id="c1", title="renamed")
        self.assertEqual(self._bound_tabs(reg, "c1"), ["t1"])

    def test_displacement_persists_to_disk(self) -> None:
        reg = TabRegistry(self.path)
        reg.update_tab("t1", chat_id="c1", create=True)
        reg.update_tab("t2", chat_id="c1", create=True)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        ids = [e["tabId"] for e in raw["tabs"]]
        self.assertEqual(ids, ["t2"])

    def test_load_keeps_first_tab_of_duplicate_chat(self) -> None:
        self.path.write_text(
            json.dumps({"tabs": [
                {"tabId": "a", "chatId": "c", "title": "a",
                 "workDir": "", "taskId": ""},
                {"tabId": "b", "chatId": "c", "title": "b",
                 "workDir": "", "taskId": ""},
                {"tabId": "d", "chatId": "", "title": "d",
                 "workDir": "", "taskId": ""},
                {"tabId": "e", "chatId": "", "title": "e",
                 "workDir": "", "taskId": ""},
            ]}),
            encoding="utf-8",
        )
        reg = TabRegistry(self.path)
        self.assertEqual(self._bound_tabs(reg, "c"), ["a"])
        # Unbound tabs are never deduped against each other.
        self.assertEqual(
            [e["tabId"] for e in reg.snapshot()], ["a", "d", "e"],
        )

    def test_merge_if_empty_keeps_first_tab_of_duplicate_chat(
        self,
    ) -> None:
        reg = TabRegistry(self.path)
        self.assertTrue(reg.merge_if_empty([
            {"tabId": "m1", "chatId": "c9", "title": "one"},
            {"tabId": "m2", "chatId": "c9", "title": "two"},
            {"tabId": "m3", "chatId": "", "title": "unbound"},
        ]))
        self.assertEqual(self._bound_tabs(reg, "c9"), ["m1"])
        self.assertEqual(
            [e["tabId"] for e in reg.snapshot()], ["m1", "m3"],
        )

    def test_concurrent_binds_leave_exactly_one_bound_tab(self) -> None:
        """Racing binds of one chat to many tabs must end with ONE."""
        reg = TabRegistry(self.path)
        start = threading.Barrier(8)

        def _bind(i: int) -> None:
            start.wait()
            reg.update_tab(f"race-{i}", chat_id="hot", create=True)

        threads = [
            threading.Thread(target=_bind, args=(i,)) for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        bound = self._bound_tabs(reg, "hot")
        self.assertEqual(
            len(bound), 1,
            f"INVARIANT VIOLATED after concurrent binds: {bound}",
        )


if __name__ == "__main__":
    unittest.main()
