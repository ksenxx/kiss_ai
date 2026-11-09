# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (server-core): tab registry sibling-path consistency.

Two places where :class:`TabRegistry`'s code paths had to agree but
did not:

* ``open_tab`` / ``has_tab`` / ``update_tab`` all pass the tab id
  through ``_clean_str`` (strip) before looking it up, but
  ``close_tab`` searched for the raw string — so a tab id a client
  sent with surrounding whitespace was registered under its stripped
  form and then could never be closed with the same string.
* ``_load`` (the on-disk registry) and ``merge_if_empty`` (a legacy
  client's ``restoredTabs``) each carried their own copy of the
  entry-sanitising loop, and the copies had drifted: every other
  creation path guarantees a non-empty title (``"new chat"``), while
  ``_load`` kept a blank one, so a hand-edited or older ``tabs.json``
  produced tabs whose ``tabs_state`` title was ``""``.
"""

from __future__ import annotations

import json
from pathlib import Path

from kiss.server.tab_registry import _MAX_TABS, OpenTabOutcome, TabRegistry


def test_close_tab_strips_like_open_tab(tmp_path: Path) -> None:
    reg = TabRegistry(tmp_path / "tabs.json")
    assert reg.open_tab(" tab-a ", "title") is OpenTabOutcome.OPENED
    assert reg.has_tab(" tab-a ")
    assert reg.close_tab(" tab-a "), (
        "BUG: close_tab looked up the raw id while open_tab stored the "
        "stripped one, so the tab could not be closed"
    )
    assert reg.snapshot() == []
    assert not reg.close_tab("   "), "a blank id closes nothing"


def test_loaded_entries_sanitised_like_merged_entries(tmp_path: Path) -> None:
    raw = {
        "tabs": [
            {"tabId": " t1 ", "chatId": "c1", "title": "", "workDir": "/w"},
            {"tabId": "t2", "chatId": "c1", "title": "dup chat"},
            {"tabId": "t1", "chatId": "", "title": "dup id"},
            {"tabId": "t3", "chatId": "", "title": "  Third  ",
             "scopeWorkDir": "/scope", "taskId": " 42 "},
            {"tabId": "", "title": "no id"},
            "junk",
        ],
    }
    path = tmp_path / "tabs.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = TabRegistry(path).snapshot()

    fresh = TabRegistry(tmp_path / "other" / "tabs.json")
    assert fresh.merge_if_empty(
        [e for e in raw["tabs"] if isinstance(e, dict)],
    )
    merged = fresh.snapshot()

    # Same input, same tabs — whichever door it came through (only the
    # task pin differs by design: legacy clients never carried one).
    assert [e["tabId"] for e in loaded] == ["t1", "t3"]
    assert [{**e, "taskId": ""} for e in loaded] == merged, (loaded, merged)
    assert loaded[0]["title"] == "new chat", (
        "BUG: _load kept a blank title while every other creation path "
        "defaults it to 'new chat'"
    )
    assert loaded[1] == {
        "tabId": "t3", "chatId": "", "title": "Third", "workDir": "",
        "scopeWorkDir": "/scope", "taskId": "42",
    }


def test_load_tolerates_non_list_and_persists_task_id(tmp_path: Path) -> None:
    path = tmp_path / "tabs.json"
    path.write_text(json.dumps({"tabs": "nope"}), encoding="utf-8")
    assert TabRegistry(path).snapshot() == []
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert TabRegistry(path).snapshot() == []
    reg = TabRegistry(path)
    reg.open_tab("t9", "nine")
    reg.update_tab("t9", chat_id="c9", task_id="task-9")
    reloaded = TabRegistry(path).snapshot()
    assert reloaded == [{
        "tabId": "t9", "chatId": "c9", "title": "nine", "workDir": "",
        "scopeWorkDir": "", "taskId": "task-9",
    }]
    # A legacy restoredTabs entry never carries a task pin.
    fresh = TabRegistry(tmp_path / "fresh" / "tabs.json")
    assert fresh.merge_if_empty([{"tabId": "t9", "taskId": "task-9"}])
    assert fresh.snapshot()[0]["taskId"] == ""
    assert not fresh.merge_if_empty([{"tabId": "t10"}]), "registry non-empty"
    assert not TabRegistry(tmp_path / "e" / "tabs.json").merge_if_empty(
        [{"tabId": "  "}],
    ), "nothing adoptable"


def test_an_emptied_registry_is_not_reseeded_by_a_stale_client(tmp_path: Path) -> None:
    """A tab closed on one surface does not come back with another surface's ``restoredTabs``.

    A registry that never had a file is seeded by the first client's
    tabs (recovery of a lost ``tabs.json``); one emptied by closes, or
    loaded from an empty ``tabs.json``, is canonical and a client's
    list of what it showed before it disconnected is stale.
    """
    path = tmp_path / "tabs.json"
    reg = TabRegistry(path)
    reg.open_tab("tab-T", "chat T")
    reg.update_tab("tab-T", chat_id="chat-C")
    stale = [{"tabId": "tab-T", "chatId": "chat-C", "title": "chat T"}]  # surface B's ready
    reg.close_tab("tab-T")  # surface A closes the last tab while B is disconnected
    assert not reg.merge_if_empty(stale), "a closed tab came back with a stale ready"
    assert reg.snapshot() == []
    reloaded = TabRegistry(path)  # a restart from the persisted empty registry
    assert not reloaded.merge_if_empty(stale), "the persisted empty registry was re-seeded"
    assert reloaded.bindings() == {}
    # Recovery stays: with no ``tabs.json`` at all the first client's tabs seed the registry.
    assert TabRegistry(tmp_path / "lost" / "tabs.json").merge_if_empty(stale)


def test_a_replayed_open_is_still_replayed_after_a_restart(tmp_path: Path) -> None:
    """An open re-sent to the daemon that replaced the one that took it does not reopen the tab.

    A page re-sends an open whose confirmation its connection lost
    before it reloads for the new daemon; the tab may have been closed
    by another surface meanwhile.  The decided tokens survive with the
    tabs on disk.
    """
    path = tmp_path / "tabs.json"
    reg = TabRegistry(path)
    reg.open_tab("keeper", "keeps the registry non-empty")
    assert reg.open_tab("tab-x", "x", ack="page-7:123") is OpenTabOutcome.OPENED
    reg.close_tab("tab-x")  # another surface closes it
    assert reg.open_tab("tab-x", "x", ack="page-7:123") is OpenTabOutcome.REPLAYED
    restarted = TabRegistry(path)
    assert restarted.open_tab("tab-x", "x", ack="page-7:123") is OpenTabOutcome.REPLAYED
    assert [e["tabId"] for e in restarted.snapshot()] == ["keeper"]
    assert restarted.open_tab("tab-x", "x", ack="page-7:124") is OpenTabOutcome.OPENED


def test_an_ack_decided_without_opening_survives_a_restart(tmp_path: Path) -> None:
    """An open decided EXISTS or FULL is as decided as one decided OPENED: persisted.

    Only the OPENED outcome used to save the registry, so an ack decided
    on an already-open tab (or a full registry) was lost at a clean
    shutdown, and the page's re-send of that very open to the next daemon
    brought back the tab another surface had closed meanwhile.
    """
    path = tmp_path / "tabs.json"
    reg = TabRegistry(path)
    assert reg.open_tab("same-tab") is OpenTabOutcome.OPENED
    assert reg.open_tab("same-tab", ack="page-1:1") is OpenTabOutcome.EXISTS
    reg.flush()  # a clean daemon shutdown
    assert "page-1:1" in json.loads(path.read_text(encoding="utf-8"))["acks"], (
        "BUG: an ack decided EXISTS was not persisted"
    )
    restarted = TabRegistry(path)
    assert restarted.close_tab("same-tab")  # another surface's user closes it
    assert restarted.open_tab("same-tab", ack="page-1:1") is OpenTabOutcome.REPLAYED
    assert restarted.snapshot() == [], "the closed tab came back with the re-sent open"

    full_path = tmp_path / "full" / "tabs.json"
    full_path.parent.mkdir()
    full_path.write_text(
        json.dumps({"tabs": [{"tabId": f"t{i}", "title": "t"} for i in range(_MAX_TABS)]}),
        encoding="utf-8",
    )
    full = TabRegistry(full_path)
    assert full.open_tab("one-more", ack="page-2:1") is OpenTabOutcome.FULL
    assert "page-2:1" in json.loads(full_path.read_text(encoding="utf-8"))["acks"], (
        "BUG: an ack decided FULL was not persisted"
    )
    restarted_full = TabRegistry(full_path)
    assert restarted_full.close_tab("t0")  # room again
    assert restarted_full.open_tab("one-more", ack="page-2:1") is OpenTabOutcome.REPLAYED
    assert not restarted_full.has_tab("one-more")
