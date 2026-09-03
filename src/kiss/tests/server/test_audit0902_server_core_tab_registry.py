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

from kiss.server.tab_registry import OpenTabOutcome, TabRegistry


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
