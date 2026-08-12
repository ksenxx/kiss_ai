# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The server-canonical shared tab registry.

Every connected client (VS Code chat webviews and remote web apps)
mirrors the same set of chat tabs.  The daemon owns the canonical,
ordered tab list; clients never persist a tab set of their own — they
reconcile against the full ``tabs_state`` snapshot the daemon
broadcasts after every mutation.

Only top-level chat tabs live here.  Sub-agent tabs are derived state
(recreated on every client from ``openSubagentTab`` broadcasts and
session replays) and content tabs are client-local stand-ins for the
VS Code editor, so neither is registered.

The registry persists to ``KISS_HOME/tabs.json`` with atomic writes
(temp file + ``os.replace``) so tabs survive daemon restarts.  A
single daemon owns the file (daemon startup already guarantees one
daemon per ``KISS_HOME`` via the UDS socket liveness check), so a
thread lock suffices for the registry's readers and writers.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_TITLE_CHARS = 200
"""Cap stored titles: clients clip for display, the wire stays small."""

_MAX_TABS = 512
"""Hard cap on registered tabs so a buggy client cannot grow the
registry (and every ``tabs_state`` broadcast) without bound."""


def _clean_str(value: Any, max_len: int = 0) -> str:
    """Return *value* as a stripped string (``""`` for non-strings)."""
    if not isinstance(value, str):
        return ""
    out = value.strip()
    if max_len and len(out) > max_len:
        out = out[:max_len]
    return out


class TabRegistry:
    """Ordered, persistent registry of the shared chat tabs.

    Every mutator returns ``True`` when it changed the registry (the
    caller then broadcasts a fresh ``tabs_state`` snapshot) and
    persists the new state before returning.
    """

    def __init__(self, path: Path) -> None:
        """Load the registry from *path* (empty when missing/corrupt).

        Args:
            path: The JSON file backing the registry
                (``KISS_HOME/tabs.json``).
        """
        self._path = path
        self._lock = threading.Lock()
        self._tabs: list[dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        """Read the persisted tab list, tolerating a missing file."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.warning(
                "Unreadable tab registry %s; starting empty",
                self._path,
                exc_info=True,
            )
            return
        entries = raw.get("tabs") if isinstance(raw, dict) else None
        if not isinstance(entries, list):
            return
        seen: set[str] = set()
        for entry in entries[:_MAX_TABS]:
            if not isinstance(entry, dict):
                continue
            tab_id = _clean_str(entry.get("tabId"))
            if not tab_id or tab_id in seen:
                continue
            seen.add(tab_id)
            self._tabs.append({
                "tabId": tab_id,
                "chatId": _clean_str(entry.get("chatId")),
                "title": _clean_str(entry.get("title"), _MAX_TITLE_CHARS),
                "workDir": _clean_str(entry.get("workDir")),
            })

    def _save_locked(self) -> None:
        """Atomically persist the tab list (caller holds the lock)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(
                json.dumps({"tabs": self._tabs}, indent=1),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except OSError:
            logger.warning(
                "Could not persist tab registry %s", self._path,
                exc_info=True,
            )

    def _find_locked(self, tab_id: str) -> dict[str, str] | None:
        """Return the entry for *tab_id* (caller holds the lock)."""
        for entry in self._tabs:
            if entry["tabId"] == tab_id:
                return entry
        return None

    def snapshot(self) -> list[dict[str, str]]:
        """Return a deep copy of the ordered tab entries."""
        with self._lock:
            return [dict(entry) for entry in self._tabs]

    def bindings(self) -> dict[str, str]:
        """Return ``{tabId: chatId}`` for every chat-bound tab."""
        with self._lock:
            return {
                entry["tabId"]: entry["chatId"]
                for entry in self._tabs
                if entry["chatId"]
            }

    def open_tab(
        self, tab_id: str, title: str = "", work_dir: str = "",
    ) -> bool:
        """Register a new tab (no-op when it already exists).

        Args:
            tab_id: The shared tab identifier.
            title: Initial tab title.
            work_dir: The tab's pinned working directory, if any.

        Returns:
            ``True`` when a new tab was appended.
        """
        tab_id = _clean_str(tab_id)
        if not tab_id:
            return False
        with self._lock:
            if self._find_locked(tab_id) is not None:
                return False
            if len(self._tabs) >= _MAX_TABS:
                logger.warning(
                    "Tab registry full (%d); refusing to open %r",
                    _MAX_TABS, tab_id,
                )
                return False
            self._tabs.append({
                "tabId": tab_id,
                "chatId": "",
                "title": _clean_str(title, _MAX_TITLE_CHARS) or "new chat",
                "workDir": _clean_str(work_dir),
            })
            self._save_locked()
            return True

    def close_tab(self, tab_id: str) -> bool:
        """Remove a tab.

        Args:
            tab_id: The shared tab identifier.

        Returns:
            ``True`` when the tab existed and was removed.
        """
        with self._lock:
            entry = self._find_locked(tab_id)
            if entry is None:
                return False
            self._tabs.remove(entry)
            self._save_locked()
            return True

    def update_tab(
        self,
        tab_id: str,
        *,
        chat_id: str | None = None,
        title: str | None = None,
        work_dir: str | None = None,
        create: bool = False,
    ) -> bool:
        """Update (or create) a tab's binding, title or work dir.

        Args:
            tab_id: The shared tab identifier.
            chat_id: New chat binding (``None`` keeps the current one).
            title: New title (``None``/empty keeps the current one).
            work_dir: New working directory (``None``/empty keeps it).
            create: Register the tab first when it is unknown.

        Returns:
            ``True`` when the registry changed.
        """
        tab_id = _clean_str(tab_id)
        if not tab_id:
            return False
        with self._lock:
            entry = self._find_locked(tab_id)
            changed = False
            if entry is None:
                if not create or len(self._tabs) >= _MAX_TABS:
                    return False
                entry = {
                    "tabId": tab_id, "chatId": "",
                    "title": "new chat", "workDir": "",
                }
                self._tabs.append(entry)
                changed = True
            if chat_id is not None:
                chat_id = _clean_str(chat_id)
                if entry["chatId"] != chat_id:
                    entry["chatId"] = chat_id
                    changed = True
            new_title = _clean_str(title, _MAX_TITLE_CHARS)
            if new_title and entry["title"] != new_title:
                entry["title"] = new_title
                changed = True
            new_wd = _clean_str(work_dir)
            if new_wd and entry["workDir"] != new_wd:
                entry["workDir"] = new_wd
                changed = True
            if changed:
                self._save_locked()
            return changed

    def merge_if_empty(self, entries: list[dict[str, str]]) -> bool:
        """Adopt a legacy client's persisted tabs into an EMPTY registry.

        One-time migration path: clients that predate the shared
        registry persisted their tab set locally and announce it via
        ``ready.restoredTabs``.  The first such client seeds the
        registry; once the registry is non-empty it is canonical and
        later announcements are ignored.

        Args:
            entries: Sanitized ``restoredTabs`` entries
                (``tabId``/``chatId`` plus optional ``title`` /
                ``workDir``).

        Returns:
            ``True`` when the registry adopted the entries.
        """
        with self._lock:
            if self._tabs:
                return False
            seen: set[str] = set()
            for entry in entries[:_MAX_TABS]:
                tab_id = _clean_str(entry.get("tabId"))
                if not tab_id or tab_id in seen:
                    continue
                seen.add(tab_id)
                self._tabs.append({
                    "tabId": tab_id,
                    "chatId": _clean_str(entry.get("chatId")),
                    "title": (
                        _clean_str(entry.get("title"), _MAX_TITLE_CHARS)
                        or "new chat"
                    ),
                    "workDir": _clean_str(entry.get("workDir")),
                })
            if not self._tabs:
                return False
            self._save_locked()
            return True
