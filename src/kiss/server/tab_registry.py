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

INVARIANT: a chat id is bound to AT MOST ONE tab.  Binding a chat to
a tab atomically displaces (removes) any other tab bound to the same
chat — the newest bind wins — and loading / merging skips duplicate
chat bindings.  Because every client mirrors this registry verbatim,
the invariant guarantees that no client ever shows two open tabs for
the same chat.

Only top-level chat tabs live here.  Sub-agent tabs are derived state
(recreated on every client from ``openSubagentTab`` broadcasts and
session replays) and content tabs are client-local stand-ins for the
VS Code editor, so neither is registered.

The registry persists to ``KISS_HOME/tabs.json`` with atomic writes
(unique sibling temp file + ``os.replace``, via
:func:`kiss.core.utils.atomic_write_text`) so tabs survive daemon
restarts and a reader never sees a torn document.  The file has
exactly ONE owner: a registry loads it once and publishes its complete
in-memory list on every save, so a second live registry on the same
file would overwrite the first one's tabs with a stale snapshot.
Daemon startup guarantees one canonical daemon per ``KISS_HOME`` (UDS
socket liveness check), and a server embedded in another process that
shares the KISS home — the channel launcher's private-UDS daemon —
owns a private registry file instead
(``VSCodeServer.use_private_tab_registry``).  A thread lock therefore
suffices for the registry's in-memory readers and writers; the unique
temp name is belt-and-braces for a sibling instance (a test next to a
live daemon) that the lock cannot see.
"""

from __future__ import annotations

import enum
import json
import logging
import threading
from pathlib import Path
from typing import Any

from kiss.core.utils import atomic_write_text, is_root_dir

logger = logging.getLogger(__name__)


class OpenTabOutcome(enum.Enum):
    """Atomic result of :meth:`TabRegistry.open_tab`.

    The three cases are decided in ONE locked section so callers never
    need an unlocked follow-up probe (``has_tab``) to tell "already
    registered" apart from "registry full" — a concurrent ``closeTab``
    between two such calls used to turn a benign re-announce of an
    existing tab into a spurious "Tab limit reached" rejection.

    Truth-value compatibility: only :attr:`OPENED` is truthy, matching
    the method's previous boolean contract (``True`` exactly when a
    new tab was appended), so existing boolean callers keep working.
    """

    OPENED = "opened"
    """A new tab was appended (and the registry persisted)."""

    EXISTS = "exists"
    """The tab id is already registered; nothing changed."""

    FULL = "full"
    """The registry is at its hard cap; the tab was refused."""

    def __bool__(self) -> bool:
        return self is OpenTabOutcome.OPENED

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


def _clean_work_dir(value: Any) -> str:
    """Return *value* as a work dir: stripped, ``""`` for roots.

    A filesystem root (``/``, ``C:\\`` — see
    :func:`kiss.core.utils.is_root_dir`) is never a legitimate tab
    work dir; one can only have been stamped by a client whose cwd
    degenerated to the root (a no-folder Dock-launched VS Code
    window).  Mapping it to ``""`` both refuses to adopt new roots
    and HEALS entries already persisted in ``tabs.json`` on the next
    load, so an old poisoned tab stops rooting its tasks and
    ``@``-mention file scans at the whole disk.
    """
    out = _clean_str(value)
    return "" if is_root_dir(out) else out


def _sanitize_entries(
    entries: list[Any], *, keep_task_id: bool,
) -> list[dict[str, str]]:
    """Return the registry entries worth adopting from raw *entries*.

    The ONE sanitising pass behind both :meth:`TabRegistry._load` (the
    on-disk ``tabs.json``) and :meth:`TabRegistry.merge_if_empty` (a
    legacy client's ``restoredTabs``): non-dict items, blank or
    repeated tab ids and later duplicates of an already-bound chat are
    dropped, every field is stripped, the title is capped and — like
    every other creation path — defaults to ``"new chat"``, and at
    most :data:`_MAX_TABS` entries survive.

    Args:
        entries: Raw entry dicts (anything else is skipped).
        keep_task_id: Adopt each entry's ``taskId`` pin.  True for the
            persisted registry; False for legacy client tabs, which
            never carried one.

    Returns:
        The sanitised entries, in input order.
    """
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    seen_chats: set[str] = set()
    for entry in entries[:_MAX_TABS]:
        if not isinstance(entry, dict):
            continue
        tab_id = _clean_str(entry.get("tabId"))
        if not tab_id or tab_id in seen:
            continue
        chat_id = _clean_str(entry.get("chatId"))
        if chat_id:
            # One tab per chat: drop later duplicates.
            if chat_id in seen_chats:
                continue
            seen_chats.add(chat_id)
        seen.add(tab_id)
        out.append({
            "tabId": tab_id,
            "chatId": chat_id,
            "title": (
                _clean_str(entry.get("title"), _MAX_TITLE_CHARS)
                or "new chat"
            ),
            "workDir": _clean_work_dir(entry.get("workDir")),
            "scopeWorkDir": _clean_work_dir(entry.get("scopeWorkDir")),
            "taskId": (
                _clean_str(entry.get("taskId")) if keep_task_id else ""
            ),
        })
    return out


class TabRegistry:
    """Ordered, persistent registry of the shared chat tabs.

    Every mutator reports whether it changed the registry (the caller
    then broadcasts a fresh ``tabs_state`` snapshot) and persists the
    new state before returning.  :meth:`update_tab` additionally
    reports the tabs it displaced to enforce the one-tab-per-chat
    invariant.
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
        self._persist_failed = False
        self._heal_pending = False
        # In-memory (never persisted) per-tab publication tokens: every
        # ``open_tab`` / ``update_tab`` that touches a tab stamps it
        # with a fresh monotonically increasing generation.  A caller
        # that wants to undo ITS OWN publication captures the token and
        # removes through :meth:`close_tab_if_generation`, which no-ops
        # when anyone else has republished the tab since — see the
        # ``_cmd_run`` compensating close (gpt-5.6-sol review 2,
        # introduced bug 1).  Entries loaded from disk carry no token
        # (generation 0), which a live capture can never equal.
        self._generations: dict[str, int] = {}
        self._generation_counter = 0
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
        self._tabs = _sanitize_entries(entries, keep_task_id=True)
        # Sanitizing may change the entries — a dropped duplicate, a
        # clipped title, or a HEALED root work dir.  Do NOT write here:
        # construction alone must never touch the file, because a
        # non-owner also constructs on the canonical path (the embedded
        # launcher's server builds its ``VSCodeServer`` before swapping
        # in its private registry), and a load-time write from it would
        # race the owning daemon.  Remember the pending heal instead;
        # the owner persists it on its first mutation or, failing that,
        # at shutdown (:meth:`flush`).
        self._heal_pending = self._tabs != entries

    def _save_locked(self) -> None:
        """Atomically persist the tab list (caller holds the lock).

        A failed write never breaks live mirroring: the in-memory
        state stays authoritative, the failure is logged loudly ONCE
        per failure streak, and because every save writes the FULL
        state, the next successful mutation (or :meth:`flush`) heals
        the file.
        """
        try:
            # The shared unique-temp writer: a fixed ``tabs.json.tmp``
            # let a sibling registry instance (another daemon, a test
            # next to a live daemon) truncate the temp inode this one
            # was about to rename into place, publishing an empty file.
            atomic_write_text(
                self._path, json.dumps({"tabs": self._tabs}, indent=1),
            )
        except OSError:
            if not self._persist_failed:
                logger.error(
                    "Could not persist tab registry %s; serving the "
                    "in-memory tabs and retrying on the next mutation "
                    "and at shutdown",
                    self._path,
                    exc_info=True,
                )
            self._persist_failed = True
            return
        if self._persist_failed:
            logger.warning(
                "Tab registry %s persisted again after earlier failures",
                self._path,
            )
        self._persist_failed = False
        # Every save writes the FULL in-memory state, so any pending
        # load-time heal is on disk now.
        self._heal_pending = False

    def flush(self) -> None:
        """Re-persist the registry if it has unsaved state.

        Called at daemon shutdown so that (a) tabs mutated while the
        disk was unwritable are not silently lost across a restart and
        (b) entries healed at load time (a sanitized root work dir)
        reach the disk even when no tab was ever mutated.  A no-op
        when the file already matches the in-memory state.
        """
        with self._lock:
            if self._persist_failed or self._heal_pending:
                self._save_locked()

    def _bump_generation_locked(self, tab_id: str) -> int:
        """Stamp *tab_id* with a fresh publication token (lock held).

        Args:
            tab_id: The shared tab identifier being (re)published.

        Returns:
            The tab's new generation token.
        """
        self._generation_counter += 1
        self._generations[tab_id] = self._generation_counter
        return self._generation_counter

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

    def bound_tabs(self) -> list[tuple[str, str, str]]:
        """Return ``(tabId, chatId, taskId)`` for every chat-bound tab.

        ``taskId`` is the specific historical task the tab was resumed
        to (``""`` when the tab tracks the chat's latest task) — the
        ready replay path passes it through so a reconnect never
        silently switches a tab to a different task.
        """
        with self._lock:
            return [
                (
                    entry["tabId"],
                    entry["chatId"],
                    entry.get("taskId", ""),
                )
                for entry in self._tabs
                if entry["chatId"]
            ]

    def has_tab(self, tab_id: str) -> bool:
        """Return whether *tab_id* is registered."""
        with self._lock:
            return self._find_locked(_clean_str(tab_id)) is not None

    def generation(self, tab_id: str) -> int:
        """Return *tab_id*'s current publication token.

        Args:
            tab_id: The shared tab identifier.

        Returns:
            The token of the tab's latest publication, or ``0`` when
            the tab is unknown or was never published by this process
            (an entry loaded from disk).
        """
        with self._lock:
            return self._generations.get(_clean_str(tab_id), 0)

    def open_tab(
        self, tab_id: str, title: str = "", work_dir: str = "",
    ) -> OpenTabOutcome:
        """Register a new tab (no-op when it already exists).

        Args:
            tab_id: The shared tab identifier.
            title: Initial tab title.
            work_dir: The tab's pinned working directory, if any.

        Returns:
            The :class:`OpenTabOutcome`, decided atomically under the
            registry lock: :attr:`~OpenTabOutcome.OPENED` (truthy)
            when a new tab was appended, :attr:`~OpenTabOutcome.EXISTS`
            when the id is already registered (also returned for a
            blank id — nothing to open, nothing to reject), and
            :attr:`~OpenTabOutcome.FULL` when the hard cap refused it.
        """
        tab_id = _clean_str(tab_id)
        if not tab_id:
            return OpenTabOutcome.EXISTS
        with self._lock:
            if self._find_locked(tab_id) is not None:
                return OpenTabOutcome.EXISTS
            if len(self._tabs) >= _MAX_TABS:
                logger.warning(
                    "Tab registry full (%d); refusing to open %r",
                    _MAX_TABS, tab_id,
                )
                return OpenTabOutcome.FULL
            self._tabs.append({
                "tabId": tab_id,
                "chatId": "",
                "title": _clean_str(title, _MAX_TITLE_CHARS) or "new chat",
                "workDir": _clean_work_dir(work_dir),
                "scopeWorkDir": "",
                "taskId": "",
            })
            self._bump_generation_locked(tab_id)
            self._save_locked()
            return OpenTabOutcome.OPENED

    def _removal_token_locked(self) -> int:
        """Draw a removal token from the shared generation clock.

        Called with the registry lock held, in the same critical
        section as the removal it stamps.  The token orders the
        removal against every publication: any LATER ``update_tab`` /
        ``open_tab`` publication of the same tab id receives a larger
        generation, so :meth:`republished_since` can tell the removal's
        out-of-lock cleanup tail whether the tab has been legitimately
        reopened since (gpt-5.6-sol review 3, missed wiring 1).

        Returns:
            The removal's clock token (always positive).
        """
        self._generation_counter += 1
        return self._generation_counter

    def republished_since(self, tab_id: str, token: int) -> bool:
        """Return whether *tab_id* was published again after *token*.

        Args:
            tab_id: The shared tab identifier.
            token: A removal token from :meth:`close_tab` or an
                ``update_tab`` displacement, or a publication
                generation.

        Returns:
            ``True`` when a publication newer than *token* stamped the
            tab — its current owner is that later publisher, so any
            cleanup keyed to *token* must not touch it.
        """
        with self._lock:
            return self._generations.get(_clean_str(tab_id), 0) > token

    def close_tab(self, tab_id: str) -> int:
        """Remove a tab.

        Args:
            tab_id: The shared tab identifier.

        Returns:
            The removal's clock token (positive, truthy) when the tab
            existed and was removed, ``0`` (falsy) otherwise.  The
            caller hands the token to its out-of-lock cleanup tail,
            which uses :meth:`republished_since` to stand down when a
            later publication has legitimately reopened the tab.
        """
        with self._lock:
            entry = self._find_locked(_clean_str(tab_id))
            if entry is None:
                return 0
            self._tabs.remove(entry)
            self._generations.pop(_clean_str(tab_id), None)
            token = self._removal_token_locked()
            self._save_locked()
            return token

    def close_tab_if_generation(self, tab_id: str, generation: int) -> bool:
        """Remove a tab only if nobody republished it since *generation*.

        The conditional twin of :meth:`close_tab` for compensating
        closes: a caller undoing its OWN ``update_tab`` publication
        passes the token that publication returned; when a later
        legitimate publication (e.g. a ``resumeSession`` reopen)
        stamped the tab with a newer token, the stale undo no-ops
        instead of deleting the newer owner's tab (gpt-5.6-sol
        review 2, introduced bug 1).  The token comparison and the
        removal are one atomic operation under the registry lock.

        Args:
            tab_id: The shared tab identifier.
            generation: The token returned by the publication being
                undone.

        Returns:
            ``True`` when the tab still carried *generation* and was
            removed.
        """
        tab_id = _clean_str(tab_id)
        if generation <= 0:
            # A loaded/adopted row has no stamped generation (reads as
            # 0); a caller could otherwise capture that 0 and "match"
            # it here.  A real publication token is always positive.
            return False
        with self._lock:
            entry = self._find_locked(tab_id)
            if entry is None or self._generations.get(tab_id, 0) != generation:
                return False
            self._tabs.remove(entry)
            self._generations.pop(tab_id, None)
            self._save_locked()
            return True

    def update_tab(
        self,
        tab_id: str,
        *,
        chat_id: str | None = None,
        title: str | None = None,
        work_dir: str | None = None,
        scope_work_dir: str | None = None,
        task_id: str | None = None,
        create: bool = False,
    ) -> tuple[bool, list[tuple[str, int]], int]:
        """Update (or create) a tab's binding, title, work dir or task.

        Binding a non-empty *chat_id* atomically DISPLACES (removes)
        any other tab bound to the same chat — the one-tab-per-chat
        invariant — and reports the displaced tabs so the caller can
        release their server-side per-tab state.  Each displaced tab
        is paired with a removal token, and the caller's own
        publication generation is returned from the SAME critical
        section that stamped it (a separate post-hoc ``generation()``
        lookup could observe a later publisher's stamp — gpt-5.6-sol
        review 3, introduced bug 1).

        Args:
            tab_id: The shared tab identifier.
            chat_id: New chat binding (``None`` keeps the current one).
            title: New title (``None``/empty keeps the current one).
            work_dir: New working directory (``None``/empty keeps it).
            scope_work_dir: The directory that scopes the tab to a
                client workspace, distinct from *work_dir* (the tab's
                execution/display directory): a standalone API
                dispatch executes in a channel/cron scratch directory
                but must appear in the CALLING workspace's tab bar, so
                its scope is pinned to that workspace while *work_dir*
                stays the scratch directory.  ``None``/empty keeps the
                current value; clients fall back to *work_dir* when it
                is empty, preserving the pre-scope behaviour.
            task_id: The specific historical task the tab shows.
                ``None`` keeps the current value; ``""`` clears it (the
                tab tracks the chat's latest task again).
            create: Register the tab first when it is unknown.

        Returns:
            ``(changed, displaced, generation)``: whether the registry
            changed, ``(tab_id, removal_token)`` pairs for the tabs
            removed because *chat_id* was bound to them (see
            :meth:`republished_since`), and this publication's own
            generation token (``0`` when nothing was published).
        """
        tab_id = _clean_str(tab_id)
        if not tab_id:
            return False, [], 0
        with self._lock:
            entry = self._find_locked(tab_id)
            changed = False
            displaced: list[tuple[str, int]] = []
            if entry is None:
                if not create or len(self._tabs) >= _MAX_TABS:
                    return False, [], 0
                entry = {
                    "tabId": tab_id, "chatId": "",
                    "title": "new chat", "workDir": "",
                    "scopeWorkDir": "",
                    "taskId": "",
                }
                self._tabs.append(entry)
                changed = True
            # Every publication — even one that changes no field —
            # re-stamps the tab: a reopen (``resumeSession``) may
            # write values identical to the current ones, yet it must
            # still invalidate any older caller's pending
            # ``close_tab_if_generation`` undo.
            generation = self._bump_generation_locked(tab_id)
            if chat_id is not None:
                chat_id = _clean_str(chat_id)
                if chat_id:
                    for other in [
                        t for t in self._tabs
                        if t is not entry and t["chatId"] == chat_id
                    ]:
                        self._tabs.remove(other)
                        self._generations.pop(other["tabId"], None)
                        displaced.append(
                            (other["tabId"], self._removal_token_locked())
                        )
                        changed = True
                if entry["chatId"] != chat_id:
                    entry["chatId"] = chat_id
                    changed = True
            if task_id is not None:
                task_id = _clean_str(task_id)
                if entry.get("taskId", "") != task_id:
                    entry["taskId"] = task_id
                    changed = True
            new_title = _clean_str(title, _MAX_TITLE_CHARS)
            if new_title and entry["title"] != new_title:
                entry["title"] = new_title
                changed = True
            new_wd = _clean_work_dir(work_dir)
            if new_wd and entry["workDir"] != new_wd:
                entry["workDir"] = new_wd
                changed = True
            new_scope = _clean_work_dir(scope_work_dir)
            if new_scope and entry.get("scopeWorkDir", "") != new_scope:
                entry["scopeWorkDir"] = new_scope
                changed = True
            if changed:
                self._save_locked()
            return changed, displaced, generation

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
            adopted = _sanitize_entries(
                list(entries), keep_task_id=False,
            )
            if not adopted:
                return False
            self._tabs = adopted
            self._save_locked()
            return True
