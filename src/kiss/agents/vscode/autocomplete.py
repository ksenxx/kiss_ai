# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Autocomplete mixin for the VS Code server.

Implements the ghost-text autocomplete pipeline and the file-path
autocomplete feature.  Split out of ``server.py`` for organisation.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from typing import TYPE_CHECKING, Any

from kiss.agents.sorcar.persistence import (
    _load_chat_context_text,
    _load_file_usage,
    _prefix_match_tasks,
)
from kiss.agents.vscode.helpers import (
    clip_autocomplete_suggestion,
    rank_file_suggestions,
)
from kiss.agents.vscode.tricks import (
    current_sentence_partial,
    prefix_match_tricks,
)

# Maximum number of fast-complete dropdown items emitted to the
# webview per ``complete`` request.  Mirrors the @-mention file
# picker's 20-item cap so the dropdown stays scrollable without UI
# tuning differences between the two pickers.
_COMPLETIONS_LIMIT = 20

if TYPE_CHECKING:
    from kiss.agents.vscode.json_printer import JsonPrinter

logger = logging.getLogger(__name__)


def _ghost_suffix(query: str, completions: list[dict[str, str]]) -> str:
    """Return the ghost-text suffix for the top completion.

    Completions carry raw suggestion text (no head splicing), so the
    ghost overlay's suffix depends on which prefix of ``query`` the
    top item actually completes:

    * ``task`` — the full query (``_prefix_match_tasks`` guarantees
      the task string starts with ``query``).
    * ``trick`` — the current sentence's leading partial as computed
      by :func:`current_sentence_partial`.
    * ``identifier`` — the trailing word/dot-chain token of ``query``.

    Returns an empty string when the top completion does not start
    with the expected prefix (e.g. an identifier candidate that
    doesn't start with the trailing token, which can only happen when
    the suggestion source disagrees with the ranker).
    """
    if not completions:
        return ""
    top = completions[0]
    text = top["text"]
    kind = top["type"]
    if kind == "task":
        prefix = query
    elif kind == "trick":
        prefix = current_sentence_partial(query)
    elif kind == "identifier":
        m = re.search(r"([\w][\w.]*)$", query)
        prefix = m.group(1) if m else ""
    else:
        prefix = query
    if not prefix or not text.startswith(prefix):
        return ""
    return text[len(prefix):]


class _AutocompleteMixin:
    """Ghost-text + file-path autocomplete methods."""

    if TYPE_CHECKING:
        printer: JsonPrinter
        work_dir: str
        _state_lock: threading.RLock
        _complete_queue: (
            queue.Queue[tuple[str, int, str, str, str, str]] | None
        )
        _complete_worker: threading.Thread | None
        _complete_seq_latest: dict[str, int]
        _file_cache: dict[str, list[str]]

    def _complete_from_active_file(
        self,
        query: str,
        snapshot_file: str = "",
        snapshot_content: str = "",
        chat_id: str = "",
    ) -> str:
        """Complete the trailing token of *query* using identifiers from the active file.

        Extracts single-word identifiers and dot-chained identifiers
        (e.g. ``self.method``, ``os.path.join``) from the active editor
        buffer (or falls back to reading from disk).  When *chat_id* is
        provided, also harvests identifiers from the ``task`` and
        ``result`` text of every prior task in that chat session, so
        completions can suggest words the user has already used earlier
        in the same conversation.  Matches the trailing token of the
        query — which may contain dots — against all candidates via
        case-sensitive prefix matching.

        Args:
            query: The full query string from the chat input.
            snapshot_file: Atomically-captured active file path.
            snapshot_content: Atomically-captured active file content.
            chat_id: Current chat session id; when non-empty, identifiers
                from previous tasks in this chat are added to the
                candidate pool.

        Returns:
            The remaining suffix to append, or empty string if no match.
        """
        content = snapshot_content
        if not content:
            active_path = snapshot_file
            if active_path:
                try:
                    with open(active_path) as f:
                        content = f.read(50000)
                except (OSError, UnicodeDecodeError):
                    # UnicodeDecodeError (a binary / non-UTF-8 active
                    # file) must not escape: it would kill the single
                    # autocomplete worker thread permanently.
                    content = ""

        if query and not (query[-1].isalnum() or query[-1] == "_" or query[-1] == "."):
            return ""
        m = re.search(r"([\w][\w.]*)$", query)
        if not m:
            return ""
        partial = m.group(1)
        if len(partial) < 2:
            return ""

        chat_text = _load_chat_context_text(chat_id)

        if not content and not chat_text:
            return ""

        combined = content + ("\n" + chat_text if chat_text else "")
        words = set(re.findall(r"\b[A-Za-z_]\w{2,}\b", combined))
        chains = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", combined))
        candidates = words | chains

        best = ""
        for candidate in candidates:
            if candidate.startswith(partial) and len(candidate) > len(partial):
                suffix = candidate[len(partial):]
                if len(suffix) > len(best):
                    best = suffix
        return best

    def _active_file_identifier_matches(
        self,
        query: str,
        snapshot_file: str = "",
        snapshot_content: str = "",
        chat_id: str = "",
    ) -> list[str]:
        """Return every identifier from the active file/chat context.

        Multi-result counterpart of :meth:`_complete_from_active_file`:
        scans the active editor buffer (or the on-disk fallback) plus
        the chat context for single-word identifiers and dot-chained
        identifiers that prefix-match the trailing token of *query*.
        The result is sorted longest-first so the dropdown shows the
        most informative completion at the top.

        Returns the *full identifier strings* (not the suffix) so the
        caller can build the textarea-replacement text by combining
        the leading non-token portion of the query with each
        identifier.
        """
        content = snapshot_content
        if not content:
            active_path = snapshot_file
            if active_path:
                try:
                    with open(active_path) as f:
                        content = f.read(50000)
                except (OSError, UnicodeDecodeError):
                    content = ""
        if query and not (
            query[-1].isalnum() or query[-1] == "_" or query[-1] == "."
        ):
            return []
        m = re.search(r"([\w][\w.]*)$", query)
        if not m:
            return []
        partial = m.group(1)
        if len(partial) < 2:
            return []
        chat_text = _load_chat_context_text(chat_id)
        if not content and not chat_text:
            return []
        combined = content + ("\n" + chat_text if chat_text else "")
        words = set(re.findall(r"\b[A-Za-z_]\w{2,}\b", combined))
        chains = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", combined))
        matches = [
            c for c in (words | chains)
            if c.startswith(partial) and len(c) > len(partial)
        ]
        # Longest-first so the dropdown's auto-selected first item is
        # the most informative completion.  Tie-breaker is
        # alphabetical for stable ordering across runs.
        matches.sort(key=lambda c: (-len(c), c))
        return matches

    def _complete_worker_loop(self) -> None:
        """Persistent worker that drains the complete queue.

        Every queued item is handed to :meth:`_complete`, which drops
        stale requests via the per-connection sequence check before
        doing any real work.  The loop must NOT blindly collapse the
        queue to its newest item: requests from *different*
        connections (VS Code windows) are interleaved on this one
        queue, and discarding everything but the newest would let one
        window's keystroke swallow another window's still-fresh
        request.
        """
        assert self._complete_queue is not None
        q = self._complete_queue
        while True:
            query, seq, snapshot_file, snapshot_content, chat_id, conn_id = (
                q.get()
            )
            try:
                self._complete(
                    query, seq, snapshot_file, snapshot_content, chat_id,
                    conn_id,
                )
            except Exception:
                # This worker is a lazily-started singleton with no
                # restart path (``_ensure_complete_worker`` sees the
                # dead thread object as "already started"), so one
                # poisoned request must never kill ghost-text
                # autocomplete for the daemon's remaining lifetime.
                logger.debug("autocomplete request failed", exc_info=True)

    def _complete(
        self,
        query: str,
        seq: int = -1,
        snapshot_file: str = "",
        snapshot_content: str = "",
        chat_id: str = "",
        conn_id: str = "",
    ) -> None:
        """Ghost text autocomplete via fast local prefix matching.

        Args:
            query: Raw query text from the chat input.
            seq: Sequence number for this request. If a newer request
                has been issued *on the same connection* (``seq`` no
                longer matches that connection's latest counter), this
                call exits early to avoid broadcasting stale results.
            snapshot_file: Atomically-captured active file path.
            snapshot_content: Atomically-captured active file content.
            chat_id: Current chat session id; passed through to the
                active-file completion so previous tasks in the same
                chat contribute identifier candidates.
            conn_id: Connection id the request arrived on (``""`` for
                direct callers).  Staleness is judged per connection so
                concurrent typing in another VS Code window never
                cancels this request.
        """
        if seq >= 0:
            with self._state_lock:
                if seq != self._complete_seq_latest.get(conn_id, -1):
                    return
        if not query or len(query) < 2:
            self._emit_ghost("", query, conn_id)
            self._emit_completions([], query, conn_id)
            return

        completions = self._complete_many(
            query, snapshot_file, snapshot_content, chat_id,
        )
        # Inline ghost text: derive the suffix from the top completion
        # so the legacy overlay keeps working for users who prefer to
        # accept with Tab without opening the dropdown.  Completions
        # are raw suggestions — a history task starts with ``query``
        # in full, a trick starts with the current sentence partial,
        # and an identifier starts with the trailing token — so the
        # ghost suffix is derived per source.  ``clip_autocomplete_
        # suggestion`` then normalises the cursor-to-ghost gap.
        fast = _ghost_suffix(query, completions)
        fast = clip_autocomplete_suggestion(query, fast)
        self._emit_ghost(fast, query, conn_id)
        self._emit_completions(completions, query, conn_id)

    def _complete_many(
        self,
        query: str,
        snapshot_file: str = "",
        snapshot_content: str = "",
        chat_id: str = "",
    ) -> list[dict[str, str]]:
        """Gather every fast-complete candidate for *query*.

        Returns up to :data:`_COMPLETIONS_LIMIT` ranked candidates, each
        a ``{"type": <kind>, "text": <full replacement>}`` dict.  The
        ``text`` field is always the *complete replacement* the chat
        textarea should hold when the candidate is accepted — never
        just a suffix — so the frontend's accept path is a single
        assignment regardless of source.

        Sources, in dropdown order:

        * ``task`` — full task strings from ``_prefix_match_tasks``
          (most recent first).
        * ``trick`` — INJECTIONS.md trick bodies from
          ``prefix_match_tricks``, emitted verbatim.
        * ``identifier`` — single-word / dot-chained identifiers
          harvested from the active editor and chat context by
          :meth:`_active_file_identifier_matches`, emitted verbatim.

        Emitted texts are the raw suggestion — never a head-spliced
        whole-input replacement.  Ghost-text and accept behaviour work
        naturally when the query is exactly the piece being completed
        (a single sentence for tricks, the trailing token for
        identifiers).

        Duplicates (same ``text``) are removed while preserving the
        earlier source's ordering so e.g. a history task that
        happens to equal a trick body never appears twice.
        """
        out: list[dict[str, str]] = []
        seen: set[str] = set()

        def _add(kind: str, text: str) -> None:
            if not text or text == query:
                return
            if text in seen:
                return
            seen.add(text)
            out.append({"type": kind, "text": text})

        for task in _prefix_match_tasks(query, limit=_COMPLETIONS_LIMIT):
            _add("task", task)
            if len(out) >= _COMPLETIONS_LIMIT:
                return out

        for trick in prefix_match_tricks(query):
            _add("trick", trick)
            if len(out) >= _COMPLETIONS_LIMIT:
                return out

        for ident in self._active_file_identifier_matches(
            query, snapshot_file, snapshot_content, chat_id,
        ):
            _add("identifier", ident)
            if len(out) >= _COMPLETIONS_LIMIT:
                return out
        return out

    def _emit_completions(
        self,
        completions: list[dict[str, str]],
        query: str,
        conn_id: str,
    ) -> None:
        """Emit one ``completions`` event for the fast-complete picker.

        Mirrors :meth:`_emit_ghost`'s connection scoping (the
        suggestion is delivered only to the typing VS Code window
        when ``conn_id`` is non-empty) and echoes ``query`` so the
        webview can drop stale replies for an input the user has
        since edited.

        Args:
            completions: List of ``{"type", "text"}`` items.
            query: The query string this list answers.
            conn_id: Requesting connection id (``""`` for direct callers).
        """
        event: dict[str, Any] = {
            "type": "completions",
            "completions": completions,
            "query": query,
        }
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

    def _emit_ghost(self, suggestion: str, query: str, conn_id: str) -> None:
        """Emit one ``ghost`` autocomplete event.

        Stamped with the requesting connection's ``conn_id`` (when
        non-empty) so the suggestion is delivered only to the VS Code
        window that is typing — never to a sibling window whose input
        happens to hold the same text.

        Args:
            suggestion: The ghost-text suffix to suggest (may be ``""``).
            query: The query string this suggestion answers.
            conn_id: Requesting connection id (``""`` for direct callers).
        """
        event: dict[str, Any] = {
            "type": "ghost", "suggestion": suggestion, "query": query,
        }
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

    def _ensure_complete_worker(self) -> None:
        """Lazily start the autocomplete worker thread on first use.

        Task processes never receive ``complete`` commands, so the
        worker thread and queue are only created for service processes
        that actually need autocomplete.

        The check-then-init is performed under ``_state_lock`` so two
        concurrent callers cannot both observe ``None`` and spawn
        duplicate worker threads (which would leak an orphan thread
        consuming from an unreferenced queue).
        """
        with self._state_lock:
            if self._complete_worker is not None:
                return
            self._complete_queue = queue.Queue()
            self._complete_worker = threading.Thread(
                target=self._complete_worker_loop, daemon=True
            )
            self._complete_worker.start()

    def _resolve_work_dir(self, work_dir: str) -> str:
        """Return *work_dir* when non-empty, else the daemon-wide work_dir.

        Used by the ``@``-mention file picker so each chat tab can scan
        its own working directory: the frontend stamps the active tab's
        ``workDir`` on ``getFiles``/``recordFileUsage`` commands, and an
        empty value falls back to the daemon-wide ``self.work_dir``
        captured from ``KISS_WORKDIR`` or the most-recent ``setWorkDir``.
        """
        return work_dir or self.work_dir

    def _refresh_file_cache(
        self,
        then_emit_for_prefix: str | None = None,
        work_dir: str = "",
        conn_id: str = "",
    ) -> None:
        """Refresh the file cache for *work_dir* in a background thread.

        When ``then_emit_for_prefix`` is set, broadcasts a ``files``
        event ranked for that prefix once the scan finishes.  This lets
        callers (``_get_files``) kick off a non-blocking refresh and
        still deliver suggestions to the UI.

        ``work_dir`` selects which directory to scan; an empty value
        defaults to ``self.work_dir`` so existing callers that omit it
        keep the daemon-wide behaviour.  Each work_dir has its own
        entry in ``self._file_cache`` (keyed by the resolved path) so
        tabs with different working directories never share file lists.

        Race protection: when invoked from ``_get_files`` (i.e.
        ``then_emit_for_prefix is not None``, meaning the cache was
        empty at the call site), this preserves the original double-
        check pattern from commit ``e49d867c`` — the scan result is
        only published if the cache is still empty when the scan
        finishes.  This prevents a slow scan from clobbering a fresher
        result published by a concurrent refresh thread.  Explicit
        refresh requests (``then_emit_for_prefix is None``) overwrite
        unconditionally, matching their callers' intent (the user just
        asked for a refresh).
        """
        from kiss.agents.vscode.diff_merge import _scan_files

        wd = self._resolve_work_dir(work_dir)
        only_if_empty = then_emit_for_prefix is not None

        def _do_refresh() -> None:
            result = _scan_files(wd)
            with self._state_lock:
                existing = self._file_cache.get(wd)
                if only_if_empty and existing is not None:
                    # A concurrent writer published a fresher value
                    # while we were scanning — emit theirs, not ours.
                    result = existing
                else:
                    self._file_cache[wd] = result
            if then_emit_for_prefix is not None:
                usage = _load_file_usage()
                ranked = rank_file_suggestions(
                    result, then_emit_for_prefix, usage,
                )
                self._emit_files(ranked, conn_id, prefix=then_emit_for_prefix)

        threading.Thread(target=_do_refresh, daemon=True).start()

    def _refresh_files_after_task(self, work_dir: str = "") -> None:
        """Refresh the ``@``-mention file cache after an agent task ends.

        The cache is populated lazily on the first ``getFiles`` for a
        ``work_dir`` and is otherwise only refreshed on a daemon-wide
        ``setWorkDir`` or an explicit refresh request.  When an agent
        creates or deletes files during its turn those changes never
        reach the cache, so the next ``@``-mention serves stale
        suggestions: brand-new files (e.g. the test file the agent
        just authored) are invisible and deleted files linger.

        This hook is invoked by :meth:`_TaskRunnerMixin._run_task_inner`
        at the tail of every task's cleanup ``finally``.  It rescans
        *work_dir* in a background thread (no caller blocking) and:

        * only updates the cache when the *set* of files actually
          changed — pure modifications never alter the picker's list
          so the rescan is a no-op; and
        * broadcasts a fresh ``files`` event (with no ``connId`` so
          every connected client receives it) only when the list
          changed, so any open ``@``-mention picker UI refreshes
          without further user action.

        When *work_dir* has no cache entry (no ``@``-mention picker
        has ever opened there) the hook is a no-op: there is nothing
        to keep fresh, and the next ``getFiles`` will scan from
        scratch anyway.  This avoids paying a directory-scan cost
        for tabs whose picker was never used.
        """
        from kiss.agents.vscode.diff_merge import _scan_files

        wd = self._resolve_work_dir(work_dir)
        with self._state_lock:
            cached = self._file_cache.get(wd)
        if cached is None:
            return
        cached_set = set(cached)

        def _do_refresh() -> None:
            result = _scan_files(wd)
            if set(result) == cached_set:
                # Only modifications (or no change at all) — nothing
                # to publish.  The cached list is still accurate so
                # no overwrite is needed either.
                return
            with self._state_lock:
                self._file_cache[wd] = result
            usage = _load_file_usage()
            ranked = rank_file_suggestions(result, "", usage)
            self._emit_files(ranked, conn_id="", prefix="")

        threading.Thread(target=_do_refresh, daemon=True).start()

    def _emit_files(
        self,
        ranked: list[dict[str, Any]],
        conn_id: str,
        loading: bool = False,
        prefix: str = "",
    ) -> None:
        """Emit one ``files`` event for the ``@``-mention picker.

        Stamped with the requesting connection's ``conn_id`` (when
        non-empty) so the file list pops the picker only in the VS
        Code window that typed ``@`` — never in a sibling window.

        Every event also echoes the ``prefix`` it was ranked for —
        the picker's analogue of the ``ghost`` event's echoed
        ``query``.  The populated reply for a cache miss arrives
        asynchronously after a background directory scan, so the
        frontend needs the prefix to drop late replies for an
        ``@``-mention the user has since edited or abandoned (a
        prefix-less reply used to re-open the picker over the input
        and swallow the next Enter keystroke).

        Args:
            ranked: Ranked file suggestion dicts to send.
            conn_id: Requesting connection id (``""`` for direct callers).
            loading: True for the immediate empty reply sent while a
                background directory scan is still running.
            prefix: The ``@``-mention query this reply was ranked for.
        """
        event: dict[str, Any] = {
            "type": "files", "files": ranked, "prefix": prefix,
        }
        if loading:
            event["loading"] = True
        if conn_id:
            event["connId"] = conn_id
        self.printer.broadcast(event)

    def _get_files(
        self, prefix: str, work_dir: str = "", conn_id: str = "",
    ) -> None:
        """Send file list for the ``@``-mention picker, scoped to *work_dir*.

        ``work_dir`` selects the directory the picker is rooted at.  An
        empty value falls back to ``self.work_dir``: the chat webview
        stamps the active tab's ``workDir`` on the ``getFiles`` command
        so tabs with different working directories see their own files,
        independent of the daemon-wide default.

        H9 — must not block the message-handling thread.  When the
        cache for the resolved work_dir is empty, kick off a background
        refresh and respond immediately with an empty ``loading=true``
        list; the same scan then emits a second ``files`` event with
        the populated list once it finishes, so the frontend gets
        results without the caller blocking.
        """
        wd = self._resolve_work_dir(work_dir)
        with self._state_lock:
            cache = self._file_cache.get(wd)
        if cache is None:
            self._refresh_file_cache(
                then_emit_for_prefix=prefix, work_dir=wd, conn_id=conn_id,
            )
            self._emit_files([], conn_id, loading=True, prefix=prefix)
            return
        usage = _load_file_usage()
        ranked = rank_file_suggestions(cache, prefix, usage)
        self._emit_files(ranked, conn_id, prefix=prefix)
