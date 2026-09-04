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
import os
import queue
import re
import stat
import threading
from typing import TYPE_CHECKING, Any

from kiss.agents.sorcar.persistence import (
    _load_chat_context_text,
    _load_file_usage,
    _prefix_match_tasks,
)
from kiss.core.models.model_info import MODEL_INFO, get_available_models
from kiss.server.helpers import (
    SUGGESTION_LIMIT as _COMPLETIONS_LIMIT,
)
from kiss.server.helpers import (
    clip_autocomplete_suggestion,
    model_vendor,
    rank_file_suggestions,
)
from kiss.server.tricks import (
    current_sentence_partial,
    prefix_match_tricks,
)

if TYPE_CHECKING:
    from kiss.server.json_printer import JsonPrinter

logger = logging.getLogger(__name__)

_TRAILING_IDENT_RE = re.compile(r"([\w][\w.]*)\Z")

_ACTIVE_FILE_READ_CAP = 50000


def trailing_identifier(query: str) -> str:
    """Return the trailing identifier token of *query*, or ``""``.

    The token is the trailing word / dot-chain (``[\\w][\\w.]*``) of
    *query*; tokens shorter than 2 characters are rejected so a lone
    letter never triggers identifier completion.

    Args:
        query: The input text being completed.

    Returns:
        The trailing token, or ``""`` when *query* does not end in a
        completable identifier prefix.
    """
    m = _TRAILING_IDENT_RE.search(query)
    if not m or len(m.group(1)) < 2:
        return ""
    return m.group(1)


def read_active_file_head(path: str) -> str:
    """Return up to the first 50 000 characters of *path*.

    Unreadable or non-UTF-8 files yield ``""`` (best effort — the
    active file is only a suggestion source).

    ``path`` is client-supplied and this function runs on the single
    autocomplete worker shared by every connection, so it must never
    block: the file is opened with ``O_NONBLOCK`` (a plain ``open``
    of a writer-less FIFO blocks forever) and anything that is not a
    regular file — FIFOs, device nodes, sockets — is rejected via
    ``fstat`` before any read.  For regular files ``O_NONBLOCK`` is a
    no-op, so normal behaviour is unchanged.  Windows has no
    ``O_NONBLOCK`` (and no FIFO open-blocking hazard); the flag
    degrades to 0 there instead of raising ``AttributeError``.

    Args:
        path: Path of the active editor file.

    Returns:
        The capped file content, or ``""`` on any read failure or when
        *path* is not a regular file.
    """
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return ""
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return ""
        f = os.fdopen(fd, encoding="utf-8")
    except OSError:
        os.close(fd)
        return ""
    try:
        with f:
            return f.read(_ACTIVE_FILE_READ_CAP)
    except (OSError, UnicodeDecodeError):
        return ""


def identifier_prefix_matches(content: str, partial: str) -> list[str]:
    """Return identifiers in *content* that strictly extend *partial*.

    Harvests single-word identifiers and dot-chained identifiers
    (e.g. ``self.method``, ``os.path.join``) from *content* and keeps
    those that case-sensitively start with *partial* and are longer
    than it.  Used by the VS Code daemon's ghost-text completion.

    Args:
        content: Text to harvest identifiers from.
        partial: The typed identifier prefix (may contain dots).

    Returns:
        The matching full identifier strings, in no particular order.
    """
    words = set(re.findall(r"\b[A-Za-z_]\w{2,}\b", content))
    chains = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", content))
    return [
        c for c in (words | chains)
        if c.startswith(partial) and len(c) > len(partial)
    ]


def model_picker_sort_key(name: str) -> tuple[int, float]:
    """Return the model picker's canonical sort key for *name*.

    Models are grouped by vendor (see
    :func:`~kiss.server.helpers.model_vendor`) with the most
    expensive model first within each vendor.  Used by the daemon's ``getModels`` reply.

    Args:
        name: A model name present in ``MODEL_INFO``.

    Returns:
        ``(vendor_order, -total_price_per_1M)``.
    """
    info = MODEL_INFO[name]
    price = float(info.input_price_per_1M) + float(info.output_price_per_1M)
    return (model_vendor(name)[1], -price)


def ranked_function_calling_models() -> list[str]:
    """Return the available function-calling models in picker order.

    Candidates are the currently-runnable models (a provider credential
    is configured) that support function calling, sorted by
    :func:`model_picker_sort_key` — the single business rule behind
    the VS Code model picker.

    Returns:
        Model names, best-vendor-first, most expensive first per vendor.
    """
    names = [
        name for name in get_available_models()
        if name in MODEL_INFO and MODEL_INFO[name].is_function_calling_supported
    ]
    names.sort(key=model_picker_sort_key)
    return names


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
    else:
        m = _TRAILING_IDENT_RE.search(query)
        prefix = m.group(1) if m else ""
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
            queue.Queue[tuple[str, int, str, str | None, str, str, str]]
            | None
        )
        _complete_worker: threading.Thread | None
        _complete_seq_latest: dict[str, int]
        _file_cache: dict[str, list[str]]
        _files_latest_request: dict[str, object]

    def _active_file_identifier_matches(
        self,
        query: str,
        snapshot_file: str = "",
        snapshot_content: str | None = None,
        chat_id: str = "",
    ) -> list[str]:
        """Return every identifier from the active file/chat context.

        Extracts single-word identifiers and dot-chained identifiers
        (e.g. ``self.method``, ``os.path.join``) from the active
        editor buffer (or the on-disk fallback) plus the chat context
        — when *chat_id* is non-empty, the ``task``/``result`` text of
        every prior task in that chat session — and returns those that
        case-sensitively prefix-match the trailing token of *query*
        (which may contain dots).  The result is sorted longest-first
        so the dropdown shows the most informative completion at the
        top.

        Returns the *full identifier strings* (not the suffix) so the
        caller can build the textarea-replacement text by combining
        the leading non-token portion of the query with each
        identifier.
        """
        if snapshot_content is not None:
            # A live editor snapshot was supplied — honour it verbatim,
            # INCLUDING the empty string (an open but empty document).
            # Falling back to the on-disk file here would resurrect
            # identifiers the user has deleted from the unsaved buffer.
            content = snapshot_content
        elif snapshot_file:
            content = read_active_file_head(snapshot_file)
        else:
            content = ""
        partial = trailing_identifier(query)
        if not partial:
            return []
        chat_text = _load_chat_context_text(chat_id)
        if not content and not chat_text:
            return []
        combined = content + ("\n" + chat_text if chat_text else "")
        matches = identifier_prefix_matches(combined, partial)
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
            (
                query, seq, snapshot_file, snapshot_content, chat_id,
                conn_id, tab_id,
            ) = q.get()
            try:
                self._complete(
                    query, seq, snapshot_file, snapshot_content, chat_id,
                    conn_id, tab_id,
                )
            except Exception:
                logger.debug("autocomplete request failed", exc_info=True)

    def _complete(
        self,
        query: str,
        seq: int = -1,
        snapshot_file: str = "",
        snapshot_content: str | None = None,
        chat_id: str = "",
        conn_id: str = "",
        tab_id: str = "",
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
            tab_id: Chat tab the request came from (``""`` for direct
                callers).  Echoed on both replies so the webview can
                tell whether the suggestion still belongs to the
                conversation on screen.
        """
        if seq >= 0:
            with self._state_lock:
                if seq != self._complete_seq_latest.get(conn_id, -1):
                    return
        if not query or len(query) < 2:
            self._emit_ghost("", query, conn_id, tab_id)
            self._emit_completions([], query, conn_id, tab_id)
            return

        completions = self._complete_many(
            query, snapshot_file, snapshot_content, chat_id,
        )
        # Re-check freshness AFTER the (potentially slow) computation:
        # a newer request on the same connection may have advanced the
        # sequence while this one was doing file/DB work, and emitting
        # now would overwrite the newer request's result with a stale
        # one (the frontend only compares the echoed query text, which
        # can be identical across the two requests).
        fast = _ghost_suffix(query, completions)
        fast = clip_autocomplete_suggestion(query, fast)
        if seq >= 0:
            # Publish while still holding the lock: invalidators
            # (``setWorkDir``, disconnect) also take ``_state_lock``,
            # so once they return no stale event can be broadcast —
            # a check-then-emit outside the lock would leave a window
            # in which an already-checked stale result still escapes.
            # Nothing in the broadcast path acquires ``_state_lock``
            # (an RLock in any case), so this cannot deadlock.
            with self._state_lock:
                if seq != self._complete_seq_latest.get(conn_id, -1):
                    return
                self._emit_ghost(fast, query, conn_id, tab_id)
                self._emit_completions(completions, query, conn_id, tab_id)
            return
        self._emit_ghost(fast, query, conn_id, tab_id)
        self._emit_completions(completions, query, conn_id, tab_id)

    def _complete_many(
        self,
        query: str,
        snapshot_file: str = "",
        snapshot_content: str | None = None,
        chat_id: str = "",
    ) -> list[dict[str, str]]:
        """Gather every fast-complete candidate for *query*.

        Returns up to :data:`_COMPLETIONS_LIMIT` ranked candidates, each
        a ``{"type": <kind>, "text": <raw suggestion>}`` dict.  The
        ``text`` field is the raw suggestion for the piece of the
        query being completed (whole query for tasks, current sentence
        partial for tricks, trailing token for identifiers); the
        frontend's ``acceptCompletion`` splices it into the input by
        replacing the longest input suffix that prefixes ``text``, so
        text the user typed before that piece is preserved.

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
        tab_id: str = "",
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
            tab_id: Requesting chat tab (``""`` for direct callers).
        """
        event: dict[str, Any] = {
            "type": "completions",
            "completions": completions,
            "query": query,
        }
        if conn_id:
            event["connId"] = conn_id
        if tab_id:
            event["tabId"] = tab_id
        self.printer.broadcast(event)

    def _emit_ghost(
        self, suggestion: str, query: str, conn_id: str, tab_id: str = "",
    ) -> None:
        """Emit one ``ghost`` autocomplete event.

        Stamped with the requesting connection's ``conn_id`` (when
        non-empty) so the suggestion is delivered only to the VS Code
        window that is typing — never to a sibling window whose input
        happens to hold the same text.  ``tab_id`` narrows that one
        window down to the chat tab that typed: the webview keeps a
        single ghost overlay and a single picker, so a reply with no
        tab of its own would render over whichever conversation the
        user has since switched to.

        Args:
            suggestion: The ghost-text suffix to suggest (may be ``""``).
            query: The query string this suggestion answers.
            conn_id: Requesting connection id (``""`` for direct callers).
            tab_id: Requesting chat tab (``""`` for direct callers).
        """
        event: dict[str, Any] = {
            "type": "ghost", "suggestion": suggestion, "query": query,
        }
        if conn_id:
            event["connId"] = conn_id
        if tab_id:
            event["tabId"] = tab_id
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

    def _files_request_map(self) -> dict[str, object]:
        """Return the per-connection latest file-picker request map.

        Maps a connection id to a unique token identifying its most
        recent ``getFiles`` request.  Entries are SHORT-LIVED: each is
        removed as soon as the request it names has been answered (see
        :meth:`_get_files` and :meth:`_refresh_file_cache`), so the map
        never accumulates departed connections and needs no teardown
        wiring.  Created lazily on first use (the daemon's ``__init__``
        predates this guard).  Callers must hold ``_state_lock``.
        """
        reqs = getattr(self, "_files_latest_request", None)
        if reqs is None:
            reqs = {}
            self._files_latest_request = reqs
        return reqs

    def _refresh_file_cache(
        self,
        then_emit_for_prefix: str,
        work_dir: str = "",
        conn_id: str = "",
        tab_id: str = "",
    ) -> None:
        """Refresh the file cache for *work_dir* in a background thread.

        Broadcasts a ``files`` event ranked for ``then_emit_for_prefix``
        once the scan finishes.  This lets the only caller
        (``_get_files``) kick off a non-blocking refresh and still
        deliver suggestions to the UI.

        ``work_dir`` selects which directory to scan; an empty value
        defaults to ``self.work_dir`` so existing callers that omit it
        keep the daemon-wide behaviour.  Each work_dir has its own
        entry in ``self._file_cache`` (keyed by the resolved path) so
        tabs with different working directories never share file lists.

        Race protection (two layers):

        * Cache publication preserves the double-check pattern from
          commit ``e49d867c`` — the scan result is only published if
          the cache is still empty when the scan finishes, so a slow
          scan never clobbers a fresher result published by a
          concurrent refresh thread.
        * Emission is guarded by the per-connection request token
          captured at call time: if the connection has since issued a
          newer ``getFiles`` (e.g. the same typed prefix from a
          different tab/work_dir), the stale scan's reply is dropped
          instead of overwriting the newer picker contents.  When the
          reply IS still the latest, its token entry is removed — the
          request is answered, so the map stays empty for idle
          connections (no per-connection teardown needed).

        ``tab_id`` is carried through to the deferred ``files`` event
        so the late reply still names the chat tab that typed ``@``.
        """
        from kiss.server.diff_merge import _scan_files

        wd = self._resolve_work_dir(work_dir)
        with self._state_lock:
            request_token = self._files_request_map().get(conn_id)

        def _do_refresh() -> None:
            result = _scan_files(wd)
            with self._state_lock:
                existing = self._file_cache.get(wd)
                if existing is not None:
                    result = existing
                else:
                    self._file_cache[wd] = result
            # Rank OUTSIDE the lock (usage is a database read), then
            # re-verify the token and emit UNDER the lock, removing the
            # token only after the emission (audit0903 F5).  The old
            # order — token removed under the lock, emission after
            # releasing it — let a newer ``getFiles`` from the same
            # connection (same typed prefix, different work dir) emit
            # first and this superseded old-workspace list land LAST;
            # the frontend validates replies only by tab and prefix,
            # so the picker showed paths from the wrong repository.
            # Serialized against ``_get_files``'s token installation,
            # a stale reply is either suppressed here or provably
            # precedes the newer request's own emission.
            usage = _load_file_usage()
            ranked = rank_file_suggestions(
                result, then_emit_for_prefix, usage,
            )
            with self._state_lock:
                reqs = self._files_request_map()
                if reqs.get(conn_id) is not request_token:
                    return
                self._emit_files(
                    ranked,
                    conn_id,
                    prefix=then_emit_for_prefix,
                    tab_id=tab_id,
                )
                reqs.pop(conn_id, None)

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
        *work_dir* in a background thread (no caller blocking) and
        only updates the cache when the *set* of files actually
        changed — pure modifications never alter the picker's list so
        the rescan is a no-op.  The next ``getFiles`` (every picker
        keystroke issues one) serves the refreshed list.

        No ``files`` event is broadcast: an unsolicited reply stamped
        ``conn_id="", prefix=""`` would be accepted by every client
        whose picker shows a bare ``@`` — including windows whose tab
        roots at a DIFFERENT work_dir — overwriting their picker with
        files from this task's workspace (fixer-5 F5-03/R5-03).

        When *work_dir* has no cache entry (no ``@``-mention picker
        has ever opened there) the hook is a no-op: there is nothing
        to keep fresh, and the next ``getFiles`` will scan from
        scratch anyway.  This avoids paying a directory-scan cost
        for tabs whose picker was never used.
        """
        from kiss.server.diff_merge import _scan_files

        wd = self._resolve_work_dir(work_dir)
        with self._state_lock:
            cached = self._file_cache.get(wd)
        if cached is None:
            return
        cached_set = set(cached)

        def _do_refresh() -> None:
            result = _scan_files(wd)
            if set(result) == cached_set:
                return
            with self._state_lock:
                if self._file_cache.get(wd) is not cached:
                    return
                self._file_cache[wd] = result

        threading.Thread(target=_do_refresh, daemon=True).start()

    def _emit_files(
        self,
        ranked: list[dict[str, Any]],
        conn_id: str,
        loading: bool = False,
        prefix: str = "",
        tab_id: str = "",
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
            tab_id: Requesting chat tab (``""`` for direct callers).  A
                window can show several chat tabs over one connection,
                and they share a single picker element, so the tab is
                what actually identifies the owner of the reply.
        """
        event: dict[str, Any] = {
            "type": "files", "files": ranked, "prefix": prefix,
        }
        if loading:
            event["loading"] = True
        if conn_id:
            event["connId"] = conn_id
        if tab_id:
            event["tabId"] = tab_id
        self.printer.broadcast(event)

    def _get_files(
        self,
        prefix: str,
        work_dir: str = "",
        conn_id: str = "",
        tab_id: str = "",
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
        token: object = object()
        with self._state_lock:
            reqs = self._files_request_map()
            reqs[conn_id] = token
            cache = self._file_cache.get(wd)
        if cache is None:
            # The placeholder must be emitted BEFORE the scan is
            # started.  Both events belong to the same request and
            # carry the same prefix, so the client cannot tell a stale
            # one from a fresh one; starting the producer first lets a
            # quick scan's populated reply be overwritten by the empty
            # placeholder that follows it (R09-3).
            self._emit_files(
                [], conn_id, loading=True, prefix=prefix, tab_id=tab_id,
            )
            self._refresh_file_cache(
                then_emit_for_prefix=prefix,
                work_dir=wd,
                conn_id=conn_id,
                tab_id=tab_id,
            )
            return
        usage = _load_file_usage()
        ranked = rank_file_suggestions(cache, prefix, usage)
        self._emit_files(ranked, conn_id, prefix=prefix, tab_id=tab_id)
        with self._state_lock:
            # This request is answered; drop its token so the map only
            # ever holds connections with a scan still in flight.
            if reqs.get(conn_id) is token:
                del reqs[conn_id]
