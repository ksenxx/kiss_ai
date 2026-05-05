"""Autocomplete mixin for the VS Code server.

Implements the ghost-text autocomplete pipeline and the file-path
autocomplete feature.  Split out of ``server.py`` for organisation.
"""

from __future__ import annotations

import queue
import re
import threading
from typing import TYPE_CHECKING

from kiss.agents.sorcar.persistence import _load_file_usage, _prefix_match_task
from kiss.agents.vscode.helpers import (
    clip_autocomplete_suggestion,
    rank_file_suggestions,
)

if TYPE_CHECKING:
    from kiss.agents.vscode.printer import VSCodePrinter


class _AutocompleteMixin:
    """Ghost-text + file-path autocomplete methods."""

    if TYPE_CHECKING:
        printer: VSCodePrinter
        work_dir: str
        _state_lock: threading.Lock
        _complete_queue: queue.Queue[tuple[str, int, str, str]] | None
        _complete_worker: threading.Thread | None
        _complete_seq_latest: int
        _file_cache: list[str] | None

    def _complete_from_active_file(
        self, query: str, snapshot_file: str = "", snapshot_content: str = ""
    ) -> str:
        """Complete the trailing token of *query* using identifiers from the active file.

        Extracts single-word identifiers and dot-chained identifiers
        (e.g. ``self.method``, ``os.path.join``) from the active editor
        buffer (or falls back to reading from disk). Matches the trailing
        token of the query — which may contain dots — against all
        candidates via case-sensitive prefix matching.

        Args:
            query: The full query string from the chat input.
            snapshot_file: Atomically-captured active file path.
            snapshot_content: Atomically-captured active file content.

        Returns:
            The remaining suffix to append, or empty string if no match.
        """
        content = snapshot_content
        if not content:
            active_path = snapshot_file
            if not active_path:
                return ""
            try:
                with open(active_path) as f:
                    content = f.read(50000)
            except OSError:
                return ""

        if query and not (query[-1].isalnum() or query[-1] == "_" or query[-1] == "."):
            return ""
        m = re.search(r"([\w][\w.]*)$", query)
        if not m:
            return ""
        partial = m.group(1)
        if len(partial) < 2:
            return ""
        words = set(re.findall(r"\b[A-Za-z_]\w{2,}\b", content))
        chains = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", content))
        candidates = words | chains

        best = ""
        for candidate in candidates:
            if candidate.startswith(partial) and len(candidate) > len(partial):
                suffix = candidate[len(partial):]
                if len(suffix) > len(best):
                    best = suffix
        return best

    def _complete_worker_loop(self) -> None:
        """Persistent worker that drains the complete queue."""
        assert self._complete_queue is not None
        q = self._complete_queue
        while True:
            item = q.get()
            while not q.empty():
                try:
                    item = q.get_nowait()
                except queue.Empty:  # pragma: no cover — race guard
                    break
            query, seq, snapshot_file, snapshot_content = item
            self._complete(query, seq, snapshot_file, snapshot_content)

    def _complete(
        self,
        query: str,
        seq: int = -1,
        snapshot_file: str = "",
        snapshot_content: str = "",
    ) -> None:
        """Ghost text autocomplete via fast local prefix matching.

        Args:
            query: Raw query text from the chat input.
            seq: Sequence number for this request. If a newer request has
                been issued (``seq`` no longer matches the counter), this
                call exits early to avoid broadcasting stale results.
            snapshot_file: Atomically-captured active file path.
            snapshot_content: Atomically-captured active file content.
        """
        if seq >= 0:
            with self._state_lock:
                if seq != self._complete_seq_latest:
                    return
        if not query or len(query) < 2:
            self.printer.broadcast({"type": "ghost", "suggestion": "", "query": query})
            return

        match = _prefix_match_task(query)
        if match:
            fast = match[len(query):]
        else:
            fast = self._complete_from_active_file(query, snapshot_file, snapshot_content)
        fast = clip_autocomplete_suggestion(query, fast)
        self.printer.broadcast({"type": "ghost", "suggestion": fast, "query": query})

    def _ensure_complete_worker(self) -> None:
        """Lazily start the autocomplete worker thread on first use.

        Task processes never receive ``complete`` commands, so the
        worker thread and queue are only created for service processes
        that actually need autocomplete.
        """
        if self._complete_worker is not None:
            return
        self._complete_queue = queue.Queue()
        self._complete_worker = threading.Thread(
            target=self._complete_worker_loop, daemon=True
        )
        self._complete_worker.start()

    def _refresh_file_cache(self, then_emit_for_prefix: str | None = None) -> None:
        """Refresh the file cache from disk in a background thread.

        When ``then_emit_for_prefix`` is set, broadcasts a ``files``
        event ranked for that prefix once the scan finishes.  This lets
        callers (``_get_files``) kick off a non-blocking refresh and
        still deliver suggestions to the UI.
        """
        from kiss.agents.vscode.diff_merge import _scan_files

        def _do_refresh() -> None:
            result = _scan_files(self.work_dir)
            with self._state_lock:
                self._file_cache = result
            if then_emit_for_prefix is not None:
                usage = _load_file_usage()
                ranked = rank_file_suggestions(
                    result, then_emit_for_prefix, usage,
                )
                self.printer.broadcast({"type": "files", "files": ranked})

        threading.Thread(target=_do_refresh, daemon=True).start()

    def _get_files(self, prefix: str) -> None:
        """Send file list for autocomplete with usage-based sorting.

        H9 — must not block the message-handling thread.  When the cache
        is empty, kick off a background refresh and respond immediately
        with an empty ``loading=true`` list; the same scan then emits a
        second ``files`` event with the populated list once it finishes,
        so the frontend gets results without the caller blocking.
        """
        with self._state_lock:
            cache = self._file_cache
        if cache is None:
            self._refresh_file_cache(then_emit_for_prefix=prefix)
            self.printer.broadcast(
                {"type": "files", "files": [], "loading": True},
            )
            return
        usage = _load_file_usage()
        ranked = rank_file_suggestions(cache, prefix, usage)
        self.printer.broadcast({"type": "files", "files": ranked})
