# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Merge-review engine for the standalone web server.

In VS Code, the TypeScript ``MergeManager`` handles per-hunk
accept/reject by modifying files through the editor API.  The
standalone web server (``web_server.py``) has no editor, so this module
provides the equivalent functionality: :class:`_WebMergeState` tracks
hunk navigation/resolution for one review tab, and the module-level
helpers surgically revert hunks on disk (with CRLF, symlink, binary,
exec-bit, and deleted-file handling that mirrors git's semantics).

Extracted from ``web_server.py``; every name is re-imported there so
``web_server._WebMergeState`` etc. keep resolving to the same objects.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.server.diff_merge import _read_lines_preserved


class _WebMergeState:
    """Tracks merge review state for a single tab in the web server.

    In VS Code, the TypeScript ``MergeManager`` handles per-hunk
    accept/reject by modifying files through the editor API.  Since the
    standalone web server has no editor, this class provides equivalent
    functionality by tracking hunk resolution state and modifying files
    on disk directly.

    Args:
        merge_data: The ``data`` dict from a ``merge_data`` event,
            containing a ``files`` list with ``name``, ``base``,
            ``current``, and ``hunks`` entries.
    """

    def __init__(self, merge_data: dict[str, Any]) -> None:
        # Full ``data`` payload of the opening ``merge_data`` event,
        # kept so an in-flight review can be replayed verbatim to a
        # client that reconnects mid-review (browser reload).  The
        # ``hunks`` dicts inside are shared (not copied): reject
        # actions adjust their ``cs`` offsets in place, so a replay
        # always reflects the current on-disk line numbers.
        self.data: dict[str, Any] = merge_data
        self.files: list[dict[str, Any]] = merge_data.get("files", [])
        # The tab's repository (or worktree) directory, stamped by the
        # backend ``_start_merge_session``.  Echoed back on the
        # ``all-done`` ``mergeAction`` so the post-merge autocommit scan
        # runs against the tab's own repo rather than the daemon-wide
        # ``self.work_dir`` (which may be a different, non-git folder).
        self.work_dir: str = merge_data.get("work_dir", "")
        self._all_hunks: list[tuple[int, int]] = [
            (fi, hi)
            for fi, f in enumerate(self.files)
            for hi in range(len(f.get("hunks", [])))
        ]
        self._pos = 0
        # Maps (file_idx, hunk_idx) -> resolution status ("accepted" or
        # "rejected"); used so the browser can render accepted hunks
        # dimmed and rejected hunks struck-through.
        self._resolved: dict[tuple[int, int], str] = {}

    @property
    def total_hunks(self) -> int:
        """Total number of hunks across all files."""
        return len(self._all_hunks)

    @property
    def remaining(self) -> int:
        """Number of unresolved hunks."""
        return self.total_hunks - len(self._resolved)

    def current(self) -> tuple[int, int] | None:
        """Return (file_idx, hunk_idx) for the current position, or None.

        M11: returns ``None`` when every hunk has been resolved, so a
        post-``accept-all``/``reject-all`` ``current()`` consultation
        is unambiguously empty rather than silently pointing at the
        last (now-resolved) hunk.
        """
        if not self.remaining:
            return None
        if self._pos >= len(self._all_hunks):
            self._pos = len(self._all_hunks) - 1
        if self.is_resolved(*self._all_hunks[self._pos]):
            # A partial ``reject-all``/``reject-file`` failure resolves
            # whole files without ever calling ``advance()``, so
            # ``_pos`` can be left on a RESOLVED hunk.  Returning it
            # would highlight a resolved hunk as current and let a
            # follow-up accept/reject re-act on it (flipping its
            # recorded status while the disk content disagrees).  Seek
            # to the next unresolved hunk instead — one is guaranteed
            # to exist because ``remaining > 0`` here.
            self._seek(1)
        return self._all_hunks[self._pos]

    def mark_resolved(self, fi: int, hi: int, status: str = "accepted") -> None:
        """Mark a hunk as resolved with the given *status*.

        Args:
            fi: File index in :attr:`files`.
            hi: Hunk index within the file.
            status: ``"accepted"`` or ``"rejected"``.  Used by the web
                frontend to render accepted hunks dimmed and rejected
                hunks struck-through after the user acts on them.
        """
        self._resolved[(fi, hi)] = status

    def is_resolved(self, fi: int, hi: int) -> bool:
        """Return True if hunk ``(fi, hi)`` has been marked resolved.

        M9: prefer this method over poking at ``_resolved`` directly so
        the resolution-tracking representation can change without
        breaking callers.
        """
        return (fi, hi) in self._resolved

    def resolutions(self) -> list[dict[str, Any]]:
        """Return the full list of resolved hunks for the browser.

        Each entry is a dict ``{"fi": ..., "hi": ..., "status": ...}``
        suitable for inclusion in a ``merge_nav`` broadcast so the
        webview can visually mark every resolved hunk.
        """
        return [
            {"fi": fi, "hi": hi, "status": status}
            for (fi, hi), status in self._resolved.items()
        ]

    def _seek(self, step: int) -> None:
        """Move *step* (+1 or -1) to the next unresolved hunk."""
        if not self.remaining:
            return
        for _ in range(len(self._all_hunks)):
            self._pos = (self._pos + step) % len(self._all_hunks)
            if not self.is_resolved(*self._all_hunks[self._pos]):
                return

    def advance(self) -> None:
        """Move to the next unresolved hunk."""
        self._seek(1)

    def go_prev(self) -> None:
        """Move to the previous unresolved hunk."""
        self._seek(-1)

    def unresolved_in_file(self, fi: int) -> list[int]:
        """Return hunk indices not yet resolved for file *fi*."""
        return [
            hi
            for ffi, hi in self._all_hunks
            if ffi == fi and not self.is_resolved(ffi, hi)
        ]

    def all_unresolved(self) -> list[tuple[int, int]]:
        """Return all (file_idx, hunk_idx) pairs not yet resolved."""
        return [
            (fi, hi)
            for fi, hi in self._all_hunks
            if not self.is_resolved(fi, hi)
        ]


def _apply_exec_bit(path: str) -> None:
    """Re-apply the executable bit to a restored file.

    Mirrors git's checkout behavior for mode-``100755`` entries:
    execute permission is granted wherever read permission is present.

    Args:
        path: File whose mode should gain exec bits.
    """
    _apply_exec_state(path, True)


def _apply_exec_state(path: str, executable: bool) -> None:
    """Set or clear the executable bits of *path*.

    Setting mirrors git's checkout behavior for mode-``100755``
    entries (execute permission granted wherever read permission is
    present); clearing removes every exec bit so a rejected
    ``chmod +x`` leaves the tree clean (git mode ``100644``).

    Args:
        path: File whose mode should be adjusted.
        executable: ``True`` to grant exec bits, ``False`` to clear.
    """
    mode = os.stat(path).st_mode
    if executable:
        os.chmod(path, mode | ((mode & 0o444) >> 2))
    else:
        os.chmod(path, mode & ~0o111)


def _exec_flag(file_data: dict[str, Any]) -> bool | None:
    """Return a manifest entry's tri-state ``exec`` flag.

    ``True`` — the base is executable (restore must set exec bits);
    ``False`` — the base is non-executable (restore must clear them);
    ``None`` — the base mode is unknown (restore must not touch it,
    e.g. legacy manifests and agent-created files).

    Args:
        file_data: File entry from merge data.

    Returns:
        The flag when present and boolean, else ``None``.
    """
    val = file_data.get("exec")
    return val if isinstance(val, bool) else None


def _restore_base_bytes(
    base_path: str,
    write_to: str,
    link_target: str | None = None,
    *,
    make_executable: bool | None = None,
) -> None:
    """Restore *write_to* to the exact bytes of *base_path*.

    Used to reject changes to binary (or undecodable) files, whose
    merge entries carry a single whole-file pseudo-hunk: line-based
    splicing is meaningless for them, so the base copy is restored
    wholesale.  A missing/unreadable base restores an empty file
    (mirroring ``_write_base_copy``'s empty-base convention).

    When *link_target* is given, the base of the entry is a SYMLINK
    blob (the agent retargeted, deleted, or replaced a tracked
    symlink): the link itself is recreated instead of writing the
    blob's target string as regular-file content.

    Args:
        base_path: Path to the pre-task base copy.
        write_to: Real workspace path to restore.
        link_target: Target string of the base symlink, or ``None``
            for regular content.
        make_executable: Tri-state base mode: ``True`` when the base
            mode is ``100755`` (the restored file gets its exec bit
            back so rejecting a deleted script leaves a clean tree and
            a runnable file), ``False`` when it is ``100644`` (an exec
            bit the agent added is cleared), ``None`` when unknown
            (the on-disk mode is left untouched).
    """
    dest = Path(write_to)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if link_target is not None:
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        os.symlink(link_target, write_to)
        return
    try:
        data = Path(base_path).read_bytes()
    except OSError:
        data = b""
    # Never write THROUGH a symlink: git tracks the link itself, not
    # its target, and the target may be a precious file (possibly
    # outside the repo) whose truncation would be silent data loss.
    if dest.is_symlink():
        dest.unlink()
    dest.write_bytes(data)
    if make_executable is not None:
        _apply_exec_state(write_to, make_executable)


def _reject_hunk_in_file(
    current_path: str,
    base_path: str,
    hunk: dict[str, int],
    target_path: str | None = None,
    *,
    binary: bool = False,
    link_target: str | None = None,
    make_executable: bool | None = None,
) -> None:
    """Revert a single hunk in the current file to the base version.

    Reads both files, replaces the hunk's lines in the current file
    with the corresponding lines from the base file, and writes the
    result back.  Files are read and written WITHOUT newline
    translation (and split on ``\\n`` only) so rejecting one hunk of a
    CRLF file does not silently rewrite every other line with LF
    endings, and line numbering matches git's ``\\n``-based hunks.

    When *binary* is true — or either file turns out not to be
    decodable text — the base bytes are restored wholesale instead:
    binary merge entries carry a single whole-file pseudo-hunk, so
    line splicing does not apply (and used to raise
    ``UnicodeDecodeError``, crashing the merge action).

    When *target_path* differs from *current_path* — which happens when
    the agent deleted a tracked file and the merge view uses a
    ``.deleted`` placeholder as the visible "current" — the rejected
    content is written to *target_path* (the real workspace location),
    so the file is actually restored on disk.  Subsequent hunks read
    from *target_path* too, so partial rejections accumulate
    correctly.

    Args:
        current_path: Path to the file with agent changes (may be a
            display placeholder for deleted files).
        base_path: Path to the pre-task base copy.
        hunk: Hunk dict with keys ``bs``, ``bc``, ``cs``, ``cc``
            (0-based line positions).
        target_path: Real workspace path to write the rejection to.
            Defaults to *current_path* for backwards compatibility.
        binary: True when the merge entry is flagged binary; restores
            the base bytes wholesale.
        link_target: Target string when the base is a symlink blob;
            the link itself is restored (see ``_restore_base_bytes``).
        make_executable: Tri-state base mode (see
            :func:`_restore_base_bytes`): ``True`` re-applies the exec
            bit, ``False`` clears it, ``None`` leaves the mode alone.
    """
    write_to = target_path or current_path
    if binary or link_target is not None:
        _restore_base_bytes(
            base_path, write_to, link_target,
            make_executable=make_executable,
        )
        return
    # Read from *write_to* (the real workspace target) when it exists
    # so that successive partial rejections accumulate against the
    # restored content rather than the (now-stale) placeholder.
    cur_lines: list[str] = []
    try:
        try:
            cur_lines = _read_lines_preserved(write_to)
        except OSError:
            try:
                cur_lines = _read_lines_preserved(current_path)
            except OSError:
                cur_lines = []
        base_lines = _read_lines_preserved(base_path)
    except UnicodeDecodeError:
        # Undecodable content that slipped past the binary sniff
        # (e.g. UTF-16 / latin-1 without NUL bytes in the first 8 KiB).
        # Restoring the base bytes wholesale beats crashing the merge
        # action with an exception.
        _restore_base_bytes(
            base_path, write_to, make_executable=make_executable,
        )
        return
    except OSError:
        base_lines = []

    new_lines = (
        cur_lines[: hunk["cs"]]
        + base_lines[hunk["bs"] : hunk["bs"] + hunk["bc"]]
        + cur_lines[hunk["cs"] + hunk["cc"] :]
    )
    dest = Path(write_to)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Replace a symlink instead of writing THROUGH it — writing through
    # would clobber the pointed-to file (which may live outside the
    # repo) while leaving the rejected link itself untouched.
    if dest.is_symlink():
        dest.unlink()
    with open(write_to, "w", encoding="utf-8", newline="") as f:
        f.write("".join(new_lines))
    if make_executable is not None:
        _apply_exec_state(write_to, make_executable)


def _record_hunk_rejected(
    hunks: list[dict[str, Any]],
    hi: int,
    still_pending: Callable[[int], bool],
) -> None:
    """Record the post-reject offset bookkeeping for ``hunks[hi]``.

    The reject splice just replaced the hunk's ``cc`` current lines
    with its ``bc`` base lines; record that in the hunk itself so a
    RETRY (and the replayed ``merge_data``) re-applies it idempotently
    and stays consistent with disk.  Without this, a mid-file write
    failure on a LATER hunk (ENOSPC/EFBIG/EIO) followed by the user
    rejecting again re-spliced this hunk with its stale ``cc``,
    duplicating ``bc - cc`` lines whenever ``bc != cc``.  Then shifts
    the ``cs`` start offset of every LATER still-pending hunk by the
    just-applied line-count delta.  Shared by
    :func:`_reject_all_hunks_in_file` and the per-hunk ``reject``
    branch of ``_apply_web_merge_action``.

    Args:
        hunks: The file's hunk dicts (mutated in place).
        hi: Index of the hunk whose reject splice just succeeded.
        still_pending: Predicate; True for a later hunk index that is
            still unresolved and therefore needs its ``cs`` shifted.
    """
    hunk = hunks[hi]
    delta = hunk["bc"] - hunk["cc"]
    hunk["cc"] = hunk["bc"]
    for later_hi in range(hi + 1, len(hunks)):
        if still_pending(later_hi):
            hunks[later_hi]["cs"] += delta


def _hunk_unresolved(state: Any, fi: int, later_hi: int) -> bool:
    """Return True when hunk ``(fi, later_hi)`` is unresolved in *state*.

    ``still_pending`` predicate (via :func:`functools.partial`) for
    :func:`_record_hunk_rejected` in the per-hunk ``reject`` branch of
    ``_apply_web_merge_action``.

    Args:
        state: The ``_WebMergeState`` of the review.
        fi: File index of the hunk.
        later_hi: Hunk index within the file.
    """
    return not state.is_resolved(fi, later_hi)


def _reject_all_hunks_in_file(
    file_data: dict[str, Any], hunk_indices: list[int] | None = None,
) -> None:
    """Surgically revert the given hunks of a file to the base version.

    Reverts only the hunks named by *hunk_indices* (all hunks when
    ``None``) via :func:`_reject_hunk_in_file`, applying the same
    ``cs`` offset fix-ups to later pending hunks as the per-hunk
    ``reject`` action.  Callers (``reject-file`` / ``reject-all``)
    pass the file's UNRESOLVED hunk indices so content the user
    already ACCEPTED stays on disk — a whole-file base copy here
    would silently wipe accepted hunks while ``resolutions()`` still
    reported them ``"accepted"``.

    Args:
        file_data: File entry from merge data with ``base``,
            ``current``, ``hunks`` and (optionally) ``target`` path
            strings.
        hunk_indices: Indices into ``file_data["hunks"]`` to revert;
            ``None`` means every hunk.
    """
    hunks = file_data.get("hunks", [])
    if hunk_indices is None:
        hunk_indices = list(range(len(hunks)))
    if file_data.get("binary"):
        # Binary entries carry a single whole-file pseudo-hunk; restore
        # the base bytes wholesale (line splicing does not apply).
        # ``link_target`` marks a symlink-base entry whose reject must
        # recreate the link itself.
        if hunk_indices:
            _restore_base_bytes(
                file_data["base"],
                file_data.get("target") or file_data["current"],
                file_data.get("link_target"),
                make_executable=_exec_flag(file_data),
            )
        return
    pending = set(hunk_indices)
    for hi in sorted(hunk_indices):
        hunk = hunks[hi]
        _reject_hunk_in_file(
            file_data["current"], file_data["base"], hunk,
            file_data.get("target"),
            make_executable=_exec_flag(file_data),
        )
        pending.discard(hi)
        _record_hunk_rejected(hunks, hi, pending.__contains__)
