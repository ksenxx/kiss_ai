# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""File-system actions behind the remote Explorer's context menu.

The remote webapp's Explorer view offers the VS Code Explorer context
menu (New File..., New Folder..., Rename..., Delete, Cut / Copy /
Paste, Find in Folder..., Compare Selected).  Each entry becomes one
``fsAction`` command that the daemon (``web_server.py``) runs through
:func:`fs_action` on a worker thread.  Every function here is a plain
synchronous operation on absolute paths the daemon already resolved;
it returns a reply dict and never raises.

Like VS Code's own Explorer, a paste that would overwrite an existing
entry is refused unless the client confirmed the replacement
(``overwrite``), and a copy into the entry's own folder gets the
"name copy" suffix VS Code uses.

The actions that change the tree run one at a time
(:data:`_mutation_lock`): each checks that its destination is free and
creates it in a separate step, and two unconfirmed requests for one
name (two surfaces' Explorers run on separate threads) that both found
it free would otherwise both create it, the second silently
overwriting the first's entry.  The search and compare actions change
nothing and run alongside.
"""

from __future__ import annotations

import os
import shutil
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.core.locked import locked
from kiss.core.processes import run_capped

FS_ACTIONS: frozenset[str] = frozenset(
    {
        "newFile",
        "newFolder",
        "rename",
        "delete",
        "copy",
        "move",
        "findInFolder",
        "compare",
    }
)
"""Every action :func:`fs_action` accepts."""

FIND_MAX_MATCHES = 2000
"""Cap on the lines one ``findInFolder`` reply lists."""

TEXT_MAX_BYTES = 4 * 1024 * 1024
"""Cap on the UTF-8 bytes of text one ``findInFolder`` / ``compare`` reply carries."""

_TOOL_TIMEOUT_S = 30
"""Deadline of the ``git grep`` / ``git diff`` child of one request."""

_FIND_EXCLUDED_DIRS = (".git", ".svn", ".hg", "node_modules", ".venv", "__pycache__")

_mutation_lock = threading.RLock()
"""Held by every action that changes the tree, from its checks to its last step."""


def fs_action(
    action: str,
    path: str,
    dest: str = "",
    name: str = "",
    query: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one Explorer context-menu action.

    Args:
        action: One of :data:`FS_ACTIONS`.
        path: The absolute path the menu was opened on (for ``newFile``
            / ``newFolder`` the parent folder; for ``compare`` the first
            file; for ``findInFolder`` the folder searched).
        dest: ``rename``: the new absolute path; ``copy`` / ``move``: the
            destination FOLDER; ``compare``: the second file.
        name: ``newFile`` / ``newFolder``: the entry name (may contain
            ``/`` to create intermediate folders, like VS Code).
        query: ``findInFolder``: the text searched for (literal).
        overwrite: ``copy`` / ``move`` / ``rename``: replace an existing
            destination instead of refusing.

    Returns:
        ``{"ok": True, "path": <resulting path>}`` (plus ``"text"`` for
        ``findInFolder`` / ``compare``, ``"exists": True`` when a paste
        was refused because the destination exists) or
        ``{"error": <message>}``.
    """
    try:
        if action == "newFile":
            return _new_entry(path, name, is_dir=False)
        if action == "newFolder":
            return _new_entry(path, name, is_dir=True)
        if action == "rename":
            return _rename(path, dest, overwrite)
        if action == "delete":
            return _delete(path)
        if action == "copy":
            return _paste(path, dest, move=False, overwrite=overwrite)
        if action == "move":
            return _paste(path, dest, move=True, overwrite=overwrite)
        if action == "findInFolder":
            return find_in_folder(path, query)
        if action == "compare":
            return compare_files(path, dest)
    except (OSError, ValueError) as exc:
        # ValueError: an embedded NUL byte, which pathlib's exists()
        # answers False to and the creating call then refuses.
        return {"error": f"{action} failed: {exc}"}
    return {"error": f"Unknown file action: {action}"}


def _safe_name(name: str) -> str:
    """Validate a new-entry name: no empty segments, no ``..``."""
    parts = name.replace("\\", "/").split("/")
    if "\0" in name or any(p in ("", ".", "..") for p in parts):
        raise OSError("invalid name")
    return "/".join(parts)


@locked(_mutation_lock)
def _new_entry(parent: str, name: str, is_dir: bool) -> dict[str, Any]:
    """Create a file or folder named *name* under *parent*."""
    if not os.path.isdir(parent):
        return {"error": f"Not a folder: {parent}"}
    try:
        rel = _safe_name(name)
    except OSError:
        return {"error": f"Not a valid name: {name!r}"}
    target = Path(parent) / rel
    # The caller confined *parent*; a nested name must not leave it
    # through a symlinked intermediate folder that mkdir/touch would
    # follow (the missing tail of the name resolves lexically).
    real_parent = os.path.realpath(parent)
    if os.path.commonpath([os.path.realpath(target.parent), real_parent]) != real_parent:
        return {"error": f"Not inside the folder: {rel}"}
    if target.exists() or target.is_symlink():
        return {"error": f"A file or folder {rel} already exists", "exists": True}
    if is_dir:
        target.mkdir(parents=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=False)
    return {"ok": True, "path": str(target)}


@locked(_mutation_lock)
def _rename(path: str, dest: str, overwrite: bool) -> dict[str, Any]:
    """Rename *path* to *dest* (an absolute path in the same folder).

    Like VS Code's Rename..., the entry stays in its folder: *dest*
    must be a sibling name (``a/b`` renames across folders are not a
    rename).  A symlink is renamed itself, never its target.
    """
    src = Path(path)
    if not os.path.lexists(src):
        return {"error": f"Not found: {path}"}
    if not dest:
        return {"error": "Not a valid name: ''"}
    target = Path(dest)
    # Compared as spelled: Windows paths compare case-insensitively, and
    # a rename that changes only the case of a name must still rename.
    if str(target) == str(src):
        return {"ok": True, "path": str(target)}
    if target.parent != src.parent or target.name in ("", ".", ".."):
        return {"error": f"Not a valid name: {target.name!r}"}
    if not os.path.lexists(target):
        os.rename(src, target)
    elif _is_same_entry(src, target):
        _rename_case(src, target)
    elif not overwrite:
        return {
            "error": f"A file or folder {target.name} already exists",
            "exists": True,
        }
    else:
        _replace(target, os.rename, src)
    return {"ok": True, "path": str(target)}


def _is_same_entry(src: Path, target: Path) -> bool:
    """Whether the existing *target* is *src* itself, its name spelled in another case.

    On a case-insensitive file system (macOS, Windows, a Linux
    ``casefold`` folder) ``A.txt`` finds ``a.txt``: the rename of
    ``a.txt`` to ``A.txt`` then finds its own source at the destination,
    while the folder holds no entry named exactly ``A.txt``.  An entry
    with that exact name is another entry, even when it is a hard link
    of the same file.
    """
    return os.path.samestat(os.lstat(src), os.lstat(target)) and (
        target.name not in os.listdir(target.parent)
    )


def _rename_case(src: Path, target: Path) -> None:
    """Rename *src* to *target*, the same entry's name in another case.

    Through a temporary name in the same folder: a Linux ``casefold``
    folder takes the direct rename of an entry onto itself as a no-op.
    A failed second step puts the entry back under *src*.
    """
    aside = src.with_name(f".{src.name}.renaming-{uuid.uuid4().hex}")
    os.rename(src, aside)
    try:
        os.rename(aside, target)
    except BaseException:
        os.rename(aside, src)
        raise


def _remove(target: Path) -> None:
    """Delete *target*, whatever it is (a symlink is unlinked, not followed)."""
    if target.is_symlink() or not target.is_dir():
        target.unlink()
    else:
        shutil.rmtree(target)


def _replace(target: Path, operation: Callable[..., Any], *args: Any) -> None:
    """Run ``operation(*args, target)`` -- which creates *target* -- over an existing *target*.

    The old entry is first renamed aside (same folder, so atomic) and
    only deleted once the operation succeeded; if it fails, whatever
    partial *target* it left is removed and the old entry is put back,
    so a confirmed overwrite never loses the previous content when the
    replacement itself cannot be produced.  The operation must leave
    its source intact when it fails (see :func:`_move_over`): a partial
    *target* is disposable only while the source still holds the data.
    """
    backup = target.with_name(f".{target.name}.replaced-{uuid.uuid4().hex}")
    os.rename(target, backup)
    try:
        operation(*args, target)
    except BaseException:
        try:
            if os.path.lexists(target):
                _remove(target)
        finally:
            os.rename(backup, target)
        raise
    _remove(backup)


@locked(_mutation_lock)
def _delete(path: str) -> dict[str, Any]:
    """Delete *path* permanently (the remote host has no trash)."""
    target = Path(path)
    if not os.path.lexists(target):
        return {"error": f"Not found: {path}"}
    if target.parent == target:
        return {"error": "Refusing to delete a file-system root"}
    _remove(target)
    return {"ok": True, "path": str(target)}


def _copy_name(dest_dir: Path, src: Path) -> Path:
    """The name VS Code gives a copy pasted into its own folder.

    ``a.txt`` becomes ``a copy.txt``, then ``a copy 2.txt`` and so on
    until a free name is found.
    """
    stem, suffix = src.stem, src.suffix
    if src.is_dir():
        stem, suffix = src.name, ""
    candidate = dest_dir / f"{stem} copy{suffix}"
    n = 2
    while candidate.exists() or candidate.is_symlink():
        candidate = dest_dir / f"{stem} copy {n}{suffix}"
        n += 1
    return candidate


@locked(_mutation_lock)
def _paste(path: str, dest_dir: str, move: bool, overwrite: bool) -> dict[str, Any]:
    """Copy or move *path* into the folder *dest_dir*."""
    src = Path(path)
    if not os.path.lexists(src):
        return {"error": f"Not found: {path}"}
    if not os.path.isdir(dest_dir):
        return {"error": f"Not a folder: {dest_dir}"}
    dest = Path(dest_dir)
    same_folder = src.parent.resolve() == dest.resolve()
    if src.is_dir() and not src.is_symlink():
        try:
            dest.resolve().relative_to(src.resolve())
        except ValueError:
            pass
        else:
            return {"error": f"Cannot paste {src.name} into itself"}
    if move and same_folder:
        # Moving an entry into the folder it is already in changes nothing.
        return {"ok": True, "path": str(src)}
    target = dest / src.name
    if same_folder:
        target = _copy_name(dest, src)
    if not os.path.lexists(target):
        _transfer(move, src, target)
    elif not overwrite:
        return {
            "error": f"A file or folder {src.name} already exists",
            "exists": True,
        }
    elif move:
        _move_over(src, target)
    else:
        _replace(target, _transfer, False, src)
    return {"ok": True, "path": str(target)}


def _transfer(move: bool, src: Path, target: Path) -> None:
    """Move or copy *src* to the not-yet-existing *target* (symlinks as links)."""
    if move:
        shutil.move(str(src), str(target))
    elif src.is_dir() and not src.is_symlink():
        shutil.copytree(src, target, symlinks=True)
    else:
        shutil.copy2(src, target, follow_symlinks=False)


def _move_over(src: Path, target: Path) -> None:
    """Move *src* over the existing *target*, never losing the source's data.

    Not :func:`_replace` around ``shutil.move``: when it cannot rename,
    ``shutil.move`` copies the source and then deletes it, so a failed
    deletion (a source whose folder entry cannot be removed, a partly
    deleted tree) leaves the only copy of the data in *target*, which
    the rollback would delete.  The source is instead renamed or, when
    that is impossible, copied over the old entry (rolled back on
    failure, the source untouched) and deleted only once *target* is
    complete: a failed deletion then leaves the data at *target*.
    """
    _replace(target, _rename_or_copy, src)
    if os.path.lexists(src):
        _remove(src)


def _rename_or_copy(src: Path, target: Path) -> None:
    """Rename *src* to the not-yet-existing *target*, or copy it when no rename is possible."""
    try:
        os.rename(src, target)
    except OSError:
        _transfer(False, src, target)


#: Stderr bytes :func:`find_in_folder` and :func:`compare_files` keep
#: for their error reply.
_STDERR_KEEP_BYTES = 65536


def find_in_folder(folder: str, query: str) -> dict[str, Any]:
    """Search *folder* recursively for the literal text *query*.

    Runs ``git grep --no-index -n -I -F`` (binary files, VCS and
    dependency folders and ``.gitignore``d files skipped) and returns
    the matches as ``relative/path:line: text`` lines — the same
    information VS Code's search results tree shows — capped at
    :data:`FIND_MAX_MATCHES`.  ``git`` is used instead of ``grep``
    because it is the one search tool present on every platform the
    daemon runs on (Windows ships no ``grep``); ``--no-index`` makes it
    walk *folder* directly, so it works outside a repository too.

    Returns:
        ``{"ok": True, "path": folder, "query": query, "text": <lines>,
        "count": <matches>, "truncated": bool}`` or ``{"error": ...}``.
    """
    if not os.path.isdir(folder):
        return {"error": f"Not a folder: {folder}"}
    if not query:
        return {"error": "Nothing to search for"}
    # ``--no-color --no-column --no-full-name`` pin the ``path:line:text``
    # shape the parser below relies on; a user's ``[color]``/``[grep]``
    # git config would otherwise change it.
    cmd = [
        "git", "grep", "--no-index", "--exclude-standard", "--no-color", "--no-column",
        "--no-full-name", "-n", "-I", "-F", "-e", query, "--", ".",
    ]
    cmd += [f":(exclude,glob)**/{d}/**" for d in _FIND_EXCLUDED_DIRS]
    try:
        # A stream cut mid-line ends with a partial match: ``last_full_line``
        # drops it.
        proc = run_capped(
            cmd, cwd=folder, limit=TEXT_MAX_BYTES, timeout=_TOOL_TIMEOUT_S,
            keep_stderr=_STDERR_KEEP_BYTES, last_full_line=True,
        )
    except OSError as exc:
        return {"error": f"git grep failed: {exc}"}
    if proc.timed_out:
        return {"error": "Search timed out"}
    if not proc.truncated and proc.returncode not in (0, 1):
        return {"error": proc.stderr.strip() or "git grep failed"}
    lines = proc.stdout.decode("utf-8", "replace").splitlines()
    truncated = proc.truncated or len(lines) > FIND_MAX_MATCHES
    kept = [
        line[2:] if line.startswith("./") else line
        for line in lines[:FIND_MAX_MATCHES]
    ]
    return {
        "ok": True,
        "path": folder,
        "query": query,
        "text": "\n".join(kept) + ("\n" if kept else ""),
        "count": len(lines),
        "truncated": truncated,
    }


def compare_files(a: str, b: str) -> dict[str, Any]:
    """Unified diff of two files (VS Code's "Compare Selected").

    Uses ``git diff --no-index`` so the output is the same diff format
    the commit graph shows, without needing either file in a repo.

    Returns:
        ``{"ok": True, "path": a, "dest": b, "text": <diff>,
        "truncated": bool}`` or ``{"error": ...}``.
    """
    for p in (a, b):
        if not os.path.isfile(p):
            return {"error": f"Not a file: {p}"}
    try:
        proc = run_capped(
            ["git", "diff", "--no-color", "--no-ext-diff", "--no-index", "--", a, b],
            limit=TEXT_MAX_BYTES, timeout=_TOOL_TIMEOUT_S, keep_stderr=_STDERR_KEEP_BYTES,
        )
    except OSError as exc:
        return {"error": f"diff failed: {exc}"}
    if proc.timed_out:
        return {"error": "diff timed out"}
    # --no-index exits 1 when the files differ, 0 when identical.
    if not proc.truncated and proc.returncode not in (0, 1):
        return {"error": proc.stderr.strip() or "git diff failed"}
    text = proc.stdout.decode("utf-8", "replace")
    if not text:
        text = f"{a} and {b} are identical\n"
    return {"ok": True, "path": a, "dest": b, "text": text, "truncated": proc.truncated}
