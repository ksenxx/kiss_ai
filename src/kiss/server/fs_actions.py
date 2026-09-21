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
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

_FIND_TIMEOUT_S = 30
_DIFF_TIMEOUT_S = 30

_FIND_EXCLUDED_DIRS = (".git", ".svn", ".hg", "node_modules", ".venv", "__pycache__")


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
    except OSError as exc:
        return {"error": f"{action} failed: {exc}"}
    return {"error": f"Unknown file action: {action}"}


def _safe_name(name: str) -> str:
    """Validate a new-entry name: no empty segments, no ``..``."""
    parts = [p for p in name.replace("\\", "/").split("/")]
    if not parts or any(p in ("", ".", "..") for p in parts):
        raise OSError("invalid name")
    return "/".join(parts)


def _new_entry(parent: str, name: str, is_dir: bool) -> dict[str, Any]:
    """Create a file or folder named *name* under *parent*."""
    if not os.path.isdir(parent):
        return {"error": f"Not a folder: {parent}"}
    try:
        rel = _safe_name(name)
    except OSError:
        return {"error": f"Not a valid name: {name!r}"}
    target = Path(parent) / rel
    if target.exists() or target.is_symlink():
        return {"error": f"A file or folder {rel} already exists", "exists": True}
    if is_dir:
        target.mkdir(parents=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=False)
    return {"ok": True, "path": str(target)}


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
    if target == src:
        return {"ok": True, "path": str(target)}
    if target.parent != src.parent or target.name in ("", ".", ".."):
        return {"error": f"Not a valid name: {target.name!r}"}
    if not os.path.lexists(target):
        os.rename(src, target)
        return {"ok": True, "path": str(target)}
    if not overwrite:
        return {
            "error": f"A file or folder {target.name} already exists",
            "exists": True,
        }
    _replace(target, os.rename, src)
    return {"ok": True, "path": str(target)}


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
    replacement itself cannot be produced.
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
    else:
        _replace(target, _transfer, move, src)
    return {"ok": True, "path": str(target)}


def _transfer(move: bool, src: Path, target: Path) -> None:
    """Move or copy *src* to the not-yet-existing *target* (symlinks as links)."""
    if move:
        shutil.move(str(src), str(target))
    elif src.is_dir() and not src.is_symlink():
        shutil.copytree(src, target, symlinks=True)
    else:
        shutil.copy2(src, target, follow_symlinks=False)


def _cut_bytes(text: str, limit: int) -> tuple[str, bool]:
    """Cut *text* to *limit* UTF-8 bytes (never inside a character)."""
    data = text.encode("utf-8", "surrogateescape")
    if len(data) <= limit:
        return text, False
    return data[:limit].decode("utf-8", "ignore"), True


class _CappedOutput:
    """Result of :func:`_run_capped`."""

    def __init__(self) -> None:
        self.returncode = -1
        self.stdout = b""
        self.stderr = ""
        self.truncated = False
        self.timed_out = False


def _drain_stderr(stream: Any, out: _CappedOutput) -> None:
    """Read *stream* (stderr) to the end, keeping its first 64 KiB."""
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        if size < 65536:
            chunks.append(chunk)
            size += len(chunk)
    out.stderr = b"".join(chunks).decode("utf-8", "replace")


def _run_capped(cmd: list[str], cwd: str | None, limit: int, timeout: float) -> _CappedOutput:
    """Run *cmd*, keeping at most *limit* bytes of its stdout.

    Unlike ``subprocess.run(capture_output=True)`` the whole output is
    never held in memory: stdout is read in chunks and the process is
    killed as soon as the cap is exceeded (``truncated``) or *timeout*
    seconds pass (``timed_out``), so a search over a huge tree or a
    diff of two huge files costs the daemon at most *limit* bytes.
    """
    out = _CappedOutput()
    proc = subprocess.Popen(  # noqa: S603 -- fixed argv, no shell
        cmd, cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    err_thread = threading.Thread(
        target=_drain_stderr, args=(proc.stderr, out), daemon=True,
    )
    err_thread.start()

    def _timed_out() -> None:
        out.timed_out = True
        proc.kill()

    timer = threading.Timer(timeout, _timed_out)
    timer.start()
    chunks: list[bytes] = []
    size = 0
    try:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                out.truncated = True
                proc.kill()
                break
        proc.stdout.close()
        proc.wait()
    finally:
        timer.cancel()
    err_thread.join(timeout=5)
    out.returncode = proc.returncode
    out.stdout = b"".join(chunks)[:limit]
    return out


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
        proc = _run_capped(cmd, folder, TEXT_MAX_BYTES, _FIND_TIMEOUT_S)
    except OSError as exc:
        return {"error": f"git grep failed: {exc}"}
    if proc.timed_out:
        return {"error": "Search timed out"}
    if not proc.truncated and proc.returncode not in (0, 1):
        return {"error": proc.stderr.strip() or "git grep failed"}
    # A stream cut mid-line ends with a partial match: drop it.
    raw = proc.stdout
    if proc.truncated:
        raw = raw[: raw.rfind(b"\n") + 1]
    lines = raw.decode("utf-8", "replace").splitlines()
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
        proc = _run_capped(
            ["git", "diff", "--no-color", "--no-index", "--", a, b],
            None, TEXT_MAX_BYTES, _DIFF_TIMEOUT_S,
        )
    except OSError as exc:
        return {"error": f"diff failed: {exc}"}
    if proc.timed_out:
        return {"error": "diff timed out"}
    # --no-index exits 1 when the files differ, 0 when identical.
    if not proc.truncated and proc.returncode not in (0, 1):
        return {"error": proc.stderr.strip() or "git diff failed"}
    text, truncated = _cut_bytes(
        proc.stdout.decode("utf-8", "replace"), TEXT_MAX_BYTES,
    )
    truncated = truncated or proc.truncated
    if not text:
        text = f"{a} and {b} are identical\n"
    return {"ok": True, "path": a, "dest": b, "text": text, "truncated": truncated}
