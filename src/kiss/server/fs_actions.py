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

import contextlib
import ctypes
import errno
import os
import shutil
import stat
import sys
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterator
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
    root: str = "",
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
        root: The workspace: the folder a new entry goes into, the
            folder of the entry renamed, deleted, copied or moved and
            the destination folder must lie in it on disk, and the
            action works in them as it resolved them (see
            :func:`_pinned`).  Empty: no such limit.

    Returns:
        ``{"ok": True, "path": <resulting path>}`` (plus ``"text"`` for
        ``findInFolder`` / ``compare``, ``"exists": True`` when a paste
        was refused because the destination exists) or
        ``{"error": <message>}``.
    """
    try:
        if action == "newFile":
            return _new_entry(path, name, False, root)
        if action == "newFolder":
            return _new_entry(path, name, True, root)
        if action == "rename":
            return _rename(path, dest, overwrite, root)
        if action == "delete":
            return _delete(path, root)
        if action == "copy":
            return _paste(path, dest, False, overwrite, root)
        if action == "move":
            return _paste(path, dest, True, overwrite, root)
        if action == "findInFolder":
            return find_in_folder(path, query)
        if action == "compare":
            return compare_files(path, dest)
    except FileExistsError as exc:
        return {
            "error": f"A file or folder {Path(exc.filename or dest).name} already exists",
            "exists": True,
        }
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
def _new_entry(parent: str, name: str, is_dir: bool, root: str) -> dict[str, Any]:
    """Create a file or folder named *name* under *parent*, a folder of the workspace *root*."""
    if not os.path.isdir(parent):
        return {"error": f"Not a folder: {parent}"}
    try:
        rel = _safe_name(name)
    except OSError:
        return {"error": f"Not a valid name: {name!r}"}
    target = Path(parent) / rel
    # A nested name must not leave *parent* through a symlinked
    # intermediate folder that mkdir/touch would follow (the missing
    # tail of the name resolves lexically).
    real_parent = _resolved(parent, root)
    if os.path.commonpath([os.path.realpath(target.parent), real_parent]) != real_parent:
        return {"error": f"Not inside the folder: {rel}"}
    if target.exists() or target.is_symlink():
        return {"error": f"A file or folder {rel} already exists", "exists": True}
    if hasattr(os, "O_PATH") and {os.open, os.mkdir} <= os.supports_dir_fd:
        _create_pinned(real_parent, target, is_dir)
    # macOS and Windows keep the creation by path, so a folder swapped for a link after the
    # checks is still followed there: no O_PATH to open an unreadable folder, no host to verify on.
    elif is_dir:
        target.mkdir(parents=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=False)
    return {"ok": True, "path": str(target)}


def _create_pinned(root: str, target: Path, is_dir: bool) -> None:
    """Create *target* and its missing parent folders below *root*, following no link.

    :func:`_new_entry` checked the name as a path, and a creation by path
    follows whatever an external writer has put there since, a link out
    of *root* in place of a folder included.  So every folder is opened
    relative to the one before it and refused if it is a link: the
    deepest existing one by its real path (:func:`_open_folder`; the
    links the check accepted resolved), the missing ones as they are
    created.
    The replies stay those of the creation by path: what it refused (a
    name through something that is no folder, a path too long or not
    encodable) is asked of the path first and refused with its error.
    """
    with contextlib.suppress(FileNotFoundError):
        os.lstat(target if is_dir else target.parent)  # fails where pathlib's first mkdir did
    names = [target.name]
    folder = target.parent
    while not folder.is_dir():
        names.insert(0, folder.name)
        folder = folder.parent
    if os.path.lexists(folder / names[0]):  # a dangling link, or a file where a folder goes
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(folder / names[0]))
    real = os.path.realpath(folder)
    Path(real).relative_to(root)  # ValueError: moved out of *root* since the check
    fd = _open_folder(real)
    spelled = folder  # an error names the entry as the request spelled it, like one by path
    try:
        for name in names[:-1]:
            spelled /= name
            try:
                below = os.open(name, _FOLDER, dir_fd=fd)
            except FileNotFoundError:
                os.mkdir(name, dir_fd=fd)
                below = os.open(name, _FOLDER, dir_fd=fd)
            os.close(fd)
            fd = below
        spelled /= names[-1]
        if is_dir:
            os.mkdir(names[-1], dir_fd=fd)
        else:
            with contextlib.suppress(FileNotFoundError):
                os.lstat(target)  # fails where the open by path did: too long, not encodable
            os.close(os.open(names[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666, dir_fd=fd))
    except OSError as exc:
        raise OSError(exc.errno, exc.strerror, str(spelled)) from exc
    finally:
        os.close(fd)


def _resolved(folder: str | Path, root: str) -> str:
    """The real path of *folder*, which must lie in the workspace *root* if one is given."""
    real = os.path.realpath(folder)
    if root and not Path(real).is_relative_to(os.path.realpath(root)):
        raise OSError(f"Not inside the workspace: {folder}")
    return real


#: How a folder is opened to work in it: never through a link, and (``O_PATH``) even when
#: it may be searched but not read.  Linux only, see :func:`_pinned`.
_FOLDER = (
    os.O_DIRECTORY | os.O_NOFOLLOW | os.O_PATH  # pyright: ignore[reportAttributeAccessIssue]
    if hasattr(os, "O_PATH")
    else 0
)

#: Whether a folder is worked in through a descriptor of it, spelled ``/proc/self/fd/N``.
_PINS = bool(_FOLDER) and os.path.isdir("/proc/self/fd")


def _open_folder(real: str) -> int:
    """Open the folder at the resolved path *real*; the caller closes the descriptor.

    Every call that takes a path resolves it again, through whatever an
    external writer has put on the way since it was checked: a link in
    place of a folder leads out of the workspace.  So every folder from
    ``/`` down is opened in the one before it and refused if it is a
    link, and the caller works in the folder it gets, wherever the path
    leads by then.
    """
    fd = os.open("/", _FOLDER)
    path = "/"
    try:
        for name in Path(real).parts[1:]:
            path = os.path.join(path, name)
            below = os.open(name, _FOLDER, dir_fd=fd)
            os.close(fd)
            fd = below
    except OSError as exc:
        os.close(fd)
        raise OSError(exc.errno, exc.strerror, path) from exc
    return fd


@contextlib.contextmanager
def _pinned(folder: str | Path, root: str) -> Iterator[Path]:
    """Yield a name of *folder* that keeps naming the folder it is now.

    The links of *folder* are resolved once, the result must lie in the
    workspace *root* (:func:`_resolved`), and the action then works
    through a descriptor of that folder (:func:`_open_folder`), spelled
    ``/proc/self/fd/N``: a folder on the way that is swapped for a link
    after the check leads no deletion, rename or paste out of the
    workspace.  An error raised meanwhile names the folder as the
    request spelled it.

    macOS and Windows have no such spelling and keep the checked name:
    a folder swapped for a link is still followed there, as it is by
    :func:`_new_entry`.
    """
    real = _resolved(folder, root)
    if not _PINS:
        yield Path(folder)
        return
    with _named(_open_folder(real), folder) as pinned:
        yield pinned


@contextlib.contextmanager
def _opened(folder: str | Path) -> Iterator[Path]:
    """Yield a name of *folder*, a folder below a pinned one, that keeps naming it.

    Like :func:`_pinned`, without the walk from ``/``: *folder* is opened as it
    is named and refused if it is a link.
    """
    if not _PINS:
        yield Path(folder)
        return
    with _named(os.open(folder, _FOLDER), folder) as pinned:
        yield pinned


@contextlib.contextmanager
def _named(fd: int, folder: str | Path) -> Iterator[Path]:
    """Yield the ``/proc/self/fd/N`` name of the open folder or file *fd*, closed at the end.

    An error raised meanwhile names it as *folder* spells it.
    """
    pinned = f"/proc/self/fd/{fd}"
    try:
        yield Path(pinned)
    except FileExistsError:
        raise  # reported by the name of the entry alone
    except OSError as exc:
        # Up to its "/" or its closing quote: /proc/self/fd/97 is not /proc/self/fd/9.
        text = str(exc).replace(pinned + "/", os.path.join(folder, ""))
        raise OSError(text.replace(repr(pinned), repr(str(folder)))) from exc
    finally:
        os.close(fd)


@locked(_mutation_lock)
def _rename(path: str, dest: str, overwrite: bool, root: str) -> dict[str, Any]:
    """Rename *path* to *dest* (an absolute path in the same folder).

    Like VS Code's Rename..., the entry stays in its folder: *dest*
    must be a sibling name (``a/b`` renames across folders are not a
    rename).  A symlink is renamed itself, never its target.  The
    folder is pinned (:func:`_pinned`) and must lie in *root*.
    """
    src = Path(path)
    if src.name in ("", "..") or not os.path.lexists(src):  # a root, "..": no entry of a folder
        return {"error": f"Not found: {path}"}
    if not dest:
        return {"error": "Not a valid name: ''"}
    target = Path(dest)
    done = {"ok": True, "path": str(target)}
    # Compared as spelled: Windows paths compare case-insensitively, and
    # a rename that changes only the case of a name must still rename.
    if str(target) == str(src):
        return done
    if target.parent != src.parent or target.name in ("", ".", ".."):
        return {"error": f"Not a valid name: {target.name!r}"}
    os.fsencode(target)  # a name that cannot be encoded is refused as the request spelled it
    with _pinned(src.parent, root) as folder:
        src, target = folder / src.name, folder / target.name
        if not os.path.lexists(target):
            _rename_noreplace(src, target)
        elif _is_same_entry(src, target):
            _rename_case(src, target)
        elif not overwrite:
            return {
                "error": f"A file or folder {target.name} already exists",
                "exists": True,
            }
        else:
            # No-clobber, as a paste publishes: a destination an editor
            # recreated after the old entry was put aside is kept, with the
            # source, and the old entry stays at the backup the error names.
            _replace(target, _rename_noreplace, src)
    return done


def _rename_noreplace(src: Path, target: Path) -> None:
    """Rename an entry without replacing a concurrently created target.

    Atomically, with the platform's rename that refuses an existing
    target.  Where there is none (NFS, 9p, CephFS and eCryptfs take no
    rename flag; a C library without ``renameat2``) the entry gets the
    new name as a hard link, which refuses an existing target too, and
    then loses the old one.  What takes no hard link (a folder, any
    entry on FAT) is renamed after a look at the target: nothing better
    exists there, and ``rename`` replaces no folder that holds anything.
    """
    if os.name == "nt":
        try:
            os.rename(src, target)
        except FileExistsError:
            # Windows reports the source in ``filename`` for this error; callers
            # must name the destination that was concurrently created.
            raise FileExistsError(
                errno.EEXIST, os.strerror(errno.EEXIST), str(target)
            ) from None
        return
    if "\0" in str(src) or "\0" in str(target):
        raise ValueError("embedded null byte")
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        args: tuple[int | bytes, ...] = (os.fsencode(src), os.fsencode(target), 4)  # RENAME_EXCL
    else:
        rename = getattr(libc, "renameat2", None)
        args = (-100, os.fsencode(src), -100, os.fsencode(target), 1)  # AT_FDCWD, NOREPLACE
    code = errno.ENOSYS
    if rename is not None:
        if rename(*args) == 0:
            return
        code = ctypes.get_errno()
    if code not in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP):
        raise OSError(code, os.strerror(code), str(target))
    try:
        os.link(src, target, follow_symlinks=False)
    except OSError as exc:
        if isinstance(exc, FileExistsError) or os.path.lexists(target):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(target)) from None
        os.rename(src, target)
        return
    try:
        os.unlink(src)
    except OSError:
        os.unlink(target)  # not left under both names
        raise


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
    No step replaces an entry an editor saved meanwhile.  A failed
    second step puts the entry back under *src* only if that name is
    still free; otherwise the entry stays at the temporary name, which
    the error gives.
    """
    aside = src.with_name(f".{src.name}.renaming-{uuid.uuid4().hex}")
    _rename_noreplace(src, aside)
    try:
        _rename_noreplace(aside, target)
    except BaseException:
        try:
            _rename_noreplace(aside, src)
        except OSError as exc:
            raise OSError(f"Source changed; retained entry at {aside}: {exc}") from exc
        raise


def _remove(target: Path) -> None:
    """Delete *target*, whatever it is (a symlink is unlinked, not followed)."""
    if target.is_symlink() or not target.is_dir():
        target.unlink()
    else:
        shutil.rmtree(str(target))  # as a str: its error names the folder the way it was given


def _replace(target: Path, operation: Callable[..., Path | OSError | None], *args: Any) -> None:
    """Run ``operation(*args, target)`` -- which creates *target* -- over an existing *target*.

    The old entry is first renamed aside (same folder, so atomic) and
    only deleted once the operation succeeded. Operations publish atomically;
    on failure the old entry is put back only if the name is still free.
    Otherwise the backup is retained and named in the error, never restored
    over a concurrent writer's entry.

    Once *target* is complete the old entry and the source a move had to
    copy (the operation's result, see :func:`_finish_move`) are deleted.
    One that cannot be deleted does not keep the other: the error names
    what stayed, and the new *target* is never rolled back for it.
    """
    backup = target.with_name(f".{target.name}.replaced-{uuid.uuid4().hex}")
    os.rename(target, backup)
    try:
        moved = operation(*args, target)
    except BaseException:
        try:
            _rename_noreplace(backup, target)
        except OSError as exc:
            raise OSError(f"Replacement failed; retained entry at {backup}: {exc}") from exc
        raise
    try:
        _remove(backup)
    except OSError as exc:
        try:
            _finish_move(moved)
        except OSError as also:
            raise OSError(f"{exc}; {also}") from also
        raise
    _finish_move(moved)


@locked(_mutation_lock)
def _delete(path: str, root: str) -> dict[str, Any]:
    """Delete *path*, an entry of a folder of *root*, permanently (the remote host has no trash)."""
    target = Path(path)
    if target.name == ".." or not os.path.lexists(target):  # "..": the folder above its folder
        return {"error": f"Not found: {path}"}
    if target.parent == target:
        return {"error": "Refusing to delete a file-system root"}
    with _pinned(target.parent, root) as folder:
        _remove(folder / target.name)
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
def _paste(path: str, dest_dir: str, move: bool, overwrite: bool, root: str) -> dict[str, Any]:
    """Copy or move *path* into the folder *dest_dir*; both folders pinned, and in *root*."""
    src = Path(path)
    if src.name in ("", "..") or not os.path.lexists(src):  # a root, "..": no entry of a folder
        return {"error": f"Not found: {path}"}
    if not os.path.isdir(dest_dir):
        return {"error": f"Not a folder: {dest_dir}"}
    with _pinned(src.parent, root) as folder, _pinned(dest_dir, root) as dest:
        src = folder / src.name
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
            return {"ok": True, "path": str(Path(path))}
        target = dest / src.name
        if same_folder:
            target = _copy_name(dest, src)
            overwrite = False
        if not os.path.lexists(target):
            _transfer(move, src, target)
        elif not overwrite:
            return {
                "error": f"A file or folder {src.name} already exists",
                "exists": True,
            }
        elif move:
            _replace(target, _rename_or_copy, src)
        else:
            _replace(target, _transfer, False, src)
    return {"ok": True, "path": str(Path(dest_dir) / target.name)}


def _transfer(move: bool, src: Path, target: Path) -> None:
    """Move or copy *src* to the not-yet-existing *target* (symlinks as links)."""
    if move:
        _finish_move(_rename_or_copy(src, target))
    else:
        staging = tempfile.TemporaryDirectory(prefix=".kiss-copy-", dir=target.parent)
        with staging as name, _opened(name) as folder:
            staged = folder / "entry"
            _copy_entry(src, staged, src.is_dir() and not src.is_symlink())
            _publish(staged, target)


def _is_closed_folder(path: Path) -> bool:
    """Whether *path* is a folder, not a link to one, whose mode closes it to writing.

    Such a folder changes its parent by no rename (its ``..`` would
    change), and no entry of it can be deleted.  Root, whom no mode
    binds, finds no folder closed; nor does a read-only filesystem
    close one (what is moved from it is still copied, see
    :func:`_rename_or_copy`).
    """
    mode = os.lstat(path).st_mode
    return stat.S_ISDIR(mode) and not mode & stat.S_IWUSR and not os.access(path, os.W_OK)


def _publish(staged: Path, target: Path) -> None:
    """Rename the complete copy *staged* to the free name *target* in the folder above its own.

    The copy of a folder closed to writing (:func:`_is_closed_folder`) is
    as closed as the folder: it is opened to its owner for the rename and
    closed again after it, through a descriptor of it where folders are
    pinned (:func:`_opened`), so a link put at either name is not followed.
    """
    if not _is_closed_folder(staged):
        _rename_noreplace(staged, target)
        return
    with _opened(staged) as copy:
        mode = stat.S_IMODE(os.stat(copy).st_mode)
        os.chmod(copy, mode | stat.S_IWUSR)
        _rename_noreplace(staged, target)
        os.chmod(copy if _PINS else target, mode)


def _copy_entry(src: Path, dst: Path, is_folder: bool) -> None:
    """Copy *src*, a folder with all below it or else a file or a link, to the new *dst*.

    An entry that is at *dst* already (put into the copy being prepared, or
    the same name in another case) fails the copy as any other error does:
    a ``FileExistsError`` would offer the destination, which is not taken,
    for replacement.
    """
    try:
        (_copy_folder if is_folder else _copy_file)(src, dst)
    except FileExistsError as exc:
        raise OSError(str(exc)) from exc


def _copy_folder(src: str | Path, dst: str | Path) -> None:
    """Copy the folder *src* and all below it to the new folder *dst*, like ``copytree``.

    ``shutil.copytree`` copies a tree by name: a folder below *src* that an
    external writer swaps for a link once it was listed is followed out of
    the workspace, and so is a folder of the copy.  Here every folder, read
    or written, is opened in the one above it and refused if it is a link
    (:func:`_opened`), and what it holds is copied through that descriptor.
    Where folders are not pinned (:func:`_pinned`) the tree is copied by name.
    """
    if not _PINS:
        shutil.copytree(src, dst, symlinks=True, copy_function=_copy_file)
        return
    os.mkdir(dst)
    with _opened(src) as source, _opened(dst) as copy:
        with os.scandir(source) as listing:
            entries = list(listing)
        for entry in entries:
            is_folder = entry.is_dir(follow_symlinks=False)
            _copy_entry(source / entry.name, copy / entry.name, is_folder)
        shutil.copystat(source, copy)


def _copy_file(src: str | Path, dst: str | Path) -> None:
    """Copy *src*, a file or a link (as a link), to *dst* with its metadata, like ``copy2``.

    ``shutil.copy2`` looks whether *src* is a pipe and then opens it by
    name: a pipe an external writer puts there in between is opened all
    the same, and without a writer that open never returns -- under
    :data:`_mutation_lock`, so every later action waits behind it.  Here
    the file is opened without waiting and without following a link, and
    read only if what was opened is a regular file.  *dst* is created, so
    a link put where it goes is not written through.  The times and
    permissions go from the file that was read to the file that was
    written, by descriptor (:func:`_named`): a link put at either name
    since is not followed.
    """
    if os.name == "nt":  # no pipe can take a file's place there
        shutil.copy2(src, dst, follow_symlinks=False)
        return
    if os.path.islink(src):
        os.symlink(os.readlink(src), dst)
        shutil.copystat(src, dst, follow_symlinks=False)
        return
    fd = os.open(src, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            what = "a named pipe" if stat.S_ISFIFO(mode) else "not a regular file"
            raise shutil.SpecialFileError(f"`{src}` is {what}")
        with open(fd, "rb", closefd=False) as fsrc, open(dst, "xb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
            fdst.flush()  # a write after this would change the times set now
            if _PINS:
                with _named(os.dup(fd), src) as old, _named(os.dup(fdst.fileno()), dst) as new:
                    shutil.copystat(old, new)
            else:  # macOS: a descriptor has no name there, see _pinned
                shutil.copystat(src, dst, follow_symlinks=False)
    finally:
        os.close(fd)


def _rename_or_copy(src: Path, target: Path) -> Path | OSError | None:
    """Rename, or copy an owned source; return its cleanup path or deferred error.

    Not ``shutil.move``, which deletes the source it had to copy at once:
    under :func:`_replace` a failed deletion would roll back *target*, by
    then the only complete copy.  The copied source is deleted by
    :func:`_finish_move`, once *target* is committed.  Until then it has
    a name of the move's own, so that what is deleted is what was copied
    and not an entry saved at *src* since.  The name is in the source's
    own folder, which the caller pinned: a folder made for it is reached
    by name, and a link put in that folder's place had its entry copied
    and deleted.

    A folder closed to writing (:func:`_is_closed_folder`) moves by that
    rename or not at all: copied, it would be left under the move's name
    beside its copy, since its entries cannot be deleted from it.
    """
    try:
        _rename_noreplace(src, target)
        return None
    except FileExistsError:
        raise
    except OSError as exc:
        if _is_closed_folder(src):
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(src)) from exc
    claimed = src.with_name(f".kiss-move-{uuid.uuid4().hex}")
    try:
        os.rename(src, claimed)
    except OSError as exc:
        # Keep the historical copy-on-rename-denial behavior, but never attempt
        # to delete a pathname we could not claim. Report after committing the copy.
        _transfer(False, src, target)
        return exc
    try:
        _transfer(False, claimed, target)
    except BaseException:
        if os.path.lexists(claimed):
            try:
                _rename_noreplace(claimed, src)
            except OSError as exc:
                raise OSError(f"Source changed; retained entry at {claimed}: {exc}") from exc
        raise
    return claimed


def _finish_move(copied: Path | OSError | None) -> None:
    """Clean only an owned copied source, after the destination is committed."""
    if isinstance(copied, OSError):
        raise copied
    if copied is not None:
        _remove(copied)


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
    # git config would otherwise change it.  ``core.quotePath=false``
    # lists a non-ASCII file name as it is spelled, not octal-escaped.
    cmd = [
        "git", "-c", "core.quotePath=false", "grep", "--no-index", "--exclude-standard",
        "--no-color", "--no-column", "--no-full-name", "-n", "-I", "-F", "-e", query, "--", ".",
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
    # Only the terminator git writes ends a row (``splitlines`` would also cut a matched
    # line at a form feed, U+2028 or a lone CR); a stream that was not cut ends with one.
    lines = proc.stdout.decode("utf-8", "replace").split("\n")
    if not lines[-1]:
        del lines[-1]
    truncated = proc.truncated or len(lines) > FIND_MAX_MATCHES
    kept = [
        (line[2:] if line.startswith("./") else line).removesuffix("\r")
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
            [  # core.quotePath=false: as in find_in_folder
                "git", "-c", "core.quotePath=false", "diff", "--no-color", "--no-ext-diff",
                "--no-index", "--", a, b,
            ],
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
