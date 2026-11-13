# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Named snapshots of the git working tree the agent works in, and their restore.

The host counterpart of the coding SEA's ``checkpoint``/``rollback``
tools, which archive a container's workdir.  A snapshot is a git tree
object holding every file git does not ignore under the working tree's
top level (tracked, changed and untracked alike), written through a
private copy of the index: HEAD, the branch, the real index and the
stash are never touched, and nothing is committed.  A restore makes
those files match the snapshot again: files changed or deleted since
are rewritten from it, files created since are deleted, and ignored
files (build output, ``tmp/``, virtualenvs) are left as they are.
A saved snapshot is the target of a ref of its own under
``refs/kiss-checkpoints/``, which keeps ``git gc`` from collecting it
for as long as its checkpoint lasts.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from kiss.agents.sorcar.git_worktree import _REPO_SCOPED_GIT_ENV
from kiss.core import tool_interrupt
from kiss.core.processes import start_process_group, terminate_process_group

CHECKPOINT_NAME = re.compile(r"[A-Za-z0-9_-]{1,40}")
"""The labels ``checkpoint`` accepts (the coding SEA's rule)."""
BEFORE_ROLLBACK = "before-rollback"
"""The checkpoint ``rollback`` saves of the state it replaces."""
GIT_TIMEOUT_SECONDS = 300
"""Deadline of one git command (hashing a large untracked tree takes a while)."""
_POLL_SECONDS = 0.2
"""How often the wait for a git command looks at the Stops."""
_KILL_GRACE_SECONDS = 2.0
"""How long a stopped git command may take to clean up after ``SIGTERM``."""
_LISTED_PATHS = 10
_GITLINK = "160000"
"""The tree-entry mode of a submodule (or an embedded repository)."""
_REFS = "refs/kiss-checkpoints"
"""Where the refs that keep saved snapshots from ``git gc`` are, a directory per task."""
_DIR_FD = (
    hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "fwalk")
    and {os.open, os.unlink, os.rmdir, os.rename} <= os.supports_dir_fd
)
"""Whether deletions and renames can go through directory descriptors (not on native Windows)."""


class CheckpointError(Exception):
    """A git command a snapshot or a restore needs failed; the message says why."""


@dataclass(frozen=True)
class Snapshot:
    """The non-ignored files of one git working tree at one moment.

    Attributes:
        root: The working tree's top level.
        tree: The git tree object holding the files.
    """

    root: Path
    tree: str


def _git(
    root: Path,
    *args: str,
    index: Path | None = None,
    is_git_dir: bool = False,
    stdin: str = "",
    stop_event: threading.Event | None = None,
) -> str:
    """Return the stdout of ``git -C root args``; raise :class:`CheckpointError` on failure.

    Input and output are file-name bytes (``os.fsencode``/``os.fsdecode``):
    text mode would read a carriage return in a ``-z`` path as a newline.
    The repo-scoping variables of the environment (``GIT_DIR``,
    ``GIT_WORK_TREE``, ``GIT_INDEX_FILE``, ...) are dropped, as every
    git runner of :mod:`kiss.agents.sorcar.git_worktree` drops them: a
    daemon started with them set would otherwise snapshot and restore
    another checkout than *root*.  They pass through temporary files,
    so the wait for git can look at
    the Stops every :data:`_POLL_SECONDS` without a pipe filling up:
    the task's *stop_event* or the tool panel's Stop ends git's process
    group at once, not after :data:`GIT_TIMEOUT_SECONDS`, and so does an
    exception injected into the wait.  A Stop pressed before the command
    starts keeps it from starting: a snapshot or restore made of fast
    commands ends at its next one.

    Args:
        root: The directory git runs in.
        args: The git subcommand and its arguments.
        index: A private index file (``GIT_INDEX_FILE``) instead of the real one.
        is_git_dir: Whether *root* is the repository's git directory itself:
            git is then told so (``GIT_DIR``) rather than left to find out,
            which ``safe.bareRepository=explicit`` forbids in a bare one.
        stdin: Text fed to the command.
        stop_event: The task's stop event, or ``None``.

    Raises:
        CheckpointError: When git fails, times out or is stopped.
        ToolCallInterrupted: When the tool panel's Stop was pressed.
    """
    _raise_if_stopped(stop_event, f"git {args[0]}")
    env = {k: v for k, v in os.environ.items() if k not in _REPO_SCOPED_GIT_ENV}
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    if is_git_dir:
        env["GIT_DIR"] = str(root)
    with (tempfile.TemporaryFile() as given, tempfile.TemporaryFile() as out,
          tempfile.TemporaryFile() as err):
        given.write(os.fsencode(stdin))
        given.seek(0)
        # The Popen is held from before the spawn, which runs inside the
        # ``try``: a Stop raised as the spawning call returns reaches the
        # handler with git in hand (see :func:`start_process_group`).
        process: subprocess.Popen[bytes] = object.__new__(subprocess.Popen)
        try:
            try:
                start_process_group(
                    process, ["git", "-C", str(root), *args],
                    stdin=given, stdout=out, stderr=err, env=env,
                )
            except OSError as error:
                raise CheckpointError(f"git {args[0]} failed: {error}") from error
            ended = _wait_for_git(process, stop_event)
        except BaseException:
            if getattr(process, "pid", None) is not None and process.poll() is None:
                terminate_process_group(process, _KILL_GRACE_SECONDS)
            raise
        if ended:
            tool_interrupt.raise_if_interrupted()
            raise CheckpointError(f"git {args[0]} {ended}")
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    if process.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        raise CheckpointError(message or f"git {args[0]} failed")
    return os.fsdecode(stdout)


def _wait_for_git(process: subprocess.Popen[bytes], stop_event: threading.Event | None) -> str:
    """Wait for *process* to exit; return ``""``, or why it was ended instead.

    A Stop (the task's *stop_event*, the tool panel's interrupt event)
    or :data:`GIT_TIMEOUT_SECONDS` ends its process group.
    """
    deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
    while True:
        try:
            process.wait(timeout=_POLL_SECONDS)
            return ""
        except subprocess.TimeoutExpired:
            pass
        if _stop_pressed(stop_event):
            ended = "was stopped"
        elif time.monotonic() >= deadline:
            ended = f"timed out after {GIT_TIMEOUT_SECONDS}s"
        else:
            continue
        terminate_process_group(process, _KILL_GRACE_SECONDS)
        return ended


def _stop_pressed(stop_event: threading.Event | None) -> bool:
    """Return whether the task's *stop_event* or the tool panel's Stop of this call is set."""
    interrupt = tool_interrupt.current_tool_interrupt_event()
    return (stop_event is not None and stop_event.is_set()) or (
        interrupt is not None and interrupt.is_set())


def _raise_if_stopped(stop_event: threading.Event | None, step: str) -> None:
    """Keep *step* (``"git add"``, ``"deleting a.txt"``) from starting once a Stop was pressed.

    Raises:
        ToolCallInterrupted: For the tool panel's Stop.
        CheckpointError: ``"<step> was stopped"``, for the task's Stop.
    """
    if _stop_pressed(stop_event):
        tool_interrupt.raise_if_interrupted()
        raise CheckpointError(f"{step} was stopped")


def snapshot(work_dir: str, stop_event: threading.Event | None = None) -> Snapshot:
    """Record the non-ignored files of the git working tree holding *work_dir*.

    The real index is copied first: its stat data spares ``git add``
    re-hashing the files that did not change, and its entries keep
    tracked files that match an ignore pattern in the snapshot.  The
    copy keeps the real index's mtime (``copy2``): git re-reads a file
    whose recorded mtime is in the same second as the index's
    ("racy git"), which is how it catches a rewrite that kept the
    file's size and mtime second.  A copy stamped with the current time
    ended that protection for the files written right after the real
    index was: a same-size rewrite of such a file within that second,
    snapshotted in a later second, was taken for its indexed content.

    Args:
        work_dir: A directory inside a git working tree.
        stop_event: The task's stop event, which ends a running git command.

    Raises:
        CheckpointError: When *work_dir* is not in a git working tree, git
            fails or a Stop ends it.
    """
    # Only git's record terminator goes: ``rstrip`` would also take the
    # newline a root named ``project\n`` ends in, naming ``project``.
    git = _git(Path(work_dir), "rev-parse", "--show-toplevel", stop_event=stop_event)
    root = Path(git.removesuffix("\n"))
    with tempfile.TemporaryDirectory(prefix="kiss-checkpoint-") as scratch:
        index = Path(scratch) / "index"
        git_path = _git(root, "rev-parse", "--git-path", "index", stop_event=stop_event)
        real_index = root / git_path.removesuffix("\n")
        if real_index.is_file():
            shutil.copy2(real_index, index)
            _clear_skip_bits(root, index, stop_event)
        add = ("add", "--all", "--", ".")
        try:
            _git(root, *add, index=index, stop_event=stop_event)
        except CheckpointError:
            # One embedded repository with no commit checked out fails the
            # whole ``git add``.  Its files are left alone like any other
            # embedded repository's, and the rest of the tree is saved.
            try:
                unborn = _unborn_repositories(root, index, stop_event)
            except CheckpointError:
                unborn = []  # a Stop, or the listing failed too: the error stays ``git add``'s
            if not unborn:
                raise
            _git(root, *add, *(f":(exclude,literal){path}" for path in unborn),
                 index=index, stop_event=stop_event)
        tree = _git(root, "write-tree", index=index, stop_event=stop_event).strip()
    return Snapshot(root, tree)


def _unborn_repositories(
    root: Path, index: Path, stop_event: threading.Event | None,
) -> list[str]:
    """Return the untracked embedded repositories under *root* with no commit checked out.

    ``git add`` records an embedded repository as the commit it has
    checked out and refuses the whole tree for one that has none (a
    fresh ``git init``).  Git lists a repository it does not look into
    as one untracked entry ending in ``/``.
    """
    others = _git(root, "ls-files", "--others", "--exclude-standard", "-z", index=index,
                  stop_event=stop_event)
    unborn: list[str] = []
    for path in others.split("\0"):
        if path.endswith("/"):
            try:
                _git(root / path, "rev-parse", "--verify", "--quiet", "HEAD",
                     stop_event=stop_event)
            except CheckpointError:
                unborn.append(path.removesuffix("/"))
    return unborn


def _clear_skip_bits(root: Path, index: Path, stop_event: threading.Event | None) -> None:
    """Clear the assume-unchanged and skip-worktree bits of the entries of *index*.

    ``git add`` trusts an entry with either bit instead of reading its
    working file, so the snapshot would hold the index's copy of a file
    the agent changed.  ``update-index`` clears one kind of bit per call.
    """
    entries = _git(root, "ls-files", "-v", "-z", index=index, stop_event=stop_event).split("\0")
    # The tag is lowercase for assume-unchanged, "S" for skip-worktree.
    flagged = "".join(f"{entry[2:]}\0" for entry in entries
                      if entry[:1].islower() or entry[:1] == "S")
    if flagged:
        for option in ("--no-assume-unchanged", "--no-skip-worktree"):
            _git(root, "update-index", option, "-z", "--stdin", index=index, stdin=flagged,
                 stop_event=stop_event)


def plan_restore(
    target: Snapshot,
    current: Snapshot,
    keep: Path,
    stop_event: threading.Event | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return what making *target*'s working tree match it again rewrites and deletes.

    Nothing is changed.  Submodules (and embedded repositories) are
    left alone: a snapshot holds only the commit they point at (nothing
    of one that has no commit yet), not their files.  A file to write
    back where something neither snapshot holds is in its way (see
    :func:`_obstacle`) stops the plan.

    Args:
        target: The snapshot to return to.
        current: A snapshot of the same working tree taken just before.
        keep: A directory never removed (the agent's work dir).
        stop_event: The task's stop event, which ends a running git command.

    Returns:
        The paths to rewrite from *target*, those of them *current* has
        no file at, and the paths to delete, relative to the working
        tree's top level.

    Raises:
        CheckpointError: When git cannot compare the trees or something
            no snapshot holds is in the way.
    """
    out = _git(target.root, "diff-tree", "-r", "-z", "--no-renames", "--raw",
               target.tree, current.tree, stop_event=stop_event)
    fields = out.split("\0")
    # Each change is ":<old mode> <new mode> <old sha> <new sha> <status>" then its path.
    changes = [(meta.lstrip(":").split(), path)
               for meta, path in zip(fields[0::2], fields[1::2], strict=False)]
    # An embedded repository on either side: neither its gitlink nor the
    # files at its path (the repository's own, held by no snapshot: ones
    # vendored plainly since, or a repository embedded over plain files)
    # are touched.
    gitlinks = tuple(f"{path}/" for (old_mode, new_mode, *_), path in changes
                     if _GITLINK in (old_mode, new_mode))
    rewrite: list[str] = []
    missing: list[str] = []
    delete: list[str] = []
    for (old_mode, new_mode, *_, status), path in changes:
        if _GITLINK in (old_mode, new_mode) or path.startswith(gitlinks):
            continue
        (delete if status == "A" else rewrite).append(path)
        if status == "D":
            missing.append(path)
    keep, deleted = keep.resolve(), set(delete)
    obstacles: dict[str, str] = {}
    for path in missing:
        found = _obstacle(target.root, path, deleted, keep)
        if found is not None:
            obstacles.setdefault(*found)
    if obstacles:
        reasons = list(obstacles.values())
        if len(reasons) > _LISTED_PATHS:
            reasons[_LISTED_PATHS:] = [f"and {len(reasons) - _LISTED_PATHS} more"]
        raise CheckpointError(
            f"nothing was changed: {'; '.join(reasons)}. A rollback replaces only files "
            "a checkpoint holds: move these away first.")
    return rewrite, missing, delete


def apply_restore(
    target: Snapshot,
    rewrite: list[str],
    missing: list[str],
    delete: list[str],
    keep: Path,
    stop_event: threading.Event | None = None,
) -> None:
    """Carry out a :func:`plan_restore`: delete *delete*, then write *rewrite* from *target*.

    Directories left empty by a deletion are removed, except *keep*.
    Nothing but what the current snapshot holds is removed: the files
    of *delete*, the file at each path of *rewrite* not in *missing*
    (see :func:`_replace`), and empty directories.  Git never writes
    with ``--force``, which clears a directory with all it holds: what
    came into a file's way after the plan (an ignored file a build
    wrote meanwhile) stays and fails the restore.  A Stop ends the
    restore before its next deletion, replacement or git command.

    Args:
        target: The snapshot to return to.
        rewrite: The paths to write back from it.
        missing: Those of *rewrite* the current snapshot has no file at.
        delete: The paths to delete.
        keep: A directory never removed (the agent's work dir).
        stop_event: The task's stop event, which ends a running git
            command and the restore.

    Raises:
        CheckpointError: When a file cannot be deleted, git cannot
            write the files or a Stop ends the restore.
    """
    keep, root = keep.resolve(), target.root
    for path in delete:
        _raise_if_stopped(stop_event, f"deleting {path}")
        _delete(root, path, keep)
    if rewrite:
        absent = set(missing)
        with tempfile.TemporaryDirectory(prefix="kiss-rollback-") as scratch:
            index = Path(scratch) / "index"
            _git(root, "read-tree", target.tree, index=index, stop_event=stop_event)
            held = [path for path in rewrite if path not in absent]
            write = missing + _replace(root, held, index, stop_event)
            for path in missing:
                _remove_empty_directories(root, path, stop_event)
            if write:
                _checkout(root, index, write, stop_event)


def _checkout(
    root: Path, index: Path, paths: list[str], stop_event: threading.Event | None,
    *options: str,
) -> None:
    """Have git write the files *paths* of *index* where nothing is (no ``--force``)."""
    _git(root, "checkout-index", *options, "-z", "--stdin", index=index,
         stdin="".join(f"{path}\0" for path in paths), stop_event=stop_event)


def _replace(
    root: Path, paths: list[str], index: Path, stop_event: threading.Event | None,
) -> list[str]:
    """Put the files *paths* of *index* in place of the files there; return those left to write.

    Git writes them into a directory of their own first (:func:`_staged`),
    then one ``rename`` moves each over its old file: the path holds the
    old file or the new one at any moment, a Stop included, and a
    directory put there since the plan is not replaced.  A file that
    cannot be moved (nothing was staged, its directory is gone, the git
    directory is on another filesystem) has its old file deleted and
    is returned, for git to write where nothing is.
    """
    if not paths:
        return []
    unmoved: list[str] = []
    with _staged(root, paths, index, stop_event) as stage:
        for path in paths:
            _raise_if_stopped(stop_event, f"rewriting {path}")
            if stage is None or not _rename_over(stage / path, root, path):
                _delete(root, path, (root / path).parent)  # its directories all stay
                unmoved.append(path)
    return unmoved


@contextlib.contextmanager
def _staged(
    root: Path, paths: list[str], index: Path, stop_event: threading.Event | None,
) -> Iterator[Path | None]:
    """Have git write the files *paths* of *index* into a directory of their own; yield it.

    The directory is in the git directory (out of the working tree and,
    almost always, on its filesystem) and is removed at the end.
    ``None`` is yielded, and nothing written, where no file can be moved
    over another through a directory descriptor (native Windows) or the
    git directory takes no new directory (it is read-only).
    """
    stage: tempfile.TemporaryDirectory[str] | None = None
    if _DIR_FD:  # pragma: no branch — false on native Windows only
        git_dir = _git(root, "rev-parse", "--absolute-git-dir", stop_event=stop_event)
        with contextlib.suppress(OSError):
            stage = tempfile.TemporaryDirectory(
                prefix="kiss-rollback-", dir=git_dir.removesuffix("\n"),
                ignore_cleanup_errors=True)
    if stage is None:
        yield None
        return
    with stage as directory:
        _checkout(root, index, paths, stop_event, f"--prefix={directory}/")
        yield Path(directory)


def _rename_over(staged: Path, root: Path, path: str) -> bool:
    """Move *staged* over the file *path* under *root*; return whether it was moved.

    Through descriptors, as in :func:`_delete`.  A ``rename`` replaces a
    file and never a directory.
    """
    *dirs, name = path.split("/")
    with contextlib.suppress(OSError), _directories(root, dirs) as fds:
        os.rename(staged, name, dst_dir_fd=fds[-1])
        return True
    return False


def _obstacle(root: Path, path: str, deleted: set[str], keep: Path) -> tuple[str, str] | None:
    """Return ``(obstacle, why)`` when something no snapshot holds is where *path* goes back.

    The restore clears a written file's way of the files the rollback
    deletes (the current snapshot holds them) and of empty directories,
    and of nothing else: an ignored file, or the work dir, at its path
    or where one of its directories goes would fail the restore
    half-way, so it is an obstacle that stops the rollback before it
    changes anything.

    Args:
        root: The working tree's top level.
        path: A file the target snapshot has and the current one lacks.
        deleted: The files the rollback deletes.
        keep: The agent's work dir (resolved), which is never removed.
    """
    parts = path.split("/")
    for depth in range(1, len(parts)):
        leading = "/".join(parts[:depth])
        if _is_file(root / leading):
            if leading in deleted:
                # The rollback deletes it first: what a link there leads to
                # is neither in the way nor the working tree's.
                return None
            return leading, (f"{leading} is now a file git ignores, where the checkpoint "
                             f"has the directory of {path}")
    where = root / path
    if _is_file(where):
        return path, f"{path} is now a file git ignores"
    if not where.is_dir():
        return None
    if where == keep or where in keep.parents:
        return path, f"{path} is now a directory holding the work dir"
    for dirpath, dirnames, filenames in os.walk(where):
        links = [name for name in dirnames if os.path.islink(os.path.join(dirpath, name))]
        for name in filenames + links:
            inside = Path(dirpath, name).relative_to(root).as_posix()
            if inside not in deleted:
                return path, f"{path} is now a directory holding {inside}, which git ignores"
    return None


def _is_file(path: Path) -> bool:
    """Whether something other than a directory is at *path* (a symlink counts)."""
    return path.is_symlink() or (path.exists() and not path.is_dir())


def _delete(root: Path, path: str, keep: Path) -> None:
    """Delete the file *path* under *root* and the directories it leaves empty, but not *keep*.

    Each directory on the way is opened from its parent's descriptor
    without following a link, and the file and the directories are
    removed through those descriptors: a directory swapped for a link
    since the rollback was planned fails the deletion instead of
    taking it to a file outside the working tree.

    Raises:
        CheckpointError: When the file cannot be deleted.
    """
    if not _DIR_FD:  # pragma: no cover — Windows-only branch
        _delete_by_name(root, path, keep)
        return
    *dirs, name = path.split("/")
    try:
        with _directories(root, dirs) as fds:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(name, dir_fd=fds[-1])
            with contextlib.suppress(OSError):
                for depth in range(len(dirs), 0, -1):
                    if root.joinpath(*dirs[:depth]) == keep:
                        break
                    os.rmdir(dirs[depth - 1], dir_fd=fds[depth - 1])
    except FileNotFoundError:
        return  # a directory of the file is gone, and the file with it
    except OSError as error:
        raise CheckpointError(f"cannot delete {path}: {error}") from error


@contextlib.contextmanager
def _directories(root: Path, dirs: list[str]) -> Iterator[list[int]]:
    """Open *root* and each of *dirs* below it, from its parent's descriptor, following no link.

    Yields the descriptors, *root*'s first; they are closed at the end.
    """
    # For its path alone where that can be asked (Linux): a deletion
    # searches and writes a directory, and needs no leave to read it.
    flags = getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY | os.O_NOFOLLOW
    fds: list[int] = []
    try:
        fds.append(os.open(root, flags))
        for part in dirs:
            fds.append(os.open(part, flags, dir_fd=fds[-1]))
        yield fds
    finally:
        for fd in fds:
            os.close(fd)


def _remove_empty_directories(
    root: Path, path: str, stop_event: threading.Event | None,
) -> None:
    """Remove the directory at *path* under *root* when it holds nothing but directories.

    What the deletions leave of a directory where a file goes back: git
    tracks no empty directory, so no snapshot holds one and no deletion
    pruned it.  Only ``rmdir`` is used, through descriptors as in
    :func:`_delete`: a directory that holds anything else stays, and so
    does what is at *path* in place of a directory, where git then
    refuses to write the file.

    Raises:
        CheckpointError: When a Stop ends the removal, before its next directory.
    """
    if not _DIR_FD:  # pragma: no cover — Windows-only branch
        _remove_empty_directories_by_name(root, path, stop_event)
        return
    *dirs, name = path.split("/")
    with contextlib.suppress(OSError), _directories(root, dirs) as fds:
        for _where, subdirs, _files, fd in os.fwalk(name, topdown=False, dir_fd=fds[-1]):
            for subdir in subdirs:
                _raise_if_stopped(stop_event, f"rewriting {path}")
                os.rmdir(subdir, dir_fd=fd)
        _raise_if_stopped(stop_event, f"rewriting {path}")
        os.rmdir(name, dir_fd=fds[-1])


def _remove_empty_directories_by_name(  # pragma: no cover — Windows-only
    root: Path, path: str, stop_event: threading.Event | None,
) -> None:
    """:func:`_remove_empty_directories` by pathname, for native Windows.

    Each directory is checked to be no link, as in :func:`_delete_by_name`,
    before it is listed and again right before it is removed.  A link at
    *path*, in it or on the way to it, or a file in it, found while
    listing ends the removal before anything is removed; a link put in
    afterwards ends it at the next directory, and only one put in
    between that last check and the removal is still followed.
    """
    found: list[Path] = []
    pending = [root / path]
    with contextlib.suppress(OSError):
        while pending:
            directory = pending.pop()
            _refuse_links(root, directory)
            found.append(directory)
            pending.extend(directory.iterdir())
        for directory in reversed(found):
            _raise_if_stopped(stop_event, f"rewriting {path}")
            _refuse_links(root, directory)
            directory.rmdir()


def _refuse_links(root: Path, directory: Path) -> None:  # pragma: no cover — Windows-only
    """Raise ``NotADirectoryError`` when *directory*, or one it is in below *root*, is a link."""
    for step in (directory, *directory.parents):
        if step == root:
            break
        if step.is_symlink() or step.is_junction():
            raise NotADirectoryError(f"{step} is a link")


def _delete_by_name(root: Path, path: str, keep: Path) -> None:  # pragma: no cover — Windows-only
    """:func:`_delete` by pathname, for native Windows, where a directory cannot be opened.

    The directories on the way are checked to be no links (a symlink, a
    junction) right before the deletion; only one put there between
    that check and the deletion is still followed.
    """
    target = root / path
    try:
        _refuse_links(root, target.parent)
        target.unlink(missing_ok=True)
    except OSError as error:
        raise CheckpointError(f"cannot delete {path}: {error}") from error
    for parent in target.parents:
        if parent in (root, keep):
            break
        try:
            parent.rmdir()
        except OSError:
            break


def _listing(paths: list[str]) -> str:
    """Return ``N file(s) (a, b, ...)`` for a tool result, the list cut at :data:`_LISTED_PATHS`."""
    shown = ", ".join(paths[:_LISTED_PATHS])
    more = f" and {len(paths) - _LISTED_PATHS} more" if len(paths) > _LISTED_PATHS else ""
    noun = "file" if len(paths) == 1 else "files"
    return f"{len(paths)} {noun}" + (f" ({shown}{more})" if paths else "")


class Checkpoints:
    """The checkpoints of one task, by name: what the ``checkpoint``/``rollback`` tools use.

    A saved snapshot is a tree nothing in the repository refers to, which
    the next ``git gc`` would collect.  Each one is therefore the target
    of a ref of its own under :data:`_REFS`, in a directory of this
    instance's: made before the snapshot gets its name, deleted when
    another snapshot takes the name, and with all that are left by
    :meth:`close`.
    """

    def __init__(self, stop_event: threading.Event | None = None) -> None:
        """Start with no checkpoints.

        Args:
            stop_event: The task's stop event, which ends a running git command.
        """
        self.saved: dict[str, Snapshot] = {}
        self.stop_event = stop_event
        self._namespace = f"{_REFS}/{uuid.uuid4().hex}"
        # Checkpoint name -> the git directory and the ref keeping its snapshot.
        self._refs: dict[str, tuple[Path, str]] = {}
        # Every git directory a ref was asked for, noted before git is asked (a
        # Stop may end git after it made the ref): where close() looks for them.
        self._git_dirs: set[Path] = set()

    def _keep(self, name: str, kept: Snapshot) -> tuple[Path, str] | None:
        """Give *kept* a ref of its own, then make it checkpoint *name*.

        The ref is made through the repository's common git directory,
        which is still there when the working tree is not (a worktree
        ``commit_and_merge`` removed), so it can be deleted then too.
        It names the tree itself, not a commit of it: a walk of the
        history, ``git log --all`` included, meets nothing new.

        Returns:
            The git directory and the ref of the snapshot *name* stood
            for until now, for :meth:`_release`; ``None`` without one.

        Raises:
            CheckpointError: When git cannot make the ref or a Stop ends
                it; *name* then stands for what it did before.
        """
        common = _git(kept.root, "rev-parse", "--git-common-dir", stop_event=self.stop_event)
        git_dir = kept.root / common.removesuffix("\n")
        ref = f"{self._namespace}/{uuid.uuid4().hex}"
        self._git_dirs.add(git_dir)
        _git(git_dir, "update-ref", ref, kept.tree, is_git_dir=True, stop_event=self.stop_event)
        replaced = self._refs.get(name)
        self.saved[name], self._refs[name] = kept, (git_dir, ref)
        return replaced

    def _release(self, replaced: tuple[Path, str] | None) -> None:
        """Delete the ref of a snapshot no checkpoint stands for any more.

        A ref git does not delete now (a Stop, a repository that is gone,
        no descriptor left for git's files) is left to :meth:`close`: what
        the caller did stands, and is not failed by this.
        """
        if replaced is not None:
            git_dir, ref = replaced
            with contextlib.suppress(CheckpointError, OSError):
                _git(git_dir, "update-ref", "-d", ref, is_git_dir=True, stop_event=self.stop_event)

    def close(self) -> None:
        """Forget every checkpoint and delete the refs that kept them from ``git gc``.

        For the end of the task, whose Stop does not keep the refs from
        being deleted.  Refs git cannot delete (their repository is gone,
        no descriptor is left for git's files) stay behind, for another
        call: nothing is raised for them.
        """
        self.saved.clear()
        self._refs.clear()
        for git_dir in self._git_dirs:
            with contextlib.suppress(CheckpointError, OSError):
                refs = _git(git_dir, "for-each-ref", "--format=delete %(refname)",
                            self._namespace, is_git_dir=True)
                _git(git_dir, "update-ref", "--stdin", is_git_dir=True, stdin=refs)

    def save(self, name: str, work_dir: str) -> str:
        """Snapshot *work_dir*'s git working tree as *name*; return the tool result.

        Args:
            name: The label (:data:`CHECKPOINT_NAME`); a later save replaces it.
            work_dir: The agent's working directory.
        """
        if not CHECKPOINT_NAME.fullmatch(name or ""):
            return "Error: name must be letters, digits, - or _ (1 to 40 characters)."
        if name == BEFORE_ROLLBACK:
            return f"Error: {BEFORE_ROLLBACK!r} is saved by rollback itself; pick another name."
        try:
            saved = snapshot(work_dir, self.stop_event)
            replaced = self._keep(name, saved)
        except CheckpointError as error:
            return f"Error: checkpoint {name!r} not saved: {error}"
        self._release(replaced)
        short = saved.tree[:12]
        return (f"Saved checkpoint {name!r} of {saved.root} (git tree {short}; "
                f"`git show {short}:<path>` prints a file as it was).")

    def roll_back(self, name: str, work_dir: str) -> str:
        """Restore *work_dir*'s git working tree from checkpoint *name*; return the tool result.

        The state replaced is saved as :data:`BEFORE_ROLLBACK` before the
        first file changes; a rollback refused before that (something no
        checkpoint holds in the way) keeps the previous one.  A checkpoint
        of another working tree (taken before ``create_worktree`` moved the
        work dir) is refused rather than restored over the wrong files.

        Args:
            name: The label given to :meth:`save`.
            work_dir: The agent's working directory.
        """
        target = self.saved.get(name)
        if target is None:
            known = ", ".join(self.saved) or "none"
            return f"Error: unknown checkpoint {name!r}. Known: {known}."
        try:
            current = snapshot(work_dir, self.stop_event)
            if current.root != target.root:
                return (f"Error: checkpoint {name!r} holds {target.root}, but the work dir "
                        f"is now in {current.root}; take a new checkpoint there.")
            if current.tree == target.tree:
                return f"Nothing to roll back: the files already match checkpoint {name!r}."
            keep = Path(work_dir)
            rewritten, missing, deleted = plan_restore(target, current, keep, self.stop_event)
            replaced = self._keep(BEFORE_ROLLBACK, current)
            try:
                apply_restore(target, rewritten, missing, deleted, keep, self.stop_event)
            finally:
                # Not before: a rollback to BEFORE_ROLLBACK restores from the
                # very snapshot it replaces, which needs its ref until then.
                self._release(replaced)
        except CheckpointError as error:
            return f"Error: rollback to {name!r} failed: {error}"
        return (f"Rolled {target.root} back to checkpoint {name!r}: rewrote "
                f"{_listing(rewritten)}; deleted {_listing(deleted)}. The replaced state "
                f"is checkpoint {BEFORE_ROLLBACK!r}.")
