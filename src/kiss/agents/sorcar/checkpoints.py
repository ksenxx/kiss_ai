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
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from kiss.core import tool_interrupt
from kiss.core.processes import popen_process_group, terminate_process_group

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
    stdin: str = "",
    stop_event: threading.Event | None = None,
) -> str:
    """Return the stdout of ``git -C root args``; raise :class:`CheckpointError` on failure.

    Input and output are file-name bytes (``os.fsencode``/``os.fsdecode``):
    text mode would read a carriage return in a ``-z`` path as a newline.
    They pass through temporary files, so the wait for git can look at
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
        stdin: Text fed to the command.
        stop_event: The task's stop event, or ``None``.

    Raises:
        CheckpointError: When git fails, times out or is stopped.
        ToolCallInterrupted: When the tool panel's Stop was pressed.
    """
    _raise_if_stopped(stop_event, f"git {args[0]}")
    env = None if index is None else {**os.environ, "GIT_INDEX_FILE": str(index)}
    with (tempfile.TemporaryFile() as given, tempfile.TemporaryFile() as out,
          tempfile.TemporaryFile() as err):
        given.write(os.fsencode(stdin))
        given.seek(0)
        try:
            process = popen_process_group(
                ["git", "-C", str(root), *args], stdin=given, stdout=out, stderr=err, env=env,
            )
        except OSError as error:
            raise CheckpointError(f"git {args[0]} failed: {error}") from error
        try:
            ended = _wait_for_git(process, stop_event)
        except BaseException:
            if process.poll() is None:
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
    git = _git(Path(work_dir), "rev-parse", "--show-toplevel", stop_event=stop_event)
    root = Path(git.rstrip("\n"))
    with tempfile.TemporaryDirectory(prefix="kiss-checkpoint-") as scratch:
        index = Path(scratch) / "index"
        git_path = _git(root, "rev-parse", "--git-path", "index", stop_event=stop_event)
        real_index = root / git_path.rstrip("\n")
        if real_index.is_file():
            shutil.copy2(real_index, index)
            _clear_skip_bits(root, index, stop_event)
        _git(root, "add", "--all", "--", ".", index=index, stop_event=stop_event)
        tree = _git(root, "write-tree", index=index, stop_event=stop_event).strip()
    return Snapshot(root, tree)


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
) -> tuple[list[str], list[str]]:
    """Return what making *target*'s working tree match it again rewrites and deletes.

    Nothing is changed.  Submodules (and embedded repositories) are
    left alone: a snapshot holds only the commit they point at, not
    their files.  A file to write back that would replace something
    neither snapshot holds (see :func:`_obstacle`) stops the plan.

    Args:
        target: The snapshot to return to.
        current: A snapshot of the same working tree taken just before.
        keep: A directory never removed (the agent's work dir).
        stop_event: The task's stop event, which ends a running git command.

    Returns:
        The paths to rewrite from *target* and the paths to delete,
        relative to the working tree's top level.

    Raises:
        CheckpointError: When git cannot compare the trees or something
            no snapshot holds is in the way.
    """
    out = _git(target.root, "diff-tree", "-r", "-z", "--no-renames", "--raw",
               target.tree, current.tree, stop_event=stop_event)
    fields = out.split("\0")
    rewrite: list[str] = []
    missing: list[str] = []
    delete: list[str] = []
    # Each change is ":<old mode> <new mode> <old sha> <new sha> <status>" then its path.
    for meta, path in zip(fields[0::2], fields[1::2], strict=False):
        old_mode, new_mode, *_, status = meta.lstrip(":").split()
        if _GITLINK in (old_mode, new_mode):
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
    return rewrite, delete


def apply_restore(
    target: Snapshot,
    rewrite: list[str],
    delete: list[str],
    keep: Path,
    stop_event: threading.Event | None = None,
) -> None:
    """Carry out a :func:`plan_restore`: delete *delete*, then write *rewrite* from *target*.

    Directories left empty by a deletion are removed, except *keep*.
    A Stop ends the restore before its next deletion or git command.

    Args:
        target: The snapshot to return to.
        rewrite: The paths to write back from it.
        delete: The paths to delete.
        keep: A directory never removed (the agent's work dir).
        stop_event: The task's stop event, which ends a running git
            command and the restore.

    Raises:
        CheckpointError: When a file cannot be deleted, git cannot
            write the files or a Stop ends the restore.
    """
    keep = keep.resolve()
    for path in delete:
        _raise_if_stopped(stop_event, f"deleting {path}")
        _delete(target.root, path, keep)
    if rewrite:
        with tempfile.TemporaryDirectory(prefix="kiss-rollback-") as scratch:
            index = Path(scratch) / "index"
            _git(target.root, "read-tree", target.tree, index=index, stop_event=stop_event)
            _git(target.root, "checkout-index", "--force", "-z", "--stdin", index=index,
                 stdin="".join(f"{path}\0" for path in rewrite), stop_event=stop_event)


def _obstacle(root: Path, path: str, deleted: set[str], keep: Path) -> tuple[str, str] | None:
    """Return ``(obstacle, why)`` when writing *path* back would destroy what no snapshot holds.

    ``checkout-index --force`` clears whatever is in a written file's
    way: a directory at its path, with everything in it, or a file where
    one of its directories goes.  That is safe only for the files the
    rollback deletes anyway (the current snapshot holds them); an ignored
    file, or the work dir, in the way is an obstacle.

    Args:
        root: The working tree's top level.
        path: A file the target snapshot has and the current one lacks.
        deleted: The files the rollback deletes.
        keep: The agent's work dir (resolved), which is never removed.
    """
    parts = path.split("/")
    for depth in range(1, len(parts)):
        leading = "/".join(parts[:depth])
        if _is_file(root / leading) and leading not in deleted:
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

    Raises:
        CheckpointError: When the file cannot be deleted.
    """
    target = root / path
    try:
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
    """The checkpoints of one task, by name: what the ``checkpoint``/``rollback`` tools use."""

    def __init__(self, stop_event: threading.Event | None = None) -> None:
        """Start with no checkpoints.

        Args:
            stop_event: The task's stop event, which ends a running git command.
        """
        self.saved: dict[str, Snapshot] = {}
        self.stop_event = stop_event

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
        except CheckpointError as error:
            return f"Error: checkpoint {name!r} not saved: {error}"
        self.saved[name] = saved
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
            rewritten, deleted = plan_restore(target, current, keep, self.stop_event)
            self.saved[BEFORE_ROLLBACK] = current
            apply_restore(target, rewritten, deleted, keep, self.stop_event)
        except CheckpointError as error:
            return f"Error: rollback to {name!r} failed: {error}"
        return (f"Rolled {target.root} back to checkpoint {name!r}: rewrote "
                f"{_listing(rewritten)}; deleted {_listing(deleted)}. The replaced state "
                f"is checkpoint {BEFORE_ROLLBACK!r}.")
