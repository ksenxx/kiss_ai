# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Git worktree operations and state.

Provides :class:`GitWorktree` (frozen dataclass for worktree state),
:class:`MergeResult` (outcome enum), and :class:`GitWorktreeOps`
(stateless helper with all git worktree operations).
"""

from __future__ import annotations

import _thread
import contextlib
import enum
import hashlib
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid
import weakref
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from kiss.core.file_lock import lock_exclusive, unlock
from kiss.core.html_to_markdown import html_to_markdown
from kiss.core.kiss_error import KISSError
from kiss.core.locked import _race_delay, locked_call
from kiss.core.processes import (
    _STOP_POLL_SECONDS,
    IS_WINDOWS,
    pid_alive,
    process_identity,
    run_capped,
    start_process_group,
    terminate_process_group,
)

logger = logging.getLogger(__name__)

#: The most seconds a taker waits for another PROCESS's hold on one of
#: the repository's cross-process locks (:func:`_file_lock`,
#: :func:`_reclaim_process_lock`).  A holder that wedged (a daemon
#: stuck in an un-stoppable merge) then costs the waiter a bounded
#: wait and a clear :class:`KISSError` instead of a silent hang.
PROCESS_LOCK_TIMEOUT_SECONDS = 300.0


def _lock_across_processes(handle: IO[Any]) -> None:
    """Take the flock on *handle*, waiting at most :data:`PROCESS_LOCK_TIMEOUT_SECONDS`.

    The bounded wait polls (see :func:`kiss.core.file_lock.lock_exclusive`),
    so an injected Stop reaches the waiting task thread too.

    Args:
        handle: The open lock file.

    Raises:
        KISSError: When another process still holds the lock after the
            bound.
    """
    try:
        lock_exclusive(handle, timeout=PROCESS_LOCK_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise KISSError(
            f"{handle.name}: {exc}; another Sorcar process (a daemon or a CLI run)"
            " is still using this repository (a merge or an orphan reclaim can"
            " take several minutes); retry later"
        ) from None


@contextmanager
def _file_lock(handle: IO[Any]) -> Iterator[None]:
    """Hold an exclusive advisory lock on *handle* for its block.

    ``repo_lock`` only serialises threads inside ONE process, but two
    Sorcar processes (the ``kiss-web`` daemon and a ``kiss`` CLI run,
    or two checkouts sharing a ``--git-common-dir``) mutate the same
    repo-local plumbing files.  A ``flock`` is held by the OS on the
    open file description, so it serialises across processes and is
    released automatically if the holder dies.  The wait is bounded by
    :data:`PROCESS_LOCK_TIMEOUT_SECONDS`.

    Args:
        handle: An open file object to lock.

    Raises:
        KISSError: When another process keeps the lock past the bound.
    """
    _lock_across_processes(handle)
    try:
        yield
    finally:
        unlock(handle)


_reclaim_lock_reentry = threading.local()


class _ReclaimLockToken:
    """Ownership token for one outer :func:`_reclaim_process_lock` hold.

    The token OWNS the open lock-file handle, so for the whole life of
    the token the kernel flock is held (the handle is closed only in
    the same ``finally`` that withdraws the token from the re-entry
    map).  The re-entry map stores only a weak reference: however the
    acquisition frame ends — normal exit, an exception, or an injected
    stop that skips the ``finally`` entirely — token and handle die
    together, releasing the flock and invalidating the marker in one
    step.  The marker can therefore never claim a hold the kernel no
    longer grants (the round-3 review's fail-open bypass).
    """

    __slots__ = ("handle", "__weakref__")

    def __init__(self, handle: IO[Any]) -> None:
        self.handle: IO[Any] | None = handle


def _held_reclaim_locks() -> dict[str, weakref.ref[_ReclaimLockToken]]:
    """Return THIS thread's map of held reclaim-lock paths to tokens.

    Backs :func:`_reclaim_process_lock`'s re-entrancy: ``flock(2)`` is
    exclusive between open file descriptions, so a thread that already
    holds the flock and re-enters through a nested taker (e.g.
    ``SorcarAgent._try_setup_worktree`` wrapping the reclaim /
    sweep / create helpers, which each take the flock themselves)
    would self-deadlock on a fresh descriptor of the same file.
    Thread-local is sufficient: every production taker acquires
    ``repo_lock`` first, so nested takes for one repo always happen on
    the thread that already owns the outer hold.  Each hold is keyed
    twice, by the resolved repo path and by the lock-file path.  Values
    are weak references to :class:`_ReclaimLockToken`; a dead
    reference means the hold is over regardless of whether its
    withdrawal code ran.
    """
    held = getattr(_reclaim_lock_reentry, "held", None)
    if held is None:
        held = {}
        _reclaim_lock_reentry.held = held
    return held


def _reclaim_lock_held(held: dict[str, weakref.ref[_ReclaimLockToken]], key: str) -> bool:
    """Whether *held* records a live outer hold under *key*.

    A live token means its handle is still open, so the kernel flock is
    genuinely held by this thread's outer acquisition.
    """
    prior = held.get(key)
    return prior is not None and prior() is not None


@contextmanager
def _reclaim_process_lock(repo: Path) -> Iterator[None]:
    """Hold the cross-process orphan-reclaim lock for *repo*.

    ``repo_lock`` is a ``threading.RLock`` — it serialises reclaim
    sweeps inside ONE process only.  Two processes (a ``kiss-web``
    daemon booting and a ``kiss`` CLI run) reclaiming the same dead
    owner's worktree can interleave their stage → cherry-pick →
    commit → cleanup sequences: one process's ``git reset`` wipes the
    other's staged cherry-pick, ``_commit_staged_merge`` then sees an
    empty index and reports SUCCESS, and ``cleanup_partial`` deletes
    the branch and directory — the orphan's work is lost.  An
    ``flock`` on ``<git_common_dir>/kiss-reclaim.lock`` serialises
    every multi-command transaction that mutates the shared main
    worktree across processes:
    :meth:`GitWorktreeOps.reclaim_orphaned_worktrees`,
    :meth:`GitWorktreeOps.sweep_orphaned_state`,
    :meth:`GitWorktreeOps.create`, and the normal merge and discard
    paths (``SorcarAgent._do_merge`` / ``discard``, whose
    failure paths run the very same ``git reset --hard``)
    (mirroring ``persistence._journal_file_lock``); the kernel
    releases it if the holder dies.  Takers that also hold
    ``repo_lock`` acquire it FIRST — the flock is always the inner
    lock.

    Re-entrant per thread (see :func:`_held_reclaim_locks`): a nested
    take of a lock file this thread already holds yields immediately,
    which lets ``SorcarAgent._try_setup_worktree`` hold the
    flock across its whole main-tree baseline snapshot (branch read +
    dirty-state copy) while still calling the reclaim / sweep / create
    helpers that take the flock internally.

    Degrades to the previous in-process-only behaviour when the common
    dir cannot be resolved or the lock file cannot be opened.

    Args:
        repo: Git repo root path.

    Yields:
        Nothing; the lock is held for the duration of the block.

    Raises:
        KISSError: When another process keeps the lock past
            :data:`PROCESS_LOCK_TIMEOUT_SECONDS`.
    """
    # The outer hold is registered under the repo path as well as the
    # lock path, so the nested takes of a setup (reclaim, sweep, create)
    # re-enter without spawning ``git rev-parse``; the lock-path key
    # still catches a nested take through another checkout of the same
    # ``--git-common-dir``.
    held = _held_reclaim_locks()
    repo_key = str(repo.resolve())
    if _reclaim_lock_held(held, repo_key):
        yield
        return
    git_common = _git_common_dir(repo)
    if git_common is None:  # pragma: no cover — not a git repo
        yield
        return
    lock_path = str(git_common / "kiss-reclaim.lock")
    if _reclaim_lock_held(held, lock_path):
        yield
        return
    try:
        handle = open(lock_path, "a+", encoding="utf-8")
    except OSError:  # pragma: no cover — unwritable git dir
        yield
        return
    # Marker registration and flock lifetime are fused into ONE frame:
    # the re-entry marker is valid exactly while `token` (the handle's
    # only owner besides this frame) is alive.  Whatever bytecode
    # boundary an injected stop lands on, marker and flock stay
    # consistent: skipping the finally leaves both held (the leaked
    # token keeps the handle open) until the token dies, which releases
    # both together; running the finally withdraws both in order.
    token = _ReclaimLockToken(handle)
    try:
        _lock_across_processes(handle)
        held[lock_path] = held[repo_key] = weakref.ref(token)
        yield
    finally:
        held.pop(lock_path, None)
        held.pop(repo_key, None)
        try:
            unlock(handle)
        finally:
            token.handle = None
            handle.close()


_repo_locks: dict[str, threading.RLock] = {}
# Re-entrant and taken through ``locked_call`` like every lock a task
# thread reaches (``repo_lock`` runs on task threads): a plain ``with``
# on it could leak it under the Stop injection (gh-74174) and hang
# ``repo_lock()`` for EVERY repository of the daemon.
_repo_locks_guard = threading.RLock()

# ``(resolved repo root, branch)`` pairs whose keep-for-review decision
# is held in THIS process only: :meth:`GitWorktreeOps.save_preserve_marker`
# could not write the durable marker file (an unwritable git dir).
# Recorded on every failed write and dropped by the next successful
# one, so :meth:`GitWorktreeOps.load_preserve_marker` — and through it
# every same-process :meth:`GitWorktreeOps.reclaim_orphaned_worktrees`,
# whose owner-pid protection deliberately exempts this process — treats
# the branch as preserved until the marker really is on disk.
# Every access is one built-in set operation (atomic under the GIL),
# so no lock guards it.
_volatile_preserve_claims: set[tuple[str, str]] = set()

_REPO_SCOPED_GIT_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
    "GIT_NAMESPACE",
    "GIT_GRAFT_FILE",
    "GIT_SHALLOW_FILE",
)


USER_PROMPT_HEADING = "\n\nUser prompt:\n"
TASK_RESULT_HEADING = "\n\nResult:\n"

_HTML_TAG_RE = re.compile(r"<[A-Za-z][^>]*>")


def result_to_commit_text(task_result: str | None) -> str:
    """Return the task result as the text stamped into a commit message.

    ``finish()`` produces the result as HTML, but a commit message is
    plain text: ``git log`` and GitHub would print the tags literally,
    and VS Code's Git hovers (blame, Timeline, Source Control Graph)
    render the message as Markdown with raw HTML disabled, so an HTML
    result body disappears entirely.  An HTML result is therefore
    converted to Markdown; a result without any HTML tag is only
    stripped.

    Args:
        task_result: The task's result summary, or ``None``/empty.

    Returns:
        The stripped Markdown (or plain) text, ``""`` when empty.
    """
    text = task_result.strip() if task_result else ""
    if text and _HTML_TAG_RE.search(text):
        return html_to_markdown(text)
    return text


def git_cleanup_whitespace(message: str) -> str:
    """Normalise *message* the way ``git commit --cleanup=whitespace`` does.

    ``git commit -m`` strips trailing whitespace from every line,
    collapses consecutive blank lines and drops leading/trailing blank
    lines before storing the message.  Any comparison between a stored
    commit message and text that is about to be committed must apply
    the same normalisation to both sides.

    Only space, tab and CR count as trailing whitespace and only
    ``\\n`` separates lines, exactly as in git: NBSP, U+2028, ``\\v``
    and ``\\f`` are ordinary characters and stay.

    Args:
        message: The raw commit message or message fragment.

    Returns:
        The normalised text without a trailing newline.
    """
    lines: list[str] = []
    for line in message.split("\n"):
        line = line.rstrip(" \t\r")
        if line or (lines and lines[-1]):
            lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _ensure_task_metadata(
    message: str,
    user_prompt: str | None,
    task_result: str | None,
) -> str:
    """Ensure the CURRENT task prompt/result are the message's suffix.

    Appends canonical ``User prompt:`` / ``Result:`` blocks carrying
    the current values to *message* unless the message ALREADY ends
    with those exact canonical blocks (the framework auto-commit path
    appends the same blocks, so a branch-HEAD message it produced must
    not be double-stamped).  The result is stamped as Markdown via
    :func:`result_to_commit_text`.

    Dedup is by exact current-value suffix — NEVER by heading
    substring: a hand-written commit body that merely mentions
    ``Result:``, a stale block from a previous task, or a prompt whose
    own text contains ``Result:`` must not suppress stamping the
    current values (gpt-5.6-sol review findings).  When the message
    ends with only the current result block, the missing prompt block
    is inserted before it to preserve canonical prompt-then-result
    order.

    *message* is the branch HEAD's stored message, which git has
    already normalised (``--cleanup=whitespace``: trailing whitespace
    stripped, blank lines collapsed), while the prompt and result are
    raw.  Both sides are normalised with :func:`git_cleanup_whitespace`
    before comparing; otherwise a prompt with a trailing space on some
    line would never match its stored copy and would be stamped twice.
    A branch HEAD stamped before results were converted to Markdown
    ends with the raw HTML result; that legacy block is replaced by
    the Markdown one instead of being kept alongside it.

    Args:
        message: The base commit message (subject + optional body).
        user_prompt: The current task's prompt, or ``None``/empty.
        task_result: The current task's result summary, or
            ``None``/empty.

    Returns:
        The commit message ending with the canonical metadata blocks
        for every non-empty current value.
    """
    msg = git_cleanup_whitespace(message)
    prompt = git_cleanup_whitespace((user_prompt or "").strip())
    result = git_cleanup_whitespace(result_to_commit_text(task_result))
    prompt_block = f"{USER_PROMPT_HEADING}{prompt}" if prompt else ""
    result_block = f"{TASK_RESULT_HEADING}{result}" if result else ""
    raw_result = git_cleanup_whitespace((task_result or "").strip())
    if raw_result and raw_result != result:
        legacy_suffix = f"{prompt_block}{TASK_RESULT_HEADING}{raw_result}"
        if msg.endswith(legacy_suffix):
            msg = msg[: -len(legacy_suffix)] + prompt_block
    if prompt and result:
        if msg.endswith(prompt_block + result_block):
            return msg
        if msg.endswith(prompt_block):
            return msg + result_block
        if msg.endswith(result_block):
            base = msg[: -len(result_block)].rstrip()
            return base + prompt_block + result_block
        return msg + prompt_block + result_block
    if prompt:
        return msg if msg.endswith(prompt_block) else msg + prompt_block
    if result:
        return msg if msg.endswith(result_block) else msg + result_block
    return msg


def repo_lock(repo: Path) -> threading.RLock:
    """Return a per-repo re-entrant lock for multi-step git operations.

    Concurrent tabs operating on the same main repository must
    serialize their checkout → stash → merge → pop sequences to
    prevent interleaving that could corrupt the working tree.

    The lock is an :class:`threading.RLock` (re-entrant) so a caller
    holding the lock for an outer multi-step operation
    (e.g. ``_try_setup_worktree`` releasing a previous worktree
    before creating a new one) can safely call inner helpers
    (``_do_merge``, ``discard``) that re-acquire the same lock on the
    same thread.  Cross-thread acquisitions still block as expected.

    A repository has one lock, whichever of its checkouts *repo* is
    (the main one, a linked worktree, the git dir itself, a path
    below one): see :func:`_repo_lock_key`.

    Args:
        repo: A checkout of the repository.

    Returns:
        The :class:`threading.RLock` of that repository.
    """
    return locked_call(_repo_locks_guard, _repo_lock_for, _repo_lock_key(repo))


def repo_locks(repos: Iterable[Path]) -> list[threading.RLock]:
    """Return the :func:`repo_lock` of each repository in *repos*, in the order to take them.

    A thread that holds several repositories' locks at once takes them
    in this order, the one of the names they go by
    (:func:`_repo_lock_key`), and each of them once.  The order of the
    checkout paths is not one: two checkouts of a repository share its
    lock, so two threads that reach the same two repositories through
    differently named checkouts would take the two locks in opposite
    orders, and each would wait for the one the other holds.

    Args:
        repos: A checkout of each repository; several of one are one lock.

    Returns:
        The locks, to be acquired first to last.
    """
    keys = sorted({_repo_lock_key(repo) for repo in repos})
    return [locked_call(_repo_locks_guard, _repo_lock_for, key) for key in keys]


def _repo_lock_key(checkout: Path) -> str:
    """Name the repository *checkout* is in by the git dir its checkouts share.

    Read from the files git itself goes by (``.git``, ``HEAD``,
    ``commondir``, ``objects``), upwards from *checkout* as git looks,
    and not asked of git: the name must not change while a lock is
    held, and git's answer can (a config it cannot parse fails every
    command).  A path in no repository names itself.
    """
    path = checkout.resolve()
    for directory in (path, *path.parents):
        # A checkout's git dir is ``.git``, or what a ``.git`` file points to (a linked
        # worktree's, a git dir kept apart); *directory* may be a git dir itself.
        pointer = (_path_in(directory / ".git") or ".git").removeprefix("gitdir: ")
        for git_dir in (directory / pointer, directory):
            # A linked worktree's git dir names the one all checkouts share.
            shared = git_dir / (_path_in(git_dir / "commondir") or ".")
            if os.path.lexists(git_dir / "HEAD") and os.path.isdir(shared / "objects"):
                return str(shared.resolve())
    return str(path)


def _path_in(file: Path) -> str | None:
    """Return the path *file* holds, or ``None`` when there is no such file to read.

    As git reads it: less every line end at the end of the file, then up to a NUL.
    """
    try:
        return os.fsdecode(file.read_bytes()).rstrip("\r\n").partition("\0")[0]
    except OSError:  # missing, a directory, unreadable
        return None


def _repo_lock_for(key: str) -> threading.RLock:
    """:func:`repo_lock`'s body under ``_repo_locks_guard``: allocate only on first use."""
    lock = _repo_locks.get(key)
    if lock is None:
        lock = _repo_locks[key] = threading.RLock()
    return lock


_GIT_TIMEOUT_SECONDS: float = 300.0

#: The longest a :func:`_git` whose cleanup a Stop cut short waits for the
#: thread that finishes it (:func:`_end_git`: two five-second graces at most).
_GIT_END_SECONDS: float = 30.0


def _git_command(
    args: tuple[str, ...], cwd: str | Path,
) -> tuple[list[str], dict[str, str]]:
    """Return the argv and environment every git runner in this module uses.

    ``core.quotepath=false`` keeps non-ASCII paths unescaped in output,
    ``-C`` pins the working directory, and the repo-scoping variables
    inherited from the environment (``GIT_DIR``, ``GIT_INDEX_FILE``, ...)
    are dropped so a caller's environment can never redirect a call to
    another repository.

    Args:
        args: Git sub-command and arguments (without the leading ``git``).
        cwd: Working directory for the git command.

    Returns:
        ``(argv, env)`` ready for ``Popen``.
    """
    cmd = ["git", "-c", "core.quotepath=false", "-C", str(cwd), *args]
    env = {k: v for k, v in os.environ.items() if k not in _REPO_SCOPED_GIT_ENV}
    return cmd, env


def _git_common_dir(repo: Path) -> Path | None:
    """Return *repo*'s ``--git-common-dir`` as an absolute path, or ``None``.

    Args:
        repo: Git repo root path.
    """
    result = _git("rev-parse", "--git-common-dir", cwd=repo)
    if result.returncode != 0:  # pragma: no cover — not a git repo
        return None
    # Only git's record terminator goes: ``strip()`` would turn a separate
    # git dir named ``metadata `` (``git init --separate-git-dir``) into
    # its neighbour ``metadata``.
    git_common = Path(result.stdout.removesuffix("\n"))
    if not git_common.is_absolute():  # pragma: no branch
        git_common = (repo / git_common).resolve()
    return git_common


def _git(
    *args: str,
    cwd: str | Path,
    env: dict[str, str] | None = None,
    input: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a git command, returning the CompletedProcess result.

    ``cwd`` is a required keyword-only argument so the type checker
    enforces that every git invocation specifies a working directory
    (passed via ``git -C <cwd>``).  This prevents accidental git
    operations against the process's current working directory.
    Repo-scoping variables inherited from the environment (``GIT_DIR``,
    ``GIT_INDEX_FILE``, ...) are always dropped; *env* adds explicit
    ones on top for the single call (e.g. a scratch ``GIT_INDEX_FILE``).

    Always passes a timeout so a hung git process (e.g. waiting on a
    credential-helper prompt or a network remote) cannot block the
    agent thread forever — the one hardened runner every server-side
    git call goes through.  On expiry git is stopped like an
    interrupted one (``SIGTERM`` first, below) and a synthesized
    non-zero ``CompletedProcess`` (returncode 124, the timeout
    convention) is returned so callers keep working.  A command that
    cannot start at all (``OSError``, e.g. an argument over the
    kernel's single-argument limit; ``ValueError`` for a NUL byte in
    an argument) is likewise returned as a failed ``CompletedProcess``
    instead of raising: returncode 127, the "command not found"
    convention, because 1 is a verdict of ``git grep``, ``git diff
    --quiet`` and ``git merge-tree``.  Output is decoded by
    :func:`_git_text`: the paths of a ``-z`` or ``rev-parse`` command
    byte for byte.

    An exception raised once git runs (the runner's Stop delivers
    ``KeyboardInterrupt`` to the task thread at any bytecode) stops git
    and its whole process group before propagating: the caller's
    rollback assumes git is finished, and a ``git commit`` left running
    behind a slow hook would otherwise land minutes later, on top of
    whatever the user has staged since.  ``SIGTERM`` first, so git
    removes its lock files; ``SIGKILL`` when it lingers.  Its pipes are
    closed with it.  A further Stop does not end this early: see the
    ``finally`` below.

    Args:
        *args: Git sub-command and arguments (without the leading ``git``).
        cwd: Working directory for the git command (required).
        env: Extra environment variables for this call only.
        input: Text fed to git's stdin (e.g. a commit message for
            ``commit -F -``, which has no argument-size limit); ``None``
            leaves stdin inherited.

    Returns:
        The completed process with stdout/stderr captured as text.
    """
    cmd, full_env = _git_command(args, cwd)
    if env:
        full_env.update(env)
    verbatim = "-z" in args or args[:1] == ("rev-parse",)
    # Before git starts: text that cannot be encoded (a lone surrogate)
    # must not leave a git behind.
    given = None if input is None else input.encode("utf-8", "surrogateescape")
    # The Popen is held from before the spawn, which runs inside the
    # ``try``: a Stop raised as the spawning call returns reaches the
    # ``finally`` with git in hand (see :func:`start_process_group`).
    proc: subprocess.Popen[bytes] = object.__new__(subprocess.Popen)
    reading = False  # ``communicate`` entered: on Windows its reader threads own the pipes
    ended = threading.Lock()  # held until :func:`_end_git` has run to its end
    ended.acquire()
    # Git reads *input* from a file, not from a pipe ``communicate`` feeds:
    # that takes its input in one call (POSIX registers stdin only for the
    # call that passes it, and a second call that passes it is refused),
    # and the wait below is taken in slices.  :func:`_end_git` closes it.
    source: IO[bytes] | None = None
    try:
        try:
            if given is not None:
                source = tempfile.TemporaryFile()
                source.write(given)
                source.seek(0)
            start_process_group(
                proc, cmd, stdin=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=full_env,
            )
        except (OSError, ValueError) as exc:
            logger.warning("git %s could not start: %s", args[:1], exc)
            return subprocess.CompletedProcess(cmd, 127, "", f"git could not start: {exc}")
        reading = True
        # In slices, like :func:`~kiss.core.processes.run_process_group`: an
        # injected Stop is raised between two of them, never inside one
        # wait for a git that writes nothing (a ``commit`` running its hooks).
        deadline = time.monotonic() + _GIT_TIMEOUT_SECONDS
        while True:
            try:
                out, err = proc.communicate(
                    timeout=min(deadline - time.monotonic(), _STOP_POLL_SECONDS),
                )
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise
        return subprocess.CompletedProcess(
            cmd, proc.returncode, _git_text(out, verbatim), _git_text(err, verbatim),
        )
    except subprocess.TimeoutExpired:
        logger.warning("git %s timed out after %ss", args, _GIT_TIMEOUT_SECONDS)
        terminate_process_group(proc)  # SIGTERM first: git removes its .lock files
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            # Only a descendant that left the process group (its own
            # setsid) can still hold the pipes: give up on its output.
            # A git that outlived its ``SIGKILL`` is not waited for
            # here; the ``finally`` stops it once more, with a bound.
            out, err = b"", b""
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=_git_text(out, verbatim),
            stderr=_git_text(err, verbatim)
            or f"git {args[0] if args else ''} timed out"
            f" after {_GIT_TIMEOUT_SECONDS}s",
        )
    finally:  # however the call ends, a Stop raised while the timeout is handled included
        try:
            _end_git(proc, source, reading, ended)
        except BaseException as stop:
            # One more Stop cut the cleanup short (the stop watchdog injects
            # twice, a tool-panel Stop may follow).  It starts over on a
            # thread of its own, into which no Stop is injected, and this
            # thread waits for it in one C call, which an injected exception
            # cannot leave early.  CPython raises one only after a call
            # returns, at a loop's jump back or on entering a Python
            # function: none can come before the call that starts the
            # thread, nor, the handler of a Stop raised as that call returns
            # making no call, between it and the wait.  The newest Stop
            # propagates once git is ended.
            newest = stop
            try:
                _thread.start_new_thread(_end_git, (proc, source, reading, ended))
            except RuntimeError:  # no thread to be had: here, where a Stop can cut it again
                _end_git(proc, source, reading, ended)
            except BaseException as again:
                newest = again
            ended.acquire(True, _GIT_END_SECONDS)
            raise newest


class _GitStepError(Exception):
    """A step of several that only count together failed; what failed is logged."""


def _git_step(repo: Path, env: dict[str, str], *args: str, input: str | None = None) -> str:
    """Run one git step on the scratch index *env* names and return what it printed.

    Raises:
        _GitStepError: git failed; its error is logged.
    """
    result = _git(*args, cwd=repo, env=env, input=input)
    if result.returncode != 0:
        logger.warning("git %s failed in %s: %s", " ".join(args), repo, result.stderr.strip())
        raise _GitStepError
    return result.stdout


def _leading_dirs(path: str) -> list[str]:
    """Return the directories the repository-relative *path* lies in, outermost first."""
    return [path[:end] for end, char in enumerate(path) if char == "/"]


def _end_git(
    proc: subprocess.Popen[bytes], source: IO[bytes] | None, reading: bool, ended: threading.Lock,
) -> None:
    """Stop git's process group, if it still runs, and close its pipes and its input.

    The cleanup of :func:`_git`.  Running it again after a run that was
    cut short is safe: a reaped git is not signalled, a closed file
    stays closed.

    Args:
        proc: The ``Popen`` that :func:`_git` held from before the spawn.
        source: The file git read its standard input from, if any.
        reading: Whether ``communicate`` was entered.
        ended: The lock :func:`_git` holds; released once the cleanup is
            complete.
    """
    if source is not None:  # git has its own descriptor
        with contextlib.suppress(OSError):
            source.close()
    if getattr(proc, "pid", None) is not None:  # None: no git was spawned
        # No ``poll()``: git may have exited with its group still running.
        if proc.returncode is None:  # once reaped, its group id may be a stranger's
            terminate_process_group(proc)  # SIGTERM first: git removes its .lock files
        # The pipes go with git (a raised error's traceback holds this
        # ``Popen``).  A close under a Windows reader thread's read would
        # park this one; one that fails is not raised.
        if not (IS_WINDOWS and reading):
            for pipe in filter(None, (proc.stdout, proc.stderr)):
                with contextlib.suppress(OSError):
                    pipe.close()
    if ended.locked():  # not in the repeat of a run that a Stop cut after its release
        ended.release()


def _git_text(data: bytes, verbatim: bool) -> str:
    """Return git output *data* as text: UTF-8, other bytes as surrogate escapes.

    Line ends are translated as text mode would (``\\r\\n`` and ``\\r`` to
    ``\\n``), except in *verbatim* output: that of a ``-z`` command,
    which lists paths NUL-terminated, and of ``rev-parse``, whose
    ``--show-toplevel`` / ``--git-common-dir`` print a path followed by
    one newline.  A carriage return in such a path is part of the name
    (translated, ``a\\rb`` would name the neighbour ``a\\nb``).
    """
    text = data.decode("utf-8", "surrogateescape")
    return text if verbatim else text.replace("\r\n", "\n").replace("\r", "\n")


def _git_stdout_head(
    *args: str,
    cwd: str | Path,
    max_bytes: int,
) -> tuple[str, bool]:
    """Run a git command and return at most *max_bytes* of its stdout.

    Stdout is streamed and the process is killed as soon as the cap is
    reached, so a command whose output is unbounded costs at most
    *max_bytes* of memory.  ``git diff --cached`` over a worktree that
    had copied in 38,081 benchmark result files (38.5 million lines)
    used to be read whole into a string by the commit-message step:
    the daemon grew to 45 GB and the merge thread, which held the
    tab's ``is_merging`` claim, never finished, so every main-tree task
    was refused with "A worktree merge is in progress" until a restart
    (``~/.kiss/kiss-web-stderr.log``, 2026-09-23 15:11 UTC).

    Args:
        *args: Git sub-command and arguments (without the leading ``git``).
        cwd: Working directory for the git command (required).
        max_bytes: Maximum number of stdout bytes to read.

    Returns:
        ``(text, truncated)``: the decoded stdout (cut back to the last
        complete line when truncated) and whether it is incomplete,
        either because more output existed or because git was killed
        after :data:`_GIT_TIMEOUT_SECONDS`.  ``("", False)`` when git
        could not start (no ``git`` on the PATH): the caller takes its
        "no output" path, as it does for a :func:`_git` that failed to
        start.
    """
    cmd, env = _git_command(args, cwd)
    try:
        out = run_capped(
            cmd, limit=max_bytes, timeout=_GIT_TIMEOUT_SECONDS, env=env, last_full_line=True,
        )
    except (OSError, ValueError) as exc:
        logger.warning("git %s could not start: %s", args[:1], exc)
        return "", False
    if out.timed_out:
        logger.warning("git %s timed out after %ss", args, _GIT_TIMEOUT_SECONDS)
    return out.stdout.decode("utf-8", "replace"), out.truncated or out.timed_out


COMMIT_MESSAGE_DIFF_LIMIT_BYTES: int = 200_000
"""Largest staged patch handed to the commit-message LLM.

See :meth:`GitWorktreeOps.staged_diff`.
"""


@dataclass(frozen=True)
class GitWorktree:
    """Immutable snapshot of a pending worktree task.

    Attributes:
        repo_root: Git repo root path.
        branch: Branch name of the worktree task.
        original_branch: The branch the user was on when the task started,
            or ``None`` if unknown (crash between creation and config write).
        wt_dir: Worktree directory path.
        baseline_commit: SHA of the initial commit that captured the user's
            dirty state (staged, unstaged, untracked files) at worktree
            creation time.  ``None`` when the main worktree was clean or
            for legacy worktrees created before baseline support.
        work_dir: The task's working directory inside the worktree
            (``wt_dir`` plus the original work dir's offset from the
            repo root).  Equals ``wt_dir`` when the task started at the
            repo root.  ``None`` for legacy snapshots created before
            this field existed.
    """

    repo_root: Path
    branch: str
    original_branch: str | None
    wt_dir: Path
    baseline_commit: str | None = None
    work_dir: Path | None = None


class MergeResult(enum.Enum):
    """Outcome of a merge operation."""

    SUCCESS = "success"
    CONFLICT = "conflict"
    CHECKOUT_FAILED = "checkout_failed"
    MERGE_FAILED = "merge_failed"
    STASH_FAILED = "stash_failed"


_WORKTREE_SUBDIR = ".kiss-worktrees"
_WORKTREE_SLUG_PREFIX = "kiss_wt-"
_WORKTREE_BRANCH_PREFIX = "kiss/wt-"

OWN_ENVIRONMENT_BLOCKER = """\
This worktree has no Python environment of its own. It shares the main checkout's:
{environment}

Run that environment's programs on the worktree's sources, for example
    PYTHONPATH={worktree}/src:{worktree}/projects {environment}/bin/pytest ...
(a `uv run PROGRAM` typed directly into the Bash tool is rewritten that way).

This file is not an environment on purpose: uv refuses to build or use one here,
in every form (bash -c, scripts, make), so every test run of this worktree is on
one Python. Do not delete or replace it.
"""
"""The text of the ``.venv`` file of a worktree (:meth:`GitWorktreeOps.block_own_environment`)."""


def _same_path(left: Path, right: Path) -> bool:
    """Return True when both paths denote the same directory.

    Symlinks (``/var`` -> ``/private/var`` on macOS) and relative
    components are resolved first; unresolvable paths fall back to a
    plain comparison so a missing directory never raises.

    Args:
        left: First path.
        right: Second path.

    Returns:
        True when the two paths are the same location.
    """
    try:
        return left.resolve() == right.resolve()
    except OSError:  # pragma: no cover — unreadable path components
        return left == right


def strip_worktree_suffix(path: str) -> str:
    """Return *path* with the ``.kiss-worktrees/kiss_wt-<slug>[/...]``
    suffix removed, leaving the parent repository path.

    Worktree directories are ephemeral — they are deleted when the
    worktree is merged or discarded.  Persisting a worktree path in
    long-lived storage (e.g. ``task_history.extra.work_dir``) would
    leave dangling references the user-visible UI cannot resolve.  Use
    this helper at every persistence boundary that records a
    ``work_dir`` for later display or filtering.

    Paths that do not contain a ``<repo>/.kiss-worktrees/kiss_wt-*``
    segment are returned unchanged, including the empty string.

    Args:
        path: An absolute or relative filesystem path string.

    Returns:
        The parent-repo path string, or the unchanged input when the
        path is not inside a KISS worktree.
    """
    parts, i = _worktree_parts(path)
    if i < 0:
        return path
    parent = "/".join(parts[:i])
    if parent:
        return parent
    return "/" if parts[0] == "" else "."


def _worktree_parts(path: str) -> tuple[list[str], int]:
    """Split *path* on either separator and locate its worktree segment.

    The single scanner behind :func:`strip_worktree_suffix` and
    :func:`inside_kiss_worktree`, so the two agree on every spelling
    (including ``\\``-separated paths read back from storage written
    on Windows).

    Args:
        path: An absolute or relative filesystem path string.

    Returns:
        ``(parts, i)`` where ``parts[i]`` is the ``.kiss-worktrees``
        segment followed by a ``kiss_wt-*`` slug, or ``i == -1`` when
        *path* is not inside a KISS worktree.
    """
    parts = path.replace("\\", "/").split("/")
    for i in range(len(parts) - 1):
        if parts[i] == _WORKTREE_SUBDIR and parts[i + 1].startswith(_WORKTREE_SLUG_PREFIX):
            return parts, i
    return parts, -1


def _porcelain_entries(output: str) -> list[tuple[str, str | None, str]]:
    """Parse ``git status --porcelain -z`` output into entries.

    The one porcelain parser, shared by
    :meth:`GitWorktreeOps.copy_dirty_state` and the merge flow's
    changed-files fallback.  Each entry is ``XY path`` ended by a NUL;
    a rename or copy (``R`` or ``C`` in ``XY``) is ``XY new``, NUL,
    ``old``, NUL.  With ``-z`` git neither quotes nor escapes a path,
    so names with spaces at either end, newlines, quotes, backslashes
    or a `` -> `` come back exactly as on disk (bytes that are not
    UTF-8 as the ``surrogateescape`` characters :func:`_git` decodes
    them to).

    Args:
        output: Raw stdout of ``git status --porcelain -z``.

    Returns:
        ``(code, old_name, new_name)`` tuples in output order: ``code``
        is the two-character status code, ``old_name`` the rename or
        copy source (``None`` for any other entry) and ``new_name`` the
        entry's current path.
    """
    entries: list[tuple[str, str | None, str]] = []
    fields = iter(output.split("\0"))
    for field in fields:
        if len(field) < 4:
            continue
        code = field[:2]
        old_name = next(fields, None) if "R" in code or "C" in code else None
        entries.append((code, old_name, field[3:]))
    return entries


def _under_symlink(root: Path, rel: str) -> bool:
    """True when an ancestor directory of *rel* inside *root* is a symlink.

    Git never lists a path beyond a symlink, so a porcelain entry whose
    ancestor is one (``D  dir/x`` next to ``A  dir`` after the tracked
    ``dir/`` was replaced by a symlink) is a stale child: ``root / rel``
    resolves *through* the link to a file outside the tree, which must
    be neither copied nor removed.
    """
    path = root
    for part in Path(rel).parts[:-1]:
        path = path / part
        if path.is_symlink():
            return True
    return False


#: The message of the stash entry a repository's merge pushed
#: (:meth:`GitWorktreeOps.stash_if_dirty`), while that entry is the merge's
#: to give back: what tells it from the user's entries, wherever their own
#: ``git stash`` moves it.  One merge runs per repository at a time
#: (``repo_lock``), and each access is one dict operation.
_merge_stash_labels: dict[Path, str] = {}

#: What the conflicted merge of a repository wrote, from
#: :meth:`GitWorktreeOps.begin_conflicted_merge` until it is committed or
#: taken back: the tree, conflict markers included, and the paths left
#: unmerged.  Kept here because any git command of the merge agent may
#: drop ``AUTO_MERGE``.  One merge runs per repository at a time
#: (``repo_lock``), and each access is one dict operation.
_conflicted_merges: dict[Path, tuple[str, list[str]]] = {}

#: How often :meth:`GitWorktreeOps._drop_stash` goes for an entry that the
#: user's stash commands keep moving.
_STASH_DROP_ATTEMPTS = 3

#: What a checkout's git directory holds while a merge, a rebase (or ``am``),
#: a cherry-pick or a revert is under way in it, one commit or a sequence.
_OPERATION_STATE = (
    "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "sequencer", "rebase-merge", "rebase-apply",
)


def _stash_entry(repo: Path, label: str) -> tuple[str, int] | None:
    """Find the stash entry of *repo* whose message ends with *label*.

    Args:
        repo: Git repo root path.
        label: The message a :meth:`GitWorktreeOps.stash_if_dirty` pushed under.

    Returns:
        ``(sha, position)``, the entry being ``stash@{position}`` right
        now; ``("", -1)`` when the stash does not hold it; None when git
        could not list the stash.
    """
    listed = _git("stash", "list", "--format=%H %gs", cwd=repo)
    if listed.returncode != 0:
        return None
    for position, line in enumerate(listed.stdout.splitlines()):
        if line.endswith(label):
            return line.partition(" ")[0], position
    return "", -1


class GitWorktreeOps:
    """Stateless helper class with all git worktree operations.

    Every method is a ``@staticmethod`` — no instance state.  All git
    interactions are encapsulated here so callers never need to parse
    returncode or stderr.
    """

    @staticmethod
    def discover_repo(path: Path) -> Path | None:
        """Find the git repo root containing *path*.

        Args:
            path: Directory to start searching from.

        Returns:
            The repo root path, or ``None`` if *path* is not in a repo.
        """
        result = _git("rev-parse", "--show-toplevel", cwd=path)
        if result.returncode != 0:
            return None
        # Only git's record terminator goes: ``strip()`` would turn a
        # root named ``project `` into its neighbour ``project``.
        return Path(result.stdout.removesuffix("\n"))

    @staticmethod
    def current_branch(repo: Path) -> str | None:
        """Return the current branch name, or ``None`` for detached HEAD.

        Args:
            repo: Git repo root path.

        Returns:
            Branch name string, or ``None`` if HEAD is detached or empty.
        """
        result = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo)
        branch = result.stdout.strip()
        if not branch or branch == "HEAD":
            return None
        return branch

    @staticmethod
    def create(repo: Path, branch: str, wt_dir: Path) -> bool:
        """Create a new worktree with a new branch.

        Args:
            repo: Git repo root path.
            branch: New branch name to create.
            wt_dir: Directory for the new worktree.

        The new worktree is registered with ``git worktree add`` and
        stamped with this process's pid (:meth:`save_owner_pid`) under
        the cross-process reclaim lock (:func:`_reclaim_process_lock`).
        ``repo_lock`` only serialises threads of ONE process, so a
        second Sorcar process running
        :meth:`reclaim_orphaned_worktrees` in between the two steps
        used to find a registered ``kiss/wt-*`` worktree with no owner
        and no ``kiss-original`` config, treat it as a legacy orphan,
        and squash-merge (a no-op) and DELETE it under the process
        that was about to run a task in it.  Holding the reclaim lock
        across both steps means a
        concurrent reclaim either finishes before the worktree exists
        or sees it already owned by a live pid and skips it.

        The owner stamp is part of a successful creation, not a
        best-effort extra: a worktree returned without it (``git
        config`` failed — e.g. another git process held
        ``.git/config.lock``) would be exactly the owner-less "legacy
        orphan" a peer process deletes the moment the lock above is
        released.  So a failed stamp removes the just-added worktree
        and branch — still under the reclaim lock — and reports
        failure, and the caller falls back the same way it does for a
        failed ``worktree add``.

        Returns:
            True if the worktree was created AND stamped with this
            process's pid, False otherwise (nothing is left on disk).
        """
        # ``repo_lock`` first, flock second — the file-wide order (see
        # ``_reclaim_process_lock``).  The failed-stamp path below calls
        # ``remove``, which takes ``repo_lock``; taking only the flock
        # here would invert the order against ``sweep_orphaned_state``
        # (repo_lock -> flock) and deadlock.  ``repo_lock`` is an RLock,
        # so callers already holding it are unaffected.
        return locked_call(
            repo_lock(repo), GitWorktreeOps._add_worktree_locked,
            repo, branch, wt_dir, ("-b", branch, str(wt_dir)), True,
        )

    @staticmethod
    def _add_worktree_locked(
        repo: Path,
        branch: str,
        wt_dir: Path,
        add_args: tuple[str, ...],
        new_branch: bool,
    ) -> bool:
        """``git worktree add`` + owner stamp under the reclaim lock.

        Shared body of :meth:`create` and :meth:`reattach`, run under
        ``repo_lock``: *add_args* follow ``git worktree add``.  When the
        owner stamp fails the add is undone: the worktree is removed,
        and so is the branch when the add created it (*new_branch*).
        """
        with _reclaim_process_lock(repo):
            result = _git("worktree", "add", *add_args, cwd=repo)
            if result.returncode != 0:
                logger.warning("Failed to add worktree: %s", result.stderr.strip())
                return False
            if not GitWorktreeOps.save_owner_pid(repo, branch):
                logger.warning(
                    "Removing freshly added worktree %s: its owner pid "
                    "could not be recorded, so another process's reclaim "
                    "would delete it under this one",
                    wt_dir,
                )
                GitWorktreeOps.remove(repo, wt_dir)
                if new_branch:
                    GitWorktreeOps.delete_branch(repo, branch)
                return False
        return True

    @staticmethod
    def reattach(repo: Path, branch: str, wt_dir: Path) -> bool:
        """Check the existing *branch* out again as the worktree *wt_dir*.

        Used after a merge failed once the worktree directory was
        already removed (its commits are safe on *branch*): the task
        gets its working copy back and can fix the problem and retry.
        Registered and owner-stamped like :meth:`create`, under the
        same locks.

        Args:
            repo: Git repo root path.
            branch: The existing ``kiss/wt-*`` branch.
            wt_dir: Directory for the worktree.

        Returns:
            True when the worktree exists and is stamped, False
            otherwise (nothing is left on disk).
        """
        return locked_call(
            repo_lock(repo), GitWorktreeOps._add_worktree_locked,
            repo, branch, wt_dir, (str(wt_dir), branch), False,
        )

    @staticmethod
    def remove(repo: Path, wt_dir: Path) -> None:
        """Remove a worktree directory (best-effort, force).

        Every caller (``discard``, ``cleanup_partial``,
        ``_finalize_worktree``) intends permanent removal of an
        agent-owned directory, so failures of ``git worktree remove``
        are escalated rather than abandoned:

        1. Plain ``--force`` (handles dirty/untracked content).
        2. ``--force --force`` (git requires force twice for worktrees
           locked via ``git worktree lock``).
        3. Direct ``rmtree`` + ``git worktree prune`` (handles
           corrupted worktrees — e.g. a deleted ``.git`` link file —
           that fail git's removal validation entirely).

        Teardown mutates the shared worktree registry under
        ``$GIT_COMMON_DIR/worktrees/``, so it takes ``repo_lock``
        itself rather than relying on every caller to remember: one
        unlocked caller is enough to let an ``rmtree``/``prune`` run
        while another tab holds the lock for ``worktree add`` /
        ``stash`` / ``checkout`` / ``cherry-pick``.  The lock is
        re-entrant, so callers that already hold it are unaffected.

        Args:
            repo: Git repo root path.
            wt_dir: Worktree directory to remove.
        """
        locked_call(repo_lock(repo), GitWorktreeOps._remove_locked, repo, wt_dir)

    @staticmethod
    def _remove_locked(repo: Path, wt_dir: Path) -> None:
        """:meth:`remove`'s body, run under ``repo_lock``."""
        if _same_path(wt_dir, repo):
            # The main working tree is not an agent worktree: git
            # rightly refuses to remove it, and the ``rmtree``
            # fallback below would delete the user's whole project.
            logger.error(
                "Refusing to remove %s: it is the main working tree, "
                "not an agent worktree",
                wt_dir,
            )
            return
        if not wt_dir.exists():
            GitWorktreeOps.prune(repo)
            return
        result = _git("worktree", "remove", str(wt_dir), "--force", cwd=repo)
        if result.returncode != 0:
            result = _git(
                "worktree", "remove", "--force", "--force", str(wt_dir),
                cwd=repo,
            )
        if result.returncode != 0:
            logger.warning(
                "worktree remove failed: %s; deleting directory directly",
                result.stderr.strip(),
            )
            GitWorktreeOps._rmtree_reporting_leftovers(wt_dir)
            GitWorktreeOps.prune(repo)

    @staticmethod
    def _rmtree_reporting_leftovers(path: Path) -> None:
        """Delete the directory *path*, logging an error for whatever survives.

        ``shutil.rmtree(ignore_errors=True)`` swallows every failure,
        so a directory the current user cannot delete (root-owned
        output a Docker container wrote into a bind-mounted worktree)
        used to linger silently under ``.kiss-worktrees/``.  The error
        names the directory so the user knows it needs ``sudo rm -rf``.

        Args:
            path: Directory to delete.
        """
        shutil.rmtree(str(path), ignore_errors=True)
        if path.exists():
            logger.error(
                "Could not delete %s completely; files remain (not owned "
                "by this user, e.g. written by a root container) — remove "
                "them by hand with sudo rm -rf",
                path,
            )

    @staticmethod
    def prune(repo: Path) -> None:
        """Prune stale worktree bookkeeping entries.

        Rewrites the shared worktree registry, so it takes the
        re-entrant ``repo_lock`` for the same reason :meth:`remove`
        does — a prune racing another tab's ``git worktree add`` can
        drop a registration that is being created.

        Args:
            repo: Git repo root path.
        """
        locked_call(repo_lock(repo), _git, "worktree", "prune", cwd=repo)

    @staticmethod
    def stage_all(wt_dir: Path) -> None:
        """Stage all changes in the worktree (``git add -A``).

        A refused ``git add`` (an embedded repository with no commit,
        an unreadable file, a held ``index.lock``) stages nothing and is
        logged: the caller's "nothing to commit" names no cause.

        Args:
            wt_dir: Worktree directory.
        """
        result = _git("add", "-A", cwd=wt_dir)
        if result.returncode != 0:
            logger.warning("git add -A failed in %s: %s", wt_dir, result.stderr.strip())

    @staticmethod
    def commit_all(wt_dir: Path, message: str) -> bool:
        """Stage all changes and commit in the worktree.

        Args:
            wt_dir: Worktree directory.
            message: Commit message.

        Returns:
            True if a commit was created, False if nothing to commit
            or the commit failed (e.g. pre-commit hook rejection).
        """
        GitWorktreeOps.stage_all(wt_dir)
        return GitWorktreeOps.commit_staged(wt_dir, message)

    @staticmethod
    def commit_staged(
        wt_dir: Path,
        message: str,
        *,
        no_verify: bool = False,
    ) -> bool:
        """Commit already-staged changes without re-staging.

        Unlike :meth:`commit_all`, this does **not** run ``git add -A``
        first.  Use when the caller has already staged the desired
        changes (e.g. via :meth:`stage_all`).

        Args:
            wt_dir: Worktree directory with pre-staged changes.
            message: Commit message.
            no_verify: If True, pass ``--no-verify`` to skip pre-commit
                and commit-msg hooks.  Use for infrastructure commits
                (e.g. baseline snapshots) that must always succeed.

        Returns:
            True if a commit was created, False if nothing was staged
            or the commit failed (e.g. pre-commit hook rejection).
        """
        if not GitWorktreeOps.has_staged_changes(wt_dir):
            return False
        return GitWorktreeOps._git_commit(wt_dir, message, no_verify=no_verify)

    @staticmethod
    def _git_commit(wt_dir: Path, message: str, *, no_verify: bool = False) -> bool:
        """:meth:`commit_staged` without the index check, for callers that just did it."""
        # The message goes on stdin: an argv string is capped at the
        # kernel's single-argument limit (128 KiB on Linux).
        # ``--cleanup=whitespace`` is git's default for a message that
        # is not edited; pinning it keeps a user's ``commit.cleanup=strip``
        # from deleting the message's ``#`` lines (Markdown headings,
        # ``#123`` references) or rejecting an all-``#`` message as empty.
        cmd = ["commit", "--cleanup=whitespace", "-F", "-"]
        if no_verify:
            cmd.append("--no-verify")
        result = _git(*cmd, cwd=wt_dir, input=message)
        if result.returncode != 0:
            logger.warning(
                "git commit failed: %s",
                result.stderr.strip(),
            )
            return False
        return True

    @staticmethod
    def has_uncommitted_changes(wt_dir: Path) -> bool:
        """Check if the working tree or index has uncommitted changes.

        Args:
            wt_dir: Git working directory to check.

        Returns:
            True if there are staged, unstaged, or untracked changes.
            A failed status command (timeout, corrupt index, ...) is
            reported as dirty so callers never destroy a worktree whose
            cleanliness could not actually be verified.
        """
        return bool(GitWorktreeOps.status_porcelain(wt_dir))

    @staticmethod
    def status_porcelain(wt_dir: Path) -> str:
        """Return the ``git status --porcelain`` output, failing closed.

        Used by failure-path warnings (e.g.
        :meth:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent._finalize_worktree`)
        to embed the exact leftover files in the log so an operator
        can tell a real pre-commit-hook rejection apart from a race
        leftover or a corrupt index without having to ssh in and
        run git themselves, and by before/after comparisons that
        decide whether a tree may be reset or cleaned.

        Args:
            wt_dir: Git working directory to inspect.

        Returns:
            The stripped porcelain output: empty exactly when git
            reports a clean tree.  When ``git status`` fails (timeout,
            corrupt index, bad config, ...) the result is a non-empty
            line naming the failure, so every caller reads the tree
            as dirty and unverifiable rather than clean.
        """
        status = GitWorktreeOps._status(wt_dir)
        if status.returncode != 0:
            logger.warning(
                "git status failed in %s (rc=%s): %s; treating as dirty",
                wt_dir, status.returncode, status.stderr.strip(),
            )
            return f"git status failed (rc={status.returncode}): {status.stderr.strip()}"
        return status.stdout.strip()

    @staticmethod
    def _status(wt_dir: Path) -> subprocess.CompletedProcess[str]:
        """Run ``git status --porcelain``, listing untracked files whatever the config says.

        ``--untracked-files=normal`` overrides a user's
        ``status.showUntrackedFiles=no`` (common in dotfile repos and
        large monorepos).  Callers decide from this output whether a
        tree is clean: a merge that read a tree with hidden untracked
        files as clean skipped the ``stash push --include-untracked``
        that keeps them out of its way.
        """
        return _git("status", "--porcelain", "--untracked-files=normal", cwd=wt_dir)

    @staticmethod
    def has_staged_changes(wt_dir: Path) -> bool:
        """Report whether the index differs from ``HEAD``.

        Uses ``git diff --cached --quiet``, which produces no output, so
        the check costs nothing however large the staged changes are.
        Its exit status is 0 for no difference and 1 for a difference;
        any other status is a git failure (corrupt index, timeout, ...),
        which is logged and reported as "staged" so the caller's commit
        attempt surfaces git's own error instead of silently dropping
        the work.

        Args:
            wt_dir: Worktree directory to inspect.

        Returns:
            True if anything is staged (or git could not tell).
        """
        result = _git("diff", "--cached", "--quiet", cwd=wt_dir)
        if result.returncode not in (0, 1):
            logger.warning(
                "git diff --cached --quiet failed in %s (rc=%s): %s; treating as staged",
                wt_dir, result.returncode, result.stderr.strip(),
            )
        return result.returncode != 0

    @staticmethod
    def staged_diff(wt_dir: Path) -> str:
        """Return the staged diff text, bounded for use in an LLM prompt.

        At most :data:`COMMIT_MESSAGE_DIFF_LIMIT_BYTES` of the patch are
        ever read (see :func:`_git_stdout_head`).  When the patch is
        larger, the result starts with a truncation note carrying the
        ``--shortstat`` totals (files changed, insertions, deletions
        over ALL changes), then the head of the ``--stat`` file list
        (bounded to a quarter of that cap), then the head of the patch.
        The model thus always sees the full scope even when the file
        list itself is too long to enumerate.

        Args:
            wt_dir: Worktree directory (must have staged changes).

        Returns:
            The diff text, or empty string if no staged changes.
        """
        max_bytes = COMMIT_MESSAGE_DIFF_LIMIT_BYTES
        patch, truncated = _git_stdout_head("diff", "--cached", cwd=wt_dir, max_bytes=max_bytes)
        if not truncated:
            return patch.strip()
        totals = _git("diff", "--cached", "--shortstat", cwd=wt_dir).stdout.strip()
        stat, _ = _git_stdout_head(
            "diff", "--cached", "--stat", cwd=wt_dir, max_bytes=max_bytes // 4,
        )
        return (
            f"[Patch truncated to its first {max_bytes:,} bytes; "
            f"the summary below covers all changes: {totals}]\n\n"
            f"{stat.strip()}\n\n{patch.strip()}"
        )

    @staticmethod
    def checkout(repo: Path, branch: str) -> tuple[bool, str]:
        """Checkout a branch in the main worktree.

        Args:
            repo: Git repo root path.
            branch: Branch name to checkout.

        Returns:
            ``(True, "")`` on success, ``(False, stderr)`` on failure.
            The stderr string describes why the checkout failed (e.g.
            dirty working tree, missing branch, an ignored file where
            *branch* tracks one: ``--no-overwrite-ignore`` keeps git
            from silently replacing it).  No checkout is attempted
            while a git operation of the user's is under way (see
            :meth:`operation_in_progress`): it would take ``HEAD`` off
            a rebase they paused.
        """
        if GitWorktreeOps.operation_in_progress(repo):
            return (False, "a merge, rebase, cherry-pick or revert is under way")
        result = _git("checkout", "--no-overwrite-ignore", branch, cwd=repo)
        if result.returncode == 0:
            return (True, "")
        return (False, result.stderr.strip())

    @staticmethod
    def operation_in_progress(repo: Path) -> bool:
        """Whether a merge, rebase, cherry-pick or revert is under way in *repo*.

        It is the user's, and a task merge keeps out of it: ``git
        stash`` would end it (its reset takes ``MERGE_HEAD`` and the
        like away), a commit made now would be taken for its own, and
        the cleanup of a conflicted merge (``cherry-pick --quit``)
        would forget the picks still to come.

        Args:
            repo: Git repo root path.

        Returns:
            True when one is, also when git could not name the
            checkout's git directory.
        """
        git_dir = _git("rev-parse", "--absolute-git-dir", cwd=repo)
        return git_dir.returncode != 0 or any(
            os.path.lexists(os.path.join(git_dir.stdout.removesuffix("\n"), name))
            for name in _OPERATION_STATE
        )

    @staticmethod
    def stash_if_dirty(repo: Path) -> bool | None:
        """Stash uncommitted changes if the working tree or index is dirty.

        Uses ``git stash push --include-untracked`` so both staged and
        unstaged changes (including new files) are saved, under a
        message of its own: :meth:`stash_pop` finds the entry by it.

        Args:
            repo: Git repo root path.

        Returns:
            True if a stash entry was created, False if the tree was
            clean, None if the tree is dirty but ``git stash push``
            failed, created no entry, or was not run because a git
            operation of the user's is under way (see
            :meth:`operation_in_progress`): the tree is still dirty.
        """
        if not GitWorktreeOps.has_uncommitted_changes(repo):
            return False
        if GitWorktreeOps.operation_in_progress(repo):
            logger.warning("a git operation is under way in %s; not stashing its changes", repo)
            return None
        label = f"kiss: auto-stash before merge {uuid.uuid4().hex[:12]}"
        _merge_stash_labels[repo] = label
        result = _git("stash", "push", "--include-untracked", "-m", label, cwd=repo)
        if result.returncode != 0:
            return None
        return True if GitWorktreeOps.stash_ref(repo) else None

    @staticmethod
    def stash_ref(repo: Path) -> str:
        """Return the SHA of the stash entry a :meth:`stash_if_dirty` of *repo* pushed.

        ``""`` when the stash holds no such entry; one that is gone
        from the stash is forgotten.  A caller that compares the value
        from before a :meth:`stash_if_dirty` with the current one knows
        whether that call created an entry even when it never returned
        (the runner's Stop can interrupt it after ``git stash push``
        completed).  The SHA ``refs/stash`` points at would not tell:
        the user's own ``git stash`` moves it too.

        Args:
            repo: Git repo root path.
        """
        label = _merge_stash_labels.get(repo)
        if label is None:
            return ""
        found = _stash_entry(repo, label)
        if found == ("", -1):
            _merge_stash_labels.pop(repo, None)
        return found[0] if found else ""

    @staticmethod
    def stash_pop(repo: Path, branch: str = "", baseline: str | None = None) -> bool:
        """Give back what the last :meth:`stash_if_dirty` of *repo* stashed.

        The entry is that call's own, not the top of the stash: the
        user's ``git stash`` takes none of the merge's locks and can
        push an entry of theirs during the merge.  After a merge of
        *branch*, the untracked files it superseded are not given back
        (see :meth:`_stash_without_superseded`).

        Tries ``git stash apply --index`` first so that files that were
        staged before the stash stay staged.  If ``--index`` fails
        (e.g. the merge changed a file that was in the index), falls
        back to a plain ``git stash apply`` which restores all changes
        as unstaged — but ONLY when the failed ``--index`` attempt left
        the working tree untouched.  A failed ``--index`` apply can
        partially apply the stash (e.g. tracked changes applied while
        an untracked-file restore failed, or a conflicted merge);
        blindly re-applying the same stash on top of that state risks
        a double-apply, so in that case the stash is left for the
        caller to surface to the user.  An applied entry is dropped.

        Args:
            repo: Git repo root path.
            branch: The task branch a successful merge just applied,
                ``""`` when none did.
            baseline: That branch's baseline commit SHA, or ``None``.

        Returns:
            True if the changes are back or that call stashed none,
            False on conflict or error, also when the entry is no
            longer in the stash or the files *branch* superseded
            could not be told: nothing of it is applied then.
        """
        label = _merge_stash_labels.get(repo)
        if label is None:
            return True
        found = _stash_entry(repo, label)
        if not found or not found[0]:
            return False
        entry = applied = found[0]
        if branch and baseline:
            copy = GitWorktreeOps._stash_without_superseded(repo, entry, branch, baseline)
            if copy is None:
                return False
            applied = copy
        before = GitWorktreeOps._status(repo)
        if _git("stash", "apply", "--index", applied, cwd=repo).returncode != 0:
            after = GitWorktreeOps._status(repo)
            if before.returncode != 0 or after.returncode != 0 or after.stdout != before.stdout:
                # The tree changed, or a status failed so nothing proves it
                # did not (two identical failure reports are not proof):
                # never risk a double-apply.
                return False
            if _git("stash", "apply", applied, cwd=repo).returncode != 0:
                return False
        GitWorktreeOps._drop_stash(repo, label, entry)
        _merge_stash_labels.pop(repo, None)
        return True

    @staticmethod
    def _stash_without_superseded(
        repo: Path, entry: str, branch: str, baseline: str,
    ) -> str | None:
        """Return *entry*, or a copy of it without the untracked files *branch* superseded.

        The baseline commit of a task holds a copy of every file that
        was untracked when the task began.  One the task then changed,
        renamed or deleted is the task's from its merge on: ``git stash
        apply`` refuses the stashed copy where the merge now tracks the
        file (and gives nothing of the entry back), and puts a deleted
        one back.  Such a copy is left out while it still is the
        baseline's, blob and mode; one the user changed since, and
        every other untracked file, stays in.  The copy is a stash
        commit no ref holds: the caller applies it and drops *entry*
        only once that worked.

        Args:
            repo: Git repo root path.
            entry: SHA of the merge's stash commit.
            branch: The task branch that was merged.
            baseline: Its baseline commit SHA.

        Returns:
            The SHA to apply: *entry* itself when it holds no such
            file; ``None`` when git failed on the way: which of its
            files the task superseded is then unknown, and the whole
            entry would put back one the task deleted.
        """
        untracked = f"{entry}^3"  # ``stash push --include-untracked`` always writes it
        names = ("diff", "--name-only", "-z", "--no-renames")
        stashed = GitWorktreeOps._tree_records(repo, untracked)
        in_baseline = GitWorktreeOps._tree_records(repo, baseline)
        added = _git(*names, "--diff-filter=A", f"{baseline}^", baseline, cwd=repo)
        changed = _git(*names, baseline, branch, cwd=repo)
        if (
            stashed is None or in_baseline is None
            or added.returncode != 0 or changed.returncode != 0
        ):
            return None
        # Stashed as the baseline has it, untracked when the task began (the
        # baseline added it) and changed by the task.
        superseded = (
            {record.partition("\t")[2] for record in stashed & in_baseline}
            & set(added.stdout.split("\0"))
            & set(changed.stdout.split("\0"))
        )
        if not superseded:
            return entry
        with tempfile.TemporaryDirectory() as scratch:
            env = {"GIT_INDEX_FILE": os.path.join(scratch, "index")}
            try:
                _git_step(repo, env, "read-tree", untracked)
                _git_step(
                    repo, env, "update-index", "--force-remove", "-z", "--stdin",
                    input="\0".join(superseded),
                )
                tree = _git_step(repo, env, "write-tree").strip()
                kept = _git_step(repo, env, "commit-tree", tree, "-m", "untracked files").strip()
                return _git_step(
                    repo, env, "commit-tree", f"{entry}^{{tree}}", "-p", f"{entry}^1",
                    "-p", f"{entry}^2", "-p", kept, "-m", "kiss: stash without superseded files",
                ).strip()
            except _GitStepError:
                return None

    @staticmethod
    def _tree_records(repo: Path, tree: str) -> set[str] | None:
        """Return the ``git ls-tree -r`` records of the files in *tree*.

        A record is ``<mode> <type> <sha><TAB><path>``: two trees hold
        the same file, blob and mode, when they share its record.

        Args:
            repo: Git repo root path.
            tree: A tree-ish.

        Returns:
            The records; ``None`` when git failed.
        """
        listed = _git("ls-tree", "-r", "-z", tree, cwd=repo)
        return set(filter(None, listed.stdout.split("\0"))) if listed.returncode == 0 else None

    @staticmethod
    def _drop_stash(repo: Path, label: str, entry: str) -> None:
        """Remove the merge's stash entry *entry*, already applied, from the stash.

        ``git stash drop`` takes a position, and the user's ``git
        stash`` moves every position between a look at the list and
        the drop.  Git names the commit it dropped: one that is not
        *entry* is stored back, on top under its message, and the drop
        is tried again.  An entry that keeps escaping stays listed.

        Until it is stored back, the user's entry is in no ref.  So the
        work runs on a thread of its own, into which no Stop is
        injected, and this thread waits for it in one C call, which an
        injected exception cannot leave early (as in :func:`_git`).

        Args:
            repo: Git repo root path.
            label: The message *entry* was pushed under.
            entry: SHA of the merge's stash commit.
        """
        done = threading.Lock()
        done.acquire()
        stop: BaseException | None = None
        try:
            _thread.start_new_thread(
                GitWorktreeOps._drop_stash_unstopped, (repo, label, entry, done),
            )
        except RuntimeError:  # no thread to be had: here, where a Stop can cut it
            GitWorktreeOps._drop_stash_unstopped(repo, label, entry, done)
        except BaseException as raised:  # a Stop as the call returned: the thread runs
            stop = raised
        done.acquire()
        if stop is not None:
            raise stop

    @staticmethod
    def _drop_stash_unstopped(repo: Path, label: str, entry: str, done: threading.Lock) -> None:
        """:meth:`_drop_stash`'s work; releases *done* when it ends."""
        try:
            for _ in range(_STASH_DROP_ATTEMPTS):
                found = _stash_entry(repo, label)
                if found is None:
                    continue
                if not found[0]:
                    return  # the user took it out
                dropped = _git("stash", "drop", f"stash@{{{found[1]}}}", cwd=repo)
                sha = re.search(r"\b[0-9a-f]{40,64}\b", dropped.stdout)
                if sha is None:
                    continue  # nothing was dropped: the position no longer exists
                if sha[0] == entry:
                    return
                logger.warning(
                    "git stash drop in %s removed %s, pushed since the stash was listed; "
                    "storing it back", repo, sha[0],
                )
                subject = _git("show", "-s", "--format=%s", sha[0], cwd=repo).stdout.strip()
                stored = _git("stash", "store", "-m", subject, sha[0], cwd=repo)
                if stored.returncode != 0:
                    logger.error(
                        "the stash entry %s of %s could not be stored back (%s); "
                        "recover it with 'git stash store %s'",
                        subject, repo, stored.stderr.strip(), sha[0],
                    )
                    return
            logger.warning(
                "the stash entry %s of %s is applied but is still in the stash; "
                "remove it with 'git stash drop'", entry, repo,
            )
        finally:
            done.release()

    @staticmethod
    def _merge_commit_message(
        repo: Path,
        branch: str,
        user_prompt: str | None = None,
        task_result: str | None = None,
    ) -> str:
        """Return the commit message to use for a squash-merge commit.

        When the agent has made at least one commit on *branch* (the
        common case — the agent auto-commits its changes before
        merging), this returns the full message (subject + body) of
        the branch's HEAD commit, so the merge into the original
        branch carries the meaningful per-task description rather
        than git's generic ``"Squashed commit of the following:"`` or
        a synthetic ``"kiss: merged from <branch>"`` placeholder.

        Falls back to ``"kiss: merged from <branch>"`` only when the
        branch's HEAD message cannot be read (corrupt branch, git
        error).  Never returns an empty string, because ``git commit``
        rejects empty messages.

        When *user_prompt* / *task_result* are provided, they are
        appended to the message under ``User prompt:`` / ``Result:``
        headings — UNLESS the message already ends with those exact
        canonical blocks carrying the CURRENT values (see
        :func:`_ensure_task_metadata`).  This guarantees the final
        commit on the user's original branch always records the task
        description and its outcome even when the branch HEAD commit
        was hand-written by the agent itself (production incident:
        commit ``dd563a7c`` — the agent committed its own work with
        its own message, the post-task auto-commit was a no-op, and
        the squash-merge reused that message verbatim, losing both
        blocks).  Dedup is by exact current-value suffix so the
        framework auto-commit path (which already appends both
        blocks) is never double-stamped, while incidental heading
        text or stale blocks from a previous task can never suppress
        the current values.

        Args:
            repo: Git repo root path (the working directory that will
                run ``git log``).
            branch: The worktree branch whose HEAD message to read.
            user_prompt: The user's task prompt to append, or ``None``
                when unavailable.
            task_result: The task's result summary to append, or
                ``None`` when unavailable.

        Returns:
            A non-empty commit message string.
        """
        result = _git("log", "-1", "--format=%B", branch, "--", cwd=repo)
        msg = result.stdout.rstrip()
        if result.returncode != 0 or not msg:
            msg = f"kiss: merged from {branch}"
        return _ensure_task_metadata(msg, user_prompt, task_result)

    @staticmethod
    def _commit_staged_merge(
        repo: Path,
        branch: str,
        user_prompt: str | None = None,
        task_result: str | None = None,
    ) -> MergeResult:
        """Commit the staged result of a squash merge, if any.

        Shared tail of :meth:`squash_merge_branch` and
        :meth:`squash_merge_from_baseline`: when the merge staged any
        changes, commit them (:meth:`_commit_merge`); on commit failure
        (e.g. a rejecting pre-commit hook) the merge is taken back by
        git's own check of each file (:meth:`_undo_merge`), not by a
        hard reset: an editor may have saved any tracked file meanwhile.

        Args:
            repo: Git repo root path.
            branch: The worktree branch the changes came from.
            user_prompt: The user's task prompt, appended to the merge
                commit message when not already present (see
                :meth:`_merge_commit_message`), or ``None``.
            task_result: The task's result summary, appended likewise,
                or ``None``.

        Returns:
            :attr:`MergeResult.SUCCESS`; :attr:`MergeResult.MERGE_FAILED`
            when the commit failed and the merge was taken back;
            :attr:`MergeResult.CONFLICT` when it could not be: what is
            left of it stays in the tree for the user to settle, and no
            merge agent is run on it (see
            :func:`~kiss.agents.sorcar.merge_conflict_resolver.resolve_merge_conflict`).
        """
        if not GitWorktreeOps.has_staged_changes(repo):
            return MergeResult.SUCCESS
        if GitWorktreeOps._commit_merge(
            repo, branch, user_prompt=user_prompt, task_result=task_result,
        ):
            return MergeResult.SUCCESS
        if GitWorktreeOps._undo_merge(repo, branch):
            return MergeResult.MERGE_FAILED
        return MergeResult.CONFLICT

    @staticmethod
    def _commit_merge(
        repo: Path,
        branch: str,
        user_prompt: str | None = None,
        task_result: str | None = None,
    ) -> bool:
        """Commit a staged merge whose index the caller has verified is non-empty.

        The commit of :meth:`_commit_staged_merge`, also used directly
        by :meth:`finish_conflicted_merge`, which has already checked
        the index: one ``git diff --cached --quiet`` per merge commit
        instead of three.

        Returns:
            Whether the merge was committed.
        """
        msg = GitWorktreeOps._merge_commit_message(
            repo, branch, user_prompt=user_prompt, task_result=task_result,
        )
        return GitWorktreeOps._git_commit(repo, msg)

    @staticmethod
    def squash_merge_branch(
        repo: Path,
        branch: str,
        user_prompt: str | None = None,
        task_result: str | None = None,
    ) -> MergeResult:
        """Squash-merge a branch and commit the result.

        Uses ``git merge --squash`` to apply all changes from *branch*,
        then commits the staged result using the HEAD commit message
        of *branch* (see :meth:`_merge_commit_message`) with the task
        prompt and result appended when provided and not already
        present in that message.

        On conflict, and when interrupted between the apply and its
        commit, what the merge applied is taken back (see
        :meth:`_undo_merge`).

        Args:
            repo: Git repo root path.
            branch: Branch to squash-merge.
            user_prompt: The user's task prompt for the merge commit
                message, or ``None``.
            task_result: The task's result summary for the merge
                commit message, or ``None``.

        Returns:
            :attr:`MergeResult.SUCCESS`, :attr:`MergeResult.CONFLICT`, or
            :attr:`MergeResult.MERGE_FAILED` when the commit failed.
        """
        return GitWorktreeOps._apply_and_commit(repo, branch, None, user_prompt, task_result)

    @staticmethod
    def _apply_and_commit(
        repo: Path,
        branch: str,
        baseline: str | None,
        user_prompt: str | None,
        task_result: str | None,
    ) -> MergeResult:
        """Apply *branch* onto HEAD (see :meth:`_apply_branch`) and commit what that staged.

        Returns:
            :attr:`MergeResult.SUCCESS`, :attr:`MergeResult.MERGE_FAILED`
            when the commit failed, or :attr:`MergeResult.CONFLICT`
            with what the merge applied taken back (see
            :meth:`_undo_merge`; :meth:`_commit_merge` for a commit
            that failed and could not be reset); a merge the main
            tree's ignored files, an index without a tree (unmerged
            entries of the user's own) or a git operation of theirs
            refuse is not attempted.
        """
        if GitWorktreeOps._refused_for_ignored_files(repo, branch, baseline):
            return MergeResult.CONFLICT
        index_before = GitWorktreeOps._index_tree(repo)
        if index_before is None:
            logger.warning("the index of %s holds no tree; not merging %s", repo, branch)
            return MergeResult.CONFLICT
        try:
            conflicted = GitWorktreeOps._apply_branch(repo, branch, baseline)
            if conflicted == []:
                return GitWorktreeOps._commit_staged_merge(
                    repo, branch, user_prompt=user_prompt, task_result=task_result,
                )
            if conflicted:  # ``None``: git applied nothing, the tree is as the user has it
                GitWorktreeOps._undo_merge(repo, branch)
            return MergeResult.CONFLICT
        except BaseException:
            GitWorktreeOps._undo_if_applied(repo, branch, index_before)
            raise

    @staticmethod
    def _undo_if_applied(repo: Path, branch: str, index_before: str) -> None:
        """Take back what an interrupted merge of *branch* applied, if anything.

        A Stop landing after the merge applied but before its commit
        (``_git`` has stopped the git child by now) would leave the
        agent's changes staged on the user's branch.  What git did is
        unknown here: an index that still holds *index_before*, the
        tree from before the merge, says it applied nothing.
        """
        if GitWorktreeOps._index_tree(repo) != index_before:
            GitWorktreeOps._undo_merge(repo, branch)

    @staticmethod
    def _index_tree(repo: Path) -> str | None:
        """Return the SHA of the tree the index of *repo* holds.

        ``None`` when it holds none (unmerged entries: a merge stopped
        on conflicts there) or git failed.
        """
        result = _git("write-tree", cwd=repo)
        return result.stdout.strip() if result.returncode == 0 else None

    @staticmethod
    def _undo_merge(repo: Path, branch: str) -> bool:
        """Take back a ``merge --squash`` / ``cherry-pick --no-commit`` that ended in no commit.

        The files the merge wrote go back to ``HEAD`` by git's own
        check of each (:meth:`_move_written_files`), from the tree the
        merge wrote: ``AUTO_MERGE``, conflict markers included, or the
        tree of *branch* where git recorded none (a fast-forward:
        ``merge --squash`` of a branch ahead of ``HEAD``).  Then the
        index is reset, and the merge's ``MERGE_MSG`` / ``SQUASH_MSG``
        with it; what the user staged meanwhile is an unstaged edit
        from then on.  A file that no longer is what the merge wrote
        (an editor that saves a file takes none of the merge's locks)
        is never reset: the checkout then stays as it is, conflicts
        included, for the user to settle, and an index that holds them
        refuses the next merge.  So it does when the index cannot be
        reset (a git command of the user's holds it): the files are
        put back.  There is no ``cherry-pick --abort``: ``--no-commit``
        starts no sequence.

        Args:
            repo: Git repo root path, with the merge applied.
            branch: The branch that was merged.

        Returns:
            Whether the merge was taken back; when not, a warning says
            what the checkout holds.
        """
        if GitWorktreeOps._take_back(repo, GitWorktreeOps._written_tree(repo, branch), "HEAD"):
            return True
        logger.warning(
            "the merge applied to %s is not taken back, or not wholly:\n%s",
            repo, GitWorktreeOps.status_porcelain(repo),
        )
        return False

    @staticmethod
    def _written_tree(repo: Path, branch: str) -> str:
        """Return the tree the merge of *branch* that git just applied wrote.

        The id of the tree ``AUTO_MERGE`` names (a later git command
        drops the name), or *branch* where git recorded none.
        """
        auto_merge = _git("rev-parse", "-q", "--verify", "AUTO_MERGE^{tree}", cwd=repo)
        return auto_merge.stdout.strip() if auto_merge.returncode == 0 else branch

    @staticmethod
    def _take_back(repo: Path, written: str, head: str) -> bool:
        """Move the working files from tree *written* to commit *head*, then the index.

        The files go by git's own check of each
        (:meth:`_move_written_files`); they are put back when the
        index then cannot be reset.

        Returns:
            Whether files and index hold *head*.
        """
        if GitWorktreeOps._move_written_files(repo, written, head):
            if _git("reset", "-q", head, cwd=repo).returncode == 0:
                return True
            GitWorktreeOps._move_written_files(repo, head, written)
        return False

    @staticmethod
    def _tree_with_files(repo: Path, tree: str, paths: list[str]) -> str | None:
        """Return *tree* with each of *paths* as the working tree holds it now.

        The file's content, or no entry where no file is.  Built in an
        index of its own.

        Returns:
            The id of that tree, or ``None`` when git failed (logged):
            it cannot take what one of the paths holds.
        """
        with tempfile.TemporaryDirectory() as scratch:
            env = {"GIT_INDEX_FILE": os.path.join(scratch, "index")}
            try:
                _git_step(repo, env, "read-tree", tree)
                _git_step(
                    repo, env, "update-index", "--add", "--remove", "-z", "--stdin",
                    input="".join(f"{path}\0" for path in paths),
                )
                return _git_step(repo, env, "write-tree").strip()
            except _GitStepError:
                return None

    @staticmethod
    def _move_written_files(repo: Path, written: str, target: str) -> bool:
        """Turn the working files of tree *written* into those of tree *target*, unless one changed.

        Git's own check of each file decides, as late as git itself
        looks before it writes (``read-tree -m -u``), in an index of
        its own that holds *written*: a file the two trees differ in
        that is no longer what *written* holds (it was saved since, or
        git cannot read it to tell) refuses the whole move, and nothing
        is written; so does a file found where the move creates one
        (:meth:`_guarded_tree`).  Only a write that fails after that
        check (a required filter, the disk) stops git half-way, with
        the files it replaced so far.  That index starts as a copy of
        the index of *repo*, date included: its stat data tells an
        untouched file without reading it, and its date which of them
        to read anyway (one written within the same second).  The index
        of *repo* is not touched, nor is a file both trees hold alike,
        whatever it holds now.

        Args:
            repo: Git repo root path.
            written: The tree-ish the working files were written from.
            target: The tree-ish they are to hold.

        Returns:
            Whether the files were moved; when not, what refused is logged.
        """
        index = _git("rev-parse", "--git-path", "index", cwd=repo)
        with tempfile.TemporaryDirectory() as scratch:
            env = {"GIT_INDEX_FILE": os.path.join(scratch, "index")}
            with contextlib.suppress(OSError):  # without the copy every file is read
                shutil.copy2(repo / index.stdout.removesuffix("\n"), env["GIT_INDEX_FILE"])
            try:
                _git_step(repo, env, "read-tree", "--reset", written)
                _git_step(repo, env, "update-index", "-q", "--refresh")
                source = GitWorktreeOps._guarded_tree(repo, env, written, target)
                _git_step(repo, env, "read-tree", "-m", "-u", source, target)
            except _GitStepError:
                return False
        return True

    @staticmethod
    def _guarded_tree(repo: Path, env: dict[str, str], written: str, target: str) -> str:
        """Return *written* as a tree whose move to *target* writes over no file git ignores.

        ``read-tree -m -u`` refuses to create a file where it finds one
        git does not track, unless git ignores that one: it is written
        over, or removed with the directory a file takes the place of.
        So each path the move creates gets a placeholder, in the index
        *env* names and in the tree returned: a symbolic link no file
        on disk is.  Git then checks the path like any tracked file,
        as it writes, and whatever it finds there refuses the move.  A
        submodule is left to git's own rule: its checkout may be there.
        A path that takes the place of files the move removes (a file
        for a directory or the reverse, a name in another case) cannot
        hold a placeholder beside them: what git does not track there
        is listed before the move instead, and refuses it.

        Raises:
            _GitStepError: git failed, or such a listing is not empty.
        """
        fields = _git_step(
            repo, env, "diff", "--raw", "-z", "--no-renames", "--diff-filter=AD",
            written, target, "--",
        ).split("\0")
        changes = list(zip(fields[::2], fields[1::2]))  # (":<modes> <ids> <A or D>", path)
        removed = {path.casefold() for change, path in changes if change.endswith("D")}
        replaced = removed.union(*map(_leading_dirs, removed))
        placeholders: list[str] = []
        clashing: list[str] = []
        for change, path in changes:
            if change.endswith("A") and change.split()[1] != "160000":
                folded = path.casefold()
                clash = folded in replaced or not removed.isdisjoint(_leading_dirs(folded))
                (clashing if clash else placeholders).append(path)
        if clashing:
            others = _git_step(
                repo, env, "--literal-pathspecs", "ls-files", "--others", "--directory",
                "--no-empty-directory", "-z", "--", *clashing,
            )
            if others:
                logger.warning(
                    "the files of %s are not moved: git does not track %s",
                    repo, others.replace("\0", " ").strip(),
                )
                raise _GitStepError
        if not placeholders:
            return written
        blob = _git_step(
            repo, env, "hash-object", "-w", "--stdin", input="kiss: git wrote no file here\n",
        ).strip()
        _git_step(
            repo, env, "update-index", "-z", "--index-info",
            input="".join(f"120000 {blob}\t{path}\0" for path in placeholders),
        )
        return _git_step(repo, env, "write-tree").strip()

    @staticmethod
    def _diff_name_only(repo: Path, *flags: str) -> list[str]:
        """Return ``git diff --name-only`` paths (empty on failure).

        Uses ``-z`` NUL-separated output and splits on NUL only, so
        paths are returned byte-exact: no C-quoting to undo (git never
        quotes ``-z`` output), no ``strip()``/``splitlines()`` that
        would mangle filenames with leading/trailing spaces or raw
        unicode line-separator sequences.  Callers compare these names
        against real on-disk paths.
        """
        result = _git("diff", "--name-only", "-z", *flags, cwd=repo)
        if result.returncode != 0:
            return []
        return [f for f in result.stdout.split("\0") if f]

    @staticmethod
    def _remove_branch_config_section(repo: Path, branch: str) -> None:
        """Remove the ``branch.<name>.*`` git config section (best-effort)."""
        _git("config", "--remove-section", f"branch.{branch}", cwd=repo)

    @staticmethod
    def delete_branch(repo: Path, branch: str) -> bool:
        """Delete a branch, its git config section and its preserve marker (best-effort).

        Uses ``-D`` (force): every caller deletes an agent-owned branch
        whose commits were merged, discarded or reclaimed, so a
        preceding safe ``-d`` could only ever add a git process.  Also
        removes the ``branch.<name>.*`` config section and the
        preserve-for-review marker file.

        Args:
            repo: Git repo root path.
            branch: Branch name to delete.

        Returns:
            True if the branch was deleted (or never existed), False
            if git refused ``-D`` — typically because the branch is the
            current HEAD of a worktree and cannot be deleted without
            first switching away.
        """
        result = _git("branch", "-D", branch, cwd=repo)
        if result.returncode != 0 and GitWorktreeOps.branch_exists(repo, branch):
            logger.warning(
                "Failed to delete branch '%s': %s",
                branch,
                result.stderr.strip(),
            )
            return False
        GitWorktreeOps._remove_branch_config_section(repo, branch)
        marker = GitWorktreeOps._preserve_marker_path(repo, branch)
        if marker is not None:
            with contextlib.suppress(OSError):
                marker.unlink()
        _volatile_preserve_claims.discard((str(repo.resolve()), branch))
        return True

    @staticmethod
    def branch_exists(repo: Path, branch: str) -> bool:
        """Check if a branch exists.

        Args:
            repo: Git repo root path.
            branch: Branch name to check.

        Returns:
            True if the branch exists.
        """
        result = _git("rev-parse", "--verify", f"refs/heads/{branch}", cwd=repo)
        return result.returncode == 0

    @staticmethod
    def new_task_branch(repo: Path) -> str:
        """Mint a unique ``kiss/wt-*`` branch name for *repo*.

        ``kiss/wt-<epoch>-<uuid8>``, with a numeric suffix appended in
        the (practically impossible) case of a collision.

        Args:
            repo: Git repo root path.

        Returns:
            A branch name that does not currently exist in *repo*.
        """
        branch = f"{_WORKTREE_BRANCH_PREFIX}{int(time.time())}-{uuid.uuid4().hex[:8]}"
        base_branch = branch
        suffix = 1
        while GitWorktreeOps.branch_exists(repo, branch):  # pragma: no branch
            branch = f"{base_branch}-{suffix}"
            suffix += 1
        return branch

    @staticmethod
    def is_ignored(repo: Path, path: Path) -> bool:
        """Whether *path* (existing or not) is git-ignored in *repo*.

        Args:
            repo: Git repo root path.
            path: Absolute path inside *repo*.

        Returns:
            True when a ``.gitignore`` / exclude rule matches *path*.
        """
        return _git("check-ignore", "-q", "--", str(path), cwd=repo).returncode == 0

    @staticmethod
    def _append_info_line(repo: Path, filename: str, entry: str) -> None:
        """Append *entry* to ``<git_common_dir>/info/<filename>`` once.

        The ``info/`` directory holds repo-local, untracked plumbing
        (``exclude``, ``attributes``) so the agent never modifies any
        tracked file in the user's repo.  Idempotent: the line is only
        appended when not already present.

        The read-then-append is performed while holding an exclusive
        ``flock`` on the file itself, so the idempotence also holds
        across PROCESSES (daemon + CLI, or two checkouts sharing a
        ``--git-common-dir``).  ``repo_lock`` alone would let both
        processes read a file without the entry and both append it,
        duplicating the line on every racing pair of runs.

        Args:
            repo: Git repo root path.
            filename: File under ``info/`` (e.g. ``"exclude"``).
            entry: The exact line to ensure is present.
        """
        locked_call(
            repo_lock(repo), GitWorktreeOps._append_info_line_locked, repo, filename, entry,
        )

    @staticmethod
    def _append_info_line_locked(repo: Path, filename: str, entry: str) -> None:
        """:meth:`_append_info_line`'s body, run under ``repo_lock``."""
        git_common = _git_common_dir(repo)
        if git_common is None:  # pragma: no cover — not a git repo
            raise OSError(f"{repo} is not a git repository")
        info_file = git_common / "info" / filename
        info_file.parent.mkdir(parents=True, exist_ok=True)
        with open(
            info_file, "a+", encoding="utf-8", errors="surrogateescape",
        ) as f, _file_lock(f):
            f.seek(0)
            content = f.read()
            if entry in content.splitlines():
                return
            _race_delay()
            prefix = "" if not content or content.endswith("\n") else "\n"
            f.write(f"{prefix}{entry}\n")
            # Flush while the flock is still held: the context
            # managers unwind in reverse order, releasing the lock
            # BEFORE ``open()``'s exit flushes the buffered write.
            # Without this a second process could grab the flock,
            # read the still-unflushed file, miss the entry, and
            # append a duplicate.
            f.flush()

    @staticmethod
    def ensure_excluded(repo: Path) -> None:
        """Add ``.kiss-worktrees/`` to local git exclude (not .gitignore).

        Uses ``<git_common_dir>/info/exclude`` so the agent never modifies
        any tracked file in the user's repo.

        Args:
            repo: Git repo root path.
        """
        GitWorktreeOps._append_info_line(repo, "exclude", ".kiss-worktrees/")

    @staticmethod
    def enable_untracked_cache(repo: Path) -> None:
        """Turn on git's untracked cache for *repo* unless already configured.

        Worktree setup, spare validation and dirty-state copying each
        run ``git status`` or ``ls-files --others`` over the whole
        tree, and on a repository with 100k+ tracked files every such
        walk costs about half a second when the directory cache is
        cold.  ``core.untrackedCache=true`` makes git record directory
        mtimes in the index so a later walk re-reads only directories
        that changed.  The setting lives in the repository's local
        config, which its worktrees share, and is written once; a
        value the user already set (``true`` or ``false``) is kept.
        ``core.fsmonitor`` is deliberately not touched: the builtin
        fsmonitor daemon is unsupported on Linux, where enabling it
        makes every ``git status`` fail.

        Args:
            repo: Git repo root path.
        """
        if _git("config", "--get", "core.untrackedCache", cwd=repo).returncode == 0:
            return
        result = _git("config", "core.untrackedCache", "true", cwd=repo)
        if result.returncode != 0:
            logger.warning(
                "Could not enable core.untrackedCache in %s: %s",
                repo, result.stderr.strip(),
            )

    @staticmethod
    def prepare_repo(repo: Path) -> None:
        """One-time repository setup before worktree work: exclude + untracked cache.

        Args:
            repo: Git repo root path.
        """
        GitWorktreeOps.ensure_excluded(repo)
        GitWorktreeOps.enable_untracked_cache(repo)

    @staticmethod
    def link_node_modules(repo: Path, wt_dir: Path) -> list[Path]:
        """Symlink the main checkout's ``node_modules`` directories into *wt_dir*.

        A fresh worktree checks out tracked files only, so every
        git-ignored ``node_modules`` of the main checkout is missing and
        JS tests and the webapp lint fail until the agent runs ``npm
        install`` (minutes, hundreds of MB) or links the directory by
        hand — a 24-hour audit of ``history.db`` found seven worktree
        tasks reinstalling and several rerunning a 340-file suite after
        "missing node_modules" failures.

        Each ignored directory named ``node_modules`` reported by ``git
        ls-files --ignored --directory`` in *repo* is linked at the same
        relative path of *wt_dir* when that path is still free and its
        parent exists.  ``.gitignore`` patterns ending in ``/``
        (``node_modules/``) match directories only, not symlinks, so the
        bare pattern ``node_modules`` is added to ``info/exclude`` (shared
        by every worktree of the repository) to keep the link out of
        ``git add -A`` and the squash merge.  Compiled output (``out/``)
        is deliberately not linked: it must be rebuilt from the
        worktree's own sources.

        Args:
            repo: Git repo root path (the main checkout).
            wt_dir: The new worktree's root.

        Returns:
            The repo-relative paths that were linked (empty when the
            main checkout has no ignored ``node_modules``).

        Raises:
            OSError: If ``info/exclude`` cannot be written (no link is
                created then) or a symlink cannot be made.
        """
        result = _git(
            "ls-files", "-z", "--others", "--ignored", "--exclude-standard",
            "--directory", cwd=repo,
        )
        if result.returncode != 0:
            return []
        wanted: list[Path] = []
        for entry in result.stdout.split("\0"):
            rel = Path(entry.rstrip("/"))
            if rel.name != "node_modules":
                continue
            src = repo / rel
            dst = wt_dir / rel
            if not src.is_dir() or dst.is_symlink() or dst.exists() or not dst.parent.is_dir():
                continue
            wanted.append(rel)
        if not wanted:
            return []
        # The exclusion goes in first: a link created before it would be a
        # plain untracked symlink that ``git add -A`` stages, so when the
        # exclude file cannot be written (OSError propagates) no link is made.
        GitWorktreeOps._append_info_line(repo, "exclude", "node_modules")
        for rel in wanted:
            os.symlink((repo / rel).resolve(), wt_dir / rel)
        return wanted

    @staticmethod
    def block_own_environment(repo: Path, wt_dir: Path) -> bool:
        """Put ``.venv`` files in *wt_dir*, so no second Python environment grows there.

        A worktree shares the main checkout's environment
        (``<repo>/.venv``).  A ``uv run`` or ``uv sync`` the Bash tool
        does not see as one (``bash -c "uv run …"``, a script, a
        Makefile) used to build the worktree an environment of its own,
        on whatever Python uv found newest: on 2026-11-11 a full test
        run and 43 helpers ran on Python 3.14.7 while the fixers ran on
        the project's 3.13.5.  uv refuses a project environment path
        that is a file ("cannot be used because expected directory but
        found a file") in every form and never replaces it, so one
        Python per worktree holds without reading any command; a call
        that names the main environment (``UV_PROJECT_ENVIRONMENT``, as
        the ``check`` script does) is unaffected.  A link to the main
        environment would not do: every such ``uv run`` syncs first and
        re-points the environment's editable install at the worktree.

        The file says where the environment is.  It goes into the
        worktree's root and into every directory that holds a tracked
        ``pyproject.toml``: uv run from or at such a directory (``cd
        sub``, ``--directory sub``, ``--project sub``) would build
        ``sub/.venv``.  Like the ``node_modules`` links the files are
        kept out of ``git add -A`` through ``info/exclude``, written
        first, and only a file git then ignores is written: a
        ``.gitignore`` that un-ignores the name (``!.venv``) wins over
        ``info/exclude``, and the file, committed with the task's work
        and merged, would replace the main checkout's environment.
        There the blocker is a directory instead: its own ``.gitignore``
        of ``*`` keeps git from staging anything in it whatever the
        repository's rules say, uv refuses a directory that is no
        environment as it refuses a file ("not a valid Python
        environment"), and the text is its ``bin``, a file, so that
        ``python -m venv`` builds nothing there either.

        Args:
            repo: Git repo root path (the main checkout).
            wt_dir: The new worktree's root.

        Returns:
            Whether a blocker was written.  None is when the main checkout
            has no ``.venv``: there is no environment to share, so a
            uv run in the worktree, typed (which
            :func:`~kiss.agents.sorcar.useful_tools.uv_in_worktree`
            then leaves as written) or not, builds the worktree's own,
            its one Python.  A directory that already has something at
            that path gets none.

        Raises:
            OSError: If ``info/exclude`` or a file cannot be written.
        """
        environment = repo / ".venv"
        if not environment.is_dir():
            return False
        tracked = _git("ls-files", "-z", "--", ":(glob)**/pyproject.toml", cwd=wt_dir)
        projects = {Path(), *(Path(entry).parent for entry in tracked.stdout.split("\0") if entry)}
        free = [
            (project / ".venv").as_posix() for project in sorted(projects)
            if not os.path.lexists(wt_dir / project / ".venv")
        ]
        if not free:
            return False
        GitWorktreeOps._append_info_line(repo, "exclude", ".venv")
        # Asked before writing: a file git would not ignore is never there for ``git add -A``.
        ignored = _git(
            "check-ignore", "-z", "--stdin", cwd=wt_dir, input="\0".join(free),
        ).stdout.split("\0")
        text = OWN_ENVIRONMENT_BLOCKER.format(environment=environment, worktree=wt_dir)
        for path in free:
            blocker = wt_dir / path
            if path not in ignored:
                blocker.mkdir()
                (blocker / ".gitignore").write_text("*\n", encoding="utf-8")
                blocker = blocker / "bin"
            blocker.write_text(text, encoding="utf-8")
        return True

    @staticmethod
    def ensure_scratch_merge_driver(repo: Path) -> None:
        """Install a merge driver that auto-resolves agent scratch files.

        ``PROGRESS.md`` is a tracked per-task agent log that every task
        wholesale rewrites ("clear PROGRESS.md when a new task begins"),
        and ``src/kiss/INJECTIONS.md`` is an agent-maintained scratch
        prompt file that can drift between task baselines.  Whenever
        main's copy diverged from the worktree's fork point, these
        scratch-file rewrites made three-way merges (``git merge
        --squash`` and ``git cherry-pick`` alike) conflict — blocking
        the entire worktree merge over non-user source state.

        This registers a repo-local ``kiss-scratch`` merge driver that
        resolves content conflicts in such files by keeping the
        incoming branch's version (``%B``): the newest task's scratch
        state wins, matching the clear-on-new-task convention.
        Installation uses
        only untracked plumbing — ``<git_common_dir>/info/attributes``
        plus repo-local config — so no tracked file is ever modified,
        and the driver equally fixes the *manual* merge/cherry-pick
        commands suggested to the user on merge failure.

        Args:
            repo: Git repo root path.
        """
        for scratch_path in ("PROGRESS.md", "src/kiss/INJECTIONS.md"):
            GitWorktreeOps._append_info_line(
                repo, "attributes", f"{scratch_path} merge=kiss-scratch"
            )
        # Called on every worktree setup (just before the orphan reclaim
        # that may merge) and merge; the values never change, so read
        # before rewriting ``.git/config``.
        driver = "cp -f %B %A"
        current = _git("config", "--get", "merge.kiss-scratch.driver", cwd=repo)
        if current.returncode == 0 and current.stdout.strip() == driver:
            return
        _git(
            "config",
            "merge.kiss-scratch.name",
            "KISS scratch file: keep the incoming task branch version",
            cwd=repo,
        )
        _git("config", "merge.kiss-scratch.driver", driver, cwd=repo)

    @staticmethod
    def _save_branch_config(
        repo: Path, branch: str, key: str, value: str, what: str,
    ) -> bool:
        """Write ``branch.<branch>.<key>`` to git config.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.
            key: Config key under the branch section.
            value: Value to store.
            what: Human-readable description for the failure log.

        Returns:
            True if config was saved successfully, False otherwise.
        """
        result = _git("config", f"branch.{branch}.{key}", value, cwd=repo)
        if result.returncode != 0:  # pragma: no cover — git config failure
            logger.warning(
                "Failed to store %s in git config: %s",
                what,
                result.stderr.strip(),
            )
            return False
        return True

    @staticmethod
    def save_original_branch(repo: Path, branch: str, original: str) -> bool:
        """Store the original branch in git config.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.
            original: The original branch to store.

        Returns:
            True if config was saved successfully, False otherwise.
        """
        return GitWorktreeOps._save_branch_config(
            repo, branch, "kiss-original", original, "original branch"
        )

    @staticmethod
    def save_baseline_commit(
        repo: Path,
        branch: str,
        sha: str,
    ) -> bool:
        """Store the baseline commit SHA in git config.

        The baseline commit captures the user's dirty state (staged,
        unstaged, untracked files) at worktree creation time.  Downstream
        operations diff against this SHA to isolate agent-only changes.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.
            sha: The baseline commit SHA to store.

        Returns:
            True if config was saved successfully, False otherwise.
        """
        return GitWorktreeOps._save_branch_config(
            repo, branch, "kiss-baseline", sha, "baseline commit"
        )

    @staticmethod
    def _load_branch_config(repo: Path, branch: str, key: str) -> str | None:
        """Read ``branch.<branch>.<key>`` from git config (best-effort).

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.
            key: The config key set previously by
                :meth:`_save_branch_config` (e.g. ``"kiss-original"``,
                ``"kiss-baseline"``).

        Returns:
            The stored value, or ``None`` when the key is absent or
            git itself failed.
        """
        result = _git("config", "--get", f"branch.{branch}.{key}", cwd=repo)
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        return value or None

    @staticmethod
    def load_original_branch(repo: Path, branch: str) -> str | None:
        """Return the original branch stored for *branch*, or ``None``.

        Reciprocal of :meth:`save_original_branch`.  Used by
        :meth:`reclaim_orphaned_worktrees` to rehydrate a worktree
        whose owning agent process died before it could publish or
        release its work.
        """
        return GitWorktreeOps._load_branch_config(repo, branch, "kiss-original")

    @staticmethod
    def load_baseline_commit(repo: Path, branch: str) -> str | None:
        """Return the baseline commit SHA stored for *branch*, or ``None``.

        Reciprocal of :meth:`save_baseline_commit`.  Used by
        :meth:`reclaim_orphaned_worktrees` to decide between
        :meth:`squash_merge_from_baseline` (baseline present) and
        :meth:`squash_merge_branch` (legacy worktree without a
        baseline).
        """
        return GitWorktreeOps._load_branch_config(repo, branch, "kiss-baseline")

    @staticmethod
    def save_preserve_marker(repo: Path, branch: str) -> bool:
        """Mark *branch* as intentionally preserved for manual review.

        When :meth:`SorcarAgent._retire_previous_worktree`
        deliberately leaves a worktree on disk (task stopped, pre-commit
        hook rejection, or a task that never merged), the
        original agent process may die before the user chooses
        merge/discard.  Reclaim on next startup must NOT silently
        publish that preserved work — the persisted marker makes
        that decision durable across process restarts.

        The marker is a file of its own under the repo's git dir
        (:meth:`_preserve_marker_path`), created with one ``touch``: it
        contends with nothing, neither a repo lock (a live owner's
        worktree is never reclaimed by another process, and waiting for
        one would park the caller behind a sibling's merge for as long
        as that merge runs) nor git's own ``.git/config.lock``, which a
        marker kept in ``git config`` lost to any concurrent git command
        holding that file.  The file goes away with the branch
        (:meth:`delete_branch`).

        When the write fails all the same (an unwritable git dir) the
        decision is recorded in the process-local
        :data:`_volatile_preserve_claims` instead, so within this
        process :meth:`load_preserve_marker` already reports the branch
        as preserved; the caller keeps its in-memory claim and retries
        later, and the successful retry drops the volatile record.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.

        Returns:
            True if the marker was saved successfully.
        """
        key = (str(repo.resolve()), branch)
        saved = False
        path = GitWorktreeOps._preserve_marker_path(repo, branch)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                saved = True
            except OSError:
                logger.warning(
                    "Failed to write the preserve-for-review marker of branch '%s'",
                    branch, exc_info=True,
                )
        if saved:
            _volatile_preserve_claims.discard(key)
        else:
            _volatile_preserve_claims.add(key)
        return saved

    @staticmethod
    def load_preserve_marker(repo: Path, branch: str) -> bool:
        """Return True when *branch* is preserved for manual review.

        Either the durable marker is on disk, or this process holds a
        volatile claim for *branch* because its marker write failed
        (see :meth:`save_preserve_marker`).
        """
        if GitWorktreeOps._preserve_marker_on_disk(repo, branch):
            return True
        return (str(repo.resolve()), branch) in _volatile_preserve_claims

    @staticmethod
    def _preserve_marker_on_disk(repo: Path, branch: str) -> bool:
        """Return True when *branch*'s durable preserve marker exists.

        The marker file, or the ``branch.<name>.kiss-preserve`` git
        config key earlier versions wrote.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.
        """
        path = GitWorktreeOps._preserve_marker_path(repo, branch)
        if path is not None and path.exists():
            return True
        return GitWorktreeOps._load_branch_config(repo, branch, "kiss-preserve") == "1"

    @staticmethod
    def _preserve_marker_path(repo: Path, branch: str) -> Path | None:
        """Return the preserve-for-review marker file of *branch*, or ``None``.

        ``<git common dir>/kiss-preserve/<branch>``, the branch name
        percent-encoded into one path component; ``None`` outside a
        git repo.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.
        """
        marker_dir = GitWorktreeOps._preserve_marker_dir(repo)
        if marker_dir is None:  # pragma: no cover — not a git repo
            return None
        return marker_dir / urllib.parse.quote(branch, safe="")

    @staticmethod
    def _preserve_marker_dir(repo: Path) -> Path | None:
        """Return ``<git common dir>/kiss-preserve``, or ``None`` outside a git repo."""
        git_common = _git_common_dir(repo)
        if git_common is None:  # pragma: no cover — not a git repo
            return None
        return git_common / "kiss-preserve"

    @staticmethod
    def _sweep_preserve_markers(repo: Path, existing: set[str]) -> None:
        """Unlink preserve markers whose ``kiss/wt-*`` branch no longer exists.

        ``git branch -D`` by hand drops the branch (and, in current git,
        its config section) but not the marker file, which would
        otherwise stay under ``kiss-preserve/`` forever.  *existing*
        comes from the caller's SUCCESSFUL ref enumeration: a marker is
        only retired because that listing lacks its branch, never because
        a per-branch lookup failed (a git error or timeout must not erase
        the user's preserve-for-review decision, or the next reclaim
        would merge the unaccepted work).

        Args:
            repo: Git repo root path.
            existing: Every ``kiss/wt-*`` branch of *repo*.
        """
        marker_dir = GitWorktreeOps._preserve_marker_dir(repo)
        if marker_dir is None or not marker_dir.is_dir():
            return
        for marker in marker_dir.iterdir():
            branch = urllib.parse.unquote(marker.name)
            if branch.startswith(_WORKTREE_BRANCH_PREFIX) and branch not in existing:
                with contextlib.suppress(OSError):
                    marker.unlink()

    @staticmethod
    def _bounded_identity(identity: str) -> str:
        """Return *identity* in at most 4096 characters, unchanged when it already fits.

        A :func:`process_identity` holds the whole command line and the
        stamp is one ``git config`` argument, which the kernel caps at
        128 KB.  A longer identity keeps its head and ends in a digest
        of the whole; applying this twice changes nothing, so both
        sides of the comparison in :meth:`_live_peer_owner` pass through it.
        """
        if len(identity) <= 4096:
            return identity
        digest = hashlib.sha256(identity.encode(errors="replace")).hexdigest()
        return f"{identity[: 4096 - 72]} sha256:{digest}"

    @staticmethod
    def save_owner_pid(repo: Path, branch: str) -> bool:
        """Record this process as the live owner of *branch*'s worktree.

        ``repo_lock`` and the reclaim exclusion sets are process-local,
        so a SECOND Sorcar process (kiss-web daemon vs. a ``kiss`` CLI
        run) used to be able to reclaim — auto-commit, squash-merge,
        and DELETE — a worktree whose owning task was still running in
        the first process (gpt-5.6-sol review finding).  The persisted
        owner pid lets :meth:`reclaim_orphaned_worktrees` in any
        process skip worktrees whose owner is still alive.

        The pid alone is ambiguous once the owner has exited and the OS
        has handed its pid to a stranger (routine on Windows, where pids
        are recycled within seconds), so the owner's
        :func:`process_identity` (start time + executable) is stored
        beside it when available and :meth:`_live_peer_owner` compares both.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.

        Returns:
            True if the config was saved successfully.
        """
        identity = process_identity(os.getpid())
        if identity is not None and not GitWorktreeOps._save_branch_config(
            repo, branch, "kiss-owner-identity",
            f"{os.getpid()}:{GitWorktreeOps._bounded_identity(identity)}",
            "owner identity",
        ):
            return False
        return GitWorktreeOps._save_branch_config(
            repo, branch, "kiss-owner-pid", str(os.getpid()), "owner pid",
        )

    @staticmethod
    def load_owner_pid(repo: Path, branch: str) -> int | None:
        """Return the owner pid stored for *branch*, or ``None``.

        Reciprocal of :meth:`save_owner_pid`.  ``None`` for legacy
        worktrees created before owner tracking, or when git failed.
        """
        raw = GitWorktreeOps._load_branch_config(
            repo, branch, "kiss-owner-pid",
        )
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @staticmethod
    def _live_peer_owner(repo: Path, branch: str) -> int | None:
        """Return the pid of another live process that stamped *branch*, else ``None``.

        The stamp is :meth:`save_owner_pid`'s.  This process's own pid
        never counts: within one process the callers' exclusion sets
        say which branches are live.  A live pid is not enough either:
        the owner may have exited and the OS may have recycled its pid.
        When the stamp also recorded the owner's
        :func:`process_identity` (prefixed with the pid it describes),
        the current holder of the pid must match it; a stamp without a
        matching identity (legacy, ``ps`` unavailable, or the pid
        re-stamped by hand) falls back to the pid probe alone, and so
        does a stamp whose current holder cannot be fingerprinted right
        now (``ps`` failing): only a different identity proves that the
        pid was recycled.

        Args:
            repo: Git repo root path.
            branch: The worktree branch name.

        Returns:
            The owner's pid while that other process still runs;
            ``None`` for no stamp, this process's stamp, or a gone owner.
        """
        owner_pid = GitWorktreeOps.load_owner_pid(repo, branch)
        if owner_pid is None or owner_pid == os.getpid() or not pid_alive(owner_pid):
            return None
        stamped = GitWorktreeOps._load_branch_config(repo, branch, "kiss-owner-identity")
        prefix = f"{owner_pid}:"
        if stamped is None or not stamped.startswith(prefix):
            return owner_pid
        current = process_identity(owner_pid)
        bounded = GitWorktreeOps._bounded_identity
        same = current is None or bounded(current) == bounded(stamped[len(prefix):])
        return owner_pid if same else None

    @staticmethod
    def load_spare_marker(repo: Path, branch: str) -> bool:
        """Return True when *branch* carries a spare-worktree marker.

        Older releases pre-created spare worktrees (a pool) and marked
        their branches ``kiss-spare``; the orphan reclaim still
        recognises a leftover spare so it is discarded, never merged.
        """
        return GitWorktreeOps._load_branch_config(
            repo, branch, "kiss-spare",
        ) == "1"

    @staticmethod
    def spare_has_content(repo: Path, branch: str, wt_dir: Path) -> bool:
        """Whether a leftover spare worktree holds anything worth preserving.

        The framework never wrote into a spare, so content in one
        means an external writer put it there — and the reclaim of an
        orphaned spare must preserve it instead of destroying it.
        The probe is one ``git status --porcelain
        --ignored`` walk: it reports staged, unstaged and untracked
        changes AND ignored files, which plain porcelain status omits
        — a fresh spare checkout has none, so any present were written
        externally.  One walk instead of two matters on the submit
        path, where a spare is validated before it is consumed.  A
        failed status counts as content: unenumerable state is
        preserved, never destroyed.

        One state the status walk cannot be trusted on: a spare whose
        gitdir has no ``index`` file.  The pool registered a spare with
        ``--no-checkout`` and populated it with ``reset --hard``; a
        process killed during that reset left the files partially written and
        the index never created, so status lists every tracked file as
        a staged deletion and the written ones as untracked.  That is
        checkout debris, not external content — treating it as content
        preserved such a spare on every reclaim pass forever — so
        :meth:`_interrupted_checkout_has_content` filters it out and
        counts only paths outside the ``HEAD`` tree and commits unique
        to the branch.

        Args:
            repo: Git repo root path.
            branch: The spare's ``kiss/wt-*`` branch name.
            wt_dir: The spare's worktree directory.

        Returns:
            True when the spare has uncommitted changes, ignored files
            (or git could not enumerate them), or commits unique to its
            branch; False when it is contentless plumbing.
        """
        index = _git("rev-parse", "--git-path", "index", cwd=wt_dir)
        if index.returncode == 0 and not (wt_dir / index.stdout.strip()).is_file():
            return GitWorktreeOps._interrupted_checkout_has_content(repo, branch, wt_dir)
        # ``--untracked-files`` is explicit: a repo-level
        # ``status.showUntrackedFiles=no`` would otherwise silence the
        # untracked AND ignored listing and make a written-to spare
        # look empty.
        status = _git(
            "status", "--porcelain", "--ignored", "--untracked-files=normal", cwd=wt_dir,
        )
        if status.returncode != 0:
            logger.warning(
                "git status failed in spare %s (rc=%s): %s; treating as content",
                wt_dir, status.returncode, status.stderr.strip(),
            )
            return True
        return bool(status.stdout.strip()) or not GitWorktreeOps._branch_is_expendable(
            repo, branch
        )

    @staticmethod
    def _interrupted_checkout_has_content(repo: Path, branch: str, wt_dir: Path) -> bool:
        """:meth:`spare_has_content` for a spare whose index was never written.

        Without an index ``git status`` cannot tell the half-written
        checkout from external writes (every ``HEAD`` file is a staged
        deletion, every written file untracked).  So the status walk
        runs against a scratch index built from ``HEAD`` with
        ``git read-tree`` (``GIT_INDEX_FILE`` in a temporary directory,
        split-index disabled so no shared index lands in the spare's
        gitdir; the spare's own index stays unwritten): git then
        reports a file the killed checkout never wrote as an unstaged
        deletion ``" D"`` — the only debris an interrupted checkout can
        leave — and anything else as what it is: ``" M"`` for a tracked
        file whose contents differ from ``HEAD``, ``??``/``!!`` for
        paths outside the tree.  Git does the content and filename
        comparison, so edited files and byte-distinct names are never
        mistaken for debris.  ``--untracked-files=all`` lists every
        file individually; the default would hide files written under a
        directory that replaced a tracked file (reported only as
        ``" D"`` for that file).  Blind spots of ``git status`` itself
        (``core.ignorecase``, ``core.filemode=false``, files inside an
        uninitialised submodule) are shared with the regular probe.

        Args:
            repo: Git repo root path.
            branch: The spare's ``kiss/wt-*`` branch name.
            wt_dir: The spare's worktree directory.

        Returns:
            True when the spare holds external files, edits or commits
            (or git could not enumerate them); False when every
            reported path is an unwritten ``HEAD`` file.
        """
        scratch = tempfile.mkdtemp(prefix="kiss-spare-index-")
        try:
            env = {"GIT_INDEX_FILE": os.path.join(scratch, "index")}
            no_split = ("-c", "core.splitIndex=false")
            read_tree = _git(*no_split, "read-tree", "HEAD", cwd=wt_dir, env=env)
            status = _git(
                *no_split, "status", "--porcelain", "-z", "--ignored",
                "--untracked-files=all", cwd=wt_dir, env=env,
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        if read_tree.returncode != 0 or status.returncode != 0:
            logger.warning(
                "git failed in index-less spare %s (read-tree rc=%s, status rc=%s); "
                "treating as content",
                wt_dir, read_tree.returncode, status.returncode,
            )
            return True
        for entry in status.stdout.split("\0"):
            if entry and entry[:2] != " D":
                return True
        logger.info(
            "Spare worktree %s has no index (interrupted checkout) and only "
            "unwritten files; treating as contentless",
            wt_dir,
        )
        return not GitWorktreeOps._branch_is_expendable(repo, branch)

    @staticmethod
    def _remove_path(path: Path) -> None:
        """Remove *path* whatever it is (symlink, dir, file, or absent).

        Checks ``is_symlink()`` first because ``is_dir()``/``exists()``
        follow symlinks: a symlink to a directory must be unlinked, not
        rmtree'd, and a broken symlink reports ``exists() == False``.
        """
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(str(path))
        elif path.exists():
            path.unlink()

    @staticmethod
    def copy_dirty_state(repo: Path, wt_dir: Path) -> bool:
        """Copy uncommitted/staged/untracked files from main worktree.

        Reads ``git status --porcelain -z`` in *repo* and mirrors every
        dirty file into *wt_dir*.  Files that exist in the main
        worktree are copied; files that were deleted are removed from
        *wt_dir*.  The caller is expected to stage and commit the
        result as a baseline commit.

        Args:
            repo: Git repo root (main worktree).
            wt_dir: Target worktree directory.

        Returns:
            True if any dirty state was copied, False if the main
            worktree was clean.

        Raises:
            OSError: If ``git status`` itself fails (timeout, corrupt
                index, ...), or a dirty tracked submodule is in the way
                (its content cannot be snapshotted into the worktree's
                gitlink), so the caller falls back to direct execution
                instead of silently running without the user's dirty
                state.
        """
        status = _git("status", "--porcelain", "-z", "-uall", cwd=repo)
        if status.returncode != 0:
            raise OSError(
                f"git status failed in {repo} (rc={status.returncode}): "
                f"{status.stderr.strip()}"
            )
        if not status.stdout.strip():
            return False

        copied = False
        for code, old_name, fname in _porcelain_entries(status.stdout):
            # Only a rename (``R``) vacates ``old_name``: a copy (``C``)
            # leaves its source in place, and that source's own entry
            # may already have mirrored it.
            if old_name is not None and "R" in code and not _under_symlink(wt_dir, old_name):
                old_dst = wt_dir / old_name
                if old_dst.is_symlink() or old_dst.exists():
                    GitWorktreeOps._remove_path(old_dst)
                    copied = True

            if _under_symlink(repo, fname) or _under_symlink(wt_dir, fname):
                # The entry's ancestor is (or became) a symlink: the
                # ancestor's own entry installs the link in *wt_dir* and
                # takes the stale real directory with it.
                continue
            src = repo / fname
            dst = wt_dir / fname

            if src.is_symlink():
                GitWorktreeOps._remove_path(dst)
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(os.readlink(src), dst)
                copied = True
            elif src.is_file():
                GitWorktreeOps._remove_path(dst)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
                copied = True
            elif src.is_dir() and not code.startswith("?"):
                # A dirty TRACKED submodule shows up in porcelain output
                # as the submodule directory itself (worktree code M/m).
                # Mirroring its working tree (minus .git) into the new
                # worktree's uninitialized gitlink directory would flatten
                # it into files git never sees there: edits to them reach
                # neither status nor auto-commit, and the finalizer would
                # remove the worktree with their only copy.  Refuse the
                # snapshot instead; the caller falls back to running in
                # the user's checkout.
                index_entry = _git("ls-files", "--stage", "-z", "--", fname, cwd=repo)
                if index_entry.stdout.startswith("160000 "):
                    raise OSError(
                        f"cannot snapshot the dirty submodule {fname!r} of "
                        f"{repo} into a worktree"
                    )
                # Any other tracked directory entry is a tracked file the
                # user replaced by a directory (`` D foo`` next to
                # ``?? foo/x``): mirror it so the worktree's stale file
                # makes way for the untracked files under it.
                # Untracked directory entries (``?? dir/``) are embedded
                # foreign repos/worktrees — git refuses to recurse into
                # them and they must not be mirrored (copying e.g. an
                # agent worktree under the repo into itself recurses
                # until "File name too long").
                if dst.is_symlink() or dst.is_file():
                    GitWorktreeOps._remove_path(dst)
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    str(src), str(dst), dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns(".git"),
                )
                copied = True
            elif dst.is_symlink() or (dst.exists() and not (dst.is_dir() and src.exists())):
                # Deleted in the main worktree, or an untracked directory
                # entry landing on a non-directory: drop the stale copy.
                GitWorktreeOps._remove_path(dst)
                copied = True

        return copied

    @staticmethod
    def head_sha(wt_dir: Path) -> str | None:
        """Return the SHA of HEAD in the given directory.

        Args:
            wt_dir: Git working directory (repo root or worktree).

        Returns:
            The full SHA string, or ``None`` on failure.
        """
        result = _git("rev-parse", "HEAD", cwd=wt_dir)
        sha = result.stdout.strip()
        return sha if result.returncode == 0 and sha else None

    @staticmethod
    def ignored_collisions(repo: Path, branch: str, baseline: str | None) -> list[str]:
        """Return the paths merging *branch* into HEAD would write over ignored files.

        The dirty-tree check and ``stash --include-untracked`` that
        precede a merge leave ignored files where they are, and both
        ``merge --squash`` and ``cherry-pick`` silently overwrite an
        ignored file at a path the branch starts tracking (or delete an
        ignored directory in the way of an incoming file, or an ignored
        file in the way of an incoming directory): a user's
        never-committed ``.env`` was replaced by the example one a task
        added, with no stash or git object to recover it from.  The
        main tree is clean when a merge runs, so whatever is on disk at
        a path HEAD does not track is ignored (an untracked file would
        have been stashed); git refuses to overwrite an untracked one
        anyway, so refusing up front loses nothing either way.  What is
        in the way is what git does not track at or under the path
        (``ls-files --others``): a tracked file the branch replaces with
        a directory, or a directory of tracked files it replaces with a
        file, is git's to merge.  A rename onto the path is the addition
        it is (``--no-renames``), not the move git would report.

        Args:
            repo: Git repo root path, checked out on the merge target.
            branch: The worktree branch to merge.
            baseline: The worktree's baseline commit SHA (only the
                branch's changes since it are applied), or ``None``
                for a legacy worktree (its changes since the merge
                base).

        Returns:
            The repository-relative paths the merge would create that
            something on disk is in the way of; empty when none.
        """
        incoming = GitWorktreeOps._diff_name_only(
            repo, "--no-renames", "--diff-filter=A", "HEAD", branch)
        applied = set(GitWorktreeOps._diff_name_only(
            repo, "--no-renames", f"{baseline or 'HEAD'}...{branch}"))
        # What is on disk where each incoming path, or a leading component
        # of it that is not a directory, goes.
        in_the_way: dict[str, str] = {}
        for path in incoming:
            if path not in applied:
                continue
            parts = path.split("/")
            for depth in range(1, len(parts) + 1):
                p = repo.joinpath(*parts[:depth])
                if os.path.lexists(p) and (
                    depth == len(parts) or os.path.islink(p) or not p.is_dir()
                ):
                    in_the_way[path] = "/".join(parts[:depth])
                    break
        if not in_the_way:
            return []
        # What git does not track under the top-level directory of each:
        # a directory it does not look into (a wholly untracked one, a
        # nested repository) is one entry ending in "/" that covers
        # whatever of ours lies below it.
        result = _git(
            "--literal-pathspecs", "ls-files", "--others", "--directory",
            "--no-empty-directory", "-z", "--",
            *sorted({blocker.split("/", 1)[0] for blocker in in_the_way.values()}), cwd=repo,
        )
        if result.returncode != 0:
            return list(in_the_way)
        others = [f for f in result.stdout.split("\0") if f]
        return [
            path for path, blocker in in_the_way.items()
            if any(
                f == blocker or f.startswith(blocker + "/")
                or (f.endswith("/") and blocker.startswith(f))
                for f in others
            )
        ]

    @staticmethod
    def _refused_for_ignored_files(repo: Path, branch: str, baseline: str | None) -> bool:
        """Log and tell whether merging *branch* must not run (:meth:`ignored_collisions`)."""
        collisions = GitWorktreeOps.ignored_collisions(repo, branch, baseline)
        if collisions:
            logger.warning(
                "merge of %s refused: it would overwrite files git ignores in %s: %s "
                "(move them away, then retry)", branch, repo, ", ".join(collisions),
            )
        return bool(collisions)

    @staticmethod
    def _theirs_option(repo: Path, baseline: str) -> list[str]:
        """Return ``["-X", "theirs"]`` when it is safe for a merge from *baseline*.

        Safe exactly when ``HEAD == baseline^`` (see
        :meth:`squash_merge_from_baseline` for why); empty otherwise,
        including on any git error — a conservative default that
        disables ``-X theirs`` rather than risk silently losing user
        commits.

        Args:
            repo: Git repo root path.
            baseline: The worktree's baseline commit SHA.
        """
        parent = _git("rev-parse", f"{baseline}^", cwd=repo)
        if parent.returncode == 0 and GitWorktreeOps.head_sha(repo) == parent.stdout.strip():
            return ["-X", "theirs"]
        return []

    @staticmethod
    def squash_merge(
        repo: Path,
        branch: str,
        baseline: str | None,
        user_prompt: str | None = None,
        task_result: str | None = None,
    ) -> MergeResult:
        """Squash-merge *branch* into HEAD, from *baseline* when one is known.

        Dispatches to :meth:`squash_merge_from_baseline` or, for a legacy
        worktree without a baseline, :meth:`squash_merge_branch`.

        Args:
            repo: Git repo root path.
            branch: The worktree branch to merge from.
            baseline: The worktree's baseline commit SHA, or ``None``.
            user_prompt: The user's task prompt for the merge commit
                message, or ``None``.
            task_result: The task's result summary for the merge
                commit message, or ``None``.

        Returns:
            The merge outcome.
        """
        if baseline:
            return GitWorktreeOps.squash_merge_from_baseline(
                repo, branch, baseline, user_prompt=user_prompt, task_result=task_result,
            )
        return GitWorktreeOps.squash_merge_branch(
            repo, branch, user_prompt=user_prompt, task_result=task_result,
        )

    @staticmethod
    def squash_merge_from_baseline(
        repo: Path,
        branch: str,
        baseline: str,
        user_prompt: str | None = None,
        task_result: str | None = None,
    ) -> MergeResult:
        """Squash-merge only the agent's changes (after baseline) into HEAD.

        Applies the agent's net change (``baseline..branch``) as ONE
        ``git cherry-pick --no-commit`` of a synthetic commit holding
        the branch's tree with *baseline* as parent (see
        :meth:`_squashed_commit`): a three-way merge with *baseline* as
        the base, so it handles cases where the user's dirty state
        (captured in the baseline) diverges from the committed HEAD
        content.  Replaying the branch's commits one by one instead
        conflicted on an intermediate edit the agent later reverted
        (when main had changed the same lines meanwhile) and refused a
        merge commit on the branch outright (``no -m option``), so a
        clean net change never merged and an orphan with such a
        history was retried on every reclaim pass.

        **The ``-X theirs`` strategy option is added precisely when —
        and only when — ``HEAD == baseline^``** (i.e. main has not
        advanced since worktree creation).  This is the common case
        and is exactly when the cherry-pick would otherwise fabricate
        a spurious modify/delete or modify/modify conflict:

        - ``_do_merge`` stashes the user's dirty edits on main before
          this call, so main HEAD == ``baseline^`` (clean).  But the
          baseline commit captured the user's dirty state — so the
          3-way merge sees:

              base   = baseline                  (user dirty edits)
              ours   = main HEAD == baseline^    (no dirty edits)
              theirs = branch tip                (dirty edits + agent diff)

        - ``base → ours`` is "revert the dirty edits"; when the agent
          deleted or modified the same hunk on the branch, plain
          cherry-pick raises a spurious conflict — even though the
          agent's actual net diff (``baseline..branch``) and main
          HEAD's actual content (``baseline^``) are perfectly
          compatible.
        - ``-X theirs`` is a hunk-level tie-breaker that resolves these
          spurious conflicts in favour of the branch tip (= the agent's
          intent), and is a NO-OP for hunks that already auto-merge.
        - Hunks cannot settle a file the task changed as a whole when
          ``HEAD`` lacks its baseline copy (it was untracked) or holds
          another one (it was dirty): such a path gets the branch
          tip's entry (see :meth:`_settle_baseline_conflicts`).

        When ``HEAD != baseline^`` (the user committed independent
        work on main between WT creation and merge), neither is done:
        any conflict in that case is a real cross-branch divergence
        between the user's main commits and the agent's branch tip,
        and silently letting the branch win would destroy the user's
        intentional commits.

        Args:
            repo: Git repo root path.
            branch: The worktree branch to merge from.
            baseline: SHA of the baseline commit to diff against.
            user_prompt: The user's task prompt for the merge commit
                message (see :meth:`_merge_commit_message`), or
                ``None``.
            task_result: The task's result summary for the merge
                commit message, or ``None``.

        Returns:
            :attr:`MergeResult.SUCCESS`, :attr:`MergeResult.CONFLICT`, or
            :attr:`MergeResult.MERGE_FAILED` when the commit failed.
        """
        log_result = _git(
            "rev-list",
            "--count",
            f"{baseline}..{branch}",
            cwd=repo,
        )
        if log_result.returncode != 0:
            logger.warning(
                "rev-list failed for baseline %s..%s: %s",
                baseline,
                branch,
                log_result.stderr.strip(),
            )
            return MergeResult.CONFLICT
        count = log_result.stdout.strip()
        if count == "0":
            return MergeResult.SUCCESS
        return GitWorktreeOps._apply_and_commit(repo, branch, baseline, user_prompt, task_result)

    @staticmethod
    def merge_has_content_conflict(
        repo: Path, branch: str, baseline: str | None,
    ) -> bool:
        """Whether squash-merging *branch* into HEAD conflicts on content.

        :meth:`squash_merge_branch` and :meth:`squash_merge_from_baseline`
        report every failed git command as :attr:`MergeResult.CONFLICT`,
        including operational failures such as a held ``index.lock``.
        This re-checks with ``git merge-tree --write-tree``, which
        merges in memory without touching the index or the working
        tree, using the same merge base (*baseline* when given, as the
        cherry-pick of ``baseline..branch`` does), the same
        ``-X theirs`` choice as :meth:`squash_merge_from_baseline` and
        its rule for what that leaves (:meth:`_baseline_induced`).

        Args:
            repo: Git repo root path (HEAD is the merge target).
            branch: Branch that would be merged.
            baseline: Baseline commit of the branch, or ``None`` to use
                the regular merge base.

        Returns:
            True only when git exits 1 (merge completed with
            conflicts) over a path the merge would not settle; False
            when the merge is clean or git failed.
        """
        args = ["merge-tree", "--write-tree", "--name-only", "-z"]
        theirs: list[str] = []
        if baseline:
            theirs = GitWorktreeOps._theirs_option(repo, baseline)
            args += [f"--merge-base={baseline}", *theirs]
        result = _git(*args, "HEAD", branch, cwd=repo)
        if result.returncode != 1:
            return False
        # ``<tree>\0<conflicted path>\0...\0\0<messages>``
        conflicted = result.stdout.split("\0\0", 1)[0].split("\0")[1:]
        return not (
            baseline and theirs and conflicted
            and GitWorktreeOps._baseline_induced(repo, branch, baseline, conflicted)
        )

    @staticmethod
    def _baseline_induced(repo: Path, branch: str, baseline: str, conflicted: list[str]) -> bool:
        """Whether *branch* changed every path of *conflicted* since *baseline*.

        While ``HEAD == baseline^`` (see :meth:`_theirs_option`),
        ``HEAD`` differs from the merge base by the user's dirty state
        alone and no path holds a competing commit: a path the task
        changed conflicts only over that state, and the branch tip's
        entry settles it.  A path the task did not change is git's own
        (``name~side``, where a file and a directory clash) and
        nothing settles it.
        """
        changed = GitWorktreeOps._diff_name_only(repo, "--no-renames", baseline, branch)
        return set(conflicted) <= set(changed)

    @staticmethod
    def _settle_baseline_conflicts(
        repo: Path, branch: str, baseline: str, conflicted: list[str],
    ) -> list[str]:
        """Give the paths a pick left unmerged while ``HEAD == baseline^`` *branch*'s entries.

        ``-X theirs`` settles hunks, not a path one side lacks: a file
        that was untracked when the task began and that the task then
        edited or renamed is modify/delete against ``HEAD``, a dirty
        file the task deleted likewise.  Each such path gets the
        branch tip's entry, mode and deletion included, in the index;
        nothing is touched unless all of them are
        :meth:`_baseline_induced`.  The working files follow by git's
        own check of each (:meth:`_move_written_files`, from the tree
        the pick wrote, ``AUTO_MERGE``): one saved since the pick
        wrote it keeps what it holds, settled in the index alone.
        Once they followed, ``AUTO_MERGE`` names the settled tree: the
        one :meth:`_undo_merge` takes back if the commit fails.

        Args:
            repo: Git repo root path, with the pick applied.
            branch: The task branch being merged.
            baseline: Its baseline commit SHA.
            conflicted: The paths the pick left unmerged.

        Returns:
            The paths still unmerged; empty when all are settled.
        """
        if not GitWorktreeOps._baseline_induced(repo, branch, baseline, conflicted):
            return conflicted
        if _git(
            "--literal-pathspecs", "reset", "-q", branch,
            "--pathspec-from-file=-", "--pathspec-file-nul",
            cwd=repo, input="\0".join(conflicted),
        ).returncode != 0:
            return GitWorktreeOps.unresolved_paths(repo) or conflicted
        # To the tree the index holds now; without one (git failed) nothing moves.
        settled = _git("write-tree", cwd=repo).stdout.strip()
        if GitWorktreeOps._move_written_files(repo, "AUTO_MERGE", settled):
            _git("update-ref", "AUTO_MERGE", settled, cwd=repo)
        return GitWorktreeOps.unresolved_paths(repo)

    @staticmethod
    def _apply_branch(repo: Path, branch: str, baseline: str | None) -> list[str] | None:
        """Apply *branch* onto HEAD without committing.

        With a *baseline* the agent's net change (``baseline..branch``)
        is ONE ``cherry-pick --no-commit`` of the
        :meth:`_squashed_commit`, with ``-X theirs`` and
        :meth:`_settle_baseline_conflicts` when :meth:`_theirs_option`
        allows them; without one, ``git merge --squash``.  Either
        runs without ``rerere``: a recorded resolution it replays
        into a conflicted file is not what the merge wrote there (see
        :meth:`_undo_merge`), and staged by
        ``rerere.autoupdate`` it leaves git failed with a merge
        applied and nothing unmerged.  And either merges with
        ``ort``, whatever ``pull.twohead`` names: the strategy the
        pick and :meth:`merge_has_content_conflict` use anyway, and
        the one that records what it wrote (``AUTO_MERGE``).

        Args:
            repo: Git repo root path, checked out on the target branch.
            branch: The task branch to merge in.
            baseline: The worktree's baseline commit SHA, or ``None`` for
                a legacy worktree.

        Returns:
            The repository-relative paths left unmerged; an empty list
            when the change applied cleanly and is staged; ``None``
            when nothing was applied: git failed for a reason other
            than conflicts, or was not run because a git operation of
            the user's is under way (:meth:`operation_in_progress`).
        """
        if GitWorktreeOps.operation_in_progress(repo):
            logger.warning("a git operation is under way in %s; not merging %s", repo, branch)
            return None
        theirs: list[str] = []
        if baseline:
            squashed = GitWorktreeOps._squashed_commit(repo, branch, baseline)
            if squashed is None:
                return None
            theirs = GitWorktreeOps._theirs_option(repo, baseline)
            args = ["cherry-pick", "--no-commit", *theirs, squashed]
        else:
            args = ["merge", "--squash", branch]
        result = _git("-c", "rerere.enabled=false", "-c", "pull.twohead=ort", *args, cwd=repo)
        if result.returncode == 0:
            return []
        conflicted = GitWorktreeOps.unresolved_paths(repo)
        if baseline and theirs and conflicted:
            conflicted = GitWorktreeOps._settle_baseline_conflicts(
                repo, branch, baseline, conflicted,
            )
            if not conflicted:
                return []
        logger.warning(
            "applying %s onto HEAD of %s failed: %s", branch, repo, result.stderr.strip(),
        )
        return conflicted or None

    @staticmethod
    def _squashed_commit(repo: Path, branch: str, baseline: str) -> str | None:
        """Return a synthetic commit holding *branch*'s tree with *baseline* as parent.

        Cherry-picking it applies the agent's net change
        (``baseline..branch``) as one three-way merge, whatever the
        branch's history (see :meth:`squash_merge_from_baseline`).
        The commit is referenced by nothing and is garbage-collected
        in due course.

        Args:
            repo: Git repo root path.
            branch: The worktree branch.
            baseline: The worktree's baseline commit SHA.

        Returns:
            The commit's SHA, or ``None`` when ``commit-tree`` failed.
        """
        squashed = _git(
            "commit-tree", f"{branch}^{{tree}}", "-p", baseline,
            "-m", f"kiss: squashed {branch}", cwd=repo,
        )
        if squashed.returncode != 0:
            logger.warning("commit-tree for %s failed: %s", branch, squashed.stderr.strip())
            return None
        return squashed.stdout.strip()

    @staticmethod
    def begin_conflicted_merge(
        repo: Path, branch: str, baseline: str | None,
    ) -> list[str] | None:
        """Apply *branch* onto HEAD, leaving conflicts in the tree to resolve.

        The conflict-resolution counterpart of
        :meth:`squash_merge_from_baseline` / :meth:`squash_merge_branch`,
        run after one of those returned :attr:`MergeResult.CONFLICT` and
        took its merge back.  Nothing is committed: the change is
        applied as on the clean path (see :meth:`_apply_branch`), with
        a *baseline* as ONE three-way merge, so the resolver sees a
        single set of conflicts instead of a stopped multi-commit
        sequencer.  What the merge wrote is noted for
        :meth:`abort_conflicted_merge` and the marker check of
        :meth:`finish_conflicted_merge`; a merge that is interrupted
        as it is applied is taken back here (:meth:`_undo_merge`).

        Args:
            repo: Git repo root path, checked out on the target branch.
            branch: The task branch to merge in.
            baseline: The worktree's baseline commit SHA, or ``None`` for
                a legacy worktree.

        Returns:
            The repository-relative paths left unmerged (the resolver's
            work list); an empty list when the merge applied cleanly
            and is staged; ``None`` when nothing was applied: git
            failed for a reason other than conflicts, the merge would
            overwrite ignored files (see :meth:`ignored_collisions`),
            the index holds unmerged entries (the user's own, or a
            merge :meth:`_undo_merge` left in place), or a git
            operation of the user's is under way.
        """
        if GitWorktreeOps._refused_for_ignored_files(repo, branch, baseline):
            return None
        index_before = GitWorktreeOps._index_tree(repo)
        if index_before is None:
            return None
        try:
            conflicted = GitWorktreeOps._apply_branch(repo, branch, baseline)
            if conflicted is not None:
                _conflicted_merges[repo] = GitWorktreeOps._written_tree(repo, branch), conflicted
            return conflicted
        except BaseException:
            # No resolver has worked on the merge: a file that is no longer
            # what it wrote is the user's, and ``_undo_merge`` keeps it.
            _conflicted_merges.pop(repo, None)
            GitWorktreeOps._undo_if_applied(repo, branch, index_before)
            raise

    @staticmethod
    def unresolved_paths(repo: Path) -> list[str]:
        """Return the repository-relative paths that are still unmerged."""
        return GitWorktreeOps._diff_name_only(repo, "--diff-filter=U")

    @staticmethod
    def paths_with_conflict_markers(repo: Path, paths: list[str]) -> list[str]:
        """Return those of *paths* whose staged content still has conflict markers.

        Only the ``<<<<<<<`` and ``>>>>>>>`` marker lines are checked,
        of the length git wrote them with (:meth:`_marker_sizes`) or
        longer, as those of a nested conflict are.  A bare ``=======``
        line is also a Markdown setext underline, so matching it would
        flag legitimately resolved files.  The paths are literal
        pathspecs, so a name containing pathspec magic cannot break
        the query, and a git failure (:meth:`_grep`) reports every
        path as unresolved: this check guards a branch deletion, so it
        fails closed.

        Args:
            repo: Git repo root path.
            paths: Repository-relative paths to inspect (the files that
                were conflicted).

        Returns:
            The subset of *paths* with markers left; empty when *paths*
            is empty.
        """
        if not paths:
            return []
        found: list[str] = []
        try:
            sizes = GitWorktreeOps._marker_sizes(repo, paths)
            by_size: dict[int, list[str]] = {}
            for path in paths:
                by_size.setdefault(sizes.get(path, 7), []).append(path)
            for size, sized in by_size.items():
                marked = GitWorktreeOps._grep(
                    repo, "-l", "--cached", "-e", f"^(<{{{size},}}|>{{{size},}})( |$)",
                    "--", *(f":(literal){p}" for p in sized),
                )
                found += [f for f in marked.split("\0") if f]
        except _GitStepError:
            logger.warning(
                "the check for conflict markers failed; treating %s as unresolved", paths,
            )
            return list(paths)
        return found

    @staticmethod
    def _marker_sizes(repo: Path, paths: list[str]) -> dict[str, int]:
        """Return the length of the conflict markers git wrote into each of *paths*.

        Read off the tree the conflicted merge of *repo* wrote
        (:meth:`begin_conflicted_merge` took note of it): the shortest
        length at which a ``<<<`` line, a ``===`` line and a ``>>>``
        line follow one another there, whatever the two sides are
        named (git says ``HEAD``, a merge driver of the user's names
        its temporary files).  The path's ``conflict-marker-size``
        attribute decided that length as git merged; nothing read
        afterwards tells what it said then.  A path without such
        lines, or with no merge noted, is left out: git's default of
        seven is checked there.

        Raises:
            _GitStepError: git failed; its error is logged.
        """
        merge = _conflicted_merges.get(repo)
        if merge is None:
            return {}
        written = merge[0]
        lines = GitWorktreeOps._grep(
            repo, "-a", "-o", "-e", "^(<+|>+)( |$)", "-e", "^=+[[:space:]]*$", written,
            "--", *(f":(literal){p}" for p in paths),
        )
        runs: dict[str, list[str]] = {}
        # ``<tree>:<path>\0<the match>\n`` a line, in the file's order.
        for name, run in re.findall(r"(?s)(.*?)\0(<+|=+|>+)[^\n]*\n", lines):
            runs.setdefault(name.removeprefix(f"{written}:"), []).append(run)
        sizes: dict[str, int] = {}
        for path, found in runs.items():
            for size in sorted({len(run) for run in found if run[0] == "<"}):
                rest = iter(found)  # ``in`` reads on from the run found last
                if all(mark * size in rest for mark in "<=>"):
                    sizes[path] = size
                    break
        return sizes

    @staticmethod
    def _grep(repo: Path, *args: str) -> str:
        """Run ``git grep -z -E`` with *args* and return what it printed, bare of decoration.

        Raises:
            _GitStepError: git failed (any exit code other than the
                0 = matches / 1 = no matches pair); its error is logged.
        """
        result = _git(
            "grep", "--no-color", "--no-line-number", "--no-column", "-z", "-E", *args,
            cwd=repo,
        )
        if result.returncode not in (0, 1):
            logger.warning("git grep failed in %s: %s", repo, result.stderr.strip())
            raise _GitStepError
        return result.stdout

    @staticmethod
    def finish_conflicted_merge(
        repo: Path,
        branch: str,
        conflicted: list[str],
        head_before: str,
        user_prompt: str | None = None,
        task_result: str | None = None,
    ) -> MergeResult:
        """Verify a resolved conflicted merge and commit it.

        Accepts the resolution only when HEAD is still *head_before*,
        every path is merged, none of the originally *conflicted* files
        still carries a marker, and the index is non-empty.  Anything
        else is refused: an empty index means the resolver discarded the
        merge, and a moved HEAD means it committed on its own — the
        commit may hold anything (a ``git reset --hard`` followed by an
        unrelated commit reproduces the case), and there is no way to
        prove it holds the merge.  Reporting SUCCESS in either case would
        let the caller delete the only branch that holds the work, so
        this check fails closed and the caller restores *head_before*.

        Args:
            repo: Git repo root path.
            branch: The task branch being merged (for the commit message).
            conflicted: The paths :meth:`begin_conflicted_merge` reported.
            head_before: ``HEAD`` SHA before the conflicted merge began.
            user_prompt: The task prompt for the commit message, or ``None``.
            task_result: The task result summary for the message, or ``None``.

        Returns:
            :attr:`MergeResult.SUCCESS` when committed,
            :attr:`MergeResult.MERGE_FAILED` when the commit itself
            failed, and :attr:`MergeResult.CONFLICT` when the
            resolution is refused.  Unless committed, the merge is
            still applied: the caller takes it back
            (:meth:`abort_conflicted_merge`).
        """
        if GitWorktreeOps.head_sha(repo) != head_before:
            logger.warning(
                "conflict resolver moved HEAD of %s away from %s; refusing "
                "the merge", repo, head_before,
            )
            return MergeResult.CONFLICT
        if GitWorktreeOps.unresolved_paths(repo):
            return MergeResult.CONFLICT
        if GitWorktreeOps.paths_with_conflict_markers(repo, conflicted):
            return MergeResult.CONFLICT
        if not GitWorktreeOps.has_staged_changes(repo):
            return MergeResult.CONFLICT
        # ``--no-commit`` leaves no sequencer state and the user had
        # none when the merge began (:meth:`operation_in_progress`):
        # one found now is the resolver's (a ``git cherry-pick
        # --continue`` half-way), and ``--quit`` keeps the index.
        _git("cherry-pick", "--quit", cwd=repo)
        if GitWorktreeOps._commit_merge(
            repo, branch, user_prompt=user_prompt, task_result=task_result,
        ):
            _conflicted_merges.pop(repo, None)
            return MergeResult.SUCCESS
        return MergeResult.MERGE_FAILED

    @staticmethod
    def abort_conflicted_merge(repo: Path, before: str, head_before: str) -> bool:
        """Take back a conflicted merge that was not completed, and nothing but the merge.

        Drops any sequencer state (the resolver's: the user had none
        when the merge began, see :meth:`operation_in_progress`) and
        puts the branch back on *head_before* (undoing any commit the
        resolver made on its own — see :meth:`finish_conflicted_merge`).
        The paths the merge wrote (:meth:`begin_conflicted_merge` took
        note of them) then go back to *head_before* whatever they hold
        by now, since the resolver worked there, and the index with
        them.  Every other file stays as it is, tracked or not: the
        user's editor takes none of the merge's locks, and a file it
        saved during the resolution cannot be told from one the
        resolver left behind.  What stays is logged.

        Args:
            repo: Git repo root path.
            before: ``status_porcelain`` output captured before
                :meth:`begin_conflicted_merge`.
            head_before: ``HEAD`` SHA captured before it; ``""`` (HEAD
                unreadable) stands for the current HEAD.

        Returns:
            Whether the merge is taken back (also when none is in
            place: :meth:`begin_conflicted_merge` applied nothing, or
            took its own merge back).  When not, the merge stays in
            the files and the index for the user to settle: git
            refused to write (a file changed as it looked, or one
            appeared where the merge had deleted one).
        """
        _git("cherry-pick", "--quit", cwd=repo)
        merge = _conflicted_merges.pop(repo, None)
        taken_back = True
        if merge is not None:
            written, conflicted = merge
            head = head_before or "HEAD"
            _git("reset", "--soft", head, cwd=repo)
            # A path the merge deleted is none of these: a file there is not the merge's.
            paths = GitWorktreeOps._diff_name_only(
                repo, "--no-renames", "--diff-filter=d", head, written, "--",
            )
            now = GitWorktreeOps._tree_with_files(repo, written, paths + conflicted)
            taken_back = now is not None and GitWorktreeOps._take_back(repo, now, head)
        kept = GitWorktreeOps.status_porcelain(repo)
        if kept != before:
            logger.warning(
                "the tree of %s differs from the one before the conflicted merge; "
                "nothing more is removed, it may be the user's:\n%s", repo, kept,
            )
        return taken_back

    @staticmethod
    def cleanup_partial(repo: Path, branch: str, wt_dir: Path) -> None:
        """Remove a partially-created worktree and branch (best-effort).

        No explicit ``prune`` is issued: :meth:`remove` already prunes
        on every path that can leave a stale registration behind, and
        a successful ``git worktree remove`` unregisters the worktree
        itself.

        Args:
            repo: Git repo root path.
            branch: The branch name to delete.
            wt_dir: The worktree directory to remove.
        """
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.delete_branch(repo, branch)

    @staticmethod
    def _config_branch_sections(repo: Path) -> set[str]:
        """Return branch names that own a ``branch.<name>.*`` config section.

        Only agent-minted ``kiss/wt-*`` names are reported; the user's
        own branch sections are never touched.

        Args:
            repo: Git repo root path.

        Returns:
            Set of ``kiss/wt-*`` branch names present in git config.
        """
        prefix = "branch."
        result = _git(
            "config", "-z", "--get-regexp", rf"^{prefix}kiss/wt-", cwd=repo,
        )
        if result.returncode != 0:
            # git config exits 1 when the regexp matches nothing.
            return set()
        # ``-z`` emits one NUL-terminated "<key>\n<value>" record per
        # match.  Splitting the raw output on newlines instead would
        # turn a multi-line config value (git stores those verbatim)
        # into bogus extra "branch names".
        names: set[str] = set()
        for record in result.stdout.split("\0"):
            if not record:
                continue
            key = record.split("\n", 1)[0]
            names.add(key[len(prefix):].rpartition(".")[0])
        return names

    @staticmethod
    def _branch_is_expendable(repo: Path, branch: str) -> bool:
        """Whether *branch* holds no work that would be lost by deleting it.

        A worktree branch is expendable once every one of its commits
        is reachable from another ref — i.e. it was merged, or it never
        diverged in the first place.  ``git branch -d`` applies exactly
        this rule but only against HEAD, so ``--merged HEAD`` would
        keep branches that were squash-merged into a different base.
        Checking reachability from *all* other refs is both stricter
        and cheaper than replaying the merge.

        ``--single-worktree`` matters: without it ``--all`` also lists
        the ``HEAD`` of every linked worktree, and a branch that is
        checked out in one (every pool spare, every live task
        worktree) is trivially "reachable" from its own worktree's
        HEAD — so a spare carrying an external commit was reported
        expendable and ``take_spare`` / ``discard_all`` / the reclaim
        pass destroyed that commit (``reset --hard``, ``branch -D``).
        Only the current working tree's HEAD is a genuinely *other*
        ref; the callers' in-use set (built from
        :meth:`registered_worktrees`) and ``_same_path`` guards already
        keep that one out of the question.

        Args:
            repo: Git repo root path.
            branch: Branch name to test.

        Returns:
            True if the branch has no commits unique to it.
        """
        result = _git(
            "rev-list",
            "--count",
            "--max-count=1",
            branch,
            "--not",
            "--single-worktree",
            "--exclude=refs/heads/" + branch,
            "--all",
            cwd=repo,
        )
        if result.returncode != 0:  # pragma: no cover — git failure
            return False
        return result.stdout.strip() == "0"

    @staticmethod
    def sweep_orphaned_state(repo: Path, *, exclude_branches: set[str] | None = None) -> int:
        """Delete leftover ``kiss/wt-*`` branches and config sections.

        A crashed, killed, or refused cleanup leaves three kinds of
        debris behind: a registered worktree whose directory is gone, a
        branch nobody has checked out, and the ``branch.kiss/wt-*.*``
        git config section that names it.  None of it is visible in
        ``git status``, so it accumulates silently — the user only
        notices as a swelling ``git branch`` listing.

        Only debris is removed.  A branch is kept when it is checked
        out by a live worktree (the task may still be running), when
        it holds commits no other ref can reach (unmerged work the user
        could still want), or when a live agent owns it: it is in
        *exclude_branches*, or another process that is still alive
        stamped it (:meth:`_live_peer_owner`).  The last two keep the
        branch of a task that removed its worktree directory and has
        not merged yet (``_finalize_worktree`` runs before
        ``_do_merge``): no worktree lists it for that moment, and
        deleting it turned a task that changed nothing into a bogus
        merge conflict.  Config sections are only purged once their
        branch is gone.

        A fourth kind of debris is a *husk*: a ``kiss_wt-*`` directory
        under ``.kiss-worktrees/`` that git does not know about — a
        finished task's directory re-created by a stray ``mkdir`` or
        by a lingering daemon, or the undeletable remnant of a failed
        removal.  Those are deleted too (:meth:`_remove_husk_dirs`).

        Args:
            repo: Git repo root path.
            exclude_branches: Branches owned by a still-live agent in
                this process, kept whatever their state.

        Returns:
            The number of branches deleted.
        """
        return locked_call(
            repo_lock(repo), GitWorktreeOps._sweep_orphaned_state_locked,
            repo, exclude_branches or set(),
        )

    @staticmethod
    def _sweep_orphaned_state_locked(repo: Path, excluded: set[str]) -> int:
        """:meth:`sweep_orphaned_state`'s body, run under ``repo_lock``."""
        with _reclaim_process_lock(repo):
            GitWorktreeOps.prune(repo)
            registered = GitWorktreeOps.registered_worktrees(repo)
            in_use = {branch for _wt_dir, branch in registered}
            result = _git(
                "for-each-ref",
                "--format=%(refname:short)",
                f"refs/heads/{_WORKTREE_BRANCH_PREFIX}*",
                cwd=repo,
            )
            if result.returncode != 0:
                # A failed enumeration (error, timeout) says nothing about
                # which branches are gone: leave every marker and config
                # section for a pass that can see the refs.
                return 0
            existing = set(result.stdout.split())
            deleted = 0
            for branch in sorted(existing - excluded):
                if branch in in_use:
                    continue
                if not GitWorktreeOps._branch_is_expendable(repo, branch):
                    continue
                if GitWorktreeOps._live_peer_owner(repo, branch) is not None:
                    continue
                # git refuses to delete a branch some worktree has
                # checked out; the in_use guard skips those first so
                # the sweep never logs a warning for a running task.
                # delete_branch removes the branch's config section and
                # marker itself, so a deleted branch stays in ``existing``.
                deleted += int(GitWorktreeOps.delete_branch(repo, branch))
            for name in GitWorktreeOps._config_branch_sections(repo) - existing:
                GitWorktreeOps._remove_branch_config_section(repo, name)
            GitWorktreeOps._sweep_preserve_markers(repo, existing)
            GitWorktreeOps._remove_husk_dirs(repo, registered)
            return deleted

    @staticmethod
    def _remove_husk_dirs(repo: Path, registered_worktrees: list[tuple[Path, str]]) -> None:
        """Delete unregistered ``kiss_wt-*`` directories under ``.kiss-worktrees/``.

        A directory is a husk when git lists no worktree at its path
        AND it carries no ``.git`` link file.  Both conditions are
        required: a registered worktree is live (a running task or a
        pool spare — ``--no-checkout`` spares are registered before
        they are populated), and a ``.git`` link marks a real checkout
        git may still re-register (a detached-HEAD worktree, which
        :meth:`registered_worktrees` skips).  Must be called under
        ``repo_lock`` and :func:`_reclaim_process_lock`, which
        :meth:`create` also holds while ``git worktree add`` writes
        the new directory and its ``.git`` file — so a worktree in the
        making is never mistaken for a husk.

        Args:
            repo: Git repo root path.
            registered_worktrees: :meth:`registered_worktrees` of *repo*,
                listed by the caller under the same locks.
        """
        root = repo / _WORKTREE_SUBDIR
        if not root.is_dir():
            return
        registered = {wt_dir.resolve() for wt_dir, _branch in registered_worktrees}
        for entry in root.iterdir():
            if not entry.name.startswith(_WORKTREE_SLUG_PREFIX):
                continue
            if entry.is_symlink() or not entry.is_dir():
                continue
            if entry.resolve() in registered or (entry / ".git").exists():
                continue
            logger.info("Removing unregistered worktree husk %s", entry)
            GitWorktreeOps._rmtree_reporting_leftovers(entry)

    @staticmethod
    def registered_worktrees(repo: Path) -> list[tuple[Path, str]]:
        """Return ``(wt_dir, branch)`` for every registered worktree.

        Parses ``git worktree list --porcelain -z``.  The main repo entry
        (whose branch name is any user branch, e.g. ``main``) is
        included alongside ``kiss/wt-*`` entries; callers filter by
        the branch-name prefix themselves.

        Detached-HEAD worktrees are skipped because they have no
        branch name to reclaim by.

        Worktree paths are emitted verbatim by git and every attribute
        is NUL-terminated (``-z``; ``_git_text`` keeps the bytes of a
        ``-z`` command as they are), so a path holding a newline or a
        carriage return survives: split on ``"\\n"``, the path of
        ``kiss_wt-a\\rb`` was read as ``kiss_wt-a``, and the reclaim
        handed that OTHER directory to :meth:`remove`, whose ``rmtree``
        fallback deleted it.  The path tail is never stripped either:
        ``strip()`` would silently rewrite a directory whose name ends
        in a space into a different existing path.

        Args:
            repo: Git repo root path.

        Returns:
            List of ``(wt_dir, branch)`` pairs in the order git
            reports them, empty on git failure.
        """
        result = _git("worktree", "list", "--porcelain", "-z", cwd=repo)
        if result.returncode != 0:  # pragma: no cover — git failure
            return []
        pairs: list[tuple[Path, str]] = []
        cur_dir: Path | None = None
        cur_branch: str | None = None
        for line in result.stdout.split("\0"):
            if line.startswith("worktree "):
                if cur_dir is not None and cur_branch is not None:
                    pairs.append((cur_dir, cur_branch))
                cur_dir = Path(line[len("worktree "):])
                cur_branch = None
            elif line.startswith("branch refs/heads/"):
                cur_branch = line[len("branch refs/heads/"):]
        if cur_dir is not None and cur_branch is not None:
            pairs.append((cur_dir, cur_branch))
        return pairs

    @staticmethod
    def reclaim_orphaned_worktrees(
        repo: Path,
        *,
        exclude_branches: set[str] | None = None,
    ) -> int:
        """Auto-commit and squash-merge orphan ``kiss/wt-*`` worktrees.

        A Sorcar process that is killed / crashes / OOMs / reboots
        while a worktree task is pending leaves its worktree
        registered on disk with dirty uncommitted work.  No in-memory
        ``self._wt`` state survives the restart, so no future
        ``_release_worktree`` call ever runs — the work stays
        stranded, invisible to :meth:`sweep_orphaned_state` (which
        reaps plumbing debris only, never real work).

        This reclaims each such worktree: it stages and commits any
        dirty state under a generic ``"kiss: reclaim orphan
        worktree"`` message, then squash-merges the branch into its
        saved ``original_branch`` (from :meth:`load_original_branch`)
        using the saved baseline commit (from
        :meth:`load_baseline_commit`) when present, then removes the
        worktree and deletes its branch.

        Safety rules — a worktree is left completely untouched when
        ANY of the following holds, so no work is ever destroyed:

        * The branch is in *exclude_branches* (caller's live-agent
          set — e.g. the current task's own worktree).
        * The saved ``kiss-owner-pid`` names a process other than this
          one that is still alive (a daemon's or CLI run's live task
          worktree or pooled spare); this is tested before every
          destructive branch below, including the spare discard.
        * The saved original branch no longer exists in the repo.  (A
          MISSING ``kiss-original`` config — a legacy worktree — is
          not a reason to skip: the main tree's current branch is used
          as the merge target, guarded by the rules below.)
        * The main tree's current branch differs from the saved
          original branch (reclaim never checks out for the user).
        * The main tree's current HEAD is detached.
        * ``git status --porcelain`` in the main tree reports dirty
          state (a reclaim now would silently include the user's
          uncommitted edits in the squash-merge commit).
        * The auto-commit inside the worktree is rejected (e.g. a
          pre-commit hook).
        * The squash-merge returns anything other than
          :attr:`MergeResult.SUCCESS` (conflict, cherry-pick failure).
          A content conflict (confirmed by
          :meth:`merge_has_content_conflict`) also writes the
          preserve-for-review marker, so later passes skip the
          worktree instead of re-running the same failing merge.

        Args:
            repo: Git repo root path.
            exclude_branches: Branches known to be owned by a still-
                live agent in this process; they must not be
                reclaimed.  ``None`` treats every registered
                ``kiss/wt-*`` worktree as reclaimable.

        Returns:
            Number of worktrees successfully reclaimed (merged and
            removed — or, for never-consumed pool spares, simply
            removed).
        """
        return locked_call(
            repo_lock(repo), GitWorktreeOps._reclaim_orphaned_worktrees_locked,
            repo, exclude_branches or set(),
        )

    @staticmethod
    def _reclaim_orphaned_worktrees_locked(repo: Path, excluded: set[str]) -> int:
        """:meth:`reclaim_orphaned_worktrees`'s body, run under ``repo_lock``."""
        reclaimed = 0
        with _reclaim_process_lock(repo):
            # ``.kiss-worktrees/`` under the main tree would otherwise
            # show up as an untracked directory in ``git status`` and
            # trip the "main tree is dirty" guard below.  ``ensure_
            # excluded`` writes it to ``.git/info/exclude`` (never a
            # tracked file) idempotently, so a reclaim can be called
            # standalone (without the ``ensure_excluded`` that
            # ``_try_setup_worktree`` normally runs just before it).
            GitWorktreeOps.prepare_repo(repo)
            GitWorktreeOps.prune(repo)
            current = GitWorktreeOps.current_branch(repo)
            if current is None:
                return 0
            if GitWorktreeOps.has_uncommitted_changes(repo):
                logger.info(
                    "Skipping orphan-worktree reclaim in %s: main "
                    "tree is dirty",
                    repo,
                )
                return 0
            for wt_dir, branch in GitWorktreeOps.registered_worktrees(repo):
                if not branch.startswith(_WORKTREE_BRANCH_PREFIX):
                    continue
                if _same_path(wt_dir, repo):
                    # ``git worktree list`` reports the main working
                    # tree too.  A user checkout that merely sits on a
                    # branch carrying the agent prefix (e.g. a machine
                    # provisioned by ./sorcar-cloud from a worktree) is
                    # not an orphan: reclaiming it would delete the
                    # project directory.
                    continue
                if branch in excluded:
                    continue
                if not wt_dir.exists():  # pragma: no cover — prune above
                    # ``prune`` at the top of this block already
                    # dropped registrations whose directory is gone,
                    # so this branch is only reachable on an rm-vs-
                    # iteration race with an external process.
                    continue
                owner_pid = GitWorktreeOps._live_peer_owner(repo, branch)
                if owner_pid is not None:
                    # Another Sorcar process (daemon vs. CLI) owns this
                    # worktree and is still alive: its task may be
                    # running right now — or its pool is holding this
                    # spare for the next task — and this process's
                    # exclusion set cannot see either.  Reclaiming
                    # would auto-commit, merge, and delete (or, for a
                    # spare, simply delete) a LIVE worktree under its
                    # owner.  Checked BEFORE the spare-marker branch,
                    # which destroys unconditionally.  Our own pid is
                    # exempt: within this process the exclusion set is
                    # authoritative for live tasks and spares, and
                    # ``load_preserve_marker`` below also honours the
                    # process-local claims of kept worktrees whose
                    # marker write failed.
                    logger.info(
                        "Skipping orphan-worktree reclaim of %s: "
                        "owning process %d is still alive",
                        wt_dir, owner_pid,
                    )
                    continue
                if GitWorktreeOps.load_spare_marker(repo, branch):
                    # A pool spare of an older release (never given
                    # to a task) is contentless plumbing:
                    # discard it, never merge.  Its branch snapshots
                    # whatever branch was checked out at refill time,
                    # so the no-config merge fallback below would
                    # squash that snapshot into the CURRENT branch
                    # even after the user switched branches.
                    if GitWorktreeOps.spare_has_content(repo, branch, wt_dir):
                        # A spare is never written to, so content here
                        # means something external happened; preserve
                        # rather than destroy.
                        logger.warning(
                            "Orphan spare worktree %s (branch '%s') "
                            "has unexpected content; preserving",
                            wt_dir, branch,
                        )
                        continue
                    GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
                    reclaimed += 1
                    continue
                if GitWorktreeOps.load_preserve_marker(repo, branch):
                    # The user (or the failed-task preserve path)
                    # deliberately parked this worktree for manual
                    # review; publishing its work here would violate
                    # the "never silently merge unverified work"
                    # contract.
                    logger.info(
                        "Skipping orphan-worktree reclaim of %s: "
                        "branch '%s' is marked preserve-for-review",
                        wt_dir, branch,
                    )
                    continue
                original_branch = GitWorktreeOps.load_original_branch(
                    repo, branch,
                )
                if not original_branch:
                    # Legacy worktree without the kiss-original
                    # config (created before that config landed, or
                    # a save that failed silently).  Fall back to
                    # the current branch of the main tree — that is
                    # what the user is on right now, so it is the
                    # branch a fresh task would merge into anyway.
                    # The dirty-main and current-branch guards below
                    # still protect against clobbering user state.
                    logger.warning(
                        "Orphan worktree %s (branch '%s') has no "
                        "kiss-original config; falling back to the "
                        "current branch '%s' of the main tree",
                        wt_dir, branch, current,
                    )
                    original_branch = current
                if not GitWorktreeOps.branch_exists(repo, original_branch):
                    logger.warning(
                        "Cannot reclaim orphan worktree %s: original "
                        "branch '%s' no longer exists; preserving",
                        wt_dir, original_branch,
                    )
                    continue
                if original_branch != current:
                    logger.info(
                        "Skipping orphan-worktree reclaim of %s: "
                        "main tree is on '%s' but the worktree "
                        "targets '%s'",
                        wt_dir, current, original_branch,
                    )
                    continue
                # Test hook (no-op in production): widens the window
                # between the guards above and the commit/merge/cleanup
                # mutations below so the cross-process reclaim race is
                # deterministic under KISS_RACE_DELAY.
                _race_delay()
                if GitWorktreeOps.has_uncommitted_changes(wt_dir):
                    # ``commit_all`` stages first; an extra
                    # ``stage_all`` here would run a second full
                    # ``git add -A`` index scan and widen the window
                    # in which a late-arriving file is staged.
                    GitWorktreeOps.commit_all(
                        wt_dir, "kiss: reclaim orphan worktree",
                    )
                    if GitWorktreeOps.has_uncommitted_changes(wt_dir):
                        logger.warning(
                            "Cannot reclaim orphan worktree %s: "
                            "auto-commit failed (a pre-commit hook "
                            "may have rejected it); preserving",
                            wt_dir,
                        )
                        continue
                baseline = GitWorktreeOps.load_baseline_commit(repo, branch)
                result = GitWorktreeOps.squash_merge(
                    repo, branch, baseline,
                    task_result="Auto-merged by orphan-worktree reclaim",
                )
                if result != MergeResult.SUCCESS:
                    logger.warning(
                        "Reclaim of orphan worktree %s: squash "
                        "merge into '%s' returned %s; preserving",
                        wt_dir, original_branch, result.value,
                    )
                    if (
                        result == MergeResult.CONFLICT
                        and GitWorktreeOps.merge_has_content_conflict(
                            repo, branch, baseline,
                        )
                    ):
                        # A content conflict needs a human; retrying
                        # it on every later pass (each task start and
                        # pool refill) only repeats a multi-second
                        # failing merge.  Park the worktree for manual
                        # review so the preserve check above skips it
                        # from now on.  CONFLICT also covers failed git
                        # commands (a held index.lock), so the content
                        # conflict is re-checked; those failures and
                        # MERGE_FAILED (the merge applied but its
                        # commit was rejected) may be transient and
                        # are retried.
                        GitWorktreeOps.save_preserve_marker(repo, branch)
                    continue
                GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)
                reclaimed += 1
        return reclaimed


MergeConflictResolver = Callable[[Any, GitWorktree, str | None, str | None], MergeResult]
"""``(agent, wt, user_prompt, task_result) -> MergeResult``: completes a
conflicted merge of ``wt.branch`` into the clean, checked-out original
branch (see :meth:`SorcarAgent._do_merge`)."""


class _WorktreeCleanupOutcome(enum.Enum):
    """Outcome of :meth:`SorcarAgent._commit_and_clean_worktree`."""

    COMMITTED_AND_REMOVED = "committed_and_removed"
    PRESERVED_COMMIT_FAILED = "preserved_commit_failed"


_PRECOMMIT_FIX_LINES = (
    "    # fix pre-commit issues, then:\n"
    "    git commit --no-verify\n"
)


def inside_kiss_worktree(path: Path | str) -> bool:
    """Whether *path* lies inside a ``<repo>/.kiss-worktrees/kiss_wt-*`` worktree.

    Args:
        path: A filesystem path (need not exist).

    Returns:
        True when a worktree segment is part of the path.
    """
    return _worktree_parts(str(path))[1] >= 0


def _path_inside(path: Path, root: Path) -> Path | None:
    """*path* relative to *root*, or ``None`` when it is not under *root*.

    Args:
        path: A resolved absolute path.
        root: A directory, resolved here.
    """
    try:
        return path.relative_to(root.resolve())
    except ValueError:
        return None


def _repo_key(wt: GitWorktree) -> Path:
    """The key of *wt* in ``SorcarAgent._wts``: its resolved repository root."""
    return Path(wt.repo_root or ".").resolve()


def _manual_merge_cmd(wt: GitWorktree) -> str:
    """Return the correct manual merge command for a worktree.

    When a baseline commit exists, the command applies the agent's net
    change (``baseline..branch``) exactly as the auto-merge does: ONE
    ``cherry-pick --no-commit`` of a synthetic commit holding the
    branch's tree with the baseline as parent (see
    :meth:`GitWorktreeOps._squashed_commit`), created inline by
    ``git commit-tree``.  Replaying the range commit by commit would
    refuse a merge commit on the branch (``no -m option``), and
    ``git merge --squash`` would incorrectly include the baseline's
    dirty-state snapshot.

    Args:
        wt: The worktree state.

    Returns:
        A shell command string for manual merge.
    """
    if wt.baseline_commit:
        squashed = (
            f"git commit-tree {shlex.quote(wt.branch + '^{tree}')}"
            f" -p {shlex.quote(wt.baseline_commit)}"
            f" -m {shlex.quote(f'kiss: squashed {wt.branch}')}"
        )
        return f'git cherry-pick --no-commit "$({squashed})"'
    return f"git merge --squash {shlex.quote(wt.branch)}"


def _merge_fix_steps(wt: GitWorktree, fix_lines: str) -> str:
    """Return the shell command block for manually completing a failed merge.

    Args:
        wt: The worktree state.
        fix_lines: Result-specific middle lines (conflict resolution or
            pre-commit fix steps), each ending in a newline.

    The final step uses ``git branch -D`` (force): a squash-merge /
    cherry-pick resolution never records the task branch as an
    ancestor of the original branch, so ``git branch -d`` would
    always refuse with "the branch ... is not fully merged".

    Returns:
        The indented multi-line command block (no trailing newline).
    """
    return (
        f"    cd {shlex.quote(str(wt.repo_root))}\n"
        f"    git checkout {shlex.quote(wt.original_branch or '')}\n"
        f"    {_manual_merge_cmd(wt)}\n"
        + fix_lines
        + f"    git branch -D {shlex.quote(wt.branch)}"
    )
