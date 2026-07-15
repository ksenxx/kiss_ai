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

import enum
import logging
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_repo_locks: dict[str, threading.RLock] = {}
_repo_locks_guard = threading.Lock()

# Repo-scoped git environment variables that OVERRIDE ``git -C``-based
# repository discovery.  When KISS is launched from a context that
# exports them — e.g. a git hook (``post-commit`` starting an agent),
# ``git rebase --exec``, or a user shell export — every git call would
# silently target the WRONG repository (the hook's repo) instead of the
# ``cwd`` passed to :func:`_git`.  This mirrors the list git itself
# clears before crossing repo boundaries (``local_repo_env`` — see
# ``git submodule``).  Author/committer/SSH/config-file variables are
# intentionally kept.
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


# Canonical commit-message metadata block headings.  Single-sourced
# here because :func:`_ensure_task_metadata`'s dedup detection depends
# on byte-exact agreement with what the writers append — both this
# module's stamping below and ``vscode/helpers._append_user_prompt`` /
# ``_append_task_result`` (which import these constants) build blocks
# as ``f"{HEADING}{text}"`` appended to an ``rstrip()``-ed message.
USER_PROMPT_HEADING = "\n\nUser prompt:\n"
TASK_RESULT_HEADING = "\n\nResult:\n"


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
    not be double-stamped).

    Dedup is by exact current-value suffix — NEVER by heading
    substring: a hand-written commit body that merely mentions
    ``Result:``, a stale block from a previous task, or a prompt whose
    own text contains ``Result:`` must not suppress stamping the
    current values (gpt-5.6-sol review findings).  When the message
    ends with only the current result block, the missing prompt block
    is inserted before it to preserve canonical prompt-then-result
    order.

    Args:
        message: The base commit message (subject + optional body).
        user_prompt: The current task's prompt, or ``None``/empty.
        task_result: The current task's result summary, or
            ``None``/empty.

    Returns:
        The commit message ending with the canonical metadata blocks
        for every non-empty current value.
    """
    msg = message.rstrip()
    prompt = user_prompt.strip() if user_prompt else ""
    result = task_result.strip() if task_result else ""
    prompt_block = f"{USER_PROMPT_HEADING}{prompt}" if prompt else ""
    result_block = f"{TASK_RESULT_HEADING}{result}" if result else ""
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

    Args:
        repo: Git repo root path.

    Returns:
        A :class:`threading.RLock` specific to the resolved repo path.
    """
    key = str(repo.resolve())
    with _repo_locks_guard:
        if key not in _repo_locks:
            _repo_locks[key] = threading.RLock()
        return _repo_locks[key]


# Local git operations can legitimately be slow in very large repositories
# (``add -A``, stash, merge/cherry-pick, and user pre-commit hooks).  Five
# minutes matches the Bash tool's normal execution budget while still putting
# a hard ceiling on a wedged hook or git process.
_GIT_TIMEOUT_SECONDS: float = 300.0


def _git(
    *args: str,
    cwd: str | Path,
) -> subprocess.CompletedProcess[str]:
    """Run a git command, returning the CompletedProcess result.

    ``cwd`` is a required keyword-only argument so the type checker
    enforces that every git invocation specifies a working directory
    (passed via ``git -C <cwd>``).  This prevents accidental git
    operations against the process's current working directory.

    Always passes a timeout so a hung git process (e.g. waiting on a
    credential-helper prompt or a network remote) cannot block the
    agent thread forever — the same protection
    ``vscode/diff_merge._git`` documents.  On expiry a synthesized
    non-zero ``CompletedProcess`` (returncode 124, the timeout
    convention) is returned so callers keep working.

    Args:
        *args: Git sub-command and arguments (without the leading ``git``).
        cwd: Working directory for the git command (required).

    Returns:
        The completed process with stdout/stderr captured as text.
    """
    # ``-c core.quotepath=false`` forces git to emit non-ASCII filenames
    # verbatim (UTF-8) rather than as C-style ``\NNN`` octal escapes,
    # regardless of the repo's local ``core.quotePath`` config.  All
    # callers parse these outputs as plain UTF-8 paths.
    cmd = [
        "git",
        "-c",
        "core.quotepath=false",
        "-C",
        str(cwd),
        *args,
    ]
    # Scrub repo-scoped GIT_* variables so an inherited GIT_DIR /
    # GIT_WORK_TREE / GIT_INDEX_FILE (e.g. from a git hook that
    # launched this process) cannot redirect the command away from
    # ``cwd`` (see :data:`_REPO_SCOPED_GIT_ENV`).
    env = {k: v for k, v in os.environ.items() if k not in _REPO_SCOPED_GIT_ENV}
    # ``errors="surrogateescape"``: git paths are byte strings and may
    # be invalid UTF-8 (e.g. Latin-1 filenames committed on Linux);
    # with ``core.quotepath=false`` such bytes are emitted verbatim and
    # a strict decode would raise UnicodeDecodeError out of EVERY git
    # call.  Surrogate escapes round-trip through ``os.fsencode`` for
    # filesystem operations, matching :func:`_unquote_git_path`.
    # Use a fresh POSIX process group rather than ``subprocess.run``.
    # Killing only the top-level git process on timeout is insufficient:
    # a hung hook can keep the captured stdout/stderr pipes open, making
    # ``run``'s post-timeout ``communicate()`` wait indefinitely even
    # after git itself has died.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        env=env,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = proc.communicate(timeout=_GIT_TIMEOUT_SECONDS)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        logger.warning("git %s timed out after %ss", args, _GIT_TIMEOUT_SECONDS)
        if os.name != "nt":
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
        else:  # pragma: no cover - Windows CI is not available here
            # ``Popen.kill`` does not terminate hook descendants on
            # Windows; taskkill's /T switch closes the whole tree.
            subprocess.run(  # noqa: S603, S607
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
                check=False,
            )
            if proc.poll() is None:
                proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            proc.kill()
            stdout = (
                exc.stdout.decode("utf-8", "surrogateescape")
                if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode("utf-8", "surrogateescape")
                if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            )
        # TimeoutExpired may expose bytes even for a text-mode process.
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "surrogateescape")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "surrogateescape")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,  # convention: timeout
            stdout=stdout or "",
            stderr=stderr
            or f"git {args[0] if args else ''} timed out"
            f" after {_GIT_TIMEOUT_SECONDS}s",
        )


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
    """

    repo_root: Path
    branch: str
    original_branch: str | None
    wt_dir: Path
    baseline_commit: str | None = None


class MergeResult(enum.Enum):
    """Outcome of a merge operation."""

    SUCCESS = "success"
    CONFLICT = "conflict"
    CHECKOUT_FAILED = "checkout_failed"
    MERGE_FAILED = "merge_failed"
    STASH_FAILED = "stash_failed"


# Sentinel directory under which the framework creates per-task worktrees.
# Layout: ``<repo>/.kiss-worktrees/kiss_wt-<slug>/...``.
_WORKTREE_SUBDIR = ".kiss-worktrees"
_WORKTREE_SLUG_PREFIX = "kiss_wt-"


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
    if not path:
        return path
    # Split on os-agnostic separators so the helper works on POSIX paths
    # carried across platforms (tests, payloads from remote daemons).
    # ``str.replace`` first folds Windows separators to ``/`` so the
    # logic is uniform.
    norm = path.replace("\\", "/")
    parts = norm.split("/")
    for i, segment in enumerate(parts):
        if (
            segment == _WORKTREE_SUBDIR
            and i + 1 < len(parts)
            and parts[i + 1].startswith(_WORKTREE_SLUG_PREFIX)
        ):
            parent = "/".join(parts[:i])
            if parent:
                # Normal case: ``/Users/x/proj/.kiss-worktrees/kiss_wt-…``
                # → ``/Users/x/proj``.
                return parent
            # No parent segments before ``.kiss-worktrees``.  For an
            # absolute path (``/.kiss-worktrees/kiss_wt-…``) the parent
            # is the filesystem root; for a relative path
            # (``.kiss-worktrees/kiss_wt-…``) it is the current dir.
            return "/" if norm.startswith("/") else "."
    return path


def _unquote_git_path(path: str) -> str:
    """Unquote a C-style quoted filename from ``git status --porcelain``.

    Git quotes filenames containing non-ASCII bytes (>0x7F), control
    characters, double-quotes, or backslashes.  The quoted form is
    surrounded by double-quotes with C-style escape sequences
    (``\\n``, ``\\t``, ``\\\\``, ``\\"``, ``\\NNN`` octal).

    When the path is not quoted (no surrounding double-quotes), it is
    returned unchanged.

    Args:
        path: Raw filename string from ``git status --porcelain`` output.

    Returns:
        The unquoted filename.
    """
    if not (path.startswith('"') and path.endswith('"')):
        return path
    inner = path[1:-1]
    raw = bytearray()
    i = 0
    _esc = {
        "n": 0x0A,
        "t": 0x09,
        "\\": 0x5C,
        '"': 0x22,
        "a": 0x07,
        "b": 0x08,
        "f": 0x0C,
        "r": 0x0D,
        "v": 0x0B,
    }
    while i < len(inner):
        if inner[i] == "\\" and i + 1 < len(inner):
            nxt = inner[i + 1]
            if nxt in _esc:
                raw.append(_esc[nxt])
                i += 2
            elif (
                nxt.isdigit()
                and i + 3 < len(inner)
                and inner[i + 2].isdigit()
                and inner[i + 3].isdigit()
            ):
                raw.append(int(inner[i + 1 : i + 4], 8))
                i += 4
            else:
                raw.append(ord("\\"))
                i += 1
        else:
            raw.extend(inner[i].encode("utf-8"))
            i += 1
    return raw.decode("utf-8", errors="surrogateescape")


def _split_rename_tail(tail: str) -> tuple[str, str]:
    """Split a porcelain rename/copy tail ``old -> new`` into raw sides.

    Each side may independently be C-style quoted (git quotes a path
    when it contains control characters, double-quotes, or
    backslashes), so the split must respect quoting and happen BEFORE
    any unquoting.  The returned sides are still raw (possibly quoted)
    and must each be passed through :func:`_unquote_git_path` exactly
    once.

    Args:
        tail: The raw status-line tail after the two status characters
            and the separating space (e.g. ``"a\\tb" -> "c\\td"``).

    Returns:
        ``(old_raw, new_raw)`` — the two sides of the rename, unsplit
        quoting intact.
    """
    if tail.startswith('"'):
        i = 1
        while i < len(tail):
            if tail[i] == "\\":
                i += 2
            elif tail[i] == '"':
                rest = tail[i + 1 :]
                if rest.startswith(" -> "):
                    return tail[: i + 1], rest[4:]
                break
            else:
                i += 1
    idx = tail.index(" -> ")
    return tail[:idx], tail[idx + 4 :]


def _porcelain_entries(output: str) -> list[tuple[str, str | None, str]]:
    """Parse ``git status --porcelain`` output into unquoted entries.

    The single shared porcelain parser — used by
    :meth:`GitWorktreeOps.copy_dirty_state` and (via
    ``merge_flow._porcelain_paths``) by the VS Code merge flow — so
    the parsers cannot drift apart.

    The output is split on ``\\n`` only (NOT ``splitlines()``, which
    would also split on a raw unicode line/paragraph-separator byte
    sequence inside an unquoted filename) and the path tail
    (``line[3:]``) is never ``strip()``-ed: space-adjacent filenames
    are legal and git leaves them unquoted.  Rename/copy entries are
    split on the `` -> `` boundary (respecting quoting) BEFORE each
    side is unquoted exactly once.

    Args:
        output: Raw stdout from a ``git status --porcelain`` command.

    Returns:
        List of ``(code, old_name, new_name)`` tuples in output order:
        ``code`` is the two-character status code, ``old_name`` is the
        rename/copy source path (``None`` for non-rename entries), and
        ``new_name`` is the entry's (current) path.
    """
    entries: list[tuple[str, str | None, str]] = []
    for line in output.split("\n"):
        if len(line) < 4:
            continue
        code = line[:2]
        tail = line[3:]
        if ("R" in code or "C" in code) and " -> " in tail:
            old_raw, new_raw = _split_rename_tail(tail)
            entries.append(
                (code, _unquote_git_path(old_raw), _unquote_git_path(new_raw))
            )
        else:
            entries.append((code, None, _unquote_git_path(tail)))
    return entries


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
        return Path(result.stdout.strip())

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

        Returns:
            True if worktree was created successfully, False otherwise.
        """
        result = _git("worktree", "add", "-b", branch, str(wt_dir), cwd=repo)
        if result.returncode != 0:
            logger.warning("Failed to create worktree: %s", result.stderr.strip())
            return False
        return True

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

        Args:
            repo: Git repo root path.
            wt_dir: Worktree directory to remove.
        """
        if not wt_dir.exists():
            # The directory may have been deleted manually or by crash
            # cleanup while git still holds a ``.git/worktrees/<name>``
            # registration; without a prune the branch stays "checked
            # out in a worktree" and ``git branch -d/-D`` refuses to
            # delete it.
            GitWorktreeOps.prune(repo)
            return
        result = _git("worktree", "remove", str(wt_dir), "--force", cwd=repo)
        if result.returncode != 0:
            result = _git(
                "worktree", "remove", "--force", "--force", str(wt_dir), cwd=repo
            )
        if result.returncode != 0:
            logger.warning(
                "worktree remove failed: %s; deleting directory directly",
                result.stderr.strip(),
            )
            shutil.rmtree(str(wt_dir), ignore_errors=True)
            GitWorktreeOps.prune(repo)

    @staticmethod
    def prune(repo: Path) -> None:
        """Prune stale worktree bookkeeping entries.

        Args:
            repo: Git repo root path.
        """
        _git("worktree", "prune", cwd=repo)

    @staticmethod
    def stage_all(wt_dir: Path) -> None:
        """Stage all changes in the worktree (``git add -A``).

        Args:
            wt_dir: Worktree directory.
        """
        _git("add", "-A", cwd=wt_dir)

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
        diff = _git("diff", "--cached", "--quiet", cwd=wt_dir)
        if diff.returncode == 0:
            return False
        cmd = ["commit", "-m", message]
        if no_verify:
            cmd.append("--no-verify")
        result = _git(*cmd, cwd=wt_dir)
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
        """
        status = _git("status", "--porcelain", cwd=wt_dir)
        return bool(status.stdout.strip())

    @staticmethod
    def status_porcelain(wt_dir: Path) -> str:
        """Return the raw ``git status --porcelain`` output.

        Used by failure-path warnings (e.g.
        :meth:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent._finalize_worktree`)
        to embed the exact leftover files in the log so an operator
        can tell a real pre-commit-hook rejection apart from a race
        leftover or a corrupt index without having to ssh in and
        run git themselves.

        Args:
            wt_dir: Git working directory to inspect.

        Returns:
            The stripped porcelain output (possibly empty).
        """
        return _git("status", "--porcelain", cwd=wt_dir).stdout.strip()

    @staticmethod
    def staged_diff(wt_dir: Path) -> str:
        """Return the staged diff text for the worktree.

        Args:
            wt_dir: Worktree directory (must have staged changes).

        Returns:
            The diff text, or empty string if no staged changes.
        """
        result = _git("diff", "--cached", cwd=wt_dir)
        return result.stdout.strip()

    @staticmethod
    def checkout(repo: Path, branch: str) -> tuple[bool, str]:
        """Checkout a branch in the main worktree.

        Args:
            repo: Git repo root path.
            branch: Branch name to checkout.

        Returns:
            ``(True, "")`` on success, ``(False, stderr)`` on failure.
            The stderr string describes why the checkout failed (e.g.
            dirty working tree, missing branch).
        """
        result = _git("checkout", branch, cwd=repo)
        if result.returncode == 0:
            return (True, "")
        return (False, result.stderr.strip())

    @staticmethod
    def stash_if_dirty(repo: Path) -> bool:
        """Stash uncommitted changes if the working tree or index is dirty.

        Uses ``git stash push --include-untracked`` so both staged and
        unstaged changes (including new files) are saved.

        Args:
            repo: Git repo root path.

        Returns:
            True if a stash entry was created, False if the tree was
            clean or nothing could be stashed.
        """
        if not GitWorktreeOps.has_uncommitted_changes(repo):
            return False
        # ``git stash push`` exits 0 WITHOUT creating a stash ("No
        # local changes to save") for dirtiness it cannot capture —
        # e.g. a submodule with modified/untracked content, or a tree
        # cleaned by a concurrent writer.  Returning True then makes
        # the caller's later ``stash_pop`` consume an unrelated,
        # pre-existing user stash.  Compare ``refs/stash`` before and
        # after so the return value honors the "a stash entry was
        # created" contract.
        before = _git("rev-parse", "-q", "--verify", "refs/stash", cwd=repo)
        result = _git(
            "stash",
            "push",
            "--include-untracked",
            "-m",
            "kiss: auto-stash before merge",
            cwd=repo,
        )
        if result.returncode != 0:
            return False
        after = _git("rev-parse", "-q", "--verify", "refs/stash", cwd=repo)
        after_sha = after.stdout.strip()
        return bool(after_sha) and after_sha != before.stdout.strip()

    @staticmethod
    def stash_pop(repo: Path) -> bool:
        """Pop the latest stash entry, preserving the staging state.

        Tries ``git stash pop --index`` first so that files that were
        staged before the stash stay staged after the pop.  If
        ``--index`` fails (e.g. the merge changed a file that was in
        the index), falls back to plain ``git stash pop`` which
        restores all changes as unstaged — but ONLY when the failed
        ``--index`` attempt left the working tree untouched.  A failed
        ``--index`` pop can partially apply the stash (e.g. tracked
        changes applied while an untracked-file restore failed, or a
        conflicted merge) while keeping the stash entry; blindly
        re-applying the same stash on top of that state risks a
        double-apply, so in that case the stash is left for the caller
        to surface to the user.

        Args:
            repo: Git repo root path.

        Returns:
            True if the pop succeeded, False on conflict or error.
        """
        before = _git("status", "--porcelain", cwd=repo).stdout
        result = _git("stash", "pop", "--index", cwd=repo)
        if result.returncode == 0:
            return True
        after = _git("status", "--porcelain", cwd=repo).stdout
        if after != before:
            # The failed ``--index`` attempt already modified the tree
            # (partial application; the stash entry is retained by
            # git).  Do not re-apply the same stash on top.
            return False
        result = _git("stash", "pop", cwd=repo)
        return result.returncode == 0

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
        # bughunt8: terminate the revision list with ``--`` — without
        # it, git refuses the command with "ambiguous argument
        # '<branch>': both revision and filename" whenever the user's
        # repo contains a file whose path equals the branch name,
        # silently degrading every merge commit message to the
        # synthetic fallback.
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
        changes, commit them with the message from
        :meth:`_merge_commit_message`; on commit failure (e.g. a
        rejecting pre-commit hook), reset hard and report
        :attr:`MergeResult.MERGE_FAILED`.

        Args:
            repo: Git repo root path.
            branch: The worktree branch the changes came from.
            user_prompt: The user's task prompt, appended to the merge
                commit message when not already present (see
                :meth:`_merge_commit_message`), or ``None``.
            task_result: The task's result summary, appended likewise,
                or ``None``.

        Returns:
            :attr:`MergeResult.SUCCESS` or :attr:`MergeResult.MERGE_FAILED`.
        """
        diff = _git("diff", "--cached", "--quiet", cwd=repo)
        if diff.returncode != 0:
            msg = GitWorktreeOps._merge_commit_message(
                repo, branch, user_prompt=user_prompt, task_result=task_result,
            )
            commit_result = _git("commit", "-m", msg, cwd=repo)
            if commit_result.returncode != 0:
                logger.warning(
                    "squash merge commit failed: %s",
                    commit_result.stderr.strip(),
                )
                _git("reset", "--hard", "HEAD", cwd=repo)
                return MergeResult.MERGE_FAILED
        return MergeResult.SUCCESS

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

        On conflict, resets to a clean state with ``git reset --hard``.

        Args:
            repo: Git repo root path.
            branch: Branch to squash-merge.
            user_prompt: The user's task prompt for the merge commit
                message, or ``None``.
            task_result: The task's result summary for the merge
                commit message, or ``None``.

        Returns:
            :attr:`MergeResult.SUCCESS` or :attr:`MergeResult.CONFLICT`.
        """
        result = _git("merge", "--squash", branch, cwd=repo)
        if result.returncode != 0:
            logger.warning(
                "squash merge failed: %s",
                result.stderr.strip(),
            )
            _git("reset", "--hard", "HEAD", cwd=repo)
            return MergeResult.CONFLICT
        return GitWorktreeOps._commit_staged_merge(
            repo, branch, user_prompt=user_prompt, task_result=task_result,
        )

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
        """Delete a branch and its git config section (best-effort).

        Tries ``-d`` first (safe delete), falls back to ``-D`` (force).
        Also removes the ``branch.<name>.*`` config section.

        Args:
            repo: Git repo root path.
            branch: Branch name to delete.

        Returns:
            True if the branch was deleted (or never existed), False
            if git refused both ``-d`` and ``-D`` — typically because
            the branch is the current HEAD of a worktree and cannot
            be deleted without first switching away.
        """
        result = _git("branch", "-d", branch, cwd=repo)
        if result.returncode != 0:
            result = _git("branch", "-D", branch, cwd=repo)
        if result.returncode == 0:
            GitWorktreeOps._remove_branch_config_section(repo, branch)
            return True
        if GitWorktreeOps.branch_exists(repo, branch):
            logger.warning(
                "Failed to delete branch '%s': %s",
                branch,
                result.stderr.strip(),
            )
            return False
        GitWorktreeOps._remove_branch_config_section(repo, branch)
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
    def _append_info_line(repo: Path, filename: str, entry: str) -> None:
        """Append *entry* to ``<git_common_dir>/info/<filename>`` once.

        The ``info/`` directory holds repo-local, untracked plumbing
        (``exclude``, ``attributes``) so the agent never modifies any
        tracked file in the user's repo.  Idempotent: the line is only
        appended when not already present.

        Args:
            repo: Git repo root path.
            filename: File under ``info/`` (e.g. ``"exclude"``).
            entry: The exact line to ensure is present.
        """
        # The read-check-append sequence below is not atomic; hold the
        # per-repo lock so concurrent tabs setting up worktrees for the
        # same repo cannot interleave and append duplicate entries.
        with repo_lock(repo):
            result = _git("rev-parse", "--git-common-dir", cwd=repo)
            git_common = Path(result.stdout.strip())
            if not git_common.is_absolute():  # pragma: no branch
                git_common = (repo / git_common).resolve()
            info_file = git_common / "info" / filename
            info_file.parent.mkdir(parents=True, exist_ok=True)
            content = ""
            if info_file.exists():
                # Git treats these files as raw bytes — non-UTF-8
                # patterns/comments are legal, so a strict decode would
                # raise UnicodeDecodeError and silently skip the entry
                # (the caller swallows exceptions), e.g. leaving
                # ``?? .kiss-worktrees/`` polluting the user's git status
                # forever.  Mirror :func:`_git`'s surrogateescape policy.
                content = info_file.read_bytes().decode(
                    "utf-8", errors="surrogateescape"
                )
                if entry in content.splitlines():
                    return
            # Prefix a newline only when the existing content lacks a
            # trailing one; unconditionally writing ``"\n{entry}\n"``
            # accumulated a blank line per append.
            prefix = "" if not content or content.endswith("\n") else "\n"
            with open(info_file, "a", encoding="utf-8") as f:
                f.write(f"{prefix}{entry}\n")

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
        _git(
            "config",
            "merge.kiss-scratch.name",
            "KISS scratch file: keep the incoming task branch version",
            cwd=repo,
        )
        # %A = temp file that must receive the merge result (starts as
        # "ours"), %B = the other branch's version; exit 0 = resolved.
        _git("config", "merge.kiss-scratch.driver", "cp -f %B %A", cwd=repo)

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

        Reads ``git status --porcelain`` in *repo* and mirrors every
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
        """
        status = _git("status", "--porcelain", "-uall", cwd=repo)
        if not status.stdout.strip():
            return False

        copied = False
        # Parse via the shared :func:`_porcelain_entries` helper (the
        # same parser backing merge_flow's ``_porcelain_paths``) so the
        # baseline-seeding and merge-flow porcelain parsers cannot
        # drift apart (split on ``\n`` only, no strip, quote-aware
        # rename splitting).
        for _code, old_name, fname in _porcelain_entries(status.stdout):
            src = repo / fname
            dst = wt_dir / fname

            if old_name is not None:
                # The rename's old path is gone from the main worktree
                # regardless of what happened to the new path, so it
                # must be removed from the task worktree even when the
                # new file was subsequently deleted (e.g. status "RD").
                old_dst = wt_dir / old_name
                # is_symlink() is checked (by _remove_path) FIRST:
                # is_dir()/exists() follow symlinks, so a symlink to a
                # directory must be unlinked (not rmtree'd) and a
                # broken symlink reports exists() == False.
                if old_dst.is_symlink() or old_dst.exists():
                    GitWorktreeOps._remove_path(old_dst)
                    copied = True

            if src.is_symlink():
                # Mirror the symlink itself (possibly broken); is_file()
                # and copy2 would follow the link instead.
                GitWorktreeOps._remove_path(dst)
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(os.readlink(src), dst)
                copied = True
            elif src.is_file():
                if dst.is_symlink():
                    # A symlink was replaced by a regular file; copy2
                    # onto the link would write THROUGH it into the
                    # link's target inside the worktree.
                    dst.unlink()
                elif dst.is_dir():
                    # A tracked directory was replaced by a same-named
                    # file; copy2 into the directory would create
                    # dst/<basename> instead of replacing dst.
                    shutil.rmtree(str(dst))
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
                copied = True
            elif dst.is_symlink():
                # The path is gone in the main worktree but the fresh
                # checkout has a (possibly broken) symlink there;
                # exists() follows the link and would miss it.
                dst.unlink()
                copied = True
            elif dst.is_dir():
                if not src.exists():
                    # The path was deleted in the main worktree but the
                    # fresh checkout has a directory there (file/dir
                    # type change); unlink() would raise on a dir.
                    shutil.rmtree(str(dst))
                    copied = True
            elif dst.exists():
                dst.unlink()
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
    def _head_matches_baseline_parent(repo: Path, baseline: str) -> bool:
        """Return True when ``HEAD == baseline^`` in *repo*.

        Used by :meth:`squash_merge_from_baseline` to decide whether
        ``-X theirs`` is safe — see that method's docstring for the
        full analysis.  Returns ``False`` on any git error: a
        conservative default that disables ``-X theirs`` rather than
        risk silently losing user commits.
        """
        head = _git("rev-parse", "HEAD", cwd=repo)
        parent = _git("rev-parse", f"{baseline}^", cwd=repo)
        if head.returncode != 0 or parent.returncode != 0:
            return False
        head_sha = head.stdout.strip()
        parent_sha = parent.stdout.strip()
        return bool(head_sha) and head_sha == parent_sha

    @staticmethod
    def squash_merge_from_baseline(
        repo: Path,
        branch: str,
        baseline: str,
        user_prompt: str | None = None,
        task_result: str | None = None,
    ) -> MergeResult:
        """Squash-merge only the agent's changes (after baseline) into HEAD.

        Uses ``git cherry-pick --no-commit`` to replay each commit
        after *baseline* onto the current HEAD.  Cherry-pick performs
        a proper three-way merge per commit (using the commit's parent
        as the merge base), so it handles cases where the user's dirty
        state (captured in the baseline) diverges from the committed
        HEAD content.

        **The ``-X theirs`` strategy option is added precisely when —
        and only when — ``HEAD == baseline^``** (i.e. main has not
        advanced since worktree creation).  This is the common case
        and is exactly when the cherry-pick would otherwise fabricate
        a spurious modify/delete or modify/modify conflict:

        - ``_do_merge`` stashes the user's dirty edits on main before
          this call, so main HEAD == ``baseline^`` (clean).  But the
          baseline commit captured the user's dirty state — so for the
          first cherry-picked commit the 3-way merge sees:

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

        When ``HEAD != baseline^`` (the user committed independent
        work on main between WT creation and merge), ``-X theirs`` is
        deliberately NOT added: any conflict in that case is a real
        cross-branch divergence between the user's main commits and
        the agent's branch tip, and silently letting the branch win
        would destroy the user's intentional commits.

        Falls back to :meth:`squash_merge_branch` when *baseline* is
        ``None`` (legacy worktrees).

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
            :attr:`MergeResult.SUCCESS` or :attr:`MergeResult.CONFLICT`.
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

        cherry_pick_args = ["cherry-pick", "--no-commit"]
        if GitWorktreeOps._head_matches_baseline_parent(repo, baseline):
            # Resolve hunk-level conflicts caused by the user's dirty
            # edits living in ``baseline`` but not in ``HEAD`` in favor
            # of the branch tip — see method docstring for the full
            # 3-way-merge analysis.  Only safe when HEAD == baseline^.
            cherry_pick_args.extend(["-X", "theirs"])
        cherry_pick_args.append(f"{baseline}..{branch}")
        result = _git(*cherry_pick_args, cwd=repo)
        if result.returncode != 0:
            logger.warning(
                "squash merge from baseline failed: %s",
                result.stderr.strip(),
            )
            _git("cherry-pick", "--abort", cwd=repo)
            return MergeResult.CONFLICT

        return GitWorktreeOps._commit_staged_merge(
            repo, branch, user_prompt=user_prompt, task_result=task_result,
        )

    @staticmethod
    def cleanup_partial(repo: Path, branch: str, wt_dir: Path) -> None:
        """Remove a partially-created worktree and branch (best-effort).

        Args:
            repo: Git repo root path.
            branch: The branch name to delete.
            wt_dir: The worktree directory to remove.
        """
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)
        GitWorktreeOps.delete_branch(repo, branch)
