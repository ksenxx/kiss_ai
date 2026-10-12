# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The git steps behind ``/sync``: fetch a remote branch, merge it, push.

:func:`sync` is the ``git_sync`` tool of
:mod:`kiss.agents.seas.sync.sync_sea`.  It runs the sequence a "pull,
merge and push" prompt used to make the chat agent type by hand,
deterministically and in one call:

1. ``git fetch <remote> +refs/heads/<branch>:refs/remotes/<remote>/<branch>``
   for the ref to merge (``origin/main`` by default; the explicit
   refspec updates the tracking ref also in a single-branch clone);
2. ``git merge --no-edit <remote>/<branch>`` when the current branch is
   behind it (a fast-forward when it has no commits of its own);
3. ``git push <remote> HEAD`` — the current branch to the remote branch
   of the same name.

A merge that stops on conflicts is left in progress and reported with
the conflicted files, so the agent can resolve them, ``git add`` and
``git commit --no-edit``, then call the tool again (which fetches,
finds nothing new, and pushes).  A push the remote rejects because the
branch moved there in the meantime is retried, after merging the
remote's copy of the branch, up to :data:`PUSH_ATTEMPTS` rounds;
nothing is ever forced.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PUSH_ATTEMPTS = 3
"""How many fetch → merge → push rounds a push race (the remote moved again) gets."""

GIT_TIMEOUT_SECONDS = 300.0
"""Seconds one git command (a fetch or push over the network) may take."""

_REJECTED_PUSH_HINTS = ("fetch first", "non-fast-forward", "cannot lock ref")
"""Fragments of ``git push`` stderr that mean the remote moved since the fetch."""

_CONFLICT_MARKER_PATTERN = r"^(<<<<<<< |>>>>>>> )"
"""The ``git grep -E`` pattern for conflict markers; ``=======`` alone is an RST underline."""


class GitSyncError(Exception):
    """A step of :func:`sync` failed; the message is the report for the agent."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``git *args`` in *repo* and return the completed process (never raises on exit code).

    The output is decoded as UTF-8, which is how git writes paths and
    messages on every platform (``text=True`` alone would use the ANSI
    code page on Windows and garble a non-ASCII file name).

    Raises:
        GitSyncError: When git cannot be run at all (not installed, or a
            command exceeded :data:`GIT_TIMEOUT_SECONDS`).
    """
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise GitSyncError(f"git {' '.join(args)} could not run: {exc}") from exc


def _output(repo: Path, *args: str) -> str:
    """Return the stripped stdout of ``git *args``; raise :class:`GitSyncError` on failure."""
    proc = _git(repo, *args)
    if proc.returncode != 0:
        raise GitSyncError(f"git {' '.join(args)} failed:\n{(proc.stderr or proc.stdout).strip()}")
    return proc.stdout.strip()


def _lines(repo: Path, *args: str) -> list[str]:
    """Return the non-empty stdout lines of ``git *args`` (empty on failure)."""
    proc = _git(repo, *args)
    return [line for line in proc.stdout.splitlines() if line] if proc.returncode == 0 else []


def _paths(repo: Path, command: str, *args: str) -> list[str]:
    """Return the paths ``git <command> -z *args`` prints: NUL-separated, never quoted."""
    proc = _git(repo, command, "-z", *args)
    return [name for name in proc.stdout.split("\0") if name] if proc.returncode == 0 else []


def _split_ref(ref: str) -> tuple[str, str]:
    """Split ``<remote>/<branch>`` into its two parts."""
    remote, slash, branch = ref.strip().partition("/")
    if not slash or not remote or not branch:
        raise GitSyncError(f"the ref to merge must be <remote>/<branch>, not {ref!r}")
    return remote, branch


def _merge(repo: Path, ref: str) -> str:
    """Merge *ref* into the current branch and return the report line.

    Raises:
        GitSyncError: The merge stopped on conflicts (left in progress,
            the conflicted files listed) or did not start (for example an
            uncommitted change would be overwritten).
    """
    before = _output(repo, "rev-parse", "HEAD")
    proc = _git(repo, "merge", "--no-edit", ref)
    if proc.returncode != 0:
        conflicted = _paths(repo, "diff", "--name-only", "--diff-filter=U")
        if conflicted:
            raise GitSyncError(
                f"Merging {ref} stopped on {len(conflicted)} conflicted file(s); the merge is "
                "left in progress:\n  " + "\n  ".join(conflicted)
                + "\nResolve every conflict, `git add` the files, `git commit --no-edit`, "
                "then call git_sync again to push."
            )
        detail = (proc.stderr or proc.stdout).strip()
        raise GitSyncError(f"git merge {ref} did not start:\n{detail}")
    after = _output(repo, "rev-parse", "HEAD")
    changed = _paths(repo, "diff", "--name-only", before, after)
    kind = "Fast-forwarded to" if after == _output(repo, "rev-parse", ref) else "Merged"
    line = f"{kind} {ref}: now at {after[:12]} ({len(changed)} files changed)."
    with_markers = []
    if changed:
        with_markers = _paths(repo, "grep", "-l", "-E", _CONFLICT_MARKER_PATTERN, "--", *changed)
    if with_markers:
        line += (
            "\nWARNING: conflict markers are committed in the merged files (they came from "
            "the remote, not from this merge): " + ", ".join(with_markers)
        )
    return line


def _fetch_and_merge(repo: Path, remote: str, name: str, branch: str, report: list[str]) -> None:
    """Fetch ``<remote>/<name>`` and merge it into *branch* (the current one) when behind.

    Appends the fetch and merge lines to *report*; the merge's errors
    propagate (see :func:`_merge`).
    """
    ref = f"{remote}/{name}"
    _output(repo, "fetch", remote, f"+refs/heads/{name}:refs/remotes/{remote}/{name}")
    ahead, behind = _output(repo, "rev-list", "--left-right", "--count", f"HEAD...{ref}").split()
    report.append(f"Fetched {ref}: {branch} is {ahead} ahead and {behind} behind.")
    if int(behind):
        report.append(_merge(repo, ref))
    else:
        report.append(f"Nothing to merge: {ref} is already part of {branch}.")


def sync(repo: str, ref: str = "origin/main") -> str:
    """Fetch *ref*, merge it into the current branch of *repo* and push the branch.

    Args:
        repo: Absolute path of the git checkout (the run's work directory).
        ref: The remote branch to merge, as ``<remote>/<branch>``.

    Returns:
        A line-per-step report: the branch, what was fetched, whether
        the merge was a fast-forward, a merge commit or nothing to do,
        and what the push did.

    Raises:
        GitSyncError: With the report of the step that failed: not a
            checkout, a detached HEAD, a merge already in progress, a
            fetch or push error, or a merge that stopped on conflicts
            (left in place for the caller to resolve).
    """
    remote, remote_branch = _split_ref(ref)
    path = Path(repo)
    if not path.is_dir():
        raise GitSyncError(f"{repo} is not a directory")
    top = _output(path, "rev-parse", "--show-toplevel")
    branch = _output(path, "symbolic-ref", "--short", "HEAD")
    if _git(path, "rev-parse", "-q", "--verify", "MERGE_HEAD").returncode == 0:
        raise GitSyncError(
            "A merge is already in progress in this checkout. Resolve its conflicts "
            "(`git diff --name-only --diff-filter=U`), `git add` the files and "
            "`git commit --no-edit`, then call git_sync again."
        )
    report = [f"Checkout {top}, branch {branch}, syncing with {ref}."]
    dirty = _lines(path, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        report.append(f"{len(dirty)} uncommitted change(s) in the working tree, left as they are.")
    attempt = 0
    while True:
        attempt += 1
        _fetch_and_merge(path, remote, remote_branch, branch, report)
        push = _git(path, "push", remote, "HEAD")
        error = (push.stderr or push.stdout).strip()
        if push.returncode == 0:
            last_line = error.splitlines()[-1] if error else "done"
            report.append(f"Pushed {branch} to {remote}: {last_line}")
            return "\n".join(report)
        if attempt < PUSH_ATTEMPTS and any(hint in error for hint in _REJECTED_PUSH_HINTS):
            report.append(
                f"{remote}/{branch} moved since the fetch; merging it, then pushing again."
            )
            if branch != remote_branch:
                _fetch_and_merge(path, remote, branch, branch, report)
            continue
        raise GitSyncError("\n".join(report) + f"\ngit push {remote} HEAD failed:\n{error}")
