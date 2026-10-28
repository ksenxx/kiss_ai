# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The git steps behind ``/sync``: commit, fetch a remote branch, merge it, push.

:func:`sync` is the ``git_sync`` tool of
:mod:`kiss.agents.seas.sync.sync_sea`.  It runs the sequence a "commit,
pull, merge and push" prompt used to make the chat agent type by hand,
deterministically and in one call:

1. ``git add -A`` and ``git commit`` when the working tree has
   uncommitted changes (edited, deleted or new files that are not
   ignored), so they ride along on the current branch instead of
   staying behind on one machine and blocking the merge of a file the
   remote changed too; the message names their number and the host
   (:func:`commit_message`);
2. ``git fetch <remote> +refs/heads/<branch>:refs/remotes/<remote>/<branch>``
   for the ref to merge (``origin/main`` by default; the explicit
   refspec updates the tracking ref also in a single-branch clone);
3. ``git merge --no-edit <remote>/<branch>`` when the current branch is
   behind it (a fast-forward when it has no commits of its own);
4. ``git push <remote> HEAD`` — the current branch to the remote branch
   of the same name.

A merge that stops on conflicts is left in progress and reported with
the conflicted files, so the agent can resolve them, ``git add`` and
``git commit --no-edit``, then call the tool again (which fetches,
finds nothing new, and pushes).  A push the remote rejects because the
branch moved there in the meantime is retried, after merging the
remote's copy of the branch, up to :data:`PUSH_ATTEMPTS` rounds;
nothing is ever forced.

A sync sent to another machine names the repository by its remote URL
rather than by a path: the checkout lives wherever that machine keeps
it (``/home/me/kiss`` on a Linux box, ``/Users/me/work/kiss`` on a
Mac).  With a *url*, :func:`sync` checks that the checkout it was
pointed at (the machine's working directory) is a clone of that
repository and, when it is not, syncs the clone among the directories
the machine's daemon has opened (:func:`locate_checkout`) instead.
"""

from __future__ import annotations

import re
import socket
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

LISTED_FILES = 20
"""How many of the committed files the report names before ``... and N more``."""


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


def commit_message(count: int, branch: str, host: str) -> str:
    """Return the message of the commit that gathers *count* uncommitted changes.

    It names the branch and the machine (``socket.gethostname()``) so
    the history shows where a change was left behind: ``sync: commit 3
    uncommitted change(s) on main (my-laptop)``.
    """
    return f"sync: commit {count} uncommitted change(s) on {branch} ({host})"


def _listed(names: list[str]) -> str:
    """Return *names* joined for a report line, at most :data:`LISTED_FILES` of them."""
    shown = ", ".join(names[:LISTED_FILES])
    rest = len(names) - LISTED_FILES
    return f"{shown} ... and {rest} more" if rest > 0 else shown


def _status_paths(top: Path) -> list[str]:
    """Return the paths ``git status`` shows as changed or untracked in the whole tree.

    ``--untracked-files=all`` names every new file, whatever
    ``status.showUntrackedFiles`` is set to; a rename's second record
    (the old name) is skipped.
    """
    records = _paths(top, "status", "--porcelain", "--untracked-files=all")
    paths: list[str] = []
    skip = False
    for record in records:
        if skip:
            skip = False
            continue
        paths.append(record[3:])
        skip = "R" in record[:2] or "C" in record[:2]
    return paths


def _commit_changes(top: Path, branch: str) -> str | None:
    """Commit the uncommitted changes of the checkout whose top-level directory is *top*.

    Every change ``git status`` shows is committed on *branch* with
    ``git add -A``: edited, deleted and renamed tracked files and new
    files that are not ignored (what the task-worktree merge commits
    too).  Run at the top so that the ``git diff`` checks cover the
    whole tree like ``status`` and ``add -A`` do, whatever
    ``diff.relative`` says.  Returns the report line (a second line
    names the changes ``git add -A`` could not stage, edits inside a
    submodule, which stay as they are), or ``None`` when the tree is
    clean.

    Raises:
        GitSyncError: When the tree has unmerged paths (a conflict left
            from an earlier merge or a failed stash pop: ``git add -A``
            would mark them resolved, markers and all), or when ``git
            commit`` fails (no committer identity, a pre-commit hook).
    """
    unmerged = _paths(top, "diff", "--name-only", "--diff-filter=U")
    if unmerged:
        raise GitSyncError(
            f"{len(unmerged)} unmerged path(s) in the working tree, left from an earlier "
            "merge: " + _listed(unmerged) + "\nResolve them, `git add` the files and "
            "`git commit`, then call git_sync again."
        )
    if not _status_paths(top):
        return None
    _output(top, "add", "-A")
    staged = _paths(top, "diff", "--cached", "--name-only")
    line = ""
    if staged:
        _output(top, "commit", "-q", "-m", commit_message(len(staged), branch, _hostname()))
        sha = _output(top, "rev-parse", "HEAD")[:12]
        line = f"Committed {len(staged)} uncommitted change(s) on {branch} as {sha}: "
        line += _listed(staged)
    left = _status_paths(top)
    if left:
        note = (
            f"{len(left)} change(s) `git add -A` cannot stage from this checkout (edits "
            f"inside a submodule?), left as they are: {_listed(left)}"
        )
        line = f"{line}\n{note}" if line else note
    return line or None


def _toplevel(path: Path) -> str:
    """Return the top-level directory of the checkout at *path*, as git prints it.

    Only git's final newline is removed, not whitespace a directory name
    may end in: the commit step runs there (:func:`_commit_changes`).

    Raises:
        GitSyncError: When *path* is not inside a git checkout.
    """
    proc = _git(path, "rev-parse", "--show-toplevel")
    if proc.returncode != 0:
        raise GitSyncError(
            f"git rev-parse --show-toplevel failed:\n{(proc.stderr or proc.stdout).strip()}"
        )
    return proc.stdout.removesuffix("\n")


def _hostname() -> str:
    """Return this machine's host name for :func:`commit_message` (``unknown`` when it has none)."""
    try:
        return socket.gethostname() or "unknown"
    except OSError:
        return "unknown"


def _merge(repo: Path, ref: str) -> str:
    """Merge *ref* into the current branch and return the report line.

    Raises:
        GitSyncError: The merge stopped on conflicts (left in progress,
            the conflicted files listed) or did not start (for example an
            ignored file would be overwritten).
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


_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:([\\/]|$)")
"""A Windows path (``C:\\repos\\r``): a drive letter, not an scp-like host."""


def _strip_dot_git(text: str) -> str:
    """Return *text* without a trailing ``.git`` (hosted repositories accept both)."""
    return text[: -len(".git")] if text.lower().endswith(".git") else text


def _host_without_user(authority: str) -> str:
    """Return *authority* (``[user[:password]@]host[:port]``) without its user part."""
    return authority.rsplit("@", 1)[1] if "@" in authority else authority


def repo_key(url: str) -> tuple[str, str]:
    """Return ``(host, path)`` identifying the repository *url* names.

    The forms git accepts for one hosted repository compare equal:
    ``https://github.com/me/kiss.git``, ``git@github.com:me/kiss``,
    ``ssh://git@github.com/me/kiss/`` and ``HTTPS://GitHub.com/me/kiss``
    all give ``("github.com", "me/kiss")``.  The scheme, the user (and
    password), the host's case, a trailing slash and a ``.git`` suffix
    are ignored for a hosted repository; the path's case and an
    explicit port (``ssh://host:2222/r`` is ``("host:2222", "r")``)
    are kept, two repositories differing only there being two.  (On a
    plain ssh server ``host:r`` is relative to the login's home while
    ``ssh://host/r`` is absolute; they still compare equal, the hosting
    services' equivalence being the common case.)  A local path gives
    ``("", <path>)`` exactly as written but for a trailing slash
    (``/srv/r`` and ``/srv/r.git`` are two directories).
    """
    text = url.strip()
    if "://" in text:
        # A URL: ``scheme://[user[:password]@]host[:port]/path``.
        authority, _, path = text.split("://", 1)[1].partition("/")
        host = _host_without_user(authority)
        if host.startswith("["):
            # ``[::1]`` or ``[::1]:2222``: the port is after the bracket.
            end = host.find("]")
            address, port = host[: end + 1], host[end + 1 :].lstrip(":")
        else:
            address, _, port = host.partition(":")
        host = f"{address.lower()}:{port}" if port else address.lower()
        return host, _strip_dot_git(path.strip("/"))
    if _WINDOWS_DRIVE.match(text):
        return "", text.replace("\\", "/").rstrip("/")
    slash = text.find("/")
    head = text if slash == -1 else text[:slash]
    at = head.find("@")
    authority_start = at + 1 if at != -1 else 0
    if head.startswith("[", authority_start):
        colon = head.find(":", head.find("]", authority_start))
    else:
        colon = head.find(":", authority_start)
    if colon != -1:
        # The scp-like form ``[user@]host:path`` (no port in this form:
        # ``host:123/r`` is the directory ``123/r``).  Only a user before
        # the host is dropped: ``git@host:team@x/r`` has no user ``team``.
        host = head[authority_start:colon]
        path = text[colon + 1 :]
        return host.lower(), _strip_dot_git(path.strip("/"))
    # A local path, as written but for a trailing slash.
    return "", text.rstrip("/") or text


def public_url(url: str) -> str:
    """Return *url* without the credentials a URL may carry (``user:token@``).

    What a report, a message to another machine or a transcript may
    show: a remote like ``https://oauth2:TOKEN@host/me/kiss.git`` is
    shown as ``https://host/me/kiss.git``.  A URL without a scheme
    (the scp-like ``git@host:path`` form, a local path) is returned as
    is, its ``user@`` being no secret.
    """
    text = url.strip()
    if "://" not in text:
        return text
    scheme, _, rest = text.partition("://")
    head, slash, path = rest.partition("/")
    if "@" in head:
        head = head.rsplit("@", 1)[1]
    return f"{scheme}://{head}{slash}{path}"


def _remote_url(path: Path, remote: str) -> str | None:
    """Return the URL of *remote* in the checkout at *path*; ``None`` when there is none.

    ``None`` also when *path* is not a directory or not inside a git
    checkout (``git remote get-url`` fails either way).
    """
    if not path.is_dir():
        return None
    proc = _git(path, "remote", "get-url", remote)
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def known_work_dirs() -> list[str]:
    """Return the directories this daemon works in: its configured one, then the recent ones.

    The ``work_dir`` and ``recent_work_dirs`` keys of ``config.json``
    (:mod:`kiss.core.user_config`), most recent first, without
    duplicates or empty entries; the places a checkout of a repository
    is looked for when the sub-task's work dir is not one
    (:func:`locate_checkout`).
    """
    # Imported here: the git steps need no configuration of their own.
    from kiss.core.user_config import load_config, recent_work_dirs

    candidates = [str(load_config().get("work_dir") or "")]
    candidates += [str(entry.get("path") or "") for entry in recent_work_dirs()]
    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def locate_checkout(repo: str, remote: str, url: str) -> tuple[Path, str]:
    """Return the checkout of the repository at *url* to sync, and a note when it is not *repo*.

    *repo* (the run's work directory) is used when its *remote* points
    at *url* (:func:`repo_key` equality).  Otherwise the first of
    :func:`known_work_dirs` whose *remote* does is used instead, and
    the note says so: a sync sent from another machine names the
    repository, not a path, because the checkout can live anywhere on
    this machine.

    URLs are shown without their credentials (:func:`public_url`).

    Raises:
        GitSyncError: When neither *repo* nor any known work dir is a
            checkout of *url*; the message says what *repo* is and how
            to point the sync at the right checkout.
    """
    path = Path(repo)
    actual = _remote_url(path, remote)
    wanted = repo_key(url)
    if actual is not None and repo_key(actual) == wanted:
        return path, ""
    what = (
        f"{repo} is not a git checkout with a remote {remote!r}" if actual is None
        else f"{repo} is a clone of {public_url(actual)}, not of {public_url(url)}"
    )
    for candidate in known_work_dirs():
        other = Path(candidate)
        if other.resolve() == path.resolve():
            continue
        found = _remote_url(other, remote)
        if found is not None and repo_key(found) == wanted:
            return other, f"{what}; syncing the checkout of that repository at {other} instead."
    raise GitSyncError(
        f"{what}, and no directory this daemon has worked in is a checkout of "
        f"{public_url(url)} either. Open the repository's checkout in a tab on this machine "
        "once (its directory is then remembered), or send the sync with the checkout's path "
        "on this machine as the work_dir option."
    )


def sync(repo: str, ref: str = "origin/main", url: str = "") -> str:
    """Commit the changes of *repo*, fetch *ref*, merge it into the current branch and push.

    Args:
        repo: Absolute path of the git checkout (the run's work directory).
        ref: The remote branch to merge, as ``<remote>/<branch>``.
        url: The URL of the repository to sync, when known (a sync sent
            from another machine names it).  *repo* must be a clone of
            it, else the clone this daemon has worked in is synced
            instead (:func:`locate_checkout`); empty skips the check.

    Returns:
        A line-per-step report: the checkout and its remote's URL, the
        branch, the uncommitted changes committed (when there were
        any), what was fetched, whether the merge was a fast-forward, a
        merge commit or nothing to do, and what the push did.

    Raises:
        GitSyncError: With the report of the step that failed: not a
            checkout (of *url*), a detached HEAD, a merge already in
            progress or unmerged paths left from one, a commit that
            failed, a fetch or push error, or a merge that stopped on
            conflicts (left in place for the caller to resolve).  When
            the checkout synced is not *repo*, the message starts by
            naming it, so a conflict is resolved in the right place.
    """
    remote, remote_branch = _split_ref(ref)
    note = ""
    if url.strip():
        path, note = locate_checkout(repo, remote, url.strip())
    else:
        path = Path(repo)
        if not path.is_dir():
            raise GitSyncError(f"{repo} is not a directory")
    try:
        return _sync_checkout(path, ref, remote, remote_branch, [note] if note else [])
    except GitSyncError as exc:
        if not note or str(exc).startswith(note):
            raise
        raise GitSyncError(
            f"{note}\n{exc}\n(The files above are relative to {path}, not to the work dir: "
            f"edit them there and run git with `-C {path}`.)"
        ) from None


def _sync_checkout(
    path: Path, ref: str, remote: str, remote_branch: str, report: list[str],
) -> str:
    """Commit, fetch, merge and push in the checkout at *path*; the steps of :func:`sync`.

    Args:
        path: The checkout (an existing directory).
        ref: The ``<remote>/<branch>`` to merge, already split into
            *remote* and *remote_branch*.
        remote: The remote to fetch from and push to.
        remote_branch: The branch of *remote* to merge.
        report: The report lines so far (a note on how the checkout was
            found, or nothing); the step lines are appended to it.

    Returns:
        The report, one line per step.

    Raises:
        GitSyncError: As :func:`sync`.
    """
    top = _toplevel(path)
    branch = _output(path, "symbolic-ref", "--short", "HEAD")
    if _git(path, "rev-parse", "-q", "--verify", "MERGE_HEAD").returncode == 0:
        raise GitSyncError(
            "A merge is already in progress in this checkout. Resolve its conflicts "
            "(`git diff --name-only --diff-filter=U`), `git add` the files and "
            "`git commit --no-edit`, then call git_sync again."
        )
    where = _remote_url(path, remote)
    shown = public_url(where) if where else "no URL"
    report.append(f"Checkout {top}, branch {branch}, syncing with {ref} ({remote} = {shown}).")
    committed = _commit_changes(Path(top), branch)
    if committed:
        report.append(committed)

    def fetch_and_merge(name: str) -> None:
        # A merge that fails after the commit says so first: the agent
        # (and the summary) must know the changes are on the branch now.
        try:
            _fetch_and_merge(path, remote, name, branch, report)
        except GitSyncError as exc:
            if committed:
                raise GitSyncError(f"{committed}\n{exc}") from None
            raise

    attempt = 0
    while True:
        attempt += 1
        fetch_and_merge(remote_branch)
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
                fetch_and_merge(branch)
            continue
        raise GitSyncError("\n".join(report) + f"\ngit push {remote} HEAD failed:\n{error}")
