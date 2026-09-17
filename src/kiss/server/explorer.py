# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Data providers for the remote webapp's Explorer and Source Control views.

The remote webapp's task-history panel carries a VS Code-like activity
bar with three views.  The *Explorer* view lists the workspace's files
and folders (:func:`list_directory`), and the *Source Control* view shows
the working tree's changes (:func:`git_status`) together with a graph of
the recent commits and the files each one modified (:func:`git_log`).
Every function here is a plain synchronous data provider: the daemon
(``web_server.py``) runs them on a worker thread and wraps the result in
a wire reply.  Nothing here writes to the repository.

All git output is read in its NUL-delimited form (``-z``), so paths,
subjects and ref names come through verbatim: no C-style quoting, and
no separator a legal file name could collide with.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from kiss.server.diff_merge import _git

EXPLORER_EXCLUDED_NAMES: frozenset[str] = frozenset(
    {".git", ".svn", ".hg", ".DS_Store", "Thumbs.db"}
)
"""Directory entries the Explorer never lists.

Mirrors VS Code's default ``files.exclude`` (``**/.git``, ``**/.svn``,
``**/.hg``, ``**/.DS_Store``, ``**/Thumbs.db``).
"""

DIR_LISTING_MAX_ENTRIES = 2000
"""Cap on the entries of one ``listDir`` reply (the rest is truncated)."""

GIT_LOG_DEFAULT_LIMIT = 50
"""Commits returned by ``gitLog`` when the client names no limit."""

GIT_LOG_MAX_LIMIT = 500
"""Hard cap on the commits one ``gitLog`` reply may carry."""

_CONFLICT_CODES: frozenset[str] = frozenset(
    {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
"""A full object name (SHA-1, or SHA-256 for such repositories)."""

_NAME_STATUS_RE = re.compile(r"^[ACDMRTUXB]\d{0,3}$")
"""A ``--name-status`` code: one letter, renames/copies with a score."""

_LOG_HEADER_FIELDS = 7
"""Fields per commit header: sha, parents, author, date, refs, subject."""


def list_directory(directory: Path) -> dict[str, Any]:
    """List *directory* for the Explorer view.

    Args:
        directory: The resolved, existing directory to list.

    Returns:
        ``{"entries": [...], "truncated": bool}`` where every entry is
        ``{"name", "path", "isDir"}`` (``path`` absolute; a symlinked
        folder also carries ``"real"``, its resolved target), folders
        first, each group sorted case-insensitively by name, names in
        :data:`EXPLORER_EXCLUDED_NAMES` skipped, and at most
        :data:`DIR_LISTING_MAX_ENTRIES` entries kept.

    Raises:
        OSError: When the directory cannot be read.
    """
    dirs: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for entry in directory.iterdir():
        if entry.name in EXPLORER_EXCLUDED_NAMES:
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:
            is_dir = False
        item = {"name": entry.name, "path": str(entry), "isDir": is_dir}
        if is_dir and entry.is_symlink():
            # The link's real target lets the Explorer recognise a
            # symlink cycle (a folder that is its own ancestor).
            try:
                item["real"] = str(entry.resolve())
            except (OSError, RuntimeError):
                pass
        (dirs if is_dir else files).append(item)
    key = _entry_sort_key
    entries = sorted(dirs, key=key) + sorted(files, key=key)
    truncated = len(entries) > DIR_LISTING_MAX_ENTRIES
    return {
        "entries": entries[:DIR_LISTING_MAX_ENTRIES],
        "truncated": truncated,
    }


def _entry_sort_key(item: dict[str, Any]) -> tuple[str, str]:
    """Sort key: case-insensitive name, then exact name for stability."""
    name: str = item["name"]
    return (name.casefold(), name)


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git, turning a missing/unlaunchable ``git`` into a failed result.

    :func:`kiss.server.diff_merge._git` raises ``OSError`` (e.g.
    ``FileNotFoundError``) when the executable itself cannot start; the
    providers here must always produce a reply, so that becomes an
    ordinary non-zero result whose ``stderr`` carries the reason.
    """
    try:
        return _git(cwd, *args)
    except OSError as exc:
        return subprocess.CompletedProcess(
            ["git", *args], 127, "", f"git could not be run: {exc}"
        )


def _chomp(text: str) -> str:
    """Drop exactly the line terminator git appends to a single-line value.

    Unlike ``str.strip()`` this preserves whitespace that is part of the
    value — a directory named ``repo `` (trailing space) is legal.
    """
    return text[:-1] if text.endswith("\n") else text


def repo_root(work_dir: str) -> str:
    """Return the git repository top level containing *work_dir*.

    Args:
        work_dir: A directory inside (or at the root of) a repository.

    Returns:
        The absolute top-level path, or ``""`` when *work_dir* is not
        inside a git work tree (or git failed).
    """
    result = _run_git(work_dir, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        return ""
    return _chomp(result.stdout)


def _same_dir(a: str, b: str) -> bool:
    """Whether *a* and *b* name the same directory (symlinks resolved)."""
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except (OSError, ValueError):
        return a == b


def git_worktrees(repo: str) -> list[dict[str, Any]]:
    """List every worktree of the repository *repo* belongs to.

    Args:
        repo: The top level of one worktree of the repository.

    Returns:
        One row per worktree, the main worktree first (git's order):
        ``{"path", "name", "head", "branch", "detached", "current"}``
        where ``name`` is the folder's base name, ``branch`` the short
        branch name (``""`` when detached), and ``current`` marks the
        worktree *repo* itself is.  Bare worktrees, worktrees git
        reports as ``prunable`` and worktrees whose folder is gone are
        skipped; a ``locked`` worktree is kept even while its folder
        is unavailable (a checkout on unmounted media) so its head
        still shows in the graph.  An unborn worktree (git prints an
        all-zero ``HEAD``) has ``head == ""``.  When ``git worktree
        list`` fails the list holds *repo* alone.
    """
    # -z terminates every attribute with NUL (records with two), which
    # is the only way a path containing a newline survives.
    result = _run_git(repo, "worktree", "list", "--porcelain", "-z")
    rows: list[dict[str, Any]] = []
    if result.returncode == 0:
        for record in result.stdout.split("\0\0"):
            row = _parse_worktree_record(record)
            if row is None or row.pop("bare", False) or row.pop("prunable"):
                continue
            if not row.pop("locked") and not os.path.isdir(row["path"]):
                continue
            row["current"] = _same_dir(row["path"], repo)
            rows.append(row)
    if not any(r["current"] for r in rows) and not _is_bare(repo):
        # An old git, or a repository whose worktree list does not name
        # this checkout: the current worktree is still what the view
        # asked about.  (A bare repository has no worktree to add.)
        head = _run_git(repo, "rev-parse", "--verify", "--quiet", "HEAD")
        branch = _run_git(repo, "symbolic-ref", "--short", "-q", "HEAD")
        rows.insert(
            0,
            {
                "path": repo,
                "name": os.path.basename(repo.rstrip("/\\")) or repo,
                "head": _chomp(head.stdout) if head.returncode == 0 else "",
                "branch": (
                    _chomp(branch.stdout) if branch.returncode == 0 else ""
                ),
                "detached": branch.returncode != 0,
                "current": True,
            },
        )
    return rows


def _is_unborn_head(value: str) -> bool:
    """Whether *value* is the all-zero object id ``git worktree list``
    prints as the HEAD of an unborn branch (40 digits for SHA-1
    repositories, 64 for SHA-256 ones)."""
    return bool(value) and set(value) == {"0"}


def _is_bare(repo: str) -> bool:
    """Whether *repo* is a bare repository (no working tree)."""
    result = _run_git(repo, "rev-parse", "--is-bare-repository")
    return result.returncode == 0 and _chomp(result.stdout) == "true"


def _parse_worktree_record(record: str) -> dict[str, Any] | None:
    """Parse one ``git worktree list --porcelain -z`` record.

    Every attribute (``worktree <path>``, ``HEAD <sha>``, ``branch
    <ref>``, ``detached``, ``bare``, ``locked [<reason>]``, ``prunable
    <reason>``) is one NUL-terminated line of the record.
    """
    path = ""
    head = ""
    branch = ""
    flags = {"detached": False, "bare": False, "locked": False, "prunable": False}
    for line in record.split("\0"):
        if not line:
            continue
        label, _, value = line.partition(" ")
        if label == "worktree":
            path = value
        elif label == "HEAD":
            head = "" if _is_unborn_head(value) else value
        elif label == "branch":
            branch = value[11:] if value.startswith("refs/heads/") else value
        elif label in flags:
            flags[label] = True
    if not path:
        return None
    return {
        "path": path,
        "name": os.path.basename(path.rstrip("/\\")) or path,
        "head": head,
        "branch": branch,
        **flags,
    }


def git_status(work_dir: str) -> dict[str, Any]:
    """Describe the working-tree changes of the repository at *work_dir*.

    Every worktree of the repository is reported, not just the one
    *work_dir* is in: a worktree task's edits live in its own checkout
    under ``.kiss-worktrees/`` and would otherwise be invisible.

    Args:
        work_dir: A directory inside the repository.

    Returns:
        On success ``{"repo", "branch", "changes": [...], "worktrees":
        [...]}``.  ``changes`` are the CURRENT worktree's changes
        (the one containing *work_dir*): each change is ``{"path",
        "absPath", "status", "group"}`` plus ``"origPath"`` for
        renames/copies.  ``status`` is a VS Code style letter (``M``
        modified, ``A`` added, ``D`` deleted, ``R`` renamed, ``C``
        copied, ``U`` untracked, ``!`` conflict, ``T`` type change)
        and ``group`` is ``"merge"``, ``"staged"`` or ``"changes"``.  A
        path that is both staged and modified again appears once per
        group.  ``worktrees`` lists every worktree (main first) as
        ``{"path", "name", "head", "branch", "detached", "current",
        "changes"}`` with the same change rows; a worktree whose
        status cannot be read carries ``"error"`` instead of
        ``changes``.  On failure ``{"error": <message>}``.
    """
    repo = repo_root(work_dir)
    if not repo:
        return {"error": f"Not a git repository: {work_dir}"}
    status = _run_git(
        repo, "status", "--porcelain=v1", "-z", "--untracked-files=all",
    )
    if status.returncode != 0:
        return {"error": status.stderr.strip() or "git status failed"}
    # symbolic-ref names the branch even before its first commit (an
    # unborn branch has no commit for rev-parse to resolve); a detached
    # HEAD has no symbolic ref, and rev-parse then reports "HEAD".
    branch = _run_git(repo, "symbolic-ref", "--short", "-q", "HEAD")
    if branch.returncode != 0:
        branch = _run_git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    branch_name = _chomp(branch.stdout) if branch.returncode == 0 else ""
    changes = parse_porcelain_status(status.stdout, repo)
    worktrees = git_worktrees(repo)
    for wt in worktrees:
        if wt["current"]:
            wt["changes"] = changes
            continue
        other = _run_git(
            wt["path"], "status", "--porcelain=v1", "-z",
            "--untracked-files=all",
        )
        if other.returncode != 0:
            wt["error"] = other.stderr.strip() or "git status failed"
            continue
        wt["changes"] = parse_porcelain_status(other.stdout, wt["path"])
    return {
        "repo": repo,
        "branch": branch_name,
        "changes": changes,
        "worktrees": worktrees,
    }


def parse_porcelain_status(text: str, repo: str) -> list[dict[str, Any]]:
    """Parse ``git status --porcelain=v1 -z`` output into change rows.

    Args:
        text: The NUL-separated porcelain output.
        repo: Absolute repository root used to build ``absPath``.

    Returns:
        Change rows as described by :func:`git_status`, in git's order,
        merge conflicts first within the same entry.
    """
    tokens = text.split("\0")
    changes: list[dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        if len(token) < 4:
            continue
        code = token[:2]
        path = token[3:]
        orig_path = ""
        if code[0] in "RC" or code[1] in "RC":
            # A rename/copy entry is followed by the original path as
            # its own NUL-terminated token.
            if i < len(tokens):
                orig_path = tokens[i]
                i += 1
        abs_path = str(Path(repo) / path)
        if code in _CONFLICT_CODES:
            changes.append(
                _change_row(path, abs_path, "!", "merge", orig_path)
            )
            continue
        if code == "??":
            changes.append(_change_row(path, abs_path, "U", "changes", ""))
            continue
        index_code, tree_code = code[0], code[1]
        # Only the side that IS the rename/copy carries the original
        # path: for ``RM`` the staged row is the rename, the unstaged
        # row is a plain modification of the new name.
        if index_code != " ":
            changes.append(
                _change_row(
                    path, abs_path, index_code, "staged",
                    orig_path if index_code in "RC" else "",
                )
            )
        if tree_code != " ":
            changes.append(
                _change_row(
                    path, abs_path, tree_code, "changes",
                    orig_path if tree_code in "RC" else "",
                )
            )
    return changes


def _change_row(
    path: str, abs_path: str, status: str, group: str, orig_path: str,
) -> dict[str, Any]:
    """Build one change row of a ``gitStatus`` reply."""
    row: dict[str, Any] = {
        "path": path,
        "absPath": abs_path,
        "status": status,
        "group": group,
    }
    if orig_path:
        row["origPath"] = orig_path
    return row


def git_log(
    work_dir: str,
    limit: int = GIT_LOG_DEFAULT_LIMIT,
    legacy_merge_diffs: bool = False,
) -> dict[str, Any]:
    """Describe the recent commits of the repository at *work_dir*.

    Args:
        work_dir: A directory inside the repository.
        limit: Maximum number of commits (clamped to
            ``1..GIT_LOG_MAX_LIMIT``).
        legacy_merge_diffs: Skip ``--diff-merges=first-parent`` and use
            the pre-2.31 ``-m`` spelling right away (what the fallback
            does when git rejects the option); exposed so the fallback
            path can be exercised on a modern git.

    Returns:
        On success ``{"repo", "head", "commits": [...]}`` where each
        commit is ``{"sha", "shortSha", "parents", "author", "date",
        "refs", "subject", "message", "files"}`` (``message`` is the
        full commit message, ``subject`` its first line); ``files`` lists
        ``{"path", "status"}`` (``status`` the ``--name-status``
        letter, ``origPath`` added for renames/copies) and ``refs`` the
        decorations (branches, tags, ``HEAD -> branch``).  A merge
        commit lists its changes against its FIRST parent, like VS
        Code's graph does.  Commits come newest first in
        ``--date-order`` (no parent before all its children), which is
        what the client's lane graph relies on.  The log starts from
        the HEAD of EVERY worktree of the repository (``worktrees``
        lists them as ``{"path", "name", "head", "branch",
        "detached", "current"}``), so a worktree task's commits show
        in the graph beside the main checkout's.  A repository without
        commits yields an empty list.  On failure ``{"error": <message>}``.
    """
    repo = repo_root(work_dir)
    if not repo:
        return {"error": f"Not a git repository: {work_dir}"}
    limit = max(1, min(int(limit), GIT_LOG_MAX_LIMIT))
    head = _run_git(repo, "rev-parse", "--verify", "--quiet", "HEAD")
    worktrees = git_worktrees(repo)
    if head.returncode != 0 and (head.returncode != 1 or head.stderr.strip()):
        # --verify --quiet exits 1 silently when HEAD names nothing (an
        # unborn branch, handled below).  Any other failure is real.
        return {"error": head.stderr.strip() or "git rev-parse HEAD failed"}
    head_sha = _chomp(head.stdout) if head.returncode == 0 else ""
    # Every worktree head is a start of the walk -- the current one
    # first.  An unborn worktree (head "") contributes nothing; when
    # all of them are unborn the graph is empty.
    starts = [head_sha] if head_sha else []
    for wt in worktrees:
        if wt["head"] and wt["head"] not in starts:
            starts.append(wt["head"])
    if not starts:
        return {"repo": repo, "head": "", "commits": [], "worktrees": worktrees}
    result = _log_from(repo, starts, limit, legacy_merge_diffs)
    if result.returncode != 0:
        return {"error": result.stderr.strip() or "git log failed"}
    commits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for commit in parse_git_log(result.stdout):
        if commit["sha"] in seen:
            continue
        seen.add(commit["sha"])
        commits.append(commit)
    # ``--max-count`` is a global cap: a worktree head far behind the
    # others can be cut off although it was a start.  The graph promises
    # every worktree's head, so a missing one is fetched on its own and
    # appended (older than everything shown, so the newest-first order
    # holds).
    for sha in starts:
        if sha in seen:
            continue
        one = _log_from(repo, [sha], 1, legacy_merge_diffs)
        if one.returncode != 0:
            continue
        for commit in parse_git_log(one.stdout):
            if commit["sha"] not in seen:
                seen.add(commit["sha"])
                commits.append(commit)
    return {
        "repo": repo,
        "head": head_sha,
        "commits": commits,
        "worktrees": worktrees,
    }


def _log_from(
    repo: str, starts: list[str], limit: int, legacy_merge_diffs: bool,
) -> subprocess.CompletedProcess[str]:
    """Run the ``git log`` :func:`git_log` parses, walking from *starts*."""
    fmt = "%x00".join(("%H", "%P", "%an", "%aI", "%D", "%s", "%B"))
    base_args = (
        "log",
        "-z",
        "--date-order",
        "--no-show-signature",
        f"--max-count={limit}",
        "--name-status",
        f"--format={fmt}",
    )
    revs = (*starts, "--")
    # --diff-merges=first-parent (git >= 2.31) makes a merge list its
    # changes against its first parent.  Older gits reject the option;
    # their ``-m`` prints a merge once per parent (the first-parent
    # diff first), which the sha de-duplication in git_log folds back
    # into one row carrying the first-parent files.
    result = None
    if not legacy_merge_diffs:
        result = _run_git(
            repo, *base_args, "--diff-merges=first-parent", *revs,
        )
    if result is None or (
        result.returncode != 0 and "diff-merges" in result.stderr
    ):
        result = _run_git(repo, *base_args, "-m", *revs)
    return result


def parse_git_log(text: str) -> list[dict[str, Any]]:
    """Parse the ``git log -z --name-status`` stream produced by :func:`git_log`.

    The stream is a flat sequence of NUL-separated tokens.  Each commit
    starts with seven header tokens (``sha``, ``parents``, ``author``,
    ``date``, ``refs``, ``subject``, ``message``); ``-z`` then terminates the header
    with a NUL and, when the commit touched files, git emits a newline
    plus ``STATUS`` / ``path`` token pairs (``R100`` / ``old`` / ``new``
    triples for renames and copies).  A token that is not a status code
    starts the next commit's header, so no separator ever has to be
    found inside a path or subject.

    Args:
        text: The raw ``git log`` output.

    Returns:
        The commit rows described by :func:`git_log`.
    """
    tokens = text.split("\0")
    commits: list[dict[str, Any]] = []
    i = 0
    while i + _LOG_HEADER_FIELDS <= len(tokens):
        sha, parents, author, date, refs, subject, body = tokens[i : i + 7]
        if not _SHA_RE.match(sha):
            break
        i += _LOG_HEADER_FIELDS
        files: list[dict[str, Any]] = []
        while i < len(tokens):
            status = tokens[i].lstrip("\n")
            if not _NAME_STATUS_RE.match(status):
                break
            i += 1
            entry: dict[str, Any] = {"status": status[:1]}
            # A status must be followed by its path(s); a stream cut
            # short (empty trailing token) ends the file list instead.
            if status[0] in "RC":
                if i + 1 >= len(tokens) or not tokens[i + 1]:
                    break
                entry["origPath"] = tokens[i]
                entry["path"] = tokens[i + 1]
                i += 2
            else:
                if i >= len(tokens) or not tokens[i]:
                    break
                entry["path"] = tokens[i]
                i += 1
            files.append(entry)
        commits.append(
            {
                "sha": sha,
                "shortSha": sha[:7],
                "parents": parents.split() if parents else [],
                "author": author,
                "date": date,
                # %D joins decorations with ", "; ref names never contain
                # a space, so that separator is unambiguous (a name with
                # a comma, e.g. "topic,comma", survives).
                "refs": [r for r in refs.split(", ") if r],
                "subject": subject,
                # %B is the raw message (subject, blank line, body);
                # "Copy Commit Message" copies all of it like VS Code.
                "message": body.strip("\n"),
                "files": files,
            }
        )
    return commits


GIT_SHOW_MAX_BYTES = 4 * 1024 * 1024
"""Cap on the text one ``gitShow`` reply may carry (a huge diff is cut)."""

_REF_NAME_RE = re.compile(r"^[^\s~^:?*\[\\]+$")
"""A branch / tag name candidate: no whitespace or ref-forbidden chars."""

_REVISION_RE = re.compile(r"^[^\s\\]+$")
"""A revision the user typed (``main``, ``v1``, ``HEAD~2``, ``abc123^``)."""


def _truncate(text: str) -> tuple[str, bool]:
    """Cut *text* to :data:`GIT_SHOW_MAX_BYTES` of UTF-8; report whether it was.

    The cap bounds what goes on the wire, so it counts encoded bytes,
    not characters (a diff of emoji or CJK text is up to four bytes a
    character).  Git output is decoded with ``surrogateescape`` (a
    non-UTF-8 file name survives as lone surrogates); the cut keeps
    those and only a multi-byte sequence split by the cut itself turns
    into (at most three) such surrogates.  The client shows its own
    "truncated" marker from the flag.
    """
    data = text.encode("utf-8", "surrogateescape")
    if len(data) <= GIT_SHOW_MAX_BYTES:
        return text, False
    cut = data[:GIT_SHOW_MAX_BYTES].decode("utf-8", "surrogateescape")
    return cut, True


def _valid_sha(sha: str) -> bool:
    """Whether *sha* looks like an object name or a short prefix of one."""
    return bool(re.match(r"^[0-9a-fA-F]{4,64}$", sha))


def _merge_diff_args(
    repo: str, *args: str, revs: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run ``git <args> <merge option> <revs>`` with a merge shown against its first parent.

    ``--diff-merges=first-parent`` needs git >= 2.31; an older git
    rejects it and gets ``--first-parent -m`` instead (``-m`` alone
    would print a merge's diff once per parent).  The option goes
    BEFORE *revs*, which end with ``--`` and any pathspec: after
    ``--`` git would read it as a path.
    """
    result = _run_git(repo, *args, "--diff-merges=first-parent", *revs)
    if result.returncode != 0 and "diff-merges" in result.stderr:
        result = _run_git(repo, *args, "--first-parent", "-m", *revs)
    return result


def git_show(work_dir: str, sha: str, path: str = "") -> dict[str, Any]:
    """Describe one commit: its patch, or one file's content at that commit.

    Backs the commit graph's "Open Changes" (the whole commit, or a
    single file's change when *path* is given together with
    ``file_at_rev=False``) and "Open File" (see :func:`git_file_at`).

    Args:
        work_dir: A directory inside the repository.
        sha: The commit (a full or abbreviated object name).
        path: When given, restrict the patch to this repository-relative
            path.

    Returns:
        ``{"repo", "sha", "subject", "text", "truncated"}`` where
        ``text`` is ``git show`` output (header + stat + unified diff,
        a merge against its first parent), or ``{"error": <message>}``.
    """
    repo = repo_root(work_dir)
    if not repo:
        return {"error": f"Not a git repository: {work_dir}"}
    if not _valid_sha(sha):
        return {"error": f"Not a commit id: {sha}"}
    if path.startswith("/"):
        return {"error": f"Not a repository path: {path}"}
    subject = _run_git(repo, "log", "-1", "--format=%s", sha, "--")
    if subject.returncode != 0:
        return {"error": subject.stderr.strip() or f"Unknown commit: {sha}"}
    # --literal-pathspecs: a file literally named ``:(glob)*.txt`` or
    # ``*.py`` is that one file, not a pattern (``--`` only ends option
    # parsing, it does not switch pathspec magic off); after ``--`` a
    # name such as ``--all`` is a path too.
    args = [
        "--literal-pathspecs",
        "show",
        "--no-color",
        "--no-show-signature",
        "--format=medium",
        "--stat",
        "--patch",
    ]
    revs = (sha, "--", path) if path else (sha, "--")
    result = _merge_diff_args(repo, *args, revs=revs)
    if result.returncode != 0:
        return {"error": result.stderr.strip() or "git show failed"}
    text, truncated = _truncate(result.stdout)
    return {
        "repo": repo,
        "sha": sha,
        "subject": _chomp(subject.stdout),
        "text": text,
        "truncated": truncated,
    }


def git_file_at(work_dir: str, sha: str, path: str) -> dict[str, Any]:
    """Return the content of *path* as committed in *sha*.

    Backs "Open File" on a file row of an expanded commit: VS Code opens
    the file at that revision (read-only), labelled ``name (shortSha)``.

    Args:
        work_dir: A directory inside the repository.
        sha: The commit.
        path: Repository-relative path of the file.

    Returns:
        ``{"repo", "sha", "path", "text", "truncated"}`` or
        ``{"error": <message>}`` (also for a binary blob).
    """
    repo = repo_root(work_dir)
    if not repo:
        return {"error": f"Not a git repository: {work_dir}"}
    if not _valid_sha(sha):
        return {"error": f"Not a commit id: {sha}"}
    if not path or path.startswith("/"):
        return {"error": f"Not a repository path: {path}"}
    result = _run_git(repo, "show", f"{sha}:{path}")
    if result.returncode != 0:
        return {"error": result.stderr.strip() or "git show failed"}
    if "\0" in result.stdout[:8192]:
        return {"error": f"Cannot display binary file: {path}"}
    text, truncated = _truncate(result.stdout)
    return {
        "repo": repo,
        "sha": sha,
        "path": path,
        "text": text,
        "truncated": truncated,
    }


def git_compare(work_dir: str, base: str, sha: str) -> dict[str, Any]:
    """Diff two revisions (``base...sha`` as VS Code's "Compare with..." does).

    Args:
        work_dir: A directory inside the repository.
        base: The revision to compare against (a ref name or commit).
        sha: The commit the user right-clicked.

    Returns:
        ``{"repo", "base", "sha", "text", "truncated"}`` (``text`` is a
        ``--stat`` summary followed by the unified diff) or
        ``{"error": <message>}``.
    """
    repo = repo_root(work_dir)
    if not repo:
        return {"error": f"Not a git repository: {work_dir}"}
    if not _valid_sha(sha):
        return {"error": f"Not a commit id: {sha}"}
    if not base or base.startswith("-") or not _REVISION_RE.match(base):
        return {"error": f"Not a revision: {base}"}
    check = _run_git(repo, "rev-parse", "--verify", "--quiet", base + "^{commit}")
    if check.returncode != 0:
        return {"error": f"Unknown revision: {base}"}
    result = _run_git(
        repo, "diff", "--no-color", "--stat", "--patch", base, sha, "--",
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip() or "git diff failed"}
    text, truncated = _truncate(result.stdout)
    return {
        "repo": repo,
        "base": base,
        "sha": sha,
        "text": text,
        "truncated": truncated,
    }


GIT_ACTIONS: frozenset[str] = frozenset(
    {"checkoutDetached", "createBranch", "createTag", "cherryPick"}
)
"""The commit-graph context-menu actions :func:`git_action` performs."""


def git_action(
    work_dir: str, action: str, sha: str, name: str = "", message: str = "",
) -> dict[str, Any]:
    """Perform a commit-graph context-menu action on the repository.

    Mirrors what VS Code's Git extension runs for the Source Control
    Graph menu:

    * ``checkoutDetached`` — ``git checkout --detach <sha>``;
    * ``createBranch`` — ``git checkout -b <name> <sha>`` (VS Code's
      "Create Branch..." creates the branch at the commit AND checks
      it out);
    * ``createTag`` — ``git tag <name> <sha>``, annotated with
      ``-a -m <message>`` when *message* is given;
    * ``cherryPick`` — ``git cherry-pick <sha>`` (``-m 1`` for a merge
      commit, i.e. its first-parent change).

    Args:
        work_dir: The worktree the action runs in.
        action: One of :data:`GIT_ACTIONS`.
        sha: The commit.
        name: Branch or tag name for ``createBranch`` / ``createTag``.
        message: Optional tag message for ``createTag``.

    Returns:
        ``{"repo", "action", "sha", "ok": True, "output": <git output>}``
        or ``{"error": <git's message>}``.
    """
    repo = repo_root(work_dir)
    if not repo:
        return {"error": f"Not a git repository: {work_dir}"}
    if action not in GIT_ACTIONS:
        return {"error": f"Unknown git action: {action}"}
    if not _valid_sha(sha):
        return {"error": f"Not a commit id: {sha}"}
    if action in ("createBranch", "createTag"):
        if not name or name.startswith("-") or not _REF_NAME_RE.match(name):
            return {"error": f"Not a valid name: {name!r}"}
        check = _run_git(repo, "check-ref-format", "--branch", name)
        if check.returncode != 0:
            return {"error": f"Not a valid name: {name!r}"}
    if action == "checkoutDetached":
        args = ["checkout", "--detach", sha, "--"]
    elif action == "createBranch":
        args = ["checkout", "-b", name, sha, "--"]
    elif action == "createTag":
        args = ["tag"]
        if message:
            args += ["-a", "-m", message]
        args += [name, sha]
    else:
        parents = _run_git(repo, "rev-list", "--parents", "-n", "1", sha)
        is_merge = (
            parents.returncode == 0 and len(parents.stdout.split()) > 2
        )
        args = ["cherry-pick"]
        if is_merge:
            args += ["-m", "1"]
        args += [sha]
    result = _run_git(repo, *args)
    # A hook can print anything; bound what goes on the wire.
    output, _ = _truncate((result.stdout + result.stderr).strip())
    if result.returncode != 0:
        return {"error": output or f"git {args[0]} failed"}
    return {
        "repo": repo,
        "action": action,
        "sha": sha,
        "ok": True,
        "output": output,
    }
