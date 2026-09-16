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

_LOG_HEADER_FIELDS = 6
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


def git_status(work_dir: str) -> dict[str, Any]:
    """Describe the working-tree changes of the repository at *work_dir*.

    Args:
        work_dir: A directory inside the repository.

    Returns:
        On success ``{"repo", "branch", "changes": [...]}`` where each
        change is ``{"path", "absPath", "status", "group"}`` plus
        ``"origPath"`` for renames/copies.  ``status`` is a VS Code
        style letter (``M`` modified, ``A`` added, ``D`` deleted, ``R``
        renamed, ``C`` copied, ``U`` untracked, ``!`` conflict, ``T``
        type change) and ``group`` is ``"merge"``, ``"staged"`` or
        ``"changes"``.  A path that is both staged and modified again
        appears once per group.  On failure ``{"error": <message>}``.
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
    return {
        "repo": repo,
        "branch": branch_name,
        "changes": parse_porcelain_status(status.stdout, repo),
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
        if index_code != " ":
            changes.append(
                _change_row(path, abs_path, index_code, "staged", orig_path)
            )
        if tree_code != " ":
            changes.append(
                _change_row(path, abs_path, tree_code, "changes", orig_path)
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
        "refs", "subject", "files"}``; ``files`` lists
        ``{"path", "status"}`` (``status`` the ``--name-status``
        letter, ``origPath`` added for renames/copies) and ``refs`` the
        decorations (branches, tags, ``HEAD -> branch``).  A merge
        commit lists its changes against its FIRST parent, like VS
        Code's graph does.  Commits come newest first in
        ``--date-order`` (no parent before all its children), which is
        what the client's lane graph relies on.  A repository without
        commits yields an empty list.  On failure ``{"error": <message>}``.
    """
    repo = repo_root(work_dir)
    if not repo:
        return {"error": f"Not a git repository: {work_dir}"}
    limit = max(1, min(int(limit), GIT_LOG_MAX_LIMIT))
    head = _run_git(repo, "rev-parse", "--verify", "--quiet", "HEAD")
    if head.returncode == 1 and not head.stderr.strip():
        # --verify --quiet exits 1 silently when HEAD names nothing: a
        # repository without commits yet.  Any other failure is real.
        return {"repo": repo, "head": "", "commits": []}
    if head.returncode != 0:
        return {"error": head.stderr.strip() or "git rev-parse HEAD failed"}
    fmt = "%x00".join(("%H", "%P", "%an", "%aI", "%D", "%s"))
    base_args = (
        "log",
        "-z",
        "--date-order",
        "--no-show-signature",
        f"--max-count={limit}",
        "--name-status",
        f"--format={fmt}",
    )
    # --diff-merges=first-parent (git >= 2.31) makes a merge list its
    # changes against its first parent.  Older gits reject the option;
    # their ``-m`` prints a merge once per parent (the first-parent
    # diff first), which the sha de-duplication below folds back into
    # one row carrying the first-parent files.
    result = None
    if not legacy_merge_diffs:
        result = _run_git(repo, *base_args, "--diff-merges=first-parent")
    if result is None or (
        result.returncode != 0 and "diff-merges" in result.stderr
    ):
        result = _run_git(repo, *base_args, "-m")
    if result.returncode != 0:
        return {"error": result.stderr.strip() or "git log failed"}
    commits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for commit in parse_git_log(result.stdout):
        if commit["sha"] in seen:
            continue
        seen.add(commit["sha"])
        commits.append(commit)
    return {"repo": repo, "head": _chomp(head.stdout), "commits": commits}


def parse_git_log(text: str) -> list[dict[str, Any]]:
    """Parse the ``git log -z --name-status`` stream produced by :func:`git_log`.

    The stream is a flat sequence of NUL-separated tokens.  Each commit
    starts with six header tokens (``sha``, ``parents``, ``author``,
    ``date``, ``refs``, ``subject``); ``-z`` then terminates the header
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
        sha, parents, author, date, refs, subject = tokens[i : i + 6]
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
                "files": files,
            }
        )
    return commits
