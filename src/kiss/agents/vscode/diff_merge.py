# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""File scanning and git diff/merge utilities."""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import _REPO_SCOPED_GIT_ENV, _unquote_git_path
from kiss.core import config as config_module

logger = logging.getLogger(__name__)


def _split_lines_keepends(text: str) -> list[str]:
    """Split *text* on ``\\n`` only, keeping the newline on each line.

    Unlike ``str.splitlines``, this does NOT split on ``\\r``, ``\\v``,
    ``\\f``, ``\\u2028`` etc., so line numbering matches git's (and the
    browser's) ``\\n``-based counting, and CRLF endings stay attached
    to their lines instead of being lost.

    Args:
        text: File content (read with newline translation disabled).

    Returns:
        List of lines, each (except possibly the last) ending in ``\\n``.
    """
    if not text:
        return []
    lines = text.split("\n")
    out = [ln + "\n" for ln in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def _read_lines_preserved(path: str | Path) -> list[str]:
    """Read a text file without newline translation and split into lines.

    Args:
        path: File to read.

    Returns:
        Lines with their original line endings (CRLF preserved).

    Raises:
        OSError: If the file cannot be read.
        UnicodeDecodeError: If the content is not decodable text.
    """
    with open(path, encoding="utf-8", newline="") as f:
        return _split_lines_keepends(f.read())


def _load_gitignore_dirs(work_dir: str) -> tuple[set[str], set[str]]:
    """Load directory names and paths to skip from .gitignore.

    Parses .gitignore for entries without glob characters.  Following
    gitignore semantics, an entry containing a slash anywhere other
    than at its end is anchored to the repository root (``/build``,
    ``src/generated``), while a bare name (``node_modules``,
    ``build/``) matches at any depth.

    Args:
        work_dir: Repository root containing .gitignore.

    Returns:
        ``(skip_names, skip_paths)`` — *skip_names* are bare directory
        names to skip at any depth (always includes ``.git``);
        *skip_paths* are root-relative directory paths to skip at
        their exact location only.
    """
    skip_names = {".git"}
    skip_paths: set[str] = set()
    try:
        gitignore = Path(work_dir) / ".gitignore"
        for raw_line in gitignore.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            if "*" in line or "?" in line:
                continue
            # A trailing "/" only marks the entry as directory-only;
            # it does not affect anchoring.
            entry = line.rstrip("/")
            if "/" in entry:
                # Anchored: match the root-relative path exactly
                # (``/build`` must NOT skip ``src/build``).
                skip_paths.add(entry.lstrip("/"))
            else:
                skip_names.add(entry)
    except OSError:
        logger.debug("Exception caught", exc_info=True)
    return skip_names, skip_paths


def _scan_files(work_dir: str) -> list[str]:
    """Scan workspace files, respecting .gitignore patterns.

    Args:
        work_dir: Repository root to scan.

    Returns:
        List of relative file and directory paths.
    """
    paths: list[str] = []
    skip_names, skip_paths = _load_gitignore_dirs(work_dir)
    wd = Path(work_dir)
    try:
        for root, dirs, files in wd.walk():
            rel_root = root.relative_to(wd)
            if len(rel_root.parts) > 10:
                dirs.clear()
                continue
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in skip_names
                and not d.startswith(".")
                and str(rel_root / d) not in skip_paths
            )
            for name in sorted(files):
                paths.append(str(rel_root / name).replace(os.sep, "/"))
                if len(paths) >= 5000:
                    return paths
            for d in dirs:
                paths.append(str(rel_root / d).replace(os.sep, "/") + "/")
                if len(paths) >= 5000:
                    return paths
    except OSError:  # pragma: no cover — Path.walk swallows OSErrors internally
        logger.debug("Exception caught", exc_info=True)
    return paths


_GIT_TIMEOUT_SECONDS: float = 30.0


def _scrubbed_git_env() -> dict[str, str]:
    """Return a copy of the environment without repo-scoped GIT_* vars.

    Strips ``GIT_DIR`` / ``GIT_WORK_TREE`` / ``GIT_INDEX_FILE`` etc.
    (see :data:`kiss.agents.sorcar.git_worktree._REPO_SCOPED_GIT_ENV`)
    so an inherited variable — e.g. from a git hook that launched this
    process — cannot redirect the command away from the ``cwd`` passed
    to :func:`_git` / :func:`_git_bytes`.  This is the same scrub
    ``git_worktree._git`` applies.

    Returns:
        Environment mapping safe to pass to a git subprocess.
    """
    return {k: v for k, v in os.environ.items() if k not in _REPO_SCOPED_GIT_ENV}


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command with captured text output.

    Always passes a 30-second timeout so a hung git process (e.g. waiting
    on a credential-helper prompt or a network remote) cannot block the
    agent thread forever (M1).  On timeout returns a non-zero
    ``CompletedProcess`` so callers don't crash.

    Repo-scoped ``GIT_*`` environment variables are scrubbed (see
    :func:`_scrubbed_git_env`) and output is decoded with
    ``errors="surrogateescape"`` because git paths are byte strings
    that may be invalid UTF-8 — a strict decode would raise
    ``UnicodeDecodeError`` out of every git call touching such a
    filename.  Both behaviors match ``git_worktree._git``.

    Args:
        cwd: Working directory for the git command.
        *args: Git sub-command and arguments.

    Returns:
        CompletedProcess with stdout/stderr as strings.
    """
    try:
        return subprocess.run(
            ["git", "-c", "core.quotepath=false", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
            cwd=cwd,
            env=_scrubbed_git_env(),
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning("git %s timed out after %ss", args, _GIT_TIMEOUT_SECONDS)
        # Synthesise a failed CompletedProcess so callers (which expect
        # a returncode/stdout/stderr triple) keep working.  Decode any
        # partial output with the same surrogateescape policy as the
        # normal path.
        stdout = (
            exc.stdout.decode("utf-8", "surrogateescape")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", "surrogateescape")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return subprocess.CompletedProcess(
            args=["git", *args],
            returncode=124,  # convention: timeout
            stdout=stdout or "",
            stderr=stderr or f"git {args[0] if args else ''} timed out",
        )


def _git_bytes(cwd: str, *args: str) -> subprocess.CompletedProcess[bytes]:
    """Run a git command and return raw bytes (for binary content).

    Applies the same repo-scoped ``GIT_*`` env scrub as :func:`_git`
    (see :func:`_scrubbed_git_env`).

    Args:
        cwd: Working directory for the git command.
        *args: Git sub-command and arguments.

    Returns:
        CompletedProcess with stdout/stderr as bytes.
    """
    try:
        return subprocess.run(
            ["git", "-c", "core.quotepath=false", *args],
            capture_output=True,
            cwd=cwd,
            env=_scrubbed_git_env(),
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=["git", *args],
            returncode=124,
            stdout=b"",
            stderr=b"timed out",
        )


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _parse_hunk_line(line: str) -> tuple[int, int, int, int] | None:
    """Parse a unified-diff @@ hunk header line.

    Returns:
        (old_start, old_count, new_start, new_count) or None if not a hunk header.
    """
    hm = _HUNK_RE.match(line)
    if not hm:
        return None
    return (
        int(hm.group(1)),
        int(hm.group(2)) if hm.group(2) is not None else 1,
        int(hm.group(3)),
        int(hm.group(4)) if hm.group(4) is not None else 1,
    )


_QUOTED_PAIR_HEADER_RE = re.compile(
    r'^diff --git "a/(?:[^"\\]|\\.)*" "b/((?:[^"\\]|\\.)*)"$',
)
_QUOTED_B_HEADER_RE = re.compile(r'^diff --git .* "b/((?:[^"\\]|\\.)*)"$')


def _diff_header_path(line: str) -> str | None:
    """Extract the (new-side) file path from a ``diff --git`` header line.

    Handles git's C-quoted form: even with ``core.quotepath=false``,
    git quotes a path containing double-quotes, backslashes, or control
    characters — e.g. ``diff --git "a/qu\\"ote.txt" "b/qu\\"ote.txt"``.
    Without quote handling, such a header matches neither plain regex,
    so the previous file's name stays current and the quoted file's
    hunks are misattributed to it.

    Args:
        line: A line beginning with ``diff --git ``.

    Returns:
        The unquoted path of the ``b/`` side, or ``None`` when the
        line cannot be parsed.
    """
    # Prefer the unambiguous symmetric form ``a/<path> b/<path>``
    # (a backreference forces both sides to be identical).  A path
    # containing the substring " b/" — e.g. a file inside a
    # directory named "x b" — makes the greedy fallback regex
    # consume up to the LAST " b/" and return a bogus suffix, so
    # the fallback is only used when the sides differ (renames).
    dm = re.match(r"^diff --git a/(.*) b/\1$", line)
    if dm:
        return dm.group(1)
    qm = _QUOTED_PAIR_HEADER_RE.match(line) or _QUOTED_B_HEADER_RE.match(line)
    if qm:
        return _unquote_git_path('"' + qm.group(1) + '"')
    dm = re.match(r"^diff --git a/.* b/(.*)", line)
    if dm:
        return dm.group(1)
    return None


def _parse_diff_hunks(
    work_dir: str,
    base_ref: str = "HEAD",
) -> dict[str, list[tuple[int, int, int, int]]]:
    """Parse ``git diff -U0 <base_ref>`` output into per-file hunk lists.

    Args:
        work_dir: Repository root directory.
        base_ref: The git ref to diff against (default ``"HEAD"``).
            Pass a baseline commit SHA to include committed changes
            between the baseline and the current working tree.

    Returns:
        Dict mapping filename to list of (old_start, old_count, new_start, new_count).
    """
    # ``--no-renames`` decomposes a rename into a full deletion of the
    # old path plus a full addition of the new path.  With rename
    # detection on (git's default), the hunks of the OLD file would be
    # keyed under the NEW name, whose blob does not exist at
    # ``base_ref`` — ``_write_base_copy``'s ``git show`` would then
    # produce an empty base while the hunks still reference old-file
    # line numbers, and the old path's deletion would be invisible in
    # the merge review entirely.
    result = _git(work_dir, "diff", "-U0", "--no-renames", base_ref, "--no-color")
    hunks: dict[str, list[tuple[int, int, int, int]]] = {}
    current_file = ""
    for line in result.stdout.split("\n"):
        if line.startswith("diff --git "):
            header_path = _diff_header_path(line)
            if header_path is not None:
                current_file = header_path
                continue
        # Detect binary files: git outputs "Binary files ... differ"
        if current_file and line.startswith("Binary files "):
            hunks.setdefault(current_file, [])
            continue
        hunk = _parse_hunk_line(line)
        if hunk and current_file:
            hunks.setdefault(current_file, []).append(hunk)
    return hunks


def _base_modes(
    work_dir: str, base_ref: str, fnames: set[str],
) -> dict[str, str]:
    """Return the git file mode at *base_ref* for each of *fnames*.

    Used by :func:`_prepare_merge_view` to detect paths that need
    non-line-based review handling: mode ``120000`` marks a symlink
    blob (whose content is the target string and whose working copy is
    read THROUGH the link), and mode ``100755`` marks an executable
    whose exec bit must be restored when a rejected deletion re-creates
    the file.

    Args:
        work_dir: Repository root directory.
        base_ref: Git ref the base content is read from.
        fnames: Candidate relative paths.

    Returns:
        Dict mapping each path present at *base_ref* to its mode
        string (e.g. ``"100644"``, ``"100755"``, ``"120000"``).
    """
    if not fnames:
        return {}
    result = _git(work_dir, "ls-tree", "-z", base_ref, "--", *sorted(fnames))
    if result.returncode != 0:
        return {}
    modes: dict[str, str] = {}
    for entry in result.stdout.split("\0"):
        if not entry:
            continue
        # "<mode> <type> <sha>\t<path>" — with ``-z`` the path is raw
        # (never C-quoted).
        meta, _, path = entry.partition("\t")
        if path:
            modes[path] = meta.split(" ", 1)[0]
    return modes


def _capture_untracked(work_dir: str) -> set[str]:
    """Return the set of untracked files in the repo.

    Args:
        work_dir: Repository root directory.

    Returns:
        Set of untracked file paths relative to work_dir.
    """
    result = _git(work_dir, "ls-files", "--others", "--exclude-standard")
    # Unquote git's C-quoted form: even with ``core.quotepath=false``,
    # paths containing double-quotes, backslashes, or control chars
    # come back quoted (e.g. ``"new\"file.txt"``) and would not match
    # any path on disk.
    #
    # Do NOT ``strip()`` the lines: a filename with a leading or
    # trailing space (legal on POSIX, and not C-quoted by git — space
    # is a printable character) would be mangled into a name that does
    # not exist on disk, silently dropping the file from the merge
    # review.  Lines are already exact ``\n``-terminated paths.
    return {
        _unquote_git_path(line)
        for line in result.stdout.split("\n")
        if line
    }


def _snapshot_files(work_dir: str, fnames: set[str]) -> dict[str, str]:
    """Return MD5 hex digests for filenames (relative to work_dir) that exist on disk.

    Args:
        work_dir: Root directory.
        fnames: Set of relative file paths to snapshot.

    Returns:
        Dict mapping filename to hex digest of its content.
    """
    result: dict[str, str] = {}
    for fname in fnames:
        fpath = Path(work_dir) / fname
        try:
            result[fname] = hashlib.md5(fpath.read_bytes()).hexdigest()
        except OSError:
            logger.debug("Exception caught", exc_info=True)
    return result


def _safe_tab_component(tab_id: str) -> str:
    """Sanitise a frontend tab id into a single safe directory name.

    The tab id arrives straight off the wire (``cmd.get("tabId")``) and
    is only coerced to ``str`` upstream.  Used verbatim as a path
    component, a hostile or malformed id such as ``"../victim"`` would
    escape the merge_dir root — and ``_cleanup_merge_data`` (run on
    every ``mergeAction all-done`` and tab close) would ``rmtree`` a
    directory OUTSIDE it.  Real tab ids are UUID-style strings and pass
    through unchanged; anything else is mapped to a collision-free safe
    name inside the parent directory.

    Args:
        tab_id: Raw frontend tab identifier (non-empty).

    Returns:
        A safe single path component, stable for a given input.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", tab_id)
    if not safe.strip("."):
        # Entirely dots ("." / "..") — never a legal directory name
        # for us; neutralise before the uniqueness suffix below.
        safe = safe.replace(".", "_") or "_"
    if safe != tab_id:
        # Keep distinct hostile ids distinct ("../x" vs ".._x").
        digest = hashlib.md5(
            tab_id.encode("utf-8", "surrogatepass"),
        ).hexdigest()[:8]
        safe = f"{safe}-{digest}"
    return safe


def _merge_data_dir(tab_id: str = "") -> Path:
    """Return the per-tab directory for merge state files.

    Uses ``{artifact_root}/merge_dir/{tab_id}/`` so merge-temp,
    untracked-base, and pending-merge.json live in the KISS artifacts
    directory, isolated per tab to prevent concurrent merge sessions
    from destroying each other's data.  The tab id is sanitised via
    :func:`_safe_tab_component` so a traversal-style id from a
    malformed client can never address a directory outside
    ``merge_dir``.

    Args:
        tab_id: Frontend tab identifier.  When non-empty, the returned
            path includes a tab-specific subdirectory.

    Returns:
        Path to the merge data directory.
    """
    base = config_module._artifact_root() / "merge_dir"
    if tab_id:
        return base / _safe_tab_component(tab_id)
    return base


def _untracked_base_dir(tab_id: str = "") -> Path:
    """Return the directory for storing pre-task base file copies.

    Uses ``{artifact_root}/merge_dir/{tab_id}/untracked-base/`` so copies
    live alongside other merge artifacts, isolated per tab.

    Only the non-worktree merge flow populates this directory (via
    :func:`_save_untracked_base`).  The worktree flow diffs against the
    baseline commit, which already captures all pre-task dirty state,
    so ``git show {baseline}:{fname}`` yields the correct base content
    and this directory remains empty.

    Args:
        tab_id: Frontend tab identifier for per-tab isolation.

    Returns:
        Path to the pre-task base-copy directory.
    """
    return _merge_data_dir(tab_id) / "untracked-base"


def _save_untracked_base(
    work_dir: str, files: set[str], tab_id: str = "",
) -> None:
    """Save copies of pre-task dirty files for later merge-view diffing.

    Despite the historical name, this is called with the union of
    untracked **and** tracked-modified files (see the ``untracked |
    set(hunks.keys())`` call site in
    :mod:`kiss.agents.vscode.task_runner`).  Each copy serves as the
    "base" against which the agent's post-task changes are diffed, so
    the merge view shows only what the agent did — on top of whatever
    dirty state the user already had.

    Only used by the non-worktree merge flow.  The worktree flow
    relies on its baseline commit instead (see
    :func:`_untracked_base_dir`).

    Args:
        work_dir: Repository root.
        files: Relative paths (to ``work_dir``) whose current on-disk
            contents should be saved as the pre-task base.
        tab_id: Frontend tab identifier for per-tab isolation.
    """
    # M5 — build the new base copy in a sibling temp directory and swap
    # it into place atomically.  This guarantees that an interrupted /
    # failing copy never leaves the merge view with a partial base set
    # (the previous good base copy is preserved).
    base_dir = _untracked_base_dir(tab_id)
    parent = base_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".untracked-base-staging-", dir=str(parent)),
    )
    try:
        # Iterate in a deterministic order so any partial-copy outcome
        # is reproducible across runs.
        for fname in sorted(files):
            fpath = Path(work_dir) / fname
            try:
                if not fpath.is_file() or fpath.stat().st_size > 2_000_000:  # pragma: no cover
                    continue
                dest = staging / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fpath, dest)
            except OSError:
                logger.debug("Exception caught", exc_info=True)
        # Atomic swap: remove the old base after the new one is fully built.
        if base_dir.exists():
            old = base_dir.with_name(base_dir.name + ".old")
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)
            os.replace(base_dir, old)
            try:
                os.replace(staging, base_dir)
            except OSError:
                # Roll back the rename of the old base on failure.
                os.replace(old, base_dir)
                raise
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.replace(staging, base_dir)
    finally:
        if staging.exists():  # pragma: no cover — only on copy failure
            shutil.rmtree(staging, ignore_errors=True)


def _cleanup_merge_data(data_dir: str) -> None:
    """Remove the entire merge data directory after merge completes.

    Args:
        data_dir: Merge data directory to remove.
    """
    d = Path(data_dir)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def _diff_files(base_path: str, current_path: str) -> list[tuple[int, int, int, int]]:
    """Compute diff hunks between two files.

    Uses Python's ``difflib.SequenceMatcher`` so no external ``diff``
    binary is required.  The output matches the ``diff -U0`` unified-diff
    hunk conventions (1-based line numbers, special handling for zero-count
    hunks on pure insertions/deletions).

    Args:
        base_path: Path to the base (pre-task) file.
        current_path: Path to the current (post-task) file.

    Returns:
        List of (base_start, base_count, current_start, current_count) tuples.
    """
    # M5 — UnicodeDecodeError is a ValueError, not OSError.  A binary
    # (or UTF-16) file would otherwise propagate and break merge for the
    # whole tab — match the more permissive guard already used in
    # ``_file_as_new_hunks``.
    # Lines are read without newline translation and split on ``\n``
    # only (see ``_read_lines_preserved``) so the hunk line numbers
    # agree with git's ``\n``-based counting and with the reject path
    # in ``web_server``, which writes the preserved endings back.
    try:
        base_lines = _read_lines_preserved(base_path)
    except (OSError, UnicodeDecodeError):
        base_lines = []
    try:
        current_lines = _read_lines_preserved(current_path)
    except (OSError, UnicodeDecodeError):
        current_lines = []
    hunks: list[tuple[int, int, int, int]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, base_lines, current_lines,
    ).get_opcodes():
        if tag == "equal":
            continue
        old_count = i2 - i1
        new_count = j2 - j1
        if old_count == 0:
            old_start = i1
        else:
            old_start = i1 + 1
        if new_count == 0:
            new_start = j1
        else:
            new_start = j1 + 1
        hunks.append((old_start, old_count, new_start, new_count))
    return hunks


def _hunk_to_dict(bs: int, bc: int, cs: int, cc: int) -> dict[str, int]:
    """Convert a raw diff hunk tuple to the merge-view dict format.

    Adjusts 1-based line numbers to 0-based for the editor.

    Args:
        bs: Base start line (1-based).
        bc: Base line count.
        cs: Current start line (1-based).
        cc: Current line count.

    Returns:
        Dict with keys bs, bc, cs, cc (0-based start lines).
    """
    return {"bs": bs if bc == 0 else bs - 1, "bc": bc, "cs": cs if cc == 0 else cs - 1, "cc": cc}


def _is_binary_file(fpath: Path) -> bool:
    """Check whether *fpath* appears to be a binary file.

    Reads the first 8 KiB and looks for null bytes — the same heuristic
    used by ``git diff``.

    Args:
        fpath: Path to the file.

    Returns:
        True when the file looks binary or cannot be read.
    """
    try:
        if not fpath.is_file():
            return False
        chunk = fpath.read_bytes()[:8192]
        return b"\x00" in chunk
    except OSError:
        return False


def _file_as_new_hunks(fpath: Path) -> list[dict[str, int]]:
    """Return a single hunk treating the entire file as newly added.

    Returns an empty list if the file doesn't exist, is too large (>2MB),
    is empty, or can't be read.

    Args:
        fpath: Absolute path to the file.

    Returns:
        List with zero or one hunk dict.
    """
    try:
        if not fpath.is_file() or fpath.stat().st_size > 2_000_000:
            return []
        # ``\n``-based counting (not ``splitlines``) so the count
        # matches git's and the browser's line numbering even for
        # content containing ``\r``/``\f``/``\u2028`` characters.
        line_count = len(_read_lines_preserved(fpath))
        return [{"bs": 0, "bc": 0, "cs": 0, "cc": line_count}] if line_count else []
    except (OSError, UnicodeDecodeError):
        logger.debug("Exception caught", exc_info=True)
        return []


def _agent_file_hunks(
    work_dir: str,
    fname: str,
    ub_dir: Path,
    pre_hunks: dict[str, list[tuple[int, int, int, int]]],
    post_file_hunks: list[tuple[int, int, int, int]] | None = None,
) -> list[dict[str, int]]:
    """Compute filtered merge-view hunk dicts for a single file.

    If a saved pre-task base copy exists in *ub_dir*, diffs against it
    to isolate the agent's changes.  Otherwise filters *post_file_hunks*
    against *pre_hunks* to exclude pre-existing changes.  If neither
    is available, treats the whole file as new.

    Args:
        work_dir: Repository root directory.
        fname: File path relative to work_dir.
        ub_dir: Directory containing saved pre-task file copies.
        pre_hunks: Pre-task diff hunks keyed by filename.
        post_file_hunks: Post-task diff hunks for this file (from git diff).
            None when the file is untracked with no git diff hunks.

    Returns:
        List of hunk dicts for the merge view.
    """
    fpath = Path(work_dir) / fname
    saved_base = ub_dir / fname
    if saved_base.is_file():
        return [_hunk_to_dict(*h) for h in _diff_files(str(saved_base), str(fpath))]
    if post_file_hunks is not None:
        pre = {(bs, bc, cc) for bs, bc, _, cc in pre_hunks.get(fname, [])}
        return [
            _hunk_to_dict(*h)
            for h in post_file_hunks
            if (h[0], h[1], h[3]) not in pre
        ]
    return _file_as_new_hunks(fpath)


def _write_base_copy(
    work_dir: str,
    merge_dir: Path,
    ub_dir: Path,
    fname: str,
    base_ref: str,
) -> Path:
    """Write the pre-task "base" copy of *fname* into *merge_dir*.

    Prefers the saved pre-task copy from *ub_dir* when one exists;
    otherwise materialises ``git show {base_ref}:{fname}``.  When git
    cannot produce the blob (e.g. a brand-new file), writes an empty
    base so the merge view diffs against nothing.

    Args:
        work_dir: Repository root directory.
        merge_dir: The merge-temp directory receiving the copy.
        ub_dir: Directory containing saved pre-task file copies.
        fname: File path relative to *work_dir*.
        base_ref: Git ref the base content is read from.

    Returns:
        The path of the written base copy inside *merge_dir*.
    """
    base_path = merge_dir / fname
    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved_base = ub_dir / fname
    if saved_base.is_file():
        shutil.copy2(saved_base, base_path)
    else:
        # Use the bytes path for text files too: a text-mode ``git
        # show`` (universal newlines) would translate CRLF to LF, so a
        # CRLF file's base copy would differ from the working file on
        # every line and the merge view would show spurious
        # whole-file hunks.  Preserving the exact committed bytes is
        # correct for both binary and text content.
        bin_result = _git_bytes(work_dir, "show", f"{base_ref}:{fname}")
        base_path.write_bytes(
            bin_result.stdout if bin_result.returncode == 0 else b"",
        )
    return base_path


def _prepare_merge_view(
    work_dir: str,
    data_dir: str,
    pre_hunks: dict[str, list[tuple[int, int, int, int]]],
    pre_untracked: set[str],
    pre_file_hashes: dict[str, str] | None = None,
    base_ref: str = "HEAD",
) -> dict[str, Any]:
    """Prepare merge-view data comparing pre-task and post-task states.

    Computes the diff between the pre-task git state and the current
    working tree, filters out pre-existing changes, and writes a
    ``pending-merge.json`` manifest with base copies and hunk data.

    Args:
        work_dir: Repository root directory.
        data_dir: Directory for merge artifacts.
        pre_hunks: Pre-task diff hunks from ``_parse_diff_hunks``.
        pre_untracked: Pre-task untracked file set.
        pre_file_hashes: Pre-task MD5 hashes for change detection.
        base_ref: Git ref to diff against (default ``"HEAD"``).
            Pass a baseline commit SHA to include changes committed
            by the agent between the baseline and the working tree.

    Returns:
        Dict with ``status``/``count``/``hunk_count`` on success,
        or ``error`` key on failure.
    """
    post_hunks = _parse_diff_hunks(work_dir, base_ref=base_ref)
    ub_dir = Path(data_dir) / "untracked-base"
    file_hunks: dict[str, list[dict[str, int]]] = {}

    def _file_changed(fname: str) -> bool:
        if pre_file_hashes is None or fname not in pre_file_hashes:
            return True
        try:
            cur = hashlib.md5((Path(work_dir) / fname).read_bytes()).hexdigest()
        except OSError:
            return True
        return cur != pre_file_hashes[fname]

    binary_files: set[str] = set()
    for fname, hunks in post_hunks.items():
        if not _file_changed(fname):
            continue
        fpath = Path(work_dir) / fname
        if fpath.is_dir():
            # A diff entry whose working-tree path is a DIRECTORY is a
            # submodule gitlink (its ``git diff`` "content" is the
            # synthetic ``Subproject commit <sha>`` line).  It cannot
            # be reviewed as file hunks: ``is_file()`` is False, so the
            # entry would be presented as a phantom DELETED file whose
            # base (``git show {base_ref}:{fname}`` fails for a
            # gitlink) and current sides are both empty — and rejecting
            # it would write regular-file content where the submodule
            # working directory lives.
            continue
        if not hunks:
            # An empty hunk list can ONLY come from git printing
            # "Binary files … differ" (mode-only changes never create
            # a post_hunks entry at all).  This covers deleted
            # binaries AND files forced binary via .gitattributes
            # whose on-disk bytes are pure text — the NUL-byte sniff
            # in _is_binary_file cannot see the attribute, so gating
            # on it here silently dropped such files from the review
            # entirely.  All of them must be reviewable as whole-file
            # binary decisions (and restorable on reject).
            binary_files.add(fname)
            continue
        filtered = _agent_file_hunks(work_dir, fname, ub_dir, pre_hunks, hunks)
        if filtered:  # pragma: no branch – changed files always produce hunks
            file_hunks[fname] = filtered
        elif fpath.is_file() and _is_binary_file(fpath):
            binary_files.add(fname)
    new_files = _capture_untracked(work_dir) - pre_untracked
    for fname in new_files:
        fpath = Path(work_dir) / fname
        if fpath.is_file() and _is_binary_file(fpath):
            binary_files.add(fname)
            continue
        filtered = _file_as_new_hunks(fpath)
        if filtered:
            file_hunks[fname] = filtered
        elif fpath.is_file():
            # A new file that yields no line hunks — EMPTY
            # (``__init__.py`` / ``.gitkeep``), oversized (>2MB), or
            # undecodable without NUL bytes — used to appear in
            # neither ``file_hunks`` nor ``binary_files``, making the
            # created file INVISIBLE in the merge review.  Present it
            # as a whole-file (binary-style) decision instead;
            # rejecting restores the empty base, consistent with the
            # existing new-file reject semantics.
            binary_files.add(fname)
    if pre_file_hashes:
        for fname in pre_untracked:
            if fname in file_hunks or fname in binary_files:
                continue
            if fname not in pre_file_hashes:
                continue
            if not _file_changed(fname):
                continue
            fpath = Path(work_dir) / fname
            if fpath.is_file() and _is_binary_file(fpath):
                binary_files.add(fname)
                continue
            filtered = _agent_file_hunks(work_dir, fname, ub_dir, pre_hunks)
            if filtered:
                file_hunks[fname] = filtered
    # Paths whose BASE blob is a symlink (the agent retargeted,
    # deleted, or replaced a tracked symlink with a regular file)
    # cannot be reviewed line-by-line: git's typechange diff emits TWO
    # entries for the same path whose hunk coordinates do not compose,
    # and the working copy is read THROUGH the link — rejecting such
    # hunks spliced the one-line blob into the followed content and
    # corrupted the path.  Review them as single whole-file decisions
    # whose ``link_target`` lets the reject path restore the symlink
    # itself.
    link_targets: dict[str, str] = {}
    base_modes = _base_modes(
        work_dir, base_ref, set(file_hunks) | binary_files,
    )
    # Submodule gitlinks (mode 160000) whose working directory was
    # REMOVED slip past the ``is_dir()`` guard above (the path no
    # longer exists on disk) and would be reviewed as a phantom
    # deleted text file — rejecting it would write a regular file
    # where the submodule directory belongs.  Drop them here.
    for fname, mode in base_modes.items():
        if mode == "160000":
            file_hunks.pop(fname, None)
            binary_files.discard(fname)
    # Executables (mode 100755) need their exec bit re-applied when a
    # rejected deletion re-creates the file — otherwise a full
    # reject-all leaves the tree dirty (old mode 100755 / new mode
    # 100644) and the restored script is no longer runnable.
    exec_files = {f for f, mode in base_modes.items() if mode == "100755"}
    for fname, mode in base_modes.items():
        if mode != "120000":
            continue
        blob = _git_bytes(work_dir, "show", f"{base_ref}:{fname}")
        if blob.returncode != 0:
            continue
        try:
            target = blob.stdout.decode()
        except UnicodeDecodeError:  # pragma: no cover — exotic target
            logger.debug("Undecodable symlink target for %s", fname)
            continue
        link_targets[fname] = target
        file_hunks.pop(fname, None)
        binary_files.add(fname)
    if not file_hunks and not binary_files:
        return {"error": "No changes"}
    merge_dir = Path(data_dir) / "merge-temp"
    if merge_dir.exists():
        shutil.rmtree(merge_dir)
    manifest_files: list[dict[str, Any]] = []
    for fname, fh in file_hunks.items():
        target_path = Path(work_dir) / fname
        current_path = target_path
        if not current_path.is_file():
            # The agent deleted a tracked file.  The merge view needs a
            # readable "current" stand-in so VS Code / the web client
            # can render the diff, but rejecting all hunks must restore
            # the file at the real workspace location (``target_path``),
            # not at the placeholder.
            deleted_dir = merge_dir / ".deleted"
            deleted_placeholder = deleted_dir / fname
            deleted_placeholder.parent.mkdir(parents=True, exist_ok=True)
            deleted_placeholder.write_text("", encoding="utf-8")
            current_path = deleted_placeholder
        base_path = _write_base_copy(
            work_dir, merge_dir, ub_dir, fname, base_ref,
        )
        text_entry: dict[str, Any] = {
            "name": fname,
            "base": str(base_path),
            "current": str(current_path),
            "target": str(target_path),
            "hunks": fh,
        }
        if fname in exec_files:
            text_entry["exec"] = True
        manifest_files.append(text_entry)
    for fname in sorted(binary_files):
        target_path = Path(work_dir) / fname
        current_path = target_path
        if not current_path.is_file():
            # The agent deleted a tracked binary file.  Like deleted
            # text files above, use an empty ``.deleted`` placeholder
            # as the visible "current" so the review UI has something
            # to render, while ``target`` keeps the real workspace
            # path so rejecting the deletion restores the file there.
            deleted_placeholder = merge_dir / ".deleted" / fname
            deleted_placeholder.parent.mkdir(parents=True, exist_ok=True)
            deleted_placeholder.write_bytes(b"")
            current_path = deleted_placeholder
        base_path = _write_base_copy(
            work_dir, merge_dir, ub_dir, fname, base_ref,
        )
        entry: dict[str, Any] = {
            "name": fname,
            "base": str(base_path),
            "current": str(current_path),
            "target": str(target_path),
            "hunks": [{"bs": 0, "bc": 0, "cs": 0, "cc": 0}],
            "binary": True,
        }
        if fname in link_targets:
            # Rejecting this entry must restore the symlink itself,
            # not write the blob's target string as file content.
            entry["link_target"] = link_targets[fname]
        if fname in exec_files:
            entry["exec"] = True
        manifest_files.append(entry)
    if not manifest_files:
        return {"error": "No changes"}
    manifest = Path(data_dir) / "pending-merge.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "branch": "HEAD",
                "files": manifest_files,
            },
        ),
        encoding="utf-8",
    )
    total_hunks = sum(len(f["hunks"]) for f in manifest_files)
    return {"status": "opened", "count": len(manifest_files), "hunk_count": total_hunks}
