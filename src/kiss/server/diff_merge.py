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
            entry = line.rstrip("/")
            if "/" in entry:
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
            returncode=124,
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
    result = _git(work_dir, "diff", "-U0", "--no-renames", base_ref, "--no-color")
    hunks: dict[str, list[tuple[int, int, int, int]]] = {}
    current_file = ""
    for line in result.stdout.split("\n"):
        if line.startswith("diff --git "):
            header_path = _diff_header_path(line)
            if header_path is not None:
                current_file = header_path
                continue
        if current_file and line.startswith("Binary files "):
            hunks.setdefault(current_file, [])
            continue
        if current_file and line.startswith("old mode "):
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
            result[fname] = _hash_path_identity(fpath)
        except OSError:
            logger.debug("Exception caught", exc_info=True)
    return result


def _hash_path_identity(fpath: Path) -> str:
    """Return an MD5 identity hash of *fpath* (symlink-aware).

    For a symlink, hashes the link IDENTITY (its target string), not
    the bytes behind it (F4-27): retargeting the link must register
    as a change, and a change to the target's content must not.
    Regular files hash their content.

    Args:
        fpath: Path to hash.

    Returns:
        Hex MD5 digest.

    Raises:
        OSError: When the path cannot be read.
    """
    if fpath.is_symlink():
        target = os.readlink(fpath)
        return hashlib.md5(
            b"symlink\x00" + target.encode("utf-8", "surrogateescape"),
        ).hexdigest()
    return hashlib.md5(fpath.read_bytes()).hexdigest()


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
        safe = safe.replace(".", "_") or "_"
    if safe != tab_id:
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
    :mod:`kiss.server.task_runner`).  Each copy serves as the
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
    base_dir = _untracked_base_dir(tab_id)
    parent = base_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".untracked-base-staging-", dir=str(parent)),
    )
    try:
        for fname in sorted(files):
            fpath = Path(work_dir) / fname
            try:
                if fpath.is_symlink():
                    # Preserve the symlink ITSELF (F4-27): copying
                    # through the link would snapshot the target's
                    # bytes, and a later reject would then replace
                    # the user's symlink with a regular file.
                    dest = staging / fname
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(os.readlink(fpath), dest)
                    continue
                if not fpath.is_file() or fpath.stat().st_size > 2_000_000:  # pragma: no cover
                    continue
                dest = staging / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fpath, dest)
            except OSError:
                logger.debug("Exception caught", exc_info=True)
        if base_dir.exists():
            old = base_dir.with_name(base_dir.name + ".old")
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)
            os.replace(base_dir, old)
            try:
                os.replace(staging, base_dir)
            except OSError:
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


def _artifact_path(root: Path, fname: str) -> Path:
    """Return a collision-safe artifact path for *fname* under *root*.

    Normally ``root / fname``.  When the agent replaced a tracked file
    with a directory (or vice versa, F4-26), the deleted file's
    artifact and the new descendants' artifacts collide (``root/node``
    cannot be both a file and a directory); the loser falls back to a
    flat hashed name under ``root/.flat/``.  The manifest stores the
    absolute path, so consumers are location-agnostic.

    Args:
        root: Artifact root directory (merge-temp or .deleted).
        fname: Relative workspace path of the reviewed file.

    Returns:
        A path whose parent directory exists and which is not a
        directory itself.
    """
    p = root / fname
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_dir():
            raise FileExistsError(str(p))
    except (FileExistsError, NotADirectoryError):
        digest = hashlib.md5(
            fname.encode("utf-8", "surrogatepass"),
        ).hexdigest()
        p = root / ".flat" / digest
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


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
    base_path = _artifact_path(merge_dir, fname)
    saved_base = ub_dir / fname
    if saved_base.is_symlink():
        # Preserve symlink identity in the base copy (F4-27).
        if base_path.is_symlink() or base_path.exists():
            base_path.unlink()
        os.symlink(os.readlink(saved_base), base_path)
    elif saved_base.is_file():
        shutil.copy2(saved_base, base_path)
    else:
        bin_result = _git_bytes(work_dir, "show", f"{base_ref}:{fname}")
        base_path.write_bytes(
            bin_result.stdout if bin_result.returncode == 0 else b"",
        )
    return base_path


def _base_exec_state(
    ub_dir: Path, base_modes: dict[str, str], fname: str,
) -> bool | None:
    """Return the pre-task executable state of *fname*, or ``None``.

    The reject path must restore the file's PRE-TASK mode: set the
    exec bits when the base was executable, clear them when it was
    not.  The pre-task truth is, in precedence order:

    1. The saved pre-task base copy in *ub_dir* (``shutil.copy2``
       preserves the mode) — this covers the non-worktree flow's
       pre-task-dirty and untracked files, whose git mode (if any)
       may not match what the user actually had on disk.
    2. The git mode at the merge's base ref: ``100755`` → executable,
       ``100644`` → not executable.

    Args:
        ub_dir: Directory holding saved pre-task base copies.
        base_modes: Git modes at the base ref, from :func:`_base_modes`.
        fname: Relative path of the reviewed file.

    Returns:
        ``True``/``False`` when the pre-task mode is known, ``None``
        when it is not (e.g. an agent-created file) — the reject path
        then leaves the on-disk mode alone.
    """
    saved = ub_dir / fname
    try:
        if saved.is_file():
            return bool(saved.stat().st_mode & 0o100)
    except OSError:  # pragma: no cover — unreadable saved copy
        return None
    mode = base_modes.get(fname)
    if mode == "100755":
        return True
    if mode == "100644":
        return False
    return None


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
            cur = _hash_path_identity(Path(work_dir) / fname)
        except OSError:
            return True
        return cur != pre_file_hashes[fname]

    binary_files: set[str] = set()
    for fname, hunks in post_hunks.items():
        if not _file_changed(fname):
            continue
        fpath = Path(work_dir) / fname
        # A tracked file replaced by a DIRECTORY is deliberately NOT
        # skipped (F4-26): git reports the file's deletion in the
        # diff, and hiding it would make the deletion invisible and
        # unrejectable.  It flows through like a deleted file (a
        # ``.deleted`` placeholder becomes the "current" side); real
        # submodule paths are filtered below via their 160000 base
        # mode.
        if not hunks:
            binary_files.add(fname)
            continue
        filtered = _agent_file_hunks(work_dir, fname, ub_dir, pre_hunks, hunks)
        if filtered:  # pragma: no branch – changed files always produce hunks
            file_hunks[fname] = filtered
        elif fpath.is_file() and _is_binary_file(fpath):
            binary_files.add(fname)
    new_files = _capture_untracked(work_dir) - pre_untracked
    created_files: set[str] = set()
    for fname in new_files:
        fpath = Path(work_dir) / fname
        if fpath.is_symlink() and not fpath.is_file():
            # A new broken or directory-target symlink (F4-28):
            # ``is_file()`` follows the link and returns False, so
            # without this branch the path is dropped from review
            # entirely even though git reports it untracked.
            binary_files.add(fname)
            created_files.add(fname)
            continue
        if fpath.is_file() and _is_binary_file(fpath):
            binary_files.add(fname)
            created_files.add(fname)
            continue
        filtered = _file_as_new_hunks(fpath)
        if filtered:
            file_hunks[fname] = filtered
            created_files.add(fname)
        elif fpath.is_file():
            binary_files.add(fname)
            created_files.add(fname)
    if pre_file_hashes:
        for fname in pre_untracked:
            if fname in file_hunks or fname in binary_files:
                continue
            if fname not in pre_file_hashes:
                continue
            if not _file_changed(fname):
                continue
            fpath = Path(work_dir) / fname
            if (ub_dir / fname).is_symlink():
                # The pre-task base is a SYMLINK (F4-27 residual): a
                # deleted or retargeted pre-task symlink must reach
                # review as a link-identity change.  Content-diffing
                # is meaningless here (the saved link is broken
                # relative to ub_dir and a deleted path has no
                # content), so route it to the binary/link path; the
                # link_targets pass below attaches the pre-task
                # target for reject to restore.
                binary_files.add(fname)
                continue
            if fpath.is_file() and _is_binary_file(fpath):
                binary_files.add(fname)
                continue
            filtered = _agent_file_hunks(work_dir, fname, ub_dir, pre_hunks)
            if filtered:
                file_hunks[fname] = filtered
    link_targets: dict[str, str] = {}
    # A saved pre-task base that is itself a symlink (F4-27) takes
    # precedence over any git blob: the user's pre-task link identity
    # (its target string) is what a reject must restore, not the git
    # baseline's target and not the target's file content.
    for fname in set(file_hunks) | set(binary_files):
        saved = ub_dir / fname
        if saved.is_symlink():
            try:
                link_targets[fname] = os.readlink(saved)
            except OSError:  # pragma: no cover — unreadable saved link
                continue
            file_hunks.pop(fname, None)
            binary_files.add(fname)
    base_modes = _base_modes(
        work_dir, base_ref, set(file_hunks) | binary_files,
    )
    for fname, mode in base_modes.items():
        if mode == "160000":
            file_hunks.pop(fname, None)
            binary_files.discard(fname)
    for fname, mode in base_modes.items():
        if mode != "120000" or fname in link_targets:
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
            deleted_placeholder = _artifact_path(
                merge_dir / ".deleted", fname,
            )
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
        if fname in created_files:
            # Agent-created file (F4-25): its "base" is a synthetic
            # empty file that never existed pre-task, so a full
            # reject must REMOVE the path, not write an empty file.
            text_entry["created"] = True
        exec_state = _base_exec_state(ub_dir, base_modes, fname)
        if exec_state is not None:
            text_entry["exec"] = exec_state
        manifest_files.append(text_entry)
    for fname in sorted(binary_files):
        target_path = Path(work_dir) / fname
        current_path = target_path
        if not current_path.is_file():
            deleted_placeholder = _artifact_path(
                merge_dir / ".deleted", fname,
            )
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
        if fname in created_files:
            entry["created"] = True
        if fname in link_targets:
            entry["link_target"] = link_targets[fname]
        else:
            exec_state = _base_exec_state(ub_dir, base_modes, fname)
            if exec_state is not None:
                entry["exec"] = exec_state
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
