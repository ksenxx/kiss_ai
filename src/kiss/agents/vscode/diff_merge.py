"""File scanning and git diff/merge utilities."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from kiss.core import config as config_module

logger = logging.getLogger(__name__)


def _log_exc() -> None:
    logger.debug("Exception caught", exc_info=True)


def _load_gitignore_dirs(work_dir: str) -> set[str]:
    """Load directory names and paths to skip from .gitignore.

    Parses .gitignore for entries without glob characters and returns
    them as a set.  Entries may be simple names (e.g. ``node_modules``)
    or paths (e.g. ``src/generated``).  Always includes ``.git``.

    Args:
        work_dir: Repository root containing .gitignore.

    Returns:
        Set of directory names/paths to skip during file scanning.
    """
    skip = {".git"}
    try:
        gitignore = Path(work_dir) / ".gitignore"
        for raw_line in gitignore.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            # Strip trailing slash (directory marker)
            name = line.rstrip("/")
            # Only use names/paths — skip glob patterns
            if "*" in name or "?" in name:
                continue
            skip.add(name)
    except OSError:
        _log_exc()
    return skip


def _scan_files(work_dir: str) -> list[str]:
    paths: list[str] = []
    skip = _load_gitignore_dirs(work_dir)
    wd = Path(work_dir)
    try:
        for root, dirs, files in wd.walk():
            rel_root = root.relative_to(wd)
            if len(rel_root.parts) - 1 > 3:
                dirs.clear()
                continue
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in skip
                and not d.startswith(".")
                and str(rel_root / d) not in skip
            )
            for name in sorted(files):
                paths.append(str(rel_root / name))
                if len(paths) >= 2000:
                    return paths
            for d in dirs:
                paths.append(str(rel_root / d) + "/")
    except OSError:  # pragma: no cover — Path.walk swallows OSErrors internally
        _log_exc()
    return paths


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command with captured text output.

    Args:
        cwd: Working directory for the git command.
        *args: Git sub-command and arguments.

    Returns:
        CompletedProcess with stdout/stderr as strings.
    """
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)


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


def _parse_diff_hunks(work_dir: str) -> dict[str, list[tuple[int, int, int, int]]]:
    result = _git(work_dir, "diff", "-U0", "HEAD", "--no-color")
    hunks: dict[str, list[tuple[int, int, int, int]]] = {}
    current_file = ""
    for line in result.stdout.split("\n"):
        dm = re.match(r"^diff --git a/.* b/(.*)", line)
        if dm:
            current_file = dm.group(1)
            continue
        hunk = _parse_hunk_line(line)
        if hunk and current_file:
            hunks.setdefault(current_file, []).append(hunk)
    return hunks


def _capture_untracked(work_dir: str) -> set[str]:
    result = _git(work_dir, "ls-files", "--others", "--exclude-standard")
    return {line.strip() for line in result.stdout.split("\n") if line.strip()}


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
            _log_exc()
    return result


def _untracked_base_dir() -> Path:
    """Return the directory for storing untracked file base copies.

    Uses ``{cwd}/.kiss.artifacts/merge_dir/untracked-base/`` so copies
    live per-project alongside other merge artifacts.

    Returns:
        Path to the untracked-base directory.
    """
    return _merge_data_dir() / "untracked-base"


def _merge_data_dir() -> Path:
    """Return the per-project directory for merge state files.

    Uses ``{cwd}/.kiss.artifacts/merge_dir/`` so merge-temp,
    untracked-base, and pending-merge.json live per-project alongside
    other artifacts.

    Returns:
        Path to the merge data directory.
    """
    artifact_dir = Path(config_module.DEFAULT_CONFIG.agent.artifact_dir)
    return artifact_dir.parent.parent / "merge_dir"


def _save_untracked_base(
    work_dir: str, untracked: set[str]
) -> None:
    """Save copies of untracked files before a task runs.

    These copies serve as the "base" for merge-view diffs when an agent
    modifies a pre-existing untracked file.

    Args:
        work_dir: Repository root.
        untracked: Set of untracked file paths (relative to work_dir).
    """
    base_dir = _untracked_base_dir()
    if base_dir.exists():
        shutil.rmtree(base_dir)
    for fname in untracked:
        fpath = Path(work_dir) / fname
        try:
            if not fpath.is_file() or fpath.stat().st_size > 2_000_000:  # pragma: no cover
                continue
            dest = base_dir / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fpath, dest)
        except OSError:
            _log_exc()


def _cleanup_merge_data(data_dir: str) -> None:
    """Remove the entire merge data directory after merge completes.

    Args:
        data_dir: Merge data directory to remove.
    """
    d = Path(data_dir)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def _diff_files(base_path: str, current_path: str) -> list[tuple[int, int, int, int]]:
    """Compute diff hunks between two files using diff -U0.

    Args:
        base_path: Path to the base (pre-task) file.
        current_path: Path to the current (post-task) file.

    Returns:
        List of (base_start, base_count, current_start, current_count) tuples.
    """
    result = subprocess.run(
        ["diff", "-U0", base_path, current_path],
        capture_output=True,
        text=True,
    )
    return [h for line in result.stdout.split("\n") if (h := _parse_hunk_line(line))]


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
    return {"bs": bs - 1, "bc": bc, "cs": cs if cc == 0 else cs - 1, "cc": cc}


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
        line_count = len(fpath.read_text().splitlines())
        return [{"bs": 0, "bc": 0, "cs": 0, "cc": line_count}] if line_count else []
    except (OSError, UnicodeDecodeError):
        _log_exc()
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


def _prepare_merge_view(
    work_dir: str,
    data_dir: str,
    pre_hunks: dict[str, list[tuple[int, int, int, int]]],
    pre_untracked: set[str],
    pre_file_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    post_hunks = _parse_diff_hunks(work_dir)
    ub_dir = _untracked_base_dir()
    file_hunks: dict[str, list[dict[str, int]]] = {}

    def _file_changed(fname: str) -> bool:
        """Return True if the agent actually modified this file."""
        if pre_file_hashes is None or fname not in pre_file_hashes:
            return True
        try:
            cur = hashlib.md5((Path(work_dir) / fname).read_bytes()).hexdigest()
        except OSError:
            _log_exc()
            return False
        return cur != pre_file_hashes[fname]

    for fname, hunks in post_hunks.items():
        if not _file_changed(fname):
            continue
        filtered = _agent_file_hunks(work_dir, fname, ub_dir, pre_hunks, hunks)
        if filtered:  # pragma: no branch – changed files always produce hunks
            file_hunks[fname] = filtered
    new_files = _capture_untracked(work_dir) - pre_untracked
    for fname in new_files:
        filtered = _file_as_new_hunks(Path(work_dir) / fname)
        if filtered:
            file_hunks[fname] = filtered
    # Detect modified pre-existing untracked files
    if pre_file_hashes:
        for fname in pre_untracked:
            if fname in file_hunks or fname not in pre_file_hashes:
                continue
            if not _file_changed(fname):
                continue
            filtered = _agent_file_hunks(work_dir, fname, ub_dir, pre_hunks)
            if filtered:
                file_hunks[fname] = filtered
    if not file_hunks:
        return {"error": "No changes"}
    merge_dir = Path(data_dir) / "merge-temp"
    if merge_dir.exists():
        shutil.rmtree(merge_dir)
    manifest_files: list[dict[str, Any]] = []
    for fname, fh in file_hunks.items():
        current_path = Path(work_dir) / fname
        if not current_path.is_file():
            continue
        base_path = merge_dir / fname
        base_path.parent.mkdir(parents=True, exist_ok=True)
        # For untracked files, use saved pre-task copy as base if available
        saved_base = ub_dir / fname
        if saved_base.is_file():
            shutil.copy2(saved_base, base_path)
        else:
            base_result = _git(work_dir, "show", f"HEAD:{fname}")
            base_path.write_text(
                base_result.stdout if base_result.returncode == 0 else ""
            )
        manifest_files.append(
            {
                "name": fname,
                "base": str(base_path),
                "current": str(current_path),
                "hunks": fh,
            }
        )
    if not manifest_files:
        return {"error": "No changes"}
    manifest = Path(data_dir) / "pending-merge.json"
    manifest.write_text(
        json.dumps(
            {
                "branch": "HEAD",
                "files": manifest_files,
            }
        )
    )
    total_hunks = sum(len(f["hunks"]) for f in manifest_files)
    return {"status": "opened", "count": len(manifest_files), "hunk_count": total_hunks}
