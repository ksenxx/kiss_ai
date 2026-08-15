"""E2E tests: the diff_merge git helper must scrub repo-scoped GIT_* env vars.

Findings-2 #1: ``diff_merge._git`` previously inherited the full process
environment, so a ``GIT_DIR`` leaked from e.g. a git hook that launched
the agent would silently redirect every git command away from the ``cwd``
repo.  It also decoded output with strict UTF-8, so any non-UTF-8 byte in
git output raised ``UnicodeDecodeError``.  These tests fail before the
fix (env scrub + ``errors="surrogateescape"``) and pass after.
"""

import subprocess
from pathlib import Path

from kiss.server.diff_merge import _git


def _make_repo(path: Path) -> None:
    """Create a git repo with one commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-q", "--allow-empty", "-m", "init"],
        cwd=path,
        check=True,
    )


def test_git_ignores_inherited_git_dir(tmp_path: Path, monkeypatch) -> None:
    """_git must operate on the cwd repo even when GIT_DIR points elsewhere."""
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    _make_repo(repo_a)
    _make_repo(repo_b)

    monkeypatch.setenv("GIT_DIR", str(repo_b / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(repo_b))

    result = _git(str(repo_a), "rev-parse", "--show-toplevel")
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == repo_a.resolve()


def test_git_decodes_non_utf8_output_with_surrogateescape(tmp_path: Path) -> None:
    """_git must not raise UnicodeDecodeError on non-UTF-8 git output."""
    repo = tmp_path / "repo"
    _make_repo(repo)

    # Store a blob whose content contains a Latin-1 (non-UTF-8) byte;
    # ``git cat-file`` emits blob bytes verbatim, so the strict-UTF-8
    # decode used before the fix raised UnicodeDecodeError here.
    hashed = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=b"caf\xe9\n",
        capture_output=True,
        check=True,
    )
    oid = hashed.stdout.decode().strip()

    result = _git(str(repo), "cat-file", "-p", oid)
    assert result.returncode == 0, result.stderr
    # Surrogate-escaped decode round-trips the original byte.
    assert result.stdout.encode("utf-8", "surrogateescape") == b"caf\xe9\n"
