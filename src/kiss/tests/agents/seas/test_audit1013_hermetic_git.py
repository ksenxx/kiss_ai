# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The SEA tests' shared ``git`` helper ignores the developer's git config and never hangs.

Audit 2026-10-13, finding N.2: ``test_git_extract_knowledge_sea`` built its
repositories with a ``_git`` helper that inherited ``~/.gitconfig`` (so a
``commit.gpgsign = true`` there spawned gpg and its pinentry) and ran
without a ``timeout``.  Both tests below failed against that helper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kiss.tests.agents.seas.conftest import git
from kiss.tests.conftest import posix_only


def test_global_git_config_does_not_reach_the_test_repositories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A global ``commit.gpgsign`` (and identity) is invisible to the helper's commits."""
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(
        "[user]\n\temail = global@example.com\n\tname = Global\n"
        "[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = /bin/false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.delenv("GIT_AUTHOR_EMAIL", raising=False)
    monkeypatch.delenv("GIT_COMMITTER_EMAIL", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "alice@example.com")
    git(repo, "config", "user.name", "Alice")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-q", "-m", "a")  # gpg.program=/bin/false would fail this
    assert git(repo, "log", "-1", "--format=%ae") == "alice@example.com"
    assert git(repo, "config", "--get", "commit.gpgsign", check=False) == ""
    # The per-call environment still wins (dates the extract tests skew).
    git(
        repo, "commit", "-q", "--allow-empty", "-m", "dated",
        env={"GIT_COMMITTER_DATE": "2030-01-01T00:00:00+00:00"},
    )
    # Compare the epoch seconds: newer git (2.54 on macOS, 2.56 on Windows) prints
    # the UTC offset of ``%cI`` as ``Z`` where git 2.43 prints ``+00:00``.
    assert git(repo, "log", "-1", "--format=%ct") == "1893456000"


@posix_only("GIT_EDITOR runs through sh")
def test_a_stuck_git_command_times_out(tmp_path: Path) -> None:
    """A git command that waits on a human (an editor here) fails instead of hanging."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "alice@example.com")
    git(repo, "config", "user.name", "Alice")
    with pytest.raises(subprocess.TimeoutExpired):
        git(repo, "commit", "--allow-empty", env={"GIT_EDITOR": "sh -c 'sleep 2'"}, timeout=0.5)
