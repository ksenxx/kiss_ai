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

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.core.processes import run_process_group, start_process_group
from kiss.tests.agents.seas.conftest import git, hermetic_git_env, seed_repo
from kiss.tests.conftest import posix_only


@posix_only("the blocking hook is a shell script")
def test_seed_repo_ignores_global_hooks_and_signing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seeded repository is built with the helper's hermetic git, so a global
    ``core.hooksPath`` whose pre-commit hook waits for ever, the same hook in an exported
    ``GIT_TEMPLATE_DIR``, a ``commit.gpgsign``, an ``init.defaultBranch`` and an identity
    exported by the launching shell never reach it; each call keeps the helper's 60 s cap."""
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Inherited Author")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "inherited@example.test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "inherited@example.test")
    hooks = tmp_path / "template" / "hooks"
    hooks.mkdir(parents=True)
    hook = hooks / "pre-commit"
    hook.write_text("#!/bin/sh\nexec sleep 3600\n", encoding="utf-8")
    hook.chmod(0o755)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(
        f"[core]\n\thooksPath = {hooks}\n[init]\n\tdefaultBranch = trunk\n"
        "[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = /bin/false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_TEMPLATE_DIR", str(hooks.parent))
    repo = tmp_path / "repo"
    seed_repo(repo)
    assert git(repo, "branch", "--show-current") == "main"
    assert git(repo, "log", "--format=%s %ae") == "seed test@example.com"
    assert (repo / "seed.txt").read_text(encoding="utf-8") == "seed\n"
    assert git(repo, "config", "--get", "commit.gpgsign") == "false"


def test_global_git_config_does_not_reach_the_test_repositories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A global ``commit.gpgsign`` (and identity) is invisible to the helper's commits, and so
    is what the launching shell exported for git: the same signing as command-scope config,
    which outranks the repository's own, and a ``GIT_DIR`` that selects another repository.
    :func:`hermetic_git_env` hides both from the SEA's git; the session's search ceiling stays."""
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(
        "[user]\n\temail = global@example.com\n\tname = Global\n"
        "[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = /bin/false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "commit.gpgsign")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    monkeypatch.setenv("GIT_CONFIG_KEY_1", "gpg.program")
    monkeypatch.setenv("GIT_CONFIG_VALUE_1", "/bin/false")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere"))
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
    # A search ceiling is no override: the folder under it is in no repository.
    inner = repo / "inner"
    inner.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(repo), prepend=os.pathsep)
    assert git(inner, "rev-parse", "--show-toplevel", check=False) == ""
    # The same for git run with the process environment, as the SEA under test runs it.
    before = dict(os.environ)
    with pytest.MonkeyPatch.context() as env:
        hermetic_git_env(env)
        commit = ["git", "-C", str(repo), "commit", "-q", "--allow-empty", "-m", "sea"]
        run_process_group(commit, 60).check_returncode()
        assert run_process_group(["git", "-C", str(inner), "rev-parse"], 60).returncode == 128
    assert git(repo, "log", "-1", "--format=%s %ae") == "sea t@t"
    assert dict(os.environ) == before
    assert not (tmp_path / "elsewhere").exists()


def _dead(pid: int) -> bool:
    """Whether no process *pid* runs any more (gone, or a zombie init has not reaped yet)."""
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text() if sys.platform == "linux" else ""
    except (ProcessLookupError, FileNotFoundError):
        return True
    return stat.rsplit(") ", 1)[-1].startswith("Z")


class _EditorUp:
    """Profile hook of the test's thread: the helper's timeout starts once the editor runs.

    ``run_process_group`` starts git and then waits for it with the
    timeout.  Between the two, at the return of ``start_process_group``,
    the hook waits (at most 30 s) until the editor has written its pid,
    so git's own start-up on a loaded machine cannot use up the short
    timeout before it has spawned the editor the test watches.
    """

    def __init__(self, pidfile: Path) -> None:
        self.pidfile = pidfile
        self.pid: int | None = None

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        """Wait for the editor's pid when git has just been started."""
        if event != "return" or frame.f_code is not start_process_group.__code__:
            return
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.pidfile.exists() and (text := self.pidfile.read_text().strip()):
                self.pid = int(text)
                return
            time.sleep(0.02)


@posix_only("GIT_EDITOR runs through sh")
def test_a_stuck_git_command_times_out(tmp_path: Path) -> None:
    """A git command that waits on a human (an editor here) fails instead of hanging.

    The editor dies with git: the helper kills git's process group, so the
    editor's ``sleep`` is gone once the timeout has been raised (audit
    2026-11-01, T4 F4: a plain ``subprocess.run`` timeout killed git alone
    and left the editor shell and its sleep running).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "alice@example.com")
    git(repo, "config", "user.name", "Alice")
    # The editor records its pid, then sleeps on in that same process.  git runs
    # it through ``sh -c``; the pid file is a quoted positional argument of the
    # inner shell, so a tmp_path with spaces (or quotes) in it stays one word.
    pidfile = tmp_path / "editor.pid"
    editor = f"sh -c 'echo $$ > \"$1\"; exec sleep 30' sh {shlex.quote(str(pidfile))}"
    up = _EditorUp(pidfile)
    previous = sys.getprofile()
    sys.setprofile(up)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            git(repo, "commit", "--allow-empty", env={"GIT_EDITOR": editor}, timeout=1.0)
    finally:
        sys.setprofile(previous)
    assert up.pid is not None, "the editor did not start within 30 s"
    deadline = time.monotonic() + 10
    while not _dead(up.pid):
        assert time.monotonic() < deadline, f"the editor (pid {up.pid}) outlived the timed-out git"
        time.sleep(0.02)
