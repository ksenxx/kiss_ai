"""E2E regression tests for the sorcar tools simplification (PART=tools).

* ``_bash_streaming`` must honour ``timeout_seconds`` when the shell closes
  its own stdout (``exec >log 2>&1; …``) and keeps running: EOF on the pipe
  used to start a fixed 5 s wait followed by a SIGKILL of the whole group,
  reported as ``exit code -9`` instead of the timeout message.
* The parent-repo Bash guard, now built on ``_worktree_roots``, keeps
  refusing parent-repo paths, keeps allowing worktree paths, and treats
  ``,``/``:`` as path terminators exactly like ``rewrite_parent_repo_paths``.
* ``Bash`` and ``run_commands_parallel`` share the guard + timeout-lift
  helper and therefore give the same refusal text.
* ``load_skill_content`` goes through ``parse_frontmatter``: BOM, frontmatter
  and malformed frontmatter are handled identically to discovery.
* The ``DockerTools.Edit`` Python fallback preserves CRLF line endings like
  the Perl fallback and the host ``Edit`` tool.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.agents.sorcar.skills import Skill, load_skill_content
from kiss.agents.sorcar.useful_tools import (
    UsefulTools,
    _bash_parent_repo_guard,
    rewrite_parent_repo_paths,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX shell/process-group semantics"
)


def test_bash_runs_to_completion_after_shell_closes_stdout(tmp_path: Path) -> None:
    """A shell that redirects its stdout away and keeps working is not killed early."""
    out = tmp_path / "out"
    started = time.monotonic()
    result = UsefulTools(work_dir=str(tmp_path)).Bash(
        "exec >/dev/null 2>&1; sleep 6; echo x > out", "p", timeout_seconds=30
    )
    elapsed = time.monotonic() - started
    assert result == "", result
    assert out.exists(), "the command was killed before it finished"
    assert 5.5 <= elapsed < 15, elapsed


def test_bash_times_out_at_deadline_after_shell_closes_stdout(tmp_path: Path) -> None:
    """Expiry of the deadline after EOF is a genuine timeout: group killed, timeout reported."""
    out = tmp_path / "out"
    started = time.monotonic()
    result = UsefulTools(work_dir=str(tmp_path)).Bash(
        "exec >/dev/null 2>&1; sleep 3; echo x > out", "p", timeout_seconds=1
    )
    elapsed = time.monotonic() - started
    assert result == "Error: Command execution timeout", result
    assert elapsed < 2.5, elapsed
    time.sleep(2.5)
    assert not out.exists(), "the process group survived the timeout"


def _make_worktree(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    wt = repo / ".kiss-worktrees" / "kiss_wt-live"
    wt.mkdir(parents=True)
    return repo, wt


def test_parent_repo_guard_refuses_parent_and_allows_worktree(tmp_path: Path) -> None:
    repo, wt = _make_worktree(tmp_path)
    err = _bash_parent_repo_guard(f"echo x > {repo}/README.md", str(wt))
    assert err is not None
    assert f"{repo}/README.md" in err and str(wt) in err
    assert f"Suggested command: echo x > {wt}/README.md" in err
    assert _bash_parent_repo_guard(f"echo x > {wt}/README.md", str(wt)) is None
    assert _bash_parent_repo_guard(f"echo {repo}-other/x", str(wt)) is None
    assert _bash_parent_repo_guard("echo hi", None) is None
    assert _bash_parent_repo_guard(f"echo {repo}/x", str(tmp_path / "elsewhere")) is None


def test_parent_repo_guard_terminators_match_rewrite(tmp_path: Path) -> None:
    """``,`` and ``:`` end a path for the guard exactly as for the rewrite."""
    repo, wt = _make_worktree(tmp_path)
    for cmd in (
        f"PYTHONPATH={repo}/src:$PYTHONPATH pytest",
        f"ls {repo}/a,{wt}/b",
        f"rsync {repo}: x",
    ):
        err = _bash_parent_repo_guard(cmd, str(wt))
        assert err is not None, cmd
        assert "Suggested command: " + rewrite_parent_repo_paths(cmd, str(wt)) in err
    assert _bash_parent_repo_guard(f"ls {wt}/a,{wt}/b", str(wt)) is None


def test_bash_and_run_commands_parallel_share_the_guard(tmp_path: Path) -> None:
    repo, wt = _make_worktree(tmp_path)
    tools = UsefulTools(work_dir=str(wt))
    cmd = f"rm {repo}/README.md"
    refused = tools.Bash(cmd, "p")
    assert refused.startswith("Error: command references the parent-repo path")
    report = tools.run_commands_parallel(f'["{cmd}"]')
    assert refused in report
    assert "exit=-1" in report or "exit code -1" in report or "-1" in report


def test_load_skill_content_matches_discovery_parsing(tmp_path: Path) -> None:
    skill_dir = tmp_path / "s"
    skill_dir.mkdir()
    path = skill_dir / "SKILL.md"
    path.write_bytes(b"\xef\xbb\xbf---\nname: s\ndescription: d\n---\n\nBody line\n")
    skill = Skill(name="s", description="d", path=str(path), source="t")
    content = load_skill_content(skill)
    assert '<skill_content name="s">\nBody line\n' in content
    assert "name: s" not in content
    path.write_text("---\n: [bad yaml\n---\nAfter bad block\n", encoding="utf-8")
    assert "After bad block" in load_skill_content(skill)
    assert "bad yaml" not in load_skill_content(skill)
    path.unlink()
    assert load_skill_content(skill) == f"Error: skill file is no longer readable: {path}"


def _local_bash(command: str, description: str) -> str:
    del description
    proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=60)
    return proc.stdout + proc.stderr


def test_docker_edit_python_fallback_preserves_crlf(tmp_path: Path) -> None:
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    result = DockerTools(_local_bash).Edit(str(path), "beta", "BETA")
    assert result.strip() == f"Successfully replaced 1 occurrence(s) in {path}"
    assert path.read_bytes() == b"alpha\r\nBETA\r\ngamma\r\n"


def test_docker_edit_python_fallback_matches_lf_old_string_in_crlf_file(tmp_path: Path) -> None:
    """A multi-line LF ``old_string`` edits a CRLF file like the host ``Edit`` does."""
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    result = DockerTools(_local_bash).Edit(str(path), "alpha\nbeta", "one\ntwo")
    assert result.strip() == f"Successfully replaced 1 occurrence(s) in {path}"
    assert path.read_bytes() == b"one\r\ntwo\r\ngamma\r\n"
