# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``rollback`` never rewrites the user's own checkout.

``rollback`` restores the whole git working tree holding the work dir and
deletes every file created since the checkpoint.  The worktree guard
refuses ``Write``/``Edit`` in a repository checkout no worktree isolates;
``rollback`` gets the same treatment, so a file the user (or another
task) created in the checkout after the checkpoint survives.  Inside the
task's worktree, in a repository whose worktree could not be created and
outside any repository the rollback still runs.

Each test drives a real ``SorcarAgent`` run against the scripted local
model server; the "user's" file is created by a Bash call between the
checkpoint and the rollback.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)


def _git(repo: Path, *args: str) -> None:
    """Run ``git -C repo args``; fail the test when it fails."""
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(path: Path) -> Path:
    """A git repository with one commit on ``main``."""
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "t@t.com")
    _git(path, "config", "user.name", "T")
    (path / "README.md").write_text("# test\n")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "initial")
    return path


def _call(tool: str, **arguments: Any) -> bytes:
    """One scripted model turn calling *tool*."""
    return tool_call_body(tool, arguments, prompt_tokens=500)


def _run(work_dir: Path, script: list[bytes]) -> list[str]:
    """Run a top-level agent in *work_dir* on *script*; return every tool result."""
    with serve(script) as (url, requests):
        SorcarAgent("rollback-guard").run(
            prompt_template="Try a change and roll it back.",
            model_name=MODEL,
            max_steps=len(script) + 2,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            work_dir=str(work_dir),
            use_memory=False,
            verbose=False,
            tool_profile="shell+edit",
        )
    return [
        str(message.get("content"))
        for message in requests[-1]["messages"]
        if message.get("role") == "tool"
    ]


def _user_edit_then_rollback(prefix: list[bytes]) -> list[bytes]:
    """*prefix*, a checkpoint, a new file created by Bash, a rollback, finish twice."""
    return [
        *prefix,
        _call("checkpoint", name="a"),
        _call("Bash", command="echo mine > user_new.txt", description="the user's edit"),
        _call("rollback", name="a"),
        finish_body("<p>done</p>"),
        finish_body("<p>done</p>"),
    ]


def test_rollback_in_the_users_checkout_is_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    results = _run(repo, _user_edit_then_rollback([]))
    assert (repo / "user_new.txt").read_text() == "mine\n"
    rollback_result = results[2]
    assert "did NOT run" in rollback_result
    assert "Error: rollback refused" in rollback_result
    assert "create_worktree()" in rollback_result


def test_rollback_inside_the_worktree_runs(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    results = _run(repo, _user_edit_then_rollback([_call("create_worktree")]))
    assert "Created worktree" in results[0]
    assert "user_new.txt" in results[3]
    assert "did NOT run" not in results[3], results[3]
    assert not list(repo.glob(".kiss-worktrees/*/user_new.txt"))
    assert not (repo / "user_new.txt").exists()


def test_rollback_in_a_repo_without_a_worktree_runs(tmp_path: Path) -> None:
    """A detached HEAD makes ``create_worktree`` fail: the task edits in place."""
    repo = _repo(tmp_path / "repo")
    _git(repo, "checkout", "--detach")
    results = _run(repo, _user_edit_then_rollback([_call("create_worktree")]))
    assert results[0].startswith("No worktree could be created"), results[0]
    assert "did NOT run" not in results[3], results[3]
    assert not (repo / "user_new.txt").exists()


def test_rollback_outside_a_repository_is_not_guarded(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    results = _run(plain, _user_edit_then_rollback([]))
    # The guard lets the call through; the tool itself reports that there
    # is no git working tree to restore.
    assert "did NOT run" not in results[2]
    assert "refused" not in results[2]
    assert (plain / "user_new.txt").exists()
