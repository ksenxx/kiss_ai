# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression: worktree directory disappears mid-task.

In ``WorktreeSorcarAgent`` the per-task git worktree directory normally
lives for the lifetime of the task.  But several real-world scenarios
cause it to vanish mid-run before the task finishes:

* a concurrent cleanup pass (``cleanup_orphans``) racing with the
  active task;
* a user-initiated discard on another tab targeting the same worktree;
* an out-of-process git operation that wipes ``.kiss-worktrees/``.

When that happens the next ``RelentlessAgent`` sub-session inherits a
``UsefulTools`` instance whose ``work_dir`` points at the missing
directory.  Without a guard, the very first ``Bash`` call inside that
sub-session raises ``FileNotFoundError: [Errno 2] No such file or
directory`` from ``subprocess.Popen(cwd=...)`` and every subsequent
Bash call dies the same way — exactly the failure pattern seen in
``~/.kiss/kiss-web-stderr.log`` ("Worktree disappeared mid-session
AGAIN" / "every Bash invocation fails immediately with [Errno 2]").

This test reproduces that failure end-to-end (real ``WorktreeSorcarAgent``
on a real on-disk git repo, only the LLM boundary stubbed) and pins the
fix: ``UsefulTools._spawn`` transparently falls back to the parent repo
root when ``self.work_dir`` no longer exists, so Bash keeps working.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.kiss_agent import KISSAgent


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t"], check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "t"], check=True
    )
    (path / "README.md").write_text("hello\n")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True
    )


def _find_tool(tools: list[Any] | None, name: str) -> Any:
    for tool in tools or []:
        if getattr(tool, "__name__", None) == name:
            return tool
    return None


def test_worktree_disappears_mid_run_bash_falls_back(tmp_path, monkeypatch):
    """End-to-end: the worktree dir is deleted between sub-sessions.

    Steps:

    1. Real on-disk git repo with one commit on ``main``.
    2. Real :class:`WorktreeSorcarAgent` — the worktree branch and
       directory are created by ``_try_setup_worktree`` for real.
    3. ``KISSAgent.run`` is stubbed at the LLM boundary so the test
       drives the agent loop without any model call:

       * On the first sub-session the stub ``shutil.rmtree``s the
         worktree directory (simulating the cleanup-race scenario)
         and returns ``is_continue: True``.
       * On the second sub-session the stub invokes the ``Bash`` tool
         from the same ``UsefulTools`` instance whose ``work_dir``
         now points at the just-deleted directory.

    The fix in ``UsefulTools._spawn`` must make the second-session
    Bash run cleanly (no ``FileNotFoundError``, no ``[Errno 2]``
    in the output) by falling back to the parent repo root.
    """
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    captured: dict[str, Any] = {
        "call_count": 0,
        "session0_work_dir": None,
        "session1_bash_output": None,
        "session1_bash_exception": None,
    }

    def stub_run(self_kiss, *args, tools=None, **kwargs):
        captured["call_count"] += 1
        call_idx = captured["call_count"] - 1
        # Pretend we used one step / no tokens / no budget so the
        # surrounding ``RelentlessAgent`` loop's accounting stays sane.
        self_kiss.step_count = 1
        self_kiss.total_tokens_used = 0
        self_kiss.budget_used = 0.0

        bash_tool = _find_tool(tools, "Bash")
        # ``Bash`` is a plain closure (grep-hint interception wrapper),
        # so reach the shared ``UsefulTools`` instance through the
        # still-bound ``Read`` tool.
        read_tool = _find_tool(tools, "Read")
        useful = getattr(read_tool, "__self__", None) if read_tool else None
        work_dir = useful.work_dir if useful else None

        if call_idx == 0:
            captured["session0_work_dir"] = work_dir
            # Simulate worktree directory vanishing mid-task.
            assert work_dir is not None
            shutil.rmtree(work_dir)
            return yaml.dump({
                "success": False,
                "is_continue": True,
                "summary": "Worktree was removed mid-session.",
            })

        # Second sub-session: ``self.work_dir`` on the shared
        # ``UsefulTools`` instance still points at the now-deleted
        # worktree.  Without the ``_spawn`` fallback fix this would
        # raise ``FileNotFoundError: [Errno 2]`` from Popen.
        try:
            output = bash_tool("pwd", "probe")
        except FileNotFoundError as exc:
            captured["session1_bash_exception"] = exc
            output = ""
        captured["session1_bash_output"] = output
        return yaml.dump({
            "success": True,
            "is_continue": False,
            "summary": f"second session ran: {output!r}",
        })

    monkeypatch.setattr(KISSAgent, "run", stub_run)

    agent = WorktreeSorcarAgent("wt-agent")
    result = agent.run(
        prompt_template="dummy prompt",
        model_name="claude-opus-4-7",
        work_dir=str(repo),
        max_sub_sessions=4,
        max_steps=5,
        max_budget=1.0,
        verbose=False,
        web_tools=False,
        is_parallel=False,
    )

    # Two sub-sessions must have been executed (the continuation
    # actually fired through the RelentlessAgent loop).
    assert captured["call_count"] == 2, (
        f"Expected 2 sessions, got {captured['call_count']}"
    )

    # First session ran inside the per-task worktree dir.
    session0_wd = captured["session0_work_dir"]
    assert session0_wd is not None
    assert ".kiss-worktrees" in session0_wd, session0_wd
    assert "kiss_wt-" in session0_wd, session0_wd

    # No FileNotFoundError must have propagated out of Bash.
    assert captured["session1_bash_exception"] is None, (
        captured["session1_bash_exception"]
    )

    # The Bash output must NOT contain ``[Errno 2]`` / "No such file
    # or directory" — the bug's exact failure signature.
    bash_out = captured["session1_bash_output"] or ""
    assert "Errno 2" not in bash_out, bash_out
    assert "No such file or directory" not in bash_out, bash_out

    # ``pwd`` must report an existing directory (the fallback target).
    pwd_path = Path(bash_out.strip())
    assert pwd_path.exists(), (
        f"Bash reported a non-existent cwd: {bash_out!r}"
    )

    # The fallback target should be the parent repo root (or anywhere
    # under it), not the deleted worktree path.
    assert ".kiss-worktrees" not in str(pwd_path.resolve()), pwd_path

    # The agent's run must complete without re-raising and must return
    # a valid YAML payload.
    payload = yaml.safe_load(result)
    assert isinstance(payload, dict), result
