# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The dispatcher's front as three pure functions, composed the way ``run_agent`` composes them.

``plan_run_agent`` resolves and checks a call, ``dispatch_command``
turns the plan into the arguments of ``daemon_client.run`` and
``daemon_client.run_command`` encodes them as the wire command.  The
command then goes through the daemon's own ``apply_sea``, so the test
covers both sides of the lock check without a daemon or a thread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.agents.sorcar import agent_dispatch, daemon_client, sea_apply, sea_commands
from kiss.agents.sorcar.sea_settings import SeaError


def _command(plan: dict[str, Any]) -> dict[str, Any]:
    """Encode a plan as the ``run`` command ``dispatch_result`` would send."""
    kwargs = agent_dispatch.dispatch_command(
        plan["prompt"], plan["sea_path"], plan["work_dir"], plan["model_name"], plan["budget"],
        plan["timeout"], plan["parent_agent"], plan["scope_work_dir"], plan["options"],
        plan["inherit"], plan["settings"], plan["workspace"], plan["timeout_explicit"],
        plan["siblings"],
    )
    kwargs["timeout"] = kwargs.pop("record_timeout")
    return daemon_client.run_command(plan["prompt"], tab_id="t", run_token="r", **kwargs)


def test_locked_profile_is_refused_before_any_command_is_built(tmp_path: Path) -> None:
    """``sh`` locks ``tool_profile='bash'``: the plan is the refusal text, nothing is encoded."""
    plan = agent_dispatch.plan_run_agent(str(tmp_path), "echo hi", "sh", tool_profile="review")
    assert isinstance(plan, str)
    assert "locks tool_profile='bash'" in plan and "asked for 'review'" in plan


def test_plan_dispatch_encode_then_daemon_apply_agree(tmp_path: Path) -> None:
    """An agreeing explicit profile and an explicit budget travel marked; the daemon admits them."""
    plan = agent_dispatch.plan_run_agent(
        str(tmp_path), "echo hi", "sh", tool_profile="bash", max_budget="0.5",
        options='{"work_dir": "sub", "use_memory": true}',
    )
    assert isinstance(plan, dict), plan
    cmd = _command(plan)
    assert cmd["seaPath"] == plan["sea_path"]
    assert cmd["provenance"] == {
        "max_budget": "explicit", "tool_profile": "explicit", "work_dir": "explicit",
        "use_memory": "explicit",
    }
    assert cmd["workDir"] == str(tmp_path / "sub")
    assert cmd["tabScopeWorkDir"] == str(tmp_path)
    assert cmd["maxBudget"] == 0.5 and cmd["toolProfile"] == "bash" and cmd["useMemory"] is True
    assert cmd["timeout"] == plan["timeout"]
    sea = sea_commands.load_sea(Path(plan["sea_path"]))
    sea_apply.apply_sea(cmd, [sea])
    record = cmd[sea_apply.RUN_CONFIG_FIELD]
    assert record["pinned"] == {}  # the explicit profile equals the SEA's; nothing replaced
    assert "tool_profile" not in record["defaulted"] and "use_memory" not in record["defaulted"]
    assert cmd["toolProfile"] == "bash" and cmd["useMemory"] is True


class _CallerWithBrokenWorktrees:
    """A calling agent whose worktree table cannot be read (a torn-down worktree)."""

    work_dir = ""

    def isolated_repos(self) -> dict[str, str]:
        """Raise the error reading the table would raise."""
        raise OSError("worktree table unreadable")


def test_dispatch_command_builds_no_directory(tmp_path: Path) -> None:
    """``dispatch_command`` is pure: the sub-task's directory is created by the sending code."""
    plan = agent_dispatch.plan_run_agent(
        str(tmp_path), "echo hi", "sh", options='{"work_dir": "sub"}',
    )
    assert isinstance(plan, dict), plan
    target = tmp_path / "sub"
    assert plan["work_dir"] == str(target) and not target.exists()
    cmd = _command(plan)
    assert cmd["workDir"] == str(target)
    assert not target.exists()


def test_dispatch_result_returns_an_error_string_when_the_caller_cannot_be_read(
    tmp_path: Path,
) -> None:
    """A caller whose ``isolated_repos`` raises gets the ``could not run`` string, not the raise.

    The directory is created before the send, as it was before the
    dispatcher's front was split, so a failed send leaves it in place.
    """
    plan = agent_dispatch.plan_run_agent(
        str(tmp_path), "echo hi", "sh", options='{"work_dir": "sub"}',
        parent_agent=_CallerWithBrokenWorktrees(),
    )
    assert isinstance(plan, dict), plan
    del plan["alias"]  # as the job thread does before ``dispatch_result(**plan)``
    result = agent_dispatch.dispatch_result(**plan)
    assert result == "Error: the sh agent task could not run: worktree table unreadable"
    assert (tmp_path / "sub").is_dir()


def test_daemon_refuses_a_marked_explicit_value_the_plan_would_refuse(tmp_path: Path) -> None:
    """A command whose explicit mark contradicts the lock is refused by ``apply_sea`` too."""
    plan = agent_dispatch.plan_run_agent(str(tmp_path), "echo hi", "sh", tool_profile="bash")
    assert isinstance(plan, dict), plan
    cmd = _command(plan)
    cmd["toolProfile"] = "review"  # what the dispatcher would never send: the pre-check refused it
    sea = sea_commands.load_sea(Path(plan["sea_path"]))
    try:
        sea_apply.apply_sea(cmd, [sea])
    except SeaError as exc:
        assert "tool_profile" in str(exc)
    else:
        raise AssertionError("the daemon admitted a value the dispatcher refuses")
