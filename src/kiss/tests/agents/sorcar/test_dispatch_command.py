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

import pytest

from kiss.agents.sorcar import agent_dispatch, daemon_client, fanout_guard, sea_apply, sea_commands
from kiss.agents.sorcar.daemon_client import TaskResult
from kiss.agents.sorcar.sea_settings import SeaError
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.kiss_agent import KISSAgent


def _command(plan: dict[str, Any], job: agent_dispatch.AgentJob | None = None) -> dict[str, Any]:
    """Encode a plan as the ``run`` command ``dispatch_result`` would send (for *job*, if any)."""
    kwargs = agent_dispatch.dispatch_command(
        plan["prompt"], plan["sea_path"], plan["work_dir"], plan["model_name"], plan["budget"],
        plan["timeout"], plan["parent_agent"], plan["scope_work_dir"], plan["options"],
        plan["inherit"], plan["settings"], plan["workspace"], plan["siblings"], job=job,
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
    for key in ("alias", "model_note"):  # as the job thread does before ``dispatch_result(**plan)``
        del plan[key]
    result = agent_dispatch.dispatch_result(**plan)
    assert result == "Error: the sh agent task could not run: worktree table unreadable"
    assert (tmp_path / "sub").is_dir()


class _Caller:
    """A calling agent with $100 left, half of it one sub-task's share; nothing else to inherit."""

    model_name = "m"
    work_dir = ""
    # Asks for sub-agents: a child on the caller's own model family is not
    # refused (``fanout_guard.same_model_conflict``), so the budget rule
    # under test is reached.
    task_description = "size the budgets of the sub-agents"

    def _subagent_budget_share(self, num_tasks: int) -> float:
        """Half of the remainder for one sub-task."""
        return 50.0 / (num_tasks + 1) * 2

    def isolated_repos(self) -> dict[str, str]:
        """No worktrees."""
        return {}


class _CallerThatSpent(_Caller):
    """A :class:`_Caller` whose task has spent *spent* so far, its sub-agents' spend included."""

    def __init__(self, spent: float) -> None:
        self.spent = spent

    def _live_budget_used(self) -> float:
        """The first number of the caller's ``Budget:`` line."""
        return self.spent


def test_review_without_a_budget_holds_an_eighth_of_the_task_spend(tmp_path: Path) -> None:
    """A review child gets 1/8 of what the calling task spent, $10 at least, within its share."""
    assert fanout_guard.review_budget(160.0) == 20.0
    assert fanout_guard.review_budget(80.0) == 10.0
    # An eighth under $10 is $10: of a small spend, of none, of one that is no number.
    assert fanout_guard.review_budget(79.0) == 10.0
    assert fanout_guard.review_budget(0.4) == 10.0
    assert fanout_guard.review_budget(0.0) == 10.0
    assert fanout_guard.review_budget(float("nan")) == 10.0

    def budget(task: str, caller: _Caller | None = None, **kwargs: Any) -> float:
        plan = agent_dispatch.plan_run_agent(
            str(tmp_path), task, "", parent_agent=caller or _CallerThatSpent(160.0), **kwargs,
        )
        assert isinstance(plan, dict), plan
        cmd = _command(plan)
        if "max_budget" not in kwargs:
            assert cmd["provenance"]["max_budget"] == "inherited", cmd["provenance"]
        return float(cmd["maxBudget"])

    assert budget("Verify the listed changes", tool_profile="review") == 20.0
    assert budget("Review the diff for bugs") == 20.0  # inferred from the task text
    assert budget("Implement the feature") == 50.0
    # A task that reviews *and* changes code is implementation work: no review cap.
    assert budget("Review the diff and fix every bug you find") == 50.0
    # A budget the call names is not a share: it stands, under the $10 too.
    assert budget("Verify the listed changes", tool_profile="review", max_budget="30") == 30.0
    assert budget("Verify the listed changes", tool_profile="review", max_budget="4") == 4.0
    assert budget("Verify the listed changes", tool_profile="shell") == 50.0
    # The cap follows the spend, and never raises the caller's share.
    assert budget("Review the diff", _CallerThatSpent(200.0)) == 25.0
    assert budget("Review the diff", _CallerThatSpent(800.0)) == 50.0
    # $10 at least: for a small spend, for none, for a caller that keeps no account of it.
    assert budget("Review the diff", _CallerThatSpent(16.0)) == 10.0
    assert budget("Review the diff", _CallerThatSpent(0.0)) == 10.0
    assert budget("Review the diff", _Caller()) == 10.0
    # The $10 is a cap like the eighth: a share under it stands ($100 among 14 and the caller).
    plan = agent_dispatch.plan_run_agent(
        str(tmp_path), "Review the diff", "", parent_agent=_CallerThatSpent(16.0), siblings=14,
    )
    assert isinstance(plan, dict), plan
    assert float(_command(plan)["maxBudget"]) == pytest.approx(100.0 / 15)

    def sent_budget(**kwargs: Any) -> float | None:
        sent = agent_dispatch.dispatch_command(
            "Verify the listed changes", agent_dispatch.DEFAULT_AGENT_PATH,
            "/srv/only-on-the-machine", "", None, 60.0, _CallerThatSpent(160.0),
            options=agent_dispatch.RunOptions(tool_profile="review"), **kwargs,
        )
        budget: float | None = sent["max_budget"]
        return budget

    # The spend is the caller's, so a review on another machine is capped like one here.
    assert sent_budget(inherit=True, machine="box") == 20.0
    # A review that inherits nothing has no share to cap: the daemon's default budget.
    assert sent_budget(inherit=False) is None


def test_a_review_is_sized_by_what_the_caller_and_its_sub_agents_spent(tmp_path: Path) -> None:
    """The spend is the caller's ledger: its sessions, the one running and its sub-agents'.

    A real caller with $1,000: $40 banked by earlier sessions, $30 in
    the session that is running and the $90 of a sub-agent that ended,
    folded in the way a dispatch folds it.  $160 spent, so a review
    holds $20 where an implementation child holds half of the $840 left.
    """
    caller = SorcarAgent("review-caller")
    caller.max_budget = 1000.0
    caller.budget_used = 40.0
    caller.task_description = _Caller.task_description
    session = KISSAgent("review-caller-session")
    session.budget_used = 30.0
    caller._current_executor = session
    assert agent_dispatch.task_spend(caller) == pytest.approx(70.0)

    def budget(task: str, **kwargs: Any) -> float:
        plan = agent_dispatch.plan_run_agent(str(tmp_path), task, "", parent_agent=caller, **kwargs)
        assert isinstance(plan, dict), plan
        return float(_command(plan)["maxBudget"])

    # An eighth of $70 is under $10: the review holds $10.
    assert budget("Verify the listed changes", tool_profile="review") == pytest.approx(10.0)
    agent_dispatch._attribute_dispatch_usage(caller, TaskResult("done", True, 90.0, 10, 2))
    assert agent_dispatch.task_spend(caller) == pytest.approx(160.0)
    assert budget("Verify the listed changes", tool_profile="review") == pytest.approx(20.0)
    assert budget("Implement the feature") == pytest.approx(420.0)
    # Two reviews of one fan-out each hold the eighth, not a part of it.
    plan = agent_dispatch.plan_run_agent(
        str(tmp_path), "Review the diff", "", parent_agent=caller, siblings=2,
    )
    assert isinstance(plan, dict), plan
    assert float(_command(plan)["maxBudget"]) == pytest.approx(20.0)


_BUDGETED_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"max_budget": 80.0}
"""


def test_a_sea_pinned_budget_is_the_one_reserved_on_the_job(tmp_path: Path) -> None:
    """A child of a SEA that pins ``max_budget`` holds that budget, not the share it was sent.

    The daemon's ``apply_sea`` pins the inherited $50 share to the
    SEA's $80 (an inherited value loses to the script's), so the
    sub-task spends up to $80 of the caller's remainder: the job's
    reservation, what the caller's next child is taken off, must be
    the $80 (``reserved_budget`` sums it), not the $50 placeholder.
    """
    (tmp_path / "budgeted.py").write_text(_BUDGETED_SEA)
    plan = agent_dispatch.plan_run_agent(
        str(tmp_path), "size the budgets", "budgeted.py", parent_agent=_Caller(),
    )
    assert isinstance(plan, dict), plan
    job = agent_dispatch.new_agent_job(plan["name"], plan, plan["parent_agent"])
    cmd = _command(plan, job)
    assert cmd["maxBudget"] == 50.0 and cmd["provenance"]["max_budget"] == "inherited"
    sea_apply.apply_sea(cmd, [sea_commands.load_sea(Path(plan["sea_path"]))])
    assert cmd["maxBudget"] == 80.0
    assert cmd[sea_apply.RUN_CONFIG_FIELD]["pinned"]["max_budget"] == [50.0, 80.0]
    assert job.budget == pytest.approx(80.0)
    assert agent_dispatch.reserved_budget([job]) == pytest.approx(80.0)
    # An explicit budget still wins over the SEA's, on the daemon and in the reservation.
    plan = agent_dispatch.plan_run_agent(
        str(tmp_path), "size the budgets", "budgeted.py", max_budget="20", parent_agent=_Caller(),
    )
    assert isinstance(plan, dict), plan
    job = agent_dispatch.new_agent_job(plan["name"], plan, plan["parent_agent"])
    cmd = _command(plan, job)
    sea_apply.apply_sea(cmd, [sea_commands.load_sea(Path(plan["sea_path"]))])
    assert cmd["maxBudget"] == 20.0 and job.budget == pytest.approx(20.0)


class _CallerWithEightLeft(_Caller):
    """A calling agent with $8 left."""

    def _subagent_budget_share(self, num_tasks: int) -> float:
        """The remainder split among the sub-tasks plus the caller's own share."""
        return 8.0 / (num_tasks + 1)


def test_a_budget_named_above_the_remainder_is_capped_at_it(tmp_path: Path) -> None:
    """A call or a SEA that names more than the caller has left gets the remainder, not a refusal.

    The dispatcher sends the named value as named, marked explicit or
    pinned by the SEA on the daemon as before, beside the caller's
    remainder (``budgetCap``); ``apply_sea`` keeps the precedence, the
    lock check and the provenance record, then lowers the effective
    budget to the cap; the job's reservation is the capped value.
    """
    (tmp_path / "budgeted.py").write_text(_BUDGETED_SEA)
    caller = _CallerWithEightLeft()

    def dispatched(agent: str, **kwargs: Any) -> tuple[dict[str, Any], agent_dispatch.AgentJob]:
        plan = agent_dispatch.plan_run_agent(
            str(tmp_path), "size the budgets", agent, parent_agent=caller, **kwargs,
        )
        assert isinstance(plan, dict), plan
        job = agent_dispatch.new_agent_job(plan["name"], plan, caller)
        cmd = _command(plan, job)
        sea_apply.apply_sea(cmd, [sea_commands.load_sea(Path(plan["sea_path"]))])
        return cmd, job

    # The call names 16 of the caller's 8: the wire carries 16 marked explicit and the cap 8.
    cmd, job = dispatched("", max_budget="16")
    assert cmd["provenance"]["max_budget"] == "explicit" and cmd["budgetCap"] == 8.0
    assert cmd["maxBudget"] == 8.0 and job.budget == pytest.approx(8.0)
    # The SEA names 80: the record still says the SEA pinned the $4 share to 80; the run gets 8.
    cmd, job = dispatched("budgeted.py")
    assert cmd[sea_apply.RUN_CONFIG_FIELD]["pinned"]["max_budget"] == [4.0, 80.0]
    assert cmd["maxBudget"] == 8.0 and job.budget == pytest.approx(8.0)
    # A budget within the remainder, named or inherited, is what it was;
    # the cap sent is the job's own bound, never more than its reservation.
    cmd, job = dispatched("", max_budget="5")
    assert cmd["maxBudget"] == 5.0 and cmd["budgetCap"] == 5.0 and job.budget == pytest.approx(5.0)
    cmd, job = dispatched("")
    assert cmd["maxBudget"] == 4.0 and cmd["budgetCap"] == 4.0 and job.budget == pytest.approx(4.0)


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
