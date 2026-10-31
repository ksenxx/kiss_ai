# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sub-tasks of a sandboxed caller stay in its sandbox; a pinning daemon keeps the model.

A Terminal-Bench trial agent whose Modal sandbox died asked ``run_parallel``
for ``{"inherit": false}`` sub-agents, which then ran on the host with the
host's tools (2026-10-30).  ``plan_run_agent`` now refuses, for a caller whose
``docker_manager`` is a container or an exec bridge, every option that would
run the sub-task anywhere but in that sandbox, and a daemon that pins the
task model (``KISS_PIN_TASK_MODEL``) refuses another model for a sub-task.
Driven through ``plan_run_agent`` as ``run_agent`` / ``run_parallel`` drive it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kiss.agents.sorcar import agent_dispatch
from kiss.agents.sorcar.agent_dispatch import (
    DEFAULT_AGENT_PATH,
    PIN_TASK_MODEL_ENV,
    RunOptions,
    confinement_conflict,
    plan_run_agent,
    run_agents_parallel,
    sandbox_of,
    slash_task_conflict,
)


class _Caller:
    """A calling agent as ``plan_run_agent`` and ``inherit_from_parent`` see it."""

    model_name = "trial-model"
    work_dir = ""

    def __init__(self, docker_manager: object = None) -> None:
        self.docker_manager = docker_manager

    def _subagent_budget_share(self, siblings: int = 1) -> float:
        return 1.0

    def isolated_repos(self) -> dict[str, str]:
        return {}


def _bridged() -> _Caller:
    """A caller whose tools run on a remote environment through an exec bridge."""
    return _Caller(SimpleNamespace(url="http://127.0.0.1:33255"))


def _contained() -> _Caller:
    """A caller whose tools run in a Docker container."""
    return _Caller(SimpleNamespace(container=SimpleNamespace(id="c0ffee")))


def test_sandbox_of_names_the_container_or_bridge() -> None:
    assert sandbox_of(_bridged()) == "exec bridge http://127.0.0.1:33255"
    assert sandbox_of(_contained()) == "container c0ffee"
    assert sandbox_of(_Caller()) == "" and sandbox_of(None) == ""
    # a manager without a started container (an image not yet attached) is no sandbox
    assert sandbox_of(_Caller(SimpleNamespace(container=None))) == ""


@pytest.mark.parametrize("caller", [_bridged(), _contained()], ids=["bridge", "container"])
@pytest.mark.parametrize(("agent", "task", "options", "reason"), [
    ("", "probe the host", '{"inherit": false}',
     "options['inherit'] = false would run it on the host"),
    ("", "probe the host", '{"docker_image": "python:3.11-slim"}',
     "options['docker_image'] would start it in another container"),
    ("sh", "probe the host", "", "a SEA file, command or channel runs as host code"),
    # the daemon rewrites a task that starts with a registered slash command into that SEA
    ("", "/sh cat /etc/hostname", "",
     "a task starting with the slash command /sh would run that command's SEA as host code"),
    ("", "probe the host", '{"work_dir": "../../"}',
     "options['work_dir'] would give it a directory of the host"),
    ("", "probe the host", '{"use_web_tools": true}',
     "options['use_web_tools'] = true would browse from the host"),
    ("", "probe the host", '{"use_memory": true}',
     "options['use_memory'] = true would read and write the host's memory"),
])
def test_sandboxed_caller_may_not_send_a_sub_task_elsewhere(
    tmp_path: Path, caller: _Caller, agent: str, task: str, options: str, reason: str,
) -> None:
    plan = plan_run_agent(str(tmp_path), task, agent, options=options, parent_agent=caller)
    assert isinstance(plan, str), plan
    assert plan.startswith("Error: ") and sandbox_of(caller) in plan and reason in plan
    assert "its sub-tasks must run there too" in plan


def test_every_reason_is_named_at_once(tmp_path: Path) -> None:
    """One call that breaks several rules gets them all, so the caller fixes them in one go."""
    plan = plan_run_agent(
        str(tmp_path), "probe", "", options='{"inherit": false, "use_web_tools": true}',
        parent_agent=_bridged(),
    )
    assert isinstance(plan, str)
    assert "inherit" in plan and "use_web_tools" in plan and plan.count("would") == 2


def test_a_remote_machine_is_refused_for_a_sandboxed_caller() -> None:
    """``machine=`` would run the sub-task on another host; the resolved name is named."""
    conflict = confinement_conflict("container c0ffee", RunOptions(), "gpu-box", DEFAULT_AGENT_PATH)
    assert "machine='gpu-box' would run it on another machine" in conflict
    assert confinement_conflict("container c0ffee", RunOptions(), "", DEFAULT_AGENT_PATH) == ""
    assert confinement_conflict("", RunOptions(inherit=False), "gpu-box", "/x.py") == ""


def test_slash_task_rule_applies_to_every_task_of_run_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_parallel`` plans once and reuses the plan: a later slash task is refused up front."""
    monkeypatch.delenv(PIN_TASK_MODEL_ENV, raising=False)
    # a slash that is no registered command is plain text (a path, a fraction)
    assert slash_task_conflict(_bridged(), "/usr/bin/env is where?") == ""
    assert slash_task_conflict(_Caller(), "/sh echo hi") == ""  # host caller, no pin
    assert slash_task_conflict(None, "/sh echo hi") == ""
    assert "slash command /sh" in slash_task_conflict(_bridged(), "/sh echo hi")
    monkeypatch.setenv(PIN_TASK_MODEL_ENV, "1")
    assert "pins every task" in slash_task_conflict(_Caller(), "/sh echo hi")
    monkeypatch.delenv(PIN_TASK_MODEL_ENV)
    # nothing is started: the error comes back alone, before the first task is planned
    result = run_agents_parallel(str(tmp_path), ["ordinary task", "/sh cat /etc/hostname"],
                                 parent_agent=_bridged())
    assert result.startswith("Error: this task runs in exec bridge")
    assert "slash command /sh" in result and "ordinary" not in result
    monkeypatch.setenv(PIN_TASK_MODEL_ENV, "1")
    result = run_agents_parallel(str(tmp_path), ["ordinary task", "/sh echo hi"],
                                 parent_agent=_Caller())
    assert result.startswith("Error: this daemon pins every task to its model, trial-model")


def test_plain_inheriting_sub_task_shares_the_sandbox(tmp_path: Path) -> None:
    """The one allowed shape: the plain sub-agent, inheriting; it attaches to the same sandbox."""
    for caller, expected in ((_bridged(), "exec:http://127.0.0.1:33255"),
                             (_contained(), "container:c0ffee")):
        plan = plan_run_agent(str(tmp_path), "run the tests", "", tool_profile="shell",
                              options='{"inherit": true, "use_memory": false}',
                              parent_agent=caller, siblings=2)
        assert isinstance(plan, dict), plan
        assert plan["inherit"] is True and plan["sea_path"] == DEFAULT_AGENT_PATH
        inherited = agent_dispatch.inherit_from_parent(
            caller, plan["model_name"], plan["budget"], plan["options"],
            script_picks_model=False, siblings=2,
        )
        assert inherited.docker_image.startswith(expected.split(":")[0] + ":")
        assert inherited.docker_image == expected
        assert inherited.model_name == "trial-model"


def test_a_host_caller_keeps_every_option(tmp_path: Path) -> None:
    """Without a sandbox nothing changes: ``inherit: false`` and a command SEA still plan."""
    plan = plan_run_agent(str(tmp_path), "probe", "", options='{"inherit": false}',
                          parent_agent=_Caller())
    assert isinstance(plan, dict) and plan["inherit"] is False
    plan = plan_run_agent(str(tmp_path), "echo hi", "sh", tool_profile="bash",
                          parent_agent=_Caller())
    assert isinstance(plan, dict) and plan["sea_path"] != DEFAULT_AGENT_PATH


def test_pinning_daemon_refuses_another_model_or_model_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PIN_TASK_MODEL_ENV, "1")
    caller = _bridged()
    plan = plan_run_agent(str(tmp_path), "review this", "", model="other-model",
                          parent_agent=caller)
    assert isinstance(plan, str)
    assert "pins every task to its model, trial-model" in plan and "'other-model'" in plan
    plan = plan_run_agent(str(tmp_path), "review this", "",
                          options='{"model_config": {"reasoning_effort": "low"}}',
                          parent_agent=caller)
    assert isinstance(plan, str) and "options['model_config'] is refused" in plan
    # the caller's own model, by name or by default, is what the pin asks for
    assert isinstance(plan_run_agent(str(tmp_path), "review this", "", model="trial-model",
                                     parent_agent=caller), dict)
    assert isinstance(plan_run_agent(str(tmp_path), "review this", "", parent_agent=caller), dict)
    # a standalone call (no calling agent) has no model to keep
    assert isinstance(plan_run_agent(str(tmp_path), "review this", "", model="other-model"), dict)
    # the sandbox rule is checked first: its message names the sandbox, not the pin
    plan = plan_run_agent(str(tmp_path), "probe", "", model="other-model",
                          options='{"inherit": false}', parent_agent=caller)
    assert isinstance(plan, str) and "inherit" in plan and "pins" not in plan
    # a slash command's SEA may pick its own model: refused under the pin for a host caller too
    plan = plan_run_agent(str(tmp_path), "/sh echo hi", "", parent_agent=_Caller())
    assert isinstance(plan, str) and "slash command /sh" in plan and "pins" in plan


def test_without_the_pin_another_model_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PIN_TASK_MODEL_ENV, raising=False)
    plan = plan_run_agent(str(tmp_path), "review this", "", model="other-model",
                          parent_agent=_bridged())
    assert isinstance(plan, dict) and plan["model_name"] == "other-model"
    monkeypatch.setenv(PIN_TASK_MODEL_ENV, "0")
    assert isinstance(plan_run_agent(str(tmp_path), "review this", "", model="other-model",
                                     parent_agent=_bridged()), dict)
