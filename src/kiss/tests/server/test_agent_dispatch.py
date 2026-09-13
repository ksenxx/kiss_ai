# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Agent-script contract tests extracted from
``kiss.tests.agents.third_party_agents.test_agent_dispatch``.

Moved here because their full dependency closure touches only
kiss.agents.sorcar (the cron agent module) and kiss.server (the
daemon's ``apply_agent_overrides`` agent-file loader) — unlike the
rest of the dispatch suite, they never create a ``run_agent`` tool or
resolve a channel, so the third-party package is not involved.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from kiss.agents.sorcar import cron_agent
from kiss.server.agent_file import AgentFileError, apply_agent_overrides


def test_cron_agent_module_is_a_valid_agent_script() -> None:
    # The contract the cron dispatch relies on: passing the cron
    # module as ``extension_agent_path`` makes it its own tools file (its
    # ``tools()`` returns the cron_job tool) and moves the session
    # to ~/.kiss/cron/work with no git lifecycle.
    cmd = {"agentPath": cron_agent.__file__, "toolsFile": ""}
    overridden = apply_agent_overrides(cmd)
    assert overridden == {"toolsFile", "workDir", "useWorktree", "autoCommit"}
    assert cmd["toolsFile"] == cron_agent.__file__
    assert cmd["workDir"] == cron_agent.work_dir()
    assert cmd["useWorktree"] is False
    assert cmd["autoCommit"] is False

def test_agent_script_tools_list_normalizes_to_own_path(
    tmp_path: Path,
) -> None:
    script = tmp_path / "self_tools_agent.py"
    script.write_text(textwrap.dedent("""
        def _hello() -> str:
            \"\"\"Say hello.

            Returns:
                A greeting.
            \"\"\"
            return "hello"

        def tools() -> list:
            return [_hello]
    """))
    cmd = {"agentPath": str(script), "toolsFile": ""}
    assert apply_agent_overrides(cmd) == {"toolsFile"}
    assert cmd["toolsFile"] == str(script)

def test_agent_script_tools_wrong_type_still_rejected(
    tmp_path: Path,
) -> None:
    script = tmp_path / "bad_tools_agent.py"
    script.write_text("def tools():\n    return 42\n")
    cmd = {"agentPath": str(script), "toolsFile": ""}
    with pytest.raises(AgentFileError, match="tools"):
        apply_agent_overrides(cmd)


def test_new_getters_override_their_wire_fields(tmp_path: Path) -> None:
    # ``scope_work_dir()``, ``use_web_tools()``, ``classify_tasks``, and
    # ``is_parallel()`` are agent-script getters: each overrides its
    # wire field on the run command.
    script = tmp_path / "new_getters_agent.py"
    script.write_text(textwrap.dedent("""
        def scope_work_dir() -> str:
            return "/tmp/caller-workspace"

        def use_web_tools():
            return False

        def classify_tasks():
            return True

        def is_parallel() -> bool:
            return False
    """))
    cmd = {
        "agentPath": str(script),
        "tabScopeWorkDir": "",
        "webTools": None,
        "classifyTasks": None,
        "useParallel": True,
    }
    overridden = apply_agent_overrides(cmd)
    assert overridden == {
        "tabScopeWorkDir", "webTools", "classifyTasks", "useParallel",
    }
    assert cmd["tabScopeWorkDir"] == "/tmp/caller-workspace"
    assert cmd["webTools"] is False
    assert cmd["classifyTasks"] is True
    assert cmd["useParallel"] is False


def test_web_and_classify_getters_accept_none(tmp_path: Path) -> None:
    # ``None`` means "no per-run override": the task runner falls back
    # to the persisted setting, like an absent wire field.
    script = tmp_path / "none_getters_agent.py"
    script.write_text(
        "def use_web_tools():\n    return None\n\n"
        "def classify_tasks():\n    return None\n"
    )
    cmd = {"agentPath": str(script), "webTools": True, "classifyTasks": False}
    assert apply_agent_overrides(cmd) == {"webTools", "classifyTasks"}
    assert cmd["webTools"] is None
    assert cmd["classifyTasks"] is None


def test_is_parallel_getter_rejects_none(tmp_path: Path) -> None:
    # Unlike the two tri-state toggles, ``is_parallel`` is a plain
    # bool parameter: ``None`` is a wrong-typed return value.
    script = tmp_path / "bad_parallel_agent.py"
    script.write_text("def is_parallel():\n    return None\n")
    cmd = {"agentPath": str(script), "useParallel": True}
    with pytest.raises(AgentFileError, match="is_parallel"):
        apply_agent_overrides(cmd)
    assert cmd["useParallel"] is True, "a broken getter must not override"


def test_scope_work_dir_getter_rejects_non_string(tmp_path: Path) -> None:
    script = tmp_path / "bad_scope_agent.py"
    script.write_text("def scope_work_dir():\n    return 7\n")
    cmd = {"agentPath": str(script), "tabScopeWorkDir": "kept"}
    with pytest.raises(AgentFileError, match="scope_work_dir"):
        apply_agent_overrides(cmd)
    assert cmd["tabScopeWorkDir"] == "kept"


def test_use_web_tools_getter_rejects_non_bool(tmp_path: Path) -> None:
    script = tmp_path / "bad_web_agent.py"
    script.write_text("def use_web_tools():\n    return 'yes'\n")
    cmd = {"agentPath": str(script), "webTools": None}
    with pytest.raises(AgentFileError, match="use_web_tools"):
        apply_agent_overrides(cmd)


def test_classify_tasks_getter_rejects_non_bool(tmp_path: Path) -> None:
    script = tmp_path / "bad_classify_agent.py"
    script.write_text("def classify_tasks():\n    return 1\n")
    cmd = {"agentPath": str(script), "classifyTasks": None}
    with pytest.raises(AgentFileError, match="classify_tasks"):
        apply_agent_overrides(cmd)
