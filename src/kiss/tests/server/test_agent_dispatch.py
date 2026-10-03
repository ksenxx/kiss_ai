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
    # ``add_to_tools()`` returns the cron_job tool on top of the basic
    # toolset) and moves the session to ~/.kiss/cron/work with no git
    # lifecycle.
    cmd = {"agentPath": cron_agent.__file__, "toolsFile": ""}
    overridden = apply_agent_overrides(cmd)
    assert overridden == {
        "toolsFile", "appendBasicTools", "workDir", "useWorktree", "autoCommit",
    }
    assert cmd["toolsFile"] == cron_agent.__file__
    assert cmd["appendBasicTools"] is True
    assert cmd["workDir"] == cron_agent.work_dir()
    assert cmd["useWorktree"] is False
    assert cmd["autoCommit"] is False


_HELLO_TOOL = '''
def _hello() -> str:
    """Say hello.

    Returns:
        A greeting.
    """
    return "hello"
'''


def test_agent_script_tools_is_own_path_without_basic_tools(
    tmp_path: Path,
) -> None:
    # ``tools()`` -> the script is its own tools file and the run gets
    # ONLY these tools (+ finish): ``appendBasicTools`` is forced off
    # whatever the client sent.
    script = tmp_path / "self_tools_agent.py"
    script.write_text(_HELLO_TOOL + "\ndef tools() -> list:\n    return [_hello]\n")
    cmd = {"agentPath": str(script), "toolsFile": "", "appendBasicTools": True}
    assert apply_agent_overrides(cmd) == {"toolsFile", "appendBasicTools"}
    assert cmd["toolsFile"] == str(script)
    assert cmd["appendBasicTools"] is False


def test_agent_script_add_to_tools_is_own_path_with_basic_tools(
    tmp_path: Path,
) -> None:
    # ``add_to_tools()`` -> same tools file, but ADDED to the basic
    # toolset: ``appendBasicTools`` is forced on.  A tuple is accepted.
    script = tmp_path / "add_tools_agent.py"
    script.write_text(_HELLO_TOOL + "\ndef add_to_tools() -> tuple:\n    return (_hello,)\n")
    cmd = {"agentPath": str(script), "toolsFile": "/client/tools.py", "appendBasicTools": False}
    assert apply_agent_overrides(cmd) == {"toolsFile", "appendBasicTools"}
    assert cmd["toolsFile"] == str(script)
    assert cmd["appendBasicTools"] is True


def test_agent_script_tools_wrong_type_still_rejected(
    tmp_path: Path,
) -> None:
    script = tmp_path / "bad_tools_agent.py"
    script.write_text("def tools():\n    return 42\n")
    cmd = {"agentPath": str(script), "toolsFile": ""}
    with pytest.raises(AgentFileError, match="tools"):
        apply_agent_overrides(cmd)


@pytest.mark.parametrize("getter", ["tools", "add_to_tools"])
def test_agent_script_tool_getters_reject_paths_and_non_callables(
    tmp_path: Path, getter: str,
) -> None:
    # A tools-file path (str or Path) is no longer a valid return value,
    # nor is a list holding a non-callable.
    for body in (
        "    return '/some/tools.py'\n",
        "    from pathlib import Path\n    return Path('/some/tools.py')\n",
        "    return [1]\n",
    ):
        script = tmp_path / f"bad_{getter}_agent.py"
        script.write_text(f"def {getter}():\n{body}")
        cmd = {"agentPath": str(script), "toolsFile": "kept", "appendBasicTools": True}
        with pytest.raises(AgentFileError, match="list of tool callables"):
            apply_agent_overrides(cmd)
        assert cmd["toolsFile"] == "kept", "a broken getter must not override"
        assert cmd["appendBasicTools"] is True


def test_agent_script_defining_both_tool_getters_is_rejected(
    tmp_path: Path,
) -> None:
    script = tmp_path / "both_tools_agent.py"
    script.write_text(
        _HELLO_TOOL
        + "\ndef tools() -> list:\n    return [_hello]\n"
        + "\ndef add_to_tools() -> list:\n    return [_hello]\n"
    )
    cmd = {"agentPath": str(script), "toolsFile": "kept"}
    with pytest.raises(AgentFileError, match="both tools"):
        apply_agent_overrides(cmd)
    assert cmd["toolsFile"] == "kept"


def test_removed_getters_are_plain_functions(tmp_path: Path) -> None:
    # ``scope_work_dir()`` and ``if_append_basic_tools()`` are no longer
    # agent-script getters: a script defining them overrides nothing
    # (and their return types are not checked).
    script = tmp_path / "legacy_getters_agent.py"
    script.write_text(
        "def scope_work_dir():\n    return 7\n\n"
        "def if_append_basic_tools():\n    return False\n"
    )
    cmd = {"agentPath": str(script), "tabScopeWorkDir": "kept", "appendBasicTools": True}
    assert apply_agent_overrides(cmd) == set()
    assert cmd["tabScopeWorkDir"] == "kept"
    assert cmd["appendBasicTools"] is True


def test_new_getters_override_their_wire_fields(tmp_path: Path) -> None:
    # ``use_web_tools()``, ``classify_tasks``, and ``is_parallel()`` are
    # agent-script getters: each overrides its wire field on the run
    # command.
    script = tmp_path / "new_getters_agent.py"
    script.write_text(textwrap.dedent("""
        def use_web_tools():
            return False

        def classify_tasks():
            return True

        def is_parallel() -> bool:
            return False
    """))
    cmd = {
        "agentPath": str(script),
        "webTools": None,
        "classifyTasks": None,
        "useParallel": True,
    }
    overridden = apply_agent_overrides(cmd)
    assert overridden == {"webTools", "classifyTasks", "useParallel"}
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


def test_use_memory_getter_overrides_wire_field(tmp_path: Path) -> None:
    # ``use_memory()`` is an agent-script getter like ``use_web_tools``:
    # a bool return overrides the run command's ``useMemory`` field.
    script = tmp_path / "memory_agent.py"
    script.write_text("def use_memory():\n    return False\n")
    cmd = {"agentPath": str(script), "useMemory": True}
    assert apply_agent_overrides(cmd) == {"useMemory"}
    assert cmd["useMemory"] is False


def test_use_memory_getter_accepts_none(tmp_path: Path) -> None:
    # ``None`` means "no per-run override": the agent then resolves the
    # persisted ``use_memory`` setting itself.
    script = tmp_path / "none_memory_agent.py"
    script.write_text("def use_memory():\n    return None\n")
    cmd = {"agentPath": str(script), "useMemory": True}
    assert apply_agent_overrides(cmd) == {"useMemory"}
    assert cmd["useMemory"] is None


def test_use_memory_getter_rejects_non_bool(tmp_path: Path) -> None:
    script = tmp_path / "bad_memory_agent.py"
    script.write_text("def use_memory():\n    return 1\n")
    cmd = {"agentPath": str(script), "useMemory": None}
    with pytest.raises(AgentFileError, match="use_memory"):
        apply_agent_overrides(cmd)
    assert cmd["useMemory"] is None, "a broken getter must not override"
