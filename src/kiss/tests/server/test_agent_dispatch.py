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
from typing import Any

import pytest

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.sea_settings import PRESETS
from kiss.server.agent_file import (
    CHANNEL_PREAMBLE,
    NO_TOOLS_PROFILE,
    AgentFileError,
    apply_agent_overrides,
)


def test_cron_agent_module_is_a_valid_agent_script() -> None:
    # The contract the cron dispatch relies on: the cron module is a
    # ``channel``-preset SEA whose ``settings()`` moves the session to
    # ~/.kiss/cron/work with no git lifecycle, whose ``add_to_tools()``
    # stages the cron_job and gateway_command tools ON TOP of the
    # built-in toolset (no tool profile is staged, so the daemon keeps
    # the basics), and whose ``add_to_system_prompt()`` follows the
    # channel preamble in the system prompt suffix.
    assert cron_agent.settings() == {"preset": "channel", "work_dir": cron_agent.cron_work_dir()}
    assert cron_agent.add_to_system_prompt() == cron_agent.CRON_DISPATCH_PREAMBLE
    cmd: dict[str, Any] = {"agentPath": cron_agent.__file__, "appendToSystemPrompt": "CALLER"}
    overridden = apply_agent_overrides(cmd)
    assert overridden == {
        "tools", "appendToSystemPrompt", "workDir",
        "useWorktree", "autoCommit", "classifyTasks", "isParallel", "useWebTools", "useMemory",
    }
    assert [t.__name__ for t in cmd["tools"]] == ["cron_job", "gateway_command"]
    assert "toolProfile" not in cmd
    assert "appendBasicTools" not in cmd
    assert "toolsFile" not in cmd
    assert cmd["workDir"] == cron_agent.cron_work_dir()
    for key, value in PRESETS["channel"].items():
        assert value is False, key
    assert cmd["useWorktree"] is False
    assert cmd["autoCommit"] is False
    assert cmd["classifyTasks"] is False
    assert cmd["isParallel"] is False
    assert cmd["useWebTools"] is False
    assert cmd["useMemory"] is False
    assert cmd["appendToSystemPrompt"] == (
        "CALLER\n\n" + CHANNEL_PREAMBLE.format(name="cron_agent")
        + "\n\n" + cron_agent.CRON_DISPATCH_PREAMBLE
    )


_HELLO_TOOL = '''
def _hello() -> str:
    """Say hello.

    Returns:
        A greeting.
    """
    return "hello"
'''


def test_agent_script_tools_stages_callables_without_basic_tools(
    tmp_path: Path,
) -> None:
    # Deprecated ``tools()`` -> the returned callables are staged on the
    # daemon-side ``tools`` field and the run gets ONLY these tools
    # (+ finish): the ``none`` tool profile is staged, which the daemon
    # turns into ``append_basic_tools = False``.  Nothing names the
    # script's path and no ``appendBasicTools`` field is written.
    script = tmp_path / "self_tools_agent.py"
    script.write_text(_HELLO_TOOL + "\ndef tools() -> list:\n    return [_hello]\n")
    cmd: dict[str, Any] = {"agentPath": str(script), "toolProfile": "full"}
    assert apply_agent_overrides(cmd) == {"tools", "toolProfile"}
    assert [t.__name__ for t in cmd["tools"]] == ["_hello"]
    assert cmd["tools"][0]() == "hello"
    assert cmd["toolProfile"] == NO_TOOLS_PROFILE == "none"
    assert "appendBasicTools" not in cmd
    assert "toolsFile" not in cmd


def test_agent_script_add_to_tools_stages_callables_with_basic_tools(
    tmp_path: Path,
) -> None:
    # ``add_to_tools()`` -> same staging, but ADDED to the built-in
    # toolset: only ``tools`` is overridden, the caller's tool profile
    # stands.  A tuple is accepted and staged as a list.
    script = tmp_path / "add_tools_agent.py"
    script.write_text(_HELLO_TOOL + "\ndef add_to_tools() -> tuple:\n    return (_hello,)\n")
    cmd: dict[str, Any] = {"agentPath": str(script), "toolProfile": "review"}
    assert apply_agent_overrides(cmd) == {"tools"}
    assert isinstance(cmd["tools"], list)
    assert [t.__name__ for t in cmd["tools"]] == ["_hello"]
    assert cmd["toolProfile"] == "review"
    assert "appendBasicTools" not in cmd
    assert "toolsFile" not in cmd


def test_client_sent_tools_field_is_replaced_by_the_getter(
    tmp_path: Path,
) -> None:
    # ``tools`` is a daemon-side field: whatever JSON a client puts
    # there is overwritten by the script's getter.  A client-sent
    # ``appendBasicTools`` is not a field the loader knows: it is left
    # exactly as sent (and ignored downstream) rather than rewritten.
    script = tmp_path / "add_tools_agent.py"
    script.write_text(_HELLO_TOOL + "\ndef add_to_tools() -> list:\n    return [_hello]\n")
    cmd: dict[str, Any] = {
        "agentPath": str(script), "tools": "/client/tools.py", "appendBasicTools": False,
    }
    assert apply_agent_overrides(cmd) == {"tools"}
    assert [t.__name__ for t in cmd["tools"]] == ["_hello"]
    assert cmd["appendBasicTools"] is False
    assert "toolProfile" not in cmd


def test_agent_script_tools_wrong_type_still_rejected(
    tmp_path: Path,
) -> None:
    script = tmp_path / "bad_tools_agent.py"
    script.write_text("def tools():\n    return 42\n")
    cmd = {"agentPath": str(script)}
    with pytest.raises(AgentFileError, match="tools"):
        apply_agent_overrides(cmd)


@pytest.mark.parametrize("getter", ["tools", "add_to_tools"])
def test_agent_script_tool_getters_reject_paths_and_non_callables(
    tmp_path: Path, getter: str,
) -> None:
    # A file path (str or Path) is not a valid return value, nor is a
    # list holding a non-callable.
    for body in (
        "    return '/some/tools.py'\n",
        "    from pathlib import Path\n    return Path('/some/tools.py')\n",
        "    return [1]\n",
    ):
        script = tmp_path / f"bad_{getter}_agent.py"
        script.write_text(f"def {getter}():\n{body}")
        cmd = {"agentPath": str(script), "appendBasicTools": True}
        with pytest.raises(AgentFileError, match="list of tool callables"):
            apply_agent_overrides(cmd)
        assert "tools" not in cmd, "a broken getter must not override"
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
    cmd = {"agentPath": str(script)}
    with pytest.raises(AgentFileError, match="both tools"):
        apply_agent_overrides(cmd)
    assert "tools" not in cmd


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
        "useWebTools": None,
        "classifyTasks": None,
        "isParallel": True,
    }
    overridden = apply_agent_overrides(cmd)
    assert overridden == {"useWebTools", "classifyTasks", "isParallel"}
    assert cmd["useWebTools"] is False
    assert cmd["classifyTasks"] is True
    assert cmd["isParallel"] is False


def test_web_and_classify_getters_accept_none(tmp_path: Path) -> None:
    # ``None`` means "no override": the key is dropped from the
    # settings, so the caller's wire value stands untouched (the task
    # runner then resolves it as it does for any client-sent value).
    script = tmp_path / "none_getters_agent.py"
    script.write_text(
        "def use_web_tools():\n    return None\n\n"
        "def classify_tasks():\n    return None\n"
    )
    cmd = {"agentPath": str(script), "useWebTools": True, "classifyTasks": False}
    assert apply_agent_overrides(cmd) == set()
    assert cmd["useWebTools"] is True
    assert cmd["classifyTasks"] is False


def test_is_parallel_getter_accepts_none(tmp_path: Path) -> None:
    # ``None`` is "no override" for EVERY settings key now, ``is_parallel``
    # included: the caller's ``isParallel`` stands and nothing is
    # reported as overridden.  (A non-bool, non-None value is still
    # rejected: see the ``settings()`` type checks.)
    script = tmp_path / "none_parallel_agent.py"
    script.write_text("def is_parallel():\n    return None\n")
    cmd = {"agentPath": str(script), "isParallel": True}
    assert apply_agent_overrides(cmd) == set()
    assert cmd["isParallel"] is True


def test_is_parallel_getter_rejects_non_bool(tmp_path: Path) -> None:
    script = tmp_path / "bad_parallel_agent.py"
    script.write_text("def is_parallel():\n    return 'no'\n")
    cmd = {"agentPath": str(script), "isParallel": True}
    with pytest.raises(AgentFileError, match=r"is_parallel\(\) must be bool, got str"):
        apply_agent_overrides(cmd)
    assert cmd["isParallel"] is True, "a broken getter must not override"


def test_use_web_tools_getter_rejects_non_bool(tmp_path: Path) -> None:
    script = tmp_path / "bad_web_agent.py"
    script.write_text("def use_web_tools():\n    return 'yes'\n")
    cmd = {"agentPath": str(script), "useWebTools": None}
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
    # ``None`` means "no override": the caller's ``useMemory`` stands
    # and the field is not reported as overridden.
    script = tmp_path / "none_memory_agent.py"
    script.write_text("def use_memory():\n    return None\n")
    cmd = {"agentPath": str(script), "useMemory": True}
    assert apply_agent_overrides(cmd) == set()
    assert cmd["useMemory"] is True


def test_use_memory_getter_rejects_non_bool(tmp_path: Path) -> None:
    script = tmp_path / "bad_memory_agent.py"
    script.write_text("def use_memory():\n    return 1\n")
    cmd = {"agentPath": str(script), "useMemory": None}
    with pytest.raises(AgentFileError, match="use_memory"):
        apply_agent_overrides(cmd)
    assert cmd["useMemory"] is None, "a broken getter must not override"
