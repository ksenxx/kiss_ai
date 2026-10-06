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
from kiss.agents.sorcar.agent_file import (
    CHANNEL_PREAMBLE,
    NO_TOOLS_PROFILE,
    AgentFileError,
    apply_agent_overrides,
    channel_workspace,
    load_layers,
)
from kiss.agents.sorcar.sea_commands import SeaScriptError
from kiss.agents.sorcar.sea_settings import WORKER_DEFAULTS, kind_defaults
from kiss.core.config import kiss_home


def test_cron_agent_module_is_a_valid_agent_script() -> None:
    # The contract the cron dispatch relies on: the cron module is a
    # ``channel``-kind SEA whose ``settings()`` moves the session to
    # ~/.kiss/cron/work with no git lifecycle, whose ``tools()`` adds
    # the cron_job and gateway_command tools ON TOP of the tools the run
    # has (no tool profile is staged, so the daemon keeps the basics),
    # and whose ``system_prompt()`` appends the cron preamble to the
    # assembled system prompt while the channel preamble goes to the
    # ``appendToSystemPrompt`` suffix.  The hook callables are written
    # on every run (``BaseSea`` roots the chain), so only the settings
    # wire fields are reported as overridden.
    sea = cron_agent.CronAgentSea()
    assert sea.settings({}) == {"kind": "channel", "work_dir": cron_agent.cron_work_dir()}
    assert sea.system_prompt("BASE") == "BASE\n\n" + cron_agent.CRON_DISPATCH_PREAMBLE
    cmd: dict[str, Any] = {"agentPath": cron_agent.__file__, "appendToSystemPrompt": "CALLER"}
    overridden = apply_agent_overrides(cmd)
    assert overridden == {
        "appendToSystemPrompt", "workDir",
        "useWorktree", "autoCommit", "classifyTasks", "isParallel", "useWebTools", "useMemory",
    }
    # The workspace a channel run holds is decided from the settings
    # alone (entered by the task runner before the tools are built).
    assert channel_workspace(cmd, load_layers(cmd)) == "default"
    # The hooks are applied by the daemon on the run's own prompt and
    # tools; nothing is evaluated at staging time.
    assert [t.__name__ for t in cmd["toolsHook"]([_hello])] == [
        "_hello", "cron_job", "gateway_command",
    ]
    assert cmd["systemPromptHook"]("BASE") == "BASE\n\n" + cron_agent.CRON_DISPATCH_PREAMBLE
    assert "tools" not in cmd
    assert "toolProfile" not in cmd
    assert "appendBasicTools" not in cmd
    assert "toolsFile" not in cmd
    assert cmd["workDir"] == cron_agent.cron_work_dir()
    for key, value in WORKER_DEFAULTS.items():
        assert value is False, key
    assert kind_defaults()["channel"] == {
        **WORKER_DEFAULTS, "work_dir": str(kiss_home() / "channel_work"),
    }
    assert cmd["useWorktree"] is False
    assert cmd["autoCommit"] is False
    assert cmd["classifyTasks"] is False
    assert cmd["isParallel"] is False
    assert cmd["useWebTools"] is False
    assert cmd["useMemory"] is False
    assert cmd["appendToSystemPrompt"] == (
        "CALLER\n\n" + CHANNEL_PREAMBLE.format(name="cron_agent")
    )
    # Applied to the base prompt plus that suffix, the hook yields the
    # caller -> channel preamble -> cron preamble order of the old contract.
    assert cmd["systemPromptHook"]("BASE\n\n" + cmd["appendToSystemPrompt"]) == (
        "BASE\n\nCALLER\n\n" + CHANNEL_PREAMBLE.format(name="cron_agent")
        + "\n\n" + cron_agent.CRON_DISPATCH_PREAMBLE
    )


def _hello() -> str:
    """Say hello.

    Returns:
        A greeting.
    """
    return "hello"


_HELLO_TOOL = '''
from kiss.agents.seas.base.base_sea import BaseSea


def _hello() -> str:
    """Say hello.

    Returns:
        A greeting.
    """
    return "hello"
'''


def test_tools_with_none_profile_stages_hook_without_basic_tools(
    tmp_path: Path,
) -> None:
    # ``tools()`` plus ``settings()["tool_profile"] == "none"`` is how a
    # SEA gets ONLY its own tools (+ finish): the method is staged as
    # the daemon-side ``toolsHook`` callable and the ``none`` tool
    # profile is staged over the caller's, which the daemon turns into
    # ``append_basic_tools = False``.  Nothing names the script's path,
    # no ``tools`` list and no ``appendBasicTools`` field is written;
    # the hook (written on every run) is not among the reported fields.
    script = tmp_path / "self_tools_agent.py"
    script.write_text(_HELLO_TOOL + """

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'tool_profile': 'none'}

    def tools(self, tools):
        return tools + [_hello]
""")
    cmd: dict[str, Any] = {"agentPath": str(script), "toolProfile": "full"}
    assert apply_agent_overrides(cmd) == {"toolProfile"}
    staged = cmd["toolsHook"]([])
    assert [t.__name__ for t in staged] == ["_hello"]
    assert staged[0]() == "hello"
    assert cmd["toolProfile"] == NO_TOOLS_PROFILE == "none"
    assert "tools" not in cmd
    assert "appendBasicTools" not in cmd
    assert "toolsFile" not in cmd


def test_agent_script_tools_hook_extends_the_tools_it_is_given(
    tmp_path: Path,
) -> None:
    # ``tools()`` alone -> same staging, but ADDED to the built-in
    # toolset: no settings field is overridden, the caller's tool
    # profile stands, and the hook keeps the tools the daemon hands it
    # ahead of the SEA's own.
    script = tmp_path / "add_tools_agent.py"
    script.write_text(_HELLO_TOOL + """

class Sea(BaseSea):
    def tools(self, tools):
        return tools + [_hello]
""")
    cmd: dict[str, Any] = {"agentPath": str(script), "toolProfile": "review"}
    assert apply_agent_overrides(cmd) == set()
    assert [t.__name__ for t in cmd["toolsHook"]([print])] == ["print", "_hello"]
    assert cmd["toolProfile"] == "review"
    assert "tools" not in cmd
    assert "appendBasicTools" not in cmd
    assert "toolsFile" not in cmd


def test_client_sent_tools_hook_field_is_replaced_by_the_method(
    tmp_path: Path,
) -> None:
    # ``toolsHook`` is a daemon-side field: whatever JSON a client puts
    # there is overwritten by the SEA's method.  A client-sent
    # ``appendBasicTools`` is not a field the loader knows: it is left
    # exactly as sent (and ignored downstream) rather than rewritten.
    script = tmp_path / "add_tools_agent.py"
    script.write_text(_HELLO_TOOL + """

class Sea(BaseSea):
    def tools(self, tools):
        return tools + [_hello]
""")
    cmd: dict[str, Any] = {
        "agentPath": str(script), "toolsHook": "/client/tools.py", "appendBasicTools": False,
    }
    assert apply_agent_overrides(cmd) == set()
    assert [t.__name__ for t in cmd["toolsHook"]([])] == ["_hello"]
    assert cmd["appendBasicTools"] is False
    assert "toolProfile" not in cmd


def test_agent_script_tools_wrong_type_rejected_when_the_hook_runs(
    tmp_path: Path,
) -> None:
    # The method runs lazily (the daemon calls the staged hook on the
    # run's tools), so a wrong return type is diagnosed by the hook, not
    # at staging time.
    script = tmp_path / "bad_tools_agent.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def tools(self, tools):
        return 42
""")
    cmd: dict[str, Any] = {"agentPath": str(script)}
    assert apply_agent_overrides(cmd) == set()
    with pytest.raises(
        SeaScriptError,
        match=(
            r"tools\(\) of agent script '.*bad_tools_agent\.py' must return "
            r"a list of tool callables \(not a file path\), got int"
        ),
    ):
        cmd["toolsHook"]([])
    assert "tools" not in cmd


def test_agent_script_tools_rejects_paths_tuples_and_non_callables(
    tmp_path: Path,
) -> None:
    # A file path (str or Path) is not a valid return value, nor is a
    # tuple or a list holding a non-callable; a method that raises is
    # reported with its exception.
    for body, message in (
        ("        return '/some/tools.py'\n", "list of tool callables.*got str"),
        (
            "        from pathlib import Path\n        return Path('/some/tools.py')\n",
            "list of tool callables.*got (Posix|Windows)Path",
        ),
        ("        return [1]\n", "list of tool callables.*got list"),
        ("        return tuple(tools)\n", "list of tool callables.*got tuple"),
        ("        return tools + 42\n", r"tools\(\) of agent script .* raised: "),
    ):
        script = tmp_path / "bad_tools_agent.py"
        script.write_text(
            "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
            f"class Sea(BaseSea):\n    def tools(self, tools):\n{body}"
        )
        cmd: dict[str, Any] = {"agentPath": str(script), "appendBasicTools": True}
        assert apply_agent_overrides(cmd) == set()
        with pytest.raises(SeaScriptError, match=message):
            cmd["toolsHook"]([])
        assert "tools" not in cmd
        assert cmd["appendBasicTools"] is True


def test_module_level_functions_are_not_sea_methods(tmp_path: Path) -> None:
    # ``scope_work_dir()`` and ``if_append_basic_tools()`` of the old
    # getter contract are plain module functions: a SEA file defining
    # them overrides nothing (and their return types are not checked).
    script = tmp_path / "legacy_getters_agent.py"
    script.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n"
        "class Sea(BaseSea):\n    pass\n\n"
        "def scope_work_dir():\n    return 7\n\n"
        "def if_append_basic_tools():\n    return False\n"
    )
    cmd = {"agentPath": str(script), "tabScopeWorkDir": "kept", "appendBasicTools": True}
    assert apply_agent_overrides(cmd) == set()
    assert cmd["tabScopeWorkDir"] == "kept"
    assert cmd["appendBasicTools"] is True


def test_file_without_a_sea_class_is_rejected(tmp_path: Path) -> None:
    # A file made of the old contract's module-level getters alone is
    # not a SEA: the loader wants exactly one BaseSea subclass.
    script = tmp_path / "getters_only_agent.py"
    script.write_text("def settings():\n    return {}\n\ndef add_to_tools():\n    return []\n")
    cmd: dict[str, Any] = {"agentPath": str(script), "appendBasicTools": True}
    with pytest.raises(
        AgentFileError,
        match=r"must define exactly one subclass of BaseSea .*; found none",
    ):
        apply_agent_overrides(cmd)
    assert cmd == {"agentPath": str(script), "appendBasicTools": True}


def test_legacy_per_field_getters_and_tools_are_plain_functions(tmp_path: Path) -> None:
    # The per-field getters of the old contract (``model()``,
    # ``use_memory()``, ``is_parallel()``, ...) and the whole-toolset
    # ``tools()`` are ordinary module functions now: a SEA file defining
    # them next to its class overrides nothing, their return values are
    # not type-checked, and the caller's wire fields stand exactly as
    # sent.
    script = tmp_path / "old_contract_agent.py"
    script.write_text(textwrap.dedent("""
        from kiss.agents.seas.base.base_sea import BaseSea


        class Sea(BaseSea):
            pass

        def model():
            return 42

        def use_memory():
            return "yes"

        def is_parallel():
            return False

        def use_web_tools():
            return 1

        def classify_tasks():
            return None

        def tools():
            return "/some/tools.py"

        def add_to_tools():
            return "/some/tools.py"

        def append_to_system_prompt():
            return 7

        def add_to_system_prompt():
            return 7

        def dispatch_timeout():
            return "soon"
    """))
    cmd: dict[str, Any] = {
        "agentPath": str(script),
        "model": "kept-model",
        "useMemory": True,
        "isParallel": True,
        "useWebTools": None,
        "classifyTasks": False,
        "appendToSystemPrompt": "CALLER",
    }
    assert apply_agent_overrides(cmd) == set()
    # The only writes are the provenance record of a script that
    # replaced nothing and the four hook callables every run gets,
    # all of them ``BaseSea``'s identities here.
    assert cmd.pop("_runConfig") == {
        "sea": "old_contract_agent", "kind": "session", "pinned": {},
    }
    assert cmd.pop("systemPromptHook")("BASE") == "BASE"
    assert cmd.pop("toolsHook")([_hello]) == [_hello]
    assert cmd.pop("llmCallHook")([1, 2]) == [1, 2]
    assert cmd.pop("toolCallHook")("Bash", {"command": "ls"}) is None
    assert cmd == {
        "agentPath": str(script),
        "model": "kept-model",
        "useMemory": True,
        "isParallel": True,
        "useWebTools": None,
        "classifyTasks": False,
        "appendToSystemPrompt": "CALLER",
    }


def test_settings_keys_override_their_wire_fields(tmp_path: Path) -> None:
    # ``use_web_tools``, ``auto_classify`` and ``allow_fan_out`` are
    # ``settings()`` keys: each is written over its wire field on the
    # run command, whatever the caller sent there.
    script = tmp_path / "settings_agent.py"
    script.write_text(textwrap.dedent("""
        from kiss.agents.seas.base.base_sea import BaseSea

        class Sea(BaseSea):
            def settings(self, settings):
                return settings | {
                    "use_web_tools": False,
                    "auto_classify": True,
                    "allow_fan_out": False,
                }


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


def test_web_and_classify_settings_accept_none(tmp_path: Path) -> None:
    # ``None`` means "no override": the key is dropped from the
    # settings, so the caller's wire value stands untouched (the task
    # runner then resolves it as it does for any client-sent value).
    script = tmp_path / "none_settings_agent.py"
    script.write_text(
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'use_web_tools': None, 'auto_classify': None}
"""
    )
    cmd = {"agentPath": str(script), "useWebTools": True, "classifyTasks": False}
    assert apply_agent_overrides(cmd) == set()
    assert cmd["useWebTools"] is True
    assert cmd["classifyTasks"] is False


def test_is_parallel_setting_accepts_none(tmp_path: Path) -> None:
    # ``None`` is "no override" for EVERY settings key, ``is_parallel``
    # included: the caller's ``isParallel`` stands and nothing is
    # reported as overridden.  (A non-bool, non-None value is still
    # rejected: see the type-check tests below.)
    script = tmp_path / "none_parallel_agent.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'allow_fan_out': None}
""")
    cmd = {"agentPath": str(script), "isParallel": True}
    assert apply_agent_overrides(cmd) == set()
    assert cmd["isParallel"] is True


def test_is_parallel_setting_rejects_non_bool(tmp_path: Path) -> None:
    script = tmp_path / "bad_parallel_agent.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'allow_fan_out': 'no'}
""")
    cmd = {"agentPath": str(script), "isParallel": True}
    with pytest.raises(
        AgentFileError,
        match=(
            r"agent script '.*bad_parallel_agent\.py': "
            r"settings\(\)\['allow_fan_out'\] must be bool, got str"
        ),
    ):
        apply_agent_overrides(cmd)
    assert cmd["isParallel"] is True, "a broken setting must not override"


def test_use_web_tools_setting_rejects_non_bool(tmp_path: Path) -> None:
    script = tmp_path / "bad_web_agent.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'use_web_tools': 'yes'}
""")
    cmd = {"agentPath": str(script), "useWebTools": None}
    with pytest.raises(
        AgentFileError, match=r"settings\(\)\['use_web_tools'\] must be bool, got str",
    ):
        apply_agent_overrides(cmd)
    assert cmd["useWebTools"] is None


def test_classify_tasks_setting_rejects_non_bool(tmp_path: Path) -> None:
    script = tmp_path / "bad_classify_agent.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'auto_classify': 1}
""")
    cmd = {"agentPath": str(script), "classifyTasks": None}
    with pytest.raises(
        AgentFileError, match=r"settings\(\)\['auto_classify'\] must be bool, got int",
    ):
        apply_agent_overrides(cmd)
    assert cmd["classifyTasks"] is None


def test_use_memory_setting_overrides_wire_field(tmp_path: Path) -> None:
    # ``use_memory`` is a ``settings()`` key like ``use_web_tools``: a
    # bool value overrides the run command's ``useMemory`` field.
    script = tmp_path / "memory_agent.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'use_memory': False}
""")
    cmd = {"agentPath": str(script), "useMemory": True}
    assert apply_agent_overrides(cmd) == {"useMemory"}
    assert cmd["useMemory"] is False


def test_use_memory_setting_accepts_none(tmp_path: Path) -> None:
    # ``None`` means "no override": the caller's ``useMemory`` stands
    # and the field is not reported as overridden.
    script = tmp_path / "none_memory_agent.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'use_memory': None}
""")
    cmd = {"agentPath": str(script), "useMemory": True}
    assert apply_agent_overrides(cmd) == set()
    assert cmd["useMemory"] is True


def test_use_memory_setting_rejects_non_bool(tmp_path: Path) -> None:
    script = tmp_path / "bad_memory_agent.py"
    script.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'use_memory': 1}
""")
    cmd = {"agentPath": str(script), "useMemory": None}
    with pytest.raises(
        AgentFileError, match=r"settings\(\)\['use_memory'\] must be bool, got int",
    ):
        apply_agent_overrides(cmd)
    assert cmd["useMemory"] is None, "a broken setting must not override"
