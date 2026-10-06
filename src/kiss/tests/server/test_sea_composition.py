# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the composed SEA contract.

What these tests pin down (``reports/sea-design-report.html``):

* ``settings()`` is the one vocabulary for run settings; ``extends`` is a
  removed key (a SEA extends another by Python inheritance, in-process or
  through :func:`~kiss.agents.sorcar.sea_commands.sea_class`); the
  ``prompt`` / ``system_prompt`` family are methods, never settings.
* Inherited settings fold base-first (the subclass's ``settings`` sees
  what its base returned, so a later value wins); a base's ``kind`` is
  never masked by the default ``session``.
* Methods chain base-first through the class hierarchy: the subclass's
  ``prompt`` receives what the base produced; a class contributes once
  even when it appears in both a picker base and the script's own MRO.
* The dispatcher resolves every agent spelling to a script path, accepts
  ``work_dir`` and ``allow_fan_out`` options, and refuses the old wire
  spellings of the ``add_to_*`` options.
* On a real daemon: a slash command on a derived SEA runs the whole chain;
  a channel SEA holds its workspace for the run and releases it; a client-
  sent daemon-side field is dropped.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.sorcar import agent_dispatch, channel_workspace, daemon_client, sea_commands
from kiss.agents.sorcar.agent_dispatch import RunOptions, inherit_from_parent, resolve_agent
from kiss.agents.sorcar.agent_file import (
    CHANNEL_PREAMBLE,
    AgentFileError,
    apply_agent_overrides,
    load_layers,
)
from kiss.agents.sorcar.agent_file import channel_workspace as held_workspace
from kiss.agents.sorcar.sea_commands import (
    SeaScriptError,
    base_settings,
    declared_settings,
    evaluate_sea,
    sea_layers,
    sea_settings,
)
from kiss.agents.sorcar.sea_settings import (
    REMOVED_SETTINGS,
    SETTING_TYPES,
    SettingsError,
    resolve_settings,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.config import kiss_home
from kiss.core.kiss_agent import KISSAgent
from kiss.server import sorcar
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness

BASE_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea

def base_tool(x: str) -> str:
    '''Base tool.'''
    return 'base ' + x

def shared(x: str) -> str:
    '''Shared (base).'''
    return 'base ' + x

class Sea(BaseSea):
    def description(self):
        return 'base'
    def settings(self, settings):
        return settings | {'kind': 'worker', 'use_web_tools': True, 'max_budget': 1}
    def prompt(self, task):
        return '[base] ' + task + ' BASE-ADD {task_id}'
    def system_prompt(self, system_prompt):
        return 'BASE SYSTEM\\n\\nBASE PROTOCOL'
    def tools(self, tools):
        return tools + [base_tool, shared]
    def llm_call_hook(self, new_messages):
        return new_messages
"""

DERIVED_SEA_TEMPLATE = """
from kiss.agents.sorcar.sea_commands import sea_class

def shared(x: str) -> str:
    '''Shared (derived).'''
    return 'derived ' + x

class Derived(sea_class({base!r})):
    def description(self):
        return 'derived'
    def settings(self, settings):
        return settings | {{'max_budget': 2.0}}
    def prompt(self, task):
        return '[derived] ' + task + ' DERIVED-ADD'
    def system_prompt(self, system_prompt):
        return system_prompt + '\\n\\nDERIVED PROTOCOL'
    def tools(self, tools):
        return [tool for tool in tools if tool.__name__ != 'shared'] + [shared]
"""

PLAIN_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {}
"""


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _mro_names(sea: BaseSea) -> list[str]:
    """Return the SEA class names base-first, the way the launcher chains them."""
    return [cls.__name__ for cls in reversed(type(sea).__mro__) if issubclass(cls, BaseSea)]


# ---------------------------------------------------------------------------
# Settings vocabulary and inheritance fold
# ---------------------------------------------------------------------------


def test_settings_vocabulary_has_no_prompt_or_extends_keys() -> None:
    assert "extends" not in SETTING_TYPES
    assert "Python inheritance" in REMOVED_SETTINGS["extends"]
    with pytest.raises(SettingsError, match="key 'extends' was removed"):
        resolve_settings({"extends": "x"})
    for key in ("prompt", "system_prompt"):
        with pytest.raises(SettingsError, match=f"unknown key {key!r}"):
            resolve_settings({key: "x"})
    # The prompt suffixes are options, not settings: refused with the method to use.
    with pytest.raises(SettingsError, match="key 'add_to_prompt' was removed: .*prompt\\(task\\)"):
        resolve_settings({"add_to_prompt": "x"})
    # The dispatcher's option vocabulary is the run-settings subset of
    # the SEA vocabulary (no script-describing keys) plus its own.
    assert set(agent_dispatch.OPTION_TYPES) == (
        set(SETTING_TYPES) - {"kind", "locked", "hidden"} - set(agent_dispatch.ARGUMENT_OPTIONS)
    ) | {"inherit", "workspace", "add_to_prompt", "add_to_system_prompt"}


class _Worker(BaseSea):
    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {"kind": "worker", "use_web_tools": True}


class _Budgeted(_Worker):
    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {"max_budget": 1}


class _Channel(_Worker):
    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return settings | {"kind": "channel"}


def test_inherited_settings_fold_base_first_and_session_never_masks_a_kind() -> None:
    assert declared_settings([]) == {}
    assert resolve_settings({}) == {"kind": "session"}
    # The subclass's settings() sees the base's and its value wins.
    assert declared_settings([_Budgeted()]) == {
        "kind": "worker", "use_web_tools": True, "max_budget": 1,
    }
    resolved = base_settings([_Budgeted()])
    assert resolved["kind"] == "worker"
    assert resolved["use_web_tools"] is True and resolved["max_budget"] == 1.0
    assert resolved["allow_fan_out"] is False  # the worker preset
    # A subclass without a kind keeps its base's; one with a kind replaces it.
    assert base_settings([_Budgeted()])["kind"] == "worker"
    assert base_settings([_Channel()])["kind"] == "channel"
    # Two layers (picker base, then the script) fold the same way.
    assert declared_settings([_Worker(), _Channel()])["kind"] == "channel"


# ---------------------------------------------------------------------------
# Inheritance: by command name and by path, base-first chaining
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[Path]:
    folder = tmp_path / "seas"
    folder.mkdir()
    home = kiss_home()
    home.mkdir(parents=True, exist_ok=True)
    seas_md = home / "SEAS.md"
    previous = seas_md.read_text(encoding="utf-8") if seas_md.is_file() else None
    seas_md.write_text(f"{folder}\n", encoding="utf-8")
    sea_commands._reset_for_tests()
    yield folder
    if previous is None:
        seas_md.unlink(missing_ok=True)
    else:
        seas_md.write_text(previous, encoding="utf-8")
    sea_commands._reset_for_tests()


def test_inheriting_by_command_name_and_by_path_chains_both_classes(registry: Path) -> None:
    base = _write(registry / "basey" / "basey_sea.py", BASE_SEA)
    by_name = _write(
        registry / "byname" / "byname_sea.py", DERIVED_SEA_TEMPLATE.format(base="basey"),
    )
    by_path = _write(
        registry / "bypath" / "bypath_sea.py",
        "from pathlib import Path\n"
        + DERIVED_SEA_TEMPLATE.format(base="../basey/basey_sea.py").replace(
            "sea_class('../basey/basey_sea.py')",
            "sea_class('../basey/basey_sea.py', relative_to=Path(__file__).parent)",
        ),
    )
    sea_commands.refresh_registry()
    for derived in (by_name, by_path):
        layers = sea_layers(derived)
        assert [layer.path for layer in layers] == [derived]
        assert _mro_names(layers[0]) == ["BaseSea", "Sea", "Derived"]
        base_module = sys.modules[type(layers[0]).__mro__[1].__module__]
        assert Path(base_module.__file__ or "") == base.resolve()
        settings = base_settings(layers)
        assert settings["kind"] == "worker"
        assert settings["use_web_tools"] is True and settings["max_budget"] == 2.0
        run = evaluate_sea(layers, "do it", task_id="T-1")
        assert run.prompt == "[derived] [base] do it BASE-ADD T-1 DERIVED-ADD"
        assert run.system_prompt_hook is not None
        assert run.system_prompt_hook("X") == "BASE SYSTEM\n\nBASE PROTOCOL\n\nDERIVED PROTOCOL"
        assert run.tools_hook is not None
        tools = run.tools_hook([])
        assert [tool.__name__ for tool in tools] == ["base_tool", "shared"]
        assert tools[1]("x") == "derived x"  # the subclass replaced the base's ``shared``
        assert run.llm_call_hook is not None and run.llm_call_hook([]) == []
        # No layer overrides ``tool_call_hook``: the staged hook is the identity.
        assert run.tool_call_hook is not None and run.tool_call_hook("x", {}) is None
        cmd: dict[str, Any] = {"agentPath": str(derived), "prompt": "do it", "parentTaskId": "T-1"}
        overridden = apply_agent_overrides(cmd)
        assert cmd["prompt"] == "[derived] [base] do it BASE-ADD T-1 DERIVED-ADD"
        assert cmd["systemPromptHook"]("X") == "BASE SYSTEM\n\nBASE PROTOCOL\n\nDERIVED PROTOCOL"
        assert cmd["maxBudget"] == 2.0 and cmd["useWebTools"] is True
        assert [tool.__name__ for tool in cmd["toolsHook"]([])] == ["base_tool", "shared"]
        assert cmd["llmCallHook"]([]) == [] and cmd["toolCallHook"]("x", {}) is None
        # The overridden set lists settings and ``prompt`` only, never the
        # four hooks (they are always staged).
        assert {"prompt", "maxBudget", "useWebTools"} <= overridden
        assert not {"systemPromptHook", "toolsHook", "llmCallHook", "toolCallHook"} & overridden


def test_inheritance_errors_name_the_script(registry: Path) -> None:
    # ``extends`` is a removed settings key: the error says what replaced it.
    extends = _write(registry / "extends_sea.py", """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'extends': 'nobody'}
""")
    with pytest.raises(
        SeaScriptError,
        match=r"settings\(\) key 'extends' was removed: a SEA extends another by Python inherit",
    ):
        sea_settings(extends)
    missing = _write(registry / "missing_sea.py", """
from kiss.agents.sorcar.sea_commands import sea_class

class Sea(sea_class('nobody')):
    pass
""")
    with pytest.raises(
        SeaScriptError,
        match=r"missing_sea.py' failed to import: .*sea_class\('nobody'\): not a registered SEA",
    ):
        sea_layers(missing)
    gone = _write(registry / "gone_sea.py", """
from kiss.agents.sorcar.sea_commands import sea_class

class Sea(sea_class('gone.py')):
    pass
""")
    with pytest.raises(
        SeaScriptError,
        match=r"gone_sea.py' failed to import: .*sea_class\('gone.py'\): not an existing Python",
    ):
        sea_layers(gone)
    # A channel SEA may inherit from a worker base; a picker base may run
    # under a channel SEA; either way the channel's settings win.
    _write(registry / "wbase" / "wbase_sea.py", BASE_SEA)
    chan2 = _write(registry / "chan2" / "chan2_sea.py", """
from kiss.agents.sorcar.sea_commands import sea_class

class Chan(sea_class('wbase')):
    def description(self):
        return 'c'
    def settings(self, settings):
        return settings | {'kind': 'channel'}
""")
    sea_commands.refresh_registry()
    layers = sea_layers(chan2)
    assert _mro_names(layers[0]) == ["BaseSea", "Sea", "Chan"]
    assert base_settings(layers)["kind"] == "channel"
    wbase = registry / "wbase" / "wbase_sea.py"
    cmd: dict[str, Any] = {"agentPath": str(chan2), "prompt": "p", "workspace": "acct"}
    assert held_workspace(cmd, load_layers(cmd)) == "acct"
    assert held_workspace({"agentPath": str(chan2)}, layers) == "default"
    assert held_workspace({"agentPath": str(wbase)}, load_layers({"agentPath": str(wbase)})) == ""
    picker_layers = sea_layers(chan2, base=wbase)
    assert [layer.path for layer in picker_layers] == [wbase.resolve(), chan2]
    assert base_settings(picker_layers)["kind"] == "channel"
    cmd = {"agentPath": str(chan2), "prompt": "p"}
    apply_agent_overrides(cmd)
    assert cmd["appendToSystemPrompt"].startswith(CHANNEL_PREAMBLE.format(name="chan2"))
    assert cmd["prompt"] == "[base] p BASE-ADD "


def test_a_class_shared_by_the_picker_base_and_the_script_contributes_once(
    registry: Path, tmp_path: Path,
) -> None:
    tally = tmp_path / "tally.txt"
    tally.write_text("", encoding="utf-8")
    counting = (
        "from pathlib import Path\n"
        f"Path({str(tally)!r}).open('a').write({{name!r}} + '\\n')\n"
    )
    _write(registry / "root" / "root_sea.py", counting.format(name="root") + """
from kiss.agents.seas.base.base_sea import BaseSea

class Root(BaseSea):
    def prompt(self, task):
        return '[root] ' + task
""")
    picker = _write(registry / "picker" / "picker_sea.py", counting.format(name="picker") + """
from kiss.agents.sorcar.sea_commands import sea_class

class Picker(sea_class('root')):
    def settings(self, settings):
        return settings | {'kind': 'worker'}
    def prompt(self, task):
        return '[picker] ' + task
""")
    direct = _write(registry / "direct" / "direct_sea.py", counting.format(name="direct") + """
from kiss.agents.sorcar.sea_commands import sea_class

class Direct(sea_class('picker')):
    def prompt(self, task):
        return '[direct] ' + task
""")
    sea_commands.refresh_registry()
    tally.write_text("", encoding="utf-8")
    layers = sea_layers(direct, base=picker)
    assert [layer.path for layer in layers] == [picker.resolve(), direct]
    assert _mro_names(layers[0]) == ["BaseSea", "Root", "Picker"]
    assert _mro_names(layers[1]) == ["BaseSea", "Root", "Picker", "Direct"]
    # Each file executes per class definition that loads it (the picker
    # base, and again inside ``direct``'s ``sea_class``) …
    assert sorted(set(tally.read_text(encoding="utf-8").split())) == ["direct", "picker", "root"]
    # … but the chain contributes a class once however many times its
    # file was executed: ``prompt`` applies root, picker and direct once each.
    assert evaluate_sea(layers, "t").prompt == "[direct] [picker] [root] t"
    assert base_settings(layers)["kind"] == "worker"
    # A common ancestor keeps its place under the base chain: a sibling
    # of the picker that shares its root still applies root once, first.
    sibling = _write(registry / "sibling" / "sibling_sea.py", counting.format(name="sibling") + """
from kiss.agents.sorcar.sea_commands import sea_class

class Sibling(sea_class('root')):
    def prompt(self, task):
        return '[sibling] ' + task
""")
    sea_commands.refresh_registry()
    layers = sea_layers(sibling, base=picker)
    assert _mro_names(layers[1]) == ["BaseSea", "Root", "Sibling"]
    assert evaluate_sea(layers, "t").prompt == "[sibling] [picker] [root] t"
    # The base IS the script: one layer.
    root = registry / "root" / "root_sea.py"
    assert [layer.path for layer in sea_layers(root, base=root)] == [root.resolve()]


def test_failed_reload_keeps_an_earlier_execution_resolving_its_annotations(
    tmp_path: Path,
) -> None:
    """A broken re-execution of a file does not unregister a running execution's module."""
    import typing

    typed_source = """
from __future__ import annotations
from dataclasses import dataclass
from kiss.agents.seas.base.base_sea import BaseSea
class Item:
    pass
@dataclass
class Box:
    item: Item
class Sea(BaseSea):
    def settings(self, settings):
        return settings | {}
"""
    sea = _write(tmp_path / "typed_sea.py", typed_source)
    first = sys.modules[type(sea_layers(sea)[0]).__module__]
    assert typing.get_type_hints(first.Box)["item"] is first.Item
    sea.write_text("def settings(:\n", encoding="utf-8")
    with pytest.raises(SeaScriptError, match="failed to import"):
        sea_layers(sea)
    # Still resolvable after the failed reload (the module entry was restored).
    assert typing.get_type_hints(first.Box)["item"] is first.Item
    # A later successful reload of the same source resolves through the new module.
    sea.write_text(typed_source, encoding="utf-8")
    second = sys.modules[type(sea_layers(sea)[0]).__module__]
    assert second is not first
    assert typing.get_type_hints(second.Box)["item"] is second.Item
    assert typing.get_type_hints(first.Box)["item"] is second.Item


def test_prompt_method_is_checked(tmp_path: Path) -> None:
    sea = _write(tmp_path / "empty_sea.py", """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def prompt(self, task):
        return '  '
""")
    with pytest.raises(SeaScriptError, match=r"prompt\(\) of agent script .* non-empty string"):
        evaluate_sea(sea_layers(sea), "t")
    sea = _write(tmp_path / "num_sea.py", """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def prompt(self, task):
        return 5
""")
    with pytest.raises(SeaScriptError, match=r"prompt\(\) of agent script .* must return a string"):
        evaluate_sea(sea_layers(sea), "t")
    # A module-level ``prompt`` is no hook: without a BaseSea subclass the
    # file is not a SEA at all.
    sea = _write(tmp_path / "const_sea.py", "def prompt(task):\n    return task\n")
    with pytest.raises(
        SeaScriptError, match=r"must define exactly one subclass of BaseSea.* found none",
    ):
        evaluate_sea(sea_layers(sea), "t")
    sea = _write(tmp_path / "raises_sea.py", """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def prompt(self, task):
        raise KeyError('k')
""")
    with pytest.raises(SeaScriptError, match=r"prompt\(\) of agent script .* raised: KeyError"):
        evaluate_sea(sea_layers(sea), "t")
    sea = _write(tmp_path / "noprompt_sea.py", PLAIN_SEA)
    run = evaluate_sea(sea_layers(sea), "kept")
    # No ``prompt()``/``tools()``: the task is kept and the tools hook is the identity.
    assert run.prompt == "kept" and run.tools_hook is not None
    assert run.tools_hook([print, len]) == [print, len]
    cmd: dict[str, Any] = {"agentPath": str(sea), "prompt": "kept"}
    assert apply_agent_overrides(cmd) == set()
    assert cmd.pop("_runConfig") == {"sea": "noprompt", "kind": "session", "pinned": {}}
    # The four hooks are always staged; here they are all identities.
    messages = [{"role": "user", "content": "hi"}]
    assert cmd.pop("systemPromptHook")("S") == "S"
    assert cmd.pop("toolsHook")([print, len]) == [print, len]
    assert cmd.pop("llmCallHook")(messages) == messages
    assert cmd.pop("toolCallHook")("x", {}) is None
    assert cmd == {"agentPath": str(sea), "prompt": "kept"}
    with pytest.raises(AgentFileError, match="must be a path string"):
        apply_agent_overrides({"agentPath": 7, "prompt": "p"})


# ---------------------------------------------------------------------------
# Dispatcher: one path, work_dir option, renamed options, allow_fan_out
# ---------------------------------------------------------------------------


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def capture_run(prompt: str, **kwargs: Any) -> daemon_client.TaskResult:
        calls.append({"prompt": prompt, **kwargs})
        return daemon_client.TaskResult(text="ok", success=True, cost=0.0, tokens=0, steps=0)

    monkeypatch.setattr(daemon_client, "run", capture_run)
    return calls


def test_every_agent_spelling_resolves_to_a_script_path() -> None:
    for spelling in ("ntfy", "NTFY", "cron", "sh", ""):
        resolved = resolve_agent(spelling, "")
        assert isinstance(resolved, tuple), resolved
        path, name = resolved
        assert Path(path).is_file() and path.endswith(".py"), spelling
        assert name == (spelling.lower() or "sorcar")
    assert Path(resolve_agent("ntfy", "")[0]).parts[-2:] == ("ntfy", "ntfy_sea.py")
    assert isinstance(resolve_agent("no-such-agent", ""), str)


def test_work_dir_option_and_script_work_dir(
    tmp_path: Path, captured: list[dict[str, Any]],
) -> None:
    caller = tmp_path / "caller"
    (caller / "sub").mkdir(parents=True)
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    plain = _write(caller / "plain.py", PLAIN_SEA)
    pinning = _write(caller / "pinning.py", f"""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {{'work_dir': {str(pinned)!r}}}
""")
    run_agent = agent_dispatch.make_run_agent_tool(str(caller))
    run_agent("t", str(plain))
    assert captured[-1]["work_dir"] == str(caller)
    run_agent("t", str(plain), options='{"work_dir": " sub "}')
    assert captured[-1]["work_dir"] == str(caller / "sub")
    assert run_agent("t", str(plain), options=json.dumps({"work_dir": str(tmp_path)})) != ""
    assert captured[-1]["work_dir"] == str(tmp_path)
    # The explicit option wins over the script's work_dir; the script's
    # applies when the option is absent, and a locked work_dir refuses the option.
    run_agent("t", str(pinning), options='{"work_dir": "sub"}')
    assert captured[-1]["work_dir"] == str(caller / "sub")
    run_agent("t", str(pinning))
    assert captured[-1]["work_dir"] == str(pinned)
    locking = _write(caller / "locking.py", f"""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {{'work_dir': {str(pinned)!r}, 'locked': ['work_dir']}}
""")
    out = run_agent("t", str(locking), options='{"work_dir": "sub"}')
    assert out.startswith("Error: locking: the script locks work_dir="), out
    assert "(asked for 'sub')" in out
    # The wire spellings are not option keys (they name the key they
    # were renamed to); the ``add_to_*`` ones are forwarded.
    out = run_agent("t", str(plain), options='{"append_to_system_prompt": "x"}')
    assert out == (
        "Error: options key 'append_to_system_prompt' was renamed to "
        "'add_to_system_prompt'; use the new name."
    )
    run_agent("t", str(plain), options='{"add_to_system_prompt": "S", "add_to_prompt": "P"}')
    assert captured[-1]["add_to_system_prompt"] == "S"
    assert captured[-1]["add_to_prompt"] == "P"
    # ``workspace`` travels as its own wire field; nothing is held here.
    run_agent("t", "ntfy", options='{"workspace": "acct-2"}')
    assert captured[-1]["workspace"] == "acct-2"
    assert Path(captured[-1]["extension_agent_path"]).parts[-2:] == ("ntfy", "ntfy_sea.py")
    assert captured[-1]["work_dir"] == str(kiss_home() / "channel_work")
    assert "KISS_CHANNEL_WORKSPACE" not in os.environ


def test_is_parallel_is_inherited_from_the_calling_agent(
    tmp_path: Path, captured: list[dict[str, Any]],
) -> None:
    parent = SorcarAgent("parent")
    parent.model_name = "gpt-6-astra"
    parent._is_parallel = False
    inherited = inherit_from_parent(parent, "", None, RunOptions())
    assert inherited.options.allow_fan_out is False
    explicit = inherit_from_parent(parent, "", None, RunOptions(allow_fan_out=True))
    assert explicit.options.allow_fan_out is True
    assert inherit_from_parent(None, "", None, RunOptions()).options.allow_fan_out is None
    plain = _write(tmp_path / "plain.py", PLAIN_SEA)
    run_agent = agent_dispatch.make_run_agent_tool(str(tmp_path), parent)
    run_agent("t", str(plain))
    assert captured[-1]["allow_fan_out"] is False
    run_agent("t", str(plain), options='{"allow_fan_out": true}')
    assert captured[-1]["allow_fan_out"] is True
    # Without a parent (standalone use) the daemon default — fan-out on — stands.
    agent_dispatch.make_run_agent_tool(str(tmp_path))("t", str(plain))
    assert captured[-1]["allow_fan_out"] is True


# ---------------------------------------------------------------------------
# Daemon: picker base, workspace hold, run_parallel(agent=)
# ---------------------------------------------------------------------------


class SeaCompositionDaemonTest(DaemonRunApiHarness):
    """Runs on a real local daemon with the executor LLM loop replaced by a recorder."""

    def setUp(self) -> None:
        super().setUp()
        sea_commands._reset_for_tests()
        self.folder = Path(self.tmpdir) / "user_seas"

    def tearDown(self) -> None:
        sea_commands._reset_for_tests()
        super().tearDown()

    def _register(self) -> None:
        home = kiss_home()
        home.mkdir(parents=True, exist_ok=True)
        (home / "SEAS.md").write_text(f"{self.folder}\n", encoding="utf-8")
        self.addCleanup((home / "SEAS.md").unlink, missing_ok=True)
        sea_commands.refresh_registry()

    def _record_runs(self, runs: list[dict[str, Any]]) -> None:
        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            if kwargs.get("is_agentic") is False:
                return ""
            arguments = dict(kwargs.get("arguments") or {})
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0001
            self_agent.step_count = 1
            if "task_description" not in arguments:
                return "result: prior progress\n"
            runs.append({
                "model_name": kwargs.get("model_name"),
                "system_prompt": str(kwargs.get("system_prompt") or ""),
                "prompt": str(arguments["task_description"]),
                "tool_names": sorted(t.__name__ for t in (kwargs.get("tools") or [])),
                "workspace": os.environ.get(channel_workspace.WORKSPACE_ENV_VAR),
            })
            raw = "success: true\nis_continue: false\nsummary: agent ok\n"
            printer = kwargs.get("printer")
            if printer is not None:  # pragma: no branch
                printer.print(
                    raw, type="result", step_count=1, total_tokens=1, cost="$0.0001",
                )
            return raw

        KISSAgent.run = stub_run  # type: ignore[assignment,method-assign]

    def _run(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        self._record_runs(runs)
        result = sorcar.run(
            prompt, work_dir=self.repo, use_worktree=False, auto_commit=False,
            endpoint_file=self.endpoint_file, timeout=60, **kwargs,
        )
        assert result.success is True, result
        assert len(runs) == 1, runs
        return runs[0]

    def _channel_source(self, extra_methods: str = "") -> str:
        return f"""
import os
from kiss.agents.seas.base.base_sea import BaseSea

class Chan(BaseSea):
    def description(self):
        return 'c'
    def settings(self, settings):
        return settings | {{'kind': 'channel', 'work_dir': {self.repo!r}}}
{extra_methods}"""

    def test_slash_command_on_a_derived_sea_runs_the_whole_chain(self) -> None:
        _write(self.folder / "basey" / "basey_sea.py", BASE_SEA)
        _write(
            self.folder / "derived" / "derived_sea.py", DERIVED_SEA_TEMPLATE.format(base="basey"),
        )
        self._register()
        run = self._run("/derived do it")
        assert run["prompt"].startswith("# Task\n[derived] [base] do it"), run["prompt"]
        assert "BASE-ADD" in run["prompt"] and "DERIVED-ADD" in run["prompt"]
        assert run["system_prompt"].startswith("BASE SYSTEM")
        assert "BASE PROTOCOL" in run["system_prompt"]
        assert "DERIVED PROTOCOL" in run["system_prompt"]
        assert "base_tool" in run["tool_names"] and "shared" in run["tool_names"]
        assert "run_parallel" not in run["tool_names"]  # the worker preset

    def test_channel_run_holds_its_workspace_for_the_run_and_releases_it(self) -> None:
        # ``tools()`` runs with the run's workspace active (a channel
        # agent loads its credentials there), so the tool it builds is
        # named after the workspace it saw.
        chan = _write(self.folder / "chan" / "chan_sea.py", self._channel_source(f"""
    def tools(self, tools):
        def probe() -> str:
            '''Probe.'''
            return 'x'
        probe.__name__ = 'ws_' + str(os.environ.get({channel_workspace.WORKSPACE_ENV_VAR!r}))
        return tools + [probe]
"""))
        assert channel_workspace.WORKSPACE_ENV_VAR not in os.environ
        run = self._run("hello", extension_agent_path=str(chan), workspace="acct-7")
        assert run["workspace"] == "acct-7"
        assert "ws_acct-7" in run["tool_names"], run["tool_names"]
        assert CHANNEL_PREAMBLE.format(name="chan") in run["system_prompt"]
        assert channel_workspace.WORKSPACE_ENV_VAR not in os.environ
        run = self._run("hello", extension_agent_path=str(chan))
        assert run["workspace"] == "default"
        assert "ws_default" in run["tool_names"], run["tool_names"]
        assert channel_workspace.WORKSPACE_ENV_VAR not in os.environ
        # A client-sent daemon-side field is dropped, never honoured.
        plain = _write(self.folder / "plain_sea.py", PLAIN_SEA)
        run = self._run("hello", extension_agent_path=str(plain), workspace="acct-7")
        assert run["workspace"] is None

    def test_conflicting_workspace_fails_within_the_bounded_wait(self) -> None:
        chan = _write(self.folder / "chan" / "chan_sea.py", self._channel_source())
        from kiss.server import task_runner

        original = task_runner.WORKSPACE_WAIT_TIMEOUT_SECONDS
        task_runner.WORKSPACE_WAIT_TIMEOUT_SECONDS = 0.2
        self.addCleanup(setattr, task_runner, "WORKSPACE_WAIT_TIMEOUT_SECONDS", original)
        assert channel_workspace.enter_workspace("other")
        try:
            runs: list[dict[str, Any]] = []
            self._record_runs(runs)
            result = sorcar.run(
                "hello", work_dir=self.repo, use_worktree=False, auto_commit=False,
                extension_agent_path=str(chan), workspace="mine",
                endpoint_file=self.endpoint_file, timeout=60,
            )
            assert result.success is False
            assert "workspace 'mine' could not be activated" in result.text, result
            assert runs == []
            assert os.environ[channel_workspace.WORKSPACE_ENV_VAR] == "other"
        finally:
            channel_workspace.exit_workspace("other")
