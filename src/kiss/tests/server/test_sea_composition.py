# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the composed SEA contract.

What these tests pin down (``reports/archive/sea-run-agent-2026-10-04/
sea-run-agent-semantics-and-defects-2026-10-04.md``,
proposals F1–F5):

* ``prompt(task)`` is the one prompt surface: it receives the task and
  returns the prompt body, ``{task_id}`` in it replaced by the calling
  task's id; ``settings()["prompt"]`` / ``["system_prompt"]`` /
  ``["add_to_prompt"]`` are rejected with a pointed message.
* ``settings()["extends"]`` lays a base script under a script, and a
  model-picker SEA is the outermost layer of every run on its tab:
  settings merge (later wins, ``session`` never masks a preset),
  prompt functions chain, system-prompt additions concatenate, tools
  union by name, hooks and ``system_prompt()`` come from the innermost
  layer.  A channel SEA cannot be a base; cycles are reported.
* Every ``run_agent`` spelling resolves to one script path and one
  dispatch; ``options`` accepts the settings vocabulary (``work_dir``
  included) and rejects the old ``append_to_*`` names by name; the
  ``workspace`` argument is forwarded as a wire field and HELD by the
  daemon for a ``channel``-preset run's lifetime.
* ``allow_fan_out`` is inherited from the calling agent; ``run_parallel``
  can name an agent script for its children.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, channel_workspace, daemon_client, sea_commands
from kiss.agents.sorcar.agent_dispatch import RunOptions, inherit_from_parent, resolve_agent
from kiss.agents.sorcar.sea_commands import (
    SeaScriptError,
    evaluate_sea,
    sea_layers,
    sea_settings,
)
from kiss.agents.sorcar.sea_settings import SETTING_TYPES, merge_settings, resolve_settings
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.config import kiss_home
from kiss.core.kiss_agent import KISSAgent
from kiss.server import sorcar
from kiss.server.agent_file import (
    CHANNEL_PREAMBLE,
    AgentFileError,
    apply_agent_overrides,
    load_layers,
)
from kiss.server.agent_file import channel_workspace as held_workspace
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness

BASE_SEA = textwrap.dedent('''
    def description() -> str:
        return "base"


    def base_tool(x: str) -> str:
        """Base tool."""
        return x


    def shared(x: str) -> str:
        """Shared name; the base's version."""
        return "base " + x


    def settings() -> dict:
        return {"kind": "worker", "use_web_tools": True}


    def prompt(task: str) -> str:
        return "[base] " + task + " BASE-ADD {task_id}"


    def system_prompt() -> str:
        return "BASE SYSTEM"


    def add_to_system_prompt() -> str:
        return "BASE PROTOCOL"


    def add_to_tools() -> list:
        return [base_tool, shared]


    def llm_call_hook():
        return base_hook


    def base_hook(messages):
        return messages
''')

DERIVED_SEA_TEMPLATE = textwrap.dedent('''
    def description() -> str:
        return "derived"


    def shared(x: str) -> str:
        """Shared name; the derived version wins."""
        return "derived " + x


    def settings() -> dict:
        return {{"extends": {base!r}, "max_budget": 2.0}}


    def prompt(task: str) -> str:
        return "[derived] " + task + " DERIVED-ADD"


    def add_to_system_prompt() -> str:
        return "DERIVED PROTOCOL"


    def add_to_tools() -> list:
        return [shared]
''')


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _register(tmp_path: Path, folder: Path) -> None:
    home = kiss_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()


# ---------------------------------------------------------------------------
# Settings vocabulary
# ---------------------------------------------------------------------------


def test_settings_vocabulary_has_no_prompt_keys() -> None:
    assert "prompt" not in SETTING_TYPES and "system_prompt" not in SETTING_TYPES
    assert "extends" in SETTING_TYPES
    for key in ("prompt", "system_prompt", "add_to_prompt"):
        with pytest.raises(ValueError, match=rf"settings\(\) has an unknown key '{key}'"):
            resolve_settings({"settings": lambda key=key: {key: "x"}})
    # The tool's options vocabulary is the settings vocabulary (minus the
    # keys that describe a script; the tool's own ``model``,
    # ``tool_profile``, ``max_budget`` and ``timeout`` arguments are
    # shortcuts for the options of the same name) plus ``inherit``, the
    # channel workspace and the two appended texts.
    assert set(agent_dispatch.OPTION_TYPES) == (
        set(SETTING_TYPES) - {"extends", "kind", "locked", "hidden"}
    ) | {"inherit", "workspace", "add_to_prompt", "add_to_system_prompt"}
    assert set(RunOptions.__dataclass_fields__) == set(agent_dispatch.OPTION_TYPES) | {
        "system_prompt"
    }


def test_merge_settings_later_wins_and_session_never_masks_a_preset() -> None:
    base = resolve_settings({"settings": lambda: {"kind": "worker", "use_web_tools": True}})
    derived = resolve_settings({"settings": lambda: {"max_budget": 1, "extends": "x"}})
    merged = merge_settings([base, derived])
    assert merged["kind"] == "worker"
    assert merged["use_web_tools"] is True and merged["max_budget"] == 1.0
    assert merged["allow_fan_out"] is False  # the worker preset's default survives
    assert "extends" not in merged
    assert merge_settings([]) == {"kind": "session"}
    # A derived preset replaces the base's preset and its defaults.
    channel = resolve_settings({"settings": lambda: {"kind": "channel"}})
    assert merge_settings([base, channel])["kind"] == "channel"


# ---------------------------------------------------------------------------
# Layers: extends, picker base, evaluation
# ---------------------------------------------------------------------------


def test_extends_by_command_name_and_by_path_evaluates_both_layers(tmp_path: Path) -> None:
    folder = tmp_path / "seas"
    base = _write(folder / "basey" / "basey_sea.py", BASE_SEA)
    by_name = _write(folder / "byname" / "byname_sea.py", DERIVED_SEA_TEMPLATE.format(base="basey"))
    by_path = _write(
        folder / "bypath" / "bypath_sea.py",
        DERIVED_SEA_TEMPLATE.format(base="../basey/basey_sea.py"),
    )
    _register(tmp_path, folder)
    for derived in (by_name, by_path):
        layers = sea_layers(derived)
        assert [layer.path for layer in layers] == [base.resolve(), derived]
        settings = sea_settings(derived)
        assert settings["kind"] == "worker"
        assert settings["use_web_tools"] is True and settings["max_budget"] == 2.0
        run = evaluate_sea(layers, "do it", task_id="T-1")
        assert run.prompt == "[derived] [base] do it BASE-ADD T-1 DERIVED-ADD"
        assert run.system_prompt == "BASE SYSTEM"
        assert run.add_to_system_prompt == "BASE PROTOCOL\n\nDERIVED PROTOCOL"
        assert [t.__name__ for t in run.tools] == ["base_tool", "shared"]
        assert run.tools[1]("x") == "derived x"
        assert run.llm_call_hook is not None and run.llm_call_hook([]) == []
        assert run.tool_call_hook is None
    # The same layers applied to a run command stage the prompt, the
    # merged settings and the daemon-side fields.
    cmd: dict[str, Any] = {"agentPath": str(by_name), "prompt": "do it", "parentTaskId": "T-1"}
    overridden = apply_agent_overrides(cmd)
    assert cmd["prompt"] == "[derived] [base] do it BASE-ADD T-1 DERIVED-ADD"
    assert "appendToPrompt" not in cmd
    assert cmd["systemPrompt"] == "BASE SYSTEM"
    assert cmd["appendToSystemPrompt"] == "BASE PROTOCOL\n\nDERIVED PROTOCOL"
    assert cmd["maxBudget"] == 2.0 and cmd["useWebTools"] is True
    assert cmd["llmCallHook"] is not None and "toolCallHook" not in cmd
    assert {"prompt", "systemPrompt", "tools", "llmCallHook"} <= overridden


def test_extends_errors_name_the_script(tmp_path: Path) -> None:
    folder = tmp_path / "seas"
    _register(tmp_path, folder)
    missing = _write(
        folder / "orphan" / "orphan_sea.py",
        "def description():\n    return 'o'\ndef settings():\n    return {'extends': 'nobody'}\n",
    )
    with pytest.raises(SeaScriptError, match="extends 'nobody' is not a registered SEA command"):
        sea_settings(missing)
    nofile = _write(
        folder / "nofile" / "nofile_sea.py",
        "def description():\n    return 'n'\ndef settings():\n    return {'extends': 'gone.py'}\n",
    )
    with pytest.raises(SeaScriptError, match="extends 'gone.py' is not an existing Python"):
        sea_settings(nofile)
    # A cycle through two scripts.
    _write(folder / "ping" / "ping_sea.py",
           "def description():\n    return 'p'\ndef settings():\n    return {'extends': 'pong'}\n")
    pong = _write(
        folder / "pong" / "pong_sea.py",
        "def description():\n    return 'q'\ndef settings():\n    return {'extends': 'ping'}\n",
    )
    sea_commands.refresh_registry()
    with pytest.raises(SeaScriptError, match="extends chain is a cycle"):
        sea_settings(pong)
    # A channel agent is never a base.
    _write(
        folder / "chan" / "chan_sea.py",
        "def description():\n    return 'c'\ndef settings():\n    return {'kind': 'channel'}\n",
    )
    onchan = _write(
        folder / "onchan" / "onchan_sea.py",
        "def description():\n    return 'o'\ndef settings():\n    return {'extends': 'chan'}\n",
    )
    sea_commands.refresh_registry()
    with pytest.raises(SeaScriptError, match="cannot extend the channel agent script"):
        sea_layers(onchan)
    # ... but a channel SEA may itself extend a worker base, and a
    # picker base under a channel SEA is allowed (the channel is last).
    base = _write(folder / "wbase" / "wbase_sea.py", BASE_SEA)
    chan2 = _write(folder / "chan2" / "chan2_sea.py",
                   "def description():\n    return 'c'\n"
                   "def settings():\n    return {'kind': 'channel', 'extends': 'wbase'}\n")
    sea_commands.refresh_registry()
    assert [layer.path for layer in sea_layers(chan2)] == [base, chan2]
    cmd: dict[str, Any] = {"agentPath": str(chan2), "prompt": "p", "workspace": " acct "}
    layers = load_layers(cmd)
    assert held_workspace(cmd, layers) == "acct"
    assert held_workspace({"agentPath": str(chan2)}, layers) == "default"
    assert held_workspace({"agentPath": str(base)}, load_layers({"agentPath": str(base)})) == ""
    apply_agent_overrides(cmd, layers)
    assert cmd["appendToSystemPrompt"].startswith(CHANNEL_PREAMBLE.format(name="chan2"))
    # No parent task: ``{task_id}`` becomes the empty string.
    assert cmd["prompt"] == "[base] p BASE-ADD "


def test_a_file_shared_by_the_base_and_the_extends_chain_runs_once(tmp_path: Path) -> None:
    """A picker base the script also extends (or a common ancestor) is one layer, executed once."""
    folder = tmp_path / "seas"
    tally = tmp_path / "tally.txt"
    counted = (
        "from pathlib import Path\n"
        f"Path({str(tally)!r}).open('a').write('{{name}}\\n')\n"
        "def description():\n    return '{name}'\n"
        "def settings():\n    return {settings}\n"
        "def prompt(task):\n    return '[{name}] ' + task\n"
    )
    root = _write(folder / "root" / "root_sea.py", counted.format(name="root", settings="{}"))
    picker = _write(
        folder / "picker" / "picker_sea.py",
        counted.format(name="picker", settings="{'extends': 'root'}"),
    )
    direct = _write(
        folder / "direct" / "direct_sea.py",
        counted.format(name="direct", settings="{'extends': 'picker'}"),
    )
    sibling = _write(
        folder / "sibling" / "sibling_sea.py",
        counted.format(name="sibling", settings="{'extends': 'root'}"),
    )
    _register(tmp_path, folder)
    sea_commands.refresh_registry()
    tally.write_text("", encoding="utf-8")
    # The base is also the script's extends: [root, picker, direct], each once.
    layers = sea_layers(direct, base=picker)
    assert [layer.path for layer in layers] == [root, picker, direct]
    assert sorted(tally.read_text().split()) == ["direct", "picker", "root"]
    assert evaluate_sea(layers, "t").prompt == "[direct] [picker] [root] t"
    # A common ancestor keeps its place under the base chain.
    tally.write_text("", encoding="utf-8")
    layers = sea_layers(sibling, base=picker)
    assert [layer.path for layer in layers] == [root, picker, sibling]
    assert sorted(tally.read_text().split()) == ["picker", "root", "sibling"]
    # The base IS the script: one layer.
    assert [layer.path for layer in sea_layers(root, base=root)] == [root]


def test_failed_reload_keeps_an_earlier_execution_resolving_its_annotations(
    tmp_path: Path,
) -> None:
    """A broken re-execution of a file does not unregister a running execution's module."""
    import typing

    sea = _write(
        tmp_path / "typed_sea.py",
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "class Item:\n    pass\n"
        "@dataclass\nclass Box:\n    item: Item\n"
        "def settings():\n    return {}\n",
    )
    first = sea_layers(sea)[0].namespace
    assert typing.get_type_hints(first["Box"])["item"] is first["Item"]
    sea.write_text("def settings(:\n", encoding="utf-8")
    with pytest.raises(SeaScriptError, match="failed to import"):
        sea_layers(sea)
    # Still resolvable after the failed reload (the module entry was restored).
    assert typing.get_type_hints(first["Box"])["item"] is first["Item"]
    # A later successful reload of the same source resolves through the new module.
    sea.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "class Item:\n    pass\n"
        "@dataclass\nclass Box:\n    item: Item\n"
        "def settings():\n    return {}\n",
        encoding="utf-8",
    )
    second = sea_layers(sea)[0].namespace
    assert typing.get_type_hints(second["Box"])["item"] is second["Item"]
    assert typing.get_type_hints(first["Box"])["item"] is second["Item"]


def test_prompt_getter_is_checked(tmp_path: Path) -> None:
    sea = _write(tmp_path / "empty_sea.py", "def prompt(task):\n    return '  '\n")
    with pytest.raises(SeaScriptError, match=r"prompt\(\) of agent script .* non-empty string"):
        evaluate_sea(sea_layers(sea), "t")
    sea = _write(tmp_path / "num_sea.py", "def prompt(task):\n    return 5\n")
    with pytest.raises(SeaScriptError, match=r"prompt\(\) of agent script .* must return a string"):
        evaluate_sea(sea_layers(sea), "t")
    sea = _write(tmp_path / "const_sea.py", "prompt = 'not callable'\n")
    with pytest.raises(SeaScriptError, match="prompt of agent script .* must be a callable"):
        evaluate_sea(sea_layers(sea), "t")
    sea = _write(tmp_path / "raises_sea.py", "def prompt(task):\n    raise KeyError('k')\n")
    with pytest.raises(SeaScriptError, match=r"prompt\(\) of agent script .* raised: KeyError"):
        evaluate_sea(sea_layers(sea), "t")
    sea = _write(tmp_path / "noprompt_sea.py", "def settings():\n    return {}\n")
    run = evaluate_sea(sea_layers(sea), "kept")
    assert run.prompt == "kept" and run.tools == []
    cmd: dict[str, Any] = {"agentPath": str(sea), "prompt": "kept"}
    assert apply_agent_overrides(cmd) == set()
    assert cmd.pop("_runConfig") == {"sea": "noprompt", "kind": "session", "pinned": {}}
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
    assert resolve_agent("ntfy", "")[0].endswith("ntfy/ntfy_sea.py")
    assert isinstance(resolve_agent("no-such-agent", ""), str)


def test_work_dir_option_and_script_work_dir(
    tmp_path: Path, captured: list[dict[str, Any]],
) -> None:
    caller = tmp_path / "caller"
    (caller / "sub").mkdir(parents=True)
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    plain = _write(caller / "plain.py", "def settings():\n    return {}\n")
    pinning = _write(
        caller / "pinning.py", f"def settings():\n    return {{'work_dir': {str(pinned)!r}}}\n",
    )
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
    locking = _write(
        caller / "locking.py",
        f"def settings():\n    return {{'work_dir': {str(pinned)!r}, 'locked': ['work_dir']}}\n",
    )
    out = run_agent("t", str(locking), options='{"work_dir": "sub"}')
    assert out.startswith("Error: locking: the script locks work_dir="), out
    assert "(asked for 'sub')" in out
    # The wire spellings are not option keys; the ``add_to_*`` ones are forwarded.
    out = run_agent("t", str(plain), options='{"append_to_system_prompt": "x"}')
    assert out.startswith("Error: options has an unknown key 'append_to_system_prompt'")
    run_agent("t", str(plain), options='{"add_to_system_prompt": "S", "add_to_prompt": "P"}')
    assert captured[-1]["append_to_system_prompt"] == "S"
    assert captured[-1]["append_to_prompt"] == "P"
    # ``workspace`` travels as its own wire field; nothing is held here.
    run_agent("t", "ntfy", options='{"workspace": "acct-2"}')
    assert captured[-1]["workspace"] == "acct-2"
    assert captured[-1]["extension_agent_path"].endswith("ntfy/ntfy_sea.py")
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
    plain = _write(tmp_path / "plain.py", "def settings():\n    return {}\n")
    run_agent = agent_dispatch.make_run_agent_tool(str(tmp_path), parent)
    run_agent("t", str(plain))
    assert captured[-1]["is_parallel"] is False
    run_agent("t", str(plain), options='{"allow_fan_out": true}')
    assert captured[-1]["is_parallel"] is True
    # Without a parent (standalone use) the daemon default — fan-out on — stands.
    agent_dispatch.make_run_agent_tool(str(tmp_path))("t", str(plain))
    assert captured[-1]["is_parallel"] is True


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

    def test_slash_command_on_an_extends_sea_runs_the_whole_chain(self) -> None:
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
        # ``add_to_tools()`` binds the account of the workspace active
        # when it runs (a channel agent loads its credentials there), so
        # the tool it builds is named after the workspace it saw.
        chan = _write(
            self.folder / "chan" / "chan_sea.py",
            "import os\n"
            "def description():\n    return 'c'\n"
            f"def settings():\n    return {{'kind': 'channel', 'work_dir': {self.repo!r}}}\n"
            "def add_to_tools():\n"
            "    def probe() -> str:\n"
            '        """Probe."""\n'
            "        return 'x'\n"
            "    probe.__name__ = 'ws_' + str(os.environ.get("
            f"{channel_workspace.WORKSPACE_ENV_VAR!r}))\n"
            "    return [probe]\n",
        )
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
        plain = _write(self.folder / "plain_sea.py", "def settings():\n    return {}\n")
        run = self._run("hello", extension_agent_path=str(plain), workspace="acct-7")
        assert run["workspace"] is None

    def test_conflicting_workspace_fails_within_the_bounded_wait(self) -> None:
        chan = _write(
            self.folder / "chan" / "chan_sea.py",
            "def description():\n    return 'c'\n"
            f"def settings():\n    return {{'kind': 'channel', 'work_dir': {self.repo!r}}}\n",
        )
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
