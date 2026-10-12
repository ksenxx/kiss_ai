# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""U2–U5 of the second SEA / run_agent reassessment
(``reports/sea-run-agent-semantics-reassessment-2-2026-10-05.html``).

U2: ``is_explicit`` reads a run command's ``provenance`` field (the
explicit-beats-classifier rule itself went with the worktree setting).
U3: a relative ``work_dir`` in a SEA's ``settings()`` is relative to the
SEA's own folder.
U4: the inferred ``review`` profile is marked on the ``ran:`` line and
``agent="reviewer"`` is refused with the spelling that means it.
U5: tool-profile aliases and close-match suggestions, renamed and
removed option keys, and tool-provided hints for stray keywords.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, daemon_client
from kiss.agents.sorcar.agent_dispatch import (
    OPTION_DOCS,
    make_run_agent_tool,
    options_keyword_hint,
    parse_run_options,
    resolve_agent,
)
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.run_config import RUN_CONFIG_KEYS, is_explicit, run_config_line
from kiss.agents.sorcar.sea_apply import apply_sea, calling_work_dir
from kiss.agents.sorcar.sea_commands import (
    declared_settings,
    own_settings,
    sea_layers,
    sea_settings,
)
from kiss.agents.sorcar.sea_docs import options_table
from kiss.agents.sorcar.sea_settings import (
    PROFILE_ALIASES,
    REMOVED_SETTINGS,
    RENAMED_SETTINGS,
    SETTING_DOCS,
    SeaError,
    anchored_work_dir,
    locked_conflicts,
    resolve_settings,
)
from kiss.agents.sorcar.sorcar_agent import (
    TOOL_PROFILES,
    canonical_tool_profile,
    resolve_tool_profile,
)
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.tests.agents.sorcar.test_tool_profiles import _bare_agent


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every dispatch instead of sending it to a daemon."""
    calls: list[dict[str, Any]] = []

    def capture_run(prompt: str, **kwargs: Any) -> daemon_client.TaskResult:
        calls.append({"prompt": prompt, **kwargs})
        return daemon_client.TaskResult(text="ok", success=True, cost=0.0, tokens=0, steps=0)

    monkeypatch.setattr(daemon_client, "run", capture_run)
    return calls


def test_is_explicit_reads_the_provenance_field() -> None:
    assert is_explicit({"max_budget": "explicit"}, "max_budget")
    assert not is_explicit({"max_budget": "inherited"}, "max_budget")
    assert not is_explicit({"model": "explicit"}, "max_budget")
    assert not is_explicit(None, "max_budget") and not is_explicit("explicit", "max_budget")
    assert "tool_profile_inferred" in RUN_CONFIG_KEYS


# --- U3: a relative work_dir is a path under the calling task's directory ------------


def test_relative_work_dir_is_under_the_calling_task_everywhere(
    tmp_path: Path, captured: list[dict[str, Any]],
) -> None:
    caller = tmp_path / "caller"
    (caller / "sub").mkdir(parents=True)
    folder = tmp_path / "seas" / "box"
    sea = _write(folder / "box_sea.py", """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'work_dir': 'sandbox'}
""")
    # The setting is kept as written at every level of the fold: the
    # launcher anchors it, with one rule for settings and options.
    assert own_settings(sea_layers(sea)[-1])["work_dir"] == "sandbox"
    assert declared_settings(sea_layers(sea))["work_dir"] == "sandbox"
    assert sea_settings(sea)["work_dir"] == "sandbox"
    assert anchored_work_dir("sandbox", str(caller)) == str(caller / "sandbox")
    assert anchored_work_dir("../shared", str(caller)) == str(caller / ".." / "shared")
    assert anchored_work_dir("~", str(caller)) == str(Path.home())
    assert anchored_work_dir(str(folder), str(caller)) == str(folder)
    # run_agent: the SEA's relative setting and the call's relative option
    # both land under the CALLER's directory, whichever caller runs it.
    run_agent = make_run_agent_tool(str(caller))
    run_agent("t", str(sea))
    assert captured[-1]["work_dir"] == str(caller / "sandbox")
    run_agent("t", str(sea), options='{"work_dir": "sub"}')
    assert captured[-1]["work_dir"] == str(caller / "sub")
    run_agent = make_run_agent_tool(str(tmp_path))
    run_agent("t", str(sea))
    assert captured[-1]["work_dir"] == str(tmp_path / "sandbox")
    # The daemon stages the same directory for a ``/box`` run (the tab's
    # ``workDir`` is the calling directory) and for a dispatched run
    # (``tabScopeWorkDir`` is the caller's; ``workDir`` is already resolved).
    cmd: dict[str, Any] = {"seaPath": str(sea), "prompt": "p", "workDir": str(caller)}
    assert "workDir" in apply_sea(cmd) and cmd["workDir"] == str(caller / "sandbox")
    cmd = {
        "seaPath": str(sea), "prompt": "p", "workDir": str(caller / "sandbox"),
        "tabScopeWorkDir": str(caller),
    }
    apply_sea(cmd)
    assert cmd["workDir"] == str(caller / "sandbox")
    assert calling_work_dir({"workDir": "/w"}) == "/w" and calling_work_dir({}) == ""
    assert calling_work_dir({"tabScopeWorkDir": "/c", "workDir": "/w"}) == "/c"
    # A relative work_dir inherited from a base class is the same path under
    # the caller as the subclass's own would be: the file that set it does
    # not matter.
    base = _write(
        tmp_path / "base" / "base_sea.py", """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'work_dir': 'data'}
""",
    )
    child = _write(folder / "child_sea.py", f"""
from kiss.agents.sorcar.sea_commands import sea_class

class Child(sea_class({str(base)!r})):
    pass
""")
    assert sea_settings(child)["work_dir"] == "data"
    run_agent("t", str(child))
    assert captured[-1]["work_dir"] == str(tmp_path / "data")
    assert "path under the calling task's directory" in SETTING_DOCS["work_dir"]
    assert "a path under the calling task's directory" in OPTION_DOCS["work_dir"]


def test_relative_work_dir_that_a_script_locks_compares_resolved(
    tmp_path: Path, captured: list[dict[str, Any]],
) -> None:
    folder = tmp_path / "seas"
    (tmp_path / "sandbox").mkdir(parents=True)
    sea = _write(
        folder / "locked_sea.py",
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'work_dir': 'sandbox', 'locked': ['work_dir']}
""",
    )
    run_agent = make_run_agent_tool(str(tmp_path))
    out = run_agent("t", str(sea), options=json.dumps({"work_dir": str(tmp_path / "sandbox")}))
    assert not out.startswith("Error:"), out
    assert captured[-1]["work_dir"] == str(tmp_path / "sandbox")
    out = run_agent("t", str(sea), options='{"work_dir": "sandbox"}')
    assert not out.startswith("Error:"), out
    out = run_agent("t", str(sea), options='{"work_dir": "elsewhere"}')
    assert out.startswith("Error: locked: the script locks work_dir="), out


# --- U4: the inferred review profile is marked; "reviewer" is not an agent -------------


def test_run_config_line_marks_an_inferred_review_profile() -> None:
    named = {"sea": "", "tool_profile": "review", "max_budget": 0.3}
    assert "tools=review budget=$0.30" in run_config_line(named)
    inferred = {**named, "tool_profile_inferred": True}
    assert "tools=review(inferred) budget=$0.30" in run_config_line(inferred)
    assert run_config_line({}).endswith("inherited=none pinned=none")


def test_reviewer_sub_agent_payload_marks_the_inferred_profile(tmp_path: Path) -> None:
    saved = DEFAULT_CONFIG.tool_profiles
    DEFAULT_CONFIG.tool_profiles = True
    try:
        agent = _bare_agent(tmp_path, _tool_profile_name="")
        agent._subagent_info = {"parent_task_id": "p", "parent_tab_id": "t", "reviewer": True}

        def payload(agent: ChatSorcarAgent, task: str) -> dict[str, object]:
            return agent._task_settings_payload(
                model="m", work_dir=str(tmp_path), is_parallel=False,
                max_budget=None, start_ts=0, task_id="x", tool_profile=agent._tool_profile(task),
            )

        got = payload(agent, "Review the diff; report issues only.")
        assert got["tool_profile"] == "review" and got["tool_profile_inferred"] is True
        # The rule did not fire (the task asks for changes): nothing to mark.
        got = payload(agent, "Fix the failing test in foo.py")
        assert got["tool_profile"] == "full" and "tool_profile_inferred" not in got
        # A named profile, alias included, is the caller's choice.
        agent._tool_profile_name = "readonly"
        got = payload(agent, "Review the diff; report issues only.")
        assert got["tool_profile"] == "review" and "tool_profile_inferred" not in got
    finally:
        DEFAULT_CONFIG.tool_profiles = saved


def test_reviewer_names_are_refused_with_the_spelling_that_means_them() -> None:
    for name in ("reviewer", "Review", "code-review", "CodeReviewer"):
        out = resolve_agent(name, "")
        assert out == (
            f"Error: {name!r} is not an agent. A reviewer is a plain sub-agent with the "
            'read-only toolset: leave agent empty and pass tool_profile="review".'
        ), name
    assert make_run_agent_tool("")("look", agent="reviewer").startswith(
        "Error: 'reviewer' is not an agent."
    )
    for name in ("assistant", "general", "analyst"):
        assert isinstance(resolve_agent(name, ""), tuple), name
    assert agent_dispatch._REVIEWER_NAMES.isdisjoint(agent_dispatch._GENERIC_AGENT_NAMES)


def test_run_agent_docstring_states_the_inferred_profile() -> None:
    doc = make_run_agent_tool("").__doc__ or ""
    assert "tools=review(inferred)" in doc
    assert "``work_dir`` (relative to this task's, as is a SEA's" in doc
    assert "SEA's folder" not in doc
    assert '``"reviewer"`` is not an agent' in doc


# --- U5: aliases, suggestions, renamed/removed keys, tool-provided keyword hints -------


def test_tool_profile_aliases_canonicalise_and_unknown_names_get_a_suggestion() -> None:
    for alias, key in PROFILE_ALIASES.items():
        assert canonical_tool_profile(alias) == key
        assert resolve_tool_profile(alias) == TOOL_PROFILES[key]
    assert canonical_tool_profile(" read_only + bash ") == "review+bash"
    assert canonical_tool_profile("") == "" and resolve_tool_profile("  ") is None
    assert parse_run_options("", tool_profile="read-only").tool_profile == "review"
    assert parse_run_options("", tool_profile=" readonly + bash ").tool_profile == "review+bash"
    # The profile is an argument only; the options key is refused.
    with pytest.raises(ValueError, match="options key 'tool_profile' is the tool_profile arg"):
        parse_run_options('{"tool_profile": "readonly"}', tool_profile="review")
    with pytest.raises(ValueError) as info:
        canonical_tool_profile("revew")
    assert str(info.value).endswith("got 'revew'. Did you mean 'review'?")
    with pytest.raises(ValueError) as info:
        canonical_tool_profile("shell+edti")
    assert str(info.value).endswith("got 'shell+edti'. Did you mean 'edit'?")
    with pytest.raises(ValueError) as info:
        canonical_tool_profile("basic")  # nothing close enough: no guess
    assert str(info.value).endswith("got 'basic'.")
    out = make_run_agent_tool("")("hi", tool_profile="revew")
    assert out.startswith("Error: tool_profile must be one of ")
    assert out.endswith("Did you mean 'review'?")


def test_a_scripts_profile_alias_is_its_key_for_locks_and_records(
    tmp_path: Path, captured: list[dict[str, Any]],
) -> None:
    assert resolve_settings(
        {"tool_profile": " readonly + bash"}
    )["tool_profile"] == "review+bash"
    sea = _write(
        tmp_path / "alias_sea.py",
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'tool_profile': 'readonly', 'locked': ['tool_profile']}
""",
    )
    assert sea_settings(sea)["tool_profile"] == "review"
    assert locked_conflicts(sea_settings(sea), {"tool_profile": "review"}) == ""
    run_agent = make_run_agent_tool(str(tmp_path))
    for spelling in ("review", "readonly", "read_only"):
        out = run_agent("t", str(sea), tool_profile=spelling)
        assert not out.startswith("Error:"), (spelling, out)
        assert captured[-1]["tool_profile"] == "review", spelling
    out = run_agent("t", str(sea), tool_profile="bash")
    assert out.startswith("Error: alias: the script locks tool_profile='review' (asked for 'bash')")


def test_the_options_table_states_the_one_relative_rule_for_work_dir() -> None:
    row = next(line for line in options_table().splitlines() if line.startswith("| `work_dir` |"))
    assert "a path under the calling task's directory" in row
    assert "as a SEA's own `work_dir` setting is" in row
    assert "settings` key" not in row


def test_unknown_option_keys_name_their_new_name_or_the_closest_key() -> None:
    run_agent = make_run_agent_tool("")
    assert run_agent("hi", options='{"model_name": "m"}') == (
        "Error: options key 'model_name' was renamed to 'model'; use the new name."
    )
    assert run_agent("hi", options='{"append_to_prompt": "x"}') == (
        "Error: options key 'append_to_prompt' was renamed to 'add_to_prompt'; use the new name."
    )
    assert run_agent("hi", options='{"append_basic_tools": false}') == (
        "Error: options key 'append_basic_tools' was removed: "
        f"{REMOVED_SETTINGS['append_basic_tools']}."
    )
    assert run_agent("hi", options='{"use_worktree": false}') == (
        "Error: options key 'use_worktree' was removed: "
        f"{REMOVED_SETTINGS['use_worktree']}."
    )
    out = run_agent("hi", options='{"use_web_tool": false}')
    assert out.startswith("Error: options has an unknown key 'use_web_tool'; known keys: ")
    assert out.endswith("Did you mean 'use_web_tools'?")
    out = run_agent("hi", options='{"colour": 1}')
    assert out.startswith("Error: options has an unknown key 'colour'") and "mean" not in out
    # The same names are refused in a script's settings(), with the same new name.
    for old, new in RENAMED_SETTINGS.items():
        message = f"settings\\(\\) key {old!r} was renamed to {new!r}"
        with pytest.raises(SeaError, match=message):
            resolve_settings({old: "x"})
    with pytest.raises(SeaError, match="'append_basic_tools' was removed"):
        resolve_settings({"append_basic_tools": True})


def test_stray_keywords_get_the_tools_own_hint(tmp_path: Path) -> None:
    agent = KISSAgent("u5-kwarg")
    agent.function_map = {"run_agent": make_run_agent_tool("/tmp")}
    _, out = agent._execute_tool({
        "name": "run_agent",
        "arguments": {
            "task": "hi", "use_memory": False, "max_budget": "1", "model_name": "m",
            "append_basic_tools": False, "use_worktree": False, "use_web_tool": True,
            "zzz": 1,
        },
    })
    assert out.startswith("Failed to call run_agent with ")
    lines = out.rstrip().split("\n")
    assert lines[-6:] == [
        "use_memory is a run setting, not an argument; pass it in the `options` JSON "
        "object: options='{\"use_memory\": false}'.",
        'model_name was renamed to model; pass model="m".',
        f"append_basic_tools was removed: {REMOVED_SETTINGS['append_basic_tools']}.",
        f"use_worktree was removed: {REMOVED_SETTINGS['use_worktree']}.",
        "use_web_tool is neither an argument nor an options key. Did you mean the option "
        "'use_web_tools'?",
        "zzz is neither an argument nor an options key.",
    ], out
    # ``run_parallel`` carries the same hint; a tool without one keeps the generic text.
    tools = _bare_agent(tmp_path)._get_tools()
    run_parallel = next(t for t in tools if t.__name__ == "run_parallel")
    assert run_parallel.unknown_arguments_hint is options_keyword_hint

    def other(task: str = "", options: str = "") -> str:
        return task + options

    agent.function_map = {"other": other}
    _, out = agent._execute_tool({"name": "other", "arguments": {"task": "x", "loud": True}})
    assert out.rstrip().endswith(
        "loud is not an argument; if it is a run setting, pass it in the `options` JSON "
        "object, e.g. options='{\"loud\": true}' (the accepted keys are listed under `options`)."
    )
    assert options_keyword_hint({}) == ""
