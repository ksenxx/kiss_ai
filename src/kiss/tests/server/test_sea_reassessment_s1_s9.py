# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The S1–S9 simplifications of ``reports/sea-run-agent-semantics-reassessment-2026-10-05.md``.

S1  one precedence sentence, ``sea_settings.PRECEDENCE_RULE``, quoted by
    both tool docstrings and rendered by ``sea docs``; ``sea lint``'s
    ``stale-prose`` rule keeps the old sentences out.
S2  the run-configuration key is ``pinned`` (an inherited or default
    value the SEA replaced), never an explicit argument.
S3  a ``channel`` SEA locks every key its kind sets.
S4  ``worker`` is a kind, not a generic agent label.
S5  ``dummy``, ``coding`` and ``oai`` are hidden; a channel is a
    third-party SEA that declares ``"kind": "channel"``.
S6  one name, "SEA", and whole argument descriptions in the tool schema.
S8  ``tool_profile`` is also an option; ``workspace`` is refused for a
    non-channel; ``run_parallel`` honours ``inherit: false``.
S9  is covered by ``tests/agents/seas/test_rsi7d_sea_tuning.py``.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import agent_dispatch, sea_commands
from kiss.agents.sorcar.agent_dispatch import (
    DEFAULT_AGENT_PATH,
    RunOptions,
    available_channels,
    make_run_agent_tool,
    parse_run_options,
    resolve_agent,
)
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.run_config import RUN_CONFIG_KEYS, run_config_line
from kiss.agents.sorcar.sea_docs import precedence_block, render
from kiss.agents.sorcar.sea_lint import PROSE_FILES, lint_prose
from kiss.agents.sorcar.sea_settings import (
    PRECEDENCE_RULE,
    declared_literal,
    declares_hidden,
    kind_defaults,
    merge_settings,
)
from kiss.core.models.model_info import model
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome


@pytest.fixture
def home() -> Iterator[IsolatedKissHome]:
    isolated = IsolatedKissHome(prefix="kiss-sea-s1s9-")
    isolated.write_config(is_worktree=False, auto_commit_mode=False, classify_tasks=False)
    try:
        yield isolated
    finally:
        isolated.cleanup()


def _bare_agent(work_dir: Path) -> ChatSorcarAgent:
    """A parent agent with the attributes ``run()`` sets before ``_get_tools``."""
    agent = ChatSorcarAgent("s1s9-parent")
    agent._use_web_tools = False
    agent._is_parallel = True
    agent._use_memory_override = None
    agent._append_basic_tools = True
    agent.web_use_tool = None
    agent._memory_tools = None
    agent.work_dir = str(work_dir)
    agent.docker_manager = None
    agent.printer = None
    return agent


# --- S1: one sentence, one source --------------------------------------------


def test_precedence_rule_is_stated_once_and_quoted_everywhere(home: IsolatedKissHome) -> None:
    assert PRECEDENCE_RULE.startswith("For every setting of a sub-task: what the call passes")
    assert "`locked`" in PRECEDENCE_RULE and "refused" in PRECEDENCE_RULE
    assert precedence_block() == "> " + PRECEDENCE_RULE
    page = "x\n<!-- sea-docs: precedence -->\nstale\n<!-- /sea-docs -->\ny\n"
    assert render(page) == (
        f"x\n<!-- sea-docs: precedence -->\n> {PRECEDENCE_RULE}\n<!-- /sea-docs -->\ny\n"
    )
    tools = {t.__name__: t for t in _bare_agent(home.repo)._get_tools()}
    for name in ("run_agent", "run_parallel"):
        doc = tools[name].__doc__ or ""
        assert PRECEDENCE_RULE in doc, name
        assert "{precedence}" not in doc and "agent script" not in doc, name


def test_stale_prose_rule_reads_prose_outside_generated_blocks(tmp_path: Path) -> None:
    page = tmp_path / PROSE_FILES[0]
    page.parent.mkdir(parents=True)
    page.write_text(
        "The SEA's settings still win over the call.\n"
        "Lives in `~/.kiss/SEAS.md`.\n"
        "<!-- sea-docs: precedence -->\n"
        "generated text may say settings win over anything\n"
        "<!-- /sea-docs -->\n"
        "Fine: $KISS_HOME/SEAS.md and the precedence rule.\n",
        encoding="utf-8",
    )
    findings = lint_prose(tmp_path)
    assert [(f.code, f.message) for f in findings] == [
        ("stale-prose", "line 1: claims a SEA's settings win over a call; state "
                        "sea_settings.PRECEDENCE_RULE instead"),
        ("stale-prose", "line 2: a `~/.kiss/` home path; write `$KISS_HOME/`"),
    ]
    assert all(f.path == page for f in findings)
    # The checkout itself is clean (the gate ``uv run check`` runs).
    assert lint_prose() == []


# --- S2: ``pinned`` --------------------------------------------------------------


def test_pinned_is_the_run_config_key_and_renders_a_reachable_example() -> None:
    assert "pinned" in RUN_CONFIG_KEYS and "overridden" not in RUN_CONFIG_KEYS
    line = run_config_line({
        "sea": "sh", "kind": "worker", "model": "gpt-5", "tool_profile": "bash",
        "max_budget": 1.0, "timeout": 3600, "inherited": ["model", "chat_id", "max_budget"],
        "pinned": {"use_worktree": [True, False]},
    })
    assert line == (
        "sh (worker) model=gpt-5 tools=bash budget=$1.00 timeout=3600s "
        "inherited=model,chat_id,max_budget pinned=use_worktree(True->False)"
    )
    assert run_config_line({}).endswith("inherited=none pinned=none")


# --- S3: a channel is closed -------------------------------------------------------


def test_channel_kind_locks_every_key_it_sets() -> None:
    channel = {**kind_defaults()["channel"], "kind": "channel", "work_dir": "/scratch"}
    merged = merge_settings([channel])  # resolved settings carry the kind's defaults
    assert merged["locked"] == sorted(kind_defaults()["channel"])
    assert set(merged["locked"]) == {
        "work_dir", "use_worktree", "auto_commit", "auto_classify", "allow_fan_out",
        "use_web_tools", "use_memory",
    }
    # A base's lock survives in the SEA that extends it; other kinds lock
    # only what they declare.
    assert merge_settings([channel, {"kind": "session", "use_memory": True}])["locked"] == (
        sorted(kind_defaults()["channel"])
    )
    worker = {**kind_defaults()["worker"], "kind": "worker"}
    assert "locked" not in merge_settings([worker])
    assert merge_settings([{**worker, "locked": ["tool_profile"]}])["locked"] == ["tool_profile"]


# --- S4 / S5: names and hidden SEAs ----------------------------------------------


def test_worker_is_a_kind_not_an_alias_and_infrastructure_seas_are_hidden() -> None:
    assert resolve_agent("", "") == (DEFAULT_AGENT_PATH, "dummy")
    assert resolve_agent("general", "") == (DEFAULT_AGENT_PATH, "dummy")
    for name in ("worker", "subagent", "helper"):
        assert str(resolve_agent(name, "")).startswith(f"Error: unknown agent '{name}'")
    commands = sea_commands.list_commands()
    for hidden in ("dummy", "coding", "oai"):
        assert hidden not in commands, hidden
    assert declares_hidden(Path(DEFAULT_AGENT_PATH))
    assert declared_literal(Path(DEFAULT_AGENT_PATH), "hidden") is True
    assert declared_literal(Path(DEFAULT_AGENT_PATH), "kind") is None
    assert declared_literal(Path("/no/such/file.py"), "kind") is None
    # A channel declares its kind literally; ``a2a`` is a command but not a channel.
    channels = available_channels()
    assert "a2a" in commands and "a2a" not in channels and "oai" not in channels
    for channel in channels:
        path = sea_commands.get_command(channel)
        assert path is not None and declared_literal(path, "kind") == "channel", channel


# --- S6: whole argument descriptions reach the model -----------------------------


def test_tool_schema_carries_whole_argument_descriptions() -> None:
    def sample(path: str, count: int = 1) -> str:
        """Read a file.

        Args:
            path: The file path to read, relative to the
                work directory; ``~`` is expanded.
            count (int): How many lines.

        Returns:
            The text.
        """
        return path * count

    from kiss.agents.sorcar.decide_tool import DEFAULT_DECISIONS_MODEL

    props = model(DEFAULT_DECISIONS_MODEL)._function_to_openai_tool(sample)["function"][
        "parameters"
    ]["properties"]
    assert props["path"]["description"] == (
        "The file path to read, relative to the work directory; ``~`` is expanded."
    )
    assert props["count"]["description"] == "How many lines."
    run_agent = make_run_agent_tool("/tmp")
    props = model(DEFAULT_DECISIONS_MODEL)._function_to_openai_tool(run_agent)["function"][
        "parameters"
    ]["properties"]
    assert props["agent"]["description"].startswith("Empty = a plain Sorcar sub-agent;")
    assert "that SEA file" in props["agent"]["description"]
    assert "``allow_fan_out``" in props["options"]["description"]


# --- S8: options -------------------------------------------------------------------


def test_tool_profile_is_an_argument_and_an_option_that_must_agree() -> None:
    assert parse_run_options("", "review").tool_profile == "review"
    assert parse_run_options('{"tool_profile": "shell"}', "").tool_profile == "shell"
    assert parse_run_options('{"tool_profile": " shell "}', "shell").tool_profile == "shell"
    with pytest.raises(ValueError, match="contradicts the tool_profile argument 'review'"):
        parse_run_options('{"tool_profile": "shell"}', "review")
    with pytest.raises(ValueError, match="tool_profile must be one of"):
        parse_run_options('{"tool_profile": "bogus"}', "")
    assert "tool_profile" in agent_dispatch.OPTION_TYPES
    assert "run_agent` only" not in agent_dispatch.OPTION_DOCS["inherit"]


def test_run_parallel_honours_inherit_false(
    home: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _bare_agent(home.repo)
    agent.model_name = "parent-model"
    agent._base_system_prompt = "PARENT PROMPT"

    def helper() -> str:
        """A tool the parent carries."""
        return "x"

    agent._extra_tools = [helper]
    fanned: list[dict[str, Any]] = []

    def fake_fanout(tasks: list[str], **kwargs: Any) -> list[str]:
        fanned.append(kwargs)
        return ["- success: true\n  summary: ok"] * len(tasks)

    monkeypatch.setattr("kiss.agents.sorcar.sorcar_agent.run_tasks_parallel", fake_fanout)
    monkeypatch.setattr(agent, "reclaim_abandoned_subagents", lambda: None)
    run_parallel = next(t for t in agent._get_tools() if t.__name__ == "run_parallel")
    assert "success: true" in run_parallel('["a"]')
    assert "success: true" in run_parallel('["a"]', options='{"inherit": false}')
    inherited, bare = fanned
    assert inherited["model_name"] == "parent-model"
    assert inherited["base_system_prompt"] == "PARENT PROMPT"
    assert inherited["inherited_tools"] == [helper]
    assert bare["model_name"] is None
    assert bare["base_system_prompt"] == ""
    assert bare["inherited_tools"] == []
    # Both are threads of this task: the budget share is never withheld.
    assert "max_budget" in bare and bare["max_budget"] == inherited["max_budget"]
    assert inspect.signature(run_parallel).parameters["options"].default == ""


def test_workspace_is_refused_for_a_non_channel_sea(tmp_path: Path) -> None:
    (tmp_path / "helper.py").write_text("def settings() -> dict:\n    return {}\n")
    run_agent = make_run_agent_tool(str(tmp_path))
    assert run_agent("hi", "helper.py", options='{"workspace": "acct"}') == (
        "Error: helper: options['workspace'] applies to a channel agent only; helper is a "
        "session SEA"
    )
    assert RunOptions().workspace == ""
