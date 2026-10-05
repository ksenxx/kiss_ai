# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""T1-T7 of ``reports/sea-run-agent-semantics-reassessment-2026-10-05.html``.

T1  ``options`` is the whole settings vocabulary: ``model``, ``max_budget``
    and ``timeout`` are options, the tool arguments are shortcuts that must
    agree with them, a renamed key names its new name, and a setting passed
    as a tool keyword is pointed at ``options``.
T2  ``/ask`` and ``/sh`` lock ``tool_profile``.
T3  the plain sub-agent SEA is ``seas/sorcar/sorcar_sea.py``.
T4  one ``timeout`` sentence: argument or option > setting > default
    (3600 s for ``run_agent``, no own limit for a ``run_parallel`` child).
T5  the precedence example is generated from ``/sh``; ``sea lint`` checks
    every "``/name`` declares ``{...}``" claim against the script.
T6  the ``ran:`` line of a path dispatch names the registered command.
T7  ``hidden`` documents why it must be the literal ``True``.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar import agent_dispatch, sea_commands
from kiss.agents.sorcar.agent_dispatch import (
    DEFAULT_AGENT_PATH,
    OPTION_TYPES,
    RunOptions,
    _run_agent,
    command_alias,
    is_agent_path,
    make_run_agent_tool,
    parse_run_options,
    resolve_agent,
    resolve_timeout,
)
from kiss.agents.sorcar.run_config import run_config_line
from kiss.agents.sorcar.sea_docs import GENERATED_FILES, REPO_ROOT, precedence_block, render
from kiss.agents.sorcar.sea_lint import lint_prose
from kiss.agents.sorcar.sea_settings import META_SETTINGS, SETTING_DOCS, SETTING_TYPES
from kiss.agents.sorcar.task_metadata import sea_name_of_agent
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import get_available_models
from kiss.tests.server.test_run_agent_subagent_tab import DaemonLocalHarness
from kiss.tests.server.test_run_config_echo import _parent

# --- T1: options = settings vocabulary ---------------------------------------------


def test_options_vocabulary_is_the_settings_vocabulary_minus_the_script_keys() -> None:
    assert set(OPTION_TYPES) == (set(SETTING_TYPES) - set(META_SETTINGS)) | {
        "inherit", "workspace", "add_to_prompt", "add_to_system_prompt",
    }
    assert {"model", "max_budget", "timeout", "tool_profile"} <= set(OPTION_TYPES)
    assert set(RunOptions.__dataclass_fields__) == set(OPTION_TYPES) | {"system_prompt"}


def test_arguments_are_shortcuts_for_the_options_of_the_same_name() -> None:
    by_options = parse_run_options(
        '{"model": "gpt-5", "max_budget": 2, "timeout": 30, "tool_profile": "review"}'
    )
    by_arguments = parse_run_options("", "review", "gpt-5", "2", "30")
    assert by_options == by_arguments
    assert by_options.model == "gpt-5" and by_options.max_budget == 2.0
    assert by_options.timeout == 30.0 and by_options.tool_profile == "review"
    # Both ways at once is fine when they agree, an error when they differ.
    assert parse_run_options('{"model": " gpt-5 "}', model="gpt-5").model == "gpt-5"
    with pytest.raises(ValueError, match=r"options\['model'\] = 'a' contradicts the model arg"):
        parse_run_options('{"model": "a"}', model="b")
    # A blank option is "not passed", so it never contradicts the argument.
    assert parse_run_options('{"model": "", "tool_profile": " "}', "review", "m").model == "m"
    assert parse_run_options('{"tool_profile": ""}', "review").tool_profile == "review"
    assert parse_run_options('{"add_to_prompt": "  "}') == RunOptions()
    # A JSON integer too large for a float is "not a number", not a crash.
    with pytest.raises(ValueError, match=r"options\['timeout'\] must be a number, got 1000"):
        parse_run_options('{"timeout": ' + "1" + "0" * 400 + "}")
    with pytest.raises(ValueError, match=r"options\['timeout'\] = 10.0 contradicts the timeout"):
        parse_run_options('{"timeout": 10}', timeout="20")
    with pytest.raises(ValueError, match=r"options\['max_budget'\] = 1.0 contradicts"):
        parse_run_options('{"max_budget": 1}', max_budget="2")
    # Numbers are validated once, whichever way they come.
    for text, bad in [("max_budget", "cheap"), ("timeout", "soon")]:
        with pytest.raises(ValueError, match=f"{text} must be a number, got '{bad}'"):
            parse_run_options("", **{text: bad})
    for key in ("max_budget", "timeout"):
        for bad in ("0", "-1", "inf", "nan"):
            with pytest.raises(ValueError, match=f"{key} must be a positive finite number"):
                parse_run_options("", **{key: bad})
        with pytest.raises(ValueError, match=rf"options\['{key}'\] must be a positive finite"):
            parse_run_options(f'{{"{key}": 0}}')
        with pytest.raises(ValueError, match=rf"options\['{key}'\] must be a number, got True"):
            parse_run_options(f'{{"{key}": true}}')
        with pytest.raises(ValueError, match=rf"options\['{key}'\] must be a number, got 'x'"):
            parse_run_options(f'{{"{key}": "x"}}')
    # ``null`` and empty strings mean "not passed".
    assert parse_run_options('{"model": null, "timeout": null}', model=" ") == RunOptions()


def test_a_renamed_setting_in_options_names_its_new_name() -> None:
    run_agent = make_run_agent_tool("/tmp")
    assert run_agent("say hi", options='{"is_parallel": false}') == (
        "Error: options key 'is_parallel' was renamed to 'allow_fan_out'; use the new name."
    )
    assert run_agent("say hi", options='{"preset": "worker"}') == (
        "Error: options key 'preset' was renamed to 'kind'; use the new name."
    )
    assert run_agent("say hi", options='{"colour": 1}').startswith(
        "Error: options has an unknown key 'colour'; known keys: work_dir, model, chat_id, "
    )
    assert run_agent("say hi", max_budget="cheap") == (
        "Error: max_budget must be a number, got 'cheap'."
    )
    assert run_agent("say hi", timeout="-5") == (
        "Error: timeout must be a positive finite number, got '-5'."
    )
    assert run_agent("say hi", options='{"model": "a"}', model="b") == (
        "Error: options['model'] = 'a' contradicts the model argument 'b'; pass one of them."
    )


def test_a_setting_passed_as_a_tool_keyword_is_pointed_at_options() -> None:
    agent = KISSAgent("t1-kwarg")
    agent.function_map = {"run_agent": make_run_agent_tool("/tmp")}
    _, out = agent._execute_tool(
        {"name": "run_agent", "arguments": {"task": "hi", "use_worktree": False, "model": "m"}}
    )
    assert out.startswith("Failed to call run_agent with ")
    assert "Expected signature: run_agent(task: str, agent: str = ''" in out
    # The tool knows its vocabulary: the hint is definite, not conditional.
    assert out.rstrip().endswith(
        "use_worktree is a run setting, not an argument; pass it in the `options` JSON "
        "object: options='{\"use_worktree\": false}'."
    )
    # A tool without an ``options`` parameter keeps the plain signature message.
    agent.function_map = {"echo": lambda text="": text}
    _, out = agent._execute_tool({"name": "echo", "arguments": {"text": "x", "loud": True}})
    assert "unexpected keyword argument 'loud'" in out and "`options`" not in out


def test_run_agent_docstring_names_the_shortcut_options() -> None:
    doc = " ".join((make_run_agent_tool("/tmp").__doc__ or "").split())
    assert (
        "keys are the SEA settings vocabulary: ``model``, ``tool_profile``, ``max_budget``, "
        "``timeout`` (the four arguments above are shortcuts for these)"
    ) in doc
    assert 'ends with ``(also agent="name")``' in doc


# --- T2: /ask and /sh lock tool_profile ---------------------------------------------


def test_ask_and_sh_lock_their_tool_profile() -> None:
    for name, profile in [("sh", "bash"), ("ask", "none")]:
        path = sea_commands.get_command(name)
        assert path is not None
        declared = sea_commands.sea_getter_value(path, "settings")
        assert declared["tool_profile"] == profile and declared["locked"] == ["tool_profile"]
        run_agent = make_run_agent_tool("/tmp")
        assert run_agent("ls", agent=name, tool_profile="review") == (
            f"Error: {name}: the script locks tool_profile='{profile}' (asked for 'review')"
        )
        assert run_agent("ls", agent=name, options='{"tool_profile": "shell"}') == (
            f"Error: {name}: the script locks tool_profile='{profile}' (asked for 'shell')"
        )


# --- T3: the plain sub-agent is seas/sorcar/sorcar_sea.py ---------------------------


def test_the_plain_sub_agent_sea_is_named_sorcar() -> None:
    default = Path(DEFAULT_AGENT_PATH)
    assert default.parts[-4:] == ("agents", "seas", "sorcar", "sorcar_sea.py")
    assert default.is_file() and not (default.parents[1] / "dummy").exists()
    assert sea_commands.sea_getter_value(default, "settings") == {"hidden": True}
    assert "`agent=\"sorcar\"`" in sea_commands.sea_getter_value(default, "description")
    for spelling in ("", "sorcar", "general", "assistant"):
        assert resolve_agent(spelling, "") == (DEFAULT_AGENT_PATH, "sorcar"), spelling
    # A reviewer is a toolset, not an agent: the name is refused with the spelling.
    assert resolve_agent("reviewer", "") == (
        "Error: 'reviewer' is not an agent. A reviewer is a plain sub-agent with the "
        'read-only toolset: leave agent empty and pass tool_profile="review".'
    )
    assert sea_commands.get_command("sorcar") is None  # hidden: no /sorcar command
    assert sea_name_of_agent("", []) == "sorcar_sea"
    for rel in ("src/kiss/server/README.md", "src/kiss/agents/third_party_agents/README.md"):
        text = (REPO_ROOT / rel).read_text("utf-8")
        assert "dummy_sea" not in text and "seas/dummy" not in text, rel


# --- T4: one timeout sentence -------------------------------------------------------


def test_timeout_has_one_sentence_and_one_resolution() -> None:
    doc = SETTING_DOCS["timeout"]
    assert doc.startswith(
        "Seconds the call blocks for the run: the call's `timeout` argument or option wins"
    )
    assert "3600 for a `run_agent` call" in doc
    assert "keeps going as an `agent_job`" in doc
    assert "no limit of its own for a `run_parallel` child" in doc
    assert "ignored by `/<name>`" in doc
    assert resolve_timeout(None, {}) == 3600.0
    assert resolve_timeout(None, {"timeout": 10}) == 10.0
    assert resolve_timeout(5.0, {"timeout": 10}) == 5.0
    assert parse_run_options('{"timeout": 7}').timeout == 7.0
    assert parse_run_options("", timeout="7").timeout == 7.0


# --- T5: the precedence example is generated; claims are linted ---------------------


def test_precedence_block_carries_a_generated_example_and_the_pages_are_current() -> None:
    block = precedence_block()
    assert "> For example, `/sh` declares " in block
    assert (
        "is refused with `Error: sh: the script locks tool_profile='bash' (asked for 'review')`"
    ) in block
    assert 'model="gpt-5")` runs it with that model' in block
    for rel in GENERATED_FILES:
        text = (REPO_ROOT / rel).read_text("utf-8")
        assert render(text) == text, rel
        assert block in text, rel


def _prose_root(tmp_path: Path, sentence: str) -> Path:
    page = tmp_path / "website/kisssorcar.github.io/docs/sea-commands.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "# page\n\n" + sentence + "\n\n<!-- sea-docs: precedence -->\n"
        "`/sh` declares `{}`\n<!-- /sea-docs -->\n",
        "utf-8",
    )
    return tmp_path


def test_lint_checks_declares_claims_against_the_script(tmp_path: Path) -> None:
    sh = '`{"kind": "worker", "tool_profile": "bash", "locked": ["tool_profile"]}`'
    # A true claim, in either word order, passes; the generated block is never read.
    assert lint_prose(_prose_root(tmp_path / "a", f"`/sh` declares {sh}.")) == []
    assert lint_prose(_prose_root(tmp_path / "b", f"for example {sh} (what `/sh` declares)")) == []
    # A Python literal is read like JSON.
    literal = "`{'kind': 'worker', 'tool_profile': 'bash', 'locked': ['tool_profile']}`"
    assert lint_prose(_prose_root(tmp_path / "c", f"`/sh` declares {literal}.")) == []
    # A stale claim names the current literal.
    [finding] = lint_prose(
        _prose_root(tmp_path / "d", '`/sh` declares `{"kind": "worker", "tool_profile": "bash"}`.')
    )
    assert finding.code == "stale-prose"
    assert finding.message == f"line 3: `/sh` declares {sh}"
    [finding] = lint_prose(
        _prose_root(tmp_path / "e", '`{"tool_profile": "none"}` (what `/ask` declares)')
    )
    assert finding.message == (
        'line 3: `/ask` declares `{"kind": "worker", "tool_profile": "none", '
        '"locked": ["tool_profile"]}`'
    )
    [finding] = lint_prose(_prose_root(tmp_path / "f", "`/no_such_sea` declares `{}`"))
    assert finding.message == "line 3: `/no_such_sea` is not a registered command"
    [finding] = lint_prose(_prose_root(tmp_path / "g", "`/sh` declares `{not a dict}`"))
    assert finding.message == "line 3: the quoted settings of `/sh` are not a dict literal"
    # The checkout's own pages are clean.
    assert lint_prose() == []


# --- T6: the ran line of a path dispatch names the command --------------------------


def test_command_alias_and_run_config_line() -> None:
    sh = str(sea_commands.get_command("sh"))
    assert is_agent_path(sh) and not is_agent_path("sh")
    assert command_alias(sh) == "sh" and command_alias("/tmp/nowhere_sea.py") == ""
    assert command_alias(DEFAULT_AGENT_PATH) == ""  # hidden: not a command
    line = run_config_line({"sea": "sh", "kind": "worker"}, "sh")
    assert line.endswith('inherited=none pinned=none (also agent="sh")')
    assert "(also" not in run_config_line({"sea": "sh", "kind": "worker"})


# --- T7: hidden documents the literal ----------------------------------------------


def test_hidden_doc_says_why_the_literal_is_required() -> None:
    doc = SETTING_DOCS["hidden"]
    assert "must be the literal `True`" in doc
    assert "reads it from the source without running the script" in doc


# --- through the real daemon: T2 and T6 -------------------------------------------


class ReassessmentDaemonTest(DaemonLocalHarness):
    """``/sh`` through ``run_agent``: the lock, and the alias of a path dispatch."""

    def setUp(self) -> None:
        super().setUp()
        self._saved_env = os.environ.get("KISS_SORCAR_LOCAL")
        os.environ["KISS_SORCAR_LOCAL"] = str(self.endpoint_file)
        if not get_available_models():
            self.skipTest("the daemon accepts a run only with a configured model")
        self.local_sea = Path(self.tmpdir) / "local_sea.py"
        self.local_sea.write_text(textwrap.dedent("""
            def description():
                return "A SEA that is not a command."
        """))

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            self_agent.total_tokens_used = 5
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            printer = kwargs.get("printer") or getattr(self_agent, "printer", None)
            if printer is not None:
                printer.print(raw, type="result", step_count=1, total_tokens=5, cost="$0.0010")
            return raw

        self._parent_class.run = stub_run

    def tearDown(self) -> None:
        if self._saved_env is None:
            os.environ.pop("KISS_SORCAR_LOCAL", None)
        else:
            os.environ["KISS_SORCAR_LOCAL"] = self._saved_env
        super().tearDown()

    def test_sh_by_path_reports_its_command_name_and_honours_its_lock(self) -> None:
        parent = _parent(self.repo)
        sh = str(sea_commands.get_command("sh"))
        out = _run_agent(self.repo, "echo hi", agent=sh, parent_agent=parent)
        ran = yaml.safe_load(out)["ran"]
        assert ran.startswith("sh (worker) ") and " tools=bash " in ran, ran
        assert ran.endswith(' (also agent="sh")'), ran
        # By name, nothing to add; a non-command path has no alias either.
        out = _run_agent(self.repo, "echo hi", agent="sh", parent_agent=parent)
        ran = yaml.safe_load(out)["ran"]
        assert ran.startswith("sh (worker) ") and "(also" not in ran, ran
        ran = yaml.safe_load(
            _run_agent(self.repo, "hi", agent=str(self.local_sea), parent_agent=parent)
        )["ran"]
        assert ran.startswith("local (session) ") and "(also" not in ran, ran
        # The lock: the same profile is fine (explicitly or as an option), another is refused.
        for kwargs in ({"tool_profile": "bash"}, {"options": '{"tool_profile": "bash"}'}):
            out = _run_agent(self.repo, "echo hi", agent="sh", parent_agent=parent, **kwargs)
            assert yaml.safe_load(out)["success"] is True, out
        out = _run_agent(
            self.repo, "echo hi", agent="sh", tool_profile="review", parent_agent=parent,
        )
        assert out == "Error: sh: the script locks tool_profile='bash' (asked for 'review')"
        # T1 through the daemon: ``model`` and ``timeout`` as options.
        out = _run_agent(
            self.repo, "echo hi", agent=sh, parent_agent=parent,
            options=f'{{"model": "{_parent(self.repo).model_name}", "timeout": 45}}',
        )
        ran = yaml.safe_load(out)["ran"]
        assert " timeout=45s " in ran and "inherited=" in ran, ran
        assert "model" not in ran.split("inherited=")[1].split(" ")[0].split(","), ran
        assert agent_dispatch.DEFAULT_DISPATCH_TIMEOUT_SECONDS == 3600.0
