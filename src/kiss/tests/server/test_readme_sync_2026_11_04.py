# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Keep the README prose of the 2026-11-04 docs sync in step with the code.

The root README, ``src/kiss/server/README.md``, ``src/kiss/SYSTEM.md`` and
``src/kiss/TIPS.md`` describe the router SEAs' models, the Browser tab's fallback to the
bundled Chromium, the general agent's finish guards and side-call
charge, the coding SEA's finish gate (verifier reuse, the reading check,
one review with a deadline, the launcher's defaults) with its two reader
modules, the shell-change reader ``shell_changes.file_changes`` and the
notes a ``Bash`` result carries.
These checks read the names and numbers from the code, so a renamed
constant, a changed model or a retired function fails here instead of
drifting silently.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from kiss.agents.seas.autorouter import autorouter_sea
from kiss.agents.seas.bestrouter import bestrouter_sea
from kiss.agents.seas.ccxbestrouter import ccxbestrouter_sea
from kiss.agents.seas.coding import coding_sea, gate_readers, shell_changes
from kiss.agents.seas.orautorouter import orautorouter_sea
from kiss.agents.seas.orbestrouter import orbestrouter_sea
from kiss.agents.sorcar import shell_scan, sorcar_agent, useful_tools
from kiss.benchmarks.terminal_bench import run as tb_run
from kiss.core import utils
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models import model as model_module
from kiss.server import browser_tab

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ROOT_README = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
_SERVER_README = (_REPO_ROOT / "src" / "kiss" / "server" / "README.md").read_text(
    encoding="utf-8"
)
#: The server README with every run of white space as one blank, for quotes that wrap.
_SERVER_README_FLAT = re.sub(r"\s+", " ", _SERVER_README)
_SYSTEM_PROMPT = (_REPO_ROOT / "src" / "kiss" / "SYSTEM.md").read_text(encoding="utf-8")
_TIPS = (_REPO_ROOT / "src" / "kiss" / "TIPS.md").read_text(encoding="utf-8")
_SEA_SOURCE = (_REPO_ROOT / "src" / "kiss" / "agents" / "seas" / "coding"
               / "coding_sea.py").read_text(encoding="utf-8")
# Read as text: importing the launcher needs the ``harbor`` package.
_LAUNCHER_SOURCE = (_REPO_ROOT / "src" / "kiss" / "benchmarks" / "terminal_bench"
                    / "harbor_agent.py").read_text(encoding="utf-8")
#: Names of the gate as it was before 2026-11-04; none may be in the code or the prose.
_RETIRED_GATE_NAMES = (
    "EvidenceGate", "HEDGE_PHRASES", "hedges_in", "HEDGE_REASON", "UNCHECKED_REASON", "JOB_WAIT",
    "EVIDENCE_VERDICT", "EVIDENCE_FIELD_VERDICT", "UNCHECKED_VERDICT", "evidence_problems",
    "evidence_field_verdict", "open_reading_verdict", "OPEN_READING_VERDICT",
    "OPEN_READING_THRESHOLD", "OPEN_READING_QUESTIONS", "leaves_a_case_open",
    "REVIEW_FAILED_VERDICT", "open_reading_refused", "_CITED", "_verifier_in",
)


def _defined(name: str, source: str) -> bool:
    """Whether *name* is a function, method, attribute, constant or class of *source*."""
    return re.search(rf"(?:def |self\.|class |^){name}\b", source, re.M) is not None


def test_readmes_name_the_router_models() -> None:
    """Both READMEs name the routers' models, and the tips window the bestrouter's."""
    for readme in (_ROOT_README, _SERVER_README):
        for name in (
            autorouter_sea.ROUTER_MODEL,
            bestrouter_sea.SIMPLE_MODEL,
            bestrouter_sea.PRIMARY_MODEL,
            bestrouter_sea.REVIEW_MODEL,
            ccxbestrouter_sea.SIMPLE_MODEL,
            ccxbestrouter_sea.PRIMARY_MODEL,
            ccxbestrouter_sea.REVIEW_MODEL,
            orautorouter_sea.ROUTER_MODEL,
            orbestrouter_sea.SIMPLE_MODEL,
            orbestrouter_sea.PRIMARY_MODEL,
            orbestrouter_sea.REVIEW_MODEL,
        ):
            assert f"`{name}`" in readme, name
    assert "`bestrouter_sea.SIMPLE_MODEL`" in _SERVER_README
    assert bestrouter_sea.BestrouterSea().settings({})["model"] == bestrouter_sea.SIMPLE_MODEL
    assert orbestrouter_sea.OrbestrouterSea().settings({})["model"] == (
        orbestrouter_sea.SIMPLE_MODEL)
    assert "runs every task on" not in _ROOT_README
    assert "claude-opus-5-5-max` (pre-run" not in _ROOT_README
    tip = next(
        line for line in _TIPS.splitlines() if line.startswith("**What `bestrouter` does.**")
    )
    for name in (
        bestrouter_sea.SIMPLE_MODEL, bestrouter_sea.PRIMARY_MODEL, bestrouter_sea.REVIEW_MODEL
    ):
        assert f"`{name}`" in tip, name


def test_server_readme_names_the_browser_fallback() -> None:
    """The browser-tab fallback and its failure summary are described as coded."""
    assert "`BrowserTabService._launch_resolved`" in _SERVER_README
    assert "`launch_failure_summary`" in _SERVER_README
    assert hasattr(browser_tab.BrowserTabService, "_launch_resolved")
    assert f"cut\n  at {browser_tab._FAILURE_LINE_LIMIT} characters" in _SERVER_README
    assert "`browserError`" in _SERVER_README
    # The README's Browser tab entry is a top-level item whose nested list ends at the next one.
    browser_tab_entry = _ROOT_README.split("\n- **Browser tab**\n", 1)[1].split("\n- ", 1)[0]
    assert "bundled Chromium" in browser_tab_entry


def test_server_readme_names_the_finish_guards_and_the_side_call_charge(tmp_path: Path) -> None:
    """The general agent guards ``finish`` and gates nothing; the coding SEA's finish is its own."""
    for name in ("_guard_tool_call", "_block_finish_with_live_jobs", "_guard_rollback",
                 "_charge_side_call_usage"):
        assert hasattr(sorcar_agent.SorcarAgent, name), name
        assert f"`{name}`" in _SERVER_README or f"`SorcarAgent.{name}`" in _SERVER_README, name
    for name in ("side_call_binding", "_attribute_side_call_usage"):
        assert hasattr(sorcar_agent, name), name
        assert f"`{name}`" in _SERVER_README, name
    assert "`KISSAgent.tool_call_guard`" in _SERVER_README
    assert "`KISSAgent._tool_call_verdict`" in _SERVER_README
    assert hasattr(KISSAgent, "_tool_call_verdict")
    # The built-in finish takes no ``evidence``; the coding SEA's own finish adds it and
    # comes first in the trial's toolset.
    built_in = list(inspect.signature(utils.finish).parameters)
    assert list(inspect.signature(coding_sea.finish).parameters) == [*built_in, "evidence"]
    assert "`coding_sea.finish`, first in `ContainerHarness.tools()`" in _SERVER_README_FLAT
    config = tmp_path / "trial.json"
    config.write_text(json.dumps({"container": "none", "prompt": "p", "model": "m",
                                  "trajectory": str(tmp_path / "trajectory.jsonl")}))
    assert coding_sea.ContainerHarness(str(config)).tools([])[0] is coding_sea.finish


def test_server_readme_names_the_framework_flags_rule() -> None:
    """A ``model_config`` of framework flags alone is added to the endpoint's configuration."""
    assert "`model.framework_flags_over`" in _SERVER_README
    assert "`core.models.model.FRAMEWORK_ONLY_CONFIG_KEYS`" in _SERVER_README
    assert "plain_io" in model_module.FRAMEWORK_ONLY_CONFIG_KEYS
    endpoint = {"base_url": "u"}
    assert model_module.framework_flags_over({"plain_io": True}, endpoint) == {
        "base_url": "u", "plain_io": True}
    assert model_module.framework_flags_over({"temperature": 0}, endpoint) == {"temperature": 0}


def test_server_readme_names_the_coding_sea_readers() -> None:
    """Every reader the README cites from the SEA's two reader modules exists and reads as told."""
    for module in (gate_readers, shell_changes):
        short = module.__name__.rsplit(".", 1)[1]
        assert f"`seas/coding/{short}.py`" in _SERVER_README, short
        cited = set(re.findall(rf"`{short}\.(\w+)`", _SERVER_README))
        assert cited, short
        for name in cited:
            assert hasattr(module, name), f"{short}.{name}"
    assert gate_readers.default_effort({"reasoning_effort": "max", "base_url": "u"}) == {
        "base_url": "u"}
    assert gate_readers.default_effort({"reasoning_effort": "max"}) is None
    assert "a later round is refused only while your verifier fails" in gate_readers.gate_refusal(
        1, 3, ["x"])
    for retired in _RETIRED_GATE_NAMES:
        for module in (gate_readers, shell_changes, coding_sea):
            assert not hasattr(module, retired), retired
        assert f"`{retired}`" not in _SERVER_README, retired
        assert retired not in _ROOT_README and retired not in _SYSTEM_PROMPT, retired


def test_server_readme_names_the_coding_sea_gate() -> None:
    """The coding SEA's verifier reuse, reading check, review call and launcher defaults."""
    for name in ("review_call", "track_change", "under_workdir", "changes", "verifier_passed_at",
                 "record_exit_codes", "verifier_verdict", "last_verifier_run", "turn_commands",
                 "reading_check", "reading_verdict", "ledger_verdict", "READING_CHECK_PROMPT",
                 "READING_VERDICT", "READING_NONE", "READING_NOTE", "BLIND_NOTE", "BLIND_MENTION",
                 "MAX_READING_REFUSALS", "MAX_LEDGER_REFUSALS", "SIDE_AGENT_EVENTS",
                 "resolve_side_model", "review_usage", "evidence_report", "unanswered_verdict",
                 "finish_verdict", "REVIEW_DEADLINE", "MAX_REVIEWS", "MAX_FAILED_REVIEWS"):
        assert _defined(name, _SEA_SOURCE), name
        assert f"`{name}`" in _SERVER_README or f"`ContainerHarness.{name}`" in _SERVER_README, name
    assert coding_sea.MAX_REVIEWS == 1 and coding_sea.MAX_READING_REFUSALS == 2
    assert f"`max_reviews` (`MAX_REVIEWS`, {coding_sea.MAX_REVIEWS})" in _SERVER_README
    assert f"`MAX_READING_REFUSALS` ({coding_sea.MAX_READING_REFUSALS})" in _SERVER_README
    assert f"`MAX_FAILED_REVIEWS`, {coding_sea.MAX_FAILED_REVIEWS}" in _SERVER_README
    assert f"`REVIEW_DEADLINE` ({coding_sea.REVIEW_DEADLINE})" in _SERVER_README
    assert f"`max_gate_rounds` ({coding_sea.MAX_GATE_ROUNDS})" in _SERVER_README
    assert "reading_check" in coding_sea.SIDE_AGENT_EVENTS
    assert coding_sea.READING_NONE.match("none") and not coding_sea.READING_NONE.match("1. x")
    assert "(passed when you ran it; nothing changed since)" in _SEA_SOURCE
    assert '"passed when you ran it; nothing changed since"' in _SERVER_README_FLAT
    assert "`(name, args, changes)` triples" in _SERVER_README
    assert "`shell_changes.writes_files`" in _SERVER_README
    assert hasattr(shell_changes, "writes_files")
    # The prompt mentions the reading and the blind check only when a model is configured.
    assert "{reading_note}" in coding_sea.SYSTEM_PROMPT
    assert "{blind_note}" in coding_sea.SYSTEM_PROMPT
    assert "{blind_mention}" in coding_sea.SYSTEM_PROMPT
    # The launcher's defaults, as the README states them.
    assert 'OTHER_FAMILY = "other"' in _LAUNCHER_SOURCE and 'NO_REVIEW = "none"' in _LAUNCHER_SOURCE
    for field, default in (("reviewer", "OTHER_FAMILY"), ("blind_verifier", "NO_REVIEW"),
                           ("reading_check", "OTHER_FAMILY")):
        assert re.search(rf"^    {field}: str = {default}$", _LAUNCHER_SOURCE, re.M), field
    assert "`reviewer` and `reading_check` default to `other`" in _SERVER_README
    assert "`blind_verifier` to `none`" in _SERVER_README
    parser = tb_run.build_parser()
    assert (parser.get_default("reviewer"), parser.get_default("reading_check"),
            parser.get_default("blind_verifier")) == ("other", "other", "none")
    for flag in ("--reviewer", "--reading-check", "--blind-verifier"):
        assert f"`{flag}`" in _SERVER_README, flag
    assert "`harbor_command`" in _SERVER_README and hasattr(tb_run, "harbor_command")


def test_server_readme_describes_file_changes_not_the_retired_regex() -> None:
    """The shell-change reader is ``file_changes``; the ``WRITING_COMMAND`` regex is gone."""
    assert not hasattr(shell_changes, "WRITING_COMMAND")
    assert "`shell_changes.file_changes`" in _SERVER_README
    assert "`is_scratch_path`" in _SERVER_README and hasattr(shell_changes, "is_scratch_path")
    for program in shell_changes.WRITING_PROGRAMS:
        assert f"`{program}`" in _SERVER_README, program
    for tool in shell_changes.BUILD_TOOLS:
        assert f"`{tool}`" in _SERVER_README, tool
    assert "`UV_VALUED_OPTIONS`" in _SERVER_README
    assert hasattr(shell_scan, "UV_VALUED_OPTIONS")
    assert f"`MAX_NESTING` ({shell_scan.MAX_NESTING})" in _SERVER_README
    for attribute in ("conditional", "after_and", "background", "negated", "heredoc_body"):
        assert hasattr(shell_scan.SimpleCommand(None, None), attribute), attribute
        assert f"`{attribute}`" in _SERVER_README or (
            f"`SimpleCommand.{attribute}`" in _SERVER_README), attribute
    assert "`handed_text`" in _SERVER_README and hasattr(shell_scan, "handed_text")
    assert "optimistic" in shell_scan.simple_commands.__code__.co_varnames


def test_server_readme_puts_the_bash_notes_after_the_result() -> None:
    """Notes follow a ``Bash`` result; the ``uv run`` rewrite and the sleep wait say so."""
    assert useful_tools.UV_REWRITTEN_NOTE.startswith("\nNote: the worktree has no environment")
    assert "`useful_tools.UV_REWRITTEN_NOTE`,\n  after the result" in _SERVER_README
    assert "the result prefixed" not in _SERVER_README
    assert "the result ends with `Waited on job" in _SERVER_README
    assert "`Edit` edits unread" in _SERVER_README
    assert hasattr(useful_tools, "uv_in_worktree")
    assert not hasattr(useful_tools, "_uv_in_worktree_guard")
