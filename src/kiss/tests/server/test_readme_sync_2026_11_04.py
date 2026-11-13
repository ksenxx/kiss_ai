# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Keep the README prose of the 2026-11-04 docs sync in step with the code.

The root README, ``src/kiss/server/README.md``, ``src/kiss/SYSTEM.md`` and
``src/kiss/TIPS.md`` describe the router SEAs' models, the Browser tab's fallback to the
bundled Chromium, the coding gate of 2026-11-04 (verifier reuse, one
review per run with a deadline, the round rule, the independent reading
of the task, the merge rule), the shell-change reader
``evidence_gate.file_changes`` and the notes a ``Bash`` result carries.
These checks read the names and numbers from the code, so a renamed
constant, a changed model or a retired function fails here instead of
drifting silently.
"""

from __future__ import annotations

import re
from pathlib import Path

from kiss.agents.seas.autorouter import autorouter_sea
from kiss.agents.seas.bestrouter import bestrouter_sea
from kiss.agents.seas.ccxbestrouter import ccxbestrouter_sea
from kiss.agents.seas.coding import coding_sea
from kiss.agents.seas.orautorouter import orautorouter_sea
from kiss.agents.seas.orbestrouter import orbestrouter_sea
from kiss.agents.sorcar import coding_gate, evidence_gate, shell_scan, useful_tools
from kiss.benchmarks.terminal_bench import run as tb_run
from kiss.core import tool_verdict
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
_KISS_AGENT_SOURCE = (_REPO_ROOT / "src" / "kiss" / "core" / "kiss_agent.py").read_text(
    encoding="utf-8")
_GATE_SOURCE = (_REPO_ROOT / "src" / "kiss" / "agents" / "sorcar" / "coding_gate.py").read_text(
    encoding="utf-8")
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
    assert "bundled Chromium" in _ROOT_README.split("- A **Browser tab**", 1)[1].split("\n", 1)[0]


def test_server_readme_names_the_gate_additions() -> None:
    """Verifier reuse, the merge rule, fingerprints and HEAD following are described."""
    for name in (
        "verifier_decides", "verifier_command", "MERGE_CAP_NOTE",
        "MERGED_WITH_FINDINGS", "MERGED_CAVEATS", "deliver_merge_caveats",
        "follow_head", "moved_head", "HEAD_MOVING", "custom_tools", "evidence_report",
        "unanswered_verdict", "refusable", "refusal", "MAX_GATE_ROUNDS", "MAX_VERIFIER_REFUSALS",
        "deliver_with_findings", "DELIVERED_CAVEATS", "UNVERIFIED_CAVEAT",
    ):
        assert _defined(name, _GATE_SOURCE), name
        assert f"`{name}`" in _SERVER_README or f"`CodingGate.{name}`" in _SERVER_README, name
    assert "`CodingGate.fingerprint`" in _SERVER_README
    assert "`CodingGate.coding`" in _SERVER_README
    assert "coding_gate" in model_module.FRAMEWORK_ONLY_CONFIG_KEYS
    assert "`model.framework_flags_over`" in _SERVER_README
    assert hasattr(model_module, "framework_flags_over")
    assert f"`MAX_GATE_ROUNDS` ({coding_gate.MAX_GATE_ROUNDS})" in _SERVER_README
    assert f"`MAX_VERIFIER_REFUSALS` ({coding_gate.MAX_VERIFIER_REFUSALS})" in _SERVER_README
    # The merge-cap note is quoted as coded, and every merge refusal ends with it.
    assert f'`MERGE_CAP_NOTE` ("{coding_gate.MERGE_CAP_NOTE}")' in _SERVER_README_FLAT
    assert "Every merge refusal ends with `MERGE_CAP_NOTE`" in _SERVER_README
    assert coding_gate.gate_refusal(1, 3, ["x"], "commit_and_merge").endswith(
        coding_gate.MERGE_CAP_NOTE)
    assert "a later round is refused only while your verifier fails" in coding_gate.gate_refusal(
        1, 3, ["x"], "finish")
    gate_bullet = _ROOT_README.split("- A **coding gate**", 1)[1].split("\n", 1)[0]
    for phrase in ("`coding_gate: false`", "a merge is refused under the same round rule",
                   "unjudged and refuse nothing", "the work differs after it",
                   "the first refusal buys one round whatever refused it",
                   "only while the verifier fails",
                   "Nothing in the `finish` call itself is refused",
                   f"{coding_gate.REVIEW_DEADLINE} s per attempt", "is retried once",
                   "`tmp/<task-id>/ledger.md`"):
        assert phrase in gate_bullet, phrase
    for stale in ("writes or deletes files", "every finding counts as answered",
                  "refused once with the phrases", "hedge", "unchecked-change",
                  "at most three completed reviews", "after three refused rounds",
                  "a merge is refused at most twice"):
        assert stale not in gate_bullet, stale
    git_verbs = re.search(r"\(\?:(.*?)\)\\b", coding_gate.HEAD_MOVING["git"].pattern)
    assert git_verbs is not None
    for verb in git_verbs.group(1).split("|"):
        assert f"`{verb}`" in _SERVER_README, verb


def test_server_readme_names_the_one_review_and_its_deadline() -> None:
    """The review is one side call at default effort with a deadline; a failed one refuses nothing.
    """
    for name in ("side_call", "default_effort", "call_stop_event", "_watch_call",
                 "REVIEW_DEADLINE", "MAX_REVIEWS", "MAX_FAILED_REVIEWS", "review_possible",
                 "answered_findings", "REVIEW_PROMPT"):
        assert _defined(name, _GATE_SOURCE), name
        assert f"`{name}`" in _SERVER_README or f"`CodingGate.{name}`" in _SERVER_README, name
    assert coding_gate.MAX_REVIEWS == 1
    assert f"`MAX_REVIEWS` ({coding_gate.MAX_REVIEWS})" in _SERVER_README
    assert f"`MAX_FAILED_REVIEWS` ({coding_gate.MAX_FAILED_REVIEWS})" in _SERVER_README
    assert f"`REVIEW_DEADLINE` ({coding_gate.REVIEW_DEADLINE})" in _SERVER_README
    assert f"`no answer within {coding_gate.REVIEW_DEADLINE} s`" in _SERVER_README_FLAT
    assert coding_gate.default_effort({"reasoning_effort": "max", "base_url": "u"}) == {
        "base_url": "u"}
    assert coding_gate.default_effort({"reasoning_effort": "max"}) is None
    assert "refuses nothing" in _SERVER_README.split("`CodingGate.side_call`", 1)[1][:1500]
    for retired in _RETIRED_GATE_NAMES:
        assert not hasattr(coding_gate, retired), retired
        assert not hasattr(evidence_gate, retired), retired
        assert not hasattr(coding_sea, retired), retired
        assert f"`{retired}`" not in _SERVER_README, retired
        assert retired not in _ROOT_README and retired not in _SYSTEM_PROMPT, retired


def test_readmes_describe_the_reading_note() -> None:
    """The independent reading of the task at the ledger's first ``Write`` is described as coded."""
    for name in ("reading_note", "READING_PROMPT", "READING_NOTE", "READING_NONE", "LEDGER_NAME",
                 "is_scratch", "numbered_items", "REVIEW_ONE_FILE_BYTES"):
        assert _defined(name, _GATE_SOURCE), name
        assert f"`{name}`" in _SERVER_README or f"`CodingGate.{name}`" in _SERVER_README, name
    assert coding_gate.LEDGER_NAME == "ledger.md"
    assert f"`LEDGER_NAME`, `{coding_gate.LEDGER_NAME}`" in _SERVER_README
    assert coding_gate.READING_NONE.match("NONE") and coding_gate.READING_NONE.match("`NONE`.\n")
    assert "`SorcarAgent._guard_tool_call`" in _SERVER_README
    assert "`tool_verdict.allow_with_note`" in _SERVER_README
    noted = tool_verdict.allow_with_note("n")
    assert noted.allowed and noted.text == "n"
    assert "`KISSAgent.tool_call_guard`" in _SERVER_README
    assert "`KISSAgent._tool_call_verdict`" in _SERVER_README
    assert _defined("_tool_call_verdict", _KISS_AGENT_SOURCE)
    assert "Callable[[str, dict[str, Any]], str | Verdict | None]" in _KISS_AGENT_SOURCE
    # The system prompt tells the agent what to do with the note, and no longer threatens
    # the refusals that are gone.
    rule = _SYSTEM_PROMPT.split("\n6. The finish gate of a run that changed code", 1)[1].split(
        "\n", 1)[0]
    for phrase in ("When you Write <scratch dir>/ledger.md", "settle each in the ledger",
                   "The first refusal buys one round whatever refused it",
                   "only while your verifier fails", "three refused rounds at most",
                   f"{coding_gate.REVIEW_DEADLINE} s to answer"):
        assert phrase in rule, phrase
    rule_5 = _SYSTEM_PROMPT.split("\n5. Put the evidence in the summary", 1)[1].split("\n", 1)[0]
    assert "the gate does not refuse on this field or on the wording of the summary" in rule_5
    for stale in ("refused once with the phrases", "must have run in this run and exited 0",
                  "unchecked-change", "admits a requirement unhandled",
                  "After three refused rounds"):
        assert stale not in _SYSTEM_PROMPT, stale


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
    assert "`evidence_gate.writes_files`" in _SERVER_README
    assert hasattr(evidence_gate, "writes_files")
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
    assert not hasattr(evidence_gate, "WRITING_COMMAND")
    assert "`evidence_gate.file_changes`" in _SERVER_README
    assert "`is_scratch_path`" in _SERVER_README and hasattr(evidence_gate, "is_scratch_path")
    for program in evidence_gate.WRITING_PROGRAMS:
        assert f"`{program}`" in _SERVER_README, program
    for tool in evidence_gate.BUILD_TOOLS:
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
