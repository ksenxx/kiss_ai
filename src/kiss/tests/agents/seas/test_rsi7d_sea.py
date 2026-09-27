# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled rsi7d agent (:mod:`kiss.agents.seas.rsi7d_sea`).

The mining tools run against tasks persisted in the test session's real
SQLite history (``KISS_HOME`` is a temporary directory, see
``conftest.py``).  The prompt editors run against a fake KISS checkout
under ``tmp_path`` (``src/kiss/agents/seas`` holding demo SEAs with a
plain, an f-string and no prompt constant) that ``_seas_dir`` resolves
through the working directory.  The agent-level test drives a real
:class:`ChatSorcarAgent` ReAct loop against the scripted local
chat-completions server so a patch really flows from the model's tool
call into the SEA file.
"""

from __future__ import annotations

import ast
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.seas import autoroute_sea
from kiss.agents.seas import rsi7d_sea as sea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _append_chat_event,
    _flush_chat_events,
    _save_task_result,
)
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

_SEA_PATH = Path(sea.__file__).resolve()
_SEAS_DIR = _SEA_PATH.parent

_PLAIN_SEA = '''"""Demo SEA with a plain prompt constant."""

SYSTEM_PROMPT = (
    "You run demo tasks. Always report the exit code. Always report the exit code. "
    "Always report the exit code. Always report the exit code. Always report the exit code. "
    "Never guess: read the file before editing it. "
)


def system_prompt() -> str:
    """Replace the default prompt."""
    return SYSTEM_PROMPT
'''

_FSTRING_SEA = '''"""Demo SEA with an f-string prompt constant."""

GATE = "uv run pytest -q"

SYSTEM_PROMPT = f"""\\
# Demo f-string agent

Run the gate `{GATE}` before you finish and report its output verbatim; when the gate
fails, fix the cause and rerun it. Never edit files you have not read. Keep the summary
short and cite every file you changed by path.
"""


def append_to_system_prompt() -> str:
    """Append to the default prompt."""
    return SYSTEM_PROMPT
'''

_TEMPLATE_SEA = '''"""Demo SEA whose prompt is a str.format template."""

SYSTEM_PROMPT = "You help {user}. Answer in {language}. Be brief."


def system_prompt() -> str:
    """Formatted at run time by a wrapper."""
    return SYSTEM_PROMPT
'''

_NOPROMPT_SEA = '''"""Demo SEA without a prompt getter."""


def tools() -> list:
    """No tools."""
    return []
'''


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake KISS checkout in *tmp_path* that ``_seas_dir`` resolves through the cwd."""
    seas = tmp_path / "src" / "kiss" / "agents" / "seas"
    seas.mkdir(parents=True)
    shutil.copy(_SEA_PATH, seas / "rsi7d_sea.py")
    shutil.copy(_SEAS_DIR / "autoroute_sea.py", seas / "autoroute_sea.py")
    (seas / "demo_sea.py").write_text(_PLAIN_SEA, encoding="utf-8")
    (seas / "fdemo_sea.py").write_text(_FSTRING_SEA, encoding="utf-8")
    (seas / "tdemo_sea.py").write_text(_TEMPLATE_SEA, encoding="utf-8")
    (seas / "noprompt_sea.py").write_text(_NOPROMPT_SEA, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return seas


def _persist(
    prompt: str,
    events: list[dict[str, Any]],
    result: str = "",
    **extra: object,
) -> str:
    """Persist a finished task row with *events*, *result* and the given columns; return its id."""
    now_ms = int(time.time() * 1000)
    payload: dict[str, object] = {
        "model": "model-a",
        "work_dir": "/work/dir",
        "startTs": now_ms - 60_000,
        "endTs": now_ms,
        "cost": 0.5,
        "steps": 5,
        "tokens": 10_000,
    }
    payload.update(extra)
    task_id, _chat = _add_task(prompt, extra=payload)
    for ev in events:
        _append_chat_event(ev, task_id=task_id)
    _flush_chat_events(task_id)
    if result:
        _save_task_result(result, task_id=task_id)
    return task_id


def _result_event(success: bool) -> dict[str, Any]:
    return {"type": "result", "text": f"success: {'true' if success else 'false'}\nsummary: done"}


def _dispatch(agent: str, task: str) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "name": "run_agent",
        "callId": 1,
        "extras": {"agent": agent, "task": task},
    }


def test_sea_getters_and_prompt_follow_the_contract() -> None:
    """The SEA replaces the system prompt, exposes its tools, and runs without a browser."""
    assert sea.system_prompt() == sea.SYSTEM_PROMPT
    assert sea.build_prompt("") == sea.build_prompt("   ")
    assert sea.build_prompt("only review_paper").endswith(
        "Additional instructions: only review_paper"
    )
    assert sea.max_budget() == 2000.0
    assert sea.use_memory() is True
    assert sea.use_web_tools() is False
    names = [t.__name__ for t in sea.tools()]
    assert names == [
        "indexed_seas",
        "sea_runs",
        "sea_findings",
        "run_findings",
        "run_overview",
        "run_transcript",
        "run_entry",
        "model_scorecard",
        "sea_prompt",
        "patch_sea_prompt",
        "write_autoroute_evidence",
    ]
    for name in names:
        assert f"`{name}" in sea.SYSTEM_PROMPT or name in sea.SYSTEM_PROMPT, name
    assert sea_commands.get_command("rsi7d") == _SEA_PATH
    assert (
        sea_commands.rewrite_prompt_if_command("/rsi7d") is None
    )  # a slash command needs task text
    rewritten = sea_commands.rewrite_prompt_if_command("/rsi7d all")
    assert rewritten is not None and rewritten[1] == _SEA_PATH


def test_sea_name_of_handles_paths_channels_and_plain_subagents() -> None:
    """A SEA path maps to its stem without ``_sea``; a channel name stays; empty stays empty."""
    assert sea.sea_name_of("/x/y/review_paper_sea.py") == "review_paper"
    assert sea.sea_name_of("C:\\\\ext\\\\slack_sea.py") == "slack"
    assert sea.sea_name_of("cron") == "cron"
    assert sea.sea_name_of("") == ""
    assert sea.sea_name_of(None) == ""


def test_indexed_seas_reports_editable_paths_and_prompt_shapes(checkout: Path) -> None:
    """Bundled SEAs are editable with their prompt shape; registered foreign SEAs are not."""
    rows = {r["name"]: r for r in json.loads(sea.indexed_seas())}
    assert rows["demo"]["editable_path"] == str(checkout / "demo_sea.py")
    assert (rows["demo"]["prompt_getter"], rows["demo"]["prompt_constant"]) == (
        "system_prompt",
        "SYSTEM_PROMPT",
    )
    assert rows["demo"]["prompt_chars"] > 100
    assert rows["fdemo"]["prompt_getter"] == "append_to_system_prompt"
    assert rows["fdemo"]["prompt_constant"] == "SYSTEM_PROMPT"
    assert rows["fdemo"]["prompt_chars"] == len(
        _FSTRING_SEA.split("SYSTEM_PROMPT = ", 1)[1].split("\n\n\ndef")[0]
    )
    assert (rows["noprompt"]["prompt_getter"], rows["noprompt"]["prompt_constant"]) == ("", "")
    assert rows["rsi7d"]["registered_path"] == str(_SEA_PATH)
    foreign = rows["slack"]
    assert foreign["registered_path"].endswith("slack_sea.py")
    assert foreign["editable_path"] == "" and foreign["prompt_constant"] == ""


def test_sea_runs_links_dispatches_and_prompt_signatures(checkout: Path) -> None:
    """A run is found through its parent's ``run_agent`` call or its system-prompt signature.

    The parent dispatches ``demo`` twice: one child row exists (matched),
    the other was never persisted (counted as unmatched).  A plain
    sub-agent child (no ``agent``) is ignored.  A side-channel child whose
    system prompt starts with ``fdemo``'s prompt is matched by signature.
    """
    agent_path = str(checkout / "demo_sea.py")
    parent = _persist(
        "/demo do the thing",
        [
            _dispatch(agent_path, "do the thing"),
            _dispatch(agent_path, "never persisted"),
            {
                "type": "tool_call",
                "name": "run_agent",
                "callId": 3,
                "extras": {"agent": "", "task": "plain"},
            },
        ],
        result="<p>ok</p>",
        cost=3.0,
        steps=12,
    )
    child = _persist(
        "do the thing",
        [_result_event(True)],
        result="<p>done</p>",
        parent_task_id=parent,
        cost=1.0,
        steps=4,
        model="model-b",
    )
    _persist("plain", [_result_event(True)], result="<p>plain</p>", parent_task_id=parent, cost=0.5)
    fdemo_prompt = sea._execute_sea(checkout / "fdemo_sea.py")["append_to_system_prompt"]()
    side = _persist(
        "What have the task done so far?",
        [
            {"type": "system_prompt", "text": "DEFAULT PROMPT\n\n" + fdemo_prompt},
            _result_event(False),
        ],
        result="<p>partial</p>",
        subagent={"parent_task_id": parent, "side_channel": True},
        cost=0.2,
        steps=2,
    )
    data = json.loads(sea.sea_runs(days=1))
    assert data["unmatched_dispatches"] >= 1
    demo = data["seas"]["demo"]
    assert demo["agents"] == [agent_path]
    assert [r["task_id"] for r in demo["runs"]] == [child]
    run = demo["runs"][0]
    assert run["status"] == "success" and run["model"] == "model-b"
    assert (
        run["cost"] == 1.0 and run["own_cost"] == 1.0 and run["steps"] == 4 and run["children"] == 0
    )
    assert run["duration_s"] == 60.0
    assert (
        demo["stats"]["runs"] == 1
        and demo["stats"]["success"] == 1
        and demo["stats"]["median_cost"] == 1.0
    )
    assert demo["stats"]["median_s_per_step"] == 15.0 and demo["stats"]["models"] == {"model-b": 1}
    fdemo = data["seas"]["fdemo"]
    assert fdemo["agents"] == ["(system prompt signature)"]
    assert [r["task_id"] for r in fdemo["runs"]] == [side]
    assert fdemo["runs"][0]["status"] == "unsuccessful"
    only = json.loads(sea.sea_runs(days=1, name="fdemo"))
    assert list(only["seas"]) == ["fdemo"]
    parent_row = json.loads(sea.sea_runs(days=1, name="demo"))
    assert "plain" not in json.dumps(parent_row)


def test_run_findings_flags_each_signal_kind_and_unknown_ids() -> None:
    """Every deterministic signal kind is detected from a persisted event stream."""
    calls: list[dict[str, Any]] = []
    for i in range(10):  # 10 tool calls, no summary -> no_summary
        calls.append({"type": "tool_call", "name": "Read", "callId": i, "path": f"f{i % 2}.py"})
        calls.append({"type": "tool_result", "content": "ok", "tool_name": "Read"})
    events: list[dict[str, Any]] = [
        {"type": "prompt", "text": "prompt"},
        {
            "type": "tool_call",
            "name": "Bash",
            "callId": 100,
            "command": "curl -sL https://example.com/page",
        },
        {
            "type": "tool_result",
            "content": "Error (exit code 7): could not resolve host",
            "tool_name": "Bash",
        },
        {
            "type": "tool_call",
            "name": "Edit",
            "callId": 101,
            "path": "a.py",
            "old_string": "x",
            "new_string": "y",
        },
        {"type": "tool_result", "content": "Error: String not found in file", "tool_name": "Edit"},
        {
            "type": "tool_call",
            "name": "run_commands_parallel",
            "callId": 102,
            "extras": {"commands": "[]"},
        },
        {
            "type": "tool_result",
            "content": "Error: Command execution timeout",
            "tool_name": "run_commands_parallel",
        },
        {"type": "tool_call", "name": "Bash", "callId": 103, "command": "ls /home/ksen/kiss"},
        {
            "type": "tool_result",
            "content": "Error: command references the parent-repo path '/home/ksen/kiss'",
            "tool_name": "Bash",
        },
        {"type": "tool_call", "name": "Read", "callId": 104, "path": "big.txt"},
        {"type": "tool_result", "content": "x" * 30_001, "tool_name": "Read"},
        *calls,
        {
            "type": "task_error",
            "text": "KISS Error: Model stream stalled: no data received for 180s",
        },
        {"type": "autocommit_done", "message": "Committed: feat: add things"},
        {"type": "followup_suggestion", "text": "Now remove things"},
        _result_event(False),
    ]
    task_id = _persist("Find things", events, result="<p>gave up</p>")
    found = json.loads(sea.run_findings(task_id))
    assert found["status"] == "unsuccessful" and found["entries"] > 20
    assert found["tool_calls"]["Read"] == 11 and found["tool_calls"]["Bash"] == 2
    counts = found["counts"]
    assert counts["shell_fetch"] == 1
    assert counts["tool_error"] == 1  # the curl exit code
    assert counts["edit_rejected"] == 1
    assert counts["timeout"] == 1
    assert counts["reviewer_misuse"] == 1
    assert counts["huge_result"] == 1
    assert counts["repeated_call"] == 2  # Read f0.py and Read f1.py, five times each
    assert counts["no_summary"] == 1
    assert counts["error_event"] == 1 and counts["stall_or_retry"] == 1
    assert counts["not_successful"] == 1
    by_kind = {s["kind"]: s for s in found["signals"]}
    assert by_kind["edit_rejected"]["detail"].startswith("Error: String not found")
    assert "x5" in by_kind["repeated_call"]["detail"]
    assert by_kind["stall_or_retry"]["detail"].startswith("TASK_ERROR:")
    assert [s["entry"] for s in found["signals"]] == sorted(s["entry"] for s in found["signals"])
    assert sea.run_findings("no-such-task") == "Error: no task with id 'no-such-task'"
    assert "Task id: " + task_id in sea.run_overview(task_id)
    assert "[0] TOOL CALL Bash(" in sea.run_transcript(task_id, 0, 5)
    assert "curl -sL https://example.com/page" in sea.run_entry(task_id, 0)


def test_sea_findings_aggregates_signals_over_a_seas_runs(checkout: Path) -> None:
    """``sea_findings`` sums tool calls and signals over the newest runs and keeps examples.

    Dispatches ``noprompt`` (no other test does, and it has no prompt
    signature) so the shared session history cannot add runs.
    """
    agent_path = str(checkout / "noprompt_sea.py")
    parent = _persist(
        "/noprompt twice", [_dispatch(agent_path, "run 1"), _dispatch(agent_path, "run 2")]
    )
    for task, ok in (("run 1", True), ("run 2", False)):
        _persist(
            task,
            [
                {"type": "tool_call", "name": "Bash", "callId": 1, "command": "pytest"},
                {
                    "type": "tool_result",
                    "content": "Error (exit code 1): 3 failed",
                    "tool_name": "Bash",
                },
                _result_event(ok),
            ],
            result="<p>r</p>",
            parent_task_id=parent,
        )
    found = json.loads(sea.sea_findings("noprompt", runs=8, days=1))
    assert len(found["runs_scanned"]) == 2
    assert found["by_status"] == {"success": 1, "unsuccessful": 1, "failed": 0}
    assert found["tool_calls"] == {"Bash": 2}
    assert (
        found["signal_counts"]["tool_error"] == 2 and found["signal_counts"]["not_successful"] == 1
    )
    assert len(found["examples"]["tool_error"]) == 2
    assert found["examples"]["tool_error"][0]["detail"].startswith("Bash: Error (exit code 1)")
    limited = json.loads(sea.sea_findings("noprompt", runs=1, days=1))
    assert len(limited["runs_scanned"]) == 1
    assert sea.sea_findings("nobody", days=1).startswith("Error: no runs of SEA 'nobody'")


def test_model_scorecard_reports_roles_costs_reliability_and_error_reasons() -> None:
    """Per-model aggregates: roles, own cost, per-step medians, tool and task errors."""
    top = _persist(
        "Build it",
        [
            {"type": "tool_call", "name": "Bash", "callId": 1, "command": "make"},
            {"type": "tool_result", "content": "Error (exit code 2): no rule", "tool_name": "Bash"},
            _result_event(True),
        ],
        result="<p>built</p>",
        model="score-a",
        cost=2.0,
        steps=10,
        tokens=50_000,
    )
    _persist(
        "Verify the listed changes: build",
        [
            {"type": "tool_call", "name": "Read", "callId": 1, "path": "x"},
            {"type": "tool_result", "content": "denied", "tool_name": "Read", "is_error": True},
            {"type": "tool_call", "name": "Bash", "callId": 2, "command": "ls"},
            {"type": "tool_result", "content": "\nError: command failed", "tool_name": "Bash"},
            _result_event(True),
        ],
        result="<p>fine</p>",
        model="score-a",
        parent_task_id=top,
        cost=0.5,
        steps=5,
        tokens=10_000,
    )
    _persist(
        "Explode",
        [],
        model="score-b",
        cost=0.1,
        steps=1,
        result="<p>KISSError: KISS Error: Agent Parallel-Follow /x Session-0 failed with 3 "
        "consecutive "
        "errors. Last error: You have no credits remaining.</p>",
    )
    _persist(
        "Explode again",
        [],
        model="score-b",
        cost=0.1,
        steps=1,
        result="<h3>Partial result: KISS Error: Agent Task update Session-0 budget exceeded.</h3>"
        "<p>The sub-agent spent $1.58 of its $1.00 budget in 3 steps.</p>",
    )
    _persist("Still running", [], model="score-b", endTs=0)
    cards = {c["model"]: c for c in json.loads(sea.model_scorecard(days=1))["models"]}
    a = cards["score-a"]
    assert a["tasks"] == 2 and a["roles"] == {"top": 1, "reviewer": 1}
    assert a["failed"] == 0 and a["unsuccessful"] == 0 and a["task_errors"] == 0
    assert a["median_own_cost"] == 1.0  # (2.0 - 0.5 child, 0.5) -> median 1.0
    # The parent's 10 steps and 50k tokens include the child's 5 and 10k:
    # own steps are 5 and 5, own tokens 40k and 10k.
    assert a["median_cost_per_step"] == 0.2  # (1.5/5, 0.5/5) -> median of 0.3 and 0.1
    assert a["median_s_per_step"] == 12.0  # 60 s / 5 own steps for both
    assert a["median_steps"] == 5 and a["median_tokens_per_step"] == 5000.0
    assert a["tool_error_rate"] == 0.3  # "Error", is_error and "\nError" results over 10 own steps
    b = cards["score-b"]
    assert b["tasks"] == 3 and b["failed"] == 2 and b["task_errors"] == 2
    reasons = list(b["error_reasons"])
    assert any(
        r.startswith("Agent failed with N consecutive errors. Last error: You have no credits")
        for r in reasons
    )
    assert any(
        r.startswith("Agent budget exceeded. The sub-agent spent N of its N budget in N steps")
        for r in reasons
    )


def test_patch_sea_prompt_edits_plain_constants_through_the_gate(checkout: Path) -> None:
    """Appending and replacing inside a plain constant rewrites the literal; the SEA still loads."""
    before = sea.sea_prompt("demo")
    assert before.startswith(
        f"# {checkout / 'demo_sea.py'}\n# system_prompt() returns SYSTEM_PROMPT, a plain string"
    )
    section = "## Lessons from recent runs (rsi7d)\n- Batch independent greps into one Bash call."
    report = sea.patch_sea_prompt("demo", "", section)
    assert report.startswith("Patched SYSTEM_PROMPT of") and "+3 lines" in report
    prompt = sea._execute_sea(checkout / "demo_sea.py")["system_prompt"]()
    assert prompt.endswith("Never guess: read the file before editing it. \n\n" + section + "\n")
    assert sea.patch_sea_prompt(
        "demo", "one Bash call.", "one Bash call, never one per grep."
    ).startswith("Patched")
    prompt = sea._execute_sea(checkout / "demo_sea.py")["system_prompt"]()
    assert "one Bash call, never one per grep." in prompt
    assert sea.sea_prompt("demo").endswith(prompt)
    assert (
        sea.patch_sea_prompt("demo", "not there", "x")
        == "Error: old text occurs 0 times in SYSTEM_PROMPT (must be exactly once)"
    )
    assert sea.patch_sea_prompt("demo", "Always report", "x").startswith(
        "Error: old text occurs 5 times"
    )
    assert sea.patch_sea_prompt("slack", "", "x").startswith(
        "Error: 'slack' is not an editable SEA under"
    )
    assert sea.patch_sea_prompt("../third_party_agents/slack", "", "x").startswith(
        "Error: '../third_party_agents/slack' is not an editable"
    )
    assert sea._editable_path("/etc/passwd") is None
    assert sea.patch_sea_prompt("noprompt", "", "x").endswith(
        "has no prompt getter returning a module-level string constant"
    )
    assert sea.sea_prompt("noprompt").startswith("Error:") and sea.sea_prompt("slack").startswith(
        "Error:"
    )


def test_patch_sea_prompt_edits_fstring_literals_and_doubles_braces(checkout: Path) -> None:
    """An f-string prompt is edited in its source literal; new text can never add a placeholder."""
    path = checkout / "fdemo_sea.py"
    shown = sea.sea_prompt("fdemo")
    assert "an f-string: the text below is its source literal" in shown
    assert 'f"""\\\n# Demo f-string agent' in shown
    report = sea.patch_sea_prompt(
        "fdemo", "", "## Lessons from recent runs (rsi7d)\n- Quote {GATE} literally."
    )
    assert report.startswith("Patched SYSTEM_PROMPT of")
    source = path.read_text(encoding="utf-8")
    # Appended as an adjacent plain literal: no doubling needed, no placeholder possible.
    assert '"""\n"""' not in source  # the f-string literal itself is untouched
    assert (
        " "
        + sea._string_literal(
            "\n\n## Lessons from recent runs (rsi7d)\n- Quote {GATE} literally.\n"
        )
        in source
    )
    prompt = sea._execute_sea(path)["append_to_system_prompt"]()
    assert "Run the gate `uv run pytest -q` before you finish" in prompt
    assert prompt.endswith("- Quote {GATE} literally.\n")
    # A replacement inside the f-string source doubles braces in the new text.
    assert sea.patch_sea_prompt(
        "fdemo", "Keep the summary\nshort", "Keep the summary {short}"
    ).startswith("Patched")
    assert "Keep the summary {{short}}" in path.read_text(encoding="utf-8")
    assert (
        "Keep the summary {short} and cite" in sea._execute_sea(path)["append_to_system_prompt"]()
    )
    ok_source = path.read_text(encoding="utf-8")
    # Closing the literal early would change code: rejected, file untouched.
    bad = sea.patch_sea_prompt("fdemo", "Never edit files", 'x"""\nimport os\ny = f"""z')
    assert bad.startswith("Error: candidate")
    assert path.read_text(encoding="utf-8") == ok_source
    # Rewriting a placeholder's expression is code: rejected.
    bad = sea.patch_sea_prompt("fdemo", "`{GATE}`", "`{__import__('os').system('true') or GATE}`")
    assert (
        bad == "Error: candidate changes code outside the SYSTEM_PROMPT constant "
        "or its {...} placeholders"
    )
    assert path.read_text(encoding="utf-8") == ok_source
    # A single-quoted f-string takes an appended paragraph too.
    (checkout / "sdemo_sea.py").write_text(
        'GATE = "make test"\nSYSTEM_PROMPT = f"Run {GATE}. Be brief."\n\n\n'
        'def system_prompt() -> str:\n    """Prompt."""\n    return SYSTEM_PROMPT\n',
        encoding="utf-8",
    )
    assert sea.patch_sea_prompt("sdemo", "", "- Cite files by path.").startswith("Patched")
    assert (
        sea._execute_sea(checkout / "sdemo_sea.py")["system_prompt"]()
        == "Run make test. Be brief.\n\n- Cite files by path.\n"
    )
    # Replacing text that lives in the appended plain segment must not double
    # braces; text in the f-string segment must; a span across segments is refused.
    assert sea.patch_sea_prompt("sdemo", "files by path", "files {by} path").startswith("Patched")
    assert sea.patch_sea_prompt("sdemo", "Be brief", "Be {brief}").startswith("Patched")
    assert (
        sea._execute_sea(checkout / "sdemo_sea.py")["system_prompt"]()
        == "Run make test. Be {brief}.\n\n- Cite files {by} path.\n"
    )
    joined = sea.sea_prompt("sdemo")
    across = joined[joined.index('." """') : joined.index('." """') + 6]
    assert sea.patch_sea_prompt("sdemo", across, "x") == (
        "Error: old text spans several string segments; replace a shorter piece"
    )


def test_patch_sea_prompt_preserves_format_fields_and_restores_on_load_failure(
    checkout: Path,
) -> None:
    """Template fields must survive; a patch that breaks loading is rolled back."""
    assert sea.patch_sea_prompt("tdemo", "Be brief.", "Be brief and cite {source}.") == (
        "Error: candidate changes the template's replacement fields: "
        "['language', 'source', 'user'] != ['language', 'user']"
    )
    assert sea.patch_sea_prompt("tdemo", "Be brief.", "Be brief {").startswith(
        "Error: candidate breaks the prompt's str.format template"
    )
    assert sea.patch_sea_prompt("tdemo", "Be brief.", "Be very brief.").startswith("Patched")
    assert (
        sea._execute_sea(checkout / "tdemo_sea.py")["system_prompt"]()
        == "You help {user}. Answer in {language}. Be very brief."
    )
    # A SEA whose prompt getter raises when the prompt contains a marker
    # word loads as a module but fails as a SEA: the file must be restored.
    fragile = checkout / "fragile_sea.py"
    fragile.write_text(
        'SYSTEM_PROMPT = "Be good."\n\n\n'
        'def system_prompt() -> str:\n    """Prompt."""\n'
        '    if "BOOM" in SYSTEM_PROMPT:\n        raise RuntimeError("boom")\n'
        "    return SYSTEM_PROMPT\n",
        encoding="utf-8",
    )
    original = fragile.read_text(encoding="utf-8")
    report = sea.patch_sea_prompt("fragile", "", "BOOM")
    assert report.startswith("Error: the patched SEA no longer loads") and report.endswith(
        "file restored"
    )
    assert fragile.read_text(encoding="utf-8") == original


def test_write_autoroute_evidence_replaces_the_marker_block(checkout: Path) -> None:
    """The evidence block is replaced in place, stamped, and never duplicated."""
    path = checkout / "autoroute_sea.py"
    assert (
        sea.EVIDENCE_START in autoroute_sea.SYSTEM_PROMPT
        and sea.EVIDENCE_END in autoroute_sea.SYSTEM_PROMPT
    )
    table = "| model | tasks |\n|---|---|\n| model-x | 12 |\n\n- model-x: no failures in 12 tasks."
    assert sea.write_autoroute_evidence(table).startswith("Patched SYSTEM_PROMPT of")
    prompt = sea._execute_sea(path)["system_prompt"]()
    stamp = time.strftime("%Y-%m-%d", time.gmtime())
    assert prompt.count(sea.EVIDENCE_START) == 1 and prompt.count(sea.EVIDENCE_END) == 1
    block = prompt[prompt.index(sea.EVIDENCE_START) : prompt.index(sea.EVIDENCE_END)]
    assert f"refreshed {stamp} by /rsi7d" in block and table in block
    assert "No evidence recorded yet" not in prompt
    assert "## Hard rules" in prompt.split(sea.EVIDENCE_END)[1]
    assert sea.write_autoroute_evidence("| model | tasks |\n|---|---|\n| model-y | 3 |").startswith(
        "Patched"
    )
    prompt = sea._execute_sea(path)["system_prompt"]()
    assert "model-x" not in prompt and "model-y | 3" in prompt
    assert prompt.count(sea.EVIDENCE_START) == 1
    # A caller that repeats the stamp line does not duplicate it; a long
    # bullet is wrapped; a table row that does not fit is refused.
    repeated = f"{sea.STAMP_PREFIX}, refreshed 2020-01-01 by /rsi7d._\n\n- " + "word " * 40
    assert sea.write_autoroute_evidence(repeated).startswith("Patched")
    prompt = sea._execute_sea(path)["system_prompt"]()
    assert prompt.count(sea.STAMP_PREFIX) == 1 and "2020-01-01" not in prompt
    assert "\n  word word" in prompt and max(len(line) for line in prompt.splitlines()) <= 92
    wide = "| model | " + "x" * 100 + " |"
    refused = sea.write_autoroute_evidence(wide)
    assert refused.startswith("Error: a line of the new text is longer than 92 characters")
    assert "| model | xxxx" not in sea._execute_sea(path)["system_prompt"]()
    # Without markers the tool refuses instead of appending.
    source = (
        path.read_text(encoding="utf-8")
        .replace(sea.EVIDENCE_START, "")
        .replace(sea.EVIDENCE_END, "")
    )
    path.write_text(source, encoding="utf-8")
    assert sea.write_autoroute_evidence("x").startswith("Error: the autoroute prompt has no")
    path.unlink()
    assert sea.write_autoroute_evidence("x").startswith("Error: 'autoroute' is not an editable SEA")


def test_wrap_markdown_wraps_prose_and_keeps_tables_and_code() -> None:
    """Long prose wraps at 92 columns with hanging bullet indents; tables and fences are kept."""
    long_words = " ".join(f"w{i}" for i in range(40))
    text = "\n".join(
        [
            "## Title",
            f"- {long_words}",
            f"  2. {long_words}",
            f"| a | {long_words} |",
            "```",
            f"code {long_words}",
            "```",
            f"plain {long_words}",
            "x" * 120,
            "short",
        ]
    )
    wrapped = sea._wrap_markdown(text)
    lines = wrapped.splitlines()
    assert lines[0] == "## Title" and lines[-1] == "short"
    assert lines[1].startswith("- w0 ") and lines[2].startswith("  w") and len(lines[1]) <= 92
    numbered = [line for line in lines if line.startswith("  2. ")]
    assert len(numbered) == 1
    after = lines[lines.index(numbered[0]) + 1]
    assert after.startswith("     w") and not after.startswith("      ")
    assert f"| a | {long_words} |" in lines  # table rows are never wrapped
    assert f"code {long_words}" in lines  # fenced code is never wrapped
    assert any(line.startswith("plain w0") and len(line) <= 92 for line in lines)
    assert "x" * 120 in lines  # nothing to wrap on
    assert sea._too_long(wrapped).startswith("| a |")
    assert sea._too_long("fits\n" + "y" * 92) == ""
    assert sea._wrap_markdown("short\n\nlines") == "short\n\nlines"
    tilde = '~~~python\nprint("' + "abc " * 30 + '")\n~~~\n' + "z " * 60
    wrapped = sea._wrap_markdown(tilde)
    assert wrapped.startswith(tilde[: tilde.index("~~~\n", 4) + 4])  # the block is untouched
    assert wrapped.splitlines()[-2].startswith("z z") and len(wrapped.splitlines()[-1]) < 92
    nested = "````\n```\n" + "x " * 60 + "\n```\n````\n" + "y " * 60
    wrapped = sea._wrap_markdown(nested)
    assert "x " * 60 in wrapped  # the inner three-backtick fence does not close the outer one
    assert len(wrapped.splitlines()) == 7  # two fences, the x line, and the y line split in two


def test_seas_dir_falls_back_to_the_bundled_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside a KISS checkout (no task, no ``src/kiss/agents/seas`` in cwd) the SEA's dir wins."""
    monkeypatch.chdir(tmp_path)
    assert sea._seas_dir() == _SEAS_DIR
    assert sea._editable_path("sh") == _SEAS_DIR / "sh_sea.py"
    assert sea._editable_path("no-such") is None
    signatures = sea._signatures()
    assert signatures["sh"] == ast.literal_eval(repr(signatures["sh"]))  # plain text
    assert len(signatures["sh"]) == sea.SIGNATURE_CHARS and "review_paper" in signatures
    assert "dummy" not in signatures  # no prompt getter


def test_agent_run_offers_the_tools_and_patches_a_sea_through_them(
    checkout: Path,
    tmp_path: Path,
) -> None:
    """A real ReAct loop: the scripted model lists the SEAs, patches one, and finishes.

    Checks the tools offered to the model, the system prompt, that the
    tool results carry real data, and that the patch landed in the file.
    """
    section = "## Lessons from recent runs (rsi7d)\n- Report the exit code before the output."
    script = [
        tool_call_body("indexed_seas", {}, prompt_tokens=500),
        tool_call_body(
            "patch_sea_prompt", {"name": "demo", "old": "", "new": section}, prompt_tokens=600
        ),
        finish_body("<p>Patched demo.</p>", prompt_tokens=700),
    ]
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("rsi7d-sea-test")
        result = agent.run(
            prompt_template=sea.build_prompt(""),
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=5,
            max_budget=sea.max_budget(),
            model_config={"base_url": url, "api_key": "local"},
            tools=sea.tools(),
            base_system_prompt=sea.system_prompt(),
            web_tools=sea.use_web_tools(),
            use_memory=False,
            is_parallel=False,
            verbose=False,
        )
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True and parsed["summary"] == "<p>Patched demo.</p>"
    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 3, [list(r) for r in requests]
    names = {t["function"]["name"] for t in agentic[0]["tools"]}
    assert {t.__name__ for t in sea.tools()} <= names
    assert {"Bash", "run_agent", "decide", "finish"} <= names
    system = next(m for m in agentic[0]["messages"] if m["role"] == "system")
    assert str(system["content"]).startswith(sea.SYSTEM_PROMPT)
    listing = [m for m in agentic[1]["messages"] if m["role"] == "tool"][-1]
    assert '"name": "demo"' in str(listing["content"])
    patched = [m for m in agentic[2]["messages"] if m["role"] == "tool"][-1]
    assert str(patched["content"]).startswith("Patched SYSTEM_PROMPT of")
    assert sea._execute_sea(checkout / "demo_sea.py")["system_prompt"]().endswith(section + "\n")
