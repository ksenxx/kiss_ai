# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled rsi7d agent (:mod:`kiss.agents.seas.rsi7d.rsi7d_sea`).

The mining tools run against tasks persisted in the test session's real
SQLite history (``KISS_HOME`` is a temporary directory, see
``conftest.py``).  The prompt editors run against a fake KISS checkout
under ``tmp_path`` (``src/kiss/agents/seas`` holding demo SEAs with a
plain, an f-string and no prompt constant) that ``_checkout_seas_dir`` resolves
through the working directory.  The agent-level test drives a real
:class:`WorktreeSorcarAgent` ReAct loop against the scripted local
chat-completions server so a patch really flows from the model's tool
call into the SEA file.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.seas.autorouter import autorouter_sea
from kiss.agents.seas.rsi7d import rsi7d_sea as sea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.git_worktree import USER_PROMPT_HEADING
from kiss.agents.sorcar.persistence import (
    _add_task,
    _append_chat_event,
    _flush_chat_events,
    _save_task_result,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.brand import HOME_DIR, render_brand
from kiss.core.utils import rmtree_force
from kiss.server import agent_state
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)
from kiss.tests.conftest import is_root, posix_only
from kiss.tests.server.parallel_agent_harness import init_repo, run_git

_SEA_PATH = Path(sea.__file__).resolve()
_SEAS_DIR = _SEA_PATH.parents[1]

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
    """A fake KISS checkout in *tmp_path* that ``_checkout_seas_dir`` resolves through the cwd."""
    seas = tmp_path / "src" / "kiss" / "agents" / "seas"
    for name in ("rsi7d", "autorouter", "demo", "fdemo", "tdemo", "noprompt"):
        (seas / name).mkdir(parents=True)
    shutil.copy(_SEA_PATH, seas / "rsi7d" / "rsi7d_sea.py")
    autorouter = _SEAS_DIR / "autorouter" / "autorouter_sea.py"
    shutil.copy(autorouter, seas / "autorouter" / "autorouter_sea.py")
    (seas / "demo" / "demo_sea.py").write_text(_PLAIN_SEA, encoding="utf-8")
    (seas / "fdemo" / "fdemo_sea.py").write_text(_FSTRING_SEA, encoding="utf-8")
    (seas / "tdemo" / "tdemo_sea.py").write_text(_TEMPLATE_SEA, encoding="utf-8")
    (seas / "noprompt" / "noprompt_sea.py").write_text(_NOPROMPT_SEA, encoding="utf-8")
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
    # The prompt names the brand's state directory (``~/{{HOME_DIR}}/...``), rendered on read.
    assert sea.system_prompt() == render_brand(sea.SYSTEM_PROMPT)
    assert "{{HOME_DIR}}" in sea.SYSTEM_PROMPT and "{{" not in sea.system_prompt()
    assert f"~/{HOME_DIR}/MODEL_INFO.json" in sea.system_prompt()
    assert "--seas-dir" in sea.description() and "--seas-dir" in sea.SYSTEM_PROMPT
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
        "write_autorouter_evidence",
        "replay_in_clone",
        "sorcar_text",
        "request_sorcar_permission",
        "patch_sorcar",
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


def test_editable_dirs_include_the_bundled_channel_seas(checkout: Path) -> None:
    """A ``third_party_agents`` SEA next to ``seas`` is editable; ``seas`` wins a name clash."""
    assert sea._editable_dirs() == [checkout]
    third_party = checkout.parent / "third_party_agents"
    (third_party / "ask").mkdir(parents=True)
    (third_party / "ask" / "ask_sea.py").write_text(_PLAIN_SEA, encoding="utf-8")
    (third_party / "demo").mkdir()
    (third_party / "demo" / "demo_sea.py").write_text(_NOPROMPT_SEA, encoding="utf-8")
    assert sea._editable_dirs() == [checkout, third_party]
    assert sea._editable_path("ask") == third_party / "ask" / "ask_sea.py"
    assert sea._editable_seas()["demo"] == checkout / "demo" / "demo_sea.py"
    listed = json.loads(sea.indexed_seas())
    assert listed["scope"] == {"seas_dir": "", "names": []}
    rows = {row["name"]: row for row in listed["seas"]}
    assert rows["ask"]["editable_path"] == str(third_party / "ask" / "ask_sea.py")
    assert rows["ask"]["prompt_constant"] == "SYSTEM_PROMPT"
    assert rows["demo"]["editable_path"] == str(checkout / "demo" / "demo_sea.py")
    assert "ask" in sea._signatures()
    section = "## Lessons from recent runs (rsi7d)\n- Cite."
    assert sea.patch_sea_prompt("ask", "", section).startswith("Patched SYSTEM_PROMPT of")
    assert "- Cite." in (third_party / "ask" / "ask_sea.py").read_text(encoding="utf-8")


def test_indexed_seas_reports_editable_paths_and_prompt_shapes(checkout: Path) -> None:
    """Bundled SEAs are editable with their prompt shape; registered foreign SEAs are not."""
    rows = {r["name"]: r for r in json.loads(sea.indexed_seas())["seas"]}
    assert rows["demo"]["editable_path"] == str(checkout / "demo" / "demo_sea.py")
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
    agent_path = str(checkout / "demo" / "demo_sea.py")
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
    fdemo = checkout / "fdemo" / "fdemo_sea.py"
    fdemo_prompt = sea._execute_sea(fdemo)["append_to_system_prompt"]()
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
    agent_path = str(checkout / "noprompt" / "noprompt_sea.py")
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
        f"# {checkout / 'demo' / 'demo_sea.py'}\n"
        "# system_prompt() returns SYSTEM_PROMPT, a plain string"
    )
    section = "## Lessons from recent runs (rsi7d)\n- Batch independent greps into one Bash call."
    report = sea.patch_sea_prompt("demo", "", section)
    assert report.startswith("Patched SYSTEM_PROMPT of") and "+3 lines" in report
    prompt = sea._execute_sea(checkout / "demo" / "demo_sea.py")["system_prompt"]()
    assert prompt.endswith("Never guess: read the file before editing it. \n\n" + section + "\n")
    assert sea.patch_sea_prompt(
        "demo", "one Bash call.", "one Bash call, never one per grep."
    ).startswith("Patched")
    prompt = sea._execute_sea(checkout / "demo" / "demo_sea.py")["system_prompt"]()
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
    path = checkout / "fdemo" / "fdemo_sea.py"
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
    (checkout / "sdemo").mkdir()
    (checkout / "sdemo" / "sdemo_sea.py").write_text(
        'GATE = "make test"\nSYSTEM_PROMPT = f"Run {GATE}. Be brief."\n\n\n'
        'def system_prompt() -> str:\n    """Prompt."""\n    return SYSTEM_PROMPT\n',
        encoding="utf-8",
    )
    assert sea.patch_sea_prompt("sdemo", "", "- Cite files by path.").startswith("Patched")
    assert (
        sea._execute_sea(checkout / "sdemo" / "sdemo_sea.py")["system_prompt"]()
        == "Run make test. Be brief.\n\n- Cite files by path.\n"
    )
    # Replacing text that lives in the appended plain segment must not double
    # braces; text in the f-string segment must; a span across segments is refused.
    assert sea.patch_sea_prompt("sdemo", "files by path", "files {by} path").startswith("Patched")
    assert sea.patch_sea_prompt("sdemo", "Be brief", "Be {brief}").startswith("Patched")
    assert (
        sea._execute_sea(checkout / "sdemo" / "sdemo_sea.py")["system_prompt"]()
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
        sea._execute_sea(checkout / "tdemo" / "tdemo_sea.py")["system_prompt"]()
        == "You help {user}. Answer in {language}. Be very brief."
    )
    # A SEA whose prompt getter raises when the prompt contains a marker
    # word loads as a module but fails as a SEA: the file must be restored.
    fragile = checkout / "fragile" / "fragile_sea.py"
    fragile.parent.mkdir()
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


def test_write_autorouter_evidence_rewrites_the_kiss_home_file(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evidence lands in ``$KISS_HOME/AUTOROUTER.md``, stamped, and the SEA file is untouched.

    The autorouter SEA splices the file into its prompt when it loads; a
    missing or empty file leaves the "no evidence yet" sentence in place.
    """
    home = tmp_path / "home"
    monkeypatch.setenv("KISS_HOME", str(home))
    path = checkout / "autorouter" / "autorouter_sea.py"
    source = path.read_text(encoding="utf-8")
    prompt = sea._execute_sea(path)["add_to_system_prompt"]()
    assert autorouter_sea.NO_EVIDENCE in prompt and "{observed_evidence()}" not in prompt
    table = "| model | tasks |\n|---|---|\n| model-x | 12 |\n\n- model-x: no failures in 12 tasks."
    # The SEA stamps the file with its own UTC date; a call straddling midnight
    # may legitimately carry either day's stamp, but the file and the report
    # must carry the same one.
    before = time.strftime("%Y-%m-%d", time.gmtime())
    report = sea.write_autorouter_evidence(table)
    after = time.strftime("%Y-%m-%d", time.gmtime())
    written = (home / "AUTOROUTER.md").read_text(encoding="utf-8")
    assert (written, report) in {
        (
            f"{sea.STAMP_PREFIX}, refreshed {stamp} by /rsi7d._\n\n{table}\n",
            f"Wrote {home / 'AUTOROUTER.md'} (5 lines, {len(written)} chars, refreshed {stamp})",
        )
        for stamp in (before, after)
    }
    assert path.read_text(encoding="utf-8") == source
    prompt = sea._execute_sea(path)["add_to_system_prompt"]()
    assert written.strip() in prompt and autorouter_sea.NO_EVIDENCE not in prompt
    assert prompt.index("## Observed model evidence") < prompt.index(table) < prompt.index(
        "## Hard rules"
    )
    table_y = "| model | tasks |\n|---|---|\n| model-y | 3 |"
    assert sea.write_autorouter_evidence(table_y).startswith("Wrote ")
    prompt = sea._execute_sea(path)["add_to_system_prompt"]()
    assert "model-x" not in prompt and "model-y | 3" in prompt
    # A caller that repeats the stamp line does not duplicate it; a long
    # bullet is wrapped; a table row that does not fit is refused; an
    # empty text is refused; nothing partial is left behind.
    repeated = f"{sea.STAMP_PREFIX}, refreshed 2020-01-01 by /rsi7d._\n\n- " + "word " * 40
    assert sea.write_autorouter_evidence(repeated).startswith("Wrote ")
    prompt = sea._execute_sea(path)["add_to_system_prompt"]()
    assert prompt.count(sea.STAMP_PREFIX) == 1 and "2020-01-01" not in prompt
    assert "\n  word word" in prompt and max(len(line) for line in prompt.splitlines()) <= 92
    wide = "| model | " + "x" * 100 + " |"
    refused = sea.write_autorouter_evidence(wide)
    assert refused.startswith("Error: a line of the new text is longer than 92 characters")
    assert sea.write_autorouter_evidence("  \n") == "Error: the evidence text is empty"
    assert "word word" in sea._execute_sea(path)["add_to_system_prompt"]()
    assert sorted(p.name for p in home.iterdir()) == ["AUTOROUTER.md"]
    # The file goes into every autorouter prompt, so the stamped text is
    # capped at the size the autorouter cuts at; a text that would exceed
    # it is refused and the file kept.  A file over the cap written by
    # other means (a hand edit, a cross-machine merge) reaches the prompt
    # cut at a line boundary with the cut marker, never whole.
    assert sea.EVIDENCE_MAX_CHARS == autorouter_sea.EVIDENCE_MAX_CHARS == 2500
    rows = "\n".join(f"| model-{i:03d} | {i} |" for i in range(200))
    refused = sea.write_autorouter_evidence(f"| model | tasks |\n|---|---|\n{rows}")
    assert refused.startswith("Error: the evidence is ") and "at most 2500" in refused
    assert "word word" in (home / "AUTOROUTER.md").read_text(encoding="utf-8")
    stamped = f"{sea.STAMP_PREFIX}, refreshed {after} by /rsi7d._\n\n{rows}\n"
    assert len(stamped) > sea.EVIDENCE_MAX_CHARS
    (home / "AUTOROUTER.md").write_text(stamped, encoding="utf-8")
    prompt = sea._execute_sea(path)["add_to_system_prompt"]()
    spliced = prompt[prompt.index(sea.STAMP_PREFIX) : prompt.index("\n## Hard rules")].strip()
    kept, _blank, marker = spliced.rsplit("\n", 2)
    assert marker == autorouter_sea.EVIDENCE_CUT and _blank == ""
    assert stamped.startswith(kept + "\n") and len(kept) <= sea.EVIDENCE_MAX_CHARS
    assert kept.endswith(" |") and "| model-199 |" not in prompt
    # A file exactly at the cap is spliced whole.
    exact = stamped[: sea.EVIDENCE_MAX_CHARS]
    (home / "AUTOROUTER.md").write_text(exact, encoding="utf-8")
    prompt = sea._execute_sea(path)["add_to_system_prompt"]()
    assert exact.strip() in prompt and autorouter_sea.EVIDENCE_CUT not in prompt
    # Blank file: the SEA falls back to the sentence.  No autorouter SEA in
    # the editable folders: the tool refuses.
    (home / "AUTOROUTER.md").write_text("\n", encoding="utf-8")
    assert autorouter_sea.NO_EVIDENCE in sea._execute_sea(path)["add_to_system_prompt"]()
    path.unlink()
    refused = sea.write_autorouter_evidence("x")
    assert refused.startswith("Error: 'autorouter' is not an editable SEA")


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
    assert sea._checkout_seas_dir() == _SEAS_DIR
    assert sea._editable_path("sh") == _SEAS_DIR / "sh" / "sh_sea.py"
    assert sea._editable_path("no-such") is None
    # An old-layout checkout (flat ``seas/rsi7d_sea.py``) is still the editable directory:
    # a replayed sweep must never fall through to the checkout the SEA was loaded from.
    old_layout = tmp_path / "src" / "kiss" / "agents" / "seas"
    old_layout.mkdir(parents=True)
    (old_layout / "rsi7d_sea.py").write_text("SYSTEM_PROMPT = 'x'\n", encoding="utf-8")
    assert sea._checkout_seas_dir() == old_layout.resolve()
    assert sea._editable_path("sh") is None
    init_repo(tmp_path)  # from a sub-directory of a checkout, the checkout's seas dir wins
    (tmp_path / "docs").mkdir()
    monkeypatch.chdir(tmp_path / "docs")
    assert sea._checkout_seas_dir() == old_layout.resolve()
    monkeypatch.chdir(tmp_path)
    shutil.rmtree(tmp_path / "src")
    rmtree_force(tmp_path / ".git")  # git's objects are read-only on Windows
    signatures = sea._signatures()
    assert signatures["sh"] == ast.literal_eval(repr(signatures["sh"]))  # plain text
    assert len(signatures["sh"]) == sea.SIGNATURE_CHARS and "review_paper" in signatures
    assert "dummy" not in signatures  # no prompt getter


def test_parse_scope_reads_leading_names_and_the_seas_dir_option(
    checkout: Path, tmp_path: Path
) -> None:
    """The scope is the leading SEA names plus ``--seas-dir``; other text ends the names."""
    assert sea.parse_scope("all") == sea.Scope()
    assert sea.parse_scope("") == sea.Scope()
    # The weekly cron relay: ``all.`` stops the names, the later ``rsi7d`` is free text.
    assert sea.parse_scope("all. Optimize every SEA including rsi7d itself.") == sea.Scope()
    assert sea.parse_scope("demo").names == ("demo",)
    assert sea.parse_scope("demo, fdemo demo. Replay the costliest run.").names == ("demo", "fdemo")
    assert sea.parse_scope("only demo").names == ()  # "only" is no SEA: unscoped
    assert sea.parse_scope("slack demo").names == ("slack", "demo")  # registered, not editable
    assert sea.parse_scope("my-sea").names == ()  # unknown; a hyphenated name would be accepted
    folder = tmp_path / "userseas"
    (folder / "alpha").mkdir(parents=True)
    (folder / "alpha" / "alpha_sea.py").write_text(_PLAIN_SEA, encoding="utf-8")
    scope = sea.parse_scope(f"--seas-dir {folder} alpha. Be brief.")
    assert scope == sea.Scope(folder.resolve(), ("alpha",))
    assert sea.parse_scope(f"alpha --seas-dir={folder}").names == ("alpha",)
    assert sea.parse_scope("--seas-dir userseas alpha") == scope  # relative to the work dir
    home = Path.home()
    tilde = Path("~") / folder.relative_to(home) if folder.is_relative_to(home) else folder
    assert sea.parse_scope(f"--seas-dir {tilde}").seas_dir == folder.resolve()
    # With a folder only its SEAs are names: neither the checkout's nor registered ones.
    assert sea.parse_scope(f"--seas-dir {folder} demo").names == ()
    assert sea.parse_scope(f"--seas-dir {folder} slack alpha").names == ()
    missing = (tmp_path / "no" / "such" / "dir").resolve()
    assert sea.parse_scope("--seas-dir no/such/dir alpha").as_dict() == {
        "seas_dir": str(missing),
        "names": [],
        "error": f"--seas-dir {missing} is not a directory",
    }
    for text in ("--seas-dir", "alpha --seas-dir", "--seas-dir=", "--seas-dir\nalpha"):
        assert sea.parse_scope(text) == sea.Scope(error="--seas-dir needs a folder"), text
    unusable = sea.parse_scope("--seas-dir ~no_such_user_rsi7d/seas alpha")
    nul = sea.parse_scope("--seas-dir /bad\x00dir")
    if os.name == "nt":
        # ntpath invents a home for any user name and resolves a NUL byte
        # without complaint; both end up as missing directories.
        assert unusable.error.endswith("no_such_user_rsi7d\\seas is not a directory")
        assert nul.error.endswith("bad\x00dir is not a directory")
    else:
        assert unusable.error.startswith("--seas-dir '~no_such_user_rsi7d/seas': ")
        assert unusable.names == () and unusable.seas_dir is None
        assert nul.error.startswith("--seas-dir '/bad\\x00dir': ")
    assert sea.parse_scope("demo").as_dict() == {"seas_dir": "", "names": ["demo"]}


def _run_rsi7d(task: str, script: list[bytes], work_dir: Path) -> tuple[Any, list[Any]]:
    """Run the rsi7d SEA on *task* against a scripted model; return the parsed result and requests.

    The agent is registered under the calling (task) thread the way the
    daemon registers a run before starting it (``commands.py``), which
    is how the SEA's tools find their task text through ``current_agent``.
    """
    agent = WorktreeSorcarAgent("rsi7d-sea-test")
    state = agent_state.AgentState(
        "rsi7d-sea-test", agent=agent, task_thread=threading.current_thread(), is_task_active=True
    )
    agent_state.register(state)
    try:
        result = _run_registered(agent, task, script, work_dir)
    finally:
        agent_state.unregister(state.task_id, state)
    return yaml.safe_load(result[0]), [r for r in result[1] if r.get("tools")]


def _run_registered(
    agent: WorktreeSorcarAgent, task: str, script: list[bytes], work_dir: Path
) -> tuple[str, list[dict[str, Any]]]:
    """Run *agent* on *task* against the scripted model; return the raw result and requests."""
    with serve(script) as (url, requests):
        result = agent.run(
            prompt_template=task,
            use_worktree=False,
            model_name=MODEL,
            work_dir=str(work_dir),
            max_steps=8,
            max_budget=sea.max_budget(),
            model_config={"base_url": url, "api_key": "local"},
            tools=sea.tools(),
            base_system_prompt=sea.system_prompt(),
            web_tools=sea.use_web_tools(),
            use_memory=False,
            is_parallel=False,
            verbose=False,
        )
    return result, requests


def _tool_results(agentic: list[Any]) -> list[str]:
    """Return the last tool result the model saw before each request after the first."""
    return [
        str([m for m in request["messages"] if m["role"] == "tool"][-1]["content"]).split(
            "\n\nSteps: "
        )[0]
        for request in agentic[1:]
    ]


def test_agent_run_offers_the_tools_and_patches_a_sea_through_them(
    checkout: Path,
    tmp_path: Path,
) -> None:
    """A real ReAct loop: the scripted model lists the SEAs, patches one, and finishes.

    Checks the tools offered to the model, the system prompt, that the
    tool results carry real data, and that the patch landed in the file.
    The task ``demo fdemo. …`` scopes the run: the listing holds those
    two SEAs only, and patching ``tdemo`` is refused.
    """
    section = "## Lessons from recent runs (rsi7d)\n- Report the exit code before the output."
    script = [
        tool_call_body("indexed_seas", {}, prompt_tokens=500),
        tool_call_body(
            "patch_sea_prompt", {"name": "tdemo", "old": "", "new": section}, prompt_tokens=550
        ),
        tool_call_body(
            "patch_sea_prompt", {"name": "demo", "old": "", "new": section}, prompt_tokens=600
        ),
        finish_body("<p>Patched demo.</p>", prompt_tokens=700),
    ]
    parsed, agentic = _run_rsi7d("demo fdemo. Patch demo only.", script, tmp_path)
    assert parsed["success"] is True and parsed["summary"] == "<p>Patched demo.</p>", parsed
    assert len(agentic) == 4
    names = {t["function"]["name"] for t in agentic[0]["tools"]}
    assert {t.__name__ for t in sea.tools()} <= names
    # ``decide`` is not asserted: the Jev tool follows the "Use Jev"
    # setting and the OpenRouter key, not the SEA's tool list.
    assert {"Bash", "run_agent", "finish"} <= names
    system = next(m for m in agentic[0]["messages"] if m["role"] == "system")
    assert str(system["content"]).startswith(sea.system_prompt())
    listing, refused, patched = _tool_results(agentic)
    scoped = json.loads(listing)
    assert scoped["scope"] == {"seas_dir": "", "names": ["demo", "fdemo"]}
    assert [row["name"] for row in scoped["seas"]] == ["demo", "fdemo"]
    assert refused == "Error: 'tdemo' is outside this run's scope ['demo', 'fdemo']"
    assert patched.startswith("Patched SYSTEM_PROMPT of")
    demo_prompt = sea._execute_sea(checkout / "demo" / "demo_sea.py")["system_prompt"]()
    assert demo_prompt.endswith(section + "\n")
    assert section not in (checkout / "tdemo" / "tdemo_sea.py").read_text(encoding="utf-8")


def test_agent_run_with_seas_dir_edits_only_that_folder(checkout: Path, tmp_path: Path) -> None:
    """``--seas-dir <folder>`` makes the folder's SEAs the editable ones and hides the rest.

    The listing holds only ``alpha``; its runs are the only ones mined;
    the checkout's ``demo`` is refused; the autorouter evidence refresh
    is refused because ``autorouter`` is out of scope; ``alpha`` is
    patched in the user folder.
    """
    folder = tmp_path / "userseas"
    (folder / "alpha").mkdir(parents=True)
    alpha = folder / "alpha" / "alpha_sea.py"
    alpha.write_text(_PLAIN_SEA, encoding="utf-8")
    parent = _persist(
        "/alpha go",
        [_dispatch(str(alpha), "go"), _dispatch(str(checkout / "demo" / "demo_sea.py"), "do")],
    )
    _persist("go", [_result_event(True)], result="<p>ok</p>", parent_task_id=parent)
    _persist("do", [_result_event(True)], result="<p>ok</p>", parent_task_id=parent)
    section = "## Lessons from recent runs (rsi7d)\n- Cite the run id."
    script = [
        tool_call_body("indexed_seas", {}, prompt_tokens=500),
        tool_call_body("sea_runs", {}, prompt_tokens=520),
        tool_call_body(
            "patch_sea_prompt", {"name": "demo", "old": "", "new": section}, prompt_tokens=550
        ),
        tool_call_body("write_autorouter_evidence", {"text": "| m | t |"}, prompt_tokens=580),
        tool_call_body(
            "patch_sea_prompt", {"name": "alpha", "old": "", "new": section}, prompt_tokens=600
        ),
        finish_body("<p>Patched alpha.</p>", prompt_tokens=700),
    ]
    parsed, agentic = _run_rsi7d(f"--seas-dir {folder}", script, tmp_path)
    assert parsed["success"] is True and parsed["summary"] == "<p>Patched alpha.</p>"
    listing, runs, refused, evidence, patched = _tool_results(agentic)
    scoped = json.loads(listing)
    assert scoped["scope"] == {"seas_dir": str(folder.resolve()), "names": []}
    assert [row["name"] for row in scoped["seas"]] == ["alpha"]
    assert scoped["seas"][0]["editable_path"] == str(alpha.resolve())
    assert list(json.loads(runs)["seas"]) == ["alpha"]
    assert refused == f"Error: 'demo' is not an editable SEA under {folder.resolve()}"
    assert evidence == f"Error: 'autorouter' is not an editable SEA under {folder.resolve()}"
    assert patched.startswith("Patched SYSTEM_PROMPT of")
    assert sea._execute_sea(alpha)["system_prompt"]().endswith(section + "\n")
    assert section not in (checkout / "demo" / "demo_sea.py").read_text(encoding="utf-8")


def test_agent_run_with_an_unusable_seas_dir_lists_mines_and_edits_nothing(
    checkout: Path, tmp_path: Path
) -> None:
    """A ``--seas-dir`` that is not a directory is an error every tool reports, not a sweep."""
    _persist("/demo go", [_dispatch(str(checkout / "demo" / "demo_sea.py"), "go")])
    script = [
        tool_call_body("indexed_seas", {}, prompt_tokens=500),
        tool_call_body("sea_runs", {}, prompt_tokens=520),
        tool_call_body("sea_findings", {"name": "demo"}, prompt_tokens=540),
        tool_call_body(
            "patch_sea_prompt", {"name": "demo", "old": "", "new": "x"}, prompt_tokens=560
        ),
        finish_body("<p>Bad folder.</p>", prompt_tokens=700),
    ]
    parsed, agentic = _run_rsi7d("--seas-dir no/such/dir demo", script, tmp_path)
    assert parsed["success"] is True
    listing, runs, findings, patched = _tool_results(agentic)
    missing = (tmp_path / "no" / "such" / "dir").resolve()
    error = f"--seas-dir {missing} is not a directory"
    assert json.loads(listing) == {
        "scope": {"seas_dir": str(missing), "names": [], "error": error},
        "seas": [],
    }
    assert json.loads(runs) == {"error": error, "seas": {}}
    assert findings == patched == f"Error: {error}"


def _commit(
    repo: Path, message: str, files: dict[str, str] | None = None, when: int = 0
) -> str:
    """Write *files* into *repo*, commit them with *message* (committed at *when*, a unix
    time, when given) and return the commit sha."""
    for name, text in (files or {}).items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")
    run_git(repo, "add", "-A")
    env = dict(os.environ, GIT_COMMITTER_DATE=f"@{when} +0000") if when else None
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", message],
        cwd=repo, env=env, capture_output=True, check=True,
    )
    return run_git(repo, "rev-parse", "HEAD").stdout.strip()


def test_prepare_replay_clone_checks_out_the_state_before_the_tasks_auto_commit(
    checkout: Path, tmp_path: Path
) -> None:
    """A file-modifying task is replayed from the first parent of its auto-commit.

    The run's work dir is a sub-directory of a removed worktree of the
    repository; the commit is found through the ``User prompt:`` block
    of the commit message (whitespace-insensitively, the whole prompt,
    made after the task started), repository and worktree paths in the
    task are rewritten to the clone in one pass although the clone lives
    inside the repository, and the replay runs in the same sub-directory
    of the clone.  A second preparation replaces the clone.
    """
    repo = tmp_path  # the clone under tmp/rsi7d/replays is inside the repository
    init_repo(repo)
    worktree = repo / ".kiss-worktrees" / "kiss_wt-gone"
    _commit(repo, "seed paper", {"docs/paper.tex": "v1\n"})
    task = f"Update the paper at {worktree}/docs/paper.tex  with the results\nin {repo}/results.md"
    start_s = int(time.time()) - 60
    _commit(repo, f"same task, earlier{USER_PROMPT_HEADING}{task}", when=start_s - 30)
    _commit(repo, f"other task{USER_PROMPT_HEADING}{task} and publish it")
    broader = _commit(repo, f"other task{USER_PROMPT_HEADING}{task} Result: provide a table")
    auto = _commit(
        repo,
        f"docs: update the paper{USER_PROMPT_HEADING}{task}\n\nResult:\nUpdated.",
        {"docs/paper.tex": "v2\n"},
    )
    task_id = _persist(task, [], work_dir=str(worktree / "docs"), sea="demo_sea")

    prepared = sea.prepare_replay_clone(task_id)
    assert isinstance(prepared, dict), prepared
    clone = Path(prepared["clone"])
    assert clone == tmp_path / sea.REPLAY_DIR / f"demo-{task_id[:8]}"
    assert prepared["commit"] == broader  # the earlier and the two broader commits do not match
    assert prepared["commit_source"] == f"first parent of the task's auto-commit {auto[:12]}"
    assert run_git(clone, "rev-parse", "HEAD").stdout.strip() == broader
    assert (clone / "docs" / "paper.tex").read_text(encoding="utf-8") == "v1\n"
    assert prepared["task"] == (
        f"Update the paper at {clone}/docs/paper.tex  with the results\nin {clone}/results.md"
    )
    assert prepared["work_dir"] == str(clone / "docs")
    assert prepared["repo"] == str(repo.resolve())
    assert prepared["sea"] == "demo" and prepared["model"] == "model-a"
    assert prepared["sea_file"] == str(checkout / "demo" / "demo_sea.py")
    assert (repo / "docs" / "paper.tex").read_text(encoding="utf-8") == "v2\n"  # untouched

    (clone / "stray.txt").write_text("x", encoding="utf-8")
    again = sea.prepare_replay_clone(task_id, name="demo")
    assert isinstance(again, dict) and again["clone"] == str(clone)
    assert not (clone / "stray.txt").exists()


def test_task_tree_resolves_live_dirs_through_git_and_removed_worktrees_by_name(
    tmp_path: Path,
) -> None:
    """An existing directory is its own checkout (nested repos, live worktrees, sub-dirs)."""
    outer = tmp_path / "outer"
    init_repo(outer)
    (outer / "docs").mkdir()
    inner = outer / "tmp" / "inner"
    init_repo(inner)
    run_git(outer, "worktree", "add", "-q", "-b", "wt", ".kiss-worktrees/kiss_wt-live")
    live = outer / ".kiss-worktrees" / "kiss_wt-live"
    (live / "docs").mkdir()
    resolved = outer.resolve()
    assert sea._task_tree(str(outer)) == (resolved, resolved, Path("."))
    assert sea._task_tree(str(outer / "docs")) == (resolved, resolved, Path("docs"))
    assert sea._task_tree(str(inner)) == (inner.resolve(), inner.resolve(), Path("."))
    assert sea._task_tree(str(live)) == (resolved, live.resolve(), Path("."))
    assert sea._task_tree(str(live / "docs")) == (resolved, live.resolve(), Path("docs"))
    nested = live / "tmp" / "nested"
    init_repo(nested)  # an independent repository under a live worktree is its own tree
    assert sea._task_tree(str(nested)) == (nested.resolve(), nested.resolve(), Path("."))
    gone = outer / ".kiss-worktrees" / "kiss_wt-gone"
    assert sea._task_tree(str(gone / "a" / "b")) == (resolved, gone, Path("a") / "b")
    assert sea._task_tree(str(tmp_path / "none" / ".kiss-worktrees" / "kiss_wt-x")) is None
    assert sea._task_tree(str(tmp_path / "missing")) is None
    assert sea._task_tree("") is None


def test_prepare_replay_clone_falls_back_to_head_and_reports_unusable_runs(
    checkout: Path, tmp_path: Path
) -> None:
    """Without an auto-commit the repository HEAD at the task's start is used; bad runs error."""
    repo = tmp_path / "repo"
    init_repo(repo)
    head = _commit(repo, "second", {"notes.md": "n\n"})
    future = int(time.time() * 1000) + 10_000
    task_id = _persist("Summarize the repo", [], work_dir=str(repo), sea="demo_sea", startTs=future)
    prepared = sea.prepare_replay_clone(task_id)
    assert isinstance(prepared, dict), prepared
    assert prepared["commit"] == head
    assert prepared["commit_source"] == "HEAD of the repository when the task started"
    assert prepared["task"] == "Summarize the repo" and prepared["work_dir"] == prepared["clone"]

    legacy = _persist("Summarize the repo", [], work_dir=str(repo), sea="demo_sea", startTs=0)
    prepared = sea.prepare_replay_clone(legacy)  # start falls back to the insertion time
    assert isinstance(prepared, dict) and prepared["commit"] == head

    # The history records a worktree run's work dir as the repository; the task text
    # still names the worktree, which is rewritten to the clone as well.
    stripped = _persist(
        f"Edit {repo}/.kiss-worktrees/kiss_wt-old/docs/paper.tex and {repo}/notes.md",
        [], work_dir=str(repo), sea="demo_sea", startTs=future,
    )
    prepared = sea.prepare_replay_clone(stripped)
    assert isinstance(prepared, dict), prepared
    assert prepared["task"] == (
        f"Edit {prepared['clone']}/docs/paper.tex and {prepared['clone']}/notes.md"
    )

    # A work dir reached through a symlink (macOS tempdirs live under /var ->
    # /private/var): git reports the real tree, the task text names the link.
    (repo / "docs").mkdir()
    link = tmp_path / "link"
    link.symlink_to(repo, target_is_directory=True)
    linked = _persist(
        f"Edit {link}/docs/paper.tex, {link}/notes.md and "
        f"{link}/.kiss-worktrees/kiss_wt-old/notes.md",
        [], work_dir=str(link / "docs"), sea="demo_sea", startTs=future,
    )
    prepared = sea.prepare_replay_clone(linked)
    assert isinstance(prepared, dict), prepared
    assert prepared["task"] == (
        f"Edit {prepared['clone']}/docs/paper.tex, {prepared['clone']}/notes.md and "
        f"{prepared['clone']}/notes.md"
    )
    clone_docs = str(Path(prepared["clone"]) / "docs")  # native separator on Windows
    assert prepared["work_dir"] == clone_docs

    # A symlink INTO the tree names only its own directory of the clone; the
    # link's parent is not part of the checkout and stays as it is.
    docs_link = tmp_path / "docs-link"
    docs_link.symlink_to(repo / "docs", target_is_directory=True)
    into = _persist(
        f"Edit {docs_link}/paper.tex; see {tmp_path}/reference.md",
        [], work_dir=str(docs_link), sea="demo_sea", startTs=future,
    )
    prepared = sea.prepare_replay_clone(into)
    assert isinstance(prepared, dict), prepared
    clone_docs = str(Path(prepared["clone"]) / "docs")
    assert prepared["task"] == f"Edit {clone_docs}/paper.tex; see {tmp_path}/reference.md"
    assert prepared["work_dir"] == clone_docs

    rooted = tmp_path / "rooted"
    rooted.mkdir()
    run_git(rooted, "init", "-q")
    run_git(rooted, "config", "user.email", "kiss-test@example.com")
    run_git(rooted, "config", "user.name", "Kiss Test")
    root = _commit(rooted, f"first{USER_PROMPT_HEADING}Start the notes", {"notes.md": "n\n"})
    first = _persist("Start the notes", [], work_dir=str(rooted), sea="demo_sea")
    prepared = sea.prepare_replay_clone(first)  # a root auto-commit has no parent to go back to
    assert isinstance(prepared, dict) and prepared["commit"] == root, prepared

    early = _persist("Summarize the repo", [], work_dir=str(repo), sea="demo_sea", startTs=1_000)
    assert sea.prepare_replay_clone(early) == (
        f"Error: {repo.resolve()} has no commit from before the task started"
    )
    assert sea.prepare_replay_clone("nope") == "Error: unknown task id 'nope'"
    no_sea = _persist("Plain sub-agent task", [], work_dir=str(repo))
    # A run without a SEA is a plain KISS Sorcar run; the fake checkout is
    # not a git repository, so there is no SYSTEM.md to replay it with.
    assert sea.prepare_replay_clone(no_sea) == (
        f"Error: run {no_sea} is a plain KISS Sorcar run and cannot be replayed from here: "
        f"{(tmp_path / 'src' / 'kiss').resolve()} is not inside a git checkout; only "
        "AGENTS.md can be changed here"
    )
    assert sea.prepare_replay_clone(no_sea, name="slack") == (
        f"Error: 'slack' is not an editable SEA under {', '.join(map(str, sea._editable_dirs()))}"
    )
    (tmp_path / "plain").mkdir()
    for work_dir in (str(tmp_path / "plain"), str(tmp_path / "missing"), ""):
        no_git = _persist("Plain task", [], work_dir=work_dir, sea="demo_sea")
        assert sea.prepare_replay_clone(no_git) == (
            f"Error: the run's work dir {work_dir!r} is not inside a git repository, so there "
            "is nothing to clone; mark the run not replay-verified"
        )
    assert sea.replay_in_clone("nope", max_budget=1.0) == "Error: unknown task id 'nope'"


@posix_only("chmod-based directory write denial")
@pytest.mark.skipif(is_root(), reason="root ignores directory permissions")
def test_prepare_replay_clone_reports_a_failed_git_command(
    checkout: Path, tmp_path: Path
) -> None:
    """A git failure while making the clone is reported, not raised."""
    repo = tmp_path / "repo"
    init_repo(repo)
    task_id = _persist(
        "Describe the seed", [], work_dir=str(repo), sea="demo_sea",
        startTs=int(time.time() * 1000) + 10_000,
    )
    replays = tmp_path / sea.REPLAY_DIR
    replays.mkdir(parents=True)
    replays.chmod(0o555)  # git cannot create the clone directory
    try:
        result = sea.prepare_replay_clone(task_id)
    finally:
        replays.chmod(0o755)
    assert isinstance(result, str) and result.startswith("Error: git clone failed:"), result
