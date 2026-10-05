# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``rsi7d``'s code, settings and eval-set tools (``sea_tuning`` and the gated editors).

The pure functions are checked on literal data; the tools run against a
fake checkout (the ``checkout`` fixture below) and the session's
temporary history database (``KISS_HOME``, see ``conftest.py``), so
``patch_sea_settings``'s acceptance test and ``export_sea_evals`` read
real persisted runs recorded with the ``sea`` column.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.rsi7d import rsi7d_sea as sea
from kiss.agents.seas.rsi7d import sea_tuning
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.persistence import (
    _add_task,
    _append_chat_event,
    _flush_chat_events,
    _save_task_result,
)


def _returned_literal(source: str, function: str) -> Any:
    """Evaluate the literal that ``def <function>`` in ``source`` returns."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            ret = node.body[-1]
            assert isinstance(ret, ast.Return) and ret.value is not None
            return ast.literal_eval(ret.value)
    raise AssertionError(f"no def {function} in source")


DEMO_SEA = '''"""Demo SEA: a worker with a tool."""

SYSTEM_PROMPT = "You run demo tasks. Report the exit code."


def description() -> str:
    """Help text."""
    return "Runs demo tasks."


def settings() -> dict:
    """Run settings."""
    return {"kind": "worker", "tool_profile": "bash"}


def system_prompt() -> str:
    """Replace the default prompt."""
    return SYSTEM_PROMPT


def count_words(text: str) -> str:
    """Return the number of words in *text*."""
    return str(len(text.split()))


def add_to_tools() -> list:
    """Extra tools."""
    return [count_words]
'''


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake KISS checkout whose ``seas/`` folder holds the editable ``demo`` SEA."""
    seas = tmp_path / "src" / "kiss" / "agents" / "seas"
    (seas / "tunedemo").mkdir(parents=True)
    (seas / "tunedemo" / "tunedemo_sea.py").write_text(DEMO_SEA, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return seas


def persist_run(
    task: str, result: str, success: bool, cost: float, duration_s: float, **extra: Any
) -> str:
    """Persist a finished ``demo`` run (the daemon records the SEA in the ``sea`` column)."""
    now_ms = int(time.time() * 1000)
    payload: dict[str, Any] = {
        "model": "model-a",
        "work_dir": "/work/dir",
        "sea": "tunedemo",
        "startTs": now_ms - int(duration_s * 1000),
        "endTs": now_ms,
        "cost": cost,
        "steps": 3,
        "tokens": 1000,
        **extra,
    }
    task_id, _chat = _add_task(task, extra=payload)
    text = f"success: {'true' if success else 'false'}\nsummary: {result}"
    _append_chat_event({"type": "result", "text": text}, task_id=task_id)
    _flush_chat_events(task_id)
    _save_task_result(text, task_id=task_id)
    return task_id


# --- pure functions -------------------------------------------------------


def test_percentile_and_limit_proposals() -> None:
    assert sea_tuning.percentile([], 0.95) == 0.0
    assert sea_tuning.percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert sea_tuning.percentile([1.0, 2.0, 3.0, 4.0, 100.0], 0.95) == 100.0
    runs = [
        {"status": "success", "duration_s": 100.0, "cost": 0.4, "result": "ok"},
        {"status": "success", "duration_s": 200.0, "cost": 0.8, "result": "ok"},
        {"status": "unsuccessful", "duration_s": 500.0, "cost": 1.0, "result": "x"},
        {"status": "running", "duration_s": 5.0, "cost": 0.1, "result": ""},
    ]
    report = sea_tuning.propose_settings(runs, {"timeout": 3600, "max_budget": 5.0})
    assert (report["runs"], report["finished"], report["stopped_by_limit"]) == (4, 3, 0)
    # 2 x p95 (500) = 1000 s rounded up to a minute; 1.5 x p95 (1.0) = 1.5 $.
    assert report["proposals"]["timeout"] == {
        "current": 3600,
        "proposed": 1020,
        "basis": "2 x p95 of 3 runs (500.0), rounded up to 60",
    }
    assert report["proposals"]["max_budget"]["proposed"] == 1.5 and report["flags"] == []
    # Too few runs: no proposal.
    assert sea_tuning.propose_settings(runs[:2], {})["proposals"] == {}
    # A run stopped by a limit never lowers that limit below the current value.
    stopped = runs + [
        {
            "status": "failed",
            "duration_s": 3600.0,
            "cost": 0.2,
            "result": "Error: the tunedemo agent task did not finish within 3600 s",
        }
    ]
    report = sea_tuning.propose_settings(stopped, {"timeout": 7200, "max_budget": 1.0})
    assert report["stopped_by_limit"] == 1
    assert "timeout" not in report["proposals"]  # 7200 stays: 2 x p95 would lower it
    assert report["proposals"]["max_budget"]["proposed"] == 1.5
    # Missing-tool failures flag the tool settings.
    tool_runs = runs + [
        {
            "status": "failed",
            "duration_s": 1,
            "cost": 0.1,
            "result": "Error: tool 'Read' is not available",
        },
        {
            "status": "failed",
            "duration_s": 1,
            "cost": 0.1,
            "result": "Error: unknown tool go_to_url",
        },
    ]
    flags = sea_tuning.propose_settings(tool_runs, {"tool_profile": "bash"})["flags"]
    assert len(flags) == 1 and flags[0].startswith(
        "2 runs failed on a missing tool (tool_profile='bash'"
    )


def test_acceptance_refuses_a_limit_that_cuts_a_successful_run() -> None:
    runs = [
        {"status": "success", "duration_s": 100.0, "cost": 0.4},
        {"status": "success", "duration_s": 900.0, "cost": 2.0},
        {"status": "failed", "duration_s": 5000.0, "cost": 9.0},
    ]
    assert sea_tuning.accepts_runs("timeout", 1000, runs) == ""
    assert sea_tuning.accepts_runs("timeout", 600, runs) == (
        "timeout=600 would have cut off 1 successful run(s) of the window "
        "(largest duration_s 900); refused"
    )
    assert sea_tuning.accepts_runs("max_budget", 1.0, runs).startswith(
        "max_budget=1 would have cut off 1"
    )
    assert sea_tuning.accepts_runs("tool_profile", 0, runs) == ""  # not a limit


def test_patch_settings_literal_is_confined_to_the_dict() -> None:
    source = (
        "X = 1\n\ndef settings() -> dict:\n"
        "    return {'kind': 'worker', 'timeout': 60,  # note\n"
        "            'tool_profile': 'bash'}\n\n\ndef other():\n    return {'timeout': 1}\n"
    )
    out = sea_tuning.patch_settings_literal(source, "timeout", 1800)
    assert "'timeout': 1800,  # note" in out and "def other():\n    return {'timeout': 1}" in out
    out = sea_tuning.patch_settings_literal(out, "max_budget", 2.5)
    assert "'tool_profile': 'bash', 'max_budget': 2.5}" in out
    assert _returned_literal(out, "settings") == {
        "kind": "worker",
        "timeout": 1800,
        "tool_profile": "bash",
        "max_budget": 2.5,
    }
    assert sea_tuning.patch_settings_literal("def settings():\n    return {}\n", "timeout", 5) == (
        'def settings():\n    return { "timeout": 5}\n'
    )
    trailing = "def settings():\n    return {\n        'kind': 'worker',\n    }\n"
    out = sea_tuning.patch_settings_literal(trailing, "timeout", 5)
    assert (
        out == "def settings():\n    return {\n        'kind': 'worker',\n"
        "        'timeout': 5,\n    }\n"
    )
    # A trailing comment never swallows the new entry; a missing comma is added.
    commented = (
        "def settings():\n    return {\n        'kind': 'worker'  # worker defaults\n    }\n"
    )
    out = sea_tuning.patch_settings_literal(commented, "timeout", 5)
    assert out == (
        "def settings():\n    return {\n        'kind': 'worker',  # worker defaults\n"
        "        'timeout': 5,\n    }\n"
    )
    assert _returned_literal(out, "settings") == {"kind": "worker", "timeout": 5}
    only_comment = "def settings():\n    return {\n        # nothing yet\n    }\n"
    out = sea_tuning.patch_settings_literal(only_comment, "timeout", 5)
    assert out.endswith('        # nothing yet\n        "timeout": 5,\n    }\n')
    with pytest.raises(ValueError, match="does not return a single dict literal"):
        sea_tuning.patch_settings_literal("def settings():\n    return build()\n", "timeout", 5)


def test_eval_candidates_and_frequent_tasks() -> None:
    finish_yaml = (
        "ran: demo (worker) model=m tools=bash budget=none timeout=none inherited=none\n"
        "  overridden=none\nsuccess: true\nis_continue: false\n"
        "summary: <p>The readme describes the install steps &amp; the test command.</p> "
        "Done at 12:30."
    )
    rows = [
        {"id": "aaaaaaaa-1", "task": "summarize the readme", "result": finish_yaml},
        {"id": "bbbbbbbb-2", "task": "  ", "result": "ignored"},
        {"id": "cccccccc-3", "task": "list tools", "result": "success: true\nsummary: ok"},
        {
            "id": "aaaaaaaa-4",  # shares the first 8 characters with the first id
            "task": "summarize the readme",
            "result": "The readme describes the install steps & the test command.",
        },
    ]
    evals = sea_tuning.eval_candidates(rows)
    assert evals["rollout"] == {"web_tools": False, "is_parallel": False, "use_memory": False}
    assert [t["id"] for t in evals["tasks"]] == [
        "hist-aaaaaaaa-1",
        "hist-cccccccc-3",
        "hist-aaaaaaaa-4",
    ]
    assert [t["split"] for t in evals["tasks"]] == ["select", "train", "train"]  # newest 30 %
    # The ``ran:`` / ``success:`` / ``is_continue:`` envelope is parsed away and the
    # sentence is normalised like ``skillopt.plain_text`` (tags removed, entities decoded).
    expect = "The readme describes the install steps & the test command."
    assert evals["tasks"][0]["expect"] == [expect] and evals["tasks"][2]["expect"] == [expect]
    assert "expect" not in evals["tasks"][1]  # too short a result to pin
    assert sea_tuning.expectations("Saved to /tmp/x. Costs were 12 dollars. Fine.") == []
    assert sea_tuning.result_summary("just text") == "just text"
    assert sea_tuning.result_summary("key: [unclosed") == "key: [unclosed"
    # A single usable row stays in ``train``; two rows split one and one.
    assert [t["split"] for t in sea_tuning.eval_candidates(rows[:1])["tasks"]] == ["train"]
    assert [t["split"] for t in sea_tuning.eval_candidates(rows[2:])["tasks"]] == [
        "select",
        "train",
    ]
    frequent = sea_tuning.frequent_task_templates(
        rows + [{"task": "summarize  the readme", "model": "m2"}],
        min_repeats=3,
    )
    assert frequent == [
        {"task": "summarize the readme", "count": 3, "models": {"unknown": 2, "m2": 1}}
    ]
    assert sea_tuning.frequent_task_templates(rows, min_repeats=3) == []


# --- the tools, end to end ---------------------------------------------------


def test_sea_source_and_patch_sea_code_through_the_gate(checkout: Path) -> None:
    path = checkout / "tunedemo" / "tunedemo_sea.py"
    listing = sea.sea_source("tunedemo", start=1, count=3)
    assert listing.startswith(f"# {path} (") and listing.endswith(
        '    1  """Demo SEA: a worker with a tool."""\n    2  \n'
        '    3  SYSTEM_PROMPT = "You run demo tasks. Report the exit code."'
    )
    assert sea.sea_source("slack").startswith("Error: 'slack' is not an editable SEA")

    # A new tool plus a guardrail hook: the file reloads and exposes both.
    report = sea.patch_sea_code(
        "tunedemo",
        'def add_to_tools() -> list:\n    """Extra tools."""\n    return [count_words]\n',
        'def shout(text: str) -> str:\n    """Return *text* upper-cased."""\n'
        "    return text.upper()\n\n\n"
        'def tool_call_hook():\n    """Refuse rm -rf."""\n    def hook(name, args):\n'
        "        if name == 'Bash' and 'rm -rf' in str(args):\n            return 'refused'\n"
        "        return None\n    return hook\n\n\n"
        'def add_to_tools() -> list:\n    """Extra tools."""\n    return [count_words, shout]\n',
    )
    assert report.startswith(f"Patched {path}:") and "+14 lines" in report
    _layers, cmd, _d = sea_commands.check_sea(path)
    assert sorted(t.__name__ for t in cmd["tools"]) == ["count_words", "shout"]
    assert callable(sea._execute_sea(path)["tool_call_hook"]())

    # Appending with an empty ``old`` adds at the end of the file.
    assert sea.patch_sea_code("tunedemo", "", "\nEXTRA = 1\n").startswith("Patched")
    assert path.read_text(encoding="utf-8").endswith("\nEXTRA = 1\n")

    # Refusals leave the file as it was.
    before = path.read_text(encoding="utf-8")
    assert (
        sea.patch_sea_code("tunedemo", "nope", "x")
        == f"Error: old text occurs 0 times in {path} (must be exactly once)"
    )
    assert sea.patch_sea_code("tunedemo", "EXTRA = 1", "EXTRA = (").startswith(
        "Error: the patched file does not compile"
    )
    assert sea.patch_sea_code("tunedemo", 'return "Runs demo tasks."', "return 123").startswith(
        "Error: the patched SEA no longer loads (",
    )
    assert sea.patch_sea_code("tunedemo", '"kind": "worker"', '"preset": "worker"').startswith(
        "Error: the patched SEA no longer loads (",  # the loader refuses the renamed key
    )
    assert sea.patch_sea_code("tunedemo", "EXTRA = 1", 'EXTRA = "~/.kiss/x"').startswith(
        "Error: sea lint rejects the patched SEA; file restored:\n",
    )
    assert path.read_text(encoding="utf-8") == before
    assert sea.patch_sea_code("slack", "", "x").startswith("Error: 'slack' is not an editable SEA")


def test_tune_patch_settings_and_export_evals_use_the_persisted_runs(checkout: Path) -> None:
    path = checkout / "tunedemo" / "tunedemo_sea.py"
    assert sea.tune_sea_settings("slack").startswith("Error:")
    assert sea.patch_sea_settings("slack", "timeout", "5").startswith("Error:")
    assert sea.export_sea_evals("slack").startswith("Error:")
    assert sea.export_sea_evals("tunedemo").startswith("Error: 0 successful run(s) of tunedemo")

    persist_run(
        "summarize the readme",
        "The readme describes the install steps and the test command.",
        True,
        0.4,
        100,
    )
    persist_run("list tools", "Listed every tool.", True, 0.8, 900)
    persist_run("count words in README", "Counted the words of the file.", True, 0.6, 300)
    persist_run(
        "flaky", "Error: the tunedemo agent task did not finish within 3600 s", False, 2.0, 3600
    )

    report = json.loads(sea.tune_sea_settings("tunedemo"))
    assert (
        report["sea"] == "tunedemo" and report["finished"] == 4 and report["stopped_by_limit"] == 1
    )
    assert report["proposals"]["timeout"]["proposed"] == 7200  # 2 x p95 (3600), no current limit
    assert report["proposals"]["max_budget"]["proposed"] == 3.0

    assert sea.patch_sea_settings("tunedemo", "timeout", "not json").startswith(
        "Error: value must be a JSON literal"
    )
    assert sea.patch_sea_settings("tunedemo", "timeout", "600") == (
        "Error: timeout=600 would have cut off 1 successful run(s) of the window "
        "(largest duration_s 900); refused"
    )
    assert sea.patch_sea_settings("tunedemo", "timeout", "7200") == (
        f"Patched settings()['timeout'] of {path} to 7200"
    )
    assert sea.patch_sea_settings("tunedemo", "max_budget", "3.0").endswith("to 3.0")
    assert sea_commands.sea_settings(path)["timeout"] == 7200
    # A value shadowed by a later ``**spread`` never reaches the run: refused and restored.
    before = path.read_text(encoding="utf-8")
    assert sea.patch_sea_code(
        "tunedemo",
        '"max_budget": 3.0}',
        '"max_budget": 3.0, **{"timeout": 60.0}}',
    ).startswith("Patched")
    assert sea.patch_sea_settings("tunedemo", "timeout", "900").startswith(
        "Error: settings()['timeout'] = 900 is shadowed (effective value 60.0"
    )
    assert "900" not in path.read_text(encoding="utf-8")
    assert sea.patch_sea_code(
        "tunedemo",
        ', **{"timeout": 60.0}',
        "",
    ).startswith("Patched")
    assert path.read_text(encoding="utf-8") == before
    assert '"max_budget": 3.0}' in path.read_text(encoding="utf-8")
    # A value the contract refuses is rolled back with the loader's words.
    assert sea.patch_sea_settings("tunedemo", "model", '"no-such-model"').startswith(
        "Error: sea lint rejects",  # unknown-model
    )
    assert sea.patch_sea_settings("tunedemo", "max_budget", '"lots"').startswith(
        "Error: the patched SEA no longer loads (",  # the contract's type check
    )
    # A redundant value (the kind's default) fails lint and is rolled back.
    assert sea.patch_sea_settings("tunedemo", "use_memory", "false").startswith(
        "Error: sea lint rejects"
    )
    assert sea_commands.sea_settings(path)["timeout"] == 7200

    out = sea.export_sea_evals("tunedemo")
    target = checkout / "tunedemo" / "evals" / "tunedemo_sea_evals_candidates.json"
    assert (
        out == f"Wrote 3 eval tasks (2 with expectations) to {target}"
    )  # "Listed every tool." is too short to pin
    evals = json.loads(target.read_text(encoding="utf-8"))
    # Newest start first: the runs were persisted with start = now - duration.
    assert [t["prompt"] for t in evals["tasks"]] == [
        "summarize the readme",
        "count words in README",
        "list tools",
    ]
    assert [t["split"] for t in evals["tasks"]] == ["select", "train", "train"]
    assert evals["tasks"][0]["expect"] == [
        "The readme describes the install steps and the test command."
    ]
    assert all(t["source_task_id"] for t in evals["tasks"])

    frequent = json.loads(sea.frequent_tasks(min_repeats=1))
    assert {"task": "list tools", "count": 1, "models": {"model-a": 1}} in frequent["tasks"]
