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
import subprocess
import time
from collections.abc import Callable
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
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon


def _returned_literal(source: str, function: str) -> Any:
    """Evaluate the dict literal ``def <function>`` in ``source`` returns (after any ``x |``)."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            ret = node.body[-1]
            assert isinstance(ret, ast.Return) and ret.value is not None
            value = ret.value.right if isinstance(ret.value, ast.BinOp) else ret.value
            return ast.literal_eval(value)
    raise AssertionError(f"no def {function} in source")


DEMO_SEA = '''"""Demo SEA: a worker with a tool."""

from kiss.agents.seas.base.base_sea import BaseSea

SYSTEM_PROMPT = "You run demo tasks. Report the exit code."


class Sea(BaseSea):
    def description(self):
        """Help text."""
        return "Runs demo tasks."

    def settings(self, settings):
        """Run settings."""
        return settings | {"kind": "worker", "tool_profile": "bash"}

    def system_prompt(self, system_prompt):
        """Replace the default prompt."""
        return SYSTEM_PROMPT

    def tools(self, tools):
        """Extra tools."""
        return tools + [count_words]


def count_words(text: str) -> str:
    """Return the number of words in *text*."""
    return str(len(text.split()))


'''


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake KISS checkout whose ``seas/`` folder holds the editable ``demo`` SEA."""
    seas = tmp_path / "src" / "kiss" / "agents" / "seas"
    (seas / "tunedemo").mkdir(parents=True)
    (seas / "tunedemo" / "tunedemo_sea.py").write_text(DEMO_SEA, encoding="utf-8")
    (seas.parents[1] / "SYSTEM.md").write_text("You are a test Sorcar.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    log = sea._change_log()
    if log.is_file():
        log.unlink()  # the session home is shared: start every test with no pending change
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
        """
from kiss.agents.seas.base.base_sea import BaseSea

X = 1

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'kind': 'worker', 'timeout': 60,  # note
                'tool_profile': 'bash'}


def other():
    return {'timeout': 1}
"""
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
    assert sea_tuning.patch_settings_literal("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {}
""", "timeout", 5) == (
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | { "timeout": 5}
"""
    )
    trailing = """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {
            'kind': 'worker',
        }
"""
    out = sea_tuning.patch_settings_literal(trailing, "timeout", 5)
    assert (
        out == """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {
            'kind': 'worker',
            'timeout': 5,
        }
"""
    )
    # A trailing comment never swallows the new entry; a missing comma is added.
    commented = (
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {
            'kind': 'worker'  # worker defaults
        }
"""
    )
    out = sea_tuning.patch_settings_literal(commented, "timeout", 5)
    assert out == (
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {
            'kind': 'worker',  # worker defaults
            'timeout': 5,
        }
"""
    )
    assert _returned_literal(out, "settings") == {"kind": "worker", "timeout": 5}
    only_comment = """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {
            # nothing yet
        }
"""
    out = sea_tuning.patch_settings_literal(only_comment, "timeout", 5)
    assert out.endswith('            # nothing yet\n            "timeout": 5,\n        }\n')
    with pytest.raises(ValueError, match="does not return a single dict literal"):
        sea_tuning.patch_settings_literal("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | build()
""", "timeout", 5)


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
        "    3  from kiss.agents.seas.base.base_sea import BaseSea"
    )
    assert sea.sea_source("slack").startswith("Error: 'slack' is not an editable SEA")

    # A new tool plus a guardrail hook: the file reloads and exposes both.
    report = sea.patch_sea_code(
        "tunedemo",
        '''
    def tools(self, tools):
        """Extra tools."""
        return tools + [count_words]
''',
        '''
    def tool_call_hook(self, name, args):
        """Refuse rm -rf."""
        if name == 'Bash' and 'rm -rf' in str(args):
            return 'refused'
        return 'OK'

    def tools(self, tools):
        """Extra tools."""
        return tools + [count_words, shout]


def shout(text: str) -> str:
    """Return *text* upper-cased."""
    return text.upper()
''',
    )
    assert report.startswith(f"Patched {path}:") and "+11 lines" in report
    seas, _cmd, _d = sea_commands.check_sea(path)
    assert sorted(t.__name__ for t in sea_commands.base_tools(seas, [])) == ["count_words", "shout"]
    patched = seas[-1]
    assert patched.tool_call_hook("Bash", {"command": "rm -rf /"}) == "refused"
    assert patched.tool_call_hook("Bash", {"command": "ls"}) == "OK"

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
        f"Patched settings()['timeout'] of {path} to 7200 (pending: replay a past run, then "
        "settle_sea_settings('tunedemo', <replay_task_id>) keeps or reverts it)"
    )
    # The change is pending: nothing else may change until it is settled,
    # and the tuner proposes nothing but the settlement.
    assert sea.patch_sea_settings("tunedemo", "max_budget", "3.0") == (
        "Error: tunedemo has a pending settings change ('timeout': None -> 7200); settle it "
        "with settle_sea_settings first"
    )
    report = json.loads(sea.tune_sea_settings("tunedemo"))
    assert report["proposals"] == {} and report["pending"]["key"] == "timeout"
    assert report["flags"][-1].startswith("settings()['timeout'] changed None -> 7200 at ")
    # Settled by a successful run of this SEA started after the change (a replay).
    replay = persist_run("replay", "Replayed fine.", True, 0.5, 0)
    assert sea.settle_sea_settings("tunedemo", replay) == (
        f"Kept settings()['timeout'] = 7200 of tunedemo (replay {replay} succeeded)"
    )
    assert sea.settle_sea_settings("tunedemo", replay) == (
        "Error: tunedemo has no pending settings change"
    )
    assert sea.patch_sea_settings("tunedemo", "max_budget", "3.0").startswith(
        f"Patched settings()['max_budget'] of {path} to 3.0 (pending"
    )
    # Settled without a replay: reverted, and the added key is removed again.
    assert sea.settle_sea_settings("tunedemo") == (
        "Reverted settings()['max_budget'] of tunedemo to None (no replay given); record why "
        "in ./tmp/rsi7d/explored-ideas.md"
    )
    assert "max_budget" not in sea_commands.sea_settings(path)
    assert sea.patch_sea_settings("tunedemo", "max_budget", "3.0").startswith("Patched")
    # A failed replay reverts too, with the replay's status as the reason.
    failed = persist_run("replay", "Error: it broke", False, 0.5, 0)
    assert sea.settle_sea_settings("tunedemo", failed).startswith(
        "Reverted settings()['max_budget'] of tunedemo to None (replay status "
    )
    assert sea.patch_sea_settings("tunedemo", "max_budget", "3.0").startswith("Patched")
    assert sea.settle_sea_settings("tunedemo", "no-such-task").startswith(
        "Reverted settings()['max_budget'] of tunedemo to None (Error: no task with id "
    )
    assert sea.patch_sea_settings("tunedemo", "max_budget", "3.0").startswith("Patched")
    # The earlier replay predates this change and does not count; a new one does.
    assert "started before the change" in sea.settle_sea_settings("tunedemo", replay)
    assert sea.patch_sea_settings("tunedemo", "max_budget", "3.0").startswith("Patched")
    time.sleep(0.01)
    fresh = persist_run("replay", "ok", True, 0.1, 0)
    assert sea.settle_sea_settings("tunedemo", fresh).startswith("Kept")
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
        out == f"Wrote 5 eval tasks (2 with expectations) to {target}"
    )  # "Listed every tool.", "Replayed fine." and "ok" are too short to pin
    evals = json.loads(target.read_text(encoding="utf-8"))
    # Newest start first: the runs were persisted with start = now - duration.
    assert [t["prompt"] for t in evals["tasks"]] == [
        "replay",
        "replay",
        "summarize the readme",
        "count words in README",
        "list tools",
    ]
    assert evals["tasks"][2]["expect"] == [
        "The readme describes the install steps and the test command."
    ]
    assert all(t["source_task_id"] for t in evals["tasks"])

    frequent = json.loads(sea.frequent_tasks(min_repeats=1))
    assert {"task": "list tools", "count": 1, "models": {"model-a": 1}} in frequent["tasks"]


# --- the closed loop: change log, revert proposals, narrowing -----------------


def test_change_log_pending_and_compare() -> None:
    """``pending_change`` / ``compare_change`` on literal records."""
    changes = [
        {"key": "timeout", "old": None, "new": 7200, "at_ms": 1000, "at": "t1",
         "replay_task_id": "", "reverted": False},
    ]
    assert sea_tuning.pending_change([]) is None
    assert sea_tuning.pending_change(changes) == changes[0]
    settled = {**changes[0], "replay_task_id": "r1"}
    assert sea_tuning.pending_change([settled]) is None
    reverted = {**changes[0], "old": 7200, "new": None, "at_ms": 2000, "reverted": True}
    assert sea_tuning.pending_change([changes[0], reverted]) is None
    # A revert of one key leaves a later pending change of another key pending.
    other = {**changes[0], "key": "max_budget", "at_ms": 3000}
    assert sea_tuning.pending_change([changes[0], reverted, other]) == other

    def run(started_ms: int, ok: bool, cost: float) -> dict[str, Any]:
        return {
            "status": "success" if ok else "failed",
            "result": "" if ok else "the agent task did not finish within 100s",
            "cost": cost, "duration_s": 10, "started_ms": started_ms,
        }

    before = [run(100, True, 1.0), run(200, True, 1.0), run(300, False, 1.0)]
    after_bad = [run(1100, False, 1.0), run(1200, False, 1.0), run(1300, True, 1.0)]
    compared = sea_tuning.compare_change(before + after_bad, settled)
    assert compared["before"] == 3 and compared["after"] == 3 and compared["worse"]
    assert compared["basis"].startswith("3 runs before / 3 after the change of t1: limit-stop "
                                        "rate 33% -> 67%")
    after_costly = [run(1100, True, 2.0), run(1200, True, 2.0), run(1300, True, 2.0)]
    assert sea_tuning.compare_change(before + after_costly, settled)["worse"]  # p95 cost x2
    after_good = [run(1100, True, 1.0), run(1200, True, 1.0), run(1300, True, 1.0)]
    assert not sea_tuning.compare_change(before + after_good, settled)["worse"]
    assert not sea_tuning.compare_change(before + after_bad[:2], settled)["worse"]  # < MIN_RUNS
    assert sea_tuning.compare_change([], settled)["basis"].endswith("p95 cost 0.00 -> 0.00")

    # ``propose_settings`` turns a settled change that made things worse into a revert
    # proposal, ignores unsettled or reverted records, and blocks on a pending one.
    report = sea_tuning.propose_settings(before + after_bad, {"timeout": 7200}, [settled])
    assert report["proposals"]["timeout"] == {
        "current": 7200, "proposed": None, "basis": "revert: " + compared["basis"],
    }
    assert report["pending"] is None
    report = sea_tuning.propose_settings(before + after_bad, {"timeout": 7200}, [reverted])
    assert "revert" not in json.dumps(report["proposals"])
    report = sea_tuning.propose_settings(before + after_bad, {"timeout": 7200}, changes)
    assert report["proposals"] == {} and report["pending"] == changes[0]
    assert report["flags"][-1].startswith("settings()['timeout'] changed None -> 7200 at t1")


def test_change_log_round_trip(tmp_path: Path) -> None:
    """``record_change`` / ``load_changes`` / ``settle_change`` on a JSONL file."""
    log = sea_tuning.change_log_path(tmp_path)
    assert log == tmp_path / "rsi7d" / "settings_changes.jsonl"
    assert sea_tuning.load_changes(log, "x") == []
    assert sea_tuning.settle_change(log, "x", "r1") is None
    first = sea_tuning.record_change(
        log, {"sea": "x", "key": "timeout", "old": None, "new": 10, "task_id": "t"}
    )
    assert first["replay_task_id"] == "" and first["reverted"] is False and first["at_ms"] > 0
    sea_tuning.record_change(log, {"sea": "y", "key": "timeout", "old": 1, "new": 2, "task_id": ""})
    assert sea_tuning.load_changes(log, "x") == [first]
    settled = sea_tuning.settle_change(log, "x", "r1")
    assert settled == {**first, "replay_task_id": "r1"}
    assert sea_tuning.load_changes(log, "x") == [settled]
    assert sea_tuning.load_changes(log, "y")[0]["new"] == 2  # the other SEA's line is intact
    assert sea_tuning.settle_change(log, "x", "r2") is None


def test_narrower_profile_is_proposed_only_on_strong_evidence() -> None:
    from collections import Counter

    done = [{"status": "success"}] * sea_tuning.NARROW_MIN_RUNS
    shell_only = Counter({"Bash": 30, "Read": 12, "finish": 10, "count_words": 2})
    # ``finish`` and a SEA's own tool are outside every profile and do not count.
    assert sea_tuning.narrower_profile(done, {}, shell_only) == {
        "current": "", "proposed": "shell",
        "basis": f"{len(done)} runs called only Bash, Read; every one is in the 'shell' profile",
    }
    assistant = sea_tuning.narrower_profile(done, {}, Counter({"Bash": 1, "talk": 1}))
    assert assistant is not None and assistant["proposed"] == "assistant"
    review = sea_tuning.narrower_profile(done, {}, Counter({"go_to_url": 1}))
    assert review is not None and review["proposed"] == "review"
    # Already narrowed, too few runs, no governed tool calls, or a tool outside every
    # narrowing profile (``Edit``, ``run_parallel``): nothing is proposed, never widened.
    assert sea_tuning.narrower_profile(done, {"tool_profile": "shell"}, shell_only) is None
    assert sea_tuning.narrower_profile(done[:-1], {}, shell_only) is None
    assert sea_tuning.narrower_profile(done, {}, Counter()) is None
    assert sea_tuning.narrower_profile(done, {}, Counter({"finish": 3})) is None
    assert sea_tuning.narrower_profile(done, {}, None) is None
    assert sea_tuning.narrower_profile(done, {}, Counter({"Bash": 1, "Edit": 1})) is None
    assert sea_tuning.narrower_profile(done, {}, Counter({"run_parallel": 1})) is None
    report = sea_tuning.propose_settings(
        [{"status": "success", "duration_s": 10, "cost": 0.1}] * sea_tuning.NARROW_MIN_RUNS,
        {"timeout": 60, "max_budget": 0.5},
        tools_used=shell_only,
    )
    assert report["proposals"]["tool_profile"]["proposed"] == "shell"


def test_settings_literal_remove(checkout: Path) -> None:
    """``REMOVE`` deletes an entry wherever it stands; an absent key is a no-op."""
    multi = """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {
            "kind": "worker",
            "timeout": 7200,  # why
            "max_budget": 3.0,
        }
"""
    assert sea_tuning.patch_settings_literal(multi, "timeout", sea_tuning.REMOVE) == (
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {
            "kind": "worker",
            "max_budget": 3.0,
        }
"""
    )
    inline = """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"kind": "worker", "timeout": 7200, "max_budget": 3.0}
"""
    assert sea_tuning.patch_settings_literal(inline, "timeout", sea_tuning.REMOVE) == (
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"kind": "worker", "max_budget": 3.0}
"""
    )
    assert sea_tuning.patch_settings_literal(inline, "max_budget", sea_tuning.REMOVE) == (
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"kind": "worker", "timeout": 7200}
"""
    )
    assert sea_tuning.patch_settings_literal(inline, "absent", sea_tuning.REMOVE) == inline


class _EditingDaemon(RecordingDaemon):
    """A daemon stand-in whose "sub-agent" edits *path* when a ``run`` command arrives.

    ``improve_sea_code`` dispatches a Sorcar run through the real
    ``daemon_client.run``; the stand-in records the command and, in
    place of the sub-agent, applies :attr:`edit` to *path* before the
    scripted ``result`` is sent, so the gate sees the file as a
    sub-agent left it.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(text="reworked")
        self.path = path
        self.edit: Callable[[Path], None] = _add_extra_helper

    async def _handle(self, ws: Any) -> None:
        self.edit(self.path)
        await super()._handle(ws)


def _add_extra_helper(path: Path) -> None:
    """Append a harmless ``extra()`` helper: a change the gate accepts."""
    path.write_text(path.read_text() + "\n\ndef extra() -> int:\n    return 1\n")


def _break_settings(path: Path) -> None:
    """Replace the SEA with one whose ``settings`` no longer loads: the gate rejects it."""
    path.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {'timeout': 'soon'}
""")


def _delete_script(path: Path) -> None:
    """Delete the SEA script outright."""
    path.unlink()


def test_improve_and_revert_sea_code_snapshot_and_gate(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``improve_sea_code`` snapshots, dispatches a full-tool Sorcar run and gates the result;
    ``revert_sea_code`` restores the snapshot.

    The dispatch is the real ``agent_dispatch.dispatch_result`` →
    ``daemon_client.run`` against a local daemon stand-in whose
    "sub-agent" edits the file between the snapshot and the gate.
    """
    path = checkout / "tunedemo" / "tunedemo_sea.py"
    assert sea.improve_sea_code("slack", "x", 1.0).startswith("Error:")
    assert sea.revert_sea_code("tunedemo") == (
        f"Error: no snapshot of tunedemo in {checkout.parents[3] / sea.SNAPSHOT_DIR / 'tunedemo'}"
        "; nothing to revert"
    )
    # An unreachable daemon (the endpoint file does not exist) is a
    # reported failure, not an exception: no task id, and the gate still
    # runs on the untouched file.
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(checkout / "no-daemon.json"))
    report = json.loads(sea.improve_sea_code("tunedemo", "add an extra() helper", 2.0, 30.0))
    assert report["task_id"] == "" and "could not run" in report["result"]
    assert "error" not in report and path.read_text() == DEMO_SEA
    assert sea.revert_sea_code("tunedemo").startswith("Restored ")

    # The run goes to the checkout the SEA folder is in (``git rev-parse
    # --show-toplevel``, which looks at parent directories too), so make
    # the fixture a real repository rooted at its top.
    root = checkout.parents[3]
    assert subprocess.run(["git", "init", "-q", str(root)], check=False).returncode == 0
    daemon = _EditingDaemon(path)
    monkeypatch.setenv("KISS_SORCAR_LOCAL", str(daemon.endpoint_file))
    try:
        report = json.loads(sea.improve_sea_code("tunedemo", "add an extra() helper", 2.0, 30.0))
        assert report["task_id"] == "task-recorded-1"
        assert report["result"] == {"success": True, "summary": "reworked", "cost": 0.0, "steps": 0}
        (command,) = daemon.run_commands
        assert "add an extra() helper" in command["prompt"] and str(path) in command["prompt"]
        # A plain Sorcar run (no agent script) in the SEA's git checkout.
        assert command["agentPath"] == "" and command["workDir"] == str(root.resolve())
        assert command["useWorktree"] is False and command["autoCommit"] is False
        assert "error" not in report and "def extra" in path.read_text()
        assert report["other_changes"] == []
        snapshot = Path(report["snapshot"])
        assert "def extra" not in (snapshot / "tunedemo_sea.py").read_text()
        assert sea.revert_sea_code("tunedemo") == f"Restored {path.parent} from {snapshot}"
        assert "def extra" not in path.read_text()

        daemon.edit = _break_settings
        report = json.loads(sea.improve_sea_code("tunedemo", "break it", 2.0, 30.0))
        assert report["error"].startswith("the patched SEA no longer loads (")
        assert report["error"].endswith("; folder restored from the snapshot")
        assert path.read_text() == DEMO_SEA

        daemon.edit = _delete_script
        report = json.loads(sea.improve_sea_code("tunedemo", "delete it", 2.0, 30.0))
        assert report["error"] == (
            "the sub-agent deleted the script; folder restored from the snapshot"
        )
        assert path.read_text() == DEMO_SEA
        assert len(daemon.run_commands) == 3
    finally:
        daemon.close()


# --- review fixes (gpt-6-astra, 2026-10-05) ----------------------------------------


def test_settle_accepts_only_a_later_successful_replay_of_the_same_sea(checkout: Path) -> None:
    path = checkout / "tunedemo" / "tunedemo_sea.py"
    earlier = persist_run("old run", "Fine.", True, 0.1, 10)
    other = persist_run("other sea", "Fine.", True, 0.1, 10, sea="slack_sea")
    time.sleep(0.01)
    assert sea.patch_sea_settings("tunedemo", "timeout", "7200").startswith("Patched")
    # Neither an unrelated SEA's run nor a run from before the change settles it.
    assert sea.settle_sea_settings("tunedemo", other).startswith(
        f"Reverted settings()['timeout'] of tunedemo to None (replay {other} ran slack, not "
        "tunedemo)"
    )
    assert sea.patch_sea_settings("tunedemo", "timeout", "7200").startswith("Patched")
    assert sea.settle_sea_settings("tunedemo", earlier).startswith(
        f"Reverted settings()['timeout'] of tunedemo to None (replay {earlier} started before "
        "the change)"
    )
    assert sea.patch_sea_settings("tunedemo", "timeout", "7200").startswith("Patched")
    later = persist_run("replay", "Fine.", True, 0.1, 0)
    assert sea.settle_sea_settings("tunedemo", later).startswith("Kept")
    assert sea_commands.sea_settings(path)["timeout"] == 7200


def test_revert_removes_a_key_that_was_absent_even_when_the_kind_gave_it_a_value(
    checkout: Path,
) -> None:
    """The record holds the literal's own value, not the kind default, so the revert of
    ``use_web_tools: true`` on a worker removes the key instead of writing the redundant
    ``False`` back (which ``sea lint`` would refuse, leaving the change pending for good)."""
    path = checkout / "tunedemo" / "tunedemo_sea.py"
    assert sea_commands.sea_settings(path)["use_web_tools"] is False  # the worker default
    assert sea.patch_sea_settings("tunedemo", "use_web_tools", "true").startswith("Patched")
    assert sea_commands.sea_settings(path)["use_web_tools"] is True
    assert sea.settle_sea_settings("tunedemo") == (
        "Reverted settings()['use_web_tools'] of tunedemo to None (no replay given); record "
        "why in ./tmp/rsi7d/explored-ideas.md"
    )
    assert path.read_text(encoding="utf-8") == DEMO_SEA
    assert sea_tuning.pending_change(sea_tuning.load_changes(sea._change_log(), "tunedemo")) is None
    # A computed entry cannot be recorded for a revert: refused before anything changes.
    source = path.read_text(encoding="utf-8")
    computed = source.replace('"tool_profile": "bash"', '"tool_profile": str("ba" + "sh")')
    path.write_text(computed, encoding="utf-8")
    assert sea.patch_sea_settings("tunedemo", "tool_profile", '"shell"') == (
        "Error: settings()['tool_profile'] is computed (str('ba' + 'sh')); change it with "
        "patch_sea_code"
    )
    assert path.read_text(encoding="utf-8") == computed


def test_only_the_change_still_in_effect_can_be_proposed_for_revert() -> None:
    def run(started_ms: int, ok: bool) -> dict[str, Any]:
        return {
            "status": "success" if ok else "failed",
            "result": "" if ok else "did not finish within 60s",
            "cost": 1.0, "duration_s": 10, "started_ms": started_ms,
        }

    runs = [run(100, True), run(200, True), run(300, True)]  # before the first change
    runs += [run(1100, False), run(1200, False), run(1300, False)]  # the 60 -> 120 change hurt
    runs += [run(2100, True), run(2200, True), run(2300, True)]  # 120 -> 240 fixed it
    first = {"key": "timeout", "old": 60, "new": 120, "at_ms": 1000, "at": "t1",
             "replay_task_id": "r1", "reverted": False}
    second = {**first, "old": 120, "new": 240, "at_ms": 2000, "at": "t2", "replay_task_id": "r2"}
    assert sea_tuning.latest_settled([first, second]) == [second]
    assert sea_tuning.latest_settled([first, {**second, "replay_task_id": ""}]) == [first]
    report = sea_tuning.propose_settings(runs, {"timeout": 240}, [first, second])
    assert "revert" not in json.dumps(report["proposals"])
    # Had the file still held 120, the superseded record would not be judged either;
    # only a settled change whose value is current can be proposed for revert.
    report = sea_tuning.propose_settings(runs, {"timeout": 120}, [first])
    assert report["proposals"]["timeout"]["basis"].startswith("revert: 3 runs before / 6 after")


def test_remove_handles_parenthesised_values_and_commas_on_the_next_line() -> None:
    assert sea_tuning.patch_settings_literal(
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"kind": "worker", "timeout": ((7200)), "x": 1}
""",
        "timeout", sea_tuning.REMOVE,
    ) == """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"kind": "worker", "x": 1}
"""
    assert sea_tuning.patch_settings_literal(
        """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"kind": "worker", "timeout": 7200
    , "max_budget": 3}
""",
        "timeout", sea_tuning.REMOVE,
    ) == """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"kind": "worker", "max_budget": 3}
"""
    assert sea_tuning.literal_value("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"a": (1)}
""", "a") == 1
    assert sea_tuning.literal_value("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def settings(self, settings):
        return settings | build()
""", "a") is None


def test_own_task_id_reads_the_persisted_id() -> None:
    """The change log names the rsi7d task by the id ``ChatSorcarAgent`` persists
    (``last_task_id``), found through ``current_agent`` on the task thread."""
    import threading

    from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
    from kiss.server import agent_state

    assert sea._own_task_id() == ""
    agent = WorktreeSorcarAgent("rsi7d-tuning-test")
    agent._last_task_id = "persisted-rsi7d-task"
    state = agent_state.AgentState(
        "rsi7d-tuning-test", agent=agent, task_thread=threading.current_thread(),
        is_task_active=True,
    )
    agent_state.register(state)
    try:
        assert agent.last_task_id == "persisted-rsi7d-task"
        assert sea._own_task_id() == "persisted-rsi7d-task"
    finally:
        agent_state.unregister(state.task_id, state)
