# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (task 4aa14638), proposal 6: fan-out results come back as digests.

A 42-child ``run_parallel`` wave and 22 full-suite ``run_commands_parallel``
runs filled 365 k tokens of one context with text the agent read once.
Over the conversation's ceiling (``DEFAULT_CONFIG.tool_output_max_chars``)
the whole report now goes to ``tool-out/`` of the caller's scratch dir
(never to its work dir) and the tool returns a digest naming the file by
its absolute path; and a sub-task's result carries its final session's
summary, not every session's handoff notes.
"""

from __future__ import annotations

import functools
import json
import uuid
from pathlib import Path

import pytest
import yaml

from kiss.agents.sorcar import agent_dispatch, cron_agent
from kiss.agents.sorcar.agent_dispatch import (
    digest_agent_results,
    result_summary,
    run_agents_parallel,
)
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.daemon_client import TaskResult
from kiss.agents.sorcar.exec_bridge import ExecBridge
from kiss.agents.sorcar.relentless_agent import _build_exhaustion_summary
from kiss.agents.sorcar.useful_tools import SPILL_DIR, UsefulTools
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_tmp import kiss_tmp_dir
from kiss.tests.agents.third_party_agents.recording_daemon import serve_recording_daemon
from kiss.tests.conftest import posix_only

pytestmark = posix_only("the host Bash tool runs bash")


def test_a_parallel_report_over_the_ceiling_is_spilled_and_digested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "tool_output_max_chars", 2000)
    work, scratch = tmp_path / "work", tmp_path / "scratch"
    work.mkdir()
    tools = UsefulTools(work_dir=str(work), scratch_dir=str(scratch))
    commands = [
        "seq 1 400",  # 1,492 characters of output: over the ceiling with the next
        "seq 1 300",
        "echo 'FAILED test_x - boom'; echo 'E  assert 1 == 2'; exit 3",
        "sleep 5",  # times out (1 s)
    ]
    report = tools.run_commands_parallel(json.dumps(commands), timeout_seconds=1,
                                         max_output_chars=4000)
    [spilled] = list((scratch / SPILL_DIR).iterdir())
    whole = spilled.read_text()
    assert spilled.name.endswith("-parallel.txt") and len(whole) > 2000
    assert whole.startswith("4 commands: 2 succeeded, 1 failed, 1 timed out.\n")
    assert "\n### [1/4] exit 0 in " in whole and "\n400" in whole  # every section, in full
    # the digest: the tally, the file (by its absolute path), one line per command, the
    # tails of the failed ones
    lines = report.splitlines()
    assert lines[0].startswith(
        "4 commands: 2 succeeded, 1 failed, 1 timed out. The whole report "
        f"(4 sections) is in {spilled.as_posix()}; this is its digest."
    )
    assert lines[1].startswith("[1/4] exit 0 in ") and lines[1].endswith(": seq 1 400")
    assert lines[2].startswith("[2/4] exit 0 in ") and lines[2].endswith(": seq 1 300")
    assert lines[3].startswith("[3/4] exit 3 in ")
    assert lines[4].startswith("[4/4] TIMED OUT after ") and lines[4].endswith(": sleep 5")
    assert "### [3/4] exit 3 in " in report and "FAILED test_x - boom\nE  assert 1 == 2" in report
    assert "### [4/4] TIMED OUT after " in report and "(no output)" in report
    assert "\n400\n" not in report and len(report) < 2000  # the successes' output stays in the file
    # under the ceiling the whole report comes back, and nothing is written
    short = tools.run_commands_parallel(json.dumps(["seq 1 3", "echo x"]))
    assert short.startswith("2 commands: 2 succeeded, 0 failed, 0 timed out.\n")
    assert "\n### [1/2] exit 0 in " in short and "1\n2\n3" in short
    assert len(list((scratch / SPILL_DIR).iterdir())) == 1
    # the work dir got nothing: no ``tmp`` folder, no report
    assert list(work.iterdir()) == []


def test_a_parallel_report_with_no_spill_directory_comes_back_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scratch dir whose ``tool-out/`` cannot be made has no file to name: nothing is cut."""
    monkeypatch.setattr(DEFAULT_CONFIG, "tool_output_max_chars", 500)
    work, scratch = tmp_path / "work", tmp_path / "scratch"
    work.mkdir()
    scratch.mkdir()
    (scratch / SPILL_DIR).write_text("a file where the directory should be\n")
    tools = UsefulTools(work_dir=str(work), scratch_dir=str(scratch))
    report = tools.run_commands_parallel(json.dumps(["seq 1 200", "seq 1 200"]))
    assert report.startswith("2 commands: 2 succeeded") and report.count("\n200") == 2
    assert "digest" not in report
    # the report is not written somewhere else instead, the work dir included
    assert list(work.iterdir()) == [] and [p.name for p in scratch.iterdir()] == [SPILL_DIR]


def test_an_agents_report_over_the_ceiling_goes_to_the_callers_scratch_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_parallel``: a host caller gets the digest of a file in its scratch dir.

    A caller whose tools run in a sandbox could not read a file of this
    machine, and a scratch dir whose ``tool-out/`` cannot be made has no
    file to name: both get the report whole.  A call with no calling
    agent has the agents' temp directory itself as scratch dir.  A real
    daemon stand-in (:class:`RecordingDaemon`) answers the two children
    of each call.
    """
    monkeypatch.setattr(DEFAULT_CONFIG, "tool_output_max_chars", 2000)
    caller = ChatSorcarAgent("caller")
    caller._last_task_id = uuid.uuid4().hex
    spill_dir = kiss_tmp_dir() / caller.last_task_id / SPILL_DIR
    spill_dir.parent.mkdir()
    spill_dir.write_text("a file where the directory should be\n")
    summary = "<p>" + "x" * 1500 + "</p>"
    run = functools.partial(run_agents_parallel, str(tmp_path), ["a", "b"], timeout="30")
    with serve_recording_daemon(cron_agent, "_daemon_endpoint_file", text=summary):
        unwritten = run(parent_agent=caller)
        spill_dir.unlink()
        digest = run(parent_agent=caller)
        caller.docker_manager = ExecBridge("exec:http://127.0.0.1:9")
        whole = run(parent_agent=caller)
        standalone = run()
    [spilled] = list(spill_dir.iterdir())  # the other calls wrote no file here
    assert spilled.name.endswith("-agents.txt")
    lines = digest.splitlines()
    assert lines[0] == (
        f"2 sub-tasks; the whole result is in {spilled.as_posix()}, this is its digest:"
    )
    assert lines[1].startswith("[1/2] ok: xxx") and lines[2].startswith("[2/2] ok: xxx")
    assert len(lines) == 3 and len(digest) < 2000
    for report in (spilled.read_text(), whole, unwritten):
        assert len(report) > 2000
        assert [yaml.safe_load(entry)["summary"] for entry in yaml.safe_load(report)] == [
            summary, summary,
        ]
    in_temp_dir = f"{(kiss_tmp_dir() / SPILL_DIR).as_posix()}/"
    assert standalone.startswith(f"2 sub-tasks; the whole result is in {in_temp_dir}")
    assert list(tmp_path.iterdir()) == []  # the callers' work dir got no ``tmp`` folder


def _result(text: str) -> TaskResult:
    return TaskResult(text=text, success=True, cost=0.0, tokens=0, steps=0)


def test_a_sub_tasks_result_is_its_final_sessions_summary() -> None:
    final = "<h3>Final Session</h3>\n<p>Done: a.py fixed, 12 tests pass.</p>"
    merged = _build_exhaustion_summary(["<p>Session one: read the code.</p>",
                                        "<p>Session two: half the fix.</p>"], final)
    assert merged.startswith("<h3>Previous Session 1</h3>\n")
    assert result_summary(_result(merged)) == (
        "(the last of 3 sessions; the earlier sessions' handoff notes are in the task's "
        "stored result)\n<p>Done: a.py fixed, 12 tests pass.</p>")
    # an exhausted run ends with the banner after the last separator
    exhausted = _build_exhaustion_summary(
        ["<p>Session one.</p>"], "Task failed after 2 sub-sessions"
    )
    assert result_summary(_result(exhausted)) == (
        "(the last of 2 sessions; the earlier sessions' handoff notes are in the task's "
        "stored result)\nTask failed after 2 sub-sessions")
    # a one-session result, with or without a separator of its own, is untouched
    assert result_summary(_result(final)) == final
    assert result_summary(_result("<p>a</p>\n\n---\n\n<p>b</p>")) == "<p>a</p>\n\n---\n\n<p>b</p>"
    assert result_summary(TaskResult(text="", success=False, cost=0, tokens=0, steps=0)) == (
        "Task failed")
    # the YAML the caller reads carries the cut summary
    text = agent_dispatch.format_dispatch_result(_result(merged), 60.0)
    assert "Session one" not in text and "12 tests pass" in text


def test_a_run_parallel_result_over_the_ceiling_is_a_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        agent_dispatch.format_dispatch_result(_result("<h3>Done</h3>\n<p>Fixed a.py.</p>"), 60.0),
        agent_dispatch.format_dispatch_result(
            TaskResult(text="<p>Could not: " + "x" * 300 + "</p>", success=False, cost=0.0,
                       tokens=0, steps=0), 60.0),
        "Started the sub-task as job agent-0123abcd; agent_job(job_id) collects its result.",
        # a dispatch that could not run: its error is final, not a job to collect
        agent_dispatch.format_dispatch_result("Error: the sorcar daemon is not running.", 60.0),
    ]
    shown = f"{(kiss_tmp_dir() / SPILL_DIR).as_posix()}/ab12cd-agents.txt"
    lines = digest_agent_results(entries, shown).splitlines()
    assert lines[0] == f"4 sub-tasks; the whole result is in {shown}, this is its digest:"
    assert lines[1] == "[1/4] ok: Done Fixed a.py."
    assert lines[2].startswith("[2/4] FAILED: Could not: xxx") and lines[2].endswith("...")
    assert len(lines[2]) < 220
    assert lines[3].startswith("[3/4] running (job agent-0123abcd): Started the sub-task as job")
    assert lines[4] == "[4/4] FAILED: Error: the sorcar daemon is not running."
