# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Tests of the evidence gate (:mod:`kiss.agents.sorcar.evidence_gate`).

The gate refuses, once per task, a ``finish(success=True)`` whose summary
hedges or that follows a file change no shell command checked; the Sorcar
agent installs it in its tool-call guard (:meth:`SorcarAgent._guard_tool_call`)
for every run that has the edit tools.
"""

from __future__ import annotations

import tempfile

from kiss.agents.sorcar import evidence_gate as eg
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent


def test_hedges_and_writing_commands_are_recognised() -> None:
    assert eg.hedges_in("All checks pass; see the table.") == []
    assert eg.hedges_in("This is my best guess; the output is NOT VERIFIED.") == [
        "best guess", "not verified"]
    for command in ("echo x > f.txt", "cat a >> log", "tee out.txt", "cp a b", "mv a b",
                    "sed -i s/a/b/ f", "sed --in-place s/a/b/ f", "git apply p.diff",
                    "pip install x", "make -j8", "touch f", "cat <<EOF\nx\nEOF", "mkdir d",
                    "a -> b"):
        assert eg.writes_files({"command": command}), command
    for command in ("pytest -q", "ls -la", "cmd 2>&1 | head", "cmd > /dev/null", "cmd &> /dev/null",
                    "grep -n x f", "git status", "git diff"):
        assert not eg.writes_files({"command": command}), command
    assert eg.writes_files({"commands": '["ls", "echo 1 > f"]'})
    assert not eg.writes_files({"commands": '["ls", "pwd"]'})
    assert eg.writes_files({"commands": "echo 1 > f"})  # not JSON: one command
    assert eg.writes_files({"commands": '{"a": "echo 1 > f"}'})  # not a list: the text itself
    assert not eg.writes_files({"commands": 7})
    assert not eg.writes_files({})


def test_gate_refuses_once_with_the_reason_then_passes() -> None:
    gate = eg.EvidenceGate(track_changes=True)
    done = {"success": True, "summary_in_html": "<p>Done, tests pass.</p>"}
    # nothing changed: a clean claim passes
    assert gate.verdict(done) is None and not gate.used
    # an edit with no check after it
    gate.observe("Edit", {"file_path": "a.py", "old_string": "x", "new_string": "y"})
    assert gate.unchecked_change
    assert gate.verdict({"success": False, "summary_in_html": "gave up"}) is None
    assert gate.verdict({}) is None  # the implicit finish
    verdict = gate.verdict(done)
    assert verdict is not None and verdict.startswith("Not finished: finish(success=True)")
    assert "1. files were changed after the last shell command" in verdict
    assert gate.used
    # the second finish passes whatever it says
    assert gate.verdict(done) is None
    # a check after the change clears it; a writing command re-marks it
    gate = eg.EvidenceGate(track_changes=True)
    gate.observe("Write", {"file_path": "a.py", "content": "x"})
    gate.observe("Bash", {"command": "pytest -q"})
    assert not gate.unchecked_change
    gate.observe("Bash", {"command": "echo fix >> a.py"})
    assert gate.unchecked_change
    gate.observe("run_commands_parallel", {"commands": '["pytest a", "pytest b"]'})
    assert not gate.unchecked_change
    gate.observe("Read", {"file_path": "a.py"})  # neither a change nor a check
    assert not gate.unchecked_change
    assert gate.verdict(done) is None
    # a hedged summary is refused even without a change; both reasons are listed together
    hedged = {"success": "true", "summary_in_html": "It should work but I could not test it."}
    verdict = gate.verdict(hedged)
    assert verdict is not None and '"could not test"' in verdict and '"should work"' in verdict
    assert "2." not in verdict
    gate = eg.EvidenceGate(track_changes=True)
    gate.observe("Edit", {"file_path": "a.py"})
    verdict = gate.verdict(hedged)
    assert verdict is not None and "1. the summary itself says" in verdict
    assert "2. files were changed" in verdict
    # without change tracking (the interactive agent) only the hedge counts
    gate = eg.EvidenceGate()
    gate.observe("Edit", {"file_path": "a.py"})
    assert gate.unchecked_change and gate.verdict(done) is None
    verdict = gate.verdict(hedged)
    assert verdict is not None and "files were changed" not in verdict


def test_sorcar_agent_guard_installs_the_gate_for_runs_with_edit_tools() -> None:
    agent = ChatSorcarAgent("evidence")
    with tempfile.TemporaryDirectory() as work_dir:
        agent.work_dir = work_dir
        done = {"success": True, "summary_in_html": "Done."}
        # a read-only run (a reviewer) is never gated
        agent._run_tool_names = frozenset({"Bash", "Read", "finish"})
        assert agent._guard_tool_call("Bash", {"command": "echo x > f"}) is None
        assert agent._guard_tool_call("finish", done) is None
        assert not agent._evidence_gate.unchecked_change
        # a full run: an unchecked edit passes (the interactive gate does not
        # track changes), a hedged success is refused once
        agent._run_tool_names = frozenset({"Bash", "Read", "Edit", "Write", "finish"})
        write_args = {"file_path": f"{work_dir}/f", "content": "x"}
        assert agent._guard_tool_call("Write", write_args) is None
        assert agent._evidence_gate.unchecked_change
        assert agent._guard_tool_call("finish", done) is None
        hedged = {"success": True, "summary_in_html": "Done, but untested."}
        refusal = agent._guard_tool_call("finish", hedged)
        assert refusal is not None and '"untested"' in refusal
        assert agent._guard_tool_call("finish", hedged) is None
        # a new task starts with a fresh gate
        agent._begin_task(None)
        assert not agent._evidence_gate.used
        assert agent._guard_tool_call("finish", hedged) is not None
