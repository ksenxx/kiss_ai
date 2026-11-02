# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The coding gate: the coding SEA's finish gate on the general Sorcar agent.

Every test runs the real gate against real shells, files and git trees;
the reviewer, where one is wanted, is the scripted local model server, and
a real ``SorcarAgent`` run against that server shows the wiring (the
tool-call guard, the exit-code ledger fed by the ``tool_result_hook``, the
refusals the model reads, the review's spend charged to the run).  The
decisions model is optional by design: without it the open-reading check
and the findings check are silent, which is what the tests rely on.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar import coding_gate as cg
from kiss.agents.sorcar import sorcar_agent as sorcar_agent_module
from kiss.agents.sorcar.agent_dispatch import digest_agent_results
from kiss.agents.sorcar.coding_gate import CodingGate, run_on_host
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import UsefulTools, digest_parallel_report
from kiss.core import user_config
from kiss.core.event_sink import bind_printer
from kiss.core.tool_interrupt import USER_INTERRUPTED_MESSAGE
from kiss.core.tool_verdict import ALLOW, Verdict, refuse
from kiss.server.json_printer import JsonPrinter
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    text_body,
    tool_call_body,
    tool_calls_body,
)

TASK = "Write hello.py that prints hello."
#: A summary that reports the outcome and its check: the decisions model, live
#: in the test environment, scores it far below the open-reading threshold
#: (a bare "Done." scores about 0.7 and would be refused once).
DONE = "<p>hello.py written and prints hello; checked by bash tmp/verify.sh (exit 0).</p>"


def _shell(work_dir: Path, command: str, timeout: int) -> tuple[int | None, str]:
    """The gate's shell for the tests: a bash subprocess in *work_dir*."""
    return run_on_host(command, str(work_dir), timeout)


def _gate(work_dir: Path, verifier: str | None = None, **kwargs: Any) -> CodingGate:
    """A gate on *work_dir* with a host shell; *verifier* is the script's body, if any."""
    path = work_dir / "tmp" / "verify.sh"
    if verifier is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(verifier)
    return CodingGate(functools.partial(_shell, work_dir), str(work_dir), "./tmp/", **kwargs)


def _host_result(command: str, work_dir: Path) -> str:
    """What the host ``Bash`` tool returns for *command*: the output, after an exit-code prefix."""
    code, output = run_on_host(command, str(work_dir), 30)
    return output if code == 0 else f"Error (exit code {code}):\n{output}"


def _turn(gate: CodingGate, name: str, args: dict, result: str, refused: bool = False) -> None:
    """One tool call and its result, as the guard and the result hook see them."""
    gate.observe(name, args)
    gate.result(name, args, result, refused)


def test_readers_pair_commands_with_exit_codes_in_both_tool_formats(tmp_path: Path) -> None:
    jobs: dict[str, str] = {}
    bash = {"command": "pytest -q"}
    # the host tool: a prefix on failure, nothing on success, its own timeout wording
    assert cg.result_exit_codes("Bash", bash, "3 passed", jobs) == [("pytest -q", "0")]
    assert cg.result_exit_codes("Bash", bash, "Error (exit code 1):\nF", jobs) == [
        ("pytest -q", "1")]
    assert cg.result_exit_codes("Bash", bash, "Error: Command execution timeout after 5s. "
                                "Output before the timeout:\nx", jobs) == [("pytest -q", "timeout")]
    # the container tool: a marker after the output, its own timeout wording
    assert cg.result_exit_codes("Bash", bash, "F\n[exit code: 2]", jobs) == [("pytest -q", "2")]
    assert cg.result_exit_codes("Bash", bash, "Error: command timed out after 5s", jobs) == [
        ("pytest -q", "timeout")]
    # a guard's refusal: the command never ran, so it is no evidence
    assert cg.result_exit_codes("Bash", bash, "REFUSED: the Bash call did NOT run", jobs) == [
        ("pytest -q", "refused")]
    # the tool's own error (a rejected argument) and the framework's (the tool
    # raised before running: a missing argument): the command never ran either
    assert cg.result_exit_codes("Bash", bash, "Error: machine= applies to background=True only.",
                                jobs) == [("pytest -q", "error")]
    assert cg.result_exit_codes("Bash", bash, "Failed to call Bash with {'command': 'pytest -q'}:"
                                " UsefulTools.Bash() missing 1 required positional argument",
                                jobs) == [("pytest -q", "error")]
    assert cg.result_exit_codes("Bash", bash, "Error (exit code 2):\nE", jobs) == [
        ("pytest -q", "2")]
    # a call the user stopped did not run to completion: not exit 0 (one tool or several)
    assert cg.result_exit_codes("Bash", bash, USER_INTERRUPTED_MESSAGE, jobs) == [
        ("pytest -q", "interrupted")]
    assert cg.result_exit_codes("run_commands_parallel", {"commands": '["a", "b"]'},
                                USER_INTERRUPTED_MESSAGE, jobs) == [
        ("a", "interrupted"), ("b", "interrupted")]
    # a background job, then its exit through bash_job (host wording), only from the status line
    started = "Started background job j1 (pid 7).\nlog: /tmp/j1.log"
    assert cg.result_exit_codes("Bash", {"command": "make"}, started, jobs) == [
        ("make", "background")]
    running = "Job j1: running (pid 7, 2s)\n$ make\n--- last 5 log lines ---\nexited with code 0"
    assert cg.result_exit_codes("bash_job", {"job_id": "j1"}, running, jobs) == []
    assert cg.result_exit_codes("bash_job", {"job_id": "j1"}, "Job j1: exited with code 3\n$ make",
                                jobs) == [("make", "3")]
    # run_commands_parallel: one section per command
    report = ("2 commands: 1 succeeded, 1 failed, 0 timed out.\n\n### [1/2] exit 0 in 0.1s\n$ a\n"
              "ok\n\n### [2/2] TIMED OUT after 9.0s\n$ b\n")
    assert cg.result_exit_codes("run_commands_parallel", {"commands": '["a", "b"]'}, report,
                                jobs) == [("a", "0"), ("b", "timeout")]
    # the digest of a report too long for the conversation: every command's
    # exit is on its status line (the sections are only the failed ones')
    digest = digest_parallel_report(["a", "b", "c"], [(0, "ok", 0.1), (None, "", 9.0),
                                                      (2, "E", 0.2)], "tmp/tool-out/x.txt")
    assert cg.result_exit_codes("run_commands_parallel", {"commands": '["a", "b", "c"]'},
                                digest, jobs) == [("a", "0"), ("b", "timeout"), ("c", "2")]
    # a status line in a command's own output (a nested parallel run) does
    # not override the command's exit, in the digest or in the full report
    outer = "echo '[1/1] exit 0 in 0.1s: inner'; exit 1"
    report = UsefulTools(work_dir=str(tmp_path)).run_commands_parallel(f'[{outer!r}]')
    inner_status: list[tuple[int | None, str, float]] = [(1, "[1/1] exit 0 in 0.1s: inner\n", 0.2)]
    for text in (report, digest_parallel_report([outer], inner_status, "tmp/tool-out/y.txt")):
        assert cg.result_exit_codes("run_commands_parallel", {"commands": f'[{outer!r}]'}, text,
                                    jobs) == [(outer, "1")]
    # the three message shapes carry tool results
    assert cg.tool_result_texts({"type": "function_call_output", "output": "x"}) == ["x"]
    assert cg.tool_result_texts({"role": "tool", "content": "y"}) == ["y"]
    anthropic = {"role": "user", "content": [
        {"type": "tool_result", "content": "a"},
        {"type": "tool_result", "content": [{"type": "text", "text": "b"}]},
        # a Read result with an image attached is one result, not two
        {"type": "tool_result", "content": [{"type": "text", "text": "Image attached"},
                                            {"type": "image", "source": {}}]}]}
    assert cg.tool_result_texts(anthropic) == ["a", "b", "Image attached"]
    assert cg.tool_result_texts({"role": "user", "content": "prompt"}) == []
    # a shell command the tool made as an Edit through the agent's tool path:
    # its exit is the follow-up's (the rest of the command), read through a
    # follow-up that was redirected again (a second file written)
    redirected = ('Applied as Edit(file_path="a.py", ...), whose result follows:\n'
                  "Successfully replaced 1 occurrence(s)\n"
                  'Then made Bash(command="pytest -q", description="rest"), whose result '
                  "follows:\nError (exit code 1):\nF")
    assert cg.bash_exit_code(redirected) == "1"
    nested = ('Applied as Edit(file_path="a.py", ...), whose result follows:\n'
              "Successfully replaced 1 occurrence(s)\n"
              'Then made Bash(command="sed -i s/x/y/ b.py && pytest -q", description="rest"), '
              "whose result follows:\n" + redirected.replace("a.py", "b.py"))
    assert cg.bash_exit_code(nested) == "1"
    assert cg.bash_exit_code(nested.replace("Error (exit code 1):\nF", "3 passed")) == "0"
    # the edit alone (no rest to run), the edit that failed, and the rest not made
    edit_only = ('Appended through Write(file_path="a.py", ...), whose result follows:\n'
                 "Successfully wrote 12 characters")
    assert cg.bash_exit_code(edit_only) == "0"
    assert cg.bash_exit_code(edit_only.replace("Successfully wrote 12 characters",
                                               "Error: file_path must be absolute")) == "error"
    not_made = ('Applied as Edit(file_path="a.py", ...), whose result follows:\n'
                "REFUSED: the Edit call did NOT run\n"
                'Bash(command="pytest -q") was not made: the Edit did not go through.')
    assert cg.bash_exit_code(not_made) == "error"
    hook_refused = ('Applied as Edit(file_path="a.py", ...), whose result follows:\n'
                    "Not now: the caller's hook said no\n"
                    'Bash(command="pytest -q") was not made: the Edit did not go through.')
    assert cg.bash_exit_code(hook_refused) == "error"
    # the outcome of a sub-task: one flag per YAML result, none for a notice or an error
    done = "ran: sorcar model=m tools=review\nsuccess: true\nsummary: fine\n"
    failed = "ran: sorcar model=m\nsuccess: false\nsummary: 'Task failed with error: x'\n"
    assert cg.dispatch_successes(done) == [True] and cg.dispatch_successes(failed) == [False]
    assert cg.dispatch_successes(yaml.safe_dump([done, failed, "Error: no model"])) == [
        True, False]
    assert cg.dispatch_successes("Started the sorcar agent task as job agent-1a2b; its tab"
                                 " is open.") == []
    assert cg.dispatch_successes("Error: unknown agent job 'agent-9'") == []
    assert cg.dispatch_successes("Job agent-1a2b (sorcar agent task) is still running.") == []
    assert cg.dispatch_successes("[unterminated: {") == []
    # the digest of a run_parallel result too long for the conversation
    notice = "Started the sub-task as job agent-0123abcd; agent_job(job_id) collects its result."
    digest = digest_agent_results([done, failed, notice], "tmp/tool-out/x-agents.txt")
    assert cg.dispatch_successes(digest) == [True, False]
    # code files arm the gate, prose does not
    assert cg.is_code_file("src/a.py") and cg.is_code_file("Makefile")
    assert cg.is_code_file("tmp/x/verify.sh") and cg.is_code_file("pyproject.toml")
    assert not cg.is_code_file("reports/a.html") and not cg.is_code_file("notes.md")
    assert not cg.is_code_file(None)
    assert cg.names_code_file({"command": "cp source.py changed.py"})
    assert cg.names_code_file({"command": "sed -i 's/x=0/x=1/' a.py && pytest -q"})
    assert cg.names_code_file({"commands": '["echo x > notes.md", "python gen.py > out.py"]'})
    assert cg.names_code_file({"command": "echo 'oops > a.py"})  # unbalanced quote: plain split
    assert not cg.names_code_file({"command": "echo 'all done' > notes.md"})
    assert not cg.names_code_file({"command": "ls -la"})
    # a review dispatch is one by profile or by the wording of every task
    assert cg.dispatches_review({"task": "do it", "tool_profile": "review"})
    assert cg.dispatches_review({"tasks": '["Review the diff of foo.py for bugs"]'})
    assert not cg.dispatches_review({"task": "Implement the parser"})
    assert not cg.dispatches_review({"tasks": '["Review the diff of foo.py for bugs", '
                                              '"Implement the parser"]'})
    assert not cg.dispatches_review({"tasks": "[]"})


def test_a_coding_run_is_gated_on_its_verifier_and_its_evidence(tmp_path: Path) -> None:
    gate = _gate(tmp_path, verifier="#!/bin/bash\ntest -f hello.py && grep -q hello hello.py\n")
    # a run that only read files is not gated, hedged or not: only the hedge check
    gate.observe("Read", {"file_path": "x"})
    done = {"success": True, "summary_in_html": DONE}
    # ... but the verifier's presence makes it a coding run
    verdict = gate.finish_verdict(dict(done), TASK)
    assert verdict is not None and "round 1 of 3" in verdict
    assert "exited 1" in verdict and "./tmp/verify.sh" in verdict
    assert gate.armed and gate.verifier_refusals == 1
    # the edit, then the check the agent ran: its exit code is read from the result
    _turn(gate, "Write", {"file_path": str(tmp_path / "hello.py"), "content": "print('bye')\n"},
          "Successfully wrote 13 characters")
    _turn(gate, "Bash", {"command": "bash tmp/verify.sh"},
          _host_result("bash tmp/verify.sh", tmp_path))
    assert gate.exit_codes == {"bash tmp/verify.sh": "1"}
    claimed = dict(done, evidence='["bash tmp/verify.sh", "pytest -q"]')
    verdict = gate.finish_verdict(claimed, TASK)
    assert verdict is not None and "round 2 of 3" in verdict
    assert "last exit code 1" in verdict and "`pytest -q`: never ran" in verdict
    assert "exited 1" in verdict  # the verifier ran again and still fails
    # the fix, the check re-run, the evidence named as run: the finish goes through unchanged
    _turn(gate, "Edit", {"file_path": str(tmp_path / "hello.py"), "old_string": "bye",
                         "new_string": "hello"}, "Successfully replaced")
    (tmp_path / "hello.py").write_text("print('hello')\n")
    _turn(gate, "Bash", {"command": "bash tmp/verify.sh"},
          _host_result("bash tmp/verify.sh", tmp_path))
    assert gate.exit_codes["bash tmp/verify.sh"] == "0"
    passing = dict(done, evidence='["bash tmp/verify.sh"]')
    assert gate.finish_verdict(passing, TASK) is None
    assert passing["summary_in_html"] == DONE and gate.gate_rounds == 2
    assert gate.last_verifier_run[0] == 0


def test_an_unchecked_edit_and_a_hedge_are_refused_once_on_a_coding_run(tmp_path: Path) -> None:
    gate = _gate(tmp_path)  # no verifier: the edit of a code file arms the gate
    gate.observe("Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"})
    assert gate.armed and gate.finish_verdict({}, TASK) == cg.IMPLICIT_FINISH_VERDICT
    assert gate.finish_verdict({}, TASK) is None  # the veto is spent
    hedged = {"success": True, "summary_in_html": DONE.replace("checked by", "untested;")}
    verdict = gate.finish_verdict(hedged, TASK)
    assert verdict is not None and '"untested"' in verdict and "files were changed" in verdict
    # the second finish passes: the evidence gate refuses once, and no stage failed
    assert gate.finish_verdict(hedged, TASK) is None
    # a prose-only run is never armed: an unchecked write and a plain summary pass
    prose = _gate(tmp_path)
    prose.observe("Write", {"file_path": str(tmp_path / "notes.md"), "content": "x"})
    assert not prose.armed
    assert prose.finish_verdict({}, TASK) is None
    assert prose.finish_verdict({"success": True, "summary_in_html": DONE}, TASK) is None
    assert prose.finish_verdict(dict(hedged), TASK) is not None  # the hedge check stays


def test_the_round_cap_delivers_with_caveats_and_the_verifier_cap_is_a_caveat(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path, verifier="echo 'hello.py missing'; exit 7\n")
    args: dict[str, Any]
    for round_number in (1, 2, 3):
        args = {"success": True, "summary_in_html": DONE}
        verdict = gate.finish_verdict(args, TASK)
        assert verdict is not None and f"round {round_number} of 3" in verdict
        assert "exited 7" in verdict and "hello.py missing" in verdict
        assert args["summary_in_html"] == DONE  # a refusal edits nothing
    assert gate.verifier_refusals == 3 and gate.gate_rounds == 3
    # the fourth finish goes through: the verifier does not run again, the
    # caveat says so, and the summary carries it
    args = {"success": True, "summary_in_html": DONE}
    assert gate.finish_verdict(args, TASK) is None
    assert args["summary_in_html"].startswith(DONE + "\n<h4>Delivered after 3 refused")
    assert "refused 3 finishes and was not run again" in args["summary_in_html"]
    assert gate.delivered_findings == [cg.UNVERIFIED_CAVEAT.format(
        verifier="./tmp/verify.sh", refusals=3)]
    # a verifier that hangs is a timeout verdict, not a hang of the gate
    slow = _gate(tmp_path, verifier="sleep 30\n", verifier_timeout=1)
    verdict = slow.finish_verdict({"success": True, "summary_in_html": DONE}, TASK)
    assert verdict is not None and "did not finish within 1 seconds" in verdict
    assert slow.last_verifier_run[0] is None


def test_a_giveup_whose_verifier_passes_is_questioned_once(tmp_path: Path) -> None:
    gate = _gate(tmp_path, verifier="exit 0\n")
    # a pause for a fresh context is not a give-up
    paused = {"success": False, "is_continue": True, "summary_in_html": "<p>So far.</p>"}
    assert gate.finish_verdict(paused, TASK) is None
    assert paused["summary_in_html"] == "<p>So far.</p>"
    giveup: dict[str, Any] = {"success": False, "summary_in_html": "<p>Could not do it.</p>"}
    verdict = gate.finish_verdict(giveup, TASK)
    assert verdict == cg.GIVEUP_RECONSIDER_VERDICT.format(verifier="./tmp/verify.sh")
    assert giveup["summary_in_html"].endswith("verifier ./tmp/verify.sh: pass.</p>")
    # the second give-up ends the run, with the check's outcome next to the verdict
    again: dict[str, Any] = {"success": False, "summary_in_html": "<p>Still no.</p>"}
    assert gate.finish_verdict(again, TASK) is None
    assert "verifier ./tmp/verify.sh: pass" in again["summary_in_html"]
    # a failing verifier questions nothing, but is reported
    failing = _gate(tmp_path, verifier="exit 2\n")
    giveup = {"success": False, "summary_in_html": "<p>No.</p>"}
    assert failing.finish_verdict(giveup, TASK) is None
    assert giveup["summary_in_html"].endswith("verifier ./tmp/verify.sh: fail (exit 2).</p>")
    # without a verifier (or any code change) a give-up is not a coding run's: untouched
    (tmp_path / "other").mkdir()
    bare = _gate(tmp_path / "other")
    giveup = {"success": False, "summary_in_html": "<p>No.</p>"}
    assert bare.finish_verdict(giveup, TASK) is None and giveup["summary_in_html"] == "<p>No.</p>"


def _record_charge(
    charged: list[tuple[float, int, int]], budget: float, tokens: int, steps: int,
    key: str = "",
) -> None:
    """The gate's ``charge`` callback for the tests: records what a review cost."""
    charged.append((budget, tokens, steps))


def _git(work_dir: Path, *args: str) -> str:
    """Run a git command in *work_dir* (user identity set) and return its output."""
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@x", HOME=str(work_dir))
    return subprocess.run(["git", *args], cwd=work_dir, env=env, check=True, text=True,
                          capture_output=True).stdout


def test_the_review_sees_the_diff_and_its_findings_must_be_answered(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "hello.py").write_text("print('bye')\n")
    (tmp_path / ".gitignore").write_text("tmp/\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "start")
    (tmp_path / "stale.py").write_text("untracked before the run\n")
    charged: list[tuple[float, int, int]] = []
    script = [
        text_body("Looks wrong.\nVERDICT: NOT DONE\n1. hello.py: prints bye, not hello."),
        text_body("VERDICT: OK"),
    ]
    with serve(script) as (url, requests):
        gate = _gate(tmp_path, verifier="grep -q . hello.py\n", reviewer=MODEL,
                     charge=functools.partial(_record_charge, charged),
                     model_config={"base_url": url, "api_key": "local"})
        gate.observe("Edit", {"file_path": str(tmp_path / "hello.py"), "old_string": "a",
                              "new_string": "b"})
        (tmp_path / "hello.py").write_text("print('bye bye')\n")
        (tmp_path / "new.py").write_text("x = 1\n")
        done = {"success": True, "summary_in_html": DONE, "evidence": "[]"}
        verdict = gate.finish_verdict(dict(done), TASK)
        assert verdict is not None and "1. hello.py: prints bye, not hello." in verdict
        assert gate.open_findings == ["1. hello.py: prints bye, not hello."]
        assert gate.reviews == 1 and len(requests) == 1 and charged and charged[0][1] > 0
        assert charged[0][2] == 1, "the review's one model call is charged as a step"
        prompt = requests[0]["messages"][-1]["content"]
        assert TASK in prompt and "grep -q . hello.py" in prompt and "(none named)" in prompt
        assert "### hello.py\n" in prompt and "-print('bye')" in prompt
        assert "+print('bye bye')" in prompt
        assert "### new.py (new file)\nx = 1" in prompt and "stale.py" not in prompt
        # nothing changed: the finish is judged on the findings without a model call
        # (without the decisions model every finding counts as answered)
        unchanged = gate.finish_verdict(dict(done), TASK)
        assert len(requests) == 1 and (unchanged is None or "not answered" in unchanged)
        # a change after the review: the diff is reviewed again, with the previous findings
        gate.observe("Edit", {"file_path": str(tmp_path / "hello.py"), "old_string": "bye bye",
                              "new_string": "hello"})
        (tmp_path / "hello.py").write_text("print('hello')\n")
        assert gate.finish_verdict(dict(done), TASK) is None
        assert len(requests) == 2 and gate.reviews == 2 and gate.open_findings == []
        assert "Findings of your previous review" in requests[1]["messages"][-1]["content"]
        assert "since your previous review" in requests[1]["messages"][-1]["content"]
    # the agent's own read-only reviewer, dispatched after the last change, replaces the
    # gate's once a sub-task of it succeeded: a dispatch that failed, a sub-task that
    # failed, or one a change followed, does not
    with serve([text_body("VERDICT: NOT DONE\n1. never asked")]) as (url, requests):
        own = _gate(tmp_path, reviewer=MODEL, model_config={"base_url": url, "api_key": "local"})
        edit = {"file_path": str(tmp_path / "hello.py"), "old_string": "a", "new_string": "b"}
        own.observe("Edit", edit)
        _turn(own, "Bash", {"command": "bash tmp/verify.sh"}, "")
        review = {"task": "Verify the listed changes: hello.py", "tool_profile": "review"}
        _turn(own, "run_agent", review, "Error: the reviewer agent task could not run: refused")
        assert not own.reviewed_by_agent and own.review_wanted()
        _turn(own, "run_agent", review,
              "ran: sorcar tools=review\nsuccess: false\nsummary: 'Task failed with error: x'")
        assert not own.reviewed_by_agent and own.review_wanted()
        _turn(own, "run_agent", review, "ran: sorcar tools=review\nsuccess: true\nsummary: ok")
        _turn(own, "Edit", edit, "Successfully replaced")
        assert not own.reviewed_by_agent and own.review_wanted()
        _turn(own, "Bash", {"command": "bash tmp/verify.sh"}, "")
        # a reviewer left running counts once agent_job reports it finished and successful
        _turn(own, "run_agent", {**review, "wait": "false"},
              "Started the sorcar agent task as job agent-1a2b; its tab is open.")
        assert not own.reviewed_by_agent and own.jobs == {"agent-1a2b": (True, own.changes)}
        _turn(own, "agent_job", {"job_id": "agent-1a2b", "action": "tail"},
              "Job agent-1a2b (sorcar agent task) is still running.")
        _turn(own, "agent_job", {"job_id": "agent-9", "action": "wait"},
              "ran: sorcar tools=review\nsuccess: true\nsummary: ok")  # not this run's job
        assert not own.reviewed_by_agent
        _turn(own, "agent_job", {"job_id": "agent-1a2b", "action": "wait"},
              "ran: sorcar tools=review\nsuccess: true\nsummary: ok")
        assert own.reviewed_by_agent and not own.review_wanted()
        assert own.finish_verdict(dict(done), TASK) is None and requests == []
        # ... until the next change
        own.observe("Bash", {"command": "cp hello.py hello2.py"})
        assert not own.reviewed_by_agent and own.review_wanted()
        # a fan-out of reviewers counts when one succeeded; a sub-task that is not a
        # review (it may change files) is a change instead, unchecked until a shell runs
        _turn(own, "run_parallel", {"tasks": '["Review hello.py", "Review hello2.py"]'},
              yaml.safe_dump(["Error: no budget", "ran: sorcar\nsuccess: true\nsummary: ok\n"]))
        assert own.reviewed_by_agent
        _turn(own, "run_agent", {"task": "Implement hello3.py so it prints hello"},
              "ran: sorcar\nsuccess: true\nsummary: written\n")
        assert own.delegated and not own.reviewed_by_agent and own.review_wanted()
        assert own.evidence.unchecked_change
        _turn(own, "Bash", {"command": "bash tmp/verify.sh"}, "")
        assert not own.evidence.unchecked_change
    # a reviewer that cannot answer refuses the finish once; a second failure
    # lets it through (a reviewer that never answers cannot hold the run)
    broken = _gate(tmp_path, reviewer="no-such-model-xyz")
    broken.observe("Edit", {"file_path": str(tmp_path / "hello.py"), "old_string": "a",
                            "new_string": "b"})
    verdict = broken.finish_verdict(dict(done), TASK)
    assert verdict is not None and "did not complete" in verdict
    assert broken.finish_verdict(dict(done), TASK) is None
    assert broken.failed_reviews == 2 and not broken.review_possible()


def test_the_gate_follows_the_work_dir_and_arms_on_shell_written_code(tmp_path: Path) -> None:
    """The verifier is looked for under the work dir of the moment, as the agent moves it
    into a worktree; a writing shell command naming a code file arms the gate as an
    ``Edit`` of it would; naming evidence does not make a prose task a coding run."""
    before, after = tmp_path / "checkout", tmp_path / "worktree"
    before.mkdir()
    after.mkdir()
    gate = _gate(before)
    assert gate.verifier_path == str(before / "tmp" / "verify.sh")
    assert gate.verifier_shown == "./tmp/verify.sh" and not gate.verifier_exists()
    # the agent moves into its worktree: the gate's shell and work dir follow
    gate.work_dir, gate.shell = str(after), functools.partial(_shell, after)
    (after / "tmp").mkdir()
    (after / "tmp" / "verify.sh").write_text("echo 'not yet'; exit 7\n")
    assert gate.verifier_path == str(after / "tmp" / "verify.sh") and gate.verifier_exists()
    verdict = gate.finish_verdict({"success": True, "summary_in_html": DONE}, TASK)
    assert verdict is not None and "exited 7" in verdict and "not yet" in verdict
    # a shell command that writes a code file arms the gate; one that writes prose does not
    prose = _gate(tmp_path)
    prose.observe("Bash", {"command": "echo 'all done' > notes.md"})
    assert not prose.armed and not prose.coding_run()
    (tmp_path / "source.py").write_text("x = 1\n")
    _turn(prose, "Bash", {"command": "cp source.py changed.py"},
          _host_result("cp source.py changed.py", tmp_path))
    assert prose.armed and prose.coding_run() and prose.evidence.unchecked_change
    verdict = prose.finish_verdict({"success": True, "summary_in_html": DONE}, TASK)
    assert verdict is not None and "files were changed after the last shell command" in verdict
    # deleting a code file is a change of the code as well: it arms the gate and
    # stays unchecked until a shell command runs after it
    (tmp_path / "old.py").write_text("y = 2\n")
    gone = _gate(tmp_path)
    _turn(gone, "Bash", {"command": "rm old.py"}, _host_result("rm old.py", tmp_path))
    assert not (tmp_path / "old.py").exists() and gone.exit_codes == {"rm old.py": "0"}
    assert gone.armed and gone.coding_run() and gone.evidence.unchecked_change
    verdict = gone.finish_verdict({"success": True, "summary_in_html": DONE}, TASK)
    assert verdict is not None and "files were changed after the last shell command" in verdict
    # evidence named on a prose-only run: checked against the ledger, but no gate stage runs
    with serve([text_body("VERDICT: NOT DONE\n1. never asked")]) as (url, requests):
        report = _gate(tmp_path, reviewer=MODEL, model_config={"base_url": url, "api_key": "local"})
        _turn(report, "Write", {"file_path": str(tmp_path / "report.md"), "content": "# T\n"},
              "Successfully wrote 4 characters")
        (tmp_path / "report.md").write_text("# T\n")
        _turn(report, "Bash", {"command": "grep -c T report.md"},
              _host_result("grep -c T report.md", tmp_path))
        claim = {"success": True, "summary_in_html": "<p>report.md written; grep -c T report.md "
                 "found the title (exit 0).</p>", "evidence": '["grep -c T report.md"]'}
        assert report.finish_verdict(dict(claim), TASK) is None
        assert not report.armed and requests == [] and report.reviews == 0
        wrong = dict(claim, evidence='["pytest -q"]')
        verdict = report.finish_verdict(wrong, TASK)
        assert verdict is not None and "`pytest -q`: never ran" in verdict and not report.armed
        assert report.gate_rounds == 1 and requests == []


def test_the_diff_is_taken_against_the_runs_base_commit(tmp_path: Path) -> None:
    """A change the agent committed (or the merge of its worktree squashed) is still in
    the diff the reviewer reads: the diff is against the commit the run started from."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    # One line-ending policy for both gits: ``_git`` hides the user's config (its
    # HOME is the repo), the gate's git reads it, and on Windows the two can
    # disagree on ``core.autocrlf``, which decides whether the CRLF file
    # ``write_text`` leaves differs from the LF blob the commit stored.
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "a.py").write_text("a = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "start")
    base = _git(repo, "rev-parse", "HEAD").strip()
    gate = _gate(repo)
    gate.observe("Edit", {"file_path": str(repo / "a.py"), "old_string": "1", "new_string": "2"})
    assert gate.base_commit == base
    (repo / "a.py").write_text("a = 2\n")
    assert "### a.py\n" in gate.diff_report() and "+a = 2" in gate.diff_report()
    _git(repo, "commit", "-q", "-am", "the change")
    (repo / "b.py").write_text("b = 1\n")
    report = gate.diff_report()
    assert "### a.py\n" in report and "-a = 1" in report and "+a = 2" in report
    assert "### b.py (new file)\nb = 1" in report
    # a later run on the same tree starts from the new head, and the untracked file
    # that was already there is not its work: nothing to review until it writes
    later = _gate(repo)
    later.observe("Bash", {"command": "ls"})
    assert later.base_commit != base and later.base_untracked == {"b.py"}
    assert later.repo_root == str(repo) and later.diff_report() == "(no changed files found)"
    (repo / "c.py").write_text("c = 1\n")
    assert later.changed_paths() == [("c.py", True)]
    # ... unless the run wrote it
    later.observe("Write", {"file_path": "b.py", "content": "b = 2\n"})
    (repo / "b.py").write_text("b = 2\n")
    assert later.root_relative("b.py") == "b.py" and later.root_relative("/elsewhere/b.py") == ""
    assert later.changed_paths() == [("b.py", True), ("c.py", True)]
    # a file edited in another repository (its worktree keeps the work dir) is in the
    # diff too, with its own repository's diff, or its content when untracked there
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    (other / "hello.py").write_text("print('bye')\n")
    _git(other, "add", "-A")
    _git(other, "commit", "-q", "-m", "start")
    later.observe("Edit", {"file_path": str(other / "hello.py"), "old_string": "bye",
                           "new_string": "hello"})
    (other / "hello.py").write_text("print('hello')\n")
    later.observe("Write", {"file_path": str(other / "new.py"), "content": "n = 1\n"})
    (other / "new.py").write_text("n = 1\n")
    report = later.diff_report()
    assert report.startswith("### b.py (new file)\nb = 2\n\n### c.py (new file)\nc = 1\n\n")
    assert f"### {other / 'hello.py'} (outside the work dir's repository)\n" in report
    assert "-print('bye')\n+print('hello')" in report and "### a.py" not in report
    assert report.endswith(f"### {other / 'new.py'} (outside the work dir's repository)\nn = 1")
    # outside git there is no base and the written files are the diff
    (tmp_path / "plain").mkdir()
    plain = _gate(tmp_path / "plain")
    plain.observe("Write", {"file_path": str(tmp_path / "plain" / "c.py"), "content": "c = 1\n"})
    (tmp_path / "plain" / "c.py").write_text("c = 1\n")
    assert plain.base_commit is None and plain.diff_report() == "### c.py (new file)\nc = 1"


def test_gate_stage_records_from_the_stage_threads_are_filed_under_the_task(
    tmp_path: Path,
) -> None:
    """The verifier and the review run on threads of their own, yet their ``gate_stage``
    records reach the task's event log: a real ``JsonPrinter`` bound to a task on the
    gate's thread records them under that task, as it does the ``round`` record."""
    printer = JsonPrinter()
    printer.adopt_thread_binding("task-9")
    printer.start_recording()
    previous = bind_printer(printer)
    try:
        with serve([text_body("VERDICT: OK")]) as (url, requests):
            gate = _gate(tmp_path, verifier="exit 0\n", reviewer=MODEL,
                         model_config={"base_url": url, "api_key": "local"})
            gate.observe("Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"})
            _turn(gate, "Bash", {"command": "bash tmp/verify.sh"}, "")
            assert gate.finish_verdict({"success": True, "summary_in_html": DONE}, TASK) is None
            assert len(requests) == 1 and gate.reviews == 1
    finally:
        bind_printer(previous)
        printer.adopt_thread_binding("")
    recorded = [e for e in printer.peek_recording_for_task("task-9") if e["type"] == "gate_stage"]
    assert sorted((e["stage"], e["outcome"], e["taskId"]) for e in recorded) == [
        ("review", "pass", "task-9"), ("verifier", "pass", "task-9")]
    review = next(e for e in recorded if e["stage"] == "review")
    assert review["model"] == MODEL and review["round"] == 1 and review["cost_usd"] >= 0


def _no_reads(name: str, args: dict[str, Any]) -> Verdict:
    """A caller's ``tool_call_hook`` that refuses every ``Read`` (the guard never sees it)."""
    return refuse("Not now: reads are off in this run.") if name == "Read" else ALLOW


def _run_agent(work_dir: Path, script: list[bytes], max_steps: int = 14,
               hook: Callable[[str, dict[str, Any]], Verdict] | None = None,
               ) -> tuple[SorcarAgent, dict, list[dict]]:
    """Run a real ``SorcarAgent`` in *work_dir* against the scripted *script*."""
    with serve(script) as (url, requests):
        agent = SorcarAgent("coding-gate")
        result = agent.run(
            prompt_template=TASK, model_name=MODEL, work_dir=str(work_dir), max_steps=max_steps,
            max_budget=1.0, model_config={"base_url": url, "api_key": "local"},
            use_memory=False, verbose=False, web_tools=False, tool_call_hook=hook,
        )
    return agent, yaml.safe_load(result), requests


def _tool_results(requests: list[dict], index: int) -> str:
    """Every tool result the model saw in its *index*-th request, joined."""
    return "\n".join(str(m.get("content", "")) for m in requests[index]["messages"]
                     if m.get("role") == "tool")


def _last_tool_result(request: dict) -> str:
    """The last tool result of *request*'s messages (``""`` for a review request)."""
    results = [str(m.get("content", "")) for m in request["messages"] if m.get("role") == "tool"]
    return results[-1] if results else ""


def _pin_reviewer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reviewer: str) -> None:
    """Point the user config at a file naming the gate's *reviewer* (``off`` for none)."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({cg.REVIEWER_CONFIG_KEY: reviewer}))
    monkeypatch.setattr(user_config, "CONFIG_PATH", config)


def test_sorcar_agent_gates_a_coding_run_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard observes, the result hook reads exit codes, the model reads the refusals.

    The ledger reads every result as the framework delivers it: a call the
    caller's hook refused before the guard saw it, a call the tool raised
    on (``Failed to call``), and a here-document into an existing file the
    ``Bash`` tool made as a ``Write`` with the rest of the command as a
    follow-up call.
    """
    _pin_reviewer(tmp_path, monkeypatch, "off")
    assert cg.gate_reviewer(MODEL) is None
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "verify.sh").write_text("test -f hello.py && grep -q hello hello.py\n")
    hello = str(tmp_path / "hello.py")
    heredoc = "cat > hello.py <<'EOF'\nprint('hello')\nEOF\nbash tmp/verify.sh"
    with_evidence = {"success": True, "is_continue": False, "summary_in_html": DONE,
                     "evidence": '["bash tmp/verify.sh", "printf gate-evidence"]'}
    script = [
        tool_call_body("Write", {"file_path": hello, "content": "print('bye')\n"}, 1000),
        finish_body(DONE, prompt_tokens=1000),  # round 1: verifier fails, edit unchecked
        tool_calls_body([("Read", {"file_path": hello}),  # refused by the caller's hook
                         ("Bash", {"command": "printf gate-evidence", "description": "d"})],
                        1000),
        tool_call_body("Bash", {"command": "bash tmp/verify.sh"}, 1000),  # no description
        tool_call_body("finish", with_evidence, 1000),  # round 2: the check never ran
        tool_call_body("Bash", {"command": heredoc, "description": "write and check"}, 1000),
        tool_call_body("finish", with_evidence, 1000),  # passes
    ]
    agent, result, requests = _run_agent(tmp_path, script, hook=_no_reads)
    assert result["success"] is True and result["summary"] == DONE
    assert len(requests) == 7
    first = _tool_results(requests, 2)
    assert "Finish refused (round 1 of 3" in first and "exited 1" in first
    assert "files were changed after the last shell command" in first
    turn = _tool_results(requests, 3)
    assert "Not now: reads are off" in turn and "gate-evidence\n" in turn
    assert _last_tool_result(requests[4]).startswith("Failed to call Bash")
    second = _last_tool_result(requests[5])
    assert "round 2 of 3" in second and "`bash tmp/verify.sh`: last exit code error" in second
    assert "printf gate-evidence" not in second  # that evidence held
    redirected = _last_tool_result(requests[6])
    assert redirected.startswith("Applied as Write(") and "Then made Bash(" in redirected
    gate = agent._coding_gate
    assert gate.exit_codes == {"printf gate-evidence": "0", "bash tmp/verify.sh": "0",
                               heredoc: "0"}
    assert gate.gate_rounds == 2 and gate.verifier_refusals == 2
    assert gate.last_verifier_run[0] == 0
    assert gate.reviewer is None and gate.verifier_path == str(tmp_path / "tmp" / "verify.sh")
    assert (tmp_path / "hello.py").read_text() == "print('hello')\n"


def test_sorcar_agent_runs_the_gates_review_and_charges_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured reviewer reads the diff at the finish; its findings hold the run."""
    _pin_reviewer(tmp_path, monkeypatch, MODEL)
    assert cg.gate_reviewer("claude-fable-5") == MODEL
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "verify.sh").write_text("grep -q hello hello.py\n")
    hello = str(tmp_path / "hello.py")
    claim = {"success": True, "is_continue": False, "summary_in_html": DONE,
             "evidence": '["bash tmp/verify.sh"]'}
    script = [
        tool_call_body("Write", {"file_path": hello, "content": "print('hello')"}, 1000),
        tool_call_body("Bash", {"command": "bash tmp/verify.sh", "description": "check"}, 1000),
        tool_call_body("finish", claim, 1000),
        text_body("VERDICT: NOT DONE\n1. hello.py: no newline at the end of the file."),
        tool_call_body("Edit", {"file_path": hello, "old_string": "')", "new_string": "')\n"},
                       1000),
        tool_call_body("Bash", {"command": "bash tmp/verify.sh", "description": "check"}, 1000),
        tool_call_body("finish", claim, 1000),
        text_body("VERDICT: OK"),
    ]
    agent, result, requests = _run_agent(tmp_path, script)
    assert result["success"] is True and result["summary"] == DONE
    assert len(requests) == 8
    reviews = [r for r in requests if "You are reviewing, read-only" in str(r["messages"][-1])]
    assert len(reviews) == 2
    assert "### hello.py (new file)\nprint('hello')" in reviews[0]["messages"][-1]["content"]
    assert "exit 0: `bash tmp/verify.sh`" in reviews[0]["messages"][-1]["content"]
    refusal = _tool_results(requests, 4)
    assert "independent review" in refusal and "no newline at the end" in refusal
    gate = agent._coding_gate
    assert gate.reviews == 2 and gate.open_findings == [] and gate.gate_rounds == 1
    # the two review calls are in the run's accounting: the executor's six
    # calls report 1,100 tokens each, the reviewer's two 600 each
    assert agent.total_tokens_used == 6 * 1100 + 2 * 600
    assert agent.budget_used > 0


def _fixed_commit_message(commit_dir: Path, user_prompt: str | None = None,
                          task_result: str | None = None, bill_to: Any = None) -> str:
    """The merge's commit message without the model call that writes it in a real run."""
    return "feat: hello"


@pytest.mark.parametrize("subdir", ["", "pkg"])
def test_sorcar_agent_gates_the_merge_of_its_worktree_and_keeps_the_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subdir: str,
) -> None:
    """In a git checkout the work happens in a worktree the agent merges before it
    finishes: the gate follows the work dir into the worktree, runs the verifier and
    the review at ``commit_and_merge`` (refusing it while they fail, so the fix is made
    in the worktree), the merge carries the verifier back to the checkout, and the
    finish re-runs it on the merged tree without another review.  A task started in
    a subdirectory of the repository has the same: the reviewer reads the diff of the
    whole repository (root-relative paths) and the verifier comes back under the
    task's own directory."""
    from kiss.tests.server.conftest import init_repo, kiss_wt_branches, run_git

    _pin_reviewer(tmp_path, monkeypatch, MODEL)
    monkeypatch.setattr(sorcar_agent_module, "_generate_commit_message", _fixed_commit_message)
    repo = tmp_path / "repo"
    init_repo(repo)
    (repo / ".gitignore").write_text("tmp/\n")
    run_git(repo, "add", ".gitignore")
    run_git(repo, "commit", "-q", "-m", "ignore tmp")
    seed = run_git(repo, "rev-parse", "HEAD").stdout.strip()
    work = repo / subdir
    work.mkdir(exist_ok=True)
    shown = f"{subdir}/hello.py" if subdir else "hello.py"
    claim = {"success": True, "is_continue": False, "summary_in_html": DONE,
             "evidence": '["bash tmp/verify.sh"]'}
    script = [
        tool_call_body("create_worktree", {}, 1000),
        tool_call_body("Write", {"file_path": "tmp/verify.sh",
                                 "content": "grep -q hello hello.py\n"}, 1000),
        tool_call_body("Write", {"file_path": "hello.py", "content": "print('bye')\n"}, 1000),
        tool_call_body("Bash", {"command": "bash tmp/verify.sh", "description": "check"}, 1000),
        tool_call_body("commit_and_merge", {}, 1000),  # refused: verifier exit 1, a finding
        text_body("VERDICT: NOT DONE\n1. hello.py: prints bye, not hello."),
        tool_call_body("Edit", {"file_path": "hello.py", "old_string": "bye",
                                "new_string": "hello"}, 1000),
        tool_call_body("Bash", {"command": "bash tmp/verify.sh", "description": "check"}, 1000),
        tool_call_body("commit_and_merge", {}, 1000),  # reviewed again, then merged
        text_body("VERDICT: OK"),
        tool_call_body("finish", claim, 1000),  # the verifier runs on the merged checkout
    ]
    agent, result, requests = _run_agent(work, script, max_steps=20)
    assert result["success"] is True and result["summary"] == DONE, result
    assert len(requests) == 11
    results = [_last_tool_result(r) for r in requests]
    refused = next(r for r in results if "Merge refused (round 1 of 3" in r)
    assert refused.startswith("REFUSED: the commit_and_merge call did NOT run")
    assert "exited 1" in refused and "1. hello.py: prints bye, not hello." in refused
    assert "before calling commit_and_merge again" in refused
    merged = next(r for r in results if r.startswith("Successfully merged branch"))
    assert f"Copied tmp/verify.sh to {work / 'tmp' / 'verify.sh'}" in merged
    reviews = [r for r in requests if "You are reviewing, read-only" in str(r["messages"][-1])]
    assert len(reviews) == 2
    assert cg.MERGE_SUMMARY_PLACEHOLDER in reviews[0]["messages"][-1]["content"]
    # at the merge no finish names evidence: the reviewer is told so, not "(none named)"
    assert cg.MERGE_EVIDENCE_PLACEHOLDER in reviews[0]["messages"][-1]["content"]
    assert "(none named)" not in reviews[0]["messages"][-1]["content"]
    assert f"### {shown} (new file)\nprint('bye')" in reviews[0]["messages"][-1]["content"]
    assert f"### {shown} (new file)\nprint('hello')" in reviews[1]["messages"][-1]["content"]
    assert "since your previous review" in reviews[1]["messages"][-1]["content"]
    # the work is on main, the worktree is gone, the verifier survived the merge
    assert (work / "hello.py").read_text() == "print('hello')\n"
    assert kiss_wt_branches(repo) == [] and list((repo / ".kiss-worktrees").iterdir()) == []
    assert run_git(repo, "log", "--format=%s", f"{seed}..HEAD").stdout.split() == ["feat:", "hello"]
    assert (work / "tmp" / "verify.sh").read_text() == "grep -q hello hello.py\n"
    assert not (repo / ".kiss-worktrees" / "tmp").exists()
    gate = agent._coding_gate
    assert gate.work_dir == str(work) and gate.base_commit == seed
    # the merge that went through left its round behind: the finish was gated afresh
    assert gate.gate_rounds == 0 and gate.reviews == 2 and gate.open_findings == []
    assert gate.last_verifier_run[0] == 0 and gate.exit_codes == {"bash tmp/verify.sh": "0"}


def test_a_merge_with_the_reviews_findings_open_and_no_change_is_refused(tmp_path: Path) -> None:
    """A second ``commit_and_merge`` with nothing changed since the review that found
    fault is refused on those findings (no model call); a change is reviewed again."""
    script = [text_body("VERDICT: NOT DONE\n1. hello.py prints bye, not hello."),
              text_body("VERDICT: OK")]
    hello = tmp_path / "hello.py"
    with serve(script) as (url, requests):
        gate = _gate(tmp_path, reviewer=MODEL, model_config={"base_url": url, "api_key": "local"})
        hello.write_text("print('bye')\n")
        _turn(gate, "Write", {"file_path": str(hello), "content": "print('bye')\n"}, "ok")
        _turn(gate, "Bash", {"command": "python hello.py"}, "bye\n")
        first = gate.merge_verdict(TASK)
        assert first is not None and "Merge refused (round 1 of 3" in first
        assert "1. hello.py prints bye, not hello." in first and "still open" not in first
        second = gate.merge_verdict(TASK)
        assert second is not None and "Merge refused (round 2 of 3" in second
        assert "still open and no file changed since" in second
        assert "1. hello.py prints bye, not hello." in second
        assert len(requests) == 1 and gate.gate_rounds == 2 and gate.reviews == 1
        _turn(gate, "Edit", {"file_path": str(hello), "old_string": "bye", "new_string": "hello"},
              "ok")
        hello.write_text("print('hello')\n")
        _turn(gate, "Bash", {"command": "python hello.py"}, "hello\n")
        assert gate.merge_verdict(TASK) is None
        assert len(requests) == 2 and gate.reviews == 2 and gate.open_findings == []


def test_a_merge_that_went_through_leaves_its_rounds_and_findings_behind(tmp_path: Path) -> None:
    """Audit 2026-11-01 (task 4aa14638): three refused merges spent the round cap, the
    merge then went through, and the finish delivered the merge's stale findings as
    caveats. ``merged()`` resets the rounds and findings; so does a review the agent
    dispatched itself after a later change."""
    script = [text_body("VERDICT: NOT DONE\n1. hello.py prints bye, not hello.")]
    hello = tmp_path / "hello.py"
    with serve(script) as (url, requests):
        gate = _gate(tmp_path, reviewer=MODEL, model_config={"base_url": url, "api_key": "local"})
        hello.write_text("print('bye')\n")
        _turn(gate, "Write", {"file_path": str(hello), "content": "print('bye')\n"}, "ok")
        _turn(gate, "Bash", {"command": "python hello.py"}, "bye\n")
        for round_number in (1, 2, 3):
            verdict = gate.merge_verdict(TASK)
            assert verdict is not None and f"round {round_number} of 3" in verdict
        assert gate.merge_verdict(TASK) is None  # past the cap the merge goes through
        assert gate.gate_rounds == 3
        assert gate.open_findings == ["1. hello.py prints bye, not hello."]
        gate.merged()
        assert gate.gate_rounds == 0 and gate.open_findings == []
        # the finish after the merge: no file changed, nothing to re-review, no caveat
        args: dict[str, Any] = {"success": True, "summary_in_html": DONE}
        assert gate.finish_verdict(args, TASK) is None
        assert args["summary_in_html"] == DONE and gate.delivered_findings == []
        assert len(requests) == 1
    # a review the agent dispatched after a later change supersedes the gate's findings
    gate = _gate(tmp_path)
    gate.gate_rounds, gate.open_findings = 2, ["1. stale"]
    _turn(gate, "Edit", {"file_path": str(hello), "old_string": "bye", "new_string": "hello"}, "ok")
    _turn(gate, "run_agent", {"task": "Verify the listed changes: hello.py. Report only "
                              "demonstrated issues.", "tool_profile": "review"},
          "ran: sorcar\nsuccess: true\nsummary: no issues\n")
    assert gate.reviewed_by_agent and gate.gate_rounds == 0 and gate.open_findings == []
    # one dispatched before the change reviewed an older diff: nothing is reset
    gate.gate_rounds, gate.open_findings = 2, ["1. stale"]
    _turn(gate, "run_agent", {"task": "Review hello.py", "tool_profile": "review"},
          "Started as job agent-abc; agent_job(job_id) collects it.")
    _turn(gate, "Edit", {"file_path": str(hello), "old_string": "hello", "new_string": "hi"}, "ok")
    _turn(gate, "agent_job", {"job_id": "agent-abc", "action": "wait"},
          "ran: sorcar\nsuccess: true\nsummary: fine\n")
    assert not gate.reviewed_by_agent and gate.gate_rounds == 2
    assert gate.open_findings == ["1. stale"]


def test_scratch_files_are_notes_not_changes_of_the_work(tmp_path: Path) -> None:
    """An ``Edit``/``Write`` under ``tmp/`` (the progress log, a probe, the verifier) and a
    redirect into ``tmp/`` leave no unchecked change, arm nothing and need no re-review."""
    gate = _gate(tmp_path, verifier="exit 0\n")
    _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"}, "ok")
    _turn(gate, "Bash", {"command": "python a.py > tmp/out.txt 2>&1"}, "")
    assert gate.armed and not gate.evidence.unchecked_change and gate.changes == 1
    for path in ("tmp/PROGRESS.md", "./tmp/c16c/probe.py", str(tmp_path / "tmp" / "verify.sh"),
                 "sub/tmp/notes.md"):
        _turn(gate, "Edit", {"file_path": path, "old_string": "a", "new_string": "b"}, "ok")
        _turn(gate, "Write", {"file_path": path, "content": "x"}, "ok")
    assert not gate.evidence.unchecked_change and gate.changes == 1
    assert gate.written_files == [str(tmp_path / "a.py")]
    done = {"success": True, "summary_in_html": DONE}
    assert gate.finish_verdict(dict(done), TASK) is None
    # a code file that merely sits next to a tmp name, or outside the work dir, is work
    for path in ("tmpl/a.py", "a.tmp", "tmp", str(tmp_path.parent / "tmp" / "x.py")):
        assert not gate.is_scratch(path), path
    _turn(gate, "Write", {"file_path": "tmpl/a.py", "content": "x"}, "ok")
    assert gate.evidence.unchecked_change and gate.changes == 2
    # a probe-only run (scratch python, no product code) is not a coding run
    (tmp_path / "other").mkdir()
    probe = _gate(tmp_path / "other")
    probe.observe("Write", {"file_path": "tmp/probe.py", "content": "print(1)\n"})
    assert not probe.armed and not probe.coding_run()


def test_a_delegated_change_is_judged_by_the_diff(tmp_path: Path) -> None:
    """A sub-task that may change files makes the run a coding run when the diff
    shows a code file changed, and its change is unchecked until a shell runs."""
    from kiss.tests.server.conftest import init_repo

    repo = tmp_path / "repo"
    init_repo(repo)
    gate = _gate(repo)
    done = {"success": True, "summary_in_html": DONE}
    _turn(gate, "run_agent", {"task": "Implement hello.py so it prints hello"},
          "ran: sorcar\nsuccess: true\nsummary: written\n")
    (repo / "notes.md").write_text("done\n")  # the child wrote prose only: not gated
    assert gate.delegated and not gate.coding_run()
    assert gate.finish_verdict(dict(done), TASK) is None
    (repo / "hello.py").write_text("print('hello')\n")  # the child changed code
    assert gate.coding_run() and gate.changed_paths() == [("hello.py", True), ("notes.md", True)]
    verdict = gate.finish_verdict(dict(done), TASK)
    assert verdict is not None and "files were changed after the last shell command" in verdict
    _turn(gate, "Bash", {"command": "python hello.py"}, "hello\n")
    assert gate.finish_verdict(dict(done), TASK) is None


def test_a_stop_during_the_review_still_charges_its_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop that ends the finish review charges what the reviewer already spent.

    The charge ran after the attempt loop, so a stop (an injected
    ``KeyboardInterrupt``) landing while the reviewer answered skipped
    it and the review's billed spend never reached the run (review of
    2026-11-01, finding 2).  Each attempt is now charged as it ends.
    """
    from kiss.core.kiss_agent import KISSAgent

    _git(tmp_path, "init", "-q")
    (tmp_path / "hello.py").write_text("print('hello')\n")
    charged: dict[str, tuple[float, int, int]] = {}  # by key: a retried charge counts once

    def charge(budget: float, tokens: int, steps: int, key: str) -> None:
        charged.setdefault(key, (budget, tokens, steps))

    real_run = KISSAgent.run

    def run(self: KISSAgent, *args: Any, **kwargs: Any) -> str:
        result: str = real_run(self, *args, **kwargs)
        if self.name == "Finish reviewer":
            raise KeyboardInterrupt("stop")
        return result

    monkeypatch.setattr(KISSAgent, "run", run)
    with serve([text_body("Looking.\nVERDICT: OK")]) as (url, requests):
        gate = _gate(tmp_path, verifier="grep -q . hello.py\n", reviewer=MODEL,
                     charge=charge,
                     model_config={"base_url": url, "api_key": "local"})
        with pytest.raises(KeyboardInterrupt):
            gate.review_verdict({"success": True, "summary_in_html": DONE}, TASK)
        assert len(requests) == 1
    assert len(charged) == 1
    ((budget, tokens, steps),) = charged.values()
    assert budget > 0 and tokens > 0 and steps == 1


@pytest.mark.parametrize("stop_after_charge", [False, True])
def test_a_stop_inside_the_review_charge_is_retried_and_counted_once(
    tmp_path: Path, stop_after_charge: bool
) -> None:
    """A stop landing inside the attempt's own charge still charges it exactly once.

    The charge runs in a ``finally``; a stop there used to lose the
    completed review's spend (review of 2026-11-01, round 2, finding
    2).  The gate charges again with the same key before the stop
    propagates, and the run's keyed ledger counts that key once.
    """
    from kiss.agents.sorcar.sorcar_agent import _agent_usage

    _git(tmp_path, "init", "-q")
    (tmp_path / "hello.py").write_text("print('hello')\n")
    run = SorcarAgent("gate-charge-retry")
    keys: list[str] = []

    def charge(budget: float, tokens: int, steps: int, key: str) -> None:
        keys.append(key)
        if len(keys) == 1 and not stop_after_charge:
            raise KeyboardInterrupt("stop before the charge")
        run._charge_gate_usage(budget, tokens, steps, key)
        if len(keys) == 1:
            raise KeyboardInterrupt("stop after the charge")

    with serve([text_body("Looking.\nVERDICT: OK")]) as (url, requests):
        gate = _gate(tmp_path, verifier="grep -q . hello.py\n", reviewer=MODEL, charge=charge,
                     model_config={"base_url": url, "api_key": "local"})
        with pytest.raises(KeyboardInterrupt):
            gate.review_verdict({"success": True, "summary_in_html": DONE}, TASK)
        assert len(requests) == 1
    assert len(keys) == 2 and keys[0] == keys[1]
    budget, tokens, steps = _agent_usage(run)
    assert budget > 0 and tokens > 0 and steps == 1
    run._charge_gate_usage(budget, tokens, steps, keys[0])  # yet another retry: still once
    assert _agent_usage(run) == (budget, tokens, steps)
