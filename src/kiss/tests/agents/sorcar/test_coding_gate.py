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
decisions model is optional by design: without it the findings check is
silent (every open finding counts as answered), so the tests that depend
on its judgement branch on ``decisions_tool_available()``.
"""

from __future__ import annotations

import functools
import json
import os
import shlex
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar import coding_gate as cg
from kiss.agents.sorcar import sorcar_agent as sorcar_agent_module
from kiss.agents.sorcar.agent_dispatch import digest_agent_results
from kiss.agents.sorcar.coding_gate import CodingGate, run_on_host
from kiss.agents.sorcar.decide_tool import decisions_tool_available
from kiss.agents.sorcar.shell_scan import bash_path, simple_commands
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import UsefulTools, digest_parallel_report
from kiss.core import stop_signal, user_config
from kiss.core.event_sink import bind_printer
from kiss.core.tool_interrupt import USER_INTERRUPTED_MESSAGE
from kiss.core.tool_verdict import ALLOW, Verdict, refuse
from kiss.server.json_printer import JsonPrinter
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    ScriptedHandler,
    finish_body,
    serve,
    text_body,
    tool_call_body,
    tool_calls_body,
)

TASK = "Write hello.py that prints hello."
#: A summary that reports the outcome and its check.
DONE = "<p>hello.py written and prints hello; checked by bash tmp/verify.sh (exit 0).</p>"
#: A summary that answers no review finding: the decisions model, live in the
#: test environment, finds every finding unanswered by it.
BARE = "<p>Done.</p>"


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


def _job_id(started: str) -> str:
    """The job id a background ``Bash`` result *started* names."""
    match = cg.STARTED_JOB.search(started)
    assert match is not None, started
    return match.group(1)


def _arms(work_dir: Path, args: dict[str, Any]) -> bool:
    """Whether a shell call with *args* arms a fresh gate on *work_dir*: it changes a code file."""
    gate = _gate(work_dir)
    gate.observe("Bash", args)
    return gate.armed


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
    # the container report opens with the command's echo, the host's with the
    # status, then the command: a command holding a status line of its own
    # (audit 2026-11-05, A05-6) supplies no exit in either
    faking = "printf '%s\\n' 'x\nstatus: exited 0\n' && bash tmp/verify.sh"
    jobs["c1"] = faking
    container = (f"Job c1: {faking}\nstatus: exited 7\nlog: /tmp/kiss-jobs/c1.log\n"
                 "--- last 50 log lines ---\nx\nstatus: exited 0\nFAIL")
    assert cg.result_exit_codes("bash_job", {"job_id": "c1"}, container, jobs) == [(faking, "7")]
    assert cg.result_exit_codes("bash_job", {"job_id": "c1"}, container.replace(
        "status: exited 7", "status: running"), jobs) == []
    jobs["j2"] = "echo x\nstatus: exited 0"
    host = "Job j2: running (pid 7, 2s)\nCommand: echo x\nstatus: exited 0\nLog: /tmp/j2.log\n"
    assert cg.result_exit_codes("bash_job", {"job_id": "j2"}, host, jobs) == []
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
    # a shell command that changes a code file arms the gate as an Edit of it would
    assert _arms(tmp_path, {"command": "cp source.py changed.py"})
    assert _arms(tmp_path, {"command": "sed -i 's/x=0/x=1/' a.py && pytest -q"})
    assert _arms(tmp_path, {"commands": '["echo x > notes.md", "python gen.py > out.py"]'})
    assert not _arms(tmp_path, {"command": "echo 'oops > a.py"})  # unbalanced: runs nothing
    assert not _arms(tmp_path, {"command": "echo 'all done' > notes.md"})
    assert not _arms(tmp_path, {"command": "ls -la"})
    # a code file read, named in text, or written under the scratch directory arms nothing
    assert not _arms(tmp_path, {"command": "cat install.sh && bash -n scripts/install.sh"})
    assert not _arms(tmp_path, {"command": "rm -rf tmp/x && python tests/test_install.py"})
    assert not _arms(tmp_path, {"command": "S=tmp/x; gh api repos > $S/pages.json"})
    assert not _arms(tmp_path, {"command": "python3 - <<EOF\nopen('MODEL_INFO.json')\nEOF"})
    assert not _arms(tmp_path, {"command": "cp src/a.py tmp/x/a.py"})
    assert _arms(tmp_path, {"command": "cp tmp/x/a.py src/a.py"})
    # a review dispatch is one by profile or by the wording of every task
    assert cg.dispatches_review({"task": "do it", "tool_profile": "review"})
    assert cg.dispatches_review({"tasks": '["Review the diff of foo.py for bugs"]'})
    assert not cg.dispatches_review({"task": "Implement the parser"})
    assert not cg.dispatches_review({"tasks": '["Review the diff of foo.py for bugs", '
                                              '"Implement the parser"]'})
    assert not cg.dispatches_review({"tasks": "[]"})


def test_a_coding_run_is_gated_on_its_verifier_and_the_evidence_named_refuses_nothing(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path, verifier="#!/bin/bash\ntest -f hello.py && grep -q hello hello.py\n")
    gate.observe("Read", {"file_path": "x"})
    done = {"success": True, "summary_in_html": DONE}
    # the verifier's presence makes it a coding run: the verifier refuses the finish
    verdict = gate.finish_verdict(dict(done), TASK)
    assert verdict is not None and verdict.startswith("Finish refused (round 1 of up to 3; ")
    assert "a later round is refused only while your verifier fails" in verdict
    assert "exited 1" in verdict and "./tmp/verify.sh" in verdict
    assert gate.armed and gate.verifier_refusals == 1
    # the edit, then the check the agent ran: its exit code is read from the result
    _turn(gate, "Write", {"file_path": str(tmp_path / "hello.py"), "content": "print('bye')\n"},
          "Successfully wrote 13 characters")
    _turn(gate, "Bash", {"command": "bash tmp/verify.sh"},
          _host_result("bash tmp/verify.sh", tmp_path))
    assert gate.exit_codes == {"bash tmp/verify.sh": "1"}
    # the evidence named (a failed command, one that never ran) refuses nothing of its
    # own: it is the reviewer's input; the verifier refuses the second round
    claimed = dict(done, evidence='["bash tmp/verify.sh", "pytest -q"]')
    verdict = gate.finish_verdict(claimed, TASK)
    assert verdict is not None and "round 2 of up to 3" in verdict and "exited 1" in verdict
    assert "pytest -q" not in verdict and "last exit code" not in verdict
    assert gate.evidence_report(claimed) == ("- exit 1: `bash tmp/verify.sh`\n"
                                             "- exit not run: `pytest -q`")
    # the fix and the check re-run: the finish goes through unchanged, whatever it names
    _turn(gate, "Edit", {"file_path": str(tmp_path / "hello.py"), "old_string": "bye",
                         "new_string": "hello"}, "Successfully replaced")
    (tmp_path / "hello.py").write_text("print('hello')\n")
    _turn(gate, "Bash", {"command": "bash tmp/verify.sh"},
          _host_result("bash tmp/verify.sh", tmp_path))
    assert gate.exit_codes["bash tmp/verify.sh"] == "0"
    assert gate.finish_verdict(claimed, TASK) is None
    assert claimed["summary_in_html"] == DONE and gate.gate_rounds == 2
    assert gate.last_verifier_run[0] == 0


def test_a_run_that_changed_no_code_is_not_gated(tmp_path: Path) -> None:
    """A prose-only run (no code file written, no verifier) gets no verifier run, no
    review and no refusal, whatever its summary says or names as evidence; a code
    file written arms the gate, whose first implicit finish is vetoed once."""
    with serve([text_body("VERDICT: NOT DONE\n1. never asked")]) as (url, requests):
        prose = _gate(tmp_path, reviewer=MODEL, model_config={"base_url": url, "api_key": "local"})
        _turn(prose, "Write", {"file_path": str(tmp_path / "notes.md"), "content": "# T\n"},
              "Successfully wrote 4 characters")
        (tmp_path / "notes.md").write_text("# T\n")
        assert not prose.armed and not prose.coding_run()
        assert prose.finish_verdict({}, TASK) is None
        for summary in (DONE, "<p>Done, but untested: an educated guess.</p>"):
            claim = {"success": True, "summary_in_html": summary, "evidence": '["pytest -q"]'}
            assert prose.finish_verdict(claim, TASK) is None
            assert claim["summary_in_html"] == summary
        assert requests == [] and prose.reviews == 0 and prose.gate_rounds == 0
        assert prose.last_verifier_run == (None, "") and not prose.armed
    gate = _gate(tmp_path)  # no verifier: the edit of a code file arms the gate
    gate.observe("Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"})
    assert gate.armed and gate.finish_verdict({}, TASK) == cg.IMPLICIT_FINISH_VERDICT
    assert gate.finish_verdict({}, TASK) is None  # the veto is spent
    # no verifier, no reviewer, an unchecked edit and a summary that hedges: nothing refuses
    hedged = {"success": True, "summary_in_html": DONE.replace("checked by", "untested;")}
    assert gate.finish_verdict(hedged, TASK) is None and gate.gate_rounds == 0


def test_the_round_cap_delivers_with_caveats_and_the_verifier_cap_is_a_caveat(
    tmp_path: Path,
) -> None:
    """A failing verifier refuses every round up to :data:`MAX_GATE_ROUNDS`."""
    gate = _gate(tmp_path, verifier="echo 'hello.py missing'; exit 7\n")
    args: dict[str, Any]
    for round_number in (1, 2, 3):
        args = {"success": True, "summary_in_html": DONE}
        verdict = gate.finish_verdict(args, TASK)
        assert verdict is not None and f"round {round_number} of up to 3" in verdict
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
    key: str = "", model: str = "",
) -> None:
    """The gate's ``charge`` callback for the tests: records what a review cost."""
    charged.append((budget, tokens, steps))


def _git(work_dir: Path, *args: str) -> str:
    """Run a git command in *work_dir* (user identity set) and return its output."""
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@x", HOME=str(work_dir))
    return subprocess.run(["git", *args], cwd=work_dir, env=env, check=True, text=True,
                          capture_output=True).stdout


def test_the_review_runs_once_and_its_findings_buy_one_round(tmp_path: Path) -> None:
    """The reviewer reads the task, the summary, the verifier, the evidence named (each
    with the exit code of its last run) and the diff against the base commit, once per
    run (:data:`MAX_REVIEWS`): a later finish is judged on its findings without a
    model call, whether or not a file changed.  The first refusal buys one round
    whatever refused it; a finish with the findings still open then goes through with
    them as caveats while the verifier passes, and is refused again while it fails."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "hello.py").write_text("print('bye')\n")
    (tmp_path / ".gitignore").write_text("tmp/\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "start")
    (tmp_path / "stale.py").write_text("untracked before the run\n")
    charged: list[tuple[float, int, int]] = []
    finding = "1. hello.py: prints bye, not hello."
    unanswered = cg.UNANSWERED_VERDICT.format(findings=finding)
    hello = {"file_path": str(tmp_path / "hello.py"), "old_string": "a", "new_string": "b"}
    with serve([text_body(f"Looks wrong.\nVERDICT: NOT DONE\n{finding}")]) as (url, requests):
        gate = _gate(tmp_path, verifier="grep -q . hello.py\n", reviewer=MODEL,
                     charge=functools.partial(_record_charge, charged),
                     model_config={"base_url": url, "api_key": "local"})
        gate.observe("Edit", hello)
        (tmp_path / "hello.py").write_text("print('bye bye')\n")
        (tmp_path / "new.py").write_text("x = 1\n")
        _turn(gate, "Bash", {"command": "python hello.py"}, "bye bye\n")
        done: dict[str, Any] = {"success": True, "summary_in_html": BARE,
                                "evidence": '["python hello.py", "pytest -q"]'}
        verdict = gate.finish_verdict(dict(done), TASK)
        assert verdict is not None and verdict.startswith("Finish refused (round 1 of up to 3; ")
        assert finding in verdict and gate.open_findings == [finding]
        assert gate.reviews == 1 and len(requests) == 1 and charged and charged[0][1] > 0
        assert charged[0][2] == 1, "the review's one model call is charged as a step"
        prompt = requests[0]["messages"][-1]["content"]
        assert TASK in prompt and BARE in prompt and "grep -q . hello.py" in prompt
        assert "- exit 0: `python hello.py`\n- exit not run: `pytest -q`" in prompt
        assert "### hello.py\n" in prompt and "-print('bye')" in prompt
        assert "+print('bye bye')" in prompt
        assert "### new.py (new file)\nx = 1" in prompt and "stale.py" not in prompt
        # the second finish, nothing changed and the finding not answered: no second
        # review and no refusal (the verifier passes); the finding is a caveat of the
        # summary where the decisions model judges it unanswered
        again = dict(done)
        assert gate.finish_verdict(again, TASK) is None
        assert len(requests) == 1 and gate.reviews == 1 and gate.gate_rounds == 1
        if decisions_tool_available():
            assert again["summary_in_html"].startswith(
                BARE + "\n<h4>Delivered after 1 refused finish rounds")
            assert finding in again["summary_in_html"]
            assert gate.delivered_findings == [unanswered.splitlines()[0]]
        else:
            assert again["summary_in_html"] == BARE
        # a change after the review brings no second review (the one review is spent);
        # while the verifier fails the finish is refused again, with the finding
        gate.observe("Edit", hello)
        (tmp_path / "hello.py").write_text("")
        verdict = gate.finish_verdict(dict(done), TASK)
        assert verdict is not None and "round 2 of up to 3" in verdict and "exited 1" in verdict
        assert len(requests) == 1 and gate.reviews == 1 and not gate.review_possible()
        assert gate.open_findings == [finding] and gate.changed_since_review
        if decisions_tool_available():
            assert unanswered in verdict
        # the fix: the verifier passes, so the open finding is a caveat again, not a refusal
        gate.observe("Edit", hello)
        (tmp_path / "hello.py").write_text("print('hello')\n")
        fixed = dict(done)
        assert gate.finish_verdict(fixed, TASK) is None and gate.gate_rounds == 2
        assert len(requests) == 1 and fixed["summary_in_html"].startswith(BARE)
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
        assert not own.reviewed_by_agent and own.jobs == {"agent-1a2b": (True, own.changes, None)}
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
        # review (it may change files) is a change instead when the work's fingerprint moved
        _turn(own, "run_parallel", {"tasks": '["Review hello.py", "Review hello2.py"]'},
              yaml.safe_dump(["Error: no budget", "ran: sorcar\nsuccess: true\nsummary: ok\n"]))
        assert own.reviewed_by_agent
        implement = {"task": "Implement hello3.py so it prints hello"}
        _turn(own, "run_agent", implement, "ran: sorcar\nsuccess: true\nsummary: written\n")
        assert not own.delegated and own.reviewed_by_agent  # the child wrote nothing
        own.observe("run_agent", implement)
        (tmp_path / "hello3.py").write_text("print('hello')\n")  # the child's write
        own.result("run_agent", implement, "ran: sorcar\nsuccess: true\nsummary: written\n",
                   False)
        assert own.delegated and not own.reviewed_by_agent and own.review_wanted()


def test_the_gate_follows_the_work_dir_and_arms_on_shell_written_code(tmp_path: Path) -> None:
    """The verifier is looked for under the work dir of the moment, as the agent moves it
    into a worktree; a writing shell command naming a code file arms the gate as an
    ``Edit`` of it would."""
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
    # audit 2026-11-04 (S8): the agent's passing run of the verifier is a run on the
    # tree of the moment: the move to another tree (the merged checkout, whose files
    # may differ) and a merge that went through drop it, a move to the same dir does not
    (after / "tmp" / "verify.sh").write_text("exit 0\n")
    verify = {"command": "bash tmp/verify.sh"}
    _turn(gate, "Bash", verify, _host_result(verify["command"], after))
    assert gate.verifier_passed_at == gate.changes
    gate.move(str(after))
    assert gate.verifier_passed_at == gate.changes
    gate.move(str(before))
    assert gate.verifier_passed_at is None and gate.work_dir == str(before)
    gate.move(str(after))
    _turn(gate, "Bash", verify, _host_result(verify["command"], after))
    assert gate.verifier_passed_at == gate.changes and gate.merged() == ""
    assert gate.verifier_passed_at is None
    # a shell command that writes a code file arms the gate; one that writes prose does not
    prose = _gate(tmp_path)
    prose.observe("Bash", {"command": "echo 'all done' > notes.md"})
    assert not prose.armed and not prose.coding_run()
    (tmp_path / "source.py").write_text("x = 1\n")
    _turn(prose, "Bash", {"command": "cp source.py changed.py"},
          _host_result("cp source.py changed.py", tmp_path))
    assert prose.armed and prose.coding_run()
    # deleting a code file is a change of the code as well: it arms the gate
    (tmp_path / "old.py").write_text("y = 2\n")
    gone = _gate(tmp_path)
    _turn(gone, "Bash", {"command": "rm old.py"}, _host_result("rm old.py", tmp_path))
    assert not (tmp_path / "old.py").exists() and gone.exit_codes == {"rm old.py": "0"}
    assert gone.armed and gone.coding_run()


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


def test_the_base_follows_head_after_a_sync_tool_unless_the_run_moved_it(tmp_path: Path) -> None:
    """Commits a sync tool pulled are not the run's work (audit 2026-11-03: the /sync
    diff armed the gate); a HEAD the run moved itself stays in the diff.  A git
    subcommand named as an argument (``echo git commit``) moves nothing; one a
    launcher runs (``uv run git commit``) does."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "a.py").write_text("a = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "start")
    gate = _gate(repo, custom_tools=frozenset({"git_sync"}))
    _turn(gate, "Bash", {"command": "echo git commit -m x; git log --oneline | head -1"}, "x\n")
    _turn(gate, "Bash", {"command": "uv run --no-sync echo git commit -m x"}, "git commit -m x\n")
    assert gate.base_commit is not None and not gate.moved_head
    # what the sync tool pulls (here: a commit made behind the gate's back) moves HEAD
    (repo / "pulled.py").write_text("p = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "pulled")
    _turn(gate, "git_sync", {"repo": str(repo)}, "merged origin/main")
    assert gate.base_commit == _git(repo, "rev-parse", "HEAD").strip()
    assert gate.diff_report() == "(no changed files found)" and not gate.coding_run()
    # a commit of the run's own is its work: the base stays behind it
    _turn(gate, "Edit", {"file_path": str(repo / "a.py"), "old_string": "1", "new_string": "2"},
          "ok")
    (repo / "a.py").write_text("a = 2\n")
    _turn(gate, "Bash", {"command": "uv run git commit -q -am 'the change'"}, "")
    _git(repo, "commit", "-q", "-am", "the change")
    base = gate.base_commit
    _turn(gate, "git_sync", {"repo": str(repo)}, "pushed")
    assert gate.moved_head and gate.base_commit == base
    assert "+a = 2" in gate.diff_report()


def test_the_agents_verifier_run_is_reused_only_when_its_exit_decided_the_commands(
    tmp_path: Path,
) -> None:
    """The gate reuses the agent's own passing run of the verifier instead of running it
    again, but only a run whose exit status the command's shows: ``bash -n``, ``|| true``,
    ``| cat``, ``true ||``, ``!``, a command after it, ``&`` or an ``exit`` / ``exec``
    before it hide a failing verifier.  Audit 2026-11-04 (S8): so do a shell option that
    reads the script from elsewhere (``bash -s``: stdin; ``--rcfile``: an option's value)
    or runs nothing (``-D``), and ``set -n`` / ``set -o noexec`` before it.  Finish review
    2026-11-04: ``--`` (or ``-``) ends the shell's options, so ``bash -- -x tmp/verify.sh``
    runs a file named ``-x`` and only passes the verifier to it as an argument.  The
    readers are the module functions ``verifier_command`` and ``verifier_decides`` (the
    coding SEA reads its own commands with them), which the gate's methods call."""
    gate = _gate(tmp_path, verifier="echo FAILURE; exit 7\n")
    work, verifier = str(tmp_path), gate.verifier_path
    _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"}, "ok")
    (tmp_path / "-x").write_text("echo OTHER; exit 0\n")  # the script ``bash -- -x`` runs
    not_run = ("bash -n tmp/verify.sh", "sh -xn ./tmp/verify.sh", "bash -s tmp/verify.sh",
               "bash -D tmp/verify.sh", "bash --rcfile tmp/verify.sh",
               "bash -- -x tmp/verify.sh", "sh - -x tmp/verify.sh", "bash -e -- -x tmp/verify.sh")
    hidden = (*not_run, "bash tmp/verify.sh || true", "bash tmp/verify.sh | cat",
              "true || bash tmp/verify.sh", "bash tmp/verify.sh; echo done",
              "bash tmp/verify.sh &", "(bash tmp/verify.sh) || true",
              "bash tmp/verify.sh && echo PASS || echo FAIL", "bash tmp/verify.sh && true &",
              "bash tmp/verify.sh && echo ok; true", "bash tmp/verify.sh && (true; true) || true",
              "bash tmp/verify.sh\necho \"exit $?\"", "ls || bash tmp/verify.sh && echo ok",
              "! bash tmp/verify.sh", "exit 0; bash tmp/verify.sh", "! (bash tmp/verify.sh)",
              "exec true; bash tmp/verify.sh", "true && ! bash tmp/verify.sh",
              "set -n; bash tmp/verify.sh", "set -o noexec; bash tmp/verify.sh",
              "set -xn; bash tmp/verify.sh", "set -eo noexec; bash tmp/verify.sh",
              "set +n -n; bash tmp/verify.sh")
    for command in hidden:
        result = _host_result(command, tmp_path)
        assert not result.startswith("Error"), (command, result)  # exit 0, verifier or not
        if command in not_run:
            assert "FAILURE" not in result, (command, result)  # the verifier did not run
        _turn(gate, "Bash", {"command": command}, result)
        assert gate.verifier_passed_at is None, command
        assert not cg.verifier_decides(command, work, verifier), command
        found = cg.verifier_command(simple_commands(command, bash_path(work)), verifier)
        assert (found is None) == (command in not_run), command
    code, output = gate.verifier_run()
    assert code == 7 and "FAILURE" in output
    # a run whose exit is the verifier's: reused (the verifier changed on disk behind
    # the gate's back would exit 5 if it ran again)
    (tmp_path / "tmp" / "verify.sh").write_text("echo RUN; exit 0\n")
    for command in (f"cd {shlex.quote(str(tmp_path))} && bash tmp/verify.sh",
                    "timeout 60 bash -x ./tmp/verify.sh 2>&1", "bash tmp/verify.sh > tmp/log",
                    "bash -xe tmp/verify.sh", "bash -- tmp/verify.sh", "bash --posix tmp/verify.sh",
                    "bash - tmp/verify.sh", "sh -e -- tmp/verify.sh",
                    # what follows the verifier under && cannot hide its failure
                    "bash tmp/verify.sh && echo VERIFIED", "true; bash tmp/verify.sh && echo ok",
                    "bash tmp/verify.sh && echo done | tee tmp/log && (echo a; echo b)",
                    "bash tmp/verify.sh && echo $(date) `true`",
                    # ``set`` arguments that leave noexec off: ``+o noexec`` turns it
                    # off, ``--`` ends the options (``-n`` is then a positional
                    # parameter), a later ``+n`` undoes ``-n`` (merge review 2026-11-04)
                    "set +o noexec; bash tmp/verify.sh", "set -- -n; bash tmp/verify.sh",
                    "set -n +n; bash tmp/verify.sh", "set -e +o noexec; bash tmp/verify.sh"):
        assert cg.verifier_decides(command, work, verifier), command
        _turn(gate, "Bash", {"command": command}, _host_result(command, tmp_path))
        assert gate.verifier_passed_at == gate.changes, command
        (tmp_path / "tmp" / "verify.sh").write_text("exit 5\n")
        assert gate.verifier_run()[0] == 0
        gate.verifier_passed_at = None
        (tmp_path / "tmp" / "verify.sh").write_text("echo RUN; exit 0\n")
    # an edit of the verifier, or a command naming it, drops the pass
    _turn(gate, "Bash", {"command": "bash tmp/verify.sh"}, _host_result("bash tmp/verify.sh",
                                                                        tmp_path))
    _turn(gate, "Bash", {"command": "chmod +x tmp/verify.sh"}, "")
    assert gate.verifier_passed_at is None


def test_a_background_verifier_run_counts_for_the_files_it_started_on(tmp_path: Path) -> None:
    """Audit 2026-11-04 (S8): a verifier started with ``Bash(background=True)`` checks the
    files as they were when it started; an edit made while it ran is not covered by the
    exit ``bash_job`` reports, so that exit is not reused at a finish.  One with no edit
    in between is, however often its report is read."""
    gate = _gate(tmp_path, verifier="grep -q 'x = 1' a.py || exit 9\n"
                 "touch tmp/read\nwhile [ ! -f tmp/release ]; do sleep .01; done\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    (tmp_path / "a.py").write_text("x = 1\n")
    _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"}, "ok")
    start: dict[str, Any] = {
        "command": "bash tmp/verify.sh", "description": "verify", "background": True,
    }
    started = tools.Bash(**start)
    job_id = _job_id(started)
    _turn(gate, "Bash", start, started)
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "tmp" / "read").exists():
            assert time.monotonic() < deadline, "the verifier did not start"
            time.sleep(0.01)
        (tmp_path / "a.py").write_text("x = 2\n")  # the edit the running verifier never saw
        _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 2\n"}, "ok")
        (tmp_path / "tmp" / "release").touch()
        wait: dict[str, Any] = {"job_id": job_id, "action": "wait", "timeout_seconds": 15}
        report = tools.bash_job(**wait)
        assert "exited with code 0" in report, report
        _turn(gate, "bash_job", wait, report)
        assert gate.verifier_passed_at is None and gate.changes == 2
        assert gate.verifier_run()[0] == 9
        # a job started after the last edit is proof; a later read of its report
        # does not carry the proof over to an edit made since
        (tmp_path / "a.py").write_text("x = 1\n")
        _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"}, "ok")
        started = tools.Bash(**start)
        wait = {"job_id": _job_id(started), "action": "wait", "timeout_seconds": 15}
        _turn(gate, "Bash", start, started)
        _turn(gate, "bash_job", wait, tools.bash_job(**wait))
        assert gate.verifier_passed_at == gate.changes == 3
        _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 3\n"}, "ok")
        tail: dict[str, Any] = {"job_id": wait["job_id"], "action": "tail"}
        _turn(gate, "bash_job", tail, tools.bash_job(**tail))
        assert gate.verifier_passed_at == 3 and gate.changes == 4
        # a job whose own command writes a file after the verifier is no proof: the
        # files it checked are not the files it left
        (tmp_path / "a.py").write_text("x = 1\n")
        _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"}, "ok")
        writing: dict[str, Any] = {
            "command": "bash tmp/verify.sh && printf 'x = 2\\n' > a.py", "description": "v",
            "background": True,
        }
        started = tools.Bash(**writing)
        wait = {"job_id": _job_id(started), "action": "wait", "timeout_seconds": 15}
        _turn(gate, "Bash", writing, started)
        report = tools.bash_job(**wait)
        assert "exited with code 0" in report, report
        _turn(gate, "bash_job", wait, report)
        assert gate.verifier_passed_at is None and gate.changes == 6
        assert gate.verifier_run()[0] == 9
    finally:
        for job in list(tools._jobs):
            tools.bash_job(job, action="kill")


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
            _turn(gate, "Bash", {"command": "python a.py"}, "")
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
    assert review["tokens"] >= 0


def _no_reads(name: str, args: dict[str, Any]) -> Verdict:
    """A caller's ``tool_call_hook`` that refuses every ``Read`` (the guard never sees it)."""
    return refuse("Not now: reads are off in this run.") if name == "Read" else ALLOW


def _run_agent(work_dir: Path, script: list[bytes], max_steps: int = 14,
               hook: Callable[[str, dict[str, Any]], Verdict] | None = None,
               **model_config: Any) -> tuple[SorcarAgent, dict, list[dict]]:
    """Run a real ``SorcarAgent`` in *work_dir* against the scripted *script*.

    Args:
        work_dir: The run's work dir.
        script: The model server's answers, in order.
        max_steps: The run's step cap.
        hook: The caller's ``tool_call_hook``, if any.
        **model_config: Model overrides besides the server's ``base_url`` and ``api_key``.
    """
    with serve(script) as (url, requests):
        agent = SorcarAgent("coding-gate")
        result = agent.run(
            prompt_template=TASK, model_name=MODEL, work_dir=str(work_dir), max_steps=max_steps,
            max_budget=1.0, model_config={"base_url": url, "api_key": "local", **model_config},
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
        finish_body(DONE, prompt_tokens=1000),  # round 1: the verifier fails
        tool_calls_body([("Read", {"file_path": hello}),  # refused by the caller's hook
                         ("Bash", {"command": "printf gate-evidence", "description": "d"})],
                        1000),
        tool_call_body("Bash", {"command": "bash tmp/verify.sh"}, 1000),  # no description
        tool_call_body("finish", with_evidence, 1000),  # round 2: the verifier still fails
        tool_call_body("Bash", {"command": heredoc, "description": "write and check"}, 1000),
        tool_call_body("finish", with_evidence, 1000),  # passes
    ]
    agent, result, requests = _run_agent(tmp_path, script, hook=_no_reads)
    assert result["success"] is True and result["summary"] == DONE
    assert len(requests) == 7
    first = _tool_results(requests, 2)
    assert "Finish refused (round 1 of up to 3" in first and "exited 1" in first
    turn = _tool_results(requests, 3)
    assert "Not now: reads are off" in turn and "gate-evidence\n" in turn
    assert _last_tool_result(requests[4]).startswith("Failed to call Bash")
    second = _last_tool_result(requests[5])
    assert "round 2 of up to 3" in second and "exited 1" in second
    assert "bash tmp/verify.sh`" not in second  # the evidence named refuses nothing
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
    """The configured reviewer reads the diff at the finish, once: its findings refuse
    one round, and the next finish is judged on them without a second review.  The
    review is a side call that keeps the run's ``base_url`` and ``api_key`` (it reaches
    the same local server) but not its ``reasoning_effort``."""
    _pin_reviewer(tmp_path, monkeypatch, MODEL)
    assert cg.gate_reviewer("claude-fable-5") == MODEL
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "verify.sh").write_text("grep -q hello hello.py\n")
    hello = str(tmp_path / "hello.py")
    finding = "1. hello.py: no newline at the end of the file."
    claim = {"success": True, "is_continue": False, "summary_in_html": BARE,
             "evidence": '["bash tmp/verify.sh"]'}
    script = [
        tool_call_body("Write", {"file_path": hello, "content": "print('hello')"}, 1000),
        tool_call_body("Bash", {"command": "bash tmp/verify.sh", "description": "check"}, 1000),
        tool_call_body("finish", claim, 1000),
        text_body(f"VERDICT: NOT DONE\n{finding}"),
        tool_call_body("Edit", {"file_path": hello, "old_string": "')", "new_string": "')\n"},
                       1000),
        tool_call_body("Bash", {"command": "bash tmp/verify.sh", "description": "check"}, 1000),
        tool_call_body("finish", claim, 1000),
    ]
    agent, result, requests = _run_agent(tmp_path, script, reasoning_effort="low")
    assert result["success"] is True and result["summary"].startswith(BARE)
    assert len(requests) == 7
    reviews = [r for r in requests if "You are reviewing, read-only" in str(r["messages"][-1])]
    assert len(reviews) == 1 and reviews[0] is requests[3]
    assert "### hello.py (new file)\nprint('hello')" in reviews[0]["messages"][-1]["content"]
    assert "exit 0: `bash tmp/verify.sh`" in reviews[0]["messages"][-1]["content"]
    assert "reasoning_effort" not in reviews[0]
    assert all(r["reasoning_effort"] == "low" for r in requests if r is not reviews[0])
    refusal = _tool_results(requests, 4)
    assert "Finish refused (round 1 of up to 3" in refusal and finding in refusal
    gate = agent._coding_gate
    assert gate.reviews == 1 and gate.open_findings == [finding] and gate.gate_rounds == 1
    # the second finish went through: the finding is a caveat of the summary where
    # the decisions model judges it unanswered (the summary says nothing of it)
    if decisions_tool_available():
        assert "Delivered after 1 refused finish rounds" in result["summary"]
        assert finding in result["summary"]
        assert gate.delivered_findings == [
            cg.UNANSWERED_VERDICT.format(findings=finding).splitlines()[0]]
    else:
        assert result["summary"] == BARE
    # the review call is in the run's accounting: the executor's six calls report
    # 1,100 tokens each, the reviewer's one 600; the decisions model's judgement of
    # the finding (a real call, when available) is charged to the run too
    if decisions_tool_available():
        assert agent.total_tokens_used > 6 * 1100 + 600
    else:
        assert agent.total_tokens_used == 6 * 1100 + 600
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
    in the worktree), the merge asked for again after the fix is not reviewed again
    (the one review is spent, and a file changed since it), the merge carries the
    verifier back to the checkout, and the finish re-runs it on the merged tree
    without another review.  A task started in
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
        tool_call_body("commit_and_merge", {}, 1000),  # not reviewed again: merged
        tool_call_body("finish", claim, 1000),  # the verifier runs on the merged checkout
    ]
    agent, result, requests = _run_agent(work, script, max_steps=20)
    assert result["success"] is True and result["summary"] == DONE, result
    assert len(requests) == 10
    results = [_last_tool_result(r) for r in requests]
    refused = next(r for r in results if "Merge refused (round 1 of up to 3" in r)
    assert refused.startswith("REFUSED: the commit_and_merge call did NOT run")
    assert "exited 1" in refused and "1. hello.py: prints bye, not hello." in refused
    assert "before calling commit_and_merge again" in refused
    assert cg.MERGE_CAP_NOTE in refused
    merged = next(r for r in results if r.startswith("Successfully merged branch"))
    assert f"Copied tmp/verify.sh to {work / 'tmp' / 'verify.sh'}" in merged
    assert "findings open" not in merged
    reviews = [r for r in requests if "You are reviewing, read-only" in str(r["messages"][-1])]
    assert len(reviews) == 1
    assert cg.MERGE_SUMMARY_PLACEHOLDER in reviews[0]["messages"][-1]["content"]
    # at the merge no finish names evidence: the reviewer is told so, not "(none named)"
    assert cg.MERGE_EVIDENCE_PLACEHOLDER in reviews[0]["messages"][-1]["content"]
    assert "(none named)" not in reviews[0]["messages"][-1]["content"]
    assert f"### {shown} (new file)\nprint('bye')" in reviews[0]["messages"][-1]["content"]
    # the work is on main, the worktree is gone, the verifier survived the merge
    assert (work / "hello.py").read_text() == "print('hello')\n"
    assert kiss_wt_branches(repo) == [] and list((repo / ".kiss-worktrees").iterdir()) == []
    assert run_git(repo, "log", "--format=%s", f"{seed}..HEAD").stdout.split() == ["feat:", "hello"]
    assert (work / "tmp" / "verify.sh").read_text() == "grep -q hello hello.py\n"
    assert not (repo / ".kiss-worktrees" / "tmp").exists()
    gate = agent._coding_gate
    assert gate.work_dir == str(work) and gate.base_commit == seed
    # the merge that went through left its round and findings behind: the finish was
    # gated afresh, on the verifier alone
    assert gate.gate_rounds == 0 and gate.reviews == 1 and gate.open_findings == []
    assert gate.last_verifier_run[0] == 0 and gate.exit_codes == {"bash tmp/verify.sh": "0"}
    assert gate.merge_caveats == [] and gate.delivered_findings == []


def test_a_merge_refused_once_goes_through_next_with_its_findings_unless_the_verifier_fails(
    tmp_path: Path,
) -> None:
    """Audit 2026-11-01 (task 4aa14638): three refused merges spent the round cap, the
    merge then went through, and the finish delivered the merge's stale findings as
    caveats; audit 2026-11-03: two runs ended with their work stranded on a branch
    after the cap.  Now every merge refusal says what happens next
    (:data:`MERGE_CAP_NOTE`); a merge asked for again with the review's findings open
    and nothing changed is not reviewed again (no model call) and, while the verifier
    passes, goes through with the findings in its result and in the finish summary as
    caveats unless the summary answers them.  ``merged()`` resets the rounds and
    findings; so does a review the agent dispatched itself after a later change.  A
    failing verifier refuses the merge again, up to the round cap."""
    finding = "1. README.md: the usage section the task asks for is missing."
    unanswered = cg.MERGE_UNANSWERED_VERDICT.format(findings=finding)
    script = [text_body(f"VERDICT: NOT DONE\n{finding}")]
    hello = tmp_path / "hello.py"
    with serve(script) as (url, requests):
        gate = _gate(tmp_path, verifier="exit 0\n", reviewer=MODEL,
                     model_config={"base_url": url, "api_key": "local"})
        hello.write_text("print('hello')\n")
        _turn(gate, "Write", {"file_path": str(hello), "content": "print('hello')\n"}, "ok")
        _turn(gate, "Bash", {"command": "python hello.py"}, "hello\n")
        first = gate.merge_verdict(TASK)
        assert first is not None and first.startswith("Merge refused (round 1 of up to 3; ")
        assert finding in first and "still open" not in first
        assert first.endswith(cg.MERGE_CAP_NOTE)
        # the second merge: no second review, no refusal; the merge goes through with
        # its findings, which the finish summary then carries
        assert gate.merge_verdict(TASK) is None
        assert len(requests) == 1 and gate.reviews == 1 and not gate.review_possible()
        assert gate.gate_rounds == 2 and gate.merge_rounds == 2
        assert gate.open_findings == [finding] and gate.merge_caveats == [unanswered]
        result = gate.merged()
        assert result.startswith("\n\nMerged after 2 refused rounds with these findings open:")
        assert finding in result and gate.gate_rounds == 0 and gate.open_findings == []
        # the finish after the merge: no file changed, nothing to re-review; the
        # merge's open finding is a caveat of a summary that says nothing of it
        # (without the decisions model, every open finding is)
        args: dict[str, Any] = {"success": True, "summary_in_html": DONE}
        assert gate.finish_verdict(args, TASK) is None
        assert args["summary_in_html"].startswith(DONE + "\n<h4>Merged after 2 refused merge")
        assert finding in args["summary_in_html"]
        assert gate.delivered_findings == [unanswered.splitlines()[0]]
        assert gate.merge_caveats == [] and len(requests) == 1
        # a merge that went through with no finding open appends nothing
        assert gate.merged() == ""
        args = {"success": True, "summary_in_html": DONE}
        assert gate.finish_verdict(args, TASK) is None and args["summary_in_html"] == DONE
        # a summary that answers the finding with a command and its output carries no caveat
        if decisions_tool_available():
            gate.merge_caveats, gate.merge_rounds = [finding], 2
            answered = DONE + ("<p>1. README.md has the usage section: <code>grep -n Usage "
                               "README.md</code> prints <code>3:## Usage</code>.</p>")
            args = {"success": True, "summary_in_html": answered}
            assert gate.finish_verdict(args, TASK) is None
            assert args["summary_in_html"] == answered and gate.merge_caveats == []
    # while the verifier fails, the merge is refused again, with the open findings,
    # up to the round cap; the next merge goes through with the findings (the verifier,
    # at its own refusal cap, does not run again)
    with serve(script) as (url, requests):
        failing = _gate(tmp_path, verifier="exit 3\n", reviewer=MODEL,
                        model_config={"base_url": url, "api_key": "local"})
        _turn(failing, "Write", {"file_path": str(hello), "content": "print('hello')\n"}, "ok")
        for round_number in (1, 2, 3):
            verdict = failing.merge_verdict(TASK)
            assert verdict is not None and f"round {round_number} of up to 3" in verdict
            assert "exited 3" in verdict and finding in verdict
            assert verdict.endswith(cg.MERGE_CAP_NOTE)
            assert ("still open and no file changed since" in verdict) == (round_number > 1)
        assert failing.merge_verdict(TASK) is None and len(requests) == 1
        assert failing.merge_rounds == 4 and failing.verifier_refusals == 3
        assert failing.verifier_skipped and failing.merge_caveats == [unanswered]
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
    redirect into ``tmp/`` are no change of the work: they arm nothing and need no re-review."""
    gate = _gate(tmp_path, verifier="exit 0\n")
    _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"}, "ok")
    _turn(gate, "Bash", {"command": "python a.py > tmp/out.txt 2>&1"}, "")
    assert gate.armed and gate.changes == 1
    for path in ("tmp/PROGRESS.md", "./tmp/c16c/probe.py", str(tmp_path / "tmp" / "verify.sh"),
                 "sub/tmp/notes.md"):
        _turn(gate, "Edit", {"file_path": path, "old_string": "a", "new_string": "b"}, "ok")
        _turn(gate, "Write", {"file_path": path, "content": "x"}, "ok")
    assert gate.changes == 1 and gate.written_files == [str(tmp_path / "a.py")]
    done = {"success": True, "summary_in_html": DONE}
    assert gate.finish_verdict(dict(done), TASK) is None
    # a code file that merely sits next to a tmp name, or outside the work dir, is work
    for path in ("tmpl/a.py", "a.tmp", "tmp", str(tmp_path.parent / "tmp" / "x.py")):
        assert not gate.is_scratch(path), path
    _turn(gate, "Write", {"file_path": "tmpl/a.py", "content": "x"}, "ok")
    assert gate.changes == 2
    # a probe-only run (scratch python, no product code) is not a coding run
    (tmp_path / "other").mkdir()
    probe = _gate(tmp_path / "other")
    probe.observe("Write", {"file_path": "tmp/probe.py", "content": "print(1)\n"})
    assert not probe.armed and not probe.coding_run()


def test_a_delegated_change_is_judged_by_the_diff(tmp_path: Path) -> None:
    """A sub-task that may change files makes the run a coding run when the diff
    shows a code file changed."""
    from kiss.tests.server.conftest import init_repo

    repo = tmp_path / "repo"
    init_repo(repo)
    gate = _gate(repo)
    done = {"success": True, "summary_in_html": DONE}
    implement = {"task": "Implement hello.py so it prints hello"}
    # a child that wrote nothing (audit 2026-11-03: a sub-agent that answered a
    # question) is no change: the work's fingerprint is the same before and after
    _turn(gate, "run_agent", implement, "ran: sorcar\nsuccess: true\nsummary: nothing to do\n")
    assert not gate.delegated and not gate.coding_run() and gate.changes == 0
    assert gate.finish_verdict(dict(done), TASK) is None
    # one sent to another machine touches no file here
    _turn(gate, "run_agent", {**implement, "machine": "gpu-box"},
          "ran: sorcar\nsuccess: true\nsummary: written\n")
    assert not gate.delegated and gate.changes == 0
    gate.observe("run_agent", implement)
    (repo / "notes.md").write_text("done\n")  # the child wrote prose only: not gated
    gate.result("run_agent", implement, "ran: sorcar\nsuccess: true\nsummary: written\n", False)
    assert gate.delegated and not gate.coding_run() and gate.changes == 1
    assert gate.finish_verdict(dict(done), TASK) is None
    (repo / "hello.py").write_text("print('hello')\n")  # the child changed code
    assert gate.coding_run() and gate.changed_paths() == [("hello.py", True), ("notes.md", True)]
    assert gate.finish_verdict(dict(done), TASK) is None  # no verifier, no reviewer: nothing fails
    _turn(gate, "Bash", {"command": "python hello.py"}, "hello\n")
    # audit 2026-11-04 (S8): a child that renamed an untracked file, its contents kept,
    # changed the work too (the fingerprint lists the untracked paths, not only their
    # contents): the verifier that passed before it is not reused
    (repo / "tmp").mkdir()
    (repo / "tmp" / "verify.sh").write_text("test -f hello.py || exit 9\n")
    _turn(gate, "Bash", {"command": "bash tmp/verify.sh"}, _host_result("bash tmp/verify.sh", repo))
    assert gate.verifier_passed_at == gate.changes == 1
    gate.observe("run_agent", implement)
    (repo / "hello.py").rename(repo / "hi.py")
    gate.result("run_agent", implement, "ran: sorcar\nsuccess: true\nsummary: renamed\n", False)
    assert gate.changes == 2 and gate.verifier_passed_at == 1
    assert gate.verifier_run()[0] == 9


def test_a_rollback_is_a_change_of_the_work(tmp_path: Path) -> None:
    """Audit 2026-11-05 (A05-5): a ``rollback`` rewrites files as an ``Edit`` would, so the
    verifier that passed before it is not reused at the finish, and the diff decides the
    coding run; one that restored nothing (the files already match the checkpoint, an
    unknown checkpoint) changes nothing."""
    from kiss.tests.server.conftest import init_repo

    repo = tmp_path / "repo"
    init_repo(repo)
    gate = _gate(repo, verifier="grep -q 'x = 1' a.py || exit 9\n")
    tools = UsefulTools(work_dir=str(repo))
    done = {"success": True, "summary_in_html": DONE}
    (repo / "a.py").write_text("x = 0\n")
    _turn(gate, "Write", {"file_path": str(repo / "a.py"), "content": "x = 0\n"}, "ok")
    _turn(gate, "checkpoint", {"name": "bad"}, tools.checkpoint("bad"))
    (repo / "a.py").write_text("x = 1\n")
    _turn(gate, "Write", {"file_path": str(repo / "a.py"), "content": "x = 1\n"}, "ok")
    _turn(gate, "Bash", {"command": "bash tmp/verify.sh"}, _host_result("bash tmp/verify.sh", repo))
    assert gate.verifier_passed_at == gate.changes == 2
    gate.observe("rollback", {"name": "bad"})
    restored = tools.rollback("bad")
    assert restored.startswith("Rolled ") and (repo / "a.py").read_text() == "x = 0\n", restored
    gate.result("rollback", {"name": "bad"}, restored, False)
    assert gate.changes == 3 and gate.verifier_passed_at == 2
    verdict = gate.finish_verdict(dict(done), TASK)
    assert verdict is not None and "exited 9" in verdict, verdict
    for name in ("bad", "nope"):  # the files already match; an unknown checkpoint
        _turn(gate, "rollback", {"name": name}, tools.rollback(name))
    assert gate.changes == 3
    # a run that never named a code file (a tool the gate does not read wrote
    # and deleted one) is armed by what a rollback restored: the diff decides
    other = tmp_path / "other"
    init_repo(other)
    gate, tools = _gate(other), UsefulTools(work_dir=str(other))
    _turn(gate, "Write", {"file_path": "notes.md", "content": "n"}, "ok")
    (other / "a.py").write_text("x = 1\n")
    assert tools.checkpoint("one").startswith("Saved ")
    (other / "a.py").unlink()
    assert not gate.coding_run()
    gate.observe("rollback", {"name": "one"})
    gate.result("rollback", {"name": "one"}, tools.rollback("one"), False)
    assert (other / "a.py").exists() and gate.changes == 2 and gate.coding_run()


def _stalling(on_request: Callable[[], None], release: threading.Event) -> type[ScriptedHandler]:
    """A model server that records each request, calls *on_request*, and never answers.

    The handler holds the connection until *release* is set (the test sets
    it when done), so the client sees a reviewer that does not answer.

    Args:
        on_request: Called when a request has arrived (a stop to raise).
        release: Lets the parked handlers return.
    """

    class Stalling(ScriptedHandler):
        def do_POST(self) -> None:  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with self.lock:
                self.requests.append(json.loads(raw))
            on_request()
            release.wait(30)

    return Stalling


def test_a_stop_during_the_review_still_propagates_and_charges_the_attempt(
    tmp_path: Path,
) -> None:
    """The run's Stop ends the review (the side call's deadline event fires on the
    thread's own stop event, and the model client gives the request up on it), the
    stop propagates out of the gate, and the attempt is charged as it ends (review of
    2026-11-01, finding 2: a stop used to skip the charge), once per key however often
    the stop lands in it; the thread's stop event is the run's again afterwards."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "hello.py").write_text("print('hello')\n")
    charged: dict[str, list[tuple[float, int, int]]] = {}

    def charge(budget: float, tokens: int, steps: int, key: str) -> None:
        charged.setdefault(key, []).append((budget, tokens, steps))

    run_stop, release = threading.Event(), threading.Event()
    outer = stop_signal.get_thread_stop_event()
    stop_signal.set_thread_stop_event(run_stop)
    try:
        handler = _stalling(run_stop.set, release)
        with serve([text_body("VERDICT: OK")], handler_base=handler) as (url, requests):
            gate = _gate(tmp_path, verifier="grep -q . hello.py\n", reviewer=MODEL,
                         charge=charge, model_config={"base_url": url, "api_key": "local"})
            with pytest.raises(KeyboardInterrupt):
                gate.review_verdict({"success": True, "summary_in_html": DONE}, TASK)
            assert len(requests) == 1 and run_stop.is_set()
        assert stop_signal.get_thread_stop_event() is run_stop
    finally:
        release.set()
        stop_signal.set_thread_stop_event(outer)
    # the attempt was charged under one key: its model call counts, no reply was billed
    assert list(charged.values()) == [[(0.0, 0, 1), (0.0, 0, 1)]]
    assert gate.reviews == 0 and gate.failed_reviews == 0


def test_a_review_past_the_deadline_refuses_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reviewer that does not answer within :data:`REVIEW_DEADLINE` is given up: the
    side call reports the deadline, the finish goes through without a review, the
    failed review counts under :data:`MAX_FAILED_REVIEWS`, and the stage record says why."""
    monkeypatch.setattr(cg, "REVIEW_DEADLINE", 1)
    release = threading.Event()
    charged: list[tuple[float, int, int]] = []
    printer = JsonPrinter()
    printer.adopt_thread_binding("task-9")
    printer.start_recording()
    previous = bind_printer(printer)
    try:
        handler = _stalling(lambda: None, release)
        with serve([text_body("VERDICT: NOT DONE\n1. never answered")],
                   handler_base=handler) as (url, requests):
            gate = _gate(tmp_path, verifier="exit 0\n", reviewer=MODEL,
                         charge=functools.partial(_record_charge, charged),
                         model_config={"base_url": url, "api_key": "local"})
            assert gate.side_call("Finish reviewer", "A question.", "probe") == (
                "", "no answer within 1 s", 0.0, 0)
            _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"},
                  "ok")
            started = time.monotonic()
            args = {"success": True, "summary_in_html": DONE}
            assert gate.finish_verdict(args, TASK) is None
            assert time.monotonic() - started < 20 and len(requests) == 2
            assert args["summary_in_html"] == DONE and gate.gate_rounds == 0
            assert (gate.reviews, gate.failed_reviews) == (0, 1) and gate.review_possible()
            assert gate.open_findings == [] and gate.changed_since_review
    finally:
        release.set()
        bind_printer(previous)
        printer.adopt_thread_binding("")
    assert charged == [(0.0, 0, 1), (0.0, 0, 1)]  # each attempt's model call is charged
    recorded = [e for e in printer.peek_recording_for_task("task-9") if e["type"] == "gate_stage"]
    review = next(e for e in recorded if e["stage"] == "review")
    assert review["outcome"] == "error" and review["error"] == "no answer within 1 s"
    assert review["round"] == 0 and review["findings"] == 0 and review["seconds"] >= 1


def test_the_ledgers_first_write_gets_the_independent_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the agent first writes ``ledger.md`` in the scratch directory, the reviewer
    model reads the task and the ledger once (:data:`READING_PROMPT`) and the sentences
    it would read differently join the ``Write``'s result (:data:`READING_NOTE`,
    through the guard's allowing verdict and ``KISSAgent``); a second ``Write`` of
    the ledger makes no call, and a ``NONE`` answer adds nothing.  The call is the
    ``reading`` gate stage."""
    _pin_reviewer(tmp_path, monkeypatch, MODEL)
    ledger = {"file_path": "tmp/ledger.md", "content": "# Ledger\n- 'prints hello': stdout.\n"}
    items = ("1. \"prints hello\": a line, so a trailing newline is expected.\n"
             "2. \"hello.py\": at the work dir's root, not under src/.")
    script = [
        tool_call_body("Write", ledger, 1000),
        text_body(items),  # the reading of the task against the ledger
        tool_call_body("Write", dict(ledger, content=ledger["content"] + "- settled.\n"), 1000),
        tool_call_body("Write", {"file_path": "hello.py", "content": "print('hello')\n"}, 1000),
        finish_body(DONE, prompt_tokens=1000),
        text_body("VERDICT: OK"),
    ]
    agent, result, requests = _run_agent(tmp_path, script)
    assert result["success"] is True and result["summary"] == DONE
    assert len(requests) == 6
    reading = requests[1]["messages"][-1]["content"]
    assert reading.startswith("An engineer is about to work on the task below")
    assert f"# Task\n{TASK}\n" in reading and ledger["content"] in reading
    first = _last_tool_result(requests[2])  # the note, then the agent's step footer
    assert first.startswith("Successfully wrote 35 characters to tmp/ledger.md"
                            + cg.READING_NOTE.format(model=MODEL, items=items))
    second = _last_tool_result(requests[3])
    assert second.startswith("Successfully wrote ") and "Independent reading" not in second
    assert (tmp_path / "tmp" / "ledger.md").read_text().endswith("- settled.\n")
    gate = agent._coding_gate
    assert gate.reading_done and gate.reviews == 1 and gate.gate_rounds == 0
    assert agent.total_tokens_used == 4 * 1100 + 2 * 600
    # a NONE answer adds nothing, and the stage records the reading; other writes, a
    # ledger outside the scratch directory, or no reviewer make no call
    printer = JsonPrinter()
    printer.adopt_thread_binding("task-9")
    printer.start_recording()
    previous = bind_printer(printer)
    try:
        with serve([text_body("NONE")]) as (url, requests):
            local = {"base_url": url, "api_key": "local"}
            gate = _gate(tmp_path, reviewer=MODEL, model_config=local)
            for name, args in (("Edit", ledger), ("Write", dict(ledger, file_path="ledger.md")),
                               ("Write", {"file_path": "tmp/notes.md", "content": "x"})):
                assert gate.reading_note(name, args, TASK) is None and not gate.reading_done
            assert _gate(tmp_path).reading_note("Write", ledger, TASK) is None
            assert gate.reading_note("Write", ledger, "") is None and requests == []
            assert gate.reading_note("Write", ledger, TASK) is None
            assert gate.reading_done and len(requests) == 1
            assert gate.reading_note("Write", ledger, TASK) is None and len(requests) == 1
    finally:
        bind_printer(previous)
        printer.adopt_thread_binding("")
    recorded = [e for e in printer.peek_recording_for_task("task-9") if e["type"] == "gate_stage"]
    assert [(e["stage"], e["outcome"], e["items"], e["model"]) for e in recorded] == [
        ("reading", "pass", 0, MODEL)]


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
