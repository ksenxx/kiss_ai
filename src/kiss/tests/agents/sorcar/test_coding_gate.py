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
import shutil
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
from kiss.agents.sorcar.agent_dispatch import (
    digest_agent_results,
    job_notice,
    new_agent_job,
    wait_any_agent_job,
)
from kiss.agents.sorcar.coding_gate import CodingGate, run_on_host
from kiss.agents.sorcar.container_jobs import JOBS_DIR, ContainerJobs
from kiss.agents.sorcar.decide_tool import decisions_tool_available
from kiss.agents.sorcar.shell_scan import MAX_NESTING, bash_path, simple_commands
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import UsefulTools, digest_parallel_report
from kiss.core import stop_signal, user_config
from kiss.core.event_sink import bind_printer
from kiss.core.models.model_info import MODEL_INFO
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
from kiss.tests.conftest import IS_WINDOWS

TASK = "Write hello.py that prints hello."
#: A summary that reports the outcome and its check.
DONE = "<p>hello.py written and prints hello; checked by bash tmp/verify.sh (exit 0).</p>"
#: A summary that answers no review finding: the decisions model, live in the
#: test environment, finds every finding unanswered by it.
BARE = "<p>Done.</p>"


def _shell(work_dir: Path, command: str, timeout: int) -> tuple[int | None, str]:
    """The gate's shell for the tests: a bash subprocess in *work_dir*."""
    return run_on_host(command, str(work_dir), timeout)


def _outside_git_shell(work_dir: Path, command: str, timeout: int) -> tuple[int | None, str]:
    """Run *command* without letting Git discover a repository above *work_dir*."""
    ceiling = shlex.quote(str(work_dir))
    return run_on_host(
        f"export GIT_CEILING_DIRECTORIES={ceiling}; {command}", str(work_dir), timeout,
    )


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


def _run(gate: CodingGate, work_dir: Path, command: str) -> None:
    """A foreground ``Bash`` call in the agent's order: observed, then run, then its result read."""
    gate.observe("Bash", {"command": command})
    gate.result("Bash", {"command": command}, _host_result(command, work_dir), False)


def _call(gate: CodingGate, tool: Any, args: dict[str, Any]) -> str:
    """One real tool call in the agent's order: observed, made, its result read by the gate."""
    gate.observe(tool.__name__, args)
    text: str = tool(**args)
    gate.result(tool.__name__, args, text, False)
    return text


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
    assert cg.result_exit_codes("Bash", bash, "F\n[exit code: 2]", jobs, container=True) == [
        ("pytest -q", "2")]
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
    # only the tools' error shapes open an error: a passing command whose
    # output starts with the word is a pass (audit 2026-11-07, A07-1)
    for passing in ("Errors: 0, Failures: 0\n3 passed", "Error-free build\n",
                    "Failed to call count: 0\n"):
        assert cg.result_exit_codes("Bash", bash, passing, jobs) == [("pytest -q", "0")]
    # a call the user stopped did not run to completion: not exit 0 (one tool or several)
    assert cg.result_exit_codes("Bash", bash, USER_INTERRUPTED_MESSAGE, jobs) == [
        ("pytest -q", "interrupted")]
    assert cg.result_exit_codes("run_commands_parallel", {"commands": '["a", "b"]'},
                                USER_INTERRUPTED_MESSAGE, jobs) == [
        ("a", "interrupted"), ("b", "interrupted")]
    # a background job, then its exit through bash_job (host wording), only from the status line
    started = "Started background job j1 (pid 7).\nlog: /tmp/j1.log"
    make = {"command": "make", "background": True}
    assert cg.result_exit_codes("Bash", make, started, jobs) == [("make", "background")]
    # a notice opens the result of a call that can return it (the follow-up's, for a
    # command made as an Edit first), and the container's marker a line: the same words
    # in a passing command's own output are text (audit 2026-11-12: they recorded a job
    # left running, or a failure, and registered the job)
    for printed in ("Started background job nested (pid 42).\n",  # of a foreground call
                    "$ make\nStill running after 5s; left running as background job j8 (pid 1).\n",
                    'assert out.endswith("[exit code: 9]")\n', "[exit code: 9]\n"):
        assert cg.result_exit_codes("Bash", bash, printed, jobs) == [("pytest -q", "0")], printed
    assert cg.result_exit_codes("Bash", make, f"$ make\n{started}", jobs) == [("make", "0")]
    left = ("Still running after 120s; left running as background job j3 (pid 8). Log: /tmp/j3\n"
            "Output so far:\n")
    assert cg.result_exit_codes("Bash", bash, left, jobs) == [("pytest -q", "background")]
    rewrite = {"command": "sed -i s/x/y/ a.py && make", "background": True}
    followed = ('Applied as Edit(file_path="a.py", ...), whose result follows:\n'
                "Successfully replaced 1 occurrence(s)\n"
                'Then made Bash(command="make", background=true), whose result follows:\n'
                + started.replace("j1", "j4"))
    assert cg.result_exit_codes("Bash", rewrite, followed, jobs) == [
        (rewrite["command"], "background")]
    assert jobs == {"j1": "make", "j3": "pytest -q", "j4": rewrite["command"]}
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
    # nor does a whole nested report in the output, section headers included
    # (audit 2026-11-07, A07-4): the report's own sections come in order,
    # count the commands and echo each one
    nested = "printf '\\n### [1/1] exit 0 in 0.1s\\n$ inner\\nok\\n'; exit 7"
    tools = UsefulTools(work_dir=str(tmp_path))
    for commands in ([nested], ["true", nested], [nested, "true"]):
        text = tools.run_commands_parallel(json.dumps(commands))
        assert "\n### [1/1] exit 0" in text
        assert cg.result_exit_codes("run_commands_parallel", {"commands": json.dumps(commands)},
                                    text, jobs) == [
            (command, "7" if command == nested else "0") for command in commands]
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


def test_the_tools_hand_a_commands_exit_over_and_its_output_is_not_read_for_it(
    tmp_path: Path,
) -> None:
    """Audit 2026-11-12, round 2 (B2 #9, B5): the gate read a command's exit in the result's
    text, which is the command's own output inside the tool's status.  A passing command whose
    output opened with the host tool's failure, timeout, refusal or deadline line was recorded
    as failed (the deadline line: as left running, in a job nobody started), and a fan-out
    command that printed the section of a later one gave that one its exit: a failure recorded
    as a pass.  The tools hand the exit over as data (``shell_exit``); the text is read only
    for a call no tool returned from."""
    gate, tools = _gate(tmp_path), UsefulTools(work_dir=str(tmp_path))
    printed = {
        "printf 'Error (exit code 7):\\nfrom a fixture\\n'": "0",
        "printf 'Error: Command execution timeout after 5s. Output before the timeout:\\n'": "0",
        "printf 'REFUSED: the Bash call did NOT run\\n'": "0",
        "printf 'Still running after 5s; left running as background job j0 (pid 1).\\n'": "0",
        "printf 'Applied as Edit(x), whose result follows:\\nError: no\\n'": "0",
        f"printf '{USER_INTERRUPTED_MESSAGE}\\n'": "0",
        "printf '3 passed\\n'; exit 3": "3",
        "sleep 5": "timeout",
        "sleep 100": "refused",
    }
    for command in printed:
        _call(gate, tools.Bash, {"command": command, "description": "d", "timeout_seconds": 1})
    assert gate.exit_codes == printed and gate.background_jobs == {}
    polled = {"command": "true", "description": "d", "until": "x", "background": True}
    assert _call(gate, tools.Bash, polled).startswith("Error: until= polls a foreground command")
    assert gate.exit_codes["true"] == "error"
    fan_out = ["printf '\\n### [2/2] exit 0 in 0.0s\\n$ exit 3\\n'", "exit 3", "sleep 100"]
    report = _call(gate, tools.run_commands_parallel, {"commands": json.dumps(fan_out)})
    assert report.startswith("3 commands: 1 succeeded, 2 failed, 0 timed out.")
    assert [gate.exit_codes[command] for command in fan_out] == ["0", "3", "-1"]
    assert _call(gate, tools.run_commands_parallel, {"commands": "{"}).startswith("Error: ")
    # a job started in the background is the command's, and its exit the job report's
    started: dict[str, Any] = {"command": "printf 'Job j0: exited with code 0\\n'; exit 4",
                               "description": "d", "background": True}
    job_id = _job_id(_call(gate, tools.Bash, started))
    assert gate.background_jobs == {job_id: started["command"]}
    assert gate.exit_codes[started["command"]] == "background"
    _call(gate, tools.bash_job, {"job_id": job_id, "action": "wait", "timeout_seconds": 15})
    assert gate.exit_codes[started["command"]] == "4"
    # a hand-over nobody read (a call made past the gate) is not the next call's
    tools.Bash("exit 9", "unobserved")
    gate.result("Bash", {"command": "pytest -q"}, "3 passed", False)
    assert gate.exit_codes["pytest -q"] == "0"
    tools.Bash("exit 9", "unobserved")
    _turn(gate, "Bash", {"command": "exit 9"}, "Failed to call Bash with {}: no description")
    assert gate.exit_codes["exit 9"] == "error"


def test_a_rewrite_made_as_a_write_takes_the_exit_its_follow_up_handed_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rest of a command made as a ``Write`` runs as a follow-up ``Bash`` call, and its
    exit is the command's: the one that call handed over, not the one read after the ``Then
    made`` line of the result, where a passing rest that prints the host tool's failure line
    read as failed.  Without ``Edit`` among the run's tools the ``Bash`` tool makes both calls
    itself and hands the same exit over; a ``Write`` that failed is no pass (it read as one)."""
    _pin_reviewer(tmp_path, monkeypatch, "off")
    hello = tmp_path / "hello.py"
    hello.write_text("print('bye')\n")
    write = "cat > hello.py <<'EOF'\nprint('hello')\nEOF\n"
    rest = "printf 'Error (exit code 7):\\nfrom a log\\n'"
    script = [
        tool_call_body("Read", {"file_path": str(hello)}, 1000),
        tool_call_body("Bash", {"command": write + rest, "description": "d"}, 1000),
        tool_call_body("Bash", {"command": write + "exit 3", "description": "d"}, 1000),
        finish_body(DONE, prompt_tokens=1000),
    ]
    agent, _, requests = _run_agent(tmp_path, script)
    assert _last_tool_result(requests[2]).startswith("Applied as Write(")
    assert agent._coding_gate.exit_codes == {write + rest: "0", rest: "0", write + "exit 3": "3",
                                             "exit 3": "3"}
    gate, tools = _gate(tmp_path), UsefulTools(work_dir=str(tmp_path))
    tools.Read(str(hello))
    for command, code in ((write + rest, "0"), (write + "exit 3", "3")):
        text = _call(gate, tools.Bash, {"command": command, "description": "d"})
        assert text.startswith("Applied as Write(") and gate.exit_codes[command] == code, text
    hello.unlink()
    hello.mkdir()  # the Write the here-document stands for fails: nothing ran
    text = _call(gate, tools.Bash, {"command": write + rest, "description": "d"})
    assert text.startswith("Not applied: ") and gate.exit_codes[write + rest] == "error", text


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
    # nor does one that does not finish in time
    slow = _gate(tmp_path, verifier="sleep 30\n", verifier_timeout=1)
    giveup = {"success": False, "summary_in_html": "<p>No.</p>"}
    assert slow.finish_verdict(giveup, TASK) is None
    assert giveup["summary_in_html"].endswith("verifier ./tmp/verify.sh: timeout.</p>")
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
        # a benefit the task claims is motivation, not a requirement to prove
        assert "is the user's motivation, not a requirement" in " ".join(prompt.split())
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
        # a wait_any result is read job by job: the finished reviewer among the jobs
        # it lists counts, a job of another run or a failed one does not
        _turn(own, "run_agent", {**review, "wait": "false"},
              "Started the sorcar agent task as job agent-2b3c; its tab is open.")
        _turn(own, "run_agent", {**review, "wait": "false"},
              "Started the sorcar agent task as job agent-3c4d; its tab is open.")
        _turn(own, "agent_job", {"job_id": "", "action": "wait_any"},
              "Finished: agent-9, agent-3c4d; running: agent-2b3c.\n\n"
              "Job agent-9 (sorcar agent task) finished:\n"
              "ran: sorcar tools=review\nsuccess: true\nsummary: ok\n\n"
              "Job agent-3c4d (sorcar agent task) finished:\n"
              "ran: sorcar tools=review\nsuccess: false\nsummary: 'Task failed with error: x'\n")
        assert not own.reviewed_by_agent and own.review_wanted()
        _turn(own, "agent_job", {"job_id": "", "action": "wait_any"},
              "None of agent-2b3c finished within 5s; they are still running.")
        assert not own.reviewed_by_agent
        _turn(own, "agent_job", {"job_id": "agent-2b3c", "action": "wait_any"},
              "Finished: agent-2b3c; running: none.\n\n"
              "Job agent-2b3c (sorcar agent task) finished:\n"
              "ran: sorcar tools=review\nsuccess: true\nsummary: ok\n")
        assert own.reviewed_by_agent and not own.review_wanted()
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
    gate.shell = functools.partial(_outside_git_shell, before)
    assert gate.verifier_path == str(before / "tmp" / "verify.sh")
    assert gate.verifier_shown == "./tmp/verify.sh" and not gate.verifier_exists()
    # the agent moves into its worktree: the gate's shell and work dir follow
    gate.work_dir, gate.shell = str(after), functools.partial(_outside_git_shell, after)
    (after / "tmp").mkdir()
    (after / "tmp" / "verify.sh").write_text("echo 'not yet'; exit 7\n")
    assert gate.verifier_path == str(after / "tmp" / "verify.sh") and gate.verifier_exists()
    verdict = gate.finish_verdict({"success": True, "summary_in_html": DONE}, TASK)
    assert verdict is not None and "exited 7" in verdict and "not yet" in verdict
    # audit 2026-11-04 (S8): the last run of the verifier is a run on the tree of the
    # moment: it stands in another tree (the merged checkout, whose files may differ)
    # only when a merge that went through finds there the files that run saw
    assert gate.verifier_memo is not None
    gate.move(str(after))
    memo = gate.verifier_memo
    assert memo is not None and memo[0] == str(after)
    gate.move(str(before))  # the key names the tree it ran on: kept, not reused here
    assert gate.verifier_memo == memo and gate.work_dir == str(before)
    gate.move(str(after))
    assert gate.verifier_run()[0] == 7
    # outside git there is no fingerprint to compare: a merge carries nothing
    assert gate.verifier_memo is not None and gate.merged() == ""
    assert gate.verifier_memo is None
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
    # so do a copy of one into a directory and one ``--`` protects (audit 2026-11-12:
    # the first listed only the directory, the second no file at all)
    for command in ("cp tmp/source.py src/", "cp tmp/source.py src", "rm -- -old.py"):
        assert _arms(tmp_path, {"command": command}), command
    assert not _arms(tmp_path, {"command": "cp notes.md docs/"})


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
    # a new file under a tmp directory is scratch (a probe, a sub-task's notes), not
    # the run's work, in a repository that does not ignore tmp/ too: it was reviewed
    # as a new file, and a sub-task's probe there made the run a coding run
    (repo / "tmp" / "sub").mkdir(parents=True)
    (repo / "tmp" / "sub" / "probe.py").write_text("p = 1\n")
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


def test_a_command_nested_past_the_scanners_limit_is_read_not_raised(tmp_path: Path) -> None:
    """A command nested deeper than ``MAX_NESTING`` made the gate's two strict scans raise
    ``TooDeepError``: out of ``observe`` (:func:`moves_head`), which the agent's guard calls
    unprotected, and out of ``result`` (:func:`verifier_decides`), which lost the call's
    ledger entry.  The gate reads such a command as far as the nesting, as its scan of the
    files changed does, and its exit shows nothing about a verifier run in it."""
    deep = "echo " + "$(echo " * (MAX_NESTING + 1) + "x" + ")" * (MAX_NESTING + 1)
    gate = _gate(tmp_path, verifier="exit 0\n")
    _turn(gate, "Bash", {"command": deep}, "x\n")
    assert gate.exit_codes == {deep: "0"} and not gate.moved_head
    command = f"git checkout -q main; bash tmp/verify.sh && {deep}"
    _turn(gate, "Bash", {"command": command}, "x\n")
    assert gate.moved_head and gate.exit_codes[command] == "0"
    assert gate.last_verifier_run == (None, "")


def test_the_agents_verifier_run_is_shown_only_when_its_exit_decided_the_commands(
    tmp_path: Path,
) -> None:
    """The agent's own passing run of the verifier is what a review is shown as its last
    output, but only a run whose exit status the command's shows: ``bash -n``, ``|| true``,
    ``| cat``, ``true ||``, ``!``, a command after it, ``&`` or an ``exit`` / ``exec``
    before it hide a failing verifier.  Audit 2026-11-04 (S8): so do a shell option that
    reads the script from elsewhere (``bash -s``: stdin; ``--rcfile``: an option's value)
    or runs nothing (``-D``), and ``set -n`` / ``set -o noexec`` before it.  Finish review
    2026-11-04: ``--`` (or ``-``) ends the shell's options, so ``bash -- -x tmp/verify.sh``
    runs a file named ``-x`` and only passes the verifier to it as an argument.  The
    readers are the module functions ``verifier_command`` and ``verifier_decides`` (the
    coding SEA reads its own commands with them), which the gate's methods call.
    Audit 2026-11-10 (B/S2-F1): a program one of the agent's commands ran may have
    changed a file no scan of the command shows.  So the run is taken for the gate's own
    last run (reused at the finish, :func:`coding_gate.verifier_ends`) only when the
    command is the call's only one, nothing runs after the verifier, no write is
    scanned and the work and the script are as they were before the call; otherwise
    the gate runs the verifier itself at the finish."""
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("tmp/\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "start")
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
        assert gate.last_verifier_run[0] != 0, command
        assert not cg.verifier_decides(command, work, verifier), command
        found = cg.verifier_command(simple_commands(command, bash_path(work)), verifier)
        assert (found is None) == (command in not_run), command
    code, output = gate.verifier_run()
    assert code == 7 and "FAILURE" in output
    # a run whose exit is the verifier's is shown; one that ends the command is the
    # gate's run too, the others the gate runs again (the script counts its runs)
    script, runs = tmp_path / "tmp" / "verify.sh", tmp_path / "tmp" / "runs"
    script.write_text(f"echo r >> {shlex.quote(str(runs))}; echo RUN; exit 0\n")
    (tmp_path / "sub").mkdir()
    run_again = ("bash tmp/verify.sh && echo VERIFIED", "true; bash tmp/verify.sh && echo ok",
                 "bash tmp/verify.sh && echo done | tee tmp/log && (echo a; echo b)",
                 "bash tmp/verify.sh && echo $(date) `true`")
    for command in (f"cd {shlex.quote(str(tmp_path))} && bash tmp/verify.sh",
                    "timeout 60 bash -x ./tmp/verify.sh 2>&1", "bash tmp/verify.sh > tmp/log",
                    # through a directory and back (audit 2026-11-10 S10-F6, review round 1)
                    "cd sub && bash ../tmp/verify.sh", "bash sub/../tmp/verify.sh",
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
                    "set -n +n; bash tmp/verify.sh", "set -e +o noexec; bash tmp/verify.sh",
                    *run_again):
        assert cg.verifier_decides(command, work, verifier), command
        runs.unlink(missing_ok=True)
        _run(gate, tmp_path, command)
        agents = gate.last_verifier_run
        assert agents[0] == 0 and ("RUN" in agents[1]) == ("> tmp/log" not in command), command
        assert (gate.verifier_memo is None) == (command in run_again), command
        reused = gate.verifier_run()
        assert reused == (agents if command not in run_again else (0, "RUN\n")), command
        assert runs.read_text() == "r\n" * (2 if command in run_again else 1), command
        gate.last_verifier_run, gate.verifier_memo = (None, ""), None  # as before any run
    # the agent's passing run is not the gate's when its call held other commands, wrote
    # a file, or changed the work some other way (a program wrote a file no scan shows)
    (tmp_path / "rewrite.sh").write_text("echo changed >> work.txt\n")
    calls: list[tuple[str, dict[str, Any]]] = [
        ("run_commands_parallel", {"commands": json.dumps(["bash tmp/verify.sh", "true"])}),
        ("Bash", {"command": "echo x > b.txt && bash tmp/verify.sh"}),
        ("Bash", {"command": "bash rewrite.sh; bash tmp/verify.sh"})]
    tools = UsefulTools(work_dir=str(tmp_path))
    for tool, call in calls:
        gate.observe(tool, call)
        if tool == "Bash":
            result = _host_result(call["command"], tmp_path)
        else:
            result = tools.run_commands_parallel(call["commands"])
        gate.result(tool, call, result, False)
        assert gate.exit_codes[cg.shell_commands(call)[0]] == "0" and gate.verifier_memo is None
    for made in ("work.txt", "rewrite.sh", "b.txt"):
        (tmp_path / made).unlink()
    # a command naming the verifier makes the gate forget its own last run of it
    assert gate.verifier_run()[0] == 0 and gate.verifier_memo is not None
    _turn(gate, "Bash", {"command": "chmod +x tmp/verify.sh"}, "")
    assert gate.verifier_memo is None
    # what leaves the change count as it was is seen by that run of the gate's: a file
    # rewritten by a program a later call runs (the fingerprint of the work differs) or,
    # git-ignored or not, by the same call after the verifier (no scan of the command
    # shows the write).  A git-ignored file a later call rewrites is outside the
    # fingerprint, for a run of the gate's as for a clean run of the agent's.
    for checked in ("work.txt", "tmp/work.txt"):
        script.write_text(f"grep -qx good {checked}\n")
        (tmp_path / "rewrite.sh").write_text(f"echo bad > {checked}\n")
        later = (("bash tmp/verify.sh", "bash rewrite.sh"),) if checked == "work.txt" else ()
        for commands in (*later, ("bash tmp/verify.sh && bash rewrite.sh",)):
            (tmp_path / checked).write_text("good\n")
            changes = gate.changes
            for command in commands:
                _run(gate, tmp_path, command)
            assert gate.changes == changes and gate.verifier_run()[0] == 1, (checked, commands)
    if not IS_WINDOWS:
        # the agent's run through a symbolic link, whose command does not name the
        # verifier, runs the same script on the same tree: an unchanged tree reuses it
        # as it reuses the gate's last run
        (tmp_path / "tmp" / "check.sh").symlink_to("verify.sh")
        script.write_text(f"echo r >> {shlex.quote(str(runs))}; exit 0\n")
        runs.unlink(missing_ok=True)
        assert gate.verifier_run()[0] == 0 and gate.verifier_run()[0] == 0
        assert runs.read_text() == "r\n"
        _run(gate, tmp_path, "bash tmp/check.sh")
        assert gate.verifier_run()[0] == 0 and runs.read_text() == "r\n" * 2
        # and the verifier rewritten through the link is run as rewritten
        changes = gate.changes
        _turn(gate, "Write", {"file_path": "tmp/check.sh", "content": "exit 7\n"}, "ok")
        (tmp_path / "tmp" / "check.sh").write_text("exit 7\n")
        assert gate.changes == changes and gate.verifier_run()[0] == 7


@pytest.mark.skipif(IS_WINDOWS, reason="a symbolic link to a directory")
def test_a_script_behind_the_parent_of_a_symbolic_link_is_not_the_verifier(tmp_path: Path) -> None:
    """Audit 2026-11-10 S10-F6, review round 1: ``link/..`` is the directory the link points
    into, so ``bash link/../tmp/verify.sh`` runs the script there, not the verifier, and its
    passing run is not the verifier's: the two paths are compared as the filesystem resolves
    them (the run through a real directory's ``..`` in the test above is the verifier's)."""
    gate = _gate(tmp_path, verifier="echo FAILURE; exit 7\n")
    (tmp_path / "other" / "deep").mkdir(parents=True)
    (tmp_path / "other" / "tmp").mkdir()
    (tmp_path / "other" / "tmp" / "verify.sh").write_text("echo OTHER; exit 0\n")
    (tmp_path / "link").symlink_to("other/deep")
    for command in ("bash link/../tmp/verify.sh", "cd link && bash ../tmp/verify.sh"):
        result = _host_result(command, tmp_path)
        assert "OTHER" in result and "FAILURE" not in result, (command, result)
        _turn(gate, "Bash", {"command": command}, result)
        assert gate.last_verifier_run[0] != 0, command
        assert not cg.verifier_decides(command, str(tmp_path), gate.verifier_path), command


def test_a_background_verifier_run_is_recorded_and_not_taken_for_the_gates(tmp_path: Path) -> None:
    """A verifier the agent runs with ``Bash(background=True)`` is recorded by its exit in
    the ledger when ``bash_job`` reports it.  Audit 2026-11-10 (B/S2-F1): its pass is
    neither reused at a finish nor shown to a review as the verifier's last output (the
    files may have changed since it started): the gate runs the verifier itself."""
    gate = _gate(tmp_path, verifier="grep -q 'x = 1' a.py || exit 9\ntouch tmp/read\n")
    gate.shell = functools.partial(_outside_git_shell, tmp_path)
    tools = UsefulTools(work_dir=str(tmp_path))
    (tmp_path / "a.py").write_text("x = 1\n")
    _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"}, "ok")
    start: dict[str, Any] = {
        "command": "bash tmp/verify.sh", "description": "verify", "background": True,
    }
    started = tools.Bash(**start)
    wait: dict[str, Any] = {"job_id": _job_id(started), "action": "wait", "timeout_seconds": 15}
    _turn(gate, "Bash", start, started)
    try:
        report = tools.bash_job(**wait)
        assert "exited with code 0" in report, report
        _turn(gate, "bash_job", wait, report)
        assert gate.exit_codes[start["command"]] == "0"
        assert gate.last_verifier_run == (None, "")
        (tmp_path / "tmp" / "read").unlink()
        assert gate.verifier_run()[0] == 0 and (tmp_path / "tmp" / "read").exists()
        (tmp_path / "a.py").write_text("x = 2\n")  # a change no tool call declared
        assert gate.verifier_run()[0] == 9
    finally:
        for job in list(tools._jobs):
            tools.bash_job(job, action="kill")


def _container_bash(work_dir: Path, command: str, description: str, timeout: int,
                    max_chars: int) -> str:
    """The ``bash_fn`` of a real :class:`ContainerJobs` for the tests: the host's bash."""
    return _shell(work_dir, command, timeout)[1][:max_chars]


def _job_report(jobs: ContainerJobs, command: str, seconds: int,
                max_chars: int = 50_000) -> tuple[str, str]:
    """What the container ``Bash(on_timeout="background")`` does: start the job, wait once."""
    job_id, started = jobs.start(command)
    assert job_id is not None, started
    return job_id, jobs.bash_job(job_id, "wait", seconds, 50, max_chars)


@pytest.mark.skipif(shutil.which("setsid") is None, reason="the job scripts need setsid")
def test_a_container_job_report_is_read_by_its_status_line(tmp_path: Path) -> None:
    """Audit 2026-11-10 (S2-F1): the container ``Bash(on_timeout="background")`` returns its
    job's report (``Job <id>: <command>``, then ``status: running`` or ``exited N``), which
    read as a passing foreground run.  It is the job's exit, or a job left running, whose
    ``bash_job`` report goes into the ledger and is not shown to a review as a pass."""
    if os.path.isdir(JOBS_DIR) and not os.access(JOBS_DIR, os.W_OK):
        pytest.skip(f"{JOBS_DIR} belongs to another user")
    gate = _gate(tmp_path, verifier="for i in $(seq 300); do [ -f tmp/go ] && break; sleep .05; "
                 "done\nexit $(cat tmp/code)\n")
    jobs = ContainerJobs(functools.partial(_container_bash, tmp_path))
    command, code = "bash tmp/verify.sh", tmp_path / "tmp" / "code"
    bash: dict[str, Any] = {"command": command, "on_timeout": "background", "timeout_seconds": 1}
    code.write_text("0")
    try:
        job_id, report = _job_report(jobs, command, 1)
        _turn(gate, "Bash", bash, report)
        assert gate.exit_codes == {command: "background"} and gate.last_verifier_run == (None, "")
        assert gate.background_jobs == {job_id: command}
        (tmp_path / "tmp" / "go").touch()
        wait = {"job_id": job_id, "action": "wait"}
        _turn(gate, "bash_job", wait, jobs.bash_job(job_id, "wait", 15))
        assert gate.exit_codes == {command: "0"} and gate.last_verifier_run == (None, "")
        # a job that ended within the call: its exit, a pass (which is shown) or a failure
        for exit_code in (0, 7):
            code.write_text(str(exit_code))
            _turn(gate, "Bash", bash, _job_report(jobs, command, 15)[1])
            assert gate.exit_codes == {command: str(exit_code)}
            if exit_code == 0:
                assert gate.last_verifier_run[0] == 0
            assert gate.verifier_run()[0] == exit_code
        # the tool echoes the command without a long sleep it spent on a running job
        # (``wait_instead_of_sleep``): the status line is the report's all the same
        slept = {**bash, "command": f"sleep 31; {command}"}
        _turn(gate, "Bash", slept, _job_report(jobs, f"true; {command}", 15)[1])
        assert gate.exit_codes[slept["command"]] == "7" and gate.last_verifier_run[0] == 7
        # cut short of its status line by a small max_output_chars, a report shows no exit
        _turn(gate, "Bash", bash, _job_report(jobs, command, 15, 10)[1])
        assert gate.exit_codes[command] == "background"
        # the host tool returns the command's own output for such a call: no job report
        printed = {**bash, "command": "printf 'Job 01234567: exited with code 7\\n'"}
        _turn(gate, "Bash", printed, _host_result(printed["command"], tmp_path))
        assert gate.exit_codes[printed["command"]] == "0" and set(gate.background_jobs) == set(
            jobs.commands)
    finally:
        (tmp_path / "tmp" / "go").touch()
        for job in jobs.commands:
            jobs.wait(job, 15)  # ended: it writes no file after these are removed
            for path in Path(JOBS_DIR).glob(f"{job}.*"):
                path.unlink()


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


#: Nested past the JSON decoder's depth: decoding it raises ``RecursionError``, no ``ValueError``.
DEEP = "[" * 100_000


def test_sorcar_agent_reads_a_list_argument_as_the_tool_that_takes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-11 (N1-N4): the gate decoded ``commands``, ``tasks`` and ``evidence``
    strictly, the tools leniently.  Text nested past the decoder's depth raised out of the
    guard (the model read a harness failure, not the tool's refusal) and out of the review
    stage (the finish went through unreviewed); an array with a lone backslash, which the
    tool runs, was read as one command that changes nothing (the run was not gated)."""
    _pin_reviewer(tmp_path, monkeypatch, MODEL)
    (tmp_path / "notes.txt").write_text("a.b\naxb\n")
    count = "grep -c 'a\\.b' notes.txt > app.py"
    finding = "1. app.py: nothing checks the count."
    claim = {"success": True, "is_continue": False, "summary_in_html": BARE, "evidence": DEEP}
    script = [
        tool_call_body("run_commands_parallel", {"commands": DEEP}, 1000),
        tool_call_body("run_parallel", {"tasks": DEEP}, 1000),
        tool_call_body("run_commands_parallel",
                       {"commands": f'["{count}", "printf done > notes.md"]'}, 1000),
        tool_call_body("finish", claim, 1000),
        text_body(f"VERDICT: NOT DONE\n{finding}"),
        tool_call_body("finish", claim, 1000),
    ]
    agent, result, requests = _run_agent(tmp_path, script)
    assert result["success"] is True and len(requests) == 6
    for request, name in ((requests[1], "commands"), (requests[2], "tasks")):
        assert _last_tool_result(request).startswith(
            f"Error: {name} must be a JSON array of strings, got '[[[")
    assert _last_tool_result(requests[3]).startswith("2 commands: 2 succeeded")
    gate = agent._coding_gate
    assert gate.armed and gate.exit_codes == {count: "0", "printf done > notes.md": "0"}
    reviews = [r for r in requests if "You are reviewing, read-only" in str(r["messages"][-1])]
    assert len(reviews) == 1 and reviews[0] is requests[4]
    assert "- exit not run: `[[[" in reviews[0]["messages"][-1]["content"]
    assert finding in _tool_results(requests, 5) and gate.reviews == 1


def test_an_array_only_the_lenient_decoder_accepts_is_read_as_the_tool_runs_it(
    tmp_path: Path,
) -> None:
    """Audit 2026-11-11 (N4): the tools repair a lone backslash (grep's ``\\.``) in their
    JSON arrays and the gate did not, so it read such an array as one command, or one task,
    its raw text: no file changed, HEAD not moved, no exit code in the ledger, the evidence
    naming it ``not run``, and a fan-out of a review and an implementation one review."""
    from kiss.tests.server.conftest import init_repo

    repo = tmp_path / "repo"
    init_repo(repo)
    (repo / "notes.txt").write_text("a.b\naxb\n")
    count, commit = "grep -c 'a\\.b' notes.txt > app.py", "git commit -q --allow-empty -m mine"
    lenient = {"commands": f'["{count}", "{commit}"]'}
    with pytest.raises(ValueError):
        json.loads(lenient["commands"])
    gate = _gate(repo, custom_tools=frozenset({"git_sync"}))
    gate.observe("run_commands_parallel", lenient)
    report = UsefulTools(work_dir=str(repo)).run_commands_parallel(lenient["commands"])
    gate.result("run_commands_parallel", lenient, report, False)
    assert (repo / "app.py").read_text() == "1\n" and gate.exit_codes == {count: "0", commit: "0"}
    assert gate.armed and gate.moved_head and gate.changes == 1
    assert gate.evidence_report({"evidence": lenient["commands"]}) == (
        f"- exit 0: `{count}`\n- exit 0: `{commit}`")
    base = gate.base_commit
    _turn(gate, "git_sync", {}, "pushed")  # the commit is the run's own: the base stays
    assert gate.base_commit == base != _git(repo, "rev-parse", "HEAD").strip()
    mixed = {"tasks": r'''["Review app.py for the regex a\.b", "Implement parser.py"]'''}
    gate.observe("run_parallel", mixed)
    (repo / "parser.py").write_text("p = 1\n")  # the implementer's write
    done = "ran: sorcar\nsuccess: true\nsummary: ok\n"
    gate.result("run_parallel", mixed, yaml.safe_dump([done, done]), False)
    assert gate.delegated and not gate.reviewed_by_agent and gate.changes == 2


def _fixed_commit_message(commit_dir: Path, user_prompt: str | None = None,
                          task_result: str | None = None, bill_to: Any = None) -> str:
    """The merge's commit message without the model call that writes it in a real run."""
    return "feat: hello"


@pytest.mark.parametrize("subdir", ["", "pkg"])
def test_sorcar_agent_gates_the_merge_of_its_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subdir: str,
) -> None:
    """In a git checkout the work happens in a worktree the agent merges before it
    finishes: the gate follows the work dir into the worktree, runs the verifier and
    the review at ``commit_and_merge`` (refusing it while they fail, so the fix is made
    in the worktree), the merge asked for again after the fix is not reviewed again
    (the one review is spent, and a file changed since it), the worktree's git-ignored
    verifier is removed with it at the merge, and the finish, with no verifier left to
    run, goes through without another review.  A task started in a subdirectory of the
    repository has the same: the reviewer reads the diff of the whole repository
    (root-relative paths)."""
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
        tool_call_body("finish", claim, 1000),  # no verifier is left in the merged checkout
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
    assert "verify.sh" not in merged and "findings open" not in merged
    reviews = [r for r in requests if "You are reviewing, read-only" in str(r["messages"][-1])]
    assert len(reviews) == 1
    assert cg.MERGE_SUMMARY_PLACEHOLDER in reviews[0]["messages"][-1]["content"]
    # at the merge no finish names evidence: the reviewer is told so, not "(none named)"
    assert cg.MERGE_EVIDENCE_PLACEHOLDER in reviews[0]["messages"][-1]["content"]
    assert "(none named)" not in reviews[0]["messages"][-1]["content"]
    assert f"### {shown} (new file)\nprint('bye')" in reviews[0]["messages"][-1]["content"]
    # the work is on main, the worktree is gone with its git-ignored verifier
    assert (work / "hello.py").read_text() == "print('hello')\n"
    assert kiss_wt_branches(repo) == [] and list((repo / ".kiss-worktrees").iterdir()) == []
    assert run_git(repo, "log", "--format=%s", f"{seed}..HEAD").stdout.split() == ["feat:", "hello"]
    assert not (work / "tmp").exists()
    gate = agent._coding_gate
    assert gate.work_dir == str(work) and gate.base_commit == seed
    # the merge that went through left its round and findings behind: the finish was
    # gated afresh and ran no verifier (the last run is the merge's, in the worktree,
    # carried into the checkout, whose files are the worktree's)
    assert gate.gate_rounds == 0 and gate.reviews == 1 and gate.open_findings == []
    assert gate.verifier_memo is not None and gate.verifier_memo[0] == str(work)
    assert gate.last_verifier_run[0] == 0 and gate.exit_codes == {"bash tmp/verify.sh": "0"}
    assert gate.merge_caveats == [] and gate.delivered_findings == []


def test_a_clean_verifier_pass_in_the_worktree_stands_at_the_merge_and_at_the_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent's own clean passing run of the verifier is the gate's run: the merge
    reuses it, and the merge carries it into the checkout when the checkout then holds
    exactly the worktree's files, so the finish there, which finds the same script (one
    the agent wrote in the checkout before its worktree existed), runs nothing either.
    The script counts its runs in a file outside the repository."""
    from kiss.tests.server.conftest import init_repo, run_git

    _pin_reviewer(tmp_path, monkeypatch, MODEL)
    monkeypatch.setattr(sorcar_agent_module, "_generate_commit_message", _fixed_commit_message)
    repo, runs = tmp_path / "repo", tmp_path / "runs"
    init_repo(repo)
    (repo / ".gitignore").write_text("tmp/\n")
    run_git(repo, "add", ".gitignore")
    run_git(repo, "commit", "-q", "-m", "ignore tmp")
    verifier = f"echo r >> {shlex.quote(str(runs))}; grep -q hello hello.py\n"
    write_verifier = tool_call_body("Write", {"file_path": "tmp/verify.sh", "content": verifier},
                                    1000)
    claim = {"success": True, "is_continue": False, "summary_in_html": DONE,
             "evidence": '["bash tmp/verify.sh"]'}
    script = [
        write_verifier,  # in the checkout
        tool_call_body("create_worktree", {}, 1000),
        write_verifier,  # in the worktree
        tool_call_body("Write", {"file_path": "hello.py", "content": "print('hello')\n"}, 1000),
        tool_call_body("Bash", {"command": "bash tmp/verify.sh", "description": "check"}, 1000),
        tool_call_body("commit_and_merge", {}, 1000),
        text_body("VERDICT: OK"),
        tool_call_body("finish", claim, 1000),
    ]
    agent, result, requests = _run_agent(repo, script, max_steps=20)
    assert result["success"] is True and result["summary"] == DONE, result
    merged = next(r for r in map(_last_tool_result, requests)
                  if r.startswith("Successfully merged branch"))
    assert "findings open" not in merged
    assert (repo / "hello.py").read_text() == "print('hello')\n"
    assert (repo / "tmp" / "verify.sh").read_text() == verifier
    assert runs.read_text() == "r\n"  # the agent's run only
    gate = agent._coding_gate
    assert gate.work_dir == str(repo) and gate.verifier_memo is not None
    assert gate.verifier_memo[0] == str(repo)


def test_a_merge_carries_the_verifier_run_only_when_the_checkout_holds_the_same_files(
    tmp_path: Path,
) -> None:
    """:meth:`CodingGate.merged` keeps the worktree's last verifier run for the checkout
    when the checkout's fingerprint is the one the run saw: the files the worktree had
    untracked are committed there, which changes nothing.  A file another merge brought,
    or a change of the work after the run, makes the finish run the script again."""
    repo, wt = tmp_path / "repo", tmp_path / "wt"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / ".gitignore").write_text("tmp/\n")
    (repo / "old.py").write_text("x = 0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "start")
    _git(repo, "worktree", "add", "-q", str(wt))
    (wt / ".gitignore").write_bytes((repo / ".gitignore").read_bytes())
    runs = tmp_path / "runs"
    verifier = f"echo r >> {shlex.quote(str(runs))}; grep -q 'x = 1' a.py\n"
    for tree in (repo, wt):
        (tree / "tmp").mkdir()
        (tree / "tmp" / "verify.sh").write_text(verifier)
    gate = _gate(wt)
    _turn(gate, "Write", {"file_path": str(wt / "a.py"), "content": "x = 1\n"}, "ok")
    (wt / "a.py").write_text("x = 1\n")
    _turn(gate, "Write", {"file_path": str(wt / "old.py"), "content": "x = 2\n"}, "ok")
    (wt / "old.py").write_text("x = 2\n")
    _run(gate, wt, "bash tmp/verify.sh")
    assert runs.read_text() == "r\n" and gate.verifier_memo is not None
    # the squash merge: the worktree's files committed in the checkout
    for name in ("a.py", "old.py"):
        shutil.copy(wt / name, repo / name)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "work")
    gate.move(str(repo))
    gate.shell = functools.partial(_shell, repo)
    assert gate.merged() == "" and gate.verifier_memo is not None
    assert gate.verifier_run()[0] == 0 and runs.read_text() == "r\n"
    # another merge's file in the checkout: the files differ, the script runs again
    (repo / "other.py").write_text("y = 1\n")
    assert gate.merged() == "" and gate.verifier_memo is None
    assert gate.verifier_run()[0] == 0 and runs.read_text() == "r\n" * 2
    # a change of the work after the run: not carried either
    gate.changed()
    assert gate.merged() == "" and gate.verifier_memo is None
    # review 2026-11-11: the fingerprint tells apart a staged rename with and without
    # its source restored (rename detection named only the destination), and the
    # same rename staged or not; and a symbolic link from a regular file holding its
    # target's bytes, or from a link to another target with the same bytes
    (repo / "other.py").unlink()
    _git(repo, "mv", "old.py", "new.py")
    staged = gate.fingerprint()
    shutil.copy(repo / "new.py", repo / "old.py")
    _git(repo, "add", "old.py")
    assert None not in (staged, gate.fingerprint()) and gate.fingerprint() != staged
    (repo / "old.py").unlink()
    _git(repo, "reset", "-q")
    assert gate.fingerprint() == staged
    if not IS_WINDOWS:
        (repo / "same.py").write_text("x = 1\n")
        (repo / "link").symlink_to("a.py")
        to_a = gate.fingerprint()
        (repo / "link").unlink()
        (repo / "link").symlink_to("same.py")
        to_same = gate.fingerprint()
        (repo / "link").unlink()
        (repo / "link").write_text("x = 1\n")
        assert len({to_a, to_same, gate.fingerprint()}) == 3 and None not in {to_a, to_same}
        # review round 2: a target name's own trailing newline is part of the link
        (repo / "link").unlink()
        (repo / "link").symlink_to("a.py\n")
        assert gate.fingerprint() not in {to_a, None}


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
        assert gate.gate_rounds == 1 and gate.merge_rounds == 1  # the one refused round
        assert gate.open_findings == [finding] and gate.merge_caveats == [unanswered]
        result = gate.merged()
        assert result.startswith("\n\nMerged after 1 refused rounds with these findings open:")
        assert finding in result and gate.gate_rounds == 0 and gate.open_findings == []
        # the finish after the merge: no file changed, nothing to re-review; the
        # merge's open finding is a caveat of a summary that says nothing of it
        # (without the decisions model, every open finding is)
        args: dict[str, Any] = {"success": True, "summary_in_html": DONE}
        assert gate.finish_verdict(args, TASK) is None
        assert args["summary_in_html"].startswith(DONE + "\n<h4>Merged after 1 refused merge")
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
        assert failing.merge_rounds == 3 and failing.verifier_refusals == 3  # the cap, not past it
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
    # a sub-agent's scratch dir is its private folder: absolute, outside the work dir
    private = tmp_path / "kiss-sub-x"
    sub = CodingGate(functools.partial(_shell, tmp_path / "other"), str(tmp_path / "other"),
                     f"{private}/")
    verifier_paths = (sub.verifier_path, sub.verifier_shown, str(private / "verify.sh"))
    assert len({os.path.normpath(path) for path in verifier_paths}) == 1
    assert sub.is_verifier(str(private / "verify.sh")) and sub.is_ledger(str(private / "ledger.md"))
    assert sub.is_scratch(str(private / "probe.py")) and not sub.is_scratch(f"{private}y/a.py")
    assert sub.verifier_decides(f"bash {shlex.quote(sub.verifier_path)}")  # its own run
    sub.observe("Write", {"file_path": str(private / "probe.py"), "content": "print(1)\n"})
    assert not sub.armed and sub.written_files == []


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
    assert gate.verifier_run()[0] == 0 and gate.changes == 1  # the gate's own run: reusable
    gate.observe("run_agent", implement)
    (repo / "hello.py").rename(repo / "hi.py")
    gate.result("run_agent", implement, "ran: sorcar\nsuccess: true\nsummary: renamed\n", False)
    assert gate.changes == 2
    assert gate.verifier_run()[0] == 9


def test_every_child_left_running_is_registered_and_its_result_read_once(tmp_path: Path) -> None:
    """Audit 2026-11-10 (S2-F2, S2-F4): a ``run_parallel`` that timed out registers every child
    still running, from the YAML list (its dump folds a notice inside ``as job <id>``, with
    escapes when the agent's name is not ASCII) and from its digest (a long name pushes the
    notice's id past the cut), so the change of a child that is not the first is seen when its
    result is collected; the same result read again (``tail`` after ``wait_any``) is no change."""
    from kiss.tests.server.conftest import init_repo

    done = "ran: sorcar\nsuccess: true\nsummary: written\n"
    fan_out = {"tasks": '["Implement a.py", "Implement b.py"]', "timeout": "0.1"}
    for shape, name in (("yaml", "révision_des_modifications"), ("digest", "n" * 150)):
        repo = tmp_path / shape
        init_repo(repo)
        gate = _gate(repo, verifier="exit 0\n")
        first, second = (new_agent_job(name, {"timeout": 0.1}, None) for _ in range(2))
        notices = [job_notice(first, 0.1), job_notice(second, 0.1)]
        _turn(gate, "run_parallel", fan_out, yaml.safe_dump(notices) if shape == "yaml"
              else digest_agent_results(notices, "tmp/tool-out/x-agents.txt"))
        assert sorted(gate.jobs) == sorted([first.job_id, second.job_id]) and gate.changes == 0
        (repo / "b.py").write_text("x = 1\n")  # the second child's work
        second.result = done
        if shape == "yaml":
            gate.result("agent_job", {"job_id": "", "action": "wait_any"},
                        wait_any_agent_job({second.job_id: second}, "", "0"), False)
        else:
            gate.result("agent_job", {"job_id": second.job_id, "action": "wait"}, done, False)
        assert gate.changes == 1 and gate.delegated
        # a reviewer approves; reading the result again changes nothing
        _turn(gate, "run_agent", {"task": "check", "tool_profile": "review"}, done)
        gate.result("agent_job", {"job_id": second.job_id, "action": "tail"}, done, False)
        assert gate.changes == 1 and gate.reviewed_by_agent


def test_a_child_left_running_that_reports_no_success_is_judged_by_the_diff(tmp_path: Path) -> None:
    """A work sub-task left running (``wait="false"``) that the run killed, that ended in an
    error text or whose result was never collected has no ``success`` flag to read, so the
    gate never compared the work with its dispatch and the code it wrote left the run
    ungated; a blocking dispatch that returned the same error text was judged by the diff.
    While such a job is unsettled the diff decides, as after any delegated change; a review
    left running changes no file and decides nothing."""
    from kiss.tests.server.conftest import init_repo

    repo = tmp_path / "repo"
    init_repo(repo)
    gate = _gate(repo)
    _turn(gate, "run_agent", {"task": "Implement hello.py", "wait": "false"},
          "Started the sorcar agent task as job agent-1a2b; its tab is open.")
    assert not gate.coding_run()  # the child wrote nothing yet
    (repo / "hello.py").write_text("print('hello')\n")  # the child's work, before it is killed
    _turn(gate, "agent_job", {"job_id": "agent-1a2b", "action": "kill"},
          "Error: the sorcar agent task could not run: boom")
    assert gate.coding_run()
    review = _gate(repo)
    _turn(review, "run_agent", {"task": "check", "tool_profile": "review", "wait": "false"},
          "Started the sorcar agent task as job agent-2b3c; its tab is open.")
    (repo / "bye.py").write_text("print('bye')\n")
    assert review.jobs and not review.coding_run()


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
    assert gate.verifier_run()[0] == 0 and gate.changes == 2  # the gate's own run: reusable
    gate.observe("rollback", {"name": "bad"})
    restored = tools.rollback("bad")
    assert restored.startswith("Rolled ") and (repo / "a.py").read_text() == "x = 0\n", restored
    gate.result("rollback", {"name": "bad"}, restored, False)
    assert gate.changes == 3
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
    """A reviewer that does not answer within :data:`REVIEW_DEADLINE` is retried once at
    low effort and then given up: the side call reports the deadline, the finish goes
    through without a review, the failed review counts under :data:`MAX_FAILED_REVIEWS`,
    and the stage record says why (the first attempt's deadline under ``retried``)."""
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
            assert time.monotonic() - started < 20 and len(requests) == 3
            assert args["summary_in_html"] == DONE and gate.gate_rounds == 0
            assert (gate.reviews, gate.failed_reviews) == (0, 1) and gate.review_possible()
            assert gate.open_findings == [] and gate.changed_since_review
    finally:
        release.set()
        bind_printer(previous)
        printer.adopt_thread_binding("")
    assert charged == [(0.0, 0, 1)] * 3  # each attempt's model call is charged
    recorded = [e for e in printer.peek_recording_for_task("task-9") if e["type"] == "gate_stage"]
    review = next(e for e in recorded if e["stage"] == "review")
    assert review["outcome"] == "error" and review["error"] == "no answer within 1 s"
    assert review["retried"] == "no answer within 1 s"
    assert review["round"] == 0 and review["findings"] == 0 and review["seconds"] >= 2


def test_a_failed_reviewer_is_retried_once_at_low_effort_then_skipped(tmp_path: Path) -> None:
    """A reviewer that answers nothing, is cut short or refused is retried once at low effort.

    The first reviewer answers nothing at the catalog's default effort (the run's own
    effort override is not applied), then at low effort (the retry carries
    ``reasoning_effort: low``, the model having a catalog thinking level) approves: one
    review, its stage record naming the retry.  The second reviewer is cut short by the
    output cap and refused on the retry: the review is skipped with both failures on its
    record, the finish goes through unchanged (refused by nothing, no caveat), and the
    failure counts as a failed review, never as a refusal against the agent."""
    thinking = next(name for name, info in MODEL_INFO.items()
                    if info.thinking and info.thinking != "low"
                    and not info.is_decisions_supported)
    efforts = [MODEL_INFO[thinking].thinking, "low"]
    cut_short = text_body("VERDICT: NOT").replace(b'"finish_reason": "stop"',
                                                  b'"finish_reason": "length"')
    filtered = text_body("").replace(b'"finish_reason": "stop"',
                                     b'"finish_reason": "content_filter"')
    charged: list[tuple[float, int, int]] = []
    printer = JsonPrinter()
    printer.adopt_thread_binding("task-9")
    printer.start_recording()
    previous = bind_printer(printer)
    try:
        with serve([text_body(""), text_body("VERDICT: OK")]) as (url, requests):
            gate = _gate(tmp_path, verifier="exit 0\n", reviewer=thinking,
                         charge=functools.partial(_record_charge, charged),
                         model_config={"base_url": url, "api_key": "local",
                                       "reasoning_effort": "high"})
            _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"},
                  "ok")
            assert gate.finish_verdict({"success": True, "summary_in_html": DONE}, TASK) is None
            assert [r.get("reasoning_effort") for r in requests] == efforts
            assert (gate.reviews, gate.failed_reviews) == (1, 0) and gate.gate_rounds == 0
        with serve([cut_short, filtered]) as (url, requests):
            gate = _gate(tmp_path, verifier="exit 0\n", reviewer=thinking,
                         charge=functools.partial(_record_charge, charged),
                         model_config={"base_url": url, "api_key": "local"})
            _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 2\n"},
                  "ok")
            args = {"success": True, "summary_in_html": DONE}
            assert gate.finish_verdict(args, TASK) is None and args["summary_in_html"] == DONE
            assert [r.get("reasoning_effort") for r in requests] == efforts
            assert (gate.reviews, gate.failed_reviews) == (0, 1) and gate.gate_rounds == 0
            assert gate.review_possible() and gate.open_findings == []
    finally:
        bind_printer(previous)
        printer.adopt_thread_binding("")
    assert len(charged) == 4  # every attempt is charged
    reviews = [e for e in printer.peek_recording_for_task("task-9")
               if e["type"] == "gate_stage" and e["stage"] == "review"]
    assert [(e["outcome"], e["retried"], e["round"]) for e in reviews] == [
        ("pass", "empty answer", 1), ("error", reviews[1]["retried"], 0)]
    assert "error" not in reviews[0] and reviews[0]["model"] == thinking
    assert reviews[1]["retried"].startswith("OutputCapError(")
    assert reviews[1]["error"].startswith("ContentFilterError(")
    # what is retried, and the retry's effort by model
    assert cg.retryable_review_failure("", "") is True
    assert cg.retryable_review_failure("I could not tell.", "") is False
    assert cg.retryable_review_failure("", "no answer within 240 s") is True
    assert cg.retryable_review_failure("", "ModelRefusalError('no')") is True
    assert cg.retryable_review_failure("", "KISSError('401')") is False
    assert cg.low_effort({"api_key": "k", "reasoning_effort": "high"}, thinking) == {
        "api_key": "k", "reasoning_effort": "low"}
    assert cg.low_effort({"reasoning_effort": "high"}, MODEL) is None


def test_the_gates_verifier_run_is_reused_while_the_work_and_the_script_are_unchanged(
    tmp_path: Path,
) -> None:
    """A finish at which neither the work (the gate's change count and the fingerprint
    of the tree) nor the verifier's text changed since the gate's last run of it reuses
    that run, exit code and output, instead of running the script again, whatever the
    exit code (the stage record says ``reused``); a new text of the script (also one
    past its review prefix, whether the gate saw the edit or not), a change of the work
    (declared by a tool call or not), or a move to another tree runs it again, and a
    give-up on the unchanged work reuses the run too.  The new text is the bytes Bash
    runs: an edit Git's ``text`` conversion hides (a CR before a newline) is one.
    Outside git nothing is reused.  The script counts its runs."""
    printer = JsonPrinter()
    printer.adopt_thread_binding("task-9")
    printer.start_recording()
    previous = bind_printer(printer)
    runs = tmp_path / "tmp" / "runs"
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("tmp/\n")
    (tmp_path / ".gitattributes").write_text("*.sh text\n")
    (tmp_path / "t.sh").write_bytes(b"true\n")
    _git(tmp_path, "config", "core.safecrlf", "false")  # no warning tells a hidden CR either
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "start")
    try:
        gate = _gate(tmp_path, verifier="echo r >> tmp/runs; echo FAIL; exit 1\n")
        _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 1\n"}, "ok")
        done = {"success": True, "summary_in_html": DONE}
        first = gate.finish_verdict(dict(done), TASK)
        assert first is not None and "exited 1" in first and runs.read_text() == "r\n"
        # nothing changed: the run stands, the script does not run, the finish is
        # refused again on the same verdict
        second = gate.finish_verdict(dict(done), TASK)
        assert second is not None and "round 2 of up to 3" in second and "FAIL" in second
        assert runs.read_text() == "r\n" and gate.verifier_refusals == 2
        assert gate.last_verifier_run == (1, "FAIL\n")
        # a new text of the script, the work unchanged: it runs again, and passes now
        (tmp_path / "tmp" / "verify.sh").write_text("echo r >> tmp/runs; echo OK; exit 0\n")
        assert gate.finish_verdict(dict(done), TASK) is None and runs.read_text() == "r\nr\n"
        # a give-up on the unchanged work and script reuses the passing run
        giveup: dict[str, Any] = {"success": False, "summary_in_html": "<p>Stuck.</p>"}
        assert gate.finish_verdict(giveup, TASK) == cg.GIVEUP_RECONSIDER_VERDICT.format(
            verifier=gate.verifier_shown)
        assert runs.read_text() == "r\nr\n" and giveup["summary_in_html"].endswith(": pass.</p>")
        # a change of the work: the script runs again
        _turn(gate, "Write", {"file_path": str(tmp_path / "a.py"), "content": "x = 2\n"}, "ok")
        assert gate.finish_verdict(dict(done), TASK) is None and runs.read_text() == "r\nr\nr\n"
        # a change no tool call declared (a file written behind the gate's back): the
        # fingerprint of the tree differs, the script runs again
        (tmp_path / "a.py").write_text("x = 3\n")
        assert gate.finish_verdict(dict(done), TASK) is None and runs.read_text() == "r\n" * 4
        assert gate.finish_verdict(dict(done), TASK) is None and runs.read_text() == "r\n" * 4
        # a move to another tree (its own verifier counting in the same file): the run
        # on the first tree does not stand there, and the move back is the tree it ran on
        other = tmp_path.parent / (tmp_path.name + "-other")
        shutil.copytree(tmp_path, other)
        gate.move(str(other))
        gate.shell = functools.partial(_shell, other)
        (other / "tmp" / "verify.sh").write_text(
            f"echo r >> {shlex.quote(str(runs))}; echo OK; exit 0\n")
        assert gate.finish_verdict(dict(done), TASK) is None and runs.read_text() == "r\n" * 5
        assert gate.finish_verdict(dict(done), TASK) is None and runs.read_text() == "r\n" * 5
        gate.move(str(tmp_path))
        gate.shell = functools.partial(_shell, tmp_path)
        assert gate.finish_verdict(dict(done), TASK) is None and runs.read_text() == "r\n" * 6
        # a script longer than its review prefix, edited past it: an edit behind the
        # gate's back is a new text (the prefix the reviewer reads is the same), and the
        # Edit the gate sees stales the run, so each runs the script again
        pad = "# pad\n" * (cg.REVIEW_ONE_FILE_BYTES // 6 + 1)
        verifier = tmp_path / "tmp" / "verify.sh"
        long_pass = pad + "echo r >> tmp/runs; echo OK; exit 0\n"
        _turn(gate, "Write", {"file_path": str(verifier), "content": long_pass}, "ok")
        verifier.write_text(long_pass)
        assert gate.finish_verdict(dict(done), TASK) is None and runs.read_text() == "r\n" * 7
        assert gate.finish_verdict(dict(done), TASK) is None and runs.read_text() == "r\n" * 7
        verifier.write_text(long_pass.replace("OK", "OK2"))  # no tool call declared it
        assert gate.finish_verdict(dict(done), TASK) is None and runs.read_text() == "r\n" * 8
        assert gate.last_verifier_run == (0, "OK2\n")
        _turn(gate, "Edit", {"file_path": str(verifier), "old_string": "exit 0",
                             "new_string": "exit 1"}, "ok")
        verifier.write_text(pad + "echo r >> tmp/runs; echo FAIL; exit 1\n")
        refused = gate.finish_verdict(dict(done), TASK)
        assert refused is not None and "exited 1" in refused and runs.read_text() == "r\n" * 9
        # an edit Git's text conversion hides: under `*.sh text`, the filtered digest of
        # `true` and of `true<CR>` is one and the same, while Bash runs the one and fails
        # the other; the gate digests the bytes Bash runs (of an untracked script of the
        # work, then of the verifier), so each edit runs the script again (at a give-up:
        # past the refusal cap a success claim runs no verifier); the LF texts are written
        # as bytes, since a text write on Windows would already end their lines with CR LF
        verifier.write_bytes((pad + "echo r >> tmp/runs; echo OK; true\n").encode())
        helper = tmp_path / "b.sh"
        _turn(gate, "Write", {"file_path": str(helper), "content": "true\n"}, "ok")
        helper.write_bytes(b"true\n")
        stuck: dict[str, Any] = {"success": False, "summary_in_html": "<p>Stuck.</p>"}
        assert gate.finish_verdict(dict(stuck), TASK) is None and runs.read_text() == "r\n" * 10
        filtered = _git(tmp_path, "hash-object", "b.sh", "tmp/verify.sh")
        helper.write_bytes(b"true\r\n")
        assert gate.finish_verdict(dict(stuck), TASK) is None and runs.read_text() == "r\n" * 11
        # ... and of a tracked script, whose hidden edit the diff of the work does not show
        # (audit 2026-11-12: of the tracked files only that diff was read)
        (tmp_path / "t.sh").write_bytes(b"true\r\n")
        assert _git(tmp_path, "diff", "HEAD", "--", "t.sh") == ""
        assert gate.finish_verdict(dict(stuck), TASK) is None and runs.read_text() == "r\n" * 12
        verifier.write_bytes(verifier.read_bytes().replace(b"true\n", b"true\r\n"))
        assert _git(tmp_path, "hash-object", "b.sh", "tmp/verify.sh") == filtered
        assert gate.finish_verdict(stuck, TASK) is None and runs.read_text() == "r\n" * 13
        # bash fails `true<CR>` with 127; Git for Windows' bash strips the CR and passes
        cr_run = ("pass", ": pass.</p>") if IS_WINDOWS else ("fail", ": fail (exit 127).</p>")
        assert stuck["summary_in_html"].endswith(cr_run[1])
    finally:
        bind_printer(previous)
        printer.adopt_thread_binding("")
    recorded = [e for e in printer.peek_recording_for_task("task-9")
                if e["type"] == "gate_stage" and e["stage"] == "verifier"]
    assert [(e["outcome"], e.get("reused"), e["giveup"]) for e in recorded] == [
        ("fail", None, False), ("reused", True, False), ("pass", None, False),
        ("reused", True, True), ("pass", None, False), ("pass", None, False),
        ("reused", True, False), ("pass", None, False), ("reused", True, False),
        ("pass", None, False), ("pass", None, False), ("reused", True, False),
        ("pass", None, False), ("fail", None, False), ("pass", None, True),
        ("pass", None, True), ("pass", None, True), (cr_run[0], None, True)]
    # outside git the gate has no fingerprint of the work: nothing is reused
    plain = tmp_path.parent / (tmp_path.name + "-plain")
    plain.mkdir()
    bare = _gate(plain, verifier="echo r >> tmp/runs; exit 0\n")
    bare.shell = functools.partial(_outside_git_shell, plain)
    _turn(bare, "Write", {"file_path": str(plain / "a.py"), "content": "x = 1\n"}, "ok")
    assert bare.finish_verdict(dict(done), TASK) is None
    assert bare.finish_verdict(dict(done), TASK) is None
    assert (plain / "tmp" / "runs").read_text() == "r\nr\n"
    _run(bare, plain, "bash tmp/verify.sh")  # nor is the agent's own passing run
    assert bare.verifier_run()[0] == 0
    assert (plain / "tmp" / "runs").read_text() == "r\n" * 4


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
    # a benefit the task claims is motivation, as in the finish review's brief
    assert "is the user's motivation, not a requirement" in " ".join(reading.split())
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
