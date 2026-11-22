# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests of what the coding SEA's finish gate reads (:mod:`kiss.agents.seas.coding.gate_readers`).

The readers are functions over what the tools returned: the exit code a
``Bash`` result shows in the host tool's and in the container tool's
format, the jobs a call started and their reports, the sections of a
parallel run, and whether a command's exit status 0 means the agent's
verifier passed.  The results read here are the real tools'
(``UsefulTools``, ``ContainerJobs``) or the shapes they return, and each
verifier command is run with bash, which says what its exit shows.

``test_coding_sea_evidence_gate.py`` tests the rest of the module: the
plain shapes of ``evidence_commands``, the review parsers, the settings
of the side calls, and every reader through the SEA's harness.
"""

from __future__ import annotations

import functools
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.coding import gate_readers as gr
from kiss.agents.sorcar.container_jobs import JOBS_DIR, ContainerJobs
from kiss.agents.sorcar.shell_scan import MAX_NESTING, bash_path, simple_commands
from kiss.agents.sorcar.useful_tools import UsefulTools, digest_parallel_report
from kiss.core.kiss_tmp import kiss_tmp_dir
from kiss.core.processes import bash_version, find_bash
from kiss.core.tool_interrupt import USER_INTERRUPTED_MESSAGE
from kiss.tests.conftest import IS_WINDOWS

#: Nested past the JSON decoder's depth: decoding it raises ``RecursionError``, no ``ValueError``.
DEEP = "[" * 100_000


def _bash(command: str, work_dir: Path) -> tuple[int, str]:
    """Run *command* with bash in *work_dir*: its exit code and its combined output."""
    done = subprocess.run(
        [find_bash() or "bash", "-c", command], cwd=work_dir, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60, check=False,
    )
    return done.returncode, done.stdout


#: The bash a command below needs, by a text only that command has (each measured with bash
#: 3.2 to 5.3): the table of the programs bash has hashed came with 4.0, and ``! !`` is a
#: command since 4.2.  macOS has bash 3.2.
_SINCE = (("BASH_CMDS[bash]=", (4, 0)), ("; ! !", (4, 2)))


def _bash_shows(command: str) -> bool:
    """Whether the bash of this machine runs *command* as the bash the readers read it for.

    Not a bash older than a construct of :data:`_SINCE`.  Nor any bash where a directory of
    ``PATH`` has a blank in its name (``Program Files`` on Windows), for the unquoted ``$PATH``
    in a word of ``env``: the blank ends the word.  Nor the one of Windows for ``env -i``:
    the ``env`` of MSYS finds no ``bash`` once ``PATH`` is gone.  What the readers say of the
    command is checked on every machine.
    """
    if "env PATH=tmp/fake:$PATH" in command and " " in os.environ.get("PATH", ""):
        return False
    if "env -i " in command and IS_WINDOWS:
        return False
    return all(bash_version() >= since for mark, since in _SINCE if mark in command)


def test_a_bash_result_shows_its_exit_in_the_format_of_the_tool_that_returned_it(
    tmp_path: Path,
) -> None:
    jobs: dict[str, str] = {}
    bash = {"command": "pytest -q"}
    # the host tool: a prefix on failure, nothing on success, its own timeout wording
    assert gr.result_exit_codes("Bash", bash, "3 passed", jobs) == [("pytest -q", "0")]
    assert gr.result_exit_codes("Bash", bash, "Error (exit code 1):\nF", jobs) == [
        ("pytest -q", "1")]
    assert gr.result_exit_codes("Bash", bash, "Error: Command execution timeout after 5s. "
                                "Output before the timeout:\nx", jobs) == [("pytest -q", "timeout")]
    # the container tool: a marker after the output, its own timeout wording
    assert gr.result_exit_codes("Bash", bash, "F\n[exit code: 2]", jobs, container=True) == [
        ("pytest -q", "2")]
    assert gr.result_exit_codes("Bash", bash, "Error: command timed out after 5s", jobs) == [
        ("pytest -q", "timeout")]
    # a guard's refusal: the command never ran, so it is no evidence
    assert gr.result_exit_codes("Bash", bash, "REFUSED: the Bash call did NOT run", jobs) == [
        ("pytest -q", "refused")]
    # the tool's own error (a rejected argument) and the framework's (the tool
    # raised before running: a missing argument): the command never ran either
    assert gr.result_exit_codes("Bash", bash, "Error: machine= applies to background=True only.",
                                jobs) == [("pytest -q", "error")]
    assert gr.result_exit_codes("Bash", bash, "Failed to call Bash with {'command': 'pytest -q'}:"
                                " UsefulTools.Bash() missing 1 required positional argument",
                                jobs) == [("pytest -q", "error")]
    # only the tools' error shapes open an error: a passing command whose
    # output starts with the word is a pass (audit 2026-11-07, A07-1)
    for passing in ("Errors: 0, Failures: 0\n3 passed", "Error-free build\n",
                    "Failed to call count: 0\n"):
        assert gr.result_exit_codes("Bash", bash, passing, jobs) == [("pytest -q", "0")]
    # a call the user stopped did not run to completion: not exit 0
    assert gr.result_exit_codes("Bash", bash, USER_INTERRUPTED_MESSAGE, jobs) == [
        ("pytest -q", "interrupted")]
    assert jobs == {}
    # Audit 2026-11-01 (S2 F11): the host prefix wins over a ``[exit code: M]`` the
    # command itself printed (a nested tool's output, a fixture), in the real tool's result
    marker = "printf '[exit code: 0]\\n'; exit 7"
    result = UsefulTools(work_dir=str(tmp_path)).Bash(marker, "exit marker", timeout_seconds=5)
    assert result.startswith("Error (exit code 7):")
    assert gr.bash_exit_code(result) == "7"
    # Each tool's sign counts in its own results only: the marker line closes a
    # container result, the prefix opens a host result, and the other's is the
    # command's output (audit 2026-11-12: a passing host command that printed a
    # marker was recorded as failed).  A marker inside a line is output in both.
    assert gr.bash_exit_code("some output\n[exit code: 3]", container=True) == "3"
    assert gr.bash_exit_code("[exit code: 0]\n", container=True) == "0"
    assert gr.bash_exit_code("fixture says [exit code: 5]\n", container=True) == "0"
    assert gr.bash_exit_code("Error (exit code 2):\n[exit code: 4]", container=True) == "4"
    assert gr.bash_exit_code("Error (exit code 2):\n[exit code: 4]") == "2"
    assert gr.bash_exit_code("F\n[exit code: 2]") == "0"
    assert gr.bash_exit_code("3 passed\nError (exit code 2):\nfrom a log\n") == "0"


def test_a_job_is_registered_by_its_notice_and_read_by_the_status_line_of_its_report(
    tmp_path: Path,
) -> None:
    jobs: dict[str, str] = {}
    bash = {"command": "pytest -q"}
    # a background job, then its exit through bash_job (host wording), only from the status line
    started = "Started background job j1 (pid 7).\nlog: /tmp/j1.log"
    make = {"command": "make", "background": True}
    assert gr.result_exit_codes("Bash", make, started, jobs) == [("make", "background")]
    # a notice opens the result of a call that can return it (the follow-up's, for a
    # command made as an Edit first), and the container's marker a line: the same words
    # in a passing command's own output are text (audit 2026-11-12: they recorded a job
    # left running, or a failure, and registered the job)
    for printed in ("Started background job nested (pid 42).\n",  # of a foreground call
                    "$ make\nStill running after 5s; left running as background job j8 (pid 1).\n",
                    'assert out.endswith("[exit code: 9]")\n', "[exit code: 9]\n"):
        assert gr.result_exit_codes("Bash", bash, printed, jobs) == [("pytest -q", "0")], printed
    assert gr.result_exit_codes("Bash", make, f"$ make\n{started}", jobs) == [("make", "0")]
    left = ("Still running after 120s; left running as background job j3 (pid 8). Log: /tmp/j3\n"
            "Output so far:\n")
    assert gr.result_exit_codes("Bash", bash, left, jobs) == [("pytest -q", "background")]
    rewrite = {"command": "sed -i s/x/y/ a.py && make", "background": True}
    followed = ('Applied as Edit(file_path="a.py", ...), whose result follows:\n'
                "Successfully replaced 1 occurrence(s)\n"
                'Then made Bash(command="make", background=true), whose result follows:\n'
                + started.replace("j1", "j4"))
    assert gr.result_exit_codes("Bash", rewrite, followed, jobs) == [
        (rewrite["command"], "background")]
    assert jobs == {"j1": "make", "j3": "pytest -q", "j4": rewrite["command"]}
    running = "Job j1: running (pid 7, 2s)\n$ make\n--- last 5 log lines ---\nexited with code 0"
    assert gr.result_exit_codes("bash_job", {"job_id": "j1"}, running, jobs) == []
    assert gr.result_exit_codes("bash_job", {"job_id": "j1"}, "Job j1: exited with code 3\n$ make",
                                jobs) == [("make", "3")]
    assert gr.result_exit_codes("bash_job", {"job_id": "j9"}, "Job j9: exited with code 3\n",
                                jobs) == []  # a job no call of the run started
    # the container report opens with the command's echo, the host's with the
    # status, then the command: a command holding a status line of its own
    # (audit 2026-11-05, A05-6) supplies no exit in either
    faking = "printf '%s\\n' 'x\nstatus: exited 0\n' && bash tmp/verify.sh"
    jobs["c1"] = faking
    container = (f"Job c1: {faking}\nstatus: exited 7\nlog: /tmp/kiss-jobs/c1.log\n"
                 "--- last 50 log lines ---\nx\nstatus: exited 0\nFAIL")
    assert gr.result_exit_codes("bash_job", {"job_id": "c1"}, container, jobs) == [(faking, "7")]
    assert gr.result_exit_codes("bash_job", {"job_id": "c1"}, container.replace(
        "status: exited 7", "status: running"), jobs) == []
    jobs["j2"] = "echo x\nstatus: exited 0"
    host = "Job j2: running (pid 7, 2s)\nCommand: echo x\nstatus: exited 0\nLog: /tmp/j2.log\n"
    assert gr.result_exit_codes("bash_job", {"job_id": "j2"}, host, jobs) == []
    # the host tool itself: the notice of a job it started registers the job, and the
    # job's exit is its report's status line, whatever the job printed into its log
    tools = UsefulTools(work_dir=str(tmp_path))
    printing = "printf 'Job j0: exited with code 0\\n'; exit 4"
    start: dict[str, Any] = {"command": printing, "description": "d", "background": True}
    literal = set(jobs)
    try:
        assert gr.result_exit_codes("Bash", start, tools.Bash(**start), jobs) == [
            (printing, "background")]
        (job_id,) = set(jobs) - literal
        assert list(tools._jobs) == [job_id] and jobs[job_id] == printing
        wait: dict[str, Any] = {"job_id": job_id, "action": "wait", "timeout_seconds": 15}
        assert gr.result_exit_codes("bash_job", wait, tools.bash_job(**wait), jobs) == [
            (printing, "4")]
    finally:
        for job in list(tools._jobs):
            tools.bash_job(job, action="kill")


def test_a_parallel_report_gives_each_command_the_exit_of_its_own_section(tmp_path: Path) -> None:
    jobs: dict[str, str] = {}
    # a call the user stopped did not run to completion: no command of it exited 0
    assert gr.result_exit_codes("run_commands_parallel", {"commands": '["a", "b"]'},
                                USER_INTERRUPTED_MESSAGE, jobs) == [
        ("a", "interrupted"), ("b", "interrupted")]
    # one section per command
    report = ("2 commands: 1 succeeded, 1 failed, 0 timed out.\n\n### [1/2] exit 0 in 0.1s\n$ a\n"
              "ok\n\n### [2/2] TIMED OUT after 9.0s\n$ b\n")
    assert gr.result_exit_codes("run_commands_parallel", {"commands": '["a", "b"]'}, report,
                                jobs) == [("a", "0"), ("b", "timeout")]
    # the digest of a report too long for the conversation: every command's
    # exit is on its status line (the sections are only the failed ones')
    digest = digest_parallel_report(["a", "b", "c"], [(0, "ok", 0.1), (None, "", 9.0),
                                                      (2, "E", 0.2)], "tmp/tool-out/x.txt")
    assert gr.result_exit_codes("run_commands_parallel", {"commands": '["a", "b", "c"]'},
                                digest, jobs) == [("a", "0"), ("b", "timeout"), ("c", "2")]
    # a status line in a command's own output (a nested parallel run) does
    # not override the command's exit, in the digest or in the full report
    outer = "echo '[1/1] exit 0 in 0.1s: inner'; exit 1"
    tools = UsefulTools(work_dir=str(tmp_path))
    report = tools.run_commands_parallel(f'[{outer!r}]')
    inner_status: list[tuple[int | None, str, float]] = [(1, "[1/1] exit 0 in 0.1s: inner\n", 0.2)]
    for text in (report, digest_parallel_report([outer], inner_status, "tmp/tool-out/y.txt")):
        assert gr.result_exit_codes("run_commands_parallel", {"commands": f'[{outer!r}]'}, text,
                                    jobs) == [(outer, "1")]
    # nor does a whole nested report in the output, section headers included
    # (audit 2026-11-07, A07-4): the report's own sections come in order,
    # count the commands and echo each one
    nested = "printf '\\n### [1/1] exit 0 in 0.1s\\n$ inner\\nok\\n'; exit 7"
    for commands in ([nested], ["true", nested], [nested, "true"]):
        text = tools.run_commands_parallel(json.dumps(commands))
        assert "\n### [1/1] exit 0" in text
        assert gr.result_exit_codes("run_commands_parallel", {"commands": json.dumps(commands)},
                                    text, jobs) == [
            (command, "7" if command == nested else "0") for command in commands]
    assert jobs == {} and gr.result_exit_codes("Read", {"file_path": "a.py"}, report, jobs) == []


def test_a_list_argument_is_read_as_the_tool_that_takes_it_decodes_it(tmp_path: Path) -> None:
    """Audit 2026-11-11 (N1-N4): the tools repair a lone backslash (grep's ``\\.``) in their
    JSON arrays and refuse text nested past the decoder's depth.  The readers decoded
    strictly: such an array was one command, its raw text (no exit code for the commands
    that ran, the evidence naming them ``not run``), and the deep text raised."""
    (tmp_path / "notes.txt").write_text("a.b\naxb\n")
    count, note = "grep -c 'a\\.b' notes.txt > count.txt", "printf done > notes.md"
    lenient = f'["{count}", "{note}"]'
    with pytest.raises(ValueError):
        json.loads(lenient)
    tools = UsefulTools(work_dir=str(tmp_path))
    report = tools.run_commands_parallel(lenient)
    assert (tmp_path / "count.txt").read_text() == "1\n"
    assert gr.result_exit_codes("run_commands_parallel", {"commands": lenient}, report, {}) == [
        (count, "0"), (note, "0")]
    assert gr.evidence_commands(lenient) == [count, note]
    refusal = tools.run_commands_parallel(DEEP)
    assert refusal.startswith("Error: commands must be a JSON array of strings, got '[[[")
    assert gr.result_exit_codes("run_commands_parallel", {"commands": DEEP}, refusal, {}) == []
    assert gr.evidence_commands(DEEP) == [DEEP]


def test_tool_results_are_found_in_the_three_message_shapes() -> None:
    assert gr.tool_result_texts({"type": "function_call_output", "output": "x"}) == ["x"]
    assert gr.tool_result_texts({"role": "tool", "content": "y"}) == ["y"]
    anthropic = {"role": "user", "content": [
        {"type": "tool_result", "content": "a"},
        {"type": "tool_result", "content": [{"type": "text", "text": "b"}]},
        # a Read result with an image attached is one result, not two
        {"type": "tool_result", "content": [{"type": "text", "text": "Image attached"},
                                            {"type": "image", "source": {}}]}]}
    assert gr.tool_result_texts(anthropic) == ["a", "b", "Image attached"]
    assert gr.tool_result_texts({"role": "user", "content": "prompt"}) == []


def test_a_command_made_as_an_edit_ended_as_its_follow_up_did() -> None:
    # a shell command the tool made as an Edit through the agent's tool path:
    # its exit is the follow-up's (the rest of the command), read through a
    # follow-up that was redirected again (a second file written)
    redirected = ('Applied as Edit(file_path="a.py", ...), whose result follows:\n'
                  "Successfully replaced 1 occurrence(s)\n"
                  'Then made Bash(command="pytest -q", description="rest"), whose result '
                  "follows:\nError (exit code 1):\nF")
    assert gr.bash_exit_code(redirected) == "1"
    nested = ('Applied as Edit(file_path="a.py", ...), whose result follows:\n'
              "Successfully replaced 1 occurrence(s)\n"
              'Then made Bash(command="sed -i s/x/y/ b.py && pytest -q", description="rest"), '
              "whose result follows:\n" + redirected.replace("a.py", "b.py"))
    assert gr.bash_exit_code(nested) == "1"
    assert gr.bash_exit_code(nested.replace("Error (exit code 1):\nF", "3 passed")) == "0"
    # the edit alone (no rest to run), the edit that failed, and the rest not made
    edit_only = ('Appended through Write(file_path="a.py", ...), whose result follows:\n'
                 "Successfully wrote 12 characters")
    assert gr.bash_exit_code(edit_only) == "0"
    assert gr.bash_exit_code(edit_only.replace("Successfully wrote 12 characters",
                                               "Error: file_path must be absolute")) == "error"
    not_made = ('Applied as Edit(file_path="a.py", ...), whose result follows:\n'
                "REFUSED: the Edit call did NOT run\n"
                'Bash(command="pytest -q") was not made: the Edit did not go through.')
    assert gr.bash_exit_code(not_made) == "error"
    hook_refused = ('Applied as Edit(file_path="a.py", ...), whose result follows:\n'
                    "Not now: the caller's hook said no\n"
                    'Bash(command="pytest -q") was not made: the Edit did not go through.')
    assert gr.bash_exit_code(hook_refused) == "error"


def test_exit_0_is_the_verifiers_pass_only_when_its_run_decides_the_commands_exit(
    tmp_path: Path,
) -> None:
    """A passing command shows that the verifier passed only when the verifier's exit status
    is the command's: ``bash -n``, ``|| true``, ``| cat``, ``true ||``, ``!``, a command
    after it, ``&`` or an ``exit`` / ``exec`` before it hide a failing verifier.  Audit
    2026-11-04 (S8): so do a shell option that reads the script from elsewhere (``bash
    -s``: stdin; ``--rcfile``: an option's value) or runs nothing (``-D``), and ``set -n``
    / ``set -o noexec`` before it.  Finish review 2026-11-04: ``--`` (or ``-``) ends the
    shell's options, so ``bash -- -x tmp/verify.sh`` runs a file named ``-x`` and only
    passes the verifier to it as an argument.  Audit 2026-11-21 (LF): the ``exec`` is the
    shell's own behind an assignment, ``command``, ``builtin`` or ``time -p`` too, and the
    program the shell becomes is the word after its options, a wrapper as well (``exec
    env``).  Audit 2026-11-21 (LF2): ``shopt -so noexec`` and a sourced file keep the
    verifier from running, a function, a ``PATH`` or a hash-table entry makes the word
    ``bash`` another program, ``BASH_ENV`` and ``SHELLOPTS`` start the verifier's shell
    otherwise, and a trap whose action leaves the shell decides the exit after it; so may
    a line only the shell knows how to run, and a wrapper's option may keep it from
    running anything.  A trap that only cleans up, an appended ``PATH`` and ``command
    bash`` past a function do not (LF3: an appended directory holds the programs no
    earlier one has, so only ``bash``, ``sh`` and a path are then what they were, with
    no wrapper in front).  Audit 2026-11-21 (LF3): nor do the options of
    ``timeout`` for its signals and its messages or those of ``env`` that take variables
    out of the environment; a wrapper at another path is another program, a word of a
    wrapper the shell expands may be an option or hold another command, and a login shell
    and zsh read other text before the script.  Bash says what each command's
    exit shows: over a failing verifier a hiding command exits 0 and a deciding one as
    the verifier."""
    script = tmp_path / "tmp" / "verify.sh"
    script.parent.mkdir()
    script.write_text("echo FAILURE; exit 7\n")
    (tmp_path / "-x").write_text("echo OTHER; exit 0\n")  # the script ``bash -- -x`` runs
    (tmp_path / "sub").mkdir()
    fake = tmp_path / "tmp" / "fake" / "bash"  # another program of the shell's name
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    (tmp_path / "tmp" / "quit.sh").write_text("exit 0\n")
    shutil.copy(fake, fake.with_name("env"))  # another program of a wrapper's name
    for missing in ("exec", "verify.sh"):  # names no directory of bash's ``PATH`` has
        shutil.copy(fake, fake.with_name(missing))
    home = tmp_path / "tmp" / "home"  # its startup files end the shell that reads them
    home.mkdir()
    for startup in (".bash_profile", ".zshenv"):
        (home / startup).write_text("exit 0\n")
    work, verifier, named = str(tmp_path), str(script), shlex.quote(str(script))
    not_run = ("bash -n tmp/verify.sh", "sh -xn ./tmp/verify.sh", "bash -s tmp/verify.sh",
               "bash -D tmp/verify.sh", "bash --rcfile tmp/verify.sh",
               "bash -- -x tmp/verify.sh", "sh - -x tmp/verify.sh", "bash -e -- -x tmp/verify.sh",
               # the word runs a function, or what only the shell knows by then
               "bash() { :; }; bash tmp/verify.sh", "function bash { true; }; bash tmp/verify.sh",
               "timeout() { :; }; timeout 60 bash tmp/verify.sh", "tmp/fake/bash tmp/verify.sh",
               # a wrapper's option under which it runs nothing
               "command -v bash tmp/verify.sh", "timeout --help 60 bash tmp/verify.sh",
               "xargs -r bash tmp/verify.sh", "timeout -k 5 --h 60 bash tmp/verify.sh",
               # a wrapper at another path is another program, and a word of a wrapper the
               # shell expands may hold another command
               "tmp/fake/env bash tmp/verify.sh",
               "timeout ${KISS_NO_SUCH_NAME:-60 true} bash tmp/verify.sh",
               "env KISS_X=1${IFS}true bash tmp/verify.sh",
               "env PATH=tmp/fake:$PATH bash tmp/verify.sh",
               # the shell reads other text first: a login shell its profiles, zsh ``.zshenv``
               "HOME=$PWD/tmp/home bash -l tmp/verify.sh",
               "HOME=$PWD/tmp/home bash --login tmp/verify.sh",
               *(["ZDOTDIR=tmp/home zsh tmp/verify.sh"] if shutil.which("zsh") else []),
               f"${{KISS_NO_SUCH_NAME:-exit}} 0; bash {named}",
               f"shopt -s expand_aliases\nalias bash=true\nbash {named}")
    # the verifier runs, as a command only the shell knows: a trap may run after any command
    unread = (f"trap 'exit 0' ERR; bash {named}",)
    hidden = (*not_run, *unread, "bash tmp/verify.sh || true", "bash tmp/verify.sh | cat",
              "true || bash tmp/verify.sh", "bash tmp/verify.sh; echo done",
              "bash tmp/verify.sh &", "(bash tmp/verify.sh) || true",
              "bash tmp/verify.sh && echo PASS || echo FAIL", "bash tmp/verify.sh && true &",
              "bash tmp/verify.sh && echo ok; true", "bash tmp/verify.sh && (true; true) || true",
              "bash tmp/verify.sh\necho \"exit $?\"", "ls || bash tmp/verify.sh && echo ok",
              "! bash tmp/verify.sh", "exit 0; bash tmp/verify.sh", "! (bash tmp/verify.sh)",
              "exec true; bash tmp/verify.sh", "true && ! bash tmp/verify.sh",
              "set -n; bash tmp/verify.sh", "set -o noexec; bash tmp/verify.sh",
              "set -xn; bash tmp/verify.sh", "set -eo noexec; bash tmp/verify.sh",
              "set +n -n; bash tmp/verify.sh",
              "command exec true; bash tmp/verify.sh", "builtin exec true; bash tmp/verify.sh",
              "X=1 command -p builtin -- exec -a v true; bash tmp/verify.sh",
              "time -p exec true; bash tmp/verify.sh",
              "exec env; bash tmp/verify.sh", "X=1 exec -l -- env; bash tmp/verify.sh",
              "shopt -so noexec; bash tmp/verify.sh", "shopt -o -qs noexec; bash tmp/verify.sh",
              "bash tmp/verify.sh; ((1))", "bash tmp/verify.sh; time", "bash tmp/verify.sh; ! !",
              f". tmp/quit.sh; bash {named}", f"command source tmp/quit.sh && bash {named}",
              "trap 'exit 0' EXIT; bash tmp/verify.sh",
              "trap -- 'true; exec true' 0; bash tmp/verify.sh",
              "trap '${KISS_NO_SUCH_NAME:-exit} 0' EXIT; bash tmp/verify.sh",
              "trap 'eval \"${KISS_NO_SUCH_NAME:-exit 0}\"' EXIT; bash tmp/verify.sh",
              "X=$(echo 'exit 0'); trap \"$X\" EXIT; bash tmp/verify.sh",
              "quit() { exit 0; }; trap quit EXIT; bash tmp/verify.sh",
              "PATH=tmp/fake:$PATH; bash tmp/verify.sh",
              'export PATH="$PWD/tmp/fake:$PATH" && bash tmp/verify.sh',
              "PATH=tmp/fake:$PATH bash tmp/verify.sh",
              "env PATH=tmp/fake bash tmp/verify.sh",
              "read PATH <<< tmp/fake; bash tmp/verify.sh",
              "export P{ATH,}=tmp/fake; bash tmp/verify.sh",
              "N=PA; export ${N}TH=tmp/fake; bash tmp/verify.sh",
              "printf 'P\\101TH' > tmp/name; read {x,$(cat tmp/name)} <<< 'a tmp/fake'; "
              "bash tmp/verify.sh",
              "PATH=$\"PATH\":tmp/fake; bash tmp/verify.sh",
              # an appended directory gives the names no earlier one has: the program
              # ``exec`` a wrapper asks for, a script named by a bare word
              "PATH=$PATH:tmp/fake; env exec bash tmp/verify.sh",
              "PATH=$PATH:$PWD/tmp/fake; cd tmp && verify.sh",
              "BASH_CMDS[bash]=tmp/fake/bash; bash tmp/verify.sh",
              "BASH_ENV=tmp/quit.sh bash tmp/verify.sh",
              "env SHELLOPTS=noexec bash tmp/verify.sh")
    for command in hidden:
        code, output = _bash(command, tmp_path)
        if _bash_shows(command):
            assert code == 0, (command, output)  # exit 0, verifier or not
            if command in not_run:
                assert "FAILURE" not in output, (command, output)  # the verifier did not run
        assert not gr.verifier_decides(command, work, verifier), command
        found = gr.verifier_command(simple_commands(command, bash_path(work)), verifier)
        assert (found is None) == (command in (*not_run, *unread)), command
    deciding = (f"cd {shlex.quote(str(tmp_path))} && bash tmp/verify.sh",
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
                # no ``exec`` of a program: a bare one behind ``command`` or with a name,
                # the one ``command -v`` describes, the program of that name ``env`` asks for
                "command exec 2>&1; bash tmp/verify.sh", "exec -a v; bash tmp/verify.sh",
                "command -v exec true; bash tmp/verify.sh", "env exec true; bash tmp/verify.sh",
                # noexec left off, a trap that only cleans up, a function of another name
                # or behind ``command``, a ``PATH`` with the directory of another ``bash``
                # appended, a variable whose name only ends in ``PATH``
                "shopt -uo noexec; bash tmp/verify.sh", "shopt -so nounset; bash tmp/verify.sh",
                "trap 'rm -f tmp/gone' EXIT; bash tmp/verify.sh", "trap - EXIT; bash tmp/verify.sh",
                "tidy() { rm -f tmp/gone; }; trap tidy EXIT; bash tmp/verify.sh",
                "sh() { :; }; bash tmp/verify.sh", "bash() { :; }; command bash tmp/verify.sh",
                "PATH=$PATH:tmp/fake; bash tmp/verify.sh",
                'export PATH="${PATH}:tmp/fake" && bash tmp/verify.sh',
                "PATH=$PATH:tmp/fake KISS_X=1 /bin/sh tmp/verify.sh",
                "PYTHONPATH=tmp/fake bash tmp/verify.sh", "time bash tmp/verify.sh",
                f"eval 'true'; bash {named}", "/bin/sh tmp/verify.sh",
                "export KISS_X=$HOME && bash tmp/verify.sh",
                "printf '%s\\n' \"$HOME\" > /dev/null; bash tmp/verify.sh",
                # the options under which a wrapper still runs its command and exits as it
                # does: ``timeout``'s for its signals and its messages, ``env``'s that take
                # variables out of the environment
                "timeout -k 5 60 bash tmp/verify.sh", "env -i bash tmp/verify.sh",
                "timeout -k5 --kill-after=5 --kill-after 5 -s KILL -sKILL --signal=KILL "
                "--signal KILL --foreground -v --verbose --preserve-status 1.5m bash tmp/verify.sh",
                "KISS_W=1 /usr/bin/env -i -u KISS_X -uKISS_Y -- KISS_Z=1 bash tmp/verify.sh",
                "nice timeout -s KILL 60 env -u KISS_X bash tmp/verify.sh")
    for command in deciding:
        assert gr.verifier_decides(command, work, verifier), command
        assert _bash(command, tmp_path)[0] == 7 or not _bash_shows(command), command
    # named by its absolute path, the verifier need not be under the work dir
    # (a sub-agent's is in its private folder)
    outside = f"bash {shlex.quote(verifier)}"
    assert gr.verifier_decides(outside, str(tmp_path / "sub"), verifier)
    assert _bash(outside, tmp_path / "sub")[0] == 7
    # a shell is its word or in /bin or /usr/bin: ``/usrbash`` is neither (no such
    # program here, so bash is not asked)
    assert not gr.verifier_decides("/usrbash tmp/verify.sh", work, verifier)
    # a NUL in a wrapper's word is no end of it: with the NUL gone this is ``env
    # -uKISS_X-u --help``, which runs nothing (bash takes no NUL in its ``-c`` text, so
    # bash is not asked)
    assert not gr.verifier_decides("env -uKISS_X\0-u --help bash tmp/verify.sh", work, verifier)
    script.write_text("echo RUN; exit 0\n")
    for command in filter(_bash_shows, deciding):
        code, output = _bash(command, tmp_path)
        assert code == 0 and ("RUN" in output) == ("> tmp/log" not in command), (command, output)
    # the commands past the scanner's nesting limit are unread and may decide the exit
    deep = "echo " + "$(echo " * (MAX_NESTING + 1) + "x" + ")" * (MAX_NESTING + 1)
    assert gr.verifier_decides("bash tmp/verify.sh && echo $(echo x)", work, verifier)
    assert not gr.verifier_decides(f"bash tmp/verify.sh && {deep}", work, verifier)


def test_a_verifier_kept_in_the_scratch_dir_is_named_by_its_absolute_path(tmp_path: Path) -> None:
    """The scratch dir is a folder of the agents' temp directory (``kiss-tmp`` in the
    system's), no longer ``tmp/`` of the work dir: the verifier kept there is run by its
    absolute path, or by its name after a ``cd`` into the scratch dir, and bash says that
    such a run decides the command's exit.  A script of that name in a ``tmp`` folder of
    the work dir is not the verifier."""
    scratch = Path(tempfile.mkdtemp(dir=kiss_tmp_dir()))
    script = scratch / "verify.sh"
    script.write_text("echo FAILURE; exit 7\n")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "verify.sh").write_text("echo OTHER; exit 0\n")
    work, verifier, named = str(tmp_path), str(script), shlex.quote(script.as_posix())
    for command in (f"bash {named}", f"bash -x {named} 2>&1 && echo VERIFIED",
                    f"cd {shlex.quote(scratch.as_posix())} && bash verify.sh"):
        assert gr.verifier_decides(command, work, verifier), command
        assert _bash(command, tmp_path)[0] == 7, command
    for command in (f"bash {named} || true", f"bash -n {named}", "bash tmp/verify.sh",
                    "cd tmp && bash verify.sh"):
        assert not gr.verifier_decides(command, work, verifier), command
        assert _bash(command, tmp_path)[0] == 0, command


@pytest.mark.skipif(IS_WINDOWS, reason="symbolic links")
def test_the_verifier_is_the_script_the_filesystem_resolves_the_path_to(tmp_path: Path) -> None:
    """Audit 2026-11-10 S10-F6, review round 1: ``link/..`` is the directory the link points
    into, so ``bash link/../tmp/verify.sh`` runs the script there, not the verifier, and its
    passing run is not the verifier's: the two paths are compared as the filesystem resolves
    them.  A symbolic link to the verifier runs the verifier under another name."""
    script = tmp_path / "tmp" / "verify.sh"
    script.parent.mkdir()
    script.write_text("echo FAILURE; exit 7\n")
    (tmp_path / "other" / "deep").mkdir(parents=True)
    (tmp_path / "other" / "tmp").mkdir()
    (tmp_path / "other" / "tmp" / "verify.sh").write_text("echo OTHER; exit 0\n")
    (tmp_path / "link").symlink_to("other/deep")
    for command in ("bash link/../tmp/verify.sh", "cd link && bash ../tmp/verify.sh"):
        assert _bash(command, tmp_path) == (0, "OTHER\n"), command
        assert not gr.verifier_decides(command, str(tmp_path), str(script)), command
    (tmp_path / "tmp" / "check.sh").symlink_to("verify.sh")
    assert _bash("bash tmp/check.sh", tmp_path) == (7, "FAILURE\n")
    assert gr.verifier_decides("bash tmp/check.sh", str(tmp_path), str(script))
    # the system's temp directory behind a symbolic link (macOS: /var -> /private/var): the
    # verifier kept in the scratch dir is the same script by either path
    scratch = Path(tempfile.mkdtemp(dir=kiss_tmp_dir()))
    (scratch / "verify.sh").write_text("echo SCRATCH; exit 7\n")
    (tmp_path / "T").symlink_to(scratch.parent)
    linked = f"bash {shlex.quote(f'{tmp_path}/T/{scratch.name}/verify.sh')}"
    assert _bash(linked, tmp_path) == (7, "SCRATCH\n")
    assert gr.verifier_decides(linked, str(tmp_path), str(scratch / "verify.sh"))


def _container_bash(work_dir: Path, command: str, description: str, timeout: int,
                    max_chars: int) -> str:
    """The ``bash_fn`` of a real :class:`ContainerJobs` for the tests: the host's bash."""
    return _bash(command, work_dir)[1][:max_chars]


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
    read as a passing foreground run.  It is the job's exit, or a job left running whose
    ``bash_job`` report gives the exit later."""
    if os.path.isdir(JOBS_DIR) and not os.access(JOBS_DIR, os.W_OK):
        pytest.skip(f"{JOBS_DIR} belongs to another user")
    (tmp_path / "tmp").mkdir()
    # the verifier runs until tmp/go releases it, however long this process takes to get
    # there; once this process is gone nobody is left to release it, and it ends by itself
    (tmp_path / "tmp" / "verify.sh").write_text(
        f"until [ -f tmp/go ] || ! kill -0 {os.getpid()} 2>/dev/null; do sleep .05; done\n"
        "exit $(cat tmp/code)\n")
    jobs = ContainerJobs(functools.partial(_container_bash, tmp_path))
    command, code = "bash tmp/verify.sh", tmp_path / "tmp" / "code"
    bash: dict[str, Any] = {"command": command, "on_timeout": "background", "timeout_seconds": 1}
    code.write_text("0")
    started: dict[str, str] = {}
    try:
        job_id, report = _job_report(jobs, command, 1)
        assert gr.result_exit_codes("Bash", bash, report, started, container=True) == [
            (command, "background")]
        assert started == {job_id: command}
        (tmp_path / "tmp" / "go").touch()
        wait = {"job_id": job_id, "action": "wait"}
        assert gr.result_exit_codes("bash_job", wait, jobs.bash_job(job_id, "wait", 15), started,
                                    container=True) == [(command, "0")]
        # a job that ended within the call: its exit, a pass or a failure
        for exit_code in ("0", "7"):
            code.write_text(exit_code)
            assert gr.result_exit_codes("Bash", bash, _job_report(jobs, command, 15)[1], started,
                                        container=True) == [(command, exit_code)]
        # the tool echoes the command without a long sleep it spent on a running job
        # (``wait_instead_of_sleep``): the status line is the report's all the same
        slept = {**bash, "command": f"sleep 31; {command}"}
        assert gr.result_exit_codes("Bash", slept, _job_report(jobs, f"true; {command}", 15)[1],
                                    started, container=True) == [(slept["command"], "7")]
        # cut short of its status line by a small max_output_chars, a report shows no exit
        assert gr.result_exit_codes("Bash", bash, _job_report(jobs, command, 15, 10)[1], started,
                                    container=True) == [(command, "background")]
        # the host tool returns the command's own output for such a call: no job report
        printed = {**bash, "command": "printf 'Job 01234567: exited with code 7\\n'"}
        assert gr.result_exit_codes("Bash", printed, _bash(printed["command"], tmp_path)[1],
                                    started) == [(printed["command"], "0")]
        assert set(started) == set(jobs.commands)
    finally:
        (tmp_path / "tmp" / "go").touch()
        for job in jobs.commands:
            # released, a job ends well within the wait; one that still runs on a stalled
            # host is killed (a second each): no job writes a file after these are removed
            if jobs.wait(job, 15) == "running":
                jobs.bash_job(job, "kill")
            for path in Path(jobs.dirs[job]).glob(f"{job}.*"):
                path.unlink()
