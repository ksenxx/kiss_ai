# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""What the coding SEA's finish gate reads: shell results, verifier runs, the review's reply.

The gate of the containerised coding SEA
(:meth:`kiss.agents.seas.coding.coding_sea.ContainerHarness.finish_verdict`)
decides on what happened, not on what the agent claims: how each shell
command the agent ran ended (:func:`result_exit_codes`, read from the
tool results in the conversation), whether an exit status 0 of a command
means the agent's verifier passed (:func:`verifier_decides`), which
commands a ``finish`` names as evidence (:func:`evidence_commands`), what
a reviewer model's reply finds (:func:`review_findings`,
:func:`numbered_items`) and whether the next finish answers it
(:func:`answered_findings`).  The module also holds the settings of the
gate's side calls (:func:`default_effort`, :func:`low_effort`,
:func:`retryable_review_failure`) and their deadline
(:func:`call_stop_event`).  Nothing here depends on the trial container.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from kiss.agents.seas.coding.shell_changes import shell_commands
from kiss.agents.sorcar.fanout_guard import _decode_json_leniently
from kiss.agents.sorcar.shell_scan import (
    SHELLS,
    Group,
    SimpleCommand,
    TooDeepError,
    bash_path,
    host_path,
    resolve,
    simple_commands,
)
from kiss.core import stop_signal
from kiss.core.tool_interrupt import USER_INTERRUPTED_MESSAGE

#: The reviewer's verdict line.
REVIEW_VERDICT_LINE = re.compile(r"VERDICT:\s*(OK|NOT DONE)\b", re.I)
#: Probability at which the decisions model's answer counts a finding as
#: answered (probed: quoted command output scores 0.95, a described change
#: 0.84, "the reviewer misread the spec" 0.06, "I believe it is fine" 0.07).
ANSWERED_THRESHOLD = 0.5
#: Question the decisions model answers per finding (see :func:`answered_findings`).
ANSWER_QUESTION = (
    "Does the engineer's summary answer this review finding with concrete evidence: a "
    "command it ran and the output quoted, or a described change to the files? Prose "
    "rebuttals without a command, output or change do not count. Finding: {finding}"
)

#: The failure marker of the container ``Bash`` tools: a line of its own after
#: the output (a note may follow it on that line).
EXIT_CODE_MARKER = re.compile(r"^\[exit code: (-?\d+)\]", re.MULTILINE)
#: The failure prefix of the host ``Bash`` tool: it opens the result.
HOST_EXIT_PREFIX = re.compile(r"Error \(exit code (-?\d+)\):")
#: How a ``Bash`` result opens when the call started a background job
#: (``background=True``) or left a foreground command running at its deadline,
#: with the job id.  The tools' notes follow the notice: the same words
#: further down are the command's own output.
STARTED_JOB = re.compile(
    r"(?:Started|Still running after \S+ left running as) background job (\w+)")
#: The status line of a ``bash_job`` report showing the job's exit: ``status:
#: exited N`` (the container tool) or ``Job <id>: exited with code N`` (the
#: host tool).  Only that line counts, read where the report puts it (its
#: first line, after the container report's echo of the command): the
#: command and the log lines of the report may say ``exited 0`` themselves.
JOB_EXITED = re.compile(r"^(?:status|Job \w+): exited (?:with code )?(-?\d+)$", re.MULTILINE)
#: The report the container ``Bash`` tool returns for ``on_timeout="background"``
#: in place of a start notice: ``Job <id>: <command>``, then the status line
#: (``exited N`` when the job ended within the call) and ``log: <dir>/<id>.log``.
#: The id is made when the job starts, so the first status line that log line
#: follows is the report's own, whatever the echo of the command before it
#: says: group 2 is the exit it shows, group 3 is set when the report holds
#: those two lines (a small ``max_output_chars`` cuts them off).
JOB_REPORT = re.compile(
    r"Job (\w+): (?:[\s\S]*?^status: (?:exited (-?\d+)|.*)\n(log): \S*/\1\.log$)?", re.MULTILINE)
#: The outcome of one command of a ``run_commands_parallel`` report:
#: ``[i/N] exit 0 in 1.2s``, ``[i/N] TIMED OUT``, ``[i/N] killed by signal 9``.
_PARALLEL_STATUS = r"\[(\d+)/(\d+)\] (?:exit (-?\d+)|(TIMED OUT)|killed by signal (\d+))"
#: The start of one section of the report, after its ``\n### `` (the header
#: ``[i/N] exit 0 in 1.2s``, then the ``$ `` of the command's echo).
PARALLEL_SECTION = re.compile(_PARALLEL_STATUS + r"[^\n]*\n\$ ")
#: The status line of one command in the preamble of the report's digest
#: (``[i/N] exit 0 in 1.2s: <command>``, ``[i/N] TIMED OUT after 9.0s:
#: <command>``; the full report was written to a file,
#: ``useful_tools.digest_parallel_report``).  Only the preamble, before the
#: first ``### `` section, is read for these: a command's own output may
#: hold such a line (a nested parallel run) and must not override its exit.
PARALLEL_DIGEST_STATUS = re.compile(
    "^" + _PARALLEL_STATUS + r" (?:in|after) [\d.]+s: ", re.MULTILINE,
)
#: Where the sections of a ``run_commands_parallel`` report or digest start.
PARALLEL_SECTIONS = "\n### "
#: A timed-out ``Bash`` result: the host tool's and the container tool's wording.
TIMED_OUT = re.compile(r"^Error: [Cc]ommand (?:execution timeout|timed out)")
#: A ``Bash`` result of a command the tool did not run (a guard's refusal).
REFUSED = re.compile(r"^REFUSED\b")
#: The result of a tool call the user stopped (``tool_interrupt``): the
#: command did not run to completion.
INTERRUPTED = re.compile("^" + re.escape(USER_INTERRUPTED_MESSAGE))
#: How a result opens when the command did not run: the tools' ``Error: ``
#: (``Error (exit code N):`` is read before this) and the framework's
#: ``Failed to call <tool> with`` (the tool raised before running it).  The
#: shapes, not the words: a passing command's output may open with ``Errors:
#: 0`` or ``Failed to call count: 0``.
FAILED = re.compile(r"Error(?::| \(exit code)|Failed to call \w+ with")
#: The first line of the result of a shell command the ``Bash`` tool made as an
#: ``Edit`` / ``Write`` through the agent's tool path (``ToolRedirect``): the
#: redirected call's result follows it, then the follow-up call running the
#: rest of the command (named by :data:`FOLLOWUP_LINE` with its result, or by
#: :data:`FOLLOWUP_NOT_MADE` when the redirected call did not go through).
REDIRECTED = re.compile(r"(?:Applied as|Appended through) .*, whose result follows:\n")
FOLLOWUP_LINE = re.compile(r"^Then made .*, whose result follows:\n", re.MULTILINE)
FOLLOWUP_NOT_MADE = re.compile(r"^.* was not made: the \w+ did not go through\.$", re.MULTILINE)
#: A shell's option under which ``bash <option> v.sh`` still runs ``v.sh``: a
#: switch that takes no value and keeps the first operand the script (``-x``,
#: ``-e``, ``--posix``).  Not ``-n`` or ``-D`` (nothing runs), ``-s`` (the
#: script comes from stdin, ``v.sh`` is ``$1``), ``-c`` (the operand is shell
#: text), ``-o`` / ``-O`` / ``--rcfile`` (the next word is their value).
SHELL_SWITCH = re.compile(r"-[xeuvlpET]+|--?|--(?:posix|norc|noprofile|login|verbose)")


def leaves_shell(command: SimpleCommand) -> bool:
    """Whether the simple *command* may keep the commands after it from running.

    ``exit`` and ``return`` end the shell; so does ``exec`` of a program
    (the shell becomes it), unlike a bare ``exec 2>&1``; after ``set -n``
    (``set -o noexec``) the shell reads the rest without running it.

    Args:
        command: A scanned simple command.
    """
    program = command.program()
    if program in ("exit", "return"):
        return True
    if program == "set":
        return _sets_noexec([word.text for word in command.argv()[1:]])
    return bool(command.words) and command.words[0].text == "exec" and bool(command.argv())


def _sets_noexec(args: list[str]) -> bool:
    """Return whether ``set args`` leaves the shell's noexec option on.

    ``-n`` (clustered too: ``-xn``) and ``-o noexec`` turn it on, ``+n``
    and ``+o noexec`` off, in order; ``--`` or ``-`` ends the options,
    and so does the first word that is no option: what follows is the
    positional parameters (``set -- -n``).  A ``--word`` is no option of
    ``set`` (an error, the shell goes on).
    """
    noexec = False
    pending = ""  # the sign of a ``-o`` / ``+o`` waiting for its option name
    for arg in args:
        if pending:
            if arg == "noexec":
                noexec = pending == "-"
            pending = ""
        elif arg in ("--", "-") or arg.startswith("--") or arg[:1] not in ("-", "+"):
            break
        else:
            for flag in arg[1:]:
                if flag == "n":
                    noexec = arg[0] == "-"
                elif flag == "o":
                    pending = arg[0]
    return noexec


def verifier_command(commands: list[SimpleCommand], verifier_path: str) -> SimpleCommand | None:
    """The first of the scanned *commands* that runs the verifier at *verifier_path*, or ``None``.

    ``bash ./tmp/<id>/verify.sh`` (``sh``, with no option but the
    switches of :data:`SHELL_SWITCH`: ``bash -n`` parses the script
    without running it, ``bash -s`` reads one from stdin instead),
    or the script itself as the command, from the work dir or after
    a ``cd`` the scanner follows (the two paths are compared as the
    filesystem resolves them: ``sub/../verify.sh`` is another file when
    ``sub`` is a symbolic link).  ``--`` or ``-`` ends the options:
    the word after it is the script whatever it starts with (``bash
    -- -x verify.sh`` runs a file named ``-x`` with the verifier as
    its argument).

    Args:
        commands: The simple commands of a shell command
            (:func:`~kiss.agents.sorcar.shell_scan.simple_commands`).
        verifier_path: The verifier's absolute path.
    """
    verifier = os.path.realpath(verifier_path)
    for simple in commands:
        argv = simple.argv()
        if not argv:
            continue
        words = argv[1:] if posixpath.basename(argv[0].text) in SHELLS else argv[:1]
        at = 0  # the words before the first operand are the shell's options
        while at < len(words) and words[at].text.startswith("-"):
            at += 1
            if words[at - 1].text in ("--", "-"):
                break
        if not all(SHELL_SWITCH.fullmatch(word.text) for word in words[:at]):
            continue
        path = resolve(simple.cwd, words[at]) if at < len(words) else None
        if path is not None and os.path.realpath(host_path(path)) == verifier:
            return simple
    return None


def verifier_decides(command: str, work_dir: str, verifier_path: str) -> bool:
    """Whether an exit status 0 of the shell *command* means its run of the verifier passed.

    So when the verifier's command is at the top level, not negated,
    its output not piped, not after ``||``, not in the background,
    nothing before it leaves the shell (``exit``, ``return``, an
    ``exec`` of a program) and everything after it follows ``&&``
    (``bash v.sh && echo ok`` fails when the verifier fails): ``bash
    v.sh || true``, ``bash v.sh | cat``, ``true || bash v.sh``,
    ``bash v.sh; echo done``, ``bash v.sh &``, ``! bash v.sh`` and
    ``exit 0; bash v.sh`` all exit 0 whatever the verifier did, so
    their exit shows nothing about it.

    Args:
        command: The command as the agent ran it.
        work_dir: The directory the command started in.
        verifier_path: The verifier's absolute path.
    """
    try:
        commands = simple_commands(command, bash_path(os.path.abspath(work_dir)))
    except TooDeepError:  # the commands past the nesting are unread: they may decide the exit
        return False
    simple = verifier_command(commands, verifier_path)
    if (simple is None or simple.group is not None or simple.reader is not None
            or simple.conditional or simple.background or simple.negated):
        return False
    at = next(index for index, candidate in enumerate(commands) if candidate is simple)
    if any(leaves_shell(earlier) for earlier in commands[:at]):
        return False
    for later in commands[at + 1:]:
        top: SimpleCommand | Group = later
        while top.group is not None:
            top = top.group
        if not top.after_and or top.background:
            return False
    return True


def tool_result_texts(message: Any) -> list[str]:
    """The tool-result texts carried by *message*.

    Handles the message shapes KISS models keep in their conversation: the
    OpenAI Responses item ``{"type": "function_call_output", "output": str}``,
    the chat-completions message ``{"role": "tool", "content": str}`` and the
    Anthropic user message whose ``content`` list holds ``tool_result``
    blocks (string or list content).

    Args:
        message: One conversation message.

    Returns:
        One text per tool result in the message (the text parts of a
        result that also carries an image are joined); empty for other
        messages.
    """
    if not isinstance(message, dict):
        return []
    if message.get("type") == "function_call_output" and isinstance(message.get("output"), str):
        return [message["output"]]
    if message.get("role") == "tool" and isinstance(message.get("content"), str):
        return [message["content"]]
    texts: list[str] = []
    if message.get("role") == "user" and isinstance(message.get("content"), list):
        for block in message["content"]:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            inner = block.get("content")
            if isinstance(inner, str):
                texts.append(inner)
            elif isinstance(inner, list):
                texts.append("\n".join(str(part.get("text", "")) for part in inner
                                       if isinstance(part, dict) and part.get("type") == "text"))
    return texts


def bash_exit_code(text: str, container: bool = False) -> str:
    """The exit code a ``Bash`` result *text* shows, as a string.

    The host tool opens a failed result with ``Error (exit code N):`` and
    the container tool closes one with a ``[exit code: N]`` line; a
    killed-at-the-deadline result is ``"timeout"``, a guard's refusal (the
    command never ran) ``"refused"``, a call the user stopped
    ``"interrupted"``, any other result the tool opens with ``Error`` (a
    rejected argument, a machine that could not be reached: the command
    did not run) or the framework with ``Failed to call`` (the tool raised
    before running it: a missing argument) ``"error"``, a command made as
    an ``Edit`` / ``Write`` the exit :func:`redirected_exit` reads, and a
    result with none of these ``"0"``.  Only the sign of the tool that
    returned the result counts: the other tool's, like a marker inside a
    line, is the command's own output (a nested tool's, a fixture).

    Args:
        text: The tool's result.
        container: Whether the container tool returned it, not the host's.
    """
    if REDIRECTED.match(text):
        return redirected_exit(text, container)
    if TIMED_OUT.match(text):
        return "timeout"
    if REFUSED.match(text):
        return "refused"
    if INTERRUPTED.match(text):
        return "interrupted"
    if container:
        markers = EXIT_CODE_MARKER.findall(text)
        if markers:
            return str(markers[-1])
    elif prefix := HOST_EXIT_PREFIX.match(text):
        return prefix.group(1)
    return "error" if FAILED.match(text) else "0"


def redirected_exit(text: str, container: bool = False) -> str:
    """The exit code of a shell command the ``Bash`` tool made as an ``Edit`` / ``Write``.

    *text* opens with :data:`REDIRECTED`; the redirected call's result
    follows that line and, when the rest of the command ran as a follow-up
    call, the :data:`FOLLOWUP_LINE` and the follow-up's result (itself of
    this shape when it wrote another file).  The command's exit is the
    follow-up's (:func:`bash_exit_code`, so nested redirects read through);
    without a follow-up it is ``"0"`` when the redirected call went through
    and ``"error"`` when it did not (its result opens with ``Error`` or
    ``Failed to call``, is a refusal, or the line after it says the
    follow-up was not made).

    Args:
        text: The ``Bash`` result.
        container: As for :func:`bash_exit_code`.
    """
    rest = text.split("\n", 1)[1] if "\n" in text else ""
    followup = FOLLOWUP_LINE.search(rest)
    if followup:
        return bash_exit_code(rest[followup.end():], container)
    if FAILED.match(rest) or REFUSED.match(rest) or FOLLOWUP_NOT_MADE.search(rest):
        return "error"
    return "0"


def job_report(args: dict[str, Any], text: str) -> re.Match[str] | None:
    """The job report (:data:`JOB_REPORT`) a ``Bash`` call with *args* returned as *text*.

    Only a call made with ``on_timeout="background"`` returns one, and it
    opens the result: it is read before :data:`STARTED_JOB`, which the
    job's log after it may hold.  It is told from a command's own output
    (the host tool's result for such a call) by its status and log lines
    or, cut short of them, by its echo of the command.  ``None`` for any
    other result.
    """
    if args.get("on_timeout") != "background":
        return None
    report = JOB_REPORT.match(text)
    if report and (report.group(3) or text.startswith(f"{args.get('command')}\n", report.end())):
        return report
    return None


def bash_result(args: dict[str, Any], text: str, container: bool = False) -> tuple[str, str]:
    """How the command of a ``Bash`` call ended, read in its result *text*, and its job.

    The result is read as the conversation recorded it: the framework's
    own text, a container job's report, or the command's output inside
    the tool's status.

    Args:
        args: The call's arguments.
        text: The call's result.
        container: As for :func:`bash_exit_code`.

    Returns:
        ``(exit code, job id)``: ``background`` and the job for one started,
        by the notice (:data:`STARTED_JOB`) that opens the result of a call
        that can return it or in a :func:`job_report` (there, the exit of a
        job that ended within the call, with its id); else
        :func:`bash_exit_code` and ``""``.
    """
    report = job_report(args, text)
    if report:  # the exit of a job that ended within the call, else a job left running
        return report.group(2) or "background", report.group(1)
    # A notice opens the result of the call that ran the command: the
    # follow-up's, when the command was made as an ``Edit`` / ``Write`` first.
    ran = text
    while REDIRECTED.match(ran) and (followup := FOLLOWUP_LINE.search(ran)):
        ran = ran[followup.end():]
    started = STARTED_JOB.match(ran)
    # ``Started ...`` is the answer to ``background=True`` only: a foreground
    # command whose output opens with the line has printed it.
    if started and (args.get("background") or not ran.startswith("Started")):
        return "background", started.group(1)
    return bash_exit_code(text, container), ""


def result_exit_codes(name: str, args: dict[str, Any], text: str,
                      jobs: dict[str, str], container: bool = False) -> list[tuple[str, str]]:
    """The ``(command, exit code)`` pairs the result *text* of a tool call shows.

    A command's own output is told from the tool's status only as far as
    the text allows.

    Args:
        name: The tool called.
        args: The call's arguments.
        text: The call's result.
        jobs: The commands of the background jobs started so far, by job id;
            a ``Bash`` result that started one adds it.
        container: Whether the run's tools are the container's, not the
            host's (see :func:`bash_exit_code`).

    Returns:
        For ``Bash``, its command with the exit :func:`bash_result` reads
        (a job it names is added to *jobs*); for
        ``bash_job``, the job's command with the exit the status line of its
        report shows (``exited N``; nothing while it runs, whatever the
        command echo and the log lines say); for ``run_commands_parallel``,
        each command with the exit of its ``### [i/N]`` section (``timeout``,
        or ``killed by signal N``), or ``interrupted`` for every command
        when the user stopped the call.  Nothing for any other tool.

        The report's sections are read in order and each must be the
        next one (``[i/N]`` with ``N`` the number of commands) and echo
        its command: a section a command printed itself (a nested
        parallel run in its output) is passed over, not read as the
        command's exit.  A digest's preamble lists every command's status,
        so its sections (the failed commands' tails) are not read.
    """
    if name == "Bash" and isinstance(args.get("command"), str):
        code, job_id = bash_result(args, text, container)
        if job_id:
            jobs[job_id] = args["command"]
        return [(args["command"], code)]
    if name == "bash_job":
        job_id = str(args.get("job_id", ""))
        command = jobs.get(job_id)
        if command is None:
            return []
        exited = JOB_EXITED.match(text.removeprefix(f"Job {job_id}: {command}\n"))
        return [] if exited is None else [(command, exited.group(1))]
    if name == "run_commands_parallel":
        commands = shell_commands(args)
        if INTERRUPTED.match(text):
            return [(command, "interrupted") for command in commands]
        preamble, *sections = text.split(PARALLEL_SECTIONS)
        found = PARALLEL_DIGEST_STATUS.findall(preamble)
        if not found:
            for section in sections:
                index = len(found) + 1
                header = PARALLEL_SECTION.match(section)
                if (header is not None and index <= len(commands)
                        and (int(header.group(1)), int(header.group(2))) == (index, len(commands))
                        and section.startswith(commands[index - 1] + "\n", header.end())):
                    found.append(header.groups(""))
        codes = {int(index): code or ("timeout" if timed_out else f"killed by signal {signal}")
                 for index, _total, code, timed_out, signal in found}
        return [(command, codes[index])
                for index, command in enumerate(commands, 1) if index in codes]
    return []


def evidence_commands(raw: Any) -> list[str]:
    """The commands a ``finish`` call's ``evidence`` argument names.

    Args:
        raw: The argument: a JSON list of strings, a list, or a plain string
            (one command).  The JSON is decoded as the tools decode their
            lists of commands (a lone backslash, grep's ``\\|``, is no
            error); text nested past the decoder's depth is a plain string.

    Returns:
        The non-empty command strings; empty when the field is empty or not
        a list of strings.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = _decode_json_leniently(text)
        except (ValueError, RecursionError):
            raw = [text]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]


def review_findings(answer: str) -> str:
    """The gaps a reviewer's *answer* lists, or ``""`` when it says ``VERDICT: OK``.

    An answer without a ``VERDICT:`` line yields no findings: the caller
    retries it once and then lets the run go (a confused reviewer must not
    hold the run, nor count as approval in the log).  So does a ``NOT
    DONE`` with nothing after it.

    Args:
        answer: The reviewer model's reply to the review prompt.
    """
    match = REVIEW_VERDICT_LINE.search(answer)
    if match is None or match.group(1).upper() == "OK":
        return ""
    return answer[match.end():].strip()


def numbered_items(findings: str) -> list[str]:
    """The numbered items of a reviewer's *findings* text, one string each.

    A line starting with ``N.`` or ``N)`` opens an item; continuation lines
    belong to the item before them.  Findings without numbers are one item.

    Args:
        findings: The text after ``VERDICT: NOT DONE``.
    """
    items: list[str] = []
    for line in findings.splitlines():
        if not line.strip():
            continue
        if re.match(r"\s*\d+[.)]\s", line) or not items:
            items.append(line.strip())
        else:
            items[-1] += " " + line.strip()
    return items


def answered_findings(summary: str, findings: list[str],
                      on_usage: Callable[[float, int], None] | None = None) -> list[bool] | None:
    """Whether *summary* answers each of *findings* with evidence, per the decisions model.

    One :func:`~kiss.agents.sorcar.decide_tool.make_decide_tool` call asks
    :data:`ANSWER_QUESTION` per finding; a finding counts as answered at a
    probability of :data:`ANSWERED_THRESHOLD` or more.  ``None`` without
    the decisions model (or when the call fails): the caller decides what
    an unjudged finding is (a refusal must not depend on an optional
    service; a caveat is shown).

    Args:
        summary: The finish summary.
        findings: The open findings (:func:`numbered_items`).
        on_usage: Called with the call's ``(cost USD, tokens)`` (the gate
            charges it to the run), or ``None``.
    """
    from kiss.agents.sorcar.decide_tool import decisions_tool_available, make_decide_tool

    if not findings:
        return []
    if not decisions_tool_available():
        return None
    questions = {f"f{i}": {"type": "noul", "instructions": ANSWER_QUESTION.format(finding=f)}
                 for i, f in enumerate(findings)}
    try:
        decide = make_decide_tool(None, on_usage=on_usage)
        answers = json.loads(decide(summary, json.dumps(questions)))["answers"]
    except Exception:
        return None
    answered = []
    for i in range(len(findings)):
        result = answers.get(f"f{i}") if isinstance(answers, dict) else None
        value = result.get("noul") if isinstance(result, dict) else None
        answered.append(not isinstance(value, int | float) or value >= ANSWERED_THRESHOLD)
    return answered


def gate_refusal(round_number: int, max_rounds: int, verdicts: list[str]) -> str:
    """The text of a refused finish: the round count and every open verdict.

    Args:
        round_number: This refusal's number within the run (from 1).
        max_rounds: Refused rounds the run gets before delivery with caveats.
        verdicts: The verdicts of the stages that refused, in gate order.
    """
    header = (f"Finish refused (round {round_number} of up to {max_rounds}; every stage ran, "
              "fix everything below before calling finish again; a later round is refused "
              "only while your verifier fails, otherwise the work goes through with the open "
              "findings as caveats):")
    return "\n\n".join([header, *verdicts])


#: Seconds between the checks for a Stop while the gate waits on its stage
#: threads and on a side call's deadline.  Short slices keep the waits in
#: Python, where an injected Stop lands; a bare ``join()`` blocks in C for
#: the whole deadline.
STOP_POLL_SECONDS = 0.2


#: Model overrides that set the effort or the thinking budget of a call: the
#: framework's ``reasoning_effort``, Anthropic's ``thinking``, OpenAI's
#: ``reasoning`` and Gemini's ``thinking_config``; a side call drops them.
EFFORT_KEYS = frozenset({"reasoning_effort", "thinking", "reasoning", "thinking_config"})


def default_effort(model_config: dict[str, Any] | None) -> dict[str, Any] | None:
    """*model_config* without its reasoning-effort override, for a side call.

    The run's overrides carry what a model needs to be reached at all (a
    ``base_url``, an ``api_key``) and the effort the run was asked for;
    a side call (the review, the reading) keeps the former and runs at
    the provider's default effort.

    Args:
        model_config: The run's model overrides, or ``None``.

    Returns:
        The overrides without ``reasoning_effort`` and without the
        providers' own effort settings (:data:`EFFORT_KEYS`, and
        ``effort`` inside Anthropic's ``output_config``), or ``None`` when
        nothing remains.
    """
    kept = {key: value for key, value in (model_config or {}).items()
            if key not in EFFORT_KEYS}
    output_config = kept.get("output_config")
    if isinstance(output_config, dict) and "effort" in output_config:
        output_config = {key: value for key, value in output_config.items() if key != "effort"}
        if output_config:
            kept["output_config"] = output_config
        else:
            del kept["output_config"]
    return kept or None


#: The effort of the one retry of a review attempt that failed (:func:`low_effort`).
REVIEW_RETRY_EFFORT = "low"
#: The errors of a review attempt that one retry at low effort may fix, by the
#: class name its ``repr`` starts with: an answer the output cap cut short and a
#: refusal (the content filter's included).  The deadline is the fourth kind.
RETRIED_REVIEW_ERRORS = ("OutputCapError", "ModelRefusalError", "ContentFilterError")
DEADLINE_ERROR = "no answer within"


def low_effort(model_config: dict[str, Any] | None, model_name: str) -> dict[str, Any] | None:
    """*model_config* for the retry of a failed review: the default effort, lowered.

    :func:`default_effort` of the overrides plus ``reasoning_effort``
    :data:`REVIEW_RETRY_EFFORT` (the framework's key, which every adapter
    maps to its provider's effort setting) when the model's catalog entry
    carries a ``thinking`` level, as :meth:`KISSAgent._retry_at_low_effort`
    does; a model that takes no effort parameter is retried as it was
    (a provider rejects the parameter on such a model).

    Args:
        model_config: The run's model overrides, or ``None``.
        model_name: The reviewer model.
    """
    from kiss.core.models.model_info import MODEL_INFO

    kept = dict(default_effort(model_config) or {})
    info = MODEL_INFO.get(model_name)
    if info is not None and info.thinking:
        kept["reasoning_effort"] = REVIEW_RETRY_EFFORT
    return kept or None


def retryable_review_failure(answer: str, error: str) -> bool:
    """Whether a review attempt failed in a way one retry at low effort may fix.

    An empty answer, an answer the output cap cut short, a refusal or
    the deadline (:data:`RETRIED_REVIEW_ERRORS`, :data:`DEADLINE_ERROR`).
    Any other error (a credential, the network) is not retried, and an
    answer without a verdict line is a judgement the gate does not ask
    for twice.

    Args:
        answer: The attempt's answer (``""`` for none).
        error: The attempt's error (``""`` when the call completed).
    """
    if error:
        return error.startswith(RETRIED_REVIEW_ERRORS) or error.startswith(DEADLINE_ERROR)
    return not answer.strip()


def call_stop_event(seconds: float, done: threading.Event) -> threading.Event:
    """An event set after *seconds*, or when the calling thread's own stop event is set.

    Bound as the stop event of the thread that makes a side call
    (:mod:`kiss.core.stop_signal`), it gives the model call a deadline
    without losing the run's Stop: a watcher thread polls both until
    *done* is set (the call ended) or it fires.

    Args:
        seconds: The deadline.
        done: Set by the caller when the call has ended, which ends the watcher.
    """
    stop = threading.Event()
    threading.Thread(target=_watch_call, name="side-call-deadline", daemon=True,
                     args=(stop, done, stop_signal.get_thread_stop_event(),
                           time.monotonic() + seconds)).start()
    return stop


def _watch_call(stop: threading.Event, done: threading.Event,
                previous: threading.Event | None, end: float) -> None:
    """The watcher of :func:`call_stop_event`: sets *stop* at *end* (monotonic) or on *previous*."""
    while not done.wait(STOP_POLL_SECONDS):
        if (previous is not None and previous.is_set()) or time.monotonic() >= end:
            stop.set()
            return
