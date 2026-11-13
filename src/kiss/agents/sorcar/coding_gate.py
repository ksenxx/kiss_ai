# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Coding gate: the coding SEA's finish gate, for every Sorcar run that changes code.

The containerised coding SEA (:mod:`kiss.agents.seas.coding.coding_sea`)
checks a ``finish`` mechanically before the run ends: the agent's own
verifier runs (or its own passing run is reused), a reviewer of another
model family reads the diff once against the task, the summary and the
commands named as evidence, a give-up is checked the same way as a success
claim, and a refused finish gets one more round, after which only a failing
verifier refuses again and the work is delivered with the open findings as
caveats.  The gate exists because success claims and give-ups are both
unreliable and a reviewer of the executor's own family shares its reading.
What it no longer does, after the measurement of 2026-11-04: refuse on
the wording of the summary, on the ``evidence`` field, on a blank review
or on review findings past the first refused round, and review the same
work again after an approval; those refusals cost rounds of work at full
context and did not change what the hidden tests found.

This module is the part of that gate that depends on nothing in the
benchmark container: :class:`CodingGate`, which
:class:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent` installs on every run
with the edit tools, and the readers of shell results both the gate and
the SEA use.  A run is a *coding run* once an ``Edit`` or ``Write`` targets
a code file (:func:`is_code_file`), a shell command changes one (the
files it changes, :func:`~kiss.agents.sorcar.evidence_gate.file_changes`:
``cp a.py b.py``, ``sed -i ... x.py``, ``python gen.py > out.py``, not a
read of one or a write under the scratch directory), or the verifier
exists; a run that writes only prose is not gated.  The verifier is
``verify.sh`` in the task's scratch directory, the path the system prompt
gives the agent (How to Work, rule 3).  The gate runs it with the run's
own shell: a subprocess in the work dir (:func:`run_on_host`), or the
task's container.  When the agent writes
the ledger the prompt asks for (``ledger.md`` in the scratch directory),
the reviewer model reads the task and the ledger once and names the
sentences it would read differently; the answer joins the ``Write``'s
result (:meth:`CodingGate.reading_note`).
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import html
import json
import logging
import os
import posixpath
import re
import shlex
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import yaml

from kiss.agents.sorcar import shell_exit
from kiss.agents.sorcar.evidence_gate import (
    EDIT_TOOLS,
    SHELL_TOOLS,
    SHELLS,
    file_changes,
    runs_subcommand,
    shell_commands,
)
from kiss.agents.sorcar.fanout_guard import (
    _decode_json_leniently,
    is_review_task,
    parse_tasks_json,
)
from kiss.agents.sorcar.shell_scan import (
    Group,
    SimpleCommand,
    TooDeepError,
    bash_path,
    host_path,
    resolve,
    simple_commands,
)
from kiss.core import stop_signal
from kiss.core.event_sink import adopt_thread_context, publish, thread_context
from kiss.core.processes import (
    exit_status_unreaped,
    find_bash,
    kill_process_tree,
    pid_handed_on,
    start_process_group,
)
from kiss.core.tool_interrupt import USER_INTERRUPTED_MESSAGE

logger = logging.getLogger(__name__)

#: ``(exit code, output)`` of a command run by the gate; the code is ``None``
#: when the command did not finish within its timeout.
ShellRun = tuple[int | None, str]
Shell = Callable[[str, int], ShellRun]

#: The agent's own verifier, in the task's scratch directory.
VERIFIER_NAME = "verify.sh"
#: Seconds the verifier gets at a finish.
VERIFIER_TIMEOUT = 900
#: Finishes the verifier may refuse; past them the finish is delivered with a
#: caveat, so a check the agent cannot make pass does not trap the run.
MAX_VERIFIER_REFUSALS = 3
#: Refused rounds of the gate per run.  The first refusal buys one round
#: whatever refused it; a later round is refused only when the verifier
#: refused, up to this cap, past which the finish goes through with the open
#: verdicts as caveats (the pass rate was flat against the number of refused
#: rounds, and each round costs a model turn at full context).
MAX_GATE_ROUNDS = 3
#: Reviews per run: one, on the whole diff.  A re-review after an approval
#: changed its verdict in 13 of 164 cases, and the rounds after a rejection
#: did not bring the refused runs up to the average; a later finish is
#: judged on the review's findings by :meth:`CodingGate.unanswered_verdict`.
MAX_REVIEWS = 1
#: Reviews that produced no verdict (or failed) before no review is asked for again.
MAX_FAILED_REVIEWS = 2
#: Seconds the reviewer gets to answer.  The provider's prompt cache lives
#: about five minutes; a gate that outlasts it makes the agent's next call
#: cost about twelve times the cached price for the same context.
REVIEW_DEADLINE = 240
#: The ledger the system prompt asks the agent to write in the scratch
#: directory; its first ``Write`` gets the independent reading.
LEDGER_NAME = "ledger.md"
#: Seconds the diff listing for the reviewer may take.
DIFF_TIMEOUT = 60
#: A digest of the bytes on disk of the files tracked at the base commit (the
#: format field): what a conversion of Git's (``text``, ``eol``, a clean filter)
#: hides from ``git diff``.  Of the base's files, not the index's, so a worktree
#: and the checkout its merge brought the same files to digest the same set; of
#: the regular files under 2 MiB, which ``find`` picks from the paths made safe
#: for it by a ``./``: a script is no larger, a link is in the diff as its
#: text, and reading every large file for each digest would cost more than
#: the diff.  A shell without these tools digests nothing.
TRACKED_BYTES_DIGEST = (
    "git ls-tree -r --name-only -z {base} | xargs -0 printf './%s\\0' | "
    "xargs -0 sh -c 'find \"$@\" -prune -type f -size -2048k -print0' sh 2>/dev/null | "
    "xargs -0 cksum | git hash-object --stdin"
)
#: Largest diff shown to the reviewer, and the largest share of it one file gets.
REVIEW_FILE_BYTES = 120_000
REVIEW_ONE_FILE_BYTES = 30_000
#: Longest verifier output quoted in a verdict or shown to the reviewer.
OUTPUT_CHARS = 4000
#: The config key naming the reviewer: ``""`` (auto: another family's
#: strongest configured model, none when every configured model is of the
#: executor's family), ``"off"``, or a model name.
REVIEWER_CONFIG_KEY = "coding_gate_reviewer"

#: Suffixes and names of the files whose change makes a run a coding run.
CODE_SUFFIXES = frozenset({
    ".py", ".pyi", ".pyx", ".ipynb", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue",
    ".svelte", ".java", ".kt", ".kts", ".scala", ".groovy", ".go", ".rs", ".c", ".h", ".cc",
    ".cpp", ".cxx", ".hpp", ".hh", ".m", ".mm", ".cs", ".fs", ".swift", ".rb", ".php", ".pl",
    ".pm", ".lua", ".r", ".jl", ".dart", ".ex", ".exs", ".erl", ".hs", ".ml", ".mli", ".clj",
    ".cljs", ".elm", ".nim", ".zig", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".proto", ".graphql", ".thrift", ".cmake", ".mk", ".gradle", ".sbt", ".toml",
    ".yaml", ".yml", ".json", ".ini", ".cfg", ".tf", ".css", ".scss", ".less",
})
CODE_FILE_NAMES = frozenset({
    "Makefile", "GNUmakefile", "Dockerfile", "CMakeLists.txt", "Justfile", "Rakefile",
    "Gemfile", "Procfile", "BUILD", "WORKSPACE", "meson.build", "SConstruct",
})

#: Answer to an implicit (text-only) finish on a coding run: an explicit
#: ``finish`` call is asked for once, which the gate then checks.
IMPLICIT_FINISH_VERDICT = (
    "Not finished: a text reply does not end a run that changed code. Call finish "
    "explicitly: with success=True only if every requirement is met and verified by a "
    "command you ran (name those commands in the evidence argument; the gate runs your "
    "verifier and a review reads the evidence), otherwise with success=False and what remains."
)
VERIFIER_FAILED_VERDICT = (
    "Not finished: your own verifier {path} exited {code} (bash {path}):\n{output}\n"
    "Fix the work it found wrong (or the verifier, if it checks something the task does "
    "not ask for), re-run it, then call finish again."
)
VERIFIER_TIMEOUT_VERDICT = (
    "Not finished: your own verifier {path} did not finish within {seconds:.0f} seconds. "
    "Make it fast enough to run at the end (or run its slow part yourself and keep the "
    "quick checks in it), then call finish again."
)
#: Shown once when a give-up's verifier passes.
GIVEUP_RECONSIDER_VERDICT = (
    "Not accepted yet: you called finish(success=False), but your own verifier {verifier} "
    "passes on the current work tree. Either name the requirement that is still unmet (one "
    "sentence, with the command that shows it) and call finish(success=False) again, or call "
    "finish(success=True) with the summary of what was done and the evidence."
)
#: Appended to the summary of every give-up on a coding run: the verifier's
#: outcome next to the agent's verdict.
GIVEUP_NOTE = (
    "\n<p><b>Checks at this finish:</b> agent: failed; verifier {verifier}: {state}.</p>"
)
#: Appended to the summary of a finish delivered past the round cap.
DELIVERED_CAVEATS = (
    "\n<h4>Delivered after {rounds} refused finish rounds; open findings</h4>\n<ol>\n{items}\n</ol>"
)
#: The caveat of a finish delivered past the verifier-refusal cap: the
#: verifier did not run again.
UNVERIFIED_CAVEAT = (
    "The verifier {verifier} refused {refusals} finishes and was not run again; the work is "
    "delivered as it stands."
)
#: What the reviewer reads as the summary when the review runs at the merge of a worktree.
MERGE_SUMMARY_PLACEHOLDER = (
    "(not written yet: this review runs at the merge of the task's worktree, before the "
    "agent's finish)"
)
#: What the reviewer reads as the evidence at a merge-time review: the finish
#: call, not yet made, is what names the commands.
MERGE_EVIDENCE_PLACEHOLDER = (
    "(not named yet: the finish call after this merge names them; judge the diff and the "
    "verifier, and do not report the missing summary or evidence as a gap)"
)

#: The reviewer's brief: the task, the agent's summary, its verifier and
#: evidence, and the diff; one ``VERDICT:`` line and the numbered gaps.
REVIEW_PROMPT = """\
You are reviewing, read-only, the work of an autonomous engineer on the task below. \
Decide whether every requirement of the task is demonstrably met by the evidence (the \
engineer's summary, their verifier script and its last output, the commands they name as \
evidence, and the diff of the files they changed). The gate runs the verifier itself at \
this finish and refuses on a failure, so judge the script by what it covers, not by the \
output shown, which is from its most recent run before this review. The evidence section \
lists the commands the engineer names as backing the claim, each with the exit code of \
its last run in this session (`not run` for one that did not run); a command is evidence \
of a requirement only when it ran, exited 0 and tests that requirement: `echo`, `cat`, a \
script the engineer wrote themselves, or the words of the \
task prove nothing about the work on their own. When the summary and evidence sections \
say they are not written yet (a review at the merge of the worktree, before the finish), \
judge the diff and the verifier against the task and do not report their absence as a \
gap. Do not invent problems and never demand \
something the task does not ask for: report only a requirement that is not met, not \
checked, or checked wrongly, each with the evidence that shows it. A benefit the task \
expects from the change ("would cut context", "lowers latency", "prevents attention \
degradation") is the user's motivation, not a requirement: do not demand that the work \
measure or prove it, unless the task explicitly asks for that measurement or names a \
target value for it; the requirement is the change itself. Take the task \
literally (exact names, formats, values, thresholds, edge cases, examples). A symptom, \
scenario or input of the task that the summary admits is not handled ("I added nothing \
for", "left as is", "known limitation") is a gap, whatever the reason given, unless the \
task itself rules it out. Do not pad the answer: the verdict and the numbered gaps only.

Answer with one line `VERDICT: OK` when every requirement is demonstrably met, or \
`VERDICT: NOT DONE` followed by a numbered list of the gaps, each one sentence of \
the form "<requirement>: <what is missing or wrong, with evidence>".

# Task
{task}

# The engineer's final summary
{summary}

# The engineer's verifier ({verifier}), exit code {code}
```
{script}
```
Its last output:
```
{output}
```

# Commands the engineer names as evidence, with the exit code of their last run
{evidence}

# The diff of the files the engineer created or changed under {workdir}
{files}
"""
#: The independent reading of the task (B1 of the 2026-11-04 analysis): the
#: reviewer model reads the task and the agent's ledger once, when the
#: ledger is written, and names the sentences it would read differently.
#: Most lost runs are a reading the checks settle the other way, and every
#: check the agent writes starts from its own reading; this one does not.
READING_PROMPT = """\
An engineer is about to work on the task below and has written the interpretation \
ledger below: the readings they chose for the task's ambiguous sentences, the inputs \
they will use, the tolerances and formats they settled, and what "done" means to them. \
You have not seen the task before and you will not see the work. Read the task on its \
own, then the ledger, and list only the places where you would read the task differently \
from the ledger, or where the task asks for something the ledger does not settle: quote \
the sentence of the task, then say in one sentence how you read it and what the \
engineer's reading would get wrong (a file, a name, a format, a value, a case, an input \
left unused). Take the task literally. Do not restate readings you agree with, do not \
invent requirements the task does not make, and do not comment on the approach. A \
benefit the task expects from the change ("would cut context", "lowers latency", \
"prevents attention degradation") is the user's motivation, not a requirement: do not \
demand that the work measure or prove it, unless the task explicitly asks for that \
measurement or names a target value for it; the requirement is the change itself.

Answer with the single line `NONE` when you read every sentence as the ledger does, \
otherwise with a numbered list, one item per sentence.

# Task
{task}

# The engineer's interpretation ledger
```
{ledger}
```
"""
#: Model answers that mean "no different reading".
READING_NONE = re.compile(r"^\s*`?NONE`?\.?\s*$", re.I)
#: Appended to the result of the ledger's ``Write``: the readings to settle.
READING_NOTE = (
    "\n\nIndependent reading of the task ({model}, from the task and this ledger): it would "
    "read these sentences differently or finds them unsettled. Settle each in the ledger "
    "before the first edit: adopt the reading, or keep yours and write next to it the words "
    "of the task that rule the other out.\n{items}"
)
REVIEW_VERDICT = (
    "Not finished: an independent review of the work against the task found these gaps:\n"
    "{findings}\n"
    "Answer every numbered finding in your next finish: either change the work or quote "
    "in summary_in_html, under the finding's number, the command you ran and its output "
    "that shows the finding wrong. An argument without a change or quoted evidence does "
    "not count."
)
#: A finding the next finish neither changed code for nor answered with evidence.
UNANSWERED_VERDICT = (
    "Not finished: these findings of the review were not answered by a change or by "
    "quoted command output:\n{findings}\n"
    "Change the work, or quote under each number the command and output that shows "
    "the finding wrong, then call finish again."
)
#: Appended to the result of a merge let through at the round cap: the open
#: verdicts, which the finish summary carries as caveats unless it answers them.
MERGED_WITH_FINDINGS = (
    "\n\nMerged after {rounds} refused rounds with these findings open:\n{items}\n"
    "They are appended to the finish summary as caveats, except those the summary answers "
    "by quoting the command and output that shows the finding wrong."
)
#: Appended to the summary of a finish after a merge that went through at the cap.
MERGED_CAVEATS = (
    "\n<h4>Merged after {rounds} refused merge rounds; findings open at the merge</h4>\n<ol>\n"
    "{items}\n</ol>"
)
#: The last line of every merge refusal: what happens to the next merge.
MERGE_CAP_NOTE = (
    "A merge asked for again goes through with its open findings attached to the result "
    "(and to the finish summary) unless your verifier fails, so fix them now."
)
#: A merge asked for while the last review's findings are open and nothing changed.
MERGE_UNANSWERED_VERDICT = (
    "Not merged: these findings of the review are still open and no file changed since:\n"
    "{findings}\n"
    "Fix the work in the worktree and call commit_and_merge again; a finding you have shown "
    "wrong is answered at the finish, by quoting under its number the command and output "
    "that shows it."
)
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
#: The first line of a ``run_parallel`` digest (the children's YAML was
#: written to a file, ``agent_dispatch.digest_agent_results``), and one
#: sub-task's line of it with its outcome.
AGENT_DIGEST_HEADER = re.compile(r"\A\d+ sub-tasks; the whole result is in ")
AGENT_DIGEST_LINE = re.compile(r"^\[\d+/\d+\] (ok|FAILED|running)\b", re.MULTILINE)
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
#: The argument of a shell tool its exits are handed over for (``shell_exit.take``).
ASKED = {"Bash": "command", "run_commands_parallel": "commands"}
#: The tools that run a sub-task: a read-only review (:func:`dispatches_review`)
#: or work that may change files.
DISPATCH_TOOLS = frozenset({"run_agent", "run_parallel"})
#: The tool that restores the files of a checkpoint: a change of the work it
#: does not name file by file (:meth:`CodingGate.restored`).
ROLLBACK_TOOL = "rollback"
#: A ``run_agent`` result for a sub-task left running (``wait="false"``, or
#: its timeout passed), with the job id ``agent_job`` takes; the digest of a
#: ``run_parallel`` result names such a child ``running (job <id>)``.
STARTED_AGENT_JOB = re.compile(r"(?:\bas |running \()job (agent-\w+)\b")
#: A shell's option under which ``bash <option> v.sh`` still runs ``v.sh``: a
#: switch that takes no value and keeps the first operand the script (``-x``,
#: ``-e``, ``--posix``).  Not ``-n`` or ``-D`` (nothing runs), ``-s`` (the
#: script comes from stdin, ``v.sh`` is ``$1``), ``-c`` (the operand is shell
#: text), ``-o`` / ``-O`` / ``--rcfile`` (the next word is their value).
SHELL_SWITCH = re.compile(r"-[xeuvlpET]+|--?|--(?:posix|norc|noprofile|login|verbose)")
#: ``git``'s subcommands that move HEAD: a commit or merge of the run's own.
HEAD_MOVING = {"git": re.compile(
    r"(?:commit|merge|pull|rebase|cherry-pick|revert|am|reset|checkout|switch)\b")}


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


def moves_head(command: str) -> bool:
    """Whether the shell *command* runs a git command that moves HEAD (:data:`HEAD_MOVING`).

    Args:
        command: The command as the agent ran it.
    """
    return any(runs_subcommand([word.text for word in simple.argv()], HEAD_MOVING)
               for simple in simple_commands(command, None, optimistic=True))


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


def verifier_ends(command: str, work_dir: str, verifier_path: str) -> bool:
    """Whether the shell *command*'s exit is its verifier run's and nothing runs after it.

    Then a passing run of *command* saw the tree as the call left it
    (:func:`verifier_decides` lets ``bash v.sh && bash rewrite.sh``
    through, whose second script may write a file no scan shows).

    Args:
        command: The command as the agent ran it.
        work_dir: The directory the command started in.
        verifier_path: The verifier's absolute path.
    """
    if not verifier_decides(command, work_dir, verifier_path):
        return False
    commands = simple_commands(command, bash_path(os.path.abspath(work_dir)))
    return verifier_command(commands, verifier_path) is commands[-1]


def is_code_file(path: Any) -> bool:
    """Whether *path* names a code, build or configuration file (:data:`CODE_SUFFIXES`).

    Args:
        path: The ``file_path`` argument of an ``Edit`` or ``Write`` call.
    """
    if not isinstance(path, str):
        return False
    name = posixpath.basename(path.replace("\\", "/"))
    return name in CODE_FILE_NAMES or posixpath.splitext(name)[1].lower() in CODE_SUFFIXES


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


def dispatch_successes(text: str) -> list[bool]:
    """The ``success`` of every sub-task result in a dispatch tool's result *text*.

    Args:
        text: The result of ``run_agent`` or ``agent_job`` (the sub-task's
            YAML ``ran`` / ``success`` / ``summary`` mapping) or of
            ``run_parallel`` (a YAML list of such texts).

    Returns:
        One flag per sub-task result found; empty for a job notice (the
        sub-task still runs), an error, or any other text.  A
        ``run_parallel`` digest (the YAML of the children was too long
        for the conversation and went to a file) is read from its
        ``[i/N] ok`` / ``[i/N] FAILED`` lines; a child still running has
        no flag.
    """
    if AGENT_DIGEST_HEADER.match(text):
        return [outcome == "ok" for outcome in AGENT_DIGEST_LINE.findall(text)
                if outcome != "running"]
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError:
        return []
    successes: list[bool] = []
    for item in loaded if isinstance(loaded, list) else [loaded]:
        if isinstance(item, str):
            try:
                item = yaml.safe_load(item)
            except yaml.YAMLError:
                continue
        if isinstance(item, dict) and "success" in item:
            successes.append(item["success"] is True)
    return successes


def started_agent_jobs(text: str) -> list[str]:
    """The ids of the jobs a ``run_agent`` / ``run_parallel`` result *text* leaves running.

    Each is named by its notice or by its line of the digest
    (:data:`STARTED_AGENT_JOB`).  The YAML list ``run_parallel`` returns is
    read decoded: its dump folds and escapes the long lines of the notices.
    """
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError:
        loaded = None
    notices = "\n".join(map(str, loaded)) if isinstance(loaded, list) else text
    return STARTED_AGENT_JOB.findall(notices)


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

    For a result no tool handed an exit over for
    (:mod:`~kiss.agents.sorcar.shell_exit`): the framework's own text, a
    container job's report, or a result recorded in a conversation.

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

    The reading of a result whose tool handed no exit over as data (see
    :meth:`CodingGate.result`); a command's own output is told from the
    tool's status only as far as the text allows.

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
    DONE`` with nothing after it: the caller tells the two apart with
    :func:`review_complete`.

    Args:
        answer: The reviewer model's reply to :data:`REVIEW_PROMPT`.
    """
    match = REVIEW_VERDICT_LINE.search(answer)
    if match is None or match.group(1).upper() == "OK":
        return ""
    return answer[match.end():].strip()


def review_complete(answer: str) -> bool:
    """Whether a reviewer's *answer* is a complete review: a verdict, plus gaps for ``NOT DONE``.

    A ``VERDICT: NOT DONE`` with nothing after it is a cut-off answer, not
    a rejection: without the gaps there is nothing to answer, and reading it
    as an empty list of findings would record it as an approval.

    Args:
        answer: The reviewer model's reply to :data:`REVIEW_PROMPT`.
    """
    match = REVIEW_VERDICT_LINE.search(answer)
    return match is not None and (
        match.group(1).upper() == "OK" or bool(answer[match.end():].strip()))


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


def gate_refusal(round_number: int, max_rounds: int, verdicts: list[str],
                 call: str = "finish") -> str:
    """The text of a refused finish (or merge): the round count and every open verdict.

    Args:
        round_number: This refusal's number within the run (from 1).
        max_rounds: Refused rounds the run gets before delivery with caveats.
        verdicts: The verdicts of the stages that refused, in gate order.
        call: The refused tool: ``finish`` or ``commit_and_merge``.
    """
    what = "Merge" if call == "commit_and_merge" else "Finish"
    header = (f"{what} refused (round {round_number} of up to {max_rounds}; every stage ran, "
              f"fix everything below before calling {call} again; a later round is refused "
              "only while your verifier fails, otherwise the work goes through with the open "
              "findings as caveats):")
    parts = [header, *verdicts]
    if call == "commit_and_merge":
        parts.append(MERGE_CAP_NOTE)
    return "\n\n".join(parts)


#: Seconds between the checks for a Stop while the gate waits: on a host
#: command (:func:`run_on_host`) and on its stage threads
#: (:meth:`CodingGate.run_stages`).  Short slices keep the waits in Python,
#: where an injected Stop lands; a bare ``communicate()`` or ``join()``
#: blocks in C for the whole deadline.
STOP_POLL_SECONDS = 0.2
#: Seconds :func:`run_on_host` waits for its killed command to be reaped and
#: for the pipe of a descendant that survived the kill to close.
REAP_GRACE_SECONDS = 2.0


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


def run_on_host(command: str, work_dir: str, timeout: int,
                env: dict[str, str] | None = None) -> ShellRun:
    """Run *command* with bash in *work_dir*: its exit code and combined output.

    The command runs in a process group of its own, killed whole (a
    verifier's test runner included) when the deadline passes, when the
    calling thread's stop event (:mod:`kiss.core.stop_signal`: the task's
    Stop, or the gate cancelling a stage) is set meanwhile, or when an
    exception cuts the wait short (a Stop landing as ``KeyboardInterrupt``);
    the output read by then is kept.  A command that has ended is not
    waited for either while a process it left behind still holds its
    output (a verifier that leaves a server running): its exit code is
    returned and what it left in its group is killed.  The wait for the
    killed group is bounded and its pipe is closed rather than read to
    its end, so a descendant that escaped into a session of its own holds
    nothing up.

    Args:
        command: The shell command.
        work_dir: Its working directory.
        timeout: Seconds it may run; past them, or past a stop, the code is ``None``.
        env: Its environment (a sub-agent's, which names its private
            folder); ``None`` for this process's.
    """
    # The Popen is held from before the spawn, which runs inside the ``try``: a
    # Stop raised as the spawning call returns reaches the ``finally`` with bash
    # in hand (see :func:`start_process_group`).
    process: subprocess.Popen[bytes] = object.__new__(subprocess.Popen)
    partial = b""
    code: int | None = None
    finished = False  # nothing of bash is left to end
    try:
        try:
            # Git for Windows keeps bash.exe off PATH: a bare "bash" fails there with WinError 2.
            start_process_group(
                process, [find_bash() or "bash", "-c", command], cwd=work_dir,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env,
            )
        except BaseException:
            # A start that failed ended what it had started; only a Stop raised
            # as it returned finds bash spawned and not reaped.
            finished = getattr(process, "pid", None) is None or process.returncode is not None
            raise
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not stop_signal.stop_requested():
            try:
                output = process.communicate(timeout=STOP_POLL_SECONDS)[0]
            except subprocess.TimeoutExpired as expired:
                # The output so far; none when only the exit was waited for.
                partial = expired.output or partial
                code = exit_status_unreaped(process)
                if code is None:
                    continue
                break  # it ended: what it left behind holds the pipe
            finished = True
            return process.returncode, output.decode("utf-8", errors="replace")
    finally:
        if not finished and not pid_handed_on(process):
            # The leader is unreaped here (one that ended is only watched), so
            # the group kill cannot hit a reused pid.  The bounded communicate
            # reaps it and returns what the pipe held; a pipe a descendant
            # still holds is closed instead.
            kill_process_tree(process)
            try:
                partial = process.communicate(timeout=REAP_GRACE_SECONDS)[0]
            except subprocess.TimeoutExpired as expired:
                partial = expired.output or partial
            if process.stdout is not None:
                process.stdout.close()
    return code, partial.decode("utf-8", errors="replace")


def gate_reviewer(model_name: str) -> str | None:
    """The reviewer model of a run on *model_name*, per :data:`REVIEWER_CONFIG_KEY`.

    Args:
        model_name: The run's model.

    Returns:
        The configured model; with no setting the strongest configured model
        of another family (:func:`~kiss.core.models.model_families.other_family_model`);
        ``None`` for ``"off"`` or when every configured model is of the
        run's own family (a same-family reviewer shares the executor's
        reading, so none is better than one).
    """
    from kiss.core.models.model_families import other_family_model
    from kiss.core.user_config import load_config

    setting = str(load_config().get(REVIEWER_CONFIG_KEY, "") or "").strip()
    if setting.lower() == "off":
        return None
    if setting:
        return setting
    try:
        return other_family_model(model_name)
    except Exception:  # an unreadable catalog must not end the run
        return None


def dispatches_review(args: dict[str, Any]) -> bool:
    """Whether a ``run_agent`` / ``run_parallel`` call with *args* dispatches a read-only review.

    Args:
        args: The call's arguments: a ``tool_profile`` of ``review`` (or
            ``readonly``), or a ``task`` / ``tasks`` every one of which reads
            as a review (:func:`~kiss.agents.sorcar.fanout_guard.is_review_task`;
            a fan-out mixing reviews with other work may change files).
            ``tasks`` is decoded as ``run_parallel`` decodes it
            (:func:`~kiss.agents.sorcar.fanout_guard.parse_tasks_json`);
            text it refuses (it dispatches nothing) is read as one task.
    """
    if str(args.get("tool_profile", "")).strip().lower() in ("review", "readonly"):
        return True
    tasks = [args["task"]] if isinstance(args.get("task"), str) else []
    raw = args.get("tasks")
    if isinstance(raw, str):
        try:
            raw = parse_tasks_json(raw)
        except ValueError:
            raw = [raw]
    if isinstance(raw, list):
        tasks.extend(t for t in raw if isinstance(t, str))
    return bool(tasks) and all(is_review_task(task) for task in tasks)


def gate_stage(stage: str, seconds: float, outcome: str, **fields: Any) -> None:
    """Record one gate stage in the task's event log (a ``gate_stage`` record).

    Args:
        stage: ``verifier``, ``review``, ``reading``, ``round`` or ``delivered``.
        seconds: The stage's wall-clock duration.
        outcome: ``pass``, ``fail``, ``error``, ``skipped``, ``missing`` or a count.
        **fields: Extra fields: the ``model``, ``cost_usd`` and ``round`` of
            a review, the ``items`` of a reading, ``giveup`` for a verifier
            run by a give-up.
    """
    logger.info("coding gate: stage=%s outcome=%s seconds=%.1f %s", stage, outcome, seconds,
                fields)
    publish("gate_stage", stage=stage, seconds=round(seconds, 3), outcome=outcome, **fields)


class CodingGate:
    """The finish gate of one Sorcar run with the edit tools.

    The agent's tool-call guard calls :meth:`observe` for every call, its
    ``tool_result_hook`` calls :meth:`result` with every call's result
    (a shell result gives the exit-code ledger its entry), and the guard
    calls :meth:`finish_verdict` for every ``finish``,
    :meth:`merge_verdict` for every ``commit_and_merge`` and
    :meth:`reading_note` for every ``Write``.  Until the run is a coding
    run (:meth:`coding_run`) a finish is not gated.

    Args:
        shell: Runs a command in the run's environment, in the run's
            current working directory: ``(exit code, output)``
            (:func:`run_on_host`, or the task's container through the
            agent).
        work_dir: The run's working directory; the agent sets
            :attr:`work_dir` again when it moves (into a worktree, back
            out of it), and the scratch directory moves with it.
        scratch_dir: The task's scratch directory, relative to the work
            dir (``tmp`` or ``tmp/<task-id>``) or absolute (a sub-agent's
            private folder); the agent's verifier is ``verify.sh`` in it.
        reviewer: The reviewer model (:func:`gate_reviewer`); ``None`` for
            no review stage.
        charge: Attributes a review attempt's spend ``(budget USD, tokens,
            model calls, key)`` to the run, once per key (a stop makes the
            gate retry it with the same key); ``None`` leaves it uncounted.
            A ``decide`` call of the gate (:func:`answered_findings`) is
            charged with its model as the ``model`` keyword too.
        model_config: The run's model overrides; its ``coding_gate: False``
            (a SEA's setting, ``/sync`` says it) declares the run not a
            coding run, whatever it changes.  The reviewer runs with the
            overrides minus their effort setting (:func:`default_effort`).
        verifier_timeout: Seconds the verifier gets at a finish.
        custom_tools: The names of the tools the gate does not model (a
            SEA's own, an MCP server's); after one HEAD is read again,
            since such a tool may have moved it (a sync that pulled commits).
        container: Whether the run's tools are the container's, not the
            host's: the two ``Bash`` tools show a failure differently
            (:func:`bash_exit_code`).
    """

    def __init__(self, shell: Shell, work_dir: str, scratch_dir: str,
                 reviewer: str | None = None,
                 charge: Callable[..., None] | None = None,
                 model_config: dict[str, Any] | None = None,
                 verifier_timeout: int = VERIFIER_TIMEOUT,
                 custom_tools: frozenset[str] = frozenset(),
                 container: bool = False) -> None:
        self.shell = shell
        self.work_dir = work_dir
        self.scratch_dir = scratch_dir.replace("\\", "/").rstrip("/").removeprefix("./")
        self.reviewer = reviewer
        self.charge = charge
        self.model_config = model_config
        self.verifier_timeout = verifier_timeout
        self.custom_tools = custom_tools
        self.container = container
        #: The run may be a coding run (``coding_gate`` of the model config).
        self.coding = (model_config or {}).get("coding_gate", True) is not False
        #: The run changed a code file or has a verifier.
        self.armed = False
        #: The commit the run started from (``git rev-parse HEAD`` at its first
        #: call that could change files), ``None`` outside git; the diff the
        #: reviewer reads is taken against it, so a change the agent committed,
        #: or the merge of its worktree squashed, is still in it.
        self.base_commit: str | None = None
        self.base_resolved = False
        #: The repository's root and its untracked files when the run began:
        #: those files are not the run's work (unless it wrote them) and stay
        #: out of the diff.
        self.repo_root: str | None = None
        self.base_untracked: set[str] = set()
        #: The run moved HEAD itself (a ``git commit`` or merge in its shell
        #: commands, the merge of its worktree): its commits are its work and
        #: stay in the diff; otherwise a moved HEAD is a sync's (:meth:`follow_head`).
        self.moved_head = False
        #: The last exit code of every shell command run, by command.
        self.exit_codes: dict[str, str] = {}
        #: The commands of the background jobs started, by job id.
        self.background_jobs: dict[str, str] = {}
        #: Per thread, ``made``: the ``(exit code, job id)`` of the last
        #: ``Bash`` / ``Edit`` / ``Write`` result read since a call was
        #: observed, which a command made as those calls takes as its own
        #: (:meth:`shell_exits`).
        self.nested = threading.local()
        #: The sub-tasks left running (``run_agent`` job id -> whether it is a
        #: read-only review, :attr:`changes` at its dispatch, and the
        #: :meth:`fingerprint` of the work then), read when ``agent_job``
        #: reports them.
        self.jobs: dict[str, tuple[bool, int, str | None]] = {}
        #: The fingerprint of the work at the dispatch or rollback being observed.
        self.dispatch_fingerprint: str | None = None
        #: The verdicts open when a merge went through at the round cap, for
        #: the finish summary (:data:`MERGED_CAVEATS`).
        self.merge_caveats: list[str] = []
        self.merge_rounds = 0
        #: A sub-task that may have changed files, or a rollback, ran: the
        #: diff decides :meth:`coding_run`.
        self.delegated = False
        #: The files written with ``Edit`` / ``Write``, in order (the diff of a run outside git).
        self.written_files: list[str] = []
        #: How many file changes the run made (:meth:`changed`).
        self.changes = 0
        #: A file changed since the gate's last review.
        self.changed_since_review = True
        #: The agent dispatched a read-only reviewer after the last change and
        #: the dispatch went through.
        self.reviewed_by_agent = False
        self.verifier_refusals = 0
        #: Past the refusal cap the verifier did not run at this finish.
        self.verifier_skipped = False
        #: The gate's last run of the verifier, or the agent's own passing run
        #: of it since: what a review is shown as its last output.
        self.last_verifier_run: ShellRun = (None, "")
        #: ``(work dir, changes, fingerprint of the work, digest of the
        #: script)`` as they were when the verifier last ran, by the gate or
        #: in a clean passing run of the agent's (:meth:`result`): the run is
        #: reused while all four are the same (:meth:`verifier_run`).
        self.verifier_memo: tuple[str, int, str | None, str] | None = None
        #: :meth:`verifier_key` before the shell call being observed, when
        #: that call is one command that ends with the verifier
        #: (:func:`verifier_ends`).
        self.verifier_key_before: tuple[str, int, str | None, str] | None = None
        self.gate_rounds = 0
        self.reviews = 0
        self.failed_reviews = 0
        #: The findings of the last review, for the next finish to answer.
        self.open_findings: list[str] = []
        #: The independent reading of the task was done (once per run, at
        #: the first ``Write`` of the ledger).
        self.reading_done = False
        self.implicit_finish_vetoed = False
        self.giveup_reconsidered = False
        self.delivered_findings: list[str] = []

    @property
    def verifier_relative(self) -> str:
        """The verifier's path as the shell commands name it.

        Relative to the work dir, or absolute when the scratch dir is.
        """
        return posixpath.join(self.scratch_dir, VERIFIER_NAME)

    @property
    def verifier_path(self) -> str:
        """The verifier's path: under the current work dir unless the scratch dir is absolute."""
        return os.path.join(self.work_dir, os.path.normpath(self.verifier_relative))

    @property
    def verifier_shown(self) -> str:
        """The verifier's path as the verdicts name it (``./tmp/<id>/verify.sh``, or absolute)."""
        absolute = os.path.isabs(self.scratch_dir)
        return self.verifier_relative if absolute else "./" + self.verifier_relative

    def relative(self, path: str) -> str:
        """*path* relative to the work dir when it is inside it.

        Args:
            path: An absolute or relative path.
        """
        relative = os.path.relpath(path, self.work_dir) if os.path.isabs(path) else path
        return path if relative.startswith("..") else relative

    def move(self, work_dir: str, fresh_base: bool = False) -> None:
        """Follow the agent into *work_dir* (a worktree, or the checkout after its merge).

        Args:
            work_dir: The run's working directory from now on; the scratch
                directory and the verifier move with it.
            fresh_base: Whether to take the base commit again at the next
                call that could change files: in a worktree just checked
                out its HEAD holds the checkout's dirty state, so the diff
                against it is the run's own work.  Leaving the worktree
                keeps the base, so the merged work is still the diff.

        The last run of the verifier (:attr:`verifier_memo`) was a run on
        the tree left behind, whose directory is part of its key: another
        tree (the merged checkout) may hold other files, so a verifier
        found there runs again, unless :meth:`merged` finds the checkout
        holding the very files that run saw.
        """
        self.work_dir = work_dir
        if fresh_base:
            self.base_resolved = False

    def verifier_stale(self) -> None:
        """Forget the gate's last run of the verifier: it, or the tree it ran on, changed."""
        self.verifier_memo = None

    def observe(self, name: str, args: dict[str, Any]) -> None:
        """Record a tool call that is about to run.

        The first call that could change files fixes :attr:`base_commit`;
        an ``Edit`` / ``Write`` of a code file, or a writing shell command
        naming one, arms the gate.

        Args:
            name: The tool name.
            args: Its arguments.
        """
        shell_exit.take()  # an exit no result took is not this call's
        self.nested.made = None
        if name in EDIT_TOOLS and self.is_scratch(args.get("file_path")):
            if self.is_verifier(str(args.get("file_path", ""))):
                self.verifier_stale()  # the verifier itself changes: its last run is stale
            return  # a scratch note (tmp/PROGRESS.md, a probe): no change of the work
        if name in EDIT_TOOLS or name in SHELL_TOOLS or name in DISPATCH_TOOLS or (
                name == ROLLBACK_TOOL):
            self.resolve_base()
        if name in EDIT_TOOLS:
            path = args.get("file_path")
            if isinstance(path, str) and path not in self.written_files:
                self.written_files.append(path)
            if is_code_file(path):
                self.armed = True
            self.changed()
        elif name in SHELL_TOOLS:
            commands = shell_commands(args)
            if any(VERIFIER_NAME in command for command in commands):
                self.verifier_stale()  # a command that may rewrite the verifier
            self.verifier_key_before = None
            if len(commands) == 1 and verifier_ends(commands[0], self.work_dir,
                                                    self.verifier_path):
                self.verifier_key_before = self.verifier_key()
            if any(moves_head(command) for command in commands):
                self.moved_head = True
            changes = file_changes(args)
            if changes:
                if any(is_code_file(path) for path in changes):
                    self.armed = True
                self.changed()
        elif name in DISPATCH_TOOLS:
            # A local sub-task that is no review may change the work: what it
            # changes is told by the fingerprint of the work before and after it.
            self.dispatch_fingerprint = None
            if not dispatches_review(args) and not str(args.get("machine", "")).strip():
                self.dispatch_fingerprint = self.fingerprint()
        elif name == ROLLBACK_TOOL:
            self.dispatch_fingerprint = self.fingerprint()

    def resolve_base(self) -> None:
        """Record the run's base commit, repository root and untracked files, once.

        Called before the run's first call that could change files.
        """
        if self.base_resolved:
            return
        self.base_resolved = True
        code, text = self.git("git rev-parse HEAD --show-toplevel && "
                              "git ls-files --others --exclude-standard -z")
        head, _, rest = text.partition("\n")
        root, _, others = rest.partition("\n")
        if code == 0 and re.fullmatch(r"[0-9a-f]{40,64}", head.strip()):
            self.base_commit = head.strip()
            # git prints the root with "/" separators, on Windows too: the OS form here.
            self.repo_root = os.path.normpath(root.strip())
            self.base_untracked = {path for path in others.split("\0") if path}

    def follow_head(self) -> None:
        """Take HEAD as the base again when it moved and the run did not move it.

        Called after a tool the gate does not model (:attr:`custom_tools`):
        a sync tool that committed and pulled moved HEAD to commits that
        are not the run's work, which the reviewer must not read as such.
        A run that moved HEAD itself (:attr:`moved_head`) keeps its base,
        since its commits are its work.
        """
        if self.base_commit is None or self.moved_head:
            return
        code, head = self.git("git rev-parse HEAD")
        if code == 0 and re.fullmatch(r"[0-9a-f]{40,64}", head.strip()):
            self.base_commit = head.strip()

    def fingerprint(self) -> str | None:
        """A digest of the work since the base: each file whose content differs from it.

        Every path the diff against the base (renames not detected, so a
        rename names its source too) or the untracked list names, with
        its bytes on disk (not what Git's ``text`` conversion would
        store) and executable bit, a symbolic link with its target text,
        or as deleted; a path whose content and mode are the base's is
        left out.  So the digest is of the files'
        contents only, whether they are tracked, staged or committed:
        the worktree before its merge and the checkout after it have the
        same digest when the merge brought exactly the worktree's files
        (:meth:`merged`).  Taken before a local sub-task is dispatched and
        when its result is read: equal digests mean the sub-task changed
        no file of the work (an untracked file renamed, its contents
        kept, is a change too).  ``None`` outside git and when a path
        cannot be read as a file (a directory: a nested repository): the
        sub-task is then read as a change.

        The diff misses a tracked file whose edit Git's text conversion
        hides (``true`` made ``true<CR>`` under ``*.sh text``), so the
        digest of the tracked files' bytes (:data:`TRACKED_BYTES_DIGEST`)
        is part of it too; trees with the same files have the same one.
        """
        base = self.base_commit or "HEAD"
        code, text = self.git(
            "set -o pipefail; export GIT_LITERAL_PATHSPECS=1; "
            f"{{ git diff --no-renames --name-only -z {base} -- && "
            "git ls-files --others --exclude-standard -z; } | sort -zu | "
            "while IFS= read -r -d '' f; do "
            # the target's own trailing newlines kept: `-n` adds none, the `x`
            # stops `$(...)` stripping them
            "if [ -L \"$f\" ]; then m=120000; t=$(readlink -n -- \"$f\" && printf x) || exit 1; "
            "h=$(printf '%s' \"${t%x}\" | git hash-object --stdin) || exit 1; "
            "elif [ -e \"$f\" ]; then "
            "h=$(git hash-object --no-filters -- \"$f\") || exit 1; "
            "m=100644; if [ -x \"$f\" ]; then m=100755; fi; "
            "else printf '%s\\0deleted\\n' \"$f\"; continue; fi; "
            f"if [ \"$(git ls-tree {base} -- \"$f\" | cut -f1)\" != \"$m blob $h\" ]; then "
            "printf '%s\\0%s %s\\n' \"$f\" \"$m\" \"$h\"; fi; done")
        if code != 0:
            return None
        tracked = self.git(TRACKED_BYTES_DIGEST.format(base=base))[1]
        return hashlib.sha1((text + tracked).encode("utf-8", "replace")).hexdigest()

    def is_verifier(self, path: str) -> bool:
        """Whether *path* (as an ``Edit`` / ``Write`` names it) is the agent's verifier."""
        return os.path.normpath(os.path.join(self.work_dir, path)) == os.path.normpath(
            self.verifier_path)

    def verifier_decides(self, command: str) -> bool:
        """Whether an exit status 0 of the shell *command* means its run of the verifier passed.

        Args:
            command: The command as the agent ran it.
        """
        return verifier_decides(command, self.work_dir, self.verifier_path)

    def root_relative(self, path: str) -> str:
        """*path*, as an ``Edit`` / ``Write`` named it, relative to the repository root.

        Args:
            path: An absolute path, or one relative to the work dir.

        Returns:
            The root-relative path with ``/`` separators, or ``""`` when the
            run has no repository root or *path* is outside it.
        """
        if self.repo_root is None:
            return ""
        relative = os.path.relpath(os.path.join(self.work_dir, path), self.repo_root)
        return "" if relative.startswith("..") else relative.replace(os.sep, "/")

    def changed(self) -> None:
        """Note a change of the files: the verifier runs again, and the review if none has run."""
        self.changes += 1
        self.changed_since_review = True
        self.reviewed_by_agent = False

    def merged(self) -> str:
        """Note that the worktree's merge went through: the finish is gated afresh.

        The rounds the merge spent and the findings of its review (met,
        or time-barred by the round cap) stay with the merge: the finish
        is refused on rounds of its own, and reviewed only when a file
        changes and no review has run (:data:`MAX_REVIEWS`).  The verdicts
        open at a merge let through at the cap are returned for the
        merge's result (:data:`MERGED_WITH_FINDINGS`) and kept for the
        finish summary (:attr:`merge_caveats`); the squash merge moved
        HEAD, which the base follows no more.

        Returns:
            The text to append to the merge's result (``""`` when no
            finding was open).
        """
        self.moved_head = True
        self.gate_rounds = 0
        self.open_findings = []
        self.carry_verifier_run()
        if not self.merge_caveats:
            return ""
        items = "\n".join(f"{i}. {v}" for i, v in enumerate(self.merge_caveats, 1))
        return MERGED_WITH_FINDINGS.format(rounds=self.merge_rounds, items=items)

    def carry_verifier_run(self) -> None:
        """Keep the worktree's last verifier run for the checkout when the merge kept its files.

        Called by :meth:`merged`, after the move into the checkout: when
        nothing was changed since that run and the checkout's
        :meth:`fingerprint` is the one the run saw (the merge brought
        exactly the worktree's files, and no other merge or edit of the
        checkout changed any), the run is taken for a run in the checkout,
        so a finish there with the same script reuses it
        (:meth:`verifier_run`).  Otherwise it is forgotten.
        """
        memo, self.verifier_memo = self.verifier_memo, None
        if memo is None or memo[1] != self.changes or memo[2] is None:
            return
        if self.fingerprint() == memo[2]:
            self.verifier_memo = (self.work_dir, *memo[1:])

    def is_ledger(self, path: str) -> bool:
        """Whether *path* (as a ``Write`` names it) is the run's ledger in its scratch dir.

        Args:
            path: The path, relative to the work dir or absolute.
        """
        return os.path.normpath(os.path.join(self.work_dir, path)) == os.path.normpath(
            os.path.join(self.work_dir, self.scratch_dir, LEDGER_NAME))

    def is_scratch(self, path: Any) -> bool:
        """Whether *path* (as an ``Edit`` / ``Write`` names it) is a scratch file of the run.

        One under a ``tmp`` directory inside the work dir, or in the
        scratch dir when that is absolute (a sub-agent's private folder):
        a progress log, a probe, the verifier.  Writing one changes no work.
        """
        if not isinstance(path, str) or not path:
            return False
        full = os.path.normpath(os.path.join(self.work_dir, path))
        if os.path.isabs(self.scratch_dir) and full.startswith(
                os.path.normpath(self.scratch_dir) + os.sep):
            return True
        relative = os.path.relpath(full, self.work_dir)
        parts = relative.replace("\\", "/").split("/")
        return not relative.startswith("..") and "tmp" in parts[:-1]

    def result(self, name: str, args: dict[str, Any], text: str, refused: bool) -> None:
        """Record a tool call's result (the agent's ``tool_result_hook``).

        A shell result gives the exit-code ledger its entries
        (:meth:`shell_exits`; the commands of a refused call are
        ``refused``).  The result of a sub-task is read by :meth:`subtask`.

        Args:
            name: The tool called.
            args: Its arguments.
            text: The result text the model reads.
            refused: Whether the call was refused (*text* is the refusal).
        """
        handed = shell_exit.take(args.get(ASKED.get(name, "")))
        nested, self.nested.made = getattr(self.nested, "made", None), None
        if name in EDIT_TOOLS:
            # whether it went through, as the framework decides it for the rest of a rewrite
            failed = refused or text.startswith(
                ("Error", "Failed to call", USER_INTERRUPTED_MESSAGE))
            self.nested.made = ("error" if failed else "0", "")
        if refused:
            if name in SHELL_TOOLS:
                self.nested.made = ("refused", "")
                for command in shell_commands(args):
                    self.exit_codes[command.strip()] = "refused"
            return
        before, self.verifier_key_before = self.verifier_key_before, None
        for command, code in self.shell_exits(name, args, text, handed, nested):
            self.exit_codes[command.strip()] = code
            if (name != "bash_job" and code == "0" and self.verifier_decides(command)
                    and not file_changes(args)):
                # the agent's own passing run of its verifier, its exit the
                # command's and no write of the call's own that may follow
                # the run (``bash v.sh && printf x > f``): shown to a review
                self.last_verifier_run = (0, text)
                # and taken for the gate's run when nothing ran after it and
                # the work and the script are as they were before the call
                # (a program in the call may change a file no scan shows)
                after = self.verifier_key() if before is not None else None
                clean = after is not None and after == before and after[2] is not None
                self.verifier_memo = after if clean else None
        if name in DISPATCH_TOOLS or name == "agent_job":
            self.subtask(name, args, text)
        elif name == ROLLBACK_TOOL:
            self.restored()
        elif name in self.custom_tools:
            self.follow_head()

    def shell_exits(
        self,
        name: str,
        args: dict[str, Any],
        text: str,
        handed: tuple[tuple[str, ...], str] | None,
        nested: tuple[str, str] | None,
    ) -> list[tuple[str, str]]:
        """The ``(command, exit code)`` pairs of a shell tool call that was not refused.

        The exits are the ones the tool handed over as data
        (:mod:`~kiss.agents.sorcar.shell_exit`): the result's text is
        the command's own output inside the tool's status, and a command
        prints what it likes.  A ``Bash`` command the agent made as an
        ``Edit`` / ``Write`` (its tool raised a ``ToolRedirect`` and
        returned nothing) ended as the last of the calls made for it did:
        the follow-up running its rest, else the ``Edit`` / ``Write``
        itself.  Only a result no tool returned is read as text
        (:func:`result_exit_codes`): the framework's own (the tool raised,
        the call was stopped), a job's report, which opens with the
        tool's lines, or one of a tool that hands nothing over.

        Args:
            name: The tool called.
            args: Its arguments.
            text: Its result.
            handed: What the tool handed over for this call, if anything.
            nested: The ``(exit code, job id)`` of the last call whose
                result was read on this thread since this one was observed.

        Returns:
            The pairs; a job a ``Bash`` call started or left running is
            added to :attr:`background_jobs`.
        """
        if name == "Bash" and isinstance(args.get("command"), str):
            if handed is not None:
                made = (handed[0][0], handed[1])
            elif nested is not None and REDIRECTED.match(text):
                made = nested
            else:
                made = bash_result(args, text, self.container)
            self.nested.made = made
            if made[1]:
                self.background_jobs[made[1]] = args["command"]
            return [(args["command"], made[0])]
        if name == "run_commands_parallel" and handed is not None:
            return list(zip(shell_commands(args), handed[0], strict=False))
        return result_exit_codes(name, args, text, self.background_jobs, self.container)

    def subtask(self, name: str, args: dict[str, Any], text: str) -> None:
        """Read the result of a sub-task: a review of the work, or a change of it.

        A read-only review the agent dispatched (:func:`dispatches_review`)
        counts the work as reviewed once a sub-task of it succeeded
        (:func:`dispatch_successes`) and no file changed since its
        dispatch; one left running (a job id in the result) is read when
        ``agent_job`` first reports it finished (a ``wait_any`` result is read
        job by job, :func:`~kiss.agents.sorcar.agent_dispatch.wait_any_results`).
        Any other sub-task may have
        changed files: when the work's :meth:`fingerprint` differs from
        the one taken at its dispatch (or none could be taken) it is a
        change (:meth:`changed`), unchecked until a shell command runs,
        and the diff decides :meth:`coding_run`.  A sub-task sent to
        another machine (``run_agent(machine=...)``) touches no file here
        and reviews nothing here: it is neither.

        Args:
            name: ``run_agent``, ``run_parallel`` or ``agent_job``.
            args: The call's arguments.
            text: Its result.
        """
        if name == "agent_job":
            from kiss.agents.sorcar.agent_dispatch import wait_any_results

            if str(args.get("action", "")) == "wait_any":
                results = wait_any_results(text)
            else:
                results = {str(args.get("job_id", "")).strip(): text}
            for job_id, result in results.items():
                successes = dispatch_successes(result)
                if job_id in self.jobs and successes:  # a job of this run, finished: read once
                    self.subtask_done(*self.jobs.pop(job_id), successes)
            return
        if str(args.get("machine", "")).strip():
            return
        review, changes = dispatches_review(args), self.changes
        fingerprint = self.dispatch_fingerprint
        for job_id in started_agent_jobs(text):
            self.jobs[job_id] = (review, changes, fingerprint)
        self.subtask_done(review, changes, fingerprint, dispatch_successes(text))

    def subtask_done(
        self, review: bool, changes: int, fingerprint: str | None, successes: list[bool],
    ) -> None:
        """Record one finished sub-task (:meth:`subtask`): a review, or a possible change.

        Args:
            review: Whether the sub-task was a read-only review.
            changes: The gate's :attr:`changes` count at its dispatch.
            fingerprint: The work's :meth:`fingerprint` at its dispatch,
                ``None`` when none could be taken.
            successes: The ``success`` flags of its results
                (:func:`dispatch_successes`); empty while it runs.
        """
        if review:
            if any(successes) and changes == self.changes:
                self.reviewed_by_agent = True
                if self.changed_since_review:
                    # A review of a diff later than the gate's own: its
                    # findings and the rounds spent on them are stale.
                    self.gate_rounds = 0
                    self.open_findings = []
        elif fingerprint is None or self.fingerprint() != fingerprint:
            self.delegated = True
            self.changed()

    def restored(self) -> None:
        """Read the result of a ``rollback``: the files it rewrote or deleted are a change.

        Told, as for a sub-task (:meth:`subtask`), by the work's
        :meth:`fingerprint` before and after it: a rollback that failed
        or restored nothing (the files already matched the checkpoint)
        changed no file.  The verifier that passed before a change is not
        reused (:meth:`changed`), and the diff decides :meth:`coding_run`,
        since the files restored need not be ones an ``Edit`` named.
        """
        if self.dispatch_fingerprint is None or self.fingerprint() != self.dispatch_fingerprint:
            self.delegated = True
            self.changed()

    def coding_run(self) -> bool:
        """Whether the run changed code or has a verifier (then it is gated).

        After a sub-task that may have changed files (:attr:`delegated`)
        the diff decides: a code file changed since the run began arms
        the gate.  So it does while a sub-task left running that is no
        review was never read as finished (:attr:`jobs`: one killed,
        ended in an error text or never collected reports no ``success``
        to settle it on).  A run declared no coding run (:attr:`coding`)
        never is.
        """
        if not self.coding:
            return False
        if not self.armed and self.verifier_exists():
            self.armed = True
        if not self.armed and (self.delegated
                               or any(not review for review, _, _ in self.jobs.values())):
            self.armed = any(is_code_file(path) for path, _ in self.changed_paths() or [])
        return self.armed

    def verifier_exists(self) -> bool:
        """Whether the agent's verifier exists and is not empty (under the current work dir)."""
        return self.shell(f"test -s {shlex.quote(self.verifier_relative)}", 10)[0] == 0

    def verifier_key(self) -> tuple[str, int, str | None, str]:
        """The tree a run of the verifier sees: ``(work dir, changes, fingerprint, digest)``.

        The digest is the whole script's, not the prefix a review reads:
        an edit past that prefix is a new script too; of the bytes Bash
        runs, not of what Git would store (a ``text`` attribute drops the
        CR before a newline that makes ``true<CR>`` a command not found).
        """
        fingerprint = self.fingerprint()
        code, digest = self.shell(
            f"git hash-object --no-filters {shlex.quote(self.verifier_relative)}", DIFF_TIMEOUT)
        return self.work_dir, self.changes, fingerprint, digest.strip() if code == 0 else ""

    def verifier_run(self, giveup: bool = False) -> ShellRun:
        """Run the agent's verifier and return ``(exit code, output)``.

        The last run, the gate's or a clean passing run of the agent's
        (:meth:`result`), is reused while the tree is the one it ran on
        (the same working directory, :attr:`changes` and
        :meth:`fingerprint` of the work, as they were when that run
        started: a move to another tree or a change no tool call declared
        is a different key) and the script's text is unchanged
        (:attr:`verifier_memo`): a round in which nothing changed runs
        nothing, and its stage record says ``reused``.  Never reused
        outside git (no fingerprint).

        Args:
            giveup: Whether a give-up runs it (recorded on the stage).
        """
        key = self.verifier_key()
        if key[2] is not None and key == self.verifier_memo:
            gate_stage("verifier", 0.0, "reused", giveup=giveup, reused=True)
            return self.last_verifier_run
        started = time.time()
        self.last_verifier_run = self.shell(f"bash {shlex.quote(self.verifier_relative)}",
                                            self.verifier_timeout)
        self.verifier_memo = key
        code = self.last_verifier_run[0]
        outcome = "pass" if code == 0 else "fail"
        gate_stage("verifier", time.time() - started, outcome, giveup=giveup)
        return self.last_verifier_run

    def finish_verdict(self, args: dict[str, Any], task: str) -> str | None:
        """Gate a ``finish``: the refusal text, or ``None`` to let the call through.

        An implicit (text-only) finish on a coding run is vetoed once with
        :data:`IMPLICIT_FINISH_VERDICT`.  A pause (``is_continue=True``)
        is never gated; a give-up is checked by :meth:`giveup_verdict`.  A
        success claim on a run that is not a coding run is not gated; on
        a coding run the verifier and the review run at once
        (:meth:`run_stages`), then the review's findings (or, once the
        review is spent, their answers) are judged, and every failed stage
        is refused together (:meth:`refusal`): the first refusal buys one
        round whatever refused it, a later round only when the verifier
        refused, up to :data:`MAX_GATE_ROUNDS`, after which the work is
        delivered with the open verdicts as caveats
        (:meth:`deliver_with_findings`).  The summary's wording and the
        ``evidence`` field refuse nothing: they are the reviewer's input.

        Args:
            args: Arguments of the ``finish`` call (empty for an implicit finish;
                edited in place when caveats are appended).
            task: The task statement.
        """
        if not args:
            if self.implicit_finish_vetoed or not self.coding_run():
                return None
            self.implicit_finish_vetoed = True
            return IMPLICIT_FINISH_VERDICT
        if str(args.get("success")).lower() != "true":
            if str(args.get("is_continue")).lower() == "true" or not self.coding_run():
                return None
            return self.giveup_verdict(args)
        if not self.coding_run():
            return None
        self.verifier_skipped = False
        stages = self.run_stages(args, task)
        verdicts = [stages["verifier"]]
        if stages.get("review") is not None:
            verdicts.append(stages["review"])
        else:  # no review this round, or one that passed or failed: its findings stand
            verdicts.append(self.unanswered_verdict(args))
        refusal = self.refusal("finish", verdicts, stages["verifier"] is not None)
        if refusal:
            return refusal
        open_verdicts = [v for v in verdicts if v]
        if self.verifier_skipped:  # not a refusal (the verifier will not run again), but a caveat
            open_verdicts.append(UNVERIFIED_CAVEAT.format(verifier=self.verifier_shown,
                                                         refusals=self.verifier_refusals))
        if open_verdicts:
            self.deliver_with_findings(args, open_verdicts)
        self.deliver_merge_caveats(args)
        return None

    def merge_verdict(self, task: str) -> str | None:
        """Gate a ``commit_and_merge``: the refusal text, or ``None`` to let the merge go.

        The merge is where the work of a worktree is declared complete,
        and the worktree is where fixing it is cheap, so the checks that
        need no summary run here first: the verifier and the review of
        the diff (:meth:`run_stages`; the reviewer is told the summary is
        not written yet).  A merge asked for again with the last review's
        findings open and no file changed since is refused with them
        (:data:`MERGE_UNANSWERED_VERDICT`: a merge has no summary to answer
        them in).  The refusal counts as a gate round like a finish's,
        under the same rule (:meth:`refusal`); a merge not refused goes
        through with its open findings attached, and its rounds and
        findings stay with it (:meth:`merged`): the finish is gated
        afresh, and reviewed only when a file changed and no review has
        run (:data:`MAX_REVIEWS`).

        Args:
            task: The task statement.
        """
        if not self.coding_run():
            return None
        self.verifier_skipped = False
        stages = self.run_stages({"summary_in_html": MERGE_SUMMARY_PLACEHOLDER}, task)
        unanswered = None
        if stages.get("review") is None and self.open_findings and not self.changed_since_review:
            unanswered = MERGE_UNANSWERED_VERDICT.format(findings="\n".join(self.open_findings))
        verdicts = [stages["verifier"], stages.get("review"), unanswered]
        open_verdicts = [v for v in verdicts if v]
        if open_verdicts and not self.refusable(stages["verifier"] is not None):
            # The round that is not refused: the merge goes through with
            # its findings attached (the merge's result, then the finish
            # summary), so no run ends with its work on a branch nobody
            # merges.
            self.merge_rounds = self.gate_rounds
            self.merge_caveats = open_verdicts
            gate_stage("delivered", 0.0, str(len(open_verdicts)), round=self.gate_rounds,
                       call="commit_and_merge")
            return None
        return self.refusal("commit_and_merge", verdicts, stages["verifier"] is not None)

    def refusable(self, verifier_failed: bool) -> bool:
        """Whether a call with open verdicts is refused now rather than delivered with caveats.

        The first refusal buys one round whatever refused it; a later
        round only when the verifier refused, up to :data:`MAX_GATE_ROUNDS`.

        Args:
            verifier_failed: Whether the verifier stage refused at this call.
        """
        return self.gate_rounds < MAX_GATE_ROUNDS and (self.gate_rounds == 0 or verifier_failed)

    def refusal(self, call: str, verdicts: list[str | None],
                verifier_failed: bool) -> str | None:
        """The refusal of *call* for the open *verdicts*, or ``None`` when it is not refused.

        Args:
            call: ``finish`` or ``commit_and_merge``.
            verdicts: The stages' verdicts (``None`` for a stage that passed).
            verifier_failed: Whether the verifier stage refused (:meth:`refusable`).
        """
        open_verdicts = [v for v in verdicts if v]
        if not open_verdicts or not self.refusable(verifier_failed):
            return None
        self.gate_rounds += 1
        gate_stage("round", 0.0, str(len(open_verdicts)), round=self.gate_rounds, call=call)
        return gate_refusal(self.gate_rounds, MAX_GATE_ROUNDS, open_verdicts, call)

    def run_stages(self, args: dict[str, Any], task: str) -> dict[str, str | None]:
        """Run the verifier and, when wanted, the review at once; each one's verdict by name.

        A stage that raises is a failed stage (its verdict names the
        error), not a failed run; each is recorded as a ``gate_stage``
        record with its duration and outcome.  The stage threads adopt the
        calling thread's event-sink context (the task's printer and
        binding), so the records they publish are filed under the task.
        A Stop ends the wait and the stages: the wait runs in short slices,
        where the task's stop event is seen and an injected
        ``KeyboardInterrupt`` lands, and either sets the stages' own stop
        event (:mod:`kiss.core.stop_signal` on their threads), on which
        the verifier's command is killed with its group
        (:func:`run_on_host`) and the reviewer's model call ends.  The
        threads start inside that protection too: a Stop (or a start
        that fails) once any stage started cancels the started ones.

        Args:
            args: Arguments of the ``finish`` call.
            task: The task statement.
        """
        stages: dict[str, Callable[[], str | None]] = {"verifier": self.verifier_verdict}
        if self.review_wanted():
            stages["review"] = functools.partial(self.review_verdict, args, task)
        verdicts: dict[str, str | None] = {}
        context = thread_context()
        cancel = threading.Event()
        # Each thread runs under its own copy of this thread's context
        # variables (a headless run's trajectory scope), a copy per thread
        # since a context may be entered by one thread at a time.
        threads = [threading.Thread(target=contextvars.copy_context().run, name=f"gate-{name}",
                                    daemon=True,
                                    args=(self.run_stage, name, stage, verdicts, context, cancel))
                   for name, stage in stages.items()]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                while thread.is_alive():
                    thread.join(STOP_POLL_SECONDS)
                    if stop_signal.stop_requested():
                        cancel.set()
        except BaseException:
            cancel.set()
            raise
        return verdicts

    def run_stage(self, name: str, stage: Callable[[], str | None],
                  verdicts: dict[str, str | None], context: Any,
                  cancel: threading.Event) -> None:
        """Run one gate stage (a thread's body): its verdict into *verdicts*.

        Args:
            name: The stage's name.
            stage: The stage.
            verdicts: Where the verdict goes, by name.
            context: The gate thread's event-sink context
                (:func:`kiss.core.event_sink.thread_context`), adopted first
                so the stage's records reach the task's event log.
            cancel: The stage's stop event, bound on its thread
                (:func:`kiss.core.stop_signal.set_thread_stop_event`):
                :meth:`run_stages` sets it when the run is stopped.
        """
        adopt_thread_context(context)
        stop_signal.set_thread_stop_event(cancel)
        started = time.time()
        try:
            verdicts[name] = stage()
        except Exception as error:
            logger.exception("coding gate: stage %s failed", name)
            # A review that could not run is no verdict (a check that did not
            # happen is never a refusal); a verifier that could not run is one.
            verdicts[name] = (None if name == "review" else
                              f"Not finished: a check of the gate could not run ({error!r}).")
            gate_stage(name, time.time() - started, "error", error=repr(error)[:500])
        except BaseException:
            # The stage was stopped (a model call raises KeyboardInterrupt on
            # its thread's stop event): the run is ending, no verdict stands.
            verdicts[name] = f"Not finished: the {name} check was stopped."
            gate_stage(name, time.time() - started, "stopped")

    def verifier_verdict(self) -> str | None:
        """Run the agent's verifier; the verdict when it fails or times out.

        No verifier means no verdict (the project's tests named as evidence
        are checked instead).  Past :data:`MAX_VERIFIER_REFUSALS` the
        verifier does not run and the finish is delivered with
        :data:`UNVERIFIED_CAVEAT`.
        """
        if not self.verifier_exists():
            gate_stage("verifier", 0.0, "missing")
            return None
        if self.verifier_refusals >= MAX_VERIFIER_REFUSALS:
            self.verifier_skipped = True
            gate_stage("verifier", 0.0, "skipped")
            return None
        code, output = self.verifier_run()
        if code == 0:
            return None
        self.verifier_refusals += 1
        if code is None:
            return VERIFIER_TIMEOUT_VERDICT.format(path=self.verifier_shown,
                                                   seconds=self.verifier_timeout)
        return VERIFIER_FAILED_VERDICT.format(path=self.verifier_shown, code=code,
                                              output=output[-OUTPUT_CHARS:])

    def evidence_report(self, args: dict[str, Any]) -> str:
        """The commands a finish names as evidence, with their last exit code, for the reviewer.

        The field refuses nothing: a command that did not run or failed
        is shown as such, and the reviewer judges what it proves.

        Args:
            args: Arguments of the ``finish`` call.
        """
        commands = evidence_commands(args.get("evidence"))
        if not commands:
            at_merge = args.get("summary_in_html") == MERGE_SUMMARY_PLACEHOLDER
            return MERGE_EVIDENCE_PLACEHOLDER if at_merge else "(none named)"
        return "\n".join(f"- exit {self.exit_codes.get(c.strip(), 'not run')}: `{c[:300]}`"
                         for c in commands[:30])

    def review_possible(self) -> bool:
        """Whether another review may run (neither the review nor the failed-review cap is hit)."""
        return self.reviews < MAX_REVIEWS and self.failed_reviews < MAX_FAILED_REVIEWS

    def review_wanted(self) -> bool:
        """Whether this round runs a review.

        A reviewer is set, a review is possible, the agent did not dispatch
        a read-only reviewer of its own after the last change, and a file
        changed since the gate's last review: a finish that changed nothing
        is judged on that review's findings by :meth:`unanswered_verdict`
        instead, without a model call.
        """
        return bool(self.reviewer and self.review_possible() and not self.reviewed_by_agent
                    and self.changed_since_review)

    def read_file(self, path: str, limit: int, in_repo: bool = False) -> str:
        """The first *limit* bytes of the file at *path* where the tools run (``""`` if none).

        Args:
            path: The file, relative to the work dir, or to the root of its
                git repository when *in_repo* (a path :meth:`changed_paths`
                listed).
            limit: The bytes to read.
            in_repo: Whether *path* is relative to the repository root.
        """
        command = f"head -c {limit} {shlex.quote(path)}"
        code, text = self.git(command) if in_repo else self.shell(command, DIFF_TIMEOUT)
        return text if code == 0 else ""

    def git(self, command: str) -> ShellRun:
        """Run the git *command* from the root of the work dir's repository.

        The paths git prints and takes are then root-relative whatever the
        work dir is (a task started in a subdirectory still has its whole
        repository reviewed).

        Args:
            command: The shell command.
        """
        return self.shell(
            f'root=$(git rev-parse --show-toplevel) && cd "$root" && ( {command} )', DIFF_TIMEOUT,
        )

    def changed_paths(self) -> list[tuple[str, bool]] | None:
        """The files changed in the work dir's repository since the run began.

        Returns:
            ``(path, untracked)`` per file, root-relative: every tracked path
            that differs from the run's :attr:`base_commit` (``HEAD`` when
            the run has none), committed or not, then every untracked file
            that was not there when the run began (:attr:`base_untracked`)
            or that the run wrote, scratch files aside (:meth:`is_scratch`:
            a repository need not ignore ``tmp/``); ``None`` when the work
            dir is not in a git repository.
        """
        code, others = self.git("git ls-files --others --exclude-standard -z")
        if code != 0:
            return None
        changed, names = self.git(f"git diff --name-only -z {self.base_commit or 'HEAD'} --")
        entries = [(path, False) for path in names.split("\0") if changed == 0 and path]
        written = {self.root_relative(path) for path in self.written_files}
        entries.extend((path, True) for path in others.split("\0")
                       if path and not self.is_scratch(path)
                       and (path not in self.base_untracked or path in written))
        return entries

    def outside_diff(self, path: str) -> str:
        """The change of the file at *path*, outside the work dir's repository.

        A task may edit another repository through its own worktree
        (``create_worktree`` of a path inside it keeps the work dir), so the
        file is shown with its ``git diff HEAD`` in its own repository when
        it is tracked there, and with its content otherwise.

        Args:
            path: The file, as the ``Edit`` / ``Write`` call named it.
        """
        directory, name = os.path.split(os.path.join(self.work_dir, path))
        code, text = self.shell(
            f"cd {shlex.quote(directory)} && git diff HEAD -- {shlex.quote(name)}", DIFF_TIMEOUT)
        return text if code == 0 and text.strip() else self.read_file(path, REVIEW_ONE_FILE_BYTES)

    def diff_report(self) -> str:
        """The diff of the files changed since the run began, for the reviewer.

        In a git repository: the files of :meth:`changed_paths`, each
        tracked one with its ``git diff`` against the run's
        :attr:`base_commit`, each untracked one with its content, then the
        files written with ``Edit`` / ``Write`` outside that repository
        (:meth:`outside_diff`).  Outside git: the content of the files
        written with ``Edit`` / ``Write``.  Each file is cut at
        :data:`REVIEW_ONE_FILE_BYTES` and the whole at
        :data:`REVIEW_FILE_BYTES`.
        """
        entries = self.changed_paths()
        in_repo = entries is not None
        if entries is None:
            kinds = [(self.relative(path), "new") for path in self.written_files]
        else:
            kinds = [(path, "new" if untracked else "tracked") for path, untracked in entries]
            kinds.extend((path, "outside") for path in self.written_files
                         if self.repo_root is not None and not self.root_relative(path))
        if not kinds:
            return "(no changed files found)"
        parts: list[str] = []
        total = 0
        for index, (path, kind) in enumerate(kinds):
            if total >= REVIEW_FILE_BYTES:
                parts.append(f"({len(kinds) - index} more changed files not shown)")
                break
            if kind == "new":
                text = self.read_file(path, REVIEW_ONE_FILE_BYTES, in_repo)
                header = f"### {path} (new file)"
            elif kind == "outside":
                text = self.outside_diff(path)
                header = f"### {path} (outside the work dir's repository)"
            else:
                base = self.base_commit or "HEAD"
                text = self.git(f"git diff {base} -- {shlex.quote(path)}")[1]
                header = f"### {path}"
            text = text[:REVIEW_ONE_FILE_BYTES]
            total += len(text)
            parts.append(f"{header}\n{text.rstrip() or '(empty)'}")
        return "\n\n".join(parts)

    def _charge_review(self, agent: Any, key: str) -> None:
        """Charge review attempt *agent*'s spend to the run under *key* (see ``charge``)."""
        spent, used, calls = agent.budget_used, agent.total_tokens_used, agent.step_count
        if self.charge is not None and (spent > 0 or used > 0 or calls > 0):
            self.charge(spent, used, calls, key)

    def _charge_decide(self, cost: float, tokens: int) -> None:
        """Charge one ``decide`` call of :func:`answered_findings` to the run, under its model.

        A failed charge is logged, never raised: it must not cost the call's answer.
        """
        from kiss.agents.sorcar.decide_tool import DEFAULT_DECISIONS_MODEL

        if self.charge is None or (cost <= 0 and tokens <= 0):
            return
        try:
            self.charge(cost, tokens, 0, f"decide:{uuid.uuid4().hex}",
                        model=DEFAULT_DECISIONS_MODEL)
        except Exception:
            logger.warning("coding gate: decide spend not charged", exc_info=True)

    def side_call(self, label: str, prompt: str, key: str,
                  lowered: bool = False) -> tuple[str, str, float, int]:
        """One non-agentic call of the reviewer model, charged to the run, with a deadline.

        The call runs at the provider's default effort (the run's model
        overrides minus the effort, :func:`default_effort`; at low effort
        when *lowered*, :func:`low_effort`) under a stop event of its own
        (:func:`call_stop_event`) that fires after :data:`REVIEW_DEADLINE`
        seconds or when the run is stopped; the model clients end the
        streaming call on it.  The attempt is charged as it ends, however
        it ends: a stop (which propagates) loses none of its spend, and a
        stop that lands inside the charge itself is charged again, once
        per key.

        Args:
            label: The agent's name in the logs.
            prompt: The whole prompt.
            key: The charge key's prefix (``finish-review``, ``reading``).
            lowered: Whether this is the retry of a failed attempt, at low effort.

        Returns:
            ``(answer, error, cost, tokens)``: the answer (``""`` for none),
            the error (``""`` when the call completed; the deadline when it
            was hit), the cost in USD and the tokens.
        """
        from kiss.core.kiss_agent import KISSAgent

        agent = KISSAgent(label)
        charge_attempt = functools.partial(self._charge_review, agent, f"{key}:{uuid.uuid4().hex}")
        config = (low_effort(self.model_config, str(self.reviewer)) if lowered
                  else default_effort(self.model_config))
        previous = stop_signal.get_thread_stop_event()
        done = threading.Event()
        deadline = call_stop_event(REVIEW_DEADLINE, done)
        stop_signal.set_thread_stop_event(deadline)
        answer, error = "", ""
        try:
            try:
                answer = str(agent.run(model_name=str(self.reviewer), prompt_template="{p}",
                                       arguments={"p": prompt}, is_agentic=False,
                                       verbose=False, model_config=config) or "")
            finally:
                done.set()
                stop_signal.set_thread_stop_event(previous)
                charge_attempt()
        except Exception as failure:
            error = repr(failure)
        except BaseException:
            if deadline.is_set() and not (previous is not None and previous.is_set()):
                error = f"{DEADLINE_ERROR} {REVIEW_DEADLINE} s"
            else:
                charge_attempt()
                raise
        return answer, error, agent.budget_used, agent.total_tokens_used

    def review_verdict(self, args: dict[str, Any], task: str) -> str | None:
        """A read-only review of the work by the reviewer model; its gaps as a verdict.

        The reviewer sees the task, the agent's summary, the verifier and
        its last output, the evidence named, and the diff
        (:meth:`diff_report`).  It runs no commands; its
        spend is charged to the run (:meth:`side_call`).  An attempt that
        answered nothing, was cut short by the output cap, was refused or
        missed the deadline (:func:`retryable_review_failure`) is retried
        once at low effort (:func:`low_effort`); a review that still
        failed, or answered without a ``VERDICT:`` line, refuses nothing:
        it keeps the previous findings open, is recorded with its error
        (and the first attempt's under ``retried``) and counts under
        :data:`MAX_FAILED_REVIEWS`, past which no review is asked for
        again.  Each ``NOT DONE`` finding is kept (:attr:`open_findings`)
        for the next finish to answer.

        Args:
            args: Arguments of the ``finish`` call.
            task: The task statement.
        """
        started = time.time()
        code, output = self.last_verifier_run
        prompt = REVIEW_PROMPT.format(
            task=task, summary=str(args.get("summary_in_html", "")),
            verifier=self.verifier_shown, code=code,
            script=self.read_file(self.verifier_relative, REVIEW_ONE_FILE_BYTES) or "(none)",
            output=output[-OUTPUT_CHARS:] or "(not run yet)",
            evidence=self.evidence_report(args),
            workdir=self.work_dir, files=self.diff_report(),
        )
        answer, error, cost, tokens = self.side_call("Finish reviewer", prompt, "finish-review")
        retried = ""
        if retryable_review_failure(answer, error):
            retried = error or "empty answer"
            answer, error, more_cost, more_tokens = self.side_call(
                "Finish reviewer", prompt, "finish-review", lowered=True)
            cost, tokens = cost + more_cost, tokens + more_tokens
        findings = review_findings(answer)
        failed = bool(error or not review_complete(answer))
        if failed:
            self.failed_reviews += 1
        else:
            self.reviews += 1
        # A report of the stage; the Spend-by-model split counts each attempt's
        # charge record instead (SorcarAgent._charge_gate_usage), so a review a
        # Stop ended is counted too and this one never twice.
        fields: dict[str, Any] = {"model": self.reviewer, "cost_usd": round(cost, 6),
                                  "tokens": tokens, "round": self.reviews,
                                  "findings": len(numbered_items(findings))}
        if retried:
            fields["retried"] = retried[:500]
        if failed:
            fields["error"] = (error or ("empty answer" if not answer.strip() else "no verdict")
                               )[:500]
        gate_stage("review", time.time() - started,
                   "error" if failed else ("fail" if findings else "pass"), **fields)
        if failed:
            return None
        self.changed_since_review = False
        self.open_findings = numbered_items(findings)
        if not findings:
            return None
        return REVIEW_VERDICT.format(findings=findings)

    def reading_note(self, name: str, args: dict[str, Any], task: str) -> str | None:
        """The independent reading of *task* to append to the ``Write`` of the agent's ledger.

        Once per run, when the agent first writes ``ledger.md`` in the
        scratch directory (:data:`LEDGER_NAME`; the system prompt asks for
        it when the task leaves points open), the reviewer model reads the
        task and the ledger's content (:data:`READING_PROMPT`, one call,
        :meth:`side_call`) and names the sentences of the task it would
        read differently.  The items join the ``Write``'s result
        (:data:`READING_NOTE`); ``NONE``, a failed call or an answer
        without items add nothing.  The call is recorded as the
        ``reading`` gate stage with its cost.

        Args:
            name: The tool called.
            args: Its arguments.
            task: The task statement.

        Returns:
            The note, or ``None`` when the call is not the ledger's first
            ``Write``, no reviewer is set, or the reading agrees.
        """
        path = args.get("file_path")
        if (name != "Write" or self.reading_done or not self.reviewer or not task
                or not isinstance(path, str) or not self.is_ledger(path)):
            return None
        self.reading_done = True
        started = time.time()
        prompt = READING_PROMPT.format(task=task,
                                       ledger=str(args.get("content", ""))[:REVIEW_ONE_FILE_BYTES])
        answer, error, cost, tokens = self.side_call("Reading check", prompt, "reading")
        items = [] if error or READING_NONE.match(answer) else numbered_items(answer)
        fields: dict[str, Any] = {"model": self.reviewer, "cost_usd": round(cost, 6),
                                  "tokens": tokens, "items": len(items)}
        if error:
            fields["error"] = error[:500]
        gate_stage("reading", time.time() - started,
                   "error" if error else ("fail" if items else "pass"), **fields)
        if not items:
            return None
        return READING_NOTE.format(model=self.reviewer, items="\n".join(items))

    def unanswered_verdict(self, args: dict[str, Any]) -> str | None:
        """The open review findings this finish does not answer by quoted evidence.

        Judged at a finish no review verdict came back for (none was
        wanted, or the one asked for failed): a finding is answered when
        the decisions model finds the summary quoting a command and its
        output for it (:func:`answered_findings`).  Without the decisions
        model every finding counts as answered.

        Args:
            args: Arguments of the ``finish`` call.
        """
        if not self.open_findings:
            return None
        answered = answered_findings(str(args.get("summary_in_html", "")), self.open_findings,
                                     self._charge_decide)
        if answered is None:
            return None
        unanswered = [f for f, ok in zip(self.open_findings, answered, strict=True) if not ok]
        if not unanswered:
            return None
        return UNANSWERED_VERDICT.format(findings="\n".join(unanswered))

    def giveup_verdict(self, args: dict[str, Any]) -> str | None:
        """Check a ``finish(success=False)`` the way a success claim is checked.

        The verifier runs (counting no refusal) and its outcome is appended
        to the summary next to the agent's own verdict (:data:`GIVEUP_NOTE`),
        so whoever reads the result sees "verifier: pass" instead of a
        bare failure notice.  When it passes, the first such give-up is
        answered once with :data:`GIVEUP_RECONSIDER_VERDICT`; the next ends
        the run whatever it says.  Give-ups are checked because many are
        wrong, with passing work on disk.  Without a verifier there is
        nothing to check.

        Args:
            args: Arguments of the ``finish`` call (edited in place).
        """
        if not self.verifier_exists():
            return None
        code = self.verifier_run(giveup=True)[0]
        state = "timeout" if code is None else "pass" if code == 0 else f"fail (exit {code})"
        args["summary_in_html"] = str(args.get("summary_in_html", "")) + GIVEUP_NOTE.format(
            verifier=self.verifier_shown, state=state)
        if code == 0 and not self.giveup_reconsidered:
            self.giveup_reconsidered = True
            return GIVEUP_RECONSIDER_VERDICT.format(verifier=self.verifier_shown)
        return None

    def deliver_with_findings(self, args: dict[str, Any], open_verdicts: list[str]) -> None:
        """Let a finish through past the round cap, with the open findings as caveats.

        The summary the caller sees gets the open verdicts appended
        (:data:`DELIVERED_CAVEATS`; the finish arguments are the dict the
        tool is then called with), and the user or the calling agent decides.

        Args:
            args: Arguments of the ``finish`` call (edited in place).
            open_verdicts: The verdicts of the stages that refused, and the
                caveat of a verifier that did not run.
        """
        self.delivered_findings = [v.splitlines()[0][:500] for v in open_verdicts]
        items = "\n".join(f"<li>{html.escape(v)}</li>" for v in open_verdicts)
        args["summary_in_html"] = str(args.get("summary_in_html", "")) + DELIVERED_CAVEATS.format(
            rounds=self.gate_rounds, items=items)
        gate_stage("delivered", 0.0, str(len(open_verdicts)), round=self.gate_rounds)

    def deliver_merge_caveats(self, args: dict[str, Any]) -> None:
        """Append to a passing finish the verdicts open at a merge let through at the cap.

        A verdict the summary answers with evidence (:func:`answered_findings`)
        is dropped; the rest (every one, without the decisions model) are
        appended as caveats (:data:`MERGED_CAVEATS`) so the user sees what
        the merge left open.

        Args:
            args: Arguments of the ``finish`` call (edited in place).
        """
        if not self.merge_caveats:
            return
        summary = str(args.get("summary_in_html", ""))
        answered = answered_findings(summary, self.merge_caveats, self._charge_decide) or [
            False] * len(self.merge_caveats)
        open_verdicts = [v for v, ok in zip(self.merge_caveats, answered, strict=True) if not ok]
        self.merge_caveats = []
        if not open_verdicts:
            return
        self.delivered_findings.extend(v.splitlines()[0][:500] for v in open_verdicts)
        items = "\n".join(f"<li>{html.escape(v)}</li>" for v in open_verdicts)
        args["summary_in_html"] = summary + MERGED_CAVEATS.format(rounds=self.merge_rounds,
                                                                 items=items)
