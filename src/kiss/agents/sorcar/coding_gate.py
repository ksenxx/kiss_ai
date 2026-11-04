# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Coding gate: the coding SEA's finish gate, for every Sorcar run that changes code.

The containerised coding SEA (:mod:`kiss.agents.seas.coding.coding_sea`)
checks a ``finish`` mechanically before the run ends: the agent's own
verifier runs, the commands the call names as evidence are compared with
the exit codes they had, a summary that hedges or admits an unhandled case
is refused, a reviewer of another model family reads the diff, a give-up
is checked the same way as a success claim, and after three refused rounds
the work is delivered with the open findings as caveats.  The gate exists
because success claims and give-ups are both unreliable, refused rounds
past the third buy nothing, and a reviewer of the executor's own family
shares its reading.

This module is the part of that gate that depends on nothing in the
benchmark container: :class:`CodingGate`, which
:class:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent` installs on every run
with the edit tools, and the readers of shell results both the gate and
the SEA use.  A run is a *coding run* once an ``Edit`` or ``Write`` targets
a code file (:func:`is_code_file`), a writing shell command names one
(:func:`names_code_file`), or the verifier exists; a run that writes only
prose keeps the hedge check of
:class:`~kiss.agents.sorcar.evidence_gate.EvidenceGate` and the check of
the commands a ``finish`` names as evidence, and nothing more.
The verifier is ``verify.sh`` in the task's scratch directory, the path the
system prompt gives the agent (How to Work, rule 3).  The gate runs it
with the run's own shell: a subprocess in the work dir
(:func:`run_on_host`), or the task's container.
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

from kiss.agents.sorcar.evidence_gate import (
    EDIT_TOOLS,
    SHELL_TOOLS,
    SHELLS,
    UNCHECKED_REASON,
    EvidenceGate,
    file_changes,
    runs_subcommand,
    shell_commands,
)
from kiss.agents.sorcar.fanout_guard import is_review_task
from kiss.agents.sorcar.shell_scan import (
    Group,
    SimpleCommand,
    bash_path,
    resolve,
    simple_commands,
)
from kiss.core import stop_signal
from kiss.core.event_sink import adopt_thread_context, publish, thread_context
from kiss.core.processes import find_bash, kill_process_tree, popen_process_group
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
#: Refused rounds of the gate per run; past them the finish goes through with
#: the open verdicts as caveats (a run refused again and again rarely recovers).
MAX_GATE_ROUNDS = 3
#: Reviews per run: the first on the whole diff, later ones after a change.
MAX_REVIEWS = 3
#: Reviews that produced no verdict (or failed) before a finish goes through without one.
MAX_FAILED_REVIEWS = 2
#: Seconds the diff listing for the reviewer may take.
DIFF_TIMEOUT = 60
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
    "verifier and checks the evidence), otherwise with success=False and what remains."
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
#: A finish whose ``evidence`` names a command that did not run or did not pass.
EVIDENCE_FIELD_VERDICT = (
    "Not finished: the evidence named in the finish call does not hold:\n{problems}\n"
    "The `evidence` argument is a JSON list of commands you ran in this run, exactly as you "
    "ran them, that exited 0. Run the check again and name it as run, or drop the item, then "
    "call finish again."
)
#: The finish refused once per run for a summary that leaves a case open.
OPEN_READING_VERDICT = (
    "Not finished: your summary says that a symptom, scenario or requirement of the task "
    "was left unhandled or settled by assumption, or that the result falls short of what "
    "the task asks for (infeasible, unsolvable, failing your own definition of done). Every "
    "such admission is a requirement unmet: the task promises an outcome, so a result short "
    "of it means a reading or a fix is still wrong, not that the data is. For each item: "
    "either handle the case and add a check for it to {verifier} (a supplied input that "
    "nothing consumes yet is the usual way in), or quote the sentence of the task or the "
    "data that rules it out. Then call finish again."
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
#: verifier did not run again.  (Worded without the evidence gate's hedge
#: phrases, which it would otherwise flag in the summary it joins.)
UNVERIFIED_CAVEAT = (
    "The verifier {verifier} refused {refusals} finishes and was not run again; the work is "
    "delivered as it stands."
)
#: The unchecked-change verdict of a merge: the evidence gate's reason, as a sentence.
UNCHECKED_VERDICT = "Not finished: {reason}."
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
lists the commands the engineer names as backing the claim, each checked to have run in \
this session and exited 0; a command is evidence of a requirement only when it tests that \
requirement: `echo`, `cat`, a script the engineer wrote themselves, or the words of the \
task prove nothing about the work on their own. When the summary and evidence sections \
say they are not written yet (a review at the merge of the worktree, before the finish), \
judge the diff and the verifier against the task and do not report their absence as a \
gap. Do not invent problems and never demand \
something the task does not ask for: report only a requirement that is not met, not \
checked, or checked wrongly, each with the evidence that shows it. Take the task \
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

# Commands the engineer names as evidence (each ran in this session and exited 0)
{evidence}
{previous}
# The diff of the files the engineer created or changed under {workdir}{scope}
{files}
"""
#: Section of a re-review: the findings of the previous round and how the
#: engineer answered them.  A re-review judges those findings; it opens no
#: new front of its own (a further spelling, input or edge case the task
#: does not name), so a round of answers closes the review instead of
#: starting the next.
REVIEW_PREVIOUS = """
# Findings of your previous review
{findings}

The engineer's answer to them (the summary above is the new one). A finding is \
settled only by a change in the diff below that meets it, or by a command the \
engineer quotes with its output that shows it wrong. It is not settled by reasoning, \
by a design document or docstring the task's symptom contradicts, or by a check added \
to the engineer's own verifier (it encodes their reading). Report every finding that is \
still unmet, in the same words if nothing changed. This review judges those findings: \
report a new one only when a sentence of the task requires it and the change since \
your previous review breaks it, quoting that sentence; do not add spellings, inputs or \
edge cases of your own that the task does not name, and do not reopen a finding the \
engineer has answered with evidence.
"""
REVIEW_VERDICT = (
    "Not finished: an independent review of the work against the task found these gaps:\n"
    "{findings}\n"
    "Answer every numbered finding in your next finish: either change the work (the "
    "review then runs again on the diff) or quote in summary_in_html, under the finding's "
    "number, the command you ran and its output that shows the finding wrong. An argument "
    "without a change or quoted evidence does not count."
)
#: A review that produced no verdict (or whose model call failed): the finish
#: is refused, not approved by silence, while another review is possible.
REVIEW_FAILED_VERDICT = (
    "Not finished: the independent review of the work did not complete (no verdict "
    "came back), so it counts as neither approval nor rejection. Re-check the work "
    "against the task and call finish again; the review runs again."
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
#: The last line of the merge refusal before the one that goes through.
MERGE_CAP_NOTE = (
    "This is the last refusal: a merge refused again goes through with its open findings "
    "attached to the result, so fix them now."
)
#: A merge asked for while the last review's findings are open and nothing changed.
MERGE_UNANSWERED_VERDICT = (
    "Not merged: these findings of the review are still open and no file changed since:\n"
    "{findings}\n"
    "Fix the work in the worktree and call commit_and_merge again (the change is reviewed "
    "again); a finding you have shown wrong is answered at the finish, by quoting under its "
    "number the command and output that shows it."
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
#: Probability at which the decisions model's answer counts the summary as
#: leaving a case of the task unhandled or the asked outcome missed (probed:
#: a summary that says "I added nothing for <a listed symptom>" scores above
#: 0.9 on the first question, one that reports the asked artifact as "not
#: feasible" above 0.9 on the second; summaries reporting the outcome
#: produced and every symptom checked score below 0.2).
OPEN_READING_THRESHOLD = 0.7
#: Questions the decisions model answers about the summary (:func:`leaves_a_case_open`).
OPEN_READING_QUESTIONS = {
    "case_open": (
        "Does the engineer explicitly admit that a symptom, scenario or requirement of the "
        "task statement is NOT handled by the delivered work (phrases like 'I added nothing "
        "for', 'left as is', 'known limitation', 'not addressed', 'did not handle', 'out of "
        "scope', 'I read X as Y' choosing one reading of a symptom while leaving another "
        "case unhandled)? Answer no when every listed symptom is reported as fixed and "
        "checked."),
    "outcome_missed": (
        "Does the delivered result, as the engineer describes it, fall short of the outcome "
        "the task statement asks for (the expected artifact is reported as infeasible, "
        "unsolvable, empty, failing its own criterion, or produced only under a weaker "
        "rule), rather than being the outcome itself? Answer no when the engineer reports "
        "the asked-for outcome as produced and checked."),
}

#: The failure marker of the container ``Bash`` tools, after the output.
EXIT_CODE_MARKER = re.compile(r"\[exit code: (-?\d+)\]")
#: The failure prefix of the host ``Bash`` tool, before the output.
HOST_EXIT_PREFIX = re.compile(r"^Error \(exit code (-?\d+)\):", re.MULTILINE)
#: A ``Bash`` result that started a background job (``background=True``, or a
#: foreground command left running at its deadline), with the job id.
STARTED_JOB = re.compile(r"(?:^Started|left running as) background job (\w+)", re.MULTILINE)
#: The status line of a ``bash_job`` report showing the job's exit: ``status:
#: exited N`` (the container tool) or ``Job <id>: exited with code N`` (the
#: host tool).  Only that line counts: the command echo and the log lines of
#: the report may say ``exited 0`` while the job runs.
JOB_EXITED = re.compile(r"^(?:status|Job \w+): exited (?:with code )?(-?\d+)$", re.MULTILINE)
#: The line that opens the log section of a ``bash_job`` report (both tools).
JOB_LOG_SECTION = re.compile(r"^--- last \d+ log lines ---$", re.MULTILINE)
#: The outcome of one command of a ``run_commands_parallel`` report:
#: ``[i/N] exit 0 in 1.2s``, ``[i/N] TIMED OUT``, ``[i/N] killed by signal 9``.
_PARALLEL_STATUS = r"\[(\d+)/\d+\] (?:exit (-?\d+)|(TIMED OUT)|killed by signal (\d+))"
#: The section header of one command of the report (``### [i/N] exit 0 in 1.2s``).
PARALLEL_EXIT = re.compile("^### " + _PARALLEL_STATUS, re.MULTILINE)
#: The status line of one command in the preamble of the report's digest
#: (``[i/N] exit 0 in 1.2s: <command>``; the full report was written to a
#: file, ``useful_tools.digest_parallel_report``).  Only the preamble, before
#: the first ``### `` section, is read for these: a command's own output may
#: hold such a line (a nested parallel run) and must not override its exit.
PARALLEL_DIGEST_STATUS = re.compile("^" + _PARALLEL_STATUS + r" in [\d.]+s: ", re.MULTILINE)
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
#: How a result opens when the command did not run: the tools' ``Error`` and
#: the framework's ``Failed to call`` (the tool raised before running it).
FAILED_PREFIXES = ("Error", "Failed to call")
#: The first line of the result of a shell command the ``Bash`` tool made as an
#: ``Edit`` / ``Write`` through the agent's tool path (``ToolRedirect``): the
#: redirected call's result follows it, then the follow-up call running the
#: rest of the command (named by :data:`FOLLOWUP_LINE` with its result, or by
#: :data:`FOLLOWUP_NOT_MADE` when the redirected call did not go through).
REDIRECTED = re.compile(r"(?:Applied as|Appended through) .*, whose result follows:\n")
FOLLOWUP_LINE = re.compile(r"^Then made .*, whose result follows:\n", re.MULTILINE)
FOLLOWUP_NOT_MADE = re.compile(r"^.* was not made: the \w+ did not go through\.$", re.MULTILINE)
#: The tools that run a sub-task: a read-only review (:func:`dispatches_review`)
#: or work that may change files.
DISPATCH_TOOLS = frozenset({"run_agent", "run_parallel"})
#: A ``run_agent`` result for a sub-task left running (``wait="false"``, or
#: its timeout passed), with the job id ``agent_job`` takes.
STARTED_AGENT_JOB = re.compile(r"\bas job (agent-\w+)\b")
#: A shell's option that parses a script without running it (``bash -n``, ``-xn``).
NOEXEC_OPTION = re.compile(r"-[a-zA-Z]*n[a-zA-Z]*|--noexec")
#: ``git``'s subcommands that move HEAD: a commit or merge of the run's own.
HEAD_MOVING = {"git": re.compile(
    r"(?:commit|merge|pull|rebase|cherry-pick|revert|am|reset|checkout|switch)\b")}


def leaves_shell(command: SimpleCommand) -> bool:
    """Whether the simple *command* may end the shell before the commands after it run.

    ``exit`` and ``return`` do; so does ``exec`` of a program (the shell
    becomes it), unlike a bare ``exec 2>&1``.

    Args:
        command: A scanned simple command.
    """
    if command.program() in ("exit", "return"):
        return True
    return bool(command.words) and command.words[0].text == "exec" and bool(command.argv())


def moves_head(command: str) -> bool:
    """Whether the shell *command* runs a git command that moves HEAD (:data:`HEAD_MOVING`).

    Args:
        command: The command as the agent ran it.
    """
    return any(runs_subcommand([word.text for word in simple.argv()], HEAD_MOVING)
               for simple in simple_commands(command, None))


def is_code_file(path: Any) -> bool:
    """Whether *path* names a code, build or configuration file (:data:`CODE_SUFFIXES`).

    Args:
        path: The ``file_path`` argument of an ``Edit`` or ``Write`` call.
    """
    if not isinstance(path, str):
        return False
    name = posixpath.basename(path.replace("\\", "/"))
    return name in CODE_FILE_NAMES or posixpath.splitext(name)[1].lower() in CODE_SUFFIXES


def names_code_file(args: dict[str, Any]) -> bool:
    """Whether a shell call changes a code file (:func:`is_code_file`).

    A command that changes one (``cp a.py b.py``, ``sed -i ... x.py``,
    ``python gen.py > out.py``) changes code as an ``Edit`` of it would;
    one that reads one (``cat install.sh``), names one in text, or
    writes one under the scratch directory (``gh api ... > tmp/x/pages.json``)
    does not (:func:`~kiss.agents.sorcar.evidence_gate.file_changes`).

    Args:
        args: The ``Bash`` / ``run_commands_parallel`` call's arguments.
    """
    return any(is_code_file(path) for path in file_changes(args))


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


def bash_exit_code(text: str) -> str:
    """The exit code a ``Bash`` result *text* shows, as a string.

    The host tool opens a failed result with ``Error (exit code N):`` and
    the container tool closes one with ``[exit code: N]``; a killed-at-the-
    deadline result is ``"timeout"``, a guard's refusal (the command never
    ran) ``"refused"``, a call the user stopped ``"interrupted"``, any
    other result the tool opens with ``Error`` (a rejected argument, a
    machine that could not be reached: the command did not run) or the
    framework with ``Failed to call`` (the tool raised before running it:
    a missing argument) ``"error"``, a command made as an ``Edit`` /
    ``Write`` the exit :func:`redirected_exit` reads, and a result with
    none of these ``"0"``.  The host prefix opens the result, so a
    marker the command itself printed (a nested tool's output, a fixture)
    does not override it; a marker is read before a prefix found later
    in the text.

    Args:
        text: The tool's result.
    """
    if REDIRECTED.match(text):
        return redirected_exit(text)
    if TIMED_OUT.match(text):
        return "timeout"
    if REFUSED.match(text):
        return "refused"
    if INTERRUPTED.match(text):
        return "interrupted"
    prefix = HOST_EXIT_PREFIX.match(text)
    if prefix is None:
        marker = EXIT_CODE_MARKER.findall(text)
        if marker:
            return str(marker[-1])
        prefix = HOST_EXIT_PREFIX.search(text)
    if prefix:
        return prefix.group(1)
    return "error" if text.startswith(FAILED_PREFIXES) else "0"


def redirected_exit(text: str) -> str:
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
    """
    rest = text.split("\n", 1)[1] if "\n" in text else ""
    followup = FOLLOWUP_LINE.search(rest)
    if followup:
        return bash_exit_code(rest[followup.end():])
    if rest.startswith(FAILED_PREFIXES) or REFUSED.match(rest) or FOLLOWUP_NOT_MADE.search(rest):
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


def result_exit_codes(name: str, args: dict[str, Any], text: str,
                      jobs: dict[str, str]) -> list[tuple[str, str]]:
    """The ``(command, exit code)`` pairs the result *text* of a tool call shows.

    Args:
        name: The tool called.
        args: The call's arguments.
        text: The call's result.
        jobs: The commands of the background jobs started so far, by job id;
            a ``Bash`` result that started one adds it.

    Returns:
        For ``Bash``, its command with the exit code of its result
        (:func:`bash_exit_code`), or ``background`` for a job started; for
        ``bash_job``, the job's command with the exit the status line of its
        report shows (``exited N``; nothing while it runs, whatever the
        command echo and the log lines say); for ``run_commands_parallel``,
        each command with the exit of its ``### [i/N]`` section (``timeout``,
        or ``killed by signal N``), or ``interrupted`` for every command
        when the user stopped the call.  Nothing for any other tool.
    """
    if name == "Bash" and isinstance(args.get("command"), str):
        command = args["command"]
        started = STARTED_JOB.search(text)
        if started:
            jobs[started.group(1)] = command
            return [(command, "background")]
        return [(command, bash_exit_code(text))]
    if name == "bash_job":
        command = jobs.get(str(args.get("job_id", "")))
        exited = JOB_EXITED.search(JOB_LOG_SECTION.split(text, maxsplit=1)[0])
        return [] if command is None or exited is None else [(command, exited.group(1))]
    if name == "run_commands_parallel":
        if INTERRUPTED.match(text):
            return [(command, "interrupted") for command in shell_commands(args)]
        preamble = text.split(PARALLEL_SECTIONS, 1)[0]
        found = PARALLEL_EXIT.findall(text) + PARALLEL_DIGEST_STATUS.findall(preamble)
        sections = {int(index): code or ("timeout" if timed_out else f"killed by signal {signal}")
                    for index, code, timed_out, signal in found}
        return [(command, sections[index])
                for index, command in enumerate(shell_commands(args), 1) if index in sections]
    return []


def evidence_commands(raw: Any) -> list[str]:
    """The commands a ``finish`` call's ``evidence`` argument names.

    Args:
        raw: The argument: a JSON list of strings, a list, or a plain string
            (one command).

    Returns:
        The non-empty command strings; empty when the field is empty or not
        a list of strings.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except ValueError:
            raw = [text]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]


def evidence_problems(commands: list[str], codes: dict[str, str]) -> list[str]:
    """The evidence *commands* that did not run, or did not exit 0, per *codes*.

    Args:
        commands: The commands a finish names (:func:`evidence_commands`).
        codes: The last exit code of every command run, by command.

    Returns:
        One ``- `command`: reason`` line per problem; empty when all hold.
    """
    problems = []
    for command in commands:
        code = codes.get(command.strip())
        if code is None:
            problems.append(f"- `{command[:200]}`: never ran as a Bash command in this run")
        elif code != "0":
            problems.append(f"- `{command[:200]}`: last exit code {code}")
    return problems


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


def leaves_a_case_open(statement: str, summary: str) -> float:
    """The decisions model's probability that *summary* admits a case of *statement* unhandled.

    The larger of the answers to :data:`OPEN_READING_QUESTIONS` (a case
    left open, the asked outcome missed); ``0.0`` without the decisions
    model or when its call fails.

    Args:
        statement: The task statement.
        summary: The finish summary.
    """
    from kiss.agents.sorcar.decide_tool import decisions_tool_available, make_decide_tool

    if not decisions_tool_available():
        return 0.0
    state = json.dumps({"task_statement": statement, "engineer_summary": summary})
    questions = {name: {"type": "noul", "instructions": text}
                 for name, text in OPEN_READING_QUESTIONS.items()}
    try:
        answers = json.loads(make_decide_tool(None)(state, json.dumps(questions)))["answers"]
        values = [answers[name]["noul"] for name in OPEN_READING_QUESTIONS]
    except Exception:
        return 0.0
    return max((float(v) for v in values if isinstance(v, int | float)), default=0.0)


def answered_findings(summary: str, findings: list[str]) -> list[bool] | None:
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
    """
    from kiss.agents.sorcar.decide_tool import decisions_tool_available, make_decide_tool

    if not findings:
        return []
    if not decisions_tool_available():
        return None
    questions = {f"f{i}": {"type": "noul", "instructions": ANSWER_QUESTION.format(finding=f)}
                 for i, f in enumerate(findings)}
    try:
        decide = make_decide_tool(None)
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
    header = (f"{what} refused (round {round_number} of {max_rounds}; every stage ran, "
              f"fix everything below before calling {call} again):")
    parts = [header, *verdicts]
    if call == "commit_and_merge" and round_number >= max_rounds - 1:
        parts.append(MERGE_CAP_NOTE)
    return "\n\n".join(parts)


def check_state(code: int | None, text: str) -> str:
    """The state of a verifier run from its ``(exit code, output)``.

    Args:
        code: Its exit code (``None`` when it did not finish in time).
        text: Its output.

    Returns:
        ``pass``, ``timeout`` or ``fail (exit N)``.
    """
    if code is None:
        return "timeout"
    return "pass" if code == 0 else f"fail (exit {code})"


#: Seconds between the checks for a Stop while the gate waits: on a host
#: command (:func:`run_on_host`) and on its stage threads
#: (:meth:`CodingGate.run_stages`).  Short slices keep the waits in Python,
#: where an injected Stop lands; a bare ``communicate()`` or ``join()``
#: blocks in C for the whole deadline.
STOP_POLL_SECONDS = 0.2
#: Seconds :func:`run_on_host` waits for its killed command to be reaped and
#: for the pipe of a descendant that survived the kill to close.
REAP_GRACE_SECONDS = 2.0


def run_on_host(command: str, work_dir: str, timeout: int) -> ShellRun:
    """Run *command* with bash in *work_dir*: its exit code and combined output.

    The command runs in a process group of its own, killed whole (a
    verifier's test runner included) when the deadline passes, when the
    calling thread's stop event (:mod:`kiss.core.stop_signal`: the task's
    Stop, or the gate cancelling a stage) is set meanwhile, or when an
    exception cuts the wait short (a Stop landing as ``KeyboardInterrupt``);
    the output read by then is kept.  The wait for the killed command is
    bounded and its pipe is closed rather than read to its end, so a
    descendant that escaped into a session of its own holds nothing up.

    Args:
        command: The shell command.
        work_dir: Its working directory.
        timeout: Seconds it may run; past them, or past a stop, the code is ``None``.
    """
    # Git for Windows keeps bash.exe off PATH: a bare "bash" fails there with WinError 2.
    process = popen_process_group(
        [find_bash() or "bash", "-c", command], cwd=work_dir, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + timeout
    partial = b""
    finished = False
    try:
        while time.monotonic() < deadline and not stop_signal.stop_requested():
            try:
                output = process.communicate(timeout=STOP_POLL_SECONDS)[0]
            except subprocess.TimeoutExpired as expired:
                # The output so far; none when only the exit was waited for.
                partial = expired.output or partial
                continue
            finished = True
            return process.returncode, output.decode("utf-8", errors="replace")
    finally:
        if not finished:
            # The leader is unreaped here, so the group kill cannot hit a reused
            # pid.  The bounded communicate reaps it and returns what the pipe
            # held; a pipe a descendant still holds is closed instead.
            kill_process_tree(process)
            try:
                partial = process.communicate(timeout=REAP_GRACE_SECONDS)[0]
            except subprocess.TimeoutExpired as expired:
                partial = expired.output or partial
            if process.stdout is not None:
                process.stdout.close()
    return None, partial.decode("utf-8", errors="replace")


def container_run(result: str) -> ShellRun:
    """Read the ``(exit code, output)`` a container ``Bash`` tool's *result* shows.

    Args:
        result: The text the tool returned (``[exit code: N]`` after the
            output of a failed command, ``Error: command timed out`` for one
            killed at its deadline).
    """
    code = bash_exit_code(result)
    if code == "timeout":
        return None, result
    if code in ("refused", "background"):
        return 1, result
    return int(code), EXIT_CODE_MARKER.sub("", result).rstrip()


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
    """
    if str(args.get("tool_profile", "")).strip().lower() in ("review", "readonly"):
        return True
    tasks = [args["task"]] if isinstance(args.get("task"), str) else []
    raw = args.get("tasks")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = [raw]
    if isinstance(raw, list):
        tasks.extend(t for t in raw if isinstance(t, str))
    return bool(tasks) and all(is_review_task(task) for task in tasks)


def gate_stage(stage: str, seconds: float, outcome: str, **fields: Any) -> None:
    """Record one gate stage in the task's event log (a ``gate_stage`` record).

    Args:
        stage: ``verifier``, ``review``, ``round`` or ``delivered``.
        seconds: The stage's wall-clock duration.
        outcome: ``pass``, ``fail``, ``skipped``, ``missing`` or a count.
        **fields: Extra fields: the ``model``, ``cost_usd`` and ``round`` of
            a review, ``giveup`` for a verifier run by a give-up.
    """
    logger.info("coding gate: stage=%s outcome=%s seconds=%.1f %s", stage, outcome, seconds,
                fields)
    publish("gate_stage", stage=stage, seconds=round(seconds, 3), outcome=outcome, **fields)


class CodingGate:
    """The finish gate of one Sorcar run with the edit tools.

    The agent's tool-call guard calls :meth:`observe` for every call, its
    ``tool_result_hook`` calls :meth:`result` with every call's result
    (a shell result gives the exit-code ledger its entry), and the guard
    calls :meth:`finish_verdict` for every ``finish`` and
    :meth:`merge_verdict` for every ``commit_and_merge``.  Until the run
    is a coding run (:meth:`coding_run`) only the check of the evidence a
    finish names and the hedge check of the
    :class:`~kiss.agents.sorcar.evidence_gate.EvidenceGate` apply, the
    latter as it did before this gate existed.

    Args:
        shell: Runs a command in the run's environment, in the run's
            current working directory: ``(exit code, output)``
            (:func:`run_on_host`, or the task's container through the
            agent).
        work_dir: The run's working directory; the agent sets
            :attr:`work_dir` again when it moves (into a worktree, back
            out of it), and the scratch directory moves with it.
        scratch_dir: The task's scratch directory, relative to the work
            dir (``tmp`` or ``tmp/<task-id>``); the agent's verifier is
            ``verify.sh`` in it.
        reviewer: The reviewer model (:func:`gate_reviewer`); ``None`` for
            no review stage.
        charge: Attributes a review attempt's spend ``(budget USD, tokens,
            model calls, key)`` to the run, once per key (a stop makes the
            gate retry it with the same key); ``None`` leaves it uncounted.
        model_config: The run's model overrides, tried first for the
            reviewer (a second attempt runs without them); its
            ``coding_gate: False`` (a SEA's setting, ``/sync`` says it)
            declares the run not a coding run, whatever it changes.
        verifier_timeout: Seconds the verifier gets at a finish.
        custom_tools: The names of the tools the gate does not model (a
            SEA's own, an MCP server's); after one HEAD is read again,
            since such a tool may have moved it (a sync that pulled commits).
    """

    def __init__(self, shell: Shell, work_dir: str, scratch_dir: str,
                 reviewer: str | None = None,
                 charge: Callable[[float, int, int, str], None] | None = None,
                 model_config: dict[str, Any] | None = None,
                 verifier_timeout: int = VERIFIER_TIMEOUT,
                 custom_tools: frozenset[str] = frozenset()) -> None:
        self.shell = shell
        self.work_dir = work_dir
        self.scratch_dir = scratch_dir.replace("\\", "/").strip("/").removeprefix("./")
        self.reviewer = reviewer
        self.charge = charge
        self.model_config = model_config
        self.verifier_timeout = verifier_timeout
        self.custom_tools = custom_tools
        #: The run may be a coding run (``coding_gate`` of the model config).
        self.coding = (model_config or {}).get("coding_gate", True) is not False
        self.evidence = EvidenceGate()
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
        #: The sub-tasks left running (``run_agent`` job id -> whether it is a
        #: read-only review, :attr:`changes` at its dispatch, and the
        #: :meth:`fingerprint` of the work then), read when ``agent_job``
        #: reports them.
        self.jobs: dict[str, tuple[bool, int, str | None]] = {}
        #: The fingerprint of the work at the dispatch being observed.
        self.dispatch_fingerprint: str | None = None
        #: :attr:`changes` when the agent's own run of the verifier last passed
        #: (the exit-code ledger); equal to :attr:`changes` at a finish, that
        #: run is reused instead of running the verifier again.
        self.verifier_passed_at: int | None = None
        #: The verdicts open when a merge went through at the round cap, for
        #: the finish summary (:data:`MERGED_CAVEATS`).
        self.merge_caveats: list[str] = []
        self.merge_rounds = 0
        #: A sub-task that may have changed files ran: the diff decides
        #: :meth:`coding_run`.
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
        self.last_verifier_run: ShellRun = (None, "")
        self.gate_rounds = 0
        self.reviews = 0
        self.failed_reviews = 0
        #: The findings of the last review, for the next finish to answer.
        self.open_findings: list[str] = []
        self.open_reading_refused = False
        self.implicit_finish_vetoed = False
        self.giveup_reconsidered = False
        self.delivered_findings: list[str] = []

    @property
    def verifier_relative(self) -> str:
        """The verifier's path relative to the work dir, as the shell commands name it."""
        return posixpath.join(self.scratch_dir, VERIFIER_NAME)

    @property
    def verifier_path(self) -> str:
        """The verifier's path under the current work dir."""
        return os.path.join(self.work_dir, *self.verifier_relative.split("/"))

    @property
    def verifier_shown(self) -> str:
        """The verifier's path as the verdicts name it (``./tmp/<task-id>/verify.sh``)."""
        return "./" + self.verifier_relative

    def relative(self, path: str, dot: bool = False) -> str:
        """*path* relative to the work dir when it is inside it (``./``-prefixed with *dot*).

        Args:
            path: An absolute or relative path.
            dot: Whether to prefix a path inside the work dir with ``./``.
        """
        relative = os.path.relpath(path, self.work_dir) if os.path.isabs(path) else path
        if relative.startswith(".."):
            return path
        return "./" + relative if dot else relative

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
        """
        self.work_dir = work_dir
        if fresh_base:
            self.base_resolved = False

    def observe(self, name: str, args: dict[str, Any]) -> None:
        """Record a tool call that is about to run.

        The first call that could change files fixes :attr:`base_commit`;
        an ``Edit`` / ``Write`` of a code file, or a writing shell command
        naming one, arms the gate.

        Args:
            name: The tool name.
            args: Its arguments.
        """
        if name in EDIT_TOOLS and self.is_scratch(args.get("file_path")):
            if self.is_verifier(str(args.get("file_path", ""))):
                self.verifier_passed_at = None  # the verifier itself changes: its last run is stale
            return  # a scratch note (tmp/PROGRESS.md, a probe): no change of the work
        self.evidence.observe(name, args)
        if name in EDIT_TOOLS or name in SHELL_TOOLS or name in DISPATCH_TOOLS:
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
                # a command that may rewrite the verifier (its run sets the
                # ledger entry again when it passes)
                self.verifier_passed_at = None
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
        """A digest of the work since the base: the tracked diff and the untracked files' contents.

        Taken before a local sub-task is dispatched and when its result
        is read: equal digests mean the sub-task changed no file of the
        work.  ``None`` outside git (the sub-task is then read as a change).
        """
        code, text = self.git(
            f"git diff {self.base_commit or 'HEAD'} -- && "
            "git ls-files --others --exclude-standard | git hash-object --stdin-paths")
        return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest() if code == 0 else None

    def is_verifier(self, path: str) -> bool:
        """Whether *path* (as an ``Edit`` / ``Write`` names it) is the agent's verifier."""
        return os.path.normpath(os.path.join(self.work_dir, path)) == os.path.normpath(
            self.verifier_path)

    def _verifier_in(self, commands: list[SimpleCommand]) -> SimpleCommand | None:
        """The first of the scanned *commands* that runs the agent's verifier, or ``None``.

        ``bash ./tmp/<id>/verify.sh`` (``sh``, with options other than
        ``-n``, which parses the script without running it), or the
        script itself as the command, from the work dir or after a
        ``cd`` the scanner follows.
        """
        verifier = posixpath.normpath(bash_path(os.path.abspath(self.verifier_path)))
        for simple in commands:
            argv = simple.argv()
            if not argv:
                continue
            words = argv[1:] if posixpath.basename(argv[0].text) in SHELLS else argv[:1]
            if any(NOEXEC_OPTION.fullmatch(word.text) for word in words):
                continue
            operands = [word for word in words if not word.text.startswith("-")]
            if operands and resolve(simple.cwd, operands[0]) == verifier:
                return simple
        return None

    def scan(self, command: str) -> list[SimpleCommand]:
        """The simple commands of the shell *command*, started in the work dir.

        Args:
            command: The command as the agent ran it.
        """
        return simple_commands(command, bash_path(os.path.abspath(self.work_dir)))

    def runs_verifier(self, command: str) -> bool:
        """Whether the shell *command* runs the agent's verifier (:meth:`_verifier_in`).

        Args:
            command: The command as the agent ran it.
        """
        return self._verifier_in(self.scan(command)) is not None

    def verifier_decides(self, command: str) -> bool:
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
        """
        commands = self.scan(command)
        simple = self._verifier_in(commands)
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
        """Note a change of the files: the next finish is reviewed again."""
        self.changes += 1
        self.changed_since_review = True
        self.reviewed_by_agent = False

    def merged(self) -> str:
        """Note that the worktree's merge went through: the finish is gated afresh.

        The rounds the merge spent and the findings of its review (met,
        or time-barred by the round cap) stay with the merge: the finish
        is refused on rounds of its own and reviewed again only when a
        file changes.  The verdicts open at a merge let through at the
        cap are returned for the merge's result (:data:`MERGED_WITH_FINDINGS`)
        and kept for the finish summary (:attr:`merge_caveats`); the
        squash merge moved HEAD, which the base follows no more.

        Returns:
            The text to append to the merge's result (``""`` when no
            finding was open).
        """
        self.moved_head = True
        self.gate_rounds = 0
        self.open_findings = []
        if not self.merge_caveats:
            return ""
        items = "\n".join(f"{i}. {v}" for i, v in enumerate(self.merge_caveats, 1))
        return MERGED_WITH_FINDINGS.format(rounds=self.merge_rounds, items=items)

    def is_scratch(self, path: Any) -> bool:
        """Whether *path* (as an ``Edit`` / ``Write`` names it) is a scratch file of the run.

        One under a ``tmp`` directory inside the work dir: a progress log,
        a probe, the verifier.  Writing one changes no work.
        """
        if not isinstance(path, str) or not path:
            return False
        relative = os.path.relpath(os.path.join(self.work_dir, path), self.work_dir)
        parts = relative.replace("\\", "/").split("/")
        return not relative.startswith("..") and "tmp" in parts[:-1]

    def result(self, name: str, args: dict[str, Any], text: str, refused: bool) -> None:
        """Record a tool call's result (the agent's ``tool_result_hook``).

        A shell result gives the exit-code ledger its entries
        (:func:`result_exit_codes`; the commands of a refused call are
        ``refused``).  The result of a sub-task is read by :meth:`subtask`.

        Args:
            name: The tool called.
            args: Its arguments.
            text: The result text the model reads.
            refused: Whether the call was refused (*text* is the refusal).
        """
        if refused:
            if name in SHELL_TOOLS:
                for command in shell_commands(args):
                    self.exit_codes[command.strip()] = "refused"
            return
        for command, code in result_exit_codes(name, args, text, self.background_jobs):
            self.exit_codes[command.strip()] = code
            if code == "0" and self.verifier_decides(command):
                # the agent's own passing run of its verifier, its exit the
                # command's: reused at a finish that nothing changed before
                self.verifier_passed_at = self.changes
                self.last_verifier_run = (0, text)
        if name in DISPATCH_TOOLS or name == "agent_job":
            self.subtask(name, args, text)
        elif name in self.custom_tools:
            self.follow_head()

    def subtask(self, name: str, args: dict[str, Any], text: str) -> None:
        """Read the result of a sub-task: a review of the work, or a change of it.

        A read-only review the agent dispatched (:func:`dispatches_review`)
        counts the work as reviewed once a sub-task of it succeeded
        (:func:`dispatch_successes`) and no file changed since its
        dispatch; one left running (a job id in the result) is read when
        ``agent_job`` reports it finished.  Any other sub-task may have
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
        successes = dispatch_successes(text)
        if name == "agent_job":
            job = self.jobs.get(str(args.get("job_id", "")).strip())
            if job is None or not successes:  # not a job of this run, or still running
                return
            review, changes, fingerprint = job
        else:
            if str(args.get("machine", "")).strip():
                return
            review, changes = dispatches_review(args), self.changes
            fingerprint = self.dispatch_fingerprint
            started = STARTED_AGENT_JOB.search(text)
            if started:
                self.jobs[started.group(1)] = (review, changes, fingerprint)
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
            self.evidence.unchecked_change = True

    def coding_run(self) -> bool:
        """Whether the run changed code or has a verifier (then it is gated).

        After a sub-task that may have changed files (:attr:`delegated`)
        the diff decides: a code file changed since the run began arms
        the gate.  A run declared no coding run (:attr:`coding`) never is.
        """
        if not self.coding:
            return False
        if not self.armed and self.verifier_exists():
            self.armed = True
        if not self.armed and self.delegated:
            self.armed = any(is_code_file(path) for path, _ in self.changed_paths() or [])
        if self.armed:
            self.evidence.track_changes = True
        return self.armed

    def verifier_exists(self) -> bool:
        """Whether the agent's verifier exists and is not empty (under the current work dir)."""
        return self.shell(f"test -s {shlex.quote(self.verifier_relative)}", 10)[0] == 0

    def verifier_run(self, giveup: bool = False) -> ShellRun:
        """Run the agent's verifier and return ``(exit code, output)``.

        The agent's own run of it is reused when it passed and nothing
        changed since (:attr:`verifier_passed_at`): the exit-code ledger
        already holds what a run now would show.

        Args:
            giveup: Whether a give-up runs it (recorded on the stage).
        """
        if self.verifier_passed_at == self.changes:
            gate_stage("verifier", 0.0, "reused", giveup=giveup)
            return self.last_verifier_run
        started = time.time()
        self.last_verifier_run = self.shell(f"bash {shlex.quote(self.verifier_relative)}",
                                            self.verifier_timeout)
        code = self.last_verifier_run[0]
        outcome = "pass" if code == 0 else "fail"
        gate_stage("verifier", time.time() - started, outcome, giveup=giveup)
        return self.last_verifier_run

    def finish_verdict(self, args: dict[str, Any], task: str) -> str | None:
        """Gate a ``finish``: the refusal text, or ``None`` to let the call through.

        An implicit (text-only) finish on a coding run is vetoed once with
        :data:`IMPLICIT_FINISH_VERDICT`.  A pause (``is_continue=True``)
        is never gated; a give-up is checked by :meth:`giveup_verdict`.  A
        success claim on a run that is not a coding run gets the check of
        the evidence it names and the hedge check; on a coding run the
        verifier and the review run at
        once (:meth:`run_stages`), then the evidence named, the hedge and
        unchecked-change check, the open-reading check and the review's
        findings are judged, and every failed stage is refused together
        (:func:`gate_refusal`) up to :data:`MAX_GATE_ROUNDS` rounds, after
        which the work is delivered with the open verdicts as caveats
        (:meth:`deliver_with_findings`).

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
            # not a coding run: the evidence named must still hold, and the
            # hedge check applies as it did before this gate existed
            return self.refusal("finish", [self.evidence_field_verdict(args)]) or (
                self.evidence.verdict(args))
        self.verifier_skipped = False
        stages = self.run_stages(args, task)
        verdicts = [stages["verifier"], self.evidence_field_verdict(args),
                    self.evidence.verdict(args), self.open_reading_verdict(args, task)]
        if "review" in stages:
            verdicts.append(stages["review"])
        else:
            verdicts.append(self.unanswered_verdict(args))
        refusal = self.refusal("finish", verdicts)
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
        need no summary run here first: the verifier, the unchecked-change
        check and the review of the diff (:meth:`run_stages`; the
        reviewer is told the summary is not written yet).  A merge asked
        for again with the last review's findings open and no file
        changed since is refused with them (:data:`MERGE_UNANSWERED_VERDICT`:
        a merge has no summary to answer them in).  The refusal counts as
        a gate round like a finish's; past the cap the merge goes through
        and its rounds and findings stay with it (:meth:`merged`): the
        finish is gated afresh and reviewed again only when a file
        changed.

        Args:
            task: The task statement.
        """
        if not self.coding_run():
            return None
        self.verifier_skipped = False
        stages = self.run_stages({"summary_in_html": MERGE_SUMMARY_PLACEHOLDER}, task)
        unchecked = None
        if self.evidence.unchecked_change:
            unchecked = UNCHECKED_VERDICT.format(reason=UNCHECKED_REASON)
        unanswered = None
        if stages.get("review") is None and self.open_findings and not self.changed_since_review:
            unanswered = MERGE_UNANSWERED_VERDICT.format(findings="\n".join(self.open_findings))
        verdicts = [stages["verifier"], unchecked, stages.get("review"), unanswered]
        open_verdicts = [v for v in verdicts if v]
        if open_verdicts and self.gate_rounds + 1 >= MAX_GATE_ROUNDS:
            # The round that would be refused last is not refused: the
            # merge goes through with its findings attached (the merge's
            # result, then the finish summary), so no run ends with its
            # work on a branch nobody merges.
            self.gate_rounds += 1
            self.merge_rounds = self.gate_rounds
            self.merge_caveats = open_verdicts
            gate_stage("delivered", 0.0, str(len(open_verdicts)), round=self.gate_rounds,
                       call="commit_and_merge")
            return None
        return self.refusal("commit_and_merge", verdicts)

    def refusal(self, call: str, verdicts: list[str | None]) -> str | None:
        """The refusal of *call* for the open *verdicts*, or ``None`` past the round cap.

        Args:
            call: ``finish`` or ``commit_and_merge``.
            verdicts: The stages' verdicts (``None`` for a stage that passed).
        """
        open_verdicts = [v for v in verdicts if v]
        if not open_verdicts or self.gate_rounds >= MAX_GATE_ROUNDS:
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
            verdicts[name] = f"Not finished: a check of the gate could not run ({error!r})."
            gate_stage(name, time.time() - started, "error")
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
        if (self.verifier_refusals >= MAX_VERIFIER_REFUSALS
                and self.verifier_passed_at != self.changes):
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

    def evidence_field_verdict(self, args: dict[str, Any]) -> str | None:
        """The verdict when a command the finish names as evidence did not run or did not pass.

        Args:
            args: Arguments of the ``finish`` call.
        """
        problems = evidence_problems(evidence_commands(args.get("evidence")), self.exit_codes)
        if not problems:
            return None
        return EVIDENCE_FIELD_VERDICT.format(problems="\n".join(problems))

    def evidence_report(self, args: dict[str, Any]) -> str:
        """The commands a finish names as evidence, with their last exit code, for the reviewer.

        Args:
            args: Arguments of the ``finish`` call.
        """
        commands = evidence_commands(args.get("evidence"))
        if not commands:
            at_merge = args.get("summary_in_html") == MERGE_SUMMARY_PLACEHOLDER
            return MERGE_EVIDENCE_PLACEHOLDER if at_merge else "(none named)"
        return "\n".join(f"- exit {self.exit_codes.get(c.strip(), 'not run')}: `{c[:300]}`"
                         for c in commands[:30])

    def open_reading_verdict(self, args: dict[str, Any], task: str) -> str | None:
        """Refuse once per run a summary that admits a case of the task is unhandled.

        The decisions model reads the task and the summary
        (:func:`leaves_a_case_open`); without it there is no verdict.

        Args:
            args: Arguments of the ``finish`` call.
            task: The task statement.
        """
        summary = str(args.get("summary_in_html", "")).strip()
        if self.open_reading_refused or not summary or not task:
            return None
        if leaves_a_case_open(task, summary) < OPEN_READING_THRESHOLD:
            return None
        self.open_reading_refused = True
        return OPEN_READING_VERDICT.format(verifier=self.verifier_shown)

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
        return self.shell(f'cd "$(git rev-parse --show-toplevel)" && {command}', DIFF_TIMEOUT)

    def changed_paths(self) -> list[tuple[str, bool]] | None:
        """The files changed in the work dir's repository since the run began.

        Returns:
            ``(path, untracked)`` per file, root-relative: every tracked path
            that differs from the run's :attr:`base_commit` (``HEAD`` when
            the run has none), committed or not, then every untracked file
            that was not there when the run began (:attr:`base_untracked`)
            or that the run wrote; ``None`` when the work dir is not in a
            git repository.
        """
        code, others = self.git("git ls-files --others --exclude-standard -z")
        if code != 0:
            return None
        changed, names = self.git(f"git diff --name-only -z {self.base_commit or 'HEAD'} --")
        entries = [(path, False) for path in names.split("\0") if changed == 0 and path]
        written = {self.root_relative(path) for path in self.written_files}
        entries.extend((path, True) for path in others.split("\0")
                       if path and (path not in self.base_untracked or path in written))
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

    def review_verdict(self, args: dict[str, Any], task: str) -> str | None:
        """A read-only review of the work by the reviewer model; its gaps as a verdict.

        The reviewer sees the task, the agent's summary, the verifier and
        its last output, the evidence named, and the diff
        (:meth:`diff_report`); a re-review also sees its previous findings.
        It runs no commands; its spend is charged to the run.  An answer
        without a ``VERDICT:`` line is retried once; a review that still
        has no verdict, or whose model call fails, refuses the finish
        (:data:`REVIEW_FAILED_VERDICT`) and keeps the previous findings
        open; it counts under :data:`MAX_FAILED_REVIEWS`, past which the
        finish goes through without a review.  Each ``NOT DONE`` finding
        is kept (:attr:`open_findings`) for the next finish to answer.

        Args:
            args: Arguments of the ``finish`` call.
            task: The task statement.
        """
        from kiss.core.kiss_agent import KISSAgent

        reviewer = self.reviewer
        if reviewer is None:  # only called when review_wanted(): a reviewer is set
            return None
        started = time.time()
        code, output = self.last_verifier_run
        previous, scope = "", ""
        if self.open_findings:
            previous = REVIEW_PREVIOUS.format(findings="\n".join(self.open_findings))
            scope = " since your previous review"
        prompt = REVIEW_PROMPT.format(
            task=task, summary=str(args.get("summary_in_html", "")),
            verifier=self.verifier_shown, code=code,
            script=self.read_file(self.verifier_relative, REVIEW_ONE_FILE_BYTES) or "(none)",
            output=output[-OUTPUT_CHARS:] or "(not run yet)",
            evidence=self.evidence_report(args), previous=previous, scope=scope,
            workdir=self.work_dir, files=self.diff_report(),
        )
        answer, cost, error = "", 0.0, ""
        for overrides in (self.model_config or None, None):
            agent = KISSAgent("Finish reviewer")
            charge_attempt = functools.partial(
                self._charge_review, agent, f"finish-review:{uuid.uuid4().hex}"
            )
            try:
                try:
                    answer = agent.run(model_name=reviewer, prompt_template="{p}",
                                       arguments={"p": prompt}, is_agentic=False,
                                       verbose=False, model_config=overrides)
                finally:
                    # Each attempt is charged as it ends, however it ends:
                    # a stop (which propagates) loses none of its spend.
                    charge_attempt()
            except Exception as failure:
                error = repr(failure)
            except BaseException:
                # A stop can land inside the finally's own charge: charge
                # again (once per key) before it propagates.
                charge_attempt()
                raise
            cost += agent.budget_used
            if error or review_complete(answer):
                break
        findings = review_findings(answer)
        failed = bool(error or not review_complete(answer))
        if failed:
            self.failed_reviews += 1
        else:
            self.reviews += 1
        fields: dict[str, Any] = {"model": self.reviewer, "cost_usd": round(cost, 6),
                                  "round": self.reviews, "findings": len(numbered_items(findings))}
        if error:
            fields["error"] = error[:500]
        gate_stage("review", time.time() - started,
                   "error" if failed else ("fail" if findings else "pass"), **fields)
        if failed:
            return REVIEW_FAILED_VERDICT if self.review_possible() else None
        self.changed_since_review = False
        self.open_findings = numbered_items(findings)
        if not findings:
            return None
        return REVIEW_VERDICT.format(findings=findings)

    def unanswered_verdict(self, args: dict[str, Any]) -> str | None:
        """The open review findings this finish answers neither by a change nor by evidence.

        A finding is answered when a file changed since the review and a
        re-review will judge the change, or when the decisions model finds
        the summary quoting a command and its output for it
        (:func:`answered_findings`).  Without the decisions model every
        finding counts as answered.

        Args:
            args: Arguments of the ``finish`` call.
        """
        if not self.open_findings:
            return None
        if self.review_possible() and self.changed_since_review:
            return None
        answered = answered_findings(str(args.get("summary_in_html", "")), self.open_findings)
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
        code, output = self.verifier_run(giveup=True)
        args["summary_in_html"] = str(args.get("summary_in_html", "")) + GIVEUP_NOTE.format(
            verifier=self.verifier_shown, state=check_state(code, output))
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
        answered = answered_findings(summary, self.merge_caveats) or [False] * len(
            self.merge_caveats)
        open_verdicts = [v for v, ok in zip(self.merge_caveats, answered, strict=True) if not ok]
        self.merge_caveats = []
        if not open_verdicts:
            return
        self.delivered_findings.extend(v.splitlines()[0][:500] for v in open_verdicts)
        items = "\n".join(f"<li>{html.escape(v)}</li>" for v in open_verdicts)
        args["summary_in_html"] = summary + MERGED_CAVEATS.format(rounds=self.merge_rounds,
                                                                 items=items)
