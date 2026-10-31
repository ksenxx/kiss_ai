# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Coding SEA: runs KISS Sorcar unattended inside a Docker container.

A trial runs the agent against one running Docker container that holds the
task's working directory.  The KISS daemon imports a tiny generated SEA file
per trial (see :func:`write_trial_sea`) that builds one
:class:`ContainerHarness` from a JSON config and exposes the harness's bound
methods through a ``TrialSea`` class.  The trial runners in ``benchmarkings/harnesstax``
are one such caller.  This module's own SEA class defines no prompt: a
harness needs a running container and a config, so the module is
``hidden`` and registers no ``/coding`` chat command.

The harness is KISS Sorcar's full built-in toolset except the browser and
persistent-memory tools, driven by a short system prompt written for
unattended engineering work in a container (:data:`SYSTEM_PROMPT`).  The shell and
file tools execute inside the trial container through the daemon's Docker mode
(``settings()["docker_image"] == "container:<id>"``), or on a remote sandbox
through an exec bridge (``"exec:<url>"``, :mod:`kiss.agents.sorcar.exec_bridge`),
so nothing the agent does touches the host.  There is no cap on the number of
steps; the only limit is the per-trial USD budget.  Every LLM call and every
tool call is appended to a JSONL trajectory file for later analysis.

The finish gate (config ``finish_gate``) is where the harness pushes back on a
``finish(success=True)`` that is not backed by evidence: the agent's own
verifier (:data:`VERIFIER_PATH`) must exist and pass, so must the independent
check a ``blind_verifier`` model wrote from the statement alone
(:data:`BLIND_VERIFIER_PATH`), every quote of the summary must be text a tool
returned during the run (:mod:`kiss.agents.seas.coding.quoted_evidence`), the
summary must not hedge or leave a reading open (the agent's evidence gate
refuses it), the findings of the previous review must be answered by a change
or quoted evidence, and a configured ``reviewer`` model reads the work (then
the diff) against the statement.  The agent is also held to an interpretation
ledger (:data:`LEDGER_PATH`) before its first edit, gets checkpoint/rollback
tools, background jobs, and a machine-written inventory at every context
restart.  Each answers a way an unattended run goes wrong: a plausible
result claimed without a check, a reading of the statement settled by
assumption, a reviewer argued down instead of answered, an edit made after
the last check, a context restart that forgets what was already tried.

The gate trusts nothing the agent can reach from its shell: the independent
check's script and the verified checkpoint's copies of the scripts it passed
live on the host (the container copy of the check is rewritten before each
run), a script runs as the argument of ``bash -c`` rather than from a file in
the container, a checkpoint archive is restored only when its SHA-256 is the
one recorded when it was saved, and the quotes of the summary are checked
against the trajectory the harness itself wrote.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shlex
import subprocess
import tarfile
import threading
import time
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import ALLOW, Verdict, WorkerSea, refuse
from kiss.agents.seas.coding import coding_test_context as test_context
from kiss.agents.seas.coding.quoted_evidence import (
    MAX_QUOTE_SHOWN_CHARS,
    MAX_QUOTES_SHOWN,
    Source,
    quote_sources,
    quotes_report,
    unsupported_quotes,
)
from kiss.agents.sorcar.checkpoints import CHECKPOINT_NAME
from kiss.agents.sorcar.container_jobs import KILL_ALL_SCRIPT
from kiss.agents.sorcar.docker_manager import _EXEC_TOKEN_VAR, kill_tagged_tree_script
from kiss.agents.sorcar.edit_guards import heredoc_guard, heredoc_targets, rewrite_guard
from kiss.agents.sorcar.evidence_gate import EDIT_TOOLS, SHELL_TOOLS, EvidenceGate
from kiss.agents.sorcar.exec_bridge import BRIDGE_PREFIX, bridge_alive, bridge_exec
from kiss.agents.sorcar.shell_guards import (
    INSTALL_COMMANDS,
    INSTALL_TIMEOUT_SECONDS,
    destructive_pattern,
)

#: Harness instances by config path (see :meth:`ContainerHarness.shared`).
_HARNESSES: dict[str, ContainerHarness] = {}
_REGISTRY_LOCK = threading.Lock()

#: The system prompt of an unattended, containerised engineering task.  It
#: replaces Sorcar's default prompt, which is written for interactive work on
#: the user's own machine (worktrees, reports, periodic summaries, web
#: research) and says nothing about how to solve a task nobody will check.
SYSTEM_PROMPT = """\
You are KISS Sorcar, an autonomous software engineer. You work alone in a \
Linux container whose working directory is {workdir}. Nobody reads anything or \
answers questions before you call finish. Complete the task in the user message \
end to end.

# Tools
- Bash runs one shell command in a fresh shell started in {workdir}; use \
absolute paths or `cd` inside the command. timeout_seconds defaults to \
{shell_timeout}; raise it for longer commands (package installs and builds: at \
least 900). run_commands_parallel runs several independent commands at once; \
its timeout applies to each. Anything that may take more than a minute \
(builds, installs, training, downloads, servers) runs as a background job: \
Bash(command, background=True) returns a job id at once; bash_job(job_id, \
action="wait", timeout_seconds=N, until="<regex>") blocks until the job exits \
or its log matches the regex (one long wait, never a loop of short polls or \
`sleep N; tail` calls); action="tail" shows the log, action="kill" stops it. \
Bash(..., on_timeout="background") keeps a command that outlives its deadline \
running as a job instead of killing it.
- Read shows a file with line numbers (use start_line and max_lines on large \
files). Edit replaces one exact string in a file; Write creates or overwrites \
a whole file. Read the region you change before you Edit it. Existing files \
are changed with Edit and Write only: a shell command that rewrites one \
through a here-document, a multi-line `sed -i` or a Python rewrite is refused \
(a here-document that creates a new file is fine).\
{test_context}
- checkpoint(name) saves a snapshot of {workdir}; rollback(name) restores it. \
The harness also snapshots the work tree whenever the finish checks pass, and \
a run that ends without an accepted finish is restored to the last snapshot \
that passed. Snapshot before a risky or large change; roll back when the \
result is not strictly better.
- finish ends the run (see step 8). summary_in_html states what you did and, \
as a checklist, each requirement of the task with the command that checked \
it and the result. There is no user to ask (ask_user_question, talk), the \
model is fixed (set_model), and progress summaries are not needed.
- Shell results may end with harness notes on facts you cannot otherwise see: \
processes a command left running, and pre-existing files under {workdir} a \
command modified or deleted. Act on them.

# How to work
1. Read the statement twice and write the interpretation ledger before the \
first edit. Nobody will answer a question, so the ledger at {ledger} (Write \
it; editing a file under {workdir} is refused until it exists) settles every \
decision in advance, one entry per line: (a) every sentence of the statement \
that admits two readings, with the reading you choose and why; (b) every \
supplied input (file, column, field, parameter, rule, example) and what in \
your solution will consume it; (c) every tolerance, format, rounding or \
ordering the statement leaves open, with the value you choose; (d) what \
"done" means for each requirement (files, names, formats, commands, \
interfaces, thresholds). Tie-breakers: an input that nothing consumes means \
your reading is wrong, go back ("no role identified" is not an entry, it is \
the reading you have not found yet); the statement promises an outcome, so a \
result that falls short of it (infeasible, unsolvable, no answer, your own \
"done" criterion unmet) is a wrong reading or a missed fix, never "the data"; the \
literal sentence beats your domain prior and beats a design note or \
docstring that the reported symptom contradicts; when a tolerance is \
undefined, take the strict one; when the statement describes an existing \
structure (file names, function names, call order, output shape), keep its \
shape. Update the ledger when you learn more; it is read by the reviewer.
2. Look before you change. Inspect the working directory (hidden files too), \
the files the task mentions, and the tools and versions available. In a code \
base, find the relevant code with grep, read it and its callers, and follow \
its conventions. Do not modify or delete the task's input files; experiment \
on copies.
3. Write your own verifier before you build. Turn the ledger's "done" entries \
into a script at {verifier} (Write it; bash runs it) with one check per \
requirement: the exact file names, formats, field names, values, thresholds, \
edge cases and interfaces the statement names, plus any example it gives. It \
must exit non-zero on the first failing check and print what failed. Prefer \
an external oracle (a reference implementation, a public dataset, a formula \
in the task's own documentation, a library that computes the same thing) over \
your own reimplementation of the answer. When a reference, golden data or a \
test set exists, hold part of it out of what you tune on and check against \
the held-out part. Keep the verifier current as your understanding improves; \
never weaken it to make it pass. finish also runs an independent check \
written from the statement alone by another agent; when the two disagree, \
resolve the disagreement with evidence, not by a vote.
4. For a task that says fix, debug or repair: reproduce the symptom, locate \
the cause, and apply the smallest change that makes the symptom disappear and \
the existing tests pass. Do not refactor, rename, reorganise or "clean up" \
around it unless the statement asks: the project's tests and callers rely on \
the code's existing structure (names, files, imports, call order, output \
shape), and every extra change is a chance to break what they rely on. Run \
the project's own tests before and after.
5. Work in small verified steps. For something to build or produce, get a \
complete working version in place first, checkpoint it, then improve it; \
never replace what works with something unverified. Keep changes minimal. \
Install standard tools and libraries rather than re-implementing them. When \
the task has a measurable objective (a score, an error, a speed) and budget \
remains, build more than one candidate and keep the one your verifier and \
held-out data rate best.
6. Verify before you finish. Re-read the task statement and the ledger and \
check every requirement directly, from a fresh shell: run the command, open \
the file, parse the output, measure the number. Run your verifier and the \
tests that cover what you changed. Treat any error, traceback or unexpected \
output in your own checks as a problem to resolve, not to explain away. Never \
weaken a test, assertion or threshold to make a check pass. Never repeat a \
failed command without changing something first.
7. Leave the environment as the task expects to find it. Keep the finished \
work in place; do not undo or reset it. Leave no stray processes or scratch \
files to confuse the next user of the container. Kill only processes you \
started; wait for or kill every background job.
8. Be honest at the end. Call finish with success=True only when every \
requirement is met and verified by a command you ran; otherwise call it with \
success=False and a precise account of what is done and what remains. A \
plausible result you could not check is not a success, and a summary that \
leaves a reading open ("if the task instead means", "a judgement call") is \
refused: resolve it with evidence first. Quote evidence verbatim from a tool \
result (a command's output, a file you read): a quote that no tool result \
of this run contains is refused. finish runs your verifier, the independent \
check and a read-only review; each finding of the review must be answered \
by a change or by quoted evidence from a command, not by argument.

# Time and effort
The task is sized for hours of expert work and you have the whole budget \
(the Budget line under each tool result shows what is left). A first \
plausible answer is usually incomplete: keep going while a requirement is \
unverified or your verifier fails, and spend remaining budget on the hardest \
requirement rather than on finishing early. Probe with short timeouts; use \
all cores where possible (`-j$(nproc)`). Keep tool output small (grep -n, \
head, tail, line ranges), batch independent commands into one call, and do \
not re-read unchanged files. Do the task yourself in one pass; sub-agents are \
rarely worth their cost here.
"""

#: Where the agent keeps its own verifier (outside the workdir: the task's
#: verifier must not find it, and a stray file there could fail a listing check).
VERIFIER_PATH = "/tmp/kiss-verify.sh"
#: Seconds the gate gives the agent's verifier.
VERIFIER_TIMEOUT = 900
#: Refusals the verifier may cause per trial (missing or failing); a verifier
#: the agent cannot make pass must not trap the run.
MAX_VERIFIER_REFUSALS = 3
#: Default ``Bash`` timeout of a trial (the tool's own default is 30 s, which
#: killed test runs and builds half way).
DEFAULT_SHELL_TIMEOUT = 120

#: Where the agent keeps its interpretation ledger (step 1 of the prompt).
LEDGER_PATH = "/tmp/kiss-ledger.md"
#: Edits refused for a missing ledger, per trial; a ledger the agent will not
#: write must not trap the run.
MAX_LEDGER_REFUSALS = 2
LEDGER_VERDICT = (
    "REFUSED: no interpretation ledger yet. Before the first edit under {workdir}, Write "
    "{ledger}: one line per ambiguous sentence of the statement with the reading you "
    "choose and why, per supplied input and what consumes it, per open tolerance, format "
    "or ordering with the value you choose, and per requirement what 'done' means. Then "
    "retry the edit."
)

#: Answer to an implicit (text-only) finish when the finish gate is on: the
#: agent is asked for an explicit ``finish`` call, which the gate checks.
IMPLICIT_FINISH_VERDICT = (
    "Not finished: a text reply does not end the run. Call finish explicitly with "
    "success=True only if every requirement is met and verified by a command you ran "
    "(the gate runs your verifier, the independent check and a review), otherwise with "
    "success=False and what remains."
)
VERIFIER_MISSING_VERDICT = (
    "Not finished: your own verifier {path} does not exist. Write it from the "
    "task statement (one check per requirement: exact file names, formats, field names, "
    "values, thresholds, edge cases, interfaces, the statement's examples; exit non-zero "
    "on the first failure and print what failed), run `bash {path}`, fix what fails, "
    "then call finish again."
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

#: Largest total size of changed files shown to the reviewer.
REVIEW_FILE_BYTES = 120_000
#: Largest size of one changed file shown to the reviewer (longer ones are cut).
REVIEW_ONE_FILE_BYTES = 30_000
#: Reviews per trial: the first on the whole work, later ones on the diff since
#: the previous verdict (``max_reviews`` in the config).
MAX_REVIEWS = 3
REVIEW_PROMPT = """\
You are reviewing, read-only, the work of an autonomous engineer on the task below. \
Nobody else will check it before it is delivered. \
Decide whether every requirement of the task statement is demonstrably met by the \
evidence (the engineer's summary and checklist, their interpretation ledger, their \
verifier script and its last output, the independent check and its output, and the \
files they changed). Every quote in the summary was checked against the run's tool \
results; the provenance section says which command or file each quote came from. A \
quote is evidence of a requirement only when its source tests that requirement: the \
output of `echo` or `cat`, the output of a script the engineer wrote themselves (marked \
as such), or the words of the statement prove nothing about the work on their own. Do \
not invent problems: report only a requirement that is not \
met, not checked, or checked wrongly, each with the evidence that shows it. Take the \
statement literally (exact names, formats, values, thresholds, edge cases, examples); \
where the ledger chooses a reading, check the choice against the statement's words. \
A symptom, scenario or input of the statement that the summary admits is not handled \
("I added nothing for", "left as is", "known limitation") is a gap, whatever the \
reason given, unless the statement itself rules it out. So is an input the task \
offers that the delivered work never uses (a data field, a setting, a clock, a method \
the ledger lists as "no role identified"): the statement promises an outcome, and a \
result short of it (infeasible, unsolvable, no answer) means a reading is wrong, not \
the data. Do not pad the answer: the verdict and the numbered gaps only.

Answer with one line `VERDICT: OK` when every requirement is demonstrably met, or \
`VERDICT: NOT DONE` followed by a numbered list of the gaps, each one sentence of \
the form "<requirement>: <what is missing or wrong, with evidence>".

# Task statement
{task}

# The engineer's final summary
{summary}

# The engineer's interpretation ledger ({ledger})
```
{ledger_text}
```

# The engineer's verifier ({verifier}), exit code {code}
```
{script}
```
Last output:
```
{output}
```

# The independent check ({blind}), exit code {blind_code}
```
{blind_script}
```
Last output:
```
{blind_output}
```

# Where the summary's quotes come from
{quotes}
{previous}
# Files the engineer created or changed under {workdir}{scope}
{files}
"""
#: Section of a re-review: the findings of the previous round and how the
#: engineer answered them.
REVIEW_PREVIOUS = """
# Findings of your previous review
{findings}

The engineer's answer to them (the summary above is the new one). A finding is \
settled only by a change in the files listed below that meets it, or by a command \
the engineer quotes with its output that shows it wrong. It is not settled by \
reasoning, by a design document or docstring the statement's symptom contradicts, \
by a check added to the engineer's own verifier (it encodes their reading), or by \
"the data makes it impossible". Report every finding that is still unmet, in the \
same words if nothing changed.
"""
REVIEW_VERDICT = (
    "Not finished: an independent review of the work against the task statement "
    "found these gaps:\n{findings}\n"
    "Answer every numbered finding in your next finish: either change the work (the "
    "review then runs again on the files changed since) or quote in summary_in_html, "
    "under the finding's number, the command you ran and its output that shows the "
    "finding wrong. An argument without a change or quoted evidence does not count."
)
#: Failed reviews (no verdict twice, or a model error) a trial tolerates before
#: a finish goes through without one; they do not use up ``max_reviews``.
MAX_FAILED_REVIEWS = 2
#: A review that produced no verdict twice (or whose model call failed): the
#: finish is refused, not approved by silence, while another review is possible.
REVIEW_FAILED_VERDICT = (
    "Not finished: the independent review of the work did not complete (no verdict "
    "came back), so it counts as neither approval nor rejection. Re-check the work "
    "against the task statement and call finish again; the review runs again."
)
#: A finding the next finish neither changed code for nor answered with evidence.
UNANSWERED_VERDICT = (
    "Not finished: these findings of the review were not answered by a change or by "
    "quoted command output:\n{findings}\n"
    "Change the work, or quote under each number the command and output that shows "
    "the finding wrong, then call finish again."
)
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
#: leaving a case of the statement unhandled or the asked outcome missed
#: (probed: a summary that says "I added nothing for <a listed symptom>"
#: scores above 0.9 on the first question, one that reports the asked
#: artifact as "not feasible" above 0.9 on the second; summaries reporting
#: the outcome produced and every symptom checked score below 0.2).
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
#: The finish refused once per trial for a summary that leaves a case open.
OPEN_READING_VERDICT = (
    "Not finished: your summary says that a symptom, scenario or requirement of the task "
    "statement was left unhandled or settled by assumption, or that the result falls short "
    "of what the statement asks for (infeasible, unsolvable, failing your own definition "
    "of done). Every such admission is a requirement unmet: the statement promises an "
    "outcome, so a result short of it means a reading or a fix is still wrong, not that "
    "the data is. For each item: either handle the case and add a "
    "check for it to {verifier} (a supplied input that nothing consumes yet is the usual "
    "way in), or quote the sentence of the statement or the data that "
    "rules it out. Then call finish again."
)

#: Where the independent check lives in the container.
BLIND_VERIFIER_PATH = "/tmp/kiss-blind-verify.sh"
#: Steps and budget share the blind verifier's agent gets.
BLIND_VERIFIER_STEPS = 40
BLIND_VERIFIER_BUDGET_FRACTION = 0.1
#: Share of the verifier timeout the author is told its script may take (the gate
#: runs the script at every finish, so a check written to the full timeout
#: would spend the whole timeout on each one).
BLIND_VERIFIER_TIME_FRACTION = 0.25
BLIND_VERIFIER_PROMPT = """\
You are an independent checker, not the engineer. An autonomous engineer is solving the \
task below in a Linux container whose working directory is {workdir}; you have a shell \
in the same container (Bash) and must write the acceptance checks for their work, from \
the task statement and the task's own inputs alone. Do not look at the engineer's solution, \
notes or scripts (anything under /tmp/kiss-* or files the task did not supply); you check \
the statement, not their reading of it.

Write a bash script that exits non-zero on the first failing check and prints what \
failed, with one check per requirement of the statement: exact file names, formats, \
field names, values, thresholds, edge cases, interfaces, the statement's examples. \
Take the statement literally. Prefer an external oracle over your own arithmetic: a \
reference implementation or library installable with pip or apt, a public dataset or \
documented formula, the project's own documentation or tests. Inspect the inputs \
with the shell first (sizes, headers, a few rows) so the checks use real column names \
and values. Keep each check fast (the whole script must run in under {timeout} \
seconds: time it once with `time bash <script>` before you deliver it, and shrink \
the inputs of a slow check rather than its coverage) and deterministic. A check must \
not modify anything under {workdir}.

Call finish(script=...) once with the complete script as its argument.

# Task statement
{task}
"""
BLIND_VERIFIER_FAILED_VERDICT = (
    "Not finished: an independent check written from the task statement alone "
    "({path}) exited {code}:\n{output}\n"
    "Resolve the disagreement with evidence: if the check is right, fix the work; if "
    "it misreads the statement, quote the statement's words that show it and add the "
    "reading to your ledger. The gate runs its own copy of the check: editing or "
    "deleting {path} changes nothing. Then call finish again."
)
#: A summary quoting text that no tool result of the run contains.  The first
#: line is also the marker by which the refusal itself, once it comes back as
#: a tool result, is kept out of the sources (it repeats the unsupported quotes).
QUOTED_EVIDENCE_MARKER = "summary_in_html quotes text that no tool result of this run contains"
QUOTED_EVIDENCE_VERDICT = (
    "Not finished: " + QUOTED_EVIDENCE_MARKER + ":\n{quotes}\n"
    "A quote is evidence only when it is the verbatim text a tool returned to you: the "
    "output of a command you ran, or the content of a file you read. For each item, run "
    "the command (or Read the file) and quote the result as returned, or remove the "
    "quote. Then call finish again."
)
#: Longest blind-verifier output shown.
BLIND_OUTPUT_CHARS = 3000

#: Where checkpoints (tar archives of the workdir) live in the container.
CHECKPOINT_DIR = "/tmp/kiss-checkpoints"
#: Largest workdir (MB) that is snapshotted.
MAX_CHECKPOINT_MB = 4096
#: Name of the automatic checkpoint taken when the finish checks pass.
VERIFIED_CHECKPOINT = "verified"
#: The digest ``sha256sum`` prints for an archive.
SHA256_HEX = re.compile(r"[0-9a-f]{64}")

#: Shell fragments that would end the trial itself: killing every process in
#: the container (the agent's own shell included) or deleting the workspace.
#: ``_END`` closes a shell word: whitespace, end of command, a separator,
#: a redirection or a comment.  ``_CMD`` is where a command name may start:
#: the beginning or a separator, then optional ``sudo`` or environment
#: assignments and an optional directory prefix.
#: The patterns live in :mod:`kiss.agents.sorcar.shell_guards`, shared with the
#: host and Docker ``Bash`` tools; the harness applies them in its tool hook
#: because a HarnessTax container's tools run outside :class:`DockerManager`.
#: A foreground ``sleep N`` of at least this many seconds is a poll: refused in
#: favour of ``bash_job`` (each poll is a model call, and a round of context,
#: spent waiting; over a long run they add up to a large share of both).
SLEEP_POLL_SECONDS = 10
SLEEP_POLL = re.compile(r"(?:^|[;&|(]|\bthen|\bdo)\s*sleep\s+(\d+(?:\.\d+)?)[a-z]?\b")
#: Quoted strings are dropped before the poll check: a sleep inside ``sh -c '...'``
#: or a here-document body belongs to the program, not to the model's waiting.
SHELL_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
SLEEP_VERDICT = (
    "REFUSED: the command did not run: a foreground `sleep {seconds}` is a model call "
    "spent waiting. Start the long-running command with Bash(command, background=True) "
    "(it returns a job id at once) and block on it in one call with "
    "bash_job(job_id, action='wait', timeout_seconds=N, until='<regex of the log line "
    "you are waiting for>'); for a job already running, bash_job(job_id, action='wait') "
    "or action='tail'. Re-issue the rest of this command without the sleep."
)
DESTRUCTIVE_VERDICT = (
    "Blocked: this command would kill every process in the container (your own "
    "shell included) or delete the working directory, which ends the task with "
    "nothing to deliver. Kill only the processes you started, by pid or exact name, "
    "and delete only files you created."
)

#: Largest file the test-context note reads back and diffs.
MAX_SNAPSHOT_BYTES = 400_000

#: Sentence added to the system prompt when the test-context notes are on.
TEST_CONTEXT_NOTE = (
    " After you change existing source code, the tool result may list the "
    "project's tests that reference the changed definitions: they define the "
    "expected behaviour, so read the relevant ones and run them."
)


#: Tools that need a human or the outside world (``run_agent`` would dispatch
#: a host-side channel agent); a headless run answers them itself instead of
#: blocking forever (the run has no wall-clock limit) or leaving the container.
UNATTENDED_TOOL_VERDICTS = {
    "ask_user_question": (
        "No user is available in this unattended run. Decide for yourself, "
        "state your assumption in the final summary, and continue."
    ),
    "talk": "No user is listening in this unattended run; continue with the task.",
    "run_agent": (
        "No external agents or messaging channels exist in this unattended, "
        "container-isolated run; do the work yourself with the other tools."
    ),
    "set_model": (
        "The model is fixed for this run (the run measures one model); continue "
        "with the current one."
    ),
}

#: Lists every process in the container as ``pid ppid starttime cmdline`` from
#: /proc alone (``ps`` is missing from many task images); the start time (stat
#: field 22) tells a reused pid from the process that had it.  ``KISS_PS=1``
#: tags the snapshot's own shell so it and its children can be dropped.
PROCESS_LIST_COMMAND = (
    "KISS_PS=1; for d in /proc/[0-9]*; do p=${d#/proc/}; "
    "s=$(sed 's/.*) //' \"$d/stat\" 2>/dev/null | cut -d' ' -f2,20); "
    "c=$(tr '\\0' ' ' < \"$d/cmdline\" 2>/dev/null | head -c 200); "
    "[ -n \"$c\" ] && [ -n \"$s\" ] && printf '%s %s %s\\n' \"$p\" \"$s\" \"$c\"; done; true"
)
#: Shell commands that try to stop a process; the survivors note is repeated
#: after them while a process the agent started is still alive.
KILL_COMMANDS = re.compile(r"\b(?:kill|pkill|killall|fuser|pgrep)\b|\bstop\b")
#: Regular files under the workdir as ``size mtime path`` (``.git``, bytecode
#: and caches excluded).  The listing is capped so a huge tree just switches
#: the input-file notes off instead of slowing every turn.
MAX_TRACKED_FILES = 40_000
FILE_LIST_COMMAND = (
    "find {workdir} -xdev \\( -name .git -o -name __pycache__ -o -name .pytest_cache "
    "-o -name node_modules -o -name .tox -o -name .mypy_cache \\) -prune -o -type f "
    "! -name '*.pyc' -printf '%s %T@ %p\\n' 2>/dev/null | head -n " + str(MAX_TRACKED_FILES + 1)
)
#: Longest ``ls -la`` shown with the task statement.
MAX_LISTING_LINES = 60
#: Processes started by the last shell call and still alive when it returned.
SURVIVORS_NOTE = (
    "\n\nHarness note: processes started by your last shell command(s) are still running: {procs}. "
    "Before you finish, stop the ones that are not meant to keep running."
)
#: Processes the agent started earlier that outlived a command meant to stop something.
STILL_RUNNING_NOTE = (
    "\n\nHarness note: processes you started earlier are still running after your last shell "
    "command(s): {procs}."
)
#: Pre-existing files a shell command changed or deleted.
INPUT_FILES_NOTE = (
    "\n\nHarness note: your last shell command(s) changed pre-existing files under {workdir}: "
    "{files}. "
    "Make sure this is a change the task asks for; if not, restore them and work on copies."
)


class CodingSea(WorkerSea):
    """The ``/coding`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/coding help``."""
        return (
            "Runs KISS Sorcar unattended inside a Docker container for benchmark trials "
            "(a ContainerHarness built by write_trial_sea from a JSON config, with the shell "
            "and file tools executing in the trial container and every call logged to a JSONL "
            "trajectory); the trial runners load the generated per-trial SEA file by path, so "
            "this module is hidden from the command list."
        )

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Return the module's own settings: hidden (only the generated per-trial SEAs run)."""
        return settings | {"hidden": True}


class ContainerHarness:
    """One trial: a container, a task prompt, a model, a trajectory log.

    Instances are created from a JSON config written by the trial runner::

        {
          "container": "<docker container name or id>",
          "workdir": "/testbed",
          "prompt": "<task instruction>",
          "model": "claude-fable-5",
          "max_budget": 50.0,
          "work_dir": "/host/scratch/dir",
          "trajectory": "/host/path/trajectory.jsonl"
        }
    """

    def __init__(self, config_path: str) -> None:
        """Load the trial configuration.

        Args:
            config_path: Absolute path of the trial's JSON config.
        """
        cfg = json.loads(Path(config_path).read_text())
        self.config_mtime_ns = Path(config_path).stat().st_mtime_ns
        self.container: str = cfg["container"]
        self.workdir: str = cfg.get("workdir", "/")
        self.task_prompt: str = cfg["prompt"]
        self.model_name: str = cfg["model"]
        self.budget: float = float(cfg.get("max_budget", 50.0))
        self.host_work_dir: str = cfg.get("work_dir", "")
        self.trajectory_path = Path(cfg["trajectory"])
        self.turns = 0
        self._lock = threading.Lock()
        #: Whether edits get the "tests that reference this code" note (off by
        #: default).
        self.test_context: bool = bool(cfg.get("test_context", False))
        #: Whether ``finish(success=True)`` is gated (:meth:`finish_verdict`).
        self.finish_gate: bool = bool(cfg.get("finish_gate", False))
        #: Model of the read-only review a gated finish gets (``""``: no review).
        self.reviewer: str = str(cfg.get("reviewer") or "")
        #: Model request overrides for this trial (``reasoning_effort`` for OpenAI-style
        #: models, ``output_config={"effort": ...}`` for Claude); empty means provider defaults.
        self.model_overrides: dict[str, Any] = dict(cfg.get("model_config") or {})
        self.implicit_finish_vetoed = False
        self.verifier_refusals = 0
        #: Whether the once-per-trial refusal of a summary leaving a case open was spent.
        self.open_reading_refused = False
        #: Whether both verifiers actually passed at the current finish (a verifier
        #: skipped because its refusals are used up does not count: the
        #: ``verified`` checkpoint is taken only when this is still true).
        self.checks_passed = False
        #: Seconds the gate gives the agent's verifier (config ``verifier_timeout``).
        self.verifier_timeout = float(cfg.get("verifier_timeout", VERIFIER_TIMEOUT))
        #: ``(script, exit code, output)`` of the verifier's last run by the gate.
        self.last_verifier_run: tuple[str, int | None, str] = ("", None, "")
        #: Model of the independent check's author (``""``: no blind verifier;
        #: config ``blind_verifier``, ``same`` for the trial's model).
        blind = str(cfg.get("blind_verifier") or "")
        self.blind_verifier: str = self.model_name if blind == "same" else blind
        #: Whether the blind verifier's agent ran (once per trial).
        self.blind_written = False
        #: The independent check's script, kept on the host: the gate runs this
        #: text, not the container's copy, which the agent can rewrite.
        self.blind_script = ""
        #: ``(script, exit code, output)`` of the blind verifier's last run by the gate.
        self.last_blind_run: tuple[str, int | None, str] = ("", None, "")
        #: Where each quote of the last finish summary came from (:func:`quote_sources`).
        self.quote_provenance: dict[str, str] = {}
        #: Reviews done so far and the open findings of the last one.  A review
        #: that fails (no verdict twice, or a model error) counts under
        #: ``failed_reviews`` instead, capped by :data:`MAX_FAILED_REVIEWS`.
        self.reviews = 0
        self.failed_reviews = 0
        self.max_reviews = int(cfg.get("max_reviews", MAX_REVIEWS))
        self.open_findings: list[str] = []
        #: Workdir snapshot (:meth:`file_snapshot`) when the last review ran.
        self.review_snapshot: dict[str, tuple[str, str]] | None = None
        self.ledger_refusals = 0
        #: Whether the ledger file was seen in the container (checked until it is).
        self.ledger_seen = False
        #: Checkpoints taken, in order; ``verified`` when the finish checks passed.
        self.checkpoints: list[str] = []
        #: SHA-256 of each checkpoint's archive when it was saved; a restore
        #: checks it first (the archive lives in the container, where the
        #: agent could replace it).
        self.checkpoint_digests: dict[str, str] = {}
        #: ``bash`` as the trial's PATH resolved it at the start, and the host's
        #: copy of that file, taken through the Docker daemon at the first model
        #: call (``None`` on a bridged trial, or when the copy failed): the gate
        #: runs every check with a fresh copy of it, not with whatever the
        #: agent left at the path (:meth:`run_script`).
        self.shell_path = "bash"
        self.host_shell: Path | None = (self.host_shell_path if self.host_shell_path.is_file()
                                        else None)
        #: Inventory for a context restart: files read, commands and their exit codes.
        self.files_read: dict[str, None] = {}
        self.commands_run: list[tuple[str, str]] = []
        #: The tool calls since the last model call, in order: ``(tool name, command)``
        #: with the command of a ``Bash`` call and ``""`` for any other tool, so the
        #: turn's results pair with the right calls (:meth:`record_exit_codes`).
        self.turn_commands: list[tuple[str, str]] = []
        #: Refuses once a success claimed without evidence (a hedged summary, an
        #: unchecked final edit).
        self.evidence = EvidenceGate(track_changes=True)
        #: Wall-clock start (the first model call), for the gate's elapsed time.
        self.started: float | None = None
        self.destructive = destructive_pattern(self.workdir)
        #: ``(path, content before the edit)`` of the edits made since the last model call.
        self.pending_edits: list[tuple[str, str]] = []
        #: Test files already pointed out to the model.
        self.shown_tests: set[str] = set()
        #: Turn of the last edit (a test-context note is only shown for edits that landed).
        self.last_edit_turn = 0
        #: Process listing taken before the first shell call of the current turn
        #: (``None`` when no shell tool ran since the last model call).
        self.processes_before: dict[tuple[int, str], tuple[int, str]] | None = None
        #: Whether a shell command of the current turn tried to stop a process.
        self.kill_attempted = False
        #: ``(pid, start time)`` of processes the agent started that outlived the call that
        #: started them.
        self.tracked_pids: set[tuple[int, str]] = set()
        #: Shell notes that found no tool result to attach to; retried on the next model call.
        self.pending_notes: list[str] = []
        #: ``path -> (size, mtime)`` of the workdir's files when the run started
        #: (``None`` until the first model call; empty when tracking is off).
        self.file_baseline: dict[str, tuple[str, str]] | None = None
        #: Pre-existing files the agent edited with Edit/Write or that were already reported.
        self.owned_files: set[str] = set()

    @classmethod
    def shared(cls, config_path: str) -> ContainerHarness:
        """Return the process-wide harness for *config_path*, creating it once.

        The daemon imports a self-contained SEA more than once per run; every
        import must see the same turn counter, so instances are cached per
        config path.  A config file rewritten since the cached instance was
        built (a rerun of the same trial directory) replaces the stale instance.

        Args:
            config_path: Absolute path of the trial's JSON config.
        """
        with _REGISTRY_LOCK:
            harness = _HARNESSES.get(config_path)
            if harness is None or harness.config_mtime_ns != Path(config_path).stat().st_mtime_ns:
                harness = cls(config_path)
                _HARNESSES[config_path] = harness
            return harness

    # ---- what the generated trial SEA's methods return ---------------------

    def prompt(self, task: str = "") -> str:
        """The task instruction shown to the model, followed by ``ls -la`` of the workdir.

        The trial SEA's ``prompt(task)``: the instruction is the config's
        ``prompt`` (the trial runner's ``run_agent`` task text is only a
        label), or *task* when the config has none.  The listing costs
        nothing and settles the first question of every run (what is
        here?), so the model sees the task's own binaries and data from
        the start.
        """
        instruction = self.task_prompt or task
        listing = self.workdir_listing()
        if not listing:
            return instruction
        return (f"{instruction}\n\n---\n`ls -la {self.workdir}` when the run started:\n"
                f"{listing}")

    def workdir_listing(self) -> str:
        """``ls -la`` of the workdir, cut to :data:`MAX_LISTING_LINES` lines ('' on failure)."""
        out = self.container_exec(["ls", "-la", self.workdir])
        if not out:
            return ""
        lines = out.splitlines()
        if len(lines) > MAX_LISTING_LINES:
            lines = lines[:MAX_LISTING_LINES] + [f"[{len(lines) - MAX_LISTING_LINES} more entries]"]
        return "\n".join(lines)

    def system_prompt(self) -> str:
        """The unattended-engineering prompt (replaces Sorcar's default prompt)."""
        tests = TEST_CONTEXT_NOTE if self.test_context else ""
        return SYSTEM_PROMPT.format(workdir=self.workdir, test_context=tests,
                                    verifier=VERIFIER_PATH, ledger=LEDGER_PATH,
                                    shell_timeout=DEFAULT_SHELL_TIMEOUT)

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """The trial's toolset: the built-ins without ``decide`` and ``summary``, plus checkpoints.

        ``decide`` is a text classifier; used as an oracle for the task's
        domain it picks a wrong answer with confidence, so an unattended
        trial does without it.  ``summary`` has no reader in an unattended
        trial, and its ten-step cadence costs two model calls per ten (the
        owed summary and the refused call before it), so it is dropped too
        (the generated SEA clears ``has_summary_tool`` with it).

        The trial's own ``checkpoint``/``rollback`` (container archives)
        replace any host built-ins of those names.

        Args:
            tools: The built-in tools of the run.
        """
        dropped = ("decide", "summary", "checkpoint", "rollback")
        kept = [t for t in tools if getattr(t, "__name__", "") not in dropped]
        return kept + [self.checkpoint, self.rollback]

    def settings(self) -> dict[str, Any]:
        """The trial's run settings (the trial SEA's ``settings``).

        A worker (its sub-agents share the trial container): the
        trial's model, hard USD cap and per-trial model
        overrides (``None`` for the provider defaults), the host scratch
        directory the daemon runs the task in (the tools run in the
        container the ``docker_image`` attaches), no host git worktree or
        auto-commit (all edits happen inside the container), no
        pre-run classification, no browser, no persistent memory.
        """
        return {
            "model": self.model_name,
            "max_budget": self.budget,
            "model_config": self.model_overrides or None,
            "work_dir": self.host_work_dir,
            "docker_image": (self.container if self.bridged else f"container:{self.container}"),
        }

    @property
    def bridged(self) -> bool:
        """Whether the trial runs on a remote environment through an exec bridge.

        The config's ``container`` is then ``exec:<url>`` (see
        :mod:`kiss.agents.sorcar.exec_bridge`) instead of a Docker container id.
        """
        return self.container.startswith(BRIDGE_PREFIX)

    # ---- hooks ------------------------------------------------------------

    def on_llm_call(self, new_messages: list) -> list:
        """Count one turn per LLM call and log the messages added since the last call.

        Args:
            new_messages: Messages appended to the conversation since the
                previous LLM call.

        Returns:
            The messages unchanged.
        """
        with self._lock:
            self.turns += 1
            turn = self.turns
            if self.started is None:
                self.started = time.time()
        if self.file_baseline is None:  # first model call: the workdir as the task provided it
            try:
                self.shell_path = (self.shell("command -v bash") or "").strip() or "bash"
                digest = self.host_digest(self.shell_path, save=self.host_shell_path)
                self.host_shell = self.host_shell_path if digest else None
                self._log({"event": "shell_baseline", "turn": turn, "path": self.shell_path,
                           "sha256": digest})
                self.file_baseline = self.file_snapshot() or {}
            except Exception as error:
                self._log({"event": "hook_error", "turn": turn, "hook": "llm_call",
                           "error": repr(error)})
                self.file_baseline = {}
        if new_messages:
            try:
                self.record_exit_codes(new_messages)
                notes = self.test_context_notes(self.landed_edits()) + self.shell_notes()
            except Exception as error:  # a note is optional; the trial goes on without it
                self._log({"event": "hook_error", "turn": turn, "hook": "llm_call",
                           "error": repr(error)})
                notes = []
            with self._lock:
                notes, self.pending_notes = self.pending_notes + notes, []
            for note in notes:
                if not append_to_last_tool_result(new_messages[-1], note):
                    with self._lock:
                        self.pending_notes.append(note)
            if turn > 1 and is_continuation_start(new_messages[0]):
                try:
                    append_user_text(new_messages[0], self.inventory())
                    self._log({"event": "inventory", "turn": turn})
                except Exception as error:
                    self._log({"event": "hook_error", "turn": turn, "hook": "inventory",
                               "error": repr(error)})
        self._log({"event": "llm_call", "turn": turn, "new_messages": _jsonable(new_messages)})
        if not self.container_alive():
            # The caller tore the container down (OOM-killed, Docker
            # restarted or the job was stopped): nothing the agent does can count any more, so
            # end the trial instead of letting the session loop retry against
            # a dead container until the budget is gone.  BudgetExceededError is
            # the one exception the Sorcar session loop treats as terminal.
            from kiss.core.kiss_error import BudgetExceededError

            self._log({"event": "container_gone", "turn": turn})
            raise BudgetExceededError(f"Container {self.container} no longer exists; run ended.")
        return new_messages

    def container_alive(self) -> bool:
        """Whether the trial container still exists (checked before every model call)."""
        if self.bridged:
            return bridge_alive(self.container[len(BRIDGE_PREFIX):])
        import docker

        try:
            return bool(_docker_client().containers.get(self.container).status == "running")
        except docker.errors.NotFound:  # type: ignore[attr-defined]
            return False
        except Exception:  # transient daemon hiccup: do not end the trial on it
            return True

    def on_tool_call(self, name: str, args: dict[str, Any]) -> Verdict:
        """Log a tool call; answer tools that need a human without running them.

        Args:
            name: Tool name.
            args: Tool arguments.

        Returns:
            ``ALLOW`` to let the call run, or ``refuse(text)`` with the text the model sees.
        """
        verdict = UNATTENDED_TOOL_VERDICTS.get(name)
        try:
            if name in ("Bash", "run_commands_parallel"):
                verdict = self.shell_verdict(args)
            elif name in ("Edit", "Write") and self.finish_gate:
                verdict = self.ledger_verdict(args)
            elif name == "finish" and self.finish_gate:
                verdict = self.finish_verdict(args)
        except Exception as error:  # a guard is optional; never end the trial on its own bug
            self._log({"event": "hook_error", "turn": self.turns, "hook": name,
                       "error": repr(error)})
        self._log({"event": "tool_call", "turn": self.turns, "tool": name,
                   "args": _jsonable(args), "blocked": verdict is not None})
        if verdict is None and name != "finish":
            self.evidence.observe(name, args)
        if verdict is None and name == "Read" and isinstance(args.get("file_path"), str):
            with self._lock:
                self.files_read[posixpath.normpath(posixpath.join(self.workdir,
                                                                  args["file_path"]))] = None
        command = args.get("command") if name == "Bash" and verdict is None else ""
        with self._lock:  # a refused call still gets a result (the refusal), so it is listed
            self.turn_commands.append((name if verdict is None else "refused",
                                       command if isinstance(command, str) else ""))
        if name in ("Edit", "Write") and verdict is None and isinstance(args.get("file_path"), str):
            self.last_edit_turn = self.turns
            path = posixpath.normpath(posixpath.join(self.workdir, args["file_path"]))
            resolved = (self.container_exec(["readlink", "-f", path]) or "").strip()
            with self._lock:
                self.owned_files.update({path, resolved} - {""})
            if self.test_context:
                try:
                    self.snapshot_edit(args["file_path"])
                except Exception as error:
                    self._log({"event": "hook_error", "turn": self.turns, "hook": name,
                               "error": repr(error)})
        if name in ("Bash", "run_commands_parallel") and verdict is None:
            try:
                self.before_shell(args)
            except Exception as error:
                self._log({"event": "hook_error", "turn": self.turns, "hook": name,
                           "error": repr(error)})
        return ALLOW if verdict is None else refuse(verdict)

    def before_shell(self, args: dict[str, Any]) -> None:
        """Remember the process table before the turn's first shell call; note kill attempts.

        Args:
            args: Arguments of the ``Bash`` or ``run_commands_parallel`` call.
        """
        text = json.dumps(args.get("command", args.get("commands", "")))
        with self._lock:
            take = self.processes_before is None
            self.kill_attempted = self.kill_attempted or bool(KILL_COMMANDS.search(text))
        if take:
            snapshot = self.process_snapshot()
            with self._lock:
                if self.processes_before is None:
                    self.processes_before = snapshot
                    # Adopt children born since the last snapshot now, while their
                    # tracked parent is still there to link them.
                    self.tracked_pids |= descendants(snapshot, self.tracked_pids)

    def shell_notes(self) -> list[str]:
        """Notes about what the turn's shell calls left behind: live processes and changed inputs.

        Empty when no shell tool ran since the last model call.  The survivors
        note names processes the calls started that are still alive; it is
        repeated after a command that tried to stop something while a tracked
        process survives.  The input-files note names pre-existing files under
        the workdir that the calls modified or deleted (files the agent edited
        with Edit/Write are its own and are skipped).
        """
        with self._lock:
            before, self.processes_before = self.processes_before, None
            kill_attempted, self.kill_attempted = self.kill_attempted, False
        if before is None:
            return []
        notes: list[str] = []
        after = self.process_snapshot()
        with self._lock:
            # Children of a process already reported (a build's compiler
            # steps, a server's workers) are tracked without a new note.
            offspring = descendants(after, self.tracked_pids)
            new = {key: info for key, info in after.items()
                   if key not in before and key not in offspring and key not in self.tracked_pids}
            self.tracked_pids |= set(new) | offspring
            self.tracked_pids &= set(after)
            alive = {key: after[key] for key in sorted(self.tracked_pids)}
        if new:
            notes.append(SURVIVORS_NOTE.format(procs=describe_processes(new)))
        elif kill_attempted and alive:
            notes.append(STILL_RUNNING_NOTE.format(procs=describe_processes(alive)))
        if self.file_baseline:
            changed = self.changed_input_files()
            if changed:
                notes.append(
                    INPUT_FILES_NOTE.format(workdir=self.workdir, files=", ".join(changed)))
        for note in notes:
            self._log({"event": "shell_note", "turn": self.turns, "note": note.strip()})
        return notes

    def process_snapshot(self) -> dict[tuple[int, str], tuple[int, str]]:
        """``(pid, start time) -> (ppid, cmdline)`` of every process in the container.

        The snapshot's own shell is left out.
        """
        out = self.container_exec(["sh", "-c", PROCESS_LIST_COMMAND]) or ""
        procs: dict[tuple[int, str], tuple[int, str]] = {}
        for line in out.splitlines():
            parts = line.split(" ", 3)
            if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                procs[(int(parts[0]), parts[2])] = (int(parts[1]), parts[3].strip())
        helper = {pid for (pid, _), (_, cmd) in procs.items() if "KISS_PS=1" in cmd}
        grew = True
        while grew:
            children = {pid for (pid, _), (ppid, _) in procs.items()
                        if ppid in helper and pid not in helper}
            helper |= children
            grew = bool(children)
        return {key: info for key, info in procs.items() if key[0] not in helper}

    def file_snapshot(self) -> dict[str, tuple[str, str]] | None:
        """``path -> (size, mtime)`` of the regular files under the workdir.

        ``None`` (tracking unavailable) for the root workdir, an unreadable
        container or more than :data:`MAX_TRACKED_FILES` files; an empty dict
        when the workdir simply holds no files.
        """
        if self.workdir.rstrip("/") == "":
            return None
        out = self.container_exec(
            ["sh", "-c", FILE_LIST_COMMAND.format(workdir=shlex.quote(self.workdir))])
        if out is None:
            return None
        files: dict[str, tuple[str, str]] = {}
        for line in out.splitlines():
            parts = line.split(" ", 2)
            if len(parts) == 3:
                files[parts[2]] = (parts[0], parts[1])
        return files if len(files) <= MAX_TRACKED_FILES else None

    def changed_input_files(self) -> list[str]:
        """Pre-existing files modified or deleted since the run started, each reported once."""
        current = self.file_snapshot()
        if current is None:
            return []
        changed: list[str] = []
        with self._lock:
            for path, (size, mtime) in (self.file_baseline or {}).items():
                if path in self.owned_files:
                    continue
                now = current.get(path)
                if now is None:
                    changed.append(f"{path} (deleted)")
                elif now != (size, mtime):
                    changed.append(f"{path} (size {size} -> {now[0]})" if now[0] != size
                                   else f"{path} (modified)")
                else:
                    continue
                self.owned_files.add(path)
        return changed[:20] + ([f"and {len(changed) - 20} more"] if len(changed) > 20 else [])

    def container_run(
        self, argv: list[str], timeout: float = 60, with_stderr: bool = False,
    ) -> tuple[int | None, bytes] | None:
        """Run *argv* in the trial environment: ``(exit code, output)``, ``None`` if it cannot run.

        The exit code is ``None`` when the command was still running at
        *timeout*; the command's process tree is then killed in the
        environment too (a Docker exec is tagged with a token in its
        environment and swept with
        :func:`kiss.agents.sorcar.docker_manager.kill_tagged_tree_script`;
        the exec bridge does the same on its side), so a verifier that hangs
        does not keep running, and modifying the work, after the verdict.

        A Docker trial returns stdout, plus stderr when *with_stderr* is set
        (verifier runs, whose tracebacks are on stderr); a bridged trial
        always returns both.
        """
        if self.bridged:
            command = " ".join(shlex.quote(a) for a in argv)
            try:
                code, output = bridge_exec(self.container[len(BRIDGE_PREFIX):], command, timeout)
            except Exception:
                return None
            return code, output.encode("utf-8", errors="replace")
        token = uuid.uuid4().hex
        try:
            completed = subprocess.run(
                ["docker", "exec", "-e", f"{_EXEC_TOKEN_VAR}={token}", self.container, *argv],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT if with_stderr else subprocess.DEVNULL,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as expired:
            kill = ["docker", "exec", self.container, "sh", "-c", kill_tagged_tree_script(token)]
            try:
                subprocess.run(kill, capture_output=True, timeout=60)
            except (subprocess.TimeoutExpired, OSError):
                pass
            return None, bytes(expired.stdout or b"")
        except (OSError, ValueError):
            return None
        return completed.returncode, completed.stdout

    def container_exec(self, argv: list[str], timeout: float = 60) -> str | None:
        """stdout of *argv* run in the trial container; ``None`` on failure or timeout."""
        result = self.container_run(argv, timeout)
        if result is None or result[0] != 0:
            return None
        return result[1].decode("utf-8", errors="replace")

    def shell(self, command: str) -> str | None:
        """stdout of a shell *command* in the trial container; ``None`` on failure."""
        return self.container_exec(["sh", "-c", command])

    def host_digest(self, path: str, timeout: float = 600, save: Path | None = None) -> str | None:
        """SHA-256 of the file at *path* in the container, read through the Docker daemon.

        ``docker cp`` streams the file out (symlinks followed) without
        running anything in the container, so the digest cannot be forged by
        replacing the container's tools.  With *save*, the file's content is
        also written there on the host (executable; removed again when the
        copy fails), which is how the trial's shell is kept for the gate
        (:meth:`run_script`).

        Args:
            path: The file's path in the container.
            timeout: Seconds to wait for the copy.
            save: Host file to write the content to.

        Returns:
            The hex digest; ``None`` on a bridged trial, for a missing path
            or when the copy fails.
        """
        if self.bridged:
            return None
        try:
            copy = subprocess.Popen(["docker", "cp", "-L", f"{self.container}:{path}", "-"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError:
            return None
        digest, found = hashlib.sha256(), False
        deadline = threading.Timer(timeout, copy.kill)  # a stalled copy must not hold the gate
        deadline.start()
        if save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(fileobj=copy.stdout, mode="r|") as stream, \
                    open(save or os.devnull, "wb") as target:
                for member in stream:
                    source = stream.extractfile(member) if member.isfile() and not found else None
                    if source is None:
                        continue
                    found = True
                    for chunk in iter(partial(source.read, 1 << 20), b""):
                        digest.update(chunk)
                        target.write(chunk)
            copy.wait()  # bounded by the timer
        except (tarfile.TarError, OSError):
            found = False
        finally:
            deadline.cancel()
            if copy.poll() is None:
                copy.kill()
                copy.wait()
        complete = found and copy.returncode == 0
        if save is not None:
            if complete:
                save.chmod(0o755)
            else:
                save.unlink(missing_ok=True)
        return digest.hexdigest() if complete else None

    def digest_of(self, path: str) -> str | None:
        """SHA-256 of a file in the trial environment: host-side under Docker, else ``sha256sum``.

        Args:
            path: The file's path in the environment.
        """
        if not self.bridged:
            return self.host_digest(path)
        output = (self.shell(f"sha256sum {shlex.quote(path)} | cut -d' ' -f1") or "").strip()
        return output if SHA256_HEX.fullmatch(output) else None

    def finish_verdict(self, args: dict[str, Any]) -> str | None:
        """Gate a successful ``finish`` on evidence: the verifiers, the summary, a review.

        Every ``finish(success=True)`` must pass, in order: the agent's own
        verifier (:meth:`verifier_verdict`), the independent check written
        from the statement alone (:meth:`blind_verdict`; both run with the
        trial's original shell, :meth:`run_script`), the quotes of the
        summary against the run's tool results (:meth:`quoted_evidence_verdict`),
        the evidence gate (a summary that hedges or leaves a reading open, an
        edit no command checked; once per trial), the answers to the open
        findings of the last review (:meth:`unanswered_verdict`) and a
        read-only review of the work (:meth:`review_verdict`, up to
        ``max_reviews`` per trial, the later ones on the files changed since
        the previous one).  There is no clock: a finish refused for being
        early buys a second round that breaks as much as it fixes.

        The work tree is checkpointed (:data:`VERIFIED_CHECKPOINT`) as soon
        as both verifiers pass, whatever the later checks say, so a run that
        then ends without an accepted finish can be restored to it
        (:func:`submit_best_checkpoint`).

        ``success`` may arrive as a bool or as the string ``"true"`` (some
        providers do not type tool arguments).  The implicit text-only finish
        consults the hook with no arguments and would end the run as a
        success, so it is vetoed once: the agent is asked to call ``finish``
        explicitly, and that call sees the verdicts.  A second implicit
        finish is logged (``implicit_finish``) and lets the run end; the
        trial runner records it as a failure, since a run that ends on text
        (a content-filter refusal, a question nobody answers) delivered
        nothing checked.  A finish let through past the verifier cap
        (:data:`MAX_VERIFIER_REFUSALS`, :attr:`checks_passed` false) is
        logged the same way (``unverified_finish``, :func:`ended_unchecked`):
        the run ends, but as a failure, and the runner submits the last
        verified tree instead (:func:`submit_best_checkpoint`), so repeating
        ``finish`` until the cap is spent delivers nothing a check refused.

        Args:
            args: Arguments of the ``finish`` call (empty for an implicit finish).

        Returns:
            ``None`` (the call may run) or the verdict the model sees.
        """
        if not args:
            if self.implicit_finish_vetoed:
                self._log({"event": "implicit_finish", "turn": self.turns})
                return None
            self.implicit_finish_vetoed = True
            return IMPLICIT_FINISH_VERDICT
        if str(args.get("success")).lower() != "true":
            return None
        self.checks_passed = True
        verdict = self.verifier_verdict() or self.blind_verdict()
        if verdict is not None:
            return verdict
        if self.checks_passed:
            self.save_checkpoint(VERIFIED_CHECKPOINT, verified=True)
        verdict = (self.quoted_evidence_verdict(args) or self.evidence.verdict(args)
                   or self.open_reading_verdict(args) or self.unanswered_verdict(args))
        if verdict is None and self.reviewer and self.review_possible():
            verdict = self.review_verdict(args)
        if verdict is None and not self.checks_passed:
            self._log({"event": "unverified_finish", "turn": self.turns,
                       "refusals": self.verifier_refusals})
        return verdict

    def quoted_evidence_verdict(self, args: dict[str, Any]) -> str | None:
        """Refuse a finish whose summary quotes text that no tool result of the run contains.

        The quotes of ``summary_in_html`` (:func:`quoted_pieces`) are looked
        up in the run's sources (:meth:`evidence_sources`): the task
        statement, every tool result and every shell command, as the
        trajectory recorded them.  The provenance of every quote is kept for
        the reviewer (:attr:`quote_provenance`) and logged as a
        ``quoted_evidence`` event (``outputs``: how many tool results the
        trajectory yielded; none at all means a message shape the harness
        does not read, and then every quote is refused until it is dropped).
        There is no cap on these refusals: a quote the agent cannot support
        is dropped or re-run, so the refusal never traps a run.

        Args:
            args: Arguments of the ``finish`` call.
        """
        sources = self.evidence_sources()
        self.quote_provenance = quote_sources(str(args.get("summary_in_html", "")), sources)
        unsupported = unsupported_quotes(self.quote_provenance)
        self._log({"event": "quoted_evidence", "turn": self.turns,
                   "outputs": sum(source.kind == "output" for source in sources),
                   "quotes": len(self.quote_provenance), "unsupported": unsupported[:50]})
        if not unsupported:
            return None
        shown = "\n".join(f'- "{piece[:MAX_QUOTE_SHOWN_CHARS]}"'
                          for piece in unsupported[:MAX_QUOTES_SHOWN])
        return QUOTED_EVIDENCE_VERDICT.format(quotes=shown)

    def evidence_sources(self) -> list[Source]:
        """The texts a summary may quote, from the trajectory: statement, tool results, commands.

        The ``tool_call`` records between two ``llm_call`` records are the
        calls whose results the second record carries, in order (the pairing
        :meth:`record_exit_codes` uses); a paired result is labelled with its
        call (:func:`result_origin`), and a shell command that names a file
        the agent wrote with ``Edit``/``Write`` says so, since the output of
        a script the agent wrote is the agent's own claim.  A turn whose
        results do not pair up with its calls contributes its results
        unlabelled.  The result of a refused call is the harness's own text,
        not evidence: it is left out, as is any result carrying the quote
        refusal (:data:`QUOTED_EVIDENCE_MARKER`), which repeats the very
        quotes it refused.
        """
        sources = [Source("statement", self.task_prompt, "the task statement")]
        calls: list[dict[str, Any]] = []
        written: list[str] = []
        for record in trajectory_events(self.trajectory_path):
            event, args = record.get("event"), record.get("args")
            if event == "tool_call":
                calls.append(record)
                if record.get("blocked"):
                    continue
                if record.get("tool") in SHELL_TOOLS:
                    for command in shell_commands(args):
                        sources.append(Source("command", command,
                                              f"a command you ran: {command[:100]}"))
                elif (record.get("tool") in EDIT_TOOLS and isinstance(args, dict)
                      and isinstance(args.get("file_path"), str)):
                    written.append(args["file_path"])
            elif event == "llm_call":
                results = [text for message in record.get("new_messages") or []
                           for text in tool_result_texts(message)]
                paired = calls if len(calls) == len(results) else [{}] * len(results)
                for call, text in zip(paired, results, strict=True):
                    if call.get("blocked") or QUOTED_EVIDENCE_MARKER in text:
                        continue
                    sources.append(Source("output", text, result_origin(call, written)))
                calls = []
        return sources

    def review_possible(self) -> bool:
        """Whether another review may run (neither the review nor the failed-review cap is hit)."""
        return self.reviews < self.max_reviews and self.failed_reviews < MAX_FAILED_REVIEWS

    def open_reading_verdict(self, args: dict[str, Any]) -> str | None:
        """Refuse once per trial a summary that admits a case of the statement is unhandled.

        The decisions model reads the task statement and the finish summary
        (:func:`leaves_a_case_open`).  A summary that admits a listed
        symptom was left alone ("I added nothing for ...") or reports the
        asked artifact as impossible names the requirement that then
        fails, and a reviewer tends to let it through once a reason is
        given; the gate does not.

        Args:
            args: Arguments of the ``finish`` call.
        """
        summary = str(args.get("summary_in_html", "")).strip()
        if self.open_reading_refused or not summary:
            return None
        score = leaves_a_case_open(self.task_prompt, summary)
        self._log({"event": "open_reading", "turn": self.turns, "score": score})
        if score < OPEN_READING_THRESHOLD:
            return None
        self.open_reading_refused = True
        return OPEN_READING_VERDICT.format(verifier=VERIFIER_PATH)

    def verifier_verdict(self) -> str | None:
        """Run the agent's own verifier; the verdict when it is missing, failing or too slow.

        At most :data:`MAX_VERIFIER_REFUSALS` refusals per trial, so a verifier
        the agent cannot make pass does not trap the run; a finish let through
        that way is not a verified one (:attr:`checks_passed`).  The script
        is read once and that text is run (:meth:`run_script`), so what the
        reviewer sees is what ran (:attr:`last_verifier_run`).
        """
        if self.verifier_refusals >= MAX_VERIFIER_REFUSALS:
            self.checks_passed = False
            return None
        script = self.read_container_file(VERIFIER_PATH)
        if script is None:
            self.verifier_refusals += 1
            self._log({"event": "verifier", "turn": self.turns, "result": "missing"})
            return VERIFIER_MISSING_VERDICT.format(path=VERIFIER_PATH)
        result = self.run_script(script, VERIFIER_PATH, self.verifier_timeout)
        code, output = (None, b"") if result is None else result
        text = output.decode("utf-8", errors="replace")
        self.last_verifier_run = (script, code, text)
        self._log({"event": "verifier", "turn": self.turns, "exit_code": code,
                   "output": text[-4000:]})
        if code == 0:
            # The verifier just checked the final state of the work.
            self.evidence.unchecked_change = False
            return None
        self.verifier_refusals += 1
        if code is None:
            return VERIFIER_TIMEOUT_VERDICT.format(path=VERIFIER_PATH,
                                                   seconds=self.verifier_timeout)
        tail = text[-3000:]
        return VERIFIER_FAILED_VERDICT.format(path=VERIFIER_PATH, code=code, output=tail)

    def review_verdict(self, args: dict[str, Any]) -> str | None:
        """A read-only review of the work by the ``reviewer`` model; its gaps as a verdict.

        The reviewer sees the task statement, the agent's summary and
        ledger, both verifiers and their last output, and the files created
        or changed under the workdir (:meth:`changed_files_report`); a
        re-review also sees its previous findings and only the files changed
        since.  It runs no commands.  Its spend is logged as a ``review``
        event, which the trial runner adds to the trial's cost.

        An answer without a ``VERDICT:`` line (adaptive thinking at the
        highest effort can use up the output cap and leave the answer
        blank) is retried once at the provider's default
        effort; a review that still has no verdict, or whose model call
        fails, is logged as failed and refuses the finish
        (:data:`REVIEW_FAILED_VERDICT`), keeping the previous findings open;
        it does not use up a review but counts under
        :data:`MAX_FAILED_REVIEWS`, past which the finish goes through (a
        reviewer that never answers cannot hold the run: the work tree the
        run ends on is what gets submitted either way, and the open
        findings are then judged against the summary by
        :meth:`unanswered_verdict`).  Each ``NOT DONE`` finding is kept
        (:attr:`open_findings`) for the next finish to answer.

        Args:
            args: Arguments of the ``finish`` call.
        """
        from kiss.core.kiss_agent import KISSAgent

        script, code, output = self.last_verifier_run
        blind_script, blind_code, blind_output = self.last_blind_run
        previous, scope = "", ""
        if self.open_findings:
            previous = REVIEW_PREVIOUS.format(findings="\n".join(self.open_findings))
            scope = " since your previous review"
        prompt = REVIEW_PROMPT.format(
            task=self.task_prompt, summary=str(args.get("summary_in_html", "")),
            ledger=LEDGER_PATH, ledger_text=(self.read_container_file(LEDGER_PATH) or "(none)")[
                :REVIEW_ONE_FILE_BYTES],
            verifier=VERIFIER_PATH, code=code, script=script[:REVIEW_ONE_FILE_BYTES],
            output=output[-4000:], blind=BLIND_VERIFIER_PATH, blind_code=blind_code,
            blind_script=blind_script[:REVIEW_ONE_FILE_BYTES], blind_output=blind_output[-4000:],
            quotes=quotes_report(self.quote_provenance),
            previous=previous, scope=scope, workdir=self.workdir,
            files=self.changed_files_report(self.review_snapshot if self.open_findings else None),
        )
        answer, cost, tokens, error, last_call = "", 0.0, 0, "", {}
        for overrides in (self.model_overrides or None, None):
            agent = KISSAgent("Finish reviewer")
            try:
                answer = agent.run(model_name=self.reviewer, prompt_template="{p}",
                                   arguments={"p": prompt}, is_agentic=False, verbose=False,
                                   model_config=overrides)
            except Exception as failure:
                error = repr(failure)
            cost += agent.budget_used
            tokens += agent.total_tokens_used
            last_call = dict(agent.last_call_usage or {})
            if error or REVIEW_VERDICT_LINE.search(answer):
                break
            self._log({"event": "review_retry", "turn": self.turns, "model": self.reviewer,
                       "answer": answer[:2000], "last_call": last_call})
        findings = review_findings(answer)
        failed = bool(error or not REVIEW_VERDICT_LINE.search(answer))
        if failed:
            self.failed_reviews += 1
        else:
            self.reviews += 1
        self._log({"event": "review", "turn": self.turns, "model": self.reviewer,
                   "round": self.reviews, "answer": answer[:8000], "findings": findings,
                   "error": error, "failed": failed, "last_call": last_call,
                   "cost_usd": cost, "tokens": tokens})
        if failed:
            return REVIEW_FAILED_VERDICT if self.review_possible() else None
        self.review_snapshot = self.file_snapshot()
        self.open_findings = numbered_items(findings)
        if not findings:
            return None
        return REVIEW_VERDICT.format(findings=findings)

    def unanswered_verdict(self, args: dict[str, Any]) -> str | None:
        """The open review findings this finish answers neither by a change nor by evidence.

        A finding is answered when a file under the workdir changed since
        the review and a re-review will judge the change (the review cap
        leaves room for one), or when the decisions model finds the finish
        summary quoting a command and its output for it
        (:func:`answered_findings`).  Without the decisions model every
        finding counts as answered.

        Args:
            args: Arguments of the ``finish`` call.
        """
        if not self.open_findings:
            return None
        if (self.review_possible() and self.review_snapshot is not None
                and self.changed_since(self.review_snapshot)):
            return None
        summary = str(args.get("summary_in_html", ""))
        answered = answered_findings(summary, self.open_findings)
        self._log({"event": "findings_answered", "turn": self.turns, "answered": answered})
        unanswered = [f for f, ok in zip(self.open_findings, answered, strict=True) if not ok]
        if not unanswered:
            return None
        return UNANSWERED_VERDICT.format(findings="\n".join(unanswered))

    def blind_verdict(self) -> str | None:
        """Run the independent check; the verdict when it fails.

        The check is written once per trial by a second agent
        (:meth:`write_blind_verifier`) that sees the task statement and the
        container but not the engineer's work, and is then run at every
        gated finish from the host's copy (:attr:`blind_script`, see
        :meth:`run_script`): the container's copy at
        :data:`BLIND_VERIFIER_PATH` is the agent's to read and run, and is
        rewritten before each run so that an edit or deletion there changes
        nothing.  Refusals share :data:`MAX_VERIFIER_REFUSALS` with the
        agent's own verifier, so a check that misreads the statement cannot
        trap the run; the disagreement is the agent's to resolve with
        evidence.  No check (no ``blind_verifier`` model, or an author that
        produced nothing) means no verdict.
        """
        if not self.blind_verifier:
            return None
        if self.verifier_refusals >= MAX_VERIFIER_REFUSALS:
            self.checks_passed = False
            return None
        if not self.blind_written:
            self.blind_written = True
            self.write_blind_verifier()
        script = self.blind_script
        if not script:
            return None
        self.write_container_file(BLIND_VERIFIER_PATH, script)
        result = self.run_script(script, BLIND_VERIFIER_PATH, self.verifier_timeout)
        code, output = (None, b"") if result is None else result
        text = output.decode("utf-8", errors="replace")
        self.last_blind_run = (script, code, text)
        self._log({"event": "blind_verifier", "turn": self.turns, "exit_code": code,
                   "output": text[-4000:]})
        if code == 0:
            return None
        self.verifier_refusals += 1
        shown = text[-BLIND_OUTPUT_CHARS:] if code is not None else (
            f"(timed out after {self.verifier_timeout:.0f} seconds)")
        return BLIND_VERIFIER_FAILED_VERDICT.format(path=BLIND_VERIFIER_PATH, code=code,
                                                    output=shown)

    def write_blind_verifier(self) -> None:
        """Have the ``blind_verifier`` model write the independent check into the container.

        The author's agent gets a shell in the container and a
        ``finish(script)`` tool, :data:`BLIND_VERIFIER_STEPS` steps and
        :data:`BLIND_VERIFIER_BUDGET_FRACTION` of the trial's budget; its
        spend is logged as a ``blind_verifier_written`` event (added to the
        trial's cost by the runner like a review's).  It sees the statement,
        not the engineer's ledger, verifier or summary.
        """
        from kiss.core.kiss_agent import KISSAgent

        harness = self
        written: list[str] = []

        def Bash(command: str, timeout_seconds: int = 60) -> str:  # noqa: N802
            """Run one shell command in the task's container and return its output.

            Args:
                command: The shell command.
                timeout_seconds: Seconds before the command is killed.
            """
            result = harness.run_script(command, "bash", timeout_seconds)
            if result is None:
                return "Error: the command could not run"
            code, out = result
            text = out.decode("utf-8", errors="replace")[-20000:]
            return text if code == 0 else f"{text}\n[exit code: {code}]"

        def finish(script: str) -> str:
            """Deliver the complete check script and end the run.

            Args:
                script: The bash script; it exits non-zero on the first failing check.
            """
            written.append(script)
            return "done"

        agent = KISSAgent("Blind verifier")
        error = ""
        try:
            agent.run(model_name=self.blind_verifier, prompt_template="{p}",
                      arguments={"p": BLIND_VERIFIER_PROMPT.format(
                          task=self.task_prompt, workdir=self.workdir,
                          timeout=int(self.verifier_timeout * BLIND_VERIFIER_TIME_FRACTION))},
                      tools=[Bash, finish], max_steps=BLIND_VERIFIER_STEPS,
                      max_budget=self.budget * BLIND_VERIFIER_BUDGET_FRACTION, verbose=False,
                      model_config=self.model_overrides or None)
        except Exception as failure:
            error = repr(failure)
        self.blind_script = written[-1] if written else ""
        script = self.blind_script
        self._log({"event": "blind_verifier_written", "turn": self.turns,
                   "model": self.blind_verifier, "script": script[:8000], "error": error,
                   "steps": agent.step_count, "last_call": dict(agent.last_call_usage or {}),
                   "cost_usd": agent.budget_used, "tokens": agent.total_tokens_used})

    def ledger_verdict(self, args: dict[str, Any]) -> str | None:
        """Refuse an edit under the workdir until the interpretation ledger exists.

        At most :data:`MAX_LEDGER_REFUSALS` refusals per trial.  Edits
        outside the workdir (the ledger and the verifier themselves, under
        ``/tmp``) are never refused.

        Args:
            args: Arguments of the ``Edit`` or ``Write`` call.
        """
        if self.ledger_seen or self.ledger_refusals >= MAX_LEDGER_REFUSALS:
            return None
        path = args.get("file_path")
        if not isinstance(path, str):
            return None
        full = posixpath.normpath(posixpath.join(self.workdir, path))
        root = self.workdir.rstrip("/")
        if not (full == root or full.startswith(root + "/")):
            return None
        if self.container_exec(["test", "-s", LEDGER_PATH]) is not None:
            self.ledger_seen = True
            return None
        self.ledger_refusals += 1
        return LEDGER_VERDICT.format(workdir=self.workdir, ledger=LEDGER_PATH)

    # ---- checkpoints -------------------------------------------------------

    def checkpoint(self, name: str) -> str:
        """Save a snapshot of the working directory under *name* (a later one replaces it).

        Args:
            name: A short label (letters, digits, ``-`` and ``_``).

        Returns:
            A confirmation, or a string starting with ``Error:``.
        """
        if not CHECKPOINT_NAME.fullmatch(name or ""):
            return "Error: name must be letters, digits, - or _ (1 to 40 characters)."
        if name == VERIFIED_CHECKPOINT:
            return (f"Error: {VERIFIED_CHECKPOINT!r} is the harness's own checkpoint; "
                    "pick another name.")
        return self.save_checkpoint(name)

    def rollback(self, name: str) -> str:
        """Restore the working directory from the snapshot *name* (see ``checkpoint``).

        Everything under the working directory is replaced by the snapshot;
        files elsewhere (installed packages, ``/tmp``) are left as they are.

        Args:
            name: The snapshot's label (``verified`` is the harness's own).

        Returns:
            A confirmation, or a string starting with ``Error:``.
        """
        if name not in self.checkpoints:
            known = ", ".join(self.checkpoints) or "none"
            return f"Error: unknown checkpoint {name!r}. Known: {known}."
        error = self.restore_checkpoint(name)
        if error:
            return error
        self.evidence.unchecked_change = True
        return f"Restored {self.workdir} from checkpoint {name!r}."

    def save_checkpoint(self, name: str, verified: bool = False) -> str:
        """Archive the workdir as ``{CHECKPOINT_DIR}/<name>.tar``; log a ``checkpoint`` event.

        The archive stays in the container, so its SHA-256 (:meth:`digest_of`,
        read through the Docker daemon where possible) is recorded here and
        in the event (:attr:`checkpoint_digests`): a restore first checks
        that the archive is the one that was saved.  A verified
        checkpoint also keeps host-side copies of the verifier scripts that
        passed (:meth:`saved_check_path`), so :func:`submit_best_checkpoint`
        re-runs the checks the checkpoint passed, not whatever scripts the
        agent left in the container later.

        Args:
            name: The checkpoint's label.
            verified: Whether the finish checks had just passed (the runner
                restores the newest such checkpoint when the run ends badly).
        """
        archive = f"{CHECKPOINT_DIR}/{name}.tar"
        wd = shlex.quote(self.workdir)
        command = (f"mkdir -p {CHECKPOINT_DIR} && mb=$(du -sm {wd} | cut -f1)"
                   f" && [ \"$mb\" -le {MAX_CHECKPOINT_MB} ] || {{ echo \"too large: ${{mb}} MB\"; "
                   f"exit 3; }}; tar -C {wd} -cf {archive}.tmp . && "
                   f"mv {archive}.tmp {archive}")
        result = self.container_run(["sh", "-c", command], timeout=600, with_stderr=True)
        code = None if result is None else result[0]
        detail = "" if result is None else result[1].decode("utf-8", errors="replace")[-500:]
        digest = (self.digest_of(archive) if code == 0 else None) or ""
        if code == 0 and not digest:
            code, detail = 6, "no digest of the archive"
        self._log({"event": "checkpoint", "turn": self.turns, "name": name, "verified": verified,
                   "exit_code": code, "detail": detail, "sha256": digest})
        if code != 0:
            return f"Error: checkpoint {name!r} not saved: {detail.strip() or 'tar failed'}"
        if verified:
            for path, script in ((VERIFIER_PATH, self.last_verifier_run[0]),
                                 (BLIND_VERIFIER_PATH, self.blind_script)):
                try:
                    self.saved_check_path(name, path).write_text(script, encoding="utf-8")
                except OSError as error:  # the host copy is for the runner; the trial goes on
                    self._log({"event": "hook_error", "turn": self.turns, "hook": "checkpoint",
                               "error": repr(error)})
        with self._lock:
            self.checkpoint_digests[name] = digest
            if name not in self.checkpoints:
                self.checkpoints.append(name)
        return f"Saved checkpoint {name!r} of {self.workdir}."

    def saved_check_path(self, checkpoint: str, script: str) -> Path:
        """The host file where a verified checkpoint keeps its copy of the verifier *script*.

        Next to the trajectory, out of the container's reach.

        Args:
            checkpoint: The checkpoint's label.
            script: The script's path in the container (``/tmp/kiss-verify.sh``).
        """
        return self.trajectory_path.parent / f"{checkpoint}.{posixpath.basename(script)}"

    @property
    def host_shell_path(self) -> Path:
        """The host file where the trial keeps its copy of the container's original bash."""
        return self.trajectory_path.parent / "shell"

    def run_script(self, script: str, name: str, timeout: float) -> tuple[int | None, bytes] | None:
        """Run a bash *script* the host holds in the trial environment, with the trial's shell.

        The script travels as the argument of ``<shell> -c``, so no file in
        the container is read; *name* becomes its ``$0``.  The shell is the
        host's copy of the container's original bash (:attr:`host_shell`),
        put into the container through the Docker daemon under a fresh
        private path right before the run and removed after it: nothing the
        agent runs or leaves at the shell's own path, and no command of the
        container, stands between the copy and the exec.  (The copy still
        loads the container's libc; the script's own commands are the
        container's.)  A bridged trial, or one whose copy failed at the
        start, runs the shell at :attr:`shell_path`; so does a run whose copy
        into the container fails, logged as a ``shell_copy_failed`` event.

        Args:
            script: The script's text.
            name: The path the script is known by (``$0``).
            timeout: Seconds before the script is killed.

        Returns:
            As :meth:`container_run` with stderr included.
        """
        shell = self.shell_path
        copied = False
        if self.host_shell is not None:
            private = f"/.kiss-shell-{uuid.uuid4().hex}"
            try:
                copied = subprocess.run(
                    ["docker", "cp", str(self.host_shell), f"{self.container}:{private}"],
                    capture_output=True, timeout=60).returncode == 0
            except (subprocess.TimeoutExpired, OSError):  # a stalled daemon, no docker binary
                copied = False
            if copied:
                shell = private
            else:
                self._log({"event": "shell_copy_failed", "turn": self.turns, "path": private})
        result = self.container_run([shell, "-c", script, name], timeout=timeout, with_stderr=True)
        if copied:
            self.container_run(["rm", "-f", shell])
        return result

    def restore_checkpoint(self, name: str) -> str:
        """Replace the workdir's content with the checkpoint *name*; ``""`` or an error.

        The archive is extracted only when its SHA-256 (:meth:`digest_of`)
        is the one recorded at the save (:attr:`checkpoint_digests`); an
        archive that was replaced or deleted in the container restores
        nothing.
        """
        if self.workdir.rstrip("/") in ("", "/"):
            return "Error: the working directory is the root; nothing is restored."
        archive = f"{CHECKPOINT_DIR}/{name}.tar"
        wd = shlex.quote(self.workdir)
        current = self.digest_of(archive)
        code: int | None
        if current is None:
            code, detail = 4, "archive missing"
        elif current != self.checkpoint_digests.get(name):
            code, detail = 5, "archive altered since it was saved"
        else:
            command = (f"find {wd} -mindepth 1 -maxdepth 1 -exec rm -rf {{}} + && "
                       f"tar -C {wd} -xf {archive}")
            result = self.container_run(["sh", "-c", command], timeout=600, with_stderr=True)
            code = None if result is None else result[0]
            detail = "" if result is None else result[1].decode("utf-8", errors="replace")[-500:]
        self._log({"event": "rollback", "turn": self.turns, "name": name, "exit_code": code,
                   "detail": detail})
        if code != 0:
            return f"Error: checkpoint {name!r} not restored: {detail.strip() or 'tar failed'}"
        return ""

    # ---- restart inventory -------------------------------------------------

    def record_exit_codes(self, new_messages: list) -> None:
        """Pair the turn's ``Bash`` commands with the exit codes their results show.

        The turn's results come in the order of its tool calls, so they are
        paired with every call of the turn and only the ``Bash`` ones kept.  A
        result without an ``[exit code: N]`` marker (the container ``Bash``
        prints it only on failure, before the usage footer) or a timeout
        message is a success.

        Args:
            new_messages: The messages added since the previous model call.
        """
        with self._lock:
            calls, self.turn_commands = self.turn_commands, []
        if not calls:
            return
        results = [text for message in new_messages for text in tool_result_texts(message)]
        if len(results) != len(calls):
            return
        for (name, command), text in zip(calls, results, strict=True):
            if name != "Bash":
                continue
            codes = EXIT_CODE_MARKER.findall(text)
            if codes:
                code = codes[-1]
            elif text.startswith("Error: command timed out"):
                code = "timeout"
            elif text.startswith("Started background job"):
                code = "background"
            else:
                code = "0"
            with self._lock:
                self.commands_run.append((command, code))
                del self.commands_run[:-INVENTORY_COMMANDS]

    def inventory(self) -> str:
        """The machine-written state block appended to a continuation session's first message.

        Files read (with size and content hash, so the new session can tell
        what changed), files the agent wrote, the last commands with their
        exit codes, the checkpoints, and whether the ledger and verifiers
        exist; the statement itself is already in the message.
        """
        with self._lock:
            read = list(self.files_read)
            commands = list(self.commands_run)
            written = sorted(self.owned_files)
        lines = ["", "# Inventory (written by the harness at this restart)"]
        if read:
            listing = self.container_exec(
                ["sh", "-c", "for f in \"$@\"; do if [ -f \"$f\" ]; then printf '%s %s %s\\n' "
                 "\"$(wc -c < \"$f\")\" \"$(md5sum \"$f\" | cut -c1-8)\" \"$f\"; "
                 "else echo \"- - $f\"; "
                 "fi; done", "sh", *read[-INVENTORY_FILES:]]) or ""
            lines.append("Files read (bytes md5 path; '-' = missing now):")
            lines += ["  " + line for line in listing.splitlines()]
        if written:
            lines.append("Files written with Edit/Write: " + ", ".join(written[:INVENTORY_FILES]))
        if commands:
            lines.append(f"Last {len(commands)} shell commands (exit code: command):")
            lines += [f"  {code}: {command[:160]}" for command, code in commands]
        present = self.container_exec(
            ["sh", "-c", f"for f in {LEDGER_PATH} {VERIFIER_PATH} {BLIND_VERIFIER_PATH}; do "
             "[ -s \"$f\" ] && echo \"$f\"; done; true"]) or ""
        lines.append("Present: " + (", ".join(present.split()) or "none of the ledger/verifiers"))
        if self.checkpoints:
            lines.append("Checkpoints: " + ", ".join(self.checkpoints))
        lines.append("Do not re-read unchanged files you already know from the summary above.")
        return "\n".join(lines)

    def changed_since(self, baseline: dict[str, tuple[str, str]]) -> list[str]:
        """The files created, modified or deleted under the workdir since *baseline*.

        Args:
            baseline: An earlier :meth:`file_snapshot`.

        Returns:
            The changed paths, sorted; empty when the current snapshot is unavailable.
        """
        current = self.file_snapshot()
        return [] if current is None else changed_paths(baseline, current)

    def write_container_file(self, path: str, content: str) -> bool:
        """Write *content* to *path* in the trial environment; whether it succeeded."""
        if self.bridged:
            command = f"printf '%s' {shlex.quote(content)} > {shlex.quote(path)}"
            result = self.container_run(["sh", "-c", command], timeout=60)
            return result is not None and result[0] == 0
        try:
            completed = subprocess.run(
                ["docker", "exec", "-i", self.container, "sh", "-c", f"cat > {shlex.quote(path)}"],
                input=content.encode("utf-8"), capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0

    def changed_files_report(self, since: dict[str, tuple[str, str]] | None = None) -> str:
        """The files created, modified or deleted under the workdir since the run started.

        Each listed with its content (text files up to
        :data:`REVIEW_ONE_FILE_BYTES`, :data:`REVIEW_FILE_BYTES` in total);
        ``(no file tracking)`` when the baseline is unavailable.

        Args:
            since: An earlier :meth:`file_snapshot` to diff against instead
                of the run's baseline (a re-review's scope).
        """
        baseline = self.file_baseline if since is None else since
        current = self.file_snapshot()
        if baseline is None or current is None:
            return "(no file tracking)"
        changed = changed_paths(baseline, current)
        if not changed:
            return "(none)"
        parts, budget = [], REVIEW_FILE_BYTES
        for path in changed:
            if path not in current:
                parts.append(f"## {path} (deleted)")
                continue
            content = self.read_container_file(path)
            label = "created" if path not in baseline else "modified"
            if content is None:
                parts.append(f"## {path} ({label}, binary or too large to show)")
                continue
            shown = content[:min(REVIEW_ONE_FILE_BYTES, budget)]
            budget -= len(shown)
            cut = " [cut]" if len(shown) < len(content) else ""
            parts.append(f"## {path} ({label}{cut})\n```\n{shown}\n```")
            if budget <= 0:
                parts.append(f"[{len(changed) - len(parts)} more changed files not shown]")
                break
        return "\n".join(parts)

    def shell_verdict(self, args: dict[str, Any]) -> str | None:
        """Block commands that would end the trial; lift the timeout of installs and builds.

        The timeout is raised in place (``args`` is the dict the tool is
        called with) so a package install or compile is never killed half
        way by ``Bash``'s 30-second default; a timeout already at or above
        :data:`INSTALL_TIMEOUT_SECONDS` (including ``run_commands_parallel``'s
        1800-second default) is left alone.

        Args:
            args: Arguments of a ``Bash`` or ``run_commands_parallel`` call.

        Returns:
            ``None`` (the call may run), or :data:`DESTRUCTIVE_VERDICT` for a blocked command.
        """
        if isinstance(args.get("command"), str):
            commands, default_timeout = [args["command"]], float(DEFAULT_SHELL_TIMEOUT)
            args.setdefault("timeout_seconds", DEFAULT_SHELL_TIMEOUT)
        elif isinstance(args.get("commands"), str):
            try:
                parsed = json.loads(args["commands"])
            except ValueError:
                parsed = [args["commands"]]
            commands = ([c for c in parsed if isinstance(c, str)] if isinstance(parsed, list)
                        else [args["commands"]])
            default_timeout = 1800.0
        else:
            return None
        if any(self.destructive.search(c) for c in commands):
            return DESTRUCTIVE_VERDICT
        foreground = not args.get("background")
        for command in commands:
            for match in SLEEP_POLL.finditer(SHELL_QUOTED.sub("", command)) if foreground else ():
                if float(match.group(1)) >= SLEEP_POLL_SECONDS:
                    return SLEEP_VERDICT.format(seconds=match.group(1))
            refusal = (heredoc_guard(command, self.workdir, self.container_path_exists)
                       or rewrite_guard(command))
            if refusal is not None:
                return refusal
            if self.finish_gate:  # a heredoc creating a file is an edit: same ledger gate
                for target in heredoc_targets(command):
                    verdict = self.ledger_verdict({"file_path": target})
                    if verdict is not None:
                        return verdict
        if any(INSTALL_COMMANDS.search(c) for c in commands):
            try:
                current = float(args.get("timeout_seconds", default_timeout))
            except (TypeError, ValueError):
                current = default_timeout
            if current < INSTALL_TIMEOUT_SECONDS:
                args["timeout_seconds"] = INSTALL_TIMEOUT_SECONDS
        return None

    def container_path_exists(self, path: str) -> bool:
        """Whether *path* exists in the trial container (the heredoc guard's probe).

        Args:
            path: An absolute path in the container.
        """
        return self.container_exec(["test", "-e", path]) is not None

    def landed_edits(self) -> list[tuple[str, str, str]]:
        """``(path, before, after)`` of the edits since the last model call that changed their file.

        Consumed once per model call; the test-context notes and the sibling
        notes share the result (see :meth:`on_llm_call`).
        """
        with self._lock:
            edits, self.pending_edits = self.pending_edits, []
        landed = []
        for path, before in edits:
            after = self.read_container_file(path)
            if after is not None and after != before:
                landed.append((path, before, after))
        return landed

    def snapshot_edit(self, file_path: str) -> None:
        """Remember the content of *file_path* before an edit so the change can be located later.

        New files and test files are ignored: the note is about tests that
        specify existing source code.

        Args:
            file_path: Path as the agent named it (relative paths are taken from the workdir).
        """
        path = posixpath.normpath(posixpath.join(self.workdir, file_path))
        if test_context.is_test_path(path):
            return
        before = self.read_container_file(path)
        if before is not None:
            with self._lock:
                self.pending_edits.append((path, before))

    def test_context_notes(self, landed: list[tuple[str, str, str]]) -> list[str]:
        """Notes about the tests that reference the code edited since the last model call.

        Args:
            landed: The edits since the last model call (:meth:`landed_edits`).
        """
        if not self.test_context:
            return []
        notes: list[str] = []
        for path, before, after in landed:
            names = test_context.changed_definitions(before, after)
            hits = test_context.find_referencing_tests(self.shell, self.workdir, names, path)
            tests = [hit for hit in hits if hit[0] not in self.shown_tests]
            if not tests:
                continue
            self.shown_tests.update(hit[0] for hit in tests)
            edited = (posixpath.relpath(path, self.workdir) if path.startswith(self.workdir)
                      else path)
            notes.append(test_context.test_context_note(edited, names, tests))
            self._log({"event": "test_context", "turn": self.turns, "file": edited,
                       "names": names, "tests": tests})
        return notes

    def read_container_file(self, path: str) -> str | None:
        """Text of the file at *path* inside the trial container.

        ``None`` when the file is missing, unreadable, binary or larger than
        :data:`MAX_SNAPSHOT_BYTES` (the test-context note diffs whole files; a
        generated or minified file is not worth the time).
        """
        result = self.container_run(["head", "-c", str(MAX_SNAPSHOT_BYTES + 1), path], timeout=30)
        if result is None or result[0] != 0:
            return None
        data = result[1]
        if len(data) > MAX_SNAPSHOT_BYTES or b"\0" in data:
            return None
        return data.decode("utf-8", errors="replace")

    def _log(self, record: dict[str, Any]) -> None:
        record["ts"] = time.time()
        line = json.dumps(record, ensure_ascii=False, default=str)
        try:
            with self._lock:
                self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                with self.trajectory_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:  # a lost log line must not end the trial
            pass


#: The reviewer's verdict line.
REVIEW_VERDICT_LINE = re.compile(r"VERDICT:\s*(OK|NOT DONE)\b", re.I)
#: The failure marker of the container ``Bash`` tools.
EXIT_CODE_MARKER = re.compile(r"\[exit code: (-?\d+)\]")
#: Entries of the restart inventory.
INVENTORY_COMMANDS = 40
INVENTORY_FILES = 60
#: First line of a continuation session's prompt (``CONTINUATION_PROMPT``).
CONTINUATION_MARKER = "# Task Progress (Continuation "
#: Side agents of the gate whose spend the trial runner adds to the trial's cost.
SIDE_AGENT_EVENTS = frozenset({"review", "blind_verifier_written"})


def review_findings(answer: str) -> str:
    """The gaps a reviewer's *answer* lists, or ``""`` when it says ``VERDICT: OK``.

    An answer without a ``VERDICT:`` line yields no findings: the caller
    retries it once and then lets the run go (a confused reviewer must not
    hold the run, nor count as approval in the log).

    Args:
        answer: The reviewer model's reply to :data:`REVIEW_PROMPT`.
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


def answered_findings(summary: str, findings: list[str]) -> list[bool]:
    """Whether *summary* answers each of *findings* with evidence, per the decisions model.

    One :func:`~kiss.agents.sorcar.decide_tool.make_decide_tool` call asks
    :data:`ANSWER_QUESTION` per finding; a finding counts as answered at a
    probability of 0.5 or more.  Without the decisions model (or when the
    call fails) every finding counts as answered: the gate must not depend
    on an optional service.

    Args:
        summary: The finish summary.
        findings: The open findings (:func:`numbered_items`).
    """
    from kiss.agents.sorcar.decide_tool import decisions_tool_available, make_decide_tool

    if not findings or not decisions_tool_available():
        return [True] * len(findings)
    questions = {f"f{i}": {"type": "noul", "instructions": ANSWER_QUESTION.format(finding=f)}
                 for i, f in enumerate(findings)}
    try:
        decide = make_decide_tool(None)
        answers = json.loads(decide(summary, json.dumps(questions)))["answers"]
    except Exception:
        return [True] * len(findings)
    answered = []
    for i in range(len(findings)):
        result = answers.get(f"f{i}") if isinstance(answers, dict) else None
        value = result.get("noul") if isinstance(result, dict) else None
        answered.append(not isinstance(value, int | float) or value >= ANSWERED_THRESHOLD)
    return answered


def changed_paths(baseline: dict[str, tuple[str, str]],
                  current: dict[str, tuple[str, str]]) -> list[str]:
    """The paths created, deleted or changed (size or mtime) between two file snapshots."""
    return sorted(set(current) ^ set(baseline)
                  | {p for p in current if p in baseline and current[p] != baseline[p]})


def user_text(message: Any) -> str:
    """The text of a user message in any of the provider shapes (``""`` for other messages)."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content
                         if isinstance(part, dict) and part.get("type") in ("text", "input_text"))
    return ""


def is_continuation_start(message: Any) -> bool:
    """Whether *message* is the first user message of a continuation session."""
    return CONTINUATION_MARKER in user_text(message)


def append_user_text(message: dict[str, Any], note: str) -> None:
    """Append *note* to the text of the user *message* (string or text-block content)."""
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = content + "\n" + note
    elif isinstance(content, list):
        kind = "input_text" if any(isinstance(p, dict) and p.get("type") == "input_text"
                                   for p in content) else "text"
        content.append({"type": kind, "text": note})


def review_usage(trajectory_path: Path, since: int = 0) -> tuple[float, int]:
    """``(USD, tokens)`` spent by the gate's side agents logged in *trajectory_path*.

    The reviewer and the blind verifier's author are model calls outside
    the daemon's accounting (:data:`SIDE_AGENT_EVENTS`), so the trial runner
    adds their spend to the trial's cost from the log.  The log is
    append-only: a runner that reuses a trial directory passes the file's
    size before its run as *since* so earlier runs' reviews are not counted
    again.

    Args:
        trajectory_path: The trial's JSONL trajectory.
        since: Byte offset at which this run's records start.
    """
    cost, tokens = 0.0, 0
    for record in trajectory_events(trajectory_path, since):
        if record.get("event") in SIDE_AGENT_EVENTS:
            cost += float(record.get("cost_usd") or 0)
            tokens += int(record.get("tokens") or 0)
    return cost, tokens


def trajectory_events(trajectory_path: Path, since: int = 0) -> list[dict[str, Any]]:
    """The JSON records of *trajectory_path* from byte offset *since* (unreadable lines skipped)."""
    try:
        lines = trajectory_path.read_bytes()[since:].decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            events.append(record)
    return events


def ended_unchecked(trajectory_path: Path, since: int = 0) -> bool:
    """Whether the run in *trajectory_path* ended on a finish that no check vouched for.

    An implicit (text-only) finish, or an explicit one let through after the
    verifier cap (:meth:`ContainerHarness.finish_verdict`).
    """
    events = trajectory_events(trajectory_path, since)
    return any(e.get("event") in ("implicit_finish", "unverified_finish") for e in events)


def submit_best_checkpoint(config_path: Path, since: int = 0) -> str:
    """Restore the trial's last verified checkpoint when the run ended without an accepted finish.

    Called by the trial runner after the run.  When the trajectory holds a
    ``verified`` checkpoint and the current work tree fails the checks that
    the checkpoint passed (the host copies of the agent's verifier and of
    the independent check, whichever exist, run with :meth:`run_script`),
    the workdir is restored to the checkpoint, provided the archive still
    has the digest the checkpoint event recorded; a work tree that still
    passes is left alone, and so is a trial without a verified checkpoint.
    The checks run with the trial's original shell, kept on the host
    (:attr:`ContainerHarness.host_shell`) and found there by this harness.
    Without this, the state delivered is whatever is on disk when the budget
    runs out, after edits that the agent itself never got to check.

    Args:
        config_path: The trial's ``config.json``.
        since: Byte offset of this run's records in the trajectory.

    Returns:
        ``"restored"``, ``"kept"`` or ``"none"`` (no verified checkpoint), for the metrics.
    """
    harness = ContainerHarness(str(config_path))
    events = trajectory_events(harness.trajectory_path, since)
    verified = [e for e in events if e.get("event") == "checkpoint" and e.get("verified")
                and e.get("exit_code") == 0]
    if not verified:
        return "none"
    harness.checkpoint_digests[VERIFIED_CHECKPOINT] = str(verified[-1].get("sha256") or "")
    # A background job still writing into the work tree would undo the restore
    # (and may be what broke the tree): stop them all first.
    killed = (harness.shell(KILL_ALL_SCRIPT) or "").split()
    baseline = next((e for e in events if e.get("event") == "shell_baseline"), {})
    harness.shell_path = str(baseline.get("path") or harness.shell_path)
    failed_check = ""
    for path in (VERIFIER_PATH, BLIND_VERIFIER_PATH):
        saved = harness.saved_check_path(VERIFIED_CHECKPOINT, path)
        script = saved.read_text(encoding="utf-8") if saved.is_file() else ""
        if not script.strip():
            continue
        result = harness.run_script(script, path, harness.verifier_timeout)
        if result is None or result[0] != 0:
            failed_check = str(saved)
            break
    if not failed_check:
        harness._log({"event": "submit", "restored": False, "failed_check": "",
                      "killed_jobs": killed})
        return "kept"
    error = harness.restore_checkpoint(VERIFIED_CHECKPOINT)
    harness._log({"event": "submit", "restored": not error, "failed_check": failed_check,
                  "error": error, "killed_jobs": killed})
    return "restored" if not error else "kept"


_DOCKER_CLIENT: Any = None


def _docker_client() -> Any:
    """The process-wide Docker client (created on first use)."""
    global _DOCKER_CLIENT
    if _DOCKER_CLIENT is None:
        import docker

        _DOCKER_CLIENT = docker.from_env()
    return _DOCKER_CLIENT


def descendants(
    procs: dict[tuple[int, str], tuple[int, str]], roots: set[tuple[int, str]]
) -> set[tuple[int, str]]:
    """Keys of the processes in *procs* whose parent chain reaches one of *roots*.

    Args:
        procs: ``(pid, start time) -> (ppid, cmdline)`` snapshot of the process table.
        roots: Keys of the processes whose descendants are wanted.

    Returns:
        The descendant keys (the roots themselves excluded).
    """
    parent_pids = {key[0] for key in roots if key in procs}
    found: set[tuple[int, str]] = set()
    grew = True
    while grew:
        grew = False
        for key, (ppid, _) in procs.items():
            if key not in found and key not in roots and ppid in parent_pids:
                found.add(key)
                parent_pids.add(key[0])
                grew = True
    return found


def describe_processes(procs: dict[tuple[int, str], tuple[int, str]], limit: int = 8) -> str:
    """``pid cmdline (parent ppid)`` for up to *limit* processes, lowest pid first."""
    items = [f"{pid} {cmd[:120]} (parent {ppid})"
             for (pid, _), (ppid, cmd) in sorted(procs.items())]
    if len(items) > limit:
        items = items[:limit] + [f"and {len(items) - limit} more"]
    return "; ".join(items)


def append_to_last_tool_result(message: Any, note: str) -> bool:
    """Append *note* to the tool result carried by *message*, if it carries one.

    Handles the message shapes KISS models keep in their conversation: the
    OpenAI Responses item ``{"type": "function_call_output", "output": str}``,
    the chat-completions message ``{"role": "tool", "content": str}`` and the
    Anthropic user message whose ``content`` list ends with a ``tool_result``
    block (string or list content).  Anything else is left untouched so the
    note never lands in the task prompt or an assistant turn.

    Args:
        message: The last message added since the previous model call.
        note: The text to append.

    Returns:
        Whether the note was appended.
    """
    if not isinstance(message, dict):
        return False
    if message.get("type") == "function_call_output" and isinstance(message.get("output"), str):
        message["output"] += "\n\n" + note
        return True
    if message.get("role") == "tool" and isinstance(message.get("content"), str):
        message["content"] += "\n\n" + note
        return True
    blocks = message.get("content")
    if message.get("role") == "user" and isinstance(blocks, list) and blocks:
        block = blocks[-1]
        if isinstance(block, dict) and block.get("type") == "tool_result":
            inner = block.get("content")
            if isinstance(inner, str):
                block["content"] = inner + "\n\n" + note
            elif isinstance(inner, list):
                inner.append({"type": "text", "text": note})
            else:
                block["content"] = note
            return True
    return False


def shell_commands(args: Any) -> list[str]:
    """The command strings of a ``Bash`` or ``run_commands_parallel`` call's *args*.

    Args:
        args: The call's arguments as the trajectory recorded them.

    Returns:
        ``[command]`` for ``Bash``; the list behind ``commands`` (a JSON list
        or a list) for ``run_commands_parallel``; empty for anything else.
    """
    if not isinstance(args, dict):
        return []
    if isinstance(args.get("command"), str):
        return [args["command"]]
    commands = args.get("commands")
    if isinstance(commands, str):
        try:
            commands = json.loads(commands)
        except ValueError:
            return [commands]
    if isinstance(commands, list):
        return [c for c in commands if isinstance(c, str)]
    return []


def result_origin(call: dict[str, Any], written: list[str]) -> str:
    """How the result of the trajectory's tool *call* record is named to the reviewer.

    Args:
        call: A ``tool_call`` record, or ``{}`` for a result no call pairs with.
        written: Paths of the files the agent wrote so far (``Edit``/``Write``);
            a shell command naming one is marked, as its output is the agent's own.
    """
    tool = str(call.get("tool") or "")
    args: dict[str, Any] = call["args"] if isinstance(call.get("args"), dict) else {}
    if not tool:
        return "a tool result"
    if tool in SHELL_TOOLS:
        command = " ; ".join(shell_commands(args))
        own = sorted({path for path in written if path in command})
        note = f" (names a file the engineer wrote: {', '.join(own)})" if own else ""
        return f"output of: {command[:100]}{note}"
    if tool == "Read":
        return f"Read of {args.get('file_path', '')}"
    return f"result of {tool}"


def tool_result_texts(message: Any) -> list[str]:
    """The tool-result texts carried by *message*.

    Same message shapes as :func:`append_to_last_tool_result`.

    Args:
        message: One conversation message.

    Returns:
        The text of every tool result in the message; empty for other messages.
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
                texts.extend(part.get("text", "") for part in inner if isinstance(part, dict))
    return texts


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of provider message objects to JSON-friendly data."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, list):
            return [_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return repr(value)


SEA_TEMPLATE = '''"""Generated per-trial SEA; see kiss.agents.seas.coding.coding_sea."""
from kiss.agents.seas.base.base_sea import Verdict, WorkerSea
from kiss.agents.seas.coding.coding_sea import ContainerHarness

_harness = ContainerHarness.shared({config_path!r})


class TrialSea(WorkerSea):
    """One coding-benchmark trial, run by the shared ContainerHarness."""

    def description(self) -> str:
        return (
            "Runs one coding-benchmark trial inside a Docker container using the "
            "harness configured in config.json next to this file; generated by "
            "kiss.agents.seas.coding.coding_sea.write_trial_sea, not meant to be "
            "invoked by hand."
        )

    def prompt(self, task: str) -> str:
        return _harness.prompt(task)

    def system_prompt(self, system_prompt: str) -> str:
        return _harness.system_prompt()

    def settings(self, settings: dict) -> dict:
        return settings | _harness.settings()

    def tools(self, tools: list) -> list:
        self.has_summary_tool = False
        return _harness.tools(tools)

    def llm_call_hook(self, new_messages: list) -> list:
        return _harness.on_llm_call(new_messages)

    def tool_call_hook(self, name: str, args: dict) -> Verdict:
        return _harness.on_tool_call(name, args)
'''


def write_trial_sea(trial_dir: Path, config: dict[str, Any]) -> Path:
    """Write ``config.json`` and the generated SEA file for one trial.

    Args:
        trial_dir: Directory that receives ``config.json``, ``sea.py`` and
            the trajectory log.
        config: The :class:`ContainerHarness` configuration (``trajectory``
            and ``work_dir`` are filled in when absent).

    Returns:
        Path of the generated SEA file.
    """
    trial_dir = trial_dir.resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config.setdefault("trajectory", str(trial_dir / "trajectory.jsonl"))
    config.setdefault("work_dir", str(trial_dir))
    config_path = trial_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=1))
    sea_path = trial_dir / "sea.py"
    sea_path.write_text(SEA_TEMPLATE.format(config_path=str(config_path)))
    return sea_path
