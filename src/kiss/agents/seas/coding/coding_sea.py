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
verifier (:data:`VERIFIER_PATH`) must exist and pass (a run of it by the agent
that passed with nothing changed since is reused), so must the independent
check a ``blind_verifier`` model wrote from the statement alone when one is
configured (:data:`BLIND_VERIFIER_PATH`), and a configured ``reviewer`` model
reads the work once against the statement, the summary and the commands named
as evidence; the findings of that review must be answered by a change or
quoted evidence.  A refused finish gets one more round; after it only a
failing verifier refuses again, and the work is delivered with the open
findings as caveats.  Before its first edit the agent is held to an
interpretation ledger (:data:`LEDGER_PATH`), and a ``reading_check`` model of
another family reads the statement and the ledger once and names the
sentences it would read differently, which the agent must settle in the
ledger first.  The agent also gets checkpoint/rollback tools, background jobs,
and a machine-written inventory at every context restart.  Each answers a
way an unattended run goes wrong: a plausible result claimed without a check,
a reading of the statement settled by assumption and shared by every check
the same model writes, a reviewer argued down instead of answered, a context
restart that forgets what was already tried.  What the gate no longer does,
after the measurement of 2026-11-04:
refuse on the wording of the summary, on the ``evidence`` field, on a review
that came back blank, or on review findings past the first refused round,
and review the same work again after an approval; those refusals cost rounds
of work at full context and did not change what the hidden tests found.

The gate trusts nothing the agent can reach from its shell: the independent
check's script and the verified checkpoint's copies of the scripts it passed
live on the host (the container copy of the check is rewritten before each
run), a script runs as the argument of ``bash -c`` rather than from a file in
the container, and a checkpoint archive is restored only when its SHA-256 is
the one recorded when it was saved.
"""

from __future__ import annotations

import hashlib
import html
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

from kiss.agents.seas.base.base_sea import ALLOW, Verdict, WorkerSea, allow_with_note, refuse
from kiss.agents.seas.coding import coding_test_context as test_context
from kiss.agents.sorcar.checkpoints import CHECKPOINT_NAME
from kiss.agents.sorcar.coding_gate import (
    EXIT_CODE_MARKER,
    REVIEW_VERDICT_LINE,
    STOP_POLL_SECONDS,
    answered_findings,
    call_stop_event,
    default_effort,
    evidence_commands,
    gate_refusal,
    low_effort,
    numbered_items,
    result_exit_codes,
    retryable_review_failure,
    review_findings,
    tool_result_texts,
    verifier_decides,
)
from kiss.agents.sorcar.container_jobs import JOBS_DIR, KILL_ALL_SCRIPT, ContainerJobs
from kiss.agents.sorcar.docker_manager import _EXEC_TOKEN_VAR, kill_tagged_tree_script
from kiss.agents.sorcar.edit_guards import heredoc_targets
from kiss.agents.sorcar.evidence_gate import shell_commands, writes_files
from kiss.agents.sorcar.exec_bridge import BRIDGE_PREFIX, bridge_alive, bridge_exec
from kiss.agents.sorcar.shell_guards import (
    INSTALL_COMMANDS,
    INSTALL_TIMEOUT_SECONDS,
    destructive_pattern,
)
from kiss.core.event_sink import adopt_thread_context, thread_context
from kiss.core.kiss_error import BudgetExceededError, KISSError, StepLimitExceededError

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
shape.{reading_note} Update the ledger when you learn more; it is read by the \
reviewer.
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
never weaken it to make it pass.{blind_note}
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
plausible result you could not check is not a success, and a reading the \
summary leaves open ("if the task instead means", "a judgement call") is a \
requirement unmet: resolve it with evidence first. Name your evidence in finish's \
`evidence` argument: a JSON list of the commands you ran (exactly as you ran \
them) whose exit code 0 backs the claim; the reviewer reads it. finish runs \
your verifier{blind_mention} and one read-only review at once and reports \
every failure together. A refused finish gets one more round; after it only \
a failing verifier refuses again (up to {rounds} rounds), and the work is \
delivered with the open findings as caveats, so fix everything a refusal \
lists before calling finish again. Each finding of the review must be \
answered by a change or by evidence from a command, not by argument.

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

#: Where the agent keeps its own verifier (outside the workdir, so it is not
#: delivered as part of the work and does not change the work tree's listing).
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
#: Sentence of the prompt's ledger step when a ``reading_check`` model is set.
READING_NOTE = (
    " When you write the ledger, an independent reader of another model family names the "
    "sentences of the statement it would read differently (the note ends the Write's "
    "result); your first edit is refused until the ledger settles each of them."
)
#: Sentences of the prompt when a ``blind_verifier`` model is set: the
#: verifier step's note and the finish step's mention of the check.
BLIND_NOTE = (
    " finish also runs an independent check written from the statement alone by "
    "another agent; when the two disagree, resolve the disagreement with evidence, not "
    "by a vote."
)
BLIND_MENTION = ", the independent check"
#: The independent reading of the statement (B1 of the 2026-11-04 analysis):
#: a model of another family than the engineer's reads the statement and the
#: ledger once, before the first edit, and names the sentences it would read
#: differently.  Five attempts of one model agree with each other, and the
#: ledger, the verifier, the independent check and a same-model reviewer all
#: start from that one reading; this is the only stage that does not.
READING_CHECK_PROMPT = """\
An autonomous engineer is about to solve the task below and has written the \
interpretation ledger below: the readings they chose for the statement's ambiguous \
sentences, the inputs they will use, the tolerances and formats they settled, and \
what "done" means to them. You have not seen the task before and you will not see the \
work. Read the statement on its own, then the ledger, and list only the places where \
you would read the statement differently from the ledger, or where the statement \
asks for something the ledger does not settle: quote the sentence of the statement, \
then say in one sentence how you read it and what the engineer's reading would get \
wrong (a file, a name, a format, a value, a case, an input left unused). Take the \
statement literally. Do not restate readings you agree with, do not invent \
requirements the statement does not make, and do not comment on the approach. A \
benefit the statement expects from the change ("would cut context", "lowers latency", \
"prevents attention degradation") is the user's motivation, not a requirement: do not \
demand that the work measure or prove it, unless the statement explicitly asks for that \
measurement or names a target value for it; the requirement is the change itself.

Answer with the single line `NONE` when you read every sentence as the ledger does, \
otherwise with a numbered list, one item per sentence.

# Task statement
{task}

# The engineer's interpretation ledger
```
{ledger}
```
"""
#: Model answers that mean "no different reading".
READING_NONE = re.compile(r"^\s*`?NONE`?\.?\s*$", re.I)
#: Edits refused for an unsettled reading, per trial.
MAX_READING_REFUSALS = 2
READING_VERDICT = (
    "REFUSED: an independent reader of the task statement (model of another family, "
    "reading the statement and your ledger) would read these sentences differently or "
    "finds them unsettled:\n{items}\n"
    "Before the first edit under {workdir}, settle each item in {ledger}: either adopt "
    "the reading, or keep yours and write next to it the words of the statement that "
    "rule the other reading out. Then retry the edit."
)
#: Appended to the result of the ledger's first ``Write``: the same items, before any edit.
READING_NOTE_RESULT = (
    "\n\nIndependent reading of the task statement (a model of another family, from the "
    "statement and this ledger): it would read these sentences differently or finds them "
    "unsettled:\n{items}\n"
    "Settle each item in {ledger} before the first edit under {workdir} (an edit there is "
    "refused until the ledger changes): adopt the reading, or keep yours and write next to "
    "it the words of the statement that rule the other reading out."
)

#: Answer to an implicit (text-only) finish when the finish gate is on: the
#: agent is asked for an explicit ``finish`` call, which the gate checks.
IMPLICIT_FINISH_VERDICT = (
    "Not finished: a text reply does not end the run. Call finish explicitly with "
    "success=True only if every requirement is met and verified by a command you ran "
    "(the gate runs your verifier and a review), otherwise with "
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
#: Appended to the verdict of a check the gate did not run again: neither a file under
#: the workdir nor the script changed since its last run (:meth:`ContainerHarness.script_run`).
REUSED_NOTE = (
    "\n(This is the result of the gate's last run of {path}, reused: neither a file under "
    "{workdir} nor the script changed since. The check runs again after a change under "
    "{workdir}; a fix outside it is seen when you run {path} yourself, whose passing run "
    "stands.)"
)

#: Largest total size of changed files shown to the reviewer.
REVIEW_FILE_BYTES = 120_000
#: Largest size of one changed file shown to the reviewer (longer ones are cut).
REVIEW_ONE_FILE_BYTES = 30_000
#: Reviews per trial (``max_reviews`` in the config): one.  The first review
#: tells shaky work from solid work at the first finish; the rounds that
#: followed a re-review did not bring the refused group up to the run
#: average and a re-review after an approval changed its verdict in 13 of
#: 164 cases (the 2026-11-04 analysis).  A later finish is judged on the
#: review's findings by :meth:`ContainerHarness.unanswered_verdict`.
MAX_REVIEWS = 1
#: Seconds a review may take.  The provider's prompt cache lives about five
#: minutes; a gate that outlasts it makes the model call after it cost about
#: twelve times the cached price for the same context, so the review runs at
#: the provider's default effort and is given up at this deadline.
REVIEW_DEADLINE = 240
REVIEW_PROMPT = """\
You are reviewing, read-only, the work of an autonomous engineer on the task below. \
Nobody else will check it before it is delivered. \
Decide whether every requirement of the task statement is demonstrably met by the \
evidence (the engineer's summary and checklist, their interpretation ledger, their \
verifier script and its last output, the independent check and its output when there \
is one, the commands they name as evidence, and the files they changed). The gate runs \
the scripts itself at this finish and refuses on a failure, so judge them by what \
they cover, not by the outputs shown, which are from their most recent run before this \
review (none on a first finish). The evidence section lists the commands the engineer \
names as backing the claim, each with the exit code of its last run in this session \
(`never ran` for one that did not run); a command is evidence of a requirement only \
when it ran, exited 0 and tests that requirement: `echo`, `cat`, a script the engineer \
wrote themselves, or the words of the statement prove nothing about the work on their \
own. Do not invent problems: report only a requirement that is not \
met, not checked, or checked wrongly, each with the evidence that shows it. A benefit \
the statement expects from the change ("would cut context", "lowers latency", "prevents \
attention degradation") is the user's motivation, not a requirement: do not demand that \
the work measure or prove it, unless the statement explicitly asks for that measurement \
or names a target value for it; the requirement is the change itself. Take the \
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

# Commands the engineer names as evidence, with the exit code of their last run
{evidence}
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
#: Failed reviews (no verdict, a deadline hit, or a model error) a trial
#: tolerates before a finish goes through without one; they do not use up
#: ``max_reviews``.  A failed review refuses nothing: a blank answer names
#: nothing to fix, so a refusal on it changes the work by chance only.
MAX_FAILED_REVIEWS = 2
#: A finding the next finish neither changed code for nor answered with evidence.
UNANSWERED_VERDICT = (
    "Not finished: these findings of the review were not answered by a change or by "
    "quoted command output:\n{findings}\n"
    "Change the work, or quote under each number the command and output that shows "
    "the finding wrong, then call finish again."
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
#: Longest blind-verifier output shown.
BLIND_OUTPUT_CHARS = 3000
#: Most evidence commands shown to the reviewer.
MAX_EVIDENCE_SHOWN = 30
#: Refused rounds of the finish gate per trial (``max_gate_rounds`` in the
#: config).  The first refusal buys one round whatever refused it; a later
#: round is refused only when the agent's own verifier refused (missing,
#: failing or too slow), up to this cap.  Past that the finish is delivered
#: with the open findings as caveats: the pass rate was flat against the
#: number of refused rounds while each round cost minutes of gate time and
#: a model turn at full context.
MAX_GATE_ROUNDS = 3
#: Shown once when a give-up passes both checks (:meth:`ContainerHarness.giveup_verdict`).
GIVEUP_RECONSIDER_VERDICT = (
    "Not accepted yet: you called finish(success=False), but your own verifier {verifier} and "
    "the independent check {blind} both pass on the current work tree. Either name the "
    "requirement that is still unmet (one sentence, with the command that shows it) and call "
    "finish(success=False) again, or call finish(success=True) with the summary of what was "
    "done and the evidence."
)
#: Appended to the summary of a finish delivered past the round cap.
DELIVERED_CAVEATS = (
    "\n<h4>Delivered after {rounds} refused finish rounds; open findings</h4>\n<ol>\n{items}\n</ol>"
)
#: The caveat of a finish delivered past the verifier-refusal cap: neither check ran.
#: (Worded without the evidence gate's hedge phrases, which it would otherwise
#: flag in the summary it is appended to.)
UNVERIFIED_CAVEAT = (
    "Checks did not run: the agent's verifier {verifier} and the independent check {blind} "
    "refused {refusals} finishes and were not run again; the work is delivered as it stands."
)
#: Appended to the summary of every give-up: the checks' outcome next to the agent's verdict.
GIVEUP_NOTE = (
    "\n<p><b>Checks at this finish:</b> agent: failed; verifier {verifier}: {verifier_state}; "
    "independent check {blind}: {blind_state}.</p>"
)

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
#: (A long foreground ``sleep`` and a here-document or ``sed -i`` rewrite are
#: the ``Bash`` tool's own business: it waits on the running job instead of
#: sleeping and applies a rewrite as a ``Write``/``Edit`` through the agent, so
#: the harness's ledger gate and bookkeeping see the edit.)
DESTRUCTIVE_VERDICT = (
    "REFUSED: the Bash call did NOT run: it would kill every process in the container "
    "(your own shell included) or delete the working directory, which ends the task with "
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
#: Each one the harness adopted as a background job is listed as ``left
#: running as job <id> (<command>, log <path>)``; the others as ``pid
#: command (parent N)``.
SURVIVORS_NOTE = (
    "\n\nHarness note: processes started by your last shell command(s) are still running: {procs}. "
    "Before you finish, stop the ones that are not meant to keep running."
)
#: Follows the survivors note when one of them became a job.
ADOPTED_HINT = (
    " A job is waited on with bash_job(<id>, 'wait', timeout_seconds=N, until='<regex>'), which "
    "returns the moment it exits or its log matches: use that instead of sleeping and polling."
)
#: A stdout redirection operator of a shell command: ``>``, ``>>``, ``&>``,
#: ``&>>``; not ``2>`` (another descriptor) nor the ``>`` of ``>&1`` (a
#: duplication).  Looked for on a copy whose quoted and escaped pieces are
#: masked (:func:`stdout_target`), so a quoted or escaped ``>`` is data.
REDIRECTION = re.compile(r"(?<![\d&>])(?:&>>?|>>?)(?!&)")
#: The pieces of a shell command line: a quoted string or an escaped
#: character (kept whole), a separator between simple commands (``&&``,
#: ``||``, ``;``, ``|``, a newline, a parenthesis, a lone ``&`` that is not
#: part of ``2>&1`` or ``&>``), or any other character.
SHELL_PIECE = re.compile(
    r"'[^']*'|\"(?:\\.|[^\"\\])*\"|\\.|(?P<sep>&&|\|\||[;|\n()]|(?<!>)&(?!>))|.", re.DOTALL)
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
        #: Model of the read-only review a gated finish gets (``""``: no review;
        #: ``other``: the strongest configured model of another family than the
        #: trial's, a second opinion that does not share the executor's reading).
        self.reviewer: str = self.resolve_side_model(str(cfg.get("reviewer") or ""))
        #: Model request overrides for this trial (``reasoning_effort`` for OpenAI-style
        #: models, ``output_config={"effort": ...}`` for Claude); empty means provider defaults.
        self.model_overrides: dict[str, Any] = dict(cfg.get("model_config") or {})
        self.implicit_finish_vetoed = False
        self.verifier_refusals = 0
        #: Whether both verifiers passed at the current finish (a verifier that
        #: failed, or was skipped because its refusals are used up, makes it false:
        #: the ``verified`` checkpoint is taken only when it is true, and a finish
        #: delivered while it is false is an unverified one).
        self.checks_passed = False
        #: Seconds the gate gives the agent's verifier (config ``verifier_timeout``).
        self.verifier_timeout = float(cfg.get("verifier_timeout", VERIFIER_TIMEOUT))
        #: ``(script, exit code, output)`` of the verifier's last run by the gate.
        self.last_verifier_run: tuple[str, int | None, str] = ("", None, "")
        #: The last run of each check the gate ran (``verifier``, ``blind``), as
        #: ``((files under the workdir after it, script text), (script, exit code,
        #: output))``: a finish at which neither changed reuses the run
        #: (:meth:`script_run`).
        self.stage_memo: dict[str, tuple[tuple[dict[str, tuple[str, str]], str],
                                         tuple[str, int | None, str]]] = {}
        #: The verifier's last verdict, by whoever ran it: ``(exit code, when,
        #: turn, who, output)``; the hand-off inventory shows it and the restart
        #: budget compares it (:meth:`restart_denied`).
        self.last_verdict: tuple[str, float, int, str, str] | None = None
        #: Hand-offs so far (config ``restart_budget``, :data:`RESTART_BUDGET`), the
        #: change count and the verifier verdict at the last one, and whether the
        #: last one was counted at the ``finish`` that asked for it.
        self.restarts = 0
        self.restart_budget = int(cfg.get("restart_budget", RESTART_BUDGET))
        self.restart_changes = 0
        self.restart_verdict: str | None = None
        self.restart_counted = False
        #: How many changes of the work tree the agent made (an edit under the
        #: workdir, a writing shell command, a rollback), and the count at the
        #: agent's own last passing run of its verifier: equal at a finish,
        #: that run stands for the gate's (:meth:`verifier_verdict`).
        self.changes = 0
        self.verifier_passed_at: int | None = None
        self.verifier_passed_snapshot: dict[str, tuple[str, str]] | None = None
        #: Model of the independent reading of the statement (``""``: none;
        #: config ``reading_check``, ``other`` for a model of another family
        #: than the trial's, and no reading at all when none is configured:
        #: a reader of the trial's own family shares its reading).  Its
        #: items, the ledger text they were raised against, and the edits
        #: refused for them (:meth:`reading_verdict`).
        reading = str(cfg.get("reading_check") or "")
        self.reading_check: str = self.resolve_side_model(reading)
        if reading == "other" and self.reading_check == self.model_name:
            self.reading_check = ""
        self.reading_items: list[str] = []
        self.reading_ledger: str | None = None
        self.reading_refusals = 0
        #: Model of the independent check's author (``""``: no blind verifier;
        #: config ``blind_verifier``, ``same`` for the trial's model).
        blind = str(cfg.get("blind_verifier") or "")
        self.blind_verifier: str = self.resolve_side_model(blind)
        #: Whether the blind verifier's agent ran (once per trial, plus one retry
        #: when it delivered nothing: on a model of another family, or on the same
        #: model when the daemon pins every task to its model).
        self.blind_written = False
        self.blind_retried = False
        #: The author's thread, started at the first model call so the check is
        #: ready before the first finish instead of written inside it.
        self.blind_thread: threading.Thread | None = None
        #: Refused rounds of the gate so far and the cap (config ``max_gate_rounds``).
        self.gate_rounds = 0
        self.max_gate_rounds = int(cfg.get("max_gate_rounds", MAX_GATE_ROUNDS))
        #: Findings left open when the finish was delivered past the cap.
        self.delivered_findings: list[str] = []
        #: Whether the one reconsideration of a passing give-up was spent.
        self.giveup_reconsidered = False
        #: Exit code of the last run of every shell command the agent ran, as a
        #: ``Bash`` command, a ``run_commands_parallel`` command or a background job
        #: waited on (the finish call's ``evidence`` is checked against it).
        self.exit_codes: dict[str, str] = {}
        #: The command of every background job started (by job id), for the exit
        #: code its ``bash_job`` wait reports.
        self.background_jobs: dict[str, str] = {}
        #: Extra fields of the next ``gate_stage`` record of a stage, by stage name:
        #: the review's and the blind author's model, cost and round, set where
        #: they are known (:meth:`review_verdict`, :meth:`write_blind_verifier`).
        self.stage_fields: dict[str, dict[str, Any]] = {}
        #: The independent check's script, kept on the host: the gate runs this
        #: text, not the container's copy, which the agent can rewrite.
        self.blind_script = ""
        #: ``(script, exit code, output)`` of the blind verifier's last run by the gate.
        self.last_blind_run: tuple[str, int | None, str] = ("", None, "")
        #: Reviews done so far and the open findings of the last one.  A review
        #: that fails (no verdict twice, or a model error) counts under
        #: ``failed_reviews`` instead, capped by :data:`MAX_FAILED_REVIEWS`.
        self.reviews = 0
        self.failed_reviews = 0
        self.max_reviews = int(cfg.get("max_reviews", MAX_REVIEWS))
        self.open_findings: list[str] = []
        #: Workdir snapshot (:meth:`file_snapshot`) when the last review ran, the
        #: claim it judged (:func:`claim_text`) and its verdict: a finish with the
        #: same claim on the same files reuses the verdict (:meth:`reused_review`).
        self.review_snapshot: dict[str, tuple[str, str]] | None = None
        self.review_claim: str | None = None
        self.review_result: str | None = None
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
        #: The tool calls since the last model call, in order: ``(tool name, args,
        #: changes so far)``, ``refused`` for a call the hook refused, so the turn's
        #: results pair with the right calls (:meth:`record_exit_codes`).
        self.turn_commands: list[tuple[str, dict[str, Any], int]] = []
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
        #: The shell commands of the current turn, in order (the log a survivor
        #: writes is read off their redirections).
        self.turn_shell_commands: list[str] = []
        #: The container's background-job registry (the one behind the run's
        #: ``bash_job`` tool), where the survivors of a shell call are adopted;
        #: ``None`` until :meth:`tools` saw the tool.
        self.jobs: ContainerJobs | None = None
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

    def resolve_side_model(self, setting: str) -> str:
        """Return the model a side agent (reviewer, check author) runs on for *setting*.

        Args:
            setting: ``""`` (no side agent), a model name, ``same`` (the
                trial's model) or ``other`` (a model of another family than
                the trial's, :func:`~kiss.core.models.model_families.other_family_model`;
                the trial's model when no other family is configured).
        """
        if setting == "same":
            return self.model_name
        if setting == "other":
            from kiss.core.models.model_families import other_family_model

            return other_family_model(self.model_name) or self.model_name
        return setting

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
        """The unattended-engineering prompt (replaces Sorcar's default prompt).

        The sentences about the independent check and the reading check
        are there only when a model is configured for them.
        """
        tests = TEST_CONTEXT_NOTE if self.test_context else ""
        blind = bool(self.blind_verifier)
        return SYSTEM_PROMPT.format(workdir=self.workdir, test_context=tests,
                                    verifier=VERIFIER_PATH, ledger=LEDGER_PATH,
                                    shell_timeout=DEFAULT_SHELL_TIMEOUT,
                                    rounds=self.max_gate_rounds,
                                    reading_note=READING_NOTE if self.reading_check else "",
                                    blind_note=BLIND_NOTE if blind else "",
                                    blind_mention=BLIND_MENTION if blind else "")

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
        # ``bash_job`` is a bound method of the container's job registry: the
        # survivors of a shell call are adopted there (:meth:`shell_notes`).
        for tool in tools:
            owner = getattr(tool, "__self__", None)
            if isinstance(owner, ContainerJobs):
                self.jobs = owner
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
            self.start_blind_author()
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
                # A hand-off the agent asked for was counted at its finish; one
                # the context limit forced is counted here, and the restart
                # budget may end the run before this session's first model call.
                denied = not self.restart_counted and self.restart_denied()
                self.restart_counted = False
                state = ""
                try:
                    state = self.inventory()
                    append_user_text(new_messages[0], state)
                    self._log({"event": "inventory", "turn": turn})
                except Exception as error:
                    self._log({"event": "hook_error", "turn": turn, "hook": "inventory",
                               "error": repr(error)})
                if denied:
                    raise RestartBudgetError(self.restart_caveat(state))
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
        note = None
        try:
            if name in ("Bash", "run_commands_parallel"):
                verdict = self.shell_verdict(args)
            elif name in ("Edit", "Write") and self.finish_gate:
                verdict = self.ledger_verdict(args)
                if verdict is None:
                    note = self.reading_note(name, args)
            elif name == "finish" and self.finish_gate:
                verdict = self.finish_verdict(args)
        except Exception as error:  # a guard is optional; never end the trial on its own bug
            self._log({"event": "hook_error", "turn": self.turns, "hook": name,
                       "error": repr(error)})
        self._log({"event": "tool_call", "turn": self.turns, "tool": name,
                   "args": _jsonable(args), "blocked": verdict is not None})
        if verdict is None and name in ("Edit", "Write", "Bash", "run_commands_parallel"):
            self.track_change(name, args)
        if verdict is None and name == "Read" and isinstance(args.get("file_path"), str):
            with self._lock:
                self.files_read[posixpath.normpath(posixpath.join(self.workdir,
                                                                  args["file_path"]))] = None
        with self._lock:  # a refused call still gets a result (the refusal), so it is listed
            self.turn_commands.append((name if verdict is None else "refused", args,
                                       self.changes))
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
        if verdict is not None:
            return refuse(verdict)
        return allow_with_note(note) if note else ALLOW

    def before_shell(self, args: dict[str, Any]) -> None:
        """Remember the process table before the turn's first shell call; note kill attempts.

        Args:
            args: Arguments of the ``Bash`` or ``run_commands_parallel`` call.
        """
        commands = shell_commands(args)  # decoded: a kill may start any line of a command
        with self._lock:
            take = self.processes_before is None
            self.kill_attempted = self.kill_attempted or any(
                KILL_COMMANDS.search(c) for c in commands)
            self.turn_shell_commands.extend(commands)
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
            commands, self.turn_shell_commands = self.turn_shell_commands, []
        if before is None:
            return []
        notes: list[str] = []
        after = self.process_snapshot()
        with self._lock:
            # Children of a process already reported (a build's compiler
            # steps, a server's workers) are tracked without a new note.
            # The processes of a background job (a started job's shell, an
            # adopted job's watcher, their children) are the registry's.
            offspring = descendants(after, self.tracked_pids)
            jobs_own = {key for key, (_, cmd) in after.items() if JOBS_DIR in cmd}
            jobs_own |= descendants(after, jobs_own)
            new = {key: info for key, info in after.items()
                   if key not in before and key not in offspring and key not in self.tracked_pids
                   and key not in jobs_own}
            self.tracked_pids |= set(new) | offspring
            self.tracked_pids &= set(after)
            alive = {key: after[key] for key in sorted(self.tracked_pids)}
        if new:
            notes.append(self.survivors_note(new, commands))
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

    def survivors_note(
        self, new: dict[tuple[int, str], tuple[int, str]], commands: list[str],
    ) -> str:
        """The survivors note for the processes *new* the turn's shell *commands* left running.

        With a job registry at hand (:attr:`jobs`), each survivor is adopted
        as a background job, with the log its command redirected its output
        to (:func:`command_logs`), and listed as ``left running as job <id>
        (<command>, log <path>)``; a survivor the registry could not adopt
        (it exited meanwhile) is listed as before.  Adopted jobs are recorded
        in :attr:`background_jobs` so a later ``bash_job`` wait on them
        reports an exit code the gate can read.

        Args:
            new: ``(pid, start time) -> (ppid, cmdline)`` of the survivors.
            commands: The shell commands of the turn, in order.

        Returns:
            The note text (:data:`SURVIVORS_NOTE`, plus :data:`ADOPTED_HINT`
            when a survivor became a job).
        """
        logs = [log for command in commands for log in command_logs(command, self.workdir)]
        items: list[str] = []
        adopted = False
        for key, (ppid, cmdline) in sorted(new.items()):
            line = ""
            if self.jobs is not None:
                job_id, line = self.jobs.adopt(key[0], cmdline, survivor_log(cmdline, logs))
                if job_id is None:
                    self._log({"event": "adopt_failed", "turn": self.turns, "pid": key[0],
                               "error": line})
                    line = ""
                else:
                    adopted = True
                    with self._lock:
                        self.background_jobs[job_id] = cmdline
            items.append(line or describe_processes({key: (ppid, cmdline)}))
        if len(items) > MAX_LISTED_SURVIVORS:
            items[MAX_LISTED_SURVIVORS:] = [f"and {len(items) - MAX_LISTED_SURVIVORS} more"]
        return SURVIVORS_NOTE.format(procs="; ".join(items)) + (ADOPTED_HINT if adopted else "")

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
        """Gate a successful ``finish`` on evidence: the scripts and a review.

        Every ``finish(success=True)`` must pass: the agent's own verifier
        (:meth:`verifier_verdict`), the independent check written from the
        statement alone when one is configured (:meth:`blind_verdict`; both
        run with the trial's original shell, :meth:`run_script`), and a
        read-only review of the work (:meth:`review_verdict`, ``max_reviews``
        per trial) or, once the review is spent, the answers to its open
        findings (:meth:`unanswered_verdict`).  The summary's wording and the
        ``evidence`` field refuse nothing: they are the reviewer's input.
        There is no clock: a finish refused for being early buys a second
        round that breaks as much as it fixes.  A round in which nothing
        changed runs nothing: each script's last run stands while neither
        the files under the workdir nor its text changed (:meth:`script_run`),
        and the review's verdict while the claim and the files are those
        it judged (:meth:`reused_review`); the stages that do run, run at
        once (:meth:`run_stages`).

        A pause (``is_continue=True``) is checked by nothing, but counts as
        a restart: from the :attr:`restart_budget`-th on, one that comes
        with no new edit and the same verifier verdict as the restart
        before it ends the run instead, as ``finish(success=False)`` with
        the state block as the caveat (:meth:`restart_denied`).

        The first refusal buys one round whatever refused it.  A later
        round is refused only when a script refused (the verifier or the
        independent check; a review finding left open is then a caveat of
        the delivery), up to ``max_gate_rounds``: refused rounds past the
        first did not raise the pass rate and cost a model turn at full
        context each.

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
        nothing checked.

        A finish refused ``max_gate_rounds`` times is delivered with the
        open verdicts as caveats (:meth:`deliver_with_findings`); so is a
        finish past the verifier cap (:data:`MAX_VERIFIER_REFUSALS`), whose
        checks no longer run (:data:`UNVERIFIED_CAVEAT`).  A finish delivered
        with :attr:`checks_passed` false is logged as unverified
        (``unverified_finish``, :func:`ended_unchecked`): the run ends, but
        as a failure, and the runner submits the last verified tree instead
        (:func:`submit_best_checkpoint`), so repeating ``finish`` until a
        cap is spent delivers nothing a check refused.

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
            if str(args.get("is_continue")).lower() == "true":
                # a pause for a fresh context, not a give-up: nothing to check,
                # unless the restart budget ends the run here instead
                if self.restart_denied():
                    args["is_continue"] = False
                    args["summary_in_html"] = (str(args.get("summary_in_html", ""))
                                               + self.restart_caveat(self.inventory()))
                else:
                    self.restart_counted = True
                return None
            return self.giveup_verdict(args)
        self.checks_passed = True
        review = self.review_wanted()
        stages = self.run_stages(args, review=review)
        checks = [stages["verifier"], stages["blind"]]
        skipped = not self.checks_passed  # past the refusal cap: the checks did not run
        self.checks_passed = not skipped and not any(checks)
        if self.checks_passed:
            self.save_checkpoint(VERIFIED_CHECKPOINT, verified=True)
        verdicts = list(checks)
        if stages.get("review") is not None:
            verdicts.append(stages["review"])
        elif not review and self.reused_review(args):  # the same claim on the same files
            verdicts.append(self.review_result)
        else:  # no review this round, or one that passed or failed: its findings stand
            verdicts.append(self.unanswered_verdict(args))
        open_verdicts = [v for v in verdicts if v]
        refusable = self.gate_rounds < self.max_gate_rounds and (
            self.gate_rounds == 0 or any(checks))
        if open_verdicts and refusable:
            self.gate_rounds += 1
            self._log({"event": "gate_round", "turn": self.turns, "round": self.gate_rounds,
                       "refusals": len(open_verdicts)})
            return gate_refusal(self.gate_rounds, self.max_gate_rounds, open_verdicts)
        if skipped:  # not a refusal (the checks will not run again), but a caveat
            open_verdicts.append(UNVERIFIED_CAVEAT.format(
                verifier=VERIFIER_PATH, blind=BLIND_VERIFIER_PATH,
                refusals=self.verifier_refusals))
        if not open_verdicts:
            return None
        self.deliver_with_findings(args, open_verdicts)
        return None

    def review_wanted(self) -> bool:
        """Whether this round runs a review.

        A reviewer is set, a review is possible, the agent's verifier exists
        (a round refused for a missing verifier does not pay for a review),
        and, once a review was done, a file under the workdir changed since
        it: a finish that changed nothing is judged on the review's findings
        by :meth:`unanswered_verdict` instead, without a model call,
        whatever refused the previous round (a clean review is not paid for
        twice because the evidence or the summary was wrong).
        """
        if not (self.reviewer and self.review_possible()
                and self.read_container_file(VERIFIER_PATH) is not None):
            return False
        if self.review_snapshot is not None:
            return bool(self.changed_since(self.review_snapshot))
        return True

    def run_stages(self, args: dict[str, Any], review: bool) -> dict[str, str | None]:
        """Run the gate's stages at once and return each one's verdict by name.

        The agent's verifier (:meth:`verifier_verdict`), the independent
        check (:meth:`blind_verdict`) and, when *review* is set, the review
        (:meth:`review_verdict`) run in threads of their own: the gate's
        time is the longest stage's, not their sum, and a refusal lists
        every failed stage so the next round fixes all of them.  Each stage
        is recorded as a ``gate_stage`` event (:meth:`gate_stage`) with its
        duration and outcome; a stage that raises is a failed stage, not a
        failed trial.

        Args:
            args: Arguments of the ``finish`` call.
            review: Whether the review stage runs.
        """
        stages: dict[str, Callable[[], str | None]] = {"verifier": self.verifier_verdict}
        if self.blind_verifier:
            stages["blind"] = self.blind_verdict
        if review:
            stages["review"] = partial(self.review_verdict, args)
        from kiss.core import stop_signal

        verdicts: dict[str, str | None] = {"blind": None}
        seconds: dict[str, float] = {}
        stop = stop_signal.get_thread_stop_event()  # the run's Stop reaches every stage
        # a review that could not run is no verdict (a check that did not happen
        # is never a refusal); a check that could not run is one
        threads = [threading.Thread(target=self.run_stage, name=f"gate-{name}", daemon=True,
                                    args=(name, stage, verdicts, seconds,
                                          self.failed_review if name == "review" else
                                          failed_stage, stop))
                   for name, stage in stages.items()]
        for thread in threads:
            thread.start()
        for thread in threads:
            while thread.is_alive():  # short slices: an injected Stop lands between them
                thread.join(STOP_POLL_SECONDS)
        # past the refusal cap the two checks do not run (:attr:`checks_passed` is false)
        skipped = self.verifier_refusals >= MAX_VERIFIER_REFUSALS and not self.checks_passed
        for name in stages:
            fields = self.stage_fields.pop(name, {})
            outcome = ("fail" if verdicts[name] is not None else
                       "error" if "error" in fields else
                       "skipped" if skipped and name != "review" else "pass")
            self.gate_stage(name, seconds[name], outcome, **fields)
        return verdicts

    def run_stage(self, name: str, stage: Callable[[], Any], results: dict[str, Any],
                  seconds: dict[str, float], failure: Callable[[Exception], Any],
                  stop: threading.Event | None = None) -> None:
        """Run one gate stage (a thread's body): its result into *results*, its time into *seconds*.

        Args:
            name: The stage's name.
            stage: The stage.
            results: Where the stage's result goes, by name.
            seconds: Where the stage's duration goes, by name.
            failure: Makes the result of a stage that raised, from the error.
            stop: The calling thread's stop event, bound as this thread's so
                the run's Stop ends a model call or a script the stage makes.
        """
        from kiss.core import stop_signal

        stop_signal.set_thread_stop_event(stop)
        started = time.time()
        try:
            results[name] = stage()
        except Exception as error:
            self._log({"event": "hook_error", "turn": self.turns, "hook": f"gate:{name}",
                       "error": repr(error)})
            results[name] = failure(error)
        except BaseException:
            # The stage was stopped (a model call raises KeyboardInterrupt on
            # its thread's stop event): the run is ending, no verdict stands.
            results[name] = f"Not finished: the {name} check was stopped."
        seconds[name] = time.time() - started

    def gate_stage(self, stage: str, seconds: float, outcome: str, **fields: Any) -> None:
        """Record one gate stage: in the trajectory and, through the running agent's
        printer, in the task's event log (``gate_stage``), so gate time, model and cost
        per stage can be read back without the trajectory file.

        Args:
            stage: ``verifier``, ``blind``, ``review`` or ``blind_author``.
            seconds: The stage's wall-clock duration.
            outcome: ``pass``, ``fail``, ``skipped`` or ``missing``.
            **fields: Extra fields: the ``model``, ``cost_usd`` and ``round`` of a
                stage that ran a model (:attr:`stage_fields`), ``giveup`` for a
                check run by a give-up, ``retried`` for the blind author.
        """
        from kiss.core.event_sink import publish

        record = {"event": "gate_stage", "turn": self.turns, "stage": stage,
                  "seconds": round(seconds, 3), "outcome": outcome, **fields}
        self._log(dict(record))
        record.pop("event")
        publish("gate_stage", **record)

    def evidence_report(self, args: dict[str, Any]) -> str:
        """The commands a finish names as evidence, with their last exit code, for the reviewer.

        The ``evidence`` argument of ``finish`` is a JSON list of the commands
        the success claim rests on (a plain string is one command); each is
        looked up among the commands the agent ran (:attr:`exit_codes`, from
        the exit codes their results showed).  The field refuses nothing:
        the reviewer reads it, and a command that never ran or failed is
        shown as such.

        Args:
            args: Arguments of the ``finish`` call.
        """
        commands = evidence_commands(args.get("evidence"))
        if not commands:
            return "(none named)"
        with self._lock:
            codes = dict(self.exit_codes)
        return "\n".join(f"- `{command[:300]}` (exit {codes.get(command.strip(), 'never ran')})"
                         for command in commands[:MAX_EVIDENCE_SHOWN])

    def deliver_with_findings(self, args: dict[str, Any], open_verdicts: list[str]) -> None:
        """Let a finish through past the round cap, with the open findings as caveats.

        The summary the caller sees gets the open verdicts appended
        (:data:`DELIVERED_CAVEATS`; the finish arguments are the dict the
        tool is then called with), the trajectory records
        ``delivered_with_findings``, and the user or the calling agent
        decides.  A finish whose checks failed or did not run
        (:attr:`checks_passed` false) is still an unverified one
        (``unverified_finish``), which the trial runner treats as before.

        Args:
            args: Arguments of the ``finish`` call (edited in place).
            open_verdicts: The verdicts of the stages that refused, and the
                caveat of checks that did not run.
        """
        self.delivered_findings = [v.splitlines()[0][:500] for v in open_verdicts]
        items = "\n".join(f"<li>{html.escape(v)}</li>" for v in open_verdicts)
        args["summary_in_html"] = str(args.get("summary_in_html", "")) + DELIVERED_CAVEATS.format(
            rounds=self.gate_rounds, items=items)
        self._log({"event": "delivered_with_findings", "turn": self.turns,
                   "rounds": self.gate_rounds, "findings": self.delivered_findings,
                   "checks_passed": self.checks_passed})
        if not self.checks_passed:
            self._log({"event": "unverified_finish", "turn": self.turns,
                       "refusals": self.verifier_refusals})
        return None

    def giveup_verdict(self, args: dict[str, Any]) -> str | None:
        """Check a ``finish(success=False)`` the way a success claim is checked.

        The agent's verifier and the independent check run (at once, without
        counting refusals); their outcome is appended to the summary next to
        the agent's own verdict (:data:`GIVEUP_NOTE`), so whoever reads the
        result sees "verifier: pass" instead of a bare failure notice, and
        a work tree that passes both is checkpointed as verified, so the
        trial runner delivers it.  When both pass, the first such give-up is
        answered once with :data:`GIVEUP_RECONSIDER_VERDICT`; the next ends
        the run whatever it says.  Give-ups are checked because many are
        wrong: the work on disk passes the checks the agent believed it had
        failed.

        Args:
            args: Arguments of the ``finish`` call (edited in place).
        """
        states = self.run_checks()
        # the agent's verifier must pass; the independent check too when there is one
        passed = states["verifier"] == "pass" and states["blind"] in ("pass", "missing")
        self._log({"event": "giveup_check", "turn": self.turns, "states": states,
                   "passed": passed})
        if passed:
            self.save_checkpoint(VERIFIED_CHECKPOINT, verified=True)
        args["summary_in_html"] = str(args.get("summary_in_html", "")) + GIVEUP_NOTE.format(
            verifier=VERIFIER_PATH, verifier_state=states["verifier"],
            blind=BLIND_VERIFIER_PATH, blind_state=states["blind"])
        if passed and not self.giveup_reconsidered:
            self.giveup_reconsidered = True
            return GIVEUP_RECONSIDER_VERDICT.format(verifier=VERIFIER_PATH,
                                                    blind=BLIND_VERIFIER_PATH)
        return None

    def run_checks(self) -> dict[str, str]:
        """Run the agent's verifier and the independent check at once, counting no refusal.

        Returns:
            The state of each check by name (``verifier``, ``blind``), as
            :func:`check_state` reads its run: ``pass``, ``missing`` (no such
            check), ``timeout``, ``fail (exit N)`` or ``error (...)`` for a
            check that could not run.
        """
        runs: dict[str, tuple[str, int | None, str]] = {}
        seconds: dict[str, float] = {}
        threads = [threading.Thread(target=self.run_stage, name=f"giveup-{name}", daemon=True,
                                    args=(name, check, runs, seconds, failed_check))
                   for name, check in (("verifier", self.verifier_run),
                                       ("blind", self.blind_run))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        states = {name: check_state(*runs[name]) for name in ("verifier", "blind")}
        for name, state in states.items():
            outcome = state if state in ("pass", "missing") else "fail"
            self.gate_stage(name, seconds[name], outcome, giveup=True,
                            **self.stage_fields.pop(name, {}))
        return states

    def review_possible(self) -> bool:
        """Whether another review may run (neither the review nor the failed-review cap is hit)."""
        return self.reviews < self.max_reviews and self.failed_reviews < MAX_FAILED_REVIEWS

    def verifier_verdict(self) -> str | None:
        """Run the agent's own verifier; the verdict when it is missing, failing or too slow.

        At most :data:`MAX_VERIFIER_REFUSALS` refusals per trial, so a verifier
        the agent cannot make pass does not trap the run; a finish let through
        that way is not a verified one (:attr:`checks_passed`).  The script
        is read once and that text is run (:meth:`run_script`), so what the
        reviewer sees is what ran (:attr:`last_verifier_run`).  The agent's
        own run of the script is reused when it passed and nothing changed
        since (:attr:`verifier_passed_at`, set by :meth:`record_exit_codes`,
        and the files under the workdir as they were then,
        :attr:`verifier_passed_snapshot`): the exit code it showed is what
        a run now would show, and a third of the gated finishes came right
        after such a run.  Such a run also stands past the refusal cap: it
        asks for no run of the script.  The gate's own last run stands
        while neither the files under the workdir nor the script changed
        since it (:meth:`script_run`); its verdict then says so
        (:data:`REUSED_NOTE`).
        """
        if (self.verifier_passed_at == self.changes and self.verifier_passed_snapshot is not None
                and self.file_snapshot() == self.verifier_passed_snapshot):
            script = self.read_container_file(VERIFIER_PATH)
            if script:
                self.last_verifier_run = (script, 0,
                                          "(passed when you ran it; nothing changed since)")
                self._log({"event": "verifier", "turn": self.turns, "exit_code": 0,
                           "reused": True})
                self.stage_fields["verifier"] = {"reused": True}
                return None
        if self.verifier_refusals >= MAX_VERIFIER_REFUSALS:
            self.checks_passed = False
            return None
        script, code, text = self.verifier_run()
        if not script:
            self.count_refusal()
            return VERIFIER_MISSING_VERDICT.format(path=VERIFIER_PATH)
        if code == 0:
            return None
        self.count_refusal()
        if code is None:
            verdict = VERIFIER_TIMEOUT_VERDICT.format(path=VERIFIER_PATH,
                                                      seconds=self.verifier_timeout)
        else:
            verdict = VERIFIER_FAILED_VERDICT.format(path=VERIFIER_PATH, code=code,
                                                     output=text[-3000:])
        return verdict + self.reused_note("verifier", VERIFIER_PATH)

    def verifier_run(self) -> tuple[str, int | None, str]:
        """Run the agent's own verifier and return ``(script, exit code, output)``.

        The script is read once and that text is run (:meth:`script_run`,
        or its last run reused), so what the reviewer sees is what ran
        (:attr:`last_verifier_run`).  A missing script is ``("", None,
        "")``; a run past ``verifier_timeout`` has exit code ``None``.
        """
        script = self.read_container_file(VERIFIER_PATH)
        if script is None:
            self._log({"event": "verifier", "turn": self.turns, "result": "missing"})
            self.record_verdict("missing", "gate", "")
            return ("", None, "")
        self.last_verifier_run = self.script_run("verifier", VERIFIER_PATH, script)
        return self.last_verifier_run

    def script_run(self, stage: str, path: str, script: str) -> tuple[str, int | None, str]:
        """Run *script*, the *stage*'s check kept at *path*, or reuse the stage's last run.

        The last run stands while neither the files under the workdir
        (:meth:`file_snapshot`, as they were when that run started: the
        state its verdict was computed on) nor the script's text changed
        since it (:attr:`stage_memo`): a round in which nothing changed
        re-runs nothing, whatever the exit code was.  A script that changes
        the tree itself leaves a different snapshot behind, so its next
        round runs it again.  No reuse without file tracking.  The run (or
        its reuse, ``reused: True``) is logged as the
        stage's event (``verifier`` / ``blind_verifier``) and marked on its
        ``gate_stage`` record (:attr:`stage_fields`); the verifier's verdict
        is remembered for the hand-off (:meth:`record_verdict`).

        Args:
            stage: ``verifier`` or ``blind``.
            path: Where the agent sees the script.
            script: The script's text.

        Returns:
            ``(script, exit code, output)``; the exit code is ``None`` past
            ``verifier_timeout``.
        """
        event = "verifier" if stage == "verifier" else "blind_verifier"
        snapshot = self.file_snapshot()
        memo = self.stage_memo.get(stage)
        if memo is not None and snapshot is not None and memo[0] == (snapshot, script):
            _script, code, text = memo[1]
            self._log({"event": event, "turn": self.turns, "exit_code": code, "reused": True})
            self.stage_fields[stage] = {"reused": True}
            return (script, code, text)
        result = self.run_script(script, path, self.verifier_timeout)
        code, output = (None, b"") if result is None else result
        text = output.decode("utf-8", errors="replace")
        if snapshot is not None:
            self.stage_memo[stage] = ((snapshot, script), (script, code, text))
        self._log({"event": event, "turn": self.turns, "exit_code": code,
                   "output": text[-4000:]})
        if stage == "verifier":
            self.record_verdict("timeout" if code is None else str(code), "gate", text)
        return (script, code, text)

    def reused_note(self, stage: str, path: str) -> str:
        """The note a reused failing verdict of *stage* ends with (``""`` when the check ran)."""
        if not self.stage_fields.get(stage, {}).get("reused"):
            return ""
        return REUSED_NOTE.format(path=path, workdir=self.workdir)

    def record_verdict(self, code: str, who: str, output: str) -> None:
        """Remember the verifier's latest verdict for the hand-off and the restart budget.

        Args:
            code: The exit code (``0``, ``1``, ...), ``timeout`` or ``missing``.
            who: ``gate`` or ``agent``.
            output: The run's output (its head is kept).
        """
        with self._lock:
            self.last_verdict = (code, time.time(), self.turns, who,
                                 "\n".join(output.splitlines()[:VERDICT_LINES]))

    def count_refusal(self) -> None:
        """Count a verifier or blind-check refusal (the stages run on threads of their own)."""
        with self._lock:
            self.verifier_refusals += 1

    def review_call(self, label: str, model: str, prompt: str, event: str = "review",
                    lowered: bool = False) -> tuple[str, str, float, int, dict[str, Any]]:
        """One non-agentic call of *model* at the provider's default effort, with a deadline.

        The call runs on the calling thread under a stop event of its own
        (:func:`~kiss.agents.sorcar.coding_gate.call_stop_event`) that fires
        after :data:`REVIEW_DEADLINE` seconds or when the run is stopped;
        the model clients watch the thread's stop event and end the
        streaming call on it, so the call cannot outlive the gate's budget
        of time, and its spend so far is still counted: a call the run's
        Stop ends logs its spend itself as a failed *event* before the Stop
        propagates (the caller never gets to log it).  The trial's model
        overrides apply without their effort
        (:func:`~kiss.agents.sorcar.coding_gate.default_effort`), or at
        low effort for the retry of a failed attempt (*lowered*,
        :func:`~kiss.agents.sorcar.coding_gate.low_effort`).

        Args:
            label: The agent's name in the logs.
            model: The model to call.
            prompt: The whole prompt.
            event: The side-agent event the caller logs the call under
                (:data:`SIDE_AGENT_EVENTS`), for the spend of a stopped call.
            lowered: Whether this is the retry of a failed attempt, at low effort.

        Returns:
            ``(answer, error, cost, tokens, last_call)``: the answer (``""``
            for none), the error (``""`` when the call completed; the deadline
            when it was hit), the call's cost in USD, its tokens, and the
            usage of the last call.
        """
        from kiss.core import stop_signal
        from kiss.core.kiss_agent import KISSAgent

        agent = KISSAgent(label)
        config = (low_effort(self.model_overrides, model) if lowered
                  else default_effort(self.model_overrides))
        previous = stop_signal.get_thread_stop_event()
        done = threading.Event()
        deadline = call_stop_event(REVIEW_DEADLINE, done)
        stop_signal.set_thread_stop_event(deadline)
        answer, error = "", ""
        try:
            answer = str(agent.run(model_name=model, prompt_template="{p}", arguments={"p": prompt},
                                   is_agentic=False, verbose=False, model_config=config) or "")
        except Exception as failure:
            error = repr(failure)
        except BaseException:
            if not deadline.is_set() or (previous is not None and previous.is_set()):
                # a Stop of the run (or a real interrupt), not the deadline
                self._log({"event": event, "turn": self.turns, "model": model, "error": "stopped",
                           "failed": True, "cost_usd": agent.budget_used,
                           "tokens": agent.total_tokens_used})
                raise
            error = f"no answer within {REVIEW_DEADLINE} s"
        finally:
            done.set()
            stop_signal.set_thread_stop_event(previous)
        return (answer, error, agent.budget_used, agent.total_tokens_used,
                dict(agent.last_call_usage or {}))

    def review_verdict(self, args: dict[str, Any]) -> str | None:
        """A read-only review of the work by the ``reviewer`` model; its gaps as a verdict.

        The reviewer sees the task statement, the agent's summary and
        ledger, the scripts and their last output, the commands named as
        evidence with their exit codes, and the files created or changed
        under the workdir (:meth:`changed_files_report`); a re-review (when
        ``max_reviews`` allows one) also sees its previous findings and only
        the files changed since.  It runs no commands.  Its spend is logged
        as a ``review`` event, which the trial runner adds to the trial's
        cost.

        The review is one model call at the provider's default effort (the
        trial's own effort override is not applied: at the highest effort
        the reviewer thought for a median 399 s, past the prompt cache's
        life, and its verdicts were no better) with :data:`REVIEW_DEADLINE`
        seconds to answer (:meth:`review_call`).  An attempt that answered
        nothing, was cut short by the output cap, was refused or missed the
        deadline (:func:`~kiss.agents.sorcar.coding_gate.retryable_review_failure`)
        is retried once at low effort.  A review that still failed, or
        answered without a ``VERDICT:`` line, is logged as failed (with the
        first attempt's failure under ``retried``) and refuses nothing: it
        keeps the previous findings open, does not use up a review, and
        counts under :data:`MAX_FAILED_REVIEWS`, past which no review is
        asked for again.  Each ``NOT DONE`` finding is kept
        (:attr:`open_findings`) for the next finish to answer, and the
        verdict with the claim it judged for :meth:`reused_review`.

        Args:
            args: Arguments of the ``finish`` call.
        """

        _script, code, output = self.last_verifier_run
        script = self.read_container_file(VERIFIER_PATH) or ""
        _blind, blind_code, blind_output = self.last_blind_run
        blind_script = self.blind_script_ready()
        previous, scope = "", ""
        if self.open_findings:
            previous = REVIEW_PREVIOUS.format(findings="\n".join(self.open_findings))
            scope = " since your previous review"
        prompt = REVIEW_PROMPT.format(
            task=self.task_prompt, summary=str(args.get("summary_in_html", "")),
            ledger=LEDGER_PATH, ledger_text=(self.read_container_file(LEDGER_PATH) or "(none)")[
                :REVIEW_ONE_FILE_BYTES],
            verifier=VERIFIER_PATH, code=code, script=script[:REVIEW_ONE_FILE_BYTES],
            output=output[-4000:] or "(not run yet)", blind=BLIND_VERIFIER_PATH,
            blind_code=blind_code, blind_script=blind_script[:REVIEW_ONE_FILE_BYTES] or "(none)",
            blind_output=blind_output[-4000:] or "(not run yet)",
            evidence=self.evidence_report(args),
            previous=previous, scope=scope, workdir=self.workdir,
            files=self.changed_files_report(self.review_snapshot if self.open_findings else None),
        )
        answer, error, cost, tokens, last_call = self.review_call("Finish reviewer",
                                                                  self.reviewer, prompt)
        retried = ""
        if retryable_review_failure(answer, error):
            retried = error or "empty answer"
            answer, error, more_cost, more_tokens, last_call = self.review_call(
                "Finish reviewer", self.reviewer, prompt, lowered=True)
            cost, tokens = cost + more_cost, tokens + more_tokens
        findings = review_findings(answer)
        failed = bool(error or not REVIEW_VERDICT_LINE.search(answer))
        if failed:
            self.failed_reviews += 1
        else:
            self.reviews += 1
        self._log({"event": "review", "turn": self.turns, "model": self.reviewer,
                   "round": self.reviews, "answer": answer[:8000], "findings": findings,
                   "error": error, "failed": failed, "last_call": last_call,
                   "cost_usd": cost, "tokens": tokens, **({"retried": retried} if retried
                                                            else {})})
        self.stage_fields["review"] = {"model": self.reviewer, "cost_usd": cost,
                                       "round": self.reviews}
        if retried:
            self.stage_fields["review"]["retried"] = retried[:500]
        if failed:
            self.stage_fields["review"]["error"] = (
                error or ("empty answer" if not answer.strip() else "no verdict"))[:500]
            return None
        self.review_snapshot = self.file_snapshot()
        self.open_findings = numbered_items(findings)
        self.review_claim = claim_text(args)
        self.review_result = REVIEW_VERDICT.format(findings=findings) if findings else None
        return self.review_result

    def reused_review(self, args: dict[str, Any]) -> bool:
        """Whether the last review's verdict stands for this finish, and record the reuse.

        It does when the finish makes the claim that review judged
        (:func:`claim_text`: the summary and the evidence named) on the
        files it saw (:attr:`review_snapshot`): the same input gets the
        same verdict (:attr:`review_result`) without a model call, and
        the stage is recorded as reused.  A changed claim on the same
        files is judged on the review's findings by
        :meth:`unanswered_verdict` instead; a changed tree gets a review
        when one is still possible (:meth:`review_wanted`).

        Args:
            args: Arguments of the ``finish`` call.
        """
        if self.review_claim is None or self.review_claim != claim_text(args):
            return False
        current = self.file_snapshot()
        if (self.review_snapshot is None or current is None
                or changed_paths(self.review_snapshot, current)):
            return False
        self.gate_stage("review", 0.0, "fail" if self.review_result else "pass", reused=True,
                        model=self.reviewer, round=self.reviews)
        return True

    def failed_review(self, error: Exception) -> None:
        """The verdict of a review stage that raised: none, the error on its stage record.

        A check that did not happen is never a refusal against the agent
        (the verifier and the independent check differ: :func:`failed_stage`).

        Args:
            error: What the stage raised.
        """
        self.stage_fields["review"] = {"model": self.reviewer, "error": repr(error)[:500]}
        return None

    def unanswered_verdict(self, args: dict[str, Any]) -> str | None:
        """The open review findings this finish does not answer by quoted evidence.

        Judged at a finish no review verdict came back for (none was
        wanted, or the one asked for failed): a finding is answered when
        the decisions model finds the finish summary quoting a command and
        its output for it (:func:`answered_findings`).  Without the
        decisions model every finding counts as answered.

        Args:
            args: Arguments of the ``finish`` call.
        """
        if not self.open_findings:
            return None
        summary = str(args.get("summary_in_html", ""))
        spent: list[tuple[float, int]] = []
        answered = answered_findings(summary, self.open_findings,
                                     lambda cost, tokens: spent.append((cost, tokens)))
        # A side-agent record (SIDE_AGENT_EVENTS): the decisions call's spend is
        # outside the daemon's accounting, like the reviewer's.
        self._log({"event": "findings_answered", "turn": self.turns, "answered": answered,
                   "cost_usd": sum(c for c, _ in spent), "tokens": sum(t for _, t in spent)})
        if answered is None:
            return None
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
        script, code, text = self.blind_run()
        if not script or code == 0:
            return None
        self.count_refusal()
        shown = text[-BLIND_OUTPUT_CHARS:] if code is not None else (
            f"(timed out after {self.verifier_timeout:.0f} seconds)")
        return BLIND_VERIFIER_FAILED_VERDICT.format(
            path=BLIND_VERIFIER_PATH, code=code, output=shown,
        ) + self.reused_note("blind", BLIND_VERIFIER_PATH)

    def blind_run(self) -> tuple[str, int | None, str]:
        """Run the independent check from the host's copy; ``(script, exit code, output)``.

        Waits for the check's author when it is still writing
        (:meth:`blind_script_ready`).  ``("", None, "")`` when there is no
        check; a run past ``verifier_timeout`` has exit code ``None``; the
        last run is reused while nothing changed (:meth:`script_run`).
        """
        script = self.blind_script_ready() if self.blind_verifier else ""
        if not script:
            return ("", None, "")
        self.write_container_file(BLIND_VERIFIER_PATH, script)
        self.last_blind_run = self.script_run("blind", BLIND_VERIFIER_PATH, script)
        return self.last_blind_run

    def start_blind_author(self) -> None:
        """Start the independent check's author on a thread of its own, once per trial.

        Called at the first model call, so the check is written while the
        engineer works instead of inside the first gated finish, which
        waited for it (:meth:`blind_script_ready`).  The author's records
        reach the trial's event log through the printer bound to the
        calling thread (:mod:`kiss.core.event_sink`).
        """
        if not self.blind_verifier or self.blind_written:
            return
        self.blind_written = True
        self.blind_thread = threading.Thread(
            target=self.author_blind_verifier, args=(thread_context(),),
            name="blind-author", daemon=True)
        self.blind_thread.start()

    def author_blind_verifier(self, context: Any) -> None:
        """Write the check (the author thread's body) and record the stage, with its retry.

        Args:
            context: The calling thread's printer and task binding
                (:func:`kiss.core.event_sink.thread_context`), so the stage's
                record is filed under the trial's task like the gate's own.
        """
        adopt_thread_context(context)
        started = time.time()
        try:
            self.write_blind_verifier()
        except Exception as error:
            self._log({"event": "hook_error", "turn": self.turns, "hook": "blind_author",
                       "error": repr(error)})
        self.gate_stage("blind_author", time.time() - started,
                        "pass" if self.blind_script else "missing", retried=self.blind_retried,
                        **self.stage_fields.pop("blind_author", {}))

    def blind_script_ready(self) -> str:
        """The independent check's script once its author has delivered it; ``""`` for none.

        Starts the author when it has not run yet (a finish before any model
        call) and waits for its thread.
        """
        self.start_blind_author()
        if self.blind_thread is not None:
            self.blind_thread.join()
        return self.blind_script

    def write_blind_verifier(self) -> None:
        """Have the ``blind_verifier`` model write the independent check into the container.

        The author's agent gets a shell in the container and a
        ``finish(script)`` tool, :data:`BLIND_VERIFIER_STEPS` steps and
        :data:`BLIND_VERIFIER_BUDGET_FRACTION` of the trial's budget; its
        spend is logged as a ``blind_verifier_written`` event (added to the
        trial's cost by the runner like a review's).  It sees the statement,
        not the engineer's ledger, verifier or summary.  An author whose run
        ends without delivering (it answered in text, or spent its steps or
        budget) is retried once (:attr:`blind_retried`) on a model of
        another family, or on the configured author model again when this
        daemon pins every task to its model
        (:func:`~kiss.agents.sorcar.agent_dispatch.task_model_pinned`: the
        retry then switches to no model the configuration did not name);
        a run that fails with an error (no such model, no credential, a
        provider down) is not retried.  The model that wrote the check and
        the authors' total cost go to the stage's record
        (:attr:`stage_fields`).
        """
        from kiss.agents.sorcar.agent_dispatch import task_model_pinned
        from kiss.core.models.model_families import other_family_model

        model = self.blind_verifier
        script, retry, cost = self.run_blind_author(model)
        if retry:
            again = model if task_model_pinned() else other_family_model(model)
            if again is not None:
                self.blind_retried = True
                model = again
                script, _retry, retry_cost = self.run_blind_author(again, retry=True)
                cost += retry_cost
        self.blind_script = script
        self.stage_fields["blind_author"] = {"model": model, "cost_usd": cost}

    def run_blind_author(self, model: str, retry: bool = False) -> tuple[str, bool, float]:
        """Run the check's author once on *model*; return ``(script, retry, cost)``.

        Args:
            model: The author's model.
            retry: Whether this is the second run, after one that delivered
                nothing (recorded as such in the ``blind_verifier_written`` event).

        Returns:
            The delivered script (``""`` for none); whether the run delivered
            nothing by its own doing (it answered in text, or ran out of
            steps or budget), which a second run may do better; and the
            run's cost in USD.  A run that failed with an error delivers
            nothing either, but another model is not the answer to that.
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
        error, exhausted = "", False
        try:
            agent.run(model_name=model, prompt_template="{p}",
                      arguments={"p": BLIND_VERIFIER_PROMPT.format(
                          task=self.task_prompt, workdir=self.workdir,
                          timeout=int(self.verifier_timeout * BLIND_VERIFIER_TIME_FRACTION))},
                      tools=[Bash, finish], max_steps=BLIND_VERIFIER_STEPS,
                      max_budget=self.budget * BLIND_VERIFIER_BUDGET_FRACTION, verbose=False,
                      model_config=self.model_overrides or None)
        except Exception as failure:
            # the step and budget caps end the run with a typed error of the
            # agent's own; the step count cannot tell the cap from a model that
            # failed at the last step (the count is advanced before the step runs)
            error = repr(failure)
            exhausted = isinstance(failure, BudgetExceededError | StepLimitExceededError)
        script = written[-1] if written else ""
        self._log({"event": "blind_verifier_written", "turn": self.turns,
                   "model": model, "retry": retry,
                   "script": script[:8000], "error": error, "exhausted": exhausted,
                   "steps": agent.step_count, "last_call": dict(agent.last_call_usage or {}),
                   "cost_usd": agent.budget_used, "tokens": agent.total_tokens_used})
        return (script, not script and (not error or exhausted), agent.budget_used)

    def track_change(self, name: str, args: dict[str, Any]) -> None:
        """Count a tool call that changes the work, for the verifier reuse of the gate.

        An ``Edit`` / ``Write`` under the workdir and a shell command that
        writes files (:func:`~kiss.agents.sorcar.evidence_gate.writes_files`;
        a write under ``/tmp``, the verifier or the ledger, is not one)
        advance :attr:`changes`.  A call that may rewrite the verifier
        itself (an edit of it, a shell command naming it) makes the
        agent's last run of it stale (:attr:`verifier_passed_at`) and
        advances the count too, so a result of that run read later in the
        turn (:meth:`record_exit_codes`) cannot stand for the new script.

        Args:
            name: The tool called (``Edit``, ``Write``, ``Bash`` or
                ``run_commands_parallel``).
            args: Its arguments.
        """
        if name in ("Edit", "Write"):
            path = args.get("file_path")
            if isinstance(path, str) and posixpath.normpath(
                    posixpath.join(self.workdir, path)) == VERIFIER_PATH:
                self.verifier_passed_at = None
                self.changes += 1
            elif self.under_workdir(args):
                self.changes += 1
            return
        if posixpath.basename(VERIFIER_PATH) in json.dumps(args, default=str):
            self.verifier_passed_at = None
            self.changes += 1
        elif writes_files(args):
            self.changes += 1

    def under_workdir(self, args: dict[str, Any]) -> bool:
        """Whether the ``file_path`` of an ``Edit`` / ``Write`` call is under the workdir.

        Args:
            args: Arguments of the call.
        """
        path = args.get("file_path")
        if not isinstance(path, str):
            return False
        full = posixpath.normpath(posixpath.join(self.workdir, path))
        root = self.workdir.rstrip("/")
        return full == root or full.startswith(root + "/")

    def ledger_verdict(self, args: dict[str, Any]) -> str | None:
        """Refuse an edit under the workdir until the interpretation ledger exists and is read.

        At most :data:`MAX_LEDGER_REFUSALS` refusals for a missing ledger
        per trial, then the independent reading of it
        (:meth:`reading_verdict`).  Edits outside the workdir (the ledger
        and the verifier themselves, under ``/tmp``) are never refused.

        Args:
            args: Arguments of the ``Edit`` or ``Write`` call.
        """
        if not self.under_workdir(args):
            return None
        if not self.ledger_seen and self.ledger_refusals < MAX_LEDGER_REFUSALS:
            if self.container_exec(["test", "-s", LEDGER_PATH]) is None:
                self.ledger_refusals += 1
                return LEDGER_VERDICT.format(workdir=self.workdir, ledger=LEDGER_PATH)
            self.ledger_seen = True
        if self.reading_check and (self.reading_ledger is None or self.reading_items):
            return self.reading_verdict()
        return None

    def read_statement(self, ledger: str) -> list[str]:
        """The independent reading of the statement against *ledger*.

        Once per trial: the ``reading_check`` model (of another family
        than the trial's) reads the statement and the ledger
        (:data:`READING_CHECK_PROMPT`, one non-agentic call,
        :meth:`review_call`).  ``NONE``, a failed call or an answer
        without items give no item.  The call's spend is logged as a
        ``reading_check`` event (:data:`SIDE_AGENT_EVENTS`) and recorded
        as the ``reading`` gate stage; the items and the ledger they were
        raised against are kept (:attr:`reading_items`,
        :attr:`reading_ledger`) for :meth:`reading_verdict`.

        Args:
            ledger: The ledger's text.
        """
        self.reading_ledger = ledger
        started = time.time()
        answer, error, cost, tokens, last_call = self.review_call(
            "Reading check", self.reading_check,
            READING_CHECK_PROMPT.format(task=self.task_prompt,
                                        ledger=ledger[:REVIEW_ONE_FILE_BYTES]),
            event="reading_check")
        if not error and not READING_NONE.match(answer):
            self.reading_items = numbered_items(answer)
        self._log({"event": "reading_check", "turn": self.turns, "model": self.reading_check,
                   "answer": answer[:8000], "items": self.reading_items, "error": error,
                   "last_call": last_call, "cost_usd": cost, "tokens": tokens})
        self.gate_stage("reading", time.time() - started,
                        "error" if error else ("fail" if self.reading_items else "pass"),
                        model=self.reading_check, cost_usd=cost, items=len(self.reading_items))
        return self.reading_items

    def reading_note(self, name: str, args: dict[str, Any]) -> str | None:
        """The independent reading to append to the result of the ledger's first ``Write``.

        The agent sees the items as soon as the ledger exists, whatever it
        edits next (a deliverable written by a script of its own under
        ``/tmp`` passes through no edit guard); :data:`READING_NOTE_RESULT`
        asks it to settle them in the ledger before the first edit, which
        :meth:`reading_verdict` then holds it to.

        Args:
            name: The tool called.
            args: Its arguments.

        Returns:
            The note, or ``None`` when the call is not the ledger's first
            ``Write``, no reading model is set, or the reading agrees.
        """
        path = args.get("file_path")
        if (name != "Write" or not self.reading_check or self.reading_ledger is not None
                or not isinstance(path, str)
                or posixpath.normpath(posixpath.join(self.workdir, path)) != LEDGER_PATH):
            return None
        items = self.read_statement(str(args.get("content", "")))
        if not items:
            return None
        return READING_NOTE_RESULT.format(items="\n".join(items), workdir=self.workdir,
                                          ledger=LEDGER_PATH)

    def reading_verdict(self) -> str | None:
        """Refuse the first edit until the ledger settles the independent reading of the statement.

        The reading is made when the ledger is first written
        (:meth:`reading_note`) or, when it was written some other way, at
        the first edit under the workdir (:meth:`read_statement` on the
        ledger read from the container).  Its items refuse the edit
        (:data:`READING_VERDICT`) until the ledger changed since they were
        raised, at most :data:`MAX_READING_REFUSALS` times (then they are
        dropped).
        """
        if self.reading_refusals >= MAX_READING_REFUSALS:
            self.reading_items = []
            return None
        ledger = self.read_container_file(LEDGER_PATH) or ""
        if self.reading_ledger is None:
            self.read_statement(ledger)
        elif ledger.strip() != self.reading_ledger.strip():
            self.reading_items = []  # the ledger answered (or at least changed)
        if not self.reading_items:
            return None
        self.reading_refusals += 1
        return READING_VERDICT.format(items="\n".join(self.reading_items),
                                      workdir=self.workdir, ledger=LEDGER_PATH)

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
        self.changes += 1
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
        wd, tar = shlex.quote(self.workdir), shlex.quote(archive)
        command = (f"mkdir -p {shlex.quote(CHECKPOINT_DIR)} && mb=$(du -sm {wd} | cut -f1)"
                   f" && [ \"$mb\" -le {MAX_CHECKPOINT_MB} ] || {{ echo \"too large: ${{mb}} MB\"; "
                   f"exit 3; }}; tar -C {wd} -cf {tar}.tmp . && "
                   f"mv {tar}.tmp {tar}")
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
                       f"tar -C {wd} -xf {shlex.quote(archive)}")
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
        """Pair the turn's shell commands with the exit codes their results show.

        The turn's results come in the order of its tool calls, so they are
        paired with every call of the turn; the ``Bash``, ``bash_job`` and
        ``run_commands_parallel`` results are read (:func:`result_exit_codes`).
        A ``Bash`` command whose exit status 0 was its run of the agent's
        verifier (:func:`verifier_decides`), that writes no file itself and
        that no edit, shell command or restore follows in the turn (the
        results are read after the whole turn, so a later call could have
        changed what the run checked) sets :attr:`verifier_passed_at` to the
        change count of its call and :attr:`verifier_passed_snapshot` to the
        files as they are now: the gate reuses that run while neither
        changes (:meth:`verifier_verdict`).  A run of the verifier that
        ends as a background job (its ``bash_job`` report shows the exit)
        records its verdict too (:attr:`last_verdict`, for the hand-off and
        the restart budget), but is never reused by the gate: the tree may
        have changed while it ran.

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
        last_change = max((at for at, (name, _args, _changes) in enumerate(calls)
                           if name not in READ_ONLY_TOOLS), default=-1)
        for at, ((name, args, changes), text) in enumerate(zip(calls, results, strict=True)):
            with self._lock:
                for command, code in result_exit_codes(
                        name, args, text, self.background_jobs, container=True):
                    self.commands_run.append((command, code))
                    del self.commands_run[:-INVENTORY_COMMANDS]
                    self.exit_codes[command.strip()] = code
                    decides = (name in ("Bash", "bash_job")
                               and not writes_files({"command": command})
                               and verifier_decides(command, self.workdir, VERIFIER_PATH))
                    if decides and code != "background":
                        # the output before the tool's exit-code marker (the usage
                        # footer follows it)
                        markers = list(EXIT_CODE_MARKER.finditer(text))
                        output = text[:markers[-1].start()] if markers else text
                        self.last_verdict = (code, time.time(), self.turns, "agent",
                                             "\n".join(output.splitlines()[:VERDICT_LINES]))
                    if decides and name == "Bash" and code == "0" and at == last_change:
                        self.verifier_passed_at = changes
                        self.verifier_passed_snapshot = self.file_snapshot()

    def inventory(self) -> str:
        """The state block appended to a continuation session's first message, in a fixed form.

        Four sections, in this order: the files changed since the run
        started (:meth:`changed_files_lines`), the verifier's last verdict
        (:meth:`verdict_lines`), the ledger's open checklist
        (:meth:`checklist_lines`) and the running background jobs
        (:meth:`running_jobs_lines`).  The statement is already in the
        message; what the last session thought is not: the new session
        reads the state, not a narrative.
        """
        lines = ["", "# Inventory (written by the harness at this restart)"]
        for title, section in (("Files changed since the start", self.changed_files_lines),
                               ("Verifier's last verdict", self.verdict_lines),
                               ("Ledger open checklist", self.checklist_lines),
                               ("Running background jobs", self.running_jobs_lines)):
            lines.append(f"## {title}")
            lines += ["  " + line for line in section()]
        return "\n".join(lines)

    def changed_files_lines(self) -> list[str]:
        """The files under the workdir created, modified or deleted since the run started.

        A created file with its line count, a modified one with its size
        before and after (the original is not kept, so line counts of a
        diff are not cheap), a deleted one as such; at most
        :data:`INVENTORY_FILES`.  ``(no file tracking)`` without a baseline.
        """
        baseline, current = self.file_baseline, self.file_snapshot()
        if baseline is None or current is None:
            return ["(no file tracking)"]
        changed = changed_paths(baseline, current)
        if not changed:
            return ["(none)"]
        shown = changed[:INVENTORY_FILES]
        created = [path for path in shown if path in current and path not in baseline]
        counts: dict[str, str] = {}
        if created:
            out = self.container_exec(["sh", "-c", "wc -l \"$@\"", "sh", *created]) or ""
            for line in out.splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[1] in current:
                    counts[parts[1]] = parts[0]
        lines = []
        for path in shown:
            if path not in current:
                lines.append(f"{path} (deleted)")
            elif path not in baseline:
                lines.append(f"{path} (created, {counts.get(path, '?')} lines)")
            else:
                lines.append(f"{path} (modified, {baseline[path][0]} -> {current[path][0]} bytes)")
        if len(changed) > len(shown):
            lines.append(f"and {len(changed) - len(shown)} more")
        return lines

    def verdict_lines(self) -> list[str]:
        """The verifier's last verdict: exit code, who ran it and when, the head of its output."""
        with self._lock:
            verdict = self.last_verdict
        if verdict is None:
            return [f"(the verifier {VERIFIER_PATH} has not run)"]
        code, at, turn, who, head = verdict
        when = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(at))
        lines = [f"exit {code}, run by the {who} at {when} (turn {turn}); output head:"]
        return lines + ["  " + line for line in head.splitlines()] if head else lines

    def checklist_lines(self) -> list[str]:
        """The ledger's unchecked ``- [ ]`` items (``(no ledger)`` without the file)."""
        ledger = self.read_container_file(LEDGER_PATH)
        if ledger is None:
            return ["(no ledger)"]
        items = [line.strip() for line in ledger.splitlines() if OPEN_ITEM.match(line)]
        return items[:INVENTORY_FILES] or ["(none open)"]

    def running_jobs_lines(self) -> list[str]:
        """The background jobs still running, as ``<id>: <command>`` (from the jobs directory)."""
        out = self.container_exec(["sh", "-c", RUNNING_JOBS_COMMAND]) or ""
        return out.splitlines()[:INVENTORY_FILES] or ["(none)"]

    def restart_denied(self) -> bool:
        """Count a hand-off; whether the restart budget denies it a session.

        It does from the :attr:`restart_budget`-th restart on, when the
        hand-off comes with no new edit (:attr:`changes` as at the previous
        restart) and no change of the verifier's verdict since it
        (:attr:`last_verdict`): the run ends with the state as the caveat
        (:meth:`restart_caveat`), and the trajectory records
        ``restart_budget``.
        """
        with self._lock:
            verdict = self.last_verdict[0] if self.last_verdict else None
            idle = self.changes == self.restart_changes and verdict == self.restart_verdict
            self.restarts += 1
            self.restart_changes, self.restart_verdict = self.changes, verdict
            restarts = self.restarts
        denied = idle and restarts >= self.restart_budget
        self._log({"event": "restart", "turn": self.turns, "restarts": restarts, "idle": idle,
                   "changes": self.changes, "verdict": verdict})
        if denied:
            self._log({"event": "restart_budget", "turn": self.turns, "restarts": restarts,
                       "changes": self.changes, "verdict": verdict})
        return denied

    def restart_caveat(self, state: str) -> str:
        """The caveat of a run the restart budget ended: why, and the state block *state*."""
        return RESTART_BUDGET_CAVEAT.format(restarts=self.restarts, state=html.escape(state))

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

        The commands are read as the tool decodes them
        (:func:`~kiss.agents.sorcar.evidence_gate.shell_commands`), so an
        array only its lenient JSON decoder accepts is guarded command by
        command too.  A here-document creating a file under the workdir is
        an edit and meets the ledger gate (:meth:`ledger_verdict`).  The timeout is
        raised in place (``args`` is the dict the tool is called with) so a
        package install or compile is never killed half way by ``Bash``'s
        30-second default; a timeout already at or above
        :data:`INSTALL_TIMEOUT_SECONDS` (including ``run_commands_parallel``'s
        1800-second default) is left alone.

        Args:
            args: Arguments of a ``Bash`` or ``run_commands_parallel`` call.

        Returns:
            ``None`` (the call may run), :data:`DESTRUCTIVE_VERDICT` for a
            blocked command or the ledger verdict for a heredoc edit.
        """
        commands = shell_commands(args)
        if isinstance(args.get("command"), str):
            default_timeout = float(DEFAULT_SHELL_TIMEOUT)
            args.setdefault("timeout_seconds", DEFAULT_SHELL_TIMEOUT)
        else:
            default_timeout = 1800.0
        if any(self.destructive.search(c) for c in commands):
            return DESTRUCTIVE_VERDICT
        if self.finish_gate:  # a heredoc creating a file is an edit: same ledger gate
            for target in (t for c in commands for t in heredoc_targets(c)):
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


#: Entries of the restart inventory.
INVENTORY_COMMANDS = 40
#: Tool calls that change no file under the workdir (``refused`` is the
#: record of a call the hook refused): a verifier run followed in its turn
#: only by these is still what the gate would see (:meth:`ContainerHarness.record_exit_codes`).
READ_ONLY_TOOLS = frozenset({"Read", "checkpoint", "finish", "refused"})
INVENTORY_FILES = 60
#: Lines of the verifier's output the hand-off inventory keeps.
VERDICT_LINES = 20
#: An unchecked item of the ledger's checklist.
OPEN_ITEM = re.compile(r"^\s*[-*]\s+\[ \]\s")
#: Prints ``<id>: <command>`` for every background job started and not yet
#: exited (the job files :mod:`kiss.agents.sorcar.container_jobs` keeps).
RUNNING_JOBS_COMMAND = (
    f"for f in {JOBS_DIR}/*.pid; do [ -e \"$f\" ] || continue; id=$(basename \"$f\" .pid); "
    f"[ -e {JOBS_DIR}/$id.exit ] && continue; printf '%s: ' \"$id\"; "
    f"head -c 200 {JOBS_DIR}/$id.cmd 2>/dev/null; echo; done; true"
)
#: Hand-offs (context restarts) a run gets before an idle one ends it: from this
#: restart on, a hand-off with no new edit and the same verifier verdict as at
#: the previous restart gets no session (config ``restart_budget``,
#: :meth:`ContainerHarness.restart_denied`).
RESTART_BUDGET = 4
#: The caveat of a run the restart budget ended, appended to its summary.
RESTART_BUDGET_CAVEAT = (
    "\n<h4>Ended by the restart budget</h4>\n<p>Restart {restarts} came with no new edit and "
    "no change of the verifier's verdict since the restart before it; the run ends with the "
    "work as it is. Its state:</p>\n<pre>{state}</pre>"
)
#: First line of a continuation session's prompt (``CONTINUATION_PROMPT``).
CONTINUATION_MARKER = "# Task Progress (Continuation "
#: Side agents of the gate whose spend the trial runner adds to the trial's cost.
SIDE_AGENT_EVENTS = frozenset({"review", "blind_verifier_written", "reading_check",
                               "findings_answered"})


def changed_paths(baseline: dict[str, tuple[str, str]],
                  current: dict[str, tuple[str, str]]) -> list[str]:
    """The paths created, deleted or changed (size or mtime) between two file snapshots."""
    return sorted(set(current) ^ set(baseline)
                  | {p for p in current if p in baseline and current[p] != baseline[p]})


def claim_text(args: dict[str, Any]) -> str:
    """The claim a ``finish`` call makes, as the reviewer judges it: its summary and its evidence.

    Args:
        args: Arguments of the ``finish`` call.
    """
    return f"{args.get('summary_in_html', '')}\n{args.get('evidence', '')}"


class RestartBudgetError(KISSError):
    """Raised at the first model call of a session the restart budget denies.

    The session loop ends the run on it with ``finish(success=False)``
    whose summary is the message: the restart budget's caveat with the
    state block (:meth:`ContainerHarness.restart_caveat`).
    """


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

    The reviewer, the blind verifier's author, the reading check and the
    decisions model's findings check are model calls outside the daemon's
    accounting (:data:`SIDE_AGENT_EVENTS`), so the trial runner
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


#: Survivors named per note; the rest are counted (every one is adopted).
MAX_LISTED_SURVIVORS = 8


def describe_processes(
    procs: dict[tuple[int, str], tuple[int, str]], limit: int = MAX_LISTED_SURVIVORS,
) -> str:
    """``pid cmdline (parent ppid)`` for up to *limit* processes, lowest pid first."""
    items = [f"{pid} {cmd[:120]} (parent {ppid})"
             for (pid, _), (ppid, cmd) in sorted(procs.items())]
    if len(items) > limit:
        items = items[:limit] + [f"and {len(items) - limit} more"]
    return "; ".join(items)


def shell_segments(command: str) -> list[str]:
    """The simple commands of *command*, as written (quotes kept), in order.

    Split at the separators :data:`SHELL_PIECE` recognises outside quotes;
    blank pieces are dropped.

    Args:
        command: The shell command as the agent wrote it.
    """
    segments: list[str] = []
    current: list[str] = []
    for piece in SHELL_PIECE.finditer(command):
        if piece.group("sep") is None:
            current.append(piece.group())
        else:
            segments.append("".join(current))
            current = []
    segments.append("".join(current))
    return [segment.strip() for segment in segments if segment.strip()]


def shell_words(text: str) -> list[str]:
    """The words of the simple command *text* as the shell would pass them (quotes removed).

    Args:
        text: One simple command; unbalanced quoting falls back to a blank split.
    """
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def command_logs(command: str, workdir: str) -> list[tuple[str, str]]:
    """``(simple command, log path)`` for each part of *command* that redirects its output.

    The parts are the simple commands :func:`shell_segments` finds; a ``cd
    DIR`` part sets the directory the relative paths of the later parts
    resolve against, *workdir* to begin with.  The log is the target of
    the part's last stdout redirection (:data:`REDIRECTION`, the one the
    shell honours); ``/dev/null`` is no log.

    Args:
        command: The shell command as the agent wrote it.
        workdir: The directory the command starts in.
    """
    logs: list[tuple[str, str]] = []
    cwd = workdir
    for text in shell_segments(command):
        words = shell_words(text)
        if len(words) == 2 and words[0] == "cd":
            cwd = posixpath.normpath(posixpath.join(cwd, words[1]))
            continue
        path = stdout_target(text)
        if path and path != "/dev/null":
            logs.append((text, posixpath.normpath(posixpath.join(cwd, path))))
    return logs


def stdout_target(text: str) -> str:
    """The file the simple command *text* redirects its output to, or ``""``.

    The operator (:data:`REDIRECTION`) is looked for outside quotes and
    escapes, the target is the shell word after it (quotes and escapes
    decoded); the last redirection is the one the shell honours.

    Args:
        text: One simple command.
    """
    masked = SHELL_PIECE.sub(
        lambda piece: "x" * len(piece.group()) if piece.group()[0] in "'\"\\" else piece.group(),
        text)
    operators = list(REDIRECTION.finditer(masked))
    words = shell_words(text[operators[-1].end():]) if operators else []
    return words[0] if words else ""


def survivor_log(cmdline: str, logs: list[tuple[str, str]]) -> str:
    """The log path among *logs* (:func:`command_logs`) of the process running *cmdline*.

    The log of the first simple command whose words (as the shell passed
    them, :func:`shell_words`) or text contain the command line, blank
    runs collapsed; when none does, the only log when there is exactly
    one; else ``""``.

    Args:
        cmdline: The process's command line as ``/proc`` shows it.
        logs: The redirected simple commands of the turn and their logs.
    """
    wanted = " ".join(cmdline.split())
    for text, path in logs:
        if wanted and (wanted in " ".join(shell_words(text)) or wanted in " ".join(text.split())):
            return path
    return logs[0][1] if len(logs) == 1 else ""


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


def failed_stage(error: Exception) -> str:
    """The verdict of a gate stage that raised *error*: a refusal naming it."""
    return f"Not finished: a check of the gate could not run ({error!r})."


def failed_check(error: Exception) -> tuple[str, int | None, str]:
    """The run record of a give-up check that raised *error*: no script, no exit code, the error.

    :func:`check_state` tells it from a missing check by its text.
    """
    return ("", None, repr(error))


def check_state(script: str, code: int | None, text: str) -> str:
    """The state of a check from its run record ``(script, exit code, output)``.

    Args:
        script: The script that ran (``""`` when none ran).
        code: Its exit code (``None`` when it did not finish or did not run).
        text: Its output, or the error of a check that could not run
            (:func:`failed_check`).

    Returns:
        ``pass``, ``fail (exit N)``, ``timeout`` (the script ran out of
        time), ``error (...)`` (the check could not run: a failure of the
        gate, not of the work) or ``missing`` (no such check).
    """
    if code == 0:
        return "pass"
    if code is not None:
        return f"fail (exit {code})"
    if script:
        return "timeout"
    return f"error ({text[:300]})" if text else "missing"


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
