# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bestrouter agent — one frontier model does the work, a second one reviews it.

The routing protocol (:data:`SYSTEM_PROMPT`) fixes the models by name:
:data:`SIMPLE_MODEL` answers simple questions and queries (the tab starts on it),
:data:`WRITER_MODEL` writes the documents, papers, reports and results a human reads,
and the final summary, :data:`PRIMARY_MODEL` runs every other task, including software
development and analysis, and the judgement steps of a task that mostly dispatches
sub-agents (the triage of a failed test run, the decision on a sub-agent's failure), and
:data:`REVIEW_MODEL` is dispatched through a non-blocking ``run_agent`` for a read-only
review and debugging pass over the work of the primary and writer models, on at most 75%
of the task budget.

Three ways to run it::

    pick ``bestrouter`` in the model picker: every task of the tab runs
    through this SEA (see ``register_as_model()`` below)

    /bestrouter add a --json flag to the export command and cover it with tests

    run_agent(agent="src/kiss/agents/seas/bestrouter/bestrouter_sea.py", task="...")

The :class:`BestrouterSea` methods (``settings()``, ``system_prompt()``,
``register_as_model()``, ...) follow the SEA contract in :mod:`kiss.agents.seas.base.base_sea`.
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea

SIMPLE_MODEL = "claude-opus-5-5-medium"
"""The model a tab starts on; it answers simple questions and queries itself."""

PRIMARY_MODEL = "claude-fable-5-2-max"
"""The model that does every task but simple questions and queries."""

REVIEW_MODEL = "gpt-6-astra-xhigh"
"""The model that reviews and debugs the primary model's work."""

WRITER_MODEL = "openrouter/google/gemini-4-carbon"
"""The model that writes the documents, papers, reports and results a human reads."""

SYSTEM_PROMPT = f"""\
## Model routing protocol (bestrouter)

Use '{SIMPLE_MODEL}' for simple questions and queries.
Use '{WRITER_MODEL}' for writing any documents, papers, reports, results which
will be read by a human.  The writing must read like human-written  with
natural tone with no AI slop.
Use '{PRIMARY_MODEL}' model for all other tasks, including software development and
analysis. A task that mostly dispatches sub-agents may dispatch, wait and log on
'{SIMPLE_MODEL}', but its judgement steps run on '{PRIMARY_MODEL}': call
`set_model("{PRIMARY_MODEL}")` before you triage a failed test run, before you act on a
sub-agent result that reports a failure or a warning.  Call
`set_model("{WRITER_MODEL}")` before you write the final summary. Use
{REVIEW_MODEL} through `run_agent(model="{REVIEW_MODEL}", tool_profile="review",
wait="false")` for a thorough read-only review and debugging of the
work of '{PRIMARY_MODEL}' and '{WRITER_MODEL}',
keep working while it runs and collect it with `agent_job(job_id, "wait")`. Check
thoroughly whether the other model missed code or wiring or introduced bugs. Use at most
75% of the task budget in {REVIEW_MODEL}, leave the reviewer's `max_budget` empty (it
gets 1/8 of what the task has spent so far, $10 at least), and ask the model not to
invent new problems. A second round gets the first round's findings and the diff since
then, not the whole change. Use the model names literally without hallucinating new
model names.

- Reviewer sub-agents: answer each finding with a change or with quoted evidence from a
command you ran, never with an argument alone. Dispatch them with run_agent(task,
model=<reviewer model>, tool_profile="review", wait="false") instead of asking the
child to call set_model (that costs a whole step on the wrong model), keep working
while the review runs (impacted tests, the progress log, the lint run) and collect
it with agent_job(job_id, "wait") when you have nothing left to do but read it, or
with agent_job("", "wait_any") when several reviewers or fixers are out and the first
to return is the one to act on; use run_parallel only for several reviewers of one
artifact. Leave max_budget empty: a review's default budget is 1/8 of what the task
has spent so far, $10 at least. Word the task as: "Verify the listed changes:
<files/functions>. Report only
demonstrated issues with file:line evidence and a reproduction or reasoning for each.
Do not seek novel regressions elsewhere and do not invent problems; say so when a
change is correct." A second round after fixes gets the first round's findings and
only the diff since then ("Check that these findings are fixed: <findings>; the fixes
are in <files>"), not the whole change again. Dispatch a reviewer only when the task
changed code, tests or configuration: a task that produced no diff (a git pull/merge,
running a script, answering a question) gets no reviewer even when the user's template
asks for one — say so in the final summary instead.
- One line per finding: when you dispatch fixers or reviewers whose results go into a
final report, name in each child's task a file of its own under your scratch dir
(<scratch dir>/findings/<child>.jsonl) and have it append one JSON object per finding,
on one line: {{"file": "src/x.py", "line": 12, "kind": "bug", "status": "fixed",
"summary": "one sentence", "test": "path::test_name"}} (kind: bug, test, doc, lint or
open; status: fixed, open or rejected; line 0 and an empty test when there is none).
`uv run findings-report <scratch dir>/findings/*.jsonl -o <report.html>` builds the
report's table and counts from those lines and fails on a malformed one, naming it;
do not assemble them from the children's prose.
- After two failed reviews of one subsystem (two rounds in which a reviewer still
demonstrates a defect in the same group of files), stop patching it finding by finding.
Give the subsystem to one owner agent: its task lists every file involved as its own
(no other agent, and not you, edits them from then on), asks first for a design note
of at most a page at a path you name under your scratch dir (the invariant, the states,
what each file does for it) and then for the fix against that note. Later rounds go
back to that same agent, not to a new one: a finished sub-agent cannot be resumed,
so start it with wait="false", name the reviewer model in its task and have it run
the review rounds itself, fixing in its own context until a review passes. If its
context ends first, its successor starts from the note.


Working rules:
- `Read` a file region before its first `Edit`, and `Read` an existing file before a
  `Write` that replaces it; an unread file is refused either way. After an `Edit` is
  rejected as unread, `Read` the file before any further `Edit` of it.
- Before each `run_parallel` / `run_agent` wave, add up the children's `max_budget`s and
  compare with your remaining budget (the `Budget: $spent/$max` footer counts the
  children). Keep at least 15% for the final verification and report. If the wave does
  not fit, do not launch it: report what is done and finish.
- Write a file longer than about 150 lines in `Write` / `Edit` calls of at most 150 lines
  each, the first chunk on disk before you compose the next. Save gathered facts to
  `<scratch dir>/` files before composing.
- When the `Steps:` footer of a tool result ends in 9, make your next call `summary`;
  otherwise the 10th, 20th, ... call is refused.
"""
"""The routing protocol added to the system prompt of every run.

A module-level f-string so ``/rsi7d`` can tune it; routers derived from
this one (``orbestrouter``) take it through :func:`routing_prompt`.
"""


def routing_prompt(router: str, simple: str, primary: str, review: str, writer: str) -> str:
    """Return :data:`SYSTEM_PROMPT` for a derived router with its own worker and reviewer.

    The heading names *router* and every mention of :data:`SIMPLE_MODEL`,
    :data:`PRIMARY_MODEL`, :data:`REVIEW_MODEL` and :data:`WRITER_MODEL` becomes
    *simple*, *primary*, *review* and *writer*; the rules are shared word for word.

    Args:
        router: The SEA's command name, shown in the heading
            ``## Model routing protocol (<router>)``.
        simple: Catalog name of the model for simple questions and queries.
        primary: Catalog name of the model that does every other task.
        review: Catalog name of the model that reviews the primary model's work.
        writer: Catalog name of the model that writes the prose a human reads.
    """
    return (
        SYSTEM_PROMPT.replace("(bestrouter)", f"({router})", 1)
        .replace(SIMPLE_MODEL, simple)
        .replace(PRIMARY_MODEL, primary)
        .replace(REVIEW_MODEL, review)
        .replace(WRITER_MODEL, writer)
    )


class BestrouterSea(BaseSea):
    """The ``/bestrouter`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/bestrouter help``."""
        return (
            f"Answers simple questions on {SIMPLE_MODEL}, writes on {WRITER_MODEL}, "
            f"runs every other task on "
            f"{PRIMARY_MODEL} and has {REVIEW_MODEL} review and debug the "
            "result read-only through a non-blocking run_agent on at most 75% of the task "
            "budget; pick "
            "`bestrouter` in the model picker, use `/bestrouter <task>` in the chat, or "
            'run_agent(agent="bestrouter", task="...").'
        )

    def register_as_model(self) -> bool:
        """List ``bestrouter`` in the model picker.

        A picked ``bestrouter`` makes the daemon run every task of the tab
        through this SEA on :data:`SIMPLE_MODEL` (``settings()`` below), with
        :data:`SYSTEM_PROMPT` added to the system prompt (``system_prompt()``).
        """
        return True

    def system_prompt(self, system_prompt: str) -> str:
        """Add the routing protocol to the default Sorcar system prompt."""
        return system_prompt + "\n\n" + SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Start every task of the tab on :data:`SIMPLE_MODEL`."""
        return settings | {"model": SIMPLE_MODEL}


