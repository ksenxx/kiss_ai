# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``/memory_quiz``: check that what finished tasks learned can be found in memory.

For each task of the user's own that ran on this machine and ended in a
report (:func:`kiss.agents.sorcar.persistence.reported_user_tasks`) the
task's model writes a few questions about the durable knowledge in the report,
each with the sentence of the report that states the answer.  A question
is used only when the report holds that sentence (:func:`grounded`) and
the judge model, of another family where one is configured, finds its
answer stated in the report (:func:`confirmed`).  Memory is then searched
as a new task in the same directory would search it, and the judge model
reads the first :data:`RECALL_PAGES` results:

* they give the answer: a **hit**;
* they do not, but the pages the task wrote or the next
  :data:`DEEP_PAGES` results do: a **retrieval gap**, the knowledge is
  stored and the search did not bring it up;
* nothing gives it: a **construction gap**, the knowledge was never stored.

Every question used goes to a log with its outcome (:func:`log_path`),
which is also how a task is known to be done.  A task is logged when its
report holds nothing to ask about, or when every question written about
it has been dropped or judged and at least one was used; a task with a
reply that could not be used, or with a question left without a verdict,
is not logged, and another run asks again.

Nothing else is written unless the task text holds ``--write``; then the
facts of a task's construction gaps are stored in one new page of that
task (:func:`store_gaps`): never an edit of an existing page, written as
``writer: quiz`` with no ``verified`` time, so every such page can be
listed, and each fact with the passage of the report it came from.  One
run at a time questions a memory.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.memory_jobs import (
    BudgetSpent,
    ModelCalls,
    count_option,
    json_objects,
    option,
)
from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.sorcar_agent import (
    _memory_settings,
    _persisted_task_id,
    _repo_memory_domains,
)
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.html_to_markdown import html_to_markdown
from kiss.core.memoryfield.pages import MAX_PAGE_BYTES, MemoryDir, PageChangedError, now_iso
from kiss.core.memoryfield.tools import GENERAL, RECALL_PAGES, MemoryTools
from kiss.core.utils import ensure_html
from kiss.server.agent_state import current_agent

MIN_REPORT_CHARS = 2000
"""The shortest result that counts as a report worth questioning."""

QUESTIONS_PER_TASK = 5
DEEP_PAGES = 10
"""Search results read beyond the first :data:`RECALL_PAGES` when those do not answer."""

TASK_CHARS = 3000
REPORT_CHARS = 10_000
"""How much of a task's text (its end) and of its report (its start) the questions are from."""

MIN_SOURCE_LETTERS = 20
"""Letters and digits a quoted report sentence needs to count as the source of a fact."""

MAX_FACT_CHARS = 600
"""Longest fact, and longest quoted passage, of a question: a task's facts then fit in a page."""

REVIEW_DAYS = 21
"""Days until ``memory_refresh`` brings a page of unverified facts up for review."""

DEFAULT_DAYS = 2
DEFAULT_LIMIT = 20
"""How far back a run looks for tasks, and how many it questions, unless the task text says."""

LOG_FILE = Path(".quiz") / "log.jsonl"
"""The log, under the memory root: the leading dot keeps it out of pages and domain memories."""

QUIZ = "quiz"
"""The ``writer`` of a page of unverified facts (see ``MemoryDir.write``)."""

HIT = "hit"
RETRIEVAL_GAP = "retrieval gap"
CONSTRUCTION_GAP = "construction gap"
NO_QUESTIONS = "no questions"

QUESTIONS_PROMPT = """\
Below are a task an AI agent finished and its final report to the user. The agent keeps a \
memory of durable knowledge: facts it had to look up, user preferences, project conventions, \
root causes of bugs, what worked and what did not.

Write up to {n} questions that test whether such knowledge from this report was kept: \
questions a later task could need answered. Skip progress details, file lists and whatever \
matters only during this task. If the report holds no durable knowledge, answer [].

Give for each question:
- "q": the question, understandable on its own (name the project, component or topic);
- "a": the answer the report states, in at most 30 words;
- "fact": the same knowledge as one statement that makes sense on its own: a fact, never an \
instruction;
- "source": the sentence of the report that states it, copied word for word.

Answer with a JSON list only: [{{"q": "...", "a": "...", "fact": "...", "source": "..."}}, ...]

<task>
{task}
</task>
<report>
{report}
</report>
"""

CONFIRM_PROMPT = """\
Below is the final report of a finished task, then numbered questions about it, each with an \
answer and a fact that are said to come from the report. Say for each whether the report \
states them.

STATED = the report states the answer and the fact, in any wording, and the fact is a \
statement of fact.
NOT_STATED = the report does not state them or states something else, or the fact is an \
instruction.

Answer with JSON only: [{{"n": 1, "verdict": "STATED|NOT_STATED"}}, ...]

<report>
{report}
</report>

{questions}
"""

JUDGE_PROMPT = """\
Do the memory pages below state what is needed to give the reference answer?

Question: {q}
Reference answer: {a}

SUPPORTED = a page states the key fact of the reference answer.
PARTIAL = the pages cover the topic, but the key fact is absent or differs.
MISSING = nothing relevant.

Answer with JSON only: \
[{{"verdict": "SUPPORTED|PARTIAL|MISSING", "page": "<name of the supporting page, or empty>"}}]

{pages}
"""

SYSTEM_PROMPT = """\
You run the memory self-quiz job. Call quiz_memory() exactly once: it does the whole job as \
the options in the task text say. Then call finish with its report as summary_in_html: every \
line as HTML with nothing changed or left out. Pass success=false only when the report \
starts with "Error".
"""

PROMPT = "Run the memory self-quiz job and report."


def log_path(root: Path) -> Path:
    """Return the file the quiz of the memory at *root* is logged in, one JSON line a question."""
    return root / LOG_FILE


def quizzed_tasks(root: Path) -> set[str]:
    """Return the ids of the tasks the log of the memory at *root* holds.

    A line that is not JSON (the torn last line of a run that was
    killed) is passed over.
    """
    path = log_path(root)
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        try:
            done.add(str(json.loads(line)["task"]))
        except (ValueError, KeyError, TypeError):
            continue
    return done


def append_log(root: Path, entries: list[dict[str, Any]]) -> None:
    """Append *entries* to the quiz log of the memory at *root*, one JSON line each.

    A torn last line (of a run that was killed) is ended first: the
    first new entry would otherwise be lost with it.
    """
    path = log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(entry) + "\n" for entry in entries)
    with path.open("a+b") as file:
        if file.tell():
            file.seek(-1, os.SEEK_END)
            if file.read(1) != b"\n":
                text = "\n" + text
        file.write(text.encode())


def _letters(text: str) -> str:
    """Return *text* as its letters and digits alone: what a quote and its source must share."""
    return re.sub(r"[\W_]+", "", text.casefold())


def grounded(items: list[dict[str, Any]], report: str) -> list[dict[str, str]]:
    """Return the questions among *items* that are complete and quote a passage of *report*.

    A model rarely copies a sentence character for character (a line
    break moves, a quotation mark changes), so quote and report are
    compared by their letters and digits alone.  The ``source`` returned,
    and stored, is the report's own passage, not the model's copy of it.

    Args:
        items: What the model answered: ``q``, ``a``, ``fact`` and ``source`` each.
        report: The report the questions were written from.

    Returns:
        The first :data:`QUESTIONS_PER_TASK` such questions, their four values stripped.
    """
    # The letters of the report, and where in the report each one is.
    marks = [(letter, at) for at, char in enumerate(report) for letter in _letters(char)]
    whole = "".join(letter for letter, _ in marks)
    kept = []
    for item in items:
        question = {key: str(item.get(key) or "").strip() for key in ("q", "a", "fact", "source")}
        quote = _letters(question["source"])
        start = whole.find(quote) if len(quote) >= MIN_SOURCE_LETTERS else -1
        if start < 0:
            continue
        question["source"] = report[marks[start][1] : marks[start + len(quote) - 1][1] + 1]
        if (
            all(question.values())
            and max(len(question["fact"]), len(question["source"])) <= MAX_FACT_CHARS
        ):
            kept.append(question)
    return kept[:QUESTIONS_PER_TASK]


def confirmed(
    calls: ModelCalls, questions: list[dict[str, str]], report: str
) -> list[dict[str, str]] | None:
    """Return the *questions* whose answer and fact the judge model finds stated in *report*.

    The questions are one model's reading of the report, and a fact of
    theirs may end up in memory: a second model reads the report too.

    Args:
        calls: The model calls of the running job.
        questions: Questions with ``q``, ``a`` and ``fact`` (:func:`grounded`).
        report: The report they were written from.

    Returns:
        The questions judged ``STATED`` and nothing else; ``None`` when
        the reply leaves a question without a verdict, as one that
        could not be used leaves them all.
    """
    listed = "\n".join(
        f"{n}. Question: {q['q']}\n   Answer: {q['a']}\n   Fact: {q['fact']}"
        for n, q in enumerate(questions, 1)
    )
    prompt = CONFIRM_PROMPT.format(report=report, questions=listed)
    # Every verdict given on a question counts: one judged twice is not judged by the last.
    given: dict[str, set[str]] = {}
    for item in json_objects(calls.ask(calls.judge, prompt)) or []:
        given.setdefault(str(item.get("n")), set()).add(str(item.get("verdict")))
    verdicts = [given.get(str(n), set()) for n in range(1, len(questions) + 1)]
    # A verdict against is final.  A question left without a verdict is not one found
    # unstated: the task, which is logged as a whole, is left for another run.
    if any("NOT_STATED" not in said and said != {"STATED"} for said in verdicts):
        return None
    return [q for q, said in zip(questions, verdicts) if said == {"STATED"}]


def own_pages(tools: MemoryTools, task_id: str) -> list[str]:
    """Return the pages of *tools*' memories that task *task_id* or a sub-task of it wrote.

    Read from the pages' ``task`` frontmatter as of the last index sync,
    so a search must have run on *tools* before.
    """
    writers = set(persistence.task_tree_ids(task_id))
    return [
        name if memory == GENERAL else f"{memory}/{name}"
        for memory, index in tools.indexes.items()
        for name, frontmatter in index.frontmatters().items()
        if frontmatter.get("task") in writers
    ]


def judge(
    calls: ModelCalls, tools: MemoryTools, question: dict[str, str], names: list[str]
) -> tuple[str, str]:
    """Ask the judge model whether the pages *names* answer *question*.

    Args:
        calls: The model calls of the running job.
        tools: The memories the pages are in.
        question: The question, with its reference answer.
        names: Page names as a search returns them.

    Returns:
        ``(verdict, page)``: ``SUPPORTED``, ``PARTIAL`` or ``MISSING``
        and the page the judge names as giving the answer; an empty
        verdict when the reply could not be used or gives two different
        ones.  No pages: ``MISSING`` without a call.
    """
    if not names:
        return "MISSING", ""
    # A page deleted since the search reads as the tool's error line, which answers nothing.
    pages = "\n\n".join(
        f"=== page {name} ===\n{tools.memory_read(name)[:MAX_PAGE_BYTES]}" for name in names
    )
    prompt = JUDGE_PROMPT.format(q=question["q"], a=question["a"], pages=pages)
    replies = json_objects(calls.ask(calls.judge, prompt)) or [{}]
    # Every verdict given counts here too: a reply that judges both ways gives none.
    verdicts = {str(reply.get("verdict")) for reply in replies}
    verdict = verdicts.pop() if len(verdicts) == 1 else ""
    if verdict not in ("SUPPORTED", "PARTIAL", "MISSING"):
        return "", ""
    return verdict, str(replies[0].get("page") or "")


def ask_memory(
    calls: ModelCalls, tools: MemoryTools, task_id: str, question: dict[str, str]
) -> dict[str, Any] | None:
    """Search memory for *question* and return its log record with the outcome.

    Args:
        calls: The model calls of the running job.
        tools: The memories a new task in the questioned task's directory searches.
        task_id: The task the question is about.
        question: ``q``, ``a``, ``fact`` and ``source``.

    Returns:
        *question* with ``first`` (the first results), ``outcome`` and
        ``page`` (the page that gives the answer, when one does);
        ``None`` when a reply of the judge could not be used.
    """
    found = [hit.name for hit in tools.search(question["q"], RECALL_PAGES + DEEP_PAGES)]
    first = found[:RECALL_PAGES]
    verdict, page = judge(calls, tools, question, first)
    outcome = HIT
    if verdict in ("PARTIAL", "MISSING"):
        written = [name for name in own_pages(tools, task_id) if name not in found]
        verdict, page = judge(calls, tools, question, [*written, *found[RECALL_PAGES:]])
        outcome = RETRIEVAL_GAP if verdict == "SUPPORTED" else CONSTRUCTION_GAP
    if not verdict:
        return None
    page = page if verdict == "SUPPORTED" else ""
    return {**question, "first": first, "outcome": outcome, "page": page}


def quiz_task(
    calls: ModelCalls, tools: MemoryTools, task: dict[str, Any]
) -> list[dict[str, Any]] | None:
    """Write questions about *task* and put each to memory.

    Args:
        calls: The model calls of the running job.
        tools: The memories a new task in *task*'s directory searches.
        task: The ``task_history`` row.

    Returns:
        What is logged of a task that is done: no record when the model
        answers ``[]`` (no durable knowledge in the report), else one
        record per question that stands (:func:`ask_memory`), at least
        one.  ``None`` when the task is to be questioned again by
        another run: a reply could not be used (refused, cut short, no
        JSON list of objects), a question was left without a verdict
        (:func:`confirmed`, :func:`judge`), or none of the questions
        written stands (:func:`grounded`, :func:`confirmed`).
    """
    report = html_to_markdown(ensure_html(str(task["result"])))[:REPORT_CHARS]
    prompt = QUESTIONS_PROMPT.format(
        n=QUESTIONS_PER_TASK, task=str(task["task"])[-TASK_CHARS:], report=report
    )
    items = json_objects(calls.ask(calls.model, prompt))
    if not items:
        return items
    questions = grounded(items, report)
    records: list[dict[str, Any]] = []
    for question in (confirmed(calls, questions, report) if questions else None) or []:
        record = ask_memory(calls, tools, str(task["id"]), question)
        if record is None:
            return None
        records.append(record)
    return records or None


def store_gaps(
    memory: MemoryDir, task: dict[str, Any], gaps: list[dict[str, Any]], job_id: str
) -> str:
    """Write the facts of *gaps* into a new page of unverified facts of *task*.

    Args:
        memory: The memory the page goes to.
        task: The ``task_history`` row the facts are from.
        gaps: The construction-gap records of the task.
        job_id: The task that runs the quiz, stored as the page's ``task``.

    Returns:
        The page's name, or ``""`` when a page of that name exists:
        nothing is written over a page.
    """
    task_id = str(task["id"])
    day = datetime.fromtimestamp(float(task["timestamp"]), UTC).date()
    about = " ".join(str(task["task"]).split())[:80]
    facts = "\n".join(
        f"- {' '.join(gap['fact'].split())}\n  Source: \"{' '.join(gap['source'].split())}\""
        for gap in gaps
    )
    body = (
        f"Facts from the final report of task {task_id} ({day}) that memory did not hold when "
        "the self-quiz job checked. Each is followed by the passage of the report it is from. "
        "One model wrote the facts and another found them stated in the report; nobody has "
        "verified them. Move a fact you confirm to the page of its topic and remove it here; "
        f"delete this page once it is empty.\n\n{facts}"
    )
    name = f"quiz-{task_id[:12]}"
    review = datetime.now(UTC).date() + timedelta(days=REVIEW_DAYS)
    try:
        memory.write(
            name,
            body,
            title=f"Unverified facts from the task: {about}",
            summary="Facts of the task's final report that memory did not hold; unverified.",
            extra={
                "task": job_id,
                "writer": QUIZ,
                "sources": f"final report of task {task_id}",
                "review_after": review.isoformat(),
            },
            expected="",
        )
    except PageChangedError:
        return ""
    return name


def task_lines(task: dict[str, Any], records: list[dict[str, Any]], note: str) -> str:
    """Return the report lines of one questioned task.

    Args:
        task: The ``task_history`` row.
        records: The task's records (:func:`ask_memory`).
        note: What was done about its construction gaps, to end the first line with.
    """
    counts = Counter(record["outcome"] for record in records)
    about = " ".join(str(task["task"]).split())[:60]
    head = f"- Task `{str(task['id'])[:12]}` ({about}): "
    if not records:
        return f"{head}nothing durable to ask about."
    lines = [
        f"{head}questions: {len(records)}; hits: {counts[HIT]}; retrieval gaps: "
        f"{counts[RETRIEVAL_GAP]}; construction gaps: {counts[CONSTRUCTION_GAP]}{note}."
    ]
    for record in records:
        if record["outcome"] == CONSTRUCTION_GAP:
            lines.append(f"  - not stored: {record['q']} Answer: {record['a']}")
        elif record["outcome"] == RETRIEVAL_GAP:
            lines.append(
                f"  - stored in `{record['page']}` but not among the first {RECALL_PAGES} "
                f"results: {record['q']}"
            )
    return "\n".join(lines)


def quiz_tasks(
    calls: ModelCalls,
    root: Path,
    tasks: list[dict[str, Any]],
    limit: int,
    write: bool,
    job_id: str,
) -> str:
    """Question memory about the first *limit* of *tasks*, log the outcomes and report.

    Args:
        calls: The model calls of the running job.
        root: The memory root.
        tasks: The ``task_history`` rows waiting to be questioned, in order.
        limit: Most tasks to question.
        write: Whether the facts of construction gaps are stored in memory.
        job_id: The task that runs the quiz.
    """
    memories: dict[str, MemoryTools] = {}
    lines: list[str] = []
    totals: Counter[str] = Counter()
    stored: list[str] = []
    for task in tasks[:limit]:
        work_dir = str(task["work_dir"])
        domains = _repo_memory_domains(work_dir) if work_dir else {}
        domain = next(iter(domains), GENERAL)
        if domain not in memories:
            memories[domain] = MemoryTools(root, domains=domains)
        try:
            records = quiz_task(calls, memories[domain], task)
        except BudgetSpent:
            break
        if records is None:
            lines.append(
                f"- Task `{str(task['id'])[:12]}`: its questions could not be written or "
                "judged; another run tries it again."
            )
            continue
        gaps = [record for record in records if record["outcome"] == CONSTRUCTION_GAP]
        note = ""
        if write and gaps:
            name = store_gaps(memories[domain].indexes[domain].memory, task, gaps, job_id)
            page = name if domain == GENERAL or not name else f"{domain}/{name}"
            note = f"; wrote `{page}`" if page else "; its page of unverified facts exists already"
            stored += [page] if page else []
            for gap in gaps:
                gap["stored"] = page
        stamp = {"time": now_iso(), "job": job_id, "task": str(task["id"])}
        entries = [stamp | record for record in records] or [stamp | {"outcome": NO_QUESTIONS}]
        append_log(root, entries)
        totals.update(record["outcome"] for record in records)
        lines.append(task_lines(task, records, note))
    head = (
        f"Memory self-quiz: tasks: {len(lines)}; questions: {sum(totals.values())}; "
        f"answered by the first {RECALL_PAGES} search results: {totals[HIT]}; "
        f"stored but not found (retrieval gaps): {totals[RETRIEVAL_GAP]}; "
        f"not stored (construction gaps): {totals[CONSTRUCTION_GAP]}."
    )
    if write:
        wrote = f"Pages of unverified facts written: {len(stored)}{': ' if stored else ''}"
        wrote += ", ".join(f"`{page}`" for page in stored) + "."
    else:
        wrote = "Nothing was written to memory; the option --write stores the facts not stored."
    tail = []
    if len(lines) < min(limit, len(tasks)):
        tail.append("Stopped early: the task's budget is nearly spent.")
    if len(lines) < len(tasks):
        tail.append(f"Tasks waiting for another run: {len(tasks) - len(lines)}.")
    tail += [f"Log: {log_path(root)}", calls.spend()]
    parts = (f"{head}\n{wrote}", "\n".join(lines), "\n".join(tail))
    return "\n\n".join(part for part in parts if part)


def quiz_memory() -> str:
    """Run the memory self-quiz job the task text asks for and return its report.

    Takes no argument: the options are the ``--`` words of the task text
    (:meth:`MemoryQuizSea.description` lists them), so nothing but the
    task itself can make the job write to memory.
    """
    caller = current_agent()
    if caller is None:
        return "Error: quiz_memory is a tool of a running /memory_quiz task"
    enabled, root = _memory_settings()
    if not enabled:
        return "Memory is switched off; there is nothing to quiz."
    try:
        calls = ModelCalls(caller, PROMPT)
        limit = count_option(calls.words, "--limit", DEFAULT_LIMIT)
        days = count_option(calls.words, "--days", DEFAULT_DAYS)
        prefix = option(calls.words, "--task")
    except ValueError as error:
        return f"Error: {error}"
    write = "--write" in calls.words
    # Held from reading which tasks are done to logging them, so that two runs never
    # question one task.  Beside the memory directory, like the lock of its writes.
    with exclusive_file_lock(Path(f"{root}.quiz.lock"), blocking=False) as alone:
        if not alone:
            return "Error: another /memory_quiz run is questioning this memory."
        if prefix:
            tasks = persistence.tasks_matching(id_prefix=prefix, limit=2)
            # A task of another machine has its work dir, and its repository's memory, there.
            if len(tasks) != 1 or not tasks[0]["success"] or tasks[0]["machine"]:
                return (
                    f"Error: --task {prefix} must name exactly one task that succeeded on this "
                    "machine."
                )
        else:
            done = quizzed_tasks(root)
            recent = persistence.reported_user_tasks(time.time() - days * 86400, MIN_REPORT_CHARS)
            tasks = [task for task in recent if task["id"] not in done]
        return quiz_tasks(calls, root, tasks, limit, write, _persisted_task_id(caller))


class MemoryQuizSea(WorkerSea):
    """``/memory_quiz``: question memory about finished tasks and log what it cannot answer."""

    def description(self) -> str:
        """One line for ``/help`` and the slash-command menu."""
        return (
            "Checks that what finished tasks learned can be found in memory. For each recent "
            "task of yours on this machine that ended in a report, a model writes questions "
            "about the durable knowledge in the report and the checking model (of another family "
            "where one is configured) reads the "
            "first search results for each: a hit, a retrieval gap (stored, but the search did "
            "not bring it up) or a construction gap (never stored). `/memory_quiz` logs every "
            "outcome in <memory>/.quiz/log.jsonl and writes nothing to memory; `--write` also "
            "stores the facts of a task's construction gaps in one new page marked `writer: "
            "quiz`, with the report sentence each came from. `--task <id prefix>` questions one "
            "task, "
            f"`--days N` (default {DEFAULT_DAYS}) and `--limit N` (default {DEFAULT_LIMIT}) "
            "bound a run, and `--judge <model>` names the checking model."
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Replace the assistant's system prompt with the job's two steps."""
        return SYSTEM_PROMPT

    def prompt(self, task: str) -> str:
        """Keep *task*, whose options the tool reads, and add the request to run the job."""
        return f"{task}\n\n{PROMPT}".strip()

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """The job and ``finish``: ``tool_profile`` is ``none``, so *tools* holds no other."""
        return [*tools, quiz_memory]

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """One tool call that may run for a while: no built-in tools, $10 budget, an hour."""
        return settings | {
            "tool_profile": "none",
            "max_budget": 10.0,
            "timeout": 3600,
            "locked": ["tool_profile"],
        }
