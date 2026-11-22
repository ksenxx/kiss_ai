# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``/memory_consolidate``: merge near-duplicate memory pages without losing a fact.

The job takes the near-duplicate families of each memory that fit in one
page (:func:`kiss.core.memoryfield.consolidate.families`), the most
similar first, and examines each with up to four model calls
(:func:`examine`, which stops at the first check that fails): the
task's model writes the page that would replace the family, or says the
pages are separate records; the judge model, of another family where one
is configured, writes questions from the original pages without seeing
the merged one, says for each question whether the merged page still
gives the answer, and last reads the originals beside the merged page
and lists every fact it no longer finds.  The questions are a sample of
the facts (on real merges with one line taken out, they noticed the loss
in about one page of five); the reading of both sides is what notices a
single lost fact.  The family is merged only when the merged page fits
in a page, answers every question and misses no fact.  One run at a time
examines a memory.

Nothing is written unless the task text holds ``--apply``.  A merge
keeps the originals in ``<memory>/.archive/`` and can be undone with
``--restore`` (:func:`kiss.core.memoryfield.consolidate.restore`); a
family left as it is for a reason that holds for its pages (separate
records, a merged page too large, a fact not kept) is recorded there
too, so the next run moves on to other families.  A family whose
examination ended in a reply that could not be used is not recorded:
another run examines it again.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
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
from kiss.agents.sorcar.sorcar_agent import _memory_settings, _persisted_task_id
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.memoryfield import consolidate
from kiss.core.memoryfield.index import VectorIndex
from kiss.core.memoryfield.pages import MAX_PAGE_BYTES, PageChangedError
from kiss.core.memoryfield.tools import GENERAL, MemoryTools
from kiss.scripts.merge_memory_pages import is_domain_memory
from kiss.server.agent_state import current_agent

DEFAULT_LIMIT = 10
"""Families a run examines unless the task text says ``--limit N``."""

QUESTIONS_PER_PAGE = 5
MAX_QUESTIONS = 20
"""Questions a merged page is checked against: per page of the family, and at most."""

FACT_CHARS = 120
"""Characters of a fact found missing that the report quotes."""

MERGE_PROMPT = """\
The memory pages below are near-duplicates, listed oldest first. Write the one page that \
replaces them.

- Keep every fact any of the pages states: names, numbers, paths, commands, dates, causes and \
caveats. Remove only repetition.
- Where pages disagree, the newer page wins; note the older value and its date beside it.
- Add nothing the pages do not state.
- At most {limit} bytes.

If the pages are separate records that must stay apart (different dates, runs, versions or \
subjects that share only a template), answer with the single word SEPARATE.

Otherwise answer in exactly this form and with nothing else:
TITLE: <one line>
SUMMARY: <one sentence>

<the page body in Markdown>

{pages}
"""

QUESTIONS_PROMPT = """\
Below are memory pages, listed oldest first. Write the questions a reader must still be able \
to answer after the pages are merged into one: up to {per_page} per page and {most} in all, \
each about one fact a page states (a name, number, path, command, date, cause or caveat), with \
the answer the page gives. Where pages disagree, give the newer page's answer. Each question \
must make sense without the pages.

Answer with a JSON list only: [{{"q": "<question>", "a": "<answer>"}}, ...]

{pages}
"""

JUDGE_PROMPT = """\
For each question below, say whether the memory page gives the expected answer.

SUPPORTED = the page states the expected answer, in any wording; an older value mentioned \
beside it as superseded does not count against it.
DIFFERENT = the page answers the question with another value.
ABSENT = the page does not answer the question.

Answer with a JSON list only, one item per question: \
[{{"n": <number>, "verdict": "SUPPORTED|DIFFERENT|ABSENT"}}, ...]

<page>
{page}
</page>

<questions>
{questions}
</questions>
"""

COMPARE_PROMPT = """\
Below are ORIGINAL memory pages, listed oldest first, and the MERGED page that is to replace \
them. List every fact an original page states that a reader can no longer learn from the \
merged page: a name, number, path, command, date, cause or caveat.

Not a loss: removed repetition; rewording; a value of an older page that a newer page \
replaced (the merged page must give the newer value).

Answer with a JSON list only, [] when nothing is lost: [{{"fact": "<the lost fact>"}}, ...]

<originals>
{pages}
</originals>

<merged>
{page}
</merged>
"""

MERGED_RE = re.compile(r"\s*TITLE:[ \t]*([^\n]+)\n+SUMMARY:[ \t]*([^\n]+)\n+(.*\S)", re.DOTALL)

SYSTEM_PROMPT = """\
You run the memory consolidation job. Call consolidate_memory() exactly once: it does the \
whole job as the options in the task text say. Then call finish with its report as \
summary_in_html: every line as HTML with nothing changed or left out, except the proposed \
page texts of a dry run, which you leave out. Pass success=false only when the report starts \
with "Error".
"""

PROMPT = "Run the memory consolidation job and report."


@dataclass(frozen=True)
class Verdict:
    """What the checks say about merging one family.

    Attributes:
        merged: ``(title, summary, body)`` of the page that may replace
            the family; ``None`` when the family is left as it is.
        reason: Why it is left as it is.
        final: Whether *reason* holds for the pages as they are (the
            family is then not examined again) and is not just a reply
            that could not be used.
        questions: How many questions the merged page answered.
    """

    merged: tuple[str, str, str] | None = None
    reason: str = ""
    final: bool = False
    questions: int = 0


def family_text(family: consolidate.Family) -> str:
    """Return the pages of *family* as the prompts quote them, oldest first."""
    return "\n\n".join(
        f"=== page {page.name} (updated {page.frontmatter.get('updated', 'unknown')}) ===\n"
        f"title: {page.title}\nsummary: {page.summary}\n\n{page.body.strip()}"
        for page in family.pages
    )


def examine(calls: ModelCalls, family: consolidate.Family) -> Verdict:
    """Have the models write the page that replaces *family* and check that it loses no fact.

    Args:
        calls: The model calls of the running job.
        family: A family that fits in one page.
    """
    pages = family_text(family)
    reply = calls.ask(calls.model, MERGE_PROMPT.format(limit=MAX_PAGE_BYTES, pages=pages))
    if reply.strip().strip(".").upper() == "SEPARATE":
        return Verdict(reason="separate records", final=True)
    match = MERGED_RE.match(reply)
    if match is None:
        return Verdict(reason="the merge reply could not be used")
    title, summary, body = (part.strip() for part in match.groups())
    size = len("\n".join((title, summary, body)).encode("utf-8"))
    if size > MAX_PAGE_BYTES:
        return Verdict(reason=f"the merged page is too large ({size} bytes)", final=True)
    asked = calls.ask(
        calls.judge,
        QUESTIONS_PROMPT.format(per_page=QUESTIONS_PER_PAGE, most=MAX_QUESTIONS, pages=pages),
    )
    questions = [q for q in json_objects(asked) or [] if q.get("q") and q.get("a")]
    questions = questions[:MAX_QUESTIONS]
    if not questions:
        return Verdict(reason="no questions could be written to check the merge")
    listed = "\n".join(f"{n}. Q: {q['q']}\n   A: {q['a']}" for n, q in enumerate(questions, 1))
    page = f"title: {title}\nsummary: {summary}\n\n{body}"
    judged = json_objects(calls.ask(calls.judge, JUDGE_PROMPT.format(page=page, questions=listed)))
    # Every verdict given on a question counts: one judged twice is not judged by the last.
    given: dict[str, set[str]] = {}
    for item in judged or []:
        given.setdefault(str(item.get("n")), set()).add(str(item.get("verdict")))
    verdicts = [given.get(str(n), set()) for n in range(1, len(questions) + 1)]
    lost = [q["q"] for q, said in zip(questions, verdicts) if said & {"DIFFERENT", "ABSENT"}]
    if lost:
        reason = f"{len(lost)} of {len(questions)} facts not kept ({'; '.join(lost[:3])})"
        return Verdict(reason=reason, final=True)
    # A question left without a verdict is not a fact found lost: another run asks again.
    if any(said != {"SUPPORTED"} for said in verdicts):
        return Verdict(reason="the check reply could not be used")
    # The questions sample the facts; reading both sides finds what none of them asked about.
    missing = json_objects(calls.ask(calls.judge, COMPARE_PROMPT.format(pages=pages, page=page)))
    if missing is None:
        return Verdict(reason="the check reply could not be used")
    if missing:
        facts = "; ".join(str(item.get("fact", item))[:FACT_CHARS] for item in missing[:3])
        reason = f"facts of the originals found missing: {len(missing)} ({facts})"
        return Verdict(reason=reason, final=True)
    return Verdict(merged=(title, summary, body), questions=len(questions))


def qualified(memory: str, name: str) -> str:
    """Return *name* (a page or a merge) of *memory* as the memory tools name it."""
    return name if memory == GENERAL else f"{memory}/{name}"


def list_merges(indexes: dict[str, VectorIndex]) -> str:
    """Return the merges recorded in the memories of *indexes*, one line each, newest first."""
    lines = []
    for memory, index in indexes.items():
        for record in consolidate.merges(index.memory):
            state = f"restored {record['restored']}" if "restored" in record else "in place"
            lines.append(
                f"- `{qualified(memory, record['id'])}`: {', '.join(record['members'])} "
                f"into {record['target']} ({state})"
            )
    return "\n".join(lines) or "No merges are recorded."


def restore_merge(indexes: dict[str, VectorIndex], merge: str) -> str:
    """Undo the merge named *merge* (as :func:`list_merges` names it) and say what came back."""
    memory, _, merge_id = merge.rpartition("/")
    index = indexes.get(memory or GENERAL)
    if index is None:
        return f"Error: no memory {memory!r}."
    try:
        restored, skipped = consolidate.restore(index.memory, merge_id)
    except (ValueError, TimeoutError) as error:
        return f"Error: {error}"
    lines = [f"Merge `{merge}`: restored {', '.join(restored) or 'no page'}."]
    lines += [f"- `{name}` not restored: {why}" for name, why in skipped.items()]
    return "\n".join(lines)


def settle(
    calls: ModelCalls,
    memory: str,
    index: VectorIndex,
    family: consolidate.Family,
    apply: bool,
    task_id: str,
) -> tuple[str, bool]:
    """Examine *family* and, when it passes and *apply* is set, merge it.

    Args:
        calls: The model calls of the running job.
        memory: The name of the family's memory.
        index: That memory's index.
        family: A family that fits in one page.
        apply: Whether the job may write.
        task_id: The task that runs the job, stored on the merged page.

    Returns:
        The report line of the family, and whether its merge passed the checks.
    """
    names = " + ".join(f"`{qualified(memory, page.name)}`" for page in family.pages)
    head = f"- {names} (similarity {family.score:.2f}): "
    verdict = examine(calls, family)
    if verdict.merged is None:
        if apply and verdict.final:
            consolidate.leave_alone(index.memory, family, verdict.reason)
        return f"{head}left as they are, {verdict.reason}.", False
    title, summary, body = verdict.merged
    before = sum(len(page.body.encode("utf-8")) for page in family.pages)
    detail = (
        f"{before} to {len(body.encode('utf-8'))} bytes; check questions answered: "
        f"{verdict.questions} of {verdict.questions}; no fact of the originals found missing"
    )
    target = qualified(memory, family.target.name)
    if not apply:
        proposed = f"TITLE: {title}\nSUMMARY: {summary}\n\n{body}"
        return (
            f"{head}would become `{target}` ({detail}).\n"
            f"Proposed page (not written):\n~~~\n{proposed}\n~~~",
            True,
        )
    try:
        merge = consolidate.commit(index.memory, family, body, title, summary, task_id)
    except (PageChangedError, ValueError, TimeoutError) as error:
        return f"{head}not merged, {error}", False
    kept = f"; {', '.join(merge.kept)} changed meanwhile and stayed" if merge.kept else ""
    return (
        f"{head}merged into `{target}` ({detail}{kept}). "
        f"Undo: `/memory_consolidate --restore {qualified(memory, merge.id)}`",
        True,
    )


def _similarity(item: tuple[str, consolidate.Family]) -> float:
    """Sort key: the similarity of the family in *item*."""
    return item[1].score


def merge_families(
    calls: ModelCalls, indexes: dict[str, VectorIndex], limit: int, apply: bool, task_id: str
) -> str:
    """Examine up to *limit* families of *indexes*, the most similar first, and report.

    Args:
        calls: The model calls of the running job.
        indexes: The memories to work on, by name.
        limit: Most families to examine.
        apply: Whether passing families are merged; a dry run otherwise.
        task_id: The task that runs the job.
    """
    waiting: list[tuple[str, consolidate.Family]] = []
    left: Counter[str] = Counter()
    for memory, index in indexes.items():
        for family in consolidate.families(index):
            if family.skip:
                left[re.split(r" \(|:", family.skip)[0]] += 1
            else:
                waiting.append((memory, family))
    waiting.sort(key=_similarity, reverse=True)
    lines: list[str] = []
    passed = 0
    for memory, family in waiting[:limit]:
        try:
            line, ok = settle(calls, memory, indexes[memory], family, apply, task_id)
        except BudgetSpent:
            break
        lines.append(line)
        passed += ok
    if apply:
        head = f"Memory consolidation: families examined: {len(lines)}; merged: {passed}."
    else:
        head = (
            f"Memory consolidation, dry run: families examined: {len(lines)}; would merge: "
            f"{passed}. Nothing was written; the option --apply merges."
        )
    tail = []
    if len(lines) < min(limit, len(waiting)):
        tail.append("Stopped early: the task's budget is nearly spent.")
    if len(waiting) > len(lines):
        tail.append(f"Families waiting for another run: {len(waiting) - len(lines)}.")
    if left:
        kinds = "; ".join(f"{kind}: {count}" for kind, count in left.most_common())
        tail.append(f"Families not examined ({kinds}).")
    tail.append(calls.spend())
    return "\n\n".join(part for part in (head, "\n".join(lines), "\n".join(tail)) if part)


def consolidate_memory() -> str:
    """Run the memory consolidation job the task text asks for and return its report.

    Takes no argument: the options are the ``--`` words of the task text
    (:meth:`MemoryConsolidateSea.description` lists them), so nothing but
    the task itself can make the job write.
    """
    caller = current_agent()
    if caller is None:
        return "Error: consolidate_memory is a tool of a running /memory_consolidate task"
    enabled, root = _memory_settings()
    if not enabled:
        return "Memory is switched off; there is nothing to consolidate."
    names = sorted(p.name for p in root.iterdir() if is_domain_memory(p)) if root.is_dir() else []
    try:
        indexes = MemoryTools(root, domains=dict.fromkeys(names, "")).indexes
        calls = ModelCalls(caller, PROMPT)
        only = option(calls.words, "--memory")
        if only and only not in indexes:
            raise ValueError(f"no memory {only!r}; memories: {', '.join(indexes)}")
        if "--restore" in calls.words:
            return restore_merge(indexes, option(calls.words, "--restore"))
        limit = count_option(calls.words, "--limit", DEFAULT_LIMIT)
    except ValueError as error:
        return f"Error: {error}"
    selected = {only: indexes[only]} if only else indexes
    if "--list" in calls.words:
        return list_merges(selected)
    apply = "--apply" in calls.words
    # Beside the memory directory, like the lock of its writes: no file of the job's is in it.
    with exclusive_file_lock(Path(f"{root}.consolidate.lock"), blocking=False) as alone:
        if not alone:
            return "Error: another /memory_consolidate run is examining this memory."
        return merge_families(calls, selected, limit, apply, _persisted_task_id(caller))


class MemoryConsolidateSea(WorkerSea):
    """``/memory_consolidate``: merge near-duplicate memory pages, checked and undoable."""

    def description(self) -> str:
        """One line for ``/help`` and the slash-command menu."""
        return (
            "Merges near-duplicate memory pages. For each group of such pages that fits in one "
            "page, a model writes the page that replaces the group, the checking model (of "
            "another family where one is configured) writes questions from the originals, "
            "checks that the new page still answers every one and reads both sides for a fact "
            "that went missing, and only then is the group "
            "replaced; the originals stay in <memory>/.archive/. `/memory_consolidate` is a dry "
            "run that reports what it would merge and writes nothing; `--apply` merges; `--list` "
            "shows past merges and "
            "`--restore <merge>` undoes one. `--memory <name>` narrows the job to one memory, "
            f"`--limit N` sets how many groups a run examines (default {DEFAULT_LIMIT}) and "
            "`--judge <model>` names the checking model."
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Replace the assistant's system prompt with the job's two steps."""
        return SYSTEM_PROMPT

    def prompt(self, task: str) -> str:
        """Keep *task*, whose options the tool reads, and add the request to run the job."""
        return f"{task}\n\n{PROMPT}".strip()

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """The job and ``finish``: ``tool_profile`` is ``none``, so *tools* holds no other."""
        return [*tools, consolidate_memory]

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """One tool call that may run for a while: no built-in tools, $10 budget, an hour."""
        return settings | {
            "tool_profile": "none",
            "max_budget": 10.0,
            "timeout": 3600,
            "locked": ["tool_profile"],
        }
