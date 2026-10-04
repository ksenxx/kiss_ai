# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Revise-and-review loop — writes a paper with ``/write_paper``, has ``/review_paper`` review it
fresh, and repeats until the review says strong accept or the paper cannot improve further.

Typed into an idle tab as ``/revise_and_review_paper <instructions>``, for example::

    /revise_and_review_paper
    Writing: Write a paper for NeurIPS 2027 at ./papers/hydra/hydra_kv.tex on Hydra-KV,
    a log-structured key-value store. Sources: ./projects/hydra_kv/. 9 pages,
    neurips_2027.sty, double-blind. Experiments may run on YCSB and on the
    ./projects/hydra_kv/bench/ workloads.
    Review: Review for NeurIPS 2027 as an area chair would, 1000 words, related
    work before June 2027, ignore the page limit.

the slash command dispatches this file as a Sorcar Extension Agent
through ``run_agent`` with the instructions as the task.  The task text
carries two blocks: how to write the paper (passed to ``/write_paper``)
and how to review it (passed to ``/review_paper``).  The agent this
file configures is a *coordinator*: it never writes or reviews the
paper itself.  Each round it runs the writer SEA on the writing
instructions (round 1) or on the latest review (later rounds), stages
a copy of the built PDF, runs the reviewer SEA on it as a fresh,
memory-free session that sees no earlier review or notes, and reads
the review's ``Recommendation:`` line.  The loop stops at ``strong
accept``, at the round cap, or when two rounds in a row fail to raise
the verdict (the paper cannot be improved further with the evidence
the user allows).

Three tools make the mechanical steps deterministic:

* :func:`writer_task` returns the task text of a writing round, with
  the user's instructions, the review to answer, the rules on
  experiments, ablations and AI discovery, and the report contract.
* :func:`reviewer_task` returns the task text of a fresh review
  round: the staged PDF only, no earlier reviews, notes or memory,
  the venue's scores and a final ``Recommendation:`` line.
* :func:`loop_status` parses the ``Recommendation:`` and score lines
  of every round's review and says whether to stop or continue.

Module-level getters (``add_to_system_prompt()``, ``add_to_tools()``,
...) follow the SEA contract in :mod:`kiss.server.agent_file`.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any

WRITE_PAPER_SEA = str(Path(__file__).resolve().parents[1] / "write_paper" / "write_paper_sea.py")
"""Absolute path of the bundled ``/write_paper`` SEA, the writer of every round."""

REVIEW_PAPER_SEA = str(Path(__file__).resolve().parents[1] / "review_paper" / "review_paper_sea.py")
"""Absolute path of the bundled ``/review_paper`` SEA, the reviewer of every round."""

VERDICTS = (
    "strong reject",
    "reject",
    "weak reject",
    "borderline",
    "weak accept",
    "accept",
    "strong accept",
)
"""Recommendations a review may end with, weakest first; the loop's target is the last."""

TARGET_VERDICT = VERDICTS[-1]
"""The recommendation that ends the loop."""

DEFAULT_MAX_ROUNDS = 6
"""Write-review rounds after which the loop stops when the task names no cap."""

DEFAULT_WORD_LIMIT = 1000
"""Word limit of each review when the task names none (room for pinpointed findings)."""

WRITER_TIMEOUT_SECONDS = 14400
"""``run_agent`` timeout of a writing round: a paper with experiments takes hours."""

REVIEWER_TIMEOUT_SECONDS = 3600
"""``run_agent`` timeout of a review round."""

DISPATCH_TIMEOUT_SECONDS = 24 * 3600
"""Wait of the slash-command relay for the whole loop (see :func:`dispatch_timeout`)."""

SYSTEM_PROMPT = f"""\
# Revise-and-review loop coordinator

You coordinate a loop that writes a research paper with the `/write_paper` agent and has
the `/review_paper` agent review it as a fresh reviewer, round after round, until the
review says "{TARGET_VERDICT}" or the paper cannot be improved further. You never write
or review the paper yourself; you run the two agents, keep the log, and decide when to stop.

## The task text

The task carries two blocks of instructions from the user:

- the writing instructions (introduced by "Writing:", "Write:", "Paper:" or similar): the
  venue, the output path, the topic, the sources of truth, the page limit, the anonymity
  mode, the house style, the writer model, and which experiments, ablations or benchmarks
  the writer may run (the user's own benchmarks, named public benchmarks, or none);
- the review instructions (introduced by "Review:", "Reviewing:" or similar): the venue
  and year, the reviewer model, the word limit (default {DEFAULT_WORD_LIMIT}), the cutoff
  date for related work, the venue rules to ignore.

Options may follow either block: the round cap (default {DEFAULT_MAX_ROUNDS}), the budget
split, a target below "{TARGET_VERDICT}". When the two blocks cannot be told apart, when
the writing instructions lack the venue, the output path or the topic of a new paper, or
when the review instructions lack the venue, ask the user once with `ask_user_question`
before the first round; the writer and the reviewer run unattended and must not ask.

## How the agents run

Both agents run through `run_agent` with the absolute path of their SEA file:

- writer: `run_agent(task=<writer_task output>, agent="{WRITE_PAPER_SEA}",
  options='{{"use_worktree": false, "auto_commit": false}}',
  timeout="{WRITER_TIMEOUT_SECONDS}", max_budget=<share>)`;
- reviewer: `run_agent(task=<reviewer_task output>, agent="{REVIEW_PAPER_SEA}",
  options='{{"use_worktree": false, "auto_commit": false, "use_memory": false}}',
  timeout="{REVIEWER_TIMEOUT_SECONDS}", max_budget=<share>)`.

Pass `model` when the user names a writer or reviewer model; use model names
literally. Never pass `chat_id` in `options`: every round is a new session, and the
paper, its sources and the review files carry the state. Pass the tool outputs of `writer_task` and
`reviewer_task` as the task text verbatim; do not paraphrase them, and never paste an
earlier review, your own opinion of the paper, or the round history into the reviewer's
task. The writer and the reviewer share your working directory (`pwd` first; when the
user names paths under the main checkout and you run in a worktree, translate them to the
worktree before anything else), so the paper the writer builds is the paper the reviewer
reads.

## A fresh review

The reviewer must judge the paper as a blind submission. Before each review round:

1. copy the built PDF to `<work dir>/tmp/revise_review/round-<N>/<stem>.pdf` (a fresh
   directory holding that file only);
2. remove notes a previous review may have left under `<work dir>/tmp/`
   (`PAPER-NOTES.md`, `information-<stem>*.md`);
3. build the task with `reviewer_task(...)`, whose text forbids the reviewer to open
   earlier reviews, notes, the git history or the paper's source directory, and run it
   with `use_memory="false"` and no `chat_id`.

Each round's review goes to `<work dir>/reports/<stem>-round<N>-review.txt`.

## The loop

1. Round 1: the writer writes the paper from the writing instructions
   (`writer_task(instructions, 1)`). When the writing instructions name an existing paper
   and ask to start from a review, skip straight to the review of that paper.
2. After each writing round find the built PDF (the writer reports it; otherwise it is
   the `.tex` path with a `.pdf` suffix). If it is missing or older than the `.tex`, run
   one more writer round asking only for the build.
3. Stage the PDF, run the fresh review, then call
   `loop_status("<work dir>/reports/<stem>-round*-review.txt", max_rounds)` and follow
   its decision. When it reports CHECK for a review without a readable recommendation,
   read that review's score lines against the venue's scale, decide the verdict yourself,
   record it in the log and call `loop_status` again with `overrides="<round>=<verdict>"`.
4. On CONTINUE, read the review and triage each weakness before the next writing round:
   a writing fix; an experiment, ablation or benchmark run the user allows (the writer
   runs it, using the AI-discovery process of its system prompt when the review asks for a
   stronger method or result); or a point the user's instructions rule out, which the
   paper states as a limitation. Put the ruled-out points, the allowed benchmarks and the
   remaining budget into `coordinator_notes` of
   `writer_task(instructions, N + 1, review_path, coordinator_notes)`.
5. Stop early, before the round cap, when the latest review's weaknesses are all points
   the user ruled out or points an earlier round already answered with evidence the
   reviewer disputes on judgment, not fact: the paper cannot be improved further with the
   evidence the user allows. Say so in the report. A STOP for two rounds without
   improvement is final unless the latest review raises a weakness no earlier round
   attempted and the user's instructions allow the evidence it needs: then run one more
   round.
6. Never ask the writer to game the review: no removing negative results, no inflated
   claims, no mention of the review, the round or earlier versions in the paper.

## Budget and time

Let B be this task's budget (in the task settings of your system prompt). Keep 10% for
coordination. Round 1's writer gets at most 40% of B (25% when revising an existing
paper); each later writer round at most 20%; each review at most 8%. Every `run_agent`
result reports the child's spend, and the `Budget:` figure of your tool results already
includes it (do not add it again); stop, with the best paper so far, when the remainder
cannot fund one writer round plus one review. A
`run_agent` that times out stops the child: check on disk what it finished (the `.tex`,
the PDF, the review file) before rerunning it once with a larger timeout.

## Log and report

Keep `<work dir>/reports/<stem>-revise-review-log.md`: one section per round with the
recommendation and score lines, each weakness and what the writer did with it (fixed
where, experiment run with its numbers, declined why), the experiments run, the child's
cost and time. Update `<work dir>/tmp/revise_review/PROGRESS.md` after each round (the
writer writes its own `./tmp/PROGRESS.md` in the shared work dir and overwrites yours).
Git add the paper directory
(`.tex`, `.bib`, figures, scripts, raw results, `.pdf`; never `.aux`, `.bbl`, `.blg`,
`.log`, `.out`), the review files and the log; the children run with auto-commit off.

Report back: the recommendation of every round, the final paper's `.tex` and `.pdf` paths
and page count, every review path, the log path, the cost of each round and the total,
why the loop stopped, and the weaknesses still open when the target was not reached.
""" """\


## Lessons from recent runs (rsi7d)

- Override a `loop_status` STOP at most once per loop, and only under the rule above (a
  weakness no earlier round attempted, with evidence the user allows). The STOP after
  that extra round is final even if the fresh review names new fixable points: every
  fresh reviewer does. Unless the user's task text itself says not to stop before every
  comment is addressed, three overrides in one loop bought rounds 4 to 6 (about $66 of
  children) and the verdicts went weak accept, weak reject, weak accept, weak reject,
  weak reject, weak reject.
- Take each child's cost and step count from the text of its `run_agent` result and
  write them into the log at once; never query `~/.kiss/sorcar.db` for them (a quoting
  error in the sqlite query cost two steps and the result was already on hand).
"""
"""The coordinator's rules, appended to the default system prompt."""

_WRITER_RULES = """\
Rules for this round:
- Run unattended: do not ask the user questions; when something is missing, choose the
  reading closest to the instructions and say so in your report.
- Pass absolute paths inside your working directory (`pwd` first) to `build_paper`,
  `check_paper` and every script.
- Edit and build: the round ends with a PDF built from the current `.tex` that passes
  `build_paper` (zero errors, zero undefined references) and `check_paper`.
- Experiments: when the instructions or the review ask for evidence the paper lacks (a
  baseline, an ablation, a benchmark, a stronger result), run it on the benchmarks the
  instructions name; when they name none, use the standard public benchmarks of the area
  and say which in the paper. Use the AI-discovery process of your system prompt when a
  stronger method or result is asked for. Commit the scripts and the raw results next to
  the paper; every new number enters the paper through its macros.
- Never remove a negative result, inflate a claim, or mention a review, a round or an
  earlier version in the paper. A point you cannot answer with evidence becomes a
  limitation stated in the paper.
- Report back: the `.tex` and `.pdf` paths, the page count, the gate counts, each
  experiment run with its numbers, and (for a revision) one line per weakness of the
  review: fixed (where), experiment run (numbers), or declined (why).
- Do not run a review of your own in this round; an independent reviewer follows.
- Spend at most the budget this session was given; when it runs short, finish with the
  paper building and report what is left undone.
"""


def description() -> str:
    """Return the one-sentence help text shown by ``/revise_and_review_paper help``."""
    return (
        "Writes a research paper with /write_paper from your writing instructions, has "
        "/review_paper review it as a fresh reviewer under your review instructions, and "
        "repeats the revise-and-review loop (with experiments, ablations or AI discovery when "
        "the review asks for evidence) until the review says strong accept or the paper cannot "
        "improve further; use `/revise_and_review_paper Writing: <...> Review: <...>`."
    )


def writer_task(
    instructions: str,
    round_number: int,
    review_path: str = "",
    coordinator_notes: str = "",
) -> str:
    """Return the task text of one ``/write_paper`` round.

    Args:
        instructions: The user's writing instructions, verbatim (venue, path, topic,
            sources, page limit, anonymity, allowed experiments and benchmarks, models).
        round_number: 1 for the first writing round, then 2, 3, ...
        review_path: Absolute path of the review this round answers; empty for a round
            that writes the paper from the instructions alone.
        coordinator_notes: Points the user ruled out, the benchmarks allowed, the budget
            left, and other notes from the coordinator; empty for none.

    Returns:
        The task text to pass verbatim to ``run_agent`` on the ``/write_paper`` SEA.
    """
    if review_path:
        head = (
            f"You ARE the paper-writing agent. Revision round {round_number}: edit the paper "
            f"described below to answer the review at {review_path}. Read the review in full, "
            "then address every weakness: fix the writing, run the experiment or ablation it "
            "asks for when the instructions allow it, or state the point as a limitation when "
            "they rule it out."
        )
    else:
        head = (
            f"You ARE the paper-writing agent. Writing round {round_number}: write the paper "
            "described below (or revise it when the instructions name an existing paper)."
        )
    parts = [head, "", "Writing instructions from the user:", instructions.strip(), ""]
    if coordinator_notes.strip():
        parts += ["Notes from the loop coordinator:", coordinator_notes.strip(), ""]
    parts.append(_WRITER_RULES.rstrip())
    return "\n".join(parts)


def reviewer_task(
    instructions: str,
    paper_path: str,
    output_path: str,
    round_number: int,
    word_limit: int = DEFAULT_WORD_LIMIT,
) -> str:
    """Return the task text of one fresh ``/review_paper`` round.

    Args:
        instructions: The user's review instructions, verbatim (venue and year, cutoff
            date, rules to ignore, reviewer model, anything about the form).
        paper_path: Absolute path of the staged PDF copy the reviewer judges.
        output_path: Absolute path the review is written to.
        round_number: The round this review belongs to (kept out of the review text).
        word_limit: Word limit of the review.

    Returns:
        The task text to pass verbatim to ``run_agent`` on the ``/review_paper`` SEA.
    """
    round_dir = os.path.dirname(paper_path)  # keeps the caller's separators, unlike Path
    choices = " | ".join(VERDICTS)
    return f"""\
You ARE the paper review agent. Review the paper at {paper_path} under the instructions \
below, then write the review to {output_path} (word_limit={word_limit}).

Review instructions from the user:
{instructions.strip()}

This is a fresh, independent review of a blind submission (round {round_number} of a \
revision cycle; do not mention rounds, earlier versions or a revision in the review). \
Judge only the file named above. Do not open any other file of the repository, any \
earlier review or review notes, the git history, memory pages, or the paper's source \
directory. Keep your notes and your research log under {round_dir}/notes/, not under \
./tmp. Run unattended: do not ask the user questions; when something is unclear, say so \
in the review.

End the review with the venue's score lines (one per line, as the venue's form names \
them) followed by one final line of exactly this form:
Recommendation: <{choices}>
Choose the recommendation the venue's area chair would read from your review; \
"{TARGET_VERDICT}" only for a paper you would argue for at the meeting."""


_RECOMMENDATION = re.compile(r"^\s*Recommendation\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
"""The final ``Recommendation: <verdict>`` line of a review."""

_VERDICT_AT_START = re.compile(
    "^[\\s*_\"'`]*(" + "|".join(sorted(VERDICTS, key=len, reverse=True)) + r")(?![a-z])"
)
"""A verdict opening the recommendation text (longest first: "borderline accept" is
borderline, "strong accept" is not accept, "unacceptable" is nothing)."""

_SCORE_LINE = re.compile(
    r"^\s*([A-Za-z][A-Za-z /()-]{1,40}?)\s*:\s*(\d+(?:\.\d+)?)(?:\s*/\s*\d+)?\s*$",
    re.MULTILINE,
)
"""A venue score line such as ``Rating: 8``, ``Soundness: 3`` or ``Overall: 4/5``."""

_ROUND_SUFFIX = re.compile(r"round(\d+)-review\.[^.]+$", re.IGNORECASE)
"""The round number in a review file named ``<stem>-round<N>-review.<ext>``."""

_TAIL_LINES = 15
"""Score lines are looked for in this many final lines of a review."""

_OVERRIDE = re.compile(r"\s*(\d+)\s*=\s*([a-z ]+?)\s*$", re.IGNORECASE)
"""One ``<round>=<verdict>`` item of ``loop_status``'s *overrides*."""


def _verdict(review: str) -> str:
    """Return the verdict opening the review's last ``Recommendation:`` line, or ``""``."""
    lines = _RECOMMENDATION.findall(review)
    if not lines:
        return ""
    m = _VERDICT_AT_START.match(lines[-1].lower())
    return m.group(1) if m else ""


def _scores(review: str) -> list[tuple[str, float]]:
    """Return the ``(name, value)`` score lines from the tail of the review."""
    tail = "\n".join(review.splitlines()[-_TAIL_LINES:])
    return [
        (name.strip(), float(value))
        for name, value in _SCORE_LINE.findall(tail)
        if name.strip().lower() != "recommendation"
    ]


def _review_paths(reviews: str) -> list[str]:
    """Expand *reviews*: a glob sorted by the round number in the names, or a list in its order."""
    if any(c in reviews for c in "*?["):
        paths = glob.glob(reviews.strip())
        return sorted(paths, key=_round_key)
    return [p.strip() for p in re.split(r"[\n,]", reviews) if p.strip()]


def _round_key(path: str) -> tuple[int, str]:
    """Sort key of a review path: its ``round<N>-review`` number, then the name."""
    m = _ROUND_SUFFIX.search(Path(path).name)
    return (int(m.group(1)) if m else 0, path)


def _overrides(overrides: str) -> dict[int, str]:
    """Parse ``"2=accept, 4=weak accept"`` into ``{2: "accept", 4: "weak accept"}``.

    Raises:
        ValueError: On an item that is not ``<round>=<verdict>`` with a known verdict.
    """
    parsed: dict[int, str] = {}
    for item in filter(str.strip, overrides.split(",")):
        m = _OVERRIDE.match(item)
        if m is None or m.group(2).lower() not in VERDICTS:
            raise ValueError(
                f"override {item.strip()!r} is not <round>=<verdict> with a verdict among "
                + ", ".join(VERDICTS)
            )
        parsed[int(m.group(1))] = m.group(2).lower()
    return parsed


def _progress(rank: int, scores: list[tuple[str, float]]) -> tuple[int, float]:
    """Return a round's progress key: the verdict rank, then the sum of its paper scores.

    Confidence lines measure the reviewer, not the paper, and are left out of the sum.
    """
    return rank, sum(value for name, value in scores if "confidence" not in name.lower())


def _decision(progress: list[tuple[int, float]], max_rounds: int) -> str:
    """Return the loop decision for the rounds' progress keys, first round first."""
    if progress[-1][0] == len(VERDICTS) - 1:
        return f"STOP: the review says {TARGET_VERDICT}; the target is reached."
    if len(progress) >= max_rounds:
        return f"STOP: {len(progress)} rounds reached the cap of {max_rounds}."
    if len(progress) >= 3:
        improved = [progress[i] > max(progress[:i]) for i in range(1, len(progress))]
        if not improved[-1] and not improved[-2]:
            return (
                "STOP: the last two rounds raised neither the recommendation nor the scores; "
                "the paper cannot be improved further with the evidence allowed. Report the "
                "best round."
            )
    return f"CONTINUE: revise the paper for round {len(progress) + 1}."


def loop_status(reviews: str, max_rounds: int = DEFAULT_MAX_ROUNDS, overrides: str = "") -> str:
    """Read the recommendation of every round's review and decide whether the loop goes on.

    Args:
        reviews: The review files in round order: an absolute glob such as
            ``/w/reports/paper-round*-review.txt`` (sorted by the ``round<N>`` number in
            the file names), or absolute paths separated by newlines or commas, kept in
            the order given.
        max_rounds: Rounds after which the loop stops whatever the verdict.
        overrides: ``<round>=<verdict>`` items separated by commas (``"2=accept"``) for
            reviews without a readable ``Recommendation:`` line, the verdict read by the
            coordinator from the score lines; empty for none.

    Returns:
        One line per round (``round N: <verdict> (rank k/6); scores: ...``), then either
        a ``CHECK`` line naming a review whose verdict is unreadable (no decision until
        the coordinator supplies it through *overrides*) or the decision: ``STOP`` at
        ``strong accept``, at the round cap, and when the last two rounds both failed to
        beat every earlier round on the verdict and then on the summed paper scores;
        ``CONTINUE`` otherwise.
    """
    paths = _review_paths(reviews)
    if not paths:
        return f"no review files match {reviews!r}; pass absolute paths"
    try:
        forced = _overrides(overrides)
    except ValueError as exc:
        return f"error: {exc}"
    lines, progress, unreadable = [], [], []
    for n, path in enumerate(paths, 1):
        file = Path(path).expanduser()
        if not file.is_file():
            return f"round {n}: no such file: {file}; fix the path and call loop_status again"
        review = file.read_text(encoding="utf-8", errors="replace")
        verdict, scores = forced.get(n) or _verdict(review), _scores(review)
        detail = "scores: " + (", ".join(f"{k}: {v:g}" for k, v in scores) or "none found")
        if verdict:
            rank = VERDICTS.index(verdict)
            progress.append(_progress(rank, scores))
            forced_note = " (verdict from overrides)" if n in forced else ""
            lines.append(
                f"round {n}: {verdict} (rank {rank}/{len(VERDICTS) - 1}){forced_note}; {detail}"
            )
        else:
            unreadable.append(n)
            lines.append(f"round {n}: no readable Recommendation line; {detail}")
    if unreadable:
        rounds = ", ".join(str(n) for n in unreadable)
        lines.append(
            f"CHECK: round {rounds} has no readable Recommendation line. Read the review's "
            "score lines against the venue's scale, decide the verdict, record it in the "
            f'log and call loop_status again with overrides="{unreadable[0]}=<verdict>".'
        )
    else:
        lines.append(_decision(progress, max_rounds))
    return "\n".join(lines)


def add_to_system_prompt() -> str:
    """Append the coordinator's rules to the default system prompt."""
    return SYSTEM_PROMPT


def add_to_tools() -> list[Any]:
    """Expose the task builders and the loop decision to the model."""
    return [writer_task, reviewer_task, loop_status]


def settings() -> dict[str, Any]:
    """A sequential coordinator with the full toolset that may run for a day.

    The rounds run as ``run_agent`` sub-tasks, so fan-out stays off; the
    ``full`` profile keeps ``run_agent`` available even when a reviewer
    dispatches the loop (the read-only ``review`` profile has none).  The
    ``timeout`` tells the dispatcher a ``/revise_and_review_paper`` run
    may take up to a day.
    """
    return {
        "use_web_tools": False,
        "is_parallel": False,
        "classify_tasks": False,
        "tool_profile": "full",
        "timeout": DISPATCH_TIMEOUT_SECONDS,
    }


