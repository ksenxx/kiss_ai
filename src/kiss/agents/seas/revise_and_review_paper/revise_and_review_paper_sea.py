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

The SEA class's methods (``system_prompt()``, ``tools()``, ...) follow
the SEA contract in :mod:`kiss.agents.seas.base.base_sea`.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea

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
"""Seconds a ``run_agent`` call waits for the whole loop (the ``timeout`` of :func:`settings`)."""

SYSTEM_PROMPT = f"""\
# Revise-and-review loop coordinator

Loop: `/write_paper` writes a research paper; `/review_paper` reviews it as a fresh
reviewer; repeat until the review says "{TARGET_VERDICT}" or the paper cannot improve.
Never write or review the paper yourself.

## The task text

Two blocks:

- writing instructions ("Writing:", "Write:", "Paper:"): venue, output path, topic,
  sources of truth, page limit, anonymity mode, house style, writer model, and the
  experiments, ablations or benchmarks the writer may run (the user's own, named
  public ones, or none);
- review instructions ("Review:", "Reviewing:"): venue and year, reviewer model, word
  limit (default {DEFAULT_WORD_LIMIT}), related-work cutoff date, venue rules to ignore.

Options after either block: round cap (default {DEFAULT_MAX_ROUNDS}), budget split, a
target below "{TARGET_VERDICT}". If the blocks are unclear, or the writing venue, output
path or topic of a new paper, or the review venue is missing,
ask the user once with `ask_user_question` before round 1. The children run unattended
and never ask.

## How the agents run

- writer: `run_agent(task=<writer_task output>, agent="{WRITE_PAPER_SEA}",
  timeout="{WRITER_TIMEOUT_SECONDS}", max_budget=<share>)`;
- reviewer: `run_agent(task=<reviewer_task output>, agent="{REVIEW_PAPER_SEA}",
  options='{{"use_memory": false}}', timeout="{REVIEWER_TIMEOUT_SECONDS}",
  max_budget=<share>)`.

Pass `model` when the user names one, literally. Never pass `chat_id` in `options`: each
round is a new session; the files carry the state. Pass `writer_task` and `reviewer_task`
outputs verbatim; add no earlier review, opinion or history to the reviewer's task. Both
agents share your working directory (`pwd` first; map user paths under the main checkout
to your worktree).

## A fresh review

The reviewer judges a blind submission. Before each review:

1. copy the built PDF alone to `<work dir>/tmp/revise_review/round-<N>/<stem>.pdf`;
2. delete earlier review notes under `<work dir>/tmp/` (`PAPER-NOTES.md`,
   `information-<stem>*.md`);
3. build the task with `reviewer_task(...)` (it forbids opening earlier reviews, notes,
   git history or the paper's source directory); run it with `use_memory="false"` and no
   `chat_id`.

Reviews go to `<work dir>/reports/<stem>-round<N>-review.txt`.

## The loop

1. Round 1: `writer_task(instructions, 1)`. If the instructions name an existing paper
   and a review to start from, skip to the review.
2. After each writing round find the PDF (reported by the writer; else the `.tex` path
   with `.pdf`). If missing or older than the `.tex`, run one writer round for the build.
3. Stage the PDF, review, then call
   `loop_status("<work dir>/reports/<stem>-round*-review.txt", max_rounds)` and obey it.
   On CHECK (no readable recommendation), read that review's score lines against the
   venue's scale, decide the verdict yourself, log it, and call `loop_status` again
   with `overrides="<round>=<verdict>"`.
4. On CONTINUE, triage each weakness: a writing fix; an allowed experiment, ablation or
   benchmark (the writer runs it, via the AI-discovery process of its system prompt when
   a stronger method or result is wanted); or a ruled-out point, stated in the paper as a
   limitation. Put ruled-out points, allowed benchmarks and the remaining budget into
   `coordinator_notes` of `writer_task(instructions, N + 1, review_path, coordinator_notes)`.
5. Stop early when every weakness is ruled out or was answered earlier with evidence the
   reviewer disputes on judgment, not fact; report that the paper cannot improve with the
   allowed evidence. A STOP for two rounds without improvement is final, unless the latest
   review raises a weakness no round attempted and the user allows its evidence: then run
   one more round, once per loop. The next STOP is final even if the review names new
   fixable points (every fresh reviewer does), unless the user's task says to address
   every comment first.
6. Never game the review: no removed negative results, no inflated claims, no mention of
   the review, round or earlier versions in the paper.

## Budget and time

Of the task budget B (in your task settings), keep 10% for coordination. Round 1's writer
gets at most 40% (25% when revising an existing paper); later writer rounds at most 20%;
each review at most 8%. Log each child's spend and steps from its `run_agent` result;
never query `history.db`. The `Budget:` figure in tool results already includes them.
Stop with the best paper when the remainder cannot fund a writer round and a review. A
timed-out `run_agent` stops the child: check what it finished on disk, then rerun it once
with a larger timeout.

## Log and report

Keep `<work dir>/reports/<stem>-revise-review-log.md`: per round, the recommendation and
score lines, each weakness and its outcome (fixed where, experiment and numbers, declined
why), the experiments run, the child's cost and time. Update
`<work dir>/tmp/revise_review/PROGRESS.md` each round (the writer overwrites
`./tmp/PROGRESS.md`). Git add the paper directory (`.tex`, `.bib`, figures, scripts, raw
results, `.pdf`; never `.aux`, `.bbl`, `.blg`, `.log`, `.out`), the reviews and the log;
the children commit nothing.

Report each round's recommendation and cost, the total cost, the final `.tex` and `.pdf`
paths and page count, every review path, the log path, why the loop stopped, and the open
weaknesses if the target was missed.
"""
"""The coordinator's rules, appended to the default system prompt."""

_WRITER_RULES = """\
Rules for this round:
- Run unattended: do not ask the user questions; when something is missing, choose the
  reading closest to the instructions and say so in your report.
- Pass absolute paths inside your working directory (`pwd` first) to `build_paper`,
  `check_paper` and every script.
- Edit and build: end the round with a PDF built from the current `.tex` that passes
  `build_paper` (zero errors, zero undefined references) and `check_paper`.
- Experiments: when the instructions or the review ask for evidence the paper lacks (a
  baseline, an ablation, a benchmark, a stronger result), run it on the benchmarks the
  instructions name; when they name none, use the area's standard public benchmarks and
  say which in the paper. Use the AI-discovery process of your system prompt when asked
  for a stronger method or result. Commit the scripts and the raw results next to the
  paper; every new number enters the paper through its macros.
- Never remove a negative result, inflate a claim, or mention a review, a round or an
  earlier version in the paper. State a point you cannot answer with evidence as a
  limitation in the paper.
- Report back: the `.tex` and `.pdf` paths, the page count, the gate counts, each
  experiment run with its numbers, and (for a revision) one line per weakness of the
  review: fixed (where), experiment run (numbers), or declined (why).
- Do not run a review of your own in this round; an independent reviewer follows.
- Stay within this session's budget; when it runs short, finish with the paper building
  and report what is left undone.
"""


class ReviseAndReviewPaperSea(BaseSea):
    """The ``/revise_and_review_paper`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/revise_and_review_paper help``."""
        return (
            "Writes a research paper with /write_paper from your writing instructions, has "
            "/review_paper review it as a fresh reviewer under your review instructions, and "
            "repeats the revise-and-review loop (with experiments, ablations or AI discovery when "
            "the review asks for evidence) until the review says strong accept or the paper cannot "
            "improve further; use `/revise_and_review_paper Writing: <...> Review: <...>`."
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Append the coordinator's rules to the default system prompt."""
        return system_prompt + "\n\n" + SYSTEM_PROMPT

    def tools(self, tools: list[Any]) -> list[Any]:
        """Expose the task builders and the loop decision to the model."""
        return tools + [writer_task, reviewer_task, loop_status]

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A sequential coordinator with the full toolset that may run for a day.

        The rounds run as ``run_agent`` sub-tasks; the ``full`` profile
        keeps ``run_agent`` available even when a reviewer
        dispatches the loop (the read-only ``review`` profile has none).  The
        ``timeout`` tells the dispatcher a ``/revise_and_review_paper`` run
        may take up to a day.
        """
        return settings | {
            "use_web_tools": False,
            "auto_classify": False,
            "tool_profile": "full",
            "timeout": DISPATCH_TIMEOUT_SECONDS,
        }


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


