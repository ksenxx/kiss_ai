# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Paper-reviewing agent — reviews a research paper for a venue as a careful human reviewer would.

Typed into an idle tab as ``/review_paper <instructions>``, for example::

    /review_paper Review ~/Downloads/fable_sol.pdf for ICLR 2027 (ignore the page
    limit). Related work before September 2026. Under 700 words, to
    ./reports/fable_sol.txt.

    /review_paper Review ./papers/hydra/hydra_kv.pdf for USENIX ATC 2027; 1000 words;
    second opinion from gpt-5.6-sol.

the slash command dispatches this file as a Sorcar Extension Agent
through ``run_agent`` with the instructions as the task.  The task text
supplies the paper (a PDF, .tex, .md or .txt file), the
venue, the output path, the word limit, the cutoff date for the
related-work search and, optionally, a second model that checks the
review.  The SEA's ``system_prompt`` adds the reviewing rules (read
everything, search the related work, judge the novelty, pinpoint
problems by page and section, suggest how to fix them, write like
Strunk and White, no AI slop) to the default system prompt, so the
agent keeps the full Sorcar toolset, the browser tools (related work,
the venue's reviewer guidelines) and ``run_parallel`` (the read-only
second opinion).

The process follows the Stanford Agentic Reviewer
(https://paperreview.ai/tech-overview): a sanity check of the input,
related-work queries at several levels of specificity from three
perspectives (the paper's benchmarks and baselines, the same problem, the
same technique) run through the arXiv and Semantic Scholar APIs with the
web tools, metadata triage with a full read of the closest papers, and
scores on seven dimensions (originality, importance, support for the
claims, soundness, clarity, value, contextualization) from which the
overall rating follows.  The related-work search uses the browser and
Bash the agent already has; no search tool is bundled.

Two tools implement the mechanical steps:

* :func:`read_paper` returns the text of the paper, page by page, with
  the line numbers that submission templates print in the margin
  removed, so the model can quote page and section.
* :func:`check_review` runs the structure and AI-slop gates on the
  review text (word limit, the five parts in order, a 2 to 3 sentence
  summary, bullets under Strengths and Weaknesses, the seven dimension
  scores, named related work, the slop and reviewer-boilerplate lists)
  and lists every hit with its line number.

The SEA class's methods (``system_prompt()``, ``tools()``, ...) follow
the SEA contract in :mod:`kiss.agents.seas.base.base_sea`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.seas.write_paper.write_paper_sea import (
    _GATES,
    _duplicated_sentences,
    _hits,
    _section,
)

SECOND_OPINION_MODEL = "gpt-5.6-sol"
"""Model that checks the review against the paper when the task does not name one."""

DEFAULT_WORD_LIMIT = 700
"""Word limit of the review when the task does not name one."""

SYSTEM_PROMPT = f"""\
# Paper-reviewing agent

You are William Strunk Jr. and E. B. White and a senior computer scientist. Review a
research paper for a conference or journal as a careful human reviewer does: read it all,
search the related work, check the claims against the evidence, and tell the authors
what to fix. Write short, natural sentences.

## The task text

The task supplies the paper (PDF, .tex, .md or .txt; download a URL to `./tmp/` first).
Pass absolute paths to `read_paper` and `check_review`: they run in the daemon process,
not in the task's working directory. Use model names literally; never invent one.

## What a review must do

1. Judge the novelty; machine reviewers skip it (Liang et al. 2024; Shin et al. 2025).
   Name the closest prior work (authors, venue, year) and say in a sentence what the
   paper adds to each, or that it adds nothing.
2. Check the claims against the paper's own evidence: abstract and introduction against
   the tables, text numbers against the tables, conclusions against the experiments run,
   baselines against their own papers.
3. Pinpoint each problem: page, section, table or figure, sentence or number. "The
   evaluation is limited" is not a finding; "Table 3 reports the median of 3 runs with
   no spread, so the 4% gap between rows 2 and 4 may be noise" is.
4. For each weakness, name the fix: which experiment, baseline, analysis, or rewrite.
5. Flag machine-written prose in the paper (the AI-slop list below, plus coined
   capitalized concepts and per-run numbers in the abstract). Quote one or two examples
   with their page, not all.
6. No stock AI-reviewer complaints ("more baselines", "larger scale", "more ablations",
   "clarity could be improved", "the authors should consider", "it would be interesting
   to"). Name the specific baseline, scale, ablation, or unclear sentence and why it
   matters for the claim.
7. Weigh the paper against the venue's bar: read its reviewer instructions on the web
   and some recent related papers there to calibrate. Score the seven dimensions under
   Scores first; the venue's own scores (rating, confidence, soundness, presentation,
   contribution, or its own names) follow, one line each. Ignore the venue rules the
   task tells you to ignore.
8. Write for the authors as much as for the area chair: ask only for what fits before a
   rebuttal (one baseline, one ablation, one statistic), never a new study.

## Process

1. Sanity check: `read_paper` page 1. Record the title and, when shown, the authors and
   venue. If the file is not a research paper (a slide deck, a blank or garbled
   conversion, no claims to check), stop and report that.
2. Read it all, appendix included, with `read_paper`, a few pages per call. Write
   `./tmp/PAPER-NOTES.md`: the claims, each number the abstract and introduction rely
   on and its source, the baselines, suspected AI slop, open questions, and the word
   limit L (700, the `check_review` default, when the task names none).
3. Search the related work before judging novelty. Write 6 to 9 queries in
   `./tmp/information-<paper stem>.md`: three perspectives (the paper's benchmarks and
   baselines; other work on the problem; the technique in other settings) at three
   levels (the paper's phrasing; the problem in general terms; the technique in general
   terms). Run each with the web tools or `curl` through the APIs, which return compact
   metadata (title, authors, date, venue, citations, abstract):
   `https://api.semanticscholar.org/graph/v1/paper/search?query=<words>&fields=title,authors,year,venue,citationCount,externalIds,abstract&publicationDateOrYear=:<cutoff>&limit=10`
   (plain words, no hyphens; on a 429 wait ten seconds and retry once, then run the
   remaining queries through the arXiv API in one Bash loop, one request per three
   seconds) and
   `https://export.arxiv.org/api/query?search_query=all:%22<phrase>%22%20AND%20submittedDate:%5B190001010000%20TO%20<YYYYMMDD>2359%5D&max_results=10`
   (quote phrases, bare words are OR-ed; keep the brackets percent-encoded, `curl` globs
   raw ones; one call per three seconds). Then Google Scholar, DBLP, the venue's
   proceedings, and the paper's citations. Log each candidate with URL, date and a
   one-line verdict; keep at least 20 distinct sources, recent and highly cited first.
   For the 5 to 8 closest, note what to look for (overlapping claim, shared baseline,
   number to compare), download the PDF to `./tmp/` (`curl -L
   https://arxiv.org/pdf/<id> -o <path>`), read it with `read_paper`, and summarize it
   against the reviewed paper; the abstract suffices for the rest. Verify all cited
   arXiv ids in one Bash call. Cite no paper you could not open.
4. Score the seven dimensions in `./tmp/PAPER-NOTES.md`, 1 to 10 each with the page,
   table or prior work that justifies it. The Summary's judgment and the venue's rating
   follow from these seven, never the reverse.
5. Allot words from L: Summary 3 sentences (0.08 L), Strengths 3 bullets (0.12 L),
   Weaknesses 5 to 7 bullets of at most 0.05 L each (0.32 L), Detailed review 0.20 L,
   Scores 7 one-clause lines plus the venue's lines (0.13 L). Draft to 85% of L and
   write the file once; `Read` an existing file at the output path first, since `Write`
   and `Edit` refuse unread files. Plain text, this order, these headings on their own
   lines:

       Summary
       (2 to 3 sentences: what the paper does, its main result with its number, your
       judgment)

       Strengths
       - (one bullet per strength, specific to this paper)

       Weaknesses
       - (one bullet per weakness, with the page or table and the fix)

       Detailed review
       (paragraphs: the novelty judgment with the named prior work, the claims you
       checked and what you found, the AI-slop examples, the improvements in the order
       the authors should make them)

       Scores
       Originality: N/10. (one clause of evidence)
       Importance of the research question: N/10. ...
       Support for the claims: N/10. ...
       Soundness of the experiments: N/10. ...
       Clarity of the writing: N/10. ...
       Value to the community: N/10. ...
       Contextualization relative to prior work: N/10. ...
       (then one line per score the venue's form asks for)

6. Count with `wc -w`. Run the `check_review` tool with the word limit and fix everything
   it flags. Before step 7 the review must be at most 95% of L (950 for L = 1000), not
   L - 1: step 7 adds words. If over, cut whole sentences or bullets in one `Edit` pass;
   do not rewrite the whole file repeatedly.
7. When the task names a second model (default `{SECOND_OPINION_MODEL}`), have it check
   the review read-only through `run_parallel(tasks, model=<second model>,
   tool_profile="review")`. It lacks `read_paper`: give it the paper and review paths
   and say `pdftotext <paper> -` in Bash (or Read for a text file) yields the text. Ask:
   does every finding hold, are the named prior works real and described correctly, is
   any AI slop left? It must report only demonstrated problems with page evidence and
   invent none. Spend at most 50% of the budget here. If `run_parallel` is absent or
   answers "You are a reviewer sub-agent and may not spawn further reviewers", do not
   retry with `run_agent`: `set_model(<second model>)`, check every quoted sentence and
   number against the .tex or `pdftotext` text yourself, then `set_model` back before
   editing. With no second model, check on your own model and report that. Never finish
   unsuccessfully because `run_agent` or `run_parallel` is missing. Apply every confirmed
   finding (one `Write` when more than three sentences change, else back-to-back `Edit`
   calls), then `wc -w`, then `check_review`. Plan three `check_review` calls per run
   (after the draft, the trim, and the fixes); rerun one only after a failure, until it
   passes.
8. Reread as a human would; remove every sentence that reads machine-written. Git add
   the review file.

Tools: in a worktree (`git rev-parse --show-toplevel`), translate a task path under the
main checkout to the worktree path before any Bash, `read_paper` or `check_review`
call; the Bash guard rejects main-checkout paths. `run_commands_parallel` takes a JSON
array of strings: double every backslash, prefer `grep -E 'a|b'` over `\\|`, and run
multi-line Python through Bash with a heredoc.

## Style (Strunk and White)

- Short declarative sentences, one idea each. Vary their length; some paragraphs are
  two sentences.
- First person singular ("I checked", "I could not find"). Present tense for what the
  paper does, past tense for what you did.
- Omit needless words. No "remarkably", "dramatically", "particularly", "fundamentally",
  "essentially", "quite"; "significantly" only with a statistical test.
- No filler praise ("well-written", "well-motivated", "interesting", "promising",
  "thorough", "extensive", "solid", "impressive"). Say what is good and why it matters.
- No hedging chains ("may potentially", "could arguably", "it seems that", "somewhat").
  State what you found; when unsure, say what you checked and what you could not.
- Definite, specific, concrete: page numbers, table numbers, paper titles, years.
- Say each thing once. The detailed review expands a bulleted weakness; it does not
  repeat it in the same words.
- American spelling. Plain ASCII: straight quotes, hyphens, three dots.

## AI slop

The review must not read as machine written. Each item is fine once; in quantity they
are the tells.

1. Em dashes: zero. Use commas, parentheses, a colon, or a new sentence.
2. Antithesis: "not X but Y", "X, not Y", "rather than", "instead of". At most 5, only
   where the contrast is technical.
3. Performative honesty: "to be clear", "honestly", "I want to be clear", "it is worth
   stating", "I say this plainly". State the point and drop the commentary.
4. Aphoristic closers and zingers at the end of paragraphs, chiasmus, "the lesson is".
5. Setup question then answer ("Does the method scale? Section 5 says yes."), "Here is
   the thing", "The picture is this".
6. Colon-pivot sentences ("The problem is simple: ..."), one-line verdict sentences after
   a long sentence, cleft sentences ("What makes this work is").
7. Reviewer boilerplate: "Overall, this paper", "In summary", "That said", "While the
   paper", "The authors should consider", "It would be interesting to", "would benefit
   from", "I encourage the authors", "minor comments", "typos:" lists, "I look forward
   to the rebuttal".
8. Rule-of-three lists, bold-label lists, a parallel cadence in every bullet.
9. Slop vocabulary: delve, leverage, pivotal, crucial, testament, landscape, tapestry,
   showcase, underscore, intricate, meticulous, seamless, vibrant, realm, myriad, foster,
   comprehensive, notably, moreover, furthermore, additionally, garner, bolster,
   interplay, multifaceted, holistic, nuanced, groundbreaking, cutting-edge,
   transformative, invaluable, commendable, noteworthy, paramount, "shed light on",
   "valuable insights", "it is worth noting", "it is important to note", "serves as",
   "stands as", "plays a crucial role", "aligns with"; "precisely", "exactly", "genuine"
   as intensifiers. Commendable, meticulous and intricate are the measured tells of
   LLM-written peer reviews (Liang et al., ICML 2024).
10. Markdown and Unicode artifacts in a plain-text review: `**bold**`, `##` headings,
    curly quotes, the Unicode ellipsis, Unicode arrows.
11. Duplicated sentences.
12. Puffery and filler: a trailing ", highlighting ..." or ", reflecting ..." clause,
    "Let us dive into", "Importantly,", "Interestingly,", "opens new avenues", "holds
    great promise", "Despite these limitations", "studies show" without naming the study.

## Report back

The output path, the word count, the gate counts before and after, the seven dimension
scores, the prior works you named with their source URLs, each second-opinion finding
and what you did with it, and anything the paper claims that you could not verify.
"""
"""The reviewing rules, appended to the default system prompt."""

_HEADINGS = ("Summary", "Strengths", "Weaknesses", "Detailed review", "Scores")
"""The five parts of a review, in the order the rules require."""

_DIMENSIONS = (
    "Originality",
    "Importance of the research question",
    "Support for the claims",
    "Soundness of the experiments",
    "Clarity of the writing",
    "Value to the community",
    "Contextualization relative to prior work",
)
"""The seven dimensions scored under Scores (paperreview.ai/tech-overview), in order."""

_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+\S")
"""A bullet line: ``- x``, ``* x``, a Unicode bullet, or ``1. x``."""

_ABBREVIATIONS = ("et al", "e.g", "i.e", "cf", "vs", "Fig", "Figs", "Sec", "Eq", "Tab", "No", "pp")
"""Abbreviations whose period does not end a sentence."""

_SENTENCE_END = re.compile(
    r"(?<=[.!?])" + "".join(rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREVIATIONS) + r"\s+"
)
"""Sentence boundary: end punctuation then whitespace, except after an abbreviation."""

_REVIEW_GATES: tuple[tuple[str, str, int, str], ...] = tuple(
    gate for gate in _GATES if gate[0] != "draft and review talk"
) + (
    (
        "reviewer boilerplate",
        r"\b(?:overall, this paper|in summary|that said|while the paper|the authors should"
        r" consider|it would be interesting|would benefit from|i encourage the authors"
        r"|minor comments|typos|look forward to the rebuttal|more baselines|larger scale"
        r"|more ablations|clarity could be improved|limited evaluation|lacks clarity"
        r"|well[- ]written|well[- ]motivated|promising|impressive)\b",
        0,
        "name the specific baseline, table, sentence or experiment instead",
    ),
    (
        "hedging",
        r"\b(?:may potentially|could potentially|could arguably|it seems that|somewhat"
        r"|arguably|appears to be|to some extent)\b",
        -1,
        "state what you checked and what you found",
    ),
    ("markdown headings", r"(?m)^\s*#+\s*[A-Za-z]", 0, "plain-text headings on their own line"),
)
"""The word gates of a review: the paper gates minus "draft talk", plus reviewer tells."""


_MIN_MARGIN_LINES = 20
"""Length of a counting chain of bare-number lines before it is taken for margin numbers."""


class ReviewPaperSea(BaseSea):
    """The ``/review_paper`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/review_paper help``."""
        return (
            "Reviews a research paper (PDF, .tex, .md or .txt) for a venue like a careful human "
            "reviewer, searching the related work and writing a Summary/Strengths/Weaknesses/"
            "Detailed review/Scores (seven dimensions, 1 to 10) review that passes the "
            "word-limit and AI-slop gates; use it as "
            "`/review_paper Review <paper path> for <venue>; <word limit> words; to <output path>` "
            "or `run_agent(agent=\"review_paper\", task=...)`."
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Append the reviewing rules to the default system prompt."""
        return system_prompt + "\n\n" + SYSTEM_PROMPT

    def tools(self, tools: list[Any]) -> list[Any]:
        """Expose the paper reader and the review checker to the model."""
        return tools + [read_paper, check_review]

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Browse (related work), skip the classifier, wait two hours.

        Reading the paper, searching the related work and the second-opinion
        round take far longer than the default ``run_agent`` wait; the
        ``timeout`` tells the dispatcher how long a ``run_agent`` call on
        ``/review_paper`` should block before returning its job id.
        """
        return settings | {
            "use_web_tools": True,
            "auto_classify": False,
            "timeout": DISPATCH_TIMEOUT_SECONDS,
        }


def _strip_margin_numbers(text: str) -> str:
    """Drop the line numbers that submission templates print in the margin of a page.

    ``pdftotext`` emits them as lines holding a single 3- or 4-digit number,
    in several runs per page, and the runs continue one another (``000`` to
    ``007``, text, ``008`` to ``013``, ...).  Such a chain is dropped when it
    is at least ``_MIN_MARGIN_LINES`` long (a template prints about 54 per
    page); a table column of a few consecutive numbers is a chain of its own
    and survives.
    """
    lines = text.splitlines()
    chains: dict[int, list[int]] = {}
    for index, line in enumerate(lines):
        if re.fullmatch(r"\s*\d{3,4}\s*", line):
            number = int(line)
            chains[number + 1] = chains.pop(number, []) + [index]
    drop = {i for chain in chains.values() if len(chain) >= _MIN_MARGIN_LINES for i in chain}
    kept = [line for index, line in enumerate(lines) if index not in drop]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _pdf_pages(path: Path, first_page: int, last_page: int) -> tuple[list[str], int]:
    """Return the text of pages *first_page* to *last_page* of a PDF and its page count."""
    pdftotext, pdfinfo = shutil.which("pdftotext"), shutil.which("pdfinfo")
    if not pdftotext or not pdfinfo:
        raise FileNotFoundError("pdftotext not found: install poppler (brew install poppler)")
    info = subprocess.run([pdfinfo, str(path)], capture_output=True, text=True, timeout=60)
    found = re.search(r"^Pages:\s+(\d+)", info.stdout, flags=re.MULTILINE)
    if not found:
        raise ValueError(f"pdfinfo could not read {path}: {info.stderr.strip()}")
    total = int(found.group(1))
    last_page = min(last_page, total) if last_page > 0 else total
    if first_page > last_page:
        return [], total
    cmd = [pdftotext, "-f", str(first_page), "-l", str(last_page), str(path), "-"]
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=300)
    if out.returncode:
        raise ValueError(f"pdftotext failed on {path}: {out.stderr.strip()}")
    return out.stdout.split("\f")[: last_page - first_page + 1], total


def read_paper(paper_path: str, first_page: int = 1, last_page: int = 0) -> str:
    """Return the text of a paper, page by page, for quoting by page and section.

    A PDF is converted with ``pdftotext``; the line numbers that submission
    templates print in the margin (a line holding only a number) are dropped.
    A ``.tex``, ``.md`` or ``.txt`` file is returned as is, as one page.

    Args:
        paper_path: Absolute path of the PDF or text file (a relative path is
            resolved against the daemon's working directory, not the task's).
        first_page: First page to return (1-based).
        last_page: Last page to return (inclusive); ``0`` means the last page of
            the paper.  Ask for a few pages per call: a full paper exceeds the
            tool-output limit.

    Returns:
        ``paper: <path> (<N> pages)`` then ``=== page K ===`` and the text of
        each requested page; or an error message starting with ``error:``.
    """
    path = Path(paper_path).expanduser()
    if not path.is_file():
        return f"error: no such file: {path}"
    if path.suffix.lower() != ".pdf":
        text = path.read_text(encoding="utf-8", errors="replace")
        return f"paper: {path} (1 page)\n=== page 1 ===\n{text}"
    try:
        pages, total = _pdf_pages(path, max(first_page, 1), last_page)
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired) as exc:
        return f"error: {exc}"
    lines = [f"paper: {path} ({total} pages)"]
    for offset, page in enumerate(pages):
        lines.append(f"=== page {max(first_page, 1) + offset} ===")
        lines.append(_strip_margin_numbers(page))
    return "\n".join(lines)


def _unwrap(review: str) -> str:
    """Join the hard-wrapped lines of each paragraph so phrases and sentences are not split.

    The joined paragraph sits on its first source line and the lines it
    absorbed become empty, so the ``L<n>`` numbers of the gates still point
    into the file.  Bullets and headings start their own paragraph.
    """
    out: list[str] = []
    head = -1
    for line in review.splitlines():
        if not line.strip() or _heading(line):
            head = -1
        elif head < 0 or _BULLET.match(line):
            head = len(out)
        else:
            out[head] = f"{out[head].rstrip()} {line.strip()}"
            out.append("")
            continue
        out.append(line)
    return "\n".join(out)


def _heading(line: str) -> str | None:
    """Return the required heading that *line* is, or ``None``.

    Headings are matched case-insensitively on their own line; a trailing
    colon and Markdown ``#`` or ``**`` decorations are tolerated here (the
    markdown gates flag them).
    """
    bare = line.strip().strip("#*").strip().rstrip(":").strip().lower()
    return next((h for h in _HEADINGS if bare == h.lower()), None)


def _parts(review: str) -> dict[str, tuple[int, list[str]]]:
    """Split *review* at the required headings into ``{heading: (line number, body lines)}``.

    A missing heading is absent from the result; a repeated heading does not
    start a new part.
    """
    parts: dict[str, tuple[int, list[str]]] = {}
    current = ""
    for number, line in enumerate(review.splitlines(), 1):
        heading = _heading(line)
        if heading is not None and heading not in parts:
            current = heading
            parts[current] = (number, [])
        elif current:
            parts[current][1].append(line)
    return parts


def _structure(review: str) -> list[str]:
    """Return the structure gate lines: headings in order, summary length, bullets."""
    parts = _parts(review)
    present = [h for h in _HEADINGS if h in parts]
    lines = _section("headings missing", [h for h in _HEADINGS if h not in parts], 0)
    starts = [parts[h][0] for h in present]
    if starts != sorted(starts):
        where = ", ".join(f"{h} L{parts[h][0]}" for h in present)
        lines.append(f"headings out of order: {where} FAIL")
    if "Summary" in parts:
        text = " ".join(parts["Summary"][1]).strip()
        count = len([s for s in _SENTENCE_END.split(text) if s.strip()])
        lines.append(f"summary sentences: {count} (2 to 3) {_verdict(2 <= count <= 3)}")
    for heading in ("Strengths", "Weaknesses"):
        if heading in parts:
            bullets = sum(1 for line in parts[heading][1] if _BULLET.match(line))
            name = heading.lower()
            lines.append(f"{name} bullets: {bullets} (at least 1) {_verdict(bullets > 0)}")
    if "Detailed review" in parts:
        words = len(" ".join(parts["Detailed review"][1]).split())
        lines.append(f"detailed review words: {words} (at least 100) {_verdict(words >= 100)}")
    if "Scores" in parts:
        missing = _unscored_dimensions("\n".join(parts["Scores"][1]))
        note = "one '<dimension>: N/10' line per dimension, N from 1 to 10"
        lines += _section("dimensions unscored", missing, 0, note)
    return lines


def _unscored_dimensions(scores: str) -> list[str]:
    """Return the dimensions without a ``<name>: N/10`` line, N in 1 to 10, in *scores*."""
    missing = []
    for name in _DIMENSIONS:
        found = re.search(rf"(?im)^\s*{re.escape(name)}\s*:\s*(\d{{1,2}})\s*/\s*10\b", scores)
        if found is None or not 1 <= int(found.group(1)) <= 10:
            missing.append(name)
    return missing


def _verdict(ok: bool) -> str:
    """Render a boolean gate result as ``PASS`` or ``FAIL``."""
    return "PASS" if ok else "FAIL"


def check_review(review_path: str, word_limit: int = DEFAULT_WORD_LIMIT) -> str:
    """Run the structure and AI-slop gates on a plain-text review and return the report.

    Args:
        review_path: Absolute path of the review text file (a relative path is
            resolved against the daemon's working directory, not the task's).
        word_limit: The review must have fewer words than this (default
            ``DEFAULT_WORD_LIMIT``).

    Returns:
        One block per gate marked ``PASS``, ``FAIL`` or ``CHECK`` (list only):
        the word count, the five headings and their order, the summary's
        sentence count, the bullets under Strengths and Weaknesses, the size
        of the detailed review, the seven dimension scores under Scores, the
        years named (prior work), the word gates, duplicated sentences and
        non-ASCII lines; then the number of failed gates.
    """
    path = Path(review_path).expanduser()
    if not path.is_file():
        return f"no such file: {path}"
    review = path.read_text(encoding="utf-8", errors="replace")
    words = len(review.split())
    lines = [f"gates for {path}"]
    lines.append(f"words: {words} (under {word_limit}) {_verdict(words < word_limit)}")
    lines += _structure(review)
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", review)))
    status = "PASS" if years else "CHECK"
    detail = ", ".join(years) or "name the closest prior work with its year"
    proxy = "a rough proxy for named prior work"
    lines.append(f"years named: {len(years)} ({proxy}) {status}; {detail}")
    # The word gates run on unwrapped paragraphs so a phrase split by a hard line
    # break still counts; the line numbers name the paragraph's first line.
    unwrapped = _unwrap(review)
    for name, pattern, limit, note in _REVIEW_GATES:
        lines += _section(name, _hits(unwrapped, pattern), limit, note)
    lines += _section("duplicated sentences", _duplicated_sentences(unwrapped), 0)
    non_ascii = [
        f"L{i}: {line.strip()[:100]}"
        for i, line in enumerate(review.splitlines(), 1)
        if any(ord(c) > 127 for c in line)
    ]
    lines += _section("non-ASCII lines", non_ascii, -1, "plain ASCII unless quoting the paper")
    failed = sum(line.endswith(" FAIL") or " FAIL; " in line for line in lines)
    lines.append(f"summary: {failed} gate(s) failed")
    return "\n".join(lines)


DISPATCH_TIMEOUT_SECONDS = 7200
"""Seconds a ``run_agent`` call waits for a ``/review_paper`` run: two hours."""

