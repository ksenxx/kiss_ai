# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Patent-reviewing agent — examines a patent draft or a published patent as an examiner would.

The ``/review_patent`` SEA reads the application (``read_patent``: PDF, Markdown
or text, page by page), searches the prior art on Google Patents, Espacenet, the
USPTO and the literature with the web tools, runs the deterministic claim gates
of ``/write_patent`` on a Markdown draft (``check_patent``), and writes a
plain-text review shaped like an Office action: Summary, Claims at issue, Prior
art, Eligibility, Novelty and obviousness, Definiteness and support, Drafting
defects, Recommendations, Scores.  ``check_patent_review`` gates the review:
headings and order, a reference with a number or URL per prior-art bullet,
a claim number per finding, scores for every dimension, and the AI-slop and
reviewer-boilerplate gates shared with ``/review_paper``.  The review is an
opinion for a registered practitioner, not legal advice.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.seas.review_paper.review_paper_sea import (
    _BULLET,
    _REVIEW_GATES,
    _SENTENCE_END,
    _parts,
    _unwrap,
    _verdict,
    read_paper,
)
from kiss.agents.seas.write_paper.write_paper_sea import _duplicated_sentences, _hits, _section
from kiss.agents.seas.write_patent.write_patent_sea import check_patent

DEFAULT_WORD_LIMIT = 2500
"""Words a review may have unless the task says otherwise."""

_HEADINGS = (
    "Summary",
    "Claims at issue",
    "Prior art",
    "Eligibility",
    "Novelty and obviousness",
    "Definiteness and support",
    "Drafting defects",
    "Recommendations",
    "Scores",
)
"""The nine parts of a patent review, in order."""

_DIMENSIONS = (
    "Claim clarity",
    "Specification support",
    "Novelty over the art found",
    "Non-obviousness",
    "Eligibility",
    "Claim scope and fallbacks",
    "Drafting quality",
)
"""The seven dimensions scored under Scores, each ``<name>: N/10``."""

_REFERENCE = re.compile(
    r"\b(?:US|EP|WO|JP|CN|KR|DE|GB)\s?-?\s?\d{4}[/-]?\d{3,}|\bUS\s?\d{1,2},?\d{3},?\d{3}\b"
    r"|https?://\S+|\barXiv[: ]\s?\d{4}\.\d{4,5}|\bdoi[: ]|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
"""A patent number, URL, arXiv id, DOI or year: what makes a prior-art bullet verifiable."""

_FINDING_PARTS = ("Eligibility", "Novelty and obviousness", "Definiteness and support")
"""Parts whose findings must each name a claim."""

_GATE_NOTES = {
    "vague attribution": "name the reference (number, title, page or column) in the same sentence",
}
"""Notes of the ``/review_paper`` word gates reworded for a patent review (no ``\\cite``)."""

SECOND_OPINION_MODEL = "gpt-5.6-sol"
"""The model that checks the review when the task names no other (``/review_paper`` uses it too)."""

SYSTEM_PROMPT = f"""\
# Patent-reviewing agent

You ARE the review agent: you examine a patent application draft (or a published
application or granted patent) the way a USPTO examiner and a prosecuting attorney would,
and you write the review to the path the task names (plain text; default
``review.txt`` next to the application).  The review is an opinion for a registered
practitioner, never legal advice; say so once, in the Summary's last sentence.  Rules
below come from 37 CFR 1.75, MPEP 2106 (eligibility, with the 2024 AI update, the
Aug 4, 2025 memo and Ex parte Desjardins, precedential Nov 4, 2025), MPEP 2131/2141
(novelty, obviousness: Graham factors, KSR rationales), MPEP 2163/2164 (written
description, enablement), MPEP 2173 (definiteness, antecedent basis) and MPEP 2181
(means-plus-function), opened on uspto.gov on 2026-10-27.

## Tools

- ``read_patent(path, first_page, last_page)``: the application page by page (PDF via
  pdftotext; ``.md``/``.txt`` as one page).  Read the whole document; claims first, then
  the description against each claim element.
- ``check_patent(draft_path, jurisdiction)``: the deterministic proofreader gates on a
  Markdown draft (numbering, one sentence, dependencies, antecedent basis, 112(f)
  triggers, relative terms, fees, terms absent from the description, figures, invention
  talk, prior-art admissions).  Run it on every ``.md`` draft and copy its FAIL lines
  into Drafting defects; on a PDF do the same checks by hand on the claims.
- ``check_patent_review(review_path, word_limit)``: the gates on your review; zero
  FAIL lines before you finish.
- The web tools for prior art (no key-gated API is needed): Google Patents
  (``https://patents.google.com/?q=(terms)&before=priority:YYYYMMDD`` filtered to the
  application's earliest priority date, then ``https://patents.google.com/patent/<number>/en``
  for the claims, description and cited-by), Espacenet
  (``https://worldwide.espacenet.com/patent/search?q=...``), USPTO Patent Public Search
  (``https://ppubs.uspto.gov/pubwebapp/``), Lens
  (``https://www.lens.org/lens/search/patent/list?q=...``), and arXiv / Semantic Scholar for
  papers (``https://export.arxiv.org/api/query?search_query=all:...``,
  ``https://api.semanticscholar.org/graph/v1/paper/search?query=...``
  ``&fields=title,year,externalIds,abstract``).

## Process

1. Sanity check: ``read_patent`` page 1 and the claims.  Note the earliest priority date,
   the jurisdiction, the statutory classes claimed (method, system, medium, composition)
   and the independent claims.  Paraphrase each independent claim as an element list
   (``1[a]``, ``1[b]``, ...) in Claims at issue; every later finding cites these labels.
2. Prior-art search from three angles: the problem solved, the claimed mechanism, and the
   field's standard components; at least 20 distinct references opened, at least 8 of
   them patents or applications, the closest 5 read in full (claims and the relevant
   passages).  For each of the closest references write a bullet under Prior art: number
   or URL, date (it must precede the priority date to count under 102; say so when it
   does not), assignee or authors, what it teaches, and which claim elements it meets.
   Element-by-element: a reference "discloses 1[a], 1[b], 1[d]; not 1[c]".
3. Eligibility (35 U.S.C. 101), per independent claim: Step 1 statutory class; Step 2A
   prong one: does the claim recite a judicial exception (name it: mathematical concept,
   mental process that can practically be performed in the mind, method of organizing
   human activity) or only involve one; prong two: is the exception integrated into a
   practical application, and which claim words and which description paragraphs state
   the technical improvement (Enfish, Desjardins); Step 2B only if needed.  A rejection
   needs a preponderance of evidence; say which way it leans and why.
4. Novelty (102) and obviousness (103), per independent claim: anticipation needs one
   reference with every element arranged as claimed; otherwise the Graham factors (scope
   of the art, differences, level of skill) and the KSR rationale that would combine
   the references, with the motivation stated and the missing element named.  Then the
   dependent claims that would survive and why.
5. Definiteness and support (112): for each claim, terms of degree without a standard
   in the description, antecedent-basis gaps, ``means for`` limitations and the structure
   the description gives them (or does not), claim terms the description never uses,
   functional claiming at the point of novelty, new matter if the task gives a parent,
   enablement of the full claimed range (Wands factors), written-description support for
   each independent claim element with the paragraph that supports it.  Every finding:
   claim, the exact words, why, and the fix.
6. Drafting defects: the ``check_patent`` FAIL lines, plus title, abstract (150 words,
   no "said"/"means"), "the present invention", prior-art admissions, absolute words,
   inconsistent element names or reference numerals, figures not described, claim
   count against the 3/20 fee thresholds, dependent claims that do not further limit.
7. Recommendations: numbered, concrete, in priority order: rewritten claim text where a
   claim should change (full claim, not a hint), description paragraphs to add, art to
   cite in the IDS (duty of candor, 37 CFR 1.56), questions for the inventor.
8. Write the review (absolute paths: a relative path resolves against the daemon's
   directory, not yours; ``Read`` an existing review before overwriting it) and run
   ``check_patent_review`` with the task's word limit.  Budget the words before drafting:
   Summary 0.06 L, Claims at issue 0.12 L, Prior art 0.18 L, Eligibility 0.12 L, Novelty
   and obviousness 0.2 L, Definiteness and support 0.14 L, Drafting defects 0.06 L,
   Recommendations 0.1 L, Scores the rest; be at 95% of L before step 9, which adds words.
   If over, cut whole bullets in one ``Edit`` pass; do not rewrite the file repeatedly.
9. Second opinion.  When the task names a second model (default ``{SECOND_OPINION_MODEL}``),
   have it check the review read-only through ``run_parallel(tasks, model=<second model>,
   tool_profile="review")``: give it the application and review paths, say that
   ``pdftotext <pdf> -`` in Bash (or ``Read`` for a text file) yields the text, and ask
   whether every finding holds against the document, whether each cited reference exists
   and is described correctly, and whether any AI slop is left; it must report only
   demonstrated problems with page or claim evidence and invent none.  Spend at most 50%
   of the budget here.  If ``run_parallel`` is absent or refuses (you are a sub-agent), do
   not retry with ``run_agent``: ``set_model(<second model>)``, check every quoted claim
   word and reference yourself, then ``set_model`` back before editing.  With no second
   model, check on your own model and say so.  Never finish unsuccessfully because a
   dispatch tool is missing.  Apply every confirmed finding, then ``check_patent_review``
   again; plan three such calls per run and fix every FAIL.
10. Finish: report the ``check_patent_review`` summary line verbatim together with the
   three most important findings and the references you could not open.

## Output

Plain text, these headings on their own lines, in this order:

   Summary                     (3 to 5 sentences: what is claimed, the field, the closest art,
                                the verdict, the "opinion for a practitioner" sentence)
   Claims at issue             (independent claims as labeled element lists)
   Prior art                   (one bullet per reference: number or URL, date, who, teaches,
                                elements met)
   Eligibility                 (per independent claim: Step 1, 2A prongs one and two, 2B)
   Novelty and obviousness     (per independent claim, then the surviving dependents)
   Definiteness and support    (per claim: term, reason, fix)
   Drafting defects            (list)
   Recommendations             (numbered)
   Scores
   {_DIMENSIONS[0]}: N/10. <one sentence>
   {_DIMENSIONS[1]}: N/10. ...
   {_DIMENSIONS[2]}: N/10. ...
   {_DIMENSIONS[3]}: N/10. ...
   {_DIMENSIONS[4]}: N/10. ...
   {_DIMENSIONS[5]}: N/10. ...
   {_DIMENSIONS[6]}: N/10. ...
   Allowance likelihood: N/10 (as filed, before amendment)
   Confidence: N/5

Every finding under Eligibility, Novelty and obviousness, and Definiteness and support
names a claim (``claim 1``).  Every prior-art bullet carries a number, URL or year.
Under {DEFAULT_WORD_LIMIT} words unless the task says otherwise.  No Markdown headings
or bold, no em dashes, no reviewer boilerplate ("the applicant should consider"), no
hedging: state what you checked and what you found.  Quote claim words exactly.

## Honesty

A reference you did not open is not in the review.  A rejection you cannot support
element by element is not a rejection; write "no art found for 1[c] in 20 references"
instead.  Breadth is not indefiniteness (MPEP 2173.04).  If the task mentions a second
model's opinion, you are still the only author of this review: read it, verify each
of its findings against the document, and keep only the ones you verified.
"""


def read_patent(patent_path: str, first_page: int = 1, last_page: int = 0) -> str:
    """Return the text of a patent or draft, one ``=== page N ===`` block per page.

    Args:
        patent_path: Path of a PDF (text extracted per page with pdftotext, layout
            kept so claims and tables stay readable), or a ``.md``/``.txt`` draft
            returned as a single page.
        first_page: First page to return (1-based; below 1 is clamped to 1).
        last_page: Last page to return; ``0`` means the end of the document.

    Returns:
        ``patent: <path> (N pages)`` followed by the requested pages, or an
        ``error: ...`` line when the file is missing, unreadable or pdftotext is
        not installed.
    """
    return read_paper(patent_path, first_page, last_page).replace("paper:", "patent:", 1)


def _bullets(body: list[str], first_line: int) -> list[tuple[int, str]]:
    """Return ``(line, text)`` for each bullet of *body*, a wrapped bullet joined into one text."""
    bullets: list[tuple[int, str]] = []
    for number, line in enumerate(body, first_line):
        if _BULLET.match(line):
            bullets.append((number, line.strip()))
        elif bullets and line.strip():
            bullets[-1] = (bullets[-1][0], bullets[-1][1] + " " + line.strip())
    return bullets


def _structure(review: str) -> list[str]:
    """Return the structure gates: headings, summary length, bullets, claim citations, scores."""
    parts = _parts(review, _HEADINGS)
    present = [h for h in _HEADINGS if h in parts]
    lines = _section("headings missing", [h for h in _HEADINGS if h not in parts], 0)
    starts = [parts[h][0] for h in present]
    if starts != sorted(starts):
        where = ", ".join(f"{h} L{parts[h][0]}" for h in present)
        lines.append(f"headings out of order: {where} FAIL")
    if "Summary" in parts:
        text = " ".join(parts["Summary"][1]).strip()
        count = len([s for s in _SENTENCE_END.split(text) if s.strip()])
        lines.append(f"summary sentences: {count} (3 to 5) {_verdict(3 <= count <= 5)}")
    if "Prior art" in parts:
        bullets = _bullets(parts["Prior art"][1], parts["Prior art"][0] + 1)
        lines.append(f"prior-art bullets: {len(bullets)} (at least 1) {_verdict(bool(bullets))}")
        bare = [
            f"L{number}: {text[:100]}" for number, text in bullets if not _REFERENCE.search(text)
        ]
        lines += _section("prior-art bullets without a number, URL or year", bare, 0)
    for heading in _FINDING_PARTS:
        if heading in parts:
            body = " ".join(parts[heading][1])
            claims = len(re.findall(r"\bclaims?\s+\d+", body, flags=re.IGNORECASE))
            lines.append(
                f"{heading.lower()} claim citations: {claims} (at least 1) {_verdict(claims > 0)}"
            )
    if "Recommendations" in parts:
        numbered = sum(
            1 for line in parts["Recommendations"][1] if re.match(r"\s*\d+[.)]\s+\S", line)
        )
        lines.append(f"recommendations numbered: {numbered} (at least 1) {_verdict(numbered > 0)}")
    if "Scores" in parts:
        scores = "\n".join(parts["Scores"][1])
        missing = []
        for name in _DIMENSIONS:
            found = re.search(rf"(?im)^\s*{re.escape(name)}\s*:\s*(\d{{1,2}})\s*/\s*10\b", scores)
            if found is None or not 1 <= int(found.group(1)) <= 10:
                missing.append(name)
        if not re.search(r"(?im)^\s*Allowance likelihood\s*:\s*(?:10|[1-9])\s*/\s*10\b", scores):
            missing.append("Allowance likelihood (N/10)")
        if not re.search(r"(?im)^\s*Confidence\s*:\s*[1-5]\s*/\s*5\b", scores):
            missing.append("Confidence (N/5)")
        lines += _section("dimensions unscored", missing, 0, "one '<dimension>: N/10' line each")
    return lines


def check_patent_review(review_path: str, word_limit: int = DEFAULT_WORD_LIMIT) -> str:
    """Run the structure and AI-slop gates on a plain-text patent review.

    Args:
        review_path: Path of the review text file.
        word_limit: The review must have fewer words than this (default
            ``DEFAULT_WORD_LIMIT``).

    Returns:
        One block per gate marked ``PASS``, ``FAIL`` or ``CHECK``: the word count,
        the nine headings and their order, the Summary's sentence count, the
        prior-art bullets and whether each carries a number, URL or year, a claim
        citation in each finding part, numbered recommendations, the seven
        dimension scores plus allowance likelihood and confidence, the word
        gates shared with ``/review_paper``, duplicated sentences and non-ASCII
        lines; then ``summary: N gate(s) failed``.
    """
    path = Path(review_path).expanduser()
    if not path.is_file():
        return f"no such file: {path}"
    review = path.read_text(encoding="utf-8", errors="replace")
    words = len(review.split())
    lines = [
        f"gates for {path}",
        f"words: {words} (under {word_limit}) {_verdict(words < word_limit)}",
    ]
    lines += _structure(review)
    unwrapped = _unwrap(review, _HEADINGS)
    for name, pattern, limit, note in _REVIEW_GATES:
        lines += _section(name, _hits(unwrapped, pattern), limit, _GATE_NOTES.get(name, note))
    lines += _section("duplicated sentences", _duplicated_sentences(unwrapped), 0)
    non_ascii = [
        f"L{i}: {line.strip()[:100]}"
        for i, line in enumerate(review.splitlines(), 1)
        if any(ord(c) > 127 for c in line)
    ]
    lines += _section(
        "non-ASCII lines", non_ascii, -1, "plain ASCII unless quoting the application"
    )
    failed = sum(line.endswith(" FAIL") or " FAIL; " in line for line in lines)
    lines.append(f"summary: {failed} gate(s) failed")
    return "\n".join(lines)


class ReviewPatentSea(BaseSea):
    """The ``/review_patent`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/review_patent help``."""
        return (
            "Reviews a patent application draft or a published patent like an examiner: prior-art "
            "search, 101/102/103/112 analysis claim by claim, drafting defects, rewritten claims; "
            "use `/review_patent <path, jurisdiction, review path>` or "
            'run_agent(agent="review_patent").'
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Append the patent-reviewing rules to the default system prompt."""
        return system_prompt + "\n\n" + SYSTEM_PROMPT

    def tools(self, tools: list[Any]) -> list[Any]:
        """Expose the page reader, the draft gates and the review gates to the model."""
        return tools + [read_patent, check_patent, check_patent_review]

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Browse for prior art, skip the classifier, route through the bestrouter."""
        return settings | {
            "use_web_tools": True,
            "auto_classify": False,
            "timeout": DISPATCH_TIMEOUT_SECONDS,
            "model": "bestrouter",
        }


DISPATCH_TIMEOUT_SECONDS = 7200
"""Seconds a ``run_agent`` call waits for a ``/review_patent`` run: two hours."""
