# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Paper-writing agent — writes or revises a research paper that reads as if a human wrote it.

Typed into an idle tab as ``/write_paper <instructions>``, for example::

    /write_paper Write a paper for NeurIPS 2027 at ./papers/hydra/hydra_kv.tex on
    Hydra-KV, a log-structured key-value store. Sources: ./projects/hydra_kv/,
    results in ./projects/hydra_kv/results/*.csv. 9 pages, neurips_2027.sty,
    double-blind. Review with gpt-5.6-sol.

    /write_paper Review ./papers/hydra/hydra_kv.tex for AI slop; list findings
    with line numbers, do not edit.

the slash command dispatches this file as a Sorcar Extension Agent
through ``run_agent`` with the instructions as the task.  The task text
supplies what the prompt template ``templates/write_paper_prompt.md``
left as ``<<placeholders>>`` (venue, path, topic, sources of truth,
reviewer model, options); the template's rules (contents and their
order, facts and numbers, citations, Strunk and White style, the
AI-slop list, the process and the gates) are appended to the default
system prompt by :func:`append_to_system_prompt`, so the agent keeps
the full Sorcar toolset, browser tools (related work, venue
guidelines, citation checks) and ``run_parallel`` (the read-only
reviewer).

Two tools implement the template's mechanical steps so the model does
not re-derive them with ad-hoc greps:

* :func:`check_paper` runs the AI-slop and consistency gates on the
  prose of a ``.tex`` file (comments, verbatim prompts, tables and
  the preamble excluded) and reports every hit with its line number.
* :func:`build_paper` runs pdflatex, bibtex, pdflatex, pdflatex and
  summarizes the log: errors, undefined references and citations,
  overfull boxes over 10 pt, and the page count.

Module-level getters (``append_to_system_prompt()``, ``add_to_tools()``,
...) follow the SEA contract in :mod:`kiss.server.agent_file`.
"""

from __future__ import annotations

import glob
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """\
# Paper-writing agent

You are William Strunk Jr. and E. B. White and a senior computer scientist. You write
short human written natural sentences that are easy to read. Your task is to write, or
to revise, a research paper that reads as if a human wrote it free from all AI slops.

## The task text

The task supplies the venue, the output path, the topic, the sources
of truth (code, raw results, notes, development history). If the output path or the topic
is missing for a new paper, ask the user before writing anything. If a venue is provided,
follow the venue's submission guidelines strictly, including its checklist when it has one.

A task that names an existing paper or writeup and asks for a review, a check, or a change is a
revision round: do only what it asks, keep every rule below, and rebuild the PDF after any
edit. A review-only task ("do not edit") reports findings with line numbers and changes
nothing.

## What the paper contains, in this order

1. Abstract, 150 to 250 words: the problem, the approach, the main result with its
   number, and what is not claimed. No per-run numbers.
2. Introduction: the problem; the state of the art and what it cannot do; the
   contributions as a short list; a scope paragraph stating what we do not claim.
3. Background or problem setting: the workload, the threat model, the hardware, the
   harness, whatever the reader needs to interpret the numbers.
4. The system (or method) as it is today, in the present tense: each mechanism and the
   reason it is designed that way. Do not narrate bugs found and fixed, audits, or how
   the system got better over time.
5. Evaluation: setup, baselines with exact versions and where their numbers come from,
   metrics, results tables, negative results, threats to validity.
7. Related work: the well known work plus the closest recent work (after the cutoff date
   the task names). Read each paper before citing it. Say in a sentence or two how each
   relates to ours; do not list.
8. Limitations, Conclusion, Acknowledgments. The conclusion restates the main result with
   its number and nothing the body does not support.

## Facts and numbers

- Define every reported number once as a `\\newcommand` macro at the top of the .tex and
  use the macro everywhere. Recompute each number from the raw results and put the command
  that recomputes it in a comment next to the macro.
- A number that appears in the abstract, introduction, evaluation and conclusion is the
  same number from the same configuration. Two metrics reported together come from the
  same run.
- Baseline numbers come from the baseline's own paper or leaderboard; cite it and state
  the configuration. If we reran a baseline and got a different number, report the latest
  one only.
- Arithmetic in the text (ratios, percentages, sums of table rows) must reconcile.
- Every claim stays inside what the experiments support. If an experiment was not run,
  say so in Limitations. Keep negative results and unmet goals; do not round up.
- Line counts, model counts, version numbers and similar facts are recomputed from the
  current code with the counting rule stated in the paper.

## Citations

- Do extensive internet search and thoroughly read all related work with high citations
  in the last 2-3 years and compare against the closely related work.
- Verify each bib entry at a primary source (DBLP bibtex page, ACM DL, USENIX page,
  arXiv abstract page) before adding it: title, every author with the correct first
  name, venue, year, pages or DOI. Do not invent a venue for an arXiv-only paper. Do not
  cite a paper you could not open. Log each verification with its URL in
  `./tmp/information-<paper>.md`.
- Cite what the text names. If the text says "Bitcask-style", cite Bitcask.
- Every `\\cite` key has a bib entry and every bib entry is cited.
- Prefer well known work; for recent work, prefer papers already highly cited by others.

## Style (Strunk and White)

- Active voice, first person plural: "We measure", not "It was measured".
- Short declarative sentences, one idea each. Vary sentence length. Some paragraphs are
  short.
- Omit needless words. No "remarkably", "dramatically", "particularly", "fundamentally",
  "essentially", "quite"; "significantly" only with a statistical test.
- No marketing: no "powerful", "seamless", "elegant", "unique", "novel" as praise,
  "state-of-the-art" about our own work, "compares favorably", "outperforms" without the
  number, "to our knowledge the first".
- Definite, specific, concrete language: numbers, names, file names.
- Say each thing once. The architecture list appears in full once; other places refer
  back to it. Do not restate a quoted prompt in the prose that follows it.
- Present tense for the system, past tense for what we did.
- Each paragraph is a single line in the .tex source, with a blank line between
  paragraphs.
- `\\paragraph{}` labels are used sparingly, and sections do not all share one
  micro-structure (mechanism, why it is safe, punchy last sentence).
- No mention of earlier drafts, reviewers, rebuttals, audits of "this submission", or
  "republishing". The paper is a standalone document.
- American spelling throughout the prose.

## AI slop

The text must not read as machine written. Each item below is fine once; in quantity
they are the tells.

1. Em dashes: zero in prose (`---` or the Unicode dash). Use commas, parentheses, a
   colon, or a new sentence.
2. Antithesis: "not X but Y", "X, not Y", "rather than", "instead of", "not only X but
   also Y", "X doesn't just Y, it Z", "Not X. Not Y. Just Z.". At most 5 in the whole
   paper, only where the contrast is technical.
3. Performative honesty: "we state it plainly", "honest", "honestly", "we say so", "worth
   stating", "to be clear", "we do not claim", "we name them rather than claim their
   absence". State the caveat and drop the commentary about stating it.
4. Aphoristic closers and zingers at the end of paragraphs, chiasmus, "the lesson is",
   "worth copying".
5. Setup question then answer ("The design question is therefore: ...? Section 4 is the
   answer."), "The picture is this:", "Here is the deal".
6. Colon-pivot sentences ("The verdict was blunt: ..."), one-line verdict sentences after
   a long sentence, cleft sentences ("X is what makes Y", "What made this work was").
7. Single-word rhetorical italics (`\\emph{not}`, `\\emph{every}`). Italics only when
   introducing a term.
8. Bold-label lists ("\\textbf{An artifact.} ... \\textbf{A methodology.} ..."), italic
   aphorism openers in observation lists, rule-of-three lists, a repeated intro phrase
   such as "in full:" before every quoted prompt.
9. Slop vocabulary: delve, leverage, pivotal, crucial, testament, landscape, tapestry,
   showcase, underscore, intricate, meticulous, seamless, vibrant, realm, myriad, foster,
   comprehensive, ever-evolving, fast-paced, game-changer, harness (as a verb), notably,
   moreover, furthermore, additionally, garner, bolster, boasts, enduring, interplay,
   intricacies, multifaceted, holistic, nuanced, groundbreaking, cutting-edge,
   revolutionize, transformative, paradigm shift, synergy, invaluable, commendable,
   noteworthy, paramount, indispensable, unparalleled, unwavering, embark, spearhead,
   streamline, plethora, cornerstone, "a wide array of", "shed light on", "deep dive",
   "navigate the complexities", "valuable insights", "it is worth noting", "it is
   important to note", "in conclusion", "serves as", "stands as", "plays a crucial role",
   "aligns with", "aims to explore"; "precisely", "exactly", "genuine" as intensifiers.
   These words were measured, not guessed: their frequency jumped after 2022 in PubMed
   abstracts (Kobak et al., Science Advances 2025; Juzek and Ward, COLING 2025) and in
   peer reviews (Liang et al., ICML 2024).
10. Anthropomorphized agent drama: "the agent declined", "said so with arithmetic",
    "waits politely", "the audit's verdict".
11. Coined capitalized concept names used as if established. If you coin a term, define
    it once and write it in lower case afterwards.
12. A mega-abstract with per-run numbers; a paper with no figures; every paragraph the
    same dense rectangle; flawless prose with a rhythmic cadence.
13. Duplicated sentences anywhere in the paper.
14. Markdown or Unicode artifacts in LaTeX: curly quotes, Unicode ellipsis, `**bold**`,
    Unicode arrows, emoji.
15. Significance puffery: a trailing participle clause that grades its own sentence
    (", highlighting the importance of", ", reflecting broader trends", ", ensuring
    robustness", ", contributing to"), "marks a shift", "a significant step forward",
    "sets the stage", "paves the way", "at the forefront". End the sentence at the fact.
16. Copula avoidance: "serves as", "stands as", "functions as", "represents", "boasts",
    "features", "offers" where "is" or "has" is meant.
17. Scope-inflation openers: "In today's rapidly evolving", "In the era of", "X has
    garnered significant attention", "X has emerged as a promising", a dictionary
    definition of the field as the first sentence. Open on the problem and a number.
18. Signposting and throat-clearing: "In this section, we will explore", "Let us dive
    into", "Let's break this down", "In summary", "To sum up", "Importantly,",
    "Interestingly,", "It should be noted", "Here's the thing", "The result? ...",
    "Think of it as", "Imagine a world where", a first sentence that restates the
    heading, a closing paragraph that recaps the section it ends.
19. Vague attribution: "studies show", "experts argue", "researchers have increasingly",
    "it is widely believed", "a growing body of work" without a `\\cite` in the same
    sentence; one citation inflated into "several works".
20. Motivational and token-caveat closers: "opens new avenues", "holds great promise",
    "exciting directions", "the possibilities are endless", "Despite these challenges,
    X remains promising", "Future work will explore" followed by a list of everything.
    A limitation is stated with its cost, not waved away.
21. Magic adverbs: truly, deeply, quietly, arguably, effortlessly, incredibly, extremely,
    seamlessly. Delete the adverb and see if the sentence changes.
22. Synonym cycling: one referent called the system, the framework, the platform, the
    pipeline and the tool in consecutive paragraphs. Pick one name and repeat it.
    Invented labels ("the verification paradox", "the context trap") count under 11.
23. Structure tells: Wh-question headings ("Why does this matter?"), a heading or
    `\\paragraph` every few sentences, bold-first bullets, one-sentence paragraphs used
    for drama, every section ending on a one-line takeaway, one point restated in
    every section.
24. Leaked LLM output and fabricated references: "As an AI language model", "as of my
    last knowledge update", "I hope this helps", "Certainly!", `oaicite`,
    `turn0search0`, `[cite: 1]`, `utm_source=chatgpt.com` in a URL. In the .bib:
    placeholder authors (John Doe, Jane Smith, Firstname Lastname), "et al." inside an
    author field, arXiv ids with XXXX, "to be updated", pages 1234-1243, a real title
    with the wrong authors, venue or year. GPTZero found 100 such citations in 51
    accepted NeurIPS 2025 papers; ICLR 2026 desk-rejects papers that contain them.
    Open every reference you cite.

## Figures and tables

At least one architecture figure and one results figure (TikZ, or matplotlib to PDF
with the script committed next to the paper). Every table caption states the
configuration. Every number in a table comes from a macro or a script. Nothing spills
into the margin.

## Process for a new paper

1. Read the sources. Write `./tmp/PLAN.md`: the section outline and the list of numbers
   to compute, each with the command that computes it.
2. Search the internet extensively for the state of the art and the related work. Visit
   at least 20 distinct sources and log them throughly in `./tmp/information-<paper>.md`
   a summary which is related to the topic of the paper.
3. Write the paper.
4. Build with the `build_paper` tool: zero errors, zero undefined references or
   citations, no overfull box over 10 pt. Render the pages to PNG (`pdftoppm -png -r 60`)
   and look at them. Fix tables that spill, prompt boxes that split badly, orphan
   headings, and `??`.
5. Run the `check_paper` tool and fix everything it flags.
6. Independent review: run the reviewer model read-only. Ask it to check, with line
   numbers, (a) every number against the raw results and the code, (b) every citation
   against a primary source, (c) internal consistency, (d) overclaims, (e) the AI-slop
   list above. Fix each finding you confirm. If a finding needs a new experiment, state
   the limitation instead of papering over it.
7. Open the venue's website and check the paper against the current guidelines: page
   limit, anonymity, checklist, fonts, margins, LLM-usage disclosure.
8. Rebuild, rerun `check_paper`, delete the planning and research files under `./tmp`,
   and git add the .tex, .bib, figures, scripts and .pdf. Never add .aux, .bbl, .blg, .log
   or .out files.

## Report back

File paths, page count, the gate counts before and after, the list of citations verified
with their source URLs, each reviewer finding and what you did with it, and anything the
paper claims that you could not verify.
""" """\


## Lessons from recent runs (rsi7d)

- Pass absolute paths inside the current worktree (`pwd` first) to `build_paper`,
  `check_paper` and every script you run; never `cd ../..` out of the worktree. A relative
  path resolves against the daemon's working directory and has rebuilt the main
  checkout's PDF and read another checkout's .tex.
- Fit a page limit in at most two rounds. After one build, measure the overflow once
  (`pdftotext -f <last main page> -l <last main page + 1> ... | wc -l`), plan cuts worth
  about 1.3 times that amount in a single edit (move whole paragraphs to the appendix
  before trimming sentences), rebuild once and re-measure.
- Report the task's total spend. The `Budget:` figure after a tool result is your own
  steps only; add the cost each `run_parallel` or `run_agent` result reports for its
  sub-agents, and use that total when the task caps the reviewer's share.
- `run_commands_parallel` takes a JSON array of strings: double every backslash, prefer
  `grep -E 'a|b'` over `\\\\|`, and run multi-line Python through Bash with a heredoc instead.
- When `run_parallel` answers "You are a reviewer sub-agent and may not spawn further
  reviewers", do not retry with `run_agent`: do the check yourself read-only and mark
  the report "independent review not run".
- A task that asks for a target review score without new experiments gets one fresh
  review round. When that round's main reason is an experiment, baseline or dataset the
  task rules out, state the limitation in the paper, run at most one more round to
  confirm the wording, and report that the score is capped by the evidence. Do not run a
  third fresh review for the same reason.
- When the task names a writer model other than yours and you run the writing as a
  sub-agent, either dispatch the reviewer yourself or copy the reviewer model name, the
  reviewer's budget share and the "do not invent new problems" clause verbatim into the
  child's task text; the child cannot see your task, so a child that is to run the review
  without them reviews with the wrong model.
- Decide who writes before the first edit: compare the writer model the task names with
  the `Model name:` line of your Task Settings and record the decision in `tmp/PLAN.md`.
  When they differ, the `run_parallel` writer sub-agent on that model is your first step
  and you make no edit to the .tex yourself; do not discover at the end that you wrote it.
  Tell the reviewer which model actually wrote the text: never describe your own edits as
  another model's work."""
"""The template's rules, appended to the default system prompt."""

# LaTeX environments whose bodies are not prose: verbatim prompts, listings, table
# cells and drawings.  The gates skip them.
_NON_PROSE_ENVIRONMENTS = (
    "lstlisting",
    "verbatim",
    "Verbatim",
    "minted",
    "prompt",
    "promptbox",
    "tabular",
    "tabular*",
    "tabularx",
    "tikzpicture",
    "filecontents",
    "filecontents*",
)

_LLM_UTM = r"utm_source=(?:chatgpt|openai|perplexity|copilot|claude|gemini)"
"""The tracking tag chatbots append to the URLs they cite."""

# (name, regex over the prose, limit, note); a limit of -1 means "list only".
_GATES: tuple[tuple[str, str, int, str], ...] = (
    ("em dashes", r"---|\u2014", 0, "use commas, parentheses, a colon or a new sentence"),
    (
        "antithesis",
        r"\brather than\b|\binstead of\b|, not |\bnot (?:\w+ ){1,3}but\b|\bnot just\b"
        r"|n't just\b",
        5,
        "only technical contrasts",
    ),
    (
        "performative honesty",
        r"\b(?:honest|honestly|plainly|genuine|precisely|exactly|to be clear|we say so"
        r"|worth stating|we do not claim|we state it)\b",
        3,
        "state the caveat, drop the commentary",
    ),
    (
        "slop vocabulary",
        r"\b(?:delve|leverag|pivotal|crucial|testament|landscape|tapestry|showcas|underscor"
        r"|intrica|meticulous|seamless|vibrant|realm|myriad|foster|comprehensive"
        r"|ever-evolving|fast-paced|game-chang|notably|moreover|furthermore|additionally"
        r"|garner|bolster|boast|enduring|interplay|multifaceted|holistic|nuanced"
        r"|groundbreaking|cutting-edge|revolutioni[sz]|transformative|paradigm shift"
        r"|synerg|invaluable|commendable|noteworthy|paramount|indispensable|unparalleled"
        r"|unwavering|embark|spearhead|streamlin|plethora|cornerstone|wide array of"
        r"|shed(?:s|ding)? light|deep dive|navigat\w* the complexit|valuable insights"
        r"|worth noting|important to note|in conclusion|serves as|stands as|aligns with"
        r"|aims to explore|(?:we|to|and|can|that) harness(?:es|ed)?\b)",
        0,
        "",
    ),
    (
        "significance puffery",
        r",\s+(?:highlighting|emphasizing|reflecting|ensuring|solidifying|cementing"
        r"|marking|contributing to|signaling|demonstrating|illustrating)\b"
        r"|\bmarks? an? (?:\w+ )?(?:shift|milestone|turning point)\b|\bsets? the stage\b"
        r"|\bpav(?:e|es|ed|ing) the way\b|\bsignificant step (?:forward|toward)"
        r"|\bat the forefront\b",
        3,
        "end the sentence at the fact; no clause that grades it",
    ),
    (
        "scope-inflation openers",
        r"\bin today's\b|\bin (?:the|an) (?:era|age) of\b"
        r"|\brapidly (?:evolving|changing|advancing) (?:field|landscape|world|area|domain)\b"
        r"|\bha(?:s|ve) (?:garnered|gained|attracted|received) (?:significant|considerable"
        r"|increasing|growing|widespread) (?:attention|interest)"
        r"|\bha(?:s|ve) emerged as (?:a|an) (?:promising|powerful|key|critical|essential)\b",
        0,
        "open on the problem and a number",
    ),
    (
        "signposting and throat-clearing",
        r"\blet(?:'s| us) (?:dive|explore|unpack|break)\b|\bin summary\b|\bto sum up\b"
        r"|\bimportantly,|\binterestingly,|\bit should be noted\b|\bit bears mentioning\b"
        r"|\bhere's (?:the|what|where|why|how)\b|\bhere is the (?:thing|kicker|catch)\b"
        r"|\bthe (?:result|catch|answer|twist)\?|\bthink of it as\b|\bimagine a world\b",
        0,
        "delete the announcement and start with the content",
    ),
    (
        "motivational closers",
        r"\bopens? (?:up )?(?:new|exciting) (?:avenues|directions|possibilities|doors)\b"
        r"|\bholds? (?:great|significant|immense|tremendous) promise\b"
        r"|\bexciting (?:new )?(?:directions|avenues|opportunities|possibilities)\b"
        r"|\bpossibilities are endless\b|\bthe future (?:is|looks) bright\b"
        r"|\bdespite (?:these|its|their|the|such) (?:challenges|limitations|hurdles),"
        r"[^.]*\b(?:promising|promise|potential|bright|exciting|thrive)",
        0,
        "state the limitation and its cost",
    ),
    (
        "vague attribution",
        r"\b(?:studies|experts|researchers|observers|critics|many works) (?:have )?"
        r"(?:show|shown|suggest|suggested|argue|argued|agree|noted|increasingly)\b"
        r"|\bit is (?:widely|generally|commonly) (?:believed|accepted|known|acknowledged"
        r"|recognized)\b|\ba growing body of\b",
        -1,
        "name the source with a \\cite in the same sentence or drop the claim",
    ),
    (
        "intensifiers and marketing",
        r"\b(?:remarkably|dramatically|particularly|fundamentally|essentially|quite"
        r"|significantly|powerful|elegant|unique|novel|state-of-the-art|compares favorably"
        r"|outperforms|to our knowledge|truly|deeply|quietly|arguably|effortlessly"
        r"|incredibly|extremely|robust|robustly)\b",
        -1,
        "review each; keep only with a number or a statistical test",
    ),
    (
        "leaked LLM output",
        r"oaicite|contentreference\[|\bturn\d+(?:search|view|news|file)\d+|"
        + _LLM_UTM
        + r"|\[cite(?:_start|:\s*\d+)\]|\bas an ai (?:language )?model\b"
        r"|\bas of my (?:last )?(?:knowledge|training)\b|\bi hope this helps\b"
        r"|\bcertainly!|\bgreat question\b",
        0,
        "chatbot residue; delete it and re-read the paragraph around it",
    ),
    (
        "draft and review talk",
        r"\b(?:draft|preliminary|this submission|reviewer|rebuttal)s?\b",
        -1,
        "zero unless the word is the paper's technical subject; the paper is standalone",
    ),
    (
        "markdown artifacts",
        r"\*\*[^*\n]+\*\*|\u201c|\u201d|\u2018|\u2019|\u2026|[\u2190-\u21ff]",
        0,
        "",
    ),
)

_BIB_SUSPECTS = (
    r"\b(?:John|Jane) (?:Doe|Smith)\b|\bfirst ?name\b|\blast ?name\b"
    r"|\bauthor\s*=\s*[{\"][^}\"]*\bet al\b|\d{4}\.X{4,5}\b|\bto be updated\b"
    r"|\bpages\s*=\s*[{\"]?\s*1234\s*-+\s*1243\b|oaicite|" + _LLM_UTM
)
"""Signatures of fabricated references (GPTZero's NeurIPS 2025 and ICLR 2026 scans)."""

_MIN_DUPLICATE_WORDS = 4
"""Shorter repeated sentences ("We measure.") are not flagged as duplicates."""

_SENTENCE_END = r"(?<=[.!?])(?<!\bet al\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bcf\.)\s+"
"""Sentence boundary: end punctuation then whitespace, except after common abbreviations."""

_MAX_ITEMS = 25
"""Hits listed per gate before the report says ``... N more``."""


def description() -> str:
    """Return the one-sentence help text shown by ``/write_paper help``."""
    return (
        "Writes or revises a LaTeX research paper from your sources and results, running "
        "AI-slop/consistency gates (`check_paper`) and a pdflatex+bibtex build (`build_paper`) "
        "with a read-only reviewer model; use `/write_paper <venue, .tex path, topic, sources, "
        'options>` or run_agent(agent="write_paper", task=...).'
    )


def _blank(match: re.Match[str]) -> str:
    """Replace a match by newlines only, so later line numbers still point into the source."""
    return "\n" * match.group(0).count("\n")


def _uncommented(source: str) -> str:
    """Return *source* without ``%`` comments (escaped ``\\%`` is kept)."""
    return re.sub(r"(?<!\\)%.*", "", source)


def _prose(source: str) -> str:
    """Return *source* with the preamble, comments and non-prose environments blanked."""
    text = _uncommented(source)
    begin = text.find(r"\begin{document}")
    if begin >= 0:
        text = "\n" * text[:begin].count("\n") + text[begin:]
    names = "|".join(re.escape(n) for n in _NON_PROSE_ENVIRONMENTS)
    return re.sub(rf"\\begin\{{({names})\}}.*?\\end\{{\1\}}", _blank, text, flags=re.DOTALL)


def _hits(text: str, pattern: str) -> list[str]:
    """Return ``L<line>: ...<context>...`` for every case-insensitive match of *pattern*.

    The context is the matched text with up to 50 characters of its line on each side,
    so a paragraph written as one long source line still shows where the hit is.
    """
    hits = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        line = text.count("\n", 0, m.start()) + 1
        start = max(text.rfind("\n", 0, m.start()) + 1, m.start() - 50)
        end = text.find("\n", m.end())
        end = min(end if end >= 0 else len(text), m.end() + 50)
        hits.append(f"L{line}: ...{text[start:end].strip()}...")
    return hits


def _section(name: str, items: list[str], limit: int, note: str = "") -> list[str]:
    """Render one gate as ``name: count (limit N) VERDICT; note`` plus the indented items.

    A negative *limit* means the gate only lists its hits for review (``CHECK``).
    """
    if limit < 0:
        verdict, bound = ("CHECK" if items else "PASS"), "list only"
    else:
        verdict, bound = ("PASS" if len(items) <= limit else "FAIL"), f"limit {limit}"
    lines = [f"{name}: {len(items)} ({bound}) {verdict}" + (f"; {note}" if note else "")]
    lines += [f"    {item[:140]}" for item in items[:_MAX_ITEMS]]
    if len(items) > _MAX_ITEMS:
        lines.append(f"    ... {len(items) - _MAX_ITEMS} more")
    return lines


def _duplicated_sentences(prose: str) -> list[str]:
    """Return ``L<a>, L<b>: <sentence>`` for every sentence that occurs more than once.

    Sentences shorter than ``_MIN_DUPLICATE_WORDS`` words are ignored; a
    paragraph is one source line, so the line numbers name the paragraphs.
    """
    seen: dict[str, list[int]] = {}
    for number, line in enumerate(prose.splitlines(), 1):
        for raw in re.split(_SENTENCE_END, line):
            sentence = re.sub(r"\s+", " ", raw).strip().lower()
            if len(sentence.split()) >= _MIN_DUPLICATE_WORDS:
                seen.setdefault(sentence, []).append(number)
    return [
        f"{', '.join(f'L{n}' for n in numbers)}: {sentence}"
        for sentence, numbers in seen.items()
        if len(numbers) > 1
    ]


def _abstract(prose: str) -> re.Match[str] | None:
    """Return the match of the abstract environment in *prose*, or ``None``."""
    return re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", prose, flags=re.DOTALL)


def _abstract_words(abstract: str) -> int:
    """Count the rendered words of *abstract*.

    Citations, references and labels are dropped with their keys; other
    commands (``\\emph{word}``, ``\\Speedup{}``) lose their name and braces
    but keep their text, as the reader sees it.
    """
    keyed = r"\\(?:cite[A-Za-z]*|[a-zA-Z]*ref|label)\*?(?:\[[^\]]*\])*\{[^}]*\}"
    text = re.sub(keyed, " ", abstract)
    text = re.sub(r"\\[A-Za-z]+\*?", " ", text)
    return len(re.sub(r"[{}\[\]~]", " ", text).split())


def _abstract_orphans(source: str, prose: str, abstract: re.Match[str]) -> list[str]:
    """Return the numbers and number macros of the abstract that the body never uses again.

    A macro counts when the preamble defines it with ``\\newcommand`` (the
    template's rule: every reported number is a macro defined once).  A
    number is reused only as a whole token (``10`` is not reused by ``100``).
    """
    body = prose[: abstract.start()] + prose[abstract.end() :]
    defined = set(re.findall(r"\\(?:re)?newcommand\*?\s*\{?(\\[A-Za-z]+)", source))
    numbers = set(re.findall(r"(?<![\w.])\d+(?:[.,]\d+)*%?(?!\w)", abstract.group(1)))
    macros = set(re.findall(r"\\[A-Za-z]+", abstract.group(1))) & defined
    orphans = [n for n in numbers if not re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])", body)]
    orphans += [m for m in macros if not re.search(rf"{re.escape(m)}(?![A-Za-z])", body)]
    return sorted(orphans)


def _keys(source: str, pattern: str) -> set[str]:
    """Return the comma-separated keys of every ``\\cmd{a,b}`` matched by *pattern*."""
    keys: set[str] = set()
    for m in re.finditer(pattern, source):
        keys |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    return keys


def _bib_entries(
    tex_path: Path, source: str, bib_path: str
) -> tuple[set[str], list[str], list[str]]:
    """Return the paper's .bib entry keys, the paths read, and the suspect bib lines.

    A suspect line matches ``_BIB_SUSPECTS`` (placeholder authors, ``XXXX``
    arXiv ids, ...) and is listed as ``<file> L<line>: ...<context>...``.
    """
    names = [bib_path] if bib_path else []
    names += sorted(_keys(source, r"\\(?:bibliography|addbibresource)\{([^}]*)\}"))
    keys: set[str] = set()
    read: list[str] = []
    suspects: list[str] = []
    for name in names:
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = tex_path.parent / path
        if path.suffix != ".bib":
            path = path.with_suffix(".bib")
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            keys |= set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", text))
            read.append(str(path))
            suspects += [f"{path.name} {hit}" for hit in _hits(text, _BIB_SUSPECTS)]
    return keys, read, suspects


def check_paper(tex_path: str, bib_path: str = "") -> str:
    """Run the AI-slop and consistency gates on a LaTeX paper and return the report.

    The word gates run on the prose only: the preamble, ``%`` comments and the
    bodies of verbatim, listing, prompt, table and TikZ environments are
    excluded.  Every hit is listed with its source line number.

    Args:
        tex_path: Path of the ``.tex`` file.
        bib_path: Path of the ``.bib`` file when the .tex does not name it with
            ``\\bibliography{}`` or ``\\addbibresource{}``.

    Returns:
        One block per gate marked ``PASS``, ``FAIL`` or ``CHECK`` (list only):
        the word gates, the abstract word count, abstract numbers unused
        elsewhere, duplicated sentences, single-word ``\\emph``, non-ASCII
        lines, dangling ``\\ref``, the ``\\cite`` keys against the bib
        entries, and bib lines with fabricated-reference signatures; then
        the number of failed gates.
    """
    path = Path(tex_path).expanduser()
    if not path.is_file():
        return f"no such file: {path}"
    source = path.read_text(encoding="utf-8")
    prose = _prose(source)
    lines = [f"gates for {path} (prose only)"]
    for name, pattern, limit, note in _GATES:
        lines += _section(name, _hits(prose, pattern), limit, note)
    abstract = _abstract(prose)
    if abstract is None:
        lines.append("abstract: not found FAIL")
    else:
        words = _abstract_words(abstract.group(1))
        verdict = "PASS" if 150 <= words <= 250 else "FAIL"
        lines.append(f"abstract words: {words} (150 to 250) {verdict}")
        lines += _section(
            "abstract numbers or macros unused elsewhere in the body",
            _abstract_orphans(source, prose, abstract),
            0,
        )
    lines += _section("duplicated sentences", _duplicated_sentences(prose), 0)
    lines += _section(
        "single-word \\emph", _hits(prose, r"\\emph\{[A-Za-z]+\}"), -1, "term introductions only"
    )
    non_ascii = [
        f"L{i}: {line.strip()}"
        for i, line in enumerate(prose.splitlines(), 1)
        if any(ord(c) > 127 for c in line)
    ]
    lines += _section("non-ASCII lines", non_ascii, -1, "none unless intended")
    # Keys are read from the comment-free source (tables and listings included);
    # ``label={...}`` covers the listings package's caption labels.
    live = _uncommented(source)
    refs = _keys(live, r"\\(?:auto|eq|page|c|C)?ref\{([^}]*)\}")
    labels = _keys(live, r"\\label\{([^}]*)\}") | _keys(live, r"\blabel=\{?([^\s,\]}]+)")
    lines += _section("\\ref without \\label", sorted(refs - labels), 0)
    cites = _keys(live, r"\\cite[A-Za-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
    entries, read, suspects = _bib_entries(path, live, bib_path)
    if read:
        lines.append(f"bib files: {', '.join(read)}")
        lines += _section("\\cite without bib entry", sorted(cites - entries), 0)
        lines += _section("bib entries never cited", sorted(entries - cites), 0)
        lines += _section(
            "fabricated-reference signatures", suspects, 0, "open the source and fix the entry"
        )
    else:
        lines.append("bib files: none found (pass bib_path to check the citations) CHECK")
    failed = sum(line.endswith(" FAIL") or " FAIL; " in line for line in lines)
    lines.append(f"summary: {failed} gate(s) failed")
    return "\n".join(lines)


def _find_tex_binary(name: str, tex_bin: str) -> str:
    """Return the path of TeX program *name*: from *tex_bin*, then PATH, then TeX Live."""
    if tex_bin:
        return str(Path(tex_bin).expanduser() / name)
    found = shutil.which(name)
    if found:
        return found
    candidates = sorted(glob.glob(f"/usr/local/texlive/*/bin/*/{name}"))
    return candidates[-1] if candidates else ""


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    """Run *cmd* in *cwd* and return ``(exit code, combined output)``."""
    if not cmd[0] or not Path(cmd[0]).is_file():
        return 127, f"{cmd[0] or 'program'}: not found"
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, errors="replace", timeout=600
    )
    return proc.returncode, proc.stdout + proc.stderr


def _status(program: str, code: int) -> str:
    """Render one build pass as ``<program>: exit <code>``, marked ``FAIL`` when non-zero."""
    return f"{program}: exit {code}" + (" FAIL" if code else "")


def _bibliography_pass(cwd: Path, stem: str, source: str, tex_bin: str) -> list[str]:
    """Run bibtex (or biber for a biblatex paper) when the first pass asked for a bibliography."""
    aux = cwd / f"{stem}.aux"
    if aux.is_file() and r"\bibdata" in aux.read_text(encoding="utf-8", errors="replace"):
        program, marks = "bibtex", ("Warning--", "error")
    elif (cwd / f"{stem}.bcf").is_file() and "biblatex" in _uncommented(source):
        program, marks = "biber", ("WARN", "ERROR")
    else:
        return ["no bibliography requested; bibtex skipped"]
    code, out = _run([_find_tex_binary(program, tex_bin), stem], cwd)
    flagged = [line.strip() for line in out.splitlines() if any(m in line for m in marks)]
    return [_status(program, code)] + [f"    {line[:140]}" for line in flagged[:_MAX_ITEMS]]


def _summarize_log(log: str) -> tuple[list[str], list[str], list[str], str]:
    """Return errors, undefined references and citations, overfull boxes over 10 pt, pages."""
    errors = [
        line.strip()
        for line in log.splitlines()
        if line.startswith("!") or re.match(r".+:\d+: ", line)
    ]
    undefined = sorted(
        set(re.findall(r"(?:Reference|Citation) `([^']*)' on page \d+ undefined", log))
    )
    overfull = []
    for m in re.finditer(r"Overfull \\[hv]box \(([\d.]+)pt too (?:wide|high)\)([^\n]*)", log):
        where = re.search(r"at lines? ([\d-]+)", m.group(2))
        if float(m.group(1)) > 10.0:
            overfull.append(f"{m.group(1)}pt at lines {where.group(1) if where else '?'}")
    pages = re.search(r"Output written on .*?\((\d+) pages?", log)
    return errors, undefined, overfull, pages.group(1) if pages else "?"


def build_paper(tex_path: str, tex_bin: str = "") -> str:
    """Build a LaTeX paper (pdflatex, bibtex, pdflatex, pdflatex) and summarize the log.

    Args:
        tex_path: Path of the main ``.tex`` file; the build runs in its directory.
        tex_bin: Directory holding ``pdflatex`` and ``bibtex`` (default: ``PATH``,
            then the newest ``/usr/local/texlive/*/bin/*``).

    Returns:
        The exit status of each pass, then the errors, the undefined references
        and citations, and the overfull boxes over 10 pt with their line
        ranges, each marked ``PASS`` or ``FAIL``; then the page count and the
        PDF and log paths.
    """
    path = Path(tex_path).expanduser().resolve()
    if not path.is_file():
        return f"no such file: {path}"
    pdflatex = _find_tex_binary("pdflatex", tex_bin)
    if not pdflatex or not Path(pdflatex).is_file():
        return "pdflatex not found: install TeX Live or pass tex_bin"
    cwd, stem = path.parent, path.stem
    latex = [pdflatex, "-interaction=nonstopmode", "-file-line-error", path.name]
    code, log = _run(latex, cwd)
    report = [_status("pdflatex 1", code)]
    report += _bibliography_pass(cwd, stem, path.read_text(encoding="utf-8"), tex_bin)
    for i in (2, 3):
        code, log = _run(latex, cwd)
        report.append(_status(f"pdflatex {i}", code))
    errors, undefined, overfull, pages = _summarize_log(log)
    report += _section("errors", errors, 0)
    report += _section("undefined references or citations", undefined, 0)
    report += _section("overfull boxes over 10pt", overfull, 0)
    pdf = cwd / f"{stem}.pdf"
    report.append(f"pages: {pages}")
    report.append(f"pdf: {pdf}" if pdf.is_file() else "pdf: not written FAIL")
    report.append(f"log: {cwd / f'{stem}.log'}")
    return "\n".join(report)


def add_to_system_prompt() -> str:
    """Append the paper-writing rules to the default system prompt."""
    return SYSTEM_PROMPT


def add_to_tools() -> list[Any]:
    """Expose the gate checker and the LaTeX builder to the model."""
    return [check_paper, build_paper]


DISPATCH_TIMEOUT_SECONDS = 21600
"""Seconds a ``run_agent`` call waits for a ``/write_paper`` run: six hours."""

def settings() -> dict[str, Any]:
    """Browse (related work, venue guidelines), fan out (the reviewer), skip the classifier.

    The ``timeout`` tells the dispatcher a ``/write_paper`` run may take
    up to :data:`DISPATCH_TIMEOUT_SECONDS`.
    """
    return {
        "use_web_tools": True,
        "is_parallel": True,
        "classify_tasks": False,
        "timeout": DISPATCH_TIMEOUT_SECONDS,
    }


