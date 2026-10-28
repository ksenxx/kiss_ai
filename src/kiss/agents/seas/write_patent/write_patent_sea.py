# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Patent-drafting agent — writes or revises a utility patent application as a practitioner would.

The ``/write_patent`` SEA turns an invention disclosure (notes, code, a paper, a
chat) into a Markdown draft laid out in the order of 37 CFR 1.77 and checks it
with two tools:

* ``check_patent`` runs the deterministic gates a claim proofreader runs:
  section order, title and abstract limits, claim numbering, one sentence per
  claim, dependency and multiple-dependent-claim rules, antecedent basis,
  112(f) triggers, relative terms, fee thresholds, claim terms with no support
  in the description, figure cross-references, reference-numeral consistency,
  "patent profanity" and prior-art admissions, and the AI-slop gates shared
  with ``/write_paper``.
* ``build_patent`` renders the draft to a filing-shaped ``.docx`` (the USPTO
  Patent Center format), a ``.pdf`` when ``pdflatex`` is installed, and a
  drawings ``.pdf`` with one ``FIG. N`` per sheet.

The ``/review_patent`` SEA imports ``check_patent`` and the draft parser so the
two agents read the same format.  Everything either agent writes is a draft for
a registered practitioner to verify and sign (37 CFR 11.18(b)); nothing here is
legal advice.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.seas.write_paper.write_paper_sea import (
    _GATES,
    _duplicated_sentences,
    _find_tex_binary,
    _hits,
    _run,
    _section,
    _summarize_log,
)

TITLE_LIMIT = 500
"""Characters a title may have (37 CFR 1.72(a))."""

ABSTRACT_LIMIT = 150
"""Words an abstract should not exceed (37 CFR 1.72(b))."""

FEE_THRESHOLDS = {"US": (3, 20), "EP": (0, 15)}
"""Independent and total claims included in the basic fee per jurisdiction (37 CFR 1.16; EPO)."""

SECOND_OPINION_MODEL = "gpt-5.6-sol"
"""The model that reviews the draft when the task names no other (``/review_paper`` uses it too)."""

SYSTEM_PROMPT = f"""\
# Patent-drafting agent

You ARE the drafting agent: you write (or revise) a utility patent application from the
user's invention disclosure, code, paper, notes or chat, in the form a registered patent
practitioner would file.  The output is a draft for that practitioner to verify and sign
(37 CFR 11.18(b)); say so in your final summary and never present it as legal advice.  The
rules below come from 37 CFR 1.52, 1.72, 1.75, 1.77 and 1.84, MPEP 608, 2106, 2173 and
2181, and the USPTO's 2024-2026 eligibility memos (Ex parte Desjardins), all opened on the
web on 2026-10-27; re-check a rule on uspto.gov or ecfr.gov when the user's jurisdiction or
filing date makes it matter.

## Inputs and outputs

The task names the invention sources, the draft path (a ``.md`` file; default
``patent/<short-name>.md``), the jurisdiction (US unless told otherwise; EP/PCT changes the
claim form) and options.  Read every source first.  Write these files, all next to the draft:

- ``<stem>.md``: the application in the 37 CFR 1.77 order, with these Markdown headings
  (uppercase, in this order; the parser accepts ``#``/``##`` headings or bare uppercase lines):
  ``# <Title>``, then ``## CROSS-REFERENCE TO RELATED APPLICATIONS`` (only if any),
  ``## TECHNICAL FIELD``, ``## BACKGROUND``, ``## SUMMARY``,
  ``## BRIEF DESCRIPTION OF THE DRAWINGS``, ``## DETAILED DESCRIPTION``, ``## CLAIMS``,
  ``## ABSTRACT``.  Under CLAIMS put the line ``What is claimed is:`` and then ``1. ...``,
  ``2. ...`` with each claim element on its own indented line.
- ``<stem>_references.md``: every patent or publication you opened while searching, one per
  line with number or URL, date, assignee or authors, and one sentence on what it teaches
  and which claim element it is closest to.  This is the practitioner's input for the
  Information Disclosure Statement (duty of candor, 37 CFR 1.56); a reference you did not
  open does not go in it.
- ``figures/fig1.png`` ...: black-and-white line drawings (37 CFR 1.84): block diagrams,
  flowcharts, data layouts; matplotlib, graphviz or TikZ to PNG/PDF; every box or arrow that
  the description names carries a reference numeral (102, 104, ...), and the figure carries
  ``FIG. N``.  Reference each as ``![FIG. N](figures/figN.png)`` in BRIEF DESCRIPTION OF THE
  DRAWINGS; ``build_patent`` assembles them into ``<stem>_drawings.pdf`` and keeps them out
  of the specification text (37 CFR 1.58(a)).
- ``<stem>.docx``, ``<stem>.pdf``, ``<stem>_drawings.pdf``: produced by ``build_patent``.

## Process

1. Understand the invention from the sources: the problem, what existed before, what is new
   and why it works (the technical improvement in the inventor's words).  Write a one-page
   invention summary in your scratch dir before drafting; list the alternatives, ranges,
   parameters and fallback designs you can see in the sources, because every one of them is
   a possible dependent claim or amendment and the specification must describe it now
   (nothing can be added after filing).
2. Search the prior art before writing a single claim: Google Patents
   (``https://patents.google.com/?q=(terms)&before=priority:YYYYMMDD``, then the
   ``/patent/<number>/en`` page for claims and cited-by), Espacenet, USPTO Patent Public
   Search, Lens, and arXiv/Semantic Scholar for papers.  At least 15 distinct references
   opened, the closest 5 read in full.  Draft claim 1 to clear the closest reference by a
   feature the sources support, not by a vague adjective.
3. Claim strategy before prose.  Three independent claims when the sources support them
   (method; system or apparatus; non-transitory computer-readable medium), each broad enough
   to be infringed by a competitor and narrow enough to clear the art you found; claim 1 the
   least restrictive.  Twenty claims total unless told otherwise (the USPTO basic fee covers 3
   independent and 20 total; each claim over the line costs money).  Dependent claims form a
   ladder: each adds one concrete limitation the description supports, in the order you would
   retreat to under rejection; group them under their parent.  Transition ``comprising``
   (open); ``consisting of`` only on purpose.  Antecedent basis: introduce every element
   with ``a``/``an`` and refer back with ``the``, with the same words each time; never
   ``said``.  No ``means for`` unless the description names the exact structure (112(f));
   no ``about``, ``substantially``, ``such as``, ``for example``, ``preferably``, ``etc.`` in
   a claim.  One sentence per claim ending in a period.  For EP/PCT drafts put reference
   signs in brackets and use the two-part form (``characterised in that``) where the
   closest art shares a clear preamble with the claim (EPC Rule 43(1): "wherever
   appropriate"); a claim with no such art stays in one part, and ``check_patent`` only
   lists the one-part claims for your review.
4. Specification that supports the claims (37 CFR 1.75(d)(1)): every claim term appears in
   the DETAILED DESCRIPTION with the same word; define any term you use in a special sense
   (you are the lexicographer); describe at least two embodiments and the alternatives for
   each element; give ranges with endpoints and example values; state the technical
   improvement and why it follows from the claimed features (this is what Step 2A prong two
   of the Alice/Mayo eligibility test and Desjardins look for: a stated improvement to how
   the computer or model operates, not a business result).  BACKGROUND: two to four short
   paragraphs on the technical problem, with no "prior art", "conventional", "well known"
   admissions and no characterization of specific references (the practitioner's call).  SUMMARY:
   mirrors the independent claims in prose ("In one aspect, a method includes...").  Say
   "the disclosure" or "some embodiments", never "the present invention" or "the invention";
   no "must", "always", "only", "essential", "critical", "necessary" unless the sources make
   the feature truly indispensable.  ABSTRACT: one paragraph under 150 words, no "said", no
   "means", no merits.  TITLE: specific, under 500 characters, no "new", "improved", "novel".
5. Run ``check_patent`` (absolute paths: a relative path resolves against the daemon's
   directory, not yours) and fix every FAIL and review every CHECK; then ``build_patent``
   and open the PDF pages that the report makes you doubt (``pdftoppm -png -r 60 -f N -l N``).
   Zero FAIL lines before you finish.
6. Second opinion: dispatch ``run_agent(agent="review_patent", task="You ARE the review
   agent. Review <absolute draft path> for <jurisdiction>; write the review to
   <scratch>/review.txt", model=<second model>, wait="false")`` with a model other than
   your own when the task names one (default ``{SECOND_OPINION_MODEL}``), keep working,
   collect it with ``agent_job``, and address every substantiated finding (claim by claim)
   before the final build.  If ``run_agent`` is absent or refuses (you are a sub-agent), do
   not retry and do not finish unsuccessfully: ``set_model(<second model>)``, read the
   claims against the description and the art yourself, then ``set_model`` back before
   editing.  A finding you reject gets one sentence of reason in the summary.  When the task
   asks to revise an existing draft, ``Read`` it before any ``Write``: a revision keeps the
   paragraphs and claims the reviewer did not fault.

## Writing style

Plain declarative sentences; a skilled engineer must be able to build the thing from the
DETAILED DESCRIPTION alone (enablement, 35 U.S.C. 112(a)).  Consistent nouns: one element,
one name, one reference numeral throughout (examiners object to inconsistent terminology,
MPEP 608.01).  Numbers and units for every parameter.  No marketing, no em dashes, no
"delve"/"leverage"/"robust", no chatbot residue.  Write ``FIG. 1`` (never ``Figure 1`` or
``Fig. 1``) and number figures from 1 in the order the description reaches them.

## Honesty

Every reference in ``<stem>_references.md`` is one you opened; a number or URL you did not
open does not appear anywhere.  If the sources do not support a feature, do not claim it;
list it in the summary as a question for the inventor.  Report the ``check_patent`` summary
line and the ``build_patent`` file list verbatim in your final summary.
"""

# Specification sections in the order of 37 CFR 1.77(b): canonical name, heading regex,
# whether a draft must have it.
_SECTIONS: tuple[tuple[str, str, bool], ...] = (
    (
        "CROSS-REFERENCE TO RELATED APPLICATIONS",
        r"cross[- ]references?(?: to related applications?)?",
        False,
    ),
    (
        "STATEMENT REGARDING FEDERALLY SPONSORED RESEARCH",
        r"statement (?:regarding|of) federally sponsored research(?: or development)?",
        False,
    ),
    ("INCORPORATION BY REFERENCE", r"incorporat\w* by reference", False),
    ("TECHNICAL FIELD", r"(?:technical )?field(?: of the (?:invention|disclosure))?", False),
    ("BACKGROUND", r"background(?: of the (?:invention|disclosure))?", True),
    ("SUMMARY", r"(?:brief )?summary(?: of the (?:invention|disclosure))?", True),
    (
        "BRIEF DESCRIPTION OF THE DRAWINGS",
        r"brief description of the (?:several views of the )?(?:drawings?|figures?)",
        False,
    ),
    (
        "DETAILED DESCRIPTION",
        r"detailed description(?: of (?:the )?(?:invention|disclosure|(?:\w+ )?embodiments))?",
        True,
    ),
    ("CLAIMS", r"claims|what is claimed is|(?:i|we) claim|the invention claimed is", True),
    ("ABSTRACT", r"abstract(?: of the disclosure)?", True),
)

_SECTION_INDEX = {name: i for i, (name, _, _) in enumerate(_SECTIONS)}

_CLAIM_START = re.compile(r"^\s*(\d+)\s*[.)]\s+(\S.*)$")
"""A line that starts a claim: its number, a period or parenthesis, text."""

_CLAIMS_PREAMBLE = re.compile(
    r"what is claimed is|(?:\bi|\bwe) claim|the invention claimed is|claimed is:", re.IGNORECASE
)

_CLAIM_REFERENCE = re.compile(
    r"\b(any\s+(?:one\s+)?of\s+|either\s+)?claims?\s+"
    r"(\d+(?:\s*,?\s*(?:,|or|and|to|through|-)\s*(?:claims?\s+)?\d+)*)",
    re.IGNORECASE,
)
"""``claim 1``, ``claim 1 or claim 2``, ``any one of claims 1 to 3 or 5``, ``claims 1, 2, or 3``."""

_FIGURE = re.compile(
    r"\b(?:FIGS?\.?|Figures?|Figs?\.?)\s*"
    r"(\d+[A-Za-z]?\b(?:\s*(?:,|and|or|to|through|-)\s*\d+[A-Za-z]?\b)*)"
)
"""``FIG. 1``, ``FIG. 1A``, ``FIGS. 1 and 2``, ``FIGS. 2-4``: the identifier list after the word."""
_FIGURE_WORDING = r"(?-i:\b(?:Figure|Figs?\.)\s*\d|\bFIG\s+\d|\bfig\.?\s*\d)"
"""Figure references not written ``FIG. N`` (the ``(?-i:...)`` keeps the gate case-sensitive)."""
_REFERENCE_NUMERAL = re.compile(r"\b([a-z][a-z-]{2,})\s+(\d{2,4}[a-z]?)\b")
_NUMERAL_SKIP = frozenset(
    "about than least approximately claim claims step steps fig figs figure figures page "
    "line lines example embodiment embodiments over under between from into onto with and "
    "the per through within for into only are was were has have had bit bits byte bytes "
    "version versions number numbers value values size sizes port ports".split()
)

_INDEFINITE_TERMS = (
    r"\b(?:about|substantially|approximately|generally|essentially|relatively|roughly"
    r"|such as|for example|e\.g\.|etc\.|preferably|optionally|or the like|and/or|suitable"
    r"|sufficient|effective amount|if desired|when desired)\b"
)
_MEANS_PLUS_FUNCTION = (
    r"\b(?:means|step) for\b|\b(?:module|mechanism|element|unit|component|device) (?:for|that"
    r"|configured to|adapted to|operable to)\b"
)
_TRANSITION = re.compile(
    r"\b(?:compris\w*|consist\w*|includ\w*|having|characteri[sz]ed)\b", re.IGNORECASE
)
_NARROWING = re.compile(r"\b(?:wherein|further|compris\w*|includ\w*|in which|where)\b", re.I)

_INVENTION_TALK = (
    r"\b(?:the|this|said|our) (?:present |instant |current |claimed )?invention\b"
    r"|\bobjects? of the (?:present )?(?:invention|disclosure)\b"
)
_PRIOR_ART_ADMISSIONS = (
    r"\bprior art\b|\badmitted\b|\bwell[- ]known\b|\bit is known\b|\bknown in the art\b"
    r"|\bconventional(?:ly)?\b|\bas is known\b|\bit is well established\b"
)
_ABSOLUTE_TERMS = (
    r"\b(?:must|always|never|only|essential|critical|necessary|required|cannot|impossible"
    r"|all embodiments|every embodiment|key (?:feature|aspect)|important (?:feature|aspect))\b"
)
_ABSTRACT_WORDING = (
    r"\bsaid\b|\bmeans\b|\b(?:the|this) (?:present )?(?:invention|disclosure) (?:relates|concerns"
    r"|describes|provides|is directed)\b|\bdisclosed herein\b|\bnovel\b|\bimproved\b"
)
_TITLE_WORDING = r"\b(?:new|improved|improvements?|novel)\b"

_PROSE_GATE_NAMES = (
    "em dashes",
    "slop vocabulary",
    "significance puffery",
    "scope-inflation openers",
    "signposting and throat-clearing",
    "motivational closers",
    "intensifiers and marketing",
    "leaked LLM output",
)
_PROSE_GATES = tuple(g for g in _GATES if g[0] in _PROSE_GATE_NAMES)
"""The ``/write_paper`` word gates that apply to patent prose (Markdown source, so no
markdown-artifact gate; claims use ``rather than`` and ``not`` on purpose, so no antithesis)."""

_STOP_WORDS = frozenset(
    "of to and or in for with that which wherein whereby by from at on is are being having "
    "comprising comprises including includes the a an said as into onto than then when where "
    "each any one more least plurality via per such between among through during before after "
    "under over not no first second third further also thereof therein thereto whether claim "
    "claims".split()
)
_INTRODUCERS = ("a", "an", "at least one", "one or more", "a plurality of", "plurality of", "each")
_DEFINITE_SKIP = frozenset(
    "following same other respective step steps claim claims invention art like foregoing "
    "above preceding".split()
)
_PHRASE_STOP = _STOP_WORDS - frozenset(
    "first second third each any one more least plurality further also no not".split()
)
"""Words that end the noun phrase after ``a``/``the`` (ordinals and quantifiers stay in it)."""
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'-]*|[,;:.()]")
_ABBREVIATIONS = r"(?:FIGS?|Figs?|No|Nos|U\.S|e\.g|i\.e|etc|vs|Dr|St|Inc|Ltd|approx|al)\."


class Draft:
    """A parsed Markdown patent draft: title, sections, claims and figures.

    ``sections`` maps a canonical section name (``"CLAIMS"``) to ``(line, text)``;
    ``claims`` holds ``(number, line, text)`` with the claim's lines joined by
    newlines; ``figures`` holds ``(line, caption, path)`` for every Markdown image.
    """

    def __init__(self, source: str) -> None:
        """Parse *source*; unknown headings are kept as text of the enclosing section."""
        self.title = ""
        self.title_line = 0
        self.sections: dict[str, tuple[int, str]] = {}
        self.order: list[tuple[str, int]] = []
        self.figures: list[tuple[int, str, str]] = []
        current = ""
        body: list[str] = []
        for number, line in enumerate(source.splitlines(), 1):
            image = re.match(r"\s*!\[([^\]]*)\]\(([^)]+)\)", line)
            if image:
                self.figures.append((number, image.group(1), image.group(2)))
                continue
            name = section_name(line)
            if name:
                if current:
                    self.sections[current] = (self.sections[current][0], "\n".join(body))
                current, body = name, []
                self.sections[name] = (number, "")
                self.order.append((name, number))
                continue
            heading = re.match(r"\s*#+\s+(.+?)\s*#*\s*$", line)
            if heading and not self.title and not current:
                self.title, self.title_line = heading.group(1).strip(), number
                continue
            body.append(line)
        if current:
            self.sections[current] = (self.sections[current][0], "\n".join(body))
        self.claims = parse_claims(*self.sections.get("CLAIMS", (0, "")))

    def text(self, name: str) -> str:
        """Return the body of section *name* (``""`` when absent)."""
        return self.sections.get(name, (0, ""))[1]

    def description(self) -> str:
        """Return every section but CLAIMS and ABSTRACT, joined: the written description."""
        return "\n".join(
            text for name, (_, text) in self.sections.items() if name not in ("CLAIMS", "ABSTRACT")
        )


def section_name(line: str) -> str | None:
    """Return the canonical 37 CFR 1.77 section a heading *line* opens, or ``None``.

    A heading is a Markdown ``#`` line, a ``**bold**`` line or a bare uppercase line
    whose text matches one of the known headings (a trailing colon is allowed).
    """
    stripped = line.strip()
    if not stripped:
        return None
    marked = stripped.startswith("#") or (stripped.startswith("**") and stripped.endswith("**"))
    text = re.sub(r"^#+\s*|\s*#+$|\*\*", "", stripped).strip().rstrip(":").strip()
    if not marked and (text != text.upper() or not re.search(r"[A-Z]", text)):
        return None
    for name, pattern, _ in _SECTIONS:
        if re.fullmatch(pattern, text, flags=re.IGNORECASE):
            return name
    return None


def parse_claims(first_line: int, text: str) -> list[tuple[int, int, str]]:
    """Return ``(number, line, text)`` for every ``N. ...`` claim in the CLAIMS section body."""
    claims: list[tuple[int, int, str]] = []
    lines: list[str] = []
    for offset, line in enumerate(text.splitlines(), 1):
        start = _CLAIM_START.match(line)
        if start:
            if claims:
                claims[-1] = (claims[-1][0], claims[-1][1], "\n".join(lines).strip())
            claims.append((int(start.group(1)), first_line + offset, ""))
            lines = [start.group(2)]
        elif claims and line.strip():
            lines.append(line.strip())
    if claims:
        claims[-1] = (claims[-1][0], claims[-1][1], "\n".join(lines).strip())
    return claims


def claim_parents(text: str) -> tuple[list[int], bool]:
    """Return the claims *text* depends from and whether it names them in the alternative.

    An independent claim returns ``([], True)``.  ``claims 1 to 3`` and ``claims 1-3``
    expand to every claim in the range (a reversed range is read the same way, so the
    dependency is never lost); a list counts as alternative only when it is joined by
    ``or`` or follows ``any (one) of``/``either``; ``claims 1 and 2`` and the bare list
    ``claims 1, 2`` are not alternative (37 CFR 1.75(c), MPEP 608.01(n)).
    """
    m = _CLAIM_REFERENCE.search(text)
    if not m:
        return [], True
    tokens = re.findall(r"\d+|\b(?:or|and|to|through)\b|-", m.group(2).lower())
    numbers: list[int] = []
    joiners: list[str] = []
    for token in tokens:
        if not token.isdigit():
            joiners.append(token)
        elif joiners and joiners[-1] in ("to", "through", "-") and numbers:
            low, high = sorted((numbers.pop(), int(token)))
            numbers += list(range(low, high + 1))
        else:
            numbers.append(int(token))
    joiners = [j for j in joiners if j not in ("to", "through", "-")]
    alternative = len(numbers) < 2 or bool(m.group(1)) or ("or" in joiners and "and" not in joiners)
    return numbers, alternative


def _ancestors(n: int, parents: dict[int, tuple[list[int], bool]]) -> set[int]:
    """Return every claim that claim *n* depends from, directly or indirectly.

    Only earlier claims are followed (a forward or self reference is reported by
    the dependency gate), so the walk always ends.
    """
    found: set[int] = set()
    pending = [p for p in parents.get(n, ([], True))[0] if p < n]
    while pending:
        p = pending.pop()
        if p not in found:
            found.add(p)
            pending += [q for q in parents.get(p, ([], True))[0] if q < p]
    return found


def _fee_weight(n: int, parents: dict[int, tuple[list[int], bool]]) -> int:
    """Return how many claims claim *n* counts as for the USPTO fee (37 CFR 1.75(c)).

    A multiple dependent claim counts as the number of claims it refers to, and a
    claim depending from one counts as that number too, so the weight of a claim
    is the sum of its parents' weights (one for an independent claim).
    """
    earlier = [p for p in parents.get(n, ([], True))[0] if p < n]
    return sum(_fee_weight(p, parents) for p in earlier) if earlier else 1


def _singular(word: str) -> str:
    """Return a crude singular of *word* (``nodes``->``node``, ``entries``->``entry``)."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


_QUANTIFIERS = (("plurality", "of"), ("at", "least", "one"), ("one", "or", "more"), ("set", "of"))


def _phrase_after(tokens: list[str], index: int) -> list[str]:
    """Return up to four singularized words after *index* until a stop word or punctuation.

    A leading quantifier (``plurality of``, ``at least one``, ``one or more``, ``set of``)
    is skipped, so ``the plurality of nodes`` and ``a plurality of nodes`` both give ``node``.
    """
    lower = [t.lower() for t in tokens[index : index + 3]]
    for quantifier in _QUANTIFIERS:
        if tuple(lower[: len(quantifier)]) == quantifier:
            index += len(quantifier)
            break
    words: list[str] = []
    for token in tokens[index : index + 4]:
        if not token[0].isalpha() or token.lower() in _PHRASE_STOP:
            break
        words.append(_singular(token.lower()))
    return words


def _spans(words: list[str]) -> set[str]:
    """Return every contiguous span of *words* (``first node`` -> ``{first, node, first node}``).

    An introduced phrase may carry a pre-modifier (``a first sorted run``) or a
    post-modifier (``a spindle substantially aligned``); registering every span lets
    ``the run``, ``the sorted run`` and ``the spindle`` all find their antecedent.
    """
    return {" ".join(words[i:j]) for i in range(len(words)) for j in range(i + 1, len(words) + 1)}


def _prefixes(words: list[str]) -> set[str]:
    """Return every prefix of *words* (``sorted run`` -> ``{sorted, sorted run}``)."""
    return {" ".join(words[: i + 1]) for i in range(len(words))}


def antecedent_problems(text: str, known: set[str]) -> tuple[list[str], set[str]]:
    """Return the ``the X``/``said X`` phrases of claim *text* with no earlier introduction.

    *known* holds the phrase spans introduced by the parent claims; the returned
    set adds this claim's introductions (and its definite phrases, so a dependent
    claim does not repeat its parent's problem).  Introductions are ``a``, ``an``,
    ``at least one``, ``one or more``, ``a plurality of`` and ``each``; a definite
    phrase resolves when one of its prefixes (``the compaction merges`` ->
    ``compaction``) is a span of an introduced phrase (``a compaction``, ``a first
    sorted run`` -> ``sorted run``, ``run``).  ``the X of the Y`` is an inherent
    feature of Y and needs no introduction of its own (MPEP 2173.05(e): ``the outer
    surface of the sphere``); the Y phrase is checked when the scan reaches it.
    """
    tokens = _TOKEN.findall(text)
    lower = [t.lower() for t in tokens]
    known = set(known)
    problems: list[str] = []
    i = 0
    while i < len(tokens):
        introducer = next((n for n in (3, 2, 1) if " ".join(lower[i : i + n]) in _INTRODUCERS), 0)
        if introducer:
            known |= _spans(_phrase_after(tokens, i + introducer))
            i += introducer
            continue
        if lower[i] in ("the", "said"):
            words = _phrase_after(tokens, i + 1)
            after = lower[i + 1 + len(words) : i + 3 + len(words)]
            inherent = len(after) == 2 and after[0] == "of" and after[1] in ("the", "said")
            if words and words[0] not in _DEFINITE_SKIP and not (_prefixes(words) & known):
                if not inherent:
                    problems.append(f"{tokens[i]} {' '.join(tokens[i + 1 : i + 1 + len(words)])}")
                known |= _spans(words)
        i += 1
    return problems, known


def _sentence_count(claim: str) -> int:
    """Return how many sentences *claim* has (abbreviations and decimals are not boundaries)."""
    text = re.sub(_ABBREVIATIONS, "", claim)
    text = re.sub(r"\d\.\d", "00", text)
    return 1 + len(re.findall(r"[.!?]\s+(?=[A-Z(\"])", text))


def _claim_gates(draft: Draft, jurisdiction: str) -> list[str]:
    """Run the claim-set gates (37 CFR 1.75, MPEP 608.01(m)(n), 2173.05, 2181)."""
    claims = draft.claims
    texts = {n: t for n, _, t in claims}
    report: list[str] = []
    preamble = (
        [] if _CLAIMS_PREAMBLE.search(draft.text("CLAIMS")) else ["no 'What is claimed is:' line"]
    )
    report += _section("claims preamble missing", preamble, 0, "MPEP 608.01(m)")
    numbers = [n for n, _, _ in claims]
    expected = list(range(1, len(numbers) + 1))
    report += _section(
        "claim numbering",
        [f"claims numbered {numbers}, expected {expected}"] if numbers != expected else [],
        0,
        "consecutive Arabic numerals from 1 (37 CFR 1.75(f))",
    )
    sentences = [
        f"claim {n} (L{line}): {_sentence_count(t)} sentences"
        for n, line, t in claims
        if _sentence_count(t) != 1
    ]
    periods = [
        f"claim {n} (L{line}): no terminal period"
        for n, line, t in claims
        if not t.rstrip().endswith(".")
    ]
    report += _section("claims not one sentence", sentences + periods, 0, "MPEP 608.01(m)")
    parents = {n: claim_parents(t) for n, _, t in claims}
    multiple = {n for n, (p, _) in parents.items() if len(p) > 1}
    bad_dependency, not_alternative, chained = [], [], []
    for n, (p, alternative) in parents.items():
        for parent in p:
            if parent >= n or parent not in parents:
                bad_dependency.append(f"claim {n} depends from claim {parent}")
        if not alternative:
            not_alternative.append(
                f"claim {n}: refer to the claims in the alternative ('claim 1 or 2')"
            )
        if n in multiple and multiple & _ancestors(n, parents):
            chained.append(f"claim {n} is a multiple dependent claim depending from one")
    report += _section(
        "dependency on a later, missing or own claim", bad_dependency, 0, "37 CFR 1.75(c)"
    )
    strict = 0 if jurisdiction == "US" else -1
    report += _section(
        "multiple dependent claims not in the alternative",
        not_alternative,
        strict,
        "37 CFR 1.75(c)",
    )
    report += _section(
        "multiple dependent claim depending from another", chained, strict, "37 CFR 1.75(c)"
    )
    independent = [n for n, (p, _) in parents.items() if not p]
    no_independent = (
        ["no claims" if not claims else "no independent claim"] if not independent else []
    )
    report += _section("independent claims missing", no_independent, 0, "35 U.S.C. 112(b)")
    max_independent, max_total = FEE_THRESHOLDS.get(jurisdiction, FEE_THRESHOLDS["US"])
    counted = sum(_fee_weight(n, parents) for n in parents) if jurisdiction == "US" else len(claims)
    fees = []
    if max_independent and len(independent) > max_independent:
        fees.append(f"{len(independent)} independent claims ({max_independent} in the basic fee)")
    if counted > max_total:
        fees.append(f"{counted} claims counted for fees ({max_total} in the basic fee)")
    if multiple and jurisdiction == "US":
        fees.append(f"multiple dependent claims {sorted(multiple)}: 37 CFR 1.16(j) fee")
    report += _section("excess-claim fees", fees, -1, "37 CFR 1.16(h)-(j) or EPO Rule 45")
    known: dict[int, set[str]] = {}
    antecedent = []
    for n, line, t in claims:
        # Each parent of a multiple dependent claim is a separate alternative: a phrase
        # must have its antecedent in every one of them (37 CFR 1.75(c)).
        alternatives = parents[n][0] or [0]
        results = {p: antecedent_problems(t, known.get(p, set())) for p in alternatives}
        known[n] = set.intersection(*(known_p for _, known_p in results.values()))
        for parent, (problems, _) in results.items():
            via = f" via claim {parent}" if len(results) > 1 else ""
            antecedent += [
                f"claim {n} (L{line}): '{phrase}' has no antecedent{via}" for phrase in problems
            ]
    report += _section(
        "antecedent basis", antecedent, 0, "introduce with 'a'/'an' first (MPEP 2173.05(e))"
    )
    report += _section(
        "indefinite terms in claims",
        _claim_hits(claims, _INDEFINITE_TERMS),
        -1,
        "MPEP 2173.05(b),(d)",
    )
    report += _section(
        "means-plus-function in claims",
        _claim_hits(claims, _MEANS_PLUS_FUNCTION),
        -1,
        "112(f): the description must name the structure (MPEP 2181)",
    )
    report += _section("'said' in claims", _claim_hits(claims, r"\bsaid\b"), -1, "use 'the'")
    no_transition = [f"claim {n}" for n in independent if not _TRANSITION.search(texts[n])]
    report += _section(
        "independent claims without a transition", no_transition, -1, "comprising / consisting of"
    )
    no_narrowing = [
        f"claim {n}" for n, (p, _) in parents.items() if p and not _NARROWING.search(texts[n])
    ]
    report += _section(
        "dependent claims without a further limitation", no_narrowing, -1, "35 U.S.C. 112(d)"
    )
    if jurisdiction != "US":
        two_part = [
            f"claim {n}"
            for n in independent
            if not re.search(r"characteri[sz]ed in that", texts[n], re.I)
        ]
        report += _section(
            "independent claims not in two-part form", two_part, -1, "EPC Rule 43(1)"
        )
    return report


def _claim_hits(claims: list[tuple[int, int, str]], pattern: str) -> list[str]:
    """Return ``claim N: <term>`` for every match of *pattern* in the claims."""
    return [
        f"claim {n}: {m.group(0)}"
        for n, _, text in claims
        for m in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]


def _figure_ids(text: str) -> set[str]:
    """Return the figure identifiers *text* refers to (``FIGS. 1A and 2`` -> ``{1A, 2}``).

    A numeric range (``FIGS. 2-4``, ``FIGS. 2 to 4``) expands to every number in it.
    """
    ids: set[str] = set()
    for group in _FIGURE.findall(text):
        for m in re.finditer(r"(\d+)([A-Za-z]?)\b(?:\s*(?:to|through|-)\s*(\d+))?", group):
            if m.group(3):
                ids |= {str(n) for n in range(int(m.group(1)), int(m.group(3)) + 1)}
            else:
                ids.add(m.group(1) + m.group(2).upper())
    return ids


def _figure_gates(draft: Draft) -> list[str]:
    """Check figure cross-references, wording and reference numerals (37 CFR 1.84, MPEP 608.01(g)).

    Figure identifiers keep their partial-view suffix (``FIG. 1A``, 37 CFR 1.84(u)); the
    integer parts must run 1..N; when every embedded image is captioned ``FIG. N`` the
    captions are matched to the described figures one by one.
    """
    brief = _figure_ids(draft.text("BRIEF DESCRIPTION OF THE DRAWINGS"))
    detailed = _figure_ids(draft.text("DETAILED DESCRIPTION"))
    problems = [
        f"FIG. {n} in BRIEF DESCRIPTION but not in DETAILED DESCRIPTION"
        for n in sorted(brief - detailed)
    ]
    problems += [
        f"FIG. {n} in DETAILED DESCRIPTION but not in BRIEF DESCRIPTION"
        for n in sorted(detailed - brief)
    ]
    described = brief | detailed
    numbers = sorted({int(re.sub(r"[A-Z]", "", n)) for n in described})
    if numbers and numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"figures numbered {numbers}, expected 1..{len(numbers)}")
    captioned = [_figure_ids(caption) for _, caption, _ in draft.figures]
    if described and captioned and all(captioned):
        # Every image names its figure: match the identifiers, not just the counts.
        images = set().union(*captioned)
        problems += [f"FIG. {n} is described but has no image" for n in sorted(described - images)]
        problems += [
            f"image captioned FIG. {n} is not described" for n in sorted(images - described)
        ]
    elif described and captioned and len(captioned) != len(described):
        problems.append(
            f"{len(captioned)} image(s) embedded for {len(described)} figure identifier(s)"
        )
    report = _section("figure cross-references", problems, 0, "every FIG. N in both descriptions")
    report += _section(
        "figure wording", _hits(draft.description(), _FIGURE_WORDING), 0, "write 'FIG. N'"
    )
    nouns: dict[str, set[str]] = {}
    for noun, numeral in _REFERENCE_NUMERAL.findall(draft.text("DETAILED DESCRIPTION").lower()):
        if noun not in _NUMERAL_SKIP:
            nouns.setdefault(numeral, set()).add(_singular(noun))
    inconsistent = [
        f"{numeral}: {', '.join(sorted(n))}" for numeral, n in sorted(nouns.items()) if len(n) > 1
    ]
    report += _section(
        "reference numerals with several names",
        inconsistent,
        -1,
        "one element, one name, one numeral",
    )
    return report


def _unsupported_claim_terms(draft: Draft) -> list[str]:
    """Return claim words (four letters or more) the description never uses (37 CFR 1.75(d)(1)).

    Transitions (``comprising``, ``includes``) and ``means`` are claim language, not
    terms the description has to use, and are skipped.
    """
    described = {_singular(w) for w in re.findall(r"[a-z][a-z-]{2,}", draft.description().lower())}
    missing: dict[str, list[int]] = {}
    for n, _, text in draft.claims:
        for word in sorted({_singular(w) for w in re.findall(r"[a-z][a-z-]{3,}", text.lower())}):
            if word in described or word in _STOP_WORDS or word == "mean":
                continue
            if not _TRANSITION.fullmatch(word):
                missing.setdefault(word, []).append(n)
    return [f"'{word}' (claims {numbers})" for word, numbers in sorted(missing.items())]


def check_patent(draft_path: str, jurisdiction: str = "US") -> str:
    """Check a Markdown patent draft against the rules a proofreader and an examiner apply.

    Args:
        draft_path: Path of the ``.md`` (or ``.txt``) draft laid out with the 37 CFR 1.77
            headings (``BACKGROUND``, ``SUMMARY``, ``BRIEF DESCRIPTION OF THE DRAWINGS``,
            ``DETAILED DESCRIPTION``, ``CLAIMS``, ``ABSTRACT``) and ``N.`` numbered claims.
        jurisdiction: ``"US"`` (default: 37 CFR 1.75 claim rules, 3/20 fee thresholds) or
            ``"EP"`` (two-part form expected, 15-claim threshold, multiple dependencies
            only listed).

    Returns:
        One ``name: count (limit) PASS|FAIL|CHECK`` line per gate with the offending
        lines or claims indented under it, then ``summary: N gate(s) failed``.  Gates:
        sections present and in order, title length and wording, abstract length and
        wording, claim numbering, one sentence per claim, dependencies, antecedent
        basis, indefinite and means-plus-function terms, fee thresholds, claim terms
        the description never uses, figure cross-references and reference numerals,
        invention talk, prior-art admissions, absolute terms, and the shared AI-slop
        gates.  A FAIL is a defect to fix; a CHECK lists what to review.
    """
    path = Path(draft_path).expanduser()
    if not path.is_file():
        return f"no such file: {path}"
    source = path.read_text(encoding="utf-8", errors="replace")
    draft = Draft(source)
    report: list[str] = []
    present = [name for name, _ in draft.order]
    missing = [name for name, _, required in _SECTIONS if required and name not in present]
    if not draft.title:
        missing.insert(0, "TITLE (a '# <title>' line before the first section)")
    if "BRIEF DESCRIPTION OF THE DRAWINGS" not in present and _FIGURE.search(source):
        missing.append("BRIEF DESCRIPTION OF THE DRAWINGS (figures are mentioned)")
    report += _section("sections missing", missing, 0, "37 CFR 1.77(b)")
    order = [f"{name} L{line}" for name, line in draft.order]
    sorted_order = [n for n, _ in sorted(draft.order, key=lambda item: _SECTION_INDEX[item[0]])]
    out_of_order = [", ".join(order)] if present != sorted_order else []
    report += _section("sections out of order", out_of_order, 0, "37 CFR 1.77(b) order")
    duplicates = [n for n in set(present) if present.count(n) > 1]
    report += _section("sections repeated", sorted(duplicates), 0)
    report += _section(
        "title characters",
        [draft.title] if len(draft.title) > TITLE_LIMIT else [],
        0,
        f"{len(draft.title)} characters (limit {TITLE_LIMIT}, 37 CFR 1.72(a))",
    )
    report += _section("title wording", _hits(draft.title, _TITLE_WORDING), 0, "MPEP 606")
    abstract = draft.text("ABSTRACT").strip()
    words = len(abstract.split())
    paragraphs = [p for p in re.split(r"\n\s*\n", abstract) if p.strip()]
    long = [f"{words} words"] if words > ABSTRACT_LIMIT else []
    report += _section(
        "abstract words over limit",
        long,
        0,
        f"{words} words (limit {ABSTRACT_LIMIT}, 37 CFR 1.72(b))",
    )
    one_paragraph = (
        []
        if len(paragraphs) == 1 or "ABSTRACT" not in draft.sections
        else [f"{len(paragraphs)} paragraphs"]
    )
    report += _section("abstract paragraphs", one_paragraph, 0, "exactly one paragraph")
    report += _section(
        "abstract wording",
        _hits(abstract, _ABSTRACT_WORDING),
        0,
        "MPEP 608.01(b): no 'said', 'means', merits",
    )
    report += _claim_gates(draft, jurisdiction)
    report += _section(
        "claim terms absent from the description",
        _unsupported_claim_terms(draft),
        -1,
        "37 CFR 1.75(d)(1)",
    )
    report += _figure_gates(draft)
    description = draft.description()
    report += _section(
        "invention talk",
        _hits(description, _INVENTION_TALK),
        0,
        "say 'the disclosure' or 'some embodiments'",
    )
    report += _section(
        "prior-art admissions",
        _hits(description, _PRIOR_ART_ADMISSIONS),
        0,
        "describe the problem, do not characterize the art",
    )
    report += _section(
        "absolute terms",
        _hits(description, _ABSOLUTE_TERMS),
        -1,
        "each one narrows the claims; keep only if truly indispensable",
    )
    prose = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("|", "```", "!["))
    )
    for name, pattern, limit, note in _PROSE_GATES:
        report += _section(name, _hits(prose, pattern), limit, note)
    report += _section("duplicated sentences", _duplicated_sentences(prose), 0)
    report.append(
        f"words: {len(description.split())} description, {words} abstract; "
        f"{len(draft.claims)} claims"
    )
    failed = sum(1 for line in report if line.endswith(" FAIL") or " FAIL; " in line)
    report.append(f"summary: {failed} gate(s) failed")
    return "\n".join(report)


def _plain(text: str) -> str:
    """Return Markdown *text* as plain text: no emphasis, code ticks or link targets."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(\*\*|__|`)", "", text)
    return re.sub(r"(?<!\w)\*(?!\s)([^*]+)\*", r"\1", text)


_TABLE_SEPARATOR = re.compile(r"\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?")
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def _table_cells(line: str) -> list[str]:
    """Split one pipe-table *line* into plain-text cells at its unescaped pipes.

    At most one outer pipe is removed at each end, so an empty first or last cell
    (``||middle||``) keeps its column; ``\\|`` inside a cell becomes a literal pipe.
    """
    line = line.removeprefix("|")
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    return [_plain(cell.strip()).replace("\\|", "|") for cell in _UNESCAPED_PIPE.split(line)]


def table_rows(paragraph: str) -> list[list[str]]:
    """Return the cell rows of a Markdown pipe table *paragraph*, or ``[]`` if it is not one.

    A table is a header line, a ``---|---`` separator line and data lines that each
    contain an unescaped pipe; outer pipes are optional.  The separator line is
    dropped and every row is padded to the width of the widest row.
    """
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if len(lines) < 2 or not _TABLE_SEPARATOR.fullmatch(lines[1]):
        return []
    if not all(_UNESCAPED_PIPE.search(line) for line in lines):
        return []
    rows = [_table_cells(line) for line in lines[:1] + lines[2:]]
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def blocks(draft: Draft) -> list[tuple[str, str]]:
    """Return the draft as ``(kind, text)`` blocks in filing order.

    Kinds: ``title``, ``heading``, ``para`` (a numbered specification paragraph),
    ``claim`` (lines joined by newlines), ``free`` (the claims preamble and the
    abstract, which are not numbered) and ``table`` (a Markdown pipe table: rows
    joined by newlines, cells by tabs, header row first).  Figures are not blocks:
    the specification must not contain drawings (37 CFR 1.58(a)); ``build_patent``
    puts them on sheets.
    """
    out: list[tuple[str, str]] = [("title", _plain(draft.title))]
    for name, _ in draft.order:
        out.append(("heading", name))
        if name == "CLAIMS":
            body = draft.text(name).splitlines()
            if draft.claims:
                body = body[: draft.claims[0][1] - draft.sections[name][0] - 1]
            out += [("free", _plain(p.strip())) for p in body if p.strip()]
            out += [("claim", f"{n}. " + _plain(t)) for n, _, t in draft.claims]
            continue
        kind = "free" if name == "ABSTRACT" else "para"
        for paragraph in re.split(r"\n\s*\n", draft.text(name)):
            rows = table_rows(paragraph)
            if rows:
                out.append(("table", "\n".join("\t".join(row) for row in rows)))
                continue
            text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
            if text:
                out.append((kind, _plain(text)))
    return out


def _docx_table(text: str) -> str:
    """Return a bordered 10 pt WordprocessingML table of the tab-separated *text* rows."""
    border = "".join(
        f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        for side in ("top", "left", "bottom", "right", "insideH", "insideV")
    )
    rows = []
    for i, line in enumerate(text.split("\n")):
        cells = "".join(
            "<w:tc><w:p><w:pPr><w:spacing w:after=\"0\"/></w:pPr>"
            + _docx_run(escape(cell), bold=i == 0, size=20)
            + "</w:p></w:tc>"
            for cell in line.split("\t")
        )
        rows.append(f"<w:tr>{cells}</w:tr>")
    return (
        f'<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/><w:tblBorders>{border}</w:tblBorders>'
        f"</w:tblPr>{''.join(rows)}</w:tbl><w:p/>"
    )


def _docx_paragraph(
    text: str, number: str = "", center: bool = False, new_page: bool = False
) -> str:
    """Return one WordprocessingML paragraph: 12 pt Times, 1.5 spacing, optional bold number."""
    props = ('<w:jc w:val="center"/>' if center else "") + (
        "<w:pageBreakBefore/>" if new_page else ""
    )
    props += '<w:spacing w:line="360" w:lineRule="auto" w:after="240"/>'
    runs = ""
    if number:
        runs += _docx_run(f"{number}    ", bold=True)
    for i, line in enumerate(text.split("\n")):
        runs += _docx_run(("    " if i else "") + escape(line), new_line=bool(i))
    return f"<w:p><w:pPr>{props}</w:pPr>{runs}</w:p>"


def _docx_run(text: str, bold: bool = False, new_line: bool = False, size: int = 24) -> str:
    """Return one Times run of *text* at *size* half-points, optionally bold, after a line break."""
    font = f'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="{size}"/>'
    props = f"<w:rPr>{'<w:b/>' if bold else ''}{font}</w:rPr>"
    br = "<w:br/>" if new_line else ""
    return f'<w:r>{props}{br}<w:t xml:space="preserve">{text}</w:t></w:r>'


_DOCX_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" '
    'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/'
    'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)
_DOCX_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
    'relationships/officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)
_DOCX_SECTION = (
    '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
    '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1440" '
    'w:header="708" w:footer="708" w:gutter="0"/>'
    "</w:sectPr>"
)


def write_docx(path: Path, parts: list[tuple[str, str]]) -> int:
    """Write *parts* to a ``.docx`` at *path* and return the number of numbered paragraphs.

    Letter paper, margins of 37 CFR 1.52(a) (top, right, bottom 2 cm; left 2.5 cm),
    1.5 line spacing, 12 pt Times; headings uppercase and centered; paragraphs
    numbered ``[0001]`` in bold; CLAIMS and ABSTRACT start new pages.
    """
    body, count = [], 0
    for kind, text in parts:
        if kind == "title":
            body.append(_docx_paragraph(text, center=True))
        elif kind == "heading":
            body.append(_docx_paragraph(text, center=True, new_page=text in ("CLAIMS", "ABSTRACT")))
        elif kind == "para":
            count += 1
            body.append(_docx_paragraph(text, number=f"[{count:04d}]"))
        elif kind == "table":
            body.append(_docx_table(text))
        else:
            body.append(_docx_paragraph(text))
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body)}{_DOCX_SECTION}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _DOCX_TYPES)
        zf.writestr("_rels/.rels", _DOCX_RELS)
        zf.writestr("word/document.xml", document)
    return count


_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    **{char: "\\" + char for char in "&%$#_{}"},
}


def _tex_escape(match: re.Match[str]) -> str:
    """Return the LaTeX escape of the special character *match* found."""
    return _TEX_ESCAPES[match.group()]


def _tex(text: str) -> str:
    """Escape *text* for LaTeX in one pass, so an escape's own braces are not re-escaped."""
    return re.sub(r"[\\~^&%$#_{}]", _tex_escape, text)


_TEX_HEAD = (
    "\\documentclass[12pt]{article}\n"
    "\\usepackage[letterpaper,top=2cm,left=2.5cm,right=2cm,bottom=2cm]{geometry}\n"
    "\\usepackage[T1]{fontenc}\n\\usepackage{graphicx}\n\\usepackage{tabularx}\n"
    "\\renewcommand{\\baselinestretch}{1.5}\n\\setlength{\\parindent}{0pt}\n"
    "\\setlength{\\parskip}{0.6em}\n\\begin{document}\n"
)


def _tex_table(text: str) -> str:
    """Return a ``tabularx`` of the tab-separated *text* rows: wrapping columns for long cells."""
    rows = [line.split("\t") for line in text.split("\n")]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    spec = "".join("X" if width > 12 else "l" for width in widths)
    if "X" not in spec:
        spec = "X" + spec[1:]
    lines = [
        " & ".join(f"\\textbf{{{_tex(cell)}}}" if i == 0 else _tex(cell) for cell in row)
        for i, row in enumerate(rows)
    ]
    body = " \\\\\n\\hline\n".join(lines)
    return (
        "\\begin{center}\\renewcommand{\\baselinestretch}{1.1}\\small\n"
        f"\\begin{{tabularx}}{{\\textwidth}}{{|{'|'.join(spec)}|}}\n\\hline\n{body} \\\\\n\\hline\n"
        "\\end{tabularx}\\end{center}\n\n"
    )


def specification_tex(parts: list[tuple[str, str]]) -> str:
    """Return the LaTeX source of the specification *parts* (same layout as the DOCX)."""
    out, count = [_TEX_HEAD], 0
    for kind, text in parts:
        if kind == "table":
            out.append(_tex_table(text))
        elif kind == "title":
            out.append(f"\\begin{{center}}{_tex(text)}\\end{{center}}\n")
        elif kind == "heading":
            page = "\\newpage\n" if text in ("CLAIMS", "ABSTRACT") else ""
            out.append(f"{page}\\begin{{center}}{_tex(text)}\\end{{center}}\n")
        elif kind == "para":
            count += 1
            out.append(f"\\textbf{{[{count:04d}]}}\\hspace{{1em}}{_tex(text)}\n\n")
        else:
            lines = [_tex(line) for line in text.split("\n")]
            out.append("\\newline\\hspace*{2em}".join(lines) + "\n\n")
    out.append("\\end{document}\n")
    return "".join(out)


def drawings_tex(figures: list[tuple[int, str, str]]) -> str:
    """Return LaTeX putting each figure on its own sheet, labeled ``FIG. N`` and ``Sheet i/n``."""
    out = [_TEX_HEAD.replace("\\begin{document}", "\\pagestyle{empty}\n\\begin{document}")]
    for i, (_, caption, path) in enumerate(figures, 1):
        label = _tex(caption) if caption else f"FIG. {i}"
        out.append(
            f"\\noindent\\hfill Sheet {i}/{len(figures)}\\par\\vfill\\begin{{center}}"
            f"\\includegraphics[width=0.9\\textwidth,height=0.75\\textheight,keepaspectratio]"
            f"{{{path}}}\\par\\vspace{{2em}}{label}\\end{{center}}\\vfill\\newpage\n"
        )
    out.append("\\end{document}\n")
    return "".join(out)


def _pdf(tex_path: Path, source: str, tex_bin: str) -> list[str]:
    """Write *source* to *tex_path*, run pdflatex twice and report errors, pages and the PDF."""
    tex_path.write_text(source, encoding="utf-8")
    pdflatex = _find_tex_binary("pdflatex", tex_bin)
    if not Path(pdflatex).is_file():
        return [
            f"{tex_path.with_suffix('.pdf').name}: pdflatex not found "
            f"({pdflatex or 'PATH and TeX Live'}), LaTeX source written to {tex_path}"
        ]
    command = [pdflatex, "-interaction=nonstopmode", "-file-line-error", tex_path.name]
    _run(command, tex_path.parent)
    code, log = _run(command, tex_path.parent)
    errors, _, _, pages = _summarize_log(log)
    report = _section(f"{tex_path.name} errors", errors, 0)
    pdf = tex_path.with_suffix(".pdf")
    report.append(
        f"pdf: {pdf} ({pages} pages)"
        if pdf.is_file() and not code
        else f"pdf: {pdf} not written FAIL"
    )
    return report


def build_patent(draft_path: str, tex_bin: str = "") -> str:
    """Render a Markdown patent draft to a filing-shaped DOCX, a PDF and a drawings PDF.

    Args:
        draft_path: Path of the ``.md`` draft (the layout ``check_patent`` expects).
        tex_bin: Directory holding ``pdflatex`` (default: ``PATH``, then TeX Live).

    Returns:
        The ``.docx`` path (always written: USPTO Patent Center format, 37 CFR 1.52
        margins and spacing, ``[0001]`` paragraph numbers, CLAIMS and ABSTRACT on new
        pages), then the specification ``.pdf`` and the ``_drawings.pdf`` (one
        ``FIG. N`` per sheet) when ``pdflatex`` is installed, with LaTeX errors and
        page counts, and the counts of numbered paragraphs, claims and abstract words.
    """
    path = Path(draft_path).expanduser().resolve()
    if not path.is_file():
        return f"no such file: {path}"
    draft = Draft(path.read_text(encoding="utf-8", errors="replace"))
    parts = blocks(draft)
    docx = path.with_suffix(".docx")
    numbered = write_docx(docx, parts)
    report = [f"docx: {docx}"]
    report += _pdf(path.with_suffix(".tex"), specification_tex(parts), tex_bin)
    if draft.figures:
        report += _pdf(
            path.with_name(f"{path.stem}_drawings.tex"), drawings_tex(draft.figures), tex_bin
        )
    else:
        report.append("drawings: none (no ![FIG. N](path) images in the draft)")
    abstract_words = len(draft.text("ABSTRACT").split())
    report.append(
        f"paragraphs numbered: {numbered}; claims: {len(draft.claims)}; "
        f"abstract words: {abstract_words}"
    )
    return "\n".join(report)


class WritePatentSea(BaseSea):
    """The ``/write_patent`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/write_patent help``."""
        return (
            "Writes or revises a utility patent application (37 CFR 1.77 layout, claims, "
            "abstract, drawings) from an invention disclosure, with proofreader gates "
            "(`check_patent`) and a DOCX/PDF build (`build_patent`); use `/write_patent "
            '<sources, draft path, jurisdiction, options>` or run_agent(agent="write_patent").'
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Append the patent-drafting rules to the default system prompt."""
        return system_prompt + "\n\n" + SYSTEM_PROMPT

    def tools(self, tools: list[Any]) -> list[Any]:
        """Expose the gate checker and the document builder to the model."""
        return tools + [check_patent, build_patent]

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Browse (prior art, rules), skip the classifier, allow a long run."""
        return settings | {
            "use_web_tools": True,
            "auto_classify": False,
            "timeout": DISPATCH_TIMEOUT_SECONDS,
        }


DISPATCH_TIMEOUT_SECONDS = 21600
"""Seconds a ``run_agent`` call waits for a ``/write_patent`` run: six hours."""
