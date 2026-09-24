# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/review_paper`` agent (``review_paper_sea``).

The reader tests run :func:`read_paper` on a small PDF written by hand
into ``tmp_path`` (three pages, margin line numbers like a conference
template, a table column) through the ``pdftotext`` installed on the
machine; they are skipped when poppler is absent, since the tool shells
out to it and there is no substitute.  The gate tests run
:func:`check_review` on real review files.  The agent-level test runs a
real :class:`ChatSorcarAgent` ReAct loop against the scripted local
chat-completions server configured as the daemon configures it from the
SEA's getters; the only replaced boundary is the LLM endpoint, and the
tools really run.

Branches not covered here: ``pdftotext`` exiting non-zero on a file that
``pdfinfo`` accepted, and its timeout; both need a corrupt PDF that
poppler half-reads, which no test double can produce faithfully.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.seas import review_paper_sea, write_paper_sea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

_SEA_PATH = Path(review_paper_sea.__file__).resolve()

# Page one imitates a conference template: 24 margin line numbers in runs of 8 around
# the text, plus a short table column (500, 501, 502) that must survive.
_PAGE_ONE = (
    ["Under review at ICLR 2027"]
    + [f"{n:03d}" for n in range(0, 8)]
    + ["Abstract", "We measure it in real use over 329 tasks."]
    + [f"{n:03d}" for n in range(8, 16)]
    + ["Table 0", "500", "501", "502"]
    + [f"{n:03d}" for n in range(16, 24)]
)
# Page two has no margin numbers: its counting column (100, 101, 102) is data.
_PAGE_TWO = ["Table 1", "runs", "100", "101", "102", "42", "43", "300", "301", "end of table"]
_PAGE_THREE = ["Appendix", "Nothing else."]


def _pdf(pages: list[list[str]]) -> bytes:
    """Return a valid single-font PDF with one text line per list item on each page."""
    objects: list[bytes] = []
    kids = [f"{4 + 2 * i} 0 R" for i in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, lines in enumerate(pages):
        content = "BT /F1 12 Tf 14 TL 72 720 Td " + " ".join(f"({t}) Tj T*" for t in lines) + " ET"
        page = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page + 1} 0 R >>".encode()
        )
        objects.append(f"<< /Length {len(content)} >>\nstream\n{content}\nendstream".encode())
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    out += trailer.encode()
    return bytes(out)


_SLOPPY_REVIEW = """\
Summary
The paper measures a writer/reviewer pairing on a two-month log --- 329 tasks. It is careful.

Strengths
- Real deployment data, 440K events, rather than a benchmark.

Weaknesses
- Table 3 has no spread; the paper would benefit from more baselines.

Detailed review
Overall, this paper is well-written. It delves into a \u201cnew\u201d protocol. It seems that
the numbers hold. The closest prior work is not
named. The closest prior work is not named.
"""

_CLEAN_REVIEW = """\
Summary
The paper measures a two-model review protocol on the full event log of a two-month
deployment (329 tasks) and reports 1,115 substantiated reviewer findings. The data is
unusual and the audit of the protocol itself is the main contribution. I lean toward
acceptance once the confidence intervals in Table 3 are added.

Strengths:
- The log is complete: every tool call and per-step cost (Section 3, Table 1).
- The leak analysis (0.14% of reviewer calls) checks the protocol, not only the model.

Weaknesses:
- Table 3 reports medians of 3 runs without spread, so the 4% gap between rows 2 and
  4 may be noise; report the range or add runs.
- The finding taxonomy (Section 4.2) was labeled by one author; a second labeler and
  an agreement statistic would settle the 1,115 count.

Detailed review
Novelty. The closest prior work is Smith and Jones (USENIX ATC 2025), which measured a
reviewer model on a benchmark of 300 synthetic tasks, and Lee (ICSE 2024), which
red-teamed a single model. This paper adds a production log and a leak audit. """ + " ".join(
    f"I checked ratio {i} in Section 5.{i} against Table 2." for i in range(12)
)


def _gate(report: str, name: str) -> str:
    """Return the header line of gate *name* in *report*."""
    return next(line for line in report.splitlines() if line.startswith(f"{name}: "))


def test_sea_getters_follow_the_user_contract() -> None:
    """The SEA appends the reviewing rules, offers the two tools, browses and fans out."""
    prompt = review_paper_sea.append_to_system_prompt()
    assert prompt == review_paper_sea.SYSTEM_PROMPT
    assert "William Strunk Jr. and E. B. White" in prompt
    assert review_paper_sea.SECOND_OPINION_MODEL in prompt
    assert f"(default {review_paper_sea.DEFAULT_WORD_LIMIT}\nwords)" in prompt
    assert "`read_paper`" in prompt and "`check_review`" in prompt
    assert "Judge the novelty" in prompt and "at least 10 distinct sources" in prompt
    assert "Em dashes: zero" in prompt
    assert [t.__name__ for t in review_paper_sea.tools()] == ["read_paper", "check_review"]
    assert review_paper_sea.use_web_tools() is True
    assert review_paper_sea.is_parallel() is True
    assert review_paper_sea.classify_tasks() is False
    assert not hasattr(review_paper_sea, "system_prompt")
    # The paper's word gates are reused minus the one about draft talk (a review
    # is allowed to say "reviewer" and "submission").
    names = [gate[0] for gate in review_paper_sea._REVIEW_GATES]
    assert "draft and review talk" not in names
    assert names[: len(write_paper_sea._GATES) - 1] == [
        g[0] for g in write_paper_sea._GATES if g[0] != "draft and review talk"
    ]


def test_slash_review_paper_resolves_to_the_bundled_sea() -> None:
    """``/review_paper <instructions>`` is rewritten into a ``run_agent`` directive on this file."""
    assert sea_commands.get_command("review_paper") == _SEA_PATH
    rewritten = sea_commands.rewrite_prompt_if_command("/review_paper Review x.pdf for ICLR")
    assert rewritten is not None
    prompt, path = rewritten
    assert path == _SEA_PATH
    assert f'agent = "{_SEA_PATH}"' in prompt
    assert prompt.endswith("TASK TEXT FOR run_agent:\nReview x.pdf for ICLR")


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler not installed")
def test_read_paper_pages_a_pdf_and_drops_margin_numbers(tmp_path: Path) -> None:
    """Pages come back one at a time with the template's line numbers gone and tables kept."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_pdf([_PAGE_ONE, _PAGE_TWO, _PAGE_THREE]))
    whole = review_paper_sea.read_paper(str(pdf))
    assert whole.startswith(f"paper: {pdf} (3 pages)\n=== page 1 ===\n")
    assert "=== page 2 ===" in whole and "=== page 3 ===\nAppendix\nNothing else." in whole
    page_one = whole.split("=== page 2 ===")[0]
    assert "We measure it in real use over 329 tasks." in page_one
    # The three runs of eight continue one another (000-023): a chain of 24
    # margin numbers, dropped.  The table column 500-502 is its own short chain
    # and stays.
    bare = [line.strip() for line in page_one.splitlines() if line.strip().isdigit()]
    assert bare == ["500", "501", "502"], bare
    page_two = whole.split("=== page 2 ===")[1].split("=== page 3 ===")[0]
    # A page without margin numbers keeps every column, the counting run of
    # three (100-102) included.
    for value in ("100", "101", "102", "42", "43", "300", "301"):
        assert f"\n{value}\n" in page_two, page_two

    middle = review_paper_sea.read_paper(str(pdf), first_page=2, last_page=2)
    assert middle.count("=== page") == 1 and "=== page 2 ===" in middle
    assert "Abstract" not in middle and "Appendix" not in middle
    tail = review_paper_sea.read_paper(str(pdf), first_page=3, last_page=99)
    assert tail.count("=== page") == 1 and "Appendix" in tail
    beyond = review_paper_sea.read_paper(str(pdf), first_page=4)
    assert beyond == f"paper: {pdf} (3 pages)"
    # first_page below 1 is clamped to 1.
    assert review_paper_sea.read_paper(str(pdf), first_page=0, last_page=1).count("=== page") == 1


def test_read_paper_handles_text_files_missing_files_and_bad_pdfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .tex is one page as is; a missing file, a fake PDF and a missing poppler are errors."""
    tex = tmp_path / "paper.tex"
    tex.write_text("\\section{Intro}\nWe measure.", encoding="utf-8")
    assert review_paper_sea.read_paper(str(tex)) == (
        f"paper: {tex} (1 page)\n=== page 1 ===\n\\section{{Intro}}\nWe measure."
    )
    missing = tmp_path / "gone.pdf"
    assert review_paper_sea.read_paper(str(missing)) == f"error: no such file: {missing}"
    fake = tmp_path / "fake.pdf"
    fake.write_text("not a pdf", encoding="utf-8")
    if shutil.which("pdftotext"):
        report = review_paper_sea.read_paper(str(fake))
        assert report.startswith(f"error: pdfinfo could not read {fake}")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert review_paper_sea.read_paper(str(fake)) == (
        "error: pdftotext not found: install poppler (brew install poppler)"
    )


def test_check_review_flags_the_sloppy_review(tmp_path: Path) -> None:
    """Every structure and word gate fires on the sloppy review with the right line numbers."""
    path = tmp_path / "review.txt"
    path.write_text(_SLOPPY_REVIEW, encoding="utf-8")
    report = review_paper_sea.check_review(str(path))
    assert _gate(report, "words").startswith("words: ") and report.count("(under 700) PASS") == 1
    assert _gate(report, "headings missing") == "headings missing: 0 (limit 0) PASS"
    assert "headings out of order" not in report
    assert _gate(report, "summary sentences") == "summary sentences: 2 (2 to 3) PASS"
    assert _gate(report, "strengths bullets") == "strengths bullets: 1 (at least 1) PASS"
    assert _gate(report, "weaknesses bullets") == "weaknesses bullets: 1 (at least 1) PASS"
    assert _gate(report, "detailed review words").endswith("(at least 100) FAIL")
    assert _gate(report, "years named").startswith("years named: 0 ")
    assert "CHECK; name the closest prior work with its year" in report
    assert _gate(report, "em dashes").startswith("em dashes: 1 (limit 0) FAIL")
    assert "    L2: ..." in report
    assert _gate(report, "antithesis").startswith("antithesis: 1 (limit 5) PASS")
    assert _gate(report, "slop vocabulary") == "slop vocabulary: 1 (limit 0) FAIL"
    assert "    L11: ...Overall, this paper is well-written. It delves into" in report
    assert _gate(report, "markdown artifacts") == "markdown artifacts: 2 (limit 0) FAIL"
    boilerplate = _gate(report, "reviewer boilerplate")
    assert boilerplate.startswith("reviewer boilerplate: 4 (limit 0) FAIL")
    assert "    L8: ...- Table 3 has no spread; the paper would benefit from" in report
    assert _gate(report, "hedging").startswith("hedging: 1 (list only) CHECK")
    assert "It seems that" in report
    assert _gate(report, "markdown headings") == (
        "markdown headings: 0 (limit 0) PASS; plain-text headings on their own line"
    )
    # The first copy is split across a hard line break; unwrapping finds it anyway,
    # and both copies are reported at the paragraph's first line.
    assert _gate(report, "duplicated sentences") == "duplicated sentences: 1 (limit 0) FAIL"
    assert "    L11, L11: the closest prior work is not named." in report
    assert _gate(report, "non-ASCII lines").startswith("non-ASCII lines: 1 (list only) CHECK")
    assert report.endswith("summary: 6 gate(s) failed"), report


def test_check_review_passes_a_clean_review_and_reports_structure_faults(tmp_path: Path) -> None:
    """A clean review fails nothing; a short limit, missing and disordered headings fail."""
    path = tmp_path / "review.txt"
    path.write_text(_CLEAN_REVIEW, encoding="utf-8")
    report = review_paper_sea.check_review(str(path))
    assert report.endswith("summary: 0 gate(s) failed"), report
    assert _gate(report, "summary sentences") == "summary sentences: 3 (2 to 3) PASS"
    assert _gate(report, "strengths bullets") == "strengths bullets: 2 (at least 1) PASS"
    assert _gate(report, "years named") == (
        "years named: 2 (a rough proxy for named prior work) PASS; 2024, 2025"
    )
    words = len(_CLEAN_REVIEW.split())
    assert _gate(report, "words") == f"words: {words} (under 700) PASS"
    report = review_paper_sea.check_review(str(path), word_limit=words)
    assert _gate(report, "words") == f"words: {words} (under {words}) FAIL"

    # Markdown headings are still recognized as headings (and flagged); a numbered
    # bullet counts; out-of-order and missing headings are reported.
    path.write_text(
        "## Weaknesses\n1. Table 2 lacks units; add them.\n\n**Strengths**\n* Real data.\n\n"
        "Detailed review:\nOne sentence only.\n",
        encoding="utf-8",
    )
    report = review_paper_sea.check_review(str(path))
    assert _gate(report, "headings missing") == "headings missing: 1 (limit 0) FAIL"
    assert "    Summary\n" in report
    assert "headings out of order: Strengths L4, Weaknesses L1, Detailed review L7 FAIL" in report
    assert "summary sentences" not in report
    assert _gate(report, "strengths bullets") == "strengths bullets: 1 (at least 1) PASS"
    assert _gate(report, "weaknesses bullets") == "weaknesses bullets: 1 (at least 1) PASS"
    assert _gate(report, "detailed review words") == "detailed review words: 3 (at least 100) FAIL"
    assert _gate(report, "markdown headings").startswith("markdown headings: 1 (limit 0) FAIL")
    assert _gate(report, "markdown artifacts").startswith("markdown artifacts: 1 (limit 0) FAIL")

    # An empty Summary and a Strengths list without bullets fail their gates; a
    # repeated heading does not start a new part.
    body = "Summary\nStrengths\nnone\nWeaknesses\n- x\nDetailed review\nSummary\n" + "w " * 100
    path.write_text(body, encoding="utf-8")
    report = review_paper_sea.check_review(str(path))
    assert _gate(report, "summary sentences") == "summary sentences: 0 (2 to 3) FAIL"
    assert _gate(report, "strengths bullets") == "strengths bullets: 0 (at least 1) FAIL"
    assert _gate(report, "detailed review words") == (
        "detailed review words: 101 (at least 100) PASS"
    )
    # Text before the first heading is ignored; parts that are absent have no gate line.
    path.write_text("Reviewer 2\nSummary\nA. B.\nStrengths\n- x\n", encoding="utf-8")
    report = review_paper_sea.check_review(str(path))
    assert _gate(report, "headings missing") == "headings missing: 2 (limit 0) FAIL"
    assert "    Weaknesses\n    Detailed review\n" in report
    assert _gate(report, "summary sentences") == "summary sentences: 2 (2 to 3) PASS"
    assert "weaknesses bullets" not in report and "detailed review words" not in report
    missing = tmp_path / "nope.txt"
    assert review_paper_sea.check_review(str(missing)) == f"no such file: {missing}"


def test_agent_gets_the_rules_and_the_tools_and_the_real_reports(tmp_path: Path) -> None:
    """With the SEA's configuration the model sees ``read_paper``/``check_review`` and real output.

    The scripted model reads a .tex paper, checks the sloppy review and finishes
    with the gate summary.  The test checks the offered tools, that the default
    system prompt was kept and the rules appended, and that both tool results
    flowed back through the tool-result messages.
    """
    paper = tmp_path / "paper.tex"
    paper.write_text("\\section{Intro}\nWe measure 329 tasks.", encoding="utf-8")
    review = tmp_path / "review.txt"
    review.write_text(_SLOPPY_REVIEW, encoding="utf-8")
    script = [
        tool_call_body("read_paper", {"paper_path": str(paper)}, prompt_tokens=500),
        tool_call_body("check_review", {"review_path": str(review)}, prompt_tokens=600),
        finish_body("<pre>summary: 6 gate(s) failed</pre>", prompt_tokens=700),
    ]
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("review-paper-sea-test")
        result = agent.run(
            prompt_template=f"Review {paper} for ICLR 2027; write to {review}.",
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=5,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            system_prompt=review_paper_sea.append_to_system_prompt(),
            tools=review_paper_sea.tools(),
            web_tools=review_paper_sea.use_web_tools(),
            is_parallel=review_paper_sea.is_parallel(),
            verbose=False,
        )
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
    assert "6 gate(s) failed" in parsed["summary"]

    agentic: list[dict[str, Any]] = [r for r in requests if r.get("tools")]
    assert len(agentic) == 3, [list(r) for r in requests]
    for request in agentic:
        names = {t["function"]["name"] for t in request["tools"]}
        expected = {"read_paper", "check_review", "Bash", "run_parallel", "go_to_url", "finish"}
        assert expected <= names, names
        system = str(next(m for m in request["messages"] if m["role"] == "system")["content"])
        assert not system.startswith(review_paper_sea.SYSTEM_PROMPT)
        assert "# Paper-reviewing agent" in system
        assert "Judge the novelty" in system
    tool_results = [m for m in agentic[2]["messages"] if m["role"] == "tool"]
    assert len(tool_results) == 2, agentic[2]["messages"]
    assert f"paper: {paper} (1 page)\n=== page 1 ===\n\\section{{Intro}}" in str(
        tool_results[0]["content"]
    )
    assert "summary: 6 gate(s) failed" in str(tool_results[1]["content"])
    assert "reviewer boilerplate: 4 (limit 0) FAIL" in str(tool_results[1]["content"])
