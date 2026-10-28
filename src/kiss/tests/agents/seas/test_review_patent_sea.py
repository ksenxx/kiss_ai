# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/review_patent`` agent.

Module under test: :mod:`kiss.agents.seas.review_patent.review_patent_sea`.  The gate
tests run ``check_patent_review`` on real review texts, ``read_patent`` on a draft and
on a PDF built by ``build_patent`` (skipped without pdflatex or poppler), and the agent
test runs the SEA against the scripted model server (``conftest.run_sea``).
"""

import shutil
from pathlib import Path

import pytest

from kiss.agents.seas.review_patent import review_patent_sea
from kiss.agents.seas.write_patent import write_patent_sea
from kiss.agents.sorcar import sea_commands
from kiss.tests.agents.seas.conftest import gate, run_sea
from kiss.tests.agents.seas.sea_contract import system_message, tool_names, tool_results
from kiss.tests.agents.seas.test_write_patent_sea import _CLEAN_DRAFT, _write
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body

_GOOD_REVIEW = """\
Summary
The draft claims a log-structured key-value store with tiered compaction. The field is storage
engines. The closest art is US 2015/0120567 A1. The independent claims are likely anticipated as
filed.

Claims at issue
Claim 1: 1[a] receiving a write request at a storage node; 1[b] appending to a write-ahead log;
1[c] storing in a first sorted run; 1[d] scheduling a compaction on a size ratio threshold.

Prior art
- US 2015/0120567 A1 (2015-04-30): RocksDB tiered compaction; discloses 1[a], 1[b], 1[c], 1[d].
- Dayan and Idreos, Dostoevsky, SIGMOD 2018, https://doi.org/10.1145/3183713.3196927:
  lazy leveling; discloses 1[c], 1[d].

Eligibility
Claim 1 is a process (Step 1). Under Step 2A prong one it recites no judicial exception;
comparing run sizes is part of a storage process, not a mental process. Eligible.

Novelty and obviousness
Claim 1 is anticipated by US 2015/0120567 A1, which discloses every element 1[a] to 1[d]
arranged as claimed. Claim 2 (threshold between 4 and 16) is obvious under KSR as routine
optimization of a result-effective variable. Claim 3 survives: no reference found that merges at
most two sorted runs at a time.

Definiteness and support
Claim 4 recites "a compaction scheduler configured to merge", functional language at the point
of novelty; FIG. 1 names the structure so 112(f) is not invoked. The description does not use
"holds" (claim 4 says "a drive that holds"); it says "held on a drive".

Drafting defects
Claim 3 is a multiple dependent claim and carries the 37 CFR 1.16(j) fee. Paragraph 3 says "the
disclosure" correctly throughout.

Recommendations
1. Amend claim 1 to recite "wherein the compaction merges at most two sorted runs at a time" and
cancel claim 3.
2. Cite US 2015/0120567 A1 and Dostoevsky in the IDS.

Scores
Claim clarity: 7/10. The claims are clear.
Specification support: 6/10. The description is thin on the scheduler.
Novelty over the art found: 3/10. Claim 1 is anticipated.
Non-obviousness: 4/10. Claim 3 only.
Eligibility: 9/10. A process on storage hardware.
Claim scope and fallbacks: 5/10. Few dependents.
Drafting quality: 7/10. Mostly clean.
Allowance likelihood: 3/10 (as filed, before amendment)
Confidence: 4/5
"""

_SLOPPY_REVIEW = """\
Scores
Claim clarity: 11/10.
Allowance likelihood: 30%
Confidence: high

Summary
Anticipated.

Prior art
- A well-known compaction paper discloses everything.
- Some blog post.

Eligibility
The claims are directed to an abstract idea \u2014 a mental process.

Novelty and obviousness
It is well known that everything is obvious. It is well known that everything is obvious.

Recommendations
Rewrite everything.
"""


def test_check_patent_review_passes_the_good_review(tmp_path: Path) -> None:
    """A nine-part review with cited art, claim citations and all scores gets zero FAIL lines."""
    path = tmp_path / "review.txt"
    path.write_text(_GOOD_REVIEW, encoding="utf-8")
    report = review_patent_sea.check_patent_review(str(path))
    assert "FAIL" not in report, report
    assert report.endswith("summary: 0 gate(s) failed")
    assert report.startswith(f"gates for {path}\nwords: ")
    assert gate(report, "headings missing") == "headings missing: 0 (limit 0) PASS"
    assert gate(report, "summary sentences") == "summary sentences: 4 (3 to 5) PASS"
    assert gate(report, "prior-art bullets") == "prior-art bullets: 2 (at least 1) PASS"
    assert (
        gate(report, "eligibility claim citations")
        == "eligibility claim citations: 1 (at least 1) PASS"
    )
    assert gate(report, "novelty and obviousness claim citations").startswith(
        "novelty and obviousness claim citations: 3 "
    )
    assert (
        gate(report, "recommendations numbered") == "recommendations numbered: 2 (at least 1) PASS"
    )
    assert gate(report, "dimensions unscored").startswith("dimensions unscored: 0 (limit 0) PASS")
    assert gate(report, "non-ASCII lines").startswith("non-ASCII lines: 0 (list only) PASS")
    assert review_patent_sea.check_patent_review(str(path), word_limit=100).count("FAIL") == 1
    # A bullet wrapped over two lines is one bullet; its identifier may sit on the second line.
    wrapped = _GOOD_REVIEW.replace(
        "- US 2015/0120567 A1 (2015-04-30)",
        "- A storage patent:\n  US 2015/0120567 A1 (2015-04-30)",
    )
    path.write_text(wrapped, encoding="utf-8")
    report = review_patent_sea.check_patent_review(str(path))
    assert gate(report, "prior-art bullets") == "prior-art bullets: 2 (at least 1) PASS"
    assert gate(report, "prior-art bullets without a number, URL or year").startswith(
        "prior-art bullets without a number, URL or year: 0 "
    )


def test_check_patent_review_flags_the_sloppy_review(tmp_path: Path) -> None:
    """Missing and misordered parts, bare art bullets, no claim citations, bad scores, slop."""
    path = tmp_path / "review.txt"
    path.write_text(_SLOPPY_REVIEW, encoding="utf-8")
    report = review_patent_sea.check_patent_review(str(path))
    assert gate(report, "headings missing").startswith("headings missing: 3 (limit 0) FAIL")
    assert "    Claims at issue" in report and "    Definiteness and support" in report
    assert "    Drafting defects" in report
    assert "headings out of order: Summary L6, Prior art L9, Eligibility L13, " in report
    assert gate(report, "summary sentences") == "summary sentences: 1 (3 to 5) FAIL"
    assert gate(report, "prior-art bullets") == "prior-art bullets: 2 (at least 1) PASS"
    assert gate(report, "prior-art bullets without a number, URL or year").startswith(
        "prior-art bullets without a number, URL or year: 2 (limit 0) FAIL"
    )
    assert "    L10: - A well-known compaction paper discloses everything." in report
    assert (
        gate(report, "eligibility claim citations")
        == "eligibility claim citations: 0 (at least 1) FAIL"
    )
    assert gate(report, "novelty and obviousness claim citations").endswith("0 (at least 1) FAIL")
    assert (
        gate(report, "recommendations numbered") == "recommendations numbered: 0 (at least 1) FAIL"
    )
    assert gate(report, "dimensions unscored").startswith("dimensions unscored: 9 (limit 0) FAIL")
    assert "    Claim clarity" in report and "    Allowance likelihood (N/10)" in report
    assert "    Confidence (N/5)" in report
    assert gate(report, "em dashes").startswith("em dashes: 1 (limit 0) FAIL")
    assert gate(report, "duplicated sentences").startswith("duplicated sentences: 1 (limit 0) FAIL")
    assert gate(report, "non-ASCII lines").startswith("non-ASCII lines: 1 (list only) CHECK")
    assert gate(report, "vague attribution").endswith(
        "; name the reference (number, title, page or column) in the same sentence"
    )
    assert "\\cite" not in report
    assert review_patent_sea.check_patent_review(str(tmp_path / "gone.txt")) == (
        f"no such file: {tmp_path / 'gone.txt'}"
    )
    # Parts that are absent get no part-specific gate, only the missing-headings line.
    path.write_text("Eligibility\nClaim 1 is a process.\n", encoding="utf-8")
    report = review_patent_sea.check_patent_review(str(path))
    assert gate(report, "headings missing").startswith("headings missing: 8 (limit 0) FAIL")
    assert gate(report, "eligibility claim citations").endswith("1 (at least 1) PASS")
    for name in (
        "summary sentences",
        "prior-art bullets",
        "recommendations numbered",
        "dimensions unscored",
    ):
        assert f"\n{name}: " not in report, name
    assert report.endswith("summary: 1 gate(s) failed")


@pytest.mark.skipif(
    not (shutil.which("pdflatex") and shutil.which("pdftotext")),
    reason="pdflatex and poppler are needed to build and read the PDF",
)
def test_read_patent_returns_pages_of_a_built_pdf_and_a_draft(tmp_path: Path) -> None:
    """A draft is one page as is; the built PDF comes back as ``=== page N ===`` blocks."""
    path = _write(tmp_path, _CLEAN_DRAFT, figures=2)
    text = review_patent_sea.read_patent(str(path))
    assert text.startswith(f"patent: {path} (1 page)\n=== page 1 ===\n# Log-Structured")
    assert "paper:" not in text
    write_patent_sea.build_patent(str(path))
    pdf = tmp_path / "draft.pdf"
    pages = review_patent_sea.read_patent(str(pdf), 3, 3)
    assert pages.startswith(f"patent: {pdf} (4 pages)\n=== page 3 ===\n")
    assert "What is claimed is:" in pages and "=== page 4 ===" not in pages
    assert "ABSTRACT" in review_patent_sea.read_patent(str(pdf), 4)
    assert review_patent_sea.read_patent(str(tmp_path / "gone.pdf")).startswith(
        "error: no such file"
    )


def test_sea_description_prompt_tools_and_settings() -> None:
    """The SEA appends its rules, exposes the three tools, browses and routes through bestrouter."""
    sea = review_patent_sea.ReviewPatentSea()
    assert sea.description().startswith("Reviews a patent application draft or a published patent")
    prompt = sea.system_prompt("DEFAULT")
    assert prompt.startswith("DEFAULT\n\n# Patent-reviewing agent")
    # The second-opinion pass names the model the SEA pins, not a literal.
    assert review_patent_sea.SECOND_OPINION_MODEL in prompt
    assert "set_model" in prompt
    assert [tool.__name__ for tool in sea.tools([])] == [
        "read_patent",
        "check_patent",
        "check_patent_review",
    ]
    settings = sea.settings({"max_budget": 1})
    assert settings == {
        "max_budget": 1,
        "use_web_tools": True,
        "auto_classify": False,
        "timeout": review_patent_sea.DISPATCH_TIMEOUT_SECONDS,
        "model": "bestrouter",
    }


def test_agent_gets_the_rules_and_the_tools_and_the_real_gate_report(tmp_path: Path) -> None:
    """With the SEA's configuration the model sees the three tools and the real review gates."""
    path = tmp_path / "review.txt"
    path.write_text(_SLOPPY_REVIEW, encoding="utf-8")
    run = sea_commands.evaluate_sea(
        [review_patent_sea.ReviewPatentSea()], f"Check the review at {path}; do not edit."
    )
    script = [
        tool_call_body("check_patent_review", {"review_path": str(path)}, prompt_tokens=500),
        finish_body("<pre>summary: 11 gate(s) failed</pre>", prompt_tokens=600),
    ]
    parsed, agentic = run_sea(run, script, tmp_path, max_steps=4)
    assert parsed["success"] is True
    assert len(agentic) == 2
    for request in agentic:
        names = tool_names(request)
        assert {
            "read_patent",
            "check_patent",
            "check_patent_review",
            "go_to_url",
            "finish",
        } <= names
        system = system_message(request)
        assert "# Patent-reviewing agent" in system and not system.startswith("# Patent")
    results = tool_results(agentic[1])
    assert len(results) == 1, agentic[1]["messages"]
    assert "dimensions unscored: 9 (limit 0) FAIL" in results[0]
    assert "summary: " in results[0] and "gate(s) failed" in results[0]
