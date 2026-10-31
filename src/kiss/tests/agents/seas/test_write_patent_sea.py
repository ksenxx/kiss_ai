# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/write_patent`` agent.

Module under test: :mod:`kiss.agents.seas.write_patent.write_patent_sea`.  The gate
tests run ``check_patent`` on real Markdown drafts; the build tests run pdflatex when it
is installed and skip otherwise; the agent test runs the SEA against a scripted model
server (``conftest.run_sea``) so the rules, the tools and the real gate report are what
the agent sees.
"""

import zipfile
from pathlib import Path

import pytest
from PIL import Image

from kiss.agents.seas.write_patent import write_patent_sea
from kiss.tests.agents.seas.conftest import gate
from kiss.tests.agents.seas.sea_contract import assert_appending_sea_run, assert_no_removed_getters
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body

_CLEAN_DRAFT = """\
# Log-Structured Key-Value Store With Tiered Compaction

## TECHNICAL FIELD

This description relates to storage engines, and more particularly to key-value stores
that keep records in sorted runs on a drive.

## BACKGROUND

A log-structured store appends incoming records to a log and later merges sorted runs.
Merging every run into one level amplifies writes by the number of levels.

## SUMMARY

In some embodiments, a storage node 102 receives a write request 104 that includes a
record 106, appends the record 106 to a write-ahead log 108, and stores the record 106 in
a first sorted run 110 of a plurality of sorted runs 110, 112 held on a drive 114. A
compaction scheduler 116 merges at most two sorted runs 110, 112 at a time when a size
ratio between them is below a threshold.

## BRIEF DESCRIPTION OF THE DRAWINGS

FIG. 1 is a block diagram of a storage node according to some embodiments.

FIG. 2 is a flow chart of a compaction method according to some embodiments.

## DETAILED DESCRIPTION

Referring to FIG. 1, a storage node 102 includes a drive 114 and a compaction scheduler
116. In some embodiments, the storage node 102 receives a write request 104 over a
network and appends the record 106 of the write request 104 to a write-ahead log 108
before it stores the record 106 in a first sorted run 110. The compaction scheduler 116
is configured to compare the size of the first sorted run 110 with the size of a second
sorted run 112 and to merge the two sorted runs 110, 112 when the size ratio is below a
threshold, for example a threshold between 4 and 16.

![FIG. 1](figures/fig1.png)

Referring to FIG. 2, a method 200 includes receiving 202 the write request 104, appending
204 the record 106 to the write-ahead log 108, storing 206 the record 106 in the first
sorted run 110, and scheduling 208 a compaction that merges at most two sorted runs at a
time. In other embodiments the storage node 102 is a virtual machine; the drive 114 may be
a solid-state drive or a network block device.

![FIG. 2](figures/fig2.png)

## CLAIMS

What is claimed is:

1. A method comprising:
    receiving, at a storage node, a write request that includes a record;
    appending the record to a write-ahead log;
    storing the record in a first sorted run of a plurality of sorted runs held on a drive; and
    scheduling a compaction that merges the first sorted run with a second sorted run of the
    plurality of sorted runs when a size ratio between the first sorted run and the second
    sorted run is below a threshold.

2. The method of claim 1, wherein the threshold is between 4 and 16.

3. The method of claim 1 or 2, wherein the compaction merges at most two sorted runs at a time.

4. A storage node comprising:
    a drive that holds a plurality of sorted runs; and
    a compaction scheduler configured to merge a first sorted run of the plurality of sorted
    runs with a second sorted run of the plurality of sorted runs when a size ratio between
    the first sorted run and the second sorted run is below a threshold.

## ABSTRACT

A storage node receives a write request that includes a record, appends the record to a
write-ahead log and stores the record in a first sorted run held on a drive. A compaction
scheduler merges the first sorted run with a second sorted run when a size ratio between
the two runs is below a threshold, so that at most two runs are merged at a time.
"""

_SLOPPY_DRAFT = (
    """\
# An Improved Widget

## SUMMARY

The present invention provides a widget 102 — it is well known that widgets are conventional.
It is essential that the widget 102 must always spin.
The widget 102 must always spin. The widget 102 must always spin.

## BACKGROUND

Figure 1 shows a prior art widget. The widget 102 must always spin.

BACKGROUND

Repeated heading.

## DETAILED DESCRIPTION

FIG. 2 shows a frame 104 of the widget 102. The gear 102 engages the frame 104, as shown
in Fig. 3. The spindle is hollow steel.

![FIG. 2](figures/fig2.png)

## ABSTRACT

Said widget comprises means for spinning. """
    + "The widget spins. " * 75
    + """

A second paragraph of the abstract.

## CLAIMS

1. A widget comprising a spindle substantially aligned with the gadget. The spindle spins
3. The widget of claim 1, wherein said spindle includes means for spinning.
4. The widget of claims 1 and 3, wherein the spindle is made of steel.
5. The widget of claim 4 or 3, wherein the spindle is hollow.
6. The widget of claim 7, wherein the spindle is red.
7. The widget of claim 1.
8. A frame.
9. A gear with teeth.
10. A base having a foot.
"""
)


def _write(tmp_path: Path, text: str, figures: int = 0) -> Path:
    """Write *text* as ``draft.md`` under *tmp_path* with *figures* 1-bit PNG figures."""
    path = tmp_path / "draft.md"
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    (tmp_path / "figures").mkdir(exist_ok=True)
    for n in range(1, figures + 1):
        Image.new("1", (200, 100), 1).save(tmp_path / "figures" / f"fig{n}.png")
    return path


def test_check_patent_passes_the_clean_draft(tmp_path: Path) -> None:
    """A draft laid out per 37 CFR 1.77 with sound claims gets zero FAIL lines."""
    path = _write(tmp_path, _CLEAN_DRAFT, figures=2)
    report = write_patent_sea.check_patent(str(path))
    assert "FAIL" not in report, report
    assert report.endswith("summary: 0 gate(s) failed")
    assert gate(report, "sections missing") == "sections missing: 0 (limit 0) PASS; 37 CFR 1.77(b)"
    assert gate(report, "antecedent basis").startswith("antecedent basis: 0 (limit 0) PASS")
    assert gate(report, "figure cross-references").startswith("figure cross-references: 0 ")
    assert gate(report, "claim numbering").startswith("claim numbering: 0 ")
    assert gate(report, "excess-claim fees").startswith("excess-claim fees: 1 (list only) CHECK")
    assert "    multiple dependent claims [3]: 37 CFR 1.16(j) fee" in report
    assert "    'hold' (claims [4])" in report
    assert gate(report, "words") == "words: 334 description, 67 abstract; 4 claims"
    # The EP rules expect two-part claims and only list multiple dependencies.
    ep = write_patent_sea.check_patent(str(path), "EP")
    assert gate(ep, "independent claims not in two-part form").startswith(
        "independent claims not in two-part form: 2 (list only) CHECK; EPC Rule 43(1)"
    )
    assert gate(ep, "excess-claim fees") == (
        "excess-claim fees: 0 (list only) PASS; 37 CFR 1.16(h)-(j) or EPO Rule 45"
    )
    assert write_patent_sea.check_patent(str(tmp_path / "gone.md")) == (
        f"no such file: {tmp_path / 'gone.md'}"
    )


def test_check_patent_flags_every_defect_of_the_sloppy_draft(tmp_path: Path) -> None:
    """Every section, claim, figure and prose gate fires on the sloppy draft."""
    path = _write(tmp_path, _SLOPPY_DRAFT)
    report = write_patent_sea.check_patent(str(path))
    assert report.endswith("summary: 20 gate(s) failed")
    assert "    BRIEF DESCRIPTION OF THE DRAWINGS (figures are mentioned)" in report
    assert gate(report, "sections out of order").startswith(
        "sections out of order: 1 (limit 0) FAIL"
    )
    assert (
        "    SUMMARY L3, BACKGROUND L9, BACKGROUND L13, DETAILED DESCRIPTION L17, ABSTRACT L24, "
        "CLAIMS L30" in report
    )
    assert gate(report, "sections repeated") == "sections repeated: 1 (limit 0) FAIL"
    assert gate(report, "title wording") == "title wording: 1 (limit 0) FAIL; MPEP 606"
    assert gate(report, "abstract words over limit") == (
        "abstract words over limit: 1 (limit 0) FAIL; 237 words (limit 150, 37 CFR 1.72(b))"
    )
    assert "    2 paragraphs" in report
    assert gate(report, "abstract wording").startswith("abstract wording: 2 (limit 0) FAIL")
    assert "    no 'What is claimed is:' line" in report
    assert (
        "    claims numbered [1, 3, 4, 5, 6, 7, 8, 9, 10], expected [1, 2, 3, 4, 5, 6, 7, 8, 9]"
        in report
    )
    assert "    claim 1 (L32): 2 sentences" in report
    assert "    claim 1 (L32): no terminal period" in report
    assert "    claim 6 depends from claim 7" in report
    assert "    claim 4: refer to the claims in the alternative ('claim 1 or 2')" in report
    assert "    claim 5 is a multiple dependent claim depending from one" in report
    assert "    4 independent claims (3 in the basic fee)" in report
    assert "    multiple dependent claims [4, 5]: 37 CFR 1.16(j) fee" in report
    assert gate(report, "antecedent basis").startswith("antecedent basis: 3 (limit 0) FAIL")
    assert "    claim 1 (L32): 'the gadget' has no antecedent" in report
    assert "    claim 6 (L36): 'The widget' has no antecedent" in report
    assert "'said spindle'" not in report  # claim 1 introduced "a spindle substantially aligned"
    assert "    claim 1: substantially" in report
    assert "    claim 3: means for" in report
    assert "    claim 3: said" in report
    assert gate(report, "independent claims without a transition").startswith(
        "independent claims without a transition: 2 (list only) CHECK"
    )
    assert "    claim 7" in report.split("dependent claims without a further limitation")[1]
    assert (
        "    'gadget' (claims [1])" in report
        and "'mean'" not in report
        and "'include'" not in report
    )
    assert "    FIG. 3 in DETAILED DESCRIPTION but not in BRIEF DESCRIPTION" in report
    assert "    figures numbered [2, 3], expected 1..2" in report
    assert "    FIG. 3 is described but has no image" in report
    assert gate(report, "figure wording") == "figure wording: 1 (limit 0) FAIL; write 'FIG. N'"
    assert "    102: gear, widget" in report
    assert gate(report, "invention talk").startswith("invention talk: 1 (limit 0) FAIL")
    assert gate(report, "prior-art admissions").startswith("prior-art admissions: 2 (limit 0) FAIL")
    assert gate(report, "absolute terms").startswith("absolute terms: 7 (list only) CHECK")
    assert gate(report, "em dashes").startswith("em dashes: 1 (limit 0) FAIL")
    assert "    L7, L7, L11: the widget 102 must always spin." in report
    assert gate(report, "words") == "words: 67 description, 237 abstract; 9 claims"
    # Under the EP rules the multiple-dependency gates only list and two-part form is expected.
    ep = write_patent_sea.check_patent(str(path), "EP")
    assert gate(ep, "multiple dependent claims not in the alternative").endswith(
        "CHECK; 37 CFR 1.75(c)"
    )
    assert gate(ep, "multiple dependent claim depending from another").endswith(
        "CHECK; 37 CFR 1.75(c)"
    )
    assert gate(ep, "independent claims not in two-part form").startswith(
        "independent claims not in two-part form: 4 (list only) CHECK"
    )
    assert "4 independent claims" not in ep


def test_check_patent_title_abstract_and_claim_count_edge_cases(tmp_path: Path) -> None:
    """A missing title, an over-long title, 21 claims and a claim set with no independent claim."""
    claims = ["1. A method comprising a step of mixing."] + [
        f"{n}. The method of claim 1, wherein the step further comprises stirring {n} times."
        for n in range(2, 22)
    ]
    path = _write(
        tmp_path,
        "## DETAILED DESCRIPTION\n\nA method mixes and stirs.\n\n## CLAIMS\n\n"
        "What is claimed is:\n\n" + "\n".join(claims) + "\n\n## ABSTRACT\n\nA method mixes.\n",
    )
    report = write_patent_sea.check_patent(str(path))
    assert "    TITLE (a '# <title>' line before the first section)" in report
    assert "    21 claims counted for fees (20 in the basic fee)" in report
    assert (
        gate(report, "title characters")
        == "title characters: 0 (limit 0) PASS; 0 characters (limit 500, 37 CFR 1.72(a))"
    )
    long_title = "# " + "Word " * 120
    path = _write(
        tmp_path,
        long_title + "\n\n## CLAIMS\n\n1. The method of claim 2.\n2. The method of claim 1.\n",
    )
    report = write_patent_sea.check_patent(str(path))
    assert gate(report, "title characters").startswith(
        "title characters: 1 (limit 0) FAIL; 599 characters"
    )
    assert "    no independent claim" in report
    assert "    claim 1 depends from claim 2" in report
    assert gate(report, "sections missing").startswith("sections missing: 4 (limit 0) FAIL")
    path = _write(tmp_path, "# Title\n\n## CLAIMS\n\nWhat is claimed is:\n")
    assert "    no claims" in write_patent_sea.check_patent(str(path))
    assert write_patent_sea.check_patent(str(path), "EP").count("two-part form: 0 ") == 1


def test_draft_parsing_sections_claims_and_dependencies() -> None:
    """Heading forms, body text, claim continuation lines and every dependency wording parse."""
    section_name = write_patent_sea.section_name
    assert section_name("**Detailed Description:**") == "DETAILED DESCRIPTION"
    assert section_name("WHAT IS CLAIMED IS:") == "CLAIMS"
    assert section_name("## Notes") is None
    assert section_name("Background") is None
    assert section_name("") is None
    draft = write_patent_sea.Draft(
        "## BACKGROUND\n# not a title\n\n## CLAIMS\n1. A method comprising:\n"
        "    a step; and\n\n    another step.\n2. The method of claim 1, wherein the step runs.\n"
    )
    assert draft.title == ""
    assert draft.text("BACKGROUND") == "# not a title\n"
    assert [(n, line) for n, line, _ in draft.claims] == [(1, 5), (2, 9)]
    assert draft.claims[0][2] == "A method comprising:\na step; and\nanother step."
    assert draft.figures == []
    parents = write_patent_sea.claim_parents
    assert parents("The widget of claims 1 and 3, wherein") == ([1, 3], False)
    assert parents("The widget of claim 4 or 3, wherein") == ([4, 3], True)
    assert parents("The method of any one of claims 1 to 3,") == ([1, 2, 3], True)
    assert parents("The method of claims 1-3,") == ([1, 2, 3], False)
    assert parents("The method of claim 1, 2 or 3,") == ([1, 2, 3], True)
    assert parents("A frame.") == ([], True)
    assert write_patent_sea._tex("A & B % $5 #1 x_y {z} ~ ^ \\ 50%") == (
        "A \\& B \\% \\$5 \\#1 x\\_y \\{z\\} \\textasciitilde{} \\textasciicircum{} "
        "\\textbackslash{} 50\\%"
    )
    singular = write_patent_sea._singular
    assert [singular(w) for w in ("entries", "batches", "nodes", "bus", "class", "ies")] == [
        "entry",
        "batch",
        "node",
        "bus",
        "class",
        "ies",
    ]


def test_antecedent_problems_resolve_quantifiers_modifiers_and_skips() -> None:
    """Introductions with quantifiers and modifiers resolve later definite phrases."""
    problems, known = write_patent_sea.antecedent_problems(
        "A method comprising: receiving a plurality of nodes and at least one first sorted run; "
        "merging the nodes with the run; and sending the same run to the gadget.",
        set(),
    )
    assert problems == ["the gadget"]
    assert {"node", "run", "sorted run", "first sorted run", "gadget"} <= known
    problems, _ = write_patent_sea.antecedent_problems(
        "The method of claim 1, wherein the spindle spins and the following step repeats.",
        {"method", "spindle"},
    )
    assert problems == []


@pytest.mark.usefixtures("tex_bin")
def test_build_patent_writes_docx_and_pdfs(tmp_path: Path) -> None:
    """The clean draft builds a numbered DOCX, a specification PDF and a drawings PDF."""
    path = _write(tmp_path, _CLEAN_DRAFT, figures=2)
    report = write_patent_sea.build_patent(str(path))
    assert report.splitlines() == [
        f"docx: {tmp_path / 'draft.docx'}",
        "draft.tex errors: 0 (limit 0) PASS",
        f"pdf: {tmp_path / 'draft.pdf'} (4 pages)",
        "draft_drawings.tex errors: 0 (limit 0) PASS",
        f"pdf: {tmp_path / 'draft_drawings.pdf'} (2 pages)",
        "paragraphs numbered: 7; claims: 4; abstract words: 67",
    ]
    with zipfile.ZipFile(tmp_path / "draft.docx") as archive:
        document = archive.read("word/document.xml").decode()
    assert "[0001]" in document and "[0007]" in document and "[0008]" not in document
    assert "What is claimed is:" in document and "ABSTRACT" in document
    tex = (tmp_path / "draft.tex").read_text(encoding="utf-8")
    assert "\\begin{center}CLAIMS\\end{center}" in tex and "1. A method comprising:" in tex
    assert "\\textbf{[0001]}" in tex and "\\textbf{[0008]}" not in tex
    drawings = (tmp_path / "draft_drawings.tex").read_text(encoding="utf-8")
    assert "figures/fig2.png" in drawings and "FIG. 2" in drawings
    # A draft without figures builds no drawings PDF; a missing image file fails that build.
    path = _write(
        tmp_path / "plain",
        "# T\n\n## CLAIMS\n\nWhat is claimed is:\n\n1. A method comprising a step.\n\n"
        "## ABSTRACT\n\nA method.\n",
    )
    report = write_patent_sea.build_patent(str(path))
    assert "drawings: none (no ![FIG. N](path) images in the draft)" in report
    assert report.endswith("paragraphs numbered: 0; claims: 1; abstract words: 2")
    path = _write(
        tmp_path / "broken",
        "# T\n\n## DETAILED DESCRIPTION\n\nFIG. 1 shows a step 102.\n\n![](figures/missing.png)\n\n"
        "## CLAIMS\n\n1. A method comprising a step.\n",
    )
    report = write_patent_sea.build_patent(str(path))
    assert "draft_drawings.tex errors: 1 (limit 0) FAIL" in report
    assert f"pdf: {tmp_path / 'broken' / 'draft_drawings.pdf'} not written FAIL" in report
    assert "FIG. 1" in (tmp_path / "broken" / "draft_drawings.tex").read_text(encoding="utf-8")


_TABLE_DRAFT = (
    "# T\n\n## DETAILED DESCRIPTION\n\nTable 1 lists the weights & thresholds.\n\n"
    "| Modality type | Event type of the hypothesis | w | v |\n|---|---|---|---|\n"
    "| **RGB camera** | fire | 0.40 | 0.40 |\n| Thermal camera | fire | 0.90 |\n\n"
    "| a | b |\n|:--|--:|\n| 1 | 2 |\n\nA paragraph after the tables.\n\n"
    "| not a table\nbecause this line has no pipe |\n\n"
    "## CLAIMS\n\nWhat is claimed is:\n\n1. A method comprising a step.\n\n"
    "## ABSTRACT\n\nA method.\n"
)


def test_build_patent_renders_markdown_tables(tmp_path: Path) -> None:
    """Pipe tables become bordered DOCX tables and LaTeX tabularx; prose is still numbered."""
    path = _write(tmp_path, _TABLE_DRAFT)
    kinds = [kind for kind, _ in write_patent_sea.blocks(write_patent_sea.Draft(_TABLE_DRAFT))]
    assert kinds == ["title", "heading", "para", "table", "table", "para", "para", "heading"] + [
        "free",
        "claim",
        "heading",
        "free",
    ]
    rows = write_patent_sea.table_rows(
        "| Modality type | Event type of the hypothesis | w | v |\n|---|---|---|---|\n"
        "| **RGB camera** | fire | 0.40 | 0.40 |\n| Thermal camera | fire | 0.90 |"
    )
    assert rows == [
        ["Modality type", "Event type of the hypothesis", "w", "v"],
        ["RGB camera", "fire", "0.40", "0.40"],
        ["Thermal camera", "fire", "0.90", ""],
    ]
    assert write_patent_sea.table_rows("| one line only |") == []
    assert write_patent_sea.table_rows("not a | table\nno separator | here") == []
    assert write_patent_sea.table_rows("A | B\n--- | ---\n1 | 2") == [["A", "B"], ["1", "2"]]
    assert write_patent_sea.table_rows("| A | B |\n|---|---|\n| left\\|right | end |") == [
        ["A", "B"],
        ["left|right", "end"],
    ]
    assert write_patent_sea.table_rows("| A | B | C |\n|:--|:-:|--:|\n||middle||") == [
        ["A", "B", "C"],
        ["", "middle", ""],
    ]
    assert write_patent_sea.table_rows("| A | B |\n|---|---|\n| - | - |\n| 1 |") == [
        ["A", "B"],
        ["-", "-"],
        ["1", ""],
    ]
    report = write_patent_sea.build_patent(str(path), tex_bin=str(tmp_path))
    assert report.endswith("paragraphs numbered: 3; claims: 1; abstract words: 2")
    with zipfile.ZipFile(tmp_path / "draft.docx") as archive:
        document = archive.read("word/document.xml").decode()
    assert document.count("<w:tbl>") == 2 and document.count("<w:tr>") == 5
    assert "<w:b/>" in document and "RGB camera" in document
    assert "weights &amp; thresholds" in document
    tex = (tmp_path / "draft.tex").read_text(encoding="utf-8")
    assert "\\usepackage{tabularx}" in tex and tex.count("\\begin{tabularx}") == 2
    assert "\\begin{tabularx}{\\textwidth}{|X|X|l|l|}" in tex  # long cells wrap, short ones do not
    assert "\\begin{tabularx}{\\textwidth}{|X|l|}" in tex  # at least one column wraps
    assert "\\textbf{Modality type} & \\textbf{Event type of the hypothesis}" in tex
    assert "Thermal camera & fire & 0.90 &  \\\\" in tex


@pytest.mark.usefixtures("tex_bin")
def test_build_patent_compiles_markdown_tables(tmp_path: Path) -> None:
    """The tabularx tables of a Markdown draft compile with no LaTeX error."""
    path = _write(tmp_path, _TABLE_DRAFT)
    assert "draft.tex errors: 0 (limit 0) PASS" in write_patent_sea.build_patent(str(path))


def test_build_patent_without_pdflatex_and_missing_draft(tmp_path: Path) -> None:
    """Without pdflatex the DOCX and LaTeX sources are still written; a missing draft errors."""
    path = _write(tmp_path, _CLEAN_DRAFT, figures=2)
    report = write_patent_sea.build_patent(str(path), tex_bin=str(tmp_path))
    assert (
        f"draft.pdf: pdflatex not found ({tmp_path / 'pdflatex'}), "
        f"LaTeX source written to {tmp_path / 'draft.tex'}" in report
    )
    assert (tmp_path / "draft.docx").is_file() and (tmp_path / "draft_drawings.tex").is_file()
    assert not (tmp_path / "draft.pdf").exists()
    assert (
        write_patent_sea.build_patent(str(tmp_path / "gone.md"))
        == f"no such file: {tmp_path / 'gone.md'}"
    )
    # A title-only draft and an empty CLAIMS section still build (the gates report them).
    path = _write(tmp_path / "title", "# Only a title\n")
    report = write_patent_sea.build_patent(str(path), tex_bin=str(tmp_path))
    assert report.endswith("paragraphs numbered: 0; claims: 0; abstract words: 0")
    path = _write(tmp_path / "empty", "# T\n\n## CLAIMS\n\nWhat is claimed is:\n")
    write_patent_sea.build_patent(str(path), tex_bin=str(tmp_path))
    tex = (tmp_path / "empty" / "draft.tex").read_text(encoding="utf-8")
    assert "What is claimed is:" in tex and "CLAIMS" in tex


def test_sea_description_prompt_tools_and_settings() -> None:
    """The SEA appends its rules, exposes both tools, browses and skips the classifier."""
    sea = write_patent_sea.WritePatentSea()
    assert sea.description().startswith("Writes or revises a utility patent application")
    prompt = sea.system_prompt("DEFAULT")
    assert prompt.startswith("DEFAULT\n\n# Patent-drafting agent")
    for rule in (
        "37 CFR 1.77",
        "150 words",
        "Antecedent basis",
        "patents.google.com",
        "Alice",
        "candor",
    ):
        assert rule in prompt, rule
    assert [tool.__name__ for tool in sea.tools([])] == ["check_patent", "build_patent"]
    settings = sea.settings({"model": "x"})
    assert settings["use_web_tools"] is True and settings["auto_classify"] is False
    assert (
        settings["timeout"] == write_patent_sea.DISPATCH_TIMEOUT_SECONDS
        and settings["model"] == "x"
    )
    assert_no_removed_getters(write_patent_sea)


def test_agent_gets_the_rules_and_the_tools_and_the_real_gate_report(tmp_path: Path) -> None:
    """With the SEA's configuration the model sees ``check_patent``/``build_patent`` and real gates.

    The scripted model calls ``check_patent`` on the sloppy draft and finishes with what the
    tool returned; the test checks the offered tools, that the default system prompt was
    kept with the rules appended, and that the real gate report flowed back to the model.
    """
    path = _write(tmp_path, _SLOPPY_DRAFT)
    results = assert_appending_sea_run(
        write_patent_sea.WritePatentSea(),
        f"Check {path} for US filing; do not edit.",
        [
            tool_call_body("check_patent", {"draft_path": str(path)}, prompt_tokens=500),
            finish_body("<pre>summary: 20 gate(s) failed</pre>", prompt_tokens=600),
        ],
        tmp_path,
        tools={"check_patent", "build_patent", "Bash", "run_agent", "go_to_url", "finish"},
        system_prompt=write_patent_sea.SYSTEM_PROMPT,
        anchors=("# Patent-drafting agent", "What is claimed is:"),
        summary="20 gate(s) failed",
    )
    assert len(results) == 1, results
    assert "summary: 20 gate(s) failed" in results[0]
    assert "antecedent basis: 3 (limit 0) FAIL" in results[0]


_CHAIN_CLAIMS = """\
What is claimed is:

1. An apparatus comprising a motor.
2. An apparatus comprising a sensor.
3. The apparatus of claim 1 or 2, wherein the motor comprises a shaft.
4. The apparatus of claim 3, wherein the shaft is steel.
5. The apparatus of claim 1 or 4, wherein the outer surface of the motor is smooth.
"""


def test_claim_gates_follow_chains_alternatives_and_fee_weights(tmp_path: Path) -> None:
    """Indirect multiple-dependent chains, per-alternative antecedents and 1.75(c) fee weights."""
    path = _write(
        tmp_path,
        "# T\n\n## DETAILED DESCRIPTION\n\nA motor, a sensor, a steel shaft and a "
        "smooth outer surface.\n\n## CLAIMS\n\n" + _CHAIN_CLAIMS + "\n## ABSTRACT\n\nA motor.\n",
    )
    report = write_patent_sea.check_patent(str(path))
    # Claim 5 depends from 4, which depends from the multiple dependent claim 3 (MPEP 608.01(n)).
    assert "    claim 5 is a multiple dependent claim depending from one" in report
    assert "claim 4 is a multiple" not in report
    # "the motor" has its antecedent in the claim-1 alternative only; "the outer surface of
    # the motor" is an inherent feature and needs no introduction (MPEP 2173.05(e)).
    assert "    claim 3 (L13): 'the motor' has no antecedent via claim 2" in report
    assert "'the motor' has no antecedent via claim 1" not in report
    assert "outer surface" not in report
    assert gate(report, "antecedent basis").startswith("antecedent basis: 1 (limit 0) FAIL")
    # Fees: claims 3 and 4 count twice each, claim 5 counts 1 + 2: 1+1+2+2+3 = 9 under 20.
    assert "claims counted for fees" not in report
    many = _CHAIN_CLAIMS + "".join(
        f"{n}. The apparatus of claim 1, wherein the motor has {n} poles.\n" for n in range(6, 18)
    )
    path = _write(tmp_path / "many", "# T\n\n## CLAIMS\n\n" + many)
    report = write_patent_sea.check_patent(str(path))
    # 17 numbered claims weigh 1+1+2+2+3 + 12 = 21 for the USPTO fee.
    assert "    21 claims counted for fees (20 in the basic fee)" in report
    # The EPO counts the claims as numbered (Rule 45): 17 claims, 15 in the basic fee.
    ep = write_patent_sea.check_patent(str(path), "EP")
    assert "    17 claims counted for fees (15 in the basic fee)" in ep


def test_figure_identifiers_empty_abstract_and_front_sections(tmp_path: Path) -> None:
    """Partial-view figures match by identifier, an empty abstract fails, front sections build."""
    draft = (
        "# T\n\n## CROSS-REFERENCE TO RELATED APPLICATIONS\n\nThis application claims priority to "
        "U.S. Provisional Application No. 63/123456.\n\n## STATEMENT REGARDING FEDERALLY SPONSORED "
        "RESEARCH\n\nNone.\n\n## BRIEF DESCRIPTION OF THE DRAWINGS\n\nFIG. 1A is a view. "
        "FIG. 1B is "
        "another view.\n\n## DETAILED DESCRIPTION\n\nFIGS. 1A and 1B show a motor 102.\n\n"
        "![FIG. 1A](figures/fig1.png)\n\n![FIG. 1B](figures/fig2.png)\n\n## CLAIMS\n\n"
        "What is claimed is:\n\n1. A motor.\n\n## ABSTRACT\n\n"
    )
    path = _write(tmp_path, draft, figures=2)
    report = write_patent_sea.check_patent(str(path))
    assert gate(report, "figure cross-references").startswith("figure cross-references: 0 ")
    assert (
        gate(report, "abstract paragraphs")
        == "abstract paragraphs: 1 (limit 0) FAIL; exactly one paragraph"
    )
    assert "    0 paragraphs" in report
    assert gate(report, "sections out of order").startswith("sections out of order: 0 ")
    parts = write_patent_sea.blocks(write_patent_sea.Draft(draft))
    assert ("heading", "CROSS-REFERENCE TO RELATED APPLICATIONS") in parts
    assert any("63/123456" in text for _, text in parts)
    # A wrongly captioned image and a range of figures.
    draft = draft.replace("![FIG. 1B](figures/fig2.png)", "![FIG. 9](figures/fig2.png)")
    report = write_patent_sea.check_patent(str(_write(tmp_path, draft)))
    assert "    FIG. 1B is described but has no image" in report
    assert "    image captioned FIG. 9 is not described" in report
    assert write_patent_sea._figure_ids("FIGS. 2-4 and FIG. 6A, Figures 7 to 8") == {
        "2",
        "3",
        "4",
        "6A",
        "7",
        "8",
    }
    # An image without a FIG. caption falls back to the count comparison.
    draft = draft.replace("![FIG. 9](figures/fig2.png)", "![](figures/fig2.png)").replace(
        "![FIG. 1A](figures/fig1.png)\n\n", ""
    )
    report = write_patent_sea.check_patent(str(_write(tmp_path, draft)))
    assert "    1 image(s) embedded for 2 figure identifier(s)" in report
