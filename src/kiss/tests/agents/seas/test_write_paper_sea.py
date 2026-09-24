# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/write_paper`` agent (:mod:`kiss.agents.seas.write_paper_sea`).

The gate tests run :func:`check_paper` on real ``.tex``/``.bib`` files
written to ``tmp_path``.  The build tests run :func:`build_paper`
against the TeX Live installed on the machine and are skipped when
there is none (``build_paper`` shells out to ``pdflatex``; there is no
substitute for it).  The agent-level test runs a real
:class:`ChatSorcarAgent` ReAct loop against the scripted local
chat-completions server configured as the daemon configures it from
the SEA's getters; the only replaced boundary is the LLM endpoint, and
the ``check_paper`` tool really runs on the fixture.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.seas import write_paper_sea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

_SEA_PATH = Path(write_paper_sea.__file__).resolve()

_ABSTRACT = " ".join(f"word{i}" for i in range(160))

_SLOPPY_TEX = (
    r"""
\documentclass{article}
% We delve into this comment --- comments are not prose.
\newcommand{\Speedup}{2.1}
\newcommand{\Runs}{12}
\usepackage{listings}
\begin{document}
\begin{abstract}
"""
    + _ABSTRACT
    + r""" The store is \Speedup{}x faster over \Runs{} runs and 3.7 GB.
\end{abstract}
\section{Introduction}
The store serves reads from an index of keys. We delve into the design rather than the history.
We use a hash rather than a tree, rather than a trie, rather than a set, rather than a heap.
It is a log, not a table --- it \emph{never} writes. The store serves reads from an index of keys.
We ran \Runs{} runs; see Figure~\ref{fig:missing} and Section~\ref{sec:intro}\label{sec:intro}.
This paper cites Bitcask~\cite{bitcask} and uses \emph{write-ahead log} for the first time.
\begin{lstlisting}
we leverage --- the seamless landscape *not prose* rather than rather than rather than
\end{lstlisting}
\begin{tabular}{ll}
delve --- & rather than \\
\end{tabular}
\bibliographystyle{plain}\bibliography{refs}
\end{document}
"""
)

_SLOPPY_BIB = """
@misc{bitcask, title={Bitcask}, author={Sheehy, Justin}, year={2010}}
@misc{unused, title={Never cited}, author={Nobody}, year={2000}}
"""

_CLEAN_TEX = (
    r"""
\documentclass{article}
\newcommand{\Speedup}{2.1}
\begin{document}
\begin{abstract}
"""
    + _ABSTRACT
    + r""" The store is \Speedup{}x faster.
\end{abstract}
\section{Design}\label{sec:design}
The store keeps one hash index in memory. It is \Speedup{}x faster than the baseline.
Section~\ref{sec:design} explains the index. We cite Bitcask~\cite{bitcask}.
\bibliographystyle{plain}\bibliography{refs}
\end{document}
"""
)

_CLEAN_BIB = "@misc{bitcask, title={Bitcask}, author={Sheehy, Justin}, year={2010}}\n"

_BIBLATEX_TEX = r"""
\documentclass{article}
\usepackage[backend=biber]{biblatex}
\addbibresource{refs.bib}
\begin{document}
We cite Bitcask~\cite{bitcask}.
\printbibliography
\end{document}
"""

_BROKEN_BUILD_TEX = r"""
\documentclass{article}
\begin{document}
\undefinedmacro
See Section~\ref{sec:missing}.
\texttt{aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}
\end{document}
"""


def _write(tmp_path: Path, tex: str, bib: str | None) -> Path:
    """Write ``paper.tex`` (and ``refs.bib`` when *bib* is given) into *tmp_path*."""
    path = tmp_path / "paper.tex"
    path.write_text(tex, encoding="utf-8")
    if bib is not None:
        (tmp_path / "refs.bib").write_text(bib, encoding="utf-8")
    return path


def _gate(report: str, name: str) -> str:
    """Return the header line of gate *name* in *report*."""
    return next(line for line in report.splitlines() if line.startswith(f"{name}: "))


def _line(tex: str, needle: str) -> int:
    """Return the 1-based line number of the first line of *tex* containing *needle*."""
    return next(i for i, line in enumerate(tex.splitlines(), 1) if needle in line)


def test_sea_getters_follow_the_user_contract() -> None:
    """The SEA appends the template's rules, offers the two tools, browses and fans out."""
    prompt = write_paper_sea.append_to_system_prompt()
    assert prompt == write_paper_sea.SYSTEM_PROMPT
    assert "William Strunk Jr. and E. B. White" in prompt
    assert write_paper_sea.REVIEWER_MODEL in prompt
    assert "`check_paper`" in prompt and "`build_paper`" in prompt
    assert "Em dashes: zero in prose" in prompt
    assert "Never add .aux" in prompt
    tools = write_paper_sea.tools()
    assert [t.__name__ for t in tools] == ["check_paper", "build_paper"]
    assert write_paper_sea.use_web_tools() is True
    assert write_paper_sea.is_parallel() is True
    assert write_paper_sea.classify_tasks() is False
    # The default system prompt is kept: the SEA only appends to it.
    assert not hasattr(write_paper_sea, "system_prompt")


def test_slash_write_paper_resolves_to_the_bundled_sea() -> None:
    """``/write_paper <instructions>`` is rewritten into a ``run_agent`` directive on this file."""
    assert sea_commands.get_command("write_paper") == _SEA_PATH
    rewritten = sea_commands.rewrite_prompt_if_command("/write_paper Write a paper on X")
    assert rewritten is not None
    prompt, path = rewritten
    assert path == _SEA_PATH
    assert f'agent = "{_SEA_PATH}"' in prompt
    assert prompt.endswith("TASK TEXT FOR run_agent:\nWrite a paper on X")


def test_check_paper_flags_the_sloppy_prose_and_skips_non_prose(tmp_path: Path) -> None:
    """Every gate fires on the prose hits and ignores comments, listings and table cells."""
    report = write_paper_sea.check_paper(str(_write(tmp_path, _SLOPPY_TEX, _SLOPPY_BIB)))
    # The em dash in the prose counts; the ones in the comment, the listing and
    # the table do not.
    assert _gate(report, "em dashes") == (
        "em dashes: 1 (limit 0) FAIL; use commas, parentheses, a colon or a new sentence"
    )
    assert f"    L{_line(_SLOPPY_TEX, '--- it')}: ...It is a log, not a table --- it" in report
    assert _gate(report, "antithesis").startswith("antithesis: 6 (limit 5) FAIL")
    assert _gate(report, "slop vocabulary") == "slop vocabulary: 1 (limit 0) FAIL"
    assert f"L{_line(_SLOPPY_TEX, 'We delve into the design')}: ..." in report
    assert "keys. We delve into the design rather than the history...." in report
    assert _gate(report, "draft and review talk").startswith(
        "draft and review talk: 0 (list only) PASS"
    )
    assert _gate(report, "markdown artifacts") == "markdown artifacts: 0 (limit 0) PASS"
    assert _gate(report, "abstract words") == "abstract words: 170 (150 to 250) PASS"
    # \Speedup and 3.7 never recur in the body; \Runs does.  Digits inside
    # words (``word10``) are not numbers.
    orphans = _gate(report, "abstract numbers or macros unused elsewhere in the body")
    assert orphans.endswith(": 2 (limit 0) FAIL")
    assert "    3.7\n" in report and "    \\Speedup\n" in report
    assert "    \\Runs\n" not in report and "    10\n" not in report
    assert _gate(report, "duplicated sentences") == "duplicated sentences: 1 (limit 0) FAIL"
    first, second = _line(_SLOPPY_TEX, "We delve into the design"), _line(_SLOPPY_TEX, "--- it")
    assert f"    L{first}, L{second}: the store serves reads from an index of keys." in report
    assert _gate(report, "single-word \\emph").startswith("single-word \\emph: 1 (list only) CHECK")
    assert _gate(report, "non-ASCII lines") == (
        "non-ASCII lines: 0 (list only) PASS; none unless intended"
    )
    assert _gate(report, "\\ref without \\label") == "\\ref without \\label: 1 (limit 0) FAIL"
    assert "    fig:missing\n" in report
    assert f"bib files: {tmp_path / 'refs.bib'}" in report
    assert _gate(report, "\\cite without bib entry") == "\\cite without bib entry: 0 (limit 0) PASS"
    assert _gate(report, "bib entries never cited") == "bib entries never cited: 1 (limit 0) FAIL"
    assert "    unused\n" in report
    assert report.endswith("summary: 7 gate(s) failed")


def test_check_paper_passes_clean_prose_and_reports_missing_inputs(tmp_path: Path) -> None:
    """A clean paper fails no gate; a missing bib, abstract or file is reported, not raised."""
    path = _write(tmp_path, _CLEAN_TEX, _CLEAN_BIB)
    report = write_paper_sea.check_paper(str(path))
    assert report.endswith("summary: 0 gate(s) failed"), report
    assert _gate(report, "antithesis") == "antithesis: 0 (limit 5) PASS; only technical contrasts"

    # The abstract's non-ASCII character is listed (CHECK), and the bib is found
    # through the explicit ``bib_path`` when the .tex does not name one.
    (tmp_path / "other.bib").write_text(_CLEAN_BIB, encoding="utf-8")
    tex = _CLEAN_TEX.replace(r"\bibliography{refs}", "").replace("one hash", "one \u201chash\u201d")
    path.write_text(tex, encoding="utf-8")
    report = write_paper_sea.check_paper(str(path))
    assert "bib files: none found (pass bib_path to check the citations) CHECK" in report
    assert _gate(report, "markdown artifacts") == "markdown artifacts: 2 (limit 0) FAIL"
    assert _gate(report, "non-ASCII lines").startswith("non-ASCII lines: 1 (list only) CHECK")
    report = write_paper_sea.check_paper(str(path), bib_path="other.bib")
    assert f"bib files: {tmp_path / 'other.bib'}" in report
    assert _gate(report, "\\cite without bib entry") == "\\cite without bib entry: 0 (limit 0) PASS"
    # An absolute bib path works too; a bib path that does not exist is not "found".
    report = write_paper_sea.check_paper(str(path), bib_path=str(tmp_path / "other.bib"))
    assert f"bib files: {tmp_path / 'other.bib'}" in report
    report = write_paper_sea.check_paper(str(path), bib_path=str(tmp_path / "gone.bib"))
    assert "bib files: none found" in report

    # No abstract, and no \begin{document} at all: reported, and the whole text is prose.
    path.write_text("Hi. We delve.", encoding="utf-8")
    report = write_paper_sea.check_paper(str(path))
    assert "abstract: not found FAIL" in report
    assert _gate(report, "slop vocabulary") == "slop vocabulary: 1 (limit 0) FAIL"
    missing = tmp_path / "nope.tex"
    assert write_paper_sea.check_paper(str(missing)) == f"no such file: {missing}"


def test_check_paper_catches_the_forms_the_rules_name(tmp_path: Path) -> None:
    """Regressions from review: forms the prompt bans but plain word lists missed."""
    tex = (
        "\\documentclass{article}\\newcommand{\\Ten}{10}\\begin{document}\n"
        "\\begin{abstract}" + " ".join(["lorem"] * 148) + " \\emph{ten} \\Ten{} words, 10 GB."
        "\\end{abstract}\n"
        "We optimize not speed but latency. We harness compiler feedback. To be clear, "
        "we do not claim causality, and we say so.\n"
        "Smith et al. show that agents fail under load. Jones et al. show that agents "
        "fail under load. We report the final result.\n"
        "The index holds 100 keys, i.e. \\Tenfold{} more. We report the final result.\n"
        "% \\cite{ghost} \\label{sec:ghost}\n"
        "\\begin{lstlisting}[caption={Agent}, label={lst:agent}]\nx\n\\end{lstlisting}\n"
        "See Listing~\\ref{lst:agent}; we cite~\\cite{real}.\n"
        "\\bibliography{refs}\\end{document}\n"
    )
    path = _write(tmp_path, tex, "@misc{real, title={Real}}\n@misc{ghost, title={Ghost}}\n")
    report = write_paper_sea.check_paper(str(path))
    assert _gate(report, "antithesis").startswith("antithesis: 1 (limit 5) PASS")
    assert "...We optimize not speed but latency." in report
    assert _gate(report, "performative honesty").startswith("performative honesty: 3 (limit 3)")
    assert _gate(report, "slop vocabulary") == "slop vocabulary: 1 (limit 0) FAIL"
    assert "We harness compiler feedback" in report
    # \emph{ten} counts as a word; a macro's expansion is not counted (148 + ten,
    # words, 10, GB).
    assert _gate(report, "abstract words") == "abstract words: 152 (150 to 250) PASS"
    # ``10`` is not reused by ``100``; ``\Ten`` is not reused by ``\Tenfold``.
    orphans = _gate(report, "abstract numbers or macros unused elsewhere in the body")
    assert orphans.endswith(": 2 (limit 0) FAIL")
    assert "    10\n" in report and "    \\Ten\n" in report
    # ``et al.`` does not end a sentence; the four-word sentence is a duplicate.
    duplicates = _gate(report, "duplicated sentences")
    assert duplicates == "duplicated sentences: 1 (limit 0) FAIL"
    assert "    L4, L5: we report the final result." in report
    # The commented \cite and \label do not count; the listings label does.
    assert _gate(report, "\\ref without \\label") == "\\ref without \\label: 0 (limit 0) PASS"
    assert _gate(report, "bib entries never cited") == "bib entries never cited: 1 (limit 0) FAIL"
    assert "    ghost\n" in report


def test_section_truncates_long_hit_lists() -> None:
    """A gate lists at most 25 hits and says how many more there are."""
    lines = write_paper_sea._section("em dashes", [f"L{i}: x" for i in range(30)], 0)
    assert lines[0] == "em dashes: 30 (limit 0) FAIL"
    assert len(lines) == 27 and lines[-1] == "    ... 5 more"


def test_summarize_log_keeps_only_boxes_over_ten_points() -> None:
    """A 5 pt overfull box is tolerated; a 20 pt one is reported with its line range."""
    log = (
        "Overfull \\hbox (5.0pt too wide) in paragraph at lines 3--4\n"
        "Overfull \\hbox (20.5pt too wide) in paragraph at lines 8--9\n"
        "Overfull \\vbox (30.0pt too high) has occurred while \\output is active\n"
        "Output written on paper.pdf (2 pages, 100 bytes).\n"
    )
    errors, undefined, overfull, pages = write_paper_sea._summarize_log(log)
    assert (errors, undefined, pages) == ([], [], "2")
    assert overfull == ["20.5pt at lines 8--9", "30.0pt at lines ?"]


def test_hits_show_context_around_long_paragraph_lines() -> None:
    """A hit inside a one-line paragraph shows 50 characters of context on each side."""
    text = "x" * 200 + " rather than " + "y" * 200
    (hit,) = write_paper_sea._hits(text, r"rather than")
    assert hit.startswith("L1: ...x") and hit.endswith("y...")
    assert "x" * 45 + " rather than " + "y" * 45 in hit
    assert len(hit) < 130
    (hit,) = write_paper_sea._hits("short --- line", r"---")
    assert hit == "L1: ...short --- line..."


def test_build_paper_reports_a_missing_file_and_a_missing_tex_install(tmp_path: Path) -> None:
    """Both failure modes come back as a message instead of an exception."""
    missing = tmp_path / "nope.tex"
    assert write_paper_sea.build_paper(str(missing)) == f"no such file: {missing}"
    path = _write(tmp_path, _CLEAN_TEX, _CLEAN_BIB)
    assert write_paper_sea.build_paper(str(path), tex_bin=str(tmp_path)) == (
        "pdflatex not found: install TeX Live or pass tex_bin"
    )
    assert write_paper_sea._run(["", "x"], tmp_path) == (127, "program: not found")


def _tex_bin() -> str:
    """Return the directory of the installed ``pdflatex`` or skip the test."""
    pdflatex = write_paper_sea._find_tex_binary("pdflatex", "")
    if not pdflatex or not Path(pdflatex).is_file():
        pytest.skip("pdflatex is not installed")
    return str(Path(pdflatex).parent)


def test_build_paper_builds_a_clean_paper_with_bibtex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pdflatex, bibtex, pdflatex, pdflatex produce a PDF with no errors, refs or overfull boxes."""
    tex_bin = _tex_bin()
    path = _write(tmp_path, _CLEAN_TEX, _CLEAN_BIB)
    report = write_paper_sea.build_paper(str(path), tex_bin=tex_bin)
    lines = report.splitlines()
    assert lines[0] == "pdflatex 1: exit 0"
    assert lines[1] == "bibtex: exit 0"
    assert "pdflatex 3: exit 0" in lines
    assert _gate(report, "errors") == "errors: 0 (limit 0) PASS"
    assert _gate(report, "undefined references or citations").endswith(": 0 (limit 0) PASS")
    assert _gate(report, "overfull boxes over 10pt") == "overfull boxes over 10pt: 0 (limit 0) PASS"
    assert "pages: 1" in lines
    assert f"pdf: {tmp_path / 'paper.pdf'}" in lines
    assert (tmp_path / "paper.pdf").is_file()
    # Without a bib request the bibliography pass is skipped; pdflatex is found on PATH.
    monkeypatch.setenv("PATH", tex_bin + os.pathsep + os.environ.get("PATH", ""))
    assert shutil.which("pdflatex") == str(Path(tex_bin) / "pdflatex")
    path.write_text(_CLEAN_TEX.replace(r"\bibliography{refs}", "").replace(r"~\cite{bitcask}", ""))
    report = write_paper_sea.build_paper(str(path))
    assert "no bibliography requested; bibtex skipped" in report
    assert _gate(report, "errors") == "errors: 0 (limit 0) PASS"
    # A stale biblatex control file alone does not select biber ...
    (tmp_path / "paper.bcf").write_text("", encoding="utf-8")
    report = write_paper_sea.build_paper(str(path))
    assert "no bibliography requested; bibtex skipped" in report
    # ... a biblatex paper does.  TeX Live may or may not ship biber; either way
    # the pass is reported by name, and a non-zero exit is marked FAIL.
    path.write_text(_BIBLATEX_TEX, encoding="utf-8")
    report = write_paper_sea.build_paper(str(path))
    assert re.search(r"^biber: exit (?:0|[1-9]\d* FAIL)$", report, flags=re.MULTILINE), report


def test_build_paper_summarizes_errors_undefined_refs_and_overfull_boxes(tmp_path: Path) -> None:
    """The log summary names the error, the dangling reference and the wide box with its lines."""
    tex_bin = _tex_bin()
    path = _write(tmp_path, _BROKEN_BUILD_TEX, None)
    report = write_paper_sea.build_paper(str(path), tex_bin=tex_bin)
    assert _gate(report, "errors").endswith(" FAIL")
    assert "Undefined control sequence" in report
    assert _gate(report, "undefined references or citations") == (
        "undefined references or citations: 1 (limit 0) FAIL"
    )
    assert "    sec:missing" in report
    assert _gate(report, "overfull boxes over 10pt").startswith(
        "overfull boxes over 10pt: 1 (limit 0) FAIL"
    )
    assert "pt at lines 5--7" in report, report
    assert f"pdf: {tmp_path / 'paper.pdf'}" in report


def test_agent_gets_the_rules_and_the_tools_and_the_real_gate_report(tmp_path: Path) -> None:
    """With the SEA's configuration the model sees ``check_paper``/``build_paper`` and real output.

    The scripted model calls ``check_paper`` on the fixture and finishes
    with what the tool returned.  The test checks the offered tools, that
    the default system prompt was kept and the rules appended, and that
    the real gate report flowed through the tool-result message.
    """
    path = _write(tmp_path, _SLOPPY_TEX, _SLOPPY_BIB)
    script = [
        tool_call_body("check_paper", {"tex_path": str(path)}, prompt_tokens=500),
        finish_body("<pre>summary: 7 gate(s) failed</pre>", prompt_tokens=600),
    ]
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("write-paper-sea-test")
        result = agent.run(
            prompt_template=f"Review {path} for AI slop; do not edit.",
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=4,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            system_prompt=write_paper_sea.append_to_system_prompt(),
            tools=write_paper_sea.tools(),
            web_tools=write_paper_sea.use_web_tools(),
            is_parallel=write_paper_sea.is_parallel(),
            verbose=False,
        )
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
    assert "7 gate(s) failed" in parsed["summary"]

    agentic: list[dict[str, Any]] = [r for r in requests if r.get("tools")]
    assert len(agentic) == 2, [list(r) for r in requests]
    for request in agentic:
        names = {t["function"]["name"] for t in request["tools"]}
        expected = {"check_paper", "build_paper", "Bash", "run_parallel", "go_to_url", "finish"}
        assert expected <= names, names
        system = str(next(m for m in request["messages"] if m["role"] == "system")["content"])
        assert not system.startswith(write_paper_sea.SYSTEM_PROMPT)
        assert "# Paper-writing agent" in system
        assert "Em dashes: zero in prose" in system
    tool_results = [m for m in agentic[1]["messages"] if m["role"] == "tool"]
    assert len(tool_results) == 1, agentic[1]["messages"]
    assert "summary: 7 gate(s) failed" in str(tool_results[0]["content"])
    assert "antithesis: 6 (limit 5) FAIL" in str(tool_results[0]["content"])
