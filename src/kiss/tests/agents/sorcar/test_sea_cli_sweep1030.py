# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``sea`` command reports what it cannot read instead of crashing or staying silent.

* ``sea lint`` crashed with a traceback (losing every other script's
  findings) on a script that is not valid UTF-8 or is nested too deeply
  for the parser: only ``OSError`` and ``SyntaxError`` were caught, and
  ``UnicodeDecodeError`` is a ``ValueError``.  Such a script is now
  ``broken``.
* ``sea docs --check docs/typo.md`` printed nothing and exited 0: every
  page, the named ones included, was filtered through ``exists()``.  A
  named page that does not exist is now reported and exits 1.
* ``sea --help`` described ``sea docs`` as regenerating a ``kind`` table
  that no longer exists; it now names the tables ``sea docs`` renders.
* ``lint_prose`` blanked generated blocks with its own copy of the block
  pattern, which did not accept the ``-`` that ``sea docs`` accepts in a
  block name; it now uses ``sea_docs.BLOCK``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_docs, sea_lint
from kiss.agents.sorcar.sea_cli import main

TOO_DEEP = "x = " + " + ".join(['"a"'] * 100_000) + "\n"
"""A valid expression nested too deeply for CPython's parser (``RecursionError``)."""


@pytest.mark.parametrize(
    "source",
    [b'# -*- coding: latin-1 -*-\nNAME = "caf\xe9"\n', TOO_DEEP.encode()],
    ids=["latin-1", "too-deep"],
)
@pytest.mark.parametrize("fix", [False, True])
def test_lint_reports_an_unreadable_script_as_broken(
    tmp_path: Path, source: bytes, fix: bool
) -> None:
    """The script is one ``broken: cannot parse`` finding, with or without ``--fix``."""
    script = tmp_path / "unreadable_sea.py"
    script.write_bytes(source)
    findings = sea_lint.lint_all([script], fix=fix)
    assert [(f.path, f.code) for f in findings] == [(script.resolve(), "broken")]
    assert findings[0].message.startswith("cannot parse: ")
    assert script.read_bytes() == source  # --fix never rewrites what it cannot read


def test_sea_docs_reports_a_named_page_that_does_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--check`` on a missing page fails, naming it."""
    missing = tmp_path / "typo.md"
    assert main(["docs", "--check", str(missing)]) == 1
    assert capsys.readouterr().out.splitlines() == [f"missing: {missing}"]


def test_sea_docs_regenerates_the_named_pages_that_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing page fails the run without stopping the others from being regenerated."""
    page = tmp_path / "page.md"
    page.write_text("<!-- sea-docs: precedence -->\nold\n<!-- /sea-docs -->\n", encoding="utf-8")
    missing = tmp_path / "typo.md"
    assert main(["docs", str(page), str(missing)]) == 1
    assert capsys.readouterr().out.splitlines() == [f"missing: {missing}", f"updated: {page}"]
    assert page.read_text(encoding="utf-8") == (
        f"<!-- sea-docs: precedence -->\n{sea_docs.precedence_block()}\n<!-- /sea-docs -->\n"
    )


def test_sea_docs_check_of_the_default_pages_reports_no_missing_page(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With no page named, only the bundled pages that exist are read; it fails only if stale."""
    code = main(["docs", "--check"])
    lines = capsys.readouterr().out.splitlines()
    assert all(line.startswith("stale: ") for line in lines), lines
    assert code == (1 if lines else 0)


def test_sea_help_names_the_tables_sea_docs_renders(capsys: pytest.CaptureFixture[str]) -> None:
    """The ``docs`` sub-command's help lists the ``TABLES`` keys and no removed table."""
    with pytest.raises(SystemExit) as exited:
        main(["--help"])
    assert exited.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert f"regenerate the sea-docs tables ({', '.join(sea_docs.TABLES)})" in text
    assert "kind" not in text


def test_lint_prose_skips_every_block_sea_docs_generates(tmp_path: Path) -> None:
    """Prose inside a block whose name has a ``-`` is generated, so it is not linted."""
    page = tmp_path / sea_docs.GENERATED_FILES[0]
    page.parent.mkdir(parents=True)
    page.write_text(
        "intro\n<!-- sea-docs: home-paths -->\nsee ~/.kiss/SEAS.md\n<!-- /sea-docs -->\n"
        "then ~/.kiss/config.json\n",
        encoding="utf-8",
    )
    home_path_message = sea_lint.STALE_PROSE[0][1]
    assert [(f.path, f.message) for f in sea_lint.lint_prose(tmp_path)] == [
        (page, f"line 5: {home_path_message}")
    ]
