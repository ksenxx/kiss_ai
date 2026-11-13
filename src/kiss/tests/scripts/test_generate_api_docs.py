# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the generate_api_docs script."""

import ast
import os
import re
import textwrap
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

from kiss.scripts import generate_api_docs
from kiss.scripts.generate_api_docs import (
    ClassInfo,
    ModuleDoc,
    _format_arg,
    _format_func_sig,
    discover_modules,
    generate_markdown,
)


class TestFormatArg:

    def test_long_default_truncated(self) -> None:
        arg = ast.arg(arg="x")
        default = ast.Constant(value="a" * 100)
        result = _format_arg(arg, default)
        assert "..." in result


class TestFormatFuncSig:
    def _parse_func(self, code: str) -> ast.FunctionDef:
        tree = ast.parse(textwrap.dedent(code))
        return tree.body[0]  # type: ignore[return-value]

    def test_vararg(self) -> None:
        func = self._parse_func("def f(*args): pass")
        assert _format_func_sig(func) == "(*args)"

    def test_positional_only_parameters_are_kept(self) -> None:
        """Audit SC1: the parameters before ``/`` were left out of the signature."""
        func = self._parse_func("def f(a, b: int = 1, /, c=2, *, d, **kw) -> None: pass")
        assert _format_func_sig(func) == "(a, b: int = 1, /, c = 2, *, d, **kw) -> None"
        # The ``self`` of a method is not part of its call, with or without the marker.
        method = self._parse_func("def m(self, a=0, /, b=1): pass")
        assert _format_func_sig(method, skip_self=True) == "(a = 0, /, b = 1)"
        only_self = self._parse_func("def m(self, /, b): pass")
        assert _format_func_sig(only_self, skip_self=True) == "(b)"


class TestGenerateMarkdown:
    def test_basic_structure(self) -> None:
        modules = [
            ModuleDoc(
                name="mypack",
                doc="My package.",
                classes=[ClassInfo(name="Foo", bases=[], doc="A foo.", init_sig="()")],
            ),
        ]
        md = generate_markdown(modules)
        assert "# KISS Framework API Reference" in md
        assert "Table of Contents" in md
        assert "`mypack`" in md
        assert "class Foo" in md


class TestEndToEnd:
    def test_generate_produces_valid_markdown(self) -> None:
        modules = discover_modules()
        md = generate_markdown(modules)
        assert len(md) > 1000
        assert md.startswith("# KISS Framework API Reference")
        assert "KISSAgent" in md
        assert "RelentlessAgent" in md
        assert "SorcarAgent" in md
        # A ``@name.setter`` is not a second method, called with the value:
        # the property is one entry, marked as one that can be assigned.
        assert "<br/>`budget_used() -> float` *(property, settable)*" in md
        assert "budget_used(value" not in md

    def test_table_of_contents_links_reach_the_module_headings(self) -> None:
        """Rendered as CommonMark, the contents list is one working link per module."""
        modules = discover_modules()
        html = MarkdownIt("commonmark").render(generate_markdown(modules))
        toc, body = html.split("</details>", 1)
        links = re.findall(r'<li><a href="#([^"]+)"><code>([^<]+)</code></a></li>', toc)
        headings = re.findall(r'<h\d><a id="([^"]+)"></a><code>([^<]+)</code>', body)
        assert "<pre>" not in toc
        assert [name for _, name in links] == [m.name for m in modules]
        assert headings == links
        assert len({target for target, _ in links}) == len(modules)

    def test_main_renames_a_whole_document_over_api_md(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Audit SC1: written in place, ``API.md`` was empty, then partial, while a generator
        filled it, and two generators left a mix of their documents.  It is renamed into place
        whole: the hard link goes on naming the document of the run before, and the file
        keeps its mode."""
        output = tmp_path / "API.md"
        monkeypatch.setattr(generate_api_docs, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(generate_api_docs, "OUTPUT", output)
        output.write_text("the document of the run before\n")
        mode = output.stat().st_mode
        os.link(output, tmp_path / "earlier")
        assert generate_api_docs.main() == 0
        modules = discover_modules()
        assert capsys.readouterr().out == f"Generated API.md ({len(modules)} modules)\n"
        assert (tmp_path / "earlier").read_text() == "the document of the run before\n"
        assert output.read_text(encoding="utf-8") == generate_markdown(modules)
        assert output.stat().st_mode == mode
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["API.md", "earlier"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
