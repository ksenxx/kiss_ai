"""Tests for the generate_api_docs script."""

import ast
import textwrap
from pathlib import Path

import pytest

from kiss.scripts.generate_api_docs import (
    ClassInfo,
    ModuleDoc,
    _find_def_in_file,
    _format_annotation,
    _format_arg,
    _format_func_sig,
    _parse_all_list,
    discover_modules,
    generate_markdown,
)


class TestFormatAnnotation:
    def test_none(self) -> None:
        assert _format_annotation(None) == ""


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

    def test_kwonly(self) -> None:
        func = self._parse_func("def f(*, key=None): pass")
        assert _format_func_sig(func) == "(*, key = None)"


class TestHasDecorator:
    def _parse_func(self, code: str) -> ast.FunctionDef:
        tree = ast.parse(textwrap.dedent(code))
        return tree.body[0]  # type: ignore[return-value]


class TestParseAllList:

    def test_non_list_all(self) -> None:
        tree = ast.parse("__all__ = ('Foo',)")
        assert _parse_all_list(tree) is None


class TestFindDefInFile:

    def test_missing_file(self, tmp_path: Path) -> None:
        assert _find_def_in_file(tmp_path / "nope.py", "Foo") is None


class TestGenerateMarkdown:
    def test_basic_structure(self) -> None:
        modules = [
            ModuleDoc(
                name="mypack",
                doc="My package.",
                all_exports=["Foo"],
                classes=[ClassInfo(name="Foo", bases=[], doc="A foo.", init_sig="()")],
                is_package=True,
            ),
        ]
        md = generate_markdown(modules)
        assert "# KISS Framework API Reference" in md
        assert "Table of Contents" in md
        assert "`mypack`" in md
        assert "from mypack import Foo" in md
        assert "class Foo" in md


class TestEndToEnd:
    def test_generate_produces_valid_markdown(self) -> None:
        modules = discover_modules()
        md = generate_markdown(modules)
        assert len(md) > 1000
        assert md.startswith("# KISS Framework API Reference")
        assert "KISSAgent" in md
        assert "DockerManager" in md
        assert "SimpleRAG" in md
        assert "GEPA" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
