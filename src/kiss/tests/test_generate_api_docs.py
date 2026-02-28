"""Tests for the generate_api_docs script."""

import ast
import textwrap
from pathlib import Path

import pytest

from kiss.scripts.generate_api_docs import (
    ClassInfo,
    FuncInfo,
    ModuleDoc,
    ParsedDoc,
    _extract_class,
    _extract_function,
    _extract_public_from_file,
    _file_to_module,
    _find_def_in_file,
    _format_annotation,
    _format_arg,
    _format_func_sig,
    _get_summary,
    _has_decorator,
    _heading_depth,
    _module_to_path,
    _parse_all_list,
    _parse_imports,
    _render_class,
    _render_function,
    _should_skip,
    _slug,
    _sort_modules,
    discover_modules,
    generate_markdown,
)


class TestFormatAnnotation:
    def test_none(self) -> None:
        assert _format_annotation(None) == ""

    def test_simple_name(self) -> None:
        node = ast.parse("int", mode="eval").body
        assert _format_annotation(node) == "int"

    def test_complex_type(self) -> None:
        node = ast.parse("dict[str, Any]", mode="eval").body
        assert _format_annotation(node) == "dict[str, Any]"


class TestFormatArg:
    def test_bare_arg(self) -> None:
        arg = ast.arg(arg="x")
        assert _format_arg(arg) == "x"

    def test_annotated_arg(self) -> None:
        arg = ast.arg(arg="x", annotation=ast.Constant(value="int"))
        result = _format_arg(arg)
        assert result.startswith("x: ")

    def test_arg_with_default(self) -> None:
        arg = ast.arg(arg="x")
        default = ast.Constant(value=42)
        assert _format_arg(arg, default) == "x = 42"

    def test_long_default_truncated(self) -> None:
        arg = ast.arg(arg="x")
        default = ast.Constant(value="a" * 100)
        result = _format_arg(arg, default)
        assert "..." in result


class TestFormatFuncSig:
    def _parse_func(self, code: str) -> ast.FunctionDef:
        tree = ast.parse(textwrap.dedent(code))
        return tree.body[0]  # type: ignore[return-value]

    def test_no_args(self) -> None:
        func = self._parse_func("def f(): pass")
        assert _format_func_sig(func) == "() -> None" or _format_func_sig(func) == "()"

    def test_positional_args(self) -> None:
        func = self._parse_func("def f(a, b): pass")
        assert _format_func_sig(func) == "(a, b)"

    def test_default_args(self) -> None:
        func = self._parse_func("def f(a, b=1): pass")
        assert _format_func_sig(func) == "(a, b = 1)"

    def test_skip_self(self) -> None:
        func = self._parse_func("def f(self, x): pass")
        assert _format_func_sig(func, skip_self=True) == "(x)"

    def test_skip_cls(self) -> None:
        func = self._parse_func("def f(cls, x): pass")
        assert _format_func_sig(func, skip_self=True) == "(x)"

    def test_vararg(self) -> None:
        func = self._parse_func("def f(*args): pass")
        assert _format_func_sig(func) == "(*args)"

    def test_kwonly(self) -> None:
        func = self._parse_func("def f(*, key=None): pass")
        assert _format_func_sig(func) == "(*, key = None)"

    def test_kwargs(self) -> None:
        func = self._parse_func("def f(**kwargs): pass")
        assert _format_func_sig(func) == "(**kwargs)"

    def test_return_annotation(self) -> None:
        func = self._parse_func("def f() -> int: pass")
        assert _format_func_sig(func) == "() -> int"


class TestGetSummary:
    def test_no_docstring(self) -> None:
        tree = ast.parse("x = 1")
        assert _get_summary(tree) == ""

    def test_module_docstring(self) -> None:
        tree = ast.parse('"""Hello world."""\nx = 1')
        assert _get_summary(tree) == "Hello world."

    def test_multiline_returns_first_line(self) -> None:
        tree = ast.parse('"""First line.\n\nMore details."""\nx = 1')
        assert _get_summary(tree) == "First line."

    def test_class_docstring(self) -> None:
        tree = ast.parse('class Foo:\n    """My class."""\n    pass')
        assert _get_summary(tree.body[0]) == "My class."  # type: ignore[arg-type]

    def test_empty_body(self) -> None:
        func = ast.parse("def f(): pass").body[0]
        assert _get_summary(func) == ""  # type: ignore[arg-type]


class TestHasDecorator:
    def _parse_func(self, code: str) -> ast.FunctionDef:
        tree = ast.parse(textwrap.dedent(code))
        return tree.body[0]  # type: ignore[return-value]

    def test_no_decorator(self) -> None:
        func = self._parse_func("def f(): pass")
        assert not _has_decorator(func, "property")

    def test_name_decorator(self) -> None:
        func = self._parse_func("@property\ndef f(): pass")
        assert _has_decorator(func, "property")

    def test_attribute_decorator(self) -> None:
        func = self._parse_func("@abc.abstractmethod\ndef f(): pass")
        assert _has_decorator(func, "abstractmethod")


class TestExtractClass:
    def test_simple_class(self) -> None:
        tree = ast.parse(textwrap.dedent("""
            class Foo:
                '''My class.'''
                def __init__(self, x: int) -> None:
                    self.x = x
                def bar(self) -> str:
                    '''Do bar.'''
                    return ''
                def _private(self):
                    pass
        """))
        cls = _extract_class(tree.body[0])  # type: ignore[arg-type]
        assert cls.name == "Foo"
        assert cls.doc == "My class."
        assert "(x: int) -> None" in cls.init_sig
        assert len(cls.methods) == 1
        assert cls.methods[0].name == "bar"
        assert cls.methods[0].parsed_doc.summary == "Do bar."

    def test_class_with_bases(self) -> None:
        tree = ast.parse("class Foo(Bar, Baz): pass")
        cls = _extract_class(tree.body[0])  # type: ignore[arg-type]
        assert cls.bases == ["Bar", "Baz"]

    def test_no_init(self) -> None:
        tree = ast.parse("class Foo: pass")
        cls = _extract_class(tree.body[0])  # type: ignore[arg-type]
        assert cls.init_sig == ""


class TestExtractFunction:
    def test_sync_function(self) -> None:
        tree = ast.parse("def foo(x: int) -> str:\n    '''Do foo.'''\n    return ''")
        func = _extract_function(tree.body[0])  # type: ignore[arg-type]
        assert func.name == "foo"
        assert func.parsed_doc.summary == "Do foo."
        assert not func.is_async
        assert "(x: int) -> str" in func.signature

    def test_async_function(self) -> None:
        tree = ast.parse("async def foo(): pass")
        func = _extract_function(tree.body[0])  # type: ignore[arg-type]
        assert func.is_async


class TestParseAllList:
    def test_no_all(self) -> None:
        tree = ast.parse("x = 1")
        assert _parse_all_list(tree) is None

    def test_simple_all(self) -> None:
        tree = ast.parse('__all__ = ["Foo", "bar"]')
        assert _parse_all_list(tree) == ["Foo", "bar"]

    def test_non_list_all(self) -> None:
        tree = ast.parse("__all__ = ('Foo',)")
        assert _parse_all_list(tree) is None


class TestParseImports:
    def test_from_imports(self) -> None:
        tree = ast.parse("from kiss.core.config import Config, DEFAULT_CONFIG")
        result = _parse_imports(tree)
        assert result == {"Config": "kiss.core.config", "DEFAULT_CONFIG": "kiss.core.config"}

    def test_aliased_import(self) -> None:
        tree = ast.parse("from foo import bar as baz")
        result = _parse_imports(tree)
        assert result == {"baz": "foo"}

    def test_no_imports(self) -> None:
        tree = ast.parse("x = 1")
        assert _parse_imports(tree) == {}


class TestModuleToPath:
    def test_package(self) -> None:
        path = _module_to_path("kiss.core")
        assert path.name == "__init__.py"
        assert "core" in str(path)

    def test_module(self) -> None:
        path = _module_to_path("kiss.core.kiss_agent")
        assert path.name == "kiss_agent.py"


class TestFileToModule:
    def test_init_file(self) -> None:
        from kiss.scripts.generate_api_docs import KISS_SRC
        path = KISS_SRC / "core" / "__init__.py"
        assert _file_to_module(path) == "kiss.core"

    def test_module_file(self) -> None:
        from kiss.scripts.generate_api_docs import KISS_SRC
        path = KISS_SRC / "core" / "kiss_agent.py"
        assert _file_to_module(path) == "kiss.core.kiss_agent"


class TestShouldSkip:
    def test_skip_tests(self) -> None:
        from kiss.scripts.generate_api_docs import KISS_SRC
        assert _should_skip(KISS_SRC / "tests" / "test_foo.py")

    def test_skip_excluded_file(self) -> None:
        from kiss.scripts.generate_api_docs import KISS_SRC
        assert _should_skip(KISS_SRC / "_version.py")

    def test_allow_core(self) -> None:
        from kiss.scripts.generate_api_docs import KISS_SRC
        assert not _should_skip(KISS_SRC / "core" / "kiss_agent.py")


class TestFindDefInFile:
    def test_find_class(self, tmp_path: Path) -> None:
        p = tmp_path / "mod.py"
        p.write_text("class Foo:\n    '''Doc.'''\n    pass\n")
        result = _find_def_in_file(p, "Foo")
        assert isinstance(result, ClassInfo)
        assert result.name == "Foo"

    def test_find_function(self, tmp_path: Path) -> None:
        p = tmp_path / "mod.py"
        p.write_text("def bar() -> int:\n    '''Doc.'''\n    return 1\n")
        result = _find_def_in_file(p, "bar")
        assert isinstance(result, FuncInfo)
        assert result.name == "bar"

    def test_not_found(self, tmp_path: Path) -> None:
        p = tmp_path / "mod.py"
        p.write_text("x = 1\n")
        assert _find_def_in_file(p, "Foo") is None

    def test_missing_file(self, tmp_path: Path) -> None:
        assert _find_def_in_file(tmp_path / "nope.py", "Foo") is None


class TestExtractPublicFromFile:
    def test_extracts_public_only(self, tmp_path: Path) -> None:
        p = tmp_path / "mod.py"
        p.write_text(textwrap.dedent("""
            class Pub: pass
            class _Priv: pass
            def pub_fn(): pass
            def _priv_fn(): pass
            def main(): pass
        """))
        classes, functions = _extract_public_from_file(p)
        assert [c.name for c in classes] == ["Pub"]
        assert [f.name for f in functions] == ["pub_fn"]


class TestSlug:
    def test_basic(self) -> None:
        assert _slug("kiss.core.kiss_agent") == "kisscorekiss_agent"

    def test_spaces(self) -> None:
        assert _slug("hello world") == "hello-world"


class TestHeadingDepth:
    def test_top_level(self) -> None:
        assert _heading_depth("kiss") == 2

    def test_nested(self) -> None:
        assert _heading_depth("kiss.core") == 3

    def test_capped_at_4(self) -> None:
        assert _heading_depth("kiss.a.b.c.d") == 4


class TestSortModules:
    def test_known_order(self) -> None:
        m1 = ModuleDoc(name="kiss.docker", doc="", all_exports=None)
        m2 = ModuleDoc(name="kiss.core", doc="", all_exports=None)
        m3 = ModuleDoc(name="kiss", doc="", all_exports=None)
        result = _sort_modules([m1, m2, m3])
        assert [m.name for m in result] == ["kiss", "kiss.core", "kiss.docker"]

    def test_unknown_modules_at_end(self) -> None:
        m1 = ModuleDoc(name="kiss", doc="", all_exports=None)
        m2 = ModuleDoc(name="kiss.unknown", doc="", all_exports=None)
        result = _sort_modules([m2, m1])
        assert result[0].name == "kiss"
        assert result[1].name == "kiss.unknown"


class TestRenderClass:
    def test_renders_class_with_methods(self) -> None:
        cls = ClassInfo(
            name="Foo", bases=["Bar"], doc="A foo.",
            init_sig="(x: int)", methods=[
                FuncInfo(name="do_it", signature="() -> str", parsed_doc=ParsedDoc("Does it.")),
            ],
        )
        lines: list[str] = []
        _render_class(lines, cls, 4)
        text = "\n".join(lines)
        assert "class Foo(Bar)" in text
        assert "A foo." in text
        assert "Foo(x: int)" in text
        assert "do_it" in text
        assert "Does it." in text

    def test_async_method(self) -> None:
        cls = ClassInfo(
            name="Foo", bases=[], doc="", init_sig="",
            methods=[FuncInfo(name="bar", signature="()", parsed_doc=ParsedDoc(""), is_async=True)],
        )
        lines: list[str] = []
        _render_class(lines, cls, 4)
        text = "\n".join(lines)
        assert "async bar" in text

    def test_no_methods_no_init(self) -> None:
        cls = ClassInfo(name="Foo", bases=[], doc="", init_sig="", methods=[])
        lines: list[str] = []
        _render_class(lines, cls, 4)
        text = "\n".join(lines)
        assert "Constructor" not in text
        assert "Methods" not in text


class TestRenderFunction:
    def test_renders_sync(self) -> None:
        func = FuncInfo(name="foo", signature="(x: int) -> str", parsed_doc=ParsedDoc("Does foo."))
        lines: list[str] = []
        _render_function(lines, func)
        text = "\n".join(lines)
        assert "def foo(x: int) -> str" in text
        assert "Does foo." in text

    def test_renders_async(self) -> None:
        func = FuncInfo(name="bar", signature="()", parsed_doc=ParsedDoc(""), is_async=True)
        lines: list[str] = []
        _render_function(lines, func)
        text = "\n".join(lines)
        assert "async def bar()" in text


class TestGenerateMarkdown:
    def test_basic_structure(self) -> None:
        modules = [
            ModuleDoc(
                name="mypack", doc="My package.", all_exports=["Foo"],
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

    def test_empty_modules(self) -> None:
        md = generate_markdown([])
        assert "# KISS Framework API Reference" in md


class TestDiscoverModules:
    def test_finds_key_modules(self) -> None:
        modules = discover_modules()
        names = {m.name for m in modules}
        assert "kiss" in names
        assert "kiss.core" in names
        assert "kiss.core.kiss_agent" in names
        assert "kiss.agents" in names
        assert "kiss.agents.gepa" in names
        assert "kiss.docker" in names
        assert "kiss.agents.kiss_evolve" in names

    def test_excludes_deprecated(self) -> None:
        modules = discover_modules()
        names = {m.name for m in modules}
        assert not any("create_and_optimize_agent" in n for n in names)
        assert not any("self_evolving_multi_agent" in n for n in names)

    def test_excludes_tests_and_scripts(self) -> None:
        modules = discover_modules()
        names = {m.name for m in modules}
        assert not any("tests" in n for n in names)
        assert not any("scripts" in n for n in names)

    def test_kiss_agent_documented(self) -> None:
        modules = discover_modules()
        agent_mod = next(m for m in modules if m.name == "kiss.core.kiss_agent")
        class_names = [c.name for c in agent_mod.classes]
        assert "KISSAgent" in class_names

    def test_no_duplicate_loss_for_finish(self) -> None:
        modules = discover_modules()
        utils_mod = next(m for m in modules if m.name == "kiss.core.utils")
        func_names = [f.name for f in utils_mod.functions]
        assert "finish" in func_names

        relentless_mod = next(
            m for m in modules if m.name == "kiss.agents.assistant.relentless_agent"
        )
        func_names2 = [f.name for f in relentless_mod.functions]
        assert "finish" in func_names2

    def test_package_exports_correct(self) -> None:
        modules = discover_modules()
        gepa_mod = next(m for m in modules if m.name == "kiss.agents.gepa")
        assert gepa_mod.is_package
        assert gepa_mod.all_exports is not None
        assert "GEPA" in gepa_mod.all_exports
        class_names = [c.name for c in gepa_mod.classes]
        assert "GEPA" in class_names

    def test_model_documented_under_core_models(self) -> None:
        modules = discover_modules()
        models_mod = next(m for m in modules if m.name == "kiss.core.models")
        class_names = [c.name for c in models_mod.classes]
        assert "Model" in class_names


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

    def test_signatures_match_source(self) -> None:
        modules = discover_modules()
        agent_mod = next(m for m in modules if m.name == "kiss.core.kiss_agent")
        kiss_agent = next(c for c in agent_mod.classes if c.name == "KISSAgent")
        assert "model_name: str" in kiss_agent.methods[0].signature
        assert "-> str" in kiss_agent.methods[0].signature

    def test_method_count_reasonable(self) -> None:
        modules = discover_modules()
        models_mod = next(m for m in modules if m.name == "kiss.core.models")
        model_cls = next(c for c in models_mod.classes if c.name == "Model")
        assert len(model_cls.methods) >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
