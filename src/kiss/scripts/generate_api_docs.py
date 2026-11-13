# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Generate API.md from kiss package source code using AST introspection."""

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

from kiss.core.utils import atomic_write_text

KISS_SRC = Path(__file__).resolve().parent.parent
PROJECT_ROOT = KISS_SRC.parent.parent
OUTPUT = PROJECT_ROOT / "API.md"

INCLUDE_FILES = (
    "core/kiss_agent.py",
    "agents/sorcar/relentless_agent.py",
    "agents/sorcar/sorcar_agent.py",
    "agents/sorcar/chat_sorcar_agent.py",
    "server/sorcar.py",
    # The synchronous client API (``run`` / ``TaskResult``), defined in
    # the sorcar layer and re-exported by ``kiss.server.sorcar``.
    "agents/sorcar/daemon_client.py",
)
"""The documented modules, as paths below the package, in the order of their sections."""


@dataclass
class ParsedDoc:
    """A Google-style docstring: its text before the sections, its ``Args`` and ``Returns``."""

    summary: str
    args: list[tuple[str, str]] = field(default_factory=list)
    returns: str = ""


@dataclass
class FuncInfo:
    """A public function or method: its rendered signature and parsed docstring."""

    name: str
    signature: str
    parsed_doc: ParsedDoc
    is_async: bool = False
    is_property: bool = False
    is_settable: bool = False


@dataclass
class ClassInfo:
    """A public class: its bases, summary line, constructor and public methods."""

    name: str
    bases: list[str]
    doc: str
    init_sig: str = ""
    init_doc: ParsedDoc = field(default_factory=lambda: ParsedDoc(""))
    methods: list[FuncInfo] = field(default_factory=list)


@dataclass
class ModuleDoc:
    """A documented module: its dotted name, summary line, public classes and functions."""

    name: str
    doc: str
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FuncInfo] = field(default_factory=list)


def _format_arg(arg: ast.arg, default: ast.expr | None = None) -> str:
    s = arg.arg
    if arg.annotation:
        s += f": {ast.unparse(arg.annotation)}"
    if default is not None:
        val = ast.unparse(default)
        if len(val) > 50:
            val = "..."
        s += f" = {val}"
    return s


def _format_func_sig(node: ast.FunctionDef | ast.AsyncFunctionDef, skip_self: bool = False) -> str:
    parts: list[str] = []
    args = node.args
    positional = [*args.posonlyargs, *args.args]  # the defaults are of the last of them
    first_default = len(positional) - len(args.defaults)
    for i, arg in enumerate(positional):
        if skip_self and i == 0 and arg.arg in ("self", "cls"):
            continue
        default = args.defaults[i - first_default] if i >= first_default else None
        parts.append(_format_arg(arg, default))
        if i + 1 == len(args.posonlyargs):
            parts.append("/")
    if args.vararg:
        parts.append(f"*{_format_arg(args.vararg)}")
    elif args.kwonlyargs:
        parts.append("*")
    for i, arg in enumerate(args.kwonlyargs):
        parts.append(_format_arg(arg, args.kw_defaults[i]))
    if args.kwarg:
        parts.append(f"**{_format_arg(args.kwarg)}")
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"({', '.join(parts)}){ret}"


def _get_summary(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    return doc.split("\n")[0].strip()


def _parse_google_docstring(raw: str) -> ParsedDoc:
    lines = raw.split("\n")
    summary_lines: list[str] = []
    args: list[tuple[str, str]] = []
    returns_parts: list[str] = []

    section = "summary"
    current_arg_name = ""
    current_arg_desc = ""
    args_base_indent = -1

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip()) if stripped else 0

        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            section = "args"
            args_base_indent = -1
            continue

        if indent == 0 and stripped.lower().startswith("returns:"):
            if current_arg_name:
                args.append((current_arg_name, current_arg_desc.strip()))
                current_arg_name = current_arg_desc = ""
            section = "returns"
            rest = stripped[len("returns:") :].strip()
            if rest:  # pragma: no branch
                returns_parts.append(rest)
            continue

        if indent == 0 and stripped.lower().startswith(
            ("raises:", "example:", "examples:", "note:", "notes:", "yields:", "see also:")
        ):
            if current_arg_name:
                args.append((current_arg_name, current_arg_desc.strip()))
                current_arg_name = current_arg_desc = ""
            section = "other"
            continue

        if section == "summary":
            if stripped:
                summary_lines.append(stripped)
        elif section == "args":
            if not stripped:
                continue
            if args_base_indent == -1:
                args_base_indent = indent
            if indent <= args_base_indent and ":" in stripped:
                if current_arg_name:
                    args.append((current_arg_name, current_arg_desc.strip()))
                name, _, desc = stripped.partition(":")
                current_arg_name = name.split("(")[0].strip()  # "name (type)" -> "name"
                current_arg_desc = desc.strip()
            elif current_arg_name:  # pragma: no branch
                current_arg_desc += " " + stripped
        elif section == "returns":
            if stripped:
                returns_parts.append(stripped)

    if current_arg_name:
        args.append((current_arg_name, current_arg_desc.strip()))

    summary = " ".join(summary_lines).strip()
    returns = " ".join(returns_parts).strip()
    return ParsedDoc(summary=summary, args=args, returns=returns)


def _has_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == name)
        or (isinstance(d, ast.Attribute) and d.attr == name)
        for d in node.decorator_list
    )


def _extract_class(node: ast.ClassDef) -> ClassInfo:
    bases = [ast.unparse(b) for b in node.bases]
    init_sig = ""
    init_doc = ParsedDoc("")
    methods: list[FuncInfo] = []
    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name == "__init__":
            init_sig = _format_func_sig(item, skip_self=True)
            init_doc = _parse_google_docstring(ast.get_docstring(item) or "")
        elif _has_decorator(item, "setter"):
            # ``@name.setter`` is no second method: the property can be assigned.
            for method in methods:
                if method.name == item.name:
                    method.is_settable = True
        elif not item.name.startswith("_"):
            methods.append(
                FuncInfo(
                    name=item.name,
                    signature=_format_func_sig(item, skip_self=True),
                    parsed_doc=_parse_google_docstring(ast.get_docstring(item) or ""),
                    is_async=isinstance(item, ast.AsyncFunctionDef),
                    is_property=_has_decorator(item, "property"),
                )
            )
    return ClassInfo(
        name=node.name,
        bases=bases,
        doc=_get_summary(node),
        init_sig=init_sig,
        init_doc=init_doc,
        methods=methods,
    )


def _extract_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FuncInfo:
    return FuncInfo(
        name=node.name,
        signature=_format_func_sig(node),
        parsed_doc=_parse_google_docstring(ast.get_docstring(node) or ""),
        is_async=isinstance(node, ast.AsyncFunctionDef),
    )


SKIP_FUNCTIONS = {"main"}


def _extract_public(tree: ast.Module) -> tuple[list[ClassInfo], list[FuncInfo]]:
    classes: list[ClassInfo] = []
    functions: list[FuncInfo] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append(_extract_class(node))
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
            and node.name not in SKIP_FUNCTIONS
        ):
            functions.append(_extract_function(node))
    return classes, functions


def discover_modules() -> list[ModuleDoc]:
    """Return the documentation of every module of :data:`INCLUDE_FILES` with a public API.

    Returns:
        One :class:`ModuleDoc` per listed file that defines a public class
        or function, in the listed order.

    Raises:
        OSError: When a listed file is missing: the list is stale.
    """
    modules: list[ModuleDoc] = []
    for rel in INCLUDE_FILES:
        tree = ast.parse((KISS_SRC / rel).read_text(encoding="utf-8"))
        classes, functions = _extract_public(tree)
        if classes or functions:
            name = ".".join((KISS_SRC.name, *Path(rel).with_suffix("").parts))
            modules.append(
                ModuleDoc(name=name, doc=_get_summary(tree), classes=classes, functions=functions)
            )
    return modules


def _slug(text: str) -> str:
    return text.replace(".", "").replace(" ", "-").lower()


def _heading_depth(module_name: str) -> int:
    depth = module_name.count(".")
    return min(depth + 2, 4)


def generate_markdown(modules: list[ModuleDoc]) -> str:
    """Return the API reference of *modules* as Markdown.

    Args:
        modules: The modules to document (:func:`discover_modules`), in
            the order of their sections.

    Returns:
        The document: a table of contents, then one section per module
        with its classes (constructor and public methods) and functions.
    """
    lines: list[str] = []

    lines.append("# KISS Framework API Reference\n")
    lines.append("> **Auto-generated** — run `uv run generate-api-docs` to regenerate.\n")

    lines.append("<details><summary><b>Table of Contents</b></summary>\n")
    for mod in modules:
        lines.append(f"- [`{mod.name}`](#{_slug(mod.name)})")
    lines.append("\n</details>\n")
    lines.append("---\n")

    for mod in modules:
        h = "#" * _heading_depth(mod.name)
        doc_part = f" — *{mod.doc}*" if mod.doc else ""
        # The anchor is the TOC link's target: a renderer's own heading id, where it makes
        # one, is derived from the whole heading text, description included.
        lines.append(f'{h} <a id="{_slug(mod.name)}"></a>`{mod.name}`{doc_part}\n')

        for cls in mod.classes:
            _render_class(lines, cls, _heading_depth(mod.name) + 1)

        for func in mod.functions:
            _render_function(lines, func)

        lines.append("---\n")

    return "\n".join(lines)


def _render_class(lines: list[str], cls: ClassInfo, depth: int) -> None:
    h = "#" * depth
    bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
    doc_part = f" — {cls.doc}" if cls.doc else ""
    lines.append(f"{h} `class {cls.name}{bases_str}`{doc_part}\n")
    if cls.init_sig:
        lines.append(f"**Constructor:** `{cls.name}{cls.init_sig}`\n")
        _render_args_returns(lines, cls.init_doc)
    for m in cls.methods:
        _render_method(lines, m)


def _render_method(lines: list[str], func: FuncInfo) -> None:
    prefix = "async " if func.is_async else ""
    prop_tag = ""
    if func.is_property:
        prop_tag = " *(property, settable)*" if func.is_settable else " *(property)*"
    sig = f"`{prefix}{func.name}{func.signature}`{prop_tag}"
    summary = f" — {func.parsed_doc.summary}" if func.parsed_doc.summary else ""
    lines.append(f"- **{func.name}**{summary}<br/>{sig}")
    _render_args_returns(lines, func.parsed_doc, indent="  ")


def _render_function(lines: list[str], func: FuncInfo) -> None:
    prefix = "async " if func.is_async else ""
    sig = f"`{prefix}def {func.name}{func.signature}`"
    summary = f" — {func.parsed_doc.summary}" if func.parsed_doc.summary else ""
    lines.append(f"**`{func.name}`**{summary}<br/>{sig}\n")
    _render_args_returns(lines, func.parsed_doc)


def _render_args_returns(lines: list[str], doc: ParsedDoc, indent: str = "") -> None:
    for name, desc in doc.args:
        lines.append(f"{indent}- `{name}`: {desc}")
    if doc.returns:
        lines.append(f"{indent}- **Returns:** {doc.returns}")
    if doc.args or doc.returns:
        lines.append("")


def main() -> int:
    """Write ``API.md`` at the repository root; return the exit code 0.

    The document is renamed into place whole: generators that run together
    (two checks of one tree) would otherwise write through each other, and
    a reader would find the file empty or partial meanwhile.
    """
    modules = discover_modules()
    markdown = generate_markdown(modules)
    atomic_write_text(OUTPUT, markdown, create_mode=0o666)
    print(f"Generated {OUTPUT.relative_to(PROJECT_ROOT)} ({len(modules)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
