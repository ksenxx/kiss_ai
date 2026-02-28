"""Generate API.md from kiss package source code using AST introspection."""

import ast
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

KISS_SRC = Path(__file__).resolve().parent.parent
PROJECT_ROOT = KISS_SRC.parent.parent
OUTPUT = PROJECT_ROOT / "API.md"
EXCLUDE_DIRS = {
    "tests", "scripts", "evals", "viz_trajectory", "demo", "__pycache__",
    "create_and_optimize_agent", "self_evolving_multi_agent",
}
EXCLUDE_FILES = {"_version.py", "conftest.py", "novelty_prompts.py"}


@dataclass
class ParsedDoc:
    summary: str
    args: list[tuple[str, str]] = field(default_factory=list)
    returns: str = ""


@dataclass
class FuncInfo:
    name: str
    signature: str
    parsed_doc: ParsedDoc
    is_async: bool = False
    is_property: bool = False


@dataclass
class ClassInfo:
    name: str
    bases: list[str]
    doc: str
    init_sig: str = ""
    init_doc: ParsedDoc = field(default_factory=lambda: ParsedDoc(""))
    methods: list[FuncInfo] = field(default_factory=list)


@dataclass
class ModuleDoc:
    name: str
    doc: str
    all_exports: list[str] | None
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FuncInfo] = field(default_factory=list)
    is_package: bool = False
    deprecated: bool = False


def _format_annotation(node: ast.expr | None) -> str:
    if node is None:
        return ""
    return ast.unparse(node)


def _format_arg(arg: ast.arg, default: ast.expr | None = None) -> str:
    s = arg.arg
    if arg.annotation:
        s += f": {_format_annotation(arg.annotation)}"
    if default is not None:
        val = ast.unparse(default)
        if len(val) > 50:
            val = "..."
        s += f" = {val}"
    return s


def _format_func_sig(node: ast.FunctionDef | ast.AsyncFunctionDef, skip_self: bool = False) -> str:
    parts: list[str] = []
    args = node.args
    n_defaults = len(args.defaults)
    n_args = len(args.args)
    for i, arg in enumerate(args.args):
        if skip_self and i == 0 and arg.arg in ("self", "cls"):
            continue
        di = i - (n_args - n_defaults)
        parts.append(_format_arg(arg, args.defaults[di] if di >= 0 else None))
    if args.vararg:
        parts.append(f"*{_format_arg(args.vararg)}")
    elif args.kwonlyargs:
        parts.append("*")
    for i, arg in enumerate(args.kwonlyargs):
        parts.append(_format_arg(arg, args.kw_defaults[i]))
    if args.kwarg:
        parts.append(f"**{_format_arg(args.kwarg)}")
    ret = f" -> {_format_annotation(node.returns)}" if node.returns else ""
    return f"({', '.join(parts)}){ret}"


def _get_summary(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    return doc.split("\n")[0].strip()


def _parse_google_docstring(raw: str) -> ParsedDoc:
    if not raw:
        return ParsedDoc("")

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
            rest = stripped[len("returns:"):].strip()
            if rest:
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
                parts = stripped.split(":", 1)
                param_part = parts[0].strip()
                paren_idx = param_part.find(" (")
                if paren_idx > 0:
                    current_arg_name = param_part[:paren_idx].strip()
                elif "(" in param_part:
                    current_arg_name = param_part.split("(")[0].strip()
                else:
                    current_arg_name = param_part
                current_arg_desc = parts[1].strip() if len(parts) > 1 else ""
            elif current_arg_name:
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
        (isinstance(d, ast.Name) and d.id == name) or
        (isinstance(d, ast.Attribute) and d.attr == name)
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
        elif not item.name.startswith("_"):
            methods.append(FuncInfo(
                name=item.name,
                signature=_format_func_sig(item, skip_self=True),
                parsed_doc=_parse_google_docstring(ast.get_docstring(item) or ""),
                is_async=isinstance(item, ast.AsyncFunctionDef),
                is_property=_has_decorator(item, "property"),
            ))
    return ClassInfo(
        name=node.name, bases=bases, doc=_get_summary(node),
        init_sig=init_sig, init_doc=init_doc, methods=methods,
    )


def _extract_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FuncInfo:
    return FuncInfo(
        name=node.name,
        signature=_format_func_sig(node),
        parsed_doc=_parse_google_docstring(ast.get_docstring(node) or ""),
        is_async=isinstance(node, ast.AsyncFunctionDef),
    )


def _parse_all_list(tree: ast.Module) -> list[str] | None:
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        return [
                            e.value for e in node.value.elts
                            if isinstance(e, ast.Constant) and isinstance(e.value, str)
                        ]
    return None


def _parse_imports(tree: ast.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                result[alias.asname or alias.name] = node.module
    return result


def _module_to_path(dotted: str) -> Path:
    parts = dotted.split(".")
    path = KISS_SRC.parent.joinpath(*parts)
    if path.is_dir():
        return path / "__init__.py"
    return path.with_suffix(".py")


def _file_to_module(path: Path) -> str:
    rel = path.relative_to(KISS_SRC.parent)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def _should_skip(path: Path) -> bool:
    rel = path.relative_to(KISS_SRC)
    return any(p in EXCLUDE_DIRS or p in EXCLUDE_FILES or p.startswith(".") for p in rel.parts)


def _find_def_in_file(path: Path, name: str) -> ClassInfo | FuncInfo | None:
    if not path.exists():
        return None
    tree = ast.parse(path.read_text())
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return _extract_class(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return _extract_function(node)
    return None


SKIP_FUNCTIONS = {"main"}


def _extract_public_from_file(path: Path) -> tuple[list[ClassInfo], list[FuncInfo]]:
    tree = ast.parse(path.read_text())
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
    """Walk the package tree and collect all public API modules."""
    modules: list[ModuleDoc] = []
    documented_per_file: dict[Path, set[str]] = {}

    for init_path in sorted(KISS_SRC.rglob("__init__.py")):
        if _should_skip(init_path):
            continue
        module_name = _file_to_module(init_path)
        source = init_path.read_text()
        tree = ast.parse(source)
        all_list = _parse_all_list(tree)
        doc = _get_summary(tree)
        deprecated = "deprecated" in source[:500].lower()
        if deprecated:
            continue

        imports = _parse_imports(tree)
        classes: list[ClassInfo] = []
        functions: list[FuncInfo] = []

        for name in all_list or []:
            defn: ClassInfo | FuncInfo | None = None
            source_path: Path = init_path
            if name in imports:
                source_path = _module_to_path(imports[name])
                defn = _find_def_in_file(source_path, name)
            if defn is None:
                source_path = init_path
                defn = _find_def_in_file(init_path, name)
            if defn is None:
                continue
            documented_per_file.setdefault(source_path, set()).add(name)
            if isinstance(defn, ClassInfo):
                classes.append(defn)
            else:
                functions.append(defn)

        modules.append(ModuleDoc(
            name=module_name, doc=doc, all_exports=all_list,
            classes=classes, functions=functions, is_package=True,
            deprecated=deprecated,
        ))

    for py_file in sorted(KISS_SRC.rglob("*.py")):
        if py_file.name == "__init__.py" or _should_skip(py_file):
            continue
        module_name = _file_to_module(py_file)
        classes, functions = _extract_public_from_file(py_file)
        already = documented_per_file.get(py_file, set())
        classes = [c for c in classes if c.name not in already]
        functions = [f for f in functions if f.name not in already]
        if not classes and not functions:
            continue
        doc = _get_summary(ast.parse(py_file.read_text()))
        modules.append(ModuleDoc(
            name=module_name, doc=doc, all_exports=None,
            classes=classes, functions=functions,
        ))

    return _sort_modules(modules)


def _sort_modules(modules: list[ModuleDoc]) -> list[ModuleDoc]:
    order = [
        "kiss", "kiss.core", "kiss.core.kiss_agent", "kiss.core.base",
        "kiss.core.config", "kiss.core.config_builder",
        "kiss.core.models", "kiss.core.models.model", "kiss.core.models.model_info",
        "kiss.core.models.openai_compatible_model", "kiss.core.models.anthropic_model",
        "kiss.core.models.gemini_model",
        "kiss.core.printer", "kiss.core.print_to_console",
        "kiss.agents.assistant.browser_ui", "kiss.agents.assistant.useful_tools",
        "kiss.agents.assistant.web_use_tool",
        "kiss.core.utils", "kiss.core.kiss_error",
        "kiss.agents", "kiss.agents.kiss",
        "kiss.agents.coding_agents",
        "kiss.agents.coding_agents.relentless_coding_agent",
        "kiss.agents.coding_agents.repo_agent", "kiss.agents.coding_agents.repo_optimizer",
        "kiss.agents.coding_agents.agent_optimizer", "kiss.agents.coding_agents.config",
        "kiss.agents.assistant", "kiss.agents.assistant.relentless_agent",
        "kiss.agents.assistant.assistant_agent", "kiss.agents.assistant.assistant",
        "kiss.agents.assistant.config",
        "kiss.agents.gepa", "kiss.agents.gepa.gepa", "kiss.agents.gepa.config",
        "kiss.agents.kiss_evolve", "kiss.agents.kiss_evolve.kiss_evolve",
        "kiss.agents.kiss_evolve.config",
        "kiss.agents.imo_agent", "kiss.agents.imo_agent.imo_agent",
        "kiss.agents.imo_agent.config",
        "kiss.docker", "kiss.docker.docker_manager",
    ]
    rank = {name: i for i, name in enumerate(order)}

    def key(m: ModuleDoc) -> tuple[int, str]:
        return (rank.get(m.name, 999), m.name)

    return sorted(modules, key=key)


def _slug(text: str) -> str:
    return text.replace(".", "").replace(" ", "-").lower()


def _heading_depth(module_name: str) -> int:
    depth = module_name.count(".")
    return min(depth + 2, 4)


def generate_markdown(modules: list[ModuleDoc]) -> str:
    lines: list[str] = []

    lines.append("# KISS Framework API Reference\n")
    lines.append("> **Auto-generated** — run `uv run generate-api-docs` to regenerate.\n")

    lines.append("<details><summary><b>Table of Contents</b></summary>\n")
    for mod in modules:
        indent = "  " * mod.name.count(".")
        lines.append(f"{indent}- [`{mod.name}`](#{_slug(mod.name)})")
    lines.append("\n</details>\n")
    lines.append("---\n")

    for mod in modules:
        h = "#" * _heading_depth(mod.name)
        doc_part = f" — *{mod.doc}*" if mod.doc else ""
        lines.append(f"{h} `{mod.name}`{doc_part}\n")

        if mod.is_package and mod.all_exports:
            exports = ", ".join(mod.all_exports)
            lines.append(f"```python\nfrom {mod.name} import {exports}\n```\n")

        for cls in mod.classes:
            _render_class(lines, cls, _heading_depth(mod.name) + 1)

        for func in mod.functions:
            _render_function(lines, func)

        lines.append("---\n")

    return "\n".join(lines)


def _render_class(lines: list[str], cls: ClassInfo, depth: int) -> None:
    h = "#" * min(depth, 6)
    bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
    doc_part = f" — {cls.doc}" if cls.doc else ""
    lines.append(f"{h} `class {cls.name}{bases_str}`{doc_part}\n")
    if cls.init_sig:
        lines.append(f"**Constructor:** `{cls.name}{cls.init_sig}`\n")
        _render_args_returns(lines, cls.init_doc)
    if cls.methods:
        for m in cls.methods:
            _render_method(lines, m)


def _render_method(lines: list[str], func: FuncInfo) -> None:
    prefix = "async " if func.is_async else ""
    prop_tag = " *(property)*" if func.is_property else ""
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
    if doc.args:
        for name, desc in doc.args:
            lines.append(f"{indent}- `{name}`: {desc}")
    if doc.returns:
        lines.append(f"{indent}- **Returns:** {doc.returns}")
    if doc.args or doc.returns:
        lines.append("")


def main() -> int:
    modules = discover_modules()
    markdown = generate_markdown(modules)
    OUTPUT.write_text(markdown)
    subprocess.run(["uv", "run", "mdformat", str(OUTPUT)], check=True)
    print(f"Generated {OUTPUT.relative_to(PROJECT_ROOT)} ({len(modules)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
