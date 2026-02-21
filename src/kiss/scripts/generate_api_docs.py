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
class FuncInfo:
    name: str
    signature: str
    doc: str
    is_async: bool = False
    is_property: bool = False


@dataclass
class ClassInfo:
    name: str
    bases: list[str]
    doc: str
    init_sig: str = ""
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


def _get_docstring(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        lines = node.body[0].value.value.strip().split("\n")
        return lines[0].strip()
    return ""


def _has_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == name) or
        (isinstance(d, ast.Attribute) and d.attr == name)
        for d in node.decorator_list
    )


def _extract_class(node: ast.ClassDef) -> ClassInfo:
    bases = [ast.unparse(b) for b in node.bases]
    init_sig = ""
    methods: list[FuncInfo] = []
    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name == "__init__":
            init_sig = _format_func_sig(item, skip_self=True)
        elif not item.name.startswith("_"):
            methods.append(FuncInfo(
                name=item.name,
                signature=_format_func_sig(item, skip_self=True),
                doc=_get_docstring(item),
                is_async=isinstance(item, ast.AsyncFunctionDef),
                is_property=_has_decorator(item, "property"),
            ))
    return ClassInfo(
        name=node.name, bases=bases, doc=_get_docstring(node),
        init_sig=init_sig, methods=methods,
    )


def _extract_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FuncInfo:
    return FuncInfo(
        name=node.name,
        signature=_format_func_sig(node),
        doc=_get_docstring(node),
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
        doc = _get_docstring(tree)
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
        doc = _get_docstring(ast.parse(py_file.read_text()))
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
        "kiss.core.printer", "kiss.core.print_to_console", "kiss.core.print_to_browser",
        "kiss.core.browser_ui", "kiss.core.useful_tools", "kiss.core.web_use_tool",
        "kiss.core.utils", "kiss.core.kiss_error",
        "kiss.agents", "kiss.agents.kiss",
        "kiss.agents.coding_agents", "kiss.agents.coding_agents.claude_coding_agent",
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
        "kiss.multiprocessing", "kiss.multiprocessing.multiprocess",
        "kiss.rag", "kiss.rag.simple_rag",
    ]
    rank = {name: i for i, name in enumerate(order)}

    def key(m: ModuleDoc) -> tuple[int, str]:
        return (rank.get(m.name, 999), m.name)

    return sorted(modules, key=key)


def _slug(text: str) -> str:
    return text.replace(".", "").replace("_", "-").replace(" ", "-").lower()


def _escape_pipe(text: str) -> str:
    return text.replace("|", "\\|")


def _heading_depth(module_name: str) -> int:
    depth = module_name.count(".")
    return min(depth + 2, 4)


def generate_markdown(modules: list[ModuleDoc]) -> str:
    lines: list[str] = []

    lines.append("# KISS Framework API Reference\n")
    lines.append("> **Auto-generated** from source code by `generate_api_docs.py`.")
    lines.append("> Run `uv run generate-api-docs` to regenerate.\n")
    lines.append("---\n")

    lines.append("## Table of Contents\n")
    for mod in modules:
        indent = "  " * mod.name.count(".")
        lines.append(f"{indent}- [`{mod.name}`](#{_slug(mod.name)})")
    lines.append("\n---\n")

    for mod in modules:
        h = "#" * _heading_depth(mod.name)
        lines.append(f"{h} `{mod.name}`\n")
        if mod.doc:
            lines.append(f"*{mod.doc}*\n")

        if mod.is_package and mod.all_exports:
            exports = ", ".join(mod.all_exports)
            lines.append(f"```python\nfrom {mod.name} import {exports}\n```\n")

        for cls in mod.classes:
            _render_class(lines, cls, _heading_depth(mod.name) + 1)

        if mod.functions:
            for func in mod.functions:
                _render_function(lines, func)

        lines.append("---\n")

    return "\n".join(lines)


def _render_class(lines: list[str], cls: ClassInfo, depth: int) -> None:
    h = "#" * min(depth, 6)
    bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
    lines.append(f"{h} `{cls.name}`\n")
    lines.append(f"```python\nclass {cls.name}{bases_str}\n```\n")
    if cls.doc:
        lines.append(f"{cls.doc}\n")
    if cls.init_sig:
        lines.append("**Constructor:**\n")
        lines.append(f"```python\n{cls.name}{cls.init_sig}\n```\n")
    if cls.methods:
        lines.append("**Methods:**\n")
        lines.append("| Method | Signature | Description |")
        lines.append("|--------|-----------|-------------|")
        for m in cls.methods:
            prefix = "async " if m.is_async else ""
            suffix = " *(property)*" if m.is_property else ""
            sig = _escape_pipe(f"{prefix}{m.name}{m.signature}")
            lines.append(f"| `{m.name}` | `{sig}`{suffix} | {_escape_pipe(m.doc)} |")
        lines.append("")


def _render_function(lines: list[str], func: FuncInfo) -> None:
    prefix = "async " if func.is_async else ""
    lines.append(f"**`{func.name}`**\n")
    lines.append(f"```python\n{prefix}def {func.name}{func.signature}\n```\n")
    if func.doc:
        lines.append(f"{func.doc}\n")


def main() -> int:
    modules = discover_modules()
    markdown = generate_markdown(modules)
    OUTPUT.write_text(markdown)
    subprocess.run(["uv", "run", "mdformat", str(OUTPUT)], check=True)
    print(f"Generated {OUTPUT.relative_to(PROJECT_ROOT)} ({len(modules)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
