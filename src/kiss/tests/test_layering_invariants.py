# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Enforce the package layering invariants.

Two invariants (user-specified) MUST always hold:

1. Code in ``src/kiss/core/`` MUST NOT depend on any code outside
   ``src/kiss/core/``.
2. Code in ``src/kiss/agents/sorcar/`` MUST NOT depend on any code
   outside ``src/kiss/agents/sorcar/`` except code in
   ``src/kiss/core/``.

"Depend on" means *any* import of a first-party ``kiss`` module —
top-level, lazy (inside a function), or conditional — so this test
walks the full AST of every Python file in the two layers and resolves
every ``import`` / ``from ... import`` statement (including relative
imports) to an absolute ``kiss.*`` module path.  Third-party and
standard-library imports are unconstrained.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
KISS_ROOT = SRC_ROOT / "kiss"


def _module_name_for(path: Path) -> str:
    """Return the absolute dotted module name for a source file.

    Args:
        path: Path to a ``.py`` file under ``src/``.

    Returns:
        Dotted module path, e.g. ``kiss.core.useful_tools``.
    """
    rel = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(module_name: str, is_package: bool, node: ast.ImportFrom) -> str:
    """Resolve a relative ``from ... import`` to an absolute module path.

    Args:
        module_name: Absolute dotted name of the importing module.
        is_package: Whether the importing module is a package
            ``__init__``.
        node: The ``ast.ImportFrom`` node (``node.level`` > 0).

    Returns:
        The absolute dotted module path being imported from.
    """
    parts = module_name.split(".")
    # For a plain module, level 1 refers to its parent package; for a
    # package __init__, level 1 refers to the package itself.
    base = parts[: len(parts) - node.level + (1 if is_package else 0)]
    if node.module:
        base = [*base, *node.module.split(".")]
    return ".".join(base)


def _dynamic_import_target(node: ast.Call) -> str | None:
    """Return the literal module path of a dynamic import call, if any.

    Recognises ``importlib.import_module("...")``, ``import_module("...")``
    and ``__import__("...")`` with a string-literal first argument.
    Non-literal arguments (e.g. a dict lookup) cannot be resolved
    statically and are ignored — the runtime alias tests and the
    static-literal check together keep the invariant honest.

    Args:
        node: The ``ast.Call`` node to inspect.

    Returns:
        The literal module path, or ``None`` when the call is not a
        recognised dynamic import with a string-literal argument.
    """
    func_name = ""
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        func_name = node.func.attr
    if func_name not in ("import_module", "__import__"):
        return None
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _kiss_imports(path: Path) -> list[tuple[int, str]]:
    """Collect every first-party ``kiss.*`` import in a file.

    Args:
        path: Path to the ``.py`` file to scan.

    Returns:
        List of ``(lineno, absolute_module_path)`` for every imported
        ``kiss`` module (lazy, top-level, and literal dynamic imports
        via ``importlib.import_module`` / ``__import__`` alike).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_name = _module_name_for(path)
    is_package = path.name == "__init__.py"
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "kiss" or alias.name.startswith("kiss."):
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                target = _resolve_relative(module_name, is_package, node)
            else:
                target = node.module or ""
            if target == "kiss" or target.startswith("kiss."):
                found.append((node.lineno, target))
        elif isinstance(node, ast.Call):
            dyn = _dynamic_import_target(node)
            if dyn is not None and (dyn == "kiss" or dyn.startswith("kiss.")):
                found.append((node.lineno, dyn))
    return found


def _violations(layer_dir: Path, allowed_prefixes: tuple[str, ...]) -> list[str]:
    """Return human-readable layering violations for a directory.

    Args:
        layer_dir: Directory whose ``.py`` files are scanned.
        allowed_prefixes: Dotted module prefixes that files in the
            layer may import (any other ``kiss.*`` import is a
            violation).

    Returns:
        List of ``"file:line imports module"`` violation strings.
    """
    bad: list[str] = []
    for path in sorted(layer_dir.rglob("*.py")):
        for lineno, target in _kiss_imports(path):
            ok = any(
                target == prefix or target.startswith(prefix + ".")
                for prefix in allowed_prefixes
            )
            if not ok:
                rel = path.relative_to(SRC_ROOT.parent)
                bad.append(f"{rel}:{lineno} imports {target}")
    return bad


def test_core_depends_only_on_core() -> None:
    """``kiss.core`` code must not import any ``kiss`` module outside core.

    Note ``import kiss`` (the bare package) is also forbidden: it
    executes ``kiss/__init__.py``, which lives outside ``kiss/core``.
    """
    violations = _violations(KISS_ROOT / "core", ("kiss.core",))
    assert not violations, (
        "kiss.core must not depend on code outside src/kiss/core/:\n"
        + "\n".join(violations)
    )


def test_sorcar_depends_only_on_sorcar_and_core() -> None:
    """Sorcar code must import only ``kiss.core`` and ``kiss.agents.sorcar``."""
    violations = _violations(
        KISS_ROOT / "agents" / "sorcar",
        ("kiss.core", "kiss.agents.sorcar"),
    )
    assert not violations, (
        "kiss.agents.sorcar must not depend on code outside "
        "src/kiss/agents/sorcar/ and src/kiss/core/:\n"
        + "\n".join(violations)
    )
