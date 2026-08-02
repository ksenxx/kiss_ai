# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end proof that ``kiss.core`` does not depend on ``kiss.agents.sorcar``.

``relentless_agent``, ``useful_tools``, ``docker_tools`` and
``docker_manager`` used to live in ``src/kiss/core/``.  They are agent
*machinery*, not framework primitives, and three of them are only ever
consumed by :class:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent`, so
they now live in ``src/kiss/agents/sorcar/``.

The regression this file guards against is the reverse dependency: if
any ``kiss.core`` module reaches back into ``kiss.agents.sorcar`` (or
anywhere else outside core), the core layer stops being a standalone,
importable foundation.  A purely static AST check is not enough — a
lazy import buried in an ``except`` branch only bites at runtime — so
these tests attack the invariant from three angles:

* the four modules really moved (and left nothing behind);
* every ``kiss.core`` module is importable in a *fresh interpreter*
  where importing ``kiss.agents`` is made to explode; and
* the whole core package still imports and does real work with the
  ``kiss.agents`` package tree deleted from the import system.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
KISS_ROOT = SRC_ROOT / "kiss"
CORE_DIR = KISS_ROOT / "core"
SORCAR_DIR = KISS_ROOT / "agents" / "sorcar"

#: Modules relocated out of ``kiss.core`` and into ``kiss.agents.sorcar``.
MOVED_MODULES = (
    "relentless_agent",
    "useful_tools",
    "docker_tools",
    "docker_manager",
)


def _core_module_names() -> list[str]:
    """Return the dotted names of every importable ``kiss.core`` module.

    Returns:
        Sorted dotted module paths such as ``kiss.core.printer``,
        including subpackages like ``kiss.core.models.model``.
    """
    names = []
    for path in sorted(CORE_DIR.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return sorted(set(names))


def _run_isolated(*fragments: str) -> subprocess.CompletedProcess[str]:
    """Execute the concatenated *fragments* in a fresh interpreter.

    Each fragment is dedented independently, so callers can mix a
    flush-left constant with an indented triple-quoted literal.

    Args:
        fragments: Python source chunks, joined with newlines.

    Returns:
        The completed process, with stdout/stderr captured as text.
    """
    source = "\n".join(textwrap.dedent(f) for f in fragments)
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(SRC_ROOT),
    )


# A meta-path hook that makes any ``import kiss.agents...`` raise loudly.
# Installed before ``kiss.core`` is imported so that even a lazy import
# inside a rarely taken branch is caught the moment it fires.
_BLOCK_AGENTS = """
import sys


class _AgentsBlocker:
    def find_module(self, fullname, path=None):
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "kiss.agents" or fullname.startswith("kiss.agents."):
            raise AssertionError(
                "kiss.core must not import " + fullname
            )
        return None


sys.meta_path.insert(0, _AgentsBlocker())
"""


def test_moved_modules_live_in_sorcar_only() -> None:
    """The four modules exist under sorcar and no longer under core."""
    missing = [m for m in MOVED_MODULES if not (SORCAR_DIR / f"{m}.py").is_file()]
    assert not missing, (
        "expected these modules in src/kiss/agents/sorcar/: " + ", ".join(missing)
    )

    left_behind = [m for m in MOVED_MODULES if (CORE_DIR / f"{m}.py").exists()]
    assert not left_behind, (
        "these modules must not remain in src/kiss/core/: " + ", ".join(left_behind)
    )


def test_no_core_source_mentions_the_moved_modules() -> None:
    """No ``kiss.core`` file may reference the old module paths.

    Catches half-finished moves where a stale ``kiss.core.useful_tools``
    string survives in a lazy import, a docstring reference, or an
    ``importlib.import_module`` call.
    """
    stale: list[str] = []
    for path in sorted(CORE_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for module in MOVED_MODULES:
                if f"kiss.core.{module}" in line or f"core/{module}.py" in line:
                    rel = path.relative_to(SRC_ROOT.parent)
                    stale.append(f"{rel}:{lineno}: {line.strip()}")
    assert not stale, "stale references to moved modules:\n" + "\n".join(stale)


def test_every_core_module_imports_without_touching_agents() -> None:
    """Import each ``kiss.core`` module with ``kiss.agents`` booby-trapped.

    A single subprocess imports every core module in turn, so a lazy
    ``from kiss.agents.sorcar...`` executed at import time anywhere in
    the layer fails the test with the offending module name.
    """
    modules = _core_module_names()
    assert "kiss.core.printer" in modules, "core module discovery is broken"

    result = _run_isolated(
        _BLOCK_AGENTS,
        f"""
        import importlib

        for name in {modules!r}:
            importlib.import_module(name)
        print("imported", len({modules!r}), "core modules")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "core modules" in result.stdout


def test_core_agent_runs_with_agents_package_absent() -> None:
    """``kiss.core`` must work when ``kiss.agents`` cannot be imported.

    This is the behavioural half of the invariant: build a real
    :class:`~kiss.core.kiss_agent.KISSAgent`, register a tool and
    execute it, all while any attempt to reach ``kiss.agents`` raises.
    """
    result = _run_isolated(
        _BLOCK_AGENTS,
        """
        from kiss.core.kiss_agent import KISSAgent
        from kiss.core.utils import finish

        def add(a: int, b: int) -> int:
            \"\"\"Add two integers.

            Args:
                a: First addend.
                b: Second addend.

            Returns:
                The sum of *a* and *b*.
            \"\"\"
            return a + b

        agent = KISSAgent("layer-check")
        agent.is_agentic = True
        agent.printer = None
        agent.function_map = {}
        agent.model = None
        agent._add_functions([add, finish])

        name, response = agent._execute_tool(
            {"name": "add", "arguments": {"a": 2, "b": 3}}
        )
        print("TOOLS", sorted(agent.function_map))
        print("CALLED", name, response)
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "CALLED add 5" in result.stdout
    assert "'add'" in result.stdout
    assert "'finish'" in result.stdout


def test_sorcar_still_owns_the_moved_modules_at_runtime() -> None:
    """The relocated classes are importable from their new home."""
    result = _run_isolated(
        """
        from kiss.agents.sorcar.docker_manager import DockerManager
        from kiss.agents.sorcar.docker_tools import DockerTools
        from kiss.agents.sorcar.relentless_agent import RelentlessAgent
        from kiss.agents.sorcar.useful_tools import UsefulTools

        for cls in (RelentlessAgent, UsefulTools, DockerTools, DockerManager):
            print(cls.__module__)
        """
    )
    assert result.returncode == 0, result.stderr
    for module in MOVED_MODULES:
        assert f"kiss.agents.sorcar.{module}" in result.stdout


def test_sorcar_agent_wires_up_the_relocated_tools() -> None:
    """``SorcarAgent`` still inherits and instantiates the moved code.

    Guards against a move that satisfies the layering rule but silently
    breaks the wiring — e.g. an import rewritten to a module that no
    longer defines the symbol.
    """
    result = _run_isolated(
        """
        from kiss.agents.sorcar.relentless_agent import RelentlessAgent
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent
        from kiss.agents.sorcar.useful_tools import UsefulTools

        assert issubclass(SorcarAgent, RelentlessAgent), "broken inheritance"
        tools = UsefulTools()
        print("TOOLNAMES", tools.Read.__name__, tools.Bash.__name__)
        print("MRO_OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "MRO_OK" in result.stdout
    assert "TOOLNAMES Read Bash" in result.stdout


def test_relentless_agent_lazy_imports_resolve() -> None:
    """The lazy imports inside ``relentless_agent`` point at real modules.

    ``relentless_agent`` imports ``UsefulTools`` and ``DockerManager``
    inside function bodies, so a stale path there survives every static
    check and only fails in production.  Resolve them for real.
    """
    source = (SORCAR_DIR / "relentless_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="relentless_agent.py")

    lazy: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.col_offset > 0:
            module = node.module or ""
            if module.startswith("kiss."):
                lazy.append((module, tuple(a.name for a in node.names)))

    assert lazy, "expected relentless_agent to keep its lazy kiss imports"

    checks = "\n".join(
        f"from {module} import {', '.join(names)}" for module, names in lazy
    )
    result = _run_isolated(checks, "print('LAZY_OK')")
    assert result.returncode == 0, result.stderr
    assert "LAZY_OK" in result.stdout


def test_get_tools_serves_file_tools_from_sorcar() -> None:
    """``SorcarAgent._get_tools()`` must hand back the relocated tools.

    This is the real wiring check: the previous tests prove the modules
    *moved*, this one proves the agent still *uses* them.  Both branches
    of ``_get_tools`` are exercised, because the docker branch reaches
    ``docker_tools`` through a lazy in-function import that no static
    scan can validate.  Every returned file tool must be a bound method
    of a class defined under ``kiss.agents.sorcar``.
    """
    from kiss.agents.sorcar.docker_tools import DockerTools
    from kiss.agents.sorcar.sorcar_agent import SorcarAgent
    from kiss.agents.sorcar.useful_tools import UsefulTools

    file_tools = ("Read", "Edit", "Write")

    agent = SorcarAgent("layer-wiring")
    # Keep the browser out of it; WebUseTool would spawn Chromium.
    agent._use_web_tools = False

    local_tools = {t.__name__: t for t in agent._get_tools()}
    for name in file_tools:
        owner = getattr(local_tools[name], "__self__", None)
        assert isinstance(owner, UsefulTools), (
            f"local {name} is not served by kiss.agents.sorcar.useful_tools"
        )
    # Bash is a thin wrapper around ``UsefulTools.Bash``; it keeps the
    # wrapped function's module, which must still be inside sorcar.
    assert local_tools["Bash"].__module__.startswith("kiss.agents.sorcar.")

    # Any truthy docker_manager flips _get_tools onto the docker branch;
    # DockerTools only stores the bash callable, so no daemon is needed.
    agent.docker_manager = object()
    docker_tools = {t.__name__: t for t in agent._get_tools()}
    for name in file_tools:
        owner = getattr(docker_tools[name], "__self__", None)
        assert isinstance(owner, DockerTools), (
            f"docker {name} is not served by kiss.agents.sorcar.docker_tools"
        )

    for registry in (local_tools, docker_tools):
        for name in file_tools:
            owner = registry[name].__self__  # type: ignore[attr-defined]
            assert type(owner).__module__.startswith("kiss.agents.sorcar."), (
                f"{name} escaped the sorcar layer: {type(owner).__module__}"
            )
