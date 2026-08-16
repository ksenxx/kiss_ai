# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end proof that ``kiss.core`` does not depend on ``kiss.agents.sorcar``.

Core-only tests (they scan, import, and exercise nothing but
``src/kiss/core``) moved here from the original root-level module; the
sorcar-side tests of the same invariant live in
``kiss.tests.agents.sorcar.test_core_sorcar_layer_separation`` and
import the shared helpers back from this module.

The regression these tests guard against is the reverse dependency: if
any ``kiss.core`` module reaches back into ``kiss.agents.sorcar`` (or
anywhere else outside core), the core layer stops being a standalone,
importable foundation.  A purely static AST check is not enough — a
lazy import buried in an ``except`` branch only bites at runtime — so
the invariant is attacked from three angles:

* no core source file references the modules that moved out of core;
* every ``kiss.core`` module is importable in a *fresh interpreter*
  where importing ``kiss.agents`` is made to explode; and
* the whole core package still imports and does real work with the
  ``kiss.agents`` package tree deleted from the import system.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import kiss

# Located via the installed ``kiss`` package rather than ``__file__`` so
# the paths stay correct no matter where this test module lives.
SRC_ROOT = Path(kiss.__file__).resolve().parents[1]
KISS_ROOT = SRC_ROOT / "kiss"
CORE_DIR = KISS_ROOT / "core"

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
