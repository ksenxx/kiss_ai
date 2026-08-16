# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sorcar-side checks that the core/sorcar layer split stayed intact.

``relentless_agent``, ``useful_tools``, ``docker_tools`` and
``docker_manager`` used to live in ``src/kiss/core/``.  They are agent
*machinery*, not framework primitives, and three of them are only ever
consumed by :class:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent`, so
they now live in ``src/kiss/agents/sorcar/``.

The core-only half of this invariant (no core source mentions the moved
modules; every core module imports and works with ``kiss.agents``
blocked) lives in
``kiss.tests.core.test_core_sorcar_layer_separation``, which also owns
the shared helpers imported below.  The tests here exercise the sorcar
side: the moved modules exist in their new home, resolve at runtime,
and ``SorcarAgent`` still wires them up.
"""

from __future__ import annotations

import ast

from kiss.tests.core.test_core_sorcar_layer_separation import (
    CORE_DIR,
    KISS_ROOT,
    MOVED_MODULES,
    _run_isolated,
)

SORCAR_DIR = KISS_ROOT / "agents" / "sorcar"


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
