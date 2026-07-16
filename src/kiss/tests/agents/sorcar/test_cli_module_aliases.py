# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end guards for the ``kiss.ui.cli`` relocation contract.

The sorcar interactive CLI implementation lives in :mod:`kiss.ui.cli`;
each old ``kiss.agents.sorcar.cli_*`` / ``mcp_cli`` module is a
backward-compatibility alias that replaces itself in ``sys.modules``
with the real module.  These tests pin the three promises the rest of
the suite (and external callers) rely on:

1. importing an old path and its new path yields the SAME module
   object, in a fresh process, in either import order;
2. a monkeypatch applied through one path is visible through the
   other (single shared module namespace);
3. importing the agent modules does not import the interactive UI
   package (the decoupling the relocation exists to provide).
"""

from __future__ import annotations

import importlib
import subprocess
import sys

MOVED_MODULES = [
    "cli_client",
    "cli_daemon_bridge",
    "cli_line_continuation",
    "cli_panel",
    "cli_printer",
    "cli_prompt",
    "cli_repl",
    "cli_steering",
    "cli_talk",
    "cli_voice",
    "mcp_cli",
]


def test_alias_identity_and_monkeypatch_propagation() -> None:
    """Old and new paths are one module; patches propagate both ways."""
    for name in MOVED_MODULES:
        old = importlib.import_module(f"kiss.agents.sorcar.{name}")
        new = importlib.import_module(f"kiss.ui.cli.{name}")
        assert old is new, f"alias broken for {name}"
        sentinel = object()
        try:
            setattr(old, "_alias_probe", sentinel)
            assert getattr(new, "_alias_probe") is sentinel, name
        finally:
            delattr(new, "_alias_probe")


def test_fresh_process_alias_identity_both_import_orders() -> None:
    """Both import orders resolve to one module object in a fresh process."""
    checks = "\n".join(
        f"import kiss.agents.sorcar.{m} as o_{m}\n"
        f"import kiss.ui.cli.{m} as n_{m}\n"
        f"assert o_{m} is n_{m}, '{m}'"
        for m in MOVED_MODULES
    )
    reversed_checks = "\n".join(
        f"import kiss.ui.cli.{m} as n_{m}\n"
        f"import kiss.agents.sorcar.{m} as o_{m}\n"
        f"assert o_{m} is n_{m}, '{m}'"
        for m in MOVED_MODULES
    )
    for code in (checks, reversed_checks):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr


def test_agent_imports_do_not_load_interactive_ui() -> None:
    """Importing the three agents never imports the kiss.ui package."""
    code = (
        "import sys\n"
        "import kiss.agents.sorcar.sorcar_agent\n"
        "import kiss.agents.sorcar.chat_sorcar_agent\n"
        "import kiss.agents.sorcar.worktree_sorcar_agent\n"
        "loaded = [m for m in sys.modules if m.startswith('kiss.ui')]\n"
        "assert not loaded, f'agents imported UI modules: {loaded}'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
