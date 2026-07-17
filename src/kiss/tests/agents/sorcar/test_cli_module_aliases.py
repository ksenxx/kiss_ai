# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end guards for the ``kiss.ui.cli`` relocation contract.

The sorcar interactive CLI implementation lives in :mod:`kiss.ui.cli`.
The old ``kiss.agents.sorcar.cli_*`` / ``mcp_cli`` module paths were
removed outright (no backward-compatibility aliases).  These tests pin
the two promises the relocation exists to provide:

1. the old ``kiss.agents.sorcar`` paths for the moved modules no
   longer exist (stale imports fail loudly instead of silently
   resolving);
2. importing the agent modules does not import the interactive UI
   package (the agents are decoupled from the CLI front end).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

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


def test_old_agent_paths_are_gone_and_new_paths_import() -> None:
    """Old kiss.agents.sorcar paths fail; kiss.ui.cli paths import."""
    import importlib

    for name in MOVED_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"kiss.agents.sorcar.{name}")
        importlib.import_module(f"kiss.ui.cli.{name}")


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
