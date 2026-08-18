# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared fixtures for model tests.

The Claude Code and Codex tests mock subprocess so they never actually
invoke the underlying CLI binaries, but ``_build_cli_args`` still calls
the locator functions and raises ``KISSError`` when the binary is not
installed.  In CI / on dev machines without the CLIs, this blocks the
streaming-, flag-, and fallback-parsing tests from even constructing a
command line.  The autouse fixture below patches the locators to return
a deterministic fake path.

Direct tests of the locator functions (e.g. ``test_find_claude_cli_missing``)
call the function via its imported name (``from ... import _find_claude_cli``)
which is bound at import time and is therefore unaffected by patches on
the module attribute, so they continue to exercise the real lookup logic.
"""

from __future__ import annotations

# The fixture body lives in kiss.tests.cli_locator_stub so test modules
# relocated out of this directory by the packaging invariants can import
# the same autouse fixture per-module.  Importing it here registers it
# for every test in this directory, exactly as before.
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
