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

from types import ModuleType

import pytest


@pytest.fixture(autouse=True)
def _stub_cli_locators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub Claude Code / Codex binary lookups for offline test runs."""
    cc_mod: ModuleType | None
    try:
        import kiss.core.models.claude_code_model as cc_mod
    except ImportError:
        pass
    else:
        monkeypatch.setattr(
            cc_mod, "_find_claude_cli", lambda: "/usr/bin/claude", raising=False,
        )

    cx_mod: ModuleType | None
    try:
        import kiss.core.models.codex_model as cx_mod
    except ImportError:
        pass
    else:
        monkeypatch.setattr(
            cx_mod, "_find_codex_cli", lambda: "/usr/bin/codex", raising=False,
        )
