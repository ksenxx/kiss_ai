# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Reusable CLI-locator stub for Claude Code / Codex model tests.

``tests/core/models/conftest.py`` applies this stub to every test in
that directory.  Tests relocated OUT of that directory (they only
depend on ``kiss.core`` / other layers, so the packaging invariants
place them elsewhere) still need it: they mock the CLI subprocess, but
``_build_cli_args`` calls the locator functions and raises ``KISSError``
on machines without the real binaries.

Such a test module opts back in with::

    from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401

which registers the autouse fixture for that module only.  Tests marked
``@pytest.mark.live_cli`` opt out: they spawn the real CLI, so a fake
``/usr/bin/...`` path would break them with ``FileNotFoundError``.
"""

from __future__ import annotations

from types import ModuleType

import pytest


@pytest.fixture(autouse=True)
def stub_cli_locators(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub Claude Code / Codex binary lookups for offline test runs.

    Mirrors the autouse fixture in ``tests/core/models/conftest.py``;
    see the module docstring for why relocated test modules import it.
    Direct tests of the locator functions call them via names bound at
    import time and are therefore unaffected by these module-attribute
    patches.
    """
    if request.node.get_closest_marker("live_cli") is not None:
        return
    cc_mod: ModuleType | None
    try:
        import kiss.core.models.claude_code_model as cc_mod
    except ImportError:
        pass
    else:
        monkeypatch.setattr(
            cc_mod, "_find_claude_cli", lambda: "/usr/bin/claude",
            raising=False,
        )

    cx_mod: ModuleType | None
    try:
        import kiss.core.models.codex_model as cx_mod
    except ImportError:
        pass
    else:
        monkeypatch.setattr(
            cx_mod, "_find_codex_cli", lambda: "/usr/bin/codex",
            raising=False,
        )
