# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `kiss_home` module fixture is imported from
#   kiss.tests.server.test_inject_panel_order and is intentionally
#   shadowed by test parameters of the same name)
"""Inject-panel tests that read the REAL bundled ``src/kiss/INJECTIONS.md``.

Split out of ``kiss.tests.server.test_inject_panel_order``: these two
tests exercise the production fallback to the bundled package data file,
which lives outside the kiss.core/kiss.agents.sorcar/kiss.server layers,
so they keep a vscode-side home while the hermetic (env-pinned) tests
moved to tests/server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.server.tricks import MY_INJECTION_DEFAULT_BODY, read_tricks
from kiss.tests.server.test_inject_panel_order import kiss_home  # noqa: F401


def test_no_injections_path_falls_back_to_bundled_package_file(
    kiss_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``KISS_INJECTIONS_PATH``, the real bundled file is used.

    Locks in the production wiring: the test merely verifies that
    *some* bundled tricks are returned when the env override is
    cleared, AND that the user-side MY_INJECTION default is still
    seeded and listed first.
    """
    monkeypatch.delenv("KISS_INJECTIONS_PATH", raising=False)
    tricks = read_tricks()
    assert tricks[0] == MY_INJECTION_DEFAULT_BODY
    assert len(tricks) >= 2


def test_default_trick_renders_at_least_once_on_fresh_install(
    kiss_home: Path,
) -> None:
    """A fresh install (no MY_INJECTION.md, no env overrides for INJECTIONS)
    still shows the default test-first trick in the inject panel."""
    tricks = read_tricks()
    assert MY_INJECTION_DEFAULT_BODY in tricks
