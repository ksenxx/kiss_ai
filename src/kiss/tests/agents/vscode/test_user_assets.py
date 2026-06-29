# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.agents.vscode.user_assets``.

Locks in the **runtime** contract of the
:func:`ensure_user_asset_from_default` helper used for
``MY_TASK_TEMPLATES.md`` (welcome-screen chips) and
``MY_INJECTION.md`` (Inject instruction panel) — both seeded from an
inline default string on first read and never overwritten thereafter.

Neither ``INJECTIONS.md`` nor ``SAMPLE_TASKS.md`` is copied into
``~/.kiss/`` by either ``install.sh`` or ``installMarkdownAssets`` in
``DependencyInstaller.ts``; both are read directly from the bundled
extension package.

The contract:

* ``KISS_HOME`` overrides ``~/.kiss``.
* A missing user copy is seeded from the inline default content on
  first read.
* An existing user copy is returned unchanged between daemon restarts
  so user edits survive every runtime re-read.
* When ``~/.kiss/`` is not writable, ``ensure_user_asset_from_default``
  returns ``None`` so callers can fall back gracefully.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiss.agents.vscode.user_assets import (
    ensure_user_asset_from_default,
    kiss_home_dir,
)


@pytest.fixture
def kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``KISS_HOME`` to a fresh ``tmp_path`` for each test."""
    home = tmp_path / ".kiss"
    monkeypatch.setenv("KISS_HOME", str(home))
    return home


def test_kiss_home_dir_honours_env(kiss_home: Path) -> None:
    assert kiss_home_dir() == kiss_home


def test_kiss_home_dir_defaults_to_dot_kiss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KISS_HOME", raising=False)
    assert kiss_home_dir() == Path.home() / ".kiss"


def test_ensure_user_asset_from_default_seeds_with_default_content(
    kiss_home: Path,
) -> None:
    """When ``~/.kiss/<name>`` is missing, write the supplied default."""
    result = ensure_user_asset_from_default(
        "MY_TASK_TEMPLATES.md", "## Task\n\nHi!\n",
    )
    expected = kiss_home / "MY_TASK_TEMPLATES.md"
    assert result == expected
    assert result is not None
    assert expected.exists()
    assert result.read_text() == "## Task\n\nHi!\n"


def test_ensure_user_asset_from_default_preserves_user_edits(
    kiss_home: Path,
) -> None:
    """An existing user copy must NEVER be overwritten by the default."""
    kiss_home.mkdir(parents=True, exist_ok=True)
    user = kiss_home / "MY_TASK_TEMPLATES.md"
    user.write_text("## Task\n\nMy curated chip\n")
    result = ensure_user_asset_from_default(
        "MY_TASK_TEMPLATES.md", "## Task\n\nHi!\n",
    )
    assert result == user
    assert result is not None
    assert result.read_text() == "## Task\n\nMy curated chip\n"


def test_ensure_user_asset_from_default_creates_kiss_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "fresh" / ".kiss"
    monkeypatch.setenv("KISS_HOME", str(home))
    assert not home.exists()
    result = ensure_user_asset_from_default(
        "MY_TASK_TEMPLATES.md", "## Task\n\nHi!\n",
    )
    assert result == home / "MY_TASK_TEMPLATES.md"
    assert result is not None
    assert home.is_dir()
    assert result.read_text() == "## Task\n\nHi!\n"


def test_ensure_user_asset_from_default_falls_back_to_none_when_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read-only KISS_HOME yields ``None`` so callers can skip cleanly."""
    if os.geteuid() == 0:  # pragma: no cover - CI runs as non-root
        pytest.skip("root cannot lose write permission via chmod")
    readonly_parent = tmp_path / "ro"
    readonly_parent.mkdir()
    readonly_parent.chmod(0o500)
    try:
        monkeypatch.setenv("KISS_HOME", str(readonly_parent / ".kiss"))
        result = ensure_user_asset_from_default(
            "MY_TASK_TEMPLATES.md", "## Task\n\nHi!\n",
        )
        assert result is None
    finally:
        readonly_parent.chmod(0o700)


def test_ensure_user_asset_from_default_my_injection_seed(
    kiss_home: Path,
) -> None:
    """``MY_INJECTION.md`` is auto-seeded on first read with the test-first trick.

    Locks in the user-facing default file content for the new
    "Inject instruction" panel.  ``test_inject_panel_order.py``
    exercises the full panel pipeline; this test pins the helper
    contract that backs it.
    """
    default_body = (
        "Write end-to-end 100% coverage tests for the feature first."
        "  Then implement the feature."
    )
    default = "## Trick\n\n" + default_body + "\n"
    result = ensure_user_asset_from_default("MY_INJECTION.md", default)
    expected = kiss_home / "MY_INJECTION.md"
    assert result == expected
    assert result is not None
    assert expected.exists()
    assert result.read_text() == default
