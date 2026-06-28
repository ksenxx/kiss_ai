# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.agents.vscode.user_assets``.

Locks in the **runtime** contract of two helpers:

* :func:`ensure_user_asset` — generic helper that seeds ``~/.kiss/<name>``
  from a bundled package copy on first read.  No production code now
  relies on it for ``INJECTIONS.md`` (the bundled tricks file is read
  directly from the package — see ``test_inject_panel_order.py``);
  the helper remains in the public API surface for future
  package-seeded assets and is exercised here generically.
* :func:`ensure_user_asset_from_default` for ``MY_TASK_TEMPLATES.md``
  and ``MY_INJECTION.md`` — seeded from an inline default string on
  first read and never overwritten thereafter.

Neither ``INJECTIONS.md`` nor ``SAMPLE_TASKS.md`` is copied into
``~/.kiss/`` by either ``install.sh`` or ``installMarkdownAssets`` in
``DependencyInstaller.ts``; both are read directly from the bundled
extension package.

The contract:

* ``KISS_HOME`` overrides ``~/.kiss``.
* A missing user copy is seeded from the package copy / default content
  on first read.
* An existing user copy is returned unchanged between daemon restarts
  so user edits survive every runtime re-read.
* When ``~/.kiss/`` is not writable, ``ensure_user_asset`` returns the
  package path and ``ensure_user_asset_from_default`` returns ``None``
  so callers can fall back gracefully.
* When neither file exists ``ensure_user_asset`` returns the package
  path (callers are expected to ``try``/``except`` around the subsequent
  ``read_text``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiss.agents.vscode.user_assets import (
    ensure_user_asset,
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


def test_returns_package_path_when_user_copy_missing_and_package_missing(
    kiss_home: Path, tmp_path: Path,
) -> None:
    pkg = tmp_path / "nonexistent.md"
    result = ensure_user_asset("nonexistent.md", pkg)
    assert result == pkg
    assert not result.exists()


def test_seeds_user_copy_from_package_on_first_read(
    kiss_home: Path, tmp_path: Path,
) -> None:
    pkg = tmp_path / "ASSET.md"
    pkg.write_text("## Section\n\nSeeded content\n")
    result = ensure_user_asset("ASSET.md", pkg)
    expected = kiss_home / "ASSET.md"
    assert result == expected
    assert expected.exists()
    assert expected.read_text() == "## Section\n\nSeeded content\n"


def test_returns_user_copy_when_present(
    kiss_home: Path, tmp_path: Path,
) -> None:
    pkg = tmp_path / "ASSET.md"
    pkg.write_text("## Section\n\nPackage default\n")
    kiss_home.mkdir(parents=True, exist_ok=True)
    user = kiss_home / "ASSET.md"
    user.write_text("## Section\n\nUser override\n")
    result = ensure_user_asset("ASSET.md", pkg)
    assert result == user
    assert result.read_text() == "## Section\n\nUser override\n"


def test_user_edits_never_clobbered_by_newer_package_copy(
    kiss_home: Path, tmp_path: Path,
) -> None:
    """``ensure_user_asset`` is read-only once the user copy exists.

    A naive mtime refresh would overwrite the user copy here — this
    test asserts the helper never silently restores the package copy
    even when its mtime is more recent.
    """
    pkg = tmp_path / "ASSET.md"
    pkg.write_text("## Section\n\nFresh package content\n")
    kiss_home.mkdir(parents=True, exist_ok=True)
    user = kiss_home / "ASSET.md"
    user.write_text("## Section\n\nUser-curated content\n")
    # Make the user copy unambiguously *older* than the package copy.
    os.utime(user, (0, 0))
    result = ensure_user_asset("ASSET.md", pkg)
    assert result == user
    assert result.read_text() == "## Section\n\nUser-curated content\n"


def test_creates_kiss_home_directory_if_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "fresh" / ".kiss"
    monkeypatch.setenv("KISS_HOME", str(home))
    assert not home.exists()
    pkg = tmp_path / "ASSET.md"
    pkg.write_text("## Section\n\nhi\n")
    result = ensure_user_asset("ASSET.md", pkg)
    assert result == home / "ASSET.md"
    assert home.is_dir()
    assert result.read_text() == "## Section\n\nhi\n"


def test_falls_back_to_package_when_kiss_home_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Point KISS_HOME at a path inside a read-only parent so the
    # ``mkdir`` raises ``OSError`` and the helper falls back to the
    # package copy without crashing.  Skip on Windows-style root-only
    # POSIX setups where chmod cannot revoke write access from the
    # effective UID (root tests).
    if os.geteuid() == 0:  # pragma: no cover - CI runs as non-root
        pytest.skip("root cannot lose write permission via chmod")
    readonly_parent = tmp_path / "ro"
    readonly_parent.mkdir()
    readonly_parent.chmod(0o500)
    try:
        monkeypatch.setenv("KISS_HOME", str(readonly_parent / ".kiss"))
        pkg = tmp_path / "ASSET.md"
        pkg.write_text("## Section\n\nfallback\n")
        result = ensure_user_asset("ASSET.md", pkg)
        assert result == pkg
        assert result.read_text() == "## Section\n\nfallback\n"
    finally:
        readonly_parent.chmod(0o700)


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
