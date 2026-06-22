# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.agents.vscode.user_assets``.

Locks in the **runtime** contract of :func:`ensure_user_asset` for
``INJECTIONS.md`` and ``SAMPLE_TASKS.md``:

* ``KISS_HOME`` overrides ``~/.kiss``.
* A missing user copy is seeded from the package copy on first read.
* An existing user copy is returned unchanged between daemon restarts
  so user edits made *between* installs survive every runtime re-read.
  Note: at *install time*, ``install.sh`` and ``installMarkdownAssets``
  in ``DependencyInstaller.ts`` always overwrite both files with the
  latest bundled copy — matching the ``MODEL_INFO.json`` pattern.
* When ``~/.kiss/`` is not writable, the package copy is returned so
  the caller still sees a valid file.
* When neither file exists the package path is returned (callers are
  expected to ``try``/``except`` around the subsequent ``read_text``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiss.agents.vscode.user_assets import ensure_user_asset, kiss_home_dir
from kiss.agents.vscode.web_server import _read_tricks


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
    pkg = tmp_path / "INJECTIONS.md"
    pkg.write_text("## Trick\n\nSeeded content\n")
    result = ensure_user_asset("INJECTIONS.md", pkg)
    expected = kiss_home / "INJECTIONS.md"
    assert result == expected
    assert expected.exists()
    assert expected.read_text() == "## Trick\n\nSeeded content\n"


def test_returns_user_copy_when_present(
    kiss_home: Path, tmp_path: Path,
) -> None:
    pkg = tmp_path / "INJECTIONS.md"
    pkg.write_text("## Trick\n\nPackage default\n")
    kiss_home.mkdir(parents=True, exist_ok=True)
    user = kiss_home / "INJECTIONS.md"
    user.write_text("## Trick\n\nUser override\n")
    result = ensure_user_asset("INJECTIONS.md", pkg)
    assert result == user
    assert result.read_text() == "## Trick\n\nUser override\n"


def test_user_edits_survive_newer_package_copy(
    kiss_home: Path, tmp_path: Path,
) -> None:
    """Runtime ``ensure_user_asset`` preserves user edits between daemon restarts.

    At *install time* ``install.sh`` and ``installMarkdownAssets`` in
    ``DependencyInstaller.ts`` always overwrite ``~/.kiss/INJECTIONS.md``
    and ``~/.kiss/SAMPLE_TASKS.md`` with the latest bundled copy — matching
    the ``MODEL_INFO.json`` pattern.

    At *runtime* (this function), however, the file is read-only: if the
    user copy already exists it is returned as-is so edits made *between*
    installs survive daemon restarts and ``uv run`` invocations.  This test
    locks in that runtime contract.
    """
    pkg = tmp_path / "INJECTIONS.md"
    pkg.write_text("## Trick\n\nFresh package content\n")
    kiss_home.mkdir(parents=True, exist_ok=True)
    user = kiss_home / "INJECTIONS.md"
    user.write_text("## Trick\n\nUser-curated content\n")
    # Make the user copy unambiguously *older* than the package copy.
    # A naive mtime refresh would overwrite the user copy here — this
    # test asserts the helper does NOT.
    os.utime(user, (0, 0))
    result = ensure_user_asset("INJECTIONS.md", pkg)
    assert result == user
    assert result.read_text() == "## Trick\n\nUser-curated content\n"


def test_creates_kiss_home_directory_if_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "fresh" / ".kiss"
    monkeypatch.setenv("KISS_HOME", str(home))
    assert not home.exists()
    pkg = tmp_path / "INJECTIONS.md"
    pkg.write_text("## Trick\n\nhi\n")
    result = ensure_user_asset("INJECTIONS.md", pkg)
    assert result == home / "INJECTIONS.md"
    assert home.is_dir()
    assert result.read_text() == "## Trick\n\nhi\n"


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
        pkg = tmp_path / "INJECTIONS.md"
        pkg.write_text("## Trick\n\nfallback\n")
        result = ensure_user_asset("INJECTIONS.md", pkg)
        assert result == pkg
        assert result.read_text() == "## Trick\n\nfallback\n"
    finally:
        readonly_parent.chmod(0o700)


def test_read_tricks_prefers_user_copy_in_kiss_home(
    kiss_home: Path,
) -> None:
    """``web_server._read_tricks`` must consult ``~/.kiss/INJECTIONS.md``.

    Locks in the contract that the kiss-web daemon (Python side) reads
    the user-local copy of ``INJECTIONS.md`` rather than the package
    copy, so authors can customise Tricks by editing
    ``~/.kiss/INJECTIONS.md`` without rebuilding.

    ``ensure_user_asset`` never refreshes an existing user copy from
    the package copy, so no mutation of the shipped
    ``src/kiss/INJECTIONS.md`` is necessary to lock in this contract.
    """
    kiss_home.mkdir(parents=True, exist_ok=True)
    (kiss_home / "INJECTIONS.md").write_text(
        "## Trick\n\nuser trick\n\n## Trick\n\nsecond user trick\n",
    )
    tricks = _read_tricks()
    assert tricks == ["user trick", "second user trick"]

