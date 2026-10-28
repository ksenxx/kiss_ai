# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``kiss.server.tricks._ensure_my_injection``.

Locks in the **runtime** contract of the :func:`_ensure_my_injection`
helper behind ``MY_INJECTION.md`` (Inject instruction panel) — seeded
from :data:`DEFAULT_MY_INJECTION` on first read and never overwritten
thereafter.

``INJECTIONS.md`` is not copied into ``~/.kiss/`` by ``install.sh``;
it is read directly from the bundled ``src/kiss/INJECTIONS.md``.

The contract:

* ``KISS_HOME`` overrides ``~/.kiss``.
* A missing user copy is seeded from the inline default content on
  first read.
* An existing user copy is returned unchanged between daemon restarts
  so user edits survive every runtime re-read.
* When ``~/.kiss/`` is not writable, ``_ensure_my_injection`` returns
  ``None`` so callers can fall back gracefully.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.server.tricks import DEFAULT_MY_INJECTION, _ensure_my_injection
from kiss.tests.conftest import is_root, posix_only


def test_ensure_my_injection_seeds_with_default_content(kiss_home: Path) -> None:
    """When ``~/.kiss/MY_INJECTION.md`` is missing, write the default."""
    result = _ensure_my_injection()
    expected = kiss_home / "MY_INJECTION.md"
    assert result == expected
    assert result is not None
    assert expected.exists()
    assert result.read_text() == DEFAULT_MY_INJECTION


def test_ensure_my_injection_preserves_user_edits(kiss_home: Path) -> None:
    """An existing user copy must NEVER be overwritten by the default."""
    kiss_home.mkdir(parents=True, exist_ok=True)
    user = kiss_home / "MY_INJECTION.md"
    user.write_text("## Trick\n\nMy curated trick\n")
    result = _ensure_my_injection()
    assert result == user
    assert result is not None
    assert result.read_text() == "## Trick\n\nMy curated trick\n"


def test_ensure_my_injection_creates_kiss_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "fresh" / ".kiss"
    monkeypatch.setenv("KISS_HOME", str(home))
    assert not home.exists()
    result = _ensure_my_injection()
    assert result == home / "MY_INJECTION.md"
    assert result is not None
    assert home.is_dir()
    assert result.read_text() == DEFAULT_MY_INJECTION


@posix_only("chmod-based permission denial")
def test_ensure_my_injection_falls_back_to_none_when_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read-only KISS_HOME yields ``None`` so callers can skip cleanly."""
    if is_root():  # pragma: no cover - CI runs as non-root
        pytest.skip("root cannot lose write permission via chmod")
    readonly_parent = tmp_path / "ro"
    readonly_parent.mkdir()
    readonly_parent.chmod(0o500)
    try:
        monkeypatch.setenv("KISS_HOME", str(readonly_parent / ".kiss"))
        assert _ensure_my_injection() is None
    finally:
        readonly_parent.chmod(0o700)


def test_ensure_my_injection_default_is_the_test_first_trick(kiss_home: Path) -> None:
    """``MY_INJECTION.md`` is auto-seeded on first read with the test-first trick.

    Locks in the user-facing default file content for the "Inject
    instruction" panel.  ``test_inject_panel_order.py`` exercises the
    full panel pipeline; this test pins the helper contract that backs
    it.
    """
    default_body = (
        "Write end-to-end 100% coverage tests for the feature first."
        "  Then implement the feature."
    )
    assert DEFAULT_MY_INJECTION == "## Trick\n\n" + default_body + "\n"
    result = _ensure_my_injection()
    assert result is not None
    assert result.read_text() == DEFAULT_MY_INJECTION
