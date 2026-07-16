# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the "Inject instruction" panel ordering and seeding.

The Inject instruction panel (sidebar button in the VS Code chat
webview) renders the strings returned by
:func:`kiss.server.tricks.read_tricks` (Python side, used by
the kiss-web daemon HTML builder).  The contract:

1. ``~/.kiss/MY_INJECTION.md`` is auto-seeded on first read with the
   default starter trick

       ## Trick

       Write end-to-end 100% coverage tests for the feature first.  Then implement the feature.

   so a fresh install shows at least one user-editable trick.
2. The panel lists **MY_INJECTION.md tricks first**, then the bundled
   ``src/kiss/INJECTIONS.md`` tricks.  No copy of ``INJECTIONS.md`` is
   ever written into ``~/.kiss/`` — the bundled package file is the
   only source of those entries, so every extension upgrade
   automatically delivers the latest bundled tricks without
   clobbering the user's curated list.
3. User edits to ``~/.kiss/MY_INJECTION.md`` are NEVER overwritten by
   the default seed once the file exists.
4. When ``~/.kiss/`` is not writable, the panel still renders the
   bundled tricks (graceful degradation).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiss.server.tricks import (
    DEFAULT_MY_INJECTION,
    MY_INJECTION_DEFAULT_BODY,
    read_tricks,
)
from kiss.server.user_assets import kiss_home_dir


@pytest.fixture
def kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``KISS_HOME`` to a fresh ``tmp_path`` for each test."""
    home = tmp_path / ".kiss"
    monkeypatch.setenv("KISS_HOME", str(home))
    return home


@pytest.fixture
def bundled_injections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point ``KISS_INJECTIONS_PATH`` at a fake bundled INJECTIONS.md.

    Lets the test pin exactly which bundled tricks the panel will
    list, independent of whatever the real ``src/kiss/INJECTIONS.md``
    happens to contain.
    """
    pkg = tmp_path / "pkg_INJECTIONS.md"
    pkg.write_text(
        "## Trick\n\nbundled trick one\n\n"
        "## Trick\n\nbundled trick two\n",
    )
    monkeypatch.setenv("KISS_INJECTIONS_PATH", str(pkg))
    return pkg


def test_default_seed_body_matches_task_spec() -> None:
    """The MY_INJECTION.md default seed body is exactly the task spec text.

    Locks in the user-visible default trick text so a regression in the
    constant cannot quietly change what a fresh install shows.
    """
    assert MY_INJECTION_DEFAULT_BODY == (
        "Write end-to-end 100% coverage tests for the feature first."
        "  Then implement the feature."
    )
    # The full file content must be parseable as a single ``## Trick``
    # section so the panel renders it as one chip.
    assert DEFAULT_MY_INJECTION == (
        "## Trick\n\n" + MY_INJECTION_DEFAULT_BODY + "\n"
    )


def test_my_injection_seeded_with_default_when_missing(
    kiss_home: Path, bundled_injections: Path,
) -> None:
    """First read auto-creates ``~/.kiss/MY_INJECTION.md`` with the default."""
    my_path = kiss_home / "MY_INJECTION.md"
    assert not my_path.exists()
    tricks = read_tricks()
    # MY_INJECTION.md exists after the call, holding the default.
    assert my_path.exists()
    assert my_path.read_text() == DEFAULT_MY_INJECTION
    # The default trick comes first, followed by the bundled tricks.
    assert tricks == [
        MY_INJECTION_DEFAULT_BODY,
        "bundled trick one",
        "bundled trick two",
    ]


def test_my_injection_first_then_bundled(
    kiss_home: Path, bundled_injections: Path,
) -> None:
    """User-curated MY_INJECTION tricks come before bundled INJECTIONS tricks."""
    kiss_home.mkdir(parents=True, exist_ok=True)
    (kiss_home / "MY_INJECTION.md").write_text(
        "## Trick\n\nmy curated trick A\n\n"
        "## Trick\n\nmy curated trick B\n",
    )
    tricks = read_tricks()
    assert tricks == [
        "my curated trick A",
        "my curated trick B",
        "bundled trick one",
        "bundled trick two",
    ]


def test_my_injection_user_edits_preserved(
    kiss_home: Path, bundled_injections: Path,
) -> None:
    """An existing MY_INJECTION.md is NEVER overwritten by the default."""
    kiss_home.mkdir(parents=True, exist_ok=True)
    user = kiss_home / "MY_INJECTION.md"
    user.write_text("## Trick\n\nuser override\n")
    # Make the user copy unambiguously older than "now" — a naive
    # mtime-refresh would clobber it; this test asserts we never do.
    os.utime(user, (0, 0))
    tricks = read_tricks()
    assert user.read_text() == "## Trick\n\nuser override\n"
    assert tricks == [
        "user override",
        "bundled trick one",
        "bundled trick two",
    ]


def test_empty_my_injection_falls_back_to_bundled_only(
    kiss_home: Path, bundled_injections: Path,
) -> None:
    """A MY_INJECTION.md with no ``## Trick`` sections contributes nothing."""
    kiss_home.mkdir(parents=True, exist_ok=True)
    (kiss_home / "MY_INJECTION.md").write_text(
        "# just a heading\n\nno trick sections here\n",
    )
    tricks = read_tricks()
    assert tricks == ["bundled trick one", "bundled trick two"]


def test_no_injections_md_written_to_kiss_home(  # noqa: N802 - test asserts on filename
    kiss_home: Path, bundled_injections: Path,
) -> None:
    """``~/.kiss/INJECTIONS.md`` must NEVER be created by the runtime.

    The bundled ``src/kiss/INJECTIONS.md`` is read directly from the
    package; only ``MY_INJECTION.md`` is written into ``~/.kiss/``.
    This test guards against an accidental re-introduction of the
    install-time copy or a runtime seed.
    """
    read_tricks()
    assert not (kiss_home / "INJECTIONS.md").exists(), (
        "read_tricks() must not create ~/.kiss/INJECTIONS.md — only "
        "MY_INJECTION.md is auto-seeded."
    )
    # But MY_INJECTION.md SHOULD have been created.
    assert (kiss_home / "MY_INJECTION.md").exists()


def test_read_tricks_handles_unreadable_my_injection_gracefully(
    kiss_home: Path, bundled_injections: Path,
) -> None:
    """A binary-corrupted MY_INJECTION.md must not crash the daemon."""
    kiss_home.mkdir(parents=True, exist_ok=True)
    # Write raw bytes that aren't valid UTF-8 to provoke a
    # UnicodeDecodeError on ``read_text``.
    (kiss_home / "MY_INJECTION.md").write_bytes(b"\xff\xfe\x00\x00bogus")
    tricks = read_tricks()
    # The bundled tricks still render — MY_INJECTION.md just contributes
    # nothing rather than killing the ghost-text worker thread.
    assert tricks == ["bundled trick one", "bundled trick two"]


def test_web_server_read_tricks_uses_my_injection_first(
    kiss_home: Path, bundled_injections: Path,
) -> None:
    """The daemon's tricks source returns MY_INJECTION first, then bundled.

    ``web_server`` builds its HTML from the public
    :func:`kiss.server.tricks.read_tricks`, which must yield
    the merged ordered list.
    """
    kiss_home.mkdir(parents=True, exist_ok=True)
    (kiss_home / "MY_INJECTION.md").write_text(
        "## Trick\n\ndaemon-side user trick\n",
    )
    tricks = read_tricks()
    assert tricks == [
        "daemon-side user trick",
        "bundled trick one",
        "bundled trick two",
    ]


def test_kiss_home_unwritable_still_returns_bundled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bundled_injections: Path,
) -> None:
    """A read-only ``~/.kiss/`` must not stop the bundled tricks from rendering."""
    if os.geteuid() == 0:  # pragma: no cover - CI runs as non-root
        pytest.skip("root cannot lose write permission via chmod")
    readonly_parent = tmp_path / "ro"
    readonly_parent.mkdir()
    readonly_parent.chmod(0o500)
    try:
        monkeypatch.setenv("KISS_HOME", str(readonly_parent / ".kiss"))
        assert kiss_home_dir() == readonly_parent / ".kiss"
        tricks = read_tricks()
        # ensure_user_asset_from_default returns None on read-only HOME;
        # MY_INJECTION tricks are skipped, bundled tricks still render.
        assert tricks == ["bundled trick one", "bundled trick two"]
    finally:
        readonly_parent.chmod(0o700)


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
    # The default MY_INJECTION trick is always at index 0 of a fresh
    # install (kiss_home fixture wipes the user copy).
    assert tricks[0] == MY_INJECTION_DEFAULT_BODY
    # The real bundled INJECTIONS.md ships several "Trick" sections,
    # so there must be at least one bundled trick after the user one.
    assert len(tricks) >= 2


def test_default_trick_renders_at_least_once_on_fresh_install(
    kiss_home: Path,
) -> None:
    """A fresh install (no MY_INJECTION.md, no env overrides for INJECTIONS)
    still shows the default test-first trick in the inject panel."""
    tricks = read_tricks()
    assert MY_INJECTION_DEFAULT_BODY in tricks


def test_missing_bundled_injections_returns_my_injection_only(
    kiss_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the bundled INJECTIONS.md is missing, MY_INJECTION still renders.

    Pointing ``KISS_INJECTIONS_PATH`` at a path that does not exist
    exercises the ``OSError`` fallback branch in
    ``_read_bundled_tricks`` — the panel must still show the
    auto-seeded user trick rather than crashing the daemon.
    """
    monkeypatch.setenv("KISS_INJECTIONS_PATH", str(tmp_path / "missing.md"))
    tricks = read_tricks()
    assert tricks == [MY_INJECTION_DEFAULT_BODY]


def test_corrupt_bundled_injections_returns_my_injection_only(
    kiss_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A binary-corrupt bundled INJECTIONS.md must not crash the daemon."""
    bundled = tmp_path / "bundled_bad.md"
    bundled.write_bytes(b"\xff\xfe\x00\x00bogus")
    monkeypatch.setenv("KISS_INJECTIONS_PATH", str(bundled))
    tricks = read_tricks()
    assert tricks == [MY_INJECTION_DEFAULT_BODY]


def test_section_with_non_trick_heading_is_ignored(
    kiss_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``## Task`` or other non-``## Trick`` heading must contribute nothing."""
    bundled = tmp_path / "mixed.md"
    bundled.write_text(
        "## Task\n\nnot a trick\n\n"
        "## Trick\n\nreal trick\n\n"
        "## Other\n\nnot a trick either\n",
    )
    monkeypatch.setenv("KISS_INJECTIONS_PATH", str(bundled))
    kiss_home.mkdir(parents=True, exist_ok=True)
    (kiss_home / "MY_INJECTION.md").write_text("")  # no user tricks
    tricks = read_tricks()
    assert tricks == ["real trick"]


def test_trick_section_with_empty_body_is_ignored(
    kiss_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``## Trick`` section whose body is whitespace contributes nothing."""
    bundled = tmp_path / "empty_body.md"
    bundled.write_text(
        "## Trick\n\n   \n\n## Trick\n\nactual trick\n",
    )
    monkeypatch.setenv("KISS_INJECTIONS_PATH", str(bundled))
    kiss_home.mkdir(parents=True, exist_ok=True)
    (kiss_home / "MY_INJECTION.md").write_text("")
    tricks = read_tricks()
    assert tricks == ["actual trick"]
