# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the SEA slash-command registry.

Covers the module in :mod:`kiss.agents.sorcar.sea_commands`:

* the registry precedence rule (bundled ``third_party_agents`` beats
  every user folder, and SEAS.md folders lower in the file override
  higher ones);
* live-reload behaviour when ``~/.kiss/SEAS.md`` or a watched folder
  changes;
* the slash-command prompt rewriter that turns ``/xxx text`` into an
  explicit ``run_agent`` directive.

Every test isolates :mod:`kiss.agents.sorcar.sea_commands` via a
``_reset_for_tests()`` fixture so a subscriber leaked from one test
cannot fire during another.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.core.config import kiss_home


@pytest.fixture(autouse=True)
def _reset_sea_commands() -> Iterator[None]:
    """Drop the module's in-memory state before and after each test."""
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()


def _touch_sea(folder: Path, name: str) -> Path:
    """Create an empty ``<name>_sea.py`` file in *folder* and return it."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_sea.py"
    path.write_text("# stub SEA for tests\n", encoding="utf-8")
    return path


def _write_seas_md(lines: list[str]) -> None:
    """Overwrite ``~/.kiss/SEAS.md`` with *lines* (one per row)."""
    home = kiss_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "SEAS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_bundled_third_party_commands_are_registered() -> None:
    """Every ``third_party_agents/*_sea.py`` is exposed as ``/<stem>``.

    The registry MUST include at least the ``slack`` and ``gmail``
    commands — both bundled with the repo — and their resolved paths
    MUST point at the real files under
    ``src/kiss/agents/third_party_agents``.
    """
    commands = sea_commands.refresh_registry()
    assert "slack" in commands
    assert "gmail" in commands
    slack_path = sea_commands.get_command("slack")
    assert slack_path is not None
    assert slack_path.name == "slack_sea.py"
    assert slack_path.parent.name == "third_party_agents"


def test_seas_md_precedence_bottom_beats_top(tmp_path: Path) -> None:
    """The folder at the BOTTOM of ``SEAS.md`` overrides higher rows.

    Also verifies the *top-most* folder still contributes a command
    that neither the bottom folder nor the bundled third-party dir
    defines.
    """
    top = tmp_path / "top"
    bottom = tmp_path / "bottom"
    top_only = _touch_sea(top, "topcmd")
    _touch_sea(top, "shared")  # will lose to bottom
    bottom_only = _touch_sea(bottom, "botcmd")
    bottom_shared = _touch_sea(bottom, "shared")
    _write_seas_md([str(top), str(bottom)])
    commands = sea_commands.refresh_registry()
    assert "topcmd" in commands and "botcmd" in commands
    assert sea_commands.get_command("topcmd") == top_only.resolve()
    assert sea_commands.get_command("botcmd") == bottom_only.resolve()
    # Precedence check: shared name resolves to the BOTTOM folder.
    assert sea_commands.get_command("shared") == bottom_shared.resolve()


def test_third_party_dir_beats_seas_md(tmp_path: Path) -> None:
    """A bundled SEA name overrides any user folder that redefines it.

    Uses ``slack``, which the third-party dir ships.  A user-folder
    ``slack_sea.py`` MUST be ignored so the daemon-installed agent
    always wins.
    """
    user = tmp_path / "override"
    _touch_sea(user, "slack")
    _write_seas_md([str(user)])
    sea_commands.refresh_registry()
    slack_path = sea_commands.get_command("slack")
    assert slack_path is not None
    assert slack_path.parent.name == "third_party_agents"


def test_seas_md_ignores_blanks_comments_and_expands_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank / ``#`` lines and env-var / ``~`` expansions must all work.

    Whitespace inside a path is preserved verbatim (no shell quoting
    required) — only a whitespace-preceded inline ``#`` starts a
    comment tail.
    """
    folder = tmp_path / "my seas"
    _touch_sea(folder, "envcmd")
    monkeypatch.setenv("SEA_TEST_ROOT", str(tmp_path))
    _write_seas_md([
        "# top comment",
        "",
        "$SEA_TEST_ROOT/my seas  # inline comment",
        "   ",
    ])
    sea_commands.refresh_registry()
    resolved = sea_commands.get_command("envcmd")
    assert resolved is not None
    assert resolved.resolve() == (folder / "envcmd_sea.py").resolve()


def test_missing_folder_in_seas_md_is_silently_skipped(
    tmp_path: Path,
) -> None:
    """A stale SEAS.md entry must never crash the registry rebuild."""
    good = tmp_path / "good"
    _touch_sea(good, "livecmd")
    ghost = tmp_path / "does-not-exist"
    _write_seas_md([str(ghost), str(good)])
    commands = sea_commands.refresh_registry()
    assert "livecmd" in commands


def test_non_sea_files_are_ignored(tmp_path: Path) -> None:
    """Only ``*_sea.py`` files surface; unrelated files never do.

    Also confirms that stems producing an invalid command name — a
    name that contains characters outside ``[A-Za-z0-9_-]`` — are
    silently skipped, because the parser and the autocomplete both
    refuse them and surfacing them as commands would mean advertising
    a slash command the user cannot type.
    """
    folder = tmp_path / "seas"
    _touch_sea(folder, "public")
    (folder / "notasea.py").write_text("", encoding="utf-8")
    (folder / "foo.bar_sea.py").write_text("", encoding="utf-8")
    (folder / "with space_sea.py").write_text("", encoding="utf-8")
    _write_seas_md([str(folder)])
    commands = sea_commands.refresh_registry()
    assert "public" in commands
    assert "notasea" not in commands
    assert "foo.bar" not in commands
    assert "with space" not in commands


def test_private_underscore_prefixed_seas_are_included(tmp_path: Path) -> None:
    """Every ``*_sea.py`` in a watched folder becomes a command.

    The requirement is "convert all *_sea.py", so ``_helper_sea.py``
    IS a command ``/_helper`` — the underscore is a valid character
    for a command name.
    """
    folder = tmp_path / "seas"
    _touch_sea(folder, "_helper")
    _write_seas_md([str(folder)])
    commands = sea_commands.refresh_registry()
    assert "_helper" in commands


def test_rewrite_prompt_for_registered_command(tmp_path: Path) -> None:
    """A ``/<name> <text>`` prompt is turned into a ``run_agent`` directive."""
    folder = tmp_path / "seas"
    sea_path = _touch_sea(folder, "myslack")
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    result = sea_commands.rewrite_prompt_if_command(
        '/myslack post "hi there" to #general',
    )
    assert result is not None
    rewritten, resolved = result
    assert resolved == sea_path.resolve()
    assert "run_agent" in rewritten
    assert str(sea_path.resolve()) in rewritten
    assert 'post "hi there" to #general' in rewritten


def test_rewrite_returns_none_for_unknown_command() -> None:
    """A slash prefix that does not match any SEA MUST NOT rewrite."""
    sea_commands.refresh_registry()
    assert sea_commands.rewrite_prompt_if_command("/definitelynot hi") is None


def test_rewrite_returns_none_without_trailing_text(tmp_path: Path) -> None:
    """``/xxx`` alone (no task text) must not rewrite; run_agent needs a task."""
    folder = tmp_path / "seas"
    _touch_sea(folder, "solo")
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    assert sea_commands.rewrite_prompt_if_command("/solo") is None
    assert sea_commands.rewrite_prompt_if_command("/solo   ") is None


def test_rewrite_ignores_prompts_that_are_not_slash_commands() -> None:
    """Plain prompts, chat text with a slash mid-line, etc. pass through."""
    sea_commands.refresh_registry()
    assert sea_commands.rewrite_prompt_if_command("hello world") is None
    assert sea_commands.rewrite_prompt_if_command("\n/slack hi") is None
    assert sea_commands.rewrite_prompt_if_command("say /slack") is None
    assert sea_commands.rewrite_prompt_if_command("") is None
    assert sea_commands.rewrite_prompt_if_command("/") is None


def test_rewrite_requires_whitespace_after_command_name(tmp_path: Path) -> None:
    """``/slackfoo bar`` must NOT match ``/slack`` — the boundary is a space."""
    folder = tmp_path / "seas"
    _touch_sea(folder, "slk")
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    assert sea_commands.rewrite_prompt_if_command("/slkextra text") is None
    assert sea_commands.rewrite_prompt_if_command("/slk text") is not None


def test_watcher_picks_up_seas_md_change(tmp_path: Path) -> None:
    """Editing ``SEAS.md`` between polls must update the registry.

    Uses a very short interval and polls for the expected update
    (bounded wait) so the test finishes quickly on slow CI.
    """
    initial_folder = tmp_path / "one"
    later_folder = tmp_path / "two"
    _touch_sea(initial_folder, "alpha")
    _touch_sea(later_folder, "beta")
    _write_seas_md([str(initial_folder)])
    seen: list[list[str]] = []
    sea_commands.subscribe(lambda cmds: seen.append(list(cmds)))
    sea_commands.start_registry_watcher(poll_interval=0.05)
    try:
        # The initial refresh MUST fire the subscriber with alpha.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not any(
            "alpha" in snap for snap in seen
        ):
            time.sleep(0.05)
        assert any("alpha" in snap for snap in seen)
        # Now add the second folder — the poller MUST notice.
        _write_seas_md([str(initial_folder), str(later_folder)])
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not any(
            "beta" in snap for snap in seen
        ):
            time.sleep(0.05)
        assert any("beta" in snap for snap in seen)
    finally:
        sea_commands.stop_registry_watcher()


def test_seas_md_preserves_backslashes_in_folder_names(
    tmp_path: Path,
) -> None:
    """Backslashes inside a folder path MUST survive parsing.

    Uses a real folder whose name contains a literal ``\\`` character
    (valid in POSIX filenames) so the assertion catches the regression
    of ``shlex.split`` turning ``C:\\Users\\alice\\seas`` into
    ``C:Usersaliceseas`` — the parser strips only whitespace-preceded
    ``#`` comment tails and leaves every other character alone.
    """
    weird = tmp_path / "with\\backslash"
    _touch_sea(weird, "backcmd")
    # Read back the parsed folder list to check the raw string
    # survived intact (independent of what the filesystem later does
    # with it) before also asserting the derived command name.
    _write_seas_md([str(weird)])
    folders = sea_commands._read_seas_md_folders()
    assert folders and "\\" in str(folders[0])
    commands = sea_commands.refresh_registry()
    assert "backcmd" in commands


def test_refresh_registry_broadcasts_once_under_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compare-and-swap on ``_last_broadcast`` MUST run under the lock.

    Reproduces the exact race the fix closes: while one thread is
    inside :func:`refresh_registry` (mid-scan, before the CAS), a
    second thread enters the same function and would — under the OLD,
    outside-the-lock implementation — pass the identical "changed"
    check and emit a duplicate ``seaCommands`` broadcast.  A slow
    ``_scan_folder`` (patched with a barrier + sleep) forces the two
    executions to overlap; the assertion below then verifies exactly
    ONE broadcast is fired.
    """
    import threading as _th
    import time as _time

    folder = tmp_path / "seas"
    _touch_sea(folder, "onlyone")
    _write_seas_md([str(folder)])

    seen: list[tuple[str, ...]] = []
    seen_lock = _th.Lock()

    def _cb(cmds: list[str]) -> None:
        with seen_lock:
            seen.append(tuple(cmds))

    sea_commands.subscribe(_cb)
    # Prime the module with an empty registry so both rescans below
    # observe the SAME "old" snapshot when they enter refresh.
    with sea_commands._lock:
        sea_commands._registry.clear()
        sea_commands._last_broadcast = ()

    # Force an overlap: the first refresh's _scan_folder call blocks
    # on a barrier until the SECOND refresh also enters the function.
    # Under the OLD (compare-and-set outside the lock) code, both
    # would then race past the identical "old snapshot" check and
    # each emit a broadcast; under the current (CAS under lock) code
    # exactly one broadcast fires because the second thread waits for
    # the lock and then observes the already-updated snapshot.
    entered = _th.Barrier(2, timeout=5.0)
    real_scan = sea_commands._scan_folder

    def _slow_scan(folder_arg):  # type: ignore[no-untyped-def]
        try:
            entered.wait()
        except _th.BrokenBarrierError:
            pass
        _time.sleep(0.05)
        return real_scan(folder_arg)

    monkeypatch.setattr(sea_commands, "_scan_folder", _slow_scan)

    start = _th.Event()

    def _worker() -> None:
        start.wait()
        sea_commands.refresh_registry()

    workers = [_th.Thread(target=_worker) for _ in range(2)]
    for w in workers:
        w.start()
    start.set()
    for w in workers:
        w.join(timeout=5)
    # Exactly one broadcast fires: the CAS under lock rejects the
    # second racer's "changed" claim.
    assert len(seen) == 1, f"expected 1 broadcast, saw {len(seen)}: {seen}"
    assert "onlyone" in seen[0]


def test_get_command_rescans_on_miss(tmp_path: Path) -> None:
    """A cache miss must trigger a rescan so a just-added SEA is found."""
    folder = tmp_path / "seas"
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    assert sea_commands.get_command("later") is None
    _touch_sea(folder, "later")
    resolved = sea_commands.get_command("later")
    assert resolved is not None
    assert resolved.name == "later_sea.py"
