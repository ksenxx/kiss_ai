# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the SEA slash-command registry.

Covers the module in :mod:`kiss.agents.sorcar.sea_commands`:

* the registry precedence rule (bundled ``third_party_agents`` beats
  every user folder, SEAS.md folders lower in the file override
  higher ones, and every user folder beats the bundled ``seas``
  package);
* live-reload behaviour when ``~/.kiss/SEAS.md`` or a watched folder
  changes;
* the slash-command splitter that turns ``/xxx text`` into the SEA to
  run and its task text, and the ``settings()`` resolver the daemon
  and dispatcher read.

Every test isolates :mod:`kiss.agents.sorcar.sea_commands` via a
``_reset_for_tests()`` fixture so a subscriber leaked from one test
cannot fire during another.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.core.config import kiss_home
from kiss.tests.conftest import is_root, posix_only


@pytest.fixture(autouse=True)
def _reset_sea_commands() -> Iterator[None]:
    """Drop the module's in-memory state before and after each test.

    Also removes the ``SEAS.md`` a test wrote into the session-wide
    ``$KISS_HOME``; a leftover file would redirect ``/merge`` (and any
    other bundled command) for every later test in the same process.
    """
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()
    (kiss_home() / "SEAS.md").unlink(missing_ok=True)


def _touch_sea(folder: Path, name: str) -> Path:
    """Create a stub SEA ``<name>/<name>_sea.py`` under *folder* and return it."""
    path = folder / name / f"{name}_sea.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# stub SEA for tests\n", encoding="utf-8")
    return path


def _write_seas_md(lines: list[str]) -> None:
    """Overwrite ``~/.kiss/SEAS.md`` with *lines* (one per row)."""
    home = kiss_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "SEAS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_bundled_third_party_commands_are_registered() -> None:
    """Every ``third_party_agents/<x>/<x>_sea.py`` is exposed as ``/<x>``.

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
    assert slack_path.parent.name == "slack"
    assert slack_path.parents[1].name == "third_party_agents"


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
    ``slack/slack_sea.py`` MUST be ignored so the daemon-installed agent
    always wins.
    """
    user = tmp_path / "override"
    _touch_sea(user, "slack")
    _write_seas_md([str(user)])
    sea_commands.refresh_registry()
    slack_path = sea_commands.get_command("slack")
    assert slack_path is not None
    assert slack_path.parents[1].name == "third_party_agents"


def test_seas_md_beats_bundled_seas_dir(tmp_path: Path) -> None:
    """The bundled ``seas/`` package has the LOWEST precedence.

    Without any ``SEAS.md`` entry ``/merge`` resolves to the bundled
    ``kiss/agents/seas/merge/merge_sea.py``.  Once a user folder listed in
    ``SEAS.md`` ships its own ``merge/merge_sea.py``, that copy MUST win.
    """
    _write_seas_md([])
    sea_commands.refresh_registry()
    bundled = sea_commands.get_command("merge")
    assert bundled is not None
    assert bundled.parent.name == "merge"
    assert bundled.parents[1].name == "seas"

    user = tmp_path / "override"
    user_merge = _touch_sea(user, "merge")
    _write_seas_md([str(user)])
    sea_commands.refresh_registry()
    assert sea_commands.get_command("merge") == user_merge.resolve()


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
    assert resolved.resolve() == (folder / "envcmd" / "envcmd_sea.py").resolve()


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
    """Only ``<x>/<x>_sea.py`` folders surface; anything else never does.

    A loose ``xxx_sea.py`` at the top of the folder (the old flat
    layout), a folder whose script is not named after it, a plain file
    and a plain sub-folder are all skipped.  Also confirms that folder
    names producing an invalid command name — a name that contains
    characters outside ``[A-Za-z0-9_-]`` — are silently skipped,
    because the parser and the autocomplete both refuse them and
    surfacing them as commands would mean advertising a slash command
    the user cannot type.
    """
    folder = tmp_path / "seas"
    _touch_sea(folder, "public")
    (folder / "loose_sea.py").write_text("", encoding="utf-8")
    (folder / "notasea.py").write_text("", encoding="utf-8")
    (folder / "plaindir").mkdir()
    (folder / "wrongname").mkdir()
    (folder / "wrongname" / "other_sea.py").write_text("", encoding="utf-8")
    _touch_sea(folder, "foo.bar")
    _touch_sea(folder, "with space")
    _write_seas_md([str(folder)])
    commands = sea_commands.refresh_registry()
    assert "public" in commands
    assert "loose" not in commands
    assert "notasea" not in commands
    assert "plaindir" not in commands
    assert "wrongname" not in commands
    assert "other" not in commands
    assert "foo.bar" not in commands
    assert "with space" not in commands


def test_private_underscore_prefixed_seas_are_included(tmp_path: Path) -> None:
    """Every ``<x>/<x>_sea.py`` in a watched folder becomes a command.

    ``_helper/_helper_sea.py`` IS a command ``/_helper`` — the
    underscore is a valid character for a command name.
    """
    folder = tmp_path / "seas"
    _touch_sea(folder, "_helper")
    _write_seas_md([str(folder)])
    commands = sea_commands.refresh_registry()
    assert "_helper" in commands


def test_slash_command_task_for_registered_command(tmp_path: Path) -> None:
    """``/<name> <text>`` is split into the trailing text and the SEA to run.

    The daemon runs the SEA directly on the trailing text (no
    ``run_agent`` directive is composed), so the task text comes back
    verbatim and the path is the registered file.
    """
    folder = tmp_path / "seas"
    sea_path = _touch_sea(folder, "myslack")
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    result = sea_commands.slash_command_task(
        '/myslack post "hi there" to #general',
    )
    assert result is not None
    task_text, resolved = result
    assert resolved == sea_path.resolve()
    assert task_text == 'post "hi there" to #general'
    assert "run_agent" not in task_text


def test_slash_command_task_is_none_for_unknown_command() -> None:
    """A slash prefix that does not match any SEA runs as an ordinary prompt."""
    sea_commands.refresh_registry()
    assert sea_commands.slash_command_task("/definitelynot hi") is None


def test_slash_command_task_is_none_without_trailing_text(tmp_path: Path) -> None:
    """``/xxx`` alone (no task text) and ``/xxx help`` are not SEA runs.

    The bare command has no task to run; ``help`` is answered by
    :func:`sea_commands.help_text_if_command` instead.
    """
    folder = tmp_path / "seas"
    _touch_sea(folder, "solo")
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    assert sea_commands.slash_command_task("/solo") is None
    assert sea_commands.slash_command_task("/solo   ") is None
    assert sea_commands.slash_command_task("/solo help") is None
    assert sea_commands.slash_command_task("/solo HELP") is None
    assert sea_commands.slash_command_task("/solo help me") is not None


def test_slash_command_task_ignores_prompts_that_are_not_slash_commands() -> None:
    """Plain prompts, chat text with a slash mid-line, etc. pass through."""
    sea_commands.refresh_registry()
    assert sea_commands.slash_command_task("hello world") is None
    assert sea_commands.slash_command_task("\n/slack hi") is None
    assert sea_commands.slash_command_task("say /slack") is None
    assert sea_commands.slash_command_task("") is None
    assert sea_commands.slash_command_task("/") is None
    assert sea_commands.slash_command_task(None) is None  # type: ignore[arg-type]


def test_slash_command_task_requires_whitespace_after_command_name(
    tmp_path: Path,
) -> None:
    """``/slackfoo bar`` must NOT match ``/slack`` — the boundary is a space."""
    folder = tmp_path / "seas"
    _touch_sea(folder, "slk")
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    assert sea_commands.slash_command_task("/slkextra text") is None
    assert sea_commands.slash_command_task("/slk text") is not None


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


def _alive_watchers() -> list[threading.Thread]:
    return [
        t for t in threading.enumerate()
        if t.name == "kiss-sea-registry-watcher" and t.is_alive()
    ]


@posix_only("FIFO-blocked read")
def test_stopped_watcher_mid_scan_does_not_resume_beside_its_successor() -> None:
    """A poller stopped while blocked in a scan must exit once the scan ends.

    ``stop_registry_watcher`` unpublishes the thread and joins it with a
    timeout; a poller still inside ``refresh_registry`` outlives a short
    join.  With one module-wide stop event, the next ``start`` cleared
    that event and the old poller resumed its loop beside the new one,
    two pollers for ever.  Each poller now owns its stop event.

    The old poller is held inside its scan by turning ``SEAS.md`` into a
    FIFO with no writer: ``read_text`` blocks until this test opens the
    write end and closes it (an empty file for the reader).
    """
    config = kiss_home() / "SEAS.md"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("", encoding="utf-8")
    sea_commands.start_registry_watcher(poll_interval=0.1)
    old = sea_commands._watcher_thread
    assert old is not None
    config.unlink()
    os.mkfifo(config)
    writer = None
    try:
        # A non-blocking open of the write end succeeds only once a
        # reader (the poller's read_text) is blocked on the FIFO.
        deadline = time.monotonic() + 10
        while writer is None:
            try:
                writer = os.open(config, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:
                assert time.monotonic() < deadline, "poller never read SEAS.md"
                time.sleep(0.01)
        config.unlink()
        config.write_text("", encoding="utf-8")
        sea_commands.stop_registry_watcher(timeout=0.05)
        assert old.is_alive(), "the old poller must still be inside its scan"
        sea_commands.start_registry_watcher(poll_interval=0.1)
        new = sea_commands._watcher_thread
        assert new is not None and new is not old
    finally:
        if writer is not None:
            os.close(writer)  # EOF: the old poller's read_text returns
    old.join(timeout=5)
    assert not old.is_alive(), "REGRESSION: stopped poller resumed beside its successor"
    assert _alive_watchers() == [new]
    sea_commands.stop_registry_watcher()
    assert _alive_watchers() == []


def test_watcher_start_and_stop_race_safely(tmp_path: Path) -> None:
    """Overlapping start/stop calls never raise and never leak a poller.

    ``start_registry_watcher`` used to publish the thread, release the
    lock, run the synchronous first rescan and only then start the
    thread.  A ``stop_registry_watcher`` in that window joined an
    unstarted thread (``RuntimeError``) and a second start saw a
    not-alive thread and spawned a second poller that nothing could
    stop.  Hammering start and stop from several threads must end
    with no error and at most one live poller, and none after the
    final stop.
    """
    folder = tmp_path / "one"
    _touch_sea(folder, "alpha")
    _write_seas_md([str(folder)])
    errors: list[BaseException] = []

    def starter() -> None:
        for _ in range(25):
            try:
                sea_commands.start_registry_watcher(poll_interval=0.1)
            except BaseException as exc:  # noqa: BLE001 — recorded, asserted below
                errors.append(exc)

    def stopper() -> None:
        for _ in range(25):
            try:
                sea_commands.stop_registry_watcher(timeout=5)
            except BaseException as exc:  # noqa: BLE001 — recorded, asserted below
                errors.append(exc)

    threads = [threading.Thread(target=starter, name=f"starter{i}") for i in range(3)]
    threads += [threading.Thread(target=stopper, name=f"stopper{i}") for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads)
    assert errors == [], f"start/stop raced into an exception: {errors!r}"

    # Pollers that were stopped exit within one poll interval; a leaked
    # second poller would survive forever.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and len(_alive_watchers()) > 1:
        time.sleep(0.05)
    assert len(_alive_watchers()) <= 1, "a second registry poller was leaked"
    sea_commands.stop_registry_watcher(timeout=5)
    assert _alive_watchers() == [], "the registry poller survived stop_registry_watcher"
    assert sea_commands._watcher_thread is None


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


_DATACLASS_SEA = '''\
"""SEA whose module-level dataclass needs ``sys.modules[__name__]``."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from kiss.agents.seas.base.base_sea import BaseSea


@dataclass
class Verdict:
    """String annotations: resolved through ``sys.modules[__module__]``."""

    worktree: bool
    note: typing.ClassVar[str] = "relay stays on the real checkout"


class Sea(BaseSea):
    def settings(self, settings):
        hints = typing.get_type_hints(Verdict)
        assert hints == {"worktree": bool, "note": typing.ClassVar[str]}, hints
        return settings | {"use_worktree": Verdict(worktree=False).worktree, "auto_commit": True}


'''


def test_dataclass_sea_with_future_annotations_loads(tmp_path: Path) -> None:
    """A ``@dataclass`` SEA under ``from __future__ import annotations`` imports.

    ``dataclasses`` resolves string annotations through
    ``sys.modules[cls.__module__].__dict__`` while the class body runs,
    so the one loader (``sea_settings.execute_python_file``) registers
    the module before executing it.  ``settings()`` also calls
    ``typing.get_type_hints`` after import, which needs the entry to
    still be there: the module stays registered under a per-path name,
    and a re-execution of the same file replaces it.
    """
    folder = tmp_path / "seas"
    sea = folder / "verdict" / "verdict_sea.py"
    sea.parent.mkdir(parents=True)
    sea.write_text(_DATACLASS_SEA, encoding="utf-8")
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()

    assert sea_commands.slash_command_task("/verdict go") == ("go", sea.resolve())
    settings = sea_commands.sea_settings(sea)
    assert settings == {"use_worktree": False, "auto_commit": True}
    loaded = sea_commands.load_sea(sea)
    name = type(loaded).__module__
    assert name.startswith("_kiss_sea_verdict_sea_")
    module = sys.modules[name]
    assert module.__dict__["Sea"] is type(loaded)
    assert module.__file__ == str(sea)
    assert loaded.path == sea
    assert module.__dict__["Verdict"].note == "relay stays on the real checkout"
    # One module per script file, not one per run.
    again = sea_commands.load_sea(sea)
    assert type(again).__module__ == name and sys.modules[name].__dict__["Sea"] is type(again)
    assert len([n for n in sys.modules if n.startswith("_kiss_sea_verdict_sea_")]) == 1


def test_failed_sea_import_leaves_no_sys_modules_entry(tmp_path: Path) -> None:
    """An SEA raising at import is reported and unregistered again."""
    sea = tmp_path / "boom_sea.py"
    sea.write_text("raise SystemExit(3)\n", encoding="utf-8")
    with pytest.raises(sea_commands.SeaError, match="SystemExit: 3") as info:
        sea_commands.sea_settings(sea)
    assert isinstance(info.value.__cause__, SystemExit)
    assert not [n for n in sys.modules if n.startswith("_kiss_sea_boom_sea_")]


_SLOW_SEA = '''\
"""Same-stem SEA whose settings() resolves a forward reference after import."""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass


from kiss.agents.seas.base.base_sea import BaseSea


@dataclass
class {cls}:
    parent: {cls} | None = None


class Sea(BaseSea):
    def settings(self, settings: dict) -> dict:
        time.sleep({delay})
        hints = typing.get_type_hints({cls})
        assert hints["parent"] == ({cls} | None), hints
        return settings | {{"use_worktree": False}}
'''


def test_concurrent_same_stem_loads_do_not_clobber_each_other(
    tmp_path: Path,
) -> None:
    """Two ``shared_sea.py`` files evaluated at once each keep their own module.

    Task threads evaluate ``settings()`` concurrently.  ``OnlyA``'s
    forward reference is resolved through ``sys.modules[__module__]``
    *after* import, while a second same-stem SEA is being loaded in
    another thread: a stem-keyed entry would by then point at the other
    file's namespace and ``get_type_hints`` would raise ``NameError``.
    """
    sea_a = tmp_path / "folder_a" / "shared_sea.py"
    sea_b = tmp_path / "folder_b" / "shared_sea.py"
    sea_a.parent.mkdir()
    sea_b.parent.mkdir()
    sea_a.write_text(_SLOW_SEA.format(cls="OnlyA", delay=0.4), encoding="utf-8")
    sea_b.write_text(_SLOW_SEA.format(cls="OnlyB", delay=0.0), encoding="utf-8")

    results: dict[str, object] = {}

    def _evaluate(label: str, sea: Path) -> None:
        try:
            results[label] = sea_commands.sea_settings(sea)["use_worktree"] is False
        except sea_commands.SeaError as exc:
            results[label] = exc

    thread_a = threading.Thread(target=_evaluate, args=("a", sea_a))
    thread_a.start()
    time.sleep(0.15)  # A has imported and is inside its sleep
    _evaluate("b", sea_b)
    thread_a.join(timeout=10)
    assert results == {"a": True, "b": True}, results
    # Same stem, different folders: two distinct module entries.
    assert len({n for n in sys.modules if n.startswith("_kiss_sea_shared_sea_")}) == 2


_DESCRIBED_SEA = '''\
"""SEA with the mandatory description() getter."""

from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def description(self):
        return "  Echoes the task back; use it as /echo <text>.  "

    def settings(self, settings):
        return settings | {"use_worktree": False}


'''


def test_help_returns_stripped_description(tmp_path: Path) -> None:
    """``/xxx help`` (any letter case) yields the SEA's stripped ``description()``.

    Anything but the bare word ``help`` is the SEA's task text, an
    unknown command and a non-command prompt yield ``None``, and the
    same prompt is never ALSO run as an SEA task by the caller because
    the task runner checks help first.
    """
    folder = tmp_path / "seas"
    sea = _touch_sea(folder, "echo")
    sea.write_text(_DESCRIBED_SEA, encoding="utf-8")
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()

    expected = "Echoes the task back; use it as /echo <text>."
    assert sea_commands.help_text_if_command("/echo help") == expected
    assert sea_commands.help_text_if_command("/echo   HELP ") == expected
    assert sea_commands.sea_description(sea_commands.load_sea(sea)) == expected
    assert sea_commands.help_text_if_command("/echo help me") is None
    assert sea_commands.help_text_if_command("/echo") is None
    assert sea_commands.help_text_if_command("/unknown help") is None
    assert sea_commands.help_text_if_command("help") is None
    assert sea_commands.help_text_if_command(None) is None  # type: ignore[arg-type]


def test_help_reports_missing_or_broken_description(tmp_path: Path) -> None:
    """A SEA without a usable ``description()`` fails ``/xxx help`` with a diagnostic."""
    folder = tmp_path / "seas"
    _write_seas_md([str(folder)])

    missing = _touch_sea(folder, "nodesc")  # no SEA class at all
    sea_commands.refresh_registry()
    with pytest.raises(sea_commands.SeaError, match="exactly one subclass.*found none"):
        sea_commands.help_text_if_command("/nodesc help")

    missing.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    pass
""", encoding="utf-8")
    with pytest.raises(sea_commands.SeaError, match="description.*non-empty string"):
        sea_commands.help_text_if_command("/nodesc help")

    # A class attribute shadowing the method: a SEA diagnostic, not a crash.
    missing.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    description = 'not callable'
""", encoding="utf-8")
    with pytest.raises(sea_commands.SeaError, match="description.*must be a method, got str"):
        sea_commands.sea_description(sea_commands.load_sea(missing))

    missing.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return 42
""", encoding="utf-8")
    with pytest.raises(sea_commands.SeaError, match="must return a string, got int"):
        sea_commands.sea_description(sea_commands.load_sea(missing))

    missing.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return '   '
""", encoding="utf-8")
    with pytest.raises(sea_commands.SeaError, match="must return a non-empty string"):
        sea_commands.sea_description(sea_commands.load_sea(missing))

    missing.write_text("""
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        raise KeyError('k')
""", encoding="utf-8")
    with pytest.raises(sea_commands.SeaError, match="KeyError") as info:
        sea_commands.sea_description(sea_commands.load_sea(missing))
    assert isinstance(info.value.__cause__, KeyError)

    missing.write_text("raise SystemExit(2)\n", encoding="utf-8")
    with pytest.raises(sea_commands.SeaError, match="SystemExit: 2"):
        sea_commands.load_sea(missing)


def test_every_bundled_sea_has_a_description() -> None:
    """Every registered bundled SEA (``seas/`` and ``third_party_agents/``) describes itself.

    ``description()`` must return one non-empty sentence, so ``/xxx
    help`` works for every command shipped with the package.
    """
    _write_seas_md([])
    commands = sea_commands.refresh_registry()
    assert len(commands) >= 58
    assert "cron" in commands
    for name in commands:
        path = sea_commands.get_command(name)
        assert path is not None
        if name in sea_commands.BUILTIN_COMMANDS:
            assert path.name == "cron_agent.py", path
        else:
            assert path.parent.name == name, path
        text = sea_commands.sea_description(sea_commands.load_sea(path))
        assert text.rstrip(".").strip(), name


_ROUTER_TRUE = """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return 'r'

    def register_as_model(self):
        return 1 == 1
"""
_ROUTER_FALSE = """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return 'r'

    def register_as_model(self):
        return 1 == 2
"""
"""Same-length sources: an edit between them keeps the file size."""
assert len(_ROUTER_TRUE) == len(_ROUTER_FALSE)


def _write_router(folder: Path, name: str, source: str) -> Path:
    """Create ``<folder>/<name>/<name>_sea.py`` with *source* and return it."""
    path = folder / name / f"{name}_sea.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Bytes, not text mode: the same-size tests compare st_size to
    # len(source), which CRLF translation on Windows would break.
    path.write_bytes(source.encode("utf-8"))
    return path


def test_model_seas_rereads_an_edit_within_the_same_second(tmp_path: Path) -> None:
    """A same-size edit whose mtime moves by one filesystem tick is seen.

    The bytecode cache keys staleness on whole-second mtime and size, so
    an import-based loader would re-run the stale ``.pyc``; the SEA loader
    compiles the source directly.  The tick is 100 ns, NTFS's resolution
    (a 1 ns bump rounds back to the old stamp there).
    """
    folder = tmp_path / "seas"
    router = _write_router(folder, "flip", _ROUTER_TRUE)
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    assert sea_commands.model_sea("flip") == router
    stamp = router.stat().st_mtime_ns
    router.write_bytes(_ROUTER_FALSE.encode("utf-8"))
    os.utime(router, ns=(stamp + 100, stamp + 100))
    assert router.stat().st_size == len(_ROUTER_TRUE)
    assert sea_commands.model_sea("flip") is None
    assert "flip" not in sea_commands.model_seas()


def test_model_seas_rereads_an_atomic_replacement_with_the_same_mtime(tmp_path: Path) -> None:
    """A file swapped in with identical size and mtime (new inode) is re-read."""
    folder = tmp_path / "seas"
    router = _write_router(folder, "swap", _ROUTER_TRUE)
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    assert sea_commands.model_sea("swap") == router
    st = router.stat()
    replacement = router.with_name("swap_sea.py.new")
    replacement.write_bytes(_ROUTER_FALSE.encode("utf-8"))
    os.utime(replacement, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(replacement, router)
    assert router.stat().st_mtime_ns == st.st_mtime_ns
    assert router.stat().st_ino != st.st_ino
    assert sea_commands.model_sea("swap") is None


@posix_only("chmod-based read denial")
@pytest.mark.skipif(is_root(), reason="root reads unreadable files")
def test_unreadable_sea_does_not_break_the_model_picker_registry(tmp_path: Path) -> None:
    """A SEA without read permission is skipped, not raised through ``model_seas()``."""
    folder = tmp_path / "seas"
    router = _write_router(folder, "ok", _ROUTER_TRUE)
    locked = _write_router(folder, "locked", _ROUTER_TRUE)
    locked.chmod(0)
    try:
        _write_seas_md([str(folder)])
        sea_commands.refresh_registry()
        seas = sea_commands.model_seas()
        assert seas["ok"] == router and "locked" not in seas
        assert {"autorouter", "bestrouter"} <= set(seas)
    finally:
        locked.chmod(0o644)


def test_model_sea_finds_a_router_installed_after_the_registry_was_built(
    tmp_path: Path,
) -> None:
    """Without a watcher, a miss rescans the folders like ``get_command`` does."""
    folder = tmp_path / "seas"
    _write_router(folder, "early", _ROUTER_TRUE)
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    assert sea_commands.model_sea("late") is None
    late = _write_router(folder, "late", _ROUTER_TRUE)
    assert sea_commands.model_sea("late") == late
