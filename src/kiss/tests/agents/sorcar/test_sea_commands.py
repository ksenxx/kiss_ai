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

Every test isolates :mod:`kiss.agents.sorcar.sea_commands` via an
``isolated_sea_registry()`` fixture so a subscriber leaked from one test
cannot fire during another.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_commands, sea_settings
from kiss.core.config import kiss_home
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.agents.sorcar.conftest import isolated_sea_registry
from kiss.tests.conftest import is_root, posix_only


@pytest.fixture(autouse=True)
def _reset_sea_commands() -> Iterator[None]:
    """Drop the module's in-memory state before and after each test.

    Also removes the ``SEAS.md`` a test wrote into the session-wide
    ``$KISS_HOME``; a leftover file would redirect ``/merge`` (and any
    other bundled command) for every later test in the same process.
    """
    with isolated_sea_registry():
        yield


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
    """A stale SEAS.md entry must never crash the registry rebuild.

    Nor a line no path can be: one with a NUL byte (what a file cut
    short by a crash is filled with), in a folder or a ``~user`` name.
    """
    good = tmp_path / "good"
    _touch_sea(good, "livecmd")
    ghost = tmp_path / "does-not-exist"
    _write_seas_md([str(ghost), "\0\0\0", f"{tmp_path}/bad\0folder", "~bad\0user/seas", str(good)])
    commands = sea_commands.refresh_registry()
    assert "livecmd" in commands and "slack" in commands


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
    if os.name == "posix":  # a name ending in a newline, which ``$`` would let through
        _touch_sea(folder, "newline\n")
    _write_seas_md([str(folder)])
    commands = sea_commands.refresh_registry()
    assert "newline\n" not in commands
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

    A stock SEA passes the empty task through, so the bare command has
    no task to run; ``help`` is answered by
    :func:`sea_commands.help_text_if_command` instead.
    """
    folder = tmp_path / "seas"
    sea_path = _touch_sea(folder, "solo")
    sea_path.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\nclass SoloSea(BaseSea):\n    pass\n",
        encoding="utf-8",
    )
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    assert sea_commands.slash_command_task("/solo") is None
    assert sea_commands.slash_command_task("/solo   ") is None
    assert sea_commands.slash_command_task("/solo help") is None
    assert sea_commands.slash_command_task("/solo HELP") is None
    assert sea_commands.slash_command_task("/solo help me") is not None


def test_slash_command_task_runs_bare_command_with_default_task(tmp_path: Path) -> None:
    """``/xxx`` alone runs a SEA whose ``prompt()`` gives the empty task text.

    The task text comes back empty (the daemon's ``apply_sea`` lets the
    SEA's ``prompt()`` supply the text); a broken SEA raises instead of
    silently running the bare command as an ordinary prompt.
    """
    folder = tmp_path / "seas"
    sea_path = _touch_sea(folder, "daily")
    sea_path.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\nclass DailySea(BaseSea):\n"
        "    def prompt(self, task):\n        return task or 'do the daily chores'\n",
        encoding="utf-8",
    )
    broken = _touch_sea(folder, "broken")
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    assert sea_commands.slash_command_task("/daily") == ("", sea_path.resolve())
    assert sea_commands.slash_command_task("/daily   ") == ("", sea_path.resolve())
    assert sea_commands.slash_command_task("/daily help") is None
    assert sea_commands.slash_command_task("/daily now") == ("now", sea_path.resolve())
    with pytest.raises(sea_commands.SeaError):
        sea_commands.slash_command_task("/broken")
    assert sea_commands.slash_command_task("/broken now") == ("now", broken.resolve())


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


def test_a_lookup_miss_inside_a_subscriber_leaves_later_subscribers_on_the_newest_list(
    tmp_path: Path,
) -> None:
    """A subscriber may look a command up (``_notify_lock`` is reentrant): a miss rescans
    and delivers the newer list to every subscriber before the callback returns.  The
    interrupted delivery then went on with its older list, and the subscribers after that
    one stayed on it for good (the next rescan finds nothing new to broadcast)."""
    folder = tmp_path / "seas"
    _write_seas_md([str(folder)])
    seen: list[list[str]] = []

    def install_and_look_up(names: list[str]) -> None:
        if "later" not in names:  # the sources change while the older list is delivered
            _touch_sea(folder, "later")
            sea_commands.get_command("later")

    sea_commands.subscribe(install_and_look_up)
    sea_commands.subscribe(seen.append)
    sea_commands.refresh_registry()
    assert seen == [sea_commands.list_commands()] and "later" in seen[0], seen


_DATACLASS_SEA = '''\
"""SEA whose module-level dataclass needs ``sys.modules[__name__]``."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from kiss.agents.seas.base.base_sea import BaseSea


@dataclass
class Verdict:
    """String annotations: resolved through ``sys.modules[__module__]``."""

    memory: bool
    note: typing.ClassVar[str] = "relay stays on the real checkout"


class Sea(BaseSea):
    def settings(self, settings):
        hints = typing.get_type_hints(Verdict)
        assert hints == {"memory": bool, "note": typing.ClassVar[str]}, hints
        return settings | {"use_memory": Verdict(memory=False).memory, "use_web_tools": True}


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
    assert settings == {"use_memory": False, "use_web_tools": True}
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
        return settings | {{"use_memory": False}}
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
            results[label] = sea_commands.sea_settings(sea)["use_memory"] is False
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


_GATED_SEA = '''\
"""SEA whose import blocks until a gate file appears (at most 20 s)."""

import os
import time

from kiss.agents.seas.base.base_sea import BaseSea

with open({started!r}, "a") as _log:
    _log.write("x")
for _ in range(400):
    if os.path.exists({gate!r}):
        break
    time.sleep(0.05)


class Sea(BaseSea):
    def settings(self, settings):
        return settings | {{"use_memory": False}}
'''


def test_slow_sea_import_does_not_block_other_files(tmp_path: Path) -> None:
    """One file's import-time code holds up only re-executions of that file.

    The daemon loads SEAs from many threads: a task's ``seaPath``, the
    ``ready`` fan-out's model-picker probe, the slash-command registry.
    A SEA that is slow to import (heavy dependencies, a script that
    blocks) used to hold a global lock for the whole execution, so
    every other loader — and with it every client's ``ready`` — waited
    for it.  Executions of the SAME file still serialise, so a failed
    re-execution restores the entry that is current when it fails.
    """
    gate = tmp_path / "gate"
    started_log = tmp_path / "started"
    slow = tmp_path / "slow_sea.py"
    quick = tmp_path / "quick_sea.py"
    slow.write_text(
        _GATED_SEA.format(gate=str(gate), started=str(started_log)), encoding="utf-8",
    )
    quick.write_text(
        _GATED_SEA.format(gate=str(tmp_path), started=str(tmp_path / "quick_started")),
        encoding="utf-8",
    )

    done: dict[str, float] = {}

    def _load(label: str, sea: Path) -> None:
        sea_commands.sea_settings(sea)
        done[label] = time.monotonic()

    slow_first = threading.Thread(target=_load, args=("slow", slow), daemon=True)
    slow_again = threading.Thread(target=_load, args=("slow_again", slow), daemon=True)
    def _marker(text: str) -> bool:
        return started_log.exists() and started_log.read_text(encoding="utf-8") == text

    try:
        slow_first.start()
        _wait_for(lambda: _marker("x"))  # inside its import-time wait
        slow_again.start()
        # The second loader is queued behind the first (it is in the
        # loader's wait for the file) before the other file is tried.
        _wait_for(lambda: sea_settings._waited_module(slow_again.ident) is not None)
        started = time.monotonic()
        _load("quick", quick)
        assert done["quick"] - started < 2.0, "a different file waited for the slow import"
        assert done.keys() == {"quick"}, "the slow file finished before its gate opened"
        # The second execution of the SAME file has not begun.
        assert _marker("x")
    finally:
        gate.write_text("", encoding="utf-8")
        for thread in (slow_first, slow_again):
            if thread.ident is not None:
                thread.join(timeout=20)
    assert done.keys() == {"quick", "slow", "slow_again"}, done
    assert _marker("xx")


def _wait_for(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    """Poll *condition* until true, failing after *timeout* seconds."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.02)


_CROSS_LOADING_SEA = '''\
"""SEA that derives from another SEA file, which in turn derives from this one."""

import os
import time

from kiss.agents.sorcar.sea_commands import sea_class

with open({started!r}, "a+") as _log:
    _log.write("x")
    _log.seek(0)
    if len(_log.read()) > 2:
        # The surviving thread re-executes the cycle on its own thread
        # until the recursion limit; cut that short.
        raise RuntimeError("loaded again")
# Both files are being imported before either loads the other.
for _ in range(400):
    if os.path.exists({other_started!r}):
        break
    time.sleep(0.05)


class Sea(sea_class({other!r})):
    pass
'''


def test_circular_sea_imports_across_threads_fail_instead_of_deadlocking(
    tmp_path: Path,
) -> None:
    """Two threads loading files that load each other both get an error.

    Each file's import waits for the other's import to have started,
    then loads the other: the thread that closes the cycle is refused
    (the other file's thread is waiting for this one), the other thread
    then re-executes the cycle on its own thread (re-entrant) until
    the fixture stops it.  Neither thread hangs.
    """
    sea_a = tmp_path / "a_sea.py"
    sea_b = tmp_path / "b_sea.py"
    started_a = tmp_path / "a_started"
    started_b = tmp_path / "b_started"
    sea_a.write_text(
        _CROSS_LOADING_SEA.format(
            started=str(started_a), other_started=str(started_b), other=str(sea_b),
        ),
        encoding="utf-8",
    )
    sea_b.write_text(
        _CROSS_LOADING_SEA.format(
            started=str(started_b), other_started=str(started_a), other=str(sea_a),
        ),
        encoding="utf-8",
    )
    errors: dict[str, str] = {}

    def _load(label: str, sea: Path) -> None:
        try:
            sea_commands.sea_settings(sea)
        except sea_commands.SeaError as exc:
            errors[label] = str(exc)
        else:
            errors[label] = "no error"

    threads = [
        threading.Thread(target=_load, args=(label, sea), daemon=True)
        for label, sea in (("a", sea_a), ("b", sea_b))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "circular SEA loads deadlocked"
    assert errors.keys() == {"a", "b"}, errors
    assert any("circular SEA imports" in message for message in errors.values()), errors
    assert all("failed to import" in message for message in errors.values()), errors


def test_a_sea_load_and_an_ordinary_import_closing_one_cycle_fail_instead_of_deadlocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cycle split between Python's import lock and a SEA's execution slot is refused.

    Thread B imports the plain module ``helper`` (Python's lock of that
    module held), whose body derives a class from SEA ``a`` through
    ``sea_class``; thread A loads ``a`` (its execution slot held), whose
    body imports ``helper``.  A waits for B's import lock, B for A's
    execution slot: a cycle that neither Python's deadlock detection nor
    a cycle check following only the execution-slot waits can see.  The
    thread closing the cycle (B, which re-checks every second) is refused
    with the circular-import error, so both threads finish.
    """
    helper = f"helper_{os.getpid()}_{tmp_path.name}".replace("-", "_")
    sea_a = tmp_path / "a_sea.py"
    started_a, started_b = tmp_path / "a_started", tmp_path / "b_started"
    (tmp_path / f"{helper}.py").write_text(
        "import os\n"
        "import time\n"
        "from pathlib import Path\n\n"
        f"Path({str(started_b)!r}).touch()\n"
        "for _ in range(1000):  # until thread A owns a_sea.py's execution slot\n"
        f"    if os.path.exists({str(started_a)!r}):\n"
        "        break\n"
        "    time.sleep(0.01)\n"
        "from kiss.agents.sorcar.sea_commands import sea_class\n\n\n"
        f"class Helper(sea_class({str(sea_a)!r})):\n"
        "    pass\n",
        encoding="utf-8",
    )
    sea_a.write_text(
        "from pathlib import Path\n\n"
        f"Path({str(started_a)!r}).touch()\n"
        f"from {helper} import Helper\n\n\n"
        "class Sea(Helper):\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, helper, raising=False)
    errors: dict[str, str] = {}

    def _import_helper() -> None:
        try:
            __import__(helper)
        except sea_commands.SeaError as exc:
            errors["b"] = str(exc)
        else:
            errors["b"] = "no error"

    def _load_a() -> None:
        try:
            sea_commands.load_sea(sea_a)
        except sea_commands.SeaError as exc:
            errors["a"] = str(exc)
        else:
            errors["a"] = "no error"

    thread_b = threading.Thread(target=_import_helper, name="ordinary-import", daemon=True)
    thread_a = threading.Thread(target=_load_a, name="sea-load", daemon=True)
    thread_b.start()
    deadline = time.monotonic() + 10
    while not started_b.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started_b.exists(), "the ordinary import never started"
    thread_a.start()
    thread_a.join(timeout=30)
    thread_b.join(timeout=30)
    assert not thread_a.is_alive() and not thread_b.is_alive(), "the mixed cycle deadlocked"
    assert errors.keys() == {"a", "b"}, errors
    assert "circular SEA imports" in errors["b"], errors
    assert "failed to import" in errors["a"], errors


_DESCRIBED_SEA = '''\
"""SEA with the mandatory description() getter."""

from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def description(self):
        return "  Echoes the task back; use it as /echo <text>.  "

    def settings(self, settings):
        return settings | {"use_memory": False}


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


_STATIC_DESCRIPTION_SEA = '''
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    @staticmethod
    def description():
        return "Describes a SEA."
'''


def test_named_commands_help_lists_the_commands_the_texts_name(tmp_path: Path) -> None:
    """Each registered command the texts name gets one help line, in order of first mention.

    A name counts at a text's start or after whitespace, an opening
    bracket, a quote or Markdown emphasis; a path, a URL or a Markdown link
    target that merely contains it does not, nor does an unknown command or
    a SEA whose file or ``description()`` is broken (a staticmethod is not
    a method).  A task that invokes a registered command itself, a bare
    ``/xxx`` that fell back to a plain prompt, gets no block.
    """
    folder = tmp_path / "seas"
    _touch_sea(folder, "echo").write_text(_DESCRIBED_SEA, encoding="utf-8")
    _touch_sea(folder, "shout").write_text(
        _DESCRIBED_SEA.replace("Echoes the task back; use it as /echo", "Shouts it; /shout"),
        encoding="utf-8",
    )
    _touch_sea(folder, "broken")  # defines no SEA class
    _touch_sea(folder, "static").write_text(_STATIC_DESCRIPTION_SEA, encoding="utf-8")
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()

    block = sea_commands.named_commands_help(
        "Fix /broken, /static and /unknown, then use **/shout**.",
        ["(\u201c/echo\u201d) and /echo.", "/shout"],
    )
    lines = block.splitlines()
    assert lines[0] == sea_commands.NAMED_COMMANDS_HEADING
    assert 'run_agent(agent="<name>", task="...")' in lines[1]
    echo = "- `/echo`: Echoes the task back; use it as /echo <text>."
    assert lines[2:] == ["- `/shout`: Shouts it; /shout <text>.", echo]
    for text in ("/echo", "`/echo`", "*/echo*", "'/echo'", "[/echo]", "Did you\n/echo?"):
        assert sea_commands.named_commands_help("", [text]).splitlines()[2:] == [echo], text
    for text in (
        "~/echo", "a/echo", "/echo/x", "/echo.py", "/echo-2", "/echo?q=1", "/echo#a",
        "[t](/echo)", "https://x.org/echo", "/broken /static /unknown", "no command",
    ):
        assert sea_commands.named_commands_help("", [text]) == "", text
    with pytest.raises(sea_commands.SeaError, match="must be a method, got staticmethod"):
        sea_commands.help_text_if_command("/static help")
    assert sea_commands.named_commands_help("/echo", ["Use /shout."]) == ""
    assert sea_commands.named_commands_help("/unknown, then /echo").splitlines()[2:] == [echo]


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
def test_unreadable_sea_does_not_break_the_model_picker_registry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A SEA without read permission is skipped, not raised through ``model_seas()``.

    So is one that loses the permission after it was read (``chmod``
    changes no revision stamp): it is reported as unreadable, not kept
    in the picker and not loaded, and is back once it can be read.
    """
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
        router.chmod(0)
        caplog.clear()
        assert "ok" not in sea_commands.model_seas()
        logged = [r.getMessage() for r in caplog.records if "ok_sea.py" in r.getMessage()]
        assert logged == [f"SEA {router}: cannot be read"]
        router.chmod(0o644)
        assert sea_commands.model_seas()["ok"] == router
    finally:
        locked.chmod(0o644)
        router.chmod(0o644)


def test_model_sea_finds_a_router_installed_after_the_registry_was_built(
    tmp_path: Path,
) -> None:
    """A miss reads the snapshot only; the next refresh (the watcher's) finds the router."""
    folder = tmp_path / "seas"
    _write_router(folder, "early", _ROUTER_TRUE)
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    assert sea_commands.model_sea("late") is None
    late = _write_router(folder, "late", _ROUTER_TRUE)
    assert sea_commands.model_sea("late") is None
    sea_commands.refresh_registry()
    assert sea_commands.model_sea("late") == late


def test_a_second_name_for_the_sea_class_is_not_a_second_class(tmp_path: Path) -> None:
    """``Main = Sea`` binds the one class twice; two classes are still refused."""
    sea = tmp_path / "alias_sea.py"
    sea.write_text(_ROUTER_TRUE + "\nMain = Sea\n", encoding="utf-8")
    assert type(sea_commands.load_sea(sea)).__name__ == "Sea"
    sea.write_text(_ROUTER_TRUE + "\nclass Other(Sea):\n    pass\n", encoding="utf-8")
    with pytest.raises(sea_commands.SeaError, match="found Sea, Other"):
        sea_commands.load_sea(sea)


@posix_only("a file name that is not UTF-8")
def test_a_sea_under_a_folder_whose_name_is_not_utf8_loads(tmp_path: Path) -> None:
    """The loader names the module after the path's bytes, whatever their encoding."""
    folder = Path(os.fsdecode(os.fsencode(tmp_path) + b"/caf\xe9"))
    try:
        folder.mkdir()
    except OSError:
        pytest.skip("this file system refuses a name that is not UTF-8")
    sea = folder / "odd_sea.py"
    sea.write_bytes(_ROUTER_TRUE.encode("utf-8"))
    assert type(sea_commands.load_sea(sea)).__name__ == "Sea"


@posix_only("a doubled separator in a sys.path entry")
def test_model_seas_rereads_a_parent_imported_through_a_doubled_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edited parent module is re-read though its ``__file__`` spells the path ``a//b``."""
    lib = tmp_path / "lib"
    lib.mkdir()
    parent = lib / "s12_doubled_parent.py"
    parent.write_bytes(_ROUTER_TRUE.encode("utf-8"))
    monkeypatch.syspath_prepend(f"{tmp_path}//lib")
    folder = tmp_path / "seas"
    source = "from s12_doubled_parent import Sea as Parent\n\n\nclass Kid(Parent):\n    pass\n"
    kid = _write_router(folder, "kid", source)
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    try:
        assert sea_commands.model_sea("kid") == kid
        assert "//" in str(sys.modules["s12_doubled_parent"].__file__)
        stamp = parent.stat().st_mtime_ns
        parent.write_bytes(_ROUTER_FALSE.encode("utf-8"))
        os.utime(parent, ns=(stamp + 100, stamp + 100))
        assert sea_commands.model_sea("kid") is None
    finally:
        sys.modules.pop("s12_doubled_parent", None)


@posix_only("chmod-based read denial")
@pytest.mark.skipif(is_root(), reason="root reads unreadable files")
def test_model_seas_rereads_a_base_once_it_can_be_read(tmp_path: Path) -> None:
    """A base that could not be read when its SEA was loaded is read once it can be.

    ``chmod`` changes no revision stamp, so the verdict cache watches a
    file it cannot read as unreadable, not by its stamp.
    """
    folder = tmp_path / "seas"
    parent = _write_router(folder, "parent", _ROUTER_TRUE)
    source = (
        "from kiss.agents.sorcar.sea_commands import sea_class\n\n\n"
        "class Kid(sea_class('parent')):\n    pass\n"
    )
    kid = _write_router(folder, "kid", source)
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    parent.chmod(0)
    try:
        assert sea_commands.model_sea("kid") is None
    finally:
        parent.chmod(0o644)
    assert sea_commands.model_sea("kid") == kid


_SELF_STR_SEA = '''
from kiss.agents.seas.base.base_sea import BaseSea


class Text(str):
    """A ``str`` whose ``__str__`` returns itself and whose other methods raise."""

    __slots__ = ()

    def __str__(self):
        return self

    def replace(self, *args):
        raise RuntimeError("replace exploded")

    def strip(self, *args):
        raise RuntimeError("strip exploded")


class Sea(BaseSea):
    def description(self):
        return Text("described")

    def prompt(self, task):
        return Text("shaped " + task)

    def system_prompt(self, system_prompt):
        return Text(system_prompt + " more")
'''


def test_text_of_an_untrusted_str_subclass_is_copied_not_called(tmp_path: Path) -> None:
    """What a method returns is an exact ``str`` from then on: no method of the script's
    own subclass runs, not even one whose ``__str__`` returns the instance itself."""
    sea = tmp_path / "text_sea.py"
    sea.write_text(_SELF_STR_SEA, encoding="utf-8")
    loaded = sea_commands.load_sea(sea)
    run = sea_commands.evaluate_sea([loaded], "task {task_id}", "t1")
    shaped = run.system_prompt_hook("base")
    assert (run.prompt, shaped) == ("shaped task t1", "base more")
    assert type(run.prompt) is str and type(shaped) is str
    assert sea_commands.sea_description(loaded) == "described"


_GATED_PICKER = """
import time
from pathlib import Path

from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def description(self):
        return 'a picker whose hook waits for the release file'

    def register_as_model(self):
        return True

    def on_picked_as_model(self, work_dir):
        with open(Path(work_dir, 'picked.log'), 'a') as log:
            log.write('LABEL\\n')
        deadline = time.monotonic() + 30
        while not Path(work_dir, 'release').exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        return ''
"""


def _picked_hook_threads(before: set[threading.Thread]) -> list[threading.Thread]:
    """Return the live ``on_picked_as_model`` threads started since *before* was taken."""
    return [t for t in threading.enumerate() if t not in before and t.name == "sea-picked-hook"]


def test_a_pick_shares_the_hook_run_under_way_instead_of_starting_another(tmp_path: Path) -> None:
    """Picks of one SEA and directory share the hook run that is under way.

    Every pick started a daemon thread of its own, so a hook that never
    returned left one blocked thread per pick for the life of the
    daemon.  A waiting pick waits for the run under way; a hook that has
    returned runs again; an edited script is another hook, which the
    blocked run of the old text does not hold back.
    """
    folder = tmp_path / "seas"
    router = _write_router(folder, "gated", _GATED_PICKER.replace("LABEL", "first"))
    _write_seas_md([str(folder)])
    sea_commands.refresh_registry()
    work = tmp_path / "work"
    work.mkdir()
    log, release = work / "picked.log", work / "release"
    before = set(threading.enumerate())
    loaded = sea_commands.load_sea(router)
    # The picker handler names the model; a run hands over the SEA it loaded.
    sea_commands.run_picked_hook("gated", str(work), wait=False)
    sea_commands.run_picked_hook("gated", str(work), wait=False, sea=loaded)
    sea_commands.run_picked_hook("gated", str(work), wait=False)
    assert len(_picked_hook_threads(before)) == 1
    _wait_for(log.exists)
    waiter = threading.Thread(
        target=sea_commands.run_picked_hook, args=("gated", str(work)), kwargs={"sea": loaded}
    )
    waiter.start()
    waiter.join(0.3)
    assert waiter.is_alive(), "the pick did not wait for the run under way"
    release.touch()
    waiter.join(10)
    assert not waiter.is_alive()
    _wait_for(lambda: not _picked_hook_threads(before))
    assert log.read_text() == "first\n"
    sea_commands.run_picked_hook("gated", str(work))
    assert log.read_text() == "first\nfirst\n"
    release.unlink()
    sea_commands.run_picked_hook("gated", str(work), wait=False)
    _wait_for(lambda: log.read_text() == "first\n" * 3)
    router.write_bytes(_GATED_PICKER.replace("LABEL", "second").encode("utf-8"))
    sea_commands.run_picked_hook("gated", str(work), wait=False)
    _wait_for(lambda: log.read_text() == "first\n" * 3 + "second\n")
    assert len(_picked_hook_threads(before)) == 2
    release.touch()
    _wait_for(lambda: not _picked_hook_threads(before))


def _stop_once_the_thread_is_launched(frame: object, event: str, arg: object) -> None:
    """Profile function of a picking thread: the clock of a Stop that lands in ``Thread.start()``.

    Injects the daemon's Stop into its own thread the moment the hook's thread exists, before
    ``start()`` has seen it run.  Nothing of the product is replaced: the injection is the real
    one, and ``start()`` checks for it at this point with or without a profile function.
    """
    if event == "c_return" and getattr(arg, "__name__", "") == "start_joinable_thread":
        inject_async_exception(threading.get_ident(), KeyboardInterrupt)


def test_a_stop_landing_in_the_start_of_the_hook_thread_leaves_one_run(tmp_path: Path) -> None:
    """A Stop that lands while a pick starts the hook's thread does not let another run start.

    The thread was entered in the table of running hooks only once ``Thread.start()`` had
    returned, so a Stop landing inside it left a running hook no later pick could find, and
    every pick started another.  The interrupted pick's thread either runs the hook as the
    one the later picks share, or finds that a later pick took its place and runs nothing.
    """
    router = _write_router(tmp_path / "seas", "gated", _GATED_PICKER.replace("LABEL", "first"))
    loaded = sea_commands.load_sea(router)
    work = tmp_path / "work"
    work.mkdir()
    before = set(threading.enumerate())
    profile = sys.getprofile()
    sys.setprofile(_stop_once_the_thread_is_launched)
    try:
        with pytest.raises(KeyboardInterrupt):
            sea_commands.run_picked_hook("gated", str(work), wait=False, sea=loaded)
    finally:
        sys.setprofile(profile)
    sea_commands.run_picked_hook("gated", str(work), wait=False, sea=loaded)
    sea_commands.run_picked_hook("gated", str(work), wait=False, sea=loaded)
    _wait_for((work / "picked.log").exists)
    _wait_for(lambda: len(_picked_hook_threads(before)) == 1)
    assert (work / "picked.log").read_text() == "first\n"
    (work / "release").touch()
    _wait_for(lambda: not _picked_hook_threads(before))


def test_a_loaded_sea_is_picked_under_the_revision_it_was_loaded_from(tmp_path: Path) -> None:
    """The hook of a script fixed since an older load of it is not held back by that load's run.

    A run loads its SEA and picks it later.  The pick was keyed by the revision the file had
    when it was picked, so the stuck hook of a SEA loaded before an edit held the key of the
    edited script, and the pick of the fixed SEA shared that run instead of running its own.
    """
    router = _write_router(tmp_path / "seas", "gated", _GATED_PICKER.replace("LABEL", "first"))
    work = tmp_path / "work"
    work.mkdir()
    log = work / "picked.log"
    before = set(threading.enumerate())
    old = sea_commands.load_sea(router)
    router.write_bytes(_GATED_PICKER.replace("LABEL", "second").encode("utf-8"))
    new = sea_commands.load_sea(router)
    sea_commands.run_picked_hook("gated", str(work), wait=False, sea=old)
    _wait_for(log.exists)
    sea_commands.run_picked_hook("gated", str(work), wait=False, sea=new)
    _wait_for(lambda: log.read_text() == "first\nsecond\n")
    sea_commands.run_picked_hook("gated", str(work), wait=False, sea=old)
    sea_commands.run_picked_hook("gated", str(work), wait=False, sea=new)
    assert len(_picked_hook_threads(before)) == 2
    (work / "release").touch()
    _wait_for(lambda: not _picked_hook_threads(before))
    assert log.read_text() == "first\nsecond\n"


_GATED_PICKER_WITH_A_GATED_IMPORT = _GATED_PICKER + """
Path(__file__).with_name('loading').touch()
deadline = time.monotonic() + 30
while not Path(__file__).with_name('loaded').exists() and time.monotonic() < deadline:
    time.sleep(0.01)
"""


def test_a_load_that_waited_for_another_is_picked_under_the_text_it_ran(tmp_path: Path) -> None:
    """A SEA is picked under the revision of the text that ran, however long its load waited.

    The revision was the file's stamp, read before the load waited for another load of the
    same script to end.  A script edited during that wait ran its new text under the old
    revision, so its pick shared the stuck hook of the old text and its own hook never ran.
    The load the edit overtook ran the old text, though it ended after the edit: it shares
    the old run.
    """
    source = _GATED_PICKER_WITH_A_GATED_IMPORT
    router = _write_router(tmp_path / "seas", "gated", source.replace("LABEL", "first"))
    loading, loaded = router.with_name("loading"), router.with_name("loaded")
    work = tmp_path / "work"
    work.mkdir()
    log = work / "picked.log"
    before = set(threading.enumerate())
    loaded.touch()
    sea_commands.run_picked_hook("gated", str(work), wait=False, sea=sea_commands.load_sea(router))
    _wait_for(log.exists)
    loaded.unlink()
    loading.unlink()
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            overtaken = pool.submit(sea_commands.load_sea, router)
            _wait_for(loading.exists)
            waiting = pool.submit(sea_commands.load_sea, router)
            _wait_for(lambda: any(
                sea_settings._waited_module(thread.ident) for thread in threading.enumerate()
            ))
            router.write_bytes(source.replace("LABEL", "second").encode("utf-8"))
        finally:
            loaded.touch()
        for sea in (overtaken.result(10), waiting.result(10)):
            sea_commands.run_picked_hook("gated", str(work), wait=False, sea=sea)
    _wait_for(lambda: log.read_text() == "first\nsecond\n")
    assert len(_picked_hook_threads(before)) == 2
    (work / "release").touch()
    _wait_for(lambda: not _picked_hook_threads(before))
    assert log.read_text() == "first\nsecond\n"


def test_an_edit_that_keeps_the_stamp_of_the_file_runs_its_own_hook(tmp_path: Path) -> None:
    """The revision a pick is keyed by is the script's text, not what a ``stat`` of it shows.

    Two writes of one size within one tick of the file system's clock, or an edit by a tool
    that puts the old time back, leave the size, the time and the inode as they were.  Picks
    were keyed by those, so the fixed script shared the stuck hook of its old text.
    """
    folder = tmp_path / "seas"
    router = _write_router(folder, "gated", _GATED_PICKER.replace("LABEL", "one"))
    _write_seas_md([str(folder)])
    work = tmp_path / "work"
    work.mkdir()
    log = work / "picked.log"
    before = set(threading.enumerate())
    stat = router.stat()
    sea_commands.run_picked_hook("gated", str(work), wait=False)
    _wait_for(log.exists)
    router.write_bytes(_GATED_PICKER.replace("LABEL", "two").encode("utf-8"))
    os.utime(router, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert sea_settings.file_stamp(router) == (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    sea_commands.run_picked_hook("gated", str(work), wait=False)
    _wait_for(lambda: log.read_text() == "one\ntwo\n")
    (work / "release").touch()
    _wait_for(lambda: not _picked_hook_threads(before))


_PICKER_WITH_A_GATED_IMPORT = """
import time
from pathlib import Path

from kiss.agents.seas.base.base_sea import BaseSea

Path(__file__).with_name('loading').touch()
while not Path(__file__).with_name('loaded').exists():
    time.sleep(0.01)


class Sea(BaseSea):
    def description(self):
        return 'a picker whose import waits for the loaded file'

    def register_as_model(self):
        return True
"""


def test_a_pick_by_name_runs_under_the_revision_its_thread_loads(tmp_path: Path) -> None:
    """Picks by name of one script revision share one run, whatever the registry held before.

    On a registry not yet filled the first pick was keyed without a revision and the second
    with one, so both ran the hook.  And a script edited between a pick and the load on its
    thread ran its new text under the old revision's key, where the next pick did not find it.
    """
    folder = tmp_path / "seas"
    router = _write_router(folder, "gated", _GATED_PICKER.replace("LABEL", "first"))
    _write_seas_md([str(folder)])
    work = tmp_path / "work"
    work.mkdir()
    log, release = work / "picked.log", work / "release"
    before = set(threading.enumerate())
    sea_commands.run_picked_hook("gated", str(work), wait=False)
    _wait_for(log.exists)
    sea_commands.run_picked_hook("gated", str(work), wait=False)
    assert len(_picked_hook_threads(before)) == 1
    release.touch()
    _wait_for(lambda: not _picked_hook_threads(before))
    assert log.read_text() == "first\n"
    # A pick whose thread is still importing the script when the script is edited.
    release.unlink()
    router.write_bytes(_PICKER_WITH_A_GATED_IMPORT.encode("utf-8"))
    sea_commands.run_picked_hook("gated", str(work), wait=False)
    _wait_for(router.with_name("loading").exists)
    router.write_bytes(_GATED_PICKER.replace("LABEL", "second").encode("utf-8"))
    router.with_name("loaded").touch()
    _wait_for(lambda: log.read_text() == "first\nsecond\n")
    sea_commands.run_picked_hook("gated", str(work), wait=False)
    release.touch()
    _wait_for(lambda: not _picked_hook_threads(before))
    assert log.read_text() == "first\nsecond\n"
    # A name that is no command has no hook: no thread is started for it.
    sea_commands.run_picked_hook("no-such-model", str(work), wait=False)
    assert not _picked_hook_threads(before)
    # Nor has a command whose SEA does not register as a model: its thread runs nothing.
    router.write_bytes(_GATED_PICKER.replace("return True", "return False").encode("utf-8"))
    sea_commands.run_picked_hook("gated", str(work))
    assert log.read_text() == "first\nsecond\n"
    # Nor has a script that is gone by the time it is picked: nothing is left to read or run.
    router.unlink()
    sea_commands.run_picked_hook("gated", str(work))
    assert log.read_text() == "first\nsecond\n" and not _picked_hook_threads(before)
