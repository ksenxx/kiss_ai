# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a hung playback wrapper dies WITH its grandchildren.

Concurrency-audit fix C3 (tmp/audit-server-scout-printer.md):
``talk_player._run_playback`` used to ``proc.kill()`` only the direct
child on timeout.  ``KISS_SORCAR_PLAY_CMD`` explicitly exists so users
and tests can substitute wrapper scripts (module docstring), and a
wrapper that runs the real player as a CHILD (``#!/bin/sh`` +
``mpg123 "$1"`` without ``exec``) left that grandchild holding the
audio device after the timeout — playing over the next clip the
serialising worker had already started, which is exactly the state
the single playback queue exists to prevent.

The fix starts each playback child in its own session
(``start_new_session`` on POSIX) and, on timeout, signals the whole
process group: SIGTERM first, escalating to SIGKILL after a grace
period.  ``KISS_SORCAR_PLAY_TIMEOUT``
makes the 600 s timeout overridable so this test forces it with a
REAL hung child — no mocks, real subprocesses, a real
:class:`TalkPlayer` worker.

Unreachable-without-fakes branches, documented instead of mocked:
``_kill_playback``'s ``proc.kill()`` fallbacks run only when
:func:`kiss.core.processes.kill_process_group` raises, which on POSIX
needs a process group owned by another user (``PermissionError``) or
one that is already gone.
"""

from __future__ import annotations

import base64
import os
import shlex
import stat
import subprocess
import sys
import threading
import time
import unittest
import uuid
from pathlib import Path

from kiss.server import talk_player
from kiss.server.talk_player import TalkPlayer

MP3_B64 = base64.b64encode(
    b"ID3\x03\x00fake-mp3-frames-" + bytes(range(64))
).decode("ascii")


def _write_script(path: Path, body: str) -> None:
    """Write an executable ``/bin/sh`` script at *path*."""
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _python_player(path: Path, body: str) -> str:
    """Write the Python player *body* at *path*; return its ``KISS_SORCAR_PLAY_CMD``.

    Run by this interpreter, so it plays on Windows too, where a
    ``/bin/sh`` script cannot be executed directly.
    """
    path.write_text("import os, time\n" + body)
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(path))}"


def _pid_alive(pid: int) -> bool:
    """Return ``True`` while *pid* exists (signal 0 probe)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for(predicate, timeout: float) -> bool:
    """Poll *predicate* until true or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class _EnvMixin(unittest.TestCase):
    """Save/restore the player env vars around each test."""

    _ENV_KEYS = ("KISS_SORCAR_PLAY_CMD", "KISS_SORCAR_PLAY_TIMEOUT")

    def setUp(self) -> None:
        """Snapshot the playback environment overrides."""
        self._saved = {k: os.environ.get(k) for k in self._ENV_KEYS}
        self._stray_pids: list[int] = []

    def tearDown(self) -> None:
        """Restore the environment and reap any surviving children."""
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for pid in self._stray_pids:
            try:
                os.kill(pid, 9)
            except (ProcessLookupError, PermissionError):
                pass


@unittest.skipUnless(os.name == "posix", "process groups are POSIX-only")
class TalkPlayerGroupKillTest(_EnvMixin):
    """Timeout kills reach the wrapper's grandchildren."""

    def test_timeout_kills_grandchild_and_worker_survives(self) -> None:
        """The wrapper's sleeping child dies; the next clip still plays.

        The wrapper spawns ``sleep 300`` and waits on it (a player run
        WITHOUT ``exec``), so it exceeds the 1 s timeout.  After the
        kill, the grandchild must be gone — and the serialising worker
        must go on to play the next queued clip.
        """
        tmp = Path(self.enterContext(_tempdir()))
        pids = tmp / "grandchild.pids"
        runs = tmp / "invocations.log"
        script = tmp / "hanging_player.sh"
        _write_script(
            script,
            f"sleep 300 &\n"
            f"echo $! >> {shlex.quote(str(pids))}\n"
            f"echo run >> {shlex.quote(str(runs))}\n"
            f"wait\n",
        )
        os.environ["KISS_SORCAR_PLAY_CMD"] = shlex.quote(str(script))
        os.environ["KISS_SORCAR_PLAY_TIMEOUT"] = "1"

        player = TalkPlayer()
        player.play(
            {"talkId": uuid.uuid4().hex, "text": "", "audioB64": MP3_B64}
        )
        self.assertTrue(
            _wait_for(pids.exists, 10.0), "wrapper never started"
        )
        grandchild = int(pids.read_text().splitlines()[0])
        self._stray_pids.append(grandchild)

        # Timeout 1 s + SIGTERM (+5 s SIGKILL grace, unused here:
        # sleep dies on SIGTERM) + slack.
        self.assertTrue(
            _wait_for(lambda: not _pid_alive(grandchild), 15.0),
            "grandchild survived the playback timeout kill — it would "
            "keep the audio device and play over the next clip",
        )

        # The worker thread must have survived the kill cycle and
        # still serialise the next clip.
        player.play(
            {"talkId": uuid.uuid4().hex, "text": "", "audioB64": MP3_B64}
        )
        self.assertTrue(
            _wait_for(
                lambda: runs.exists()
                and len(runs.read_text().splitlines()) >= 2,
                20.0,
            ),
            "worker did not play the next clip after a timeout kill",
        )
        for line in pids.read_text().splitlines():
            self._stray_pids.append(int(line))

    def test_sigkill_escalation_for_term_ignoring_wrapper(self) -> None:
        """A wrapper that traps SIGTERM is SIGKILLed after the grace.

        Covers ``_kill_playback``'s ``TimeoutExpired`` arm: the group
        SIGTERM does not end the wrapper (it traps the signal and
        loops), so the escalation must SIGKILL the group.
        """
        tmp = Path(self.enterContext(_tempdir()))
        pids = tmp / "wrapper.pid"
        script = tmp / "term_ignoring_player.sh"
        _write_script(
            script,
            f"trap '' TERM\n"
            f"echo $$ >> {shlex.quote(str(pids))}\n"
            f"i=0\n"
            f"while [ $i -lt 600 ]; do sleep 1; i=$((i+1)); done\n",
        )
        os.environ["KISS_SORCAR_PLAY_CMD"] = shlex.quote(str(script))
        os.environ["KISS_SORCAR_PLAY_TIMEOUT"] = "1"

        player = TalkPlayer()
        player.play(
            {"talkId": uuid.uuid4().hex, "text": "", "audioB64": MP3_B64}
        )
        self.assertTrue(
            _wait_for(pids.exists, 10.0), "wrapper never started"
        )
        wrapper = int(pids.read_text().splitlines()[0])
        self._stray_pids.append(wrapper)

        # Timeout 1 s + grace 5 s + slack for the SIGKILL to land.
        self.assertTrue(
            _wait_for(lambda: not _pid_alive(wrapper), 20.0),
            "SIGTERM-ignoring wrapper survived: SIGKILL escalation "
            "never reached its process group",
        )


@unittest.skipUnless(os.name == "posix", "process groups are POSIX-only")
class TalkPlayerShutdownTest(_EnvMixin):
    """A shutdown takes the playing wrapper, its grandchild and the temp clip with it.

    The worker is a daemon thread and each player runs in a session of
    its own, so a daemon that stopped or exited while a clip played left
    the player running with nobody to time it out (audit 2026-11-10
    B/V7-F6).
    """

    def _hung_wrapper(self, tmp: Path) -> Path:
        """Make the player a wrapper that hangs on a grandchild; return its record file.

        Each run adds one line: the wrapper's pid, the grandchild's and the clip's path.
        """
        record = tmp / "runs"
        script = tmp / "hanging_player.sh"
        _write_script(
            script, f'sleep 300 &\necho "$$ $! $1" >> {shlex.quote(str(record))}\nwait\n'
        )
        os.environ["KISS_SORCAR_PLAY_CMD"] = shlex.quote(str(script))
        os.environ.pop("KISS_SORCAR_PLAY_TIMEOUT", None)
        return record

    def _assert_nothing_left(self, record: Path) -> None:
        """The wrapper ran once, and neither process nor the clip is left of that run."""
        (line,) = record.read_text().splitlines()
        wrapper, grandchild, clip = line.split(" ", 2)
        self._stray_pids += [int(wrapper), int(grandchild)]
        self.addCleanup(Path(clip).unlink, missing_ok=True)
        self.assertFalse(_pid_alive(int(wrapper)), "the playing wrapper outlived its player")
        self.assertTrue(
            _wait_for(lambda: not _pid_alive(int(grandchild)), 15.0),
            "the wrapper's grandchild outlived its player",
        )
        self.assertFalse(os.path.exists(clip), "the temp clip outlived its player")

    def test_shutdown_ends_the_playing_group_and_drops_the_queue(self) -> None:
        """The daemon's shutdown; the next server of the process gets a fresh player."""
        tmp = Path(self.enterContext(_tempdir()))
        record = self._hung_wrapper(tmp)
        talk_player.shutdown()  # an earlier test's shared player, if any
        self.addCleanup(talk_player.shutdown)
        player = talk_player.shared_player()
        for _ in range(2):  # the second clip waits in the queue behind the hung one
            player.play({"talkId": uuid.uuid4().hex, "text": "", "audioB64": MP3_B64})
        self.assertTrue(
            _wait_for(lambda: record.exists() and record.read_text().endswith("\n"), 10.0),
            "wrapper never started",
        )
        worker = player._worker
        assert worker is not None

        talk_player.shutdown()

        self.assertFalse(worker.is_alive())
        self._assert_nothing_left(record)
        # The shut-down player admits nothing more; the process's next talk
        # goes to a fresh one.
        played = tmp / "played"
        os.environ["KISS_SORCAR_PLAY_CMD"] = _python_player(
            tmp / "player.py", f"open({str(played)!r}, 'a').write('run\\n')\n"
        )
        event = {"talkId": uuid.uuid4().hex, "text": "", "audioB64": MP3_B64}
        self.assertTrue(player.play(event))
        self.assertIs(player._worker, worker)
        fresh = talk_player.shared_player()
        self.assertIsNot(fresh, player)
        fresh.play(event)
        self.assertTrue(_wait_for(played.exists, 10.0), "the fresh shared player never played")
        time.sleep(0.3)  # a clip the shut-down player had still admitted would play here too
        self.assertEqual(played.read_text(), "run\n")

    def test_interpreter_exit_ends_the_playing_group(self) -> None:
        """A process that exits while its shared player plays leaves nothing of the clip."""
        tmp = Path(self.enterContext(_tempdir()))
        record = self._hung_wrapper(tmp)
        code = (
            "import os, sys, time\n"
            "from kiss.server.talk_player import shared_player\n"
            "shared_player().play({'talkId': 't', 'audioB64': sys.argv[2]})\n"
            "deadline = time.monotonic() + 60\n"
            "while time.monotonic() < deadline and not (\n"
            "    os.path.exists(sys.argv[1]) and open(sys.argv[1]).read().endswith('\\n')\n"
            "):\n"
            "    time.sleep(0.05)\n"
        )
        done = subprocess.run([sys.executable, "-c", code, str(record), MP3_B64], timeout=120)
        self.assertEqual(done.returncode, 0)
        self._assert_nothing_left(record)


class PlayerResolvedOncePerEventTest(_EnvMixin):
    """``play`` resolves the machine's player once, when the event is queued."""

    def test_player_resolved_at_enqueue_is_the_one_used_at_playback(self) -> None:
        """The argv found at enqueue travels with the entry to the worker.

        The player used to be looked up twice per talk: by the daemon
        before queueing and again by the worker at playback (an
        environment read and up to six ``PATH`` scans each).  Now the
        lookup happens once, at enqueue: a player override changed
        while the clip waits in the queue does not apply to it.
        """
        tmp = Path(self.enterContext(_tempdir()))
        runs = tmp / "invocations.log"
        gate = tmp / "release"
        first = _python_player(
            tmp / "first_player.py",
            f"open({str(runs)!r}, 'a').write('first\\n')\n"
            f"while not os.path.exists({str(gate)!r}):\n    time.sleep(0.05)\n",
        )
        second = _python_player(
            tmp / "second_player.py", f"open({str(runs)!r}, 'a').write('second\\n')\n"
        )
        os.environ["KISS_SORCAR_PLAY_CMD"] = first

        player = TalkPlayer()
        queued_first = player.play(
            {"talkId": uuid.uuid4().hex, "text": "", "audioB64": MP3_B64}
        )
        self.assertTrue(_wait_for(runs.exists, 10.0), "first player never started")
        # The worker is busy with the first clip: the second waits in
        # the queue, its player already chosen.
        queued_second = player.play(
            {"talkId": uuid.uuid4().hex, "text": "", "audioB64": MP3_B64}
        )
        os.environ["KISS_SORCAR_PLAY_CMD"] = second
        gate.write_text("go")
        self.assertTrue(
            _wait_for(lambda: len(runs.read_text().splitlines()) >= 2, 20.0),
            "the second clip never played",
        )
        self.assertEqual(runs.read_text().split(), ["first", "first"])
        self.assertEqual((queued_first, queued_second), (True, True))

    def test_no_player_means_nothing_queued_and_nothing_remembered(self) -> None:
        """Without a player ``play`` answers ``False`` and the talk may be re-sent later."""
        tmp = Path(self.enterContext(_tempdir()))
        runs = tmp / "invocations.log"
        script = _python_player(tmp / "player.py", f"open({str(runs)!r}, 'a').write('run\\n')\n")
        os.environ["KISS_SORCAR_PLAY_CMD"] = "'unterminated"  # malformed: no player
        player = TalkPlayer()
        event = {"talkId": uuid.uuid4().hex, "text": "", "audioB64": MP3_B64}
        self.assertFalse(player.play(event))
        self.assertFalse(player.play({"talkId": uuid.uuid4().hex, "text": ""}))
        self.assertIsNone(player._worker)
        # The same talk re-sent once a player exists plays: it was not
        # remembered as spoken.
        os.environ["KISS_SORCAR_PLAY_CMD"] = script
        self.assertTrue(player.play(event))
        self.assertTrue(_wait_for(runs.exists, 10.0), "the re-sent talk never played")
        # Spoken once (the browser client's ``spokenTalkIds`` behaviour):
        # the repeat is accepted but nothing is queued for it.
        self.assertTrue(player.play(event))
        self.assertTrue(player._queue.empty())


class PlaybackTimeoutEnvTest(_EnvMixin):
    """``_playback_timeout`` parsing (all branches, real env)."""

    def test_timeout_parsing(self) -> None:
        """Valid overrides win; junk and non-positive values fall back."""
        os.environ.pop("KISS_SORCAR_PLAY_TIMEOUT", None)
        self.assertEqual(talk_player._playback_timeout(), 600.0)
        os.environ["KISS_SORCAR_PLAY_TIMEOUT"] = "2.5"
        self.assertEqual(talk_player._playback_timeout(), 2.5)
        os.environ["KISS_SORCAR_PLAY_TIMEOUT"] = "not-a-number"
        self.assertEqual(talk_player._playback_timeout(), 600.0)
        os.environ["KISS_SORCAR_PLAY_TIMEOUT"] = "0"
        self.assertEqual(talk_player._playback_timeout(), 600.0)
        os.environ["KISS_SORCAR_PLAY_TIMEOUT"] = "-3"
        self.assertEqual(talk_player._playback_timeout(), 600.0)
        # Review finding 6: non-finite values passed the ``> 0`` gate
        # and handed ``Popen.wait`` an infinite timeout, disabling the
        # hung-playback group kill.  They must fall back too.
        for raw in ("inf", "+inf", "Infinity", "1e999", "nan"):
            os.environ["KISS_SORCAR_PLAY_TIMEOUT"] = raw
            self.assertEqual(
                talk_player._playback_timeout(), 600.0,
                f"non-finite override {raw!r} must fall back",
            )

    def test_run_playback_oserror(self) -> None:
        """A missing player binary is skipped cleanly (OSError branch)."""
        talk_player._run_playback(["/nonexistent/sorcar-player-xyz"], threading.Event())


def _tempdir():
    """Return a TemporaryDirectory context manager (test-local dirs)."""
    import tempfile

    return tempfile.TemporaryDirectory(prefix="conc2026-talk-")


if __name__ == "__main__":
    unittest.main()
