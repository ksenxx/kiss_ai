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
period (``voice_wake_control`` pattern).  ``KISS_SORCAR_PLAY_TIMEOUT``
makes the 600 s timeout overridable so this test forces it with a
REAL hung child — no mocks, real subprocesses, a real
:class:`TalkPlayer` worker.

Unreachable-without-fakes branches, documented instead of mocked:
``_signal_group``'s ``killpg is None`` path and ``_kill_playback``'s
``proc.kill()`` fallback only run on Windows (``os.killpg`` always
exists on POSIX), and ``_signal_group``'s ``PermissionError`` arm
needs a process group owned by another user.
"""

from __future__ import annotations

import base64
import os
import shlex
import stat
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
        """A missing player binary fails cleanly (OSError branch)."""
        self.assertFalse(
            talk_player._run_playback(["/nonexistent/sorcar-player-xyz"])
        )


def _tempdir():
    """Return a TemporaryDirectory context manager (test-local dirs)."""
    import tempfile

    return tempfile.TemporaryDirectory(prefix="conc2026-talk-")


if __name__ == "__main__":
    unittest.main()
