# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the mic listener exits when its spawner dies abruptly.

The listener is spawned in its own session with stdin on /dev/null
(``voice_wake_control.py`` and ``voiceWake.ts`` both do this), so a
spawner that is SIGKILLed or crashes without running its shutdown path
leaves the listener no SIGHUP and no stdin EOF.  Before the fix
``run_mic`` only blocked on its audio queue and fed Vosk; its first
write to the dead stdout pipe was ``emit("WAKE")``, so the orphan kept
PortAudio's input stream open (mic indicator on, device held, CPU
spent on Vosk) until somebody said the wake word.  ``run_mic`` now
checks ``os.getppid()`` on every loop iteration and returns as soon as
the process has been reparented.

Real listener subprocess, real sounddevice/PortAudio stream, no mocks:
a wrapper Python process spawns the listener exactly like the
controller does, reports the listener pid once ``READY`` arrives, and
is then SIGKILLed by the test.  Before the fix the listener survived
this indefinitely (the test's poll ran out and the orphan had to be
killed by hand); after the fix it is gone within one audio block.

Skipped on hosts without a microphone: ``run_mic`` cannot open its
input stream without one, so there is nothing to orphan.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Spawns the listener the way voice_wake_control.py does (own session,
# stdin on /dev/null, stdout piped to the spawner), then prints the
# listener pid after READY and idles forever — until the test kills it.
SPAWNER_SCRIPT = """
import subprocess, sys, time
proc = subprocess.Popen(
    [sys.executable, "-m", "kiss.server.voice_wake",
     "--mic-watchdog-timeout", "2"],
    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL, start_new_session=True,
)
line = proc.stdout.readline().decode().strip()
print(f"{line} {proc.pid}", flush=True)
while True:
    time.sleep(1)
"""


def _have_input_device() -> bool:
    """Return True when PortAudio reports a default input device."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sounddevice; sounddevice.query_devices(kind='input')",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return probe.returncode == 0


def _pid_gone(pid: int) -> bool:
    """Return True once *pid* has exited (a reaped process or a zombie)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return not state or state.startswith("Z")


@unittest.skipUnless(os.name == "posix", "os.getppid() only tracks reparenting on POSIX")
class TestListenerExitsWhenSpawnerDies(unittest.TestCase):
    """A SIGKILLed spawner must not leave a mic-holding listener behind."""

    @pytest.mark.slow
    def test_listener_exits_after_spawner_sigkill(self) -> None:
        if not _have_input_device():
            self.skipTest("no audio input device available")

        spawner = subprocess.Popen(
            [sys.executable, "-c", SPAWNER_SCRIPT],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        listener_pid = 0
        try:
            assert spawner.stdout is not None
            first = spawner.stdout.readline().split()
            self.assertEqual(first[:1], ["READY"], msg=f"spawner said {first!r}")
            listener_pid = int(first[1])
            self.assertFalse(_pid_gone(listener_pid))

            spawner.kill()
            spawner.wait(timeout=30)

            deadline = time.monotonic() + 10
            while not _pid_gone(listener_pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertTrue(
                _pid_gone(listener_pid),
                msg=f"listener {listener_pid} outlived its SIGKILLed spawner",
            )
        finally:
            if spawner.poll() is None:
                spawner.kill()
                spawner.wait(timeout=30)
            if listener_pid and not _pid_gone(listener_pid):
                os.kill(listener_pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
