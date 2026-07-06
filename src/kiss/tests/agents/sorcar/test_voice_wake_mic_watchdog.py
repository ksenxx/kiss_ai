# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test for the mic-stream-health watchdog in the listener.

Reproduces the "wake word silently stops working" production failure:
on macOS, PortAudio input streams can silently stop delivering
callbacks after an audio device/route change (confirmed live — two
long-running listeners were each 100% blocked in ``blocks.get()``
while a freshly opened stream received audio fine).  The old
``run_mic`` looped on ``blocks.get()`` with no timeout, so the process
stayed alive but deaf and the host believed it was still listening.

Real listener subprocess, real sounddevice/PortAudio stream, no mocks:
the ``KISS_VOICE_MIC_BLOCK_SIZE`` hook sets a block size worth a full
minute of audio, so the (healthy, real) input stream cannot deliver a
single block within the short ``--mic-watchdog-timeout``.  From the
listener's point of view this is exactly what a silently dead stream
looks like: the stream object is alive, but no callback ever fires.
The watchdog must detect the stall, close and reopen the stream the
capped number of times (the reopened streams stay "silent" for the
same reason), then give up with a diagnostic on stderr and a nonzero
exit code so the extension host can surface the error instead of a
silently deaf microphone.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]

# One minute of audio per block: a real stream that can never deliver
# a block within the ~1s watchdog window used by the test.
SILENT_BLOCK_SIZE = 16000 * 60
WATCHDOG_TIMEOUT_SECONDS = 1.0


def _have_input_device() -> bool:
    """Return True when PortAudio reports a default input device."""
    probe = subprocess.run(
        [
            "uv", "run", "python", "-c",
            "import sounddevice; sounddevice.query_devices(kind='input')",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return probe.returncode == 0


class TestMicWatchdog(unittest.TestCase):
    """A silent mic stream is detected, retried, and reported."""

    def test_silent_stream_exits_nonzero_with_diagnostic(self) -> None:
        if not _have_input_device():
            self.skipTest("no audio input device available")

        env = dict(os.environ)
        env["KISS_VOICE_MIC_BLOCK_SIZE"] = str(SILENT_BLOCK_SIZE)
        # Expected runtime: ~4 watchdog windows + 3 reopen delays +
        # model load; the 120s cap guarantees the old no-timeout
        # ``blocks.get()`` regression fails fast instead of hanging.
        proc = subprocess.run(
            [
                "uv", "run", "python", "-m",
                "kiss.agents.vscode.voice_wake",
                "--mic-watchdog-timeout", str(WATCHDOG_TIMEOUT_SECONDS),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        detail = f"stdout={proc.stdout!r}\nstderr={proc.stderr[-4000:]}"
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]

        # The first stream opened fine: READY was emitted exactly once
        # (reopens must not re-emit it — the host counts on that).
        self.assertEqual(lines.count("READY"), 1, msg=detail)

        # The watchdog noticed the stall and tried the capped number
        # of reopens, logging each attempt.
        self.assertEqual(
            proc.stderr.count("reopening the input stream (attempt"),
            3,
            msg=detail,
        )

        # After the reopened streams stayed silent it gave up with a
        # diagnostic and a nonzero exit code, so the extension host
        # reports an error state instead of a silently dead mic.
        self.assertIn("mic watchdog", proc.stderr, msg=detail)
        self.assertIn("giving up", proc.stderr, msg=detail)
        self.assertNotEqual(proc.returncode, 0, msg=detail)


class TestMicWatchdogTimeoutArg(unittest.TestCase):
    """The watchdog timeout CLI rejects values that break queue.get()."""

    def test_rejects_non_positive_and_non_finite_timeout(self) -> None:
        for raw in ("0", "-1", "nan", "inf", "-inf", "abc"):
            proc = subprocess.run(
                [
                    "uv", "run", "python", "-m",
                    "kiss.agents.vscode.voice_wake",
                    f"--mic-watchdog-timeout={raw}",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            detail = (
                f"raw={raw!r} stdout={proc.stdout!r} "
                f"stderr={proc.stderr[-1000:]}"
            )
            self.assertEqual(proc.returncode, 2, msg=detail)
            self.assertNotIn("READY", proc.stdout, msg=detail)
            self.assertIn("mic-watchdog-timeout", proc.stderr, msg=detail)
            self.assertIn("positive finite", proc.stderr, msg=detail)


if __name__ == "__main__":
    unittest.main()
