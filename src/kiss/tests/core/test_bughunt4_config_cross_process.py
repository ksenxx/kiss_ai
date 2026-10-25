# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: ``save_config`` must be safe across *processes*.

``user_config.save_config`` serialises its read-merge-replace of
``~/.kiss/config.json`` with ``_config_lock`` — a ``threading.Lock``,
which only protects callers inside ONE process.  In production two
processes routinely write the same config concurrently (the ``kiss-web``
daemon persisting ``remote_password``/``tunnel`` state while a second
daemon or the CLI persists ``last_model`` or a settings toggle).
Two processes that each read the same old file, overlay only their own
key, and ``os.replace`` it back drop one another's update.

Reproduction: two real subprocesses hammer ``save_config`` with
*different* keys; after every save each process reads the config back
and checks its own key still holds the value it just wrote (no other
process ever writes that key, so any regression proves a stale-read
clobber).  With only the in-process threading lock this fails almost
immediately; with a cross-process file lock it never fails.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import TestCase

_WORKER = r"""
import json, os, sys
key = sys.argv[1]
n = int(sys.argv[2])
start_file = sys.argv[3]
# Wait for the parent to release both workers at once.
import time
deadline = time.time() + 10
while not os.path.exists(start_file):
    if time.time() > deadline:
        sys.exit(2)
    time.sleep(0.001)
from kiss.core.user_config import load_config, save_config
for i in range(n):
    value = f"{key}-{i}"
    save_config({key: value})
    seen = load_config().get(key)
    if seen != value:
        print(f"LOST UPDATE: wrote {value!r} read back {seen!r}", flush=True)
        sys.exit(1)
sys.exit(0)
"""

_RECENT_WORKER = r"""
import os, sys, time
from kiss.core.user_config import record_recent_work_dir
dirs = sys.argv[1].split(os.pathsep)
rounds = int(sys.argv[2])
start_file = sys.argv[3]
open(sys.argv[4], "w").close()  # imported: ready to race
deadline = time.time() + 10
while not os.path.exists(start_file):
    if time.time() > deadline:
        sys.exit(2)
    time.sleep(0.001)
for _ in range(rounds):
    for d in dirs:
        record_recent_work_dir(d)
sys.exit(0)
"""


class TestRecordRecentWorkDirCrossProcess(TestCase):
    """``record_recent_work_dir`` must merge under the cross-process lock.

    Two workers each record their own ten directories over and over
    (twenty in all, under the thirty-row cap).  A record that reads
    the list outside the file lock and then stores it whole replaces
    the other worker's rows with its stale copy; in lock-step the two
    processes keep clobbering each other, and the file ends up holding
    one worker's rows only.  Every row of both must survive.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_two_processes_keep_each_others_entries(self) -> None:
        env = dict(os.environ)
        env["KISS_HOME"] = str(Path(self.tmpdir) / ".kiss")
        start_file = str(Path(self.tmpdir) / "go")
        procs = []
        ready_files = []
        for name in ("a", "b"):
            dirs = [Path(self.tmpdir) / f"{name}{i}" for i in range(10)]
            for d in dirs:
                d.mkdir()
            ready_files.append(Path(self.tmpdir) / f"ready-{name}")
            procs.append(subprocess.Popen(
                [sys.executable, "-c", _RECENT_WORKER,
                 os.pathsep.join(str(d) for d in dirs), "300", start_file,
                 str(ready_files[-1])],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
        # Release both only once both have imported, so the loops overlap.
        deadline = time.monotonic() + 60
        while not all(f.exists() for f in ready_files) and time.monotonic() < deadline:
            time.sleep(0.01)
        Path(start_file).touch()
        for proc in procs:
            out, err = proc.communicate(timeout=120)
            self.assertEqual(proc.returncode, 0, f"stdout={out!r} stderr={err[-500:]!r}")
        stored = json.loads((Path(env["KISS_HOME"]) / "config.json").read_text())
        self.assertEqual(
            sorted(e["path"] for e in stored["recent_work_dirs"]),
            sorted(str(Path(self.tmpdir) / f"{name}{i}") for name in "ab" for i in range(10)),
        )


class TestSaveConfigCrossProcess(TestCase):
    """Concurrent ``save_config`` from two processes must not lose updates."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_two_processes_do_not_clobber_each_other(self) -> None:
        env = dict(os.environ)
        env["KISS_HOME"] = str(Path(self.tmpdir) / ".kiss")
        start_file = str(Path(self.tmpdir) / "go")
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", _WORKER, key, "300", start_file],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for key in ("work_dir", "last_model")
        ]
        Path(start_file).touch()
        failures: list[str] = []
        for proc in procs:
            out, err = proc.communicate(timeout=120)
            if proc.returncode != 0:
                failures.append(
                    f"exit={proc.returncode} stdout={out.strip()!r} "
                    f"stderr={err.strip()[-500:]!r}"
                )
        self.assertEqual(
            failures, [],
            "save_config lost updates between two concurrent processes: "
            + "; ".join(failures),
        )
