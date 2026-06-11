# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: ``save_config`` must be safe across *processes*.

``vscode_config.save_config`` serialises its read-merge-replace of
``~/.kiss/config.json`` with ``_config_lock`` — a ``threading.Lock``,
which only protects callers inside ONE process.  In production two
daemons routinely write the same config concurrently (the ``kiss-web``
daemon persisting ``remote_password``/``tunnel`` state while a VS Code
window's service daemon persists ``last_model`` or a settings toggle).
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

import os
import shutil
import subprocess
import sys
import tempfile
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
from kiss.agents.vscode.vscode_config import load_config, save_config
for i in range(n):
    value = f"{key}-{i}"
    save_config({key: value})
    seen = load_config().get(key)
    if seen != value:
        print(f"LOST UPDATE: wrote {value!r} read back {seen!r}", flush=True)
        sys.exit(1)
sys.exit(0)
"""


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
