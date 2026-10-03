# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-26 (core): ``MY_MODELS.json`` edits use ``exclusive_file_lock``.

``model_info._my_models_flock`` inlined its own open / ``lock_exclusive``
/ ``try: ... finally: unlock`` sequence, a copy of
``file_lock.exclusive_file_lock`` that the 2026-09-24 audit had already
folded out of ``vscode_config``.  It now delegates to the helper.  The
contract the custom-model CRUD relies on is pinned end to end with real
processes sharing one home directory:

* two processes adding models concurrently never drop each other's entry;
* the sidecar lock is released after each edit, so a later edit in the
  same process (and a delete) still goes through;
* the sidecar lives next to the registry, never on the registry itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_WORKER = r"""
import os, sys, time
from kiss.core.models.model_info import save_custom_model
tag, start_file, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
deadline = time.time() + 30
while not os.path.exists(start_file):
    if time.time() > deadline:
        sys.exit(2)
    time.sleep(0.001)
for i in range(n):
    err = save_custom_model(f"audit-{tag}-{i}", endpoint="http://127.0.0.1:1/v1")
    if err is not None:
        print(err, file=sys.stderr)
        sys.exit(3)
"""


def _run_workers(home: Path, n: int, tags: list[str]) -> None:
    """Start one ``save_custom_model`` worker per tag with ``HOME=home`` and wait.

    The registry lives in ``$KISS_HOME`` (``~/.kiss`` by default), so the
    test runner's own ``KISS_HOME`` is dropped: ``HOME`` alone decides.
    """
    env = dict(os.environ, HOME=str(home), USERPROFILE=str(home))
    env.pop("KISS_HOME", None)
    start_file = home / "start"
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, tag, str(start_file), str(n)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for tag in tags
    ]
    time.sleep(1.0)  # let the interpreters import before releasing them together
    start_file.write_text("go")
    for proc in procs:
        _, err = proc.communicate(timeout=180)
        assert proc.returncode == 0, err.decode(errors="replace")


def test_concurrent_processes_never_drop_a_custom_model() -> None:
    """Every model added by either process survives in ``MY_MODELS.json``."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        home = Path(tmp_dir)
        _run_workers(home, n=25, tags=["a", "b"])
        registry = home / ".kiss" / "MY_MODELS.json"
        names = set(json.loads(registry.read_text(encoding="utf-8")))
        expected = {f"audit-{tag}-{i}" for tag in ("a", "b") for i in range(25)}
        assert expected <= names
        assert (home / ".kiss" / ".MY_MODELS.json.kiss.lock").is_file()


_SEQUENTIAL = r"""
import json
from pathlib import Path
from kiss.core.models.model_info import (
    USER_MY_MODELS_PATH, delete_custom_model, save_custom_model,
)
assert save_custom_model("audit-one", endpoint="http://127.0.0.1:1/v1") is None
assert save_custom_model("audit-two", endpoint="http://127.0.0.1:1/v1") is None
assert save_custom_model("audit-three", original_name="audit-one") is None
assert delete_custom_model("audit-two") is None
names = set(json.loads(USER_MY_MODELS_PATH.read_text(encoding="utf-8")))
assert "audit-three" in names and not names & {"audit-one", "audit-two"}, names
print("ok")
"""


def test_lock_released_between_edits_in_one_process() -> None:
    """Back-to-back add / rename / delete in one process never self-deadlock."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        env = dict(os.environ, HOME=tmp_dir, USERPROFILE=tmp_dir)
        result = subprocess.run(
            [sys.executable, "-c", _SEQUENTIAL],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"
