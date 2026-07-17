# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Race-condition regression test for ``vscode_config.save_config``.

``save_config`` performs a read-modify-write on ``~/.kiss/config.json``:
it loads the existing file, overlays the caller's keys, and atomically
replaces the file.  When several threads call it concurrently (e.g. an
agent's ``update_settings`` tool persisting ``last_model`` while the VS
Code command handler persists a settings toggle), each thread reads the
SAME old file, applies ONLY its own key, and the last ``os.replace``
wins — silently dropping every other thread's update (a lost-update
race).

These tests force the interleaving by widening the window between the
read and the atomic replace (a small sleep inside the patched
``mkstemp``), then assert that every concurrent writer's value survives.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import kiss.core.vscode_config as vc

# Distinct (key, value) pairs drawn from ``DEFAULTS`` so each writer
# touches a different key with a non-default value.  ``save_config``
# only persists keys present in ``DEFAULTS``.
_DISTINCT_WRITES: dict[str, Any] = {
    "max_budget": 222,
    "custom_endpoint": "http://endpoint.example",
    "custom_headers": "X-Race:1",
    "custom_api_key": "race-api-key",
    "remote_password": "race-password",
    "work_dir": "/race/work/dir",
    "last_model": "race-model",
    "demo_mode": True,
    "auto_commit_mode": False,
    "is_parallel": False,
    "is_worktree": False,
    "use_web_browser": False,
}


class TestSaveConfigConcurrencyNoLostUpdate(unittest.TestCase):
    """Concurrent ``save_config`` calls must not lose updates."""

    def setUp(self) -> None:
        self._orig_dir = vc.CONFIG_DIR
        self._orig_path = vc.CONFIG_PATH
        self._orig_mkstemp = vc.tempfile.mkstemp
        self._tmpdir = tempfile.mkdtemp()
        vc.CONFIG_DIR = Path(self._tmpdir)
        vc.CONFIG_PATH = Path(self._tmpdir) / "config.json"

    def tearDown(self) -> None:
        vc.CONFIG_DIR = self._orig_dir
        vc.CONFIG_PATH = self._orig_path
        vc.tempfile.mkstemp = self._orig_mkstemp
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _install_slow_mkstemp(self) -> None:
        """Widen the read-modify-write window.

        ``mkstemp`` is called AFTER ``save_config`` has read the old
        file and built its merged dict, but BEFORE ``os.replace``.
        Sleeping here lets every concurrent thread read the same old
        file before any of them publishes, maximising the lost-update
        window so the race reproduces deterministically.
        """
        orig = self._orig_mkstemp

        def slow_mkstemp(*args: Any, **kwargs: Any) -> Any:
            time.sleep(0.05)
            return orig(*args, **kwargs)

        vc.tempfile.mkstemp = slow_mkstemp

    def test_concurrent_writes_all_survive(self) -> None:
        # Seed an initial file so every writer reads a real baseline.
        vc.save_config({"max_budget": 100})
        self._install_slow_mkstemp()

        barrier = threading.Barrier(len(_DISTINCT_WRITES))
        errors: list[BaseException] = []

        def writer(key: str, value: Any) -> None:
            try:
                barrier.wait()
                vc.save_config({key: value})
            except BaseException as exc:  # pragma: no cover - defensive
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(k, v))
            for k, v in _DISTINCT_WRITES.items()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"writer threads raised: {errors}"

        cfg = vc.load_config()
        missing = {
            k: v for k, v in _DISTINCT_WRITES.items() if cfg.get(k) != v
        }
        assert not missing, (
            f"lost updates from concurrent save_config: {missing} "
            f"(config on disk: { {k: cfg.get(k) for k in _DISTINCT_WRITES} })"
        )


if __name__ == "__main__":
    unittest.main()
