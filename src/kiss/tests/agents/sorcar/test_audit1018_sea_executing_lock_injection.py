# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``sea_settings._EXECUTING`` survives the runner's Stop injection (audit 2026-10-18).

:func:`~kiss.agents.sorcar.sea_settings.execute_python_file` is the one
loader of SEAs, and it runs on task threads: the dispatcher's
``run_agent`` with a ``.py`` path, the model-picker probe, a SEA that
loads another script.  It serialises executions of one file under
``_EXECUTING``, a daemon-wide ``Condition``.  Taken with a plain
``with``, a Stop delivered right after its ``__enter__`` returned
(CPython gh-74174, reliable when the acquire blocked on another holder)
left the lock owned by the dying thread, and every later SEA load in
the daemon hung on it.  Both critical sections now go through
:func:`~kiss.core.locked.locked_call`.

The hammer runs in a child process (a leaked lock would wedge pytest):
a contender holds ``_EXECUTING`` in 1 ms bursts so the worker's acquire
blocks, the worker loops a real SEA load while hundreds of interrupts
are injected into it, and afterwards a fresh thread must still load a
SEA.  Fails on the pre-fix ``with`` sites, passes on the fixed ones.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_settings
from kiss.tests.lock_injection import hammer


class LoadSea:
    """One real ``execute_python_file`` of a tiny script on the worker."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.loaded = 0

    def __call__(self) -> None:
        try:
            namespace = sea_settings.execute_python_file(self.path)
        except sea_settings.SeaError:
            return  # an interrupt that landed inside the exec, wrapped by the loader
        assert namespace["ANSWER"] == 42
        self.loaded += 1


class Loader(threading.Thread):
    """A fresh thread loading the script once; ``namespace`` is set when it got through."""

    def __init__(self, path: str) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.namespace: dict[str, object] | None = None

    def run(self) -> None:
        self.namespace = sea_settings.execute_python_file(self.path)


def run_hammer() -> None:
    """Child-process body: hammer the SEA loader and check ``_EXECUTING`` stays usable."""
    logging.getLogger(sea_settings.__name__).setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory() as folder:
        script = Path(folder) / "audit1018_sea.py"
        script.write_text("ANSWER = 42\n", encoding="utf-8")
        driver = LoadSea(str(script))
        # The lock is checked once the contender is gone: it holds it most of the time.
        hammer(driver, {"_EXECUTING": sea_settings._EXECUTING}, contend=sea_settings._EXECUTING)
        assert driver.loaded > 0, "the worker never completed a load"
        # The daemon's next SEA load, on another thread, still goes through.
        loader = Loader(str(script))
        loader.start()
        loader.join(10.0)
        assert not loader.is_alive(), "a later SEA load hung on the leaked lock"
        assert loader.namespace is not None and loader.namespace["ANSWER"] == 42
        assert not sea_settings._EXECUTING_OWNER


def test_sea_executing_lock_survives_injection() -> None:
    """No ``_EXECUTING`` site a task thread runs in the SEA loader leaks it under injection."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer()"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]


if __name__ == "__main__":
    pytest.main([__file__])
