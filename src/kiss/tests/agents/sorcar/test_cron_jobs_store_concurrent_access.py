# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The cron job store survives readers and writers racing on Windows.

``save_jobs`` used to stage its temp file and ``os.replace`` it itself.
On Windows a ``load_jobs`` in flight (the web UI listing jobs, a CLI
``list``) holds the store open without ``FILE_SHARE_DELETE``, so that
replace raised ``PermissionError``, the scheduler tick logged
``Scheduler tick failed: [WinError 5]`` and the job's result was lost.
Both functions now go through the shared helpers in
:mod:`kiss.core.utils`, which wait out the other side.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.cron_agent import load_jobs, save_jobs
from kiss.tests.agents.sorcar.test_cron_agent import _isolated_kiss_home  # noqa: F401
from kiss.tests.conftest import HOT_READER_PAUSE

_WRITERS = 4
_ROUNDS = 60


def _writer(tag: int, errors: list[str]) -> None:
    for i in range(_ROUNDS):
        try:
            save_jobs([{"id": f"job-{tag}", "last_summary": f"round {i}", "pad": "x" * 2000}])
        except OSError as exc:
            errors.append(f"writer {tag}: {exc!r}")
            return


@pytest.mark.timeout(120)
def test_concurrent_save_and_load_never_fail_or_read_empty(tmp_path: Path) -> None:
    """Writers never raise and a hot reader never sees a missing store."""
    save_jobs([{"id": "seed"}])
    stop = threading.Event()
    errors: list[str] = []
    empties = [0]
    reads = [0]

    def _reader() -> None:
        while not stop.is_set():
            time.sleep(HOT_READER_PAUSE)
            jobs = load_jobs()
            reads[0] += 1
            if not jobs:
                empties[0] += 1

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    writers = [threading.Thread(target=_writer, args=(i, errors)) for i in range(_WRITERS)]
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    reader.join(timeout=10)

    assert errors == []
    assert reads[0] > 0, "reader never ran"
    assert empties[0] == 0, f"{empties[0]} of {reads[0]} reads saw an empty store"
    assert load_jobs()[0]["id"].startswith("job-")
    assert sorted(p.name for p in (tmp_path / "cron").iterdir()) == ["jobs.json"], (
        "temp residue left next to jobs.json"
    )
