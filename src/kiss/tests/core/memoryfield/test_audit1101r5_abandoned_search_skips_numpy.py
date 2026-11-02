# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""A memory search abandoned by a Stop ends after its embedding batch; it does not search.

``MemoryTools._search`` leaves its sync-and-search worker running when
the task thread is stopped (``cancel`` set, pool shut down without a
join).  The worker used to go on to ``VectorIndex.search`` once the sync
had finished: a result nobody reads, computed seconds later on a pool
thread, whose lazy ``import numpy`` then ran in the background of
whatever the process did next.  In the test suite that half-initialised
``numpy`` module was found by ``pytest.approx`` (which reads
``sys.modules`` without the import lock) and failed an unrelated pricing
test with ``partially initialized module 'numpy' has no attribute
'isscalar'``.  The worker now returns as soon as the sync ends when the
search was cancelled.

The scenario runs in a fresh interpreter so that "numpy was never
imported" is observable: the sync path does not import numpy, only the
search path does.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from kiss.core.memoryfield.index import hashed_embedding
from kiss.core.memoryfield.tools import MemoryTools
from kiss.tests.lock_injection import ignore_thread_interrupt, inject

EMBED_SECONDS = 1.0
STOP_AFTER_SECONDS = 0.3


def slow_embedding(text: str) -> list[float]:
    """The offline hashed embedding, as slow as one network request per text."""
    time.sleep(EMBED_SECONDS)
    return hashed_embedding(text)


def _pool_threads() -> list[str]:
    return [
        t.name for t in threading.enumerate()
        if t.name.startswith("ThreadPoolExecutor") and t.is_alive()
    ]


def run_child() -> None:
    """Stop a first-time search during its sync, wait out the abandoned worker, report."""
    threading.excepthook = ignore_thread_interrupt
    root = Path(tempfile.mkdtemp(prefix="kiss-abandoned-search-"))
    tools = MemoryTools(root / "memory", embed=slow_embedding)
    tools.memory.write("page", "a page the first search has to embed")
    outcome: list[object] = []

    def call() -> None:
        try:
            outcome.append(tools.memory_search("a page"))
        except BaseException as e:  # noqa: BLE001 - the injected interrupt is the point
            outcome.append(e)

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    time.sleep(STOP_AFTER_SECONDS)
    assert worker.is_alive(), "the search finished before the stop"
    assert inject(worker) == 1
    worker.join(5.0)
    assert not worker.is_alive(), "the stop did not land while the sync ran"
    assert isinstance(outcome[0], KeyboardInterrupt), outcome
    # The abandoned pool workers finish their embedding and exit.
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and _pool_threads():
        time.sleep(0.05)
    print(json.dumps({"pool_threads": _pool_threads(), "numpy_imported": "numpy" in sys.modules}))


def test_a_stopped_search_does_not_search_after_its_sync() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_child()"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["pool_threads"] == [], report
    assert report["numpy_imported"] is False, report
