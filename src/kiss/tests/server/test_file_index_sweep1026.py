"""``FileIndex.scan`` leaves the process-wide cyclic collector alone.

It used to ``gc.disable()`` for the walk, which stopped every other
thread's cyclic garbage from being collected for the whole scan.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path

from kiss.server.file_index import FileIndex


def test_scan_keeps_the_collector_enabled(tmp_path: Path) -> None:
    for d in range(60):
        folder = tmp_path / f"dir{d}"
        folder.mkdir()
        for f in range(100):
            (folder / f"file{f}.txt").write_text("x")
    assert gc.isenabled()
    scanning = threading.Event()
    finished = threading.Event()
    samples: list[bool] = []

    def sample_collector_state() -> None:
        scanning.wait(10)
        while not finished.is_set():
            samples.append(gc.isenabled())

    sampler = threading.Thread(target=sample_collector_state, daemon=True)
    sampler.start()
    scanning.set()
    try:
        index = FileIndex.scan(str(tmp_path))
    finally:
        finished.set()
        sampler.join(10)
    assert len(index.paths) == 6000 + 60
    assert samples, "the sampler never ran during the scan"
    assert all(samples), f"the collector was off for {samples.count(False)} samples"
    assert gc.isenabled()
