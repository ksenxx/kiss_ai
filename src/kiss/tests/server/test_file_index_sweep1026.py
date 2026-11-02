"""``FileIndex.scan`` leaves the process-wide cyclic collector alone.

It used to ``gc.disable()`` for the walk, which stopped every other
thread's cyclic garbage from being collected for the whole scan.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from types import FrameType
from typing import Any

from kiss.server import file_index
from kiss.server.file_index import FileIndex


def test_scan_keeps_the_collector_enabled(tmp_path: Path) -> None:
    """The collector is on at every line the scan executes.

    The state is sampled by a trace function on the scanning thread,
    at each line of ``file_index`` code the scan runs: a sampler
    thread could go unscheduled for the whole 100 ms scan under a
    full-suite load (2026-11-01 run), and a sample taken by the scan
    itself cannot miss a ``gc.disable()`` anywhere in the walk.
    """
    for d in range(60):
        folder = tmp_path / f"dir{d}"
        folder.mkdir()
        for f in range(100):
            (folder / f"file{f}.txt").write_text("x")
    assert gc.isenabled()
    samples: list[bool] = []

    def trace(frame: FrameType, event: str, _arg: Any) -> Any:
        """Sample the collector's state at every line of the scan's own code."""
        if event == "line" and frame.f_code.co_filename == file_index.__file__:
            samples.append(gc.isenabled())
        return trace

    sys.settrace(trace)
    try:
        index = FileIndex.scan(str(tmp_path))
    finally:
        sys.settrace(None)
    assert len(index.paths) == 6000 + 60
    assert len(samples) > 6000, "the scan ran fewer lines than it has files"
    assert all(samples), f"the collector was off for {samples.count(False)} samples"
    assert gc.isenabled()
