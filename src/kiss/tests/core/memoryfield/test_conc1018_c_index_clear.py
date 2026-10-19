"""``VectorIndex.clear`` tolerates a concurrent clear of the same index file.

The old exists-then-unlink raised ``FileNotFoundError`` when another
thread or process (a parallel ``evaluate.py`` run) removed the file
between the two calls.  Eight threads released by a barrier hit that
window in most rounds, so a few rounds pin the fix.
"""

import sys
import threading
from pathlib import Path

from kiss.core.memoryfield.index import VectorIndex, hashed_embedding
from kiss.core.memoryfield.pages import MemoryDir


def test_concurrent_clear_never_raises(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    memory.write("p", "A page to index.")
    index = VectorIndex(memory, embed=hashed_embedding)
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(40):
            index.sync()
            assert index.path.exists()
            barrier = threading.Barrier(8)
            errors: list[BaseException] = []
            threads = [
                threading.Thread(target=_clear_after, args=(barrier, index, errors))
                for _ in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert errors == []
            assert not index.path.exists()
    finally:
        sys.setswitchinterval(previous)
    # Clearing an index that was never built is a no-op too.
    index.clear()
    assert index.sync().added == 1


def _clear_after(barrier: threading.Barrier, index: VectorIndex, errors: list[Exception]) -> None:
    barrier.wait()
    try:
        index.clear()
    except Exception as exc:
        errors.append(exc)
