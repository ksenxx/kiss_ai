# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``persistence._close_db`` bumps ``_db_generation`` under ``_init_tables_lock``.

The generation counter retires every other thread's cached connection
lazily: ``_get_db`` compares its thread-local snapshot against it.
``_recover_orphaned_sidecars_locked`` bumps it under
``_init_tables_lock``; ``_close_db`` used to bump it with a bare
``_db_generation += 1``.  Two unsynchronised read-modify-writes lose
one increment, and a thread that reconnected at the lost generation
keeps a handle to a database the close meant to abandon.

The test runs ``_close_db`` from several threads at once (the same
read-modify-write as the sidecar-recovery pairing, without needing a
sidecar deletion to line up) and checks that no increment is lost.

Honest limit: on the GIL build of CPython 3.14 this test also passed
BEFORE the fix.  ``dis`` shows ``LOAD_GLOBAL / BINARY_OP /
STORE_GLOBAL`` with no eval-breaker check point in between, so the
GIL cannot switch threads inside the read-modify-write; the lost
increment needs a free-threaded build.  The lock makes the invariant
explicit instead of relying on that bytecode detail, and this test
guards it where it can be observed.
"""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

import kiss.agents.sorcar.persistence as th

_THREADS = 8
_CALLS = 3000


def _close_repeatedly(barrier: threading.Barrier) -> None:
    barrier.wait()
    for _ in range(_CALLS):
        th._close_db()


def test_concurrent_close_db_never_loses_a_generation_bump() -> None:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    old_interval = sys.getswitchinterval()
    with tempfile.TemporaryDirectory() as tmp:
        kiss_dir = Path(tmp) / ".kiss"
        kiss_dir.mkdir()
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None
        try:
            th._get_db()
            th._close_db()
            before = th._db_generation
            barrier = threading.Barrier(_THREADS)
            threads = [
                threading.Thread(target=_close_repeatedly, args=(barrier,), daemon=True)
                for _ in range(_THREADS)
            ]
            sys.setswitchinterval(1e-6)
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=120)
            assert not any(t.is_alive() for t in threads)
            assert th._db_generation - before == _THREADS * _CALLS
        finally:
            sys.setswitchinterval(old_interval)
            th._close_db()
            th._DB_PATH, th._db_conn, th._KISS_DIR = saved
