# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``_atomic_publish`` survives a concurrent reader on Windows.

Windows refuses ``Path.replace`` over a file that another handle holds
open without ``FILE_SHARE_DELETE`` (``[WinError 5] Access is denied``).
Python's ``open()`` never sets that flag, so a process polling the file
(``Path.read_text`` in a loop, an editor, a virus scanner) made the
web editor's save fail intermittently on Windows.  ``_atomic_publish``
now retries the rename for up to a second; this test runs a real
reader thread against it.  On POSIX the rename never fails, so the
test just proves the retry does not change the happy path.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from kiss.server.web_server import _atomic_write_text


def _poll_reads(path: Path, stop: threading.Event) -> None:
    # A poller like the content-tab tests' ``_wait_for_disk`` (or an
    # editor's file watcher): with plain ``Path.replace`` this makes a
    # few of every 200 publishes fail on Windows.
    while not stop.is_set():
        try:
            path.read_bytes()
        except (PermissionError, FileNotFoundError):
            pass
        time.sleep(0.002)


def test_publish_succeeds_while_another_thread_polls_the_file(tmp_path: Path) -> None:
    target = tmp_path / "held.txt"
    target.write_text("v0", encoding="utf-8")
    stop = threading.Event()
    reader = threading.Thread(target=_poll_reads, args=(target, stop), daemon=True)
    reader.start()
    try:
        for i in range(1, 201):
            _atomic_write_text(target, f"v{i}")
    finally:
        stop.set()
        reader.join(timeout=5)
    assert target.read_text(encoding="utf-8") == "v200"
    assert [p.name for p in tmp_path.iterdir()] == ["held.txt"]
