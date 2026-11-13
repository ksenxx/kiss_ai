"""Failure paths of ``persistence`` side files (audit S1, 2026-10-30).

1. ``_process_owner_token`` creates ``<KISS_HOME>/task-owners/<token>.lock``
   and writes the pid into it.  When the open succeeded but the write
   failed (full disk, quota, file-size limit), it returned ``""`` with the
   file still on disk and its handle left open.  ``_owner_state`` stayed
   empty, so every later ``_add_task`` minted another token and left one
   more file that no row names and nothing ever deletes.  The write
   failure here is real: the child process lowers its own
   ``RLIMIT_FSIZE`` to 0 with ``SIGXFSZ`` ignored, so the marker's flush
   fails with ``EFBIG`` the way it fails on a full disk.
2. ``_journal_file_lock`` logged "proceeding without it" when a peer kept
   the journal lock past the timeout, but no caller proceeds: an append
   goes to a spill file and a replay or prune skips the round.  The lock
   is held for real, through a second open file description (``flock``
   locks conflict between those even inside one process).
3. ``_spill_journal_lines`` stages a spill as ``<sidecar>.writing-*`` and
   renames it into place.  A write the OS refused left the staging file
   behind for good: nothing reads or deletes that name again.  Refused
   for real, as in 1.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence
from kiss.core.file_lock import lock_exclusive

_FAILING_MARKER_CHILD = """
import json, resource, signal
from kiss.agents.sorcar import persistence

def markers():
    return sorted(p.name for p in persistence._owner_dir().iterdir())

signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
hard = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
resource.setrlimit(resource.RLIMIT_FSIZE, (0, hard))
failed = [persistence._process_owner_token() for _ in range(3)]
after_failures = markers()
resource.setrlimit(resource.RLIMIT_FSIZE, (hard, hard))
token = persistence._process_owner_token()
print(json.dumps({
    "failed": failed, "after_failures": after_failures,
    "token": token, "after_success": markers(),
}))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX RLIMIT_FSIZE")
def test_failed_marker_write_leaves_no_marker_file(tmp_path: Path) -> None:
    """Failed publishes return "" and leave no file; a later one succeeds."""
    proc = subprocess.run(
        [sys.executable, "-c", _FAILING_MARKER_CHILD],
        env={**os.environ, "KISS_HOME": str(tmp_path)},
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["failed"] == ["", "", ""]
    assert report["after_failures"] == [], "a failed publish left its marker file"
    assert "could not publish task-owner liveness marker" in proc.stderr
    assert "Exception ignored" not in proc.stderr, "a failed marker's handle was left open"
    assert report["token"]
    assert report["after_success"] == [f"{report['token']}.lock"]


_FAILING_SPILL_CHILD = """
import os, resource, signal, sys
from kiss.agents.sorcar import persistence

signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (0, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))
try:
    persistence._spill_journal_lines(sys.argv[1], ["a record\\n"])
except OSError:
    print("refused", sorted(os.listdir(os.path.dirname(sys.argv[1]))))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX RLIMIT_FSIZE")
def test_failed_spill_write_leaves_no_staging_file(tmp_path: Path) -> None:
    """A spill the OS refuses to write raises and leaves no ``.writing-*`` file."""
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", _FAILING_SPILL_CHILD, str(journal_dir / "events.journal")],
        env={**os.environ, "KISS_HOME": str(tmp_path / "home")},
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.stdout.strip() == "refused []", proc.stdout + proc.stderr
    # A staging file that was never created has nothing to remove: the open's failure is raised.
    with pytest.raises(FileNotFoundError):
        persistence._spill_journal_lines(str(tmp_path / "gone" / "events.journal"), ["x\n"])


def test_journal_lock_timeout_warning_names_the_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The timeout warning says what the caller does instead of the locked write."""
    sidecar = str(tmp_path / "events.journal")
    with open(sidecar + ".lock", "a+", encoding="utf-8") as holder:
        assert lock_exclusive(holder, blocking=False)
        with (
            caplog.at_level(logging.WARNING, logger=persistence.logger.name),
            persistence._journal_file_lock(sidecar, timeout=0.1) as locked,
        ):
            assert locked is False
    [message] = [r.getMessage() for r in caplog.records if "journal lock" in r.getMessage()]
    assert "proceeding without it" not in message
    assert "spill file" in message and "next round" in message
