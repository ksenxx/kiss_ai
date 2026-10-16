# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``read_bytes_waiting_for_writer`` rides out Windows's non-atomic replace.

``os.replace`` on Windows unlinks the old name before the staged file
takes it, so a reader racing :func:`atomic_write_text` can get
``FileNotFoundError`` for a file that is never meant to be missing (seen
on a loaded Windows VM: a hot reader of ``cron/jobs.json`` found "no such
file" on most of its reads and the cron store looked empty).  The reader
now waits while the writer's staged ``.<name>-*`` sibling exists; a file
that is genuinely absent still raises at once, on every platform.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from kiss.core.utils import read_bytes_waiting_for_writer


def _publish_after(stage: Path, target: Path, delay: float) -> None:
    time.sleep(delay)
    os.replace(stage, target)


def test_missing_target_with_staged_sibling_waits_for_the_publish(tmp_path: Path) -> None:
    """The gap inside a replace: target gone, ``.<name>-*`` sibling present."""
    # Brackets in the name: the sibling check must not treat it as a glob.
    target = tmp_path / "jobs[1].json"
    stage = tmp_path / ".jobs[1].json-abc123"
    stage.write_bytes(b"[1]")
    if os.name != "nt":
        # POSIX replaces atomically, so the gap never exists and nothing
        # is retried: nobody is publishing yet, so this must raise.
        with pytest.raises(FileNotFoundError):
            read_bytes_waiting_for_writer(target)
        os.replace(stage, target)
    else:
        publisher = threading.Thread(target=_publish_after, args=(stage, target, 0.15))
        publisher.start()
        try:
            assert read_bytes_waiting_for_writer(target) == b"[1]"
        finally:
            publisher.join()
    assert target.read_bytes() == b"[1]"


def test_missing_target_without_staged_sibling_raises_at_once(tmp_path: Path) -> None:
    """A genuinely absent file costs no retry budget (10 s on Windows otherwise)."""
    # An unrelated staged sibling (another file's write) must not count.
    (tmp_path / ".jobs1.json-abc123").write_bytes(b"[]")
    start = time.monotonic()
    with pytest.raises(FileNotFoundError):
        read_bytes_waiting_for_writer(tmp_path / "jobs[1].json")
    assert time.monotonic() - start < 1.0
