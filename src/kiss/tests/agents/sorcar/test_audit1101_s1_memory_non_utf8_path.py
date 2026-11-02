# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A repository whose path is not valid UTF-8 still gets its domain memory.

``_repo_memory_domains`` reads the checkout path from ``git rev-parse``.
A POSIX path is bytes; one holding ``0xff`` is legal, and git prints it as
is.  Decoding that output strictly raised ``UnicodeDecodeError`` out of
the run's setup, failing every memory-enabled task in such a checkout
before its first model step (audit 2026-11-01, S1 F1).  The path is now
decoded the way Python decodes file names (``os.fsdecode``), so the
repository is named like any other.  Runs against a real git repository.
"""

from __future__ import annotations

import errno
import os
import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar.sorcar_agent import _repo_memory_domains
from kiss.tests.conftest import posix_only

pytestmark = posix_only("a non-UTF-8 path is a POSIX file name")


def test_a_non_utf8_checkout_path_names_its_repository_memory(tmp_path: Path) -> None:
    """The byte ``0xff`` in the checkout's name neither aborts nor changes the memory name."""
    repo_bytes = os.path.join(os.fsencode(tmp_path), b"My_Project-\xff-v2")
    try:
        os.mkdir(repo_bytes)
    except OSError as exc:  # APFS (macOS) refuses names that are not valid UTF-8
        if exc.errno != errno.EILSEQ:
            raise
        pytest.skip("filesystem rejects non-UTF-8 file names")
    repo = os.fsdecode(repo_bytes)
    subprocess.run(["git", "init", "-q", repo], check=True, capture_output=True, timeout=30)
    (Path(repo) / "src").mkdir()
    expected = {"my-project-v2": f"memory of the repository {Path(repo).resolve()}"}
    assert _repo_memory_domains(repo) == expected
    assert _repo_memory_domains(str(Path(repo) / "src")) == expected
