# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests reproducing bugs found in the vscode helper modules.

No mocks, patches, fakes, or test doubles. All tests use real objects,
real temp files, and real git repositories.
"""

from __future__ import annotations

from pathlib import Path

from kiss.server.server import VSCodeServer


def test_autocomplete_survives_binary_active_file(tmp_path: Path) -> None:
    """A non-UTF-8 active file must not crash the autocomplete pipeline.

    Bug: ``_active_file_identifier_matches`` reads the active editor
    file from disk in text mode and catches only ``OSError``.  A binary
    (non-UTF-8) active file raises ``UnicodeDecodeError``, which
    propagates out of ``_complete`` and permanently kills the single
    ``_complete_worker_loop`` thread — autocomplete then stays dead for
    the rest of the daemon's life because ``_ensure_complete_worker``
    sees the (dead) worker as already started.
    """
    binary_file = tmp_path / "image.bin"
    binary_file.write_bytes(b"\xff\xfe\x00\x01\x80binary\x00data\xff")
    server = VSCodeServer()

    matches = server._active_file_identifier_matches("fo", str(binary_file), "")

    assert matches == []
