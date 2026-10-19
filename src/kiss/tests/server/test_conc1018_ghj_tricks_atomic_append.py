# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a reader never sees a cut ``## Trick`` body while the Add button appends one.

``read_tricks()`` (the autocomplete worker on every keystroke, the page
bootstrap) reads ``~/.kiss/MY_INJECTION.md`` with no lock; the delete
and edit rewrites are therefore atomic (temp file + rename).  The append
was not: ``open("a")`` + ``write`` lets a concurrent reader hit the end
of the file while the body is only partly there, and the parser returns
the prefix as a trick, which the panel then shows and the composer
offers as a ghost completion.  Reproduced on the old code with real
threads: at 4 MiB bodies about one append in ten was observed cut
(usually at a 4096-byte page boundary).  The append is now
``atomic_write_text`` like its siblings.

The test is a race test by nature: it stops at the first torn read and
fails; the fixed code runs every round clean.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from kiss.server.tricks import (
    append_my_injection_trick,
    delete_my_injection_trick,
    read_tricks,
)

_ROUNDS = 80
_BODY_SIZE = 4 * 1024 * 1024
_READERS = 3


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("KISS_HOME", str(home))
    bundled = tmp_path / "INJECTIONS.md"
    bundled.write_text("", encoding="utf-8")
    monkeypatch.setenv("KISS_INJECTIONS_PATH", str(bundled))
    return home


def _is_whole(trick: str) -> bool:
    return trick.startswith("trick ") and trick.endswith("|end") and len(trick) > _BODY_SIZE


def test_readers_never_observe_a_partly_appended_trick(isolated_home: Path) -> None:
    seed = set(read_tricks())
    assert seed
    torn: list[str] = []
    done = threading.Event()

    def reader() -> None:
        while not done.is_set():
            for trick in read_tricks():
                if trick not in seed and not _is_whole(trick):
                    torn.append(f"{trick[:12]}... len={len(trick)}")
                    done.set()
                    return

    readers = [threading.Thread(target=reader, daemon=True) for _ in range(_READERS)]
    for t in readers:
        t.start()
    try:
        for i in range(_ROUNDS):
            if done.is_set():
                break
            body = f"trick {i} " + "x" * _BODY_SIZE + "|end"
            assert append_my_injection_trick(body) is None
            assert delete_my_injection_trick(body) is None
    finally:
        done.set()
        for t in readers:
            t.join(timeout=30)
    assert torn == []
    assert set(read_tricks()) == seed
