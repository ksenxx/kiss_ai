# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: ``tabs.json`` is never published torn by two registry instances.

``TabRegistry._save_locked`` staged every save in the FIXED sibling
``tabs.json.tmp`` (``Path.write_text`` truncates, then fills) before
``os.replace``-ing it into place.  The instance lock cannot serialise
a second instance — a sibling daemon, or a test next to a live daemon
— so writer B could truncate the temp inode writer A was about to
rename onto ``tabs.json``, publishing an empty or half-written
document.  The save must stage through a unique temp file (the shared
``kiss.core.utils.atomic_write_text``), which also guarantees a failed
publish leaves no residue behind.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from kiss.server.tab_registry import OpenTabOutcome, TabRegistry

_WRITERS = 8
_ROUNDS = 150


def _open_close_loop(
    reg: TabRegistry, tag: str, start: threading.Barrier,
) -> None:
    start.wait()
    title = f"title-{tag}-" + "x" * 190
    for i in range(_ROUNDS):
        tab_id = f"tab-{tag}-{i}"
        assert reg.open_tab(tab_id, title, f"/work/{tag}") is OpenTabOutcome.OPENED
        assert reg.close_tab(tab_id)


@pytest.mark.timeout(120)
def test_two_registry_instances_never_publish_a_torn_file(tmp_path: Path) -> None:
    path = tmp_path / "tabs.json"
    registries = [TabRegistry(path) for _ in range(_WRITERS)]
    start = threading.Barrier(_WRITERS + 1)
    stop = threading.Event()
    torn: list[str] = []
    reads = [0]

    def _reader() -> None:
        start.wait()
        while not stop.is_set():
            try:
                raw = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            reads[0] += 1
            try:
                doc = json.loads(raw)
            except json.JSONDecodeError:
                torn.append(raw)
                return
            if not isinstance(doc, dict) or not isinstance(doc.get("tabs"), list):
                torn.append(raw)
                return

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    writers = [
        threading.Thread(
            target=_open_close_loop, args=(registries[i], str(i), start),
        )
        for i in range(_WRITERS)
    ]
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    reader.join(timeout=10)
    assert reads[0] > 0, "reader never observed tabs.json"
    assert torn == [], (
        f"BUG: torn tabs.json observed ({len(torn[0])} bytes): {torn[0][:80]!r}"
    )
    final = json.loads(path.read_text(encoding="utf-8"))
    assert final == {"tabs": []}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tabs.json"], (
        "temp residue left next to tabs.json"
    )


def test_failed_save_leaves_no_temp_residue_and_heals(tmp_path: Path) -> None:
    """A save that cannot land leaves nothing behind and heals on the next one."""
    path = tmp_path / "tabs.json"
    reg = TabRegistry(path)
    # A non-empty DIRECTORY at the target makes the final rename fail.
    path.mkdir()
    (path / "occupant").write_text("x", encoding="utf-8")
    assert reg.open_tab("t1", "one") is OpenTabOutcome.OPENED
    assert reg.open_tab("t2", "two") is OpenTabOutcome.OPENED  # second failure: quiet
    assert path.is_dir()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tabs.json"], (
        "failed publish left temp residue"
    )
    # The in-memory registry stays authoritative and the next
    # successful mutation heals the file.
    (path / "occupant").unlink()
    path.rmdir()
    assert reg.close_tab("t2")
    assert json.loads(path.read_text(encoding="utf-8"))["tabs"][0]["tabId"] == "t1"
    # ``flush`` after a healed save is a no-op.
    reg.flush()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tabs.json"]


def test_flush_retries_a_failed_save(tmp_path: Path) -> None:
    path = tmp_path / "tabs.json"
    reg = TabRegistry(path)
    path.mkdir()
    (path / "occupant").write_text("x", encoding="utf-8")
    assert reg.open_tab("t1", "one") is OpenTabOutcome.OPENED
    (path / "occupant").unlink()
    path.rmdir()
    reg.flush()
    assert json.loads(path.read_text(encoding="utf-8"))["tabs"][0]["tabId"] == "t1"
