# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A sync that started before a clear/rebuild writes nothing into the rebuilt index.

Audit 2026-11-01, C6 F1.  ``_create_schema`` restarts the ``revision``
counter at zero, so a row of a rebuilt index can carry the same revision
as the row the old index held for that page.  A sync that snapshotted the
old index and only then wrote (``evaluate.py --rebuild`` runs ``clear()``
+ ``sync()`` while the daemon's own syncs are in flight) matched the new
row with its stale CAS token and deleted the recreated page.  The slow
sync's embedder performs the recreate + rebuild during its embedding
window, which is deterministic: the snapshot is taken before the first
embedding and every write after it.
"""

from pathlib import Path

from kiss.core.memoryfield.index import VectorIndex, hashed_embedding
from kiss.core.memoryfield.pages import MemoryDir


def test_sync_started_before_a_rebuild_skips_its_stale_writes(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path / "memory")
    memory.write("a", "Original version")
    fast = VectorIndex(memory, embed=hashed_embedding)
    fast.sync()
    memory.delete("a")  # the slow sync will snapshot a.md as stale ...
    memory.write("b", "Trigger a slow embedding batch")  # ... and embed b
    rebuilt: list[bool] = []

    def embed_after_rebuilding_once(text: str) -> list[float]:
        if not rebuilt:
            rebuilt.append(True)
            memory.write("a", "Newly recreated version")
            fast.clear()
            fast.sync()  # the new index holds a.md at revision 1 again
        return hashed_embedding(text)

    slow = VectorIndex(memory, embed=embed_after_rebuilding_once)
    report = slow.sync()
    assert rebuilt
    assert (report.added, report.updated, report.removed) == (0, 0, 0)
    assert memory.page_names() == ["a", "b"]
    assert sorted(fast.frontmatters()) == ["a", "b"]
    assert fast.frontmatters()["a"]["title"] == "a"
    # The rebuilt index is complete and consistent: a further sync changes nothing.
    assert fast.sync().unchanged == 2
    assert [hit.name for hit in fast.search("Newly recreated version", k=1)] == ["a"]
