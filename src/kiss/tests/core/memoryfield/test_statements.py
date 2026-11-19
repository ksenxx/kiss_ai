# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for statement vectors: the page split, their storage and two-stage search.

The search ranks a page by the mean of two cosine similarities, its own and
its best statement's.

Everything runs through ``MemoryDir`` and ``VectorIndex`` on real page files
and a real index file, with the offline hashed embedder.

Not tested here: a search ignores the statement vectors of an index file that
was deleted and rebuilt after it read the page rows (the statement query
compares the index ``generation``).  The rebuild has to land between two reads
of one search, which a test cannot time without a test double; the query has
no branch of its own, and a mismatch takes the fallback that
``test_a_row_rewritten_without_statements_outdates_them`` covers.
"""

import math
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from kiss.core.memoryfield import index as index_module
from kiss.core.memoryfield.index import (
    HASHED_EMBEDDING_DIMS,
    SCHEMA_VERSION,
    SyncReport,
    VectorIndex,
    hashed_embedding,
    serialize_float16,
)
from kiss.core.memoryfield.pages import MemoryDir
from kiss.core.memoryfield.statements import (
    MAX_CODE_BYTES,
    MAX_PAGE_STATEMENTS,
    MAX_PREFIX_BYTES,
    MAX_STATEMENT_BYTES,
    page_statements,
    statement_text,
)

PAGE = """\
---
title: Daemon notes
---
Intro paragraph that runs
over two lines.

# Ports
- the daemon listens on port 8787
  unless KISS_WEB_PORT is set
1. numbered items are bullets too
* so are stars

| name | value |
|------|-------|
| port | 8787 |

## Restart
```bash
systemctl --user restart kiss-web

```
```
```
Text right after a fence.
```
an open fence runs to the end
"""


def stored(index: VectorIndex) -> dict[str, tuple[int, int, int]]:
    """``{filename: (page revision, statements revision, number of vectors)}`` per stored row."""
    with closing(sqlite3.connect(index.path)) as conn:
        rows = conn.execute(
            "SELECT p.filename, p.revision, s.revision, LENGTH(s.embeddings)"
            " FROM pages p JOIN statements s ON s.filename = p.filename"
        ).fetchall()
    return {row[0]: (row[1], row[2], row[3] // (2 * HASHED_EMBEDDING_DIMS)) for row in rows}


def statement_rows(index: VectorIndex) -> int:
    """Number of rows in the statements table, orphans included."""
    with closing(sqlite3.connect(index.path)) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM statements").fetchone()[0])


def ranking(index: VectorIndex) -> list[tuple[str, float]]:
    """Names and scores of the pages *index* returns for the saffron query."""
    return [(hit.name, hit.score) for hit in index.search("saffron threads", k=3)]


def cooking_memory(root: Path) -> MemoryDir:
    """Three pages; only one statement of the long ``kitchen`` page is about saffron."""
    memory = MemoryDir(root)
    filler = "\n".join(f"- shelf {i} holds flour sugar salt pepper rice beans" for i in range(40))
    memory.write("kitchen", f"# Shelves\n{filler}\n\n# Spices\n- saffron threads cost a fortune\n")
    memory.write("saffron-shop", "# Shop\n- the spice shop sells paprika and saffron powder\n")
    memory.write("garage", "# Tools\n- the garage holds a hammer and a saw\n")
    return memory


def test_a_page_is_cut_into_bullets_paragraphs_table_rows_and_code_blocks() -> None:
    assert page_statements("daemon", PAGE) == [
        "Daemon notes\nIntro paragraph that runs over two lines.",
        "Daemon notes - Ports\n- the daemon listens on port 8787 unless KISS_WEB_PORT is set",
        "Daemon notes - Ports\n1. numbered items are bullets too",
        "Daemon notes - Ports\n* so are stars",
        "Daemon notes - Ports\n| name | value |",
        "Daemon notes - Ports\n| port | 8787 |",
        "Daemon notes - Restart\nsystemctl --user restart kiss-web\n",
        "Daemon notes - Restart\nText right after a fence.",
        "Daemon notes - Restart\nan open fence runs to the end",
    ]
    # The statement as written in the page, without the prefix it is embedded with.
    assert statement_text(PAGE, 1) == (
        "- the daemon listens on port 8787 unless KISS_WEB_PORT is set"
    )
    assert statement_text(PAGE, 9) == "" and statement_text(PAGE, -1) == ""
    # A page without a title is titled by its name; headings alone are no statement.
    assert page_statements("bare-name", "one line") == ["bare-name\none line"]
    assert page_statements("empty", "# Only\n## Headings\n\n") == []
    # A fence closes on a line of its own character and at least its length; a line
    # that starts with a hashtag is text: a heading has a space after its marks.
    nested = (
        "````markdown\n```python\n# not a heading\n```\n````\n"
        "~~~\ntilde `fence`\n~~~\n#incident-123 reopens\n"
    )
    assert page_statements("p", nested) == [
        "p\n```python\n# not a heading\n```",
        "p\ntilde `fence`",
        "p\n#incident-123 reopens",
    ]


def test_statements_are_cut_to_their_byte_limits_and_counted_per_page() -> None:
    long_page = "é" * 3000 + "\n\n```\n" + "x" * 3000 + "\n```\n"
    paragraph, code = page_statements("pg", long_page)
    assert len(paragraph.encode()) == MAX_STATEMENT_BYTES - 1  # never half a character
    assert len(code.encode()) == MAX_CODE_BYTES
    many = "\n".join(f"- item {i}" for i in range(MAX_PAGE_STATEMENTS + 5))
    assert len(page_statements("p", many)) == MAX_PAGE_STATEMENTS
    assert statement_text(many, MAX_PAGE_STATEMENTS) == ""
    # A long title and heading are cut, never the statement they introduce.
    crowded = f"---\ntitle: {'T' * 3000}\n---\n# {'H' * 3000}\nThe fact.\n"
    assert page_statements("p", crowded) == ["T" * MAX_PREFIX_BYTES + "\nThe fact."]


def test_float16_blob_holds_unit_vectors_end_to_end() -> None:
    blob = serialize_float16([[3.0, 4.0], [0.0, 2.0]])
    values = np.frombuffer(blob, dtype="<f2").tolist()
    assert values == pytest.approx([0.6, 0.8, 0.0, 1.0], abs=1e-3)
    assert serialize_float16([]) == b""


def test_search_ranks_a_page_by_itself_and_its_best_statement(tmp_path: Path) -> None:
    memory = cooking_memory(tmp_path)
    index = VectorIndex(memory, embed=hashed_embedding, model_code="m")
    assert index.sync() == SyncReport(added=3, updated=0, removed=0, unchanged=0)
    assert {name: row[2] for name, row in stored(index).items()} == {
        "garage.md": 1,
        "kitchen.md": 41,
        "saffron-shop.md": 1,
    }
    hits = index.search("saffron threads", k=3)
    assert [hit.name for hit in hits] == ["kitchen", "saffron-shop"]
    # The hit names the statement that matched: the last of the page's 41.
    assert hits[0].statement == 40
    assert statement_text(memory.read("kitchen").raw, 40) == "- saffron threads cost a fortune"
    assert hits[0].score > hits[1].score > 0
    # One vector per page dilutes the saffron line among forty shelves.
    by_page = VectorIndex(memory, embed=hashed_embedding, model_code="m", statements=False)
    assert by_page.sync() == SyncReport(added=0, updated=0, removed=0, unchanged=3)
    page_hits = by_page.search("saffron threads", k=3)
    assert [hit.name for hit in page_hits] == ["saffron-shop", "kitchen"]
    assert all(hit.statement is None for hit in page_hits)
    # The score is the mean of the page's cosine and its best statement's.
    statement = page_statements("kitchen", memory.read("kitchen").raw)[40]
    cosine = math.sumprod(hashed_embedding(statement), hashed_embedding("saffron threads"))
    assert hits[0].score == pytest.approx((page_hits[1].score + cosine) / 2, abs=1e-3)
    # Asked to, an index with statements ranks by page embedding alone.
    assert ranking(by_page) == [
        (hit.name, hit.score) for hit in index.search("saffron threads", k=3, by_page=True)
    ]


def test_only_the_shortlist_of_the_page_scan_is_ranked_by_statement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = cooking_memory(tmp_path)
    index = VectorIndex(memory, embed=hashed_embedding, model_code="m")
    index.sync()
    monkeypatch.setattr(index_module, "SHORTLIST_PAGES", 1)
    # The page scan puts saffron-shop first; kitchen never reaches the second stage.
    assert [hit.name for hit in index.search("saffron threads", k=1)] == ["saffron-shop"]
    # A larger k widens the shortlist, read here in chunks of two pages.
    monkeypatch.setattr(index_module, "_STATEMENT_QUERY_PAGES", 2)
    assert [hit.name for hit in index.search("saffron threads", k=3)] == ["kitchen", "saffron-shop"]
    # The shortlist is exactly that long: a page whose score ties with its last page
    # (a copy, here without statement vectors) is not read and causes no fallback.
    memory.write("zz-copy", memory.read("saffron-shop").body, title="saffron shop")
    index.sync()
    with closing(sqlite3.connect(index.path)) as conn, conn:
        conn.execute("DELETE FROM statements WHERE filename = 'zz-copy.md'")
    (hit,) = index.search("saffron threads", k=1)
    assert (hit.name, hit.statement, hit.by_page) == ("saffron-shop", 0, False)
    assert all(hit.by_page for hit in index.search("saffron threads", k=2))


def test_search_ranks_by_page_until_the_whole_shortlist_has_statements(tmp_path: Path) -> None:
    memory = cooking_memory(tmp_path)
    by_page = VectorIndex(memory, embed=hashed_embedding, model_code="m", statements=False)
    assert by_page.sync().added == 3
    assert stored(by_page) == {}  # an index without statements stores none
    index = VectorIndex(memory, embed=hashed_embedding, model_code="m")
    # No time for the backfill: nothing is embedded and the ranking is the page ranking.
    assert index.sync(backfill_seconds=0) == SyncReport(0, 0, 0, 0)
    assert stored(index) == {} and ranking(index) == ranking(by_page)
    # A modified page is embedded whatever the backfill is allowed, statements included;
    # the two other pages still lack theirs, so the ranking stays the page ranking.
    memory.write("garage", "# Tools\n- the garage holds a hammer and a saffron grinder\n")
    assert index.sync(backfill_seconds=0) == SyncReport(0, 1, 0, 0)
    assert list(stored(index)) == ["garage.md"]
    assert len(ranking(by_page)) == 3 and ranking(index) == ranking(by_page)
    assert all(hit.statement is None for hit in index.search("saffron threads", k=3))
    # The backfill embeds the rest; each row and its statements share one revision.
    assert index.sync() == SyncReport(added=0, updated=2, removed=0, unchanged=1)
    assert all(page == statements for page, statements, _ in stored(index).values())
    assert [name for name, _ in ranking(index)] == ["kitchen", "saffron-shop", "garage"]
    assert [name for name, _ in ranking(by_page)] == ["saffron-shop", "garage", "kitchen"]
    assert index.sync() == SyncReport(added=0, updated=0, removed=0, unchanged=3)


def test_a_row_rewritten_without_statements_outdates_them(tmp_path: Path) -> None:
    """A process that predates statement vectors rewrites a page row and leaves the statements."""
    memory = cooking_memory(tmp_path)
    index = VectorIndex(memory, embed=hashed_embedding, model_code="m")
    index.sync()
    before = stored(index)["kitchen.md"]
    memory.write("kitchen", "# Spices\n- saffron threads cost a fortune\n- so does vanilla\n")
    old_process = VectorIndex(memory, embed=hashed_embedding, model_code="m", statements=False)
    assert old_process.sync().updated == 1
    page_revision, statements_revision, vectors = stored(index)["kitchen.md"]
    assert (statements_revision, vectors) == before[1:] and page_revision > statements_revision
    # The 41 stale vectors are not used: the shortlist falls back to the page ranking.
    assert all(hit.statement is None for hit in index.search("saffron threads", k=3))
    assert index.sync() == SyncReport(added=0, updated=1, removed=0, unchanged=2)
    assert stored(index)["kitchen.md"][2] == 2
    assert index.search("vanilla", k=1)[0].statement == 1


def test_statement_vectors_follow_deletes_clears_and_orphans(tmp_path: Path) -> None:
    memory = cooking_memory(tmp_path)
    index = VectorIndex(memory, embed=hashed_embedding, model_code="m")
    index.sync()
    # A process that predates statement vectors drops the row of a deleted page
    # and leaves its statements, which the next removal by this code sweeps.
    memory.delete("garage")
    with closing(sqlite3.connect(index.path)) as conn, conn:
        conn.execute("DELETE FROM pages WHERE filename = 'garage.md'")
    assert index.sync() == SyncReport(added=0, updated=0, removed=0, unchanged=2)
    assert statement_rows(index) == 3
    memory.delete("saffron-shop")
    assert index.sync() == SyncReport(added=0, updated=0, removed=1, unchanged=1)
    assert sorted(stored(index)) == ["kitchen.md"] and statement_rows(index) == 1
    index.clear()
    assert statement_rows(index) == 0


def test_pages_without_usable_statement_vectors_keep_their_page_score(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    memory.write("headings", "# Saffron\n## Saffron threads\n", title="Saffron threads")
    memory.write("damaged", "- saffron threads are red\n")
    memory.write("whole", "- saffron threads are dried stigmas\n")
    index = VectorIndex(memory, embed=hashed_embedding, model_code="m")
    index.sync()
    assert stored(index)["headings.md"][2] == 0
    with closing(sqlite3.connect(index.path)) as conn, conn:
        conn.execute("UPDATE statements SET embeddings = x'0102' WHERE filename = 'damaged.md'")
    hits = {hit.name: hit for hit in index.search("saffron threads", k=3)}
    assert hits["whole"].statement == 0
    assert hits["headings"].statement is None and hits["damaged"].statement is None
    # The search did rank with statements: these two pages just have none to offer.
    assert not any(hit.by_page for hit in hits.values())
    by_page = VectorIndex(memory, embed=hashed_embedding, model_code="m", statements=False)
    page_hits = by_page.search("saffron threads", k=3)
    assert all(hit.by_page for hit in page_hits)
    page_scores = {hit.name: hit.score for hit in page_hits}
    # Their own cosine stands in for a best statement's, to the last bit.
    assert hits["headings"].score == page_scores["headings"]
    assert hits["damaged"].score == page_scores["damaged"]
    assert hits["whole"].score > page_scores["whole"]


def test_an_index_from_before_statement_vectors_is_migrated(tmp_path: Path) -> None:
    memory = cooking_memory(tmp_path)
    VectorIndex(memory, embed=hashed_embedding, model_code="m").sync()
    path = tmp_path / "m.sqlite3"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("DROP TABLE statements")
        conn.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")
    index = VectorIndex(memory, embed=hashed_embedding, model_code="m")
    assert [hit.name for hit in index.search("saffron threads", k=1)] == ["saffron-shop"]
    assert index.sync() == SyncReport(added=0, updated=3, removed=0, unchanged=0)
    assert [hit.name for hit in index.search("saffron threads", k=1)] == ["kitchen"]
    with closing(sqlite3.connect(path)) as conn:
        version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    assert version[0] == SCHEMA_VERSION == "4"


def test_sync_of_named_pages_leaves_the_others_alone(tmp_path: Path) -> None:
    memory = cooking_memory(tmp_path)
    index = VectorIndex(memory, embed=hashed_embedding, model_code="m")
    assert index.sync(names={"garage"}) == SyncReport(added=1, updated=0, removed=0, unchanged=0)
    assert index.count() == 1
    assert index.sync() == SyncReport(added=2, updated=0, removed=0, unchanged=1)
    memory.delete("garage")
    memory.delete("kitchen")
    assert index.sync(names={"garage", "no-such-page"}) == SyncReport(0, 0, 1, 0)
    assert sorted(stored(index)) == ["kitchen.md", "saffron-shop.md"]


def test_similar_pages_lists_near_duplicates_of_one_page(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    body = "The daemon listens on port 8787 by default."
    for name in ("ports", "ports-copy", "ports-again"):
        memory.write(name, body, title="Ports")
    memory.write("ports-near", body + " Usually it does.", title="Ports")
    memory.write("garage", "The garage holds a hammer and a saw.")
    index = VectorIndex(memory, embed=hashed_embedding, model_code="m")
    index.sync()
    similar = index.similar_pages("ports", threshold=0.6, k=10)
    # Best first, equal scores in name order, the page itself left out.
    assert [name for name, _ in similar] == ["ports-again", "ports-copy", "ports-near"]
    assert similar[0][1] == similar[1][1] == pytest.approx(1.0)
    assert 0.6 <= similar[2][1] < 0.999
    assert index.similar_pages("ports", threshold=0.6, k=1) == similar[:1]
    assert index.similar_pages("ports", threshold=0.999) == similar[:2]
    assert index.similar_pages("garage") == [] and index.similar_pages("no-such-page") == []


class ShortBatches:
    """An embedding service that answers a batch with one vector too few."""

    def __call__(self, text: str) -> list[float]:
        return hashed_embedding(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed all but the last of *texts*."""
        return [hashed_embedding(text) for text in texts[:-1]]


def test_a_batch_answered_with_too_few_vectors_stores_nothing(tmp_path: Path) -> None:
    index = VectorIndex(cooking_memory(tmp_path), embed=ShortBatches(), model_code="m")
    with pytest.raises(ValueError, match="returned 45 vectors for 46 inputs"):
        index.sync()
    assert index.count() == 0 and stored(index) == {}


class StopAfterOneBatch:
    """An embedder that sets *cancel* while it answers its first batch."""

    def __init__(self, cancel: threading.Event) -> None:
        self.cancel = cancel
        self.batches: list[int] = []

    def __call__(self, text: str) -> list[float]:
        return hashed_embedding(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* and ask the sync to stop."""
        self.cancel.set()
        self.batches.append(len(texts))
        return [hashed_embedding(text) for text in texts]


def test_a_page_and_its_statements_travel_in_one_request(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    for i in range(4):
        memory.write(f"page-{i}", "\n".join(f"- page {i} fact {j}" for j in range(99)))
    cancel = threading.Event()
    embedder = StopAfterOneBatch(cancel)
    index = VectorIndex(memory, embed=embedder, model_code="m")
    # 100 inputs a page: two pages fill a request of at most 256 inputs, and the
    # stop that arrives during it keeps both pages whole.
    assert index.sync(cancel=cancel) == SyncReport(added=2, updated=0, removed=0, unchanged=0)
    assert embedder.batches == [200]
    assert {name: row[2] for name, row in stored(index).items()} == {
        "page-0.md": 99,
        "page-1.md": 99,
    }
    cancel.clear()
    assert index.sync().added == 2 and embedder.batches == [200, 200]
