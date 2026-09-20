"""End-to-end tests for the SQLite vector index (offline embedder + live model)."""

import hashlib
import logging
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from kiss.core.memoryfield.index import (
    DEFAULT_EMBEDDING_MODEL,
    HASHED_EMBEDDING_MODEL_CODE,
    ModelEmbedder,
    SyncReport,
    VectorIndex,
    default_embedder,
    deserialize_float32,
    hashed_embedding,
    model_code_for_filename,
    normalize,
    serialize_float32,
)
from kiss.core.memoryfield.pages import MAX_PAGE_BYTES, MemoryDir

live_api = pytest.mark.live_api
requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; embedding live test needs it",
)


def small_embedding(text: str) -> list[float]:
    """A 64-dimensional offline embedding, used to provoke dimension mismatches."""
    return hashed_embedding(text, dims=64)


# --- vector helpers ------------------------------------------------------


def test_hashed_embedding_properties() -> None:
    vector = hashed_embedding("Carbon fibre woks conduct heat unevenly")
    assert len(vector) == 1024
    assert math.isclose(math.sumprod(vector, vector), 1.0, rel_tol=1e-6)
    assert hashed_embedding("same text") == hashed_embedding("same text")
    assert hashed_embedding("") == [0.0] * 1024
    # Lexically similar texts are closer than unrelated ones.
    a, b, c = (
        hashed_embedding("carbon fibre wok heat"),
        hashed_embedding("heat of a carbon fibre wok"),
        hashed_embedding("finnish bureaucracy id number"),
    )
    assert math.sumprod(a, b) > math.sumprod(a, c)


def test_normalize_and_float32_roundtrip() -> None:
    assert normalize([0.0, 0.0]) == [0.0, 0.0]
    unit = normalize([3.0, 4.0])
    assert unit == pytest.approx([0.6, 0.8])
    blob = serialize_float32(unit)
    assert len(blob) == 8
    assert deserialize_float32(blob) == pytest.approx(unit, abs=1e-7)


def test_model_code_for_filename() -> None:
    assert model_code_for_filename("text-embedding-3-small") == "text-embedding-3-small"
    assert model_code_for_filename("BAAI/bge-base-en-v1.5") == "BAAI-bge-base-en-v1.5"
    assert model_code_for_filename("../x") == "x"


# --- VectorIndex ---------------------------------------------------------


def test_index_sync_is_incremental_and_search_ranks(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    memory.write(
        "woks",
        "Carbon fibre woks conduct heat unevenly but heat up quickly.",
        title="Carbon Fibre Woks",
    )
    memory.write(
        "dvv", "Getting a Finnish personal identity code requires visiting DVV.", title="Finnish ID"
    )
    index = VectorIndex(memory, embed=hashed_embedding)
    assert index.path == tmp_path / f"{HASHED_EMBEDDING_MODEL_CODE}.sqlite3"

    report = index.sync()
    assert (report.added, report.updated, report.removed, report.unchanged) == (2, 0, 0, 0)
    assert index.count() == 2
    with closing(sqlite3.connect(index.path)) as conn:
        assert conn.execute("SELECT value FROM meta WHERE key='model_code'").fetchone() == (
            HASHED_EMBEDDING_MODEL_CODE,
        )
        filenames = {row[0] for row in conn.execute("SELECT filename FROM pages")}
    assert filenames == {"woks.md", "dvv.md"}

    # Nothing changed: no re-embedding.
    assert index.sync().unchanged == 2

    hits = index.search("carbon fibre woks conduct heat", k=5)
    assert hits[0].name == "woks" and hits[0].title == "Carbon Fibre Woks" and hits[0].summary == ""
    assert hits[0].score > 0
    # Only title, summary and body are embedded (volatile frontmatter such as
    # the uuid and timestamps is excluded), so scores are deterministic: the
    # true match scores ~0.8 while the unrelated page's residual
    # hash-collision score is ~0.07, so min_score=0.1 separates them.
    assert [
        h.name for h in index.search("carbon fibre woks conduct heat", k=5, min_score=-1.0)
    ] == ["woks", "dvv"]
    assert [h.name for h in index.search("carbon fibre woks conduct heat", k=5, min_score=0.1)] == [
        "woks"
    ]
    assert index.search("woks", k=0) == []
    assert index.search("woks", k=1)[0].name == "woks"

    # Modify one page, delete the other, add a third.
    memory.write("woks", "Now about induction hobs and copper pans.")
    memory.delete("dvv")
    memory.write("git", "Rebase before merging a worktree branch.")
    report = index.sync()
    assert (report.added, report.updated, report.removed, report.unchanged) == (1, 1, 1, 0)
    assert index.count() == 2
    assert index.search("copper pans induction")[0].name == "woks"
    assert index.search("Finnish DVV", min_score=0.05) == []

    index.clear()
    assert not index.path.exists()
    index.clear()  # idempotent
    assert index.count() == 0
    assert index.sync().added == 2


def test_embedding_ignores_volatile_frontmatter(tmp_path: Path) -> None:
    """Rewriting identical content must reproduce the exact same vector.

    ``MemoryDir.write`` generates a fresh ``uuid`` and timestamps for every
    new page, so this only holds because ``embedding_input`` embeds nothing
    but the title, the summary and the body.
    """
    memory = MemoryDir(tmp_path)
    index = VectorIndex(memory, embed=hashed_embedding)
    memory.write("woks", "Carbon fibre woks conduct heat.", title="Woks", summary="wok care")
    index.sync()
    with closing(sqlite3.connect(index.path)) as conn:
        first = conn.execute("SELECT embedding FROM pages WHERE filename='woks.md'").fetchone()[0]

    memory.delete("woks")
    assert index.sync().removed == 1
    memory.write("woks", "Carbon fibre woks conduct heat.", title="Woks", summary="wok care")
    assert index.sync().added == 1
    with closing(sqlite3.connect(index.path)) as conn:
        second = conn.execute("SELECT embedding FROM pages WHERE filename='woks.md'").fetchone()[0]
    assert bytes(first) == bytes(second)

    raw = memory.read("woks").raw
    assert index.embedding_input(raw) == "Woks\nwok care\nCarbon fibre woks conduct heat.\n"


def test_old_input_format_rows_are_reembedded(tmp_path: Path) -> None:
    """Rows embedded under an older input format are re-embedded exactly once.

    Each row records the input-format version it was embedded under, so a
    row marked with the pre-frontmatter-exclusion format ``'1'`` (what a
    still-running old process writes, even AFTER a current-format sync) is
    picked up and healed by the next sync.  The ``sha256`` column stays the
    plain content hash, which is what keeps that old process from seeing
    current rows as changed and rebuilding the index right back (the
    mixed-version ping-pong found in review).
    """
    memory = MemoryDir(tmp_path)
    memory.write("woks", "Carbon fibre woks conduct heat.")
    memory.write("dvv", "Getting a Finnish personal identity code requires visiting DVV.")
    index = VectorIndex(memory, embed=hashed_embedding)
    assert index.sync().added == 2
    raw_bytes = memory.page_path("dvv").read_bytes()
    with closing(sqlite3.connect(index.path)) as conn:
        good, sha = conn.execute(
            "SELECT embedding, sha256 FROM pages WHERE filename='dvv.md'"
        ).fetchone()
    # Old-reader compatibility: current code stores the PLAIN content hash,
    # so pre-format code (which compares plain sha256) sees rows unchanged.
    assert bytes(sha) == hashlib.sha256(raw_bytes).digest()

    # Overwrite one row the way pre-format code did: same plain sha256, a
    # vector of the whole raw file (frontmatter included), and — via the
    # column default — input_format '1'.
    legacy_vector = normalize(hashed_embedding(raw_bytes.decode("utf-8")))
    with closing(sqlite3.connect(index.path)) as conn, conn:
        conn.execute(
            "UPDATE pages SET embedding = ?, input_format = '1' WHERE filename = 'dvv.md'",
            (serialize_float32(legacy_vector),),
        )

    report = index.sync()
    assert (report.added, report.updated, report.removed, report.unchanged) == (0, 1, 0, 1)
    with closing(sqlite3.connect(index.path)) as conn:
        healed, fmt = conn.execute(
            "SELECT embedding, input_format FROM pages WHERE filename='dvv.md'"
        ).fetchone()
    assert bytes(healed) == bytes(good) and fmt == "2"
    assert index.sync().unchanged == 2  # healing converges: no rebuild loop
    assert index.search("Finnish DVV", k=1)[0].name == "dvv"


def test_pre_column_index_schema_is_migrated_and_reembedded(tmp_path: Path) -> None:
    """An index file created before the input_format/revision columns is upgraded.

    Connecting adds the missing columns.  ``input_format`` defaults to
    ``'1'``, which marks every existing row as embedded under the old
    whole-raw-file mapping, so the next sync re-embeds all pages;
    ``revision`` defaults to ``0`` and is replaced by the next values of
    the index-wide counter (which the baseline sync left at 2) on that
    first rewrite.
    """
    memory = MemoryDir(tmp_path)
    memory.write("woks", "Carbon fibre woks conduct heat.")
    memory.write("dvv", "Getting a Finnish personal identity code requires visiting DVV.")
    index = VectorIndex(memory, embed=hashed_embedding)
    assert index.sync().added == 2
    with closing(sqlite3.connect(index.path)) as conn, conn:
        conn.execute("ALTER TABLE pages DROP COLUMN input_format")  # simulate the old schema
        conn.execute("ALTER TABLE pages DROP COLUMN revision")

    report = index.sync()
    assert (report.added, report.updated, report.removed, report.unchanged) == (0, 2, 0, 0)
    with closing(sqlite3.connect(index.path)) as conn:
        revisions = [row[0] for row in conn.execute("SELECT revision FROM pages ORDER BY revision")]
        counter = conn.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()[0]
    assert revisions == [3, 4] and counter == "4"
    assert index.sync().unchanged == 2
    with closing(sqlite3.connect(index.path)) as conn:  # a no-op sync does not burn revisions
        assert conn.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()[0] == "4"
    assert index.search("Finnish DVV", k=1)[0].name == "dvv"


def test_index_embedding_input_truncates_to_page_limit(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    index = VectorIndex(memory, embed=hashed_embedding)
    short = "x" * 100
    assert index.embedding_input(short) == short
    long_text = "é" * MAX_PAGE_BYTES  # 2 bytes per character
    truncated = index.embedding_input(long_text)
    assert len(truncated.encode("utf-8")) <= MAX_PAGE_BYTES
    assert truncated == "é" * (MAX_PAGE_BYTES // 2)


def test_index_skips_rows_with_foreign_dimension(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    memory = MemoryDir(tmp_path)
    memory.write("one", "Alpha beta gamma.")
    VectorIndex(memory, embed=hashed_embedding, model_code="shared").sync()
    # A differently sized embedder pointed at the same index file must not crash.
    other = VectorIndex(memory, embed=small_embedding, model_code="shared")
    with caplog.at_level(logging.WARNING):
        assert other.search("alpha") == []
    assert "stored dimension 1024 != query dimension 64" in caplog.text


def test_index_default_model_code_comes_from_embedder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory = MemoryDir(tmp_path)
    index = VectorIndex(memory, embed=ModelEmbedder("text-embedding-3-small"))
    assert index.model_code == "text-embedding-3-small"
    assert index.path.name == "text-embedding-3-small.sqlite3"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    default = VectorIndex(memory)
    assert default.model_code == "text-embedding-3-small"
    monkeypatch.delenv("OPENAI_API_KEY")
    offline = VectorIndex(memory)
    assert offline.model_code == HASHED_EMBEDDING_MODEL_CODE
    assert offline.path.name == "hashed-bow-v1.sqlite3"


def test_default_embedder_uses_model_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    embed = default_embedder()
    assert isinstance(embed, ModelEmbedder)
    assert embed.model_name == DEFAULT_EMBEDDING_MODEL


def test_default_embedder_falls_back_offline_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert default_embedder() is hashed_embedding
    monkeypatch.setenv("OPENAI_API_KEY", "   ")  # blank counts as absent
    assert default_embedder() is hashed_embedding


def test_index_rejects_foreign_model_code_and_symlinked_path(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    memory.write("one", "Alpha beta gamma.")
    VectorIndex(memory, embed=hashed_embedding, model_code="provider/a:b").sync()
    # A different model code that sanitises to the same filename must not reuse the index.
    other = VectorIndex(memory, embed=hashed_embedding, model_code="provider/a/b")
    assert other.path == tmp_path / "provider-a-b.sqlite3"
    with pytest.raises(ValueError, match="was built with embedding model 'provider/a:b'"):
        other.sync()
    with pytest.raises(ValueError, match="was built with embedding model"):
        other.search("alpha")

    external = tmp_path / "elsewhere.sqlite3"
    linked = VectorIndex(memory, embed=hashed_embedding, model_code="linked")
    linked.path.symlink_to(external)
    with pytest.raises(ValueError, match="is a symlink"):
        linked.sync()
    assert not external.exists()


def test_index_connections_are_closed_and_hold_no_transaction(tmp_path: Path) -> None:
    import gc
    import warnings

    memory = MemoryDir(tmp_path)
    memory.write("one", "Alpha beta gamma.")
    index = VectorIndex(memory, embed=hashed_embedding)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        for _ in range(50):
            index.sync()
            index.search("alpha")
            index.count()
        gc.collect()
    # A concurrent writer is not blocked by a reader-style search having run.
    with closing(sqlite3.connect(index.path, timeout=0.1)) as conn, conn:
        conn.execute("INSERT INTO meta (key, value) VALUES ('probe', '1')")
    index.clear()


def test_index_skips_page_modified_during_embedding(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    memory.write("volatile", "first version")
    path = memory.page_path("volatile")

    def embed_and_modify(text: str) -> list[float]:
        # Simulates another process rewriting the page while its embedding is computed.
        if "first version" in text:
            path.write_text(path.read_text().replace("first version", "second version"))
        return hashed_embedding(text)

    index = VectorIndex(memory, embed=embed_and_modify, model_code="volatile-test")
    report = index.sync()
    assert (report.added, report.updated, report.unchanged) == (0, 0, 0)
    assert index.count() == 0
    report = index.sync()
    assert report.added == 1
    assert index.search("second version")[0].name == "volatile"


def test_index_handles_undecodable_bytes(tmp_path: Path) -> None:
    (tmp_path / "bin.md").write_bytes(b"---\ntitle: B\n---\n\xff\xfe raw bytes")
    index = VectorIndex(MemoryDir(tmp_path), embed=hashed_embedding)
    assert index.sync().added == 1
    assert index.search("raw bytes")[0].title == "B"


@live_api
@requires_openai
def test_model_embedder_live_search(tmp_path: Path) -> None:
    """Real embeddings rank a paraphrase above a lexically overlapping distractor."""
    memory = MemoryDir(tmp_path)
    memory.write(
        "cookware",
        "Carbon fibre woks heat up very fast but distribute heat unevenly; "
        "season them like cast iron.",
    )
    memory.write(
        "carbon-tax",
        "The carbon tax on fibre-optic cable imports was raised; heat pumps are exempt.",
    )
    embedder = ModelEmbedder()
    index = VectorIndex(memory, embed=embedder)
    assert index.path.name == "text-embedding-3-small.sqlite3"
    assert index.sync().added == 2
    hits = index.search("how should I look after a new wok made of carbon fiber?", k=2)
    assert hits[0].name == "cookware"
    assert len(embedder("hello")) == 1536


# --- concurrent syncs (optimistic compare-and-swap) ------------------------


def _row_state(index: VectorIndex, filename: str) -> tuple[bytes, bytes] | None:
    with closing(sqlite3.connect(index.path)) as conn:
        row = conn.execute(
            "SELECT sha256, embedding FROM pages WHERE filename = ?", (filename,)
        ).fetchone()
    return None if row is None else (bytes(row[0]), bytes(row[1]))


def test_concurrent_sync_does_not_overwrite_fresher_rows(tmp_path: Path) -> None:
    """A slow sync must not clobber rows a faster concurrent sync refreshed.

    Sync A snapshots the index, embeds page ``a`` (v1) and passes the
    post-embed re-read check, then keeps embedding page ``c``.  Meanwhile
    page ``a`` is rewritten to v2 and a second sync B indexes ``a`` (v2)
    and inserts the new page ``c``.  When A finally writes, its stale
    ``a`` (v1) row and duplicate ``c`` row must be dropped in favour of B's.
    """
    memory = MemoryDir(tmp_path)
    memory.write("a", "alpha v0")
    memory.write("b", "beta v0")
    VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync()
    memory.write("a", "alpha v1")
    memory.write("c", "gamma new")
    path_a = memory.page_path("a")

    def embed_and_race(text: str) -> list[float]:
        if "gamma new" in text:  # page a's v1 row is already built and verified
            path_a.write_text(path_a.read_text().replace("alpha v1", "alpha v2"))
            assert VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync() == (
                SyncReport(added=1, updated=1, removed=0, unchanged=1)
            )
        return hashed_embedding(text)

    slow = VectorIndex(memory, embed=embed_and_race, model_code="cas")
    v0_row = _row_state(slow, "a.md")
    report = slow.sync()
    assert (report.added, report.updated, report.removed, report.unchanged) == (0, 0, 0, 1)
    assert _row_state(slow, "a.md") != v0_row  # B's v2 row is in place ...
    assert _row_state(slow, "a.md")[0] == hashlib.sha256(path_a.read_bytes()).digest()  # type: ignore[index]
    assert slow.count() == 3
    # ... and the index has converged: nothing left to heal.
    assert VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync().unchanged == 3
    assert slow.search("alpha v2", min_score=0.1)[0].name == "a"


def test_concurrent_sync_does_not_delete_recreated_page_row(tmp_path: Path) -> None:
    """A stale-row delete computed before a page was recreated must not fire.

    Sync A lists the pages after ``a`` was deleted, so it plans to drop
    ``a``'s row.  While A embeds ``b``, page ``a`` is recreated and a
    concurrent sync B refreshes its row.  A's delete must leave B's row.
    """
    memory = MemoryDir(tmp_path)
    memory.write("a", "alpha v0")
    memory.write("b", "beta v0")
    VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync()
    memory.delete("a")
    memory.write("b", "beta v1")

    def embed_and_recreate(text: str) -> list[float]:
        if "beta v1" in text:
            memory.write("a", "alpha reborn")
            VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync()
        return hashed_embedding(text)

    slow = VectorIndex(memory, embed=embed_and_recreate, model_code="cas")
    report = slow.sync()
    assert (report.added, report.updated, report.removed, report.unchanged) == (0, 0, 0, 0)
    assert slow.count() == 2
    assert slow.search("alpha reborn", min_score=0.1)[0].name == "a"
    assert VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync().unchanged == 2


def test_concurrent_sync_cas_survives_byte_identical_rewrite(tmp_path: Path) -> None:
    """The CAS token must not repeat when a page cycles back to identical bytes.

    Sync A snapshots ``a`` (v0), embeds v1 and pauses on page ``c``.  Sync B
    then indexes v2, the exact v0 bytes are restored, and sync C indexes v0
    again.  A's snapshot ``(sha256, input_format)`` now matches C's fresh
    row byte for byte, so a content-keyed CAS would let A overwrite it with
    stale v1; the per-write ``revision`` token must reject the write.
    """
    memory = MemoryDir(tmp_path)
    memory.write("a", "alpha v0")
    memory.write("b", "beta v0")
    VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync()
    path_a = memory.page_path("a")
    v0_bytes = path_a.read_bytes()
    path_a.write_bytes(v0_bytes.replace(b"alpha v0", b"alpha v1"))
    memory.write("c", "gamma new")

    def embed_and_cycle(text: str) -> list[float]:
        if "gamma new" in text:  # A's stale v1 row for `a` is already built
            path_a.write_bytes(v0_bytes.replace(b"alpha v0", b"alpha v2"))
            assert VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync().updated == 1
            path_a.write_bytes(v0_bytes)
            assert VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync().updated == 1
        return hashed_embedding(text)

    slow = VectorIndex(memory, embed=embed_and_cycle, model_code="cas")
    report = slow.sync()
    assert (report.added, report.updated, report.removed, report.unchanged) == (0, 0, 0, 1)
    assert _row_state(slow, "a.md")[0] == hashlib.sha256(v0_bytes).digest()  # type: ignore[index]
    assert VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync().unchanged == 3


def test_concurrent_sync_cas_survives_byte_identical_recreation(tmp_path: Path) -> None:
    """A planned stale delete must not fire on a byte-identical re-inserted row.

    Sync A snapshots ``a`` and finds it gone from disk, so it plans to drop
    the row.  While A embeds ``b``, sync B drops the row, the exact original
    bytes of ``a`` are restored, and sync C inserts a fresh row for it whose
    ``(sha256, input_format)`` equals A's snapshot.  A's conditional delete
    must miss because C's row carries a new ``revision``.
    """
    memory = MemoryDir(tmp_path)
    memory.write("a", "alpha v0")
    memory.write("b", "beta v0")
    VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync()
    path_a = memory.page_path("a")
    v0_bytes = path_a.read_bytes()
    path_a.unlink()
    memory.write("b", "beta v1")

    def embed_and_recreate(text: str) -> list[float]:
        if "beta v1" in text:
            assert VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync().removed == 1
            path_a.write_bytes(v0_bytes)
            assert VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync().added == 1
        return hashed_embedding(text)

    slow = VectorIndex(memory, embed=embed_and_recreate, model_code="cas")
    report = slow.sync()
    assert (report.added, report.updated, report.removed, report.unchanged) == (0, 0, 0, 0)
    assert slow.count() == 2
    assert _row_state(slow, "a.md")[0] == hashlib.sha256(v0_bytes).digest()  # type: ignore[index]
    assert VectorIndex(memory, embed=hashed_embedding, model_code="cas").sync().unchanged == 2
