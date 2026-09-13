"""End-to-end tests for the SQLite vector index (offline embedder + live model)."""

import logging
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from kiss.agents.memoryfield.index import (
    DEFAULT_EMBEDDING_MODEL,
    HASHED_EMBEDDING_MODEL_CODE,
    ModelEmbedder,
    VectorIndex,
    default_embedder,
    deserialize_float32,
    hashed_embedding,
    model_code_for_filename,
    normalize,
    serialize_float32,
)
from kiss.agents.memoryfield.pages import MAX_PAGE_BYTES, MemoryDir

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
    # The unrelated page is (near-)orthogonal: the random uuid/timestamps in its
    # frontmatter can collide with query buckets, so only its rank is fixed.
    assert [
        h.name for h in index.search("carbon fibre woks conduct heat", k=5, min_score=-1.0)
    ] == ["woks", "dvv"]
    assert [
        h.name for h in index.search("carbon fibre woks conduct heat", k=5, min_score=0.05)
    ] == ["woks"]
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
