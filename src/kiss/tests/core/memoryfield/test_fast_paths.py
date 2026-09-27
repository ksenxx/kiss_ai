"""End-to-end tests for the memoryfield's latency and scalability paths.

Covers the stat-validated sync (git's racy-clean rule), batched embedding
with per-batch commits, the decoded-row cache and its invalidation, the
read-only connection fast path, the query-embedding cache, the overlap of
query embedding with sync, and blocked pairwise near-duplicate detection.

The Windows branches of :func:`kiss.core.memoryfield.index.stat_key`
(``os.name == "nt"``) cannot run on POSIX without patching ``os.name``;
they are exercised by running this suite on Windows.
"""

import math
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import pytest

from kiss.core.memoryfield import index as index_module
from kiss.core.memoryfield.index import (
    EMBED_BATCH_INPUTS,
    QUERY_CACHE_SIZE,
    RACY_WINDOW_NS,
    ModelEmbedder,
    VectorIndex,
    hashed_embedding,
    stat_key,
)
from kiss.core.memoryfield.pages import MemoryDir
from kiss.core.memoryfield.tools import MemoryTools

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


class CountingEmbedder:
    """The offline hashed embedding, counting single and batched calls."""

    model_name = "counting-hashed"

    def __init__(self, fail_on_batch: int = 0, fail_on_text: str = "") -> None:
        self.calls: list[str] = []
        self.batches: list[int] = []
        self.fail_on_batch = fail_on_batch
        self.fail_on_text = fail_on_text

    def __call__(self, text: str) -> list[float]:
        if self.fail_on_text and self.fail_on_text in text:
            raise RuntimeError("embedding service unavailable")
        self.calls.append(text)
        return hashed_embedding(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        if len(self.batches) == self.fail_on_batch:
            raise RuntimeError("embedding service unavailable")
        return [hashed_embedding(text) for text in texts]


def stat_keys(index: VectorIndex) -> dict[str, str]:
    with closing(sqlite3.connect(index.path)) as conn:
        return dict(conn.execute("SELECT filename, stat_key FROM pages").fetchall())


def wait_past_racy_window() -> None:
    time.sleep(RACY_WINDOW_NS / 1e9 + 0.3)


def test_stat_key_is_empty_inside_racy_window(tmp_path: Path) -> None:
    page = tmp_path / "a.md"
    page.write_text("alpha", encoding="utf-8")
    st = page.stat()
    digest = bytes(range(32))
    assert stat_key(st, time.time_ns(), digest) == ""
    later = max(st.st_mtime_ns, st.st_ctime_ns) + RACY_WINDOW_NS + 1
    key = stat_key(st, later, digest)
    assert key.startswith(f"{st.st_size}:{st.st_mtime_ns}:")
    assert key.endswith(f":{st.st_ino}#{digest.hex()}")


def test_sync_trusts_stat_keys_and_verify_rehashes(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    memory.write("alpha", "Alpha page about carbon woks.")
    memory.write("beta", "Beta page about Finnish DVV.")
    index = VectorIndex(memory, embed=hashed_embedding)
    assert index.sync().added == 2
    # Just written: racy, so no stat key yet and every sync hashes them.
    assert set(stat_keys(index).values()) == {""}
    assert index.sync().unchanged == 2
    wait_past_racy_window()
    # The next sync hashes once more and records the keys without re-embedding.
    report = index.sync()
    assert (report.added, report.updated, report.unchanged) == (0, 0, 2)
    assert all(stat_keys(index).values())
    # A corrupted digest no longer matches the digest bound into the key.
    with closing(sqlite3.connect(index.path)) as conn, conn:
        conn.execute("UPDATE pages SET sha256 = x'00' WHERE filename = 'alpha.md'")
    assert index.sync().updated == 1
    # A row whose key vouches for the current stat data but whose content
    # differs (an edit that kept size and all timestamps): a stat-key match
    # skips hashing, so only verify=True notices.
    st = memory.page_path("alpha").stat()
    forged = stat_key(st, time.time_ns() + 2 * RACY_WINDOW_NS, b"\x00")
    with closing(sqlite3.connect(index.path)) as conn, conn:
        conn.execute(
            "UPDATE pages SET sha256 = x'00', stat_key = ? WHERE filename = 'alpha.md'",
            (forged,),
        )
    assert index.sync().unchanged == 2
    report = index.sync(verify=True)
    assert (report.updated, report.unchanged) == (1, 1)
    assert index.sync(verify=True).unchanged == 2
    # Changing only the permissions changes the ctime: rehashed, not re-embedded.
    memory.page_path("beta").chmod(0o600)
    assert index.sync().unchanged == 2
    # A still-running older process rewrites a row's content with SQL that
    # does not know the stat_key column: the key must stop vouching for it.
    stale_vector = index_module.serialize_float32(hashed_embedding("unrelated text"))
    with closing(sqlite3.connect(index.path)) as conn, conn:
        conn.execute(
            "UPDATE pages SET sha256 = x'01', embedding = ?, revision = revision + 100"
            " WHERE filename = 'alpha.md'",
            (stale_vector,),
        )
    wait_past_racy_window()
    report = index.sync()
    assert (report.updated, report.unchanged) == (1, 1)
    assert index.search("carbon woks")[0].name == "alpha"


def test_nonpositive_k_never_embeds_the_query(tmp_path: Path) -> None:
    tools = MemoryTools(tmp_path, embed=CountingEmbedder(fail_on_text="never"), model_code="c")
    assert tools.memory_search("never needed", k=0) == "No memory pages yet."
    tools.memory_write("woks", "Carbon fibre woks conduct heat.")
    assert tools.memory_pull("never needed", k=0) == "No matches."


def test_same_size_rewrite_with_restored_mtime_is_detected(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    path = tmp_path / "wok.md"
    path.write_text("copper pans heat evenly\n", encoding="utf-8")
    index = VectorIndex(memory, embed=hashed_embedding)
    index.sync()
    before = path.stat()
    # Same size, mtime put back: only the racy rule / ctime can reveal it.
    path.write_text("carbon woks heat evenly\n", encoding="utf-8")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert index.sync().updated == 1
    assert index.search("carbon woks")[0].name == "wok"


def test_sync_batches_embeddings_and_commits_each_batch(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    total = EMBED_BATCH_INPUTS + 10
    for i in range(total):
        memory.write(f"page-{i:04d}", f"Note number {i} about topic {i % 7}.")
    failing = CountingEmbedder(fail_on_batch=2)
    index = VectorIndex(memory, embed=failing, model_code="counting")
    with pytest.raises(RuntimeError, match="unavailable"):
        index.sync()
    # The first batch was committed before the second request failed.
    assert failing.batches == [EMBED_BATCH_INPUTS, 10]
    assert index.count() == EMBED_BATCH_INPUTS
    embedder = CountingEmbedder()
    index = VectorIndex(memory, embed=embedder, model_code="counting")
    report = index.sync()
    assert (report.added, report.unchanged) == (10, EMBED_BATCH_INPUTS)
    assert embedder.batches == [10] and embedder.calls == []


def test_batches_split_on_byte_limit(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    for i in range(40):
        memory.write(f"big-{i:02d}", f"page {i} " + "x" * 7900)
    embedder = CountingEmbedder()
    VectorIndex(memory, embed=embedder, model_code="counting").sync()
    assert sum(embedder.batches) == 40
    assert len(embedder.batches) == 2
    assert all(n * 7900 <= index_module.EMBED_BATCH_BYTES for n in embedder.batches)


def test_embedder_without_embed_many_is_called_per_page(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    memory.write("one", "first")
    memory.write("two", "second")
    calls: list[str] = []

    def plain(text: str) -> list[float]:
        calls.append(text)
        return hashed_embedding(text)

    VectorIndex(memory, embed=plain, model_code="plain").sync()
    assert len(calls) == 2


def test_row_cache_follows_deletes_updates_and_rebuilds(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    memory.write("woks", "Carbon fibre woks conduct heat.")
    memory.write("dvv", "Finnish DVV identity code.")
    index = VectorIndex(memory, embed=hashed_embedding)
    index.sync()
    assert index.search("carbon woks")[0].name == "woks"
    # Another process's sync deletes a row: only the revision/count move.
    memory.delete("woks")
    VectorIndex(memory, embed=hashed_embedding).sync()
    assert [h.name for h in index.search("carbon woks", min_score=-1.0)] == ["dvv"]
    memory.write("dvv", "Finnish DVV now also issues carbon woks.")
    VectorIndex(memory, embed=hashed_embedding).sync()
    assert index.search("issues carbon woks")[0].name == "dvv"
    # Rebuilt file with the same row count and revision numbers.
    index.clear()
    memory.delete("dvv")
    memory.write("pans", "Copper pans for induction.")
    fresh = VectorIndex(memory, embed=hashed_embedding)
    fresh.sync()
    assert [h.name for h in index.search("copper pans")] == ["pans"]


def test_search_opens_index_without_write_lock(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    memory.write("alpha", "Alpha beta gamma.")
    index = VectorIndex(memory, embed=hashed_embedding)
    index.sync()
    with closing(sqlite3.connect(index.path, timeout=0.1)) as writer:
        writer.execute("BEGIN IMMEDIATE")  # another process mid-write
        other = VectorIndex(memory, embed=hashed_embedding)
        start = time.monotonic()
        assert other.search("alpha")[0].name == "alpha"
        assert other.count() == 1
        assert time.monotonic() - start < 5
        writer.rollback()


def test_query_embedding_cache_and_overlap(tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    tools = MemoryTools(tmp_path, embed=embedder, model_code="counting")
    assert tools.memory_search("anything") == "No memory pages yet."
    tools.memory_write("woks", "Carbon fibre woks conduct heat.")
    assert "woks" in tools.memory_search("carbon woks")
    assert "woks.md" in tools.memory_pull("carbon woks")
    assert embedder.calls.count("carbon woks") == 1
    for i in range(QUERY_CACHE_SIZE + 1):
        tools.index.embed_query(f"query {i}")
    tools.index.embed_query("carbon woks")  # evicted: embedded again
    assert embedder.calls.count("carbon woks") == 2
    tools.index.embed_query(f"query {QUERY_CACHE_SIZE}")  # still cached
    assert embedder.calls.count(f"query {QUERY_CACHE_SIZE}") == 1


def test_query_embedding_failure_propagates_from_search(tmp_path: Path) -> None:
    tools = MemoryTools(tmp_path, embed=CountingEmbedder(fail_on_text="boom"), model_code="c")
    tools.memory_write("woks", "Carbon fibre woks conduct heat.")
    with pytest.raises(RuntimeError, match="unavailable"):
        tools.memory_search("boom query")
    assert tools.index.count() == 1  # the concurrent sync still ran


def test_near_duplicates_match_brute_force_across_blocks(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    count = index_module._PAIR_BLOCK_ROWS + 30
    for i in range(count):
        memory.write(f"p{i:05d}", f"topic {i % 40} shared words {i % 3}", title="Note")
    index = VectorIndex(memory, embed=hashed_embedding)
    index.sync()
    pairs = index.near_duplicates(threshold=0.95)
    names = memory.page_names()
    expected = set()
    embedded = {n: hashed_embedding(index.embedding_input(memory.read(n).raw)) for n in names}
    for a_i, a in enumerate(names):
        for b in names[a_i + 1 :]:
            if math.sumprod(embedded[a], embedded[b]) >= 0.95 + 1e-5:
                expected.add((a, b))
    got = {(a, b) for a, b, _ in pairs}
    assert expected <= got
    assert all(a < b for a, b in got)
    assert any(int(b[1:]) >= index_module._PAIR_BLOCK_ROWS > int(a[1:]) for a, b in got)
    assert [s for _, _, s in pairs] == sorted((s for _, _, s in pairs), reverse=True)


def test_page_stats_handles_missing_and_file_roots(tmp_path: Path) -> None:
    assert MemoryDir(tmp_path / "missing").page_stats() == {}
    file_root = tmp_path / "file"
    file_root.write_text("x", encoding="utf-8")
    assert MemoryDir(file_root).page_stats() == {}
    root = tmp_path / "mem"
    (root / "sub.md").mkdir(parents=True)
    (root / "Bad_Name.md").write_text("x", encoding="utf-8")
    (root / "ok.md").write_text("x", encoding="utf-8")
    (root / "ok.txt").write_text("x", encoding="utf-8")
    assert list(MemoryDir(root).page_stats()) == ["ok"]


@requires_openai
@pytest.mark.live_api
def test_model_embedder_embed_many_matches_single_calls() -> None:
    embedder = ModelEmbedder()
    texts = ["carbon fibre woks", "Finnish DVV identity code"]
    batch = embedder.embed_many(texts)
    assert len(batch) == 2
    for text, vector in zip(texts, batch, strict=True):
        single = embedder(text)
        assert math.sumprod(vector, single) / math.sqrt(
            math.sumprod(vector, vector) * math.sumprod(single, single)
        ) > 0.999
    assert embedder.embed_many([]) == []
