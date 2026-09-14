# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for memory_refresh: re-indexing, duplicate and stale reporting."""

from pathlib import Path

from kiss.core.memoryfield.index import VectorIndex, hashed_embedding
from kiss.core.memoryfield.pages import MemoryDir
from kiss.core.memoryfield.tools import MemoryTools


def make_tools(root: Path) -> MemoryTools:
    return MemoryTools(root, embed=hashed_embedding)


def test_refresh_on_empty_memory(tmp_path: Path) -> None:
    tools = make_tools(tmp_path / "memory")
    out = tools.memory_refresh()
    assert "0 added, 0 updated, 0 removed, 0 unchanged" in out
    assert "No near-duplicate or stale pages." in out


def test_refresh_reports_near_duplicates_only(tmp_path: Path) -> None:
    tools = make_tools(tmp_path / "memory")
    tools.memory_write("wok-care-v1", "Season carbon steel woks with flaxseed oil at 250C.")
    tools.memory_write("wok-care-v2", "Season carbon steel woks with flaxseed oil at 240C.")
    tools.memory_write("pg-port", "The staging Postgres listens on port 6433.")
    out = tools.memory_refresh(duplicate_threshold=0.7)
    assert "Near-duplicate pages" in out
    assert "wok-care-v1 ~ wok-care-v2" in out
    assert "pg-port" not in out.split("Near-duplicate pages")[1]
    # A stricter threshold drops the pair.
    assert "Near-duplicate" not in tools.memory_refresh(duplicate_threshold=0.999)


def test_refresh_reindexes_pages_edited_outside_the_agent(tmp_path: Path) -> None:
    tools = make_tools(tmp_path / "memory")
    tools.memory_write("editor-page", "original body about parsers")
    tools.memory_search("parsers")  # settle the index
    page_path = tools.memory.page_path("editor-page")
    page_path.write_text(
        page_path.read_text(encoding="utf-8").replace("parsers", "linkers"),
        encoding="utf-8",
    )
    (tmp_path / "memory" / "new-page.md").write_text("written by a human", encoding="utf-8")
    out = tools.memory_refresh()
    assert "1 added, 1 updated, 0 removed" in out


def test_refresh_reports_stale_pages_and_skips_unparseable_dates(tmp_path: Path) -> None:
    tools = make_tools(tmp_path / "memory")
    tools.memory_write("fresh", "written just now")
    tools.memory_write("old", "written long ago")
    tools.memory_write("dateless", "timestamp got mangled")
    for name, updated in (("old", "2020-01-05T00:00:00Z"), ("dateless", "not-a-date")):
        path = tools.memory.page_path(name)
        text = path.read_text(encoding="utf-8")
        start = text.index("updated:")
        end = text.index("\n", start)
        path.write_text(text[:start] + f"updated: {updated!r}" + text[end:], encoding="utf-8")
    out = tools.memory_refresh(stale_days=365)
    assert "Pages not updated in 365 days" in out
    stale_section = out.split("Pages not updated")[1]
    assert "old" in stale_section
    assert "fresh" not in stale_section
    assert "dateless" not in stale_section


def test_near_duplicates_skips_mismatched_dimensions(tmp_path: Path) -> None:
    """Rows embedded at different dimensions in one index file are not compared."""
    memory = MemoryDir(tmp_path / "memory")
    narrow = VectorIndex(
        memory, embed=lambda text: hashed_embedding(text, dims=64), model_code="shared"
    )
    memory.write("first-page", "alpha beta gamma")
    narrow.sync()
    wide = VectorIndex(
        memory, embed=lambda text: hashed_embedding(text, dims=128), model_code="shared"
    )
    memory.write("second-page", "alpha beta gamma")
    wide.sync()  # re-embeds only the new page, leaving a 64-dim row behind
    assert wide.near_duplicates(threshold=-1.0) == []
