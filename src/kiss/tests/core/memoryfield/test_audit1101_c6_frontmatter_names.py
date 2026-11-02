# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Frontmatter and page-name edge cases from audit 2026-11-01, C6 F2-F5.

* F2: an invalid unquoted YAML timestamp (``updated: 2026-13-01``) is
  malformed frontmatter, not an exception that disables search over the
  memory's healthy pages.
* F3: a valid unquoted timestamp (parsed by YAML into a ``datetime``) is
  as stale as its quoted twin.
* F4: ``extra`` cannot replace the stored ``uuid`` / ``created`` of an
  existing page.
* F5: a page or memory name ending in a newline is invalid.
"""

from pathlib import Path

import pytest

from kiss.core.memoryfield.index import hashed_embedding
from kiss.core.memoryfield.pages import MemoryDir, is_valid_page_name, split_frontmatter
from kiss.core.memoryfield.tools import MemoryTools

INVALID_TIMESTAMPS = (
    "---\nupdated: 2026-13-01\n---\nA malformed frontmatter date.\n",
    "---\nupdated: 0001-01-01T00:00:00+99:00\n---\nAn impossible zone offset.\n",
)


@pytest.mark.parametrize("text", INVALID_TIMESTAMPS)
def test_invalid_yaml_timestamp_is_malformed_frontmatter(text: str) -> None:
    assert split_frontmatter(text) == ({}, text)


def test_invalid_yaml_timestamp_page_does_not_disable_search(tmp_path: Path) -> None:
    tools = MemoryTools(tmp_path / "memory", embed=hashed_embedding)
    tools.memory_write("good", "Postgres database listens on port 6433.")
    (tools.memory.root / "bad.md").write_text(INVALID_TIMESTAMPS[0], encoding="utf-8")
    hits = tools.memory_search("Postgres database", memory="general")
    assert "good" in hits and not hits.startswith("Error:")
    assert "bad  —  bad" in tools.memory_list()
    assert "A malformed frontmatter date." in tools.memory_read("bad")


def test_unquoted_yaml_timestamp_counts_as_stale(tmp_path: Path) -> None:
    tools = MemoryTools(tmp_path / "memory", embed=hashed_embedding)
    tools.memory.root.mkdir()
    stamps = {
        "quoted": "'2000-01-01T00:00:00Z'",
        "unquoted": "2000-01-01T00:00:00Z",
        "offset": "2000-01-01T00:00:00+02:00",
        "naive": "2000-01-01T00:00:00",
        "mangled": "not-a-date",
    }
    for name, stamp in stamps.items():
        (tools.memory.root / f"{name}.md").write_text(
            f"---\ntitle: {name}\nupdated: {stamp}\n---\nDatabase notes {name}.\n",
            encoding="utf-8",
        )
    tools.memory_write("fresh", "Database notes written just now.")
    out = tools.memory_refresh(stale_days=30)
    stale_section = out.split("Pages not updated in 30 days")[1]
    for name in ("quoted", "unquoted", "offset", "naive"):
        assert f"  {name}  (updated " in stale_section, name
    assert "mangled" not in stale_section
    assert "fresh" not in stale_section


def test_extra_cannot_replace_the_identity_of_an_existing_page(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path / "memory")
    first = memory.write("page", "Original body")
    extra = {"uuid": "replacement", "created": "1999-01-01T00:00:00Z", "source": "s"}
    second = memory.write("page", "Updated body", extra=extra)
    assert second.frontmatter["uuid"] == first.frontmatter["uuid"]
    assert second.frontmatter["created"] == first.frontmatter["created"]
    assert second.frontmatter["source"] == "s"
    assert memory.read("page").frontmatter["uuid"] == first.frontmatter["uuid"]
    assert extra == {"uuid": "replacement", "created": "1999-01-01T00:00:00Z", "source": "s"}
    # On create, extra may still carry the identity (importing from another memoryfield).
    imported = memory.write("imported", "body", extra=extra)
    assert imported.frontmatter["uuid"] == "replacement"
    assert imported.frontmatter["created"] == "1999-01-01T00:00:00Z"


def test_names_ending_in_a_newline_are_invalid(tmp_path: Path) -> None:
    assert not is_valid_page_name("notes\n")
    assert not is_valid_page_name("notes\r\n")
    with pytest.raises(ValueError):
        MemoryDir(tmp_path / "memory").write("notes\n", "Database notes")
    tools = MemoryTools(tmp_path / "memory", embed=hashed_embedding)
    assert tools.memory_write("notes\n", "Database notes").startswith("Error:")
    assert tools.memory_list() == "No memory pages yet."
    assert [path.name for path in (tmp_path / "memory").glob("*.md")] == []
    with pytest.raises(ValueError, match="Invalid memory name"):
        MemoryTools(tmp_path / "memory", embed=hashed_embedding, domains={"kiss\n": "d"})
