"""End-to-end tests for the agent-facing memory tools (offline + one live agent run)."""

import os
from pathlib import Path

import pytest

from kiss.agents.memoryfield.index import hashed_embedding
from kiss.agents.memoryfield.tools import MEMORY_PROTOCOL, PULL_CHAR_LIMIT, MemoryTools
from kiss.core.kiss_agent import KISSAgent

live_api = pytest.mark.live_api
requires_keys = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("OPENAI_API_KEY")),
    reason="ANTHROPIC_API_KEY and OPENAI_API_KEY needed for the live agent memory test",
)


def make_tools(root: Path) -> MemoryTools:
    return MemoryTools(root, embed=hashed_embedding)


def test_tools_full_lifecycle(tmp_path: Path) -> None:
    tools = make_tools(tmp_path / "memory")
    names = [t.__name__ for t in tools.tools()]
    assert names == [
        "memory_search",
        "memory_pull",
        "memory_read",
        "memory_write",
        "memory_list",
        "memory_delete",
        "memory_refresh",
    ]
    assert tools.memory_search("anything") == "No memory pages yet."
    assert tools.memory_pull("anything") == "No memory pages yet."
    assert tools.memory_list() == "No memory pages yet."

    assert tools.memory_write(
        "carbon-fibre-woks",
        "Carbon fibre woks conduct heat unevenly.\nSource: https://example.com",
        title="Carbon Fibre Woks",
        summary="Thermal properties of carbon fibre cookware",
    ).startswith("Wrote carbon-fibre-woks.md (")
    assert tools.memory_write(
        "finnish-id", "Getting a Finnish identity code requires visiting DVV."
    ).startswith("Wrote finnish-id.md")

    listing = tools.memory_list()
    assert "carbon-fibre-woks  —  Carbon Fibre Woks: Thermal properties" in listing
    assert "finnish-id  —  finnish id" in listing and "finnish id:" not in listing

    search = tools.memory_search("carbon fibre woks heat")
    assert search.splitlines()[0].endswith(
        "carbon-fibre-woks  —  Carbon Fibre Woks: Thermal properties of carbon fibre cookware"
    )
    assert tools.memory_search("carbon fibre woks heat", k=1).count("\n") == 0

    pulled = tools.memory_pull("Finnish identity code DVV", k=1)
    assert pulled.startswith("### finnish-id.md  (score ")
    assert "Getting a Finnish identity code requires visiting DVV." in pulled

    read = tools.memory_read("carbon-fibre-woks.md")
    assert read.startswith("---\ntitle: Carbon Fibre Woks\n") and read.endswith(
        "Source: https://example.com\n"
    )
    assert tools.memory_read("missing") == "Error: no memory page named 'missing'."
    assert tools.memory_read("../escape").startswith("Error: Invalid page name")

    assert tools.memory_write("Bad Name", "x").startswith("Error: Invalid page name")
    assert tools.memory_write("blank", "  ") == "Error: Refusing to write an empty page."

    assert tools.memory_delete("finnish-id") == "Deleted finnish-id."
    assert tools.memory_delete("finnish-id") == "Error: no memory page named 'finnish-id'."
    assert tools.memory_delete("../x").startswith("Error: Invalid page name")
    # After deletion the index drops the row on the next search.
    assert "finnish-id" not in tools.memory_search("Finnish identity code DVV", k=5)
    # An empty query embeds to the zero vector, so every page is orthogonal and dropped.
    assert tools.memory_search("", k=5) == "No matches."
    assert tools.memory_pull("") == "No matches."
    assert sorted(p.name for p in (tmp_path / "memory").iterdir()) == [
        "carbon-fibre-woks.md",
        "hashed-bow-v1.sqlite3",
    ]


def test_write_warns_when_page_exceeds_embedding_limit(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory_write("big", "word " * 3000)
    assert "Warning: page is" in result and "Split it into several pages." in result
    assert tools.memory_search("word")[0:5] != "No ma"


def test_pull_caps_total_output(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    body = "alpha beta gamma delta " * 500  # ~11.5 KB per page
    for i in range(4):
        tools.memory_write(f"page-{i}", body)
    pulled = tools.memory_pull("alpha beta gamma", k=4)
    assert len(pulled) <= PULL_CHAR_LIMIT + 200
    assert pulled.count("### page-") == 2
    assert "more page(s) omitted" in pulled


def test_pull_truncates_an_oversized_first_hit(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write("huge", "alpha beta " * 6000)  # ~66 KB
    tools.memory_write("small", "alpha beta gamma")
    pulled = tools.memory_pull("alpha beta", k=2)
    assert len(pulled) <= PULL_CHAR_LIMIT + 200
    assert "[page truncated; use memory_read for the rest]" in pulled
    assert "[1 more page(s) omitted" in pulled
    alone = tools.memory_pull("alpha beta", k=1)
    assert "[page truncated" in alone and "omitted" not in alone


def test_protocol_mentions_every_tool() -> None:
    for name in (
        "memory_search",
        "memory_pull",
        "memory_write",
        "memory_delete",
        "memory_refresh",
    ):
        assert name in MEMORY_PROTOCOL


@live_api
@requires_keys
def test_agent_remembers_across_sessions(tmp_path: Path) -> None:
    """A KISSAgent stores a fact through the tools; a fresh agent recalls it by semantic search."""
    tools = MemoryTools(tmp_path / "memory")
    system_prompt = MEMORY_PROTOCOL + "\nBe brief. Do not ask questions."

    writer = KISSAgent("memory-writer")
    writer.run(
        model_name="claude-haiku-4-5",
        prompt_template=(
            "Store this in memory as a page (name it yourself), then call finish: "
            "The staging Postgres for project Kestrel listens on port 6433 and its "
            "read-only role is named kestrel_ro."
        ),
        system_prompt=system_prompt,
        tools=tools.tools(),
        max_steps=8,
        verbose=False,
    )
    names = tools.memory.page_names()
    assert names, "the agent did not write a memory page"
    assert "6433" in tools.memory.read(names[0]).raw

    reader = KISSAgent("memory-reader")
    answer = reader.run(
        model_name="claude-haiku-4-5",
        prompt_template=(
            "Which port does the Kestrel staging database use and what is the read-only "
            "role called? Look it up in memory first, then answer in one line via finish."
        ),
        system_prompt=system_prompt,
        tools=tools.tools(),
        max_steps=8,
        verbose=False,
    )
    assert "6433" in answer and "kestrel_ro" in answer
