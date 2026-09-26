"""End-to-end tests for the agent-facing memory tools (offline + one live agent run)."""

import os
from pathlib import Path

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.memoryfield.index import hashed_embedding
from kiss.core.memoryfield.tools import MEMORY_PROTOCOL, PULL_CHAR_LIMIT, MemoryTools

live_api = pytest.mark.live_api
requires_keys = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("OPENAI_API_KEY")),
    reason="ANTHROPIC_API_KEY and OPENAI_API_KEY needed for the live agent memory test",
)


def make_tools(root: Path) -> MemoryTools:
    return MemoryTools(root, embed=hashed_embedding)


def test_tools_default_embedder_is_offline_without_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no OPENAI_API_KEY, MemoryTools works end-to-end fully offline."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tools = MemoryTools(tmp_path / "memory")
    assert tools.memory_write(
        "offline-page", "Memory works without any API key via the hashed embedder."
    ).startswith("Wrote offline-page.md")
    assert "offline-page" in tools.memory_search("hashed embedder without API key")
    assert tools.index.path.name == "hashed-bow-v1.sqlite3"


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
    assert "Warning: the page's searchable text" in result
    assert "Split it into several pages." in result
    assert tools.memory_search("word")[0:5] != "No ma"
    # The warning keys on the embedded text (title + summary + body), not the
    # raw file size: frontmatter overhead alone must not trigger it.
    result = tools.memory_write("fits", "word " * 1600, summary="s" * 100)
    assert "Warning" not in result


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


def make_domain_tools(root: Path) -> MemoryTools:
    """A general memory with two domain memories, ``kiss`` and ``finance``."""
    return MemoryTools(
        root, embed=hashed_embedding,
        domains={"kiss": "memory of the repository /r/kiss", "finance": "household finances"},
    )


def test_domain_pages_live_in_sub_directories_and_carry_their_memory_in_the_name(
    tmp_path: Path,
) -> None:
    tools = make_domain_tools(tmp_path / "memory")
    assert tools.memory_write("daily-note", "General lesson about pytest flakes.").startswith(
        "Wrote daily-note.md ("
    )
    out = tools.memory_write("kiss/daemon-ports", "The daemon listens on port 8765.", title="Ports")
    assert out.startswith("Wrote kiss/daemon-ports.md (")
    assert (tmp_path / "memory" / "kiss" / "daemon-ports.md").exists()
    tools.memory_write("finance/mortgage-rate", "The mortgage rate is fixed at 3.1 percent.")
    # The general memory sees only its own pages; each domain sees only its own.
    assert tools.memory.page_names() == ["daily-note"]
    assert tools.indexes["kiss"].memory.page_names() == ["daemon-ports"]
    assert tools.memory_read("kiss/daemon-ports").startswith("---\ntitle: Ports")
    assert tools.memory_read("daemon-ports") == "Error: no memory page named 'daemon-ports'."
    assert tools.memory_read("nope/daemon-ports") == (
        "Error: unknown memory 'nope' in page name 'nope/daemon-ports'; "
        "memories: general, kiss, finance."
    )
    assert tools.memory_write("nope/x", "body").startswith("Error: unknown memory 'nope'")
    assert tools.memory_write("kiss/Bad Name", "body").startswith("Error: Invalid page name")
    assert tools.memory_write("kiss/a/b", "body").startswith("Error: Invalid page name")
    listed = tools.memory_list()
    assert "daily-note  —  daily note" in listed
    assert "kiss/daemon-ports  —  Ports" in listed
    assert "finance/mortgage-rate  —  mortgage rate" in listed
    assert tools.memory_list(memory="kiss") == "kiss/daemon-ports  —  Ports"
    assert tools.memory_list(memory="general") == "daily-note  —  daily note"
    assert tools.memory_list(memory="nope") == (
        "Error: unknown memory 'nope'; memories: general, kiss, finance."
    )


def test_search_spans_every_memory_unless_narrowed(tmp_path: Path) -> None:
    tools = make_domain_tools(tmp_path / "memory")
    tools.memory_write("daily-note", "General lesson about pytest flakes.")
    tools.memory_write("kiss/daemon-ports", "The daemon listens on port 8765.")
    tools.memory_write("finance/mortgage-rate", "The mortgage rate is fixed at 3.1 percent.")
    everything = tools.memory_search("daemon listens on port", k=5)
    assert everything.splitlines()[0].split()[1] == "kiss/daemon-ports"
    assert tools.memory_search("pytest flakes", k=1).split()[1] == "daily-note"
    assert tools.memory_search("daemon listens on port", memory="finance") == "No matches."
    assert "kiss/daemon-ports" in tools.memory_search("daemon listens on port", memory="kiss")
    assert tools.memory_search("daemon", memory="nope") == (
        "Error: unknown memory 'nope'; memories: general, kiss, finance."
    )
    pulled = tools.memory_pull("daemon listens on port", k=1)
    assert pulled.startswith("### kiss/daemon-ports.md  (score ")
    assert "port 8765" in pulled
    assert tools.memory_pull("mortgage", memory="general") == "No matches."
    assert tools.memory_pull("mortgage", memory="nope").startswith("Error: unknown memory")
    # Empty memories answer "No memory pages yet." only when every selected memory is empty.
    empty = make_domain_tools(tmp_path / "empty")
    assert empty.memory_search("anything") == "No memory pages yet."
    empty.memory_write("kiss/one", "one page in kiss")
    assert empty.memory_search("zzz", memory="finance") == "No memory pages yet."
    assert empty.memory_search("zzz") == "No matches."
    (tmp_path / "empty" / "kiss" / "one.md").unlink()
    assert empty.memory_pull("one page in kiss") == "No memory pages yet."


def test_delete_and_refresh_address_domain_memories(tmp_path: Path) -> None:
    tools = make_domain_tools(tmp_path / "memory")
    tools.memory_write("kiss/daemon-ports", "The daemon listens on port 8765.", title="Ports")
    tools.memory_write("kiss/daemon-port-copy", "The daemon listens on port 8765.", title="Ports")
    tools.memory_write("finance/mortgage-rate", "The mortgage rate is fixed at 3.1 percent.")
    report = tools.memory_refresh()
    assert "Index of general refreshed: 0 added" in report
    assert "Index of kiss refreshed: 2 added" in report
    assert "Index of finance refreshed: 1 added" in report
    assert "kiss/daemon-port-copy ~ kiss/daemon-ports  (similarity" in report
    only_kiss = tools.memory_refresh(memory="kiss")
    assert only_kiss.startswith(
        "Index of kiss refreshed: 0 added, 0 updated, 0 removed, 2 unchanged."
    )
    assert "finance" not in only_kiss
    assert tools.memory_refresh(memory="nope").startswith("Error: unknown memory 'nope'")
    assert tools.memory_delete("kiss/daemon-port-copy") == "Deleted kiss/daemon-port-copy."
    assert tools.memory_delete("kiss/daemon-port-copy") == (
        "Error: no memory page named 'kiss/daemon-port-copy'."
    )
    assert tools.memory_delete("nope/x").startswith("Error: unknown memory 'nope'")
    assert tools.indexes["kiss"].memory.page_names() == ["daemon-ports"]
    # Stale pages and search hits are reported under their qualified names, with summaries.
    stale = tools.memory_refresh(stale_days=0, memory="kiss")
    assert "Pages not updated in 0 days (re-verify or delete):\n  kiss/daemon-ports  (updated " in (
        stale
    )
    tools.memory_write("kiss/review-round3-notes", "Round notes.", summary="Notes of round 3")
    assert "Warning: the name looks like a per-round/per-session note" in tools.memory_write(
        "kiss/review-round3-notes", "Round notes.", summary="Notes of round 3"
    )
    hit = tools.memory_search("round notes", k=1, memory="kiss")
    assert hit.endswith("kiss/review-round3-notes  —  review round3 notes: Notes of round 3")
    # Without domain memories the refresh report keeps its single-memory wording.
    plain = make_tools(tmp_path / "plain")
    plain.memory_write("only-page", "body")
    assert plain.memory_refresh().startswith("Index refreshed: 1 added")


def test_protocol_lists_the_attached_domain_memories(tmp_path: Path) -> None:
    plain = make_tools(tmp_path / "plain")
    assert plain.protocol() == MEMORY_PROTOCOL
    tools = make_domain_tools(tmp_path / "memory")
    protocol = tools.protocol()
    assert protocol.startswith(MEMORY_PROTOCOL)
    assert "   - `kiss/`: memory of the repository /r/kiss" in protocol
    assert "   - `finance/`: household finances" in protocol
    assert "`<memory>/<page>`" in protocol


def test_invalid_domain_names_are_rejected(tmp_path: Path) -> None:
    for bad in ("general", "Kiss", "a/b", "", "-x"):
        with pytest.raises(ValueError, match="Invalid memory name"):
            MemoryTools(tmp_path / "memory", embed=hashed_embedding, domains={bad: "d"})


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
