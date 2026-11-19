"""End-to-end tests for the agent-facing memory tools (offline + one live agent run)."""

import os
import re
import threading
import time
from pathlib import Path

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.memoryfield import tools as tools_module
from kiss.core.memoryfield.index import VectorIndex, hashed_embedding
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
    # Under each hit: the statement of the page that matched, on one line.
    assert tools.memory_search("carbon fibre woks heat", k=1).splitlines()[1:] == [
        "       > Carbon fibre woks conduct heat unevenly. Source: https://example.com"
    ]

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
    # The deletion left a tombstone, so a sync with another machine deletes the page there too.
    assert sorted(p.name for p in (tmp_path / "memory").iterdir()) == [
        ".tombstones",
        "carbon-fibre-woks.md",
        "hashed-bow-v1.sqlite3",
    ]
    assert [p.name for p in (tmp_path / "memory" / ".tombstones").iterdir()] == ["finnish-id"]


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


def test_list_counts_pages_not_lines_of_a_multiline_summary(tmp_path: Path) -> None:
    """A summary with newlines is one page: listed on one line, counted once.

    The listing's cut used to count physical lines, so a five-line summary
    made ``memory_list`` report ``-1 more pages not shown`` while the
    second page was missing.
    """
    tools = make_tools(tmp_path)
    tools.memory_write("first", "body", title="First", summary="one\ntwo\nthree\nfour\nfive")
    tools.memory_write("second", "body", title="Second", summary="short")
    whole = tools.memory_list()
    assert whole == "first  —  First: one two three four five\nsecond  —  Second: short"
    cut = tools.memory_list(max_output_chars=45)
    assert cut.startswith("first  —  First: one two three four five\n... [1 more pages not shown")


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
    # Each write indexed its page, so the refresh finds nothing to embed.
    assert "Index of general refreshed: 0 added, 0 updated, 0 removed, 0 unchanged." in report
    assert "Index of kiss refreshed: 0 added, 0 updated, 0 removed, 2 unchanged." in report
    assert "Index of finance refreshed: 0 added, 0 updated, 0 removed, 1 unchanged." in report
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
    assert (
        "Pages not verified or updated in 0 days (re-verify or delete):\n"
        "  kiss/daemon-ports  (verified "
    ) in stale
    tools.memory_write("kiss/review-round3-notes", "Round notes.", summary="Notes of round 3")
    assert "Warning: the name looks like a per-round/per-session note" in tools.memory_write(
        "kiss/review-round3-notes", "Round notes.", summary="Notes of round 3"
    )
    hit = tools.memory_search("round notes", k=1, memory="kiss")
    assert hit.splitlines()[0].endswith(
        "kiss/review-round3-notes  —  review round3 notes: Notes of round 3"
    )
    assert hit.splitlines()[1:] == ["       > Round notes."]
    # Without domain memories the refresh report keeps its single-memory wording.
    plain = make_tools(tmp_path / "plain")
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "only-page.md").write_text("body", encoding="utf-8")
    assert plain.memory_refresh().startswith("Index refreshed: 1 added")


def test_memories_are_merged_on_one_score_scale_while_statement_vectors_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = make_domain_tools(tmp_path / "memory")
    shelves = "\n".join(f"- shelf {i} holds flour sugar salt pepper rice beans" for i in range(40))
    tools.memory_write("kitchen", f"{shelves}\n- saffron threads cost a fortune")
    # The kiss memory was indexed by a process that predates statement vectors.
    kiss = tools.indexes["kiss"].memory
    kiss.write("saffron-shop", "- the spice shop sells paprika and saffron powder")
    VectorIndex(kiss, embed=hashed_embedding, statements=False).sync()
    monkeypatch.setattr(tools_module, "BACKFILL_SECONDS", 0.0)
    # kiss can only be ranked by page embedding, so the general memory is too: scored
    # with its saffron statement, the kitchen page would outrank the shop's page score.
    by_page = [line.split("  ", 1) for line in tools.memory_search("saffron threads").splitlines()]
    assert [name for _, name in by_page] == [
        "kiss/saffron-shop  —  saffron shop",
        "kitchen  —  kitchen",
    ]
    # Once the search may write the missing statement vectors, both are scored with them.
    monkeypatch.setattr(tools_module, "BACKFILL_SECONDS", 5.0)
    scored = [line.split("  ", 1) for line in tools.memory_search("saffron threads").splitlines()]
    assert [name for _, name in scored] == [
        "kitchen  —  kitchen",
        "     > - saffron threads cost a fortune",
        "kiss/saffron-shop  —  saffron shop",
        "     > - the spice shop sells paprika and saffron powder",
    ]
    assert float(scored[0][0]) > float(scored[2][0]) >= float(by_page[0][0]) > float(by_page[1][0])
    # A memory whose only hit is a page of headings, which has no statement, is not a
    # memory waiting for its vectors: the others keep their statement scores.
    tools.memory_write("finance/spice-budget", "# Saffron threads\n## Budget\n")
    with_headings = [line.strip() for line in tools.memory_search("saffron threads").splitlines()]
    assert [line.split("  ", 1)[-1] for line in with_headings] == [
        "finance/spice-budget  —  spice budget",
        "kitchen  —  kitchen",
        "> - saffron threads cost a fortune",
        "kiss/saffron-shop  —  saffron shop",
        "> - the spice shop sells paprika and saffron powder",
    ]


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
    with pytest.raises(ValueError, match="Invalid writer 'robot': use one of agent, "):
        MemoryTools(tmp_path / "memory", embed=hashed_embedding, writer="robot")


PORTS = "The daemon listens on port 8765 and is restarted through its systemd user service."


def test_write_indexes_the_page_and_names_the_pages_it_nearly_duplicates(tmp_path: Path) -> None:
    tools = make_domain_tools(tmp_path / "memory")
    first = tools.memory_write("kiss/daemon-ports", PORTS, title="Ports")
    assert re.fullmatch(r"Wrote kiss/daemon-ports\.md \(\d+ bytes\)\.", first)
    assert tools.indexes["kiss"].count() == 1  # indexed by the write, before any search
    copy = tools.memory_write("kiss/daemon-port-copy", PORTS, title="Ports")
    assert copy.endswith(
        " bytes). Similar pages: kiss/daemon-ports (1.00). If this page replaces or extends "
        "one of them, write to that name or memory_delete it; pages about different dates "
        "or different occurrences of the same thing stay separate."
    )
    # Only the pages of the memory written to are compared.
    assert "Similar pages" not in tools.memory_write("daemon-ports", PORTS, title="Ports")
    assert "Similar pages" not in tools.memory_write("kiss/garage", "A hammer and a saw.")
    # At most three are named, the closest first; a rewrite does not name its own page.
    tools.memory_write("kiss/daemon-ports-near", PORTS + " Usually it is.", title="Ports")
    tools.memory_write("kiss/daemon-ports-b", PORTS, title="Ports")
    rewrite = tools.memory_write("kiss/daemon-ports", PORTS, title="Ports")
    listed = re.search(r"Similar pages: (.*?)\. If this page", rewrite)
    assert listed is not None
    names = [entry.split(" ")[0] for entry in listed.group(1).split(", ")]
    assert names == ["kiss/daemon-port-copy", "kiss/daemon-ports-b", "kiss/daemon-ports-near"]
    assert re.fullmatch(r"kiss/daemon-ports-near \(0\.9\d\)", listed.group(1).split(", ")[2])
    # A page deleted since the last search is still in the index, and is not named.
    for name in names[:2]:
        assert tools.memory_delete(name) == f"Deleted {name}."
    rewrite = tools.memory_write("kiss/daemon-ports", PORTS, title="Ports")
    assert re.search(r"Similar pages: kiss/daemon-ports-near \(0\.9\d\)\. If this page", rewrite)


def test_write_warns_about_tool_call_markup_in_title_or_summary(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    warning = " Warning: the title or summary contains tool-call markup (such as </summary>)"
    assert warning in tools.memory_write("leaked-title", "body one", title="Ports</title>")
    assert warning in tools.memory_write(
        "leaked-summary", "body two", summary='Ports</parameter>\n<parameter name="x">'
    )
    assert "Warning" not in tools.memory_write(
        "html-notes", "The <summary> tag opens a <details> box.", title="HTML <details> notes"
    )


class FailingEmbedding:
    """An embedding service that is down."""

    def __call__(self, text: str) -> list[float]:
        raise RuntimeError("embedding service unavailable")


class HeldEmbedding:
    """An embedding service that answers only once ``release`` is set."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.worker: threading.Thread | None = None

    def __call__(self, text: str) -> list[float]:
        self.worker = threading.current_thread()
        self.release.wait(60)
        return hashed_embedding(text)


def test_a_failed_or_slow_embedding_costs_the_notice_never_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_tools(tmp_path).memory_write("daemon-ports", PORTS, title="Ports")
    down = MemoryTools(tmp_path, embed=FailingEmbedding())
    reply = down.memory_write("daemon-port-copy", PORTS, title="Ports")
    assert re.fullmatch(r"Wrote daemon-port-copy\.md \(\d+ bytes\)\.", reply)
    assert down.memory.read("daemon-port-copy").body == PORTS + "\n"
    assert down.index.count() == 1  # left to the next search
    assert "daemon-port-copy" in make_tools(tmp_path).memory_search("daemon port")

    held = HeldEmbedding()
    slow = MemoryTools(tmp_path, embed=held)
    monkeypatch.setattr(tools_module, "WRITE_INDEX_SECONDS", 0.05)
    started = time.monotonic()
    reply = slow.memory_write("daemon-ports-b", PORTS, title="Ports")
    assert time.monotonic() - started < 30
    assert re.fullmatch(r"Wrote daemon-ports-b\.md \(\d+ bytes\)\.", reply)
    assert slow.index.count() == 2
    # The request left behind is the page's one: it stores the row when it is answered.
    held.release.set()
    assert held.worker is not None
    held.worker.join(60)
    assert not held.worker.is_alive() and slow.index.count() == 3


class DeletesPageDuringSearch:
    """The hashed embedder; its query request deletes a page once the sync has scanned it."""

    def __init__(self, query: str, page: Path) -> None:
        self.query = query
        self.page = page
        self.scanned = threading.Event()

    def __call__(self, text: str) -> list[float]:
        if text == self.query:
            assert self.scanned.wait(60), "the sync never embedded the new page"
            self.page.unlink()
        else:
            self.scanned.set()  # the sync embeds only after its scan of the directory
        return hashed_embedding(text)


def test_search_lists_a_page_deleted_since_the_scan_without_its_statement(tmp_path: Path) -> None:
    query = "daemon port"
    embedding = DeletesPageDuringSearch(query, tmp_path / "daemon-ports.md")
    tools = MemoryTools(tmp_path, embed=embedding)
    tools.memory_write("daemon-ports", PORTS, title="Ports")
    (tmp_path / "new-page.md").write_text("The daemon port is in the unit file.", encoding="utf-8")
    embedding.scanned.clear()
    lines = tools.memory_search(query).splitlines()
    assert not (tmp_path / "daemon-ports.md").exists()
    gone = next(i for i, line in enumerate(lines) if "  daemon-ports  —  Ports" in line)
    assert not any(line.startswith("       > ") for line in lines[gone + 1 : gone + 2])
    assert "       > The daemon port is in the unit file." in lines


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
