# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for memory_refresh: re-indexing, duplicate, stale and review reporting."""

import sys
from pathlib import Path
from types import FrameType

from kiss.core.memoryfield.index import VectorIndex, hashed_embedding
from kiss.core.memoryfield.pages import MemoryDir
from kiss.core.memoryfield.tools import MemoryTools


def make_tools(root: Path) -> MemoryTools:
    return MemoryTools(root, embed=hashed_embedding)


def set_stamp(tools: MemoryTools, name: str, key: str, value: str) -> None:
    """Rewrite frontmatter line *key* of page *name* by hand, as an editor would."""
    path = tools.memory.page_path(name)
    text = path.read_text(encoding="utf-8")
    start = text.index(f"{key}:")
    end = text.index("\n", start)
    path.write_text(text[:start] + f"{key}: {value!r}" + text[end:], encoding="utf-8")


class _AgentWritesFirst:
    """Profile hook: *agent* writes page *name* as another write is about to take the memory's lock.

    The write of an agent that commits while a job's write waits for the lock, made at
    that exact place on the job's own thread.  Events between two threads cannot place
    it there: the lock is the only thing the job's write waits for, and a test thread
    holding it would keep the agent's write out as well.
    """

    def __init__(self, agent: MemoryTools, name: str) -> None:
        self.agent = agent
        self.name = name
        self.verified = ""

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "call" and frame.f_code is MemoryDir._mutation_lock.__code__:
            self.agent.memory_write(self.name, "Checked by the agent.")
            self.verified = self.agent.memory.read(self.name).frontmatter["verified"]


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
    set_stamp(tools, "old", "verified", "2020-01-05T00:00:00Z")
    set_stamp(tools, "dateless", "verified", "not-a-date")
    out = tools.memory_refresh(stale_days=365)
    assert "Pages not verified or updated in 365 days" in out
    stale_section = out.split("Pages not verified or updated")[1]
    assert "  old  (verified 2020-01-05T00:00:00Z)" in stale_section
    assert "fresh" not in stale_section
    assert "dateless" not in stale_section


def test_a_job_rewrite_does_not_make_a_page_look_verified(tmp_path: Path) -> None:
    agent = MemoryTools(tmp_path / "memory", embed=hashed_embedding, task_id="task-7")
    agent.memory_write("ports", "The daemon listens on port 8765.", sources="install.sh")
    first = agent.memory.read("ports").frontmatter
    assert (first["task"], first["writer"], first["sources"]) == ("task-7", "agent", "install.sh")
    assert first["created"] <= first["verified"] <= first["updated"]
    set_stamp(agent, "ports", "verified", "2020-01-05T00:00:00Z")
    job = MemoryTools(
        tmp_path / "memory", embed=hashed_embedding, task_id="job-1", writer="consolidation"
    )
    # A verified stamp in the body is not the job's to set either.
    job.memory_write("ports", "---\nverified: '2030-01-01T00:00:00Z'\n---\nPort 8765 (merged).")
    merged = agent.memory.read("ports").frontmatter
    assert (merged["task"], merged["writer"]) == ("job-1", "consolidation")
    assert merged["verified"] == "2020-01-05T00:00:00Z" < merged["updated"]
    assert merged["sources"] == "install.sh"
    assert "  ports  (verified 2020-01-05T00:00:00Z)" in agent.memory_refresh(stale_days=365)
    # Search and recall say who wrote the page, and recall dates it by its verification.
    assert agent.memory_search("port 8765").splitlines()[0].endswith(
        "ports  —  ports [written by a consolidation job]"
    )
    assert "- `ports` (2020-01-05): ports [written by a consolidation job]" in agent.recall(
        "port 8765"
    )
    # An agent that writes the page again has verified it.
    agent.memory_write("ports", "The daemon listens on port 8765 (checked).")
    checked = agent.memory.read("ports").frontmatter
    assert (checked["task"], checked["writer"]) == ("task-7", "agent")
    assert first["verified"] <= checked["verified"] <= checked["updated"]
    assert "No near-duplicate or stale pages." in agent.memory_refresh(stale_days=365)
    assert "written by" not in agent.memory_search("port 8765") + agent.recall("port 8765")
    # A page from before provenance was last written by an agent, at its update: a
    # job's rewrite keeps that time as the page's verification.
    legacy = "---\ntitle: Legacy\nupdated: '2020-01-01T00:00:00Z'\n---\nAn old fact about tides.\n"
    (tmp_path / "memory" / "legacy.md").write_text(legacy, encoding="utf-8")
    assert "  legacy  (updated 2020-01-01T00:00:00Z)" in agent.memory_refresh(stale_days=365)
    job.memory_write("legacy", "An old fact about tides (merged).")
    merged = agent.memory.read("legacy").frontmatter
    assert merged["verified"] == "2020-01-01T00:00:00Z" < merged["updated"]
    assert "  legacy  (verified 2020-01-01T00:00:00Z)" in agent.memory_refresh(stale_days=365)
    assert "- `legacy` (2020-01-01): Legacy [written by a consolidation job]" in agent.recall(
        "old fact about tides"
    )
    # A page a job creates, and rewrites, was never verified.
    job.memory_write("digest", "A digest of the tide tables.")
    job.memory_write("digest", "A digest of the tide tables (second pass).")
    assert "verified" not in agent.memory.read("digest").frontmatter
    # Recall and the stale listing have no time to date it by.
    assert "- `digest`: digest [written by a consolidation job]" in agent.recall("tide tables")
    assert "digest" not in agent.memory_refresh(stale_days=0)
    # The time a job keeps is the one stored when its write holds the memory's lock: an
    # agent's verification committed while the job waited for the lock survives the job's
    # write, on a page verified long ago and on one that did not exist when the job began.
    set_stamp(agent, "ports", "verified", "2020-01-05T00:00:00Z")
    for name in ("ports", "tides"):
        hook = _AgentWritesFirst(agent, name)
        previous = sys.getprofile()
        sys.setprofile(hook)
        try:
            job.memory_write(name, "Rewritten by the job.")
        finally:
            sys.setprofile(previous)
        raced = agent.memory.read(name)
        assert raced.body == "Rewritten by the job.\n"
        assert raced.frontmatter["writer"] == "consolidation"
        assert raced.frontmatter.get("verified") == hook.verified >= checked["verified"]


def test_refresh_lists_pages_past_their_review_date(tmp_path: Path) -> None:
    tools = make_tools(tmp_path / "memory")
    assert tools.memory_write("oops", "x", review_after="next week") == (
        "Error: review_after must be a date (YYYY-MM-DD), not 'next week'."
    )
    assert tools.memory_list() == "No memory pages yet."
    tools.memory_write("price", "The plan costs 20 dollars a month.", review_after="2020-06-30")
    tools.memory_write("licence", "The licence runs out next decade.", review_after="2999-01-01")
    tools.memory_write("mangled", "Somebody edited the date.", review_after="2020-06-30")
    tools.memory_write("plain", "A fact that does not expire.")
    set_stamp(tools, "mangled", "review_after", "soon")
    assert tools.memory.read("price").frontmatter["review_after"] == "2020-06-30"
    assert "review_after" not in tools.memory.read("plain").frontmatter
    # Written after its review date: the writer has just verified it.
    assert "No near-duplicate or stale pages." in tools.memory_refresh()
    set_stamp(tools, "price", "verified", "2020-01-05T00:00:00Z")
    due = "Pages past their review_after date (re-verify and rewrite, or delete):\n  price  ("
    out = tools.memory_refresh(stale_days=100_000)
    assert due + "review after 2020-06-30)" in out
    assert "licence" not in out and "mangled" not in out and "No near-duplicate" not in out
    # Rewriting the page verifies it and keeps the date; a new date replaces it.
    tools.memory_write("price", "The plan costs 25 dollars a month.")
    assert tools.memory.read("price").frontmatter["review_after"] == "2020-06-30"
    assert "review_after" not in tools.memory_refresh()
    tools.memory_write("price", "The plan costs 25 dollars a month.", review_after="2031-01-01")
    assert tools.memory.read("price").frontmatter["review_after"] == "2031-01-01"
    # Dates are read whole and compared on their UTC day: text after a date makes it
    # no date, and a verification counts on the day it happened in UTC.
    tools.memory_write("same-day", "The ferry leaves at dawn.", review_after="2020-06-30")
    tools.memory_write("day-after", "Invoices are due in March.", review_after="2020-06-30")
    tools.memory_write("tail", "The printer needs toner.", review_after="2020-06-30")
    set_stamp(tools, "same-day", "verified", "2020-07-01T00:30:00+02:00")  # 22:30 UTC, June 30
    set_stamp(tools, "day-after", "verified", "2020-06-30T23:30:00-02:00")  # 01:30 UTC, July 1
    set_stamp(tools, "tail", "verified", "2020-01-05T00:00:00Z")
    set_stamp(tools, "tail", "review_after", "2020-06-30NOT_A_DATE")
    out = tools.memory_refresh(stale_days=100_000)
    assert due.replace("price", "same-day") + "review after 2020-06-30)" in out
    assert "day-after" not in out and "tail" not in out
    # A time that falls on no UTC day (outside the years 1 to 9999) is no date either:
    # as a review date it is skipped, as a verification it does not count.
    set_stamp(tools, "licence", "review_after", "9999-12-31T23:30:00-02:00")
    set_stamp(tools, "mangled", "review_after", "0001-01-01T00:30:00+02:00")
    set_stamp(tools, "tail", "review_after", "2020-06-30")
    set_stamp(tools, "tail", "verified", "9999-12-31T23:30:00-02:00")
    out = tools.memory_refresh(stale_days=100_000)
    listed = "review after 2020-06-30)\n  tail  (review after 2020-06-30)"
    assert due.replace("price", "same-day") + listed in out
    assert "licence" not in out and "mangled" not in out


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
