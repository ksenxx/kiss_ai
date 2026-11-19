# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of merging near-duplicate pages: families, commit, archive and restore."""

import hashlib
import json
import sys
from pathlib import Path
from types import FrameType

import pytest

from kiss.core.memoryfield import consolidate
from kiss.core.memoryfield.index import VectorIndex, hashed_embedding
from kiss.core.memoryfield.pages import MemoryDir, PageChangedError
from kiss.core.memoryfield.tools import MemoryTools

PG = (
    "The staging Postgres listens on port 6433 and the replica on 6434. "
    "Backups run nightly at 02:00 UTC to the bucket pg-backups. "
)


def make_tools(root: Path, writer: str = "agent") -> MemoryTools:
    return MemoryTools(root, embed=hashed_embedding, task_id="task-1", writer=writer)


def pg_family(tools: MemoryTools) -> None:
    """Three near-duplicate pages about one database, written by an agent."""
    tools.memory_write("pg-a", PG + "Owner is the platform team.", title="Staging Postgres",
                       sources="runbook.md", review_after="2027-03-01")
    tools.memory_write("pg-b", PG + "Owner is the platform team since March.",
                       title="Staging Postgres", sources="runbook.md, wiki/pg")
    tools.memory_write("pg-c", PG + "The owner is the platform team; pager is #pg.",
                       title="Staging Postgres", summary="ports", review_after="2027-01-15")


def set_stamp(memory: MemoryDir, name: str, key: str, value: str) -> None:
    """Rewrite frontmatter line *key* of page *name* by hand, as an editor would."""
    path = memory.page_path(name)
    text = path.read_text(encoding="utf-8")
    start = text.index(f"{key}:")
    path.write_text(text[:start] + f"{key}: {value!r}" + text[text.index("\n", start):],
                    encoding="utf-8")


class _DeletesAfterListing:
    """Profile hook: deletes page *name* as the index hands out its near-duplicate pairs.

    A page another task deletes between the listing of the pairs and the
    reading of the pages, made at that exact place.
    """

    def __init__(self, memory: MemoryDir, name: str) -> None:
        self.memory = memory
        self.name = name

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "return" and frame.f_code is VectorIndex.near_duplicates.__code__:
            self.memory.delete(self.name)


class _StopsAtDelete:
    """Profile hook: a Stop (``KeyboardInterrupt``) that lands as a merge starts to delete."""

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "call" and frame.f_code is MemoryDir.delete.__code__:
            raise KeyboardInterrupt


UNTOUCHED = "a page of this name exists that the merge did not write"


def test_families_are_connected_groups_and_say_why_one_is_left_alone(tmp_path: Path) -> None:
    tools = make_tools(tmp_path / "memory")
    memory, index = tools.memory, tools.index
    pg_family(tools)
    tools.memory_write("wok", "Season carbon steel woks with flaxseed oil at 250C.")
    # Too large to merge without shortening: two pages of about 4.5 KB each.
    for name in ("big-a", "big-b"):
        tools.memory_write(name, "Deploy checklist of the billing service. " * 110 + name)
    set_stamp(memory, "pg-b", "updated", "2026-01-01T00:00:00Z")
    set_stamp(memory, "pg-a", "updated", "2026-02-01T00:00:00Z")

    # pg-b and pg-c are each close to pg-a only: one family through it, the best pair first.
    assert [(a, b) for a, b, _ in index.near_duplicates(0.9)] == [
        ("big-a", "big-b"), ("pg-a", "pg-b"), ("pg-a", "pg-c")]
    big, pg = consolidate.families(index)
    assert [page.name for page in pg.pages] == ["pg-b", "pg-a", "pg-c"]  # oldest first
    assert pg.target.name == "pg-c" and pg.skip == "" and 0.9 < pg.score < big.score
    assert [page.name for page in big.pages] == ["big-a", "big-b"]
    assert big.skip.startswith("does not fit in one page (")
    assert consolidate.families(index, threshold=0.9999) == []

    # A family examined and not merged is not examined again until one of its pages changes.
    consolidate.leave_alone(memory, big, "separate records")
    consolidate.leave_alone(memory, pg, "2 of 9 facts not kept")
    assert consolidate.families(index)[1].skip == "left alone before: 2 of 9 facts not kept"
    assert consolidate.families(index)[0].skip.startswith("does not fit")
    tools.memory_write("pg-b", PG + "Owner is the platform team since March.")  # a new stamp
    assert consolidate.families(index)[1].skip == ""

    # A page that is not UTF-8 is not merged: what is read from it cannot be archived as it is.
    path = memory.page_path("pg-c")
    path.write_bytes(path.read_bytes() + b"Pager \xff.\n")
    assert consolidate.families(index)[1].skip == "holds a page that is not UTF-8 (pg-c)"
    tools.memory_write("pg-c", PG + "The owner is the platform team; pager is #pg.")
    assert consolidate.families(index)[1].skip == ""

    # A page a merge wrote is not merged again.
    make_tools(tmp_path / "memory", writer="consolidation").memory_write(
        "pg-a", PG + "Owner is the platform team.")
    assert consolidate.families(index)[1].skip == (
        "holds a page a merge wrote (pg-a), which is not merged again")

    # A member deleted while the families are read: the pair that is left is no family.
    tools.memory_delete("pg-b")
    tools.memory_delete("big-b")
    tools.memory_write("pg-a", PG + "Owner is the platform team.")
    hook = _DeletesAfterListing(memory, "pg-a")
    previous = sys.getprofile()
    sys.setprofile(hook)
    try:
        assert consolidate.families(index) == []
    finally:
        sys.setprofile(previous)


def test_commit_keeps_the_originals_and_restore_brings_them_back(tmp_path: Path) -> None:
    tools = make_tools(tmp_path / "memory")
    memory, index = tools.memory, tools.index
    pg_family(tools)
    # pg-a is from before provenance (no writer, no verified): its evidence is its last update.
    path = memory.page_path("pg-a")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if not line.startswith(("task:", "writer:", "verified:", "updated:"))]
    lines.insert(1, "updated: '2026-03-01T00:00:00Z'")
    # newline: a page is stored with "\n" on every OS, and read back byte for byte.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (family,) = consolidate.families(index)
    originals = {page.name: page for page in family.pages}
    assert [page.name for page in family.pages] == ["pg-a", "pg-b", "pg-c"]

    merge = consolidate.commit(memory, family, PG + "Owner: platform team; pager #pg.",
                               "Staging Postgres (merged)", "ports, backups, owner", "task-9")

    assert memory.page_names() == ["pg-c"] and merge.deleted == ["pg-a", "pg-b"]
    assert merge.kept == [] and merge.id.endswith("-pg-c")
    merged = memory.read("pg-c").frontmatter
    assert merged == merge.page.frontmatter
    assert (merged["writer"], merged["task"]) == ("consolidation", "task-9")
    assert (merged["title"], merged["summary"]) == ("Staging Postgres (merged)",
                                                    "ports, backups, owner")
    # As verified as the least verified member; every source; the earliest review date.
    assert merged["verified"] == "2026-03-01T00:00:00Z"
    assert merged["sources"] == "runbook.md, wiki/pg" and merged["review_after"] == "2027-01-15"
    assert (merged["uuid"], merged["created"]) == (
        originals["pg-c"].frontmatter["uuid"], originals["pg-c"].frontmatter["created"])
    archive = tmp_path / "memory" / ".archive" / merge.id
    assert {f.name: f.read_text() for f in archive.glob("*.md")} == {
        f"{name}.md": page.raw for name, page in originals.items()}
    assert sorted(p.name for p in (tmp_path / "memory" / ".tombstones").iterdir()) == [
        "pg-a", "pg-b"]
    (record,) = consolidate.merges(memory)
    assert record["id"] == merge.id and record["members"] == ["pg-a", "pg-b", "pg-c"]
    assert (record["target"], record["task"]) == ("pg-c", "task-9") and "restored" not in record
    assert record["deleted"] == ["pg-a", "pg-b"]
    assert record["written"] == hashlib.sha256(
        memory.page_path("pg-c").read_bytes()).hexdigest()
    assert consolidate.families(index) == []  # also: the archive holds no pages

    restored, skipped = consolidate.restore(memory, merge.id)

    assert (restored, skipped) == (["pg-a", "pg-b", "pg-c"], {})
    assert not list((tmp_path / "memory" / ".tombstones").iterdir())
    for name, original in originals.items():
        page = memory.read(name)
        assert page.body == original.body
        changed = {k for k in page.frontmatter.keys() | original.frontmatter.keys()
                   if page.frontmatter.get(k) != original.frontmatter.get(k)}
        assert changed <= {"updated", "verified"}, (name, changed)
        assert page.frontmatter["updated"] >= record["time"]
    # A fresh ``updated`` does not make the page from before provenance look verified now.
    assert memory.read("pg-a").frontmatter["verified"] == "2026-03-01T00:00:00Z"
    assert memory.read("pg-b").frontmatter["verified"] == originals["pg-b"].frontmatter["verified"]
    assert "restored" in consolidate.merges(memory)[0]
    # The job does not merge again what the user took apart.
    (again,) = consolidate.families(index)
    assert again.skip == "a merge of these pages was undone"
    # A second restore finds every page in place and touches none.
    assert consolidate.restore(memory, merge.id) == (
        [], dict.fromkeys(["pg-a", "pg-b", "pg-c"], UNTOUCHED))
    for bad in ("no-such-merge", "../" + merge.id, ""):
        with pytest.raises(ValueError, match="no merge"):
            consolidate.restore(memory, bad)


def test_a_merge_never_destroys_what_another_task_wrote_meanwhile(tmp_path: Path) -> None:
    tools = make_tools(tmp_path / "memory")
    memory, index = tools.memory, tools.index
    pg_family(tools)
    tools.memory_write("pg-d", PG + "Owner is the platform team since April.",
                       title="Staging Postgres")
    (family,) = consolidate.families(index)
    assert family.target.name == "pg-d" and len(family.pages) == 4

    # The newest member was rewritten while the merged page was being made: nothing happens.
    tools.memory_write("pg-d", PG + "Owner is the platform team since May.")
    with pytest.raises(PageChangedError):
        consolidate.commit(memory, family, "merged", "t", "s")
    assert memory.page_names() == ["pg-a", "pg-b", "pg-c", "pg-d"]
    assert consolidate.merges(memory) == []
    assert not list((tmp_path / "memory" / ".archive").iterdir())

    # Other members rewritten or deleted meanwhile are left as they are.
    (family,) = consolidate.families(index)
    tools.memory_write("pg-a", PG + "Owner is the platform team. Pager rota: weekly.")
    tools.memory_delete("pg-b")
    merge = consolidate.commit(memory, family, "merged", "", "")
    assert (merge.deleted, merge.kept) == (["pg-c"], ["pg-a", "pg-b"])
    assert memory.page_names() == ["pg-a", "pg-d"]
    assert "Pager rota: weekly." in memory.read("pg-a").body
    # A merge that names no title keeps the page's; without a task id it stores none.
    assert merge.page.title == "Staging Postgres" and "task" not in memory.read("pg-d").frontmatter

    # Restore brings back what the merge deleted and gives way to what was written since:
    # a line added to the merged page by hand, which leaves its frontmatter as the merge
    # wrote it, is in no archive.
    with memory.page_path("pg-d").open("a", encoding="utf-8", newline="\n") as file:
        file.write("Added by hand: the pager rota is weekly.\n")
    assert memory.read("pg-d").frontmatter["writer"] == "consolidation"
    assert consolidate.restore(memory, merge.id) == (["pg-c"], {"pg-d": UNTOUCHED})
    assert memory.page_names() == ["pg-a", "pg-c", "pg-d"]  # pg-b: deleted by the other task
    assert memory.read("pg-d").body == "merged\nAdded by hand: the pager rota is weekly.\n"


def test_a_merge_stopped_halfway_can_still_be_undone(tmp_path: Path) -> None:
    tools = make_tools(tmp_path / "memory")
    memory = tools.memory
    pg_family(tools)
    (family,) = consolidate.families(tools.index)
    previous = sys.getprofile()
    sys.setprofile(_StopsAtDelete())
    try:
        with pytest.raises(KeyboardInterrupt):
            consolidate.commit(memory, family, "merged", "", "")
    finally:
        sys.setprofile(previous)
    # The merged page is written, no member is deleted, and the archive is complete.
    assert memory.page_names() == ["pg-a", "pg-b", "pg-c"]
    assert memory.read("pg-c").body == "merged\n"
    (record,) = consolidate.merges(memory)
    assert "deleted" not in record
    assert consolidate.restore(memory, record["id"]) == (
        ["pg-c"], dict.fromkeys(["pg-a", "pg-b"], UNTOUCHED))
    assert memory.read("pg-c").body == family.target.body


def test_restore_reports_a_page_it_cannot_write_back_and_merged_pages_claim_no_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory"
    jobs = make_tools(root, writer="quiz")
    memory = jobs.memory
    # Two pages only a job has written (no evidence), one of them with an empty body.
    jobs.memory_write("note-a", "Shared note about the release train.", title="Release train")
    jobs.memory_write("note-b", "Shared note about the release train.", title="Release train")
    path = memory.page_path("note-a")
    path.write_text(path.read_text().replace("Shared note about the release train.\n", ""))
    (family,) = consolidate.families(jobs.index, threshold=0.5)
    assert [page.name for page in family.pages] == ["note-a", "note-b"]
    assert consolidate.merged_provenance(family) == {
        "task": "", "writer": "consolidation", "verified": "", "sources": "", "review_after": ""}

    merge = consolidate.commit(memory, family, "One note about the release train.", "", "")
    assert "verified" not in memory.read("note-b").frontmatter
    assert consolidate.restore(memory, merge.id) == (
        ["note-b"], {"note-a": "Refusing to write an empty page."})
    assert memory.read("note-b").frontmatter["writer"] == "quiz"
    assert "verified" not in memory.read("note-b").frontmatter
    assert "restored" in json.loads((root / ".archive" / merge.id / "merge.json").read_text())

    # The archive is never written through a link out of the memory.
    (root / ".archive").rename(root / "elsewhere")
    (root / ".archive").symlink_to(root / "elsewhere")
    with pytest.raises(ValueError, match="archive directory must not be"):
        consolidate.merges(memory)
