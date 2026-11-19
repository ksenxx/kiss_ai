# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Merging near-duplicate pages: which pages, and how a merge is committed and undone.

The part of the consolidation job that needs no language model (the job
itself, :mod:`kiss.agents.seas.memory_consolidate.memory_consolidate_sea`,
has a model write the merged page and checks it against the originals).

Pages whose embeddings are near-duplicates
(:meth:`~kiss.core.memoryfield.index.VectorIndex.near_duplicates`) form a
*family*: a connected group of such pairs.  A family is merged only when
its pages fit in one page as they are (:data:`FIT_BYTES`), so that a merge
drops repetition and never has to shorten anything; only when none of its
pages is itself a merged page, since rewriting a rewrite erodes it; only
when all of them are UTF-8, since no other page can be archived as it
is; and never again once a merge of its pages was undone.  A family the job
examined and did not merge is recorded (:func:`leave_alone`) and not
examined again until one of its pages changes, so a scheduled job works
its way through a memory instead of asking about the same pages each run.

A merge (:func:`commit`) first copies every member, byte for byte, into
``<memory>/.archive/<time>-<page>/``, a directory no page listing, index
or machine sync reads; then writes the merged page under the newest
member's name and records what it wrote; then deletes the other members,
which leaves tombstones.  Each of the two steps happens only while its
page is still the text the job read, so a page another task rewrote
during the merge is neither overwritten nor deleted.  :func:`restore`
writes the archived pages back through
:meth:`~kiss.core.memoryfield.pages.MemoryDir.write`: a file copied back
by hand keeps its old ``updated`` time, which the tombstone that already
travelled to another machine outlives at the next sync.  (A sync treats
a page and a deletion of the same moment, within the difference it
measures between the two clocks, as a conflict it names and leaves to a
person: so it does a restore made that soon after its merge.)
"""

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kiss.core.memoryfield.index import VectorIndex
from kiss.core.memoryfield.pages import (
    MAX_PAGE_BYTES,
    PROVENANCE_KEYS,
    MemoryDir,
    Page,
    PageChangedError,
    evidence_time,
    read_page_text,
    split_frontmatter,
)
from kiss.core.utils import atomic_write_text

# The ``writer`` of a merged page (see ``MemoryDir.write``).
CONSOLIDATION = "consolidation"

# Sub-directory of a memory that holds the pages merges replaced; the
# leading dot keeps it out of page names and domain-memory names.
ARCHIVE_DIR = ".archive"
RECORD_FILE = "merge.json"
# In the archive: the families examined and not merged, with the reason.
LEFT_ALONE_FILE = "left-alone.json"

# Most bytes the bodies of a family may hold together to be merged: one
# page, less room for its title and summary.
FIT_BYTES = MAX_PAGE_BYTES - 700


def _moment(value: object) -> datetime | None:
    """The time a frontmatter timestamp names (no zone: UTC), ``None`` for anything else."""
    try:
        moment = datetime.fromisoformat(str(value or ""))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _age_key(page: Page) -> tuple[datetime, str]:
    """Sort key that puts the page updated last at the end (no ``updated``: oldest)."""
    return _moment(page.frontmatter.get("updated")) or datetime.min.replace(tzinfo=UTC), page.name


@dataclass(frozen=True)
class Family:
    """Near-duplicate pages of one memory that a merge would turn into one page.

    Attributes:
        pages: The members as read from disk, oldest first by ``updated``.
        score: The highest similarity between two of them.
        skip: Why the job leaves the family as it is; empty for a family
            it may merge.
    """

    pages: tuple[Page, ...]
    score: float
    skip: str = ""

    @property
    def target(self) -> Page:
        """The newest member: the merged page takes its name."""
        return self.pages[-1]


def _family_key(pages: Iterable[Page]) -> str:
    """Identify a family by its pages and their text: once one changes it is another family."""
    texts = sorted(f"{page.name}\0{page.raw}" for page in pages)
    return hashlib.sha256("\0\0".join(texts).encode("utf-8")).hexdigest()


def _left_alone(memory: MemoryDir) -> dict[str, str]:
    """Return the families of *memory* recorded as examined and not merged: key to reason."""
    file = _archive_dir(memory) / LEFT_ALONE_FILE
    if not file.is_file():
        return {}
    recorded: dict[str, str] = json.loads(file.read_text(encoding="utf-8"))
    return recorded


def _skip_reason(pages: list[Page], undone: list[set[str]], left: dict[str, str]) -> str:
    """Why *pages* are not merged.

    Args:
        pages: The pages of a family.
        undone: The members of each merge that was restored.
        left: The families examined and not merged (:func:`_left_alone`).
    """
    merged = [page.name for page in pages if page.frontmatter.get("writer") == CONSOLIDATION]
    if merged:
        return f"holds a page a merge wrote ({', '.join(merged)}), which is not merged again"
    # Read with a replacement character where its bytes are no text.
    damaged = [page.name for page in pages if "\ufffd" in page.raw]
    if damaged:
        return f"holds a page that is not UTF-8 ({', '.join(damaged)})"
    names = {page.name for page in pages}
    if any(len(names & members) > 1 for members in undone):
        return "a merge of these pages was undone"
    size = sum(len(page.body.encode("utf-8")) for page in pages)
    if size > FIT_BYTES:
        return f"does not fit in one page ({size} bytes of {FIT_BYTES})"
    reason = left.get(_family_key(pages))
    return f"left alone before: {reason}" if reason else ""


def _read_existing(memory: MemoryDir, names: list[str]) -> list[Page]:
    """Read the pages *names* that still exist (one may be deleted after the index listed it)."""
    pages = []
    for name in names:
        try:
            pages.append(memory.read(name))
        except FileNotFoundError:
            continue
    return pages


def families(index: VectorIndex, threshold: float = 0.9) -> list[Family]:
    """Return the near-duplicate families of *index*'s memory, the most similar first.

    Syncs the index, then groups the pairs at or above *threshold* into
    connected groups: a page belongs to the family of every page it
    nearly duplicates.  A family is marked as one to leave alone
    (:attr:`Family.skip`) when it holds a page a merge wrote or a page
    that is not UTF-8, when a merge of two of its pages was undone
    (:func:`restore`), when its pages do
    not fit in one page, or when it was examined before as it is now and
    not merged (:func:`leave_alone`).

    Args:
        index: The index of the memory.
        threshold: Cosine similarity at or above which two pages are
            near-duplicates (the default of ``memory_refresh``).
    """
    index.sync()
    pairs = index.near_duplicates(threshold)
    undone = [set(record["members"]) for record in merges(index.memory) if "restored" in record]
    left = _left_alone(index.memory)
    neighbours: dict[str, set[str]] = {}
    for a, b, _ in pairs:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    found: list[Family] = []
    seen: set[str] = set()
    # Pairs come best first, so the first pair of a group carries its best score.
    for start, _, score in pairs:
        if start in seen:
            continue
        seen.add(start)
        group, stack = [], [start]
        while stack:
            name = stack.pop()
            group.append(name)
            reached = neighbours[name] - seen
            seen |= reached
            stack.extend(sorted(reached))
        pages = sorted(_read_existing(index.memory, group), key=_age_key)
        if len(pages) > 1:
            found.append(Family(tuple(pages), score, _skip_reason(pages, undone, left)))
    return found


def leave_alone(memory: MemoryDir, family: Family, reason: str) -> None:
    """Record that *family* was examined and not merged, so the job does not examine it again.

    The record holds while the family's pages stay as they are: with a
    page changed, added or gone it is another family.

    Args:
        memory: The memory that holds the family.
        family: The family, as :func:`families` read it.
        reason: Why it was not merged, in a few words.
    """
    recorded = _left_alone(memory) | {_family_key(family.pages): reason}
    file = _archive_dir(memory) / LEFT_ALONE_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(file, json.dumps(recorded, indent=2), create_mode=0o666)


def merged_provenance(family: Family, task_id: str = "") -> dict[str, str]:
    """Return the provenance of the page that replaces *family*.

    Nobody verifies anything in a merge, so the page is as verified as its
    least verified member: ``verified`` is the oldest evidence time
    (:func:`~kiss.core.memoryfield.pages.evidence_time`) of the members,
    and empty when one of them has none.  ``sources`` are those of all
    members, ``review_after`` is the earliest of theirs.

    Args:
        family: The pages being merged.
        task_id: The task that runs the merge, stored as ``task``.
    """
    times = [evidence_time(page.frontmatter)[1] for page in family.pages]
    dated = [(moment, time) for time in times if (moment := _moment(time)) is not None]
    verified = min(dated)[1] if len(dated) == len(times) else ""
    sources: list[str] = []
    for page in family.pages:
        for source in str(page.frontmatter.get("sources") or "").split(","):
            if source.strip() and source.strip() not in sources:
                sources.append(source.strip())
    reviews = sorted(str(p.frontmatter["review_after"]) for p in family.pages
                     if p.frontmatter.get("review_after"))
    return {
        "task": task_id,
        "writer": CONSOLIDATION,
        "verified": verified,
        "sources": ", ".join(sources),
        "review_after": reviews[0] if reviews else "",
    }


@dataclass(frozen=True)
class Merge:
    """A committed merge.

    Attributes:
        id: Name of its folder in the memory's archive; :func:`restore` takes it.
        page: The merged page as written.
        deleted: The members that were deleted.
        kept: The members left in place because another task rewrote or
            deleted them while the merge was being made.
    """

    id: str
    page: Page
    deleted: list[str]
    kept: list[str]


def _archive_dir(memory: MemoryDir) -> Path:
    """Return the archive directory of *memory*, refusing one that is a symlink."""
    directory = memory.root / ARCHIVE_DIR
    if directory.is_symlink():
        raise ValueError(f"{directory} is a symlink; the archive directory must not be.")
    return directory


def commit(
    memory: MemoryDir, family: Family, body: str, title: str, summary: str, task_id: str = ""
) -> Merge:
    """Replace the pages of *family* by one page, keeping the originals in the archive.

    Args:
        memory: The memory that holds the family.
        family: The pages to replace, as :func:`families` read them.
        body: Markdown body of the merged page.
        title: Its title.
        summary: Its one-sentence summary.
        task_id: The task that runs the merge.

    Returns:
        The merge, with the archive folder that undoes it.

    Raises:
        PageChangedError: If the newest member is no longer the text
            that was read; no page is written or deleted.
        ValueError: If *body* is empty or the archive directory is a symlink.
        TimeoutError: If the memory's lock stays held; as above.
    """
    time = datetime.now(UTC)
    folder = _archive_dir(memory) / f"{time:%Y%m%dT%H%M%SZ}-{family.target.name}"
    folder.mkdir(parents=True)
    names = [page.name for page in family.pages]
    for page in family.pages:
        atomic_write_text(folder / f"{page.name}.md", page.raw, create_mode=0o666)
    record: dict[str, Any] = {
        "target": family.target.name,
        "members": names,
        "time": f"{time:%Y-%m-%dT%H:%M:%SZ}",
        "task": task_id,
    }
    atomic_write_text(folder / RECORD_FILE, json.dumps(record, indent=2), create_mode=0o666)
    try:
        merged = memory.write(
            family.target.name,
            body,
            title=title,
            summary=summary,
            extra=merged_provenance(family, task_id),
            expected=family.target.raw,
        )
    except (PageChangedError, ValueError, TimeoutError):
        # The failures of a write that has written nothing: no merge, no
        # record of one.  After any other (a Stop landing in the write) the
        # archive stays, since the page may have been replaced.
        shutil.rmtree(folder)
        raise
    # What restore may replace: the page as long as it is this text.
    record["written"] = hashlib.sha256(merged.raw.encode("utf-8")).hexdigest()
    atomic_write_text(folder / RECORD_FILE, json.dumps(record, indent=2), create_mode=0o666)
    deleted, kept = [], []
    for page in family.pages[:-1]:
        try:
            memory.delete(page.name, expected=page.raw)
        except (FileNotFoundError, PageChangedError):
            kept.append(page.name)
        else:
            deleted.append(page.name)
    record["deleted"] = deleted
    atomic_write_text(folder / RECORD_FILE, json.dumps(record, indent=2), create_mode=0o666)
    return Merge(folder.name, merged, deleted, kept)


def merges(memory: MemoryDir) -> list[dict[str, Any]]:
    """Return the records of the merges in *memory*'s archive, newest first.

    Each record holds ``id`` (the folder :func:`restore` takes),
    ``target``, ``members``, ``time``, ``task``, ``written`` (the SHA-256
    of the merged page as written), ``deleted`` (the members the merge
    deleted; absent when the merge was interrupted before it had deleted
    them all) and, once undone, ``restored``.
    """
    records = []
    for file in sorted(_archive_dir(memory).glob(f"*/{RECORD_FILE}"), reverse=True):
        records.append({"id": file.parent.name} | json.loads(file.read_text(encoding="utf-8")))
    return records


def restore(memory: MemoryDir, merge_id: str) -> tuple[list[str], dict[str, str]]:
    """Bring back the pages the merge *merge_id* replaced.

    Each archived page is written back with its own frontmatter and a
    fresh ``updated`` time, so it outlives the tombstone of its deletion
    on every machine; its ``verified`` time is the evidence time it had,
    so a page from before provenance does not pass as verified now.
    Only the pages the merge deleted and the page it replaced come back,
    and nothing written since the merge is touched: a deleted page comes
    back only where no page of its name exists, and the merged page gives
    way to the page it replaced only while it is, byte for byte, what the
    merge wrote (a line added to it by hand is in no archive).

    Args:
        memory: The memory the merge was made in.
        merge_id: The ``id`` of a record of :func:`merges`.

    Returns:
        The names restored, and for every other member why it was left out.

    Raises:
        ValueError: If the archive holds no such merge.
    """
    folder = _archive_dir(memory) / merge_id
    if Path(merge_id).name != merge_id or not (folder / RECORD_FILE).is_file():
        raise ValueError(f"no merge {merge_id!r} in {_archive_dir(memory)}")
    record = json.loads((folder / RECORD_FILE).read_text(encoding="utf-8"))
    restored: list[str] = []
    skipped: dict[str, str] = {}
    # Members come oldest first, the target last.
    for name in [*record.get("deleted", record["members"][:-1]), record["target"]]:
        frontmatter, body = split_frontmatter(read_page_text(folder / f"{name}.md"))
        current = _read_existing(memory, [name])
        stored = current[0].raw if current else ""
        merged = name == record["target"] and (
            hashlib.sha256(stored.encode("utf-8")).hexdigest() == record.get("written")
        )
        if stored and not merged:
            skipped[name] = "a page of this name exists that the merge did not write"
            continue
        # Every key the page had, and none of the merged page's: an empty
        # value removes a key, and ``updated`` is the time of this write.
        extra = {key: "" for key in ("title", "summary", *PROVENANCE_KEYS)}
        extra |= {key: value for key, value in frontmatter.items() if key != "updated"}
        extra["verified"] = evidence_time(frontmatter)[1]
        try:
            memory.write(name, body, extra=extra, expected=stored)
        except (PageChangedError, ValueError, TimeoutError) as error:
            skipped[name] = str(error)
        else:
            restored.append(name)
    if restored:
        record["restored"] = f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}"
        atomic_write_text(folder / RECORD_FILE, json.dumps(record, indent=2), create_mode=0o666)
    return restored, skipped
