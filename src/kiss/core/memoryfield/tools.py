# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Agent tools over one or more memoryfields: search, pull, read, write, list, delete.

Pass the bound methods of :class:`MemoryTools` to ``KISSAgent.run(tools=...)``
and append :meth:`MemoryTools.protocol` to the system prompt. Every search
first runs an incremental index sync, so pages written by any process (the
agent, a human in an editor, a sync script) are searchable without a
separate reindex.  A write indexes the page it wrote at once and names the
pages that nearly duplicate it, and stamps the page with its provenance (the
writing task, the kind of writer, when an agent last verified it).

A :class:`MemoryTools` serves a *general* memory (its ``root`` directory)
plus any number of *domain* memories, each a sub-directory ``root/<name>``
with its own pages and index -- for example the memory of the repository the
agent works in.  Page names carry the memory: ``<page>`` is a page of the
general memory, ``<memory>/<page>`` a page of that domain memory.
"""

import hashlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from kiss.core import config as config_module
from kiss.core import tool_interrupt
from kiss.core.memoryfield.index import Embedder, SearchHit, VectorIndex, embedding_text
from kiss.core.memoryfield.pages import (
    AGENT,
    MAX_PAGE_BYTES,
    MemoryDir,
    evidence_time,
    is_valid_page_name,
)
from kiss.core.memoryfield.statements import statement_text
from kiss.core.utils import read_bytes_waiting_for_writer

MEMORY_PROTOCOL = """\
## Memory
You have a persistent memory: a directory of short Markdown pages with a semantic
search index. Follow this protocol:
1. The pages most relevant to the task are already listed at the end of the
   task message ("Memory pages relevant to this task", name, date last
   verified and summary): read the ones you need with `memory_pull` or
   `memory_read`; call `memory_search` only for a sub-topic that list does
   not cover. When a page verified within the last 30 days answers a factual
   question, re-verify only the facts that change (prices, versions, dates)
   on at most three sites.
2. While working, record durable knowledge with `memory_write`: facts you had to
   look up, user preferences, project conventions, root causes of bugs, what
   worked and what did not. One topic per page, under ~8 KB, with source URLs or
   file paths so the fact can be re-verified later. Prefer updating an existing
   page over creating a near-duplicate.
3. Do not store secrets, credentials or raw transcripts. Delete pages that turn
   out to be wrong or obsolete with `memory_delete`. Per-round or per-session
   working notes (review round N findings, fix lists, progress) are not durable
   knowledge: keep them in the `PROGRESS.md` of your scratch dir, not in memory.
4. When search results look redundant or outdated, call `memory_refresh`: it
   re-indexes pages edited outside the agent and lists near-duplicate and stale
   pages. Merge duplicates with `memory_write` + `memory_delete`, and re-verify
   or delete stale pages.
"""

DOMAINS_PROTOCOL = """\
5. You have several memories. A page of the general memory is named `<page>`; a
   page of a domain memory is named `<memory>/<page>`, and `memory_search`,
   `memory_pull`, `memory_list` and `memory_refresh` cover every memory unless
   `memory=` narrows them (`general` is the general memory). Domain memories:
{domains}
   Store what you learn about a domain in its memory (for a repository: its
   code, architecture, conventions, root causes of its bugs) and keep the
   general memory for user preferences and lessons that cross domains.
"""

GENERAL = "general"
"""The ``memory=`` value that narrows a tool to the general memory."""

WRITERS = (AGENT, "consolidation", "quiz")
"""Who writes pages through a :class:`MemoryTools` (the ``writer`` frontmatter
key): an agent, which holds the evidence for what it writes, or a
maintenance job, which rewrites pages without verifying their facts."""

# memory_write names up to DUPLICATE_PAGES pages of the same memory whose
# page embedding is at least this similar to the page it wrote (the default
# threshold of memory_refresh).
DUPLICATE_THRESHOLD = 0.9
DUPLICATE_PAGES = 3
# Longest memory_write waits for the embedding of the page it wrote.
WRITE_INDEX_SECONDS = 10.0

# Characters of the matching statement memory_search prints under a hit.
STATEMENT_CHARS = 240

# Longest the sync before a search spends, per memory, on pages that are
# indexed and only lack statement vectors (all of them right after an
# upgrade): the rest waits for later searches or a memory_refresh, and the
# search ranks by page embeddings meanwhile.
BACKFILL_SECONDS = 5.0

_MARKUP_RE = re.compile(r"</?(title|parameter|summary|invoke)\b", re.IGNORECASE)
"""Tool-call markup that leaked into a title or summary."""

_EPHEMERAL_NAME_RE = re.compile(r"(^|-)(round|session|iteration|pass)-?\d+(-|$)")
"""Page names such as ``muse-auth-round4-review`` or ``server-fixes-round-7``
are per-round working notes (36 of 260 pages in the 2026-09-19 audit)."""

# Cap on characters returned by memory_pull so a broad query cannot flood the context.
PULL_CHAR_LIMIT = 24_000

# Pages listed in the block :meth:`MemoryTools.recall` appends to a task.
RECALL_PAGES = 5
# Characters of the task text embedded as the recall query; a longer
# task keeps its head and its tail (a continuation puts the actual task
# after the previous session's summary).
RECALL_QUERY_CHARS = 4_000
RECALL_HEADING = "# Memory pages relevant to this task"


def recall_query(task: str) -> str:
    """Return the part of *task* embedded as the recall query: all of it, or its head and tail."""
    if len(task) <= RECALL_QUERY_CHARS:
        return task
    half = RECALL_QUERY_CHARS // 2
    return task[:half] + "\n...\n" + task[-half:]


def _job_note(frontmatter: dict[str, Any]) -> str:
    """The mark of a page last written by a maintenance job, ``""`` for an agent's page."""
    writer = str(frontmatter.get("writer") or AGENT)
    return "" if writer == AGENT else f" [written by a {writer} job]"


def _utc_date(value: object) -> date | None:
    """The UTC day of a frontmatter date or timestamp (no zone: UTC), ``None`` for anything else.

    Anything else is a value that is no ISO 8601 date, or a time whose
    offset puts its UTC day outside the years 1 to 9999
    (``9999-12-31T23:30:00-02:00``), which the conversion cannot hold.
    """
    try:
        moment = datetime.fromisoformat(str(value or ""))
        return (moment if moment.tzinfo is None else moment.astimezone(UTC)).date()
    except (ValueError, OverflowError):
        return None


def _index_and_compare(index: VectorIndex, name: str) -> list[tuple[str, float]]:
    """Index page *name* now and return the pages of *index* that nearly duplicate it.

    Only *name* is synced, so a page deleted since the last search still
    has its row: the pages returned are the closest that still exist.
    """
    index.sync(names={name})
    existing = index.memory.page_stats()
    similar = index.similar_pages(name, DUPLICATE_THRESHOLD, k=index.count())
    return [pair for pair in similar if pair[0] in existing][:DUPLICATE_PAGES]


def _qualified(memory: str, name: str) -> str:
    """The page name an agent sees: ``name`` in the general memory, else ``memory/name``."""
    return name if memory == GENERAL else f"{memory}/{name}"


class MemoryTools:
    """Tool set giving an agent read/write access to a general memory and its domain memories.

    Args:
        root: The general memory directory (created on first write).
        embed: Embedding function; defaults to
            :func:`kiss.core.memoryfield.index.default_embedder` — the
            framework's ``text-embedding-3-small`` model when an
            ``OPENAI_API_KEY`` is available, else the fully offline
            :func:`kiss.core.memoryfield.index.hashed_embedding`.
        model_code: Overrides the embedding-model identifier used to name the
            index files.
        domains: Domain memories to attach, ``{name: description}``; each
            lives in the sub-directory ``root/<name>`` and its pages are
            addressed as ``<name>/<page>``.  A name must follow the page
            naming rules and must not be ``"general"``.
        task_id: Id of the task these tools serve, stored as ``task`` in
            every page it writes (its events are in history.db); empty
            when the caller is no task.
        writer: One of :data:`WRITERS`, stored as ``writer`` in every page
            written.  Only an ``"agent"`` write sets the page's
            ``verified`` time (see :meth:`MemoryDir.write`).

    Raises:
        ValueError: On an invalid domain name or writer.
    """

    def __init__(
        self,
        root: str | Path,
        embed: Embedder | None = None,
        model_code: str | None = None,
        domains: dict[str, str] | None = None,
        task_id: str = "",
        writer: str = AGENT,
    ) -> None:
        if writer not in WRITERS:
            raise ValueError(f"Invalid writer {writer!r}: use one of {', '.join(WRITERS)}.")
        self.task_id = task_id
        self.writer = writer
        self.memory = MemoryDir(root)
        self.index = VectorIndex(self.memory, embed=embed, model_code=model_code)
        self.domains: dict[str, str] = dict(domains or {})
        self.indexes: dict[str, VectorIndex] = {GENERAL: self.index}
        for name in self.domains:
            if name == GENERAL or not is_valid_page_name(name):
                raise ValueError(
                    f"Invalid memory name {name!r}: use lowercase letters, digits and "
                    f"hyphens, starting and ending with a letter or digit, not {GENERAL!r}."
                )
            self.indexes[name] = VectorIndex(
                MemoryDir(self.memory.root / name),
                embed=self.index.embed,
                model_code=self.index.model_code,
            )

    def tools(self) -> list:
        """The callables to register with ``KISSAgent.run(tools=...)``."""
        return [
            self.memory_search,
            self.memory_pull,
            self.memory_read,
            self.memory_write,
            self.memory_list,
            self.memory_delete,
            self.memory_refresh,
        ]

    def protocol(self) -> str:
        """The system-prompt block governing these tools.

        :data:`MEMORY_PROTOCOL`, followed by a description of the attached
        domain memories and how their pages are named when there are any.
        """
        if not self.domains:
            return MEMORY_PROTOCOL
        lines = "\n".join(
            f"   - `{name}/`: {description}" for name, description in self.domains.items()
        )
        return MEMORY_PROTOCOL + DOMAINS_PROTOCOL.format(domains=lines)

    def _selected(self, memory: str) -> dict[str, VectorIndex]:
        """The indexes a ``memory=`` argument selects: all when empty, else that one.

        Raises:
            ValueError: If *memory* names no attached memory.
        """
        if not memory:
            return self.indexes
        if memory not in self.indexes:
            raise ValueError(f"unknown memory {memory!r}; memories: {', '.join(self.indexes)}.")
        return {memory: self.indexes[memory]}

    def _locate(self, name: str) -> tuple[str, MemoryDir, str]:
        """Split a page name into ``(memory, its MemoryDir, bare page name)``.

        Raises:
            ValueError: If the name's memory prefix names no attached memory.
        """
        memory, sep, page = name.partition("/")
        if not sep or not is_valid_page_name(memory):
            # No prefix, or a prefix (such as "..") that cannot name a memory:
            # the general MemoryDir validates the whole name.
            return GENERAL, self.memory, name
        if memory not in self.indexes:
            raise ValueError(
                f"unknown memory {memory!r} in page name {name!r}; "
                f"memories: {', '.join(self.indexes)}."
            )
        return memory, self.indexes[memory].memory, page

    @staticmethod
    def _search_one(
        index: VectorIndex, query: str, k: int, cancel: threading.Event
    ) -> list[SearchHit]:
        """Sync one index and search it, embedding the query during the sync.

        The query's embedding request runs on a worker thread while the
        sync scans the directory (and embeds any changed pages), so the
        call costs about one embedding round trip instead of two.
        Once *cancel* is set the sync starts no further embedding batch
        and the search is skipped: nobody reads an abandoned worker's
        hits (see :meth:`search`), and the search's lazy numpy import
        would otherwise run on this thread, in the background, long
        after the task thread has moved on.
        """
        if k <= 0:
            index.sync(cancel=cancel, backfill_seconds=BACKFILL_SECONDS)
            return []
        with ThreadPoolExecutor(max_workers=1) as pool:
            query_vector = pool.submit(index.embed_query, query)
            index.sync(cancel=cancel, backfill_seconds=BACKFILL_SECONDS)
            query_vector.result()  # re-raises an embedding failure
        if cancel.is_set():
            return []
        return index.search(query, k=k)

    def search(self, query: str, k: int, memory: str = "") -> list[SearchHit]:
        """Search the selected memories concurrently and merge the best *k* hits.

        What ``memory_search`` prints and ``memory_pull`` reads, as hits:
        for a caller that works with the pages found (the self-quiz job
        reads them to a judge).

        Each memory's sync-and-search runs on its own worker thread; the
        task thread waits in short slices
        (:func:`~kiss.core.tool_interrupt.result_interruptible`) and does not
        join the pool, so a Stop injected into it propagates at once.  An
        abandoned worker finishes only its current embedding batch in the
        background (safe: the index writes each batch in its own
        transaction) and then stops, so process exit, which joins pool
        threads, waits for at most that batch.

        Hit names come back qualified with the memory they were found in.
        A memory whose statement vectors are still being written ranks by
        page embedding (see :meth:`VectorIndex.search`), and those scores
        are lower than scores made with a statement: unless the memories
        agree, all of them are searched again by page embedding.

        Args:
            query: What to look for, as a question or topic.
            k: Maximum number of hits.
            memory: Search only this memory; empty searches every memory.

        Returns:
            The hits, best first.

        Raises:
            ValueError: If *memory* names no attached memory.
        """
        selected = self._selected(memory)
        cancel = threading.Event()
        pool = ThreadPoolExecutor(max_workers=len(selected))
        try:
            # ``submit`` starts a thread and takes the pool's semaphore:
            # a Stop raised as either takes its lock would strand a worker.
            futures = [
                tool_interrupt.call_without_injection(
                    pool.submit, self._search_one, index, query, k, cancel,
                )
                for index in selected.values()
            ]
            found = [tool_interrupt.result_interruptible(future) for future in futures]
        finally:
            cancel.set()
            pool.shutdown(wait=False)
        if len({memory_hits[0].by_page for memory_hits in found if memory_hits}) > 1:
            found = [index.search(query, k=k, by_page=True) for index in selected.values()]
        hits = [
            replace(hit, name=_qualified(name, hit.name))
            for name, memory_hits in zip(selected, found, strict=True)
            for hit in memory_hits
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.name))
        return hits[:k]

    def _format_hits(self, hits: list[SearchHit]) -> str:
        """One line per hit (score, name, title, summary), then the statement that matched."""
        lines = []
        for hit in hits:
            line = f"{hit.score:.3f}  {hit.name}  —  {hit.title}"
            if hit.summary:
                line += f": {hit.summary}"
            lines.append(line + _job_note(hit.frontmatter))
            statement = self._statement(hit)
            if statement:
                lines.append(f"       > {statement}")
        return "\n".join(lines)

    def _statement(self, hit: SearchHit) -> str:
        """The statement of *hit*'s page that matched the query, on one line, or ``""``.

        ``""`` too for a page rewritten since it was indexed: the hit's
        position is one in the indexed text, and another sentence, or
        none, in the page as it is now.
        """
        if hit.statement is None:
            return ""
        _, memory_dir, page = self._locate(hit.name)
        try:
            data = read_bytes_waiting_for_writer(memory_dir.page_path(page))
        except FileNotFoundError:
            return ""  # deleted by another process since the search
        if hashlib.sha256(data).digest() != hit.digest:
            return ""  # rewritten by another process since it was indexed
        raw = data.decode("utf-8", errors="replace")
        text = " ".join(statement_text(raw, hit.statement).split())
        return text if len(text) <= STATEMENT_CHARS else text[: STATEMENT_CHARS - 3] + "..."

    def _no_hits(self, memory: str) -> str:
        return (
            "No memory pages yet."
            if all(index.count() == 0 for index in self._selected(memory).values())
            else "No matches."
        )

    def recall(self, task: str, k: int = RECALL_PAGES) -> str:
        """Return the block of pages relevant to *task* that is appended to the task message.

        The search the memory protocol used to ask the model to make
        as its first turn, made here instead: the full task text is the
        query (a paraphrase missed the relevant page), and the result is
        one line per page — name, the date an agent last verified it (its
        last update for a page without one) and summary — so the
        model reads a body only when it wants one.

        Args:
            task: The task text (clipped to :data:`RECALL_QUERY_CHARS`).
            k: Maximum pages listed.

        Returns:
            The Markdown block headed :data:`RECALL_HEADING`, or ``""``
            when nothing matches or the search fails (an embedding
            outage must not block the task).
        """
        try:
            hits = self.search(recall_query(task), k)
        except Exception:
            return ""
        if not hits:
            return ""
        lines = [RECALL_HEADING]
        for hit in hits:
            frontmatter = hit.frontmatter
            verified = evidence_time(frontmatter)[1][:10]
            line = f"- `{hit.name}`"
            if verified:
                line += f" ({verified})"
            line += f": {hit.title}"
            if hit.summary:
                line += f" — {hit.summary}"
            lines.append(line + _job_note(frontmatter))
        lines.append("Read a page with memory_read(name) or several with memory_pull(query).")
        return "\n".join(lines)

    def memory_search(self, query: str, k: int = 5, memory: str = "") -> str:
        """Semantic search over memory pages; returns the best-matching page names with scores.

        Args:
            query: What you want to recall, phrased as a question or topic.
            k: Maximum number of pages to return (default 5).
            memory: Search only this memory ('general' or a domain memory's
                name); empty (default) searches every memory.
        """
        try:
            hits = self.search(query, k, memory)
        except ValueError as e:
            return f"Error: {e}"
        if not hits:
            return self._no_hits(memory)
        return self._format_hits(hits)

    def memory_pull(self, query: str, k: int = 3, memory: str = "") -> str:
        """Semantic search that returns the full contents of the matching pages in one call.

        Args:
            query: What you want to recall, phrased as a question or topic.
            k: Maximum number of pages to return (default 3).
            memory: Search only this memory ('general' or a domain memory's
                name); empty (default) searches every memory.
        """
        try:
            hits = self.search(query, k, memory)
        except ValueError as e:
            return f"Error: {e}"
        if not hits:
            return self._no_hits(memory)
        chunks: list[str] = []
        used = 0
        vanished = 0
        for hit in hits:
            _, memory_dir, page = self._locate(hit.name)
            try:
                raw = memory_dir.read(page).raw
            except FileNotFoundError:
                # Another process deleted the page after sync() indexed it.
                vanished += 1
                continue
            chunk = f"### {hit.name}.md  (score {hit.score:.3f})\n{raw}"
            if used + len(chunk) > PULL_CHAR_LIMIT:
                if not chunks:
                    chunk = (
                        chunk[:PULL_CHAR_LIMIT] + "\n[page truncated; use memory_read for the rest]"
                    )
                    chunks.append(chunk)
                omitted = len(hits) - vanished - len(chunks)
                if omitted:
                    chunks.append(f"[{omitted} more page(s) omitted; lower k or refine the query]")
                break
            chunks.append(chunk)
            used += len(chunk)
        if not chunks:
            return "No matches."
        return "\n\n".join(chunks)

    def memory_read(self, name: str) -> str:
        """Return the full text of one memory page.

        Args:
            name: Page name as shown by memory_search or memory_list (with or
                without .md); '<memory>/<page>' for a page of a domain memory.
        """
        try:
            _, memory_dir, page = self._locate(name)
            return memory_dir.read(page).raw
        except FileNotFoundError:
            return f"Error: no memory page named {name!r}."
        except ValueError as e:
            return f"Error: {e}"

    def memory_write(
        self,
        name: str,
        content: str,
        title: str = "",
        summary: str = "",
        sources: str = "",
        review_after: str = "",
    ) -> str:
        """Create or replace a memory page (Markdown body; frontmatter is generated for you).

        Args:
            name: Page name: lowercase letters, digits and hyphens, e.g.
                'postgres-agent-auth-flow'; prefix it with '<memory>/' to write
                into a domain memory.
            content: Markdown body. Keep it under ~8 KB; one topic per page; cite sources.
            title: Human-readable title (defaults to the name on first write).
            summary: One-sentence summary shown in search results.
            sources: Where the page's facts come from (URLs, file paths),
                comma-separated; empty keeps the stored value.
            review_after: Date (YYYY-MM-DD) after which the facts need
                re-verifying, for facts known to expire (prices, versions,
                deadlines); empty keeps the stored value.
        """
        # Who writes; the page's ``verified`` time follows from it in the write.
        extra = {"task": self.task_id, "writer": self.writer}
        if sources:
            extra["sources"] = sources
        try:
            if review_after:
                extra["review_after"] = date.fromisoformat(review_after).isoformat()
        except ValueError:
            return f"Error: review_after must be a date (YYYY-MM-DD), not {review_after!r}."
        try:
            memory, memory_dir, page_name = self._locate(name)
            page = memory_dir.write(page_name, content, title=title, summary=summary, extra=extra)
        except (ValueError, TimeoutError) as e:
            return f"Error: {e}"
        size = len(page.raw.encode("utf-8"))
        embedded = len(embedding_text(page.raw).encode("utf-8"))
        note = ""
        if _EPHEMERAL_NAME_RE.search(page.name):
            note = (
                " Warning: the name looks like a per-round/per-session note; such "
                "notes belong in the PROGRESS.md of the scratch dir, not in durable "
                "memory. Consider memory_delete once the task is over."
            )
        if embedded > MAX_PAGE_BYTES:
            note += (
                f" Warning: the page's searchable text (title, summary and body) is "
                f"{embedded} bytes; only the first {MAX_PAGE_BYTES} bytes are embedded. "
                "Split it into several pages."
            )
        if _MARKUP_RE.search(f"{page.title}\n{page.summary}"):
            note += (
                " Warning: the title or summary contains tool-call markup (such as "
                "</summary>); write the page again with a clean title and summary."
            )
        similar = self._index_page(memory, page.name)
        if similar:
            listed = ", ".join(
                f"{_qualified(memory, other)} ({score:.2f})" for other, score in similar
            )
            note += (
                f" Similar pages: {listed}. If this page replaces or extends one of them, "
                "write to that name or memory_delete it; pages about different dates or "
                "different occurrences of the same thing stay separate."
            )
        return f"Wrote {_qualified(memory, page.name)}.md ({size} bytes).{note}"

    def _index_page(self, memory: str, name: str) -> list[tuple[str, float]]:
        """Index the page just written and return the pages of its memory that nearly duplicate it.

        The page's row is stored at once, so the next search has nothing
        to embed for it.  The embedding request runs on a worker thread
        and is given :data:`WRITE_INDEX_SECONDS`: a failed or slow request
        costs the notice, never the write (the next search then indexes
        the page).

        Returns:
            Up to :data:`DUPLICATE_PAGES` ``(page name, similarity)`` pairs
            at :data:`DUPLICATE_THRESHOLD` or above, best first.
        """
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = tool_interrupt.call_without_injection(
                pool.submit, _index_and_compare, self.indexes[memory], name,
            )
            return tool_interrupt.result_interruptible(future, WRITE_INDEX_SECONDS)
        except Exception:
            return []
        finally:
            pool.shutdown(wait=False)

    def memory_list(self, memory: str = "", max_output_chars: int = 0) -> str:
        """List every memory page with its title and summary.

        Args:
            memory: List only this memory ('general' or a domain memory's
                name); empty (default) lists every memory.
            max_output_chars: Characters of listing kept (0 = the
                configured default budget, 8,000); a longer listing is
                cut and says how many pages it holds.  Capped at the
                configured ceiling.
        """
        try:
            selected = self._selected(memory)
        except ValueError as e:
            return f"Error: {e}"
        config = config_module.DEFAULT_CONFIG
        budget = max_output_chars or config.tool_output_default_chars
        budget = max(1, min(int(budget), config.tool_output_max_chars))
        lines = []
        for memory_name, index in selected.items():
            for name in index.memory.page_names():
                try:
                    page = index.memory.read(name)
                except FileNotFoundError:
                    # Deleted by another process after page_names() listed it.
                    continue
                line = f"{_qualified(memory_name, name)}  —  {page.title}"
                if page.summary:
                    line += f": {page.summary}"
                # One physical line per page: the cut below counts pages
                # by newlines, and a title or summary may contain some.
                lines.append(" ".join(line.split("\n")))
        if not lines:
            return "No memory pages yet."
        listing = "\n".join(lines)
        if len(listing) <= budget:
            return listing
        kept = listing[:budget].rsplit("\n", 1)[0]
        shown = kept.count("\n") + 1
        return (
            f"{kept}\n... [{len(lines) - shown} more pages not shown: pass memory=<name> "
            f"to list one memory, memory_search(query) to find a page, or "
            f"max_output_chars up to {config.tool_output_max_chars}]"
        )

    def memory_refresh(
        self, stale_days: int = 30, duplicate_threshold: float = 0.9, memory: str = ""
    ) -> str:
        """Re-index the memory and report pages that need maintenance.

        Re-embeds pages changed on disk by any process (the agent, a human in
        an editor, a sync script), drops index rows for deleted pages, and
        lists pages past their ``review_after`` date, near-duplicate page
        pairs and pages not verified (or, without a ``verified`` time,
        updated) for long, so they can be merged, re-verified, or deleted.

        Args:
            stale_days: Pages not verified or updated in this many days are
                listed as stale (default 30).
            duplicate_threshold: Cosine similarity at or above which two pages
                are reported as near-duplicates (default 0.9).
            memory: Refresh only this memory ('general' or a domain memory's
                name); empty (default) refreshes every memory.
        """
        try:
            selected = self._selected(memory)
        except ValueError as e:
            return f"Error: {e}"
        lines = []
        for memory_name, index in selected.items():
            report = index.sync(verify=True)
            label = (
                "Index"
                if memory_name == GENERAL and len(selected) == 1
                else (f"Index of {memory_name}")
            )
            lines.append(
                f"{label} refreshed: {report.added} added, {report.updated} updated, "
                f"{report.removed} removed, {report.unchanged} unchanged."
            )
            due = self._review_due(index)
            if due:
                lines.append(
                    "Pages past their review_after date (re-verify and rewrite, or delete):"
                )
                lines.extend(
                    f"  {_qualified(memory_name, name)}  (review after {review_after})"
                    for name, review_after in due[:20]
                )
            duplicates = index.near_duplicates(duplicate_threshold)
            if duplicates:
                lines.append(
                    "Near-duplicate pages (merge with memory_write, then memory_delete the loser):"
                )
                lines.extend(
                    f"  {_qualified(memory_name, a)} ~ {_qualified(memory_name, b)}  "
                    f"(similarity {score:.3f})"
                    for a, b, score in duplicates[:20]
                )
            stale = self._stale_pages(index, stale_days)
            if stale:
                lines.append(
                    f"Pages not verified or updated in {stale_days} days (re-verify or delete):"
                )
                lines.extend(
                    f"  {_qualified(memory_name, name)}  ({key} {value})"
                    for name, key, value in stale[:20]
                )
            if not due and not duplicates and not stale:
                lines.append("No near-duplicate or stale pages.")
        return "\n".join(lines)

    @staticmethod
    def _review_due(index: VectorIndex) -> list[tuple[str, str]]:
        """Return ``(name, review_after)`` of the pages due for review, earliest date first.

        A page is due once its ``review_after`` date has passed, unless an
        agent verified it after that date (both taken as UTC days).  A
        ``review_after`` that is no date is skipped, like an unparseable
        time in :meth:`_stale_pages`.
        """
        today = datetime.now(UTC).date()
        due: list[tuple[str, str]] = []
        for name, frontmatter in index.frontmatters().items():
            review_after = _utc_date(frontmatter.get("review_after"))
            if review_after is None or review_after >= today:
                continue
            verified = _utc_date(frontmatter.get("verified"))
            if verified is None or verified <= review_after:
                due.append((name, review_after.isoformat()))
        due.sort(key=lambda entry: (entry[1], entry[0]))
        return due

    @staticmethod
    def _stale_pages(index: VectorIndex, stale_days: int) -> list[tuple[str, str, str]]:
        """Return ``(name, key, time)`` for pages older than *stale_days* days, oldest first.

        A page's age runs from its evidence time
        (:func:`~kiss.core.memoryfield.pages.evidence_time`):
        ``verified``, when an agent that held the evidence last wrote it,
        or ``updated`` on a page from before provenance; *key* names the
        one used.  A maintenance job's rewrite refreshes ``updated`` only
        and keeps the evidence time as ``verified``, so it does not make a
        page look verified.

        Reads the frontmatter the just-finished sync stored in *index*
        rather than every page file again.  The time is the spec's
        quoted ``YYYY-MM-DDTHH:MM:SSZ`` or, on a hand-written page, an
        unquoted YAML timestamp (stored by the index as ``str(datetime)``,
        ``2000-01-01 00:00:00+00:00``); both are ISO 8601, a value without
        a zone is taken as UTC.  Pages whose time is
        missing or unparseable are skipped: staleness cannot be
        established for them, and reporting them would train the agent to
        "fix" pages that may be fresh.
        """
        cutoff = datetime.now(UTC) - timedelta(days=stale_days)
        stale: list[tuple[datetime, str, str, str]] = []
        for name, frontmatter in index.frontmatters().items():
            key, raw_time = evidence_time(frontmatter)
            try:
                moment = datetime.fromisoformat(raw_time)
            except ValueError:
                continue
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=UTC)
            if moment < cutoff:
                stale.append((moment, name, key, raw_time))
        # By the instant, then the name: the text of a time does not order
        # times written with different offsets.
        stale.sort()
        return [entry[1:] for entry in stale]

    def memory_delete(self, name: str) -> str:
        """Delete a memory page that is wrong or obsolete.

        Args:
            name: Page name (with or without .md); '<memory>/<page>' for a
                page of a domain memory.
        """
        try:
            _, memory_dir, page = self._locate(name)
            memory_dir.delete(page)
        except FileNotFoundError:
            return f"Error: no memory page named {name!r}."
        except (ValueError, TimeoutError) as e:
            return f"Error: {e}"
        return f"Deleted {name}."
