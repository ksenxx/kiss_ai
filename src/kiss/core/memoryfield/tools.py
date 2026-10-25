# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Agent tools over one or more memoryfields: search, pull, read, write, list, delete.

Pass the bound methods of :class:`MemoryTools` to ``KISSAgent.run(tools=...)``
and append :meth:`MemoryTools.protocol` to the system prompt. Every search
first runs an incremental index sync, so pages written by any process (the
agent, a human in an editor, a sync script) are searchable without a
separate reindex.

A :class:`MemoryTools` serves a *general* memory (its ``root`` directory)
plus any number of *domain* memories, each a sub-directory ``root/<name>``
with its own pages and index -- for example the memory of the repository the
agent works in.  Page names carry the memory: ``<page>`` is a page of the
general memory, ``<memory>/<page>`` a page of that domain memory.
"""

import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kiss.core.memoryfield.index import Embedder, SearchHit, VectorIndex, embedding_text
from kiss.core.memoryfield.pages import MAX_PAGE_BYTES, MemoryDir, is_valid_page_name

MEMORY_PROTOCOL = """\
## Memory
You have a persistent memory: a directory of short Markdown pages with a semantic
search index. Follow this protocol:
1. The pages most relevant to the task are already listed at the end of the
   task message ("Memory pages relevant to this task", name, date and summary):
   read the ones you need with `memory_pull` or `memory_read`; call
   `memory_search` only for a sub-topic that list does not cover. When a page
   updated within the last 30 days answers a factual question, re-verify only
   the facts that change (prices, versions, dates) on at most three sites.
2. While working, record durable knowledge with `memory_write`: facts you had to
   look up, user preferences, project conventions, root causes of bugs, what
   worked and what did not. One topic per page, under ~8 KB, with source URLs or
   file paths so the fact can be re-verified later. Prefer updating an existing
   page over creating a near-duplicate.
3. Do not store secrets, credentials or raw transcripts. Delete pages that turn
   out to be wrong or obsolete with `memory_delete`. Per-round or per-session
   working notes (review round N findings, fix lists, progress) are not durable
   knowledge: keep them in `./tmp/PROGRESS.md`, not in memory.
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

# How long the task thread blocks in one ``Future.result`` wait on a search
# worker.  The runner's Stop injects ``KeyboardInterrupt`` with
# ``PyThreadState_SetAsyncExc``, which lands only when the thread runs
# bytecode: an unbounded wait would make a search deaf to Stop for the
# whole of a first-time index build (minutes of embedding requests).
_WAIT_POLL_SECONDS = 0.1


def recall_query(task: str) -> str:
    """Return the part of *task* embedded as the recall query: all of it, or its head and tail."""
    if len(task) <= RECALL_QUERY_CHARS:
        return task
    half = RECALL_QUERY_CHARS // 2
    return task[:half] + "\n...\n" + task[-half:]


def _format_hits(hits: list[SearchHit]) -> str:
    lines = []
    for hit in hits:
        line = f"{hit.score:.3f}  {hit.name}  —  {hit.title}"
        if hit.summary:
            line += f": {hit.summary}"
        lines.append(line)
    return "\n".join(lines)


def _qualified(memory: str, name: str) -> str:
    """The page name an agent sees: ``name`` in the general memory, else ``memory/name``."""
    return name if memory == GENERAL else f"{memory}/{name}"


def _result[T](future: Future[T]) -> T:
    """``future.result()`` in :data:`_WAIT_POLL_SECONDS` slices, so a Stop can land between them.

    A ``TimeoutError`` from a slice means "keep waiting" only while the
    future is still running: a worker whose own failure is a
    ``TimeoutError`` (an embedder's socket read timing out) makes
    ``future.result()`` raise that same class instantly, so it must be
    re-raised, not polled for again.
    """
    while not future.done():
        try:
            return future.result(timeout=_WAIT_POLL_SECONDS)
        except TimeoutError:
            continue
    return future.result()


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

    Raises:
        ValueError: On an invalid domain name.
    """

    def __init__(
        self,
        root: str | Path,
        embed: Embedder | None = None,
        model_code: str | None = None,
        domains: dict[str, str] | None = None,
    ) -> None:
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
    def _search_one(memory: str, index: VectorIndex, query: str, k: int) -> list[SearchHit]:
        """Sync one index and search it, embedding the query during the sync.

        The query's embedding request runs on a worker thread while the
        sync scans the directory (and embeds any changed pages), so the
        call costs about one embedding round trip instead of two.  Hit
        names come back qualified with the memory they were found in.
        """
        if k <= 0:
            index.sync()
            return []
        with ThreadPoolExecutor(max_workers=1) as pool:
            query_vector = pool.submit(index.embed_query, query)
            index.sync()
            query_vector.result()  # re-raises an embedding failure
        return [replace(hit, name=_qualified(memory, hit.name)) for hit in index.search(query, k=k)]

    def _search(self, query: str, k: int, memory: str) -> list[SearchHit]:
        """Search the selected memories concurrently and merge the best *k* hits.

        Each memory's sync-and-search runs on its own worker thread; the
        task thread waits in short slices (:func:`_result`) and does not
        join the pool, so a Stop injected into it propagates at once while
        a worker finishes its current embedding batch in the background
        (safe: the index writes each batch in its own transaction).
        """
        selected = self._selected(memory)
        pool = ThreadPoolExecutor(max_workers=len(selected))
        try:
            futures = [
                pool.submit(self._search_one, name, index, query, k)
                for name, index in selected.items()
            ]
            hits = [hit for future in futures for hit in _result(future)]
        finally:
            pool.shutdown(wait=False)
        hits.sort(key=lambda hit: (-hit.score, hit.name))
        return hits[:k]

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
        one line per page — name, last update date and summary — so the
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
            hits = self._search(recall_query(task), k, "")
        except Exception:
            return ""
        if not hits:
            return ""
        lines = [RECALL_HEADING]
        for hit in hits:
            updated = str(hit.frontmatter.get("updated") or "")[:10]
            line = f"- `{hit.name}`"
            if updated:
                line += f" ({updated})"
            line += f": {hit.title}"
            if hit.summary:
                line += f" — {hit.summary}"
            lines.append(line)
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
            hits = self._search(query, k, memory)
        except ValueError as e:
            return f"Error: {e}"
        if not hits:
            return self._no_hits(memory)
        return _format_hits(hits)

    def memory_pull(self, query: str, k: int = 3, memory: str = "") -> str:
        """Semantic search that returns the full contents of the matching pages in one call.

        Args:
            query: What you want to recall, phrased as a question or topic.
            k: Maximum number of pages to return (default 3).
            memory: Search only this memory ('general' or a domain memory's
                name); empty (default) searches every memory.
        """
        try:
            hits = self._search(query, k, memory)
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

    def memory_write(self, name: str, content: str, title: str = "", summary: str = "") -> str:
        """Create or replace a memory page (Markdown body; frontmatter is generated for you).

        Args:
            name: Page name: lowercase letters, digits and hyphens, e.g.
                'postgres-agent-auth-flow'; prefix it with '<memory>/' to write
                into a domain memory.
            content: Markdown body. Keep it under ~8 KB; one topic per page; cite sources.
            title: Human-readable title (defaults to the name on first write).
            summary: One-sentence summary shown in search results.
        """
        try:
            memory, memory_dir, page_name = self._locate(name)
            page = memory_dir.write(page_name, content, title=title, summary=summary)
        except (ValueError, TimeoutError) as e:
            return f"Error: {e}"
        size = len(page.raw.encode("utf-8"))
        embedded = len(embedding_text(page.raw).encode("utf-8"))
        note = ""
        if _EPHEMERAL_NAME_RE.search(page.name):
            note = (
                " Warning: the name looks like a per-round/per-session note; such "
                "notes belong in ./tmp/PROGRESS.md, not in durable memory. Consider "
                "memory_delete once the task is over."
            )
        if embedded > MAX_PAGE_BYTES:
            note += (
                f" Warning: the page's searchable text (title, summary and body) is "
                f"{embedded} bytes; only the first {MAX_PAGE_BYTES} bytes are embedded. "
                "Split it into several pages."
            )
        return f"Wrote {_qualified(memory, page.name)}.md ({size} bytes).{note}"

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
        from kiss.core.config import DEFAULT_CONFIG

        try:
            selected = self._selected(memory)
        except ValueError as e:
            return f"Error: {e}"
        budget = max_output_chars or DEFAULT_CONFIG.tool_output_default_chars
        budget = max(1, min(int(budget), DEFAULT_CONFIG.tool_output_max_chars))
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
                lines.append(line)
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
            f"max_output_chars up to {DEFAULT_CONFIG.tool_output_max_chars}]"
        )

    def memory_refresh(
        self, stale_days: int = 30, duplicate_threshold: float = 0.9, memory: str = ""
    ) -> str:
        """Re-index the memory and report pages that need maintenance.

        Re-embeds pages changed on disk by any process (the agent, a human in
        an editor, a sync script), drops index rows for deleted pages, and
        lists near-duplicate page pairs plus pages whose ``updated`` timestamp
        is old, so they can be merged, re-verified, or deleted.

        Args:
            stale_days: Pages not updated in this many days are listed as
                stale (default 30).
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
                lines.append(f"Pages not updated in {stale_days} days (re-verify or delete):")
                lines.extend(
                    f"  {_qualified(memory_name, name)}  (updated {updated})"
                    for name, updated in stale[:20]
                )
            if not duplicates and not stale:
                lines.append("No near-duplicate or stale pages.")
        return "\n".join(lines)

    @staticmethod
    def _stale_pages(index: VectorIndex, stale_days: int) -> list[tuple[str, str]]:
        """Return ``(name, updated)`` for pages older than *stale_days* days.

        Reads the frontmatter the just-finished sync stored in *index*
        rather than every page file again.  Pages whose ``updated``
        frontmatter is missing or unparseable are skipped: staleness
        cannot be established for them, and reporting them would train
        the agent to "fix" pages that may be fresh.
        """
        cutoff = datetime.now(UTC) - timedelta(days=stale_days)
        stale: list[tuple[str, str]] = []
        for name, frontmatter in index.frontmatters().items():
            raw_updated = str(frontmatter.get("updated", ""))
            try:
                updated = datetime.strptime(raw_updated, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            except ValueError:
                continue
            if updated < cutoff:
                stale.append((name, raw_updated))
        stale.sort(key=lambda entry: entry[1])
        return stale

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
