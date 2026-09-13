# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Agent tools over a memoryfield: search, pull, read, write, list, delete.

Pass the bound methods of :class:`MemoryTools` to ``KISSAgent.run(tools=...)``
and prepend :data:`MEMORY_PROTOCOL` to the system prompt. Every search first
runs an incremental index sync, so pages written by any process (the agent,
a human in an editor, git pull) are searchable without a separate reindex.
"""

from pathlib import Path

from kiss.agents.memoryfield.index import Embedder, SearchHit, VectorIndex
from kiss.agents.memoryfield.pages import MAX_PAGE_BYTES, MemoryDir

MEMORY_PROTOCOL = """\
## Memory
You have a persistent memory: a directory of short Markdown pages with a semantic
search index. Follow this protocol:
1. Before starting work, call `memory_search` with a description of the task to
   recall relevant lessons, corrections, decisions and past results. Use
   `memory_pull` to read the matching pages in one call.
2. While working, record durable knowledge with `memory_write`: facts you had to
   look up, user preferences, project conventions, root causes of bugs, what
   worked and what did not. One topic per page, under ~8 KB, with source URLs or
   file paths so the fact can be re-verified later. Prefer updating an existing
   page over creating a near-duplicate.
3. Do not store secrets, credentials or raw transcripts. Delete pages that turn
   out to be wrong or obsolete with `memory_delete`.
"""

# Cap on characters returned by memory_pull so a broad query cannot flood the context.
PULL_CHAR_LIMIT = 24_000


def _format_hits(hits: list[SearchHit]) -> str:
    lines = []
    for hit in hits:
        line = f"{hit.score:.3f}  {hit.name}  —  {hit.title}"
        if hit.summary:
            line += f": {hit.summary}"
        lines.append(line)
    return "\n".join(lines)


class MemoryTools:
    """Tool set giving an agent read/write access to one memory directory.

    Args:
        root: The memory directory (created on first write).
        embed: Embedding function; defaults to the framework's
            ``text-embedding-3-small`` model. Pass
            :func:`kiss.agents.memoryfield.index.hashed_embedding` for a
            fully offline memory.
        model_code: Overrides the embedding-model identifier used to name the
            index file.
    """

    def __init__(
        self,
        root: str | Path,
        embed: Embedder | None = None,
        model_code: str | None = None,
    ) -> None:
        self.memory = MemoryDir(root)
        self.index = VectorIndex(self.memory, embed=embed, model_code=model_code)

    def tools(self) -> list:
        """The callables to register with ``KISSAgent.run(tools=...)``."""
        return [
            self.memory_search,
            self.memory_pull,
            self.memory_read,
            self.memory_write,
            self.memory_list,
            self.memory_delete,
        ]

    def memory_search(self, query: str, k: int = 5) -> str:
        """Semantic search over memory pages; returns the best-matching page names with scores.

        Args:
            query: What you want to recall, phrased as a question or topic.
            k: Maximum number of pages to return (default 5).
        """
        self.index.sync()
        hits = self.index.search(query, k=k)
        if not hits:
            return "No memory pages yet." if self.index.count() == 0 else "No matches."
        return _format_hits(hits)

    def memory_pull(self, query: str, k: int = 3) -> str:
        """Semantic search that returns the full contents of the matching pages in one call.

        Args:
            query: What you want to recall, phrased as a question or topic.
            k: Maximum number of pages to return (default 3).
        """
        self.index.sync()
        hits = self.index.search(query, k=k)
        if not hits:
            return "No memory pages yet." if self.index.count() == 0 else "No matches."
        chunks: list[str] = []
        used = 0
        for hit in hits:
            raw = self.memory.read(hit.name).raw
            chunk = f"### {hit.name}.md  (score {hit.score:.3f})\n{raw}"
            if used + len(chunk) > PULL_CHAR_LIMIT:
                if not chunks:
                    chunk = (
                        chunk[:PULL_CHAR_LIMIT] + "\n[page truncated; use memory_read for the rest]"
                    )
                    chunks.append(chunk)
                omitted = len(hits) - len(chunks)
                if omitted:
                    chunks.append(f"[{omitted} more page(s) omitted; lower k or refine the query]")
                break
            chunks.append(chunk)
            used += len(chunk)
        return "\n\n".join(chunks)

    def memory_read(self, name: str) -> str:
        """Return the full text of one memory page.

        Args:
            name: Page name as shown by memory_search or memory_list (with or without .md).
        """
        try:
            return self.memory.read(name).raw
        except FileNotFoundError:
            return f"Error: no memory page named {name!r}."
        except ValueError as e:
            return f"Error: {e}"

    def memory_write(self, name: str, content: str, title: str = "", summary: str = "") -> str:
        """Create or replace a memory page (Markdown body; frontmatter is generated for you).

        Args:
            name: Page name: lowercase letters, digits and hyphens, e.g. 'postgres-agent-auth-flow'.
            content: Markdown body. Keep it under ~8 KB; one topic per page; cite sources.
            title: Human-readable title (defaults to the name on first write).
            summary: One-sentence summary shown in search results.
        """
        try:
            page = self.memory.write(name, content, title=title, summary=summary)
        except ValueError as e:
            return f"Error: {e}"
        size = len(page.raw.encode("utf-8"))
        note = ""
        if size > MAX_PAGE_BYTES:
            note = (
                f" Warning: page is {size} bytes; only the first {MAX_PAGE_BYTES} bytes are "
                "embedded. Split it into several pages."
            )
        return f"Wrote {page.name}.md ({size} bytes).{note}"

    def memory_list(self) -> str:
        """List every memory page with its title and summary."""
        names = self.memory.page_names()
        if not names:
            return "No memory pages yet."
        lines = []
        for name in names:
            page = self.memory.read(name)
            line = f"{name}  —  {page.title}"
            if page.summary:
                line += f": {page.summary}"
            lines.append(line)
        return "\n".join(lines)

    def memory_delete(self, name: str) -> str:
        """Delete a memory page that is wrong or obsolete.

        Args:
            name: Page name (with or without .md).
        """
        try:
            self.memory.delete(name)
        except FileNotFoundError:
            return f"Error: no memory page named {name!r}."
        except ValueError as e:
            return f"Error: {e}"
        return f"Deleted {name}."
