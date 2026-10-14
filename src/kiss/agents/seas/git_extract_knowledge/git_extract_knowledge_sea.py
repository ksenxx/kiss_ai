# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Git knowledge extractor — builds and refreshes the durable memory of a repository.

``/git_extract_knowledge <repo>`` (a local path or a clone URL) makes the
agent go over every tracked file and every commit of the repository and
leave behind a memory that answers questions about the domain the
repository addresses and about the repository itself.  The memory has
two tiers, both in the repository's *domain memory* directory
(``$KISS_HOME/memories/<repo-slug>/``), which every Sorcar run inside that
repository attaches through its ``memory_*`` tools (the slug is the
checkout's directory name, so two unrelated repositories checked out
under the same name share one memory and rebuild it in turn):

* **Curated pages** — Markdown pages the agent writes with
  :func:`write_knowledge_page`: overview, domain glossary, architecture,
  conventions, history, one page per module, recent changes.  These are
  ordinary memory pages, found by ``memory_search`` / ``memory_pull``.
* **The block store** — ``knowledge.sqlite3``, a full-text (FTS5) index
  the :func:`index_repo` tool builds deterministically
  (:mod:`kiss.agents.seas.git_extract_knowledge.git_knowledge_index`): one block per file,
  per 80-line chunk, per symbol definition, per commit, per (commit,
  file) change with its patch, per tag, branch, contributor and
  directory.  It holds millions of blocks and is queried with
  :func:`knowledge_search` or, from any shell, with this module's CLI
  (``python -m kiss.agents.seas.git_extract_knowledge.git_extract_knowledge_sea search ...``).
  The tool-written page ``knowledge-lookup`` tells other agents how.

The first run also registers a daily cron job (:func:`schedule_daily_update`)
that runs this SEA again with the task ``update <repo>`` at
:data:`DAILY_UPDATE_HOUR_PACIFIC` o'clock America/Los_Angeles time, so
the memory follows the repository.  ``update`` runs are incremental:
only changed files and new commits are indexed, and the agent revises
the pages the changes affect.  ``ask <question>`` answers from the
memory.

The SEA class's methods (``system_prompt()``, ``tools()``, ...) follow the
SEA contract in :mod:`kiss.agents.seas.base.base_sea`.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.git_extract_knowledge.git_knowledge_index import (
    IndexReport,
    KnowledgeError,
    format_report,
    is_url,
    memory_location,
    now_iso,
    resolve_repo,
    store_path,
)
from kiss.agents.seas.git_extract_knowledge.git_knowledge_index import index_repo as _index_repo
from kiss.agents.seas.git_extract_knowledge.git_knowledge_store import KINDS, Block, KnowledgeStore
from kiss.core.memoryfield.pages import MemoryDir
from kiss.core.memoryfield.tools import MemoryTools

MODULE = "kiss.agents.seas.git_extract_knowledge.git_extract_knowledge_sea"
"""Importable name of this module, for the CLI commands quoted in prompts and pages."""

LOOKUP_PAGE = "knowledge-lookup"
"""Name of the tool-written page that tells agents how to query the block store."""

DAILY_UPDATE_HOUR_PACIFIC = 4
"""Hour (America/Los_Angeles) of the daily memory refresh."""

DAILY_UPDATE_BUDGET_USD = 25.0
"""Default budget (USD) of one daily refresh: the SEA run that re-indexes the
repository and revises the pages.  Indexing costs nothing; the revision of
the pages of a busy day (dozens of commits, several pages rewritten) is what
this pays for."""

DAILY_UPDATE_RELAY_BUDGET_USD = 2.0
"""Extra budget of the cron job's relay session — the one that only calls
``run_agent`` and relays its result — on top of the refresh's own budget.
The nested run's spending is attributed to the relay when it returns, so the
job's budget is the sum of the two."""

DAILY_UPDATE_MODEL = "claude-fable-5-1"
"""Model of the daily refresh (the cron job and the nested SEA run).  Pinned:
an unattended job must not silently follow the daemon's default model."""

DAILY_UPDATE_TIMEOUT_SECONDS = 4 * 3600
"""Per-run timeout of the daily refresh job (a big repository can take a while)."""

JOB_NAME_PREFIX = "git-knowledge daily update: "
"""Cron job names are this prefix plus the repository's memory slug."""

DAILY_PROMPT = (
    "Call the run_agent tool IMMEDIATELY, as your first action, with these "
    "arguments and no others:\n"
    "  agent      = {sea!r}\n"
    "  task       = {task!r}\n"
    "  timeout    = {timeout!r}\n"
    "  max_budget = {max_budget!r}\n"
    "  model      = {model!r}\n"
    "Do not explore source code, paraphrase the task, or call any other tool first. "
    "When run_agent returns, relay its result as your final summary."
)
"""Prompt of the daily cron job: a ``run_agent`` directive to this SEA.

The ``timeout``, ``max_budget`` and ``model`` arguments matter: the
nested run gets the daemon's defaults for whatever the directive leaves out
— a 300-second timeout (``agent_dispatch``), the configured default budget
(thousands of dollars, not the job's) and the daemon's current model —
whatever the cron job's own settings say.
"""

SYSTEM_PROMPT = """You build and maintain a git repository's durable memory: its domain, \
build and history. Take every fact from the repository (files, commits, tags) and cite \
it. Never guess.

# The memory
Both tiers live in the repository's domain memory directory, which every Sorcar agent \
working there searches:
1. Curated pages, written with `write_knowledge_page`: Markdown, one topic per page, \
under 8 KB (split longer topics), every claim backed by a path (and line numbers when \
useful), a symbol or a commit hash.
2. The block store, built by `index_repo`: a full-text index of every tracked file \
(`file`), 80-line chunk (`chunk`), definition (`symbol`), commit on any branch or tag \
(`commit`), per-file change with patch (`change`), tag, branch, contributor (`author`) \
and directory (`dir`). Nothing is sampled; only text files are indexed, and a file or \
per-commit diff above 64 MB is marked `content: not indexed` / `patch: truncated`. Query \
it with `knowledge_search` / `knowledge_read`: keyword search (identifiers, file names, \
commit-message words; `auth*` matches a prefix; all-word matches rank first, then \
any-word matches).

# The task text
It names the repository (local path or clone URL) and optionally a mode word:
- `build` or none: BUILD the memory, then schedule the daily update;
- `update`: REFRESH the memory incrementally (the daily cron run); schedule nothing;
- `ask <question>`: ANSWER from the memory.
Tools run in the daemon, not in your working directory, so pass a local repository as \
an ABSOLUTE path; for a relative path or `.`, prepend the output of `pwd` run with \
`Bash`. Pass the same `repo` to every tool.

# BUILD
1. `index_repo(repo)`; read its report (files, languages, top-level directories, \
commit count and span, contributors, tags, block counts).
2. With `Read`/`Bash` in the reported path, read README and docs, CONTRIBUTING, build \
manifests (pyproject.toml, package.json, Cargo.toml, go.mod, pom.xml, CMakeLists.txt, \
Makefile, ...), CI and release configuration, changelogs. Use `knowledge_search` for \
the rest: where a concept lives, which commits introduced it, who works on what.
3. Write these pages (fixed names, so later runs find them):
   - `overview`: what the software does and for whom, the domain in one paragraph, the \
stack, entry points (binaries, CLIs, services, public APIs), how to build, test and run.
   - `domain-glossary`: the domain's terms as the repository uses them, each with a \
one-or-two-sentence definition and the paths where it is central; continue in \
`domain-glossary-2`, ... above 8 KB.
   - `domain-concepts`: the ideas, models, algorithms, data formats, standards and \
protocols the code implements, with the implementing paths.
   - `architecture`: major components, their dependencies and interactions, the data \
flow of the main use cases, persistence, configuration, extension points.
   - `conventions`: code style, layout, testing and how to run tests, lint and type \
checks, branching, versioning, release process, CI.
   - `history`: eras with dates and commit ranges, releases (tags), the largest or most \
consequential changes, the current focus; cite commit hashes.
   - `module-<slug>` per top-level directory and significant package in it \
(`module-src-kiss-core`): purpose, key files and their roles, key symbols, dependencies \
in and out, invariants and gotchas, recent changes.
   - `faq`: what a newcomer or maintainer asks about the domain and the repository, \
each with a short answer and its evidence path or hash.
   In a large repository, fan out the module pages with `run_parallel`: give each \
sub-agent the repository path, its directories, the naming rule and the exact CLI \
commands below; it reads code with `Read`/`Bash` and writes pages through the CLI. \
Write the other pages yourself.
4. Verify with at least eight `search_knowledge_pages` and `knowledge_search` queries a \
user might ask; fix missing or wrong pages.
5. `schedule_daily_update(repo)`.
6. `finish` with success and an HTML summary: memory directory, block counts, pages \
written, the cron job, what a user can now ask.

# UPDATE (daily, unattended: never ask questions)
1. `index_repo(repo)`; its report lists the changed files and the new commits since \
the previous run. If nothing changed, finish with the one-line summary \
"No changes since the last update."
2. Read the new commits (`knowledge_read` on `commit:<sha>` and `change:<sha>:<path>`, \
or `git show` in the repository path) and the changed files.
3. Revise every affected page (`read_knowledge_page`, then `write_knowledge_page` with \
the whole corrected page): module pages, `architecture`, `conventions`, `overview`; \
`domain-*` when vocabulary or concepts moved; `history` for a tagged release or a large \
change; `faq` when an answer changed. Add a dated entry (what, why, commit hashes) at \
the top of `recent-changes`; when it nears 8 KB, move its oldest entries into `history`.
4. `finish` with an HTML summary of what changed in the repository and in the memory.

# ASK
Search the pages (`search_knowledge_pages`, `read_knowledge_page`) and the block store \
(`knowledge_search`, `knowledge_read`), then `finish` with the answer in HTML, citing \
pages, paths and commit hashes. Say plainly when the memory lacks the answer.

# Sub-agents
If `index_repo` is not among your tools, you are a fan-out sub-agent; run the knowledge \
tools as shell commands with `Bash`:
  {python} -m {module} search <repo> "<query>" [--k 10] [--kinds symbol,commit] [--path src/]
  {python} -m {module} read <repo> <block-key>
  {python} -m {module} pages <repo>
  {python} -m {module} read-page <repo> <name>
  {python} -m {module} write-page <repo> <name> --title "<title>" --summary "<one sentence>" \
--file <markdown file>
Quote `<repo>` when the path has spaces. Write each page's Markdown to a file first, then \
pass it with `--file`.

# Rules
- Cite evidence for every fact: `path:line`, `symbol`, `commit abc1234`, `tag v1.2`.
- State only what you verified: a number from a command you ran (`git rev-list --count \
main`, `git shortlog -sn`, `wc -l`, `grep -c`), with its scope; a command, CLI or \
console-script name from the manifest (`[project.scripts]`, `bin`, `Makefile` targets), \
never from habit; a commit hash from `git log`; "A depends on B" from an import or a \
call, not from a docstring.
- Describe the code as it is today; a removed behaviour is not current.
- Never edit, commit or push in the repository; the memory is the only output.
- Plain, specific prose. No filler, no marketing, no emoji.
- When the `summary` tool exists, call it every ten steps: when a tool result's `Steps:` \
footer ends in 9, make the next call `summary`; otherwise the 10th, 20th, ... call is \
refused and must be repeated.
- The final `finish` summary is HTML (`<h3>`, `<p>`, `<ul>`), never Markdown.
"""

_resolved: dict[str, Path] = {}


class GitExtractKnowledgeSea(WorkerSea):
    """The ``/git_extract_knowledge`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/git_extract_knowledge help``."""
        return (
            "Indexes every tracked file and commit of a git repository into its domain memory "
            "(curated Markdown pages plus a full-text block store) and schedules a daily "
            "incremental refresh; use `/git_extract_knowledge <repo path or clone URL>`, "
            "`/git_extract_knowledge update <repo>`, `/git_extract_knowledge ask <question>` or "
            'run_agent(agent="git_extract_knowledge", task=...).'
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Return :data:`SYSTEM_PROMPT` with the interpreter and module filled in."""
        return SYSTEM_PROMPT.replace(
            "{python}", shlex.quote(sys.executable),
        ).replace("{module}", MODULE)

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A worker with the full toolset, on the real checkout.

        No worktree or auto-commit (it indexes repositories, it does not
        change them), no classifier, no browser, no memory; the full
        toolset keeps ``run_parallel`` for the per-repository indexing
        sub-agents.
        """
        return settings | {
            "tool_profile": "full",
        }

    def tools(self, tools: list[Any]) -> list[Any]:
        """Return the knowledge tools added to the built-in toolset."""
        return tools + [
            index_repo, knowledge_status, knowledge_search, knowledge_read,
            list_knowledge_pages, read_knowledge_page, search_knowledge_pages,
            write_knowledge_page, delete_knowledge_page, schedule_daily_update,
        ]


def _repo(spec: str) -> Path:
    """Resolve *spec* once per process (a URL is fetched on the first call only)."""
    if spec not in _resolved:
        _resolved[spec] = resolve_repo(spec)
    return _resolved[spec]


def canonical_spec(spec: str) -> str:
    """The form of *spec* a later run can rely on: the URL itself, or the checkout's root."""
    return spec.strip() if is_url(spec) else str(_repo(spec))


def _memory_tools(repo: Path) -> tuple[str, MemoryTools]:
    """Return ``(slug, MemoryTools)`` addressing the repository's domain memory."""
    from kiss.agents.sorcar.sorcar_agent import _memory_settings, _repo_memory_domains

    domains = _repo_memory_domains(str(repo))
    slug = next(iter(domains))
    return slug, MemoryTools(_memory_settings()[1], domains=domains)


def note_block(name: str, page_text: str) -> Block:
    """The ``note`` block mirroring a curated page into the block store."""
    return Block("note", f"note:{name}", f"page {name}", page_text)


def sync_note_blocks(memory_dir: Path, store: KnowledgeStore) -> int:
    """Mirror every curated page of *memory_dir* into ``note`` blocks; return the count."""
    pages = MemoryDir(memory_dir)
    store.delete(kind="note")
    blocks = [note_block(name, pages.read(name).raw) for name in pages.page_names()]
    return store.upsert(blocks, now_iso())


def lookup_page(repo: Path, slug: str, report: IndexReport) -> str:
    """The body of the ``knowledge-lookup`` page for *repo*."""
    counts = ", ".join(f"{kind} {report.counts.get(kind, 0)}" for kind in KINDS)
    total = sum(report.counts.values())
    python = shlex.quote(sys.executable)
    target = shlex.quote(str(repo))
    return f"""# How to look up knowledge about {repo.name}

The memory of the repository `{repo}` has two tiers.

1. Curated pages in this domain memory (`{slug}/...`): `overview`, `domain-glossary`,
   `domain-concepts`, `architecture`, `conventions`, `history`, `module-*` (one per
   module), `faq`, `recent-changes`. Search them with `memory_search` / `memory_pull`.
2. The block store `{report.store_path}`: {total} blocks ({counts}). It indexes
   every tracked file (`file`), every 80-line chunk of every text file (`chunk`), every
   definition (`symbol`), every commit on any branch or tag (`commit`), every per-file
   change with its patch (`change`), every tag, branch, contributor (`author`) and
   directory (`dir`). Query it from the shell:

       {python} -m {MODULE} search {target} "<query>" [--k 10] [--kinds symbol,commit] [--path src/]
       {python} -m {MODULE} read {target} <block-key>
       {python} -m {MODULE} status {target}

   Queries are keyword searches (BM25): use identifiers, file names and words from
   commit messages; `auth*` matches a prefix; all words must match first, then any-word
   matches fill the remaining slots. Block keys look like `file:src/a.py`,
   `chunk:src/a.py:81`, `symbol:src/a.py:12:run`, `commit:<sha>`, `change:<sha>:<path>`
   (a long patch continues in `change:<sha>#2:<path>`, ...), `tag:v1.0`, `branch:main`,
   `author:<email>`, `dir:src`.

Last indexed {report.head[:12]} on {now_iso()} ({report.mode} run, {report.seconds} s).
Refreshed every morning at {DAILY_UPDATE_HOUR_PACIFIC:02d}:00 America/Los_Angeles (PDT or
PST) by the cron job "{JOB_NAME_PREFIX}{slug}" while the kiss-web daemon runs.
"""


# ----- tools ------------------------------------------------------------------


def index_repo(repo: str) -> str:
    """Index the repository into its block store and report what was indexed.

    The first run indexes every tracked file and every commit on ``HEAD``;
    later runs index only files whose content changed and commits since
    the previous run (a rewritten history triggers a full rebuild).  The
    curated pages are mirrored into ``note`` blocks and the
    ``knowledge-lookup`` page is (re)written.

    Args:
        repo: A local path inside the repository or a clone URL (a URL is
            cloned under ``$KISS_HOME/knowledge/checkouts`` and reset to
            the remote's default branch on every run).

    Returns:
        The index report: repository path, memory directory, mode,
        HEAD, block counts per kind, languages, top-level directories,
        contributors, tags, the files and commits indexed in this run.
    """
    try:
        path = _repo(repo)
        store = KnowledgeStore(store_path(path))
        report = _index_repo(path, store)
        slug, memory_dir = memory_location(path)
        MemoryDir(memory_dir).write(
            LOOKUP_PAGE, lookup_page(path, slug, report),
            title=f"How to look up knowledge about {path.name}",
            summary=(
                f"Where the memory of the repository {path.name} lives (curated pages and "
                f"the knowledge.sqlite3 block store) and the shell commands that query it."
            ),
        )
        notes = sync_note_blocks(memory_dir, store)
    except KnowledgeError as exc:
        return f"Error: {exc}"
    return format_report(report) + f"\ncurated pages mirrored as note blocks: {notes}"


def knowledge_status(repo: str) -> str:
    """Report the state of the repository's memory without indexing anything.

    Args:
        repo: A local path inside the repository or a clone URL.

    Returns:
        The memory directory, block store path, last indexed HEAD and
        time, block counts per kind, and the names of the curated pages.
    """
    try:
        path = _repo(repo)
        slug, memory_dir = memory_location(path)
    except KnowledgeError as exc:
        return f"Error: {exc}"
    db = store_path(path)
    if not db.exists():
        return f"No memory yet for {path}: run index_repo first (would live in {memory_dir})."
    store = KnowledgeStore(db)
    counts = store.counts()
    lines = [
        f"repository: {path}",
        f"memory slug: {slug}",
        f"memory directory: {memory_dir}",
        f"block store: {db}",
        f"last indexed HEAD: {store.get_meta('head')} at {store.get_meta('indexed_at')} "
        f"({store.get_meta('mode')} run)",
        f"blocks: {sum(counts.values())} (" + ", ".join(
            f"{kind} {counts[kind]}" for kind in KINDS if kind in counts
        ) + ")",
        "curated pages: " + (", ".join(MemoryDir(memory_dir).page_names()) or "(none)"),
    ]
    return "\n".join(lines)


def knowledge_search(
    repo: str, query: str, k: int = 10, kinds: str = "", path_prefix: str = "",
) -> str:
    """Search the repository's block store by keywords.

    Args:
        repo: A local path inside the repository or a clone URL.
        query: Keywords: identifiers, file names, words from commit
            messages; ``auth*`` matches a prefix.  Every word must match
            first; when that yields fewer than *k* hits, blocks matching
            some of the words fill the rest.
        k: Maximum number of hits (at most 100).
        kinds: Comma-separated block kinds to restrict to (``symbol,commit``);
            empty searches every kind.
        path_prefix: Restrict to blocks about paths under this prefix.

    Returns:
        One hit per paragraph: key, title, and a snippet with the matched
        words in brackets.  Read a whole block with ``knowledge_read``.
    """
    try:
        store = KnowledgeStore(store_path(_repo(repo)))
    except KnowledgeError as exc:
        return f"Error: {exc}"
    wanted = [kind.strip() for kind in kinds.split(",") if kind.strip()]
    unknown = [kind for kind in wanted if kind not in KINDS]
    if unknown:
        return f"Error: unknown block kinds {unknown}; choose from {', '.join(KINDS)}."
    hits = store.search(query, k=min(max(k, 1), 100), kinds=wanted, path_prefix=path_prefix)
    if not hits:
        return f"No blocks match {query!r}."
    parts = [
        f"{i}. {hit.block.key}\n   {hit.block.title}\n   {hit.snippet.replace(chr(10), ' | ')}"
        for i, hit in enumerate(hits, start=1)
    ]
    return "\n".join(parts)


def knowledge_read(repo: str, key: str) -> str:
    """Return the full text of one block of the repository's block store.

    Args:
        repo: A local path inside the repository or a clone URL.
        key: The block key as shown by ``knowledge_search`` (for example
            ``file:src/a.py``, ``commit:<sha>``, ``change:<sha>:<path>``).
    """
    try:
        block = KnowledgeStore(store_path(_repo(repo))).get(key)
    except KnowledgeError as exc:
        return f"Error: {exc}"
    if block is None:
        return f"No block with key {key!r}."
    return f"{block.key}\n{block.title}\n\n{block.text}"


def list_knowledge_pages(repo: str) -> str:
    """List the curated pages of the repository's memory with their summaries.

    Args:
        repo: A local path inside the repository or a clone URL.
    """
    try:
        slug, tools = _memory_tools(_repo(repo))
    except KnowledgeError as exc:
        return f"Error: {exc}"
    return tools.memory_list(slug)


def read_knowledge_page(repo: str, name: str) -> str:
    """Return the full text of one curated page of the repository's memory.

    Args:
        repo: A local path inside the repository or a clone URL.
        name: The page name (``overview``, ``module-src-core``, ...).
    """
    try:
        slug, tools = _memory_tools(_repo(repo))
    except KnowledgeError as exc:
        return f"Error: {exc}"
    return tools.memory_read(f"{slug}/{name}")


def search_knowledge_pages(repo: str, query: str, k: int = 5) -> str:
    """Search the curated pages semantically, as any Sorcar agent's ``memory_search`` does.

    Args:
        repo: A local path inside the repository or a clone URL.
        query: A question or topic in natural language.
        k: Maximum number of pages to return.
    """
    try:
        slug, tools = _memory_tools(_repo(repo))
    except KnowledgeError as exc:
        return f"Error: {exc}"
    return tools.memory_search(query, k=k, memory=slug)


def write_knowledge_page(
    repo: str, name: str, content: str, title: str = "", summary: str = "",
) -> str:
    """Create or replace a curated page of the repository's memory.

    The page is written into the repository's domain memory (where every
    Sorcar agent working in the repository finds it) and mirrored into
    the block store as a ``note`` block.

    Args:
        repo: A local path inside the repository or a clone URL.
        name: Page name: lowercase letters, digits and hyphens
            (``overview``, ``module-src-core``).
        content: The Markdown body — the whole page, under ~8 KB, with
            evidence (paths, symbols, commit hashes) for every fact.
        title: Human-readable title.
        summary: One sentence shown in search results.

    Returns:
        A confirmation with the page size, or an error message.
    """
    try:
        path = _repo(repo)
        slug, tools = _memory_tools(path)
    except KnowledgeError as exc:
        return f"Error: {exc}"
    result = tools.memory_write(f"{slug}/{name}", content, title=title, summary=summary)
    if result.startswith("Error"):
        return result
    page = MemoryDir(memory_location(path)[1]).read(name)
    KnowledgeStore(store_path(path)).upsert([note_block(name, page.raw)], now_iso())
    return result


def delete_knowledge_page(repo: str, name: str) -> str:
    """Delete a curated page of the repository's memory and its ``note`` block.

    Args:
        repo: A local path inside the repository or a clone URL.
        name: The page name.
    """
    try:
        path = _repo(repo)
        slug, tools = _memory_tools(path)
    except KnowledgeError as exc:
        return f"Error: {exc}"
    result = tools.memory_delete(f"{slug}/{name}")
    if not result.startswith("Error"):
        KnowledgeStore(store_path(path)).delete(key=f"note:{name}")
    return result


def daily_update_schedule() -> str:
    """The 5-field cron expression of the daily refresh.

    The cron scheduler (:data:`kiss.agents.sorcar.cron_agent.SCHEDULE_TZ`)
    evaluates every cron expression in America/Los_Angeles wall-clock time,
    whatever the daemon machine's own time zone, so the expression names
    :data:`DAILY_UPDATE_HOUR_PACIFIC` o'clock directly and follows daylight
    saving by itself.
    """
    return f"0 {DAILY_UPDATE_HOUR_PACIFIC} * * *"


def daily_update_job(repo: str, max_budget: float = DAILY_UPDATE_BUDGET_USD) -> dict[str, str]:
    """The ``cron_job("create", ...)`` arguments of the daily refresh of *repo*.

    Args:
        repo: A local path inside the repository or a clone URL.
        max_budget: Budget (USD) of the nested SEA run; the job gets
            :data:`DAILY_UPDATE_RELAY_BUDGET_USD` more for its relay session.

    Raises:
        KnowledgeError: When *repo* is not a git repository or clone URL.
    """
    slug = memory_location(_repo(repo))[0]
    target = canonical_spec(repo)
    return {
        "name": JOB_NAME_PREFIX + slug,
        "schedule": daily_update_schedule(),
        "prompt": DAILY_PROMPT.format(
            sea=str(Path(__file__).resolve()), task=f"update {target}",
            timeout=str(DAILY_UPDATE_TIMEOUT_SECONDS), max_budget=str(max_budget),
            model=DAILY_UPDATE_MODEL,
        ),
        "model_name": DAILY_UPDATE_MODEL,
        "max_budget": str(max_budget + DAILY_UPDATE_RELAY_BUDGET_USD),
        "timeout": str(DAILY_UPDATE_TIMEOUT_SECONDS),
    }


def schedule_daily_update(repo: str, max_budget: float = DAILY_UPDATE_BUDGET_USD) -> str:
    """Register the daily cron job that refreshes the repository's memory.

    The job runs this SEA with the task ``update <repo>`` every day at
    :data:`DAILY_UPDATE_HOUR_PACIFIC` o'clock America/Los_Angeles (the
    kiss-web daemon must be running for cron jobs to fire).  Idempotent:
    an existing up-to-date job for this repository is reported, not
    duplicated; an existing job for it whose schedule, model, budget or
    prompt differ (created by an older version of this SEA, or with
    another budget) is replaced.

    Args:
        repo: A local path inside the repository or a clone URL; the job
            stores the URL, or the checkout's root directory, so it does
            not depend on any working directory.
        max_budget: Budget of the nested SEA run in USD (the job itself
            gets :data:`DAILY_UPDATE_RELAY_BUDGET_USD` more for its relay).

    Returns:
        The created or existing job (id, schedule, next run), or an error.
    """
    from kiss.agents.sorcar.cron_agent import SCHEDULE_TZ, cron_job, load_jobs

    try:
        wanted = daily_update_job(repo, max_budget)
        task_marker = f"= {f'update {canonical_spec(repo)}'!r}\n"  # in every prompt version
    except KnowledgeError as exc:
        return f"Error: {exc}"
    replaced: list[str] = []
    current: dict[str, Any] | None = None
    for job in load_jobs():
        if job.get("name") != wanted["name"] or task_marker not in str(job.get("prompt", "")):
            continue  # another repository (two repositories may share a slug)
        if (
            current is None
            and job.get("prompt") == wanted["prompt"]
            and job.get("schedule") == wanted["schedule"]
            and job.get("model_name") == wanted["model_name"]
            and float(job.get("max_budget") or 0) == float(wanted["max_budget"])
            and float(job.get("timeout") or 0) == float(wanted["timeout"])
        ):
            current = job
            continue
        cron_job("remove", job_id=str(job["id"]))
        replaced.append(str(job["id"]))
    note = f"Removed outdated job(s) {', '.join(replaced)}.\n" if replaced else ""
    if current is not None:
        when = datetime.fromtimestamp(float(current["next_run_at"]), SCHEDULE_TZ)
        return (
            f"{note}Already scheduled: job {current['id']} ({wanted['name']}), schedule "
            f"{current['schedule']!r} Pacific time, next run "
            f"{when.isoformat(timespec='minutes')}."
        )
    return note + cron_job(
        "create", name=wanted["name"], schedule=wanted["schedule"], prompt=wanted["prompt"],
        model_name=wanted["model_name"], max_budget=wanted["max_budget"],
        timeout=wanted["timeout"],
    )


# ----- the SEA class --------------------------------------------------------------


# ----- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the knowledge tools from the shell (used by fan-out sub-agents and any agent).

    Sub-commands: ``index``, ``status``, ``search``, ``read``, ``pages``,
    ``read-page``, ``write-page``, ``delete-page``, ``schedule``.

    Args:
        argv: Command-line arguments (default ``sys.argv[1:]``).

    Returns:
        The process exit code: 0, or 1 when the tool reported an error.
    """
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}", description="Query and maintain a repository's memory.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("index", "status", "pages", "schedule"):
        sub.add_parser(command).add_argument("repo")
    search = sub.add_parser("search")
    search.add_argument("repo")
    search.add_argument("query")
    search.add_argument("--k", type=int, default=10)
    search.add_argument("--kinds", default="")
    search.add_argument("--path", default="")
    for command in ("read", "read-page", "delete-page"):
        cmd = sub.add_parser(command)
        cmd.add_argument("repo")
        cmd.add_argument("key" if command == "read" else "name")
    write = sub.add_parser("write-page")
    write.add_argument("repo")
    write.add_argument("name")
    write.add_argument("--title", default="")
    write.add_argument("--summary", default="")
    write.add_argument("--file", required=True, help="Markdown file with the page body")
    args = parser.parse_args(argv)
    if not is_url(args.repo):
        args.repo = os.path.abspath(os.path.expanduser(args.repo))  # relative to the shell's cwd
    if args.command == "index":
        out = index_repo(args.repo)
    elif args.command == "status":
        out = knowledge_status(args.repo)
    elif args.command == "search":
        out = knowledge_search(args.repo, args.query, args.k, args.kinds, args.path)
    elif args.command == "read":
        out = knowledge_read(args.repo, args.key)
    elif args.command == "pages":
        out = list_knowledge_pages(args.repo)
    elif args.command == "read-page":
        out = read_knowledge_page(args.repo, args.name)
    elif args.command == "delete-page":
        out = delete_knowledge_page(args.repo, args.name)
    elif args.command == "write-page":
        out = write_knowledge_page(
            args.repo, args.name, Path(args.file).read_text(encoding="utf-8"),
            args.title, args.summary,
        )
    else:
        out = schedule_daily_update(args.repo)
    print(out)
    return 1 if out.startswith("Error") else 0


if __name__ == "__main__":
    sys.exit(main())
