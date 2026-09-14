# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Recall evaluation of the memoryfield index on real past Sorcar tasks.

The evaluation turns finished tasks from ``~/.kiss/sorcar.db`` into memory
pages (one page per task: the request plus the final result as text), then
asks: given a later question, does retrieval surface the page of the task
that answered it? Each probe query has exactly one gold page, so we report
Recall@1/3/5 (was the gold page in the top k?) and MRR (1/rank of the gold
page) for several retrievers:

* ``vector``  the memoryfield index with the default embedding model
* ``hashed``  the same index with the offline feature-hashing embedding
* ``bm25``    SQLite FTS5 keyword search (the "just grep it" baseline)
* ``hybrid``  reciprocal-rank fusion of ``vector`` and ``bm25``

Probe queries come from two sources: a hand-written set that references
specific known tasks, and LLM-written paraphrases of sampled pages.

Users repeat tasks ("run all tests", "update the README"), so the corpus
contains families of near-identical pages that no retriever can tell apart
from the query alone. Besides strict recall of the exact gold page, we also
report *family* recall: a page whose request text, after whitespace and case
normalisation, equals the gold page's request appears in the top k. Family
recall is what a curated memory would achieve after merging repeats into one
page per topic.

Run::

    uv run python -m kiss.core.memoryfield.evaluate --limit 300 --llm-probes 40
"""

import argparse
import hashlib
import json
import logging
import random
import re
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

from kiss.core.memoryfield.index import (
    DEFAULT_EMBEDDING_MODEL,
    HASHED_EMBEDDING_MODEL_CODE,
    ModelEmbedder,
    VectorIndex,
    hashed_embedding,
)
from kiss.core.memoryfield.pages import MAX_PAGE_BYTES, MemoryDir, slugify

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("~/.kiss/sorcar.db")
DEFAULT_PROBE_MODEL = "claude-fable-5-1"
RECALL_KS = (1, 3, 5)

_BLOCK_TAGS = frozenset(
    {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "pre", "table"}
)
_FAILURE_PREFIXES = ("Task failed", "Task interrupted")
_FAILURE_EXACT = frozenset(
    {
        "Agent Failed Abruptly",
        "Task terminated unexpectedly (process killed)",
        "Task stopped by user",
    }
)


class _TextExtractor(HTMLParser):
    """Collect the text of an HTML fragment, one line per block element."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(html: str) -> str:
    """Convert an HTML fragment (a task result) to plain text.

    Args:
        html: HTML or plain text.

    Returns:
        Text with block elements separated by newlines and runs of blank
        lines collapsed.
    """
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    text = "".join(extractor.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@dataclass(frozen=True)
class PastTask:
    """A finished top-level task loaded from ``task_history``."""

    task_id: str
    timestamp: float
    task: str
    result: str

    @property
    def page_name(self) -> str:
        """Stable page name: a slug of the request plus the first 8 hex digits of the id."""
        first_line = self.task.strip().splitlines()[0] if self.task.strip() else "task"
        return f"{slugify(first_line, 48)}-{self.task_id[:8]}"

    @property
    def title(self) -> str:
        """The first line of the request, trimmed to 120 characters."""
        first_line = self.task.strip().splitlines()[0] if self.task.strip() else self.task_id
        return first_line[:120]

    @property
    def family(self) -> str:
        """Tasks with the same normalised request text form one family (repeated tasks)."""
        return " ".join(self.task.lower().split())


def load_past_tasks(db_path: Path, limit: int, min_result_chars: int = 300) -> list[PastTask]:
    """Load the most recent successful top-level tasks from a Sorcar database.

    Args:
        db_path: Path to ``sorcar.db`` (opened read-only).
        limit: Maximum number of tasks.
        min_result_chars: Skip tasks whose result is shorter than this.

    Returns:
        Tasks ordered newest first.
    """
    uri = db_path.expanduser().resolve().as_uri() + "?mode=ro"
    tasks: list[PastTask] = []
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        rows = conn.execute(
            "SELECT id, timestamp, task, result FROM task_history"
            " WHERE (parent_task_id IS NULL OR parent_task_id = '') AND length(result) >= ?"
            " ORDER BY timestamp DESC",
            (min_result_chars,),
        )
        for task_id, timestamp, task, result in rows:
            if result.startswith(_FAILURE_PREFIXES) or result in _FAILURE_EXACT:
                continue
            tasks.append(PastTask(task_id=task_id, timestamp=timestamp, task=task, result=result))
            if len(tasks) >= limit:
                break
    return tasks


def task_page_body(task: PastTask, max_bytes: int = MAX_PAGE_BYTES - 700) -> str:
    """Render a past task as a Markdown page body that fits the embedding limit.

    Args:
        task: The past task.
        max_bytes: Budget for the body; the remainder of :data:`MAX_PAGE_BYTES`
            is left for frontmatter.
    """
    date = datetime.fromtimestamp(task.timestamp, UTC).strftime("%Y-%m-%d")
    request = task.task.strip()
    result = html_to_text(task.result)
    header = f"# Task ({date})\n\n"
    body = f"{header}{request}\n\n# Result\n\n{result}\n"
    data = body.encode("utf-8")
    if len(data) > max_bytes:
        body = data[:max_bytes].decode("utf-8", errors="ignore").rstrip() + "\n\n[truncated]\n"
    return body


def build_memory_from_tasks(memory: MemoryDir, tasks: list[PastTask]) -> list[str]:
    """Make *memory* contain exactly one page per task; returns the page names.

    Pages that already exist are left untouched (a task's request and result
    never change, so rewriting would only refresh ``updated`` and force a
    re-embedding); pages for tasks not in *tasks* are deleted. The directory
    is therefore dedicated to the evaluation corpus.

    Args:
        memory: Target memory directory.
        tasks: Tasks to convert.
    """
    wanted = {task.page_name: task for task in tasks}
    existing = set(memory.page_names())
    for name in existing - wanted.keys():
        memory.delete(name)
    for name, task in wanted.items():
        if name in existing:
            continue
        memory.write(
            name,
            task_page_body(task),
            title=task.title,
            extra={"source": f"sorcar.db task_history {task.task_id}"},
        )
    return list(wanted)


class KeywordIndex:
    """FTS5/BM25 keyword search over the pages of a memory directory (baseline).

    Holds one in-memory SQLite connection; call :meth:`close` (or use
    ``closing(KeywordIndex(memory))``) when done.
    """

    def __init__(self, memory: MemoryDir) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE VIRTUAL TABLE pages USING fts5("
            "name UNINDEXED, body, tokenize='porter unicode61')"
        )
        for name in memory.page_names():
            self.conn.execute(
                "INSERT INTO pages (name, body) VALUES (?, ?)", (name, memory.read(name).raw)
            )
        self.conn.commit()

    def search(self, query: str, k: int) -> list[str]:
        """Return up to *k* page names ranked by BM25 for the words in *query*.

        Args:
            query: Natural-language query; punctuation is stripped and terms
                are OR-ed so any overlap counts.
            k: Maximum number of results.
        """
        words = re.findall(r"[A-Za-z0-9_]+", query)
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words)
        rows = self.conn.execute(
            "SELECT name FROM pages WHERE pages MATCH ? ORDER BY rank LIMIT ?", (match, k)
        ).fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        """Release the in-memory database."""
        self.conn.close()


def reciprocal_rank_fusion(rankings: list[list[str]], k: int, constant: int = 60) -> list[str]:
    """Fuse several ranked lists with reciprocal rank fusion.

    Args:
        rankings: Ranked page-name lists from different retrievers.
        k: Number of fused results to return.
        constant: RRF smoothing constant (60 is the value from the original paper).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, name in enumerate(ranking, start=1):
            scores[name] = scores.get(name, 0.0) + 1.0 / (constant + rank)
    ordered = sorted(scores.items(), key=_score_of_item, reverse=True)
    return [name for name, _ in ordered[:k]]


def _score_of_item(item: tuple[str, float]) -> float:
    return item[1]


@dataclass(frozen=True)
class Probe:
    """One evaluation query and the page it must retrieve."""

    query: str
    gold: str
    source: str


# Hand-written probes: later questions a user might plausibly ask, with the
# id prefix of the past task whose page must come back. They are phrased
# differently from the original requests on purpose.
HAND_PROBES: tuple[tuple[str, str], ...] = (
    ("which memory architecture did we conclude is best for an AI agent", "bf690e27"),
    ("how do I put the car through an emissions readiness drive cycle", "f69f2770"),
    ("nonstick cookware brand comparison for safety and durability", "e16312e8"),
    ("is that Chinese cookware brand safe and where is it made", "09171e64"),
    ("how to authenticate the google docs agent", "7a9756ac"),
    ("implement the whatsapp channel agent like the other messaging agents", "86a9f305"),
    ("add a green border around the settings panel", "c3d202a8"),
    ("fix race conditions, hangs and deadlocks across the codebase", "20a1fa03"),
    ("why did the previous run stop responding and hang the web app", "53c1b3d4"),
    ("show images from tool results inline in the event panel", "164dd587"),
    ("publish website changes to the github pages repo", "2f08a2de"),
    ("task classifier should use structured output and not be agentic", "a1406415"),
    ("merge conflict help", "918eb380"),
    ("rename SEA methods to drop the get_ prefix", "72326fb9"),
    ("MCP servers for brave search, notion, postgres, firecrawl and google workspace", "0604f0ce"),
)

PROBE_PROMPT = """\
Below is a memory page written by an AI coding assistant about a past task.
Write ONE short question (8-20 words) that the same user might ask weeks later,
which this page would answer. Paraphrase: do not copy distinctive phrases,
file names or identifiers from the page verbatim. Output only the question.

<page>
{page}
</page>
"""


def hand_probes(tasks: list[PastTask]) -> list[Probe]:
    """Resolve :data:`HAND_PROBES` against the loaded tasks; unknown ids are skipped.

    Args:
        tasks: The evaluation corpus.
    """
    by_prefix = {task.task_id[:8]: task for task in tasks}
    probes: list[Probe] = []
    for query, prefix in HAND_PROBES:
        task = by_prefix.get(prefix)
        if task is None:
            logger.warning("Hand probe skipped: task %s not in corpus", prefix)
            continue
        probes.append(Probe(query=query, gold=task.page_name, source="hand"))
    return probes


def llm_probes(memory: MemoryDir, names: list[str], model_name: str) -> list[Probe]:
    """Ask an LLM to write one paraphrased question per page in *names*.

    Args:
        memory: The memory directory.
        names: Page names to write probes for.
        model_name: Generation model.
    """
    from kiss.core.models.model_info import model as create_model

    probes: list[Probe] = []
    for name in names:
        page = memory.read(name)
        model = create_model(model_name)
        model.initialize(PROBE_PROMPT.format(page=page.raw[:6000]))
        text, _ = model.generate()
        query = text.strip().strip('"').splitlines()[0].strip() if text.strip() else ""
        if not query:
            logger.warning("Empty probe for %s; skipped", name)
            continue
        probes.append(Probe(query=query, gold=name, source="llm"))
    return probes


def cached_llm_probes(memory: MemoryDir, names: list[str], args: argparse.Namespace) -> list[Probe]:
    """Return LLM-written probes, reusing ``llm_probes.json`` when it matches the request.

    The cache records the probe model, seed, count and a hash of the corpus
    page names; any difference (or ``--rebuild``) regenerates the probes.

    Args:
        memory: The memory directory.
        names: Sorted corpus page names.
        args: Parsed CLI arguments (``out``, ``llm_probes``, ``probe_model``,
            ``seed``, ``rebuild``).
    """
    cache_path = args.out.with_name("llm_probes.json")
    meta = {
        "probe_model": args.probe_model,
        "seed": args.seed,
        "count": args.llm_probes,
        "corpus_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
    }
    if cache_path.exists() and not args.rebuild:
        cached = json.loads(cache_path.read_text())
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return [Probe(**p) for p in cached["probes"]]
        logger.info("Probe cache %s does not match this run; regenerating", cache_path)
    sample = random.Random(args.seed).sample(names, min(args.llm_probes, len(names)))
    generated = llm_probes(memory, sample, args.probe_model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"meta": meta, "probes": [asdict(p) for p in generated]}, indent=1)
    )
    return generated


@dataclass
class RetrieverResult:
    """Aggregate metrics for one retriever."""

    name: str
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    family_recall: dict[int, float] = field(default_factory=dict)
    family_mrr: float = 0.0
    mean_latency_ms: float = 0.0
    ranks: list[int | None] = field(default_factory=list)
    family_ranks: list[int | None] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)


def _recall_and_mrr(ranks: list[int | None]) -> tuple[dict[int, float], float]:
    n = len(ranks) or 1
    recall = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in RECALL_KS}
    return recall, sum(1.0 / r for r in ranks if r is not None) / n


def score_ranks(
    name: str,
    ranks: list[int | None],
    latencies: list[float],
    family_ranks: list[int | None] | None = None,
) -> RetrieverResult:
    """Turn per-probe gold ranks into Recall@k and MRR (strict and family-level).

    Args:
        name: Retriever label.
        ranks: 1-based rank of the gold page per probe, or None when absent.
        latencies: Per-probe search latency in milliseconds.
        family_ranks: 1-based rank of the first page from the gold page's
            family per probe; defaults to *ranks*.
    """
    if family_ranks is None:
        family_ranks = list(ranks)
    result = RetrieverResult(
        name=name, ranks=ranks, family_ranks=family_ranks, latencies_ms=latencies
    )
    result.recall, result.mrr = _recall_and_mrr(ranks)
    result.family_recall, result.family_mrr = _recall_and_mrr(family_ranks)
    result.mean_latency_ms = sum(latencies) / (len(latencies) or 1)
    return result


def rank_of(gold: str, ranking: list[str]) -> int | None:
    """1-based position of *gold* in *ranking*, or None."""
    return ranking.index(gold) + 1 if gold in ranking else None


def task_families(tasks: list[PastTask]) -> dict[str, str]:
    """Map each task's page name to its family key (the normalised request text)."""
    return {task.page_name: task.family for task in tasks}


def family_rank_of(gold: str, ranking: list[str], families: dict[str, str]) -> int | None:
    """1-based position of the first page in *ranking* from *gold*'s family, or None.

    Args:
        gold: The gold page name.
        ranking: Ranked page names.
        families: Page name -> family key; pages missing from it are their own family.
    """
    family = families.get(gold, gold)
    for position, name in enumerate(ranking, start=1):
        if families.get(name, name) == family:
            return position
    return None


def duplicate_family_count(families: dict[str, str]) -> int:
    """Number of pages whose family contains more than one page.

    Args:
        families: Page name -> family key.
    """
    counts: dict[str, int] = {}
    for family in families.values():
        counts[family] = counts.get(family, 0) + 1
    return sum(count for count in counts.values() if count > 1)


def run_evaluation(
    memory: MemoryDir,
    probes: list[Probe],
    vector_index: VectorIndex,
    hashed_index: VectorIndex,
    keyword_index: KeywordIndex,
    families: dict[str, str] | None = None,
    k: int = max(RECALL_KS),
) -> tuple[dict[str, RetrieverResult], list[dict[str, object]]]:
    """Run every probe through every retriever.

    Args:
        memory: The memory directory under test.
        probes: Queries with gold pages.
        vector_index: Index built with the neural embedding model.
        hashed_index: Index built with :func:`hashed_embedding`.
        keyword_index: FTS5 baseline.
        families: Page name -> family key for family-level recall; by default
            every page is its own family.
        k: Depth of the ranked lists.

    Returns:
        ``(results_by_retriever, per_probe_rows)``.
    """
    if families is None:
        families = {}
    ranks: dict[str, list[int | None]] = {"vector": [], "hashed": [], "bm25": [], "hybrid": []}
    family_ranks: dict[str, list[int | None]] = {key: [] for key in ranks}
    latencies: dict[str, list[float]] = {key: [] for key in ranks}
    rows: list[dict[str, object]] = []
    for probe in probes:
        t0 = time.perf_counter()
        vector_ranking = [hit.name for hit in vector_index.search(probe.query, k=k)]
        t1 = time.perf_counter()
        hashed_ranking = [hit.name for hit in hashed_index.search(probe.query, k=k)]
        t2 = time.perf_counter()
        bm25_ranking = keyword_index.search(probe.query, k=k)
        t3 = time.perf_counter()
        hybrid_ranking = reciprocal_rank_fusion([vector_ranking, bm25_ranking], k=k)
        t4 = time.perf_counter()
        rankings = {
            "vector": vector_ranking,
            "hashed": hashed_ranking,
            "bm25": bm25_ranking,
            "hybrid": hybrid_ranking,
        }
        for key, ranking in rankings.items():
            ranks[key].append(rank_of(probe.gold, ranking))
            family_ranks[key].append(family_rank_of(probe.gold, ranking, families))
        latencies["vector"].append((t1 - t0) * 1000)
        latencies["hashed"].append((t2 - t1) * 1000)
        latencies["bm25"].append((t3 - t2) * 1000)
        latencies["hybrid"].append(((t1 - t0) + (t3 - t2) + (t4 - t3)) * 1000)
        rows.append(
            {
                "query": probe.query,
                "gold": probe.gold,
                "source": probe.source,
                "gold_title": memory.read(probe.gold).title,
                **{f"rank_{key}": ranks[key][-1] for key in ranks},
                **{f"family_rank_{key}": family_ranks[key][-1] for key in ranks},
                "top_vector": vector_ranking[:3],
                "top_bm25": bm25_ranking[:3],
            }
        )
    results = {
        key: score_ranks(key, ranks[key], latencies[key], family_ranks[key]) for key in ranks
    }
    return results, rows


def format_results_table(results: dict[str, RetrieverResult], n_probes: int) -> str:
    """Render the metrics as a Markdown table.

    Args:
        results: Output of :func:`run_evaluation`.
        n_probes: Number of probes evaluated.
    """
    lines = [
        "| retriever | R@1 | R@3 | R@5 | MRR | family R@1 | family R@3 | family R@5 | family MRR "
        "| mean latency (ms) | probes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results.values():
        lines.append(
            f"| {r.name} | {r.recall[1]:.2f} | {r.recall[3]:.2f} | {r.recall[5]:.2f} | {r.mrr:.2f} "
            f"| {r.family_recall[1]:.2f} | {r.family_recall[3]:.2f} | {r.family_recall[5]:.2f} "
            f"| {r.family_mrr:.2f} | {r.mean_latency_ms:.0f} | {n_probes} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="path to sorcar.db")
    parser.add_argument("--memory-dir", type=Path, default=Path("./tmp/memoryfield-eval/memory"))
    parser.add_argument("--out", type=Path, default=Path("./tmp/memoryfield-eval/results.json"))
    parser.add_argument("--limit", type=int, default=300, help="number of past tasks to index")
    parser.add_argument(
        "--min-result-chars", type=int, default=300, help="skip tasks with shorter results"
    )
    parser.add_argument("--llm-probes", type=int, default=40, help="number of LLM-written probes")
    parser.add_argument("--probe-model", default=DEFAULT_PROBE_MODEL)
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"embedding model name, or '{HASHED_EMBEDDING_MODEL_CODE}' for the offline embedder",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rebuild", action="store_true", help="rewrite pages and indexes")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tasks = load_past_tasks(args.db, args.limit, args.min_result_chars)
    memory = MemoryDir(args.memory_dir)
    if args.rebuild:
        for name in memory.page_names():
            memory.delete(name)
    before = set(memory.page_names())
    names = build_memory_from_tasks(memory, tasks)
    logger.info(
        "Corpus: %d pages in %s (%d written, %d removed)",
        len(names),
        memory.root,
        len(set(names) - before),
        len(before - set(names)),
    )
    families = task_families(tasks)

    if args.embedding_model == HASHED_EMBEDDING_MODEL_CODE:
        vector_index = VectorIndex(
            memory, embed=hashed_embedding, model_code=HASHED_EMBEDDING_MODEL_CODE
        )
    else:
        vector_index = VectorIndex(memory, embed=ModelEmbedder(args.embedding_model))
    hashed_index = VectorIndex(
        memory, embed=hashed_embedding, model_code=HASHED_EMBEDDING_MODEL_CODE
    )
    if args.rebuild:
        vector_index.clear()
        hashed_index.clear()
    t0 = time.perf_counter()
    report = vector_index.sync()
    logger.info("vector index sync: %s in %.1fs", report, time.perf_counter() - t0)
    hashed_index.sync()

    probes = hand_probes(tasks)
    if args.llm_probes > 0:
        probes += cached_llm_probes(memory, sorted(names), args)

    with closing(KeywordIndex(memory)) as keyword_index:
        results, rows = run_evaluation(
            memory, probes, vector_index, hashed_index, keyword_index, families=families
        )
    by_source: dict[str, dict[str, RetrieverResult]] = {}
    for source in sorted({p.source for p in probes}):
        subset = [i for i, p in enumerate(probes) if p.source == source]
        by_source[source] = {
            key: score_ranks(
                key,
                [result.ranks[i] for i in subset],
                [result.latencies_ms[i] for i in subset],
                [result.family_ranks[i] for i in subset],
            )
            for key, result in results.items()
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "corpus_pages": len(names),
                "pages_in_duplicate_families": duplicate_family_count(families),
                "embedding_model": args.embedding_model,
                "probe_model": args.probe_model,
                "results": {key: asdict(r) for key, r in results.items()},
                "results_by_source": {
                    src: {key: asdict(r) for key, r in res.items()}
                    for src, res in by_source.items()
                },
                "probes": rows,
            },
            indent=1,
            default=str,
        )
    )
    print(
        f"\nCorpus: {len(names)} pages from {args.db} "
        f"({duplicate_family_count(families)} pages belong to repeated task families); "
        f"{len(probes)} probes\n"
    )
    print(format_results_table(results, len(probes)))
    for source, res in by_source.items():
        n = sum(1 for p in probes if p.source == source)
        print(f"\nProbes from source '{source}':\n")
        print(format_results_table(res, n))
    print(f"\nDetails: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
