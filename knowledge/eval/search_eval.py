"""Measure how well memory search answers a fixed question set over a knowledge base.

Each question in the YAML file lists the pages that answer it (``gold``). The script
syncs the same index the agent's ``memory_search`` tool uses (``VectorIndex`` over
``MemoryDir``), runs every question, and prints the rank of the best gold page,
Recall@1/3/5 and MRR. It exits 1 when Recall@5 is below ``--min-recall5``.

Run::

    uv run python knowledge/eval/search_eval.py            # OpenAI embedder if key set
    uv run python knowledge/eval/search_eval.py --hashed   # offline embedder
"""

import argparse
import sys
from pathlib import Path

import yaml

from kiss.core.memoryfield import MemoryDir, VectorIndex, hashed_embedding
from kiss.core.memoryfield.index import HASHED_EMBEDDING_MODEL_CODE

HERE = Path(__file__).resolve().parent
FETCH_K = 10  # results fetched per question; must be >= 5 for R@5


def best_rank(ranked: list[str], gold: list[str]) -> int:
    """Return the 1-based rank of the first gold page in *ranked*, or 0 when absent.

    Args:
        ranked: Page names in search order.
        gold: Page names that answer the question.
    """
    for i, name in enumerate(ranked, 1):
        if name in gold:
            return i
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the question set and print per-question ranks and summary metrics.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 when Recall@5 reaches ``--min-recall5``, else 1.
    """
    parser = argparse.ArgumentParser(description="Search test for a memory knowledge base.")
    parser.add_argument("--dir", type=Path, default=HERE.parent, help="knowledge directory")
    parser.add_argument("--questions", type=Path, default=HERE / "questions.yaml")
    parser.add_argument("--hashed", action="store_true", help="use the offline embedder")
    parser.add_argument("--min-recall5", type=float, default=0.0)
    args = parser.parse_args(argv)

    memory = MemoryDir(args.dir)
    if args.hashed:
        index = VectorIndex(memory, embed=hashed_embedding, model_code=HASHED_EMBEDDING_MODEL_CODE)
    else:
        index = VectorIndex(memory)
    index.sync()
    questions = yaml.safe_load(args.questions.read_text())
    unknown = sorted({g for q in questions for g in q["gold"]} - set(memory.page_names()))
    if unknown:
        print(f"Gold pages missing from {memory.root}: {', '.join(unknown)}")
        return 1

    ranks = []
    for item in questions:
        ranked = [hit.name for hit in index.search(item["q"], k=FETCH_K)]
        rank = best_rank(ranked, item["gold"])
        ranks.append(rank)
        mark = f"#{rank}" if rank else "miss"
        top = ranked[0] if ranked else "-"
        print(f"{mark:>5}  {item['q']}  (top: {top})")

    n = len(ranks)
    recall = {k: sum(1 for r in ranks if 0 < r <= k) / n for k in (1, 3, 5)}
    mrr = sum(1 / r for r in ranks if r) / n
    print(
        f"\n{n} questions, {len(memory.page_names())} pages, embedder {index.model_code}: "
        f"R@1 {recall[1]:.2f}  R@3 {recall[3]:.2f}  R@5 {recall[5]:.2f}  MRR {mrr:.2f}"
    )
    return 0 if recall[5] >= args.min_recall5 else 1


if __name__ == "__main__":
    sys.exit(main())
