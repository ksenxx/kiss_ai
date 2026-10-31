# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``VectorIndex.search`` with a k-th-best cut returns what an exact scan returns.

float32 BLAS scores only pre-select the rows that are rescored exactly.
With ``s = float32_dot_error(dims)`` bounding each row's error, a true
top-k row can score ``2 * s`` below the k-th best approximate score (its
own error and the k-th row's pointing in opposite directions), but the
floor used to be ``kth_best - s``; it is now ``kth_best - 2 * s``.

Why no test fails on the old floor: dropping a row needs two rows whose
float32 errors are opposite and together exceed ``s``, while every
rounding in a dot product of unit vectors is far below the worst case
``s`` assumes (a brute-force search over 200,000 candidates per query at
3 dimensions never got past ``s - 2 * 2**-24``).  So this test pins the
guarantee itself on a corpus of near-ties, where float32 order and exact
order disagree and every candidate is rescored.
"""

from __future__ import annotations

from pathlib import Path

from kiss.core.memoryfield.index import VectorIndex, hashed_embedding
from kiss.core.memoryfield.pages import MemoryDir

PAGES = 120


def near_tie_embedding(text: str) -> list[float]:
    """One shared topic plus a jitter far below float32 resolution of a dot product."""
    base = hashed_embedding("one shared topic for every page")
    jitter = hashed_embedding(text)
    return [b + 1e-6 * j for b, j in zip(base, jitter, strict=True)]


def test_cut_search_equals_the_uncut_exact_scan(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path / "memory")
    for i in range(PAGES):
        memory.write(f"page-{i}", f"note {i} with its own words {i * 7919}")
    index = VectorIndex(memory, embed=near_tie_embedding)
    index.sync()
    for query in ("first query", "second query", "a third, longer query text"):
        exact = [(hit.name, hit.score) for hit in index.search(query, k=PAGES, min_score=-1.0)]
        assert len(exact) == PAGES
        for k in (1, 2, 5, 17):
            assert [(hit.name, hit.score) for hit in index.search(query, k=k)] == exact[:k]
