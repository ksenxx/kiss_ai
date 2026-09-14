# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""File-based agent memory with a SQLite vector index (the memoryfield pattern).

A memory is a flat directory of short Markdown pages with YAML frontmatter
(:mod:`kiss.core.memoryfield.pages`) plus one regenerable SQLite index of
embeddings (:mod:`kiss.core.memoryfield.index`). :class:`MemoryTools` turns
that into seven agent tools; :mod:`kiss.core.memoryfield.evaluate` measures
recall against real past Sorcar tasks.
"""

from kiss.core.memoryfield.index import (
    DEFAULT_EMBEDDING_MODEL,
    ModelEmbedder,
    SearchHit,
    SyncReport,
    VectorIndex,
    default_embedder,
    hashed_embedding,
)
from kiss.core.memoryfield.pages import MemoryDir, Page
from kiss.core.memoryfield.tools import MEMORY_PROTOCOL, MemoryTools

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "MEMORY_PROTOCOL",
    "MemoryDir",
    "MemoryTools",
    "ModelEmbedder",
    "Page",
    "SearchHit",
    "SyncReport",
    "VectorIndex",
    "default_embedder",
    "hashed_embedding",
]
