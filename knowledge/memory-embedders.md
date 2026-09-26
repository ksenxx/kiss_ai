---
title: Memory embedders (OpenAI text-embedding-3-small vs offline hashed-bow-v1)
uuid: afa10fd9-4ea3-4919-ab8f-524ef8ffdd3c
summary: default_embedder picks ModelEmbedder text-embedding-3-small when OPENAI_API_KEY
  is set, else offline hashed_embedding hashed-bow-v1 (1024 dims); separate index
  files per embedder, model_code
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Memory embedders

An `Embedder` is any `Callable[[str], list[float]]`. `VectorIndex` accepts one explicitly or calls
`default_embedder()`.

## default_embedder
- `OPENAI_API_KEY` non-empty in the environment -> `ModelEmbedder()` with
  `DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"`. The daemon loads the API-key store into the
  environment at startup, so this is the normal case under kiss-web.
- Otherwise -> `hashed_embedding`, fully offline.

## ModelEmbedder
Wraps a KISS model from the catalog (`kiss.core.models.model_info.model`), created lazily once
under a lock and initialised with `initialize("")`, so the HTTP pool is reused and it is
thread-safe. `__call__` uses `get_embedding`; `embed_many` uses `get_embeddings` for batched
requests (used by `VectorIndex.sync`). Any catalog embedding model works, e.g.
`gemini-embedding-001`.

## hashed_embedding (hashed-bow-v1)
Deterministic feature hashing of lowercase alphanumeric unigrams plus bigrams with log-scaled
counts into `HASHED_EMBEDDING_DIMS = 1024` dims, normalized to unit length (zeros for empty text).
It captures lexical overlap only; it is the no-key fallback and the baseline a neural embedder
must beat in `memory-recall-evaluation`.

## One index file per embedder
`VectorIndex.model_code` defaults to the embedder's `model_name` attribute, else
`HASHED_EMBEDDING_MODEL_CODE = "hashed-bow-v1"`. The file name is
`model_code_for_filename(model_code) + ".sqlite3"` (non `[A-Za-z0-9._-]` runs become `-`, so
`BAAI/bge-base-en-v1.5` -> `BAAI-bge-base-en-v1.5`). A store used first without a key and later with
one simply grows a second index beside the first. Opening an index whose `meta.model_code` differs
raises `ValueError`.

Gotcha: a custom embedder callable without a `model_name` attribute gets model code
`hashed-bow-v1` unless you pass `model_code=`, which would collide with the real hashed index;
the dimension check in `search` then skips mismatched rows with a warning.

## Sources
- `src/kiss/core/memoryfield/index.py` (`default_embedder`, `ModelEmbedder`, `hashed_embedding`, `model_code_for_filename`, `DEFAULT_EMBEDDING_MODEL`, `HASHED_EMBEDDING_MODEL_CODE`, `HASHED_EMBEDDING_DIMS`)
