---
title: Embeddings through the model layer (get_embedding, get_embeddings, emb catalog
  flag)
uuid: c3579353-1c9a-4d5b-bf66-65b89d126718
summary: 'Embeddings in KISS: get_embedding/get_embeddings on OpenAI-compatible and
  Gemini (not Anthropic/CLI/decisions), emb catalog flag, initialize before embedding,
  memoryfield embedder.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Embeddings through the model layer

## API
`Model.get_embedding(text, embedding_model=None) -> list[float]` and
`Model.get_embeddings(texts, embedding_model=None) -> list[list[float]]`. The base
`get_embeddings` loops over `get_embedding`; providers with a batch endpoint override it.

## Per provider
| Adapter | Support | Notes |
|---|---|---|
| `OpenAICompatibleBase` (v1 and v2) | yes | `client.embeddings.create(model=..., input=...)`; `get_embeddings` sends one batched request (at most 2048 inputs, 300k tokens total) and re-orders results by `item.index`. Errors -> `KISSError("Embedding generation failed for model ...")`. |
| `GeminiModel` | yes | `client.models.embed_content`; batch = base loop. |
| `AnthropicModel` | no | raises `KISSError("Anthropic does not provide an embeddings API.")` |
| `CLITextModel` (`cc/*`, `codex/*`) | no | raises `KISSError` |
| `DecisionsModel` | no | not an embedding model |

The model used is `embedding_model` if given, else **the instance's own model name**. So create
the model from the embedding model's catalog name, e.g. `model("text-embedding-3-small")`; asking
a chat model to embed fails rather than silently using some other embedding model (this was an
explicit Gemini fix: it used to default to `gemini-embedding-001`).

## Catalog
Embedding models carry `"emb": true, "gen": false, "fc": false` in `MODEL_INFO.json`, e.g.
`text-embedding-3-small` (context 8191, $0.02 per 1M input, output price 0). `gen: false` keeps
them out of `get_available_models()` and skips cache-pricing defaults. Routing uses the normal
prefix rules (`text-embedding` is an OpenAI prefix).

## Client must exist first
SDK clients are created in `initialize()` (OpenAI via `_ensure_client`, Gemini builds
`genai.Client` there). Call `m.initialize("")` before `get_embedding`, as
`src/kiss/core/memoryfield/index.py` does in its lazy `_client()`:
```python
m = model("text-embedding-3-small"); m.initialize("")
vec = m.get_embedding("hello")
vecs = m.get_embeddings(["a", "b"])
```

## Users in the repo
- `kiss.core.memoryfield.index` embedder: `__call__` -> `get_embedding`, `embed_many` ->
  `get_embeddings` (the persistent memory / knowledge-base search).
- `kiss.scripts.update_models` probes embedding models with `get_embedding("Hello world")`.

## Cost
Embedding calls are not routed through `KISSAgent` accounting; `calculate_cost(name, input, 0)`
applies if a caller wants to bill them.

## Sources
- `src/kiss/core/models/model.py` (`Model.get_embedding`, `Model.get_embeddings`, `CLITextModel.get_embedding`)
- `src/kiss/core/models/openai_compatible_model.py` (`OpenAICompatibleBase.get_embedding`, `OpenAICompatibleBase.get_embeddings`)
- `src/kiss/core/models/gemini_model.py` (`GeminiModel.get_embedding`)
- `src/kiss/core/models/anthropic_model.py` (`AnthropicModel.get_embedding`)
- `src/kiss/core/memoryfield/index.py` (embedder `_client`, `__call__`, `embed_many`)
- `src/kiss/scripts/update_models.py`
