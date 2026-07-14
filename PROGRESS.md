# PROGRESS — Why does claude-fable-5 return empty responses? (COMPLETE)

## Root cause (reproduced live, twice)

Full-fidelity replay of the exact failing production request (15KB Sorcar system
prompt + 10KB SWEdefend task prompt from `~/.kiss/sorcar.db`, real
`AnthropicModel` adapter + real Sorcar toolset):

- `claude-fable-5` → **`stop_reason: "refusal"`, ZERO content blocks**
  (usage in=1 out=9), on turn 1 AND on the retry turn.
- `claude-opus-4-8`, identical request → works (thinking + tool_use).

So fable-5's **safety layer refuses** the SWEdefend security-research prompt
("detect these types of attacks...") deterministically; opus-4-8 does not.
Ruled out: adapter thinking config (adaptive + display=summarized is correct),
top-level `cache_control` (a legit typed param in anthropic SDK 0.83.0),
streaming parsing.

## Project code issue (fixed)

The adapter ignored `stop_reason == "refusal"` → returned `([], "", resp)` →
KISSAgent burned a useless "MUST have at least one function call" retry →
raised `_EmptyModelResponseError` blaming "a streaming or reasoning-block
parsing issue" → fell back with the misleading reason "repeated empty
responses" (the System-Prompt panel message the user saw).

## Fix landed (all in this worktree, verified)

1. `src/kiss/core/kiss_error.py`: new `ModelRefusalError(KISSError)`.
1. `src/kiss/core/models/anthropic_model.py`: new `_raise_on_refusal()` called
   from both `generate()` and `generate_and_process_with_tools()` — raises
   `ModelRefusalError` on `stop_reason=="refusal"` (before any conversation
   append, so the fallback model replays clean history).
1. `src/kiss/core/kiss_agent.py` `_run_agentic_loop`: `ModelRefusalError` now
   triggers an IMMEDIATE `_try_switch_to_fallback(reason='a safety refusal (stop_reason="refusal")')` — no wasted retry turn, accurate diagnostic.
1. New e2e tests (no mocks; real anthropic SDK against a local SSE server
   replaying the exact production refusal stream shape):
   `src/kiss/tests/core/models/test_anthropic_refusal_fallback.py` — 6 tests,
   all pass (adapter raises on tools/plain paths, conversation untouched,
   agent swaps primary→fallback with exactly one refused request, no-fallback
   case surfaces `ModelRefusalError`).

## Verification

- Live API: replay through fixed adapter now raises `ModelRefusalError`
  immediately for claude-fable-5 on the exact production request.
- `uv run pytest` on new file: 6 passed; impacted suites
  (test_fable5_fallback, anthropic thinking/interleaved/callback suites,
  empty-response silent-death, no-tool-call loop): 42 + 4 passed.
- `uv run check --full`: ALL checks pass.
- Temp files in ./tmp cleaned; new test file added to git.
