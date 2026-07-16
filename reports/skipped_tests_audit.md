# Audit of the 51 Skipped Tests (full-suite run, July 2026)

Goal: confirm each of the 51 skips observed in the full test-suite run is genuinely
environment/credential-gated and not silently masking a real issue.

Method: the full suite (6,662 collected tests) was re-run in 8 parallel shards with
`pytest -rs` to capture every skip reason, then every skip site was read and verified —
by installing the missing dependency and re-running where possible, and by probing the
live provider API where the skip depended on an external service.

## Summary verdict

| # skips | Category | Verdict |
|---|---|---|
| 18 | jsdom not installed (VS Code UI tests) | Genuine env gating — all pass once `npm ci` is run in `src/kiss/agents/vscode` (verified: 220/220 pass) |
| 20 | "No standard config functions" (`test_new_channel_agents.py::test_config_roundtrip`) | **NOT env-gated — test was 100% dead after the ChannelConfig refactor. Fixed: 19/20 revived and passing; 1 (Google Chat) is a genuine by-design skip** |
| 6 | "No required params for this agent" (`test_auth_flow.py`) | Genuine by-design skip (agents have no required credential params) |
| 3 | "Running on macOS — platform check passes" (`test_auth_flow.py`) | Genuine platform gating (tests assert non-macOS error paths) |
| 3 | Optional deps `twilio` / `pynostr` / `mattermostdriver` not installed | Genuine optional-dependency gating (verified by installing and running) |
| 1 | "Embedding model not available … 404" (`test_model_implementations.py`) | **NOT env-gated — permanently skipping and masking a real production bug. Fixed.** |

Bottom line: 30 of the 51 skips were genuinely environment/credential/platform/design
gated. 21 were not: 1 skip masked a real production bug (dead Gemini embedding default)
and 20 skips were a silently dead test (config-roundtrip coverage lost in a refactor).
Both problems were fixed in this task.

## Details per category

### 1. jsdom skips — 18 tests, genuine (verified)

Files: `src/kiss/tests/agents/vscode/test_*` (history panel, fast-complete picker,
run-parallel panel, subagent input, thought panel, web-speech fallback, web server).
Each test checks for `jsdom` under `src/kiss/agents/vscode/node_modules` and skips with
"jsdom is not installed … run `npm install` there".

Verification: ran `npm ci` in `src/kiss/agents/vscode`, then re-ran every affected test
file: **220 passed, 0 failed**. The skips gate only on the local Node dev-dependency
install and mask nothing. (Note: `npm install` — as opposed to `npm ci` — drifts
`package-lock.json`; use `npm ci`.)

### 2. Channel-agent config roundtrip — 20 tests, dead test, FIXED

`test_new_channel_agents.py::test_config_roundtrip` looked for legacy module-level
`_config_path`/`_load_config`/`_clear_config` functions. After the refactor that moved
all 20 channel agents (telegram, discord, signal, msteams, matrix, feishu, line,
mattermost, irc, bluebubbles, imessage, nextcloud_talk, nostr, synology_chat, tlon,
twitch, zalo, phone_control, sms, googlechat) to the shared
`ChannelConfig` object in `_channel_agent_utils.py`, **no module had those functions any
more, so the test skipped for every single agent** — corrupt-config robustness
(bad JSON → `None`, non-dict JSON → `None`, clear/reload semantics) had zero per-agent
coverage. The shared `TestChannelConfig` in `test_refactoring_tasks.py` only covers the
save/load/clear happy path, not corrupt input.

Fix: the test now falls back to the module-level `_config` `ChannelConfig` object
(mirroring the existing `_reset_config` helper in the same file). Result: **19 of the 20
agents now run the roundtrip test and pass**. `googlechat_agent` still skips — genuinely
by design: it stores Google OAuth artifacts (`credentials.json` / `token.json` managed
by google-auth), not a JSON string-dict config, so the dict-config roundtrip semantics
do not apply to it.

### 3. "No required params" auth skips — 6 tests, genuine

`test_auth_flow.py::test_authenticate_rejects_empty_required_params` and
`::test_authenticate_rejects_whitespace_params` skip for `GoogleChatAgent`,
`IMessageAgent`, and `GmailAgent` because their metadata lists
`"required_params": []`. Verified against the agent sources:
`authenticate_googlechat(service_account_json_path: str = "")` (optional param),
`authenticate_imessage()` and `authenticate_gmail()` (no params — platform/OAuth-file
based auth). There is genuinely no required credential parameter whose empty/whitespace
rejection could be tested; nothing is masked.

### 4. macOS platform skips — 3 tests, genuine

`test_auth_flow.py::TestPlatformSpecificAuth` (lines 393/406/418) assert that
iMessage/BlueBubbles auth returns a "requires macOS" error **on non-Darwin platforms**;
on this macOS machine the tested condition cannot occur, so they skip. The macOS-side
(positive) behavior of both agents is covered by the other parametrized tests in the
same file, which do run here. Inherent, correct platform gating.

### 5. Optional third-party SDKs — 3 tests, genuine (verified)

- `test_bughunt_signal_sms_nostr.py` skips 2 tests via
  `unittest.skipIf(find_spec(...) is None)` for `twilio` and `pynostr`.
- `test_bughunt_teams_mm_gchat.py` skips 1 test via `pytest.mark.skipif` for
  `mattermostdriver`.

None of these packages are in `[project] dependencies` in `pyproject.toml` (they appear
only in the type-checker ignore overrides), and the production modules import them
lazily — they are deliberately optional SDKs. Verification: ran the twilio and
mattermostdriver tests with `uv run --with twilio --with mattermostdriver` —
**both pass**. `pynostr` could not even be installed in this environment (its `coincurve`
21.0.0 dependency fails to build), which confirms why it must remain an optional,
skip-gated dependency. No project behavior is masked; the same backends have
non-SDK-dependent tests (bot detection, config handling) that always run.

### 6. Gemini embedding 404 skip — 1 test, MASKED A REAL BUG, FIXED

`test_model_implementations.py::TestGeminiModel::test_get_embedding` requested
`models/text-embedding-005` and skipped whenever the API returned 404. Investigation
showed this skip fired for **everyone, always**, regardless of credentials:

- Live `ListModels` on the Gemini v1beta API returns only `gemini-embedding-001`,
  `gemini-embedding-2`, and `gemini-embedding-2-preview` as embedding models;
  `text-embedding-005` has never existed there (it is a Vertex AI-only name).
- Worse, the production default in `GeminiModel.get_embedding`
  (`src/kiss/core/models/gemini_model.py`) was `text-embedding-004`, which Google
  **shut down on January 14, 2026** (deprecation announced December 3, 2025 — see the
  Gemini API release notes, https://ai.google.dev/gemini-api/docs/changelog). A live
  probe confirmed: `text-embedding-004` → 404, `text-embedding-005` → 404,
  `gemini-embedding-001` → OK (3072-dim vector).

So the "environment-gated" skip was permanently hiding the fact that
`GeminiModel.get_embedding()` with the default model has been broken since mid-January
2026.

Fix (test-first):
1. Rewrote the test to call `m.get_embedding("Hello world")` (the production default)
   and removed the 404→skip guard — reproduced the failure (404 KISSError).
2. Changed the production default in `gemini_model.py` from `text-embedding-004` to
   `gemini-embedding-001` — test now passes and returns a real embedding.
   The test remains credential-gated by the class-level `@requires_gemini_api_key`,
   which is the correct gating; a future Google-side model retirement will now fail
   loudly instead of silently skipping.

Note: `model_info.py` retains `text-embedding-004` in its **name-routing** tables (the
name still routes to the Gemini provider if requested explicitly); those are offline
routing rules exercised by routing tests and are unaffected by the retirement.
`MODEL_INFO.json` already lists `gemini-embedding-001` / `gemini-embedding-2`.

## Changes made

- `src/kiss/core/models/gemini_model.py` — default embedding model
  `text-embedding-004` → `gemini-embedding-001` (production bug fix).
- `src/kiss/tests/core/models/test_model_implementations.py` — Gemini embedding test
  now exercises the production default and no longer skips on 404.
- `src/kiss/tests/agents/channels/test_new_channel_agents.py` — `test_config_roundtrip`
  supports the shared `ChannelConfig` style, reviving 19 previously dead tests.
- `src/kiss/agents/vscode/media/main.js` — fixed a pre-existing prettier lint error
  surfaced while running the mandatory full check.

Verification: affected test files re-run (194 passed + 211 vscode UI tests passed after
the main.js fix; 220 jsdom tests passed with jsdom installed); `uv run check --full`
passes cleanly.

## Expected residual skips going forward

On a machine without the optional extras: 18 (jsdom) + 3 (optional SDKs) + 3 (non-macOS
platform tests, when on macOS) + 6 (no-required-params by design) + 1 (googlechat
OAuth-file config by design) = **31 skips**, every one of them verified genuine.
