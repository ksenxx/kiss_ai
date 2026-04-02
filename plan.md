# Plan: Remove Novita Models Completely
# Plan: Remove Novita Models Completely
# Plan: Remove Novita Models Completely
# Plan: Remaining Bugs and Race Conditions Still Missing
Remove **all Novita model support** from the project so that Novita is no
longer exposed in runtime configuration, model selection, implementation code,
tests, or user-facing documentation. This plan replaces the earlier Novita bug
entries with a full removal plan.

The removal should be done consistently across the main source tree and the VS
Code `kiss_project` mirror so the repository does not retain dead provider
paths, stale documentation, or failing references.

---

## Scope of removal

The following Novita-related surfaces currently exist and must be removed or
updated:

### Runtime/configuration
- `src/kiss/core/config.py`
- `src/kiss/agents/vscode/kiss_project/src/kiss/core/config.py`

### Model implementation and registration
- `src/kiss/core/models/novita_model.py`
- `src/kiss/core/models/__init__.py`
- `src/kiss/core/models/model_info.py`
- `src/kiss/agents/vscode/kiss_project/src/kiss/core/models/novita_model.py`
- `src/kiss/agents/vscode/kiss_project/src/kiss/core/models/__init__.py`
- `src/kiss/agents/vscode/kiss_project/src/kiss/core/models/model_info.py`

### Tests
- `src/kiss/tests/conftest.py`
- `src/kiss/tests/core/models/test_model_implementations.py`
- `src/kiss/agents/vscode/kiss_project/src/kiss/tests/conftest.py`
- `src/kiss/agents/vscode/kiss_project/src/kiss/tests/core/models/test_model_implementations.py`

### User-facing documentation and provider lists
- `README.md`
- `src/kiss/agents/vscode/README.md`
- `src/kiss/agents/vscode/kiss_project/README.md`
- `src/kiss/core/RELENTLESS_AGENT.md`
- `src/kiss/agents/vscode/kiss_project/src/kiss/core/RELENTLESS_AGENT.md`

---

## Precise code and documentation changes

## 1. Remove the Novita API key from configuration
- Document that any remaining forced-interrupt path is best-effort and must be
  run-id guarded.
- `src/kiss/core/config.py`
- mirror copy under `src/kiss/agents/vscode/kiss_project/src/kiss/core/config.py`

**Current issue:**
The configuration schema still exposes `NOVITA_API_KEY`, which advertises a
provider that should no longer exist in the product.

**Required changes:**
- Delete the `NOVITA_API_KEY` field from the settings/config model.
- Remove its environment-variable description text.
- Verify no helper or validation logic still expects that field.

**Reason:**
Keeping the config key after provider removal creates dead configuration surface
area, confuses users, and encourages unsupported setups.

---

## 2. Remove the Novita model implementation module entirely

## 5. File-cache refresh can overwrite newer state with stale results
- `src/kiss/core/models/novita_model.py`
- mirror copy under `src/kiss/agents/vscode/kiss_project/src/kiss/core/models/novita_model.py`

**Required changes:**
- Delete the Novita model implementation file.
- Ensure no imports anywhere still reference `NovitaModel`.

**Reason:**
Once Novita support is removed from dispatch and model metadata, the provider
implementation becomes dead code and should not remain in-tree.

---

## 3. Remove Novita exports from the lazy-import model package
- Record only to the recording associated with the current run/thread context.
- Pass a recording id/run id through printer calls or store it in thread-local
- `src/kiss/core/models/__init__.py`
- mirror copy under `src/kiss/agents/vscode/kiss_project/src/kiss/core/models/__init__.py`

**Current issue:**
`NovitaModel` is part of the public export list and lazy-import table.

**Required changes:**
- Remove `"NovitaModel"` from `__all__`.
- Remove the `"NovitaModel": "kiss.core.models.novita_model"` lazy-import map
  entry.
- Confirm `__getattr__` or related lazy-loading logic still works cleanly after
  the removal.

**Reason:**
Public exports must match the supported provider surface.

---

## 4. Remove Novita models from the model catalog
serialized.

- `src/kiss/core/models/model_info.py`
- mirror copy under `src/kiss/agents/vscode/kiss_project/src/kiss/core/models/model_info.py`

**Current Novita entries found:**
- `novita/deepseek/deepseek-v3.2`
- `novita/minimax/minimax-m2.5`
- `novita/zai-org/glm-5`

**Required changes:**
- Remove all `novita/...` entries from the model metadata table.
- Remove any comments naming a Novita model as the default or preferred option.
- Remove the `model_name.startswith("novita/")` dispatch branch from the model
  factory.
- Remove the `NovitaModel` import in that dispatch branch.
- Remove the use of `keys.NOVITA_API_KEY` from model construction.
- Re-check nearby provider dispatch ordering so branch simplification does not
  leave unreachable code or formatting debris.

**Reason:**
Model discovery and model construction must no longer recognize Novita-prefixed
model names.

---

## 5. Remove Novita-specific test plumbing

**Files:**
- `src/kiss/tests/conftest.py`
- `src/kiss/tests/core/models/test_model_implementations.py`
- mirror copies under `src/kiss/agents/vscode/kiss_project/src/kiss/tests/...`

**Current issue:**
The test suite still contains provider-detection helpers and skip markers for
`NOVITA_API_KEY`, along with a Novita implementation test case.

**Required changes in `conftest.py`:**
- Remove `has_novita_api_key()`.
- Remove `NOVITA_API_KEY` handling from any helper that maps model names to
  required environment variables.
- Remove `requires_novita_api_key` skip marker.
- Simplify any now-redundant branching created by the removal.

**Required changes in model implementation tests:**
- Remove the Novita parametrized case that expects `NovitaModel`.
- Verify the remaining parametrized cases still cover all supported provider
  branches without duplication gaps.

**Reason:**
Tests should validate only supported providers and must not preserve dead
runtime interfaces.

---

## 6. Remove Novita from README provider lists

**Files:**
- `README.md`
- `src/kiss/agents/vscode/README.md`
- `src/kiss/agents/vscode/kiss_project/README.md`

**Current issue:**
The README files still list Novita and specific Novita-hosted models as
supported.

**Required changes:**
- Remove the `Novita` bullet from model/provider lists.
- Remove any Novita model examples.
- Reword surrounding text if needed so punctuation and list formatting stay
  natural after deletion.

**Reason:**
User-facing docs must not advertise a removed provider.

---

## 7. Remove Novita from provider comparison / marketing text

**Files:**
- `src/kiss/core/RELENTLESS_AGENT.md`
- mirror copy under
  `src/kiss/agents/vscode/kiss_project/src/kiss/core/RELENTLESS_AGENT.md`

**Current issue:**
The document still says the agent runs on providers including `Novita`.

**Required changes:**
- Remove `Novita` from the provider list in the comparison text.
- Read the full sentence after editing to make sure the grammar still works.

**Reason:**
Provider comparisons should reflect actual supported backends.

---

## 8. Audit for any remaining Novita references after removal

**Files:** repository-wide

**Required verification pass:**
After the edits above, run a repository-wide search for all of the following:
- `novita`
- `Novita`
- `NOVITA_API_KEY`
- `novita_model`
- `NovitaModel`

**Expected result:**
- No runtime code references remain.
- No tests reference Novita.
- No docs advertise Novita.
- No mirror-only stale references remain.

**Reason:**
Provider removal is easy to do incompletely because this repository contains a
mirrored source tree under the VS Code extension.

---

## 9. Keep plan consistency with earlier bug list

The previous plan included Novita-specific bug entries about:
- provider prefix stripping,
- unavailable model listing,
- missing Novita handling in helper selection,
- and incorrect Novita labeling.

Since the new goal is **complete removal**, those bug-fix items should no
longer exist as independent work items. They should be replaced conceptually by
this larger removal task because deleting the provider is the root fix.

---

## Implementation order

1. Remove `NOVITA_API_KEY` from config and mirror config.
2. Remove `novita_model.py` and its mirror copy.
3. Remove `NovitaModel` exports from both `models/__init__.py` files.
4. Remove Novita entries and dispatch logic from both `model_info.py` files.
5. Remove Novita test helpers/markers/cases from both test trees.
6. Remove Novita mentions from all README and `RELENTLESS_AGENT.md` files.
7. Run a final repository-wide search to confirm there are no lingering
   references.

This order prevents transient broken imports while the deletion is in progress.
`yaml.add_representer(str, _str_presenter)` changes PyYAML’s global representer
for **all** string dumping in the process. Any other module using `yaml.dump`
now inherits this formatting side effect.

**Root cause:**
Global serializer customization is being registered at import time rather than on
an isolated dumper instance.
- No `NOVITA_API_KEY` field exists in the config model.
- No `NovitaModel` symbol is exported or imported anywhere.
- No `novita/...` models remain in model metadata.
- The model factory rejects Novita names naturally because no Novita branch
  remains.
- No tests reference Novita-specific skip markers or env vars.
- No README or marketing text mentions Novita.
- The VS Code `kiss_project` mirror matches the primary source tree.
- A repository-wide grep for Novita-related strings returns no relevant hits.

---

## Notes on simplicity

This should be implemented as a **true deletion**, not as a deprecated stub:
- do not keep compatibility aliases,
- do not keep hidden config keys,
- do not keep unreachable dispatch branches,
- and do not keep skipped tests for a removed provider.

A clean removal is simpler, easier to maintain, and avoids future regressions
where a partially removed provider accidentally reappears in docs or UI.
`NovitaModel` is part of the public export list and lazy-import table.

**Required changes:**
- Remove `"NovitaModel"` from `__all__`.
- Remove the `"NovitaModel": "kiss.core.models.novita_model"` lazy-import map
  entry.
- Confirm `__getattr__` or related lazy-loading logic still works cleanly after
  the removal.

**Reason:**
Public exports must match the supported provider surface.

---
- `src/kiss/core/models/model_info.py`
- mirror copy under `src/kiss/agents/vscode/kiss_project/src/kiss/core/models/model_info.py`

**Current Novita entries found:**
- `novita/deepseek/deepseek-v3.2`
- `novita/minimax/minimax-m2.5`
- `novita/zai-org/glm-5`

**Required changes:**
- Remove all `novita/...` entries from the model metadata table.
- Remove any comments naming a Novita model as the default or preferred option.
- Remove the `model_name.startswith("novita/")` dispatch branch from the model
  factory.
- Remove the `NovitaModel` import in that dispatch branch.
- Remove the use of `keys.NOVITA_API_KEY` from model construction.
- Re-check nearby provider dispatch ordering so branch simplification does not
  leave unreachable code or formatting debris.

**Reason:**
Model discovery and model construction must no longer recognize Novita-prefixed
model names.

---

## 5. Remove Novita-specific test plumbing

**Files:**
- `src/kiss/tests/conftest.py`
- `src/kiss/tests/core/models/test_model_implementations.py`
- mirror copies under `src/kiss/agents/vscode/kiss_project/src/kiss/tests/...`

**Current issue:**
The test suite still contains provider-detection helpers and skip markers for
`NOVITA_API_KEY`, along with a Novita implementation test case.

**Required changes in `conftest.py`:**
- Remove `has_novita_api_key()`.
- Remove `NOVITA_API_KEY` handling from any helper that maps model names to
  required environment variables.
- Remove `requires_novita_api_key` skip marker.
- Simplify any now-redundant branching created by the removal.

**Required changes in model implementation tests:**
- Remove the Novita parametrized case that expects `NovitaModel`.
- Verify the remaining parametrized cases still cover all supported provider
  branches without duplication gaps.

**Reason:**
Tests should validate only supported providers and must not preserve dead
runtime interfaces.

---

## 6. Remove Novita from README provider lists

**Files:**
- `README.md`
- `src/kiss/agents/vscode/README.md`
- `src/kiss/agents/vscode/kiss_project/README.md`

**Current issue:**
The README files still list Novita and specific Novita-hosted models as
supported.

**Required changes:**
- Remove the `Novita` bullet from model/provider lists.
- Remove any Novita model examples.
- Reword surrounding text if needed so punctuation and list formatting stay
  natural after deletion.

**Reason:**
User-facing docs must not advertise a removed provider.

---

## 7. Remove Novita from provider comparison / marketing text

**Files:**
- `src/kiss/core/RELENTLESS_AGENT.md`
- mirror copy under
  `src/kiss/agents/vscode/kiss_project/src/kiss/core/RELENTLESS_AGENT.md`

**Current issue:**
The document still says the agent runs on providers including `Novita`.

**Required changes:**
- Remove `Novita` from the provider list in the comparison text.
- Read the full sentence after editing to make sure the grammar still works.

**Reason:**
Provider comparisons should reflect actual supported backends.

---

## 8. Audit for any remaining Novita references after removal

**Files:** repository-wide

**Required verification pass:**
After the edits above, run a repository-wide search for all of the following:
- `novita`
- `Novita`
- `NOVITA_API_KEY`
- `novita_model`
- `NovitaModel`

**Expected result:**
- No runtime code references remain.
- No tests reference Novita.
- No docs advertise Novita.
- No mirror-only stale references remain.

**Reason:**
Provider removal is easy to do incompletely because this repository contains a
mirrored source tree under the VS Code extension.

---

## 9. Keep plan consistency with earlier bug list

The previous plan included Novita-specific bug entries about:
- provider prefix stripping,
- unavailable model listing,
- missing Novita handling in helper selection,
- and incorrect Novita labeling.

Since the new goal is **complete removal**, those bug-fix items should no
longer exist as independent work items. They should be replaced conceptually by
this larger removal task because deleting the provider is the root fix.

---

## Implementation order

1. Remove `NOVITA_API_KEY` from config and mirror config.
2. Remove `novita_model.py` and its mirror copy.
3. Remove `NovitaModel` exports from both `models/__init__.py` files.
4. Remove Novita entries and dispatch logic from both `model_info.py` files.
5. Remove Novita test helpers/markers/cases from both test trees.
6. Remove Novita mentions from all README and `RELENTLESS_AGENT.md` files.
7. Run a final repository-wide search to confirm there are no lingering
   references.

This order prevents transient broken imports while the deletion is in progress.
- `novita`
- `Novita`
- `NOVITA_API_KEY`
- `novita_model`
- `NovitaModel`

**Expected result:**
- No runtime code references remain.
- No tests reference Novita.
- No docs advertise Novita.
- No mirror-only stale references remain.

**Reason:**
Provider removal is easy to do incompletely because this repository contains a
mirrored source tree under the VS Code extension.

---

## 9. Keep plan consistency with earlier bug list

The previous plan included Novita-specific bug entries about:
- provider prefix stripping,
- unavailable model listing,
- missing Novita handling in helper selection,
- and incorrect Novita labeling.

Since the new goal is **complete removal**, those bug-fix items should no
longer exist as independent work items. They should be replaced conceptually by
this larger removal task because deleting the provider is the root fix.

---

## Implementation order

1. Remove `NOVITA_API_KEY` from config and mirror config.
2. Remove `novita_model.py` and its mirror copy.
3. Remove `NovitaModel` exports from both `models/__init__.py` files.
4. Remove Novita entries and dispatch logic from both `model_info.py` files.
5. Remove Novita test helpers/markers/cases from both test trees.
6. Remove Novita mentions from all README and `RELENTLESS_AGENT.md` files.
7. Run a final repository-wide search to confirm there are no lingering
   references.

This order prevents transient broken imports while the deletion is in progress.
- `src/kiss/agents/sorcar/sorcar_agent.py`
- `src/kiss/agents/sorcar/stateful_sorcar_agent.py`
- `src/kiss/agents/create_and_optimize_agent/agent_evolver.py`
- many channel agent CLI `main()` functions

**Bug:**
Multiple entry points do:
- `old_cwd = os.getcwd()`
- `os.chdir(work_dir)`
- run agent
- `os.chdir(old_cwd)`

`os.chdir()` is process-global, so concurrent threads or embedded use can break
file resolution in unrelated code.

**Root cause:**
Working-directory control is implemented with global process mutation instead of
passing explicit paths to file/subprocess operations.

**Fix plan:**
- Remove `os.chdir()` from agent and channel CLIs.
- Pass `work_dir` explicitly to tools/subprocesses and use absolute paths.
- Keep cwd unchanged for the host process.

---

- No `NOVITA_API_KEY` field exists in the config model.
- No `NovitaModel` symbol is exported or imported anywhere.
- No `novita/...` models remain in model metadata.
- The model factory rejects Novita names naturally because no Novita branch
  remains.
- No tests reference Novita-specific skip markers or env vars.
- No README or marketing text mentions Novita.
- The VS Code `kiss_project` mirror matches the primary source tree.
- A repository-wide grep for Novita-related strings returns no relevant hits.

---

## Notes on simplicity

This should be implemented as a **true deletion**, not as a deprecated stub:
- do not keep compatibility aliases,
- do not keep hidden config keys,
- do not keep unreachable dispatch branches,
- and do not keep skipped tests for a removed provider.

A clean removal is simpler, easier to maintain, and avoids future regressions
where a partially removed provider accidentally reappears in docs or UI.
## 18. `AgentEvolver._evaluate_variant()` mutates `sys.path` and `sys.modules` globally during evaluation

**File:** `src/kiss/agents/create_and_optimize_agent/agent_evolver.py`

**Bug:**
Variant evaluation temporarily inserts into `sys.path`, injects a module into
`sys.modules`, and also changes cwd. If evaluations ever overlap, or if the
process hosts other Python activity, imports can resolve against the wrong
variant code.

**Root cause:**
Evaluation isolation relies on global interpreter state mutation.

**Fix plan:**
- Remove cwd changes.
- Isolate variant execution in a subprocess, or at minimum in a cleaner import
  boundary that does not leak into global `sys.path`/`sys.modules`.
- Ensure each evaluation has explicit filesystem and import context.

---

## 19. Channel daemon still mutates `StatefulSorcarAgent._chat_id` directly

**File:** `src/kiss/channels/background_agent.py`

**Bug:**
The daemon resumes sessions by directly assigning `agent._chat_id`. That bypasses
agent invariants and couples the daemon to private state.

**Root cause:**
Session restoration lacks a public API for setting/resuming by chat id.

**Fix plan:**
- Add a documented public method on `StatefulSorcarAgent` for resuming by
  `chat_id`.
- Use that method from the daemon.
- Keep private attributes private.

---

## 20. Channel daemon sender state maps are still unsynchronized

**File:** `src/kiss/channels/background_agent.py`

**Bug:**
`_sender_locks` and `_sender_chat_ids` are mutated from the polling thread and
read/written from handler threads without any protecting lock. The code relies
on incidental GIL behavior and still has check-then-insert races.

**Root cause:**
Per-sender session bookkeeping uses shared dicts with no synchronization.

**Fix plan:**
- Protect sender maps with a dedicated lock.
- Atomically create/fetch the sender lock and chat id mapping.
- Consider replacing parallel dicts with one sender-state structure.

---

## 21. Channel daemon spawns unbounded detached threads

**File:** `src/kiss/channels/background_agent.py`

**Bug:**
`_dispatch_message()` starts a new daemon thread for every inbound message, even
for senders that are already busy. Under message bursts this creates needless
thread churn and potential thread storms.

**Root cause:**
Busy-skipping is checked inside the worker thread after the thread has already
been created.

**Fix plan:**
- Check sender busy state before spawning a thread.
- Use a bounded executor or per-sender worker queue.
- Ensure the daemon has predictable concurrency limits.

---

## 22. Channel daemon `stop()` does not unblock reconnect sleeps or poll-loop sleeps promptly

**File:** `src/kiss/channels/background_agent.py`

**Bug:**
`run()` and `_connect_and_poll()` use `time.sleep(...)` directly for reconnect
backoff and polling. Calling `stop()` sets an event but does not interrupt those
sleeps, so shutdown can be delayed by up to the full sleep interval/backoff.

**Root cause:**
Stop-aware waiting is implemented with raw sleep instead of
`Event.wait(timeout)`.

**Fix plan:**
- Replace all daemon sleeps with `_stop_event.wait(timeout)`.
- Exit immediately when stop is requested during backoff or idle polling.

---

## 23. Several channel backends have infinite `wait_for_reply()` loops with no shutdown or timeout path

**Files:** representative examples
- `src/kiss/channels/slack_agent.py`
- `src/kiss/channels/irc_agent.py`
- `src/kiss/channels/whatsapp_agent.py`
- `src/kiss/channels/synology_chat_agent.py`
- `src/kiss/channels/line_agent.py`
- `src/kiss/channels/zalo_agent.py`
- `src/kiss/channels/tlon_agent.py`

**Bug:**
Many backends implement `wait_for_reply()` as `while True: sleep(); poll();`
without timeout, cancellation, or daemon stop integration. If a user never
replies, the agent can block forever.

**Root cause:**
Reply waiting is modeled as an endless busy-poll loop.

**Fix plan:**
- Add timeout and stop-event parameters to the ChannelBackend contract or an
  equivalent backend-owned cancellation mechanism.
- Make all backend implementations honor it.
- Return a structured timeout result instead of hanging indefinitely.

---

## 24. IRC backend reader thread/socket lifecycle is incomplete and can leak resources

**File:** `src/kiss/channels/irc_agent.py`

**Bug:**
`IRCChannelBackend.connect()` starts `_reader_thread`, but there is no explicit
close/disconnect lifecycle integrated with the daemon. The loop uses
`while self._sock:` and blocking `recv()`, so shutdown/reconnect can leave old
reader threads and sockets around until the socket fails on its own.

**Root cause:**
The backend protocol lacks a disconnect/cleanup hook, and the IRC backend starts
background I/O without lifecycle management.

**Fix plan:**
- Add backend cleanup/disconnect support.
- Ensure daemon reconnect/stop closes backend resources before reconnecting.
- Make the IRC reader loop exit promptly on shutdown.

---

## 25. WhatsApp webhook server lifecycle is unmanaged and can conflict on reconnect

**File:** `src/kiss/channels/whatsapp_agent.py`

**Bug:**
`WhatsAppChannelBackend.connect()` starts an embedded `HTTPServer` on port 8080
but there is no corresponding shutdown. Reconnects or repeated daemon starts can
leave the old server alive and cause bind conflicts.

**Root cause:**
Background server startup is not paired with deterministic teardown.

**Fix plan:**
- Add backend disconnect/cleanup support.
- Shut down the webhook server on daemon stop/reconnect.
- Avoid hardcoding one shared port when multiple instances may exist.

---

## 26. Relentless-agent summarizer temp file is created outside the project temp area

**File:** `src/kiss/core/relentless_agent.py`

**Bug:**
The summarizer fallback uses `tempfile.mkstemp(...)` directly, creating temp
files outside the project’s `WORK_DIR/tmp` convention. That is inconsistent with
project rules and makes cleanup/locality harder.

**Root cause:**
Temporary artifact creation is ad hoc.

**Fix plan:**
- Centralize temp-file creation under the configured work/project temp dir.
- Keep continuation/summarizer artifacts in a predictable location and clean
  them up reliably.

---

## 27. Docker manager still has no timeout/cancellation path for non-streaming execs

**File:** `src/kiss/docker/docker_manager.py`

**Bug:**
`DockerManager.Bash()` calls `container.exec_run(...)` synchronously with no
timeout or cooperative stop support. A hanging command inside the container can
block forever.

**Root cause:**
Docker command execution does not mirror the timeout/stop semantics of the host
`UsefulTools.Bash()` implementation.

**Fix plan:**
- Add timeout support to Docker bash execution.
- Add cancellation/cleanup of hanging exec sessions where possible.
- Keep behavior aligned with non-Docker Bash tool semantics.

---

## 28. Docker streaming and non-streaming output formatting still inject unconditional blank separators

**File:** `src/kiss/docker/docker_manager.py`

**Bug:**
Both Bash paths build output as `stdout + "\n" + stderr` even when one side is
empty. That creates spurious blank lines and can mis-shape empty-output results.

**Root cause:**
Output concatenation is unconditional rather than join-on-present-parts.

**Fix plan:**
- Join only non-empty stdout/stderr parts.
- Keep exit-code formatting separate from payload formatting.

---

## 29. `UsefulTools._clean_env()` caches the environment forever and can use stale credentials/config

**File:** `src/kiss/agents/sorcar/useful_tools.py`

**Bug:**
`_clean_env()` caches a copy of `os.environ` the first time it is called and
reuses it forever. If the process updates environment variables later (for
example credentials, PATH adjustments, or test setup), child commands launched
through Bash still see stale values.

**Root cause:**
Environment sanitization is implemented as an immortal cache over mutable global
state.

**Fix plan:**
- Either rebuild the clean environment per call, or cache only pieces proven
  immutable.
- Keep the `VIRTUAL_ENV` stripping behavior without freezing the whole env.

---

## 30. Keep the Novita-removal work as a separate implementation track, not as the only plan
- No `NOVITA_API_KEY` field exists in the config model.
- No `NovitaModel` symbol is exported or imported anywhere.
- No `novita/...` models remain in model metadata.
- The model factory rejects Novita names naturally because no Novita branch
  remains.
- No tests reference Novita-specific skip markers or env vars.
- No README or marketing text mentions Novita.
- The VS Code `kiss_project` mirror matches the primary source tree.
- A repository-wide grep for Novita-related strings returns no relevant hits.

---

## Notes on simplicity

This should be implemented as a **true deletion**, not as a deprecated stub:
- do not keep compatibility aliases,
- do not keep hidden config keys,
- do not keep unreachable dispatch branches,
- and do not keep skipped tests for a removed provider.

A clean removal is simpler, easier to maintain, and avoids future regressions
where a partially removed provider accidentally reappears in docs or UI.
   - safe stop semantics,
   - synchronized generation checks,
   - file-refresh sequencing,
   - nonblocking user-answer handoff.
3. Fix browser-printer recording ownership and context reset.
4. Fix global-state hazards:
   - `Base.global_budget_used` locking,
   - YAML representer isolation,
   - package-relative `SORCAR.md` loading.
5. Remove process-global cwd mutation from CLIs and evaluation paths.
6. Fix channel daemon synchronization, bounded concurrency, and stop-aware waits.
7. Add backend cleanup/disconnect lifecycle and cancelable `wait_for_reply()`
   across channels.
8. Fix Docker timeout/cancellation and output formatting behavior.
9. Reattach the Novita-removal plan as a separate product cleanup track.

---

## Validation checklist

After implementation, verify all of the following:

- No `NOVITA_API_KEY` field exists in the config model.
- No `NovitaModel` symbol is exported or imported anywhere.
- No `novita/...` models remain in model metadata.
- The model factory rejects Novita names naturally because no Novita branch
  remains.
- No tests reference Novita-specific skip markers or env vars.
- No README or marketing text mentions Novita.
- The VS Code `kiss_project` mirror matches the primary source tree.
- A repository-wide grep for Novita-related strings returns no relevant hits.

---

## Notes on simplicity

This should be implemented as a **true deletion**, not as a deprecated stub:
- do not keep compatibility aliases,
- do not keep hidden config keys,
- do not keep unreachable dispatch branches,
- and do not keep skipped tests for a removed provider.

A clean removal is simpler, easier to maintain, and avoids future regressions
where a partially removed provider accidentally reappears in docs or UI.
- Two concurrent VS Code `run` commands cannot start two tasks.
- `stop` cannot interrupt a newer run after an older run exits.
- Follow-up suggestions and persisted events attach to the correct run even when
  task text is duplicated.
- SQLite persistence remains correct under concurrent history/event/file/model
  updates.
- History substring search treats `%` and `_` literally.
- No process-global `os.chdir()` is required for agent/channel CLIs.
- Channel daemon shutdown is prompt even during backoff/poll sleep.
- Channel backends do not block forever in `wait_for_reply()`.
- IRC/WhatsApp background resources are cleaned up on stop/reconnect.
- Global budget accounting is atomic under concurrent agents.
- YAML dumping changes do not leak globally to unrelated code.
- `SORCAR.md` loading no longer depends on the current working directory.
- Docker Bash commands can be timed out/cancelled and format output cleanly.
- Main tree and VS Code mirror stay synchronized.
