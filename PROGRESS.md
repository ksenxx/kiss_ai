# Progress

## Task: review previous model-picker fix via gpt-5.5, write tests for reported bugs, fix using claude-opus-4-7, repeat until no more bugs

### Pass 1 (gpt-5.5 review of original fix)

Found:
- `_get_models` could broadcast `selected` value that is not present in the available model list when only a custom endpoint is configured (because `get_default_model()` cannot name a custom model).
- `_get_models` mutated daemon-global `self._default_model` without synchronization and emitted it directly to the broadcast event.

Fix (`src/kiss/agents/vscode/server.py::_get_models`):
- Wrapped mutation in `_state_lock` and emitted a local `selected` snapshot captured under the lock.
- When the cached/persisted/refreshed model is unavailable but `models_list` is non-empty (custom-only case), pick `models_list[0]` so the picker never emits an unavailable selection.

Test added (`src/kiss/tests/agents/vscode/test_model_picker_last_model_persistence.py::test_get_models_selects_custom_model_when_cached_default_is_invalid`).

### Pass 2 (gpt-5.5 review of pass-1 fix)

Found: race between `_cmd_select_model` and `_get_models`.
- `_cmd_select_model` set `self._default_model = newmodel` inside `_state_lock` but called `_record_model_usage` (which persists via `_save_last_model`) AFTER releasing the lock.
- `_get_models` read `_load_last_model()` OUTSIDE the lock, then inside the lock did `self._default_model = persisted`. A concurrent select left a stale on-disk value that clobbered the just-picked in-memory selection.

Test added (`test_concurrent_get_models_does_not_revert_just_picked_model`) using a `_save_last_model` mock that blocks until signalled, plus thread synchronization.

Fix:
- `src/kiss/agents/vscode/commands.py::_cmd_select_model`: moved `_record_model_usage(model)` INSIDE the `with self._state_lock:` block so disk is written before lock release.
- `src/kiss/agents/vscode/server.py::_get_models`: moved `persisted = _load_last_model()` INSIDE `_state_lock` so it observes the new on-disk value.

### Pass 3 (gpt-5.5 review of pass-2 fix)

Found: same race exists in `_new_chat`.
- `_new_chat` read `_load_last_model()` outside `_state_lock` and assigned `self._default_model = persisted` outside the lock. A concurrent `_cmd_select_model` could be clobbered, with the bug appearing on both the daemon-wide default AND the new tab's `selected_model`.

Test added (`test_concurrent_new_chat_does_not_revert_just_picked_model`).

Fix (`src/kiss/agents/vscode/server.py::_new_chat`):
- Moved `persisted = _load_last_model()` and `self._default_model = persisted` INSIDE the existing `with self._state_lock:` block.
- Snapshotted `welcome_model = self._default_model` under the lock and used the snapshot in the `showWelcome` broadcast (mirrors the `selected` snapshot pattern from `_get_models`).

### Pass 4 (gpt-5.5 review of pass-3 fix)

No new reproducible bugs reported. Lock ordering is consistent (`_state_lock` outer → DB `_rw_lock` / config `_config_lock` inner) matching the established convention elsewhere in the codebase.

### Verification

Final model-picker regression tests:

```text
uv run pytest -q --no-cov src/kiss/tests/agents/vscode/test_model_picker_last_model_persistence.py
4 passed
```

Broader impacted tests:

```text
uv run pytest -q --no-cov src/kiss/tests/agents/vscode/ src/kiss/tests/agents/sorcar/test_change_model.py
1083 passed, 4 skipped, 27 deselected, 436 subtests passed
```

Full test suite, run in 8 parallel splits (vscode-A/B, sorcar-A/B, core, channels, bench/scripts/docker/viz, root tests):

```text
338 + 732 + 711 + 918 + 433 + 474 + 75 + 97 passed
= 3778 passed total
```

Full project checks:

```text
uv run check --full
✅ All checks passed!
```

### Files changed

- `src/kiss/agents/vscode/commands.py` — `_cmd_select_model` now persists under `_state_lock`.
- `src/kiss/agents/vscode/server.py` — `_get_models` reads persisted under lock; `_new_chat` reads persisted under lock and snapshots welcome_model; custom-only fallback for `selected`.
- `src/kiss/tests/agents/vscode/test_model_picker_last_model_persistence.py` — 3 new regression tests (custom-only, concurrent-get-models, concurrent-new-chat).
