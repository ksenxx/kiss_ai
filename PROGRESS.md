# Progress

- Started a new task to diagnose why reopening VS Code shows the model picker as `gpt-5.5` instead of the model selected in the last session (`claude-opus-4-8`).
- Read `SORCAR.md` first as required; it was empty.
- Cleared the previous task log from `PROGRESS.md` for this new task.
- Read model-picker related code in `SorcarSidebarView.ts`, `SorcarTab.ts`, `server.py`, `commands.py`, `vscode_config.py`, `persistence.py`, `model_info.py`, and `media/main.js`.
- Confirmed the likely root cause: `VSCodeServer.__init__` reads persisted `last_model` once into `self._default_model`, but a long-lived `kiss-web` daemon can survive VS Code window restarts. A new VS Code activation sends `getModels`, and `_get_models()` broadcasts stale `self._default_model` unless that model is invalid. Therefore an old `gpt-5.5` default can overwrite the webview picker even if `config.json` was later changed to `claude-opus-4-8`.
- Read the user's actual `~/.kiss/config.json` and observed `"last_model": "gpt-5.5"`, which explains the currently visible picker state.
- Added regression test `src/kiss/tests/agents/vscode/test_model_picker_last_model_persistence.py` that starts a server with persisted `gpt-5.5`, changes persisted `last_model` to `claude-opus-4-8` while the server remains alive, calls `_get_models()`, and expects the `models` broadcast to select `claude-opus-4-8`.
- Ran the new regression before the fix; it failed as expected with `selected='gpt-5.5'` instead of `claude-opus-4-8`.
- Fixed `VSCodeServer._get_models()` to call `_load_last_model()` on each model-list refresh and update `self._default_model` to the persisted last model whenever it is present in the currently available/runnable model list. Kept the existing invalid-selection recovery afterward.
- Ran impacted model-picker tests after the fix:
  ```bash
  uv run pytest -q \
    src/kiss/tests/agents/vscode/test_model_picker_last_model_persistence.py \
    src/kiss/tests/agents/vscode/test_model_picker_refresh.py \
    src/kiss/tests/agents/vscode/test_new_chat_model_picker.py
  ```
  They passed: `6 passed in 0.85s`.
- Ran `uv run check --full`; ruff reported one long line in the new regression test.
- Wrapped the long `monkeypatch.setattr(vc, "CONFIG_PATH", ...)` line in `test_model_picker_last_model_persistence.py`.
