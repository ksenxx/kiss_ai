---
title: Webview settings panel and ~/.kiss/config.json round-trip
uuid: f3d07e56-926a-4591-b489-cf3a1ea09af6
summary: Webview Settings drawer (#config-form, cfg-* ids) - getConfig/configData
  load, saveConfig to ~/.kiss/config.json, edited-field tracking, editor-tabs toggle,
  adding a checkbox.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Webview settings panel and ~/.kiss/config.json round-trip

Agent settings are not VS Code settings. They live in `~/.kiss/config.json` (owned by the daemon's config module) and are edited in the webview's Settings drawer, identical in VS Code and the remote webapp. Only `kissSorcar.defaultModel`, `kissSorcar.kissProjectPath` and `kissSorcar.editorTabsMode` are VS Code settings.

## Opening
- In-webview: the gear/menu calls `openSettingsPanel()` in `main.js`.
- From VS Code: command `kissSorcar.openSettings` -> `SorcarPanelManager.openSettings()` or `SorcarSidebarView.openSettingsUI()`, which focuses the chat, waits up to 15 x 200 ms for `ready`, and posts `{type: 'openSettings'}`.
- `openSettingsPanel()` resets `configFormPopulated` and `settingsEditedFields`, collapses the API Keys / Custom Models subpanels, and sends `api.getConfig()` and `api.getMyModels()`.

## Load
The daemon replies `configData` (`config`, `apiKeys`, `machine`). `main.js` `case 'configData'` shows the machine name in `#status-machine`, calls `populateConfigForm(cfg, apiKeys)` and updates recent work dirs. In VS Code the host overwrites `config.work_dir` with the window's workspace folder before relaying (`_installClientListener`). The host also polls `~/.kiss/config.json` every 2 s (`_watchConfigFile`/`_checkConfigFile`, started on `ready`) and re-requests config when `remote_password` changes.

## Save
`closeSettingsPanel()` flushes via `saveSettingsIfPopulated()`: a populated form sends `collectConfigForm()`; a form closed before `configData` arrived sends only the fields the user edited (`collectConfigForm(edited)`); an untouched, unpopulated form sends nothing. `api.saveConfig({config, apiKeys})` is forwarded by the host (`FORWARDED_COMMANDS.saveConfig = ['config', 'apiKeys']`). The daemon merges partial payloads, so omitted keys keep their stored values. Edits are tracked by a delegated `input`/`change` listener calling `markSettingsFieldEdited(id)`, so a later `configData` does not overwrite a field the user is editing.

## Fields (ids in `media/chat.html` `#config-form`)
`cfg-remote-password`, `cfg-max-budget`, `cfg-auto-commit`, `cfg-use-worktree`, `cfg-use-web-tools`, `cfg-classify-tasks`, `cfg-classify-with-decisions`, `cfg-use-memory`, `cfg-editor-tabs-mode`, `cfg-voice-auto-submit`, `cfg-voice-sensitivity`, API key inputs `cfg-key-<ENV_NAME>` (Anthropic, Anthropic workspace id, OpenAI, Z.AI, Moonshot, OpenRouter, Together, Gemini), custom model inputs (`cfg-custom-model-name`, `cfg-custom-endpoint`, `cfg-custom-api-key`, `cfg-custom-headers`), and buttons `cfg-update-btn`, `cfg-server-reset-btn`, `cfg-update-models-btn`. There are no work-dir or memory-dir fields; the work dir is chosen in the separate Working-directory panel, and `memory_dir` is edited in `config.json` directly.

`cfg-editor-tabs-mode` is special: its change posts the host-only message `setEditorTabsMode` (`postToHost`), and the host writes `kissSorcar.editorTabsMode` to the most specific scope that already holds a value.

## Adding a checkbox bound to a config key
1. Add `<label class="config-label config-checkbox"><input type="checkbox" id="cfg-..."> ...</label>` to `#config-form` in `chat.html`.
2. In `main.js`: look up the element, set it in `populateConfigForm`, and emit it in `collectConfigForm` (guarded by the `onlyIds` filter).
3. The daemon's config sanitizer must know the key and its default. No host change is needed.
4. Test in jsdom (e.g. extend `test/configToggleInit.test.js`) and, if the setting has a consumer, with a Python test against the daemon.

## Sources
- `src/kiss/agents/vscode/media/main.js` (`openSettingsPanel`, `closeSettingsPanel`, `saveSettingsIfPopulated`, `populateConfigForm`, `collectConfigForm`, `markSettingsFieldEdited`, `case 'configData'`, `case 'openSettings'`)
- `src/kiss/agents/vscode/media/chat.html` (`#config-form`)
- `src/kiss/agents/vscode/src/SorcarSidebarView.ts` (`openSettingsUI`, `_watchConfigFile`, `_checkConfigFile`, `FORWARDED_COMMANDS`, `case 'setEditorTabsMode'`)
- `src/kiss/agents/vscode/test/configToggleInit.test.js`, `test/settingsWorkDirField.test.js`
