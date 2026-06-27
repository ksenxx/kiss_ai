# Progress

- Started task: reproduce and fix stale merge/diff UI on remote webapp tab refresh with an end-to-end test.
- Read `SORCAR.md` first as required; it was empty.
- Switched primary work model to `claude-opus-4-7` as requested for coding, bug fixing, and test creation.
- Inspected relevant remote-web merge review code in `src/kiss/agents/vscode/web_server.py` and webview rendering code in `src/kiss/agents/vscode/media/main.js`.
- Added a Python end-to-end regression in `src/kiss/tests/agents/vscode/test_bughunt4_merge_replay_on_reconnect.py` reproducing the bug: on remote webapp refresh, the active `tabId` also appears in `ready.restoredTabs`, causing `_handle_ready` to replay `merge_data`/`merge_started`/`merge_nav` twice. The first diff panel becomes stale because later `merge_nav` updates only the newest merge panel.
- Verified the new Python test failed before the fix with duplicate merge events: `['merge_data', 'merge_started', 'merge_nav', 'merge_data', 'merge_started', 'merge_nav']`.
- Fixed `RemoteAccessServer._handle_ready` to track merge-review tabs replayed during a single `ready` command and skip duplicate replay for restored tabs already replayed as the active tab.
- Ran the new impacted Python test after the fix; it passed. Ran the full impacted Python file `src/kiss/tests/agents/vscode/test_bughunt4_merge_replay_on_reconnect.py`; all 3 tests passed.
- Ran `uv run check --full`; first run exposed a pre-existing Prettier issue in `src/kiss/agents/vscode/src/DependencyInstaller.ts` (missing trailing comma in `showInformationNotification(...)`). Fixed that formatting issue and reran `uv run check --full`; all checks passed.
- Switched to `gpt-5.5` for review. Review finding: a redundant JS jsdom test had been added but did not reproduce the actual server duplicate replay bug and was not included in npm test scripts; removed that redundant file and kept the Python end-to-end WSS test as the authoritative regression.
- Review also noted the restored-tab replay guard should retain the previous non-empty tab-id condition; updated the condition to require `rt_id` to be non-empty before consulting the duplicate-replay set.
