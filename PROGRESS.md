# Task: Halve the gap between composer buttons (burger menu, model picker, attach, promptlet, mic) in the REMOTE web app; test visually.

## Findings so far (no code changes made yet)

- Buttons live in `src/kiss/agents/vscode/media/chat.html` inside `#input-footer > #model-picker`:
  ids `menu-btn` (burger), `model-btn` (model picker pill), `upload-btn` (attach), `tricks-btn` (promptlet), `voice-btn` (mic).
- Remote-only styling lives in `src/kiss/agents/vscode/media/remote-codex.css`; EVERY rule must be scoped under `body.remote-chat` (only the remote page adds this class via `BODY_CLASS_ATTR` in `web_server.py:_build_html`, ~line 2797). The VS Code webview must NOT be affected -> do NOT change main.css.
- main.css line ~1162: `#model-picker { ... gap: 1px; ... }` — the flex gap. remote-codex.css has NO `#model-picker` gap override, so remote uses 1px gap; buttons are 36x36 circles (remote-codex.css ~lines 189-206), model-btn is a 32px-high pill.
- Since gap is only 1px, the VISIBLE gap comes from button padding (36px button around 16px icon = ~10px inner padding per side => ~21px visual icon-to-icon gap). To "halve the gap between buttons", plan: add to remote-codex.css a scoped rule, e.g.
  `body.remote-chat #model-picker { gap: 0; } body.remote-chat #menu-btn, ...#upload-btn, ...#tricks-btn, ...#voice-btn { margin-left: -5px; margin-right: -5px; }` — OR simpler/cleaner: keep 36px hit area but pull buttons together with negative horizontal margins ~ -5px each side so edge-to-edge icon spacing halves (~21px -> ~10.5px). First take a BEFORE screenshot and measure actual bounding boxes via Playwright, then decide exact values (measure gap = box[i+1].x - (box[i].x + box[i].width) between each adjacent pair, halve it).
- Visual test harness pattern (copy from `src/kiss/tests/agents/vscode/test_codex_uniform_font_size.py` lines ~303-410):
  - `_start_live_server(tmp_path, ready, done, state)` thread: imports `RemoteAccessServer, _generate_self_signed_cert` from `kiss.agents.vscode.web_server`; generates cert/key in tmp_path; `RemoteAccessServer(host="127.0.0.1", port=0, work_dir=str(tmp_path), certfile=..., keyfile=..., url_file=tmp_path/"remote-url.json", uds_path=tmp_path/"sorcar.sock")`; `await server.start_async()`; port = `next(iter(server._ws_server.sockets)).getsockname()[1]`; loop until done; `await server.stop_async()`.
  - Playwright: `p.chromium.launch(args=["--ignore-certificate-errors"])`, `browser.new_page(ignore_https_errors=True, viewport={...})`, `page.goto(f"https://127.0.0.1:{port}/", wait_until="domcontentloaded")`, `page.wait_for_selector("#output", state="attached")`.
  - No auth password appears needed in that test flow for page load.

## DONE

- Edited `src/kiss/agents/vscode/media/remote-codex.css`: added body.remote-chat-scoped rule after the `.active` rule giving #menu-btn/#upload-btn/#tricks-btn/#voice-btn `margin-left:-5px; margin-right:-5px` with menu-btn margin-left:0 and voice-btn margin-right:0 (outer edges pinned).
- Measured live with tmp/measure_gaps.py (RemoteAccessServer + Playwright): visible icon gaps BEFORE: 11/11/21/21 px; AFTER: 6/6/11/11 px — halved. Screenshots verified (tmp/composer_before.png, composer_after.png): looks good, no glyph overlap.

## Final verification session (mobile)

- Confirmed no `@media` rule in remote-codex.css overrides the negative margins at mobile widths (only `#output`/`#input-area` padding change at \<=600px), so the halving applies on mobile.
- Ran `uv run pytest src/kiss/tests/agents/vscode/test_remote_composer_button_gap.py -v` (mobile 420x900 viewport): 1 passed.
- Fresh visual check at true mobile viewport (390x844, dpr=2, is_mobile=True) via live RemoteAccessServer + Playwright: measured visible gaps 6/6/11/11 px (historical 11/11/21/21 px) — halved; screenshot of `#input-footer` inspected — burger, model pill, attach, promptlet, mic evenly tight, no glyph overlap, send button intact.
- `uv run check --full`: all checks passed.

## Follow-up task: "can you halve it further?" (DONE)

- Read `remote-codex.css` (lines 183-256) and the existing e2e test; current margins were `-5px` per inner edge, live gaps 6/6/11/11 px.
- Edited `src/kiss/agents/vscode/media/remote-codex.css`: changed the scoped negative margins on #menu-btn/#upload-btn/#tricks-btn/#voice-btn from `-5px` to `-8px` (outer pins `menu-btn { margin-left: 0 }` / `voice-btn { margin-right: 0 }` unchanged), updated the comment. -8px per inner edge halves the once-halved gaps again: pill seams 6 -> 3 px, circle seams 11 -> 5 px.
- Updated `src/kiss/tests/agents/vscode/test_remote_composer_button_gap.py`: renamed test to `test_remote_composer_button_gaps_quartered`, assertion now `gap <= old_gap / 4 + 0.5` (quarter of historical 11/11/21/21 px), still requires gap > 0 and 36x36 touch targets. Test passes.
- Fresh visual check at mobile viewport (390x844, dpr=2, is_mobile=True) via live RemoteAccessServer + Playwright: measured visible gaps 3/3/5/5 px (was 6/6/11/11); screenshots of `#input-footer` and full page inspected — icons tight but distinct, no overlap, touch targets 36x36, model pill/send button intact.
- `uv run check --full`: all checks pass (also mdformat-fixed this PROGRESS.md).

## Remaining plan (was)

1. Write a small scratch script in ./tmp/ (based on the harness above) to launch server + Playwright, measure adjacent gaps between #menu-btn, #model-btn, #upload-btn, #tricks-btn, #voice-btn bounding boxes, and take a BEFORE screenshot of the composer.
1. Edit `remote-codex.css` (Read it first around lines 180-230) adding a `body.remote-chat`-scoped rule that halves the measured gap (likely negative margins or reduced button width, preserving 36px height/hit comfort).
1. Re-run script: verify each adjacent gap ≈ half of before; take AFTER screenshot; visually inspect both screenshots (Read/screenshot tool).
1. Add a permanent e2e test `src/kiss/tests/agents/vscode/test_remote_composer_button_gap.py` using the live-server pattern asserting the halved gaps (both static CSS check + live computed geometry).
1. Run `uv run check --full` and the new test; clean ./tmp; git add the new test; finish.

# Task: Simplify ./install.sh so it never touches kiss-web (restart owned by extension); test by actual installation

## DONE

- `install.sh` step \[5/5\]: removed ALL kiss-web handling (~200 lines): the `uv sync` pre-build of the new extension's venv, the `check-kiss-web-active-tasks.py` guard + `KISS_FORCE_RESTART`, the supervisor-binary (plist/systemd `ExecStart`) probing, the `lsof -nP -ti tcp:8787 -sTCP:LISTEN | xargs kill` stop loop, `rm -f ~/.kiss/sorcar.sock`, the `launchctl kickstart -k` / `systemctl --user restart` defense, and the bounded socket-comeback wait. Replaced with a comment stating kiss-web is deliberately NOT touched: the extension's `DependencyInstaller.restartKissWebDaemon()` restarts it after the `.extension-updated`-marker reload (fingerprint mismatch), deferring while tasks are in flight (`daemonHasActiveTasks`). install.sh now only: installs software (steps 1–3), builds the VSIX (step 4), installs it + writes the marker (step 5).
- Updated stale comments (setsid header item 3, `handle_hup`, exec-tee rationale) that referenced the removed "Stopping old kiss-web daemon" flow; kept all test-pinned needles (`subshell`/`reset`/`trap`, `NOT stuck`, MODEL_INFO.json / INJECTIONS.md rationale comments, BEGIN/END extraction markers, `NPM_CI_FLAGS`, `update_repo()`).
- Tests: deleted `test_install_script_daemon_restart.py` and `test_install_script_kickstart_after_rm.py` (pinned the old kill/kickstart behavior). Rewrote `test_install_script_terminal_freeze_notice.py`: kept freeze-notice-before-`--install-extension` and completes-after-PTY-death e2e tests, added `test_step_5_5_never_touches_kiss_web` (real dummy daemon + logging launchctl/systemctl stubs + pre-existing sorcar.sock: daemon must survive, socket must remain, supervisors never invoked).
- INVARIANTS.md: daemon is started by the extension's DependencyInstaller (never install.sh); new invariant: install.sh MUST NOT touch kiss-web.
- Verification: 15 + 48 impacted tests pass; `uv run check --full` all green.
- ACTUAL INSTALL TEST on this machine: ran `bash install.sh` from the worktree while this very task was running inside the kiss-web daemon (PID 26105). Install completed ("=== Source bootstrap complete ==="), extension ksenxx.kiss-sorcar@2026.7.21 freshly installed (extension dir mtime updated), daemon PID and `~/.kiss/sorcar.sock` inode (184036613) unchanged, zero "Stopping/kickstart/Pre-building/Waiting" lines in the log. The `.extension-updated` marker was consumed by the reloaded extension (`ensureDependencies`), and `~/.kiss/.kiss-web.fingerprint` was NOT rewritten → DependencyInstaller deferred the kiss-web restart because this task is active; the fingerprint mismatch persists, so the extension restarts kiss-web at the next activation/idle — exactly the designed extension-owned restart.
