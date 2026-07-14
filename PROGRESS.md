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

## Remaining plan (was)

1. Write a small scratch script in ./tmp/ (based on the harness above) to launch server + Playwright, measure adjacent gaps between #menu-btn, #model-btn, #upload-btn, #tricks-btn, #voice-btn bounding boxes, and take a BEFORE screenshot of the composer.
1. Edit `remote-codex.css` (Read it first around lines 180-230) adding a `body.remote-chat`-scoped rule that halves the measured gap (likely negative margins or reduced button width, preserving 36px height/hit comfort).
1. Re-run script: verify each adjacent gap ≈ half of before; take AFTER screenshot; visually inspect both screenshots (Read/screenshot tool).
1. Add a permanent e2e test `src/kiss/tests/agents/vscode/test_remote_composer_button_gap.py` using the live-server pattern asserting the halved gaps (both static CSS check + live computed geometry).
1. Run `uv run check --full` and the new test; clean ./tmp; git add the new test; finish.
