---
title: Browser stealth and bot-protection handling (Patchright, Xvfb, challenges)
uuid: 5aad799a-0b58-4dfe-9bc2-a3c8177b13d9
summary: 'How web_stealth passes Cloudflare/Akamai/Anubis checks: Patchright, Chrome
  channel, headed Chromium on Xvfb, human mouse/typing, challenge_vendor detection,
  Google-to-Bing fallback.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Browser stealth and bot-protection handling

Motivation recorded in the `web_stealth.py` docstring: 95 hosts in the task history answered
the browser with a bot-protection page. On 14 of those URLs the stock setup (Playwright,
headless Chromium) loaded 4; Patchright with headed Chromium on a virtual display loaded 11.

## Engine

- `playwright_api()` imports `patchright.sync_api` if available, else `playwright.sync_api`.
  Patchright avoids the `Runtime.enable` / `Console.enable` CDP leaks anti-bot scripts look for.
  `playwright_package()` returns the matching package name for `install chromium`.
- `chrome_channel()` returns `"chrome"` when Google Chrome is installed (`google-chrome` on PATH
  or a known install path), else `"chromium"` (full Chromium in new headless mode, not
  chrome-headless-shell).
- No locale, timezone or device-scale override is set (`WebUseTool._context_args`), so the
  browser reports the machine's real values; a fixed timezone next to a different egress IP is
  a fingerprinting signal.

## "Headless" is headed on Xvfb (Linux)

`virtual_display()` starts one Xvfb per process (Linux only, `Xvfb` binary required) and returns
its `DISPLAY`. `_ensure_browser` then launches Chromium *headed* with that display, so nothing
shows on screen yet the browser is not in headless mode. `_chromium_headless` is True only when
no display could be started (macOS, Windows, Linux without Xvfb). Headed windows use
`no_viewport=True`; real headless gets an explicit viewport. `stop_virtual_display` tears the
Xvfb down at exit; a wrapper shell loop kills it if the process dies.

## Human-like input

`mouse_path` makes Bezier pointer paths from the last position (`_mouse_xy`) to the target;
`_human_click` glides onto the element before clicking. `typing_chunks` splits text into
chunks with uneven delays (`type_text` scales its watchdog by `typing_duration_secs`).

## Challenge pages

`go_to_url` calls `_settle_challenge(response)` after navigation:
- Detects an interstitial from the `cf-mitigated: challenge` response header or
  `challenge_vendor(title, body_head)`, which matches title patterns ("Just a moment...",
  "Access Denied", "Robot or human?", ...) and body pattern pairs (Cloudflare, Anubis, Akamai,
  Imperva Incapsula, PerimeterX, Google "unusual traffic").
- Waits an initial `_CHALLENGE_WAIT_SECS` (12 s) with small idle pointer movements for the page to
  clear itself; a Cloudflare Turnstile checkbox is pressed after `_TURNSTILE_REACTION_SECS`, and the
  press extends the deadline to at least 12 s after the click.
- Google "unusual traffic" (`/sorry/`) rates the IP, not the browser, so it is not waited on:
  `search_fallback_url` rewrites a Google `/search?q=` (or the `continue` URL of a `/sorry/`
  page) to `https://www.bing.com/search?q=...`.
- If the page stays blocked, the tool prepends a `Note:` line naming the vendor to the tree,
  so the agent can call `show_browser()` and ask the user instead of retrying blindly.

## Sources
- `src/kiss/agents/sorcar/web_stealth.py` (`playwright_api`, `chrome_channel`, `virtual_display`, `stop_virtual_display`, `mouse_path`, `typing_chunks`, `challenge_vendor`, `search_fallback_url`)
- `src/kiss/agents/sorcar/web_use_tool.py` (`WebUseTool._settle_challenge`, `_challenge_vendor`, `_turnstile_checkbox`, `_human_click`, `_context_args`, `_mask_headless_user_agent`)
- Commit `4bcbd1dc0` "defeat bot-protection blocks with human-like browser automation"
