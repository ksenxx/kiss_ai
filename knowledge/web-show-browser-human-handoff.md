---
title: show_browser, close_browser and handing the browser to a human
uuid: 10447e21-903c-4b3e-9833-7804accb3ad3
summary: show_browser relaunches the same profile visible or headless, carrying cookies
  and URL across; when to hand the browser to a human (login, CAPTCHA); close_browser
  vs close.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# show_browser, close_browser and handing the browser to a human

Browsing starts with no visible window (`headless=True`, implemented as headed Chromium on Xvfb
where possible; see `web-stealth-bot-protection`). Chromium cannot switch between headless and
visible without a restart, so `show_browser(visible)` does a controlled relaunch:

1. If the requested state already holds and the browser is alive, return
   `"Browser is already visible."` / `"... headless."`.
2. `_capture_session()` adopts a tab the user may have opened (`_check_for_new_tab`), records the
   current URL (empty for `about:` pages) and `context.cookies()`.
3. `_close_browser_only()`, flip `self._headless`, relaunch via `_try_ensure_browser`.
4. `_restore_cookies()` adds the cookies back. This matters because a restart drops
   session-only cookies, exactly the ones a login or bot-check flow is in the middle of setting;
   the persistent profile alone would lose them.
5. Re-navigate to the saved URL and return its accessibility tree, or
   `"Browser is now visible."` when no page was open.

A visible launch records the frontmost app first (`_get_frontmost_app`, macOS) and re-activates
it afterwards (`_activate_app`) so the user's focus is restored.

## When to call it

The tool docstring (which is the model-facing description) lists: interactive login or OAuth
consent, a CAPTCHA, an "unusual traffic" bot check, or a user asking to watch. The
`_settle_challenge` note for a non-clearing bot page explicitly tells the agent to call
`show_browser()`. The Sorcar system prompt adds: call `show_browser()` first, then ask the user
for help, then `show_browser(visible=False)` when done. On a headless server with no display a
visible window cannot be shown to anyone; the user must be on the machine.

## close_browser vs close

- `close_browser()` (model tool): closes Chromium only; the next web call relaunches with the
  same profile, so logins persist. Use it when browsing is finished in a long task.
- `close()` (not a model tool): also stops Playwright, unregisters the `atexit` hook and deletes
  an ephemeral profile. `SorcarAgent.run` calls it in its `finally` block.

## Sources
- `src/kiss/agents/sorcar/web_use_tool.py` (`WebUseTool.show_browser`, `_capture_session`, `_restore_cookies`, `close_browser`, `close`, `_get_frontmost_app`, `_activate_app`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`SorcarAgent.run`)
- Commit `c7b9d7bfe` "make browser headless by default with show_browser() escape hatch"
