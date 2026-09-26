---
title: Browser launch, persistent profile, locks and crash/hang recovery
uuid: 1542e623-6b82-46e0-991c-68619643eb12
summary: 'WebUseTool launch: ~/.kiss/browser_profile, _N escalation dirs, cross-process
  profile lock, SingletonLock cleanup, Chromium auto-install, crash and input-hang
  watchdogs.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Browser launch, persistent profile, locks and crash/hang recovery

## Lazy launch

No browser starts until a tool needs one. Every public tool except `close_browser` (which never launches one)
calls `_try_ensure_browser`, which calls `_ensure_browser`:
- If the current page is alive, return. If the page died but the context still has open pages,
  adopt the newest one.
- Otherwise tear down (`_close_browser_only`) and launch. Launch args include
  `--disable-blink-features=AutomationControlled`, `--disable-dev-shm-usage`,
  `--window-size=<viewport>` and `--ignore-gpu-blocklist` (so WebGL exists without a GPU).
- If launch fails with "Executable doesn't exist" / "playwright install" / "patchright install",
  it runs `python -m <patchright|playwright> install chromium` (900 s timeout) and retries. Other
  errors are re-raised so the real cause (profile lock, missing libs) is not hidden.

## Profile directory

- Default: `kiss_home()/browser_profile` (that is `~/.kiss/browser_profile`, or
  `$KISS_HOME/browser_profile`). It persists logins and bot-check clearance cookies.
- `user_data_dir=None` gives a non-persistent context (`launcher.launch` + `new_context`).
- `ephemeral=True` creates a `kiss_web_profile_*` temp dir, deleted by `close()`. Sorcar uses
  this for sub-agents (see `web-sorcar-integration`).
- If the profile is locked by a live Chromium (`_is_profile_in_use` reads the `SingletonLock`
  symlink target), `_resolve_user_data_dir` tries `<dir>_1`, `<dir>_2`, ... up to 99.
  `_cleanup_stale_escalation_dirs` deletes `_N` dirs whose `SingletonLock` pid is provably
  dead (never the base profile, live or lock-less dirs), so they do not pile up after crashes.
- `_launch_browser` runs under a process-local `_LAUNCH_LOCK` and a machine-wide file lock
  `<profile>.lock` (`_profile_lock`). Without the cross-process lock two kiss processes could
  both see a free profile, one would delete the other's live `SingletonLock`, and Chromium would
  either open one profile twice (corrupting logins) or fail with "Failed to create a
  ProcessSingleton".
- `_clean_singleton_locks` deletes stale `SingletonLock/Cookie/Socket` before launch.

After launch the context routes `accounts.google.com` requests to `route.abort()`
(`_ACCOUNTS_GOOGLE_URL_RE`, `_abort_route`), and `_mask_headless_user_agent` rewrites the
`HeadlessChrome` UA token to `Chrome` when real headless mode is used.

## Crash and hang recovery

- `_on_page_crash`: clears `_page`/`_elements` only when the crashed page is the current one
  (background tab crashes are ignored), keeping the context so the main process shuts down
  cleanly. The next tool call relaunches or adopts.
- `_on_browser_lost`: drops references on context close/browser exit.
- Raw input calls (`keyboard.press/type`, `mouse.move/wheel`, `Locator.count`) have no timeout.
  `_require_responsive_renderer` probes with `wait_for_function("() => 1")` first, and
  `_input_hang_watchdog` arms a `threading.Timer` (`_INPUT_WATCHDOG_SECS` = 15 s) that kills the
  Chromium pid (`_watchdog_kill`, escalating SIGTERM to SIGKILL, identity-checked) if the input
  wedges; the tool returns an error and the next call relaunches. Added in commit "harden
  WebUseTool against renderer hangs on unresponsive pages".
- `close()` / `_close_browser_only` use `_CLOSE_WATCHDOG_SECS` (15 s) the same way, and
  `atexit` is registered so processes do not leak Chromium windows.

## Sources
- `src/kiss/agents/sorcar/web_use_tool.py` (`WebUseTool.__init__`, `_ensure_browser`, `_launch_browser`, `_resolve_user_data_dir`, `_profile_lock`, `_clean_singleton_locks`, `_is_profile_in_use`, `_on_page_crash`, `_require_responsive_renderer`, `_input_hang_watchdog`, `_watchdog_kill`, `close`)
- `src/kiss/agents/sorcar/persistence.py` (`_default_kiss_dir`)
