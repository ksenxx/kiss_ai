# PROGRESS — startup health check + Slack monitoring alert (current task)

## Task

Add a startup health check that verifies `get_available_models()` returns
a non-empty list before the poller starts, so future shell/env
misconfiguration is caught immediately and loudly (monitoring alert)
instead of silently retrying with empty results.

## Design (from web research — 10 sites)

Best practices synthesized: fail fast in an isolated preflight; actively
signal failures to a real-time channel (Slack) with actionable content
(job name, cause, env, log path, recovery step); deduplicate alerts to
avoid alert fatigue (cron fires every minute); post a recovery notice on
state change; make the alert path itself robust (never let alert errors
mask the failure); test the monitoring by intentionally breaking it.

## Implementation (slack_channel_sorcar_poller.py)

- New `HEALTH_ALERT_FILE` (`~/.kiss/slack_channel_sorcar_poller/health_alert.json`)
  and `HEALTH_ALERT_COOLDOWN = 3600.0`.
- `_post_channel_alert(text) -> bool`: posts to #sorcar with the existing
  bot token; never raises (logs and returns False on any error).
- `_load_health_alert()`: reads the persisted alert marker.
- `_startup_health_check()`: called from `main()` right after
  `source_shell_env()` (replacing the previous inline check):
  - models available + marker present → unlink marker FIRST (concurrent
    invocations race on recovery; unlink-before-post makes at most one
    announce), then post `:white_check_mark:` recovery notice.
  - models empty → log ERROR; post `:rotating_light:` alert with SHELL
    value, log path, and recovery hint — only if the last alert is older
    than the 1-hour cooldown (marker persists `last_alert_ts`);
    `sys.exit(1)` so the failure also lands in cron.log.

## Verification (end-to-end, real Slack)

- Failure path: ran `_startup_health_check()` under a cron-identical env
  (`env -i`, SHELL=/bin/bash, no \*\_API_KEY, CLI-model entries removed to
  reproduce zero available models) → exit code 1, ONE alert posted in
  #sorcar, marker written; second run within cooldown → exit 1 and NO
  second alert (marker ts unchanged).
- Recovery path: ran with real env + marker present → recovery notice
  posted, marker cleared, no exit. (A concurrent real cron tick also
  recovered in the same second — the race produced a duplicate notice,
  which motivated the unlink-before-post fix; deleted the duplicate
  Slack message.)
- Live cron ticks after redeploy: poller.log shows normal
  "Polling channel=sorcar" lines; cron.log clean (0 bytes).
- `uv run check --full` → all checks passed.
- Redeployed byte-identical module to the installed extension tree that
  the cron venv python imports.

______________________________________________________________________

# PROGRESS — fix (A): never post empty replies from the sorcar poller (prior task)

## Task

Empty "(empty response)" replies were posted in #sorcar because the
cron-launched Sorcar task never produced a result, making
`run_agent_via_kiss_web` return "" which `_extract_summary` turned into
"(empty response)" and posted, while `state["threads"][ts]` was saved so
the message was never retried.

## Fix implemented (slack_channel_sorcar_poller.py)

- Added `_is_empty_reply(reply)` helper: True when the reply is blank or
  the literal "(empty response)" placeholder.
- In `_handle_top_level`: after `_run_sorcar`, if `_is_empty_reply(reply)`
  → log a warning, do NOT post, do NOT save thread state → message stays
  unanswered and the next cron tick retries it.
- In `_handle_thread_replies`: same guard for thread continuations.

## ACTUAL ROOT CAUSE found and fixed (session 2)

Fix (A) alone made the poller retry forever: the log showed the same
message retried every 3 s with "Empty Sorcar result" and one
`AssertionError` in `run_agent_via_kiss_web`. Traced by re-running
`_run_task_inner` synchronously under a cron-identical environment
(`env -i HOME PATH` without `SHELL`) with a traced
`printer.broadcast`, which surfaced the real failure:

```
BROADCAST result 'No model available.  Set at least one API key in the environment.'
```

Cause chain: cron leaves `$SHELL` unset → `_get_user_shell()` falls
back to "bash" → `source_shell_env()` sources `~/.bashrc` (which has
only OPENAI_API_KEY) instead of `~/.zshrc` (which has
ANTHROPIC_API_KEY) → `get_available_models()` lacks the configured
default model `claude-fable-5` → `_run_task_inner` aborts with "No
model available" before `agent.run` → `last_run_result` stays "" →
empty reply.

Fixes:

- `scripts/kiss-slack-sorcar-cron.sh` (`run_poller`): export
  `SHELL="$(dscl . -read /Users/$(id -un) UserShell)"` (fallback
  `/bin/zsh`) before exec'ing the poller; redeployed the wrapper to
  `~/.kiss/slack_channel_sorcar_poller/kiss-slack-sorcar-cron.sh`.
- `slack_channel_sorcar_poller.py` `main()`: defence in depth — when
  `$SHELL` is unset, set it from `pwd.getpwuid(os.getuid()).pw_shell`
  before `source_shell_env()`; and if `get_available_models()` is
  still empty, log an error and `sys.exit(1)` instead of running
  tasks that can only produce empty replies.

## Verification (end-to-end, real cron)

- `uv run check --full` → all checks passed.
- Deployed the fixed module to the installed extension tree
  (`~/.vscode/extensions/.../src/kiss/...`, the path the cron venv
  python actually imports) and the fixed wrapper to the stable cron
  location; killed the stale poller running with the old env.
- The next real cron tick (01:31) picked up the pending ksen message
  "List three prime numbers between 10 and 30..." and this time ran
  the task successfully (model claude-fable-5, 6.2 s, $0.10) and
  posted the threaded mrkdwn reply:
  `Three prime numbers between 10 and 30: *11*, *17*, and *23*.`
- state.json now maps ts 1783844064.022589 → chat
  129ec20fe6e74d0b84672786afb6133a; no duplicate replies; cron.log
  clean.

______________________________________________________________________

# PROGRESS — create "kiss-slack" cron job for #sorcar channel poller (prior task)

## Task

Create a cron job named with prefix "kiss-slack" that effectively checks
every 3 seconds for latest unanswered messages from ksen in the #sorcar
channel using the Slack agent, runs each message as a Sorcar task
one-by-one in arrival order, and replies with the result formatted for
Slack (mrkdwn).

## Plan

- The repo already ships
  `src/kiss/agents/third_party_agents/slack_channel_sorcar_poller.py`,
  which is DESIGNED exactly for this: invoked once per minute from cron,
  holds an fcntl lock, polls the channel every POLL_INTERVAL=3s for
  RUN_DURATION=57s, processes messages from KISS_SLACK_USER (default
  "ksen") in KISS_SLACK_CHANNEL (default "sorcar") oldest-first,
  runs each as a Sorcar task via the kiss-web launcher, and posts a
  threaded reply converted to Slack mrkdwn (SLACK_FORMATTING_HINT).
- The installed extension venv
  (~/.vscode/extensions/ksenxx.kiss-sorcar-2026.7.18/kiss_project/.venv)
  has an identical copy of the module — verified via diff (SAME) — and
  the Slack token for workspace "learningsystems" authenticates fine
  ("sorcar2" in "Sky Computing").
- Cron entry name prefix: crontab lines have no names, so the "name" is
  carried by a wrapper script `kiss-slack-sorcar-cron.sh` committed under
  `scripts/` plus a `# kiss-slack-sorcar-poller` comment marker in the
  crontab (used for idempotent install/removal).

## Steps completed

1. Read SORCAR.md, slack_agent.py, slack_channel_sorcar_poller.py, prior
   PROGRESS entries (token already saved for workspace "learningsystems";
   channel #sorcar = C0AKYSNLB7W exists and bot is a member).
1. Verified `crontab -l` is currently empty; verified installed venv
   python can import the poller module and the token auth_test passes.
1. Created `scripts/kiss-slack-sorcar-cron.sh` (the named "kiss-slack"
   job) that execs the installed venv python on the poller module with
   KISS_SLACK_WORKSPACE=learningsystems, USER=ksen, CHANNEL=sorcar.
1. Installed crontab entry running the script every minute (poller
   internally polls every 3 s for 57 s → effective 3-second checking).
1. IMPORTANT: worktree paths are discarded after the task, so the live
   crontab was repointed at a stable copy of the wrapper at
   `~/.kiss/slack_channel_sorcar_poller/kiss-slack-sorcar-cron.sh`
   (the canonical source stays committed in `scripts/`). Final crontab:
   \`\* * * * * ~/.kiss/slack_channel_sorcar_poller/kiss-slack-sorcar-cron.sh
   > > ~/.kiss/slack_channel_sorcar_poller/cron.log 2>&1
   # kiss-slack-sorcar-poller\`
1. Smoke-tested one run of the stable wrapper; poller.log shows
   "Polling channel=sorcar (C0AKYSNLB7W) bot=U0AKVLSTFGD
   user=UD7PM70GG (ksen)" and min_ts watermark initialization; a second
   invocation (the real cron tick at 23:41) started while the first held
   the fcntl lock, confirming lock exclusion works; state.json created.
1. Committed `scripts/kiss-slack-sorcar-cron.sh` + PROGRESS.md.

## Operations

- Logs: `~/.kiss/slack_channel_sorcar_poller/poller.log` (poller),
  `cron.log` (wrapper stdout/stderr).
- Uninstall: `~/.kiss/slack_channel_sorcar_poller/kiss-slack-sorcar-cron.sh --uninstall`
- Re-install: same script with `--install`.
- Config via env in the wrapper: KISS_SLACK_WORKSPACE=learningsystems,
  KISS_SLACK_USER=ksen, KISS_SLACK_CHANNEL=sorcar; optional
  KISS_SLACK_MODEL / KISS_SLACK_BUDGET (default $5/task).

## TASK COMPLETE
