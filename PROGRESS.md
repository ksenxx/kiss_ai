# PROGRESS — create "kiss-slack" cron job for #sorcar channel poller (current task)

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
2. Verified `crontab -l` is currently empty; verified installed venv
   python can import the poller module and the token auth_test passes.
3. Created `scripts/kiss-slack-sorcar-cron.sh` (the named "kiss-slack"
   job) that execs the installed venv python on the poller module with
   KISS_SLACK_WORKSPACE=learningsystems, USER=ksen, CHANNEL=sorcar.
4. Installed crontab entry running the script every minute (poller
   internally polls every 3 s for 57 s → effective 3-second checking).
5. IMPORTANT: worktree paths are discarded after the task, so the live
   crontab was repointed at a stable copy of the wrapper at
   `~/.kiss/slack_channel_sorcar_poller/kiss-slack-sorcar-cron.sh`
   (the canonical source stays committed in `scripts/`). Final crontab:
   `* * * * * ~/.kiss/slack_channel_sorcar_poller/kiss-slack-sorcar-cron.sh
   >> ~/.kiss/slack_channel_sorcar_poller/cron.log 2>&1
   # kiss-slack-sorcar-poller`
6. Smoke-tested one run of the stable wrapper; poller.log shows
   "Polling channel=sorcar (C0AKYSNLB7W) bot=U0AKVLSTFGD
   user=UD7PM70GG (ksen)" and min_ts watermark initialization; a second
   invocation (the real cron tick at 23:41) started while the first held
   the fcntl lock, confirming lock exclusion works; state.json created.
7. Committed `scripts/kiss-slack-sorcar-cron.sh` + PROGRESS.md.

## Operations

- Logs: `~/.kiss/slack_channel_sorcar_poller/poller.log` (poller),
  `cron.log` (wrapper stdout/stderr).
- Uninstall: `~/.kiss/slack_channel_sorcar_poller/kiss-slack-sorcar-cron.sh --uninstall`
- Re-install: same script with `--install`.
- Config via env in the wrapper: KISS_SLACK_WORKSPACE=learningsystems,
  KISS_SLACK_USER=ksen, KISS_SLACK_CHANNEL=sorcar; optional
  KISS_SLACK_MODEL / KISS_SLACK_BUDGET (default $5/task).

## TASK COMPLETE
