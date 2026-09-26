---
title: Cron agent dispatch (run_agent agent="cron"), gateway_command and kiss-cron
  CLI
uuid: 98953bee-d2a6-4d56-a621-7d3886a217ea
summary: run_agent(agent='cron') dispatch, CRON_DISPATCH_PREAMBLE, cron_job + gateway_command
  tools, gateways scheduled as command jobs with --quiet, kiss-cron CLI.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Cron agent dispatch, gateways, and the `kiss-cron` CLI

## Dispatch
The main Sorcar agent does **not** carry the `cron_job` tool. `cron_agent.py` is an *agent
script*: a scheduling request is `run_agent(task, agent="cron")` (commit `b8ca0fefe`).
`agent_dispatch._run_agent` prepends `CRON_DISPATCH_PREAMBLE` and dispatches the module file
with `git_lifecycle=False` and `classify=False`. The module's agent-script getters:
- `tools()` → `[cron_job, gateway_command]` (the module is its own tools file);
- `work_dir()` → `$KISS_HOME/cron/work` (created); `use_worktree()` / `auto_commit()` → `False`.

The preamble instructs the session to use `cron_job` immediately without reading source,
translate natural-language schedules itself, prefer no-LLM `command` jobs for polls (curl/grep
that prints only on news, with `until_delivered=True` for one-time alerts), never schedule a
gateway as a prompt job, set `work_dir` (+ `use_worktree`/`auto_commit`, larger `timeout`) for
project jobs, report duplicates instead of renaming, never call `run_agent` (recursion), and
never edit source or debug the channel CLI (report the failing command instead).

## Always-on gateways are command jobs
A gateway is a recurring poll tick of a channel CLI (see `channels-gateway-runner`). It must be
a `command` job: a tick that finds nothing starts no LLM session and costs nothing, whereas a
prompt job would pay for a session every tick. `gateway_command(channel, chat, pairing=True, workspace="")`:
- matches `channel` against `available_channels()` with `_squash`;
- refuses channels whose module has no `_make_backend` ("has no gateway (poll) mode") and an
  empty `chat`;
- builds `<cli> --channel=<shlex-quoted chat> [--pairing] [--workspace=...] --quiet`, where
  `<cli>` comes from `_channel_cli_name` (installed `console_scripts` entry for
  `...<channel>_sea:main`, e.g. `kiss-gchat`, `kiss-ha`; fallback `kiss-<channel>`).
Typical schedule `every 2m`, deliver `none`. `--quiet` keeps idle ticks silent, so only
activity and failures are logged/delivered.

Chat ids: only Slack, Discord, Matrix and Google Chat resolve names in `find_channel`; for
other channels (e.g. Telegram's numeric chat id) the *top-level* session must first look the
id up through the channel agent, then dispatch the cron agent, because the cron session is
forbidden to call `run_agent`.

## `kiss-cron` CLI (`main`)
`--daemon [--interval S]`, `--tick`, `--list`,
`--create NAME --schedule S (--prompt P | --command C) [--deliver T] [-m MODEL] [-b BUDGET] [--until-delivered] [--work-dir DIR] [--worktree] [--auto-commit] [--timeout S]`,
`--remove ID`, `--pause ID`, `--resume ID`, `--run ID`. `--daemon` prints a banner and calls
`run_scheduler`; `--tick` calls `tick()` and prints `ran N job(s)`. The other flags map onto
`cron_job(...)` and print the tool's YAML.

## History
- `3e36a5bc3`: cron moved into the sorcar package and runs as a daemon thread.
- `4170abed6`: concurrent jobs in isolated scratch dirs; gateway command scheduling.
- `6ed74a748`: dedupe on `cron_job` create.
- `ec8d624ad`: per-job `work_dir`/`use_worktree`/`auto_commit`/`timeout`.

## Sources
- `src/kiss/agents/sorcar/cron_agent.py` (`CRON_DISPATCH_PREAMBLE`, `tools`, `work_dir`, `use_worktree`, `auto_commit`, `gateway_command`, `_channel_cli_name`, `main`)
- `src/kiss/agents/sorcar/agent_dispatch.py` (`_run_agent`, cron branch)
- `src/kiss/agents/third_party_agents/README.md` (examples 16-19)
