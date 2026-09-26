---
title: 'kiss-<channel> CLI: channel_main flags, interactive vs poll mode'
uuid: 7ac5d219-1d0b-458c-913a-683a0187ca13
summary: 'kiss-<channel> CLI via channel_main: -t/-f task mode, --channel poll tick,
  --workspace, --allow-users, --pairing, --quiet, --approve, KISS_WORKDIR, api_keys.env.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# `kiss-<channel>` CLI

Every channel's `main()` delegates to `channel_main(agent_cls, cli_name, channel_name=..., make_backend=..., extra_usage=...)`
in `_channel_agent_utils.py`. Parser base: `_channel_cli._build_arg_parser()` (`allow_abbrev=False`,
so `--para` is rejected rather than expanded).

## Flags
Common: `-m/--model_name`, `-e/--endpoint`, `--header K:V` (repeatable), `-b/--max_budget`
(validated by `_parse_budget_value`), `-w/--work_dir` (default: launch dir), `--no-web`,
`-p/--parallel` / `--no-parallel`, `-t/--task`, `-f/--file`, `-V`, and `--workspace` (default
`default`). Poll-capable channels (`make_backend` not `None`) add `--channel`,
`--allow-users a,b`, `--pairing`, `--quiet`, `--approve CODE`, `--list-pending`. No arguments
prints a usage line and exits 1. Slack adds `--list-workspaces` / `--delete-workspace WS`.

## Modes
1. **Pairing admin** — `--approve` or `--list-pending`: edits the state file and returns.
2. **Poll tick** — `--channel CH`: one `ChannelRunner.run_once()`; prints
   `Processed N message(s).` With `--quiet` it prints nothing unless N > 0, so an idle tick
   scheduled as a cron command job produces empty stdout and delivers nothing.
3. **Interactive / one-shot** — otherwise: builds the agent (passing `workspace` when the
   constructor accepts it), `run_agent_via_kiss_web(agent, prompt, **_build_run_kwargs(args))`,
   then `_print_run_stats`.

## Environment details
- Before anything else `channel_main` calls `kiss.core.vscode_config.load_api_keys()` (falling
  back to `load_api_keys_readonly()` on `OSError`, e.g. read-only `$KISS_HOME`) so
  `KISS_MUSE_AUTH` and channel tokens from `$KISS_HOME/api_keys.env` are in `os.environ` before
  the first `muse_auth_enabled()` check. A CLI started from cron does not inherit the daemon's
  environment.
- The installed wrappers run `uv run --directory <kiss_project> ...`, which changes cwd; they
  export the user's original `$PWD` as `KISS_WORKDIR`, and `_launch_work_dir()` prefers it.
- The CLI name printed in pairing replies is `cli_name` (e.g. `kiss-telegram`).
  `cron_agent._channel_cli_name` finds the real console-script name from entry points
  (`kiss-gchat`, `kiss-ha`) with `kiss-<channel>` fallback.

## Sources
- `src/kiss/agents/third_party_agents/_channel_agent_utils.py` (`channel_main`, `_handle_pairing_admin`)
- `src/kiss/agents/third_party_agents/_channel_cli.py` (`_build_arg_parser`, `_launch_work_dir`, `_parse_budget_value`, `_build_run_kwargs`, `_print_run_stats`)
- `src/kiss/agents/third_party_agents/slack_sea.py` (`main`, `_list_workspaces`)
