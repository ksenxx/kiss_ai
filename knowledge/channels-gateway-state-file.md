---
title: 'Gateway state file: ledger, circuit breaker, DM pairing, thread continuity'
uuid: bc83fb32-0330-44d5-b534-0db46caf2189
summary: 'Gateway state file channel_state_*.json: threads, delivery ledger, circuit
  breaker, DM pairing codes, cursor; derive_state_path, lock, --approve/--list-pending.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Gateway state file

Poll mode persists "Hermes-gateway" state per `(workspace, channel)` so cron ticks can be
stateless processes.

## Location (`derive_state_path`)
File name `channel_state_<ws[:24]>_<channel[:24]>_<sha256(ws\0channel)[:10]>.json` (components
sanitized by `sanitize_state_component`; the hash of the raw strings keeps `a-b` and `a_b`
apart). It sits next to the module's `_config` `config.json` when the module has a
`ChannelConfig`, else under `$KISS_HOME/third_party_agents/channel_state/<Label>/`. A sibling
`.lock` file is the per-channel lock.

## Schema (`default_channel_state`)
| key | meaning |
| --- | --- |
| `threads` | `{thread_ts: {chat_id, last_reply_ts, updated_at}}` — daemon chat to resume per thread; pruned after 7 days |
| `ledger` | pending replies `{channel_id, thread_ts, text, created}` |
| `failures`, `paused_until` | circuit breaker |
| `approved_users`, `pending_pairing` | DM pairing (`{user_id: {code, ts}}`) |
| `cursor` | backend poll cursor (e.g. Telegram `getUpdates` offset) |
| `thread_rotation` | offset for the 20-threads-per-tick cap |
| `pending_envelopes` | messages consumed by a destructive poll (Signal) but not yet handled |

`load_channel_state` never trusts the file: `_normalize_state` validates each field (types,
finite floats, non-negative ints) and substitutes defaults, so a JSON-valid but malformed file
cannot crash a tick. `save_channel_state` writes atomically with 0600 (`write_private_file`).

## Behaviours
- **At-least-once delivery**: `_send_reply` appends a ledger entry and saves *before* sending
  (`_send_with_retry`: 2 attempts, 1 s apart); success removes the entry. Leftovers are resent
  next tick by `_redeliver_pending` with the prefix `(recovered reply) `.
- **Circuit breaker**: only exceptions that escape `_run_tick` count. After
  `_BREAKER_FAILURE_LIMIT` = 5 consecutive failures the channel pauses for
  `_BREAKER_PAUSE_SECONDS` = 900 s. To resume early, delete `paused_until` from the file.
  Thread-poll or continuation-launch failures do not count; they only keep the old cursor.
- **DM pairing** (`--pairing`): the allow set becomes closed (`--allow-users` ∪
  `approved_users`). An unknown sender gets one code (`secrets.token_hex(4)`) and the exact
  approval command, e.g. `kiss-telegram --channel mychan --approve ab12cd34` (with
  `--workspace` when not default); repeat messages from a pending sender are ignored silently.
- **Admin flags**: `--approve CODE` and `--list-pending` require `--channel` and run
  `_handle_pairing_admin` under `_channel_state_lock_with_deadline` (polls the non-blocking
  lock for up to 30 s; a tick holds the lock for its whole agent run, so a blocking acquire
  would look hung). Timeout exits 1 asking to retry.

## Why the lock covers the whole tick
Every runner save happens while `run_once` holds the lock, and the admin path takes the same
lock, so an approval can never be overwritten by a stale in-memory save from a running tick.

## Sources
- `src/kiss/agents/third_party_agents/_channel_agent_utils.py` (`derive_state_path`, `default_channel_state`, `_normalize_state`, `load_channel_state`, `save_channel_state`, `channel_state_lock`, `_channel_state_lock_with_deadline`, `ChannelRunner._send_reply`, `_redeliver_pending`, `_record_transport_failure`, `_effective_allow`, `_maybe_send_pairing`, `_pairing_reply_text`, `_handle_pairing_admin`)
