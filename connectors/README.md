# KISS Sorcar Connectors

Curated, privacy-first MCP connectors for KISS Sorcar. Every connector here
follows one rule: **credentials and data go only to the service that already
holds them** — local open-source servers with your own keys, or no-auth remote
endpoints. No hosted gateway, aggregator, or third-party runtime ever sits in
the path.

Sorcar discovers servers from `~/.kiss/mcp.json` (all projects),
`<project>/.mcp.json` (Claude-Code compatible), and `<project>/.kiss/mcp.json`.
Each server's tools appear to the agent as `<server>_<tool>` and are filtered
by the `mcp_permissions` wildcard rules in `~/.kiss/config.json`.

## Quick start

```bash
uv run python connectors/enable.py list          # see everything + status
uv run python connectors/enable.py enable slack  # checks prereqs/env, writes config
uv run python connectors/enable.py disable slack
uv run python connectors/verify.py               # connect to all configured servers
```

## Enabled by default (verified working, no credentials needed)

| Connector | What it gives the agent | Runs |
|---|---|---|
| `fetch` | Fetch any web page as Markdown | local (`uvx`) |
| `time` | Current time, timezone conversion | local (`uvx`) |
| `memory` | Persistent knowledge-graph memory across runs | local (`npx`) |
| `sequential-thinking` | Structured reasoning scratchpad | local (`npx`) |
| `deepwiki` | Q&A over any public GitHub repo | remote, no auth |
| `context7` | Up-to-date library docs for coding | remote, no auth |
| `github` | GitHub's official server: repos, issues, PRs, actions | local binary |

The `github` connector mints its token at launch with `gh auth token`
(`brew install gh github-mcp-server`; `gh auth login` once) — the token is
never written to any file — and runs with `--read-only` so the server itself
refuses writes; drop that flag in `~/.kiss/mcp.json` if you want the agent to
file issues or PRs, and rely on the `mcp_permissions` deny rules instead.

Privacy note: `deepwiki` and `context7` are the only remote entries — their
operators see the queries you send them (repo names, library names) and
nothing else. Disable either with `enable.py disable <name>` if that matters.
All packages are version-pinned in `catalog.json`; bump them deliberately.

## Available with your own credentials (enable when ready)

| Connector | Service | Credential (env var, read from your shell) |
|---|---|---|
| `google` | Gmail, Calendar, Drive, Docs, Sheets, ... | `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` (your own GCP OAuth client) |
| `slack` | Search/read Slack; posting off by default | `SLACK_MCP_XOXP_TOKEN` (or browser `xoxc`/`xoxd` tokens) |
| `twilio-sms` | Send SMS via your Twilio account | `TWILIO_ACCOUNT_SID`, `TWILIO_API_KEY`, `TWILIO_API_SECRET` |
| `whatsapp` | Personal WhatsApp (QR-paired, all data local) | none — pair by QR code |
| `brave-search` | Web/news/image search | `BRAVE_API_KEY` |
| `notion` | Notion pages and databases | `NOTION_TOKEN` (internal integration) |
| `postgres` | Your PostgreSQL databases (restricted mode) | `DATABASE_URI` |
| `firecrawl` | Crawling/scraping via Firecrawl cloud | `FIRECRAWL_API_KEY` |
| `playwright` | Second isolated browser (Sorcar has one natively) | none |

`enable.py` refuses to enable a connector whose executables or env vars are
missing and prints the exact setup steps (from `catalog.json`). Export
credentials in your shell profile — Sorcar's stdio launcher passes your
environment to the server at launch, so **no secret is ever stored in
`mcp.json` or this repository**. Restart Sorcar after changing env vars or
configs: servers launch with the environment Sorcar started with.

Twilio's team advises against running community MCP servers alongside their
official one (prompt-injection isolation); if you enable `twilio-sms`, prefer
project scope (`--scope project`) in a project that has no third-party servers.

### WhatsApp in three steps

```bash
brew install go
uv run python connectors/enable.py enable whatsapp   # clones lharries/whatsapp-mcp
cd ~/.kiss/connectors/whatsapp-mcp/whatsapp-bridge && go run main.go   # scan QR once
```

Messages sync into a local SQLite DB; nothing new sees your traffic (it is the
normal end-to-end-encrypted WhatsApp Web protocol). Unofficial API — use
judiciously. Re-pair about every 20 days.

## Anthropic & OpenAI billing — no MCP server needed

The most private connector is no connector: one curl from your machine to the
vendor's official Admin API. Both scripts read the key from your environment:

```bash
export ANTHROPIC_ADMIN_KEY=sk-ant-admin...   # Console -> org admin key
connectors/bin/anthropic_costs.sh 30         # cost report, last 30 days

export OPENAI_ADMIN_KEY=sk-admin...          # platform.openai.com -> Admin keys
connectors/bin/openai_costs.sh 30
```

Both require an *organization* account. On an individual account, ask Sorcar to
open the vendor console in its browser and read the usage page instead.

## Fidelity & Bank of America — deliberately not connectors

Neither offers a retail API, and every aggregator path (Plaid, SnapTrade)
routes your balances — or your password — through a third party. The
zero-third-party answer is Sorcar's own browser: ask the agent to open
fidelity.com or bankofamerica.com, it calls `show_browser()` so **you** type
the password and 2FA yourself, and it reads balances/positions/statements from
the logged-in session. Keep it read-only; never automate transfers or trades.

## Safety defaults

`~/.kiss/config.json` ships deny rules so destructive tools never even reach
the agent (last matching rule wins):

```json
"mcp_permissions": {
  "*": "allow",
  "*_delete*": "deny",
  "github_create_or_update_file": "deny",
  "github_push_files": "deny",
  "github_create_repository": "deny",
  "github_fork_repository": "deny",
  "github_merge_pull_request": "deny"
}
```

Loosen or tighten per taste — e.g. add `"slack_*": "deny"` to make a connector
read-only for a while without disabling it. Remember the standing rules:
treat remote tool *results* as untrusted input, keep secrets out of tool
arguments, and gate anything that can send data out.

## Files

- `catalog.json` — machine-readable catalog (config, prereqs, env vars, setup steps)
- `enable.py` — enable/disable/list CLI (writes config via Sorcar's own writer)
- `verify.py` — connects to every configured server through Sorcar's `MCPManager`
- `bin/anthropic_costs.sh`, `bin/openai_costs.sh` — billing via official Admin APIs
