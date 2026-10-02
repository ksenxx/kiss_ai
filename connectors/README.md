# KISS Sorcar Connectors

Curated, privacy-first MCP connectors for KISS Sorcar. Every connector here
follows one rule: **credentials and data go only to the service that already
holds them** — local open-source servers with your own keys, or no-auth remote
endpoints. No hosted gateway, aggregator, or third-party runtime ever sits in
the path.

Sorcar discovers servers from `~/.kiss/mcp.json` (all projects),
`<project>/.mcp.json` (Claude-Code compatible), and `<project>/.kiss/mcp.json`,
in that order of increasing precedence (on a name clash the later file wins).
The files are re-read at the start of each task, and the servers' tools are
added only to runs with the full tool profile. Each server's tools appear to the agent as
`<server>_<tool>` and are filtered by the `mcp_permissions` wildcard rules in
`~/.kiss/config.json`.

## Quick start

```bash
uv run python connectors/enable.py list          # see everything + status
uv run python connectors/enable.py enable slack  # checks prereqs/env, writes config
uv run python connectors/enable.py enable slack --scope project   # <project>/.kiss/mcp.json instead of ~/.kiss/mcp.json
uv run python connectors/enable.py disable slack
uv run python connectors/verify.py               # connect to all configured servers
uv run python connectors/verify.py fetch time    # ... or only the named ones
```

`enable` writes to `~/.kiss/mcp.json` (`--scope user`, the default) or
`<project>/.kiss/mcp.json` (`--scope project`); `disable` removes the entry
from every config file it appears in, including `<project>/.mcp.json`.

## Default set (verified working; no API keys, except `github` needs `gh auth login`)

These are flagged `"default": true` in `catalog.json`. Nothing turns them on
for you: run `enable.py enable <name>` for each one you want.

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
The npm/uvx package specs are version-pinned in `catalog.json`; bump them
deliberately. Two exceptions are not pinned by the catalog: the `whatsapp`
Git clone (`git clone --depth 1`, no ref) and the separately installed `gh` /
`github-mcp-server` executables.

## Additional connectors (most need your own credentials or a pairing step; enable when ready)

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
| `you-search` | You.com web search + URL content extraction (keyless free tier) | none — free profile; see setup for the optional key upgrade |
| `playwright` | Second isolated browser (Sorcar has one natively) | none |

`enable.py` refuses to enable a connector whose executables or env vars are
missing: it names the missing items and, for env vars, prints the setup steps
from `catalog.json`; `--force` writes the entry anyway. Export
credentials in your shell profile — Sorcar's stdio launcher passes your
environment to the server at launch, so **no secret is ever stored in
`mcp.json` or this repository**. Restart Sorcar after changing env vars:
servers launch with the environment Sorcar started with. Edits to the
`mcp.json` files take effect at the next task (a changed entry gets a fresh
connection; unchanged entries keep their live one).

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

## Remote servers with OAuth sign-in (Notion, Linear, Asana, Zoom, ...)

Hosted MCP servers that follow the MCP authorization spec are not in
`catalog.json`; Sorcar signs in to them directly, with no KISS-owned app and
no broker in between. Ask the agent to connect (it has the
`connect_mcp_server(name, url="", transport="")` and
`finish_mcp_server_connect(name)` tools) or run it yourself:

```bash
uv run python -m kiss.agents.sorcar.mcp_oauth notion                 # known name
uv run python -m kiss.agents.sorcar.mcp_oauth acme https://mcp.acme.com/mcp http   # any URL; transport http (default) or sse
```

The names `notion`, `linear`, `asana` (`sse`) and `zoom` map to the vendors'
endpoints; any other name needs its URL. A server that is not yet configured
is written to `~/.kiss/mcp.json` first (user scope). Sorcar then registers
itself with the server's authorization server as a public PKCE client (Client
ID Metadata Document or Dynamic Client Registration), opens the authorization
URL in the Browser tab that every KISS surface switches to when the kiss-web
daemon can open it (else in your default browser), says where it went and
also returns the URL, and waits for the redirect on
the fixed loopback address `http://localhost:53683/callback`. You sign in and
click Allow; `finish_mcp_server_connect` reports `pending` until then and the
sign-in gives up after 10 minutes. One sign-in runs at a time (starting
another cancels the first). Tokens and the registered client are stored in
`~/.kiss/mcp_auth/<server>.json` (mode `0600`); later runs reuse and refresh
them and never open a browser: a remote server without stored tokens fails
with a hint to run the sign-in. Servers whose authorization server allows
neither registration method (Zoom) need your own OAuth app: export
`KISS_MCP_<NAME>_CLIENT_ID` (and `KISS_MCP_<NAME>_CLIENT_SECRET` for a
confidential app; `<NAME>` is the server name upper-cased with non-alphanumerics
as `_`) and register the redirect URI above with it.
`KISS_MCP_CLIENT_METADATA_URL` points at a hosted Client ID Metadata Document
when you prefer CIMD. Local `stdio` servers need no sign-in and the tools
refuse them. The `notion` catalog entry above is the other route to Notion: a
local server with an internal-integration token instead of the hosted one.

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

Add deny rules to `~/.kiss/config.json` so destructive tools never even reach
the agent (nothing installs these for you; last matching rule wins):

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
