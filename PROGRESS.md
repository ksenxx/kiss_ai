# Task: Research — features/ideas for Sorcar CLI to be comparable with OpenCode (round 2)

Research-only task; no source code changed. Context: a prior session already did an
OpenCode comparison, and since then Agent Skills (SKILL.md + .claude/skills compat +
`skill` tool + `/skills` command + pattern permissions) were IMPLEMENTED in Sorcar.
This round focused on OpenCode doc pages NOT covered before.

## Web research (10 sites, all via go_to_url, notes kept in tmp/information-oc2.md)

1. https://opencode.ai/docs/cli/ — full CLI surface (run/serve/web/acp/attach/session/
   stats/export/import/plugin/pr/db/uninstall/upgrade/agent create/mcp/models/auth/github;
   --pure; env vars incl. OPENCODE_CONFIG_CONTENT, DISABLE_PRUNE).
2. https://opencode.ai/docs/tui/ — /undo //redo (git snapshots), /compact, /connect,
   /editor, /export, /init, /details, /thinking + ctrl+t variants, leader-key keybinds,
   command palette, tui.json attention notifications/sounds, @alias refs, !cmd shell.
3. https://opencode.ai/docs/policies/ — experimental allow/deny policy statements for
   `provider.use`; wildcards; last-match-wins; GLOBAL config beats project (repo can't
   re-enable a globally denied provider).
4. https://opencode.ai/docs/custom-tools/ — drop-in tool files in .opencode/tools/ and
   ~/.config/opencode/tools/; filename=tool name; multi-export; Zod arg schemas;
   same-name overrides built-in tools; context {agent, sessionID, directory, worktree}.
5. https://opencode.ai/docs/references/ — `references` config: alias → local dir or
   git repo (cloned to cache); description advertised in agent context; @alias and
   @alias/ in TUI; auto-allowed through external-directory permission boundary.
6. https://opencode.ai/docs/formatters/ — opt-in post-edit auto-format; ~26 built-ins
   keyed by extension with requirement detection; custom {command with $FILE, extensions,
   environment, disabled}.
7. https://opencode.ai/docs/zen/ — curated/benchmarked model gateway; cheap-model
   auto session titles; auto-reload, monthly limits per workspace/member; team roles,
   admin model allow/deny, BYOK passthrough; public /zen/v1/models metadata endpoint.
8. https://opencode.ai/docs/ecosystem/ — plugin/project ecosystem: context pruning,
   secret redaction (vibeguard), PTY background processes, scheduler, supermemory,
   notifications, Daytona sandboxes, subscription-auth plugins, Discord/Neovim/Obsidian
   clients — all enabled by server API + plugin hooks.
9. https://opencode.ai/docs/network/ — HTTPS_PROXY/NO_PROXY support, proxy basic auth,
   custom CA via NODE_EXTRA_CA_CERTS (enterprise networking).
10. https://opencode.ai/docs/go/ — $10/mo open-model subscription; dollar-denominated
    rolling usage limits ($12/5h, $30/wk, $60/mo) with fallback to free models or
    overflow to credits.

## Synthesis

Produced a tiered feature-gap list in the finish summary (Tier 1: undo/redo, plan/build
toggle, headless run, granular permissions w/ doom_loop+external_directory, ! shell,
/compact, session lifecycle/export/stats; Tier 2: markdown agents, command upgrades,
plugins/hooks, custom Python tools, references, formatters, LSP, MCP; Tier 3:
serve/attach/acp, TUI polish, policies, budget windows, networking, ecosystem).
Marked Agent Skills as already implemented. Cleaned up tmp/information-oc2.md.
