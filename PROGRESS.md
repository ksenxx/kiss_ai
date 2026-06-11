# Task: Features/ideas to add to Sorcar CLI to make it comparable with OpenCode (use internet search extensively)

## Progress log
1. Read SORCAR.md (empty), README.md → captured Sorcar CLI current features (REPL, steering, custom commands, MCP, skills, worktrees, persistence, parallel subagents, web/browser, 529 models, messaging agents, budget, VS Code ext, web/mobile app).
2. Listed src/kiss/agents/sorcar/ modules: chat_sorcar_agent, cli_helpers, cli_panel, cli_repl, cli_steering, custom_commands, git_worktree, mcp_cli, mcp_servers, persistence, running_agent_state, skills, useful_tools, web_use_tool, worktree_sorcar_agent.
3. Research notes file: tmp/information-opencode-research.md (counter inside). Visited so far:
   - [1] opencode.ai/docs/ (intro)
   - [2] opencode.ai/docs/cli/ (full CLI reference)
   - [3] opencode.ai/docs/tui/ (TUI commands/keybinds)
4. Visited [4] /docs/agents/ (primary/subagents, permissions, agent create), [5] /docs/plugins/ (hook system, custom tools, compaction hooks), [6] /docs/lsp/ (30 built-in LSP servers, diagnostics feedback). Notes appended to tmp/information-opencode-research.md (counter 6/10).
4b. Visited [7] /docs/share/ (public share links, auto-share, unshare, enterprise self-host). Counter now 7/10 in tmp/information-opencode-research.md.
4c. NEXT (3 more sites): (a) https://opencode.ai/docs/server/ or /docs/sdk/ (headless HTTP server + typed SDK), (b) https://github.com/anomalyco/opencode (README), (c) a third-party comparison blog or HN thread on "opencode vs claude code". After each: append notes + bump counter in tmp/information-opencode-research.md.
5. Visited [8] /docs/server/ (headless HTTP server, OpenAPI 3.1, REST API, /tui remote-control, SSE events), [9] DuckDuckGo search results (3rd-party reviews: 95K+ stars, model-comparison workflow, benchmark themes), [10] builder.io/blog/opencode-vs-claude-code (MCP context tax, Workspaces, subscription auth, head-to-head benchmark). Counter reached 10/10.
6. Synthesized ~20 feature recommendations for Sorcar CLI (grouped: agent system/permissions, session management, terminal UX, server/SDK architecture, ecosystem/integrations, model access) and delivered in finish() summary. Deleted tmp/information-opencode-research.md. Research-only task — no code changes made.
5. After 10 sites: synthesize feature-gap list comparing OpenCode vs Sorcar CLI and deliver recommendations in finish() summary. This is a research/advice task — no code changes needed. Delete tmp files before final finish.
