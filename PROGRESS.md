# Progress

## Task: Research — features/ideas to make Sorcar CLI comparable with Claude Code and OpenCode

### Steps Done

1. Read SORCAR.md (empty), README.md (CLI section), `src/kiss/agents/sorcar/cli_repl.py`
   and listed `src/kiss/agents/sorcar/` to inventory current Sorcar CLI features:
   REPL with @-mentions, ghost completion, slash commands (/help /clear /new /resume
   /model /cost /usage /context /commands /exit), custom markdown commands
   (~/.kiss/commands, .kiss/commands, ~/.claude/commands compat), steering, git
   worktrees, browser, parallel sub-agents, budget flag, chat resume.
1. Web research — visited 10 sites with go_to_url, tracked in
   tmp/information-sorcarcli.md (deleted at end):
   1. Claude Code CLI reference (code.claude.com/docs/en/cli-reference)
   1. Claude Code Interactive mode reference
   1. OpenCode intro docs (opencode.ai/docs)
   1. OpenCode CLI reference (opencode.ai/docs/cli)
   1. Claude Code Checkpointing docs
   1. OpenCode LSP Servers docs
   1. DuckDuckGo search: opencode vs claude code comparisons
   1. Firecrawl deep comparison blog (June 2026)
   1. Claude Code Commands reference (full slash-command list)
   1. OpenCode Agents docs (primary/subagent system, permissions)
1. Synthesized a prioritized feature-gap list for Sorcar CLI (delivered in the final
   answer): headless -p/pipe mode + JSON output, checkpoint/rewind (/undo /redo),
   permission modes + fine-grained tool permission rules, background session
   management (bg/attach/logs/stop + agent view), ! shell mode, MCP support, hooks,
   session export/import/fork/branch, LSP diagnostics, /init memory generation,
   /compact + /context visualization, image paste, vim mode, custom subagents with
   per-agent models, /goal validator loop, code-review skills, stats, self-update,
   server/SDK mode, plugins, etc.
1. No code changes were made; research-only task. Temp research file removed.
