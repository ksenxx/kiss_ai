# Task: Features/ideas to make Sorcar CLI comparable with OpenCode (internet research)

## Steps done
1. Read SORCAR.md (empty). Created tmp/information-opencode2.md research log.
2. Visited 10 sites with go_to_url, logging findings after each (counter 10/10):
   1. opencode.ai/docs/ (intro: /init, plan/build Tab toggle, /undo //redo, /share, image drag-drop)
   2. opencode.ai/docs/cli/ (run, serve, attach, agent create, mcp, models, session, stats, export/import, web, acp, plugin, pr, db, uninstall, upgrade, --pure, env vars)
   3. opencode.ai/docs/tui/ (! bash mode, leader-key keybinds, /compact /details /editor /export /thinking, tui.json, attention sounds/notifications)
   4. opencode.ai/docs/lsp/ (34 built-in LSP servers, lazy start, diagnostics fed to agent)
   5. opencode.ai/docs/permissions/ (allow/ask/deny, glob rules on tool input, external_directory, doom_loop, .env deny default, "always" session patterns)
   6. opencode.ai/docs/agents/ (primary vs subagent, build/plan/general/explore/scout, markdown agents, steps limit, task permissions, child-session navigation)
   7. opencode.ai/docs/commands/ ($ARGUMENTS/$1..$N, !`cmd` injection, @file refs, agent/model/subtask per command)
   8. opencode.ai/docs/plugins/ (JS/TS hook plugins, npm auto-install, tool.execute.before/after, shell.env, custom tools, compaction hook)
   9. opencode.ai/docs/skills/ (SKILL.md on-demand loading, .claude/skills compat, skill permissions)
   10. opencode.ai/docs/share/ (manual/auto/disabled share modes, /unshare deletes data, enterprise self-host)
3. Verified Sorcar CLI current commands via grep on src/kiss/agents/sorcar/cli_repl.py: /help /clear /new /resume /model /cost /usage /context /commands /exit /quit.
4. Synthesized OpenCode-vs-Sorcar gap analysis; delivered in finish summary. Cleaned tmp/.
