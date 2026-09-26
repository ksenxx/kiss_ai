---
title: Agent Skills (SKILL.md) support in Sorcar
uuid: 50c28274-3af4-439c-a5e1-31f53a1446c6
summary: 'Agent Skills (SKILL.md) in Sorcar: discovery paths and precedence, progressive
  disclosure via the skill tool, skill_permissions wildcards, bundled plugin skills.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Agent Skills

A *skill* is a directory holding a `SKILL.md` whose YAML frontmatter gives `name` and `description`
(Agent Skills standard, agentskills.io, compatible with Claude Code).

## Discovery (`discover_skills(work_dir)`), low to high precedence; later wins on a name clash
1. bundled: `kiss/agents/claude_skills/<plugin>/skills/<name>/SKILL.md`, namespaced `<plugin>:<name>`
2. `~/.claude/skills/` (respects `CLAUDE_CONFIG_DIR`), source "claude-user"
3. `~/.agents/skills/`, source "agents-user"
4. `~/.kiss/skills/` (respects `KISS_HOME`), source "user"
5. `<work_dir>/.claude/skills/`, then `<work_dir>/.agents/skills/`, then `<work_dir>/.kiss/skills/` (project)

Project skills override user skills, and a native `.kiss` skill beats a Claude Code skill at the same level.
`_parse_skill_file` is lenient: the directory-derived name wins over a mismatched frontmatter name, and a missing
description falls back to the first body paragraph. A skill is **skipped only if neither exists**.

## Progressive disclosure
1. Catalog: only names and descriptions, embedded as `<available_skills>` XML in the `skill` tool's docstring (`_catalog_xml`).
2. Instructions: calling the tool loads the full body (`load_skill_content`). It re-reads `SKILL.md` at activation,
   so edits are picked up, strips the frontmatter, and wraps the body in `<skill_content>` with the skill dir.
3. Resources: files under the skill dir are listed, not read (`_list_resources`, capped at 50, skipping .git,
   node_modules and __pycache__).

`make_skill_tool(work_dir)` returns `None`, registering no tool, when no **user or project** skills exist.
Bundled plugin skills alone do not register the tool.

## Permissions
The `skill_permissions` key in `~/.kiss/config.json` maps shell-style wildcards to "allow"/"deny", e.g.
`{"*": "allow", "internal-*": "deny"}`. The **last** matching rule wins (OpenCode semantics), and no match means allow.
Denied skills are hidden from the catalog rather than blocked at activation. MCP tools use the same rule shape
(`load_permission_rules`).

## Wiring
`SorcarAgent._get_tools` adds the skill tool only for the `full` profile.

## Sources
- `src/kiss/agents/sorcar/skills.py` (`discover_skills`, `_parse_skill_file`, `make_skill_tool`, `load_skill_content`, `skill_permission`, `load_skill_permissions`, `load_permission_rules`, `_catalog_xml`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`_get_tools`)
