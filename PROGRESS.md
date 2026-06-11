# Task: Add "Agent Skills with Claude compatibility" to Sorcar CLI

User task: Add on-demand SKILL.md loading via a `skill` tool (only name+description
visible until loaded — token-efficient). Read `.claude/skills/` and `~/.claude/skills/`
like OpenCode does; also expose bundled `claude_skills`; pattern-based skill
permissions (`internal-*: deny`). Use internet search extensively (10 sites,
information file in tmp/). Test extensively by actually running sorcar CLI.

## Session 1 — exploration done (steps 1-8)

### Facts discovered

- Worktree: /Users/ksen/work/kiss/.kiss-worktrees/kiss_wt-1781216479-53e41cd0
- SORCAR.md is empty.
- Bundled skills live at `src/kiss/agents/claude_skills/` — NOT present in repo/worktree
  (downloaded by scripts/fetch-claude-skills.sh). Present in installed extension:
  `~/.vscode/extensions/ksenxx.kiss-sorcar-2026.6.18/kiss_project/src/kiss/agents/claude_skills/`
  Structure: `<plugin>/skills/<skill-name>/SKILL.md` with YAML frontmatter
  `name:` and `description:`. Plugins also have agents/ commands/ README.md.
  Also `/Users/ksen/work/kiss/src/kiss/agents/claude_skills` does NOT exist (empty in main repo too — first `ls` returned nothing).
  Worktree has no claude_skills dir. So bundled discovery must tolerate absence.
- `src/kiss/agents/sorcar/custom_commands.py` is the perfect pattern to copy:
  frontmatter regex `_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)`,
  discovery dirs claude-user (`~/.claude/commands`, honours CLAUDE_CONFIG_DIR),
  user (`~/.kiss/commands` via `_default_kiss_dir` from persistence.py:143),
  claude-project (`<wd>/.claude/commands`), project (`<wd>/.kiss/commands`);
  precedence low→high: claude-user, user, claude-project, project.
- `src/kiss/agents/sorcar/sorcar_agent.py::SorcarAgent._get_tools()` builds tools;
  appends ask_user_question, update_settings, set_model; run_parallel if \_is_parallel.
  Tools are plain named functions with full docstrings (docstring = tool schema).
- `cli_repl.py`: `SLASH_COMMANDS: dict[str,str]` at line ~114; `_handle_slash()` at ~446
  handles commands; `_print_help` ~413; completer `_slash_matches` ~233.
  Existing commands: /help /clear /new /resume /model /cost /usage /context /commands /exit /quit.
- Tests live in `src/kiss/tests/` (e.g. src/kiss/tests/test_install_claude_skills.py,
  src/kiss/tests/agents/...). Lint: `uv run check --full`. Tests: `uv run pytest -v`.

### Design (planned, not yet implemented)

1. NEW `src/kiss/agents/sorcar/skills.py`:
   - `@dataclass(frozen=True) Skill`: name, description, path (SKILL.md abs path),
     dir, source ("bundled"|"claude-user"|"user"|"claude-project"|"project").
   - `bundled_skills_dir()` = Path(__file__).parent.parent / "claude_skills".
     Glob `*/skills/*/SKILL.md` AND `*/SKILL.md` (flat layout) for robustness.
   - `claude_user_skills_dir()` honours CLAUDE_CONFIG_DIR → `~/.claude/skills`.
   - `user_skills_dir()` = `_default_kiss_dir()/skills`; project dirs:
     `<wd>/.claude/skills`, `<wd>/.kiss/skills`; each dir globs `*/SKILL.md`
     (skill dir name = fallback name; frontmatter name/description preferred).
   - `discover_skills(work_dir) -> dict[str, Skill]` with precedence as above
     (bundled lowest).
   - Permissions: `skill_permission(name, rules) -> "allow"|"deny"`;
     rules from `load_skill_permissions()`: merge `~/.kiss/config.json` key
     `skill_permissions` (dict pattern→"allow"/"deny") with project
     `<wd>/.kiss/skills.json`?? — SIMPLER: config.json `skill_permissions` only
     (use kiss.agents.vscode.vscode_config.load_config). fnmatch patterns,
     LAST matching rule wins (OpenCode semantics), default allow.
     Denied skills are excluded from discovery listing entirely.
   - `SkillsTool` class: builds a `skill(name: str) -> str` closure?? — NO,
     code style says prefer named functions; make a class with method `skill`
     whose `__doc__` is set dynamically at init listing available skills
     (name + description lines). On invocation: returns SKILL.md full body
     (frontmatter stripped) + header noting skill dir + listing of bundled
     resource files (rglob, relative paths) so agent can Read/run them.
     Unknown/denied name → helpful error listing available names.
   - `format_skill_listing(skills)` for the `/skills` CLI command.
1. EDIT `sorcar_agent.py::_get_tools`: instantiate SkillsTool(work_dir=self.work_dir);
   if it found skills, `tools.append(skills_tool.skill)`. NOTE: dynamic __doc__ on a
   bound method — must verify the schema builder reads instance docstring; the
   pattern used by DockerTools/UsefulTools methods is in those files; check how
   docstrings are read (kiss/core/models/\*.\_build_openai_tools_schema). If bound
   method __doc__ is class-level only, define a per-instance plain function in
   __init__ assigned as attribute `self.skill` with __doc__ set, named function
   style (module-level `_make_skill_tool` returning a def with docstring set).
1. EDIT `cli_repl.py`: add `/skills` to SLASH_COMMANDS + handler printing
   `format_skill_listing(discover_skills(work_dir))`.
1. Tests NEW `src/kiss/tests/agents/sorcar/test_sorcar_skills.py`:
   - tmp HOME/KISS_HOME + CLAUDE_CONFIG_DIR fixtures; create skills in all 4 dirs;
     test discovery precedence, frontmatter parsing, missing-frontmatter fallback,
     deny patterns (`internal-*: deny`), last-rule-wins, listing format,
     skill() load returns body + resources, unknown skill error.
   - CLI integration: run `./sorcar` (check how the `sorcar` script invokes
     cli_repl) with piped stdin `/skills\n/exit\n` and assert listing appears.
     Check env needed (model key?) — /skills should work before any LLM call.
   - Agent end-to-end (cheap model) optional: agent invokes skill tool. Decide
     in session 2 based on harness conventions (look at existing tests for
     running CLI, e.g. tests for cli_repl).
1. Web research FIRST in session 2 (10 sites via go_to_url, file
   tmp/information-skills.md, counter 0/10→10/10):
   opencode.ai/docs/skills, agentskills spec (agentskills.io?),
   anthropic docs agent skills (docs.claude.com/en/docs/agents-and-tools/agent-skills),
   anthropic engineering blog skills, github anthropics/skills, claude code skills docs,
   opencode permissions docs, simonwillison skills post, etc. Extract: SKILL.md format
   fields (name, description, allowed-tools, license, metadata), discovery dirs,
   permission patterns, tool description format ("Skills: name: desc" lines).

## Session 2 — web research DONE (10/10 sites, see tmp/information-skills.md)

Key decisions from research (synthesis in info file):

- Catalog (name+description only) embedded in the `skill` tool docstring as
  `<available_skills>` XML; tool returns `<skill_content name=...>` with
  frontmatter-stripped body + `Skill directory: <abs>` + `<skill_resources>`
  relative file listing (capped at 50), per agentskills.io implementor guide.
- Discovery (low→high): bundled `src/kiss/agents/claude_skills/<plugin>/skills/<name>/SKILL.md`
  (name namespaced `<plugin>:<name>`), `~/.claude/skills` (CLAUDE_CONFIG_DIR),
  `~/.agents/skills`, `~/.kiss/skills`, `<wd>/.claude/skills`, `<wd>/.agents/skills`,
  `<wd>/.kiss/skills` — each `<name>/SKILL.md`.
- Lenient parse: skip skill if description empty/unparseable YAML; name falls back to dir name.
- Permissions: `skill_permissions` dict in ~/.kiss/config.json (via vscode_config.load_config),
  pattern→"allow"|"deny", `*`/`?` wildcards (fnmatch), LAST matching rule wins, default allow.
  Denied skills hidden entirely from catalog.
- No skills → don't register tool. `/skills` slash command lists catalog.
- Verified: schema builder uses `inspect.getdoc(func)` (model.py:584) so a module-level
  factory function with dynamically assigned `__doc__` + `__name__="skill"` works.
- `_default_kiss_dir()` in persistence.py honors KISS_HOME.

## Session 2 — IMPLEMENTATION COMPLETE

- NEW `src/kiss/agents/sorcar/skills.py`: Skill dataclass; discovery from bundled
  claude_skills (`<plugin>:<name>` namespaced), `~/.claude/skills` (CLAUDE_CONFIG_DIR),
  `~/.agents/skills`, `~/.kiss/skills` (KISS_HOME), and project `.claude/.agents/.kiss`
  skills dirs; lenient frontmatter parsing (description falls back to first body
  paragraph; skill skipped only when no description at all); `skill_permissions`
  pattern rules from `~/.kiss/config.json` (fnmatch, last-match-wins, default allow,
  denied skills hidden from catalog); `make_skill_tool()` builds a `skill(name)` tool
  whose docstring embeds the `<available_skills>` name+description catalog and whose
  result wraps the frontmatter-stripped body in `<skill_content>` + skill directory +
  `<skill_resources>` listing (capped 50, never inlined); returns None with no skills.
- EDIT `sorcar_agent.py::_get_tools`: registers the skill tool when skills exist.
- EDIT `cli_repl.py`: `/skills` lists, `/skills <name>` shows one; help + docstring.
- NEW `src/kiss/tests/agents/sorcar/test_sorcar_skills.py`: 36 integration tests
  (discovery/precedence/parsing/permissions/tool/listing/agent toolset/REPL
  subprocess) — all pass.
- `uv run check --full` all green (had to mdformat PROGRESS.md).
- Existing impacted tests pass: test_cli_repl, test_custom_commands, test_sorcar_agent (73).
- Live CLI verified: `./sorcar -w <dir>` REPL `/skills`, `/skills pdf-tricks`,
  denied `internal-secret` hidden; real-model one-shot run (`sorcar -t ...`,
  claude-opus-4-7, $0.06) where the agent itself called `skill(name="pdf-tricks")`
  and reported the planted magic word from the skill body.
- Cleaned tmp/ artifacts; staged changes in git.

### TODO checklist

- [x] Web research 10 sites → tmp/information-skills.md
- [ ] Write skills.py
- [ ] Wire skill tool into sorcar_agent.py
- [ ] Add /skills to cli_repl.py (+ module docstring mention, help)
- [ ] Tests (unit + run actual sorcar CLI via subprocess)
- [ ] uv run check --full
- [ ] Run impacted tests
- [ ] Update README if skills feature should be documented (check README for /commands precedent)
- [ ] Clean tmp/, delete this PROGRESS content? (keep PROGRESS.md log per rules)
