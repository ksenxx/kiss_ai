# Progress

## Task: Add Custom Commands (.md files) support to Sorcar CLI — DONE

### Steps Done

1. Read cli_repl.py, cli_helpers.py, persistence.py (`_default_kiss_dir` respects
   KISS_HOME), and the existing test layout.
1. Web research (10 sites, tracked in tmp/information-customcmds.md, now deleted):
   OpenCode (old + new SST docs), Claude Code skills/commands docs + commands
   reference, Gemini CLI custom commands, Crush/Agent Skills spec, Codex CLI slash
   commands, python-frontmatter, builder.io practical post.
1. Implemented `src/kiss/agents/sorcar/custom_commands.py`:
   - `CustomCommand` dataclass; discovery from `~/.kiss/commands/**/*.md` (user)
     and `<work_dir>/.kiss/commands/**/*.md` (project, overrides user).
   - Subdirectory namespacing: `git/commit.md` → `/git:commit`.
   - Optional YAML frontmatter: `description`, `argument-hint`.
   - Expansion: `@{path}` file injection → !\`cmd\` shell injection →
     `$1..$9` (shlex quoting) → `$ARGUMENTS`; args appended after two
     newlines when no placeholder (Gemini default).
1. Wired into `cli_repl.py`: `/commands` built-in lists custom commands;
   `_handle_slash` dispatches unknown `/name` to custom commands via `_run_one`;
   `CliCompleter._slash_matches` Tab-completes custom names after built-ins;
   `/help` shows a "Custom commands:" section.
1. Tests: `src/kiss/tests/agents/sorcar/test_custom_commands.py` (25 tests:
   discovery, override, namespacing, frontmatter, all expansion paths, listing,
   completion, REPL subprocess for /commands, /help, unknown). All pass; existing
   28 REPL tests pass; `uv run check --full` passes.
1. Verified by actually running the sorcar CLI: created
   `.kiss/commands/greet.md` with `$1` placeholder, ran `/commands` (listed
   `/greet [name] (project) Greet someone by name`) and `/greet World` — the
   agent ran the expanded prompt and finished with summary "HELLO World".

## Follow-up task: also load commands from .claude/commands — DONE

1. Extended `custom_commands.py` with `claude_user_commands_dir()`
   (honours `CLAUDE_CONFIG_DIR`, defaults to `~/.claude/commands`) and
   `claude_project_commands_dir()` (`<work_dir>/.claude/commands`).
1. `discover_commands` now loads four directories, precedence low→high:
   claude-user → user → claude-project → project (project beats user; native
   `.kiss` beats `.claude` at the same level). Sources labelled
   `claude-user` / `claude-project` in `/commands` listings.
1. Updated module/REPL docstrings and the empty-listing hint to mention the
   Claude Code directories.
1. Tests: fixture now isolates `CLAUDE_CONFIG_DIR`; added 5 tests (claude
   user/project discovery, kiss-over-claude same level, claude-project over
   kiss-user, namespacing/frontmatter for claude files, REPL `/commands`
   listing of claude commands). 30/30 pass; 27 existing REPL tests pass;
   `uv run check --full` green.
1. Verified with the real sorcar CLI: a `.claude/commands/status.md` in the
   work dir was listed as `/status (claude-project) Summarize repo status`.
