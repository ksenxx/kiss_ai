# Task Progress — kisssorcar.github.io README-aligned redesign

## Task
Make `~/work/kisssorcar.github.io/index.html` have the same sections as
`PWD/README.md`. Use a card for each section with eye-catching animation that
does NOT look like generic Claude/GPT output.

## Repos
- README source: `/Users/ksen/work/kiss/.kiss-worktrees/kiss_wt-1782164548-e4e973a6/README.md`
- Site repo: `/Users/ksen/work/kisssorcar.github.io/` (branch `main`, clean tree)
- Backup of original site: `/tmp/index.bak.html`
- Site assets dir contents: `KISS-Sorcar-UI.png`, `KISS-Sorcar.png`,
  `kiss_sorcar.pdf`, `og-card.png`, `se_kiss_sorcar.pdf`, `sorcar-main.mp4`,
  `writing_paper.pdf`.

## README section ordering (target for site)
1. Hero / one-line install (already in site hero — keep)
2. KISS Sorcar vs Claude Code vs Cursor  → `#compare`
3. What is in the Name                   → `#name`   (NEW)
4. Installation                          → `#install` (expand: full source, pip/uv, model keys, VS Code)
5. CLI Interface                         → `#cli`    (NEW — CLI options table, interactive features, `sorcar mcp`)
6. Messaging & Third-Party Agents        → `#agents`
7. Models Supported                      → `#models`
8. Contributing                          → `#contributing` (NEW)
9. License                               → `#license`     (NEW)
10. Citation                             → `#citation`

Drop the current `#features`, `#architecture`, `#papers`, `#demo` sections
since they are not present as headings in README.md. (Keep the demo video as
the hero screenshot since it already lives above the first section.)

## Design system — eye-catching, non-template-y animations
- Each top-level section wrapped in a `.section-card` whose `::before` paints
  an animated rotating conic-gradient border via `@property --angle` →
  `@keyframes spin { to { --angle: 360deg; } }`. Slow, ~9 s rotation.
- Scroll-triggered reveal: IntersectionObserver toggles a class that runs a
  "tilt-rise" keyframe: `from { opacity: 0; transform: translateY(36px) rotateX(-10deg); transform-origin: bottom; } to { opacity: 1; transform: none; }`.
  Each child card inside the section gets a stagger delay via `--i` custom
  property (set in JS or via nth-child).
- Inner cards: cursor spotlight. `onmousemove` updates CSS vars `--mx`/`--my`
  → a `radial-gradient(circle 240px at var(--mx) var(--my), ...)` overlay.
- Hero: subtle "magic dust" canvas — 60 floating particles rising slowly,
  thematic for Sorcar (the magician). Pauses when offscreen.
- Keep existing palette (Material blue/teal). Add a second accent for the
  rotating border: gold `#f7c948` blending with `--primary` and `--accent`.
- Respect `prefers-reduced-motion: reduce` — disable spin, particles, tilt
  reveal in that mode.

## CLI Interface section content (from README)
Three sub-cards laid out in a vertical stack inside the section:

### CLI Options sub-card — table with these rows
| Flag | Description |
| `-t`, `--task` | Task description; non-interactive mode |
| `-f`, `--file` | Task from a file; non-interactive mode |
| `-m`, `--model_name` | LLM model name; defaults to best available |
| `-e`, `--endpoint` | Custom base URL for local/self-hosted model |
| `--header` | Custom HTTP header `Key:Value`; may repeat |
| `-b`, `--max_budget` | Max spend in USD |
| `-w`, `--work_dir` | Working directory |
| `-v`, `--verbose` | Print Rich panels (default true) |
| `-p`/`--parallel` / `--no-parallel` | Toggle parallel sub-agents |
| `--worktree` / `--no-worktree` | Interactive only — git worktree isolation |
| `--auto-commit` / `--no-auto-commit` | Interactive only — auto-commit worktree |
| `--no-web` | Disable browser/web tools |

### Interactive features sub-card — bulleted list
- `@` file mentions with ranked completion
- Slash commands: `/help`, `/clear` (`/new`), `/resume`, `/model`,
  `/model list`, `/cost` (`/usage`, `/context`), `/commands`, `/skills`,
  `/mcp`, `/autocommit`, `/exit` (`/quit`)
- Custom Markdown slash commands from `~/.kiss/commands`,
  `<project>/.kiss/commands`, `~/.claude/commands`,
  `<project>/.claude/commands`
- Agent Skills from `~/.kiss/skills`, `<project>/.kiss/skills`, Claude skill
  dirs, `.agents/skills`, bundled Sorcar skills
- MCP server discovery from `~/.kiss/mcp.json`, `<project>/.kiss/mcp.json`,
  `<project>/.mcp.json`
- VS Code "Tricks" button entries from `~/.kiss/INJECTIONS.md`
- VS Code welcome-screen sample-task chips from `~/.kiss/SAMPLE_TASKS.md`

### `sorcar mcp` sub-card — table with these rows
- `sorcar mcp add <name> <cmd…>` — Register stdio/`--transport http`/`sse`
  server in `--scope user` or `--scope project`; `--env KEY=VALUE`,
  `--header 'Key: Value'` (repeatable).
- `sorcar mcp list [--ping]` — List configured servers; `--ping` reports
  live status and tool counts.
- `sorcar mcp get <name>` — Print one server's config as JSON.
- `sorcar mcp remove <name>` — Delete a server from every writable config.
- `sorcar mcp auth <name> [--no-browser]` — OAuth 2.1 browser flow (DCR +
  PKCE); tokens under `~/.kiss/mcp_auth/`.
- `sorcar mcp logout <name>` — Delete stored OAuth tokens.
- `sorcar mcp debug <name>` — Dump capabilities, tools, resources, prompts.

## Models section content (from README)
- OpenAI 70, Anthropic 13, Gemini/Google 23, Together AI 77, MiniMax 5,
  OpenRouter 303, Claude Code CLI (`cc/*`) 3, Codex CLI (`codex/*`) 7.
- Totals: 485 gen, 321 fc, 7 emb.

## Messaging section content (from README)
BlueBubbles · Discord · Feishu · Gmail · Google Chat · iMessage · IRC · LINE ·
Matrix · Mattermost · Microsoft Teams · Nextcloud Talk · Nostr · Phone
Control · Signal · Slack · SMS · Synology Chat · Telegram · Tlon · Twitch ·
WhatsApp · Zalo. Plus Govee smart-home CLI. Lives in
`src/kiss/agents/third_party_agents/`.

## What is in the Name content (from README)
KISS Agent Framework — small agent runtime around the KISS principle.
KISS Sorcar named after P. C. Sorcar, Bengali magician. "Sorcar" also means
government in Bengali.

## Contributing / License (from README)
- Contributing: issues welcome; KISS Sorcar can help implement/review.
- License: Apache-2.0, link to `LICENSE`.

## Steps already completed
1. Read `README.md` end-to-end and inventoried its sections.
2. Read current `index.html` (1162 lines) — captured every existing section
   (`#install`, `#compare`, `#features`, `#architecture`, `#models`,
   `#agents`, `#papers`, `#demo`, `#citation`), the nav, the hero, the
   benchmark band, the demo video, and the footer.
3. Saved backup of original site at `/tmp/index.bak.html`.

## Steps remaining (next session)
1. Rewrite `/Users/ksen/work/kisssorcar.github.io/index.html` from scratch:
   - Preserve `<head>` SEO/meta/og tags exactly.
   - Preserve hero block (logo, Einstein quote, h1, lead, benchmark band,
     screenshot video) — they are great as-is.
   - Update nav links: Compare · Name · Install · CLI · Agents · Models ·
     Contributing · License · Citation.
   - Insert sections in the README order documented above.
   - Add `@property --angle` rotating conic-gradient border on
     `.section-card`, `prefers-reduced-motion` guard, IntersectionObserver
     scroll-reveal with stagger, cursor-spotlight on inner cards, magic-dust
     canvas in hero.
2. Open the file in a local file:// URL to sanity-check rendering, fix
   any visual issues.
3. Commit and push? — not asked; only modify the file. Leave staging to the
   user.
4. Clean up `PWD/tmp/` and `PROGRESS.md` is the project-level tracker so it
   stays.
