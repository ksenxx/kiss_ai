# Task

Make `INJECTIONS.md` and `SAMPLE_TASKS.md` follow the same `~/.kiss/`
copy+read pattern as `MODEL_INFO.json` — but **without** the
clobber-on-update semantics, since these two files are user-editable.

## Final design

- `~/.kiss/INJECTIONS.md` and `~/.kiss/SAMPLE_TASKS.md` are the runtime
  source of truth; bundled package copies under `src/kiss/` are the
  seed.
- Seed-only-if-missing: once a user copy exists, **user edits win
  unconditionally** — no mtime-based refresh, no install-time clobber.
  Regenerating from defaults is an explicit user action: remove the
  file under `~/.kiss/` and re-launch.
- `KISS_HOME` overrides `~/.kiss` in both the install script and the
  runtime helpers, matching the rest of the kiss codebase.

## Files changed

1. **NEW** `src/kiss/agents/vscode/user_assets.py` — Python helper
   exporting `kiss_home_dir()` and `ensure_user_asset(name, package_path)`.
1. **NEW** `src/kiss/agents/vscode/src/userAssets.ts` — TypeScript
   counterpart `kissHomeDir()` / `ensureUserAsset(name, packagePath)`.
1. **MODIFY** `src/kiss/agents/vscode/web_server.py` — `_read_tricks`
   now routes through `ensure_user_asset` so the kiss-web daemon
   reads `~/.kiss/INJECTIONS.md`.
1. **MODIFY** `src/kiss/agents/vscode/src/SorcarTab.ts` — `getTricks`
   and `readSampleTasks` go through `ensureUserAsset`.
1. **MODIFY** `install.sh` — eagerly seeds both files under
   `${KISS_HOME:-$HOME/.kiss}` only when absent (idempotent, no
   clobber).
1. **MODIFY** `src/kiss/agents/vscode/test/sampleTasks.test.js` —
   rewritten with a `withTempKissHome` helper; covers missing-file,
   lazy seed, user-override-preferred, no-clobber-on-newer-package,
   ordering, multi-line bodies, non-`Task` heading skip, empty-body
   skip, mdformat unescape, shipped-file sanity.
1. **NEW** `src/kiss/tests/agents/vscode/test_user_assets.py` —
   Python tests for the helper + `_read_tricks` integration.
1. **MODIFY** `README.md` — adds two bullets under the Customization
   section so users discover they can edit
   `~/.kiss/INJECTIONS.md` / `~/.kiss/SAMPLE_TASKS.md`.

## Review (gpt-5 non-codex subagent)

Initial verdict: **NEEDS WORK**. Addressed:

- MED-1 — added Customization bullets to README.
- MED-2 — removed `os.utime` mutation of the shipped
  `src/kiss/INJECTIONS.md` from the Python integration test.
- MED-3 + MED-4 — dropped the mtime-based refresh entirely (both
  helpers) and switched `install.sh` to no-clobber so user edits
  survive every install/`git pull`. Updated docstrings and added
  the no-clobber test on both sides.
- LOW-1 — `install.sh` now honours `KISS_HOME`.

## Verification

- `uv run check --full` → ✅ all checks pass (ruff, mypy, pyright,
  mdformat, VS Code extension typecheck + eslint + stylelint +
  htmlhint, 17 node tests).
- `uv run pytest -x test_user_assets.py test_welcome_suggestions_not_broadcast.py` → 10 passed.
- `bash -n install.sh` → syntax OK.
