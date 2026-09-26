---
title: Building and linting the VS Code extension (npm run compile, out/, eslint)
uuid: ca9ef8c4-6962-4f96-815c-a6c572da85f3
summary: npm scripts of the VS Code extension - compile (tsc src -> out, allowJs),
  typecheck, lint:ts/css/html scopes, eslint config, uv run check --full integration,
  worktree out/ setup.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Building and linting the VS Code extension (npm run compile, out/, eslint)

All commands run in `src/kiss/agents/vscode/`.

## npm scripts (`package.json`)
| Script | Command | Notes |
|---|---|---|
| `compile` | `tsc -p ./` | `src/**/*` -> `out/` |
| `watch` | `tsc -watch -p ./` | |
| `typecheck` | `tsc --noEmit -p ./` | |
| `lint` | `lint:ts && lint:css && lint:html` | |
| `lint:ts` | `eslint 'src/**/*.ts' 'media/**/*.js' --ignore-pattern 'media/**/*.min.js'` | `test/` and `scripts/` are NOT linted |
| `lint:css` | `stylelint 'media/**/*.css'` (minus `*.min.css`) | `stylelint-config-standard` |
| `lint:html` | `htmlhint 'src/**/*.html' 'media/**/*.html'` | `chat.html` is kept htmlhint-clean |
| `lint:fix` | eslint + stylelint with `--fix` | |
| `test` | `npm run compile && node test/run-all.js` | see `vscode-js-tests` |
| `check` | `typecheck && lint && test` | |
| `package` | `node scripts/package-vsix.js --no-dependencies --allow-missing-repository -o kiss-sorcar.vsix` | see `vscode-packaging-and-branding` |
| `vscode:prepublish` | `npm run compile && npm run copy-kiss` | |

## TypeScript config (`tsconfig.json`)
`target` ES2022, `rootDir` `src`, `outDir` `out`, `strict: true`, `allowJs: true`, `include: ["src/**/*"]`. The `allowJs` setting is why plain-JS host modules (`UpdateChecker.js`, `reloadGuard.js`, `daemonHealth.js`, `daemonRestartVerify.js`, `installerPath.js`, `macLaunchd.js`) live in `src/` and are copied into `out/`. `media/*.js` is NOT compiled: webview scripts are loaded as-is.

`out/` is git-ignored. Many JS tests `require('../out/...')` and skip or fail with "compiled extension missing" until `npm run compile` has run.

## ESLint (`eslint.config.mjs`)
Flat config based on gts-style rules: `eslint:recommended`, prettier (`prettier/prettier: error`), `eqeqeq`, `no-var`, `prefer-const`, `prefer-arrow-callback`, single quotes, plus `typescript-eslint` recommended for `src/**/*.ts`. Ignored: `out/`, `node_modules/`, `media/marked.min.js`, `media/highlight.min.js`, `media/vosk.js`. Running eslint directly on `test/*.js` or `scripts/*.js` reports `no-undef` for `require`/`process`; that is expected, not a project failure.

## Integration with the Python checker
`uv run check --full` (`src/kiss/scripts/check.py`) adds `npm --prefix src/kiss/agents/vscode run typecheck` and `run lint` only when `node_modules` exists and `npm` is on `PATH`. It also calls `_sync_extension_version()`, which copies the version from `src/kiss/core/_version.py` into `package.json`.

## Worktrees
`node_modules/` and `out/` are git-ignored, so a fresh git worktree has neither. The Sorcar worktree setup symlinks the main checkout's `node_modules`; `out/` must be built with `npm run compile` before JS tests or the lint. Do not run `npm install`/`npm ci` in a worktree: it would rewrite the symlinked `node_modules` of the main checkout.

## Sources
- `src/kiss/agents/vscode/package.json` (`scripts`, `devDependencies`)
- `src/kiss/agents/vscode/tsconfig.json`
- `src/kiss/agents/vscode/eslint.config.mjs`
- `src/kiss/scripts/check.py` (`_sync_extension_version`, `--full` stages)
- `.gitignore` (`node_modules/`, `src/kiss/agents/vscode/out/`)
