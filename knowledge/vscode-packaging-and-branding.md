---
title: VSIX packaging, copy-kiss.sh, hashed icons and branding
uuid: b1f14981-21b1-4f48-a490-869bbafdedf8
summary: kiss-sorcar VSIX build - copy-kiss.sh bundles kiss_project and syncs version,
  apply-brand.js from media/brand.json, package-vsix.js with content-hashed icons.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# VSIX packaging, copy-kiss.sh, hashed icons and branding

## Pipeline
`npm run package` = `node scripts/package-vsix.js --no-dependencies --allow-missing-repository -o kiss-sorcar.vsix`. vsce runs the `vscode:prepublish` script first (`npm run compile && npm run copy-kiss`). `install.sh` and the release script build through `npm run package`.

## `copy-kiss.sh`
1. Version: `$KISS_EXP_VERSION` if set, else `__version__` from `src/kiss/core/_version.py`; written into `package.json` `version`.
2. Runs `node scripts/apply-brand.js`.
3. Recreates `kiss_project/` next to the manifest: a copy of the root `pyproject.toml` with the wheel `force-include` section removed, `uv.lock`, `README.md`, and every git-tracked file under `src/kiss/` except the extension's own non-Python files (`src/kiss/agents/vscode/*` is skipped unless it is a `.py` file or under `media/`, because the daemon serves `media/` to the remote webapp).
4. Copies the root `LICENSE` into the extension directory.
`kiss_project/`, the extension `LICENSE` and `README.md` are git-ignored. At runtime `findKissProject()` falls back to `<extension>/kiss_project`, where `DependencyInstaller` creates the `.venv`.

## `.vscodeignore`
Excludes `src/**` (TypeScript sources; compiled JS is in `out/`), `test/**`, `node_modules/**`, `tsconfig.json`, `copy-kiss.sh`, `install.sh`, `scripts/**`, `*.vsix`, `__pycache__`, and `kiss_project/src/kiss/tests/**`.

## Content-hashed icons (`scripts/hash-icons.js`)
Problem: the VS Code server serves extension files with `Cache-Control: public, max-age=31536000` under a URL that contains only the extension version, so a rebuilt VSIX with the same version but a changed icon kept showing the old icon (activity bar, editor tabs, Extensions view) even after a window reload.
Fix: `applyHashedIcons(root)` copies each image directly under `media/` to `media/hashed/<name>-<md5[:8]>.<ext>`, writes the map to `media/hashed/index.json`, rewrites `package.json` to the hashed paths, and returns a restore function. `package-vsix.js` (`packWithHashedIcons`) wraps `vsce pack` and restores the original manifest in `finally` and on SIGINT/SIGTERM/SIGHUP; stale hashed references left by a killed build are recognized and resolved. The tracked manifest keeps plain names; `media/hashed/` is git-ignored. At runtime `mediaIconPath(extensionRoot, name)` (`src/brand.ts`) maps through the index and falls back to `media/<name>` when running from source; `SorcarPanelManager` uses it for editor-tab icons. Webview assets do not need this: they carry `?v=<sha256>` queries.

## Branding (`media/brand.json`)
Keys: `product_name` ("KISS Sorcar"), `short_name` ("KISS"), `tagline`, `identity`, `extension_description`.
- Runtime: `src/brand.ts` `loadBrand()` reads it synchronously at module load, falling back key by key to stock values, and exports `BRAND`, `PRODUCT_NAME`, `SHORT_NAME`, `renderBrand()` (fills `{{PRODUCT_NAME}}`, `{{SHORT_NAME}}`, `{{TAGLINE}}`). It mirrors `kiss.core.brand` on the Python side.
- Manifest: `scripts/apply-brand.js` rewrites display strings in `package.json` ("KISS Sorcar" -> `product_name`, a leading "KISS: " command prefix -> `short_name`, `description` -> `extension_description`). With the stock brand it is a byte-for-byte no-op; it is idempotent for custom brands; command ids, view ids and setting keys are never touched.
- `media/brand.css` carries brand styling for the webview.

## Sources
- `src/kiss/agents/vscode/copy-kiss.sh`
- `src/kiss/agents/vscode/scripts/package-vsix.js` (`packWithHashedIcons`), `scripts/hash-icons.js` (`hashedIconIndex`, `applyHashedIcons`), `scripts/apply-brand.js` (`brandManifest`)
- `src/kiss/agents/vscode/src/brand.ts` (`loadBrand`, `renderBrand`, `mediaIconPath`)
- `src/kiss/agents/vscode/.vscodeignore`, `media/brand.json`
- `src/kiss/agents/vscode/test/hashIcons.test.js`
