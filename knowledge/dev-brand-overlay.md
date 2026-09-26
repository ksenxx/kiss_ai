---
title: 'White-label brand overlay: .brand/ directory applied by install.sh'
uuid: bafd6eb1-def8-4c64-9cd2-172b9411cfd5
summary: 'White-label rebranding via git-ignored .brand/ (brand.json, brand.css, icons,
  thumbnail): install.sh apply/restore_brand_overlay around copy-kiss and vsce package;
  tracked files stay stock.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# White-label brand overlay (`.brand/`)

## Brand data
The product name, tagline and agent identity come from one file,
`src/kiss/agents/vscode/media/brand.json` (`product_name`, `short_name`, `tagline`, `identity`,
`extension_description`). It is read by `kiss.core.brand`, by the extension host
(`src/brand.ts`) and by the chat page. `media/brand.css` is loaded after the stock stylesheet
and is an empty skin hook. The runtime side of branding belongs to the core and extension
areas. This page covers the build-time overlay.

## The overlay
A white-label distribution puts its own `brand.json`, `brand.css`, `kiss-icon.svg`,
`kiss-icon.png` and `thumbnail.jpeg` (`BRAND_OVERLAY_FILES`) into `.brand/` at the checkout
root. `.brand/` is gitignored (`/.brand/`). In `install.sh` step `[4/5]`:
1. `apply_brand_overlay "$PROJECT_DIR"` runs after `npm run compile`. It does nothing without
   `.brand/`. Otherwise it snapshots the checkout's `src/kiss/agents/vscode/package.json` and
   each media file the overlay overrides into a `mktemp -d` backup, then copies the overlay over
   `src/kiss/agents/vscode/media/`.
2. `npm run copy-kiss` runs `src/kiss/agents/vscode/copy-kiss.sh`, which runs
   `src/kiss/agents/vscode/scripts/apply-brand.js` to rewrite
   `package.json`'s display strings from `media/brand.json` and bundles the branded media into
   `kiss_project`.
3. `npm run package` builds the VSIX.
4. `restore_brand_overlay` always runs, including after a failed build. It copies the
   snapshot back. `package.json` gets its pre-build content plus the version that
   `copy-kiss.sh` synced, which is the only change a stock build makes to the manifest (a small
   python snippet merges them).

## Why not edit the tracked files
The Update button's preflight (`git stash`, `git reset --hard @{upstream}`, `git stash pop`)
would conflict on every release, because the branded display strings in `package.json` sit
next to the version line each release bumps. Restoring from the snapshot rather than with
`git checkout` keeps unrelated local edits, and also works in a plain directory copy that is
not a git checkout. The checked-in files always carry the stock KISS Sorcar brand (commit "keep
KISS Sorcar as the default brand; rebrand only via install.sh overlay").

## Test
`src/kiss/tests/test_install_brand_overlay.py` extracts the two functions and the step-4 build
block from `install.sh` and runs them under bash against a throwaway git repo with a stub `npm`.
It checks that the installed build is branded and the checkout ends up unchanged, with and
without `.brand/`.

## Sources
- `install.sh` (`apply_brand_overlay`, `restore_brand_overlay`, `BRAND_OVERLAY_FILES`, step `[4/5]`)
- `.gitignore` (`/.brand/`)
- `README.md` (Branding paragraph)
- `src/kiss/tests/test_install_brand_overlay.py`
