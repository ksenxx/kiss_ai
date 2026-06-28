# Progress

## Task

> When the user clicks on a filepath in the web chatview, detect it
> and — if it exists — open it in the VS Code editor or a native
> viewer. Tests first, then implementation.

Models:

- **claude-opus-4-7** drove test creation, implementation, and
  bug-fix iterations.
- **gpt-5.5** drove the post-implementation review for missed sites
  and regressions.

## Tests (claude-opus-4-7)

Two end-to-end test files were added.

### `src/kiss/agents/vscode/test/clickFilePathOpens.test.js` (JSDOM)

Drives the real `media/chat.html` + `panelCopy.js` + `main.js` in
jsdom and asserts that rendered chat content surfaces filepath
tokens as clickable `[data-path]` elements which post `openFile` to
the extension on click. 21 sub-tests cover:

1. Absolute, dot-relative, workspace-relative, parent-relative, and
   home-relative paths in `prompt` events.
1. Path with `:line` suffix.
1. Paths in `result` summary and `result` text.
1. Paths in streamed `text_delta` / `text_end` flow.
1. Paths in bash `system_output` stream and non-Bash `tool_result`
   content (success branch's `bash-panel`).
1. Paths in `tool_result` error content (`.tr-content`).
1. Paths inside fenced code blocks.
1. URL path slices (`https://x/y`) are NOT linkified (lookbehind
   guard).
1. Bare filenames (no slash) are NOT linkified.
1. Trailing punctuation (`,`, `.`) is stripped from path matches.
1. Click on the linkified span posts `{type:'openFile', path}` and
   parses `:NN` into `line:NN`.
1. Text inside an `<a>` element is NOT double-linkified.
1. Existing tool_call `path:` argument (`.tp[data-path]`) keeps
   posting `openFile` (no regression).
1. Linkifier is idempotent — no nested `data-path` elements.

### `src/kiss/agents/vscode/test/openFileNativeViewer.test.js`

Drives the compiled `out/SorcarSidebarView.js` with a stubbed
`vscode` module and a stub UDS daemon, mirroring the harness from
`autocommitProgressSticky.test.js`. 7 sub-tests cover the
extension's `openFile` handler:

1. Text source file → `openTextDocument` + `showTextDocument` only.
1. Text source file with `line:N` → cursor on 0-indexed line `N-1`,
   `revealRange` fires.
1. Image file (`.png`) → `vscode.commands.executeCommand('vscode.open', uri)`,
   NOT `openTextDocument`.
1. PDF file (`.pdf`) → same as image.
1. Path outside workspace → silently refused (no call into vscode).
1. Non-existent path → silently refused.
1. Directory → silently refused (`isFile()` guard).

## Implementation (claude-opus-4-7)

### Webview linkifier — `src/kiss/agents/vscode/media/main.js`

New `linkifyFilePaths(root)` helper (just above `hlBlock`) walks
text nodes under `root`, skipping nodes inside `<a>`, `<script>`,
`<style>`, `<textarea>`, `<input>`, `<button>`, `<select>` or any
ancestor already carrying a `data-path` attribute. Each matching
token is replaced with
`<span class="kiss-filelink" data-path="..." title="Open ...">`.

The matcher uses one regex:

```js
const _LINK_FILEPATH_RE =
  /(?<![\w@:%/.~-])((?:(?:~|\.{1,2})?\/|[A-Za-z0-9_+-]+\/)[A-Za-z0-9_./+-]*[A-Za-z0-9_+/-](?::\d+)?)/g;
```

- The pattern accepts absolute (`/tmp/a.py`), home-relative
  (`~/work/a.py`), dot-relative (`./src/a.py`, `../README.md`), and
  workspace-relative paths with a directory component
  (`src/kiss/INJECTIONS.md`).
- The lookbehind rejects matches inside URLs (`m/foo` etc.) so
  `https://x.com/foo` is NOT mis-linkified as the path `/foo`.
- The inner pattern requires at least one `/` and either an explicit
  absolute/home/dot-relative prefix or a leading directory component,
  so bare filenames like `package.json` and ambiguous tokens like
  `v1.0` are ignored.
- The final character class (`[A-Za-z0-9_+\-/]`) excludes trailing
  `.`/`,` so sentence punctuation does not leak into the path.
- `(?::\d+)?` captures an optional `:NN` line suffix.

The linkifier is wired in at every chat-content render site:

| Site | Where |
| --- | --- |
| `prompt` / `system_prompt` body | `bodyEl` after `innerHTML` |
| `text_end` (streamed text) | `tState.txtEl` |
| `tool_result` success bash-panel | `opContent` |
| `tool_result` error `.tr-content` | inside `r` |
| `system_output` (bash stream RAF flush) | `tState.bashPanel` |
| `system_output` (non-bash) `<div class="ev sys">` | `s` |
| `tool_call` flushing prior bash buffer | `tState.bashPanel` |
| `tool_result` flushing prior bash buffer | `tState.bashPanel` |
| `result` panel `.rc-body` | `rcBody` in `createResultPanel` |

The existing global click handler (`document.addEventListener('click'…)`)
already routes `[data-path]` clicks to `vscode.postMessage({type:'openFile', …})`
with `path` / `line` parsed from a `path:NN` suffix, so the
linkified spans inherit that wiring for free.

CSS hook in `media/main.css`:

```css
.kiss-filelink {
  color: var(--cyan);
  cursor: pointer;
  text-decoration: underline dotted;
  word-break: break-all;
}
.kiss-filelink:hover {
  color: color-mix(in srgb, var(--accent) 82%, transparent);
  text-decoration: underline solid;
}
```

`NodeFilter` was added to `eslint.config.mjs` webview globals
since the TreeWalker constructor uses it.

### Native-viewer routing — `src/kiss/agents/vscode/src/SorcarSidebarView.ts`

The `openFile` handler now dispatches based on the file extension:

- Text-like (default): `vscode.workspace.openTextDocument` +
  `vscode.window.showTextDocument`, with optional cursor positioning
  on the requested 1-indexed line.
- Native-viewer-only (binary): `vscode.commands.executeCommand('vscode.open', uri)`
  delegates to whatever viewer VS Code has registered for the type
  (image preview, PDF preview, default OS app, …).

A new module-level constant `NATIVE_VIEWER_EXTENSIONS` enumerates the
binary / preview-only extensions (images, PDFs, archives, Office
docs, native binaries, audio/video, fonts, compiled artefacts).
A new helper `isTextLikeExtension(filePath)` returns `true` when the
extension is absent or not in the set. Defaulting to text means
config / dotfiles / unknown source extensions keep opening in the
editor.

All of the prior safety gates (`isPathInside`, `fs.existsSync`,
`isFile`) still run before either branch so the new wiring inherits
the original H4 path-traversal defence.

## Review (gpt-5.5)

- Verified every chat-rendering site in `media/main.js` now feeds
  its rendered body through `linkifyFilePaths`:
  - `handleOutputEvent` text_end (≈ line 2425): ✔
  - `handleOutputEvent` tool_call bash-buffer flush (≈ 2443): ✔
  - `handleOutputEvent` tool_result bash-buffer flush + finalise
    (≈ 2526): ✔
  - `handleOutputEvent` tool_result error `.tr-content` (≈ 2546): ✔
  - `handleOutputEvent` tool_result success `opContent` (≈ 2554): ✔
  - `handleOutputEvent` system_output bash RAF flush (≈ 2567): ✔
  - `handleOutputEvent` system_output non-bash sys panel (≈ 2581): ✔
  - `handleOutputEvent` prompt / system_prompt body (≈ 2614): ✔
  - `createResultPanel` `.rc-body` (≈ 2129): ✔
- Replay (`replayEventsInto`) and background-tab streaming
  (`processOutputEventForBgTab`) both delegate to `handleOutputEvent`
  so they pick up the linkifier without separate wiring; spot-grepped
  to confirm no chat-content innerHTML lives outside that helper.
- Path detector accepts workspace-relative paths such as
  `src/kiss/INJECTIONS.md` while still requiring a directory
  component, so bare filenames remain unlinked.
- Path detector lookbehind covers URL hosts (`m/foo`), arbitrary
  port:port chains (`:8080/foo`), and prior `.`-/`-`-/`~`-bearing
  identifiers, so cases that would otherwise emit spurious
  `data-path` tokens are rejected.
- `_LINK_FILEPATH_RE` `lastIndex` is reset to 0 before every
  use to avoid the stateful-regex gotcha across multiple text
  nodes.
- Extension-side `openFile` still runs `isPathInside`, `fs.existsSync`
  and `isFile()` BEFORE either branch — non-existent paths,
  directories, and outside-workspace paths are silently refused, as
  the previous behaviour required.
- `NATIVE_VIEWER_EXTENSIONS` defaults to **text** for unknown
  extensions; this preserves the prior behaviour for source code,
  dotfiles, and config files while only routing well-known binary
  formats to `vscode.open`.
- Continuation review found one missed usability case: common
  workspace-relative paths like `src/kiss/INJECTIONS.md` were not
  linkified by the initial absolute/dot/home-relative regex. Added
  an E2E assertion for that case first, then broadened the matcher to
  accept a leading directory component while keeping bare filenames
  unlinked.
- Ran the full VS Code extension suite (`npm test`, 50+ test files
  including the two new ones) and `uv run check --full` — both pass
  (ruff, mypy, pyright, ESLint, stylelint, htmlhint, mdformat, all
  JSDOM/UDS tests).

## Result

`uv run check --full` ⇒ **All checks passed**.

New tests:

- `clickFilePathOpens.test.js`: 21 passed.
- `openFileNativeViewer.test.js`: 7 passed.
