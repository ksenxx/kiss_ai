---
title: VS Code extension JS tests - running, harnesses, compiled out/, common breakages
uuid: c6e4e43f-4096-4d99-8217-e20c8af88667
summary: 'VS Code extension JS tests: run-all.js (one node per suite, V8 flags, 10
  min timeout), needs compiled out/, vscode stubs, jsdom harness, fake sockets, coverage
  gates, common breakages.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# VS Code extension JS tests

## Running
- `npm test` in `src/kiss/agents/vscode/` = `npm run compile && node test/run-all.js` (`compile` is `tsc -p ./`).
- `test/run-all.js` discovers every `*.test.js` and `*.coverage.js` in `test/` on disk (about 360), sorts them, and runs each in its own node process (`spawnSync`, cwd = extension dir) so suites cannot leak globals, timers or listeners. Failed suites are listed at the end.
- `V8_FLAGS` = `--no-maglev --no-concurrent-sparkplug`: on Node 24+, Maglev or background Sparkplug could leave a V8 worker parked at `process.exit()`, and about one suite in 700 hung after printing "passed" (nodejs/node#64274). They must be on the command line: `NODE_OPTIONS` rejects V8 flags.
- `SUITE_TIMEOUT_MS` = 10 minutes; a hung suite is killed (SIGKILL) and reported as a failure by name.
- Run one suite from the extension dir: `node --no-maglev --no-concurrent-sparkplug test/<name>.test.js`.
- There is no mocha/jest: each suite is a plain node script using `assert`, exiting non-zero on failure.
- `check --full` runs the extension typecheck and lint but never these tests.

## Prerequisite: compiled `out/`
Host-side suites `require(path.join(EXT_ROOT, 'out', ...))`, the compiled TypeScript. In an agent worktree `node_modules` is symlinked but `out/` is not, so run `npm run compile` first; otherwise those suites fail on a missing module. Media-only jsdom suites do not need `out/`.

## Kinds of suites
1. **Webview (jsdom)**, about 230 suites: load `media/chat.html` + `media/main.js` into `JSDOM` with `runScripts: 'dangerously'`. `test/simplify2_harness.js` provides `makeWebview(opts)` (strips `{{...}}` placeholders and script tags, stubs `scrollIntoView`/`scrollTo`, installs an `acquireVsCodeApi` that records posted messages and keeps state, evaluates `panelCopy.js`, `api.js`, `main.js`), `send(win, data)` (dispatches a `MessageEvent`) and `sleep(ms)`. `opts.beforeScripts(win)` runs before `main.js` (e.g. to add `remote-chat` or `editor-tab-mode` to `<body>`, or to wrap timers).
2. **Extension host**: `require` compiled `out/*.js` with no real VS Code API. The common pattern (about 80 suites): build a `vscodeStub`, store it in `global.__kissVscodeStub`, and patch `Module._resolveFilename` so `require('vscode')` resolves to `test/_vscode-stub.js`, which exports `global.__kissVscodeStub` (see `activeTaskSinkPanelManager.test.js`). A few suites instead patch `Module._load` to return the stub when `request === 'vscode'` (`activationUpdateNotificationAction.test.js`).
3. **Daemon transport**: a real `net` server as a fake kiss-web daemon. `test/fakeSock.js` `fakeSockPath(dir, name)` returns a Unix socket path, or a named pipe on Windows (node cannot `listen()` on a filesystem path there); `SOCK_FILE_OPS` marks cases needing a real socket file (false on win32). Point clients at it with `KISS_SORCAR_SOCK`; isolate state with `KISS_HOME` / `HOME` in child processes.
4. **Coverage gates** (`*.coverage.js`): run a functional suite with `NODE_V8_COVERAGE` and require 100% line coverage of a region in `out/*.js` fenced by comments like `// audit0903-coverage:start` / `:end` (`runGate` in `audit0903_voice_intent.coverage.js`). Deleting or moving a fence breaks the gate.

## jsdom pitfalls
- Objects posted by `main.js` live in the jsdom realm; `assert.deepStrictEqual` against a Node object fails on prototype identity. Compare `JSON.parse(JSON.stringify(x))` or individual fields.
- Wrap `win.setInterval`/`setTimeout` before evaluating `main.js` to observe or fire polls without waiting.
- Probe scripts must `win.close()` and `process.exit(0)`, or pending timers keep node alive.

## Common breakages
- Production code gains a new `vscode.*` call (e.g. `vscode.Uri.joinPath` in a constructor) and suites whose stubs lack it fail with `TypeError`. Add the member to those stubs; about 60 suites already define `joinPath`, e.g. `(base, ...parts) => makeUri(path.join(base.fsPath, ...parts))`, so copy one.
- `extension.ts` starts calling a new public method of `SorcarPanelManager`/`SorcarSidebarView`: suites that replace them with hand-written stand-ins must be updated.
- `media/main.js` gains a new `monaco.editor.*` call and suites with a fake `monaco` break; `grep -ln 'monaco = ' test/*.js` finds them.
- Python tests that extract a single function from `media/main.js` break when it calls a new collaborator; stub the collaborator in the extracted snippet.

## Python-side UI tests
`src/kiss/tests/agents/vscode/` (~115 files) holds pytest suites for the same UI: jsdom scripts driven from Python (jsdom found through the extension's `node_modules`), tests that extract one JS function and evaluate it in node, Playwright tests against a real daemon's remote webapp, and protocol tests such as `test_server_api.py` (api.js list vs daemon catalog).

## Sources
- `src/kiss/agents/vscode/package.json` (`test`, `compile`)
- `src/kiss/agents/vscode/test/run-all.js` (`testFiles`, `main`, `V8_FLAGS`, `SUITE_TIMEOUT_MS`)
- `src/kiss/agents/vscode/test/simplify2_harness.js` (`makeWebview`, `send`, `sleep`)
- `src/kiss/agents/vscode/test/_vscode-stub.js`
- `src/kiss/agents/vscode/test/fakeSock.js` (`fakeSockPath`, `SOCK_FILE_OPS`)
- `src/kiss/agents/vscode/test/activeTaskSinkPanelManager.test.js` (`Module._resolveFilename` stub)
- `src/kiss/agents/vscode/test/activationUpdateNotificationAction.test.js` (`Module._load` stub)
- `src/kiss/agents/vscode/test/audit0903_voice_intent.coverage.js` (`runGate`)
- `src/kiss/scripts/check.py` (extension typecheck and lint only)
- `src/kiss/tests/agents/vscode/`
