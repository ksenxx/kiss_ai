---
title: SCM commit message generation from the VS Code extension
uuid: 7f1363b2-fdd7-49bc-a82a-4ed9268c497c
summary: kissSorcar.generateCommitMessage (also hijacking git/copilot ids) - per-repo
  scm:<root> tab ids, 20s countdown in the SCM input box, join-in-flight, serialized
  writes.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# SCM commit message generation from the VS Code extension

## Entry points
- Command `kissSorcar.generateCommitMessage` (SCM title button, `when: scmProvider == git`).
- The same handler is also registered for `github.copilot.git.generateCommitMessage` and `git.generateCommitMessage` (wrapped in `try`, since registering an id another extension owns can throw). VS Code passes the clicked repository's `rootUri` as the first argument and a `CancellationToken` as the third.

## Flow (`extension.ts`)
1. `triggerCommitMessageGeneration(rootUri, _ctx, token)`: the in-flight map `commitGenInFlight` is keyed by `scmTabIdFor(root)` = `scm:<repoRoot>`. A second call for the same repository joins the running promise. The promise is registered synchronously before the first `await`, so two calls in the same tick cannot both start.
2. `runCommitMessageGeneration`: if the repository has no staged changes (`repo.state.indexChanges`), write `Error: nothing staged` into the SCM box and stop. Otherwise create a `CommitGeneration {repoRoot, pending}` in `commitGens`, start a 20 s countdown (`Generating in Ns ...`) in that repository's input box, and call `sidebarView.generateCommitMessage(token, tabId, repoRoot)`, which sends `generateCommitMessage {model, tabId, workDir}` to the daemon (model = the view's selected model). That promise resolves on the matching `commitMessage`, on cancellation, or after a 30 s safety timeout; a duplicate call for a tab already pending returns an already-resolved promise, which is why `extension.ts` joins in-flight generations itself instead of relying on it.
3. The daemon answers `commitMessage {message, error, tabId}`; `SorcarSidebarView` fires `onCommitMessage` for its own tabs. The listener ignores results for canceled or unknown generations (`!gen.pending`), stops the countdown, and writes the message (or clears the box and shows a warning on error).
4. Cancellation (token) marks the generation not pending and clears the countdown.

## Why per-repository state
A workspace can contain several repositories (multi-root, or a nested checkout). Earlier code shared the countdown, the target input box and the request id, and wrote one repository's message into another's box. Now each part is keyed by the repository root: the daemon claims one generation per `tabId`, so different repositories must use different ids.

## SCM input writes
`setScmMessage(message, reveal, repoRoot)` goes through a promise chain (`scmWriteChain`) so an older, slower write (a stale countdown tick) cannot land after a newer one. `pickRepo` selects the repository by `rootUri.fsPath`, falling back to the first. The git API comes from `getGitApi()` (`src/gitApi.ts`, the `vscode.git` extension export).

## Failure paths
- A `generateCommitMessage` command dropped by `AgentClient` (daemon unreachable past the TTL) fires an `onCommitMessage` error "The agent was unreachable", so the countdown and promise do not hang.
- If the git API is unavailable, `hasStagedChanges` returns true (lets the daemon decide).

`kissSorcar.gitCommit` is different: it reveals the chat and runs the webview's manual Git Commit flow (the settings drawer's commit button).

## Sources
- `src/kiss/agents/vscode/src/extension.ts` (`triggerCommitMessageGeneration`, `runCommitMessageGeneration`, `scmTabIdFor`, `setScmMessage`, `startCommitCountdown`, `hasStagedChanges`)
- `src/kiss/agents/vscode/src/SorcarSidebarView.ts` (`generateCommitMessage`, `onCommitMessage`, `_handleDroppedCommand`)
- `src/kiss/agents/vscode/src/gitApi.ts` (`getGitApi`)
- `src/kiss/agents/vscode/package.json` (`menus.scm/title`)
