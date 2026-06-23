# PROGRESS — Per-workspace secondary-sidebar bootstrap fix

## Task

When a NEW VS Code workspace is opened, the secondary sidebar should
(a) widen to approximately one-third of the VS Code window and
(b) auto-select the KISS Sorcar tab. Before this fix, both behaviours
only fired on the very first activation per machine because the
gates lived in `context.globalState`.

## Root cause

`src/kiss/agents/vscode/src/extension.ts` gated the widen + auto-open
logic on `context.globalState.get('sidebarWidened')` and
`context.globalState.get('firstLaunchDone')`. `globalState` is
per-install (shared across every workspace on the machine), so the
second and later workspaces on the same machine inherited the
flags from the first workspace and the activate handler skipped both
the `widenToOneThird()` call AND the `focusChatInput()` call.

VS Code stores sidebar widths per workspace, so without the
bootstrap call each new workspace landed at VS Code's default
narrow secondary-sidebar width and at whichever view happened to be
selected — not necessarily the KISS Sorcar tab.

## Fix

`src/kiss/agents/vscode/src/extension.ts`: switched both flags from
`context.globalState` to `context.workspaceState`. The gate is now
per-workspace (the natural granularity), so the bootstrap fires
exactly once per workspace and never overwrites the user's manual
width adjustments on later reopens.

Diff (logical):

```ts
// before
if (!context.globalState.get<boolean>('sidebarWidened')) { ... }
let shouldAutoOpen = !context.globalState.get<boolean>('firstLaunchDone');
void context.globalState.update('firstLaunchDone', undefined);
await context.globalState.update('sidebarWidened', true);
await context.globalState.update('firstLaunchDone', true);

// after
if (!context.workspaceState.get<boolean>('sidebarWidened')) { ... }
let shouldAutoOpen = !context.workspaceState.get<boolean>('firstLaunchDone');
void context.workspaceState.update('firstLaunchDone', undefined);
await context.workspaceState.update('sidebarWidened', true);
await context.workspaceState.update('firstLaunchDone', true);
```

## Reproduction / regression test

`src/kiss/agents/vscode/test/secondarySidebarPerWorkspace.test.js`
loads the real compiled `out/extension.js`, stubs only the modules
that would pull in VS Code/jsdom/the daemon socket (`vscode`,
`SorcarSidebarView`, `DependencyInstaller`, `gitApi`, `reloadGuard`),
and runs `activate()` three times:

1. Workspace 1 — fresh `workspaceState` and fresh `globalState`.
   Asserts `widenToOneThird` fires once and `focusChatInput` fires.
1. Workspace 2 — fresh `workspaceState`, SAME `globalState` memento
   as workspace 1. Asserts both fire again. This is the case that
   reproduces the original bug; before the fix the `globalState`
   flags from workspace 1 suppressed both calls.
1. Workspace 2 reopened — same `workspaceState` memento as #2.
   Asserts neither fires (idempotency — the user's manual width
   tweaks must survive reopens).

Verified by temporarily reverting the compiled bundle to use
`globalState` and confirming the test fails with
`workspace 2: widenToOneThird must fire exactly once on a NEW workspace (got 0)`, then restoring the fix and confirming the test
passes.

## Verification

- `npm run compile` clean.
- `node test/secondarySidebarPerWorkspace.test.js` — 3/3 OK.
- `npm run check` (typecheck + lint + all 27 vscode tests) green.
- `uv run check --full` green at the repo root (ruff, mypy, pyright,
  vscode-check, mdformat).

## Incidental clean-ups

- `src/kiss/agents/vscode/src/DependencyInstaller.ts` — auto-fixed a
  pre-existing prettier violation flagged by `npm run check`.
- `RECIPES.md` — auto-fixed a pre-existing `mdformat` violation
  flagged by `uv run check --full`.
