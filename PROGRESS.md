# Progress Log — Review-driven follow-up task

## Current task

Review the update-notification work from the previous task, reproduce bugs
reported by the review with end-to-end tests, fix those bugs, run the tests in
parallel, review the fixes, and repeat until no reported bugs remain.

Model note: `set_model("gpt-5.5")` and `set_model("claude-opus-4-7")` both
reported deferred model changes only; neither model is live in this environment.
The review and fix passes were therefore performed by the active model.

## Reviewed change set from previous task

- `src/kiss/agents/vscode/media/main.js`: added visible sticky state for
  notifications, SVG object actions, and the permanent
  `kiss-update-available` toast for `update_available` daemon broadcasts.
- `src/kiss/agents/vscode/media/main.css`: added icon/action alignment CSS.
- `src/kiss/agents/vscode/test/updateNotification.test.js`: added the original
  end-to-end update-toast regression test.
- `src/kiss/agents/vscode/package.json`: added that test to the extension test
  chain.

## Review findings

The thorough follow-up review identified these actionable bugs/regression risks:

1. `showNotification` registered `mouseleave`/`focusout` listeners only on first
   toast creation, and those listeners closed over the first render's
   `severity` and `sticky` values. Reusing a notification id with different
   sticky semantics could leave a now-transient toast alive forever or dismiss a
   now-sticky toast.
1. The new in-webview-only update notification used an object action with a
   local `onClick`, but the X close button still called
   `removeNotification(id, undefined, actions.length > 0)`. Closing the update
   toast therefore posted an extension `notificationAction` message that has no
   useful consumer.
1. Re-broadcasting `update_available` with a newer `latest` version should keep
   one stable toast while refreshing both the visible message and the action
   `aria-label`. Review suggested this was likely correct but needed a guard.

Lower-severity notes were reviewed and not converted into fixes in this pass:
error swallowing in local action handlers, `status`/`polite` ARIA semantics, and
CSS layout changes for existing string action buttons. Existing tests cover the
legacy string-action path.

## Tests added

Added `src/kiss/agents/vscode/test/updateNotification_review.test.js`, a real
JSDOM webview end-to-end test that evaluates `media/panelCopy.js` and
`media/main.js` and sends real `message` events. It covers:

- sticky-state refresh for a reused notification id:
  `sticky:true` → `sticky:false` → hover clears the timer → mouseleave must
  schedule a new 10-second auto-dismiss timer using the current state;
- X-close on the update toast dismisses locally and does not post
  `{type: "notificationAction"}`;
- repeated `update_available` broadcasts with `latest: "9999.1.0"` reuse the
  single toast and refresh the visible text plus `aria-label`.

Before the fix, the new test failed with:

```text
AssertionError [ERR_ASSERTION]: leaving a reused now-transient toast must reschedule auto-dismiss using current sticky state
```

This reproduced the stale-closure bug.

## Fixes implemented

### `src/kiss/agents/vscode/media/main.js`

- Added local-action detection:

```js
const hasLocalActions = actions.some(
  action =>
    action &&
    typeof action === 'object' &&
    !Array.isArray(action) &&
    typeof action.onClick === 'function',
);
const notifyOnClose = actions.length > 0 && !hasLocalActions;
```

- Changed hover/focus lifecycle listeners to read current toast state instead of
  stale first-render closure values:

```js
toast.addEventListener('mouseleave', () => {
  const state = toast.kissNotificationState || {
    id: id,
    severity: 'info',
    sticky: false,
  };
  scheduleNotificationDismiss(state.id, state.severity, state.sticky);
});
toast.addEventListener('focusout', () => {
  const state = toast.kissNotificationState || {
    id: id,
    severity: 'info',
    sticky: false,
  };
  scheduleNotificationDismiss(state.id, state.severity, state.sticky);
});
```

- Refreshed `toast.kissNotificationState` on every render:

```js
toast.kissNotificationState = {id: id, severity: severity, sticky: sticky};
```

- Changed the close button to use the current render's notification-close
  policy:

```js
closeBtn.addEventListener('click', () =>
  removeNotification(id, undefined, notifyOnClose),
);
```

### `src/kiss/agents/vscode/package.json`

Added `node test/updateNotification_review.test.js` to the `npm test` chain.

## Verification performed

- `cd src/kiss/agents/vscode && node test/updateNotification_review.test.js`:
  failed before the fix, then passed after the fix.
- `cd src/kiss/agents/vscode && npm run compile && find test -name '*.test.js' -print0 | xargs -0 -n 1 -P 8 node`: all 32 VS Code extension
  tests passed in parallel after compiling.
- `uv run check --full`: all code, type, VS Code extension, tests, and
  markdown formatting passed. The first run exposed an unformatted
  `PROGRESS.md`; after formatting this file, the full check was re-run and
  passed.

## Final review loop status

After the fixes, the review no longer found an actionable reproduced bug in the
update-notification changes. The local-action close semantics remain compatible
with existing string actions, and the stable update-toast id refreshes message
and `aria-label` without stacking duplicates.
