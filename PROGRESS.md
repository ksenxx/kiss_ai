# Progress

## Task: Review last notification-system updates, reproduce/fix review bugs, and verify

Started a new task, so cleared the previous task progress log.

### GPT-5.5 review of last notification-system updates

Reviewed the full notification implementation and call sites, not just the diff. The review found two concrete bugs in `src/kiss/agents/vscode/src/WebviewNotifications.ts`:

1. **Pending action promises can hang forever when the webview/poster is cleared.**

   - `showWarningNotification(..., 'Action')` stores a resolver in `pendingActions`.
   - If the chat webview is disposed or `setWebviewNotificationPoster(undefined)` is called before the user clicks/dismisses the toast, no `notificationAction` message can ever arrive.
   - The promise remains pending, which can hang flows such as API-key prompts.

1. **Progress notifications are not closed if the progress task throws synchronously.**

   - `withWebviewNotificationProgress` currently evaluates `task(progress, source.token)` before `Promise.resolve(...)` is constructed:
     ```ts
     return Promise.resolve(task(progress, source.token)).finally(() => {
       poster?.({type: 'notification', id, close: true});
       source.dispose();
     });
     ```
   - A synchronous throw skips the `.finally(...)`, leaving the progress notification stuck and the cancellation source undisposed.

Planned changes:

- Extend `src/kiss/agents/vscode/test/webviewNotifications.test.js` with E2E regression coverage for both bugs using the real compiled `WebviewNotifications.js` module and the existing real notification/webview test harness.
- Fix `WebviewNotifications.ts` by resolving pending action promises with `undefined` when the poster is cleared, and by wrapping progress task invocation in a promise chain that also catches synchronous throws while preserving rejection semantics.
- Compile, run the impacted tests, then run all VS Code extension tests in parallel after counting them.
- Review fixes again with `gpt-5.5` and repeat if new reproducible bugs are found.

### Reproduction tests added before fixing

Extended `src/kiss/agents/vscode/test/webviewNotifications.test.js` with two E2E-style regressions against the real compiled `WebviewNotifications.js` module:

```js
const pendingActionPromise = notificationsApi.showWarningNotification(
  'Choose a pending action.',
  'Continue',
);
const pendingBeforeDispose = await Promise.race([
  pendingActionPromise.then(value => ({settled: true, value})),
  new Promise(resolve => setTimeout(() => resolve({settled: false}), 80)),
]);
assert.deepStrictEqual(pendingBeforeDispose, {settled: false});
view.dispose();
const pendingAfterDispose = await Promise.race([
  pendingActionPromise.then(value => ({settled: true, value})),
  new Promise(resolve => setTimeout(() => resolve({settled: false}), 200)),
]);
assert.deepStrictEqual(pendingAfterDispose, {settled: true, value: undefined});
```

and:

```js
await notificationsApi.withWebviewNotificationProgress(
  {
    location: vscodeStub.ProgressLocation.Notification,
    title: 'Sync failing progress',
  },
  () => {
    throw thrown;
  },
);
```

The assertions require the original error to be rejected, the cancellation source to be disposed, and a `{type:'notification', close:true}` message to be posted.

Ran `cd src/kiss/agents/vscode && npm run compile && node test/webviewNotifications.test.js` after installing npm deps. The new pending-action test reproduced the first review bug exactly:

```text
AssertionError [ERR_ASSERTION]: disposing the active webview must resolve pending action notifications undefined
+ actual - expected

  {
+   settled: false
-   settled: true,
-   value: undefined
  }
```

The first assertion also verified the action promise was genuinely pending before dispose.
