// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * VS Code-like notifications rendered inside the KISS Sorcar chat webview.
 */

import * as vscode from 'vscode';

type Severity = 'info' | 'warning' | 'error';
type NotificationPost = (message: Record<string, unknown>) => void;

let poster: NotificationPost | undefined;
let nextId = 1;
const pendingActions = new Map<string, (value: string | undefined) => void>();

function resolveAllPendingActions(): void {
  const resolvers = Array.from(pendingActions.values());
  pendingActions.clear();
  for (const resolve of resolvers) resolve(undefined);
}

/**
 * Register the active chat webview poster used by notification helpers.
 *
 * When no webview is registered, helpers fall back to VS Code's native
 * notification APIs so startup/install messages are still visible.
 */
export function setWebviewNotificationPoster(
  notificationPoster: NotificationPost | undefined,
): void {
  // Any poster change (clear OR replace) must resolve pending action
  // promises that were registered against the previous poster.  The
  // previous webview is no longer reachable so its toasts can never
  // produce a notificationAction reply, and the new webview (if any)
  // does not know about the old IDs.  Leaving them pending would hang
  // flows like API-key prompts forever.
  if (notificationPoster !== poster) {
    resolveAllPendingActions();
  }
  poster = notificationPoster;
}

/** Resolve a pending webview-notification action selected by the user. */
export function resolveWebviewNotificationAction(
  id: string,
  action: string | undefined,
): void {
  const resolve = pendingActions.get(id);
  if (!resolve) return;
  pendingActions.delete(id);
  resolve(action);
}

function splitMessageArgs(items: readonly unknown[]): {
  options: vscode.MessageOptions | undefined;
  actions: string[];
} {
  let options: vscode.MessageOptions | undefined;
  const actions: string[] = [];
  for (const item of items) {
    if (typeof item === 'string') {
      actions.push(item);
    } else if (item && typeof item === 'object' && !Array.isArray(item)) {
      options = item as vscode.MessageOptions;
    }
  }
  return {options, actions};
}

function nativeShow(
  severity: Severity,
  message: string,
  options: vscode.MessageOptions | undefined,
  actions: readonly string[],
): Thenable<string | undefined> {
  if (severity === 'error') {
    return vscode.window.showErrorMessage(message, options || {}, ...actions);
  }
  if (severity === 'warning') {
    return vscode.window.showWarningMessage(message, options || {}, ...actions);
  }
  return vscode.window.showInformationMessage(
    message,
    options || {},
    ...actions,
  );
}

function showNotification(
  severity: Severity,
  message: string,
  ...items: unknown[]
): Thenable<string | undefined> {
  const {options, actions} = splitMessageArgs(items);
  if (!poster) {
    return nativeShow(severity, message, options, actions);
  }
  const id = String(nextId++);
  poster({
    type: 'notification',
    id,
    severity,
    message,
    actions,
    sticky: !!options?.modal || actions.length > 0,
  });
  if (actions.length === 0) return Promise.resolve(undefined);
  return new Promise(resolve => {
    pendingActions.set(id, resolve);
  });
}

/** Show an informational KISS notification in the chat webview. */
export function showInformationNotification(
  message: string,
  ...items: unknown[]
): Thenable<string | undefined> {
  return showNotification('info', message, ...items);
}

/** Show a warning KISS notification in the chat webview. */
export function showWarningNotification(
  message: string,
  ...items: unknown[]
): Thenable<string | undefined> {
  return showNotification('warning', message, ...items);
}

/** Show an error KISS notification in the chat webview. */
export function showErrorNotification(
  message: string,
  ...items: unknown[]
): Thenable<string | undefined> {
  return showNotification('error', message, ...items);
}

/**
 * Run a task with progress shown in the chat webview notification stack.
 */
export function withWebviewNotificationProgress<R>(
  options: vscode.ProgressOptions,
  task: (
    progress: vscode.Progress<{message?: string; increment?: number}>,
    token: vscode.CancellationToken,
  ) => Thenable<R>,
): Thenable<R> {
  if (!poster || options.location !== vscode.ProgressLocation.Notification) {
    return vscode.window.withProgress(options, task);
  }
  const id = String(nextId++);
  const title = options.title || 'KISS Sorcar';
  poster({
    type: 'notification',
    id,
    severity: 'info',
    message: title,
    progress: true,
    sticky: true,
  });
  const progress: vscode.Progress<{message?: string; increment?: number}> = {
    report: value => {
      poster?.({
        type: 'notification',
        id,
        severity: 'info',
        message: title,
        progress: true,
        progressMessage: value.message || '',
        sticky: true,
      });
    },
  };
  const source = new vscode.CancellationTokenSource();
  return Promise.resolve()
    .then(() => task(progress, source.token))
    .finally(() => {
      poster?.({type: 'notification', id, close: true});
      source.dispose();
    });
}
