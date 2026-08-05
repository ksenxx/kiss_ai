// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

export function setWebviewNotificationPoster(
  notificationPoster: NotificationPost | undefined,
): void {
  if (notificationPoster !== poster) {
    resolveAllPendingActions();
  }
  poster = notificationPoster;
}

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

export function showInformationNotification(
  message: string,
  ...items: unknown[]
): Thenable<string | undefined> {
  return showNotification('info', message, ...items);
}

export function showWarningNotification(
  message: string,
  ...items: unknown[]
): Thenable<string | undefined> {
  return showNotification('warning', message, ...items);
}

export function showErrorNotification(
  message: string,
  ...items: unknown[]
): Thenable<string | undefined> {
  return showNotification('error', message, ...items);
}

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
