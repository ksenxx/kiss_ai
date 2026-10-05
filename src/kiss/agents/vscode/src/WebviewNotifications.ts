// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import {ToWebviewMessage} from './types';
import {PRODUCT_NAME} from './brand';

type Severity = 'info' | 'warning' | 'error';
// The toast protocol is part of ToWebviewMessage, so a field renamed here
// and not in media/main.js (or vice versa) fails to compile instead of
// silently rendering an empty toast.
type NotificationMessage = Extract<ToWebviewMessage, {type: 'notification'}>;
type NotificationPost = (message: NotificationMessage) => void;

let poster: NotificationPost | undefined;
let nextId = 1;
const pendingActions = new Map<string, (value: string | undefined) => void>();
// Cancellation sources of the progress operations still running, keyed
// by toast id.  Unlike pendingActions these are not one-shot: closing
// the toast (action undefined) does not end the operation, whose next
// report re-posts the toast with its Cancel button, and that button
// must still cancel.  They also outlive a poster replacement, so the
// surface that receives the re-posted toast can cancel too.
const progressSources = new Map<string, vscode.CancellationTokenSource>();

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

/**
 * Clear the poster only when *notificationPoster* is the one installed.
 *
 * Two surfaces can hold the poster in turn (the sidebar chat view and
 * the editor-tabs panel manager); an unconditional clear on one
 * surface's teardown would silence toasts the other surface had just
 * claimed — e.g. the sidebar webview disposing right after the mode
 * switch to editor tabs installed the panels' poster.
 *
 * @param notificationPoster The poster the caller installed earlier.
 */
export function clearWebviewNotificationPoster(
  notificationPoster: NotificationPost,
): void {
  if (poster === notificationPoster) {
    resolveAllPendingActions();
    poster = undefined;
  }
}

/**
 * Deliver the webview's answer to a toast: the clicked action's label,
 * or undefined when the toast was closed without choosing one.
 *
 * A message toast resolves its pending show*Notification() promise once.
 * A progress toast's Cancel cancels the running operation; closing it
 * leaves the operation (and its Cancel) alone.
 */
export function resolveWebviewNotificationAction(
  id: string,
  action: string | undefined,
): void {
  const source = progressSources.get(id);
  if (source) {
    if (action === 'Cancel') source.cancel();
    return;
  }
  const resolve = pendingActions.get(id);
  if (!resolve) return;
  pendingActions.delete(id);
  resolve(action);
}

/**
 * Options accepted by the show*Notification helpers: VS Code's message
 * options plus `tabId`, the chat tab whose task the toast reports on.
 * A tagged toast is shown only over that tab (and, in editor-tabs mode,
 * only in that tab's panel); native VS Code notifications are
 * window-level and ignore it.
 */
export type NotificationOptions = vscode.MessageOptions & {tabId?: string};

function splitMessageArgs(items: readonly unknown[]): {
  options: NotificationOptions | undefined;
  actions: string[];
} {
  let options: NotificationOptions | undefined;
  const actions: string[] = [];
  for (const item of items) {
    if (typeof item === 'string') {
      actions.push(item);
    } else if (item && typeof item === 'object' && !Array.isArray(item)) {
      options = item as NotificationOptions;
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
  const toast: NotificationMessage = {
    type: 'notification',
    id,
    severity,
    message,
    actions,
    // Errors never auto-dismiss: the user must be able to read the cause
    // and act on it however long the failure takes to notice.
    sticky: severity === 'error' || !!options?.modal || actions.length > 0,
  };
  if (options?.tabId) toast.tabId = options.tabId;
  poster(toast);
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

/**
 * Options for withWebviewNotificationProgress: VS Code's progress options
 * plus `tabId`, the chat tab whose task the progress toast reports on
 * (see NotificationOptions).
 */
export type ProgressNotificationOptions = vscode.ProgressOptions & {
  tabId?: string;
};

export function withWebviewNotificationProgress<R>(
  options: ProgressNotificationOptions,
  task: (
    progress: vscode.Progress<{message?: string; increment?: number}>,
    token: vscode.CancellationToken,
  ) => Thenable<R>,
): Thenable<R> {
  const {tabId, ...nativeOptions} = options;
  if (!poster || options.location !== vscode.ProgressLocation.Notification) {
    return vscode.window.withProgress(nativeOptions, task);
  }
  const id = String(nextId++);
  const title = options.title || PRODUCT_NAME;
  // Every post of the toast's lifecycle (open, update, close) carries
  // the same tab, so a task's progress never pops over another task's
  // chat and the close reaches the toast wherever it was shown.
  const tab = tabId ? {tabId} : {};
  // A cancellable progress toast carries a 'Cancel' action wired to the
  // token the task receives, exactly like the native progress
  // notification's Cancel button.  Every re-post must repeat the
  // actions: the webview re-renders the action row from each event.
  // The source stays registered until the task ends (see
  // progressSources), not until the first click or dismissal.
  const actions = options.cancellable ? ['Cancel'] : [];
  const source = new vscode.CancellationTokenSource();
  if (options.cancellable) progressSources.set(id, source);
  poster({
    type: 'notification',
    id,
    ...tab,
    severity: 'info',
    message: title,
    actions,
    progress: true,
    sticky: true,
  });
  const progress: vscode.Progress<{message?: string; increment?: number}> = {
    report: value => {
      poster?.({
        type: 'notification',
        id,
        ...tab,
        severity: 'info',
        message: title,
        actions,
        progress: true,
        progressMessage: value.message || '',
        sticky: true,
      });
    },
  };
  return Promise.resolve()
    .then(() => task(progress, source.token))
    .finally(() => {
      progressSources.delete(id);
      poster?.({type: 'notification', id, ...tab, close: true});
      source.dispose();
    });
}
