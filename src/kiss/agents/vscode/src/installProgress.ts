// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Live "Installing KISS Sorcar" notification while ./install.sh runs.
//
// install.sh publishes its current step to $KISS_HOME/.install-progress
// (`report_step` in install.sh): line 1 is the installer's pid, line 2 the
// step text ("[4/5] Building VS Code extension...").  It removes the file
// when it exits.  This module polls that file and mirrors it as a native,
// non-blocking VS Code progress notification: a spinner toast in the
// bottom-right corner whose message follows the step, closed when the file
// disappears or the installer process is gone.
//
// The toast is deliberately a native VS Code notification and not a chat
// webview toast (WebviewNotifications.ts): while install.sh runs the chat
// webview may be collapsed, not yet created, or mid-reload, and the user
// running install.sh from a terminal has no chat open at all.

import * as vscode from 'vscode';
import * as fs from 'fs';
import {PRODUCT_NAME} from './brand';

/** File name of the progress file inside $KISS_HOME. */
export const INSTALL_PROGRESS_FILE = '.install-progress';

/** Poll interval of the progress file, in milliseconds. */
const INSTALL_PROGRESS_POLL_MS = 1000;

/** One published install step. */
export interface InstallProgress {
  /** Pid of the running install.sh. */
  pid: number;
  /** Step text, e.g. "[4/5] Building VS Code extension...". */
  message: string;
}

/**
 * Parse the progress file written by install.sh.
 *
 * Returns undefined when the file is missing, still being written (no
 * step line yet) or malformed.
 *
 * @param progressPath Absolute path of the progress file.
 * @returns The published step, or undefined.
 */
export function readInstallProgress(
  progressPath: string,
): InstallProgress | undefined {
  let text: string;
  try {
    text = fs.readFileSync(progressPath, 'utf8');
  } catch {
    return undefined;
  }
  const [pidLine, ...rest] = text.split('\n');
  const pid = Number.parseInt(pidLine, 10);
  const message = rest.join('\n').trim();
  if (!Number.isInteger(pid) || pid <= 0 || !message) {
    return undefined;
  }
  return {pid, message};
}

/**
 * Whether the installer process is still running.
 *
 * `kill(pid, 0)` sends no signal and only checks existence; EPERM means
 * the process exists but belongs to another user.  On Windows install.sh
 * runs under an MSYS bash whose `$$` is not a Windows pid, so the check is
 * skipped and the file's presence alone keeps the toast open.
 *
 * @param pid Pid published by install.sh.
 * @returns True when the process exists.
 */
export function isInstallerAlive(pid: number): boolean {
  if (process.platform === 'win32') {
    return true;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return (err as NodeJS.ErrnoException).code === 'EPERM';
  }
}

/**
 * One native progress notification whose message follows the install step.
 *
 * VS Code hands the progress reporter to the task callback asynchronously,
 * so the toast tracks its own state: a message set before the reporter
 * arrives is shown once it does, and a close before then resolves the task
 * as soon as it starts, so no spinner is left open.
 */
class InstallToast {
  private progress: vscode.Progress<{message?: string}> | undefined;
  private closed = false;
  private resolveTask: (() => void) | undefined;

  /**
   * Open the notification with `message` as its first step.
   *
   * @param message Step text shown next to the title.
   */
  constructor(public message: string) {
    void vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Installing ${PRODUCT_NAME}`,
        cancellable: false,
      },
      progress => this.run(progress),
    );
  }

  private run(progress: vscode.Progress<{message?: string}>): Promise<void> {
    return new Promise<void>(resolve => {
      if (this.closed) {
        resolve();
        return;
      }
      this.progress = progress;
      this.resolveTask = resolve;
      progress.report({message: this.message});
    });
  }

  /**
   * Show a new step, if it differs from the current one.
   *
   * @param message Step text to show.
   */
  update(message: string): void {
    if (message === this.message) {
      return;
    }
    this.message = message;
    this.progress?.report({message});
  }

  /** Close the notification. */
  close(): void {
    this.closed = true;
    this.resolveTask?.();
    this.resolveTask = undefined;
  }
}

/**
 * Mirrors install.sh's progress file as a native progress notification.
 *
 * Polls `progressPath` every `pollMs`.  While the file names a live
 * installer an "Installing <product>" toast shows its current step; the
 * toast closes when the file is removed (install.sh exited) or the
 * installer pid is gone (install.sh was killed and could not clean up); a
 * file left behind by a dead installer is ignored until the next install
 * overwrites it.
 */
export class InstallProgressWatcher implements vscode.Disposable {
  private toast: InstallToast | undefined;
  private readonly timer: NodeJS.Timeout;

  /**
   * Start polling.
   *
   * @param progressPath Absolute path of the progress file.
   * @param pollMs Poll interval in milliseconds.
   */
  constructor(
    private readonly progressPath: string,
    pollMs: number = INSTALL_PROGRESS_POLL_MS,
  ) {
    this.tick();
    // unref: a poll timer must never keep the process alive on its own.
    this.timer = setInterval(() => this.tick(), pollMs);
    this.timer.unref();
  }

  /** Re-read the progress file and open, update or close the toast. */
  tick(): void {
    const current = readInstallProgress(this.progressPath);
    if (!current) {
      this.closeToast();
      return;
    }
    if (!isInstallerAlive(current.pid)) {
      // Left behind by a killed installer.  Ignored, not deleted: the
      // next install overwrites the same path, and a delayed removal
      // could take that install's first step with it.
      this.closeToast();
      return;
    }
    if (this.toast) {
      this.toast.update(current.message);
    } else {
      this.toast = new InstallToast(current.message);
    }
  }

  private closeToast(): void {
    this.toast?.close();
    this.toast = undefined;
  }

  /** Stop polling and close an open toast. */
  dispose(): void {
    clearInterval(this.timer);
    this.closeToast();
  }
}
