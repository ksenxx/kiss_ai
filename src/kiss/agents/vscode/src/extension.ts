// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * KISS Sorcar VS Code Extension entry point.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {SorcarSidebarView} from './SorcarSidebarView';
import {isReloadReady} from './reloadGuard';

import {ensureDependencies, ensureLocalBinInPath} from './DependencyInstaller';

let sidebarView: SorcarSidebarView | undefined;

export function activate(context: vscode.ExtensionContext): void {
  ensureLocalBinInPath();
  console.log('KISS Sorcar extension activating...');

  // --- Secondary sidebar chat view ---
  sidebarView = new SorcarSidebarView(context.extensionUri);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'kissSorcar.chatViewSecondary',
      sidebarView,
      {webviewOptions: {retainContextWhenHidden: true}},
    ),
  );
  context.subscriptions.push({dispose: () => sidebarView?.dispose()});

  // --- Commands ---

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.openPanel', () => {
      void sidebarView!.focusChatInput();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.newConversation', async () => {
      await sidebarView!.focusChatInput();
      sidebarView!.newConversation();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.stopTask', () => {
      sidebarView!.stopTask();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.runSelection', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const sel = editor.document.getText(editor.selection);
      if (!sel || !sel.trim()) {
        vscode.window.showInformationMessage('No text selected');
        return;
      }
      sidebarView!.submitTask(sel.trim());
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.insertSelectionToChat', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const sel = editor.selection;
      const text = editor.document.getText(sel);
      if (!text || !text.trim()) {
        vscode.window.showInformationMessage('No text selected');
        return;
      }
      const filePath = vscode.workspace.asRelativePath(editor.document.uri);
      const startLine = sel.start.line + 1;
      const lineCount = sel.end.line - sel.start.line + 1;
      const hunkRef = `text from (line, col)=(${startLine},${lineCount}) to (line, col)=(${startLine},${lineCount}) in PWD/${filePath}`;
      void sidebarView!.appendToInput(hunkRef);
    }),
  );

  let _focusToggling = false;
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.toggleFocus', async () => {
      if (_focusToggling) return;
      _focusToggling = true;
      try {
        if (sidebarView!.hasFocus) {
          // Webview chat has focus → switch to the text editor
          await vscode.commands.executeCommand(
            'workbench.action.focusFirstEditorGroup',
          );
        } else {
          // Editor (or anything else) has focus → focus the sidebar chat
          await sidebarView!.focusChatInput();
        }
      } finally {
        _focusToggling = false;
      }
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.focusEditor', () => {
      vscode.commands.executeCommand('workbench.action.focusFirstEditorGroup');
    }),
  );

  // Commit message generation — sets the Git SCM input box
  const setScmMessage = async (message: string) => {
    try {
      const gitExt = vscode.extensions.getExtension('vscode.git');
      if (gitExt) {
        const git = gitExt.isActive ? gitExt.exports : await gitExt.activate();
        const api = git.getAPI(1);
        if (api.repositories.length > 0) {
          api.repositories[0].inputBox.value = message;
          vscode.commands.executeCommand('workbench.view.scm');
        }
      }
    } catch (err) {
      console.error('[kissSorcar] Failed to set SCM message:', err);
    }
  };

  // Countdown shown in the SCM input box while a commit message is being
  // generated. Starts at 15s, decrements every second, stays at 0 if the
  // agent is still generating. Cleared/replaced when the result arrives.
  const commitCountdownSeconds = 20;
  let stopCommitCountdown: (() => void) | undefined;
  const startCommitCountdown = () => {
    stopCommitCountdown?.();
    let seconds = commitCountdownSeconds;
    void setScmMessage(`Generating in ${seconds}s ...`);
    const interval = setInterval(() => {
      seconds = Math.max(seconds - 1, 0);
      void setScmMessage(`Generating in ${seconds}s ...`);
    }, 1000);
    stopCommitCountdown = () => {
      clearInterval(interval);
      stopCommitCountdown = undefined;
    };
  };

  context.subscriptions.push(
    sidebarView!.onCommitMessage(ev => {
      const countdownWasRunning = stopCommitCountdown !== undefined;
      stopCommitCountdown?.();
      if (ev.error) {
        vscode.window.showWarningMessage(`Commit message: ${ev.error}`);
        if (countdownWasRunning) void setScmMessage('');
      } else if (ev.message) {
        void setScmMessage(ev.message);
      } else if (countdownWasRunning) {
        void setScmMessage('');
      }
    }),
  );

  // Returns true iff the first Git repository has at least one staged change.
  const hasStagedChanges = async (): Promise<boolean> => {
    try {
      const gitExt = vscode.extensions.getExtension('vscode.git');
      if (!gitExt) return true; // Can't check — let generation proceed.
      const git = gitExt.isActive ? gitExt.exports : await gitExt.activate();
      const api = git.getAPI(1);
      if (api.repositories.length === 0) return true;
      return api.repositories[0].state.indexChanges.length > 0;
    } catch (err) {
      console.error('[kissSorcar] Failed to check staged changes:', err);
      return true;
    }
  };

  const triggerCommitMessageGeneration = async (
    _rootUri?: unknown,
    _context?: unknown,
    token?: vscode.CancellationToken,
  ): Promise<void> => {
    if (!(await hasStagedChanges())) {
      await setScmMessage('Error: nothing staged');
      return;
    }
    startCommitCountdown();
    token?.onCancellationRequested(() => stopCommitCountdown?.());
    return sidebarView!.generateCommitMessage(token);
  };

  context.subscriptions.push(
    vscode.commands.registerCommand(
      'kissSorcar.generateCommitMessage',
      triggerCommitMessageGeneration,
    ),
  );

  for (const cmdId of [
    'github.copilot.git.generateCommitMessage',
    'git.generateCommitMessage',
  ]) {
    try {
      context.subscriptions.push(
        vscode.commands.registerCommand(cmdId, triggerCommitMessageGeneration),
      );
    } catch {
      // Already registered by another extension — ignored
    }
  }

  // Merge commands — route to the active tab's MergeManager
  for (const cmd of [
    'acceptChange',
    'rejectChange',
    'prevChange',
    'nextChange',
    'acceptAll',
    'rejectAll',
    'acceptFile',
    'rejectFile',
  ] as const) {
    context.subscriptions.push(
      vscode.commands.registerCommand(`kissSorcar.${cmd}`, () => {
        sidebarView!.handleMergeCommand(cmd);
      }),
    );
  }

  // Auto-reload when this extension's files are replaced (e.g. VSIX reinstall).
  // fs.watchFile uses stat-polling so it works even when the file is deleted
  // and recreated, which is what happens during VSIX installation.
  //
  // Two watchers are needed:
  //   1. extension.js — fires when the VSIX is reinstalled with the *same*
  //      version (files overwritten in-place).
  //   2. ~/.kiss/.extension-updated marker — fires when build-extension.sh
  //      (or install.sh / release.sh) writes the marker *after* installing a
  //      *new* version.  A version bump puts the new extension in a separate
  //      directory, so the old extension.js is never modified and watcher #1
  //      never fires.
  const extJsPath = path.join(context.extensionPath, 'out', 'extension.js');
  const markerPath = path.join(os.homedir(), '.kiss', '.extension-updated');
  const sockPath = path.join(os.homedir(), '.kiss', 'sorcar.sock');

  let reloadTriggered = false;
  let settleTimer: ReturnType<typeof setInterval> | undefined;

  const doReload = () => {
    if (reloadTriggered) return;
    reloadTriggered = true;
    if (settleTimer) {
      clearInterval(settleTimer);
      settleTimer = undefined;
    }
    fs.unwatchFile(extJsPath);
    fs.unwatchFile(markerPath);
    vscode.commands.executeCommand('workbench.action.reloadWindow');
  };

  // A same-version ``code --install-extension --force`` first DELETES the
  // extension directory and then re-extracts it, so ``out/extension.js`` is
  // transiently missing (and briefly partially written) for a noticeable
  // window.  Reloading during that window brings the chat webview up against
  // a half-installed extension (its ``media`` resources are gone) and before
  // ``install.sh`` has restarted the kiss-web daemon — the chat view renders
  // blank.  Instead of reloading the instant a watcher fires, poll until the
  // reinstall has fully settled: ``out/extension.js`` present, non-empty and
  // size-stable across two consecutive polls, AND the daemon socket is back so
  // the reloaded webview can reconnect immediately.  A hard timeout reloads
  // anyway so a daemon that never comes back can't strand the user on stale
  // code.
  const RELOAD_SETTLE_INTERVAL_MS = 500;
  const RELOAD_SETTLE_TIMEOUT_MS = 60_000;
  const triggerReload = () => {
    if (reloadTriggered || settleTimer) return;
    let prevSize = -1;
    let waited = 0;
    settleTimer = setInterval(() => {
      waited += RELOAD_SETTLE_INTERVAL_MS;
      const {ready, size} = isReloadReady(extJsPath, sockPath, prevSize);
      prevSize = size;
      if (ready || waited >= RELOAD_SETTLE_TIMEOUT_MS) {
        doReload();
      }
    }, RELOAD_SETTLE_INTERVAL_MS);
  };

  // Two watchers feed the same settle gate:
  //   1. extension.js — fires when the VSIX is reinstalled with the *same*
  //      version (files overwritten in-place).
  //   2. ~/.kiss/.extension-updated marker — fires when install.sh /
  //      build-extension.sh write the marker *after* installing the
  //      extension and restarting the daemon.
  fs.watchFile(extJsPath, {interval: 2000}, (curr, prev) => {
    if (curr.mtimeMs !== prev.mtimeMs || curr.ino !== prev.ino) {
      triggerReload();
    }
  });

  fs.watchFile(markerPath, {interval: 2000}, (curr, prev) => {
    // Only reload when the marker is *created* or *modified* (size > 0),
    // not when ensureDependencies() deletes it (size === 0).
    if (curr.size > 0 && curr.mtimeMs !== prev.mtimeMs) {
      triggerReload();
    }
  });

  context.subscriptions.push({
    dispose: () => {
      if (settleTimer) {
        clearInterval(settleTimer);
        settleTimer = undefined;
      }
      fs.unwatchFile(extJsPath);
      fs.unwatchFile(markerPath);
    },
  });

  // Register tree view so the activity-bar icon opens the sidebar on click.
  const treeView = vscode.window.createTreeView('kissSorcar.chatView', {
    treeDataProvider: {
      getTreeItem: (el: string) => new vscode.TreeItem(el),
      getChildren: () => [],
    },
  });
  context.subscriptions.push(treeView);

  treeView.onDidChangeVisibility(async e => {
    if (e.visible) {
      // Switch primary sidebar away from the KS tree view so the icon never
      // toggles/closes the sidebar on repeated clicks.
      await vscode.commands.executeCommand('workbench.view.explorer');
      // Show the KISS Sorcar chat in the secondary sidebar
      await sidebarView!.focusChatInput();
    }
  });

  // Widen the secondary sidebar on first activation so the chat panel's
  // width is approximately one-third of the VS Code window.  We can't read
  // the workbench window width from an extension, but the webview can
  // report its own ``window.innerWidth`` (= sidebar width) and
  // ``screen.availWidth`` (close proxy for window width when maximized).
  // ``widenToOneThird`` runs an iterative measure→increase/decrease loop
  // until the sidebar is within ~6 % of the target.
  if (!context.globalState.get<boolean>('sidebarWidened')) {
    sidebarView!.onFirstResolve(() => {
      setTimeout(async () => {
        await vscode.commands.executeCommand(
          'workbench.action.focusAuxiliaryBar',
        );
        await sidebarView!.widenToOneThird();
        await vscode.commands.executeCommand(
          'workbench.action.focusFirstEditorGroup',
        );
        await context.globalState.update('sidebarWidened', true);
      }, 500);
    });
  }

  // Decide whether to auto-open the secondary sidebar.
  // True on first-ever launch (firstLaunchDone is undefined) or after a
  // rebuild/reinstall (marker written by build-extension.sh).  We use a
  // local boolean because globalState.update() is async and the get()
  // below would still see the stale value.
  const extensionUpdatedMarker = path.join(
    os.homedir(),
    '.kiss',
    '.extension-updated',
  );
  let shouldAutoOpen = !context.globalState.get<boolean>('firstLaunchDone');
  if (fs.existsSync(extensionUpdatedMarker)) {
    shouldAutoOpen = true;
    void context.globalState.update('firstLaunchDone', undefined);
  }

  // On first launch after install, auto-open the secondary sidebar chat
  // and focus the input so the user can start typing immediately.
  if (shouldAutoOpen) {
    setTimeout(async () => {
      await sidebarView!.focusChatInput();
      await context.globalState.update('firstLaunchDone', true);
    }, 1000);
  }

  // Auto-install dependencies in background
  ensureDependencies().catch(err => {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[KISS Sorcar] Dependency setup error:', err);
    vscode.window.showErrorMessage(
      `KISS Sorcar: Setup failed — ${msg}. Check ~/.kiss/install.log for details.`,
    );
  });

  console.log('KISS Sorcar extension activated');
}

export function deactivate(): void {
  sidebarView?.dispose();
  sidebarView = undefined;
  console.log('KISS Sorcar extension deactivated');
}
