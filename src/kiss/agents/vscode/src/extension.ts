// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {MERGE_ACTIONS, SorcarSidebarView} from './SorcarSidebarView';
import {getGitApi} from './gitApi';
import {isReloadReady} from './reloadGuard';

import {ensureDependencies, ensureLocalBinInPath} from './DependencyInstaller';
import {findKissProject} from './kissPaths';
import {resetTipsOnExtensionUpdate} from './SorcarTab';
import {checkForExtensionUpdate} from './UpdateChecker';
import {
  showErrorNotification,
  showInformationNotification,
  showWarningNotification,
} from './WebviewNotifications';

let sidebarView: SorcarSidebarView | undefined;

export function activate(context: vscode.ExtensionContext): void {
  ensureLocalBinInPath();
  console.log('KISS Sorcar extension activating...');

  sidebarView = new SorcarSidebarView(context.extensionUri);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'kissSorcar.chatViewSecondary',
      sidebarView,
      {webviewOptions: {retainContextWhenHidden: true}},
    ),
  );
  context.subscriptions.push({dispose: () => sidebarView?.dispose()});

  sidebarView.syncWorkDir();

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
    vscode.commands.registerCommand('kissSorcar.runSelection', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const sel = editor.document.getText(editor.selection);
      if (!sel || !sel.trim()) {
        showInformationNotification('No text selected');
        return;
      }
      await sidebarView!.submitTask(sel.trim());
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.insertSelectionToChat', () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const sel = editor.selection;
      const text = editor.document.getText(sel);
      if (!text || !text.trim()) {
        showInformationNotification('No text selected');
        return;
      }
      const filePath = vscode.workspace.asRelativePath(editor.document.uri);
      const startLine = sel.start.line + 1;
      const startCol = sel.start.character + 1;
      const endLine = sel.end.line + 1;
      const endCol = sel.end.character + 1;
      const hunkRef = `text from (line, col)=(${startLine},${startCol}) to (line, col)=(${endLine},${endCol}) in ./${filePath}`;
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
          await vscode.commands.executeCommand(
            'workbench.action.focusFirstEditorGroup',
          );
        } else {
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

  const setScmMessage = async (message: string, reveal = false) => {
    try {
      const api = await getGitApi();
      if (api && api.repositories.length > 0) {
        api.repositories[0].inputBox.value = message;
        if (reveal) vscode.commands.executeCommand('workbench.view.scm');
      }
    } catch (err) {
      console.error('[kissSorcar] Failed to set SCM message:', err);
    }
  };

  const commitCountdownSeconds = 20;
  let stopCommitCountdown: (() => void) | undefined;
  const startCommitCountdown = () => {
    stopCommitCountdown?.();
    let seconds = commitCountdownSeconds;
    void setScmMessage(`Generating in ${seconds}s ...`, true);
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
      if ((ev.tabId ?? '') !== '') return;
      const countdownWasRunning = stopCommitCountdown !== undefined;
      stopCommitCountdown?.();
      if (ev.error) {
        showWarningNotification(`Commit message: ${ev.error}`);
        if (countdownWasRunning) void setScmMessage('');
      } else if (ev.message) {
        void setScmMessage(ev.message, true);
      } else if (countdownWasRunning) {
        void setScmMessage('');
      }
    }),
  );

  const hasStagedChanges = async (): Promise<boolean> => {
    try {
      const api = await getGitApi();
      if (!api || api.repositories.length === 0) return true;
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
      await setScmMessage('Error: nothing staged', true);
      return;
    }
    startCommitCountdown();
    const teardown = () => {
      if (stopCommitCountdown) {
        stopCommitCountdown();
        void setScmMessage('');
      }
    };
    token?.onCancellationRequested(teardown);
    return sidebarView!.generateCommitMessage(token).finally(teardown);
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
    } catch {}
  }

  for (const cmd of Object.values(MERGE_ACTIONS)) {
    context.subscriptions.push(
      vscode.commands.registerCommand(`kissSorcar.${cmd}`, () => {
        sidebarView!.handleMergeCommand(cmd);
      }),
    );
  }

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
    fs.unwatchFile(markerPath);
    vscode.commands.executeCommand('workbench.action.reloadWindow');
  };

  const RELOAD_SETTLE_INTERVAL_MS = 500;
  const RELOAD_SOCKET_GRACE_MS = 3_000;
  const RELOAD_SETTLE_TIMEOUT_MS = 15_000;
  const triggerReload = () => {
    if (reloadTriggered || settleTimer) return;
    let prevSize = -1;
    let waited = 0;
    let codeReadySince = -1;
    settleTimer = setInterval(() => {
      waited += RELOAD_SETTLE_INTERVAL_MS;
      const {codeReady, socketUp, size} = isReloadReady(
        extJsPath,
        sockPath,
        prevSize,
      );
      prevSize = size;
      if (codeReady && codeReadySince < 0) codeReadySince = waited;
      const codeStableFor = codeReadySince < 0 ? 0 : waited - codeReadySince;
      if (
        (codeReady && (socketUp || codeStableFor >= RELOAD_SOCKET_GRACE_MS)) ||
        waited >= RELOAD_SETTLE_TIMEOUT_MS
      ) {
        doReload();
      }
    }, RELOAD_SETTLE_INTERVAL_MS);
  };

  fs.watchFile(markerPath, {interval: 2000}, (curr, prev) => {
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
      fs.unwatchFile(markerPath);
    },
  });

  const treeView = vscode.window.createTreeView('kissSorcar.chatView', {
    treeDataProvider: {
      getTreeItem: (el: string) => new vscode.TreeItem(el),
      getChildren: () => [],
    },
  });
  context.subscriptions.push(treeView);

  treeView.onDidChangeVisibility(async e => {
    if (e.visible) {
      await vscode.commands.executeCommand('workbench.view.explorer');
      await sidebarView!.focusChatInput();
    }
  });

  if (!context.workspaceState.get<boolean>('sidebarWidened')) {
    sidebarView!.onFirstResolve(() => {
      setTimeout(async () => {
        await vscode.commands.executeCommand(
          'workbench.action.focusAuxiliaryBar',
        );
        await sidebarView!.widenToOneThird();
        await vscode.commands.executeCommand(
          'workbench.action.focusFirstEditorGroup',
        );
        await context.workspaceState.update('sidebarWidened', true);
      }, 500);
    });
  }

  const extensionUpdatedMarker = path.join(
    os.homedir(),
    '.kiss',
    '.extension-updated',
  );
  let shouldAutoOpen = !context.workspaceState.get<boolean>('firstLaunchDone');
  if (fs.existsSync(extensionUpdatedMarker)) {
    shouldAutoOpen = true;
    void context.workspaceState.update('firstLaunchDone', undefined);
  }
  resetTipsOnExtensionUpdate();

  if (shouldAutoOpen) {
    setTimeout(async () => {
      await sidebarView!.focusChatInput();
      await context.workspaceState.update('firstLaunchDone', true);
    }, 1000);
  }

  ensureDependencies().catch(err => {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[KISS Sorcar] Dependency setup error:', err);
    showErrorNotification(
      `KISS Sorcar: Setup failed — ${msg}. Check ~/.kiss/install.log for details.`,
    );
  });

  void checkForExtensionUpdate({
    kissProjectPath: findKissProject() || undefined,
    notify: ({latest, current}: {latest: string; current: string}) => {
      void showInformationNotification(
        `KISS Sorcar: a new release (${latest}) is available. ` +
          `You are on ${current}.`,
        'Update now',
      ).then(action => {
        if (action === 'Update now') {
          sidebarView?.runUpdate();
        }
      });
    },
  }).catch(err => {
    console.error('[KISS Sorcar] Update check failed:', err);
  });

  console.log('KISS Sorcar extension activated');
}

export function deactivate(): void {
  sidebarView?.dispose();
  sidebarView = undefined;
  console.log('KISS Sorcar extension deactivated');
}
