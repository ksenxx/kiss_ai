// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import {MERGE_ACTIONS, SorcarSidebarView} from './SorcarSidebarView';
import {getGitApi} from './gitApi';
import {isReloadReady} from './reloadGuard';

import {ensureDependencies, ensureLocalBinInPath} from './DependencyInstaller';
import {findKissProject} from './kissPaths';
import {kissHomeDir} from './userAssets';
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
      void sidebarView!.appendToInput(text);
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

  const repoRootOf = (rootUri: unknown): string | undefined => {
    const fsPath = (rootUri as {fsPath?: unknown} | undefined)?.fsPath;
    return typeof fsPath === 'string' ? fsPath : undefined;
  };

  type GitRepoLike = {
    rootUri?: {fsPath?: string};
    inputBox: {value: string};
    state: {indexChanges: unknown[]};
  };

  const pickRepo = (
    repositories: GitRepoLike[],
    repoRoot?: string,
  ): GitRepoLike | undefined => {
    if (repoRoot) {
      const match = repositories.find(r => r.rootUri?.fsPath === repoRoot);
      if (match) return match;
    }
    return repositories[0];
  };

  // Serialize SCM input-box writes: an older, slower write (e.g. a stale
  // countdown tick) must never land after — and overwrite — a newer one.
  let scmWriteChain: Promise<void> = Promise.resolve();
  const setScmMessage = (
    message: string,
    reveal = false,
    repoRoot?: string,
  ): Promise<void> => {
    scmWriteChain = scmWriteChain.then(async () => {
      try {
        const api = await getGitApi();
        const repo = api
          ? pickRepo(api.repositories as GitRepoLike[], repoRoot)
          : undefined;
        if (repo) {
          repo.inputBox.value = message;
          if (reveal) vscode.commands.executeCommand('workbench.view.scm');
        }
      } catch (err) {
        console.error('[kissSorcar] Failed to set SCM message:', err);
      }
    });
    return scmWriteChain;
  };

  const commitCountdownSeconds = 20;
  let stopCommitCountdown: (() => void) | undefined;
  // Repository root of the generation currently in flight, or undefined
  // when none is (used to route the result and ignore canceled ones).
  let pendingCommitRepoRoot: string | undefined;
  let commitGenPending = false;
  const startCommitCountdown = (repoRoot?: string) => {
    stopCommitCountdown?.();
    let seconds = commitCountdownSeconds;
    void setScmMessage(`Generating in ${seconds}s ...`, true, repoRoot);
    const interval = setInterval(() => {
      seconds = Math.max(seconds - 1, 0);
      void setScmMessage(`Generating in ${seconds}s ...`, false, repoRoot);
    }, 1000);
    stopCommitCountdown = () => {
      clearInterval(interval);
      stopCommitCountdown = undefined;
    };
  };

  context.subscriptions.push(
    sidebarView!.onCommitMessage(ev => {
      if ((ev.tabId ?? '') !== '') return;
      // A canceled generation must not apply a late backend result.
      if (!commitGenPending) return;
      commitGenPending = false;
      const repoRoot = pendingCommitRepoRoot;
      const countdownWasRunning = stopCommitCountdown !== undefined;
      stopCommitCountdown?.();
      if (ev.error) {
        showWarningNotification(`Commit message: ${ev.error}`);
        if (countdownWasRunning) void setScmMessage('', false, repoRoot);
      } else if (ev.message) {
        void setScmMessage(ev.message, true, repoRoot);
      } else if (countdownWasRunning) {
        void setScmMessage('', false, repoRoot);
      }
    }),
  );

  const hasStagedChanges = async (repoRoot?: string): Promise<boolean> => {
    try {
      const api = await getGitApi();
      if (!api || api.repositories.length === 0) return true;
      const repo = pickRepo(api.repositories as GitRepoLike[], repoRoot);
      if (!repo) return true;
      return repo.state.indexChanges.length > 0;
    } catch (err) {
      console.error('[kissSorcar] Failed to check staged changes:', err);
      return true;
    }
  };

  const triggerCommitMessageGeneration = async (
    rootUri?: unknown,
    _context?: unknown,
    token?: vscode.CancellationToken,
  ): Promise<void> => {
    const repoRoot = repoRootOf(rootUri);
    if (!(await hasStagedChanges(repoRoot))) {
      await setScmMessage('Error: nothing staged', true, repoRoot);
      return;
    }
    pendingCommitRepoRoot = repoRoot;
    commitGenPending = true;
    startCommitCountdown(repoRoot);
    const teardown = () => {
      if (stopCommitCountdown) {
        stopCommitCountdown();
        void setScmMessage('', false, repoRoot);
      }
    };
    const cancelSub = token?.onCancellationRequested(() => {
      commitGenPending = false;
      teardown();
    });
    return sidebarView!.generateCommitMessage(token).finally(() => {
      cancelSub?.dispose();
      commitGenPending = false;
      pendingCommitRepoRoot = undefined;
      teardown();
    });
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
  const markerPath = path.join(kissHomeDir(), '.extension-updated');
  const sockPath =
    process.env.KISS_SORCAR_SOCK || path.join(kissHomeDir(), 'sorcar.sock');

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
      // Reset the stability clock whenever the bundle changes again, so
      // time spent through unstable writes never counts as "stable".
      if (!codeReady) codeReadySince = -1;
      else if (codeReadySince < 0) codeReadySince = waited;
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
      const widenTimer = setTimeout(async () => {
        // The extension may have been deactivated before this fires.
        if (!sidebarView) return;
        await vscode.commands.executeCommand(
          'workbench.action.focusAuxiliaryBar',
        );
        await sidebarView.widenToOneThird();
        await vscode.commands.executeCommand(
          'workbench.action.focusFirstEditorGroup',
        );
        await context.workspaceState.update('sidebarWidened', true);
      }, 500);
      context.subscriptions.push({dispose: () => clearTimeout(widenTimer)});
    });
  }

  const extensionUpdatedMarker = markerPath;
  let shouldAutoOpen = !context.workspaceState.get<boolean>('firstLaunchDone');
  if (fs.existsSync(extensionUpdatedMarker)) {
    shouldAutoOpen = true;
    void context.workspaceState.update('firstLaunchDone', undefined);
  }
  resetTipsOnExtensionUpdate();

  if (shouldAutoOpen) {
    const autoOpenTimer = setTimeout(async () => {
      if (!sidebarView) return;
      await sidebarView.focusChatInput();
      await context.workspaceState.update('firstLaunchDone', true);
    }, 1000);
    context.subscriptions.push({dispose: () => clearTimeout(autoOpenTimer)});
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
