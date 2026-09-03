// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import {SorcarSidebarView} from './SorcarSidebarView';
import {getGitApi} from './gitApi';
import {isReloadReady} from './reloadGuard';

import {ensureDependencies, ensureLocalBinInPath} from './DependencyInstaller';
import {findKissProject} from './kissPaths';
import {kissHomeDir, sorcarSockPath} from './userAssets';
import {resetTipsOnExtensionUpdate} from './SorcarTab';
import {
  checkForExtensionUpdate,
  snoozeUpdateNotification,
} from './UpdateChecker';
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

  // A generation in flight for one repository.
  //
  // A workspace can hold several -- a multi-root workspace, or a repo
  // with a vendored sub-checkout -- and VS Code tells the callback which
  // one the user clicked in.  Every part of a generation therefore
  // belongs to a repository: which folder the daemon diffs, which SCM
  // input box shows the countdown and the answer, and which request the
  // answer belongs to.  Sharing any of them wrote one repository's
  // message into another's box.
  interface CommitGeneration {
    repoRoot?: string;
    pending: boolean;
    stopCountdown?: () => void;
  }

  // The daemon claims one generation per tabId, so two repositories
  // asking at once must ask under different ids or the second is
  // dropped.  The id is derived from the root, so a second click on the
  // same repository still joins the first.
  const scmTabIdFor = (repoRoot?: string): string =>
    repoRoot ? `scm:${repoRoot}` : '';

  const commitGens = new Map<string, CommitGeneration>();
  const commitGenInFlight = new Map<string, Promise<void>>();

  const startCommitCountdown = (gen: CommitGeneration) => {
    gen.stopCountdown?.();
    let seconds = commitCountdownSeconds;
    void setScmMessage(`Generating in ${seconds}s ...`, true, gen.repoRoot);
    const interval = setInterval(() => {
      seconds = Math.max(seconds - 1, 0);
      void setScmMessage(`Generating in ${seconds}s ...`, false, gen.repoRoot);
    }, 1000);
    gen.stopCountdown = () => {
      clearInterval(interval);
      gen.stopCountdown = undefined;
    };
  };

  context.subscriptions.push(
    sidebarView!.onCommitMessage(ev => {
      const gen = commitGens.get(ev.tabId ?? '');
      // A canceled generation must not apply a late backend result.
      if (!gen || !gen.pending) return;
      gen.pending = false;
      const repoRoot = gen.repoRoot;
      const countdownWasRunning = gen.stopCountdown !== undefined;
      gen.stopCountdown?.();
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

  const runCommitMessageGeneration = async (
    rootUri?: unknown,
    token?: vscode.CancellationToken,
  ): Promise<void> => {
    const repoRoot = repoRootOf(rootUri);
    if (!(await hasStagedChanges(repoRoot))) {
      await setScmMessage('Error: nothing staged', true, repoRoot);
      return;
    }
    const tabId = scmTabIdFor(repoRoot);
    const gen: CommitGeneration = {repoRoot, pending: true};
    commitGens.set(tabId, gen);
    startCommitCountdown(gen);
    const teardown = () => {
      if (gen.stopCountdown) {
        gen.stopCountdown();
        void setScmMessage('', false, repoRoot);
      }
    };
    const cancelSub = token?.onCancellationRequested(() => {
      gen.pending = false;
      teardown();
    });
    return sidebarView!
      .generateCommitMessage(token, tabId, repoRoot)
      .finally(() => {
        cancelSub?.dispose();
        gen.pending = false;
        teardown();
        if (commitGens.get(tabId) === gen) commitGens.delete(tabId);
      });
  };

  // The generation in flight for each repository, if any.
  //
  // A second invocation for the SAME repository while one is running --
  // a double click on the SCM sparkle, or one of the two hijacked ids
  // below firing -- must JOIN it rather than start a competing one.  The
  // sidebar already de-duplicates per tab, but it does so by handing back
  // an already-resolved promise meaning "someone else owns this";
  // treating that as "my generation finished" tore down the real one and
  // dropped its result on the floor.  The promise is registered
  // synchronously, before the first `await`, so two calls made in the
  // same tick cannot both slip past this guard.
  //
  // A request for a DIFFERENT repository is not a duplicate and must not
  // be swallowed by it.
  const triggerCommitMessageGeneration = (
    rootUri?: unknown,
    _context?: unknown,
    token?: vscode.CancellationToken,
  ): Promise<void> => {
    const tabId = scmTabIdFor(repoRootOf(rootUri));
    const existing = commitGenInFlight.get(tabId);
    if (existing) return existing;
    const running = runCommitMessageGeneration(rootUri, token).finally(() => {
      if (commitGenInFlight.get(tabId) === running) {
        commitGenInFlight.delete(tabId);
      }
    });
    commitGenInFlight.set(tabId, running);
    return running;
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

  const extJsPath = path.join(context.extensionPath, 'out', 'extension.js');
  const markerPath = path.join(kissHomeDir(), '.extension-updated');
  const sockPath = sorcarSockPath();

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

  let shouldAutoOpen = !context.workspaceState.get<boolean>('firstLaunchDone');
  if (fs.existsSync(markerPath)) {
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
      `KISS Sorcar: Setup failed — ${msg}. ` +
        `Check ${path.join(kissHomeDir(), 'install.log')} for details.`,
    );
  });

  void checkForExtensionUpdate({
    kissProjectPath: findKissProject() || undefined,
    notify: ({latest, current}: {latest: string; current: string}) => {
      void showInformationNotification(
        `KISS Sorcar: a new release (${latest}) is available. ` +
          `You are on ${current}.`,
        'Update now',
        'Remind me later',
      ).then(action => {
        if (action === 'Update now') {
          sidebarView?.runUpdate();
        } else if (action === 'Remind me later') {
          snoozeUpdateNotification({latest});
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
