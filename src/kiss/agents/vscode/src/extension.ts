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

  // Synchronise the kiss-web daemon's ``work_dir`` for THIS window
  // with the workspace folder that VS Code launched in.  The daemon
  // tracks one work_dir per client connection (one connection per VS
  // Code window), so every window keeps its own work_dir — the folder
  // open in that window — no matter how many windows share the daemon.
  // ``AgentClient`` re-sends the setWorkDir preamble on every
  // (re)connect, so calling this before ``ensureDependencies()`` is
  // safe even when the daemon is restarted later.
  sidebarView.syncWorkDir();

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
    vscode.commands.registerCommand('kissSorcar.runSelection', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const sel = editor.document.getText(editor.selection);
      if (!sel || !sel.trim()) {
        showInformationNotification('No text selected');
        return;
      }
      // Copy the highlighted text, paste it into the chat webview's
      // input textbox and submit it to the agent (opens/resolves the
      // sidebar webview first when it is closed).
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

  // Commit message generation — sets the Git SCM input box.  ``reveal``
  // opens the SCM view; it must be requested only on user-visible
  // transitions (generation start, final message), NOT on every
  // countdown tick — a per-second ``workbench.view.scm`` steals the
  // sidebar focus once per second for the whole generation.
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

  // Countdown shown in the SCM input box while a commit message is being
  // generated. Starts at ``commitCountdownSeconds``, decrements every
  // second, stays at 0 if the agent is still generating. Cleared/replaced
  // when the result arrives.
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

  // Returns true iff the first Git repository has at least one staged change.
  const hasStagedChanges = async (): Promise<boolean> => {
    try {
      const api = await getGitApi();
      // Can't check — let generation proceed.
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
    // Teardown for the no-reply paths: if the generation ends without
    // ever firing ``onCommitMessage`` (daemon never replies and the
    // 30 s promise timeout resolves, or the user cancels), the
    // countdown interval must be cleared — otherwise it overwrites the
    // SCM input box with "Generating in 0s ..." every second FOREVER —
    // and the stale countdown text must be wiped.  When a real result
    // arrived, the ``onCommitMessage`` handler above already stopped
    // the countdown (``stopCommitCountdown`` is undefined), so this
    // teardown is a no-op and never clobbers the generated message.
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
    } catch {
      // Already registered by another extension — ignored
    }
  }

  // Merge commands — route to the active tab's MergeManager
  for (const cmd of Object.values(MERGE_ACTIONS)) {
    context.subscriptions.push(
      vscode.commands.registerCommand(`kissSorcar.${cmd}`, () => {
        sidebarView!.handleMergeCommand(cmd);
      }),
    );
  }

  // Auto-reload when this extension has been (re)installed.
  //
  // We watch *only* ``~/.kiss/.extension-updated``.  ``install.sh``,
  // ``scripts/build-extension.sh`` and ``scripts/release.sh`` all touch
  // that marker as their *final* step, after ``code --install-extension``
  // has returned and the kiss-web daemon has been restarted.  A reload
  // triggered by the marker is therefore guaranteed to bring up fully
  // installed extension files.
  //
  // We deliberately do NOT watch ``out/extension.js``.  An earlier version
  // did, but ``install.sh``'s ``tsc`` step rewrites ``out/extension.js``
  // ~5–15 s *before* the rest of the install (``copy-kiss.sh``, daemon
  // restart, ``code --install-extension``) completes.  Polling that file
  // raced with the in-flight install and reloaded the window mid-step,
  // tearing down the integrated terminal that was running ``install.sh``
  // (the terminal shutdown writes ``\x03`` to the PTY, which is why users
  // saw an unexplained ``^C`` and an aborted install).  Marker-only
  // watching breaks that race: the reload can only fire after install.sh
  // is done.
  //
  // ``fs.watchFile`` uses stat-polling, so it works even though the marker
  // is repeatedly deleted and recreated (``ensureDependencies()`` clears
  // it after consuming it).
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

  // A same-version ``code --install-extension --force`` first DELETES the
  // extension directory and then re-extracts it, so ``out/extension.js`` is
  // transiently missing (and briefly partially written) for a noticeable
  // window.  Reloading during that window brings the chat webview up against
  // a half-installed extension (its ``media`` resources are gone) — the chat
  // view renders blank.  So we never reload until ``out/extension.js`` is
  // present, non-empty and size-stable across two consecutive polls
  // (``codeReady``).
  //
  // We *prefer* to also wait for the kiss-web daemon's socket to be back so
  // the reloaded webview can reconnect immediately, but we must NOT block on
  // it indefinitely.  ``install.sh`` deliberately kills the daemon and deletes
  // the socket before writing the update marker, and on a source install the
  // socket only returns once the *post-reload* ``ensureDependencies()``
  // respawns the daemon.  Hard-gating the reload on the socket therefore
  // dead-locks: the reload waits for a socket that can only come back after
  // the reload.  That stranded users on stale code until the 60 s hard
  // timeout, so they restarted VS Code by hand.  Instead, once the code is
  // stable we give the socket a short grace window and then reload regardless
  // — the webview's ``AgentClient`` auto-reconnects when the daemon comes back.
  const RELOAD_SETTLE_INTERVAL_MS = 500;
  // How long to keep waiting for the daemon socket after the extension code
  // has settled before reloading anyway.
  const RELOAD_SOCKET_GRACE_MS = 3_000;
  // Absolute ceiling: reload no later than this even if the code never settles
  // (e.g. an interrupted reinstall) so the user is never stranded.
  const RELOAD_SETTLE_TIMEOUT_MS = 15_000;
  const triggerReload = () => {
    if (reloadTriggered || settleTimer) return;
    let prevSize = -1;
    let waited = 0;
    // Wall-clock at which the extension code first became stable; -1 until
    // then.  Used to bound the post-code-ready wait for the daemon socket.
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
      // Reload when the code is stable AND either the socket is back or the
      // grace window has elapsed; or when the absolute timeout is hit.
      if (
        (codeReady && (socketUp || codeStableFor >= RELOAD_SOCKET_GRACE_MS)) ||
        waited >= RELOAD_SETTLE_TIMEOUT_MS
      ) {
        doReload();
      }
    }, RELOAD_SETTLE_INTERVAL_MS);
  };

  // Single watcher: the ``~/.kiss/.extension-updated`` marker file.
  //
  // install.sh / build-extension.sh / release.sh all write this marker as
  // their final step (after ``code --install-extension`` has returned and
  // the kiss-web daemon has been restarted), so a stat change on it
  // signals "extension is fully installed; safe to reload".  This covers
  // both same-version reinstalls (where ``out/extension.js`` would be
  // overwritten in place) and version bumps (where the new extension is
  // extracted into a fresh directory).
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
  // The gate is per-WORKSPACE (not per-machine) so that every new
  // workspace gets its secondary sidebar widened to ~1/3 of the
  // window on the first open, while preserving any width the user
  // manually tweaks on subsequent reopens of the same workspace.
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

  // Decide whether to auto-open the secondary sidebar.
  // True on first launch in THIS workspace (firstLaunchDone is
  // undefined) or after a rebuild/reinstall (marker written by
  // build-extension.sh).  We use a local boolean because
  // workspaceState.update() is async and the get() below would still
  // see the stale value.  The gate is per-WORKSPACE so every new
  // workspace auto-selects the KISS Sorcar tab in the secondary
  // panel on its first open.
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
  // Re-arm the Tips window when the extension has just been rebuilt or
  // reinstalled (install.sh / build-extension.sh / release.sh restarted
  // kiss-web and wrote the update marker): remove ~/.kiss/TIPS_SHOWN so
  // the next chat webview render auto-opens the Tips window once again.
  // Must run before ensureDependencies(), which consumes the marker.
  resetTipsOnExtensionUpdate();

  // On first launch in this workspace, auto-open the secondary sidebar
  // chat and focus the input so the user can start typing immediately.
  if (shouldAutoOpen) {
    setTimeout(async () => {
      await sidebarView!.focusChatInput();
      await context.workspaceState.update('firstLaunchDone', true);
    }, 1000);
  }

  // Auto-install dependencies in background
  ensureDependencies().catch(err => {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[KISS Sorcar] Dependency setup error:', err);
    showErrorNotification(
      `KISS Sorcar: Setup failed — ${msg}. Check ~/.kiss/install.log for details.`,
    );
  });

  // Active upstream update check.
  //
  // ``ensureDependencies()`` only restarts the kiss-web daemon on the
  // slow (install) path; on the fast "all deps present" path the
  // already-running daemon's PyPI poll is the only thing watching
  // PyPI — and that poll's cached answer can be up to an hour stale.
  // We additionally probe PyPI directly here so a fresh VS Code launch
  // always sees the most recent release without waiting for the next
  // daemon poll cycle.  All side effects are guarded:
  //   * the helper rate-limits itself via ~/.kiss/.update-check.json
  //     so we hit PyPI at most a few times per day;
  //   * any error (network, malformed payload) is swallowed so update
  //     checking can never break extension activation.
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
