// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import {SorcarSidebarView} from './SorcarSidebarView';
import {CHAT_PANEL_VIEW_TYPE, SorcarPanelManager} from './SorcarPanelManager';
import {getGitApi} from './gitApi';
import {isReloadReady} from './reloadGuard';

import {ensureDependencies, ensureLocalBinInPath} from './DependencyInstaller';
import {findKissProject} from './kissPaths';
import {kissHomeDir, sorcarSockPath} from './userAssets';
import {
  HISTORY_PANEL_TAB_ID,
  historyPanelBodyAttrs,
  resetTipsOnExtensionUpdate,
} from './SorcarTab';
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
let panelManager: SorcarPanelManager | undefined;
let historyView: SorcarSidebarView | undefined;

// workspaceState key: root tab ids of the chat editor panels open at
// the previous session's shutdown (see priorPanelTabIds in activate).
const PANEL_TAB_IDS_KEY = 'kissSorcar.editorPanelTabIds';

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

  // Root tab ids of the chat panels open when the previous session
  // shut down — the registry tabs behind whatever serialized chat
  // placeholders the workbench restores after a reload. Read BEFORE
  // the manager can persist this session's first value; undefined
  // until a session of this extension version has recorded one.
  const priorPanelTabIds =
    context.workspaceState.get<string[]>(PANEL_TAB_IDS_KEY);

  // Fold one panel open/close into the persisted record. Per-tab
  // deltas, not whole-set writes: see the manager's _recordPanelTab
  // contract (a whole-set write mid-revival would replace the record
  // with a partial one).
  const recordPanelTab = (tabId: string, open: boolean): void => {
    const stored = context.workspaceState.get<string[]>(PANEL_TAB_IDS_KEY);
    const ids = stored ?? [];
    const next = open
      ? ids.includes(tabId)
        ? ids
        : [...ids, tabId]
      : ids.filter(id => id !== tabId);
    void context.workspaceState.update(PANEL_TAB_IDS_KEY, next);
  };

  // Panel closes retire their chat tabs through the sidebar
  // controller's long-lived daemon client: a panel's own client dies
  // with the panel, which could lose a closeTab queued while the
  // daemon was briefly unreachable.
  panelManager = new SorcarPanelManager(
    context.extensionUri,
    tabId => sidebarView?.closeChatTab(tabId),
    recordPanelTab,
  );
  context.subscriptions.push(panelManager.registerSerializer());
  context.subscriptions.push({
    dispose: () => {
      panelManager?.dispose();
      panelManager = undefined;
    },
  });

  const editorTabsMode = () => SorcarPanelManager.modeEnabled();

  // The chat surface commands act on: in editor-tabs mode the active
  // chat panel (opening one when asked to), otherwise the sidebar view.
  const chatController = (createIfMissing: boolean) => {
    if (!editorTabsMode()) return sidebarView!;
    if (createIfMissing) return panelManager!.revealActiveOrCreate();
    return panelManager!.activeController();
  };

  const workspaceDir = (): string => {
    const folders = vscode.workspace.workspaceFolders;
    return folders && folders.length > 0 ? folders[0].uri.fsPath : '';
  };

  // Switching the editor-tabs mode (from the settings UI toggle or
  // settings.json): ON migrates the registry's chats of this workspace
  // into editor tabs (the sidebar view hides via its `when` clause)
  // and CLOSES the secondary sidebar — the chats now live in editor
  // tabs, so the bar the sidebar chat occupied must not linger empty;
  // OFF closes the panels without retiring their chats and CLOSES the
  // secondary sidebar — the sidebar view re-adopts the chats from
  // `tabs_state` when a KS button (or anything else) next reveals it.
  // Guarded like the other optional host APIs (see
  // registerWebviewPanelSerializer): absent only in test stubs.
  let modeSwitchAt = 0;
  if (typeof vscode.workspace.onDidChangeConfiguration === 'function') {
    context.subscriptions.push(
      vscode.workspace.onDidChangeConfiguration(e => {
        if (!e.affectsConfiguration('kissSorcar.editorTabsMode')) return;
        modeSwitchAt = Date.now();
        if (editorTabsMode()) {
          panelManager!.enterMode(
            sidebarView!.getRegistryTabEntries(),
            workspaceDir(),
          );
          void vscode.commands.executeCommand(
            'workbench.action.closeAuxiliaryBar',
          );
        } else {
          panelManager!.closeAll();
          void vscode.commands.executeCommand(
            'workbench.action.closeAuxiliaryBar',
          );
        }
      }),
    );
  }

  sidebarView.syncWorkDir();

  // How many editor tabs host a chat — live panels AND the serialized
  // placeholders a window reload restores (indistinguishable in the
  // tabGroups API). -1 when the API is absent (test stubs).
  const chatEditorTabCount = (): number => {
    const groups = vscode.window.tabGroups?.all;
    if (!groups) return -1;
    let count = 0;
    for (const group of groups) {
      for (const tab of group.tabs) {
        const viewType = (tab.input as {viewType?: unknown} | null)?.viewType;
        if (
          typeof viewType === 'string' &&
          viewType.includes(CHAT_PANEL_VIEW_TYPE)
        ) {
          count += 1;
        }
      }
    }
    return count;
  };

  // True when ANY editor tab hosts a chat panel. The workbench's
  // restored chat tabs count too: after a window reload they exist as
  // serialized placeholders long before the panel manager adopts them,
  // and opening a "first" chat next to them would be a duplicate.
  const hasChatEditorTab = (): boolean => {
    const count = chatEditorTabCount();
    // Guarded like the other optional host APIs (absent in test stubs).
    if (count < 0) return panelManager!.panelCount > 0;
    return count > 0;
  };

  // A chat tab another client created — or first ran a task in — since
  // the last registry snapshot (e.g. a task run in the remote web app)
  // opens as an editor tab. Sidebar mode needs no host help: its
  // webview adopts the tab from the same `tabs_state` snapshot. The
  // long-lived sidebar controller is the listener because it is
  // connected to the daemon from activation on, panels or not (and it
  // requests a baseline snapshot on every daemon connect, so remote
  // tabs created before activation or during an outage arrive too).
  // The FIRST snapshot's tabs may be duplicated by this window's own
  // chat editor tabs — serialized placeholders are invisible to the
  // panel manager until the workbench revives them — so while any
  // chat tab exists, only first-snapshot tabs the previous session
  // verifiably did NOT hold (per the persisted panel tab ids) may
  // open: exactly the tabs a remote client created in the meantime.
  context.subscriptions.push(
    sidebarView.onRegistryTabsState(delta => {
      if (!editorTabsMode()) return;
      let toAdopt = delta.added;
      if (
        delta.firstSnapshot &&
        (panelManager!.panelCount > 0 || hasChatEditorTab())
      ) {
        const tabCount = chatEditorTabCount();
        if (tabCount >= 0 && tabCount === panelManager!.panelCount) {
          // Every chat editor tab is a LIVE panel the manager already
          // knows — no serialized placeholder is pending revival, so
          // nothing a first-snapshot tab could duplicate exists;
          // adoptRegistryTabs dedupes against the open panels by tab
          // and chat id. (A window that opened a fresh chat while the
          // daemon was down still adopts a remote tab from its late
          // first snapshot this way.)
        } else if (priorPanelTabIds) {
          const prior = new Set(priorPanelTabIds);
          toAdopt = delta.added.filter(e => !prior.has(e.tabId));
        } else {
          // Unrevived placeholders whose ids are unknowable (no
          // persisted record — first session of this version): adopt
          // nothing rather than risk duplicating them, but still let
          // the snapshot sync the open panels' chat bindings below.
          toAdopt = [];
        }
      }
      panelManager!.adoptRegistryTabs(toAdopt, workspaceDir(), delta.listed);
    }),
  );

  // A KS button brought the history panel on screen; an editor window
  // with no chat tab at all also gets a fresh conversation.
  const openChatIfNoneOpen = (): void => {
    if (!editorTabsMode()) return;
    if (panelManager!.panelCount > 0 || hasChatEditorTab()) return;
    void panelManager!.openNewChat().focusChatInput();
  };

  // The primary-sidebar history panel (editor-tabs mode): the same
  // chat webview in history-only mode (see historyPanelBodyAttrs), so
  // search, filters, deletes and live refreshes all come from main.js
  // unchanged. Its history clicks arrive as `openChatPanel` messages
  // and open editor tabs through the panel manager.
  historyView = new SorcarSidebarView(context.extensionUri, {
    rootTabId: HISTORY_PANEL_TAB_ID,
    bodyAttrs: historyPanelBodyAttrs(),
    onEvent: event => {
      if (event.kind === 'openChat') panelManager?.openChat(event);
    },
  });
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'kissSorcar.historyView',
      {
        resolveWebviewView: (view, resolveContext, token) => {
          historyView!.resolveWebviewView(view, resolveContext, token);
          view.onDidChangeVisibility(() => {
            if (view.visible) openChatIfNoneOpen();
          });
          openChatIfNoneOpen();
        },
      },
      {webviewOptions: {retainContextWhenHidden: true}},
    ),
  );
  context.subscriptions.push({dispose: () => historyView?.dispose()});

  // The editor-title KS button (editor-tabs mode): show the history
  // panel in the primary sidebar, and make sure a chat tab is open.
  // In non-editor-tabs mode a KS button only reveals the chat in the
  // secondary sidebar: no history panel, no new chat.
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.showHistory', async () => {
      if (!editorTabsMode()) {
        await sidebarView!.focusChatInput();
        return;
      }
      await vscode.commands.executeCommand('kissSorcar.historyView.focus');
      openChatIfNoneOpen();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.openPanel', () => {
      void chatController(true)!.focusChatInput();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.openSettings', () => {
      if (editorTabsMode()) {
        void panelManager!.openSettings();
      } else {
        void sidebarView!.openSettingsUI();
      }
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.newConversation', async () => {
      if (editorTabsMode()) {
        // Every conversation is its own editor tab in this mode. Route
        // through the active chat panel's webview when one exists: its
        // createNewTab posts openChatPanel WITH the composer draft, so
        // Cmd+T carries the drafted text into the new tab exactly like
        // the sidebar path below does.
        const active = panelManager!.activeController();
        if (active) {
          await active.focusChatInput();
          active.newConversation();
          return;
        }
        await panelManager!.openNewChat().focusChatInput();
        return;
      }
      await sidebarView!.focusChatInput();
      sidebarView!.newConversation();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.stopTask', () => {
      chatController(false)?.stopTask();
    }),
  );

  // The editor-title git-commit button (editor-tabs mode): run the
  // manual Git Commit of the active chat panel's working tree — the
  // same daemon autocommitAction flow the settings drawer's Git
  // Commit button uses. Also usable from the command palette in
  // sidebar mode, where it acts on the sidebar chat.
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.gitCommit', () => {
      void chatController(true)!.gitCommit();
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
      await chatController(true)!.submitTask(sel.trim());
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
      void chatController(true)!.appendToInput(text);
    }),
  );

  let _focusToggling = false;
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.toggleFocus', async () => {
      if (_focusToggling) return;
      _focusToggling = true;
      try {
        const controller = chatController(false);
        if (controller?.hasFocus) {
          // In editor-tabs mode the chat IS an editor, so "back to the
          // editor" means the previously used one; in sidebar mode the
          // first editor group.
          await vscode.commands.executeCommand(
            editorTabsMode()
              ? 'workbench.action.openPreviousRecentlyUsedEditor'
              : 'workbench.action.focusFirstEditorGroup',
          );
        } else {
          await chatController(true)!.focusChatInput();
        }
      } finally {
        _focusToggling = false;
      }
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.focusEditor', () => {
      // In editor-tabs mode the chat IS an editor in the first group;
      // "focus the editor" then means the previously used one.
      vscode.commands.executeCommand(
        editorTabsMode()
          ? 'workbench.action.openPreviousRecentlyUsedEditor'
          : 'workbench.action.focusFirstEditorGroup',
      );
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

  // The KS activity-bar button in non-editor-tabs mode shows this
  // dummy tree: never leave anything (history panel or otherwise) up
  // in the primary sidebar — close it back and reveal the existing
  // chat in the secondary sidebar instead, creating no new chat.
  // Bumped on EVERY visibility flip: the continuation below parks in two
  // awaits, and a newer user action (hiding the tree again, switching
  // primary-sidebar views) during that window must win over the stale
  // continuation instead of having focus stolen back from it.
  let treeVisGen = 0;
  treeView.onDidChangeVisibility(async e => {
    treeVisGen += 1;
    if (!e.visible) return;
    await vscode.commands.executeCommand('workbench.action.closeSidebar');
    // Event delivery is ordered, so the hide flip our own closeSidebar
    // just caused has arrived by now: snapshot AFTER it, and only newer
    // (user-driven) flips invalidate this continuation.
    const gen = treeVisGen;
    // The tree also pops up when editorTabsMode flips OFF while the
    // KISS container is the active primary-sidebar view (the history
    // panel hides, the tree takes its spot). That flip must leave the
    // secondary sidebar CLOSED, so give its config handler — which may
    // run in this same tick — a moment to record itself, then bail.
    await new Promise(r => setTimeout(r, 50));
    if (Date.now() - modeSwitchAt < 2000) return;
    if (gen !== treeVisGen) return;
    // Deactivation during either await disposes the view and clears the
    // module slot; a disposed surface must not be focused (and the old
    // `sidebarView!` assertion threw an unhandled TypeError here).
    if (!sidebarView) return;
    await sidebarView.focusChatInput();
  });

  if (!context.workspaceState.get<boolean>('sidebarWidened')) {
    sidebarView!.onFirstResolve(() => {
      const widenTimer = setTimeout(async () => {
        // The extension may have been deactivated before this fires.
        if (!sidebarView) return;
        // The one-time widening belongs to the sidebar surface only.
        if (editorTabsMode()) return;
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

  // A genuine first launch in this workspace — as opposed to the
  // auto-open replayed after an extension update (markerPath below).
  const firstLaunch = !context.workspaceState.get<boolean>('firstLaunchDone');
  let shouldAutoOpen = firstLaunch;
  if (fs.existsSync(markerPath)) {
    shouldAutoOpen = true;
    void context.workspaceState.update('firstLaunchDone', undefined);
  }
  resetTipsOnExtensionUpdate();

  if (shouldAutoOpen) {
    const autoOpenTimer = setTimeout(async () => {
      if (!sidebarView) return;
      if (firstLaunch) {
        // The workbench's default layout (code-server and recent VS
        // Code) starts with the secondary sidebar open on the built-in
        // Chat view. KISS Sorcar's chat replaces it: close the bar
        // before opening the chat surface. In sidebar mode the
        // focusChatInput below reopens it on the KISS chat view; in
        // editor-tabs mode the chat is an editor tab and the bar
        // stays closed.
        await vscode.commands.executeCommand(
          'workbench.action.closeAuxiliaryBar',
        );
      }
      await chatController(true)!.focusChatInput();
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
  // Shutdown first: the panel disposals below (and any the workbench
  // triggers) must not retire chats from the daemon's registry.
  panelManager?.markShutdown();
  panelManager?.dispose();
  panelManager = undefined;
  sidebarView?.dispose();
  sidebarView = undefined;
  historyView?.dispose();
  historyView = undefined;
  console.log('KISS Sorcar extension deactivated');
}
