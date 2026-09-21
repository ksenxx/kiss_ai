// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import {SorcarSidebarView} from './SorcarSidebarView';
import {SorcarPanelManager} from './SorcarPanelManager';
import {getGitApi} from './gitApi';
import {isReloadReady} from './reloadGuard';
import {syncEditorActionsLocation} from './editorActionsLocation';

import {ensureDependencies, ensureLocalBinInPath} from './DependencyInstaller';
import {findKissProject} from './kissPaths';
import {kissHomeDir, sorcarSockPath} from './userAssets';
import {
  HISTORY_PANEL_TAB_ID,
  historyPanelBodyAttrs,
  META_PANEL_TAB_ID,
  metaPanelBodyAttrs,
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
let metaView: SorcarSidebarView | undefined;

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

  // The secondary-sidebar Task Info view (editor-tabs mode): the
  // remote webapp's rightmost desktop panel — live task metadata plus
  // the running task's tmp/PROGRESS.md — rendered by the same chat
  // webview in meta-panel-mode. It mirrors the ACTIVE chat editor
  // panel: the panel manager relays each active panel's metaUpdate
  // reports into it through setMetaSink below.
  metaView = new SorcarSidebarView(context.extensionUri, {
    rootTabId: META_PANEL_TAB_ID,
    bodyAttrs: metaPanelBodyAttrs(),
    // The view is display-only: it opens no chats and owns no panel.
    onEvent: () => {},
  });
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'kissSorcar.metaViewSecondary',
      metaView,
      {webviewOptions: {retainContextWhenHidden: true}},
    ),
  );
  context.subscriptions.push({dispose: () => metaView?.dispose()});
  panelManager.setMetaSink((values, progressMd) => {
    metaView?.postMetaState(values, progressMd);
  });

  // Bring the Task Info view on screen in the secondary sidebar
  // without stealing the keyboard focus from wherever the caller
  // means to leave it (the `.focus` command both opens the bar and
  // focuses the view, so callers refocus the chat afterwards).
  const revealMetaView = async (): Promise<void> => {
    try {
      await vscode.commands.executeCommand(
        'kissSorcar.metaViewSecondary.focus',
      );
    } catch (err) {
      console.error('[KISS Sorcar] Task Info reveal failed:', err);
    }
  };

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
  // and KEEPS the secondary sidebar open, now showing the Task Info
  // view — the remote webapp's rightmost desktop panel — in the spot
  // the sidebar chat occupied;
  // OFF closes the panels without retiring their chats and OPENS the
  // secondary sidebar on the KISS Sorcar chat view, focusing its
  // composer — the chats moved there, so that is where the user must
  // land (the view re-adopts them from `tabs_state` as it resolves).
  // Guarded like the other optional host APIs (see
  // registerWebviewPanelSerializer): absent only in test stubs.
  //
  // Set by the OFF branch: its reveal must END with the chat composer
  // focused. A window that started in editor-tabs mode resolves the
  // sidebar view for the first time on that reveal, which arms the
  // one-time widening below; its focus handoff then goes back to the
  // chat instead of the editor group.
  let refocusChatAfterWiden = false;
  // The chat / task ids the sidebar chat view last reported showing
  // (see the activeTask wiring below): the OFF branch replays them
  // into the history panel, whose highlight the closing editor panels
  // would otherwise leave on the last ACTIVE panel's task.
  let sidebarActiveTask = {chatId: '', taskId: ''};
  if (typeof vscode.workspace.onDidChangeConfiguration === 'function') {
    context.subscriptions.push(
      vscode.workspace.onDidChangeConfiguration(e => {
        // The title-bar placement of the Sorcar buttons depends on the
        // custom title bar being visible: when its visibility setting
        // flips, re-sync (an owned "titleBar" value is restored while
        // the custom title bar is "never" — the actions would be
        // hidden — and re-applied when it comes back).
        if (
          e.affectsConfiguration('window.customTitleBarVisibility') &&
          editorTabsMode()
        ) {
          void syncEditorActionsLocation(context, true);
        }
        if (!e.affectsConfiguration('kissSorcar.editorTabsMode')) return;
        // Follow the mode with the editor-actions toolbar: ON moves
        // the four Sorcar editor-title buttons into the window title
        // bar, OFF restores the user's own actions location.
        void syncEditorActionsLocation(context, editorTabsMode());
        if (editorTabsMode()) {
          panelManager!.enterMode(
            sidebarView!.getRegistryTabEntries(),
            workspaceDir(),
          );
          // The secondary sidebar stays OPEN: the chat view it hosted
          // just hid (its `when` clause flipped false), and the Task
          // Info view takes its place. The reveal's focus then goes
          // back to the chat the user is migrating to.
          const revealManager = panelManager;
          void revealMetaView().then(() => {
            // The reveal resolves later; a mode flipped back OFF in
            // the meantime owns the focus now (focusChatInput below,
            // in the else branch), and this stale continuation must
            // not steal it. Nor may it outlive its own activation:
            // deactivate() clears panelManager, and a reactivation
            // installs a new one.
            if (!editorTabsMode() || panelManager !== revealManager) {
              return undefined;
            }
            return panelManager?.activeController()?.focusChatInput();
          });
        } else {
          panelManager!.closeAll();
          historyView?.postActiveTask(
            sidebarActiveTask.chatId,
            sidebarActiveTask.taskId,
          );
          // focusChatInput runs kissSorcar.chatViewSecondary.focus
          // (the view is contributable again — its `when` clause just
          // flipped true), which opens the secondary sidebar on the
          // KISS Sorcar view, waits for the webview to resolve and
          // focuses the composer.
          refocusChatAfterWiden = true;
          void sidebarView!.focusChatInput();
        }
      }),
    );
  }

  sidebarView.syncWorkDir();

  // In editor-tabs mode the Sorcar editor-title buttons (+, git
  // commit, gear, KS) live in the window title bar rather than the
  // editor tab bar; apply that on activation too, so the placement
  // holds in fresh windows. Activation with the mode OFF deliberately
  // does NOT restore: another window may have the mode ON (e.g. a
  // workspace-level override turns it off only here), and the setting
  // is global — restores happen only on explicit mode flips.
  if (editorTabsMode()) {
    void syncEditorActionsLocation(context, true);
    // Editor-tabs mode always shows the secondary sidebar: the Task
    // Info view is part of the mode's layout (the chat is an editor
    // tab, the bar shows the running task's metadata beside it), so a
    // window that starts in the mode brings it on screen too. The
    // reveal's focus goes back to the chat composer — or, before any
    // panel has revived, to the editor group the user started in.
    const revealManager = panelManager;
    void revealMetaView().then(() => {
      // A mode flipped off while the reveal was in flight owns the
      // focus now; this stale continuation must not steal it — nor may
      // it outlive its own activation (deactivate() clears
      // panelManager, a reactivation installs a new one).
      if (!editorTabsMode() || panelManager !== revealManager) {
        return undefined;
      }
      const active = panelManager?.activeController();
      if (active) return active.focusChatInput();
      // Back to the group the user started in — not group 1, where
      // focusFirstEditorGroup would drag a user working in group 2/3.
      return vscode.commands.executeCommand(
        'workbench.action.focusActiveEditorGroup',
      );
    });
  }

  // The editor-tabs-mode invariant — at least one chat editor tab is
  // always open, as the sidebar strip always keeps one chat tab. The
  // panel manager re-establishes it after every panel close; this
  // covers the two closes it cannot observe from a panel: a restored
  // placeholder closed before revival (tabGroups backstop) and a window
  // that starts with the mode on and no chat tab at all (restored
  // placeholders count, so a reload never gets a duplicate). The
  // activation open stays in the background — the user did not ask
  // for a chat right now.
  context.subscriptions.push(panelManager.watchEditorTabs());
  panelManager.ensureChatOpen({preserveFocus: true});

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
      if (delta.firstSnapshot && panelManager!.hasChatEditorTab()) {
        const tabCount = panelManager!.chatEditorTabCount();
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
  // with no chat tab at all also gets a fresh conversation (the
  // invariant normally holds already — this is the user asking for
  // the chat, so a fresh one takes the focus).
  const openChatIfNoneOpen = (): void => {
    // Optional: a history-view visibility flip may land after
    // deactivation cleared the module slot.
    panelManager?.ensureChatOpen();
  };

  // The primary-sidebar history panel (BOTH modes): the same chat
  // webview in history-only mode (see historyPanelBodyAttrs), so
  // search, filters, deletes and live refreshes all come from main.js
  // unchanged. Its history clicks arrive as `openChatPanel` messages;
  // in editor-tabs mode they open editor tabs through the panel
  // manager, in sidebar mode they open the chat in the secondary
  // sidebar's chat view.
  historyView = new SorcarSidebarView(context.extensionUri, {
    rootTabId: HISTORY_PANEL_TAB_ID,
    bodyAttrs: historyPanelBodyAttrs(),
    onEvent: event => {
      if (event.kind !== 'openChat') return;
      if (editorTabsMode()) {
        panelManager?.openChat(event);
      } else {
        void sidebarView?.openChatFromHistory(event);
      }
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
  // The history panel highlights (and scrolls to) the row of the task
  // the chat surface on screen shows: in editor-tabs mode the panel
  // manager relays the ACTIVE editor panel's ids, in sidebar mode the
  // secondary sidebar's chat view reports its own visible tab. Each
  // relay is gated on its mode so the hidden surface cannot override
  // the one the user is looking at.
  panelManager.setActiveTaskSink((chatId, taskId) => {
    if (editorTabsMode()) historyView?.postActiveTask(chatId, taskId);
  });
  sidebarView.onActiveTask = (chatId, taskId) => {
    sidebarActiveTask = {chatId, taskId};
    if (!editorTabsMode()) historyView?.postActiveTask(chatId, taskId);
  };

  // The KS button: show the history panel in the primary sidebar —
  // the same surface in both modes. Editor-tabs mode also makes sure
  // a chat tab is open (the chat lives in editor tabs there; in
  // sidebar mode it lives in the secondary sidebar and needs no help).
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.showHistory', async () => {
      await vscode.commands.executeCommand('kissSorcar.historyView.focus');
      if (editorTabsMode()) openChatIfNoneOpen();
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

  if (!context.workspaceState.get<boolean>('sidebarWidened')) {
    sidebarView!.onFirstResolve(() => {
      const widenTimer = setTimeout(async () => {
        // The extension may have been deactivated before this fires.
        // Clearing the timer cannot stop a callback that has already
        // started, so the view is captured here and its liveness
        // re-checked after every await: deactivation during one of
        // them clears the module slot (and disposes the view).
        const view = sidebarView;
        if (!view) return;
        // The one-time widening belongs to the sidebar surface only.
        if (editorTabsMode()) return;
        try {
          await vscode.commands.executeCommand(
            'workbench.action.focusAuxiliaryBar',
          );
          if (sidebarView !== view) return;
          await view.widenToOneThird();
          if (sidebarView !== view) return;
          // Hand focus back: to the editor group the user came from,
          // or — when the view first resolved because the user just
          // switched editor-tabs mode OFF — to the chat composer that
          // reveal promised (the widening's own focusAuxiliaryBar and
          // resize commands landed on top of it).
          if (refocusChatAfterWiden) {
            refocusChatAfterWiden = false;
            await view.focusChatInput();
          } else {
            await vscode.commands.executeCommand(
              'workbench.action.focusFirstEditorGroup',
            );
          }
          if (sidebarView !== view) return;
          await context.workspaceState.update('sidebarWidened', true);
        } catch (err) {
          console.error('[KISS Sorcar] sidebar widening failed:', err);
        }
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
      // Same discipline as the widen timer above: deactivation during
      // an await clears `sidebarView`/`panelManager`, and the old
      // `chatController(true)!` then dereferenced undefined and left
      // this callback rejecting unhandled.
      const view = sidebarView;
      if (!view) return;
      try {
        if (firstLaunch) {
          // The workbench's default layout (code-server and recent VS
          // Code) starts with the secondary sidebar open on the
          // built-in Chat view. KISS Sorcar replaces it: in sidebar
          // mode close the bar (the focusChatInput below reopens it on
          // the KISS chat view); in editor-tabs mode reveal the Task
          // Info view — the chat itself is an editor tab, and the bar
          // shows the running task's metadata beside it.
          if (editorTabsMode()) {
            await revealMetaView();
          } else {
            await vscode.commands.executeCommand(
              'workbench.action.closeAuxiliaryBar',
            );
          }
          if (sidebarView !== view) return;
        }
        const controller = chatController(true);
        if (!controller) return;
        await controller.focusChatInput();
        if (sidebarView !== view) return;
        await context.workspaceState.update('firstLaunchDone', true);
      } catch (err) {
        console.error('[KISS Sorcar] first-launch chat open failed:', err);
      }
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
        'Update when idle',
        'Remind me later',
      ).then(action => {
        if (action === 'Update now') {
          sidebarView?.runUpdate();
        } else if (action === 'Update when idle') {
          sidebarView?.updateWhenIdle();
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
  metaView?.dispose();
  metaView = undefined;
  console.log('KISS Sorcar extension deactivated');
}
