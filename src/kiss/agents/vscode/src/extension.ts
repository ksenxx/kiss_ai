/**
 * KISS Sorcar VS Code Extension entry point.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { TabManager } from './SorcarTab';
import { SorcarSidebarView } from './SorcarSidebarView';
import { ensureDependencies, ensureLocalBinInPath } from './DependencyInstaller';

let tabManager: TabManager | undefined;
let sidebarView: SorcarSidebarView | undefined;

export function activate(context: vscode.ExtensionContext): void {
  ensureLocalBinInPath();
  console.log('KISS Sorcar extension activating...');

  tabManager = new TabManager(context.extensionUri);
  context.subscriptions.push({ dispose: () => tabManager?.dispose() });

  // --- Secondary sidebar chat view ---
  sidebarView = new SorcarSidebarView(context.extensionUri, tabManager.mergeManager);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'kissSorcar.chatViewSecondary',
      sidebarView,
      { webviewOptions: { retainContextWhenHidden: true } },
    )
  );
  context.subscriptions.push({ dispose: () => sidebarView?.dispose() });

  // --- Commands ---

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.openPanel', () => {
      const tab = tabManager!.getActiveTab();
      if (tab) {
        tab.panel.reveal();
      } else {
        tabManager!.createTab();
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.newConversation', () => {
      tabManager!.createTab();
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.stopTask', () => {
      tabManager!.getActiveTab()?.stopTask();
    })
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
      const tab = tabManager!.createTab();
      tab.submitTask(sel.trim());
    })
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
      const hunkRef = `WORK_DIR/${filePath}:@@ -${startLine},${lineCount} +${startLine},${lineCount} @@ `;
      const tab = tabManager!.getActiveTab() || tabManager!.createTab();
      tab.appendToInput(hunkRef);
    })
  );

  let _focusToggling = false;
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.toggleFocus', async () => {
      if (_focusToggling) return;
      _focusToggling = true;
      try {
        const tab = tabManager!.getActiveTab();
        if (tab && tab.panel.active) {
          // Chat tab is currently focused → switch to text editor
          await vscode.commands.executeCommand('workbench.action.focusFirstEditorGroup');
        } else {
          // Text editor is focused → switch to chat tab
          if (tab) {
            await tab.focusChatInput();
          } else {
            const newTab = tabManager!.createTab();
            await newTab.focusChatInput();
          }
        }
      } finally {
        _focusToggling = false;
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.focusEditor', () => {
      vscode.commands.executeCommand('workbench.action.focusFirstEditorGroup');
    })
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

  context.subscriptions.push(
    tabManager.onCommitMessage((ev) => {
      if (ev.error) {
        vscode.window.showWarningMessage(`Commit message: ${ev.error}`);
      } else if (ev.message) {
        setScmMessage(ev.message);
      }
    })
  );

  context.subscriptions.push(
    sidebarView!.onCommitMessage((ev) => {
      if (ev.error) {
        vscode.window.showWarningMessage(`Commit message: ${ev.error}`);
      } else if (ev.message) {
        setScmMessage(ev.message);
      }
    })
  );

  const triggerCommitMessageGeneration = (
    _rootUri?: unknown,
    _context?: unknown,
    token?: vscode.CancellationToken
  ): Thenable<void> | void => {
    return tabManager!.generateCommitMessage(token);
  };

  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.generateCommitMessage', triggerCommitMessageGeneration)
  );

  for (const cmdId of [
    'github.copilot.git.generateCommitMessage',
    'git.generateCommitMessage',
  ]) {
    try {
      context.subscriptions.push(
        vscode.commands.registerCommand(cmdId, triggerCommitMessageGeneration)
      );
    } catch {
      // Already registered by another extension — ignored
    }
  }

  // Merge commands
  for (const cmd of ['acceptChange', 'rejectChange', 'prevChange', 'nextChange', 'acceptAll', 'rejectAll', 'acceptFile', 'rejectFile'] as const) {
    context.subscriptions.push(
      vscode.commands.registerCommand(`kissSorcar.${cmd}`, () => {
        tabManager!.mergeManager[cmd]();
      })
    );
  }

  // Auto-reload when this extension's files are replaced (e.g. VSIX reinstall).
  // fs.watchFile uses stat-polling so it works even when the file is deleted
  // and recreated, which is what happens during VSIX installation.
  const extJsPath = path.join(context.extensionPath, 'out', 'extension.js');
  fs.watchFile(extJsPath, { interval: 2000 }, (curr, prev) => {
    if (curr.mtimeMs !== prev.mtimeMs || curr.ino !== prev.ino) {
      fs.unwatchFile(extJsPath);
      vscode.commands.executeCommand('workbench.action.reloadWindow');
    }
  });
  context.subscriptions.push({ dispose: () => fs.unwatchFile(extJsPath) });

  // Register tree view so the activity-bar icon creates a new chat tab on click.
  // After creating the tab we close the sidebar, so the next click re-opens it
  // (visibility flips true again) and the cycle repeats.
  const treeView = vscode.window.createTreeView('kissSorcar.chatView', {
    treeDataProvider: {
      getTreeItem: (el: string) => new vscode.TreeItem(el),
      getChildren: () => [],
    },
  });
  context.subscriptions.push(treeView);

  let lastTabCreatedAt = Date.now(); // guards against double-creation on activation
  treeView.onDidChangeVisibility(e => {
    if (e.visible && Date.now() - lastTabCreatedAt > 1000) {
      lastTabCreatedAt = Date.now();
      tabManager!.createTab();
      vscode.commands.executeCommand('workbench.action.closeSidebar');
    }
  });

  // Restore chat tabs from the previous VSCode session.
  // VSCode calls deserializeWebviewPanel for each saved panel of this viewType.
  let restoredCount = 0;
  context.subscriptions.push(
    vscode.window.registerWebviewPanelSerializer('kissSorcar.chat', {
      async deserializeWebviewPanel(panel: vscode.WebviewPanel, state: unknown) {
        restoredCount++;
        const task = (state as Record<string, unknown> | null)?.task as string | undefined;
        tabManager!.createTab(true, panel, task);
      },
    })
  );

  // Auto-open a chat tab only on the very first activation after installation.
  // On subsequent launches, rely on VSCode's built-in tab serializer to restore
  // previously open chat tabs.
  const ACTIVATED_KEY = 'kissSorcar.hasBeenActivated';
  const hasBeenActivated = context.globalState.get<boolean>(ACTIVATED_KEY, false);
  if (!hasBeenActivated) {
    context.globalState.update(ACTIVATED_KEY, true);
    setTimeout(() => {
      if (restoredCount === 0) {
        tabManager!.createTab(true);
      }
    }, 200);
  }

  // Auto-install dependencies in background
  ensureDependencies().catch(err => {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[KISS Sorcar] Dependency setup error:', err);
    vscode.window.showErrorMessage(
      `KISS Sorcar: Setup failed — ${msg}. Check ~/.kiss/install.log for details.`
    );
  });

  console.log('KISS Sorcar extension activated');
}

export function deactivate(): void {
  tabManager?.dispose();
  tabManager = undefined;
  sidebarView?.dispose();
  sidebarView = undefined;
  console.log('KISS Sorcar extension deactivated');
}
