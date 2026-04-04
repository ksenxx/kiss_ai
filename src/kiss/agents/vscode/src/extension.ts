/**
 * KISS Sorcar VS Code Extension entry point.
 */

import * as vscode from 'vscode';
import { TabManager } from './SorcarTab';
import { ensureDependencies, ensureLocalBinInPath } from './DependencyInstaller';

let tabManager: TabManager | undefined;

export function activate(context: vscode.ExtensionContext): void {
  ensureLocalBinInPath();
  console.log('KISS Sorcar extension activating...');

  tabManager = new TabManager(context.extensionUri);
  context.subscriptions.push({ dispose: () => tabManager?.dispose() });

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

  let _chatFocused = false;
  let _focusToggling = false;
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(() => {
      _chatFocused = false;
    })
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('kissSorcar.toggleFocus', async () => {
      if (_focusToggling) return;
      _focusToggling = true;
      try {
        if (_chatFocused) {
          _chatFocused = false;
          await vscode.commands.executeCommand('workbench.action.focusActiveEditorGroup');
        } else {
          _chatFocused = true;
          const tab = tabManager!.getActiveTab();
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
      _chatFocused = false;
      vscode.commands.executeCommand('workbench.action.focusActiveEditorGroup');
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

  const triggerCommitMessageGeneration = (
    _rootUri?: unknown,
    _context?: unknown,
    token?: vscode.CancellationToken
  ): Thenable<void> | void => {
    let tab = tabManager!.getActiveTab();
    if (!tab) tab = tabManager!.createTab();
    return tab.generateCommitMessage(token);
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

  // Auto-open a chat tab on activation, restoring the last session
  tabManager.createTab(true);

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
  console.log('KISS Sorcar extension deactivated');
}
