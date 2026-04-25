/**
 * KISS Sorcar VS Code Extension entry point.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import {SorcarSidebarView} from './SorcarSidebarView';

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
      const hunkRef = `hunk @@ -${startLine},${lineCount} +${startLine},${lineCount} @@ in PWD/${filePath}:`;
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
  const extJsPath = path.join(context.extensionPath, 'out', 'extension.js');
  fs.watchFile(extJsPath, {interval: 2000}, (curr, prev) => {
    if (curr.mtimeMs !== prev.mtimeMs || curr.ino !== prev.ino) {
      fs.unwatchFile(extJsPath);
      vscode.commands.executeCommand('workbench.action.reloadWindow');
    }
  });
  context.subscriptions.push({dispose: () => fs.unwatchFile(extJsPath)});

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
